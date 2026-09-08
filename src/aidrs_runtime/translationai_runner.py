"""In-process TranslationAIRunner.

Replaces per-(Chr, Strand) subprocess fan-out with a single Python object that
loads the 5 .h5 keras models ONCE and serves all sequences. Output
_predTIS_*.txt / _predTTS_*.txt / _predORFs_*.txt are byte-identical with the
legacy `translationai -I ... -t 0.5,0.5` subprocess pipeline so that
aggr_translationai_result in src/protein_coding_ability.py continues to work
unchanged.

Critical defenses (per F3b plan):
  - TF_CPP_MIN_LOG_LEVEL='3' is set BEFORE any keras/tf import.
  - CUDA_VISIBLE_DEVICES='-1' is preserved (no GPU on this baseline machine).
  - h5py.File is used inside a `with` context manager (avoid EMFILE leaks
    that the subprocess fan-out was hiding).
  - model.predict(..., verbose=0) suppresses the Keras progress bar.
  - 5 models are loaded once at __init__; load_model raises immediately on a
    corrupted .h5 (Fail-Fast) — no silent fallback.
"""

import logging
import os
import sys
import time

# CRITICAL: silence TF BEFORE any keras / tensorflow / h5py import.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Repo root is needed to import the translationai package (it ships under
# `repo/aided/TranslationAI/`, not as an installed package).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from keras.models import load_model  # noqa: E402
from pkg_resources import resource_filename  # noqa: E402
from translationai.fa_to_h5_converter import convert_fa_to_h5  # noqa: E402
from translationai.utils import categorical_crossentropy_2d, clip_datapoints  # noqa: E402

from src.aidrs_runtime.device_manager import device_manager

logger = logging.getLogger("AIDRS")


class TranslationAIRunner:
    """Load 5 .h5 keras models once; serve all sequences in-process."""

    MODEL_SCALE = "2000"
    MODELS_USED = ["l1", "l2", "l3", "l4", "l5"]
    BATCH_SIZE = 6
    # Ensemble averaging matches the legacy __main__.py.
    N_VERSIONS = 5

    def __init__(self):
        # Telemetry: ModelLoad start
        self._t0 = time.perf_counter()

        # Load 5 models ONCE. Fail-Fast on any corrupted .h5.
        self.models = []
        self.use_gpu = device_manager.transai_use_gpu
        self.batch_size = 128 if self.use_gpu else self.BATCH_SIZE
        for v in self.MODELS_USED:
            model_name = f"models/translationAI_{self.MODEL_SCALE}_{v}.h5"
            model_path = resource_filename("translationai", model_name)
            m = load_model(model_path)
            m.compile(loss=categorical_crossentropy_2d, optimizer="adam")
            self.models.append(m)

        # Telemetry: ModelLoad done
        self._t1 = time.perf_counter()
        logger.info(
            f"[Telemetry TranslationAIRunner] ModelLoad: {self._t1 - self._t0:.3f}s"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_fasta(self, fa_path, threshold_str="0.5,0.5"):
        """Run TranslationAI on the FASTA at fa_path.

        Produces _predTIS_<t>.txt, _predTTS_<t>.txt, _predORFs_<t>.txt next
        to fa_path with content byte-identical to the legacy `translationai
        -I fa_path -t threshold_str` subprocess.

        Args:
            fa_path: input FASTA file path. The legacy protein_coding_ability
                path uses {tmp}/TranslationAI_temp/{Chr}_{Strand}.fasta.
            threshold_str: e.g. "0.5,0.5" (cutoff) or "2,2" (top-k). The
                default matches AIDRS's translationai_score_threshold=0.9
                call site (which uses 0.5,0.5 in run_translationai).

        Returns:
            Path to the produced _predORFs_*.txt (or None if fa_path was
            empty / produced zero sequences).
        """
        t1 = self._t1

        threshold_str = str(threshold_str)
        # P1-4: validate threshold_str has exactly two comma-separated
        # values before indexing. A malformed single-value or empty input
        # would otherwise raise IndexError with no actionable context.
        parts = threshold_str.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"threshold_str must be 'TIS,TTS' comma-separated; got {threshold_str!r}"
            )
        TIS_score_cutoff = float(parts[0])
        TTS_score_cutoff = float(parts[1])
        if TIS_score_cutoff >= 1:
            TIS_score_cutoff = int(TIS_score_cutoff)
        if TTS_score_cutoff >= 1:
            TTS_score_cutoff = int(TTS_score_cutoff)

        # ---- DataPrep: in-process .fa -> .h5 conversion ----
        h5f_name = fa_path[:-3] + ".h5"
        if not os.path.exists(h5f_name):
            convert_fa_to_h5(fa_path, h5f_name)
        # Telemetry: DataPrep done
        t2 = time.perf_counter()

        # Output file paths matching the legacy CLI schema exactly.
        fn_tis = fa_path[:-3] + f"_predTIS_{threshold_str.split(',')[0]}.txt"
        fn_tts = fa_path[:-3] + f"_predTTS_{threshold_str.split(',')[1]}.txt"
        fn_orf = fa_path[:-3] + f"_predORFs_{'_'.join(threshold_str.split(','))}.txt"

        # ---- Inference ----
        # 'with' context manager (CRITICAL: avoid EMFILE in in-process mode).
        with h5py.File(h5f_name, "r") as h5f:
            num_idx = len(h5f.keys()) // 2
            with open(fa_path, "r") as fh_seq:
                seq_lines = fh_seq.readlines()
            seq_num = len(seq_lines) // 2
            if num_idx != seq_num:
                raise Exception(
                    "!!!ERROR: The sequence numbers from the .h5 file and "
                    "the .fa file do not match!"
                )

            pred_tis_lines = []  # list of "{header}\t{pos,score}\t..." lines
            pred_tts_lines = []
            for idx in range(num_idx):
                X = h5f["X" + str(idx)][:]
                Y = h5f["Y" + str(idx)][:]
                Xc, Yc = clip_datapoints(X, Y, int(self.MODEL_SCALE), 1)
                Yps = [np.zeros(Yc[0].shape)]
                # Ensemble averaging across all 5 models. CRITICAL: verbose=0
                # suppresses Keras progress bar.
                for m in self.models:
                    Yp = m.predict(Xc, batch_size=self.batch_size, verbose=0)
                    if not isinstance(Yp, list):
                        Yp = [Yp]
                    Yps[0] += Yp[0] / self.N_VERSIONS

                seq_len = len(seq_lines[idx * 2 + 1])
                is_expr = (Yc[0].sum(axis=(1, 2)) >= 1)

                # --- TIS ---
                Y_pred_TIS = Yps[0][is_expr, :, 2].flatten()
                argsorted_y_pred_TIS = np.argsort(Y_pred_TIS[0:seq_len])[::-1]
                if TIS_score_cutoff < 1:  # cutoff
                    ind_threshold = len(argsorted_y_pred_TIS)
                    for i in range(len(argsorted_y_pred_TIS)):
                        if Y_pred_TIS[argsorted_y_pred_TIS[i]] < TIS_score_cutoff:
                            ind_threshold = i
                            break
                else:  # top-k
                    ind_threshold = TIS_score_cutoff
                idx_pred_tis = argsorted_y_pred_TIS[: int(ind_threshold)]
                pred_TIS_pos_score = [
                    f"{str(ind)},{Y_pred_TIS[ind]}" for ind in idx_pred_tis
                ]
                pred_tis_lines.append(
                    seq_lines[idx * 2].strip("\n")
                    + "\t"
                    + "\t".join(pred_TIS_pos_score)
                    + "\n"
                )

                # --- TTS ---
                Y_pred_TTS = Yps[0][is_expr, :, 1].flatten()
                argsorted_y_pred_TTS = np.argsort(Y_pred_TTS[0:seq_len])[::-1]
                if TTS_score_cutoff < 1:  # cutoff
                    ind_threshold = len(argsorted_y_pred_TTS)
                    for i in range(len(argsorted_y_pred_TTS)):
                        if Y_pred_TTS[argsorted_y_pred_TTS[i]] < TTS_score_cutoff:
                            ind_threshold = i
                            break
                else:  # top-k
                    ind_threshold = TTS_score_cutoff
                idx_pred_tts = argsorted_y_pred_TTS[: int(ind_threshold)]
                pred_TTS_pos_score = [
                    f"{str(ind)},{Y_pred_TTS[ind]}" for ind in idx_pred_tts
                ]
                pred_tts_lines.append(
                    seq_lines[idx * 2].strip("\n")
                    + "\t"
                    + "\t".join(pred_TTS_pos_score)
                    + "\n"
                )

        # Telemetry: Inference done
        t3 = time.perf_counter()

        # ---- DiskIO: write _predTIS_, _predTTS_, _predORFs_ files ----
        with open(fn_tis, "w") as fh:
            for line in pred_tis_lines:
                fh.write(line)
        with open(fn_tts, "w") as fh:
            for line in pred_tts_lines:
                fh.write(line)

        # ORF post-processing — exactly mirrors legacy __main__.py.
        self._write_orfs(
            fn_tis=fn_tis,
            fn_tts=fn_tts,
            fn_out=fn_orf,
            TIS_score_cutoff=TIS_score_cutoff,
            TTS_score_cutoff=TTS_score_cutoff,
        )

        # Telemetry: DiskIO done
        t4 = time.perf_counter()
        logger.info(
            f"[Telemetry TranslationAIRunner] ModelLoad: {self._t1 - self._t0:.3f}s | "
            f"DataPrep: {t2 - t1:.3f}s | Inference: {t3 - t2:.3f}s | "
            f"DiskIO: {t4 - t3:.3f}s | Total: {t4 - self._t0:.3f}s"
        )

        device_manager.cleanup_tf_vram()
        return fn_orf if num_idx > 0 else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _write_orfs(fn_tis, fn_tts, fn_out, TIS_score_cutoff, TTS_score_cutoff):
        """Combine per-seq TIS/TTS hits into _predORFs_*.txt (legacy schema).

        This is a direct port of the legacy __main__.py ORF-finding block,
        kept here so that aggr_translationai_result's glob on
        *_predORFs_0.5_0.5.txt continues to see byte-identical content.
        """
        # Parse _predTIS_*.txt
        header_list = []
        predTIS_pos_score_list = []
        with open(fn_tis, "r") as fpr1:
            line = 1
            while line:
                line = fpr1.readline()
                if not line:
                    break
                header_list.append(line.split("\t")[0])
                words = line.strip().split("\t")
                predTIS_pos_score = []
                if len(words) > 1:
                    TIS_pos_score_predict = [
                        c.split(",") for c in words[1:]
                    ]
                    TIS_pos_score_predict = [
                        [int(c[0]), float(c[1])] for c in TIS_pos_score_predict
                    ]
                    TIS_pos_score_predict = sorted(
                        TIS_pos_score_predict, key=lambda x: x[1], reverse=True
                    )
                    if TIS_score_cutoff >= 1:
                        predTIS_pos_score = TIS_pos_score_predict[
                            0 : int(TIS_score_cutoff)
                        ]
                    else:
                        for c in TIS_pos_score_predict:
                            if c[1] >= TIS_score_cutoff:
                                predTIS_pos_score.append(c)
                            else:
                                break
                predTIS_pos_score_list.append(predTIS_pos_score)

        # Parse _predTTS_*.txt
        predTTS_pos_score_list = []
        with open(fn_tts, "r") as fpr2:
            line = 1
            while line:
                line = fpr2.readline()
                if not line:
                    break
                words = line.strip().split("\t")
                predTTS_pos_score = []
                if len(words) > 1:
                    TTS_pos_score_predict = [
                        c.split(",") for c in words[1:]
                    ]
                    TTS_pos_score_predict = [
                        [int(c[0]), float(c[1])] for c in TTS_pos_score_predict
                    ]
                    TTS_pos_score_predict = sorted(
                        TTS_pos_score_predict, key=lambda x: x[1], reverse=True
                    )
                    if TTS_score_cutoff >= 1:
                        predTTS_pos_score = TTS_pos_score_predict[
                            0 : int(TTS_score_cutoff)
                        ]
                    else:
                        for c in TTS_pos_score_predict:
                            if c[1] >= TTS_score_cutoff:
                                predTTS_pos_score.append(c)
                            else:
                                break
                predTTS_pos_score_list.append(predTTS_pos_score)

        # Write _predORFs_*.txt
        tot_predicted_ORF_num = 0
        seq_num = len(predTIS_pos_score_list)
        with open(fn_out, "w") as fhOut:
            for seq_i in range(seq_num):
                predicted_ORF_list = []
                for TIS_j in range(len(predTIS_pos_score_list[seq_i])):
                    TIS_pos_j, TIS_score_j = predTIS_pos_score_list[seq_i][TIS_j]
                    for TTS_k in range(len(predTTS_pos_score_list[seq_i])):
                        TTS_pos_k, TTS_score_k = predTTS_pos_score_list[seq_i][TTS_k]
                        CDR_len = TTS_pos_k - TIS_pos_j
                        if CDR_len > 0 and not CDR_len % 3:
                            tot_predicted_ORF_num += 1
                            predicted_ORF_list.append(
                                [
                                    [TIS_pos_j, TTS_pos_k],
                                    [TIS_score_j, TTS_score_k],
                                    TIS_score_j * TTS_score_k,
                                ]
                            )
                predicted_ORF_list = sorted(
                    predicted_ORF_list, key=lambda x: x[2], reverse=True
                )
                predicted_ORF_list = [c[0] + c[1] for c in predicted_ORF_list]
                predicted_ORF_list = [
                    list(map(lambda x: str(x), c)) for c in predicted_ORF_list
                ]
                predicted_ORF_str_list = [
                    ",".join(c) for c in predicted_ORF_list
                ]
                content = (
                    header_list[seq_i]
                    + "\t"
                    + "\t".join(predicted_ORF_str_list)
                )
                if len(predicted_ORF_list) > 0:
                    fhOut.write(content + "\n")