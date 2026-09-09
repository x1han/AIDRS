"""In-process PuffinRunner.

Replaces per-(Chr, Strand) subprocess fan-out with a single Python object that
holds the Puffin model and the selene_sdk.Genome once, and serves all TSS sites
via batched encoding + sequential single-sample forward. Output puffin_*.csv
files in puffin_out/ are byte-identical with the legacy puffin.py
--no_interpretation pipeline so that aggr_puffin_result in
src/tss_annotation.py continues to work unchanged.

Note on batching: a naive single-call `model.forward(batch_tensor)` is *not*
byte-identical with the legacy per-sample forward (`model(seq[None,:,:].T)`)
because torch's FFT-based convolutions pick different BLAS algorithms for
batch sizes > 1, which produces ~3e-6 float32 differences. Those differences
can flip a peak height across a rounding boundary (`round(_, 4)`) and
propagate into the final assessment.tsv. Sequential single-sample forward
is the only way to keep the output byte-identical while still saving the
subprocess fan-out cost (one model load + no per-(Chr, Strand) process
startup vs. N model loads in the legacy code).
"""

import logging
import os
import sys
import time
import warnings

import pandas as pd
import torch
import selene_sdk
from selene_sdk import sequences

# Repo root is needed to import the aided.Puffin.puffin module (the `aided`
# tree is not an installed package, it ships next to `src/`).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aided.Puffin.puffin import Puffin  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="selene_sdk")

logger = logging.getLogger("AIDRS")


class PuffinRunner:
    """Load Puffin model and selene_sdk.Genome once; serve batched inference."""

    # Sequence length required by Puffin's predict() before trimming 325 bp from
    # each side (matches puffin.py main: < 651 bp is rejected).
    MIN_SEQ_LEN = 651
    # Trim length applied to each side of the predicted profile.
    TRIM = 325
    # Half-window used to expand a TSS into a fetch window.
    HALF_WINDOW = 500

    def __init__(self, genome_path, num_threads=4):
        # Telemetry: ModelLoad
        self._t0 = time.perf_counter()

        # Be conservative on threads; sub-process fan-out used 4.
        # CLI override: aidrs --threads N -> tss_annotation.num_processes -> here.
        # qsub -V can inject OMP_NUM_THREADS from the submit host before our
        # os.environ write below fires, so we explicitly set here AFTER any
        # external propagation and pin torch to the same value.
        self._num_threads = max(1, int(num_threads))
        os.environ["OMP_NUM_THREADS"] = str(self._num_threads)
        os.environ["MKL_NUM_THREADS"] = str(self._num_threads)
        torch.set_num_threads(self._num_threads)

        # Load model once (CPU only — matches legacy puffin.py path on this box).
        self.model = Puffin(use_cuda=False)
        self.model.eval()  # disable dropout / BN running stats updates

        # Load genome once.
        self.genome = selene_sdk.sequences.Genome(input_path=genome_path)

        # Cached forward target indices.
        self._fwd_keys = list(self.model.targeti_map.keys())
        self._rev_keys = list(self.model.targeti_rev_map.keys())

        # Telemetry: ModelLoad done
        self._t1 = time.perf_counter()
        logger.info(
            f"[Telemetry PuffinRunner] ModelLoad: {self._t1 - self._t0:.3f}s | "
            f"num_threads={self._num_threads} (OMP={os.environ.get('OMP_NUM_THREADS')}, "
            f"torch={torch.get_num_threads()})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_sites(self, sites_df, puffin_out_path, half_window=None):
        """Run Puffin on every unique (Chr, TSS) site in sites_df.

        Args:
            sites_df: DataFrame containing at least Chr, Strand, and either
                TrStart (when Strand == '+') or TrEnd (when Strand == '-').
                Typically the merged SSC table from Stage 2.1.
            puffin_out_path: directory where puffin_*.csv files are written.
                Created if missing.
            half_window: half-window around the TSS. Defaults to
                PuffinRunner.HALF_WINDOW (500) which yields a 1001 bp window.

        Returns:
            dict mapping (Chrom, Start, End, Strand) -> trimmed_pred (np.ndarray
            of shape [10, L-650]).
        """
        if half_window is None:
            half_window = self.HALF_WINDOW

        t1 = self._t1

        os.makedirs(puffin_out_path, exist_ok=True)

        # Build the deduped list of (Chrom, Start, End, Strand) per
        # puffin.py's region-mode logic.
        sites = self._collect_sites(sites_df, half_window)

        # Fetch & filter sequences.
        sequences_bp = []  # list of (Chrom, Start, End, Strand, seq_bp)
        for key, seq_bp in self._fetch_sequences(sites):
            sequences_bp.append((key, seq_bp))

        # Telemetry: DataPrep done
        t2 = time.perf_counter()

        if not sequences_bp:
            t3 = t2
            t4 = t3
            logger.info(
                f"[Telemetry PuffinRunner] ModelLoad: {self._t1 - self._t0:.3f}s | "
                f"DataPrep: {t2 - t1:.3f}s | Inference: {t3 - t2:.3f}s | "
                f"DiskIO: {t4 - t3:.3f}s | Total: {t4 - self._t0:.3f}s"
            )
            return {}

        # Sequential single-sample forward: the only path to byte-identical
        # output with the legacy per-(Chr, Strand) subprocess pipeline.
        # Each sample gets its own `model(seq[None,:,:].transpose(1,2))` call.
        # We still benefit from loading the model and genome once (the
        # legacy code paid an N-model-load cost).
        all_preds = []
        # T1 instrumentation: per-PROGRESS_EVERY records, emit a flush=True
        # print so SGE logs surface real-time inference progress. At 431k
        # sites this loop can run for hours with no other log output.
        PROGRESS_EVERY = 5000
        total_seqs = len(sequences_bp)
        with torch.no_grad():
            for seq_idx, (_, seq_bp) in enumerate(sequences_bp):
                enc = sequences.sequence_to_encoding(
                    seq_bp,
                    base_to_index={
                        "A": 0, "a": 0,
                        "C": 1, "c": 1,
                        "G": 2, "g": 2,
                        "T": 3, "t": 3,
                    },
                    bases_arr="ACGT",
                )
                # Match legacy predict(): FloatTensor + .transpose(1,2) → [1, 4, L]
                x = torch.FloatTensor(enc)[None, :, :].transpose(1, 2)
                pred = self.model(x)  # [1, 10, L]
                all_preds.append(pred.detach().cpu().numpy()[0])
                done = seq_idx + 1
                if done % PROGRESS_EVERY == 0 or done == total_seqs:
                    elapsed = time.perf_counter() - t2
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (total_seqs - done) / rate if rate > 0 else float("inf")
                    print(
                        f"[puffin][T1] predict_sites progress: "
                        f"{done}/{total_seqs} rate={rate:.2f} seqs/s "
                        f"elapsed={elapsed:.1f}s eta={eta:.0f}s",
                        flush=True,
                    )

        # Telemetry: Inference done
        t3 = time.perf_counter()

        # Trim, write per-site CSVs in the legacy puffin.py format, and build
        # the result dict.
        results = {}
        for k_i, (key, seq_bp) in enumerate(sequences_bp):
            seq_len = len(seq_bp)
            pred_np = all_preds[k_i]
            trimmed = pred_np[:, self.TRIM : seq_len - self.TRIM]
            self._write_csv(puffin_out_path, key, seq_bp, trimmed)
            results[key] = trimmed

        # Telemetry: DiskIO done
        t4 = time.perf_counter()
        logger.info(
            f"[Telemetry PuffinRunner] ModelLoad: {self._t1 - self._t0:.3f}s | "
            f"DataPrep: {t2 - t1:.3f}s | Inference: {t3 - t2:.3f}s | "
            f"DiskIO: {t4 - t3:.3f}s | Total: {t4 - self._t0:.3f}s"
        )

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _collect_sites(self, sites_df, half_window):
        """Build a deterministic, deduped list of (Chrom, Start, End, Strand)."""
        seen = {}
        for (chrom, strand), sub in sites_df.groupby(["Chr", "Strand"], observed=True, sort=False):
            tss_col = "TrStart" if strand == "+" else "TrEnd"
            for tss_site in sub[tss_col].unique():
                tss_site = int(tss_site)
                start = tss_site - half_window
                end = tss_site + half_window
                key = (chrom, start, end, strand)
                if key not in seen:
                    seen[key] = tss_site
        # Stable order: sort by (Chr, Start, Strand) for reproducibility.
        return sorted(seen.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3]))

    def _fetch_sequences(self, sites):
        """Fetch genome sequences for each site; skip with warning if too short.

        `sites` is the list of ((Chrom, Start, End, Strand), tss_site) tuples
        produced by _collect_sites.
        """
        for key, tss_site in sites:
            (chrom, start, end, strand) = key
            if strand == "-":
                offset = 1
            else:
                offset = 0
            seq_bp = self.genome.get_sequence_from_coords(
                chrom, start + offset, end + offset, strand
            )
            if len(seq_bp) < self.MIN_SEQ_LEN:
                logger.warning(
                    f"Skipping {chrom}_{start}_{end}_{strand}: sequence length "
                    f"{len(seq_bp)} < {self.MIN_SEQ_LEN} bp (tss={tss_site})"
                )
                continue
            yield key, seq_bp

    def _write_csv(self, puffin_out_path, key, seq_bp, trimmed):
        """Write a single puffin_*.csv file matching puffin.py's predict() schema."""
        chrom, start, end, strand = key
        strand_token = "minus" if strand == "-" else "plus"
        name = f"puffin_{chrom}_{start}_{end}_{strand_token}"
        out_file = os.path.join(puffin_out_path, name + ".csv")

        seq_bp_trimmed = seq_bp[self.TRIM : -self.TRIM]
        lines = {}
        lines["Coordinate"] = list(range(len(seq_bp_trimmed)))
        lines["Sequence"] = list(seq_bp_trimmed)
        for k in self._fwd_keys:
            lines["Prediciton " + k] = trimmed[self.model.targeti_map[k], :]
        for k in self._rev_keys:
            lines["Prediciton rev strand " + k] = trimmed[
                self.model.targeti_rev_map[k], :
            ]

        df = pd.DataFrame.from_dict(lines, orient="index")
        df.to_csv(out_file)