#!/usr/bin/env python

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
from pyfaidx import Fasta
import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
from functools import partial

# F3b: import the in-process TranslationAIRunner. The runner loads the 5 .h5
# keras models once and serves all sequences; orf_predict_by_translationai
# no longer spawns one subprocess per (Chr, Strand).
from .aidrs_runtime.translationai_runner import TranslationAIRunner
from .aidrs_runtime.concurrency import drain_futures_loud, get_process_pool

class TranslationAI_ORF:
    def __init__(self, genome, tmp_path='temp', translationai_score_threshold=0.9, num_processes=8):
        # P7 fix: keep the original path string so we can ship it across the
        # multiprocessing pickling boundary (a live pyfaidx.Fasta is a
        # BufferedReader and cannot be pickled).
        self.genome_path = genome
        self.genome = Fasta(genome)
        self.tmp_path = tmp_path
        self.translationai_score_threshold = translationai_score_threshold
        self.num_processes = num_processes
        # F3b: instantiate the in-process runner ONCE. Loading the 5 .h5
        # models here (instead of once-per-(Chr, Strand) subprocess) is the
        # whole point of F3b.
        self._runner = TranslationAIRunner()

    @staticmethod
    def fetch_exon(row):
        chrom = str(row['Chr'])
        # Defensive int() on coordinates: skip the row (return None) when
        # TrStart/TrEnd are NaN/missing instead of crashing with
        # "int() can't convert non-string with explicit base".
        try:
            start = int(row['TrStart'])      # 1-based
            end   = int(row['TrEnd'])
        except (TypeError, ValueError):
            logger.debug("Skipping fetch_exon for %s:%s-%s due to NaN/inf coordinates",
                         row.get('Chr', 'NA'), row.get('TrStart', 'NA'), row.get('TrEnd', 'NA'))
            return None
        strand = str(row['Strand'])
        
        # Parse SSC column to get exon ranges
        ssc = str(row['SSC'])
        positions = [start] + list(map(int, ssc.split('-'))) + [end]
        
        # Convert position list to exon ranges (start, end)
        exon_ranges = []
        for i in range(0, len(positions), 2):
            if i + 1 < len(positions):
                exon_start = positions[i]
                exon_end = positions[i + 1]
                exon_ranges.append((exon_start, exon_end))
        
        # If on negative strand, reverse the entire exon list order and reverse coordinates of each exon
        if strand == '-':
            exon_ranges.reverse()
            # Reverse coordinate order within each exon
            exon_ranges = [(end, start) for start, end in exon_ranges]

        return exon_ranges

    
    @staticmethod
    def fetch_seq(row, genome):
        chrom = str(row['Chr'])
        # Defensive int() on coordinates: skip the row (return '') when
        # TrStart/TrEnd are NaN/missing instead of crashing with
        # "int() can't convert non-string with explicit base". Caller
        # (run_translationai) treats empty string as a no-op sequence.
        try:
            start = int(row['TrStart'])      # 1-based
            end   = int(row['TrEnd'])
        except (TypeError, ValueError):
            logger.debug("Skipping fetch_seq for %s:%s-%s due to NaN/inf coordinates",
                         row.get('Chr', 'NA'), row.get('TrStart', 'NA'), row.get('TrEnd', 'NA'))
            return ''
        strand = str(row['Strand'])

        # Parse SSC column to get exon ranges
        ssc = str(row['SSC'])
        try:
            positions = [start] + list(map(int, ssc.split('-'))) + [end]
        except ValueError:
            logger.debug("Skipping fetch_seq for %s:%s-%s due to unparseable SSC %s",
                         chrom, start, end, ssc)
            return ''
        
        # Convert position list to exon ranges (start, end)
        exon_ranges = []
        for i in range(0, len(positions), 2):
            if i + 1 < len(positions):
                exon_start = positions[i]
                exon_end = positions[i + 1]
                exon_ranges.append((exon_start, exon_end))
        
        # Extract exon sequences
        exon_sequences = []
        for exon_start, exon_end in exon_ranges:
            # Ensure exon range is within transcript range
            if exon_start >= start and exon_end <= end:
                # pyfaidx slice is 0-based, left-closed, right-open
                exon_seq_str = genome[chrom][exon_start-1:exon_end].seq
                exon_sequences.append(exon_seq_str)
        
        # Concatenate all exon sequences
        seq_str = ''.join(exon_sequences)
        seq = Seq(seq_str)                         # turn into Biopython Seq

        if strand == '-':
            seq = seq.reverse_complement()

        return str(seq)

    @staticmethod
    def run_translationai(df, genome, tmp_path, runner=None, worker_id=None):
        if df.empty:
            return

        df_fasta = df.copy()

        Chrom = df_fasta['Chr'].unique()[0]
        Strand = df_fasta['Strand'].unique()[0]

        df_fasta['seq'] = df_fasta.apply(lambda row: TranslationAI_ORF.fetch_seq(row, genome), axis=1)

        # P7 fanout: write the per-(Chr, Strand) FASTA directly into tmp_path
        # (which, when called from a worker process, is the per-worker subdir
        # {fanout_root}/worker_{pid}/). The pred* files then sit next to it.
        os.makedirs(tmp_path, exist_ok=True)
        fasta_out_path = os.path.join(tmp_path, f"{Chrom}_{Strand}.fasta")

        with open(fasta_out_path, "w") as fh:
            pass

        with open(fasta_out_path, "a") as fh:
            keys = (
                df_fasta.reset_index(drop=True)               # Row index becomes 0,1,2,...
                .pipe(lambda d:                       # Concatenate key
                    d['TrStart'].astype(str) + '~' +
                    d['SSC'].astype(str) + '~' +
                    d['TrEnd'].astype(str)
                )
            )
            for idx, key in keys.items():
                output_entries = []

                # Generate FASTA header and sequence in a single try block so
                # that exceptions in seq access (KeyError), seq is None/NaN,
                # or numeric conversion errors all skip the row instead of
                # writing a corrupt literal "None"/"nan" into the FASTA file.
                try:
                    seq = df_fasta.iloc[idx]['seq']
                    if seq is None or (isinstance(seq, float) and pd.isna(seq)) or seq == "":
                        logger.debug("Skipping FASTA entry %d: empty/None sequence", idx)
                        continue
                    seq_name = (
                        f">{df_fasta.iloc[idx]['Chr']}:"
                        f"{int(df_fasta.iloc[idx]['TrStart'])}-"
                        f"{int(df_fasta.iloc[idx]['TrEnd'])}"
                        f"({df_fasta.iloc[idx]['Strand']})"
                        f"({key})"
                        f"({int(0)}, "
                        f"{int(0)},)"
                    )
                    output_entries.append(f"{seq_name}\n{seq}\n")
                except (KeyError, ValueError, TypeError) as e:
                    # Handle missing fields, type errors, or unparseable SSCs
                    logger.warning("Skipping FASTA entry at index %d: %s", idx, e)
                    continue

                # Batch write to file
                if output_entries:
                    fh.writelines(output_entries)
        # F3b: in-process prediction. The 5 .h5 models live in `runner`; we
        # just hand it the per-(Chr, Strand) FASTA and let it produce the
        # _predTIS_*/_predTTS_*/_predORFs_* files that aggr_translationai_result
        # already knows how to read.
        if runner is None:
            raise RuntimeError(
                "TranslationAI runner not initialised; F3b requires an "
                "in-process TranslationAIRunner."
            )
        runner.predict_fasta(fasta_out_path, threshold_str="0.5,0.5")

    def orf_predict_by_translationai(self, df):
        # P7 fanout: distribute (Chr, Strand) groups across worker processes
        # via ProcessPoolExecutor (dynamic scheduling — the pool reclaims a
        # slot the moment any worker finishes, so a tiny Chr1+ group doesn't
        # stall behind a giant Chr3- group sitting on a busy worker). Each
        # worker owns its own 5-model ensemble and writes into a deterministic
        # per-(Chr, Strand) subdir under TranslationAI_temp/. Groups assigned
        # to the same worker are processed sequentially WITHIN that worker
        # (preserves BLAS determinism per-strand); across workers the order
        # is free (as_completed).
        # LPT scheduling: longest (chr,strand) groups first to minimise straggler wait.
        # Secondary key str(name) makes order independent of pandas groupby internal order.
        # We iterate as (name, group) tuples so the key is always accessible — pandas
        # groupby objects in 2.x do not expose `.name` on the DataFrame slice.
        df_groups = sorted(
            [(name, g) for name, g in df.groupby(['Chr','Strand'], observed=True)],
            key=lambda ng: (-len(ng[1]), str(ng[0])),
        )
        # Unpack for downstream code that expects list[DataFrame] but uses
        # df_group['Chr'].unique()[0] / df_group['Strand'].unique()[0] — still works on the DataFrame half.
        df_groups = [g for _, g in df_groups]
        if not df_groups:
            return
        num_workers = max(1, min(self.num_processes, len(df_groups)))

        # P7.2 re-run safety: blow away any stale outputs from a prior
        # crashed run BEFORE we kick off the pool, so aggr_translationai_result
        # never mixes old _pred* files with new ones. fanout_root itself is
        # the directory aggr_translationai_result is called against;
        # workers nest their per-(Chr, Strand) subdirs inside it.
        fanout_root = os.path.join(self.tmp_path, "TranslationAI_temp")
        shutil.rmtree(fanout_root, ignore_errors=True)
        os.makedirs(fanout_root, exist_ok=True)

        # P7.1 dynamic chunking: one task per (Chr, Strand) group, executor
        # does the scheduling. No round-robin pre-slicing — groups are
        # submitted individually so a fast-finishing worker immediately
        # picks up the next group. Each task carries the group wrapped in a
        # one-element list to preserve the worker's "iterable of groups"
        # contract without re-shaping the worker signature.
        tasks = [
            ([df_group], self.genome_path, fanout_root, idx)
            for idx, df_group in enumerate(df_groups)
        ]

        # spawn context: avoid forking the parent's already-loaded TF / h5py
        # state into every worker (which would defeat the "no model sharing
        # across processes" constraint and risk CUDA / BLAS re-init races).
        ctx = mp.get_context("spawn")
        try:
            with get_process_pool(num_workers=num_workers, mp_context=ctx) as executor:
                futures = [
                    executor.submit(TranslationAI_ORF._run_translationai_worker, task)
                    for task in tasks
                ]
                # Fail-loud: aggregate ALL worker exceptions (no silent swallow)
                # and post-condition rglob to catch zero-output silent path.
                drain_futures_loud(futures, stage_name="2.4 TranslationAI")

                # Block on silent zero-output path: workers exit 0 but produce
                # zero _predORFs_*.txt files (e.g., empty model loading).
                pred_files = list(Path(fanout_root).rglob("*_predORFs_0.5_0.5.txt"))
                if len(df_groups) > 0 and not pred_files:
                    raise RuntimeError(
                        f"[FATAL] TranslationAI completed with exit code 0, but "
                        f"produced ZERO _predORFs_0.5_0.5.txt files under "
                        f"{fanout_root}! This indicates silent failure during "
                        f"inference or empty model loading."
                    )
        except BrokenProcessPool as e:
            raise RuntimeError(
                f"TranslationAI worker pool broken: {e}"
            ) from e

    @staticmethod
    def _run_translationai_worker(args):
        """Per-process worker: load 5 models, process its assigned
        (Chr, Strand) group(s) sequentially, write outputs to a deterministic
        per-(Chr, Strand) subdir of fanout_root.

        The df_groups argument is an iterable of per-(Chr, Strand)
        DataFrames (today: a one-element list, since the executor submits
        one group per task). Groups are processed sequentially so BLAS
        stays deterministic per-strand; sibling worker processes run in
        parallel but each holds an independent 5-model ensemble.

        Tuple: (chr_strand_groups, genome_path, fanout_root, runner_id).
        """
        df_groups, genome_path, fanout_root, runner_id = args

        # CRITICAL: collapse TF intra/inter-op threads to 1. With multiple
        # worker processes, leaving TF to grab "all" cores would cause thread
        # contention that breaks per-strand determinism (different run → same
        # byte output is a P7 hard requirement).
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        # P7 fix: open pyfaidx.Fasta inside the worker (a live Fasta handle
        # cannot survive pickle across the spawn boundary).
        genome = Fasta(genome_path)

        # Each worker instantiates its own TranslationAIRunner — own 5-model
        # ensemble. No model sharing across processes (per the P7 contract).
        runner = TranslationAIRunner()

        for df_group in df_groups:
            Chrom = str(df_group['Chr'].unique()[0])
            Strand = str(df_group['Strand'].unique()[0])
            # P7.2 deterministic, content-addressed output dir: same
            # (Chr, Strand) → same path on rerun, so the outer rmtree in
            # orf_predict_by_translationai guarantees a clean slate without
            # pid-name races.
            worker_dir = os.path.join(fanout_root, f"{Chrom}_{Strand}")
            os.makedirs(worker_dir, exist_ok=True)
            TranslationAI_ORF.run_translationai(
                df_group,
                genome=genome,
                tmp_path=worker_dir,
                runner=runner,
                worker_id=runner_id,
            )
    
    @staticmethod
    def check_nmd(df, translationai_score_threshold=0.9):
        """
        Determine NMD (Nonsense-Mediated Decay)
        Judge whether NMD is triggered based on EJC-dependent and EJC-independent mechanisms
        """
        def determine_nmd_status(row):
            # If no ORF is predicted, return 'no_orf'
            if (row['TIS_related_location'] == 'no' or 
                row['TTS_related_location'] == 'no'):
                return 'no_orf'
            
            # If either TIS_score or TTS_score is less than translationai_score_threshold, return 'no_orf'
            try:
                tis_score = float(row['TIS_score']) if row['TIS_score'] != 'no' else 0.0
                tts_score = float(row['TTS_score']) if row['TTS_score'] != 'no' else 0.0
                if tis_score < translationai_score_threshold or tts_score < translationai_score_threshold:
                    return 'no_orf'
            except (ValueError, TypeError):
                return 'no_orf'
            
            try:
                tis_pos = int(row['TIS_related_location'])
                tts_pos = int(row['TTS_related_location'])

                
                # Get exon ranges
                exon_ranges = TranslationAI_ORF.fetch_exon(row)
                if exon_ranges is None:
                    # fetch_exon returned None due to NaN/inf coordinates (P1-9).
                    # Treat as 'no_orf' rather than crashing the worker.
                    # Note: writing via df.at[idx, ...] from inside df.apply is a
                    # pandas anti-pattern AND `idx` is not in scope here — drop it.
                    # The return below is what gets assigned to Predict_NMD by
                    # the outer df.apply(determine_nmd_status, axis=1).
                    return 'no_orf'

                # EJC-dependent NMD determination
                ejc_dependent_nmd = False
                if len(exon_ranges) >= 2:
                    # Calculate cumulative exon length to find each exon-exon junction position
                    cumulative_length = 0
                    junction_positions = []
                    
                    for i, (start, end) in enumerate(exon_ranges[:-1]):  # Exclude last exon
                        cumulative_length += abs(end - start + 1)
                        junction_positions.append(cumulative_length)
                    
                    # Check if stop codon is located upstream of any exon-exon junction by ≥55nt
                    if tts_pos <= junction_positions[-1] - 55:
                        ejc_dependent_nmd = True
                
                # EJC-independent NMD determination - check 3' UTR length
                ejc_independent_nmd = False
                
                # Calculate total CDS length (from TIS to TTS)
                total_cds_length = tts_pos - tis_pos + 1
                
                # Calculate total transcript length
                total_transcript_length = sum(abs(end - start + 1) for start, end in exon_ranges)
                
                # Calculate 3' UTR length (total transcript length - TTS position)
                utr3_length = total_transcript_length - tts_pos
                
                # If 3' UTR length > 1kb (1000nt), then EJC-independent NMD may be triggered
                if utr3_length > 1000:
                    ejc_independent_nmd = True
                
                # Comprehensive NMD status determination
                if (ejc_dependent_nmd or ejc_independent_nmd):
                    return 'NMD'
                else:
                    return 'Normal'
                    
            except (ValueError, TypeError):
                return 'unknown'
        
        # Apply NMD determination function
        df['Predict_NMD'] = df.apply(determine_nmd_status, axis=1)
        
        return df
    
    def aggr_translationai_result(self, df, translationai_out_path):
        all_lines = []
        # P7 fanout: workers nest outputs under
        # {translationai_out_path}/{Chr}_{Strand}/, so walk the tree to
        # collect every _predORFs_0.5_0.5.txt regardless of which worker
        # produced it.
        for root, _, files in os.walk(translationai_out_path):
            for name in files:
                if name.endswith("_predORFs_0.5_0.5.txt"):
                    file_path = os.path.join(root, name)

                    # Check if file exists and is not empty
                    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                        continue
                    try:
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                            # Filter empty lines and lines with only whitespace
                            non_empty_lines = [line for line in lines if line.strip()]
                            if non_empty_lines:
                                all_lines.extend(non_empty_lines)
                            else:
                                print(f"Warning: File contains no valid data: {file_path}")
                    except (IOError, OSError) as e:
                        print(f"Error reading file {file_path}: {e}")
                        continue
        
        # Parse translationai output results and create DataFrame
        results = []
        
        # Parse data from all_lines
        for line in all_lines:
            line = line.strip()
            if line and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    # Parse header part
                    header = parts[0]
                    # Parse numerical part
                    values = parts[1].split(',')
                    
                    if len(values) >= 4:
                        # Extract chr
                        chr_part = header.split(':')[0].replace('>', '')
                        
                        # Extract strand (first parenthesis)
                        strand_start = header.find('(')
                        strand_end = header.find(')', strand_start)
                        strand = header[strand_start+1:strand_end] if strand_start != -1 and strand_end != -1 else ''
                        
                        # Extract key (second parenthesis)
                        key_start = header.find('(', strand_end+1)
                        key_end = header.find(')', key_start)
                        key = header[key_start+1:key_end] if key_start != -1 and key_end != -1 else ''
                        
                        # Split key into 3 values (format: start~ssc~end)
                        key_parts = key.split('~') if key else ['', '', '']
                        tr_start = key_parts[0] if len(key_parts) > 0 else ''
                        ssc = key_parts[1] if len(key_parts) > 1 else ''
                        tr_end = key_parts[2] if len(key_parts) > 2 else ''
                        
                        # Extract 4 numbers
                        try:
                            tis_location = int(values[0])
                            tts_location = int(values[1])
                            tis_score = float(values[2])
                            tts_score = float(values[3])
                            
                            results.append({
                                'Chr': chr_part,
                                'Strand': strand,
                                'TrStart': tr_start,
                                'TrEnd': tr_end,
                                'SSC': ssc,
                                'TIS_related_location': tis_location,
                                'TTS_related_location': tts_location,
                                'TIS_score': tis_score,
                                'TTS_score': tts_score
                            })
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing line: {line}, error: {e}")
                            continue
        
        dtypes_df = {
            "Chr": "category",
            "Strand": "category",
            "TrStart": "int32",
            "TrEnd": "int32",
            "SSC": "string",       # pandas ≥1.5 recommends using string
        }
        # Create DataFrame and return
        if results:
            meriged_translationai_res = pd.DataFrame(results)

            # LPT scheduling: longest (chr,strand) groups first to minimise straggler wait.
            # Secondary key str(name) makes order independent of pandas groupby internal order.
            df_groups = sorted(
                [(name, g) for name, g in df.groupby(['Chr','Strand'], observed=True)],
                key=lambda ng: (-len(ng[1]), str(ng[0])),
            )
            df_groups = [g for _, g in df_groups]

            merged_df_list = []
            for df_group in df_groups:
                Chrom = df_group['Chr'].unique()[0]
                Strand = df_group['Strand'].unique()[0]

                tss_col = 'TrStart' if Strand == '+' else 'TrEnd'

                # F-008 fix: drop TranslationAI result columns from df_group
                # BEFORE merge. df_group enters this stage with TIS/TTS
                # columns pre-populated with the 'no' sentinel (see the
                # default-fill loop at the end of this function). If left
                # in place, the merge on Chr/Strand/TrStart/SSC/TrEnd would
                # produce pandas _x/_y suffix columns for TIS/TTS and the
                # downstream `if col not in df.columns` check would then
                # create a fresh 'no'-filled TIS_related_location column,
                # silently discarding every TranslationAI prediction.
                # Dropping here lets translationai_subset's values land in
                # clean column names so check_nmd at the end sees real TIS/TTS.
                _tai_result_cols = [
                    'TIS_related_location', 'TTS_related_location',
                    'TIS_score', 'TTS_score',
                ]
                df_group = df_group.drop(
                    columns=[c for c in _tai_result_cols if c in df_group.columns]
                )

                translationai_subset = meriged_translationai_res[
                    (meriged_translationai_res['Chr'] == Chrom) &
                    (meriged_translationai_res['Strand'] == Strand)
                ]

                translationai_subset = translationai_subset.astype(dtypes_df)

                # TranslationAI can return multiple predicted ORFs (different
                # TIS/TTS positions) for the same physical transcript model
                # (TrStart, SSC, TrEnd). With how='outer' the merge would
                # duplicate the transcript row per predicted ORF, inflating
                # the dataframe (Case 1 +1 vs Case 3 in the chr1 factorial
                # run showed 2408 phantom rows, 1591 of them truncation=yes).
                # Fix: keep only the highest-TIS_score ORF per physical model,
                # then how='left' so annotation NEVER adds new transcript rows.
                if "TIS_score" in translationai_subset.columns:
                    translationai_subset = translationai_subset.sort_values(
                        by="TIS_score", ascending=False
                    ).drop_duplicates(
                        subset=["Chr", "Strand", "TrStart", "SSC", "TrEnd"],
                        keep="first",
                    )
                df_merged = df_group.merge(
                    translationai_subset,
                    on=["Chr", "Strand", "TrStart", "SSC", "TrEnd"],
                    how='left'
                )
                merged_df_list.append(df_merged)

            if merged_df_list:
                df = pd.concat(merged_df_list, ignore_index=True)
        else:
            meriged_translationai_res = pd.DataFrame()
            # LPT scheduling: longest (chr,strand) groups first to minimise straggler wait.
            # Secondary key str(name) makes order independent of pandas groupby internal order.
            df_groups = sorted(
                [(name, g) for name, g in df.groupby(['Chr','Strand'], observed=True)],
                key=lambda ng: (-len(ng[1]), str(ng[0])),
            )
            df_groups = [g for _, g in df_groups]

            merged_df_list = []
            for df_group in df_groups:
                merged_df_list.append(df_group)

            if merged_df_list:
                df = pd.concat(merged_df_list, ignore_index=True)

        # Check if the required columns exist in the DataFrame before trying to fill them
        required_columns = ['TIS_related_location', 'TTS_related_location', 'TIS_score', 'TTS_score']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 'no'
        
        # Now safely fill NaN values for existing columns
        existing_required_columns = [col for col in required_columns if col in df.columns]
        if existing_required_columns:
            df[existing_required_columns] = df[existing_required_columns].fillna('no')
        
        def to_int_or_no(x):
            if pd.isna(x) or str(x).strip() == 'no':
                return 'no'
            try:
                return int(float(x))
            except (ValueError, TypeError):
                return 'no'

        cols = ['TIS_related_location', 'TTS_related_location']
        
        # Only process columns that exist in the DataFrame
        existing_cols = [col for col in cols if col in df.columns]
        for col in existing_cols:
            df[col] = df[col].apply(to_int_or_no)

        if existing_cols:
            df[existing_cols] = df[existing_cols].astype('object')

        df = TranslationAI_ORF.check_nmd(df, self.translationai_score_threshold)
        
        # Delete generated temp folder
        import shutil
        if os.path.exists(translationai_out_path):
            shutil.rmtree(translationai_out_path, ignore_errors=True)
        
        return df
