#!/usr/bin/env python

import os
import logging
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool
from typing import Optional, Dict, List, Union, Tuple, Any
from collections import defaultdict
import glob
import polars as pl

logger = logging.getLogger(__name__)

from .common import read_flnc
from .gene_grouping import GeneClustering
from .isoform_quantification import IsoformQuantifier

# Low-level DataFrame / column utilities (Group A)
from .aidrs_runtime import column_standardize
# Mid-level consolidation helpers (Group B)
from .aidrs_runtime import transcript_consolidate
# High-level report writers (Group C)
from .aidrs_runtime import report_writers

# P0-C: column registry in aidrs_runtime.column_registry centralizes the
# schema so future stages opt-in new columns without touching
# generate_reports.py. Order preserves byte-identical output for
# legacy runs (17-col C107 100k baseline SHA a8469106...).
from .aidrs_runtime.column_registry import resolve_assessment_columns


class IsoformAnnotator:
    """
    1. IsoformAnnotator(num_processes=20, reference)
    2. save_results(df_result, output_dir, ref_anno=None)
    """
    def __init__(self, num_processes: int = 20, terminal_tolerance: int = 50):
        self.num_processes = num_processes
        self.terminal_tolerance = terminal_tolerance

    # ------------------------------------------------------------------
    # 1. Unique external entry point (keeping signature unchanged)
    # ------------------------------------------------------------------
    def save_results(self,
                     df_result: pd.DataFrame,
                     output_dir: str,
                     reference: str,
                     ref_anno: Optional[pd.DataFrame] = None,
                     args: Optional[Any] = None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        # ---- 1. Annotation ----
        df_result.to_csv(os.path.join(output_dir,
                                         'temp/aidrs.transcript.result_df.tsv'),
                            sep='\t', index=False)
        annotated_df = self.annotate(df_result, ref_anno, self.num_processes)
        annotated_df.to_csv(os.path.join(output_dir,
                                         'temp/aidrs.transcript.annotated_df.before_quantification.tsv'),
                            sep='\t', index=False)

        # ---- 2. Quantification ----
        # Extract sample names from args.bam
        sample_names = []
        if args is not None and hasattr(args, 'bam'):
            # Extract sample names using the same method as in isoform_assembling function
            sample_names = [os.path.splitext(os.path.basename(bam))[0] for bam in args.bam]

        # Get quantification parameters from args
        min_samples_expr = getattr(args, 'min_expressed_samples', 1) if args is not None else 1

        # Create quantifier with options
        quantifier = IsoformQuantifier(
            num_processes=self.num_processes,
            min_expressed_samples=min_samples_expr
        )

        # Only proceed with quantification if we have sample names
        quantification_success = False
        if sample_names:
            try:
                # Quantify all samples
                quantification_matrices = quantifier.quantify_all_samples(
                    sample_names,
                    annotated_df,
                    output_dir
                )

                # Intersect matrices with model
                # intersect_matrices_with_model returns 3-tuple:
                #   (filtered_matrices, filtered_transcript_model, filtered_quant_df).
                # The caller previously unpacked only 2 values → ValueError caught
                # silently by the outer `except Exception` → count/cpm matrices never
                # written.
                (quantification_matrices,
                 filtered_annotated_df,
                 _filtered_quant_df) = quantifier.intersect_matrices_with_model(
                    quantification_matrices, annotated_df
                )

                # Save quantification matrices (consistent dotted naming:
                # aidrs.{name}.tsv — matches aidrs.transcript.assessment.tsv convention).
                for name, matrix in quantification_matrices.items():
                    output_path = os.path.join(output_dir, f'aidrs.{name}.tsv')
                    matrix.to_csv(output_path, sep='\t')

                # Update annotated_df with the filtered version
                annotated_df = filtered_annotated_df
                quantification_success = True
            except Exception as e:
                print(f"Warning: Quantification failed: {e}")

        annotated_df.to_csv(os.path.join(output_dir,
                                         'temp/aidrs.transcript.annotated_df.tsv'),
                            sep='\t', index=False)
        # Merge df_result and annotated_df, keeping only one copy of duplicate columns
        # Common columns between df_result and annotated_df
        common_cols = ['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd']
        # Columns unique to annotated_df that we want to add
        annotated_cols = ['TrID', 'GeneID', 'GeneName']

        # Merge the dataframes on common columns
        df_result_after_quant = df_result.merge(
            annotated_df[common_cols + annotated_cols],
            on=common_cols,
            how='inner'
        )
        df_result_after_quant.to_csv(os.path.join(output_dir,
                                         'temp/aidrs.transcript.assessment.tsv'),
                            sep='\t', index=False)
        # ---- 3. Output GTF ----
        self.to_gtf(annotated_df, output_dir)
        # ---- 4. Output FASTA ----
        df_result_after_quant = self.to_fasta(os.path.join(output_dir, 'aidrs.transcript_model.gtf'), reference, output_dir, df_result_after_quant)

        # Expand df_result_after_quant with a 'sample' column
        # For each sample, duplicate df_result_after_quant and add the sample name as a column
        if sample_names:
            # Create a list to hold dataframes for each sample
            sample_dfs = []
            for sample in sample_names:
                # Copy df_result_after_quant and add sample column
                sample_df = df_result_after_quant.copy()
                sample_df['sample'] = sample
                sample_dfs.append(sample_df)

            # Concatenate all sample dataframes
            if sample_dfs:
                df_result_after_quant = pd.concat(sample_dfs, ignore_index=True)
            else:
                # If no sample dataframes were created, add an empty sample column
                df_result_after_quant['sample'] = ''

        df_result_after_quant['sites'] = df_result_after_quant.apply(
            lambda r: sorted(
                list(map(int, r['SSC'].split('-'))) +
                [int(r['TrStart']), int(r['TrEnd'])]
            ), axis=1
        )

        df_result_after_quant, polyA_tables = self.polyA_len_profile(df_result_after_quant, output_dir)
        # polyA sidecars: written as Parquet so raw_polyA_lengths can be a
        # native Arrow List<Float32> column (1NF-safe, single-cell-per-row).
        # TSV would force comma-separated strings inside a single cell,
        # violating 1NF and blocking downstream SQL/polars aggregation.
        for name, table in polyA_tables.items():
            out_path = os.path.join(output_dir, f'aidrs.{name}.parquet')
            try:
                table.to_parquet(out_path, index=False)
                logger.info(f"Wrote polyA sidecar: {out_path} "
                            f"({len(table)} rows, raw_polyA_lengths as List<Float32>)")
            except Exception as e:
                logger.error(f"[CRITICAL FAIL-LOUD] Failed to write {out_path}: "
                             f"{type(e).__name__}: {e}", exc_info=True)
                raise
        # ---- 6. Original assessment table ----
        # P0-C: column registry in aidrs_runtime.column_registry centralizes
        # the schema so future stages opt-in new columns without touching
        # generate_reports.py. Order preserves byte-identical output for
        # legacy runs (17-col C107 100k baseline SHA a8469106...).
        _ASSESS_COLS = resolve_assessment_columns(df_result_after_quant)
        df_result_after_quant[_ASSESS_COLS].drop_duplicates(
            subset=["Chr", "Strand", "TrStart", "TrEnd", "SSC"]
        ).to_csv(os.path.join(output_dir,
                      'aidrs.transcript.assessment.tsv'), sep='\t', index=False)


    # ------------------------------------------------------------------
    # 2.1 Main Annotation Logic
    # ------------------------------------------------------------------

    def annotate(self,
                  df_result: pd.DataFrame,
                  ref_anno: Optional[pd.DataFrame],
                  num_processes: int) -> pd.DataFrame:
        # 3.1 Generate uniqueTr
        df = df_result.copy()
        df['uniqueTr'] = 'Tr' + df.groupby(
            ['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd'],
            observed=True
        ).ngroup().astype(str)
        # P1-3 fix: dedup on the physical-coordinate subset that uniquely
        # determines uniqueTr (groupby ngroup above). The other columns
        # (frequency, TIS_*, TTS_*, Predict_NMD) may legitimately differ for
        # the same physical transcript; full-row dedup would silently hide
        # annotation disagreement.
        df_unique = df[['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd', 'frequency', 'uniqueTr', 'TIS_related_location', 'TTS_related_location', 'Predict_NMD']].drop_duplicates(
            subset=['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd']
        )
        # 3.2 Cluster to get Group
        gene_clustering = GeneClustering(num_processes=num_processes)
        df_unique = gene_clustering.cluster(df_unique)

        # 3.3 Merge with reference annotation
        if ref_anno is not None:
            ref_anno_model = ref_anno[
                ref_anno['SSC'].isin(df_unique['SSC'].unique())
            ].copy()
            merged = df_unique.merge(
                ref_anno_model,
                on=['Chr', 'Strand', 'SSC'],
                how='left',
                suffixes=('', '_ref')
            )
        else:
            merged = df_unique.copy()

        # 3.4 Run concurrently by Group
        df_groups = [g for _, g in merged.groupby('Group', observed=True)]
        with Pool(num_processes) as pool:
            # Use static method instead of instance method to solve multiprocessing serialization issues
            func = partial(IsoformAnnotator._annotate_one_group, ref_anno=ref_anno, terminal_tolerance=self.terminal_tolerance)
            results = pool.map(func, df_groups)
        # Handle case where results is empty

        if not results:
            # Create empty dataframe with expected columns
            empty_df = pd.DataFrame(columns=['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd', 'frequency', 'uniqueTr', 'TIS_related_location', 'TTS_related_location', 'Predict_NMD', 'Group', 'TrID', 'GeneID', 'GeneName', 'TrStart_ref', 'TrEnd_ref'])
            return empty_df
        final_df = pd.concat(results, ignore_index=True)

        # 3.5 Deduplicate key
        def merge_fusion_genes(df):
            df['tr_key'] = (
                df['Chr'].astype(str) + df['Strand'].astype(str) + ':' +
                df['TrStart'].astype(str) + '-' + df['SSC'].astype(str) + '-' +
                df['TrEnd'].astype(str)
            )
            agg_rules = {}
            for col in df.columns:
                if col == 'tr_key': continue
                if col in ['GeneID', 'GeneName']:
                    agg_rules[col] = lambda x: '-'.join(x.astype(str).unique())
                else:
                    agg_rules[col] = 'first'
            return df.groupby('tr_key', as_index=False).agg(agg_rules).drop(columns=['tr_key'])

        final_df = merge_fusion_genes(final_df) # Gene fusion

        final_df['key'] = final_df['TrID']
        cnt = final_df.groupby('key').cumcount().add(1).astype(str)
        final_df['TrID'] = np.where(
            cnt != '1',
            final_df['TrID'] + '_' + cnt,
            final_df['TrID']
        ) # Multiple terminal isoforms
        final_df = final_df.drop(columns=['key'])

        if ref_anno is not None:
            return self._novel_gene_remapping(final_df, ref_anno)
        else:
            return final_df

    def annotate_one_group(self,
                           df_group: pd.DataFrame,
                           ref_anno: Optional[pd.DataFrame]) -> pd.DataFrame:
        """With provided pure function logic completely consistent, only indentation level changes"""
        if ref_anno is not None:
            # Split data into query (novel) and reference parts
            query_df = df_group[df_group.isna().any(axis=1)].copy()
            ref_df = df_group[~df_group.isna().any(axis=1)].copy()

            # Process reference data if exists
            if not ref_df.empty and not (ref_df.shape[0] == len(ref_df.uniqueTr.unique()) == len(ref_df.TrID.unique())):
                ref_df = self._map_transcript_1to1(ref_df)  # Solve 1 uniqueTr vs. multiple TrIDs and multiple uniqueTrs vs. 1 TrID

            # Process both query and reference data
            if not query_df.empty and not ref_df.empty:
                # Update reference data with TSS/TES flags
                self._update_ref_with_flags(ref_df)
                # Build reference dictionary and map query to reference
                ref_dict = self._build_ref_dict(ref_df)
                query_df = self._map_query_to_ref(query_df, ref_dict)
                result_df = pd.concat([ref_df, query_df], ignore_index=True)
                return result_df.drop_duplicates()
            elif ref_df.empty:   # All novel
                return self._fill_novel(df_group).drop_duplicates()
            else:                # All reference
                # Update reference data with TSS/TES flags
                self._update_ref_with_flags(ref_df)
                return ref_df.drop_duplicates()
        else:
            return self._fill_novel(df_group).drop_duplicates()

    @staticmethod
    def _annotate_one_group(
        df_group: pd.DataFrame,
        ref_anno: Optional[pd.DataFrame],
        terminal_tolerance: int
    ) -> pd.DataFrame:
        """Static method version of annotate_one_group, used for multiprocessing"""
        # Create temporary instance to reuse existing methods
        temp_instance = IsoformAnnotator(terminal_tolerance=terminal_tolerance)
        return temp_instance.annotate_one_group(df_group, ref_anno)

    # ------------------------------------------------------------------
    # 2.2 Annotation Helper Functions (thin shims delegating to modules)
    # ------------------------------------------------------------------
    # These methods were extracted to column_standardize.py and
    # transcript_consolidate.py. The shims preserve the existing class
    # API so aidrs.py and the characterization tests keep working.

    def _update_ref_with_flags(self, ref_df: pd.DataFrame) -> None:
        """Thin shim -> transcript_consolidate.update_ref_with_flags."""
        return transcript_consolidate.update_ref_with_flags(
            ref_df, self.terminal_tolerance
        )

    def _transcript_1to1_processor(self, uni_tr_mappings):
        """Thin shim -> transcript_consolidate.transcript_1to1_processor."""
        return transcript_consolidate.transcript_1to1_processor(
            uni_tr_mappings, self.terminal_tolerance
        )

    def _map_transcript_1to1(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thin shim -> transcript_consolidate.map_transcript_1to1."""
        return transcript_consolidate.map_transcript_1to1(
            df, self.terminal_tolerance
        )

    def _fill_novel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thin shim -> column_standardize.fill_novel."""
        return column_standardize.fill_novel(df)

    def _build_ref_dict(self, ref_df: pd.DataFrame, include_term: bool = True) -> dict:
        """Thin shim -> column_standardize.build_ref_dict."""
        return column_standardize.build_ref_dict(ref_df, include_term=include_term)

    def _map_query_to_ref(self, query_df: pd.DataFrame, ref_dict: dict) -> pd.DataFrame:
        """Thin shim -> transcript_consolidate.map_query_to_ref."""
        return transcript_consolidate.map_query_to_ref(query_df, ref_dict)

    def _novel_gene_remapping(self, df_result: pd.DataFrame,
                               ref_anno: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Thin shim -> transcript_consolidate.novel_gene_remapping."""
        return transcript_consolidate.novel_gene_remapping(
            df_result, ref_anno, self.terminal_tolerance
        )

    # ------------------------------------------------------------------
    # 3. GTF & FASTA Generation (thin shims delegating to report_writers)
    # ------------------------------------------------------------------
    # These methods were extracted to report_writers.py. The shims
    # preserve the existing class API so aidrs.py and the
    # characterization tests keep working.

    def to_gtf(self, df: pd.DataFrame, output_dir: str) -> None:
        """Thin shim -> report_writers.to_gtf."""
        return report_writers.to_gtf(df, output_dir)

    def to_fasta(self, gtf_file: str, genome_fasta: str, output_dir: str,
                  df_result_after_quant: Optional[pd.DataFrame] = None
) -> Optional[pd.DataFrame]:
        """Thin shim -> report_writers.to_fasta."""
        return report_writers.to_fasta(
            gtf_file, genome_fasta, output_dir, df_result_after_quant
        )

    def _reverse_complement(self, seq: str) -> str:
        """Thin shim -> column_standardize.reverse_complement."""
        return column_standardize.reverse_complement(seq)


    # ------------------------------------------------------------------
    # 4. Quantification and PolyA Profiling
    # ------------------------------------------------------------------
    # polyA_len_profile is intentionally kept in this module because it
    # is tightly coupled to filesystem layout (reads temp/*_flnc_correct.ssc
    # written by earlier stages) and to polars, neither of which is part
    # of the column_standardize / transcript_consolidate / report_writers
    # responsibility split.

    # def quantify(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    #     df['length'] = df['sites'].apply(
    #         lambda s: sum(s[i+1]-s[i] for i in range(0, len(s), 2))
    #     )
    #     df['rpk'] = df['quantification'] / (df['length'] / 1000)
    #     tpm_scale = 1e6 / df['rpk'].sum()
    #     df['TPM'] = (df['rpk'] * tpm_scale).round(2)
    #     # Transcript weights
    #     df['weight'] = df['length'] / df.groupby(['GeneID', 'sample'])['length'].transform('sum')
    #     df['weighted_tpm'] = df['TPM'] * df['weight']  # weighted TPM for genes
    #     return {
    #         "transcript_counts": df.pivot_table(
    #             index=['TrID', 'GeneID', 'GeneName'],
    #             columns='sample',
    #             values='quantification',
    #             fill_value=0
    #         ),
    #         "transcript_tpm": df.pivot_table(
    #             index=['TrID', 'GeneID', 'GeneName'],
    #             columns='sample',
    #             values='TPM',
    #             fill_value=0
    #         ),
    #         "gene_counts": df.groupby(['GeneID', 'GeneName', 'sample'],
    #                                  observed=True)['quantification'].sum().unstack(fill_value=0),
    #         "gene_tpm": df.groupby(['GeneID', 'GeneName', 'sample'],
    #                               observed=True)['weighted_tpm'].sum().unstack(fill_value=0)
    #     }
    # quantify method has been moved to the IsoformQuantifier class
    # def quantify(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    #     # Calculate transcript length from exon sites (sum of exon lengths)
    #     df['length'] = df['sites'].apply(
    #         lambda s: sum(s[i+1] - s[i] for i in range(0, len(s), 2))
    #     )
    #
    #     transcript_tpm_list = []
    #     gene_tpm_list = []
    #     transcript_cpm_list = []
    #     gene_cpm_list = []
    #
    #     # Process each sample independently
    #     for sample, group in df.groupby('sample'):
    #         group = group.copy()
    #
    #         # --- Transcript-level metrics ---
    #         # 1. Calculate transcript CPM (normalize by total reads per sample)
    #         total_reads = group['quantification'].sum()  # Total mapped reads for sample
    #         group['CPM'] = (group['quantification'] / total_reads * 1e6).round(2)
    #
    #         # 2. Calculate transcript TPM (normalize by total RPK per sample)
    #         group['rpk'] = group['quantification'] / (group['length'] / 1000)
    #         total_rpk = group['rpk'].sum()
    #         group['TPM'] = (group['rpk'] / total_rpk * 1e6).round(2)
    #         transcript_tpm_list.append(group)
    #         transcript_cpm_list.append(group[['TrID', 'GeneID', 'GeneName', 'CPM', 'sample']])
    #
    #         # --- Gene-level metrics (recalculate from raw counts) ---
    #         # Aggregate transcript counts to gene level
    #         gene_agg = group.groupby(['GeneID', 'GeneName'], observed=True).agg(
    #             gene_count=('quantification', 'sum'),  # Sum of transcript counts
    #             gene_length=('length', lambda x: np.sum(x * group.loc[x.index, 'quantification']) / x.sum())  # Expression-weighted length
    #         ).reset_index()
    #
    #         # Calculate gene CPM (using total sample reads)
    #         gene_agg['gene_CPM'] = (gene_agg['gene_count'] / total_reads * 1e6).round(2)
    #
    #         # Calculate gene TPM (using gene-level RPK)
    #         gene_agg['gene_rpk'] = gene_agg['gene_count'] / (gene_agg['gene_length'] / 1000)
    #         total_gene_rpk = gene_agg['gene_rpk'].sum()  # Should equal total_rpk
    #         gene_agg['gene_TPM'] = (gene_agg['gene_rpk'] / total_gene_rpk * 1e6).round(2)
    #         gene_agg['sample'] = sample
    #         gene_tpm_list.append(gene_agg)
    #         gene_cpm_list.append(gene_agg[['GeneID', 'GeneName', 'gene_CPM', 'sample']])
    #
    #     # Merge results
    #     df_transcript = pd.concat(transcript_tpm_list, ignore_index=True)
    #     df_gene = pd.concat(gene_tpm_list, ignore_index=True)
    #     df_transcript_cpm = pd.concat(transcript_cpm_list, ignore_index=True)
    #     df_gene_cpm = pd.concat(gene_cpm_list, ignore_index=True)
    #
    #     return {
    #         # Transcript counts
    #         "transcript_counts": df.pivot_table(
    #             index=['TrID', 'GeneID', 'GeneName'],
    #             columns='sample',
    #             values='quantification',
    #             fill_value=0
    #         ),
    #
    #         # Transcript TPM
    #         "transcript_tpm": df_transcript.pivot_table(
    #             index=['TrID', 'GeneID', 'GeneName'],
    #             columns='sample',
    #             values='TPM',
    #             fill_value=0
    #         ),
    #
    #         # Transcript CPM
    #         "transcript_cpm": df_transcript_cpm.pivot_table(
    #             index=['TrID', 'GeneID', 'GeneName'],
    #             columns='sample',
    #             values='CPM',
    #             fill_value=0
    #         ),
    #
    #         # Gene counts
    #         "gene_counts": df_gene.pivot_table(
    #             index=['GeneID', 'GeneName'],
    #             columns='sample',
    #             values='gene_count',
    #             fill_value=0
    #         ),
    #
    #         # Gene TPM
    #         "gene_tpm": df_gene.pivot_table(
    #             index=['GeneID', 'GeneName'],
    #             columns='sample',
    #             values='gene_TPM',
    #             fill_value=0
    #         ),
    #
    #         # Gene CPM
    #         "gene_cpm": df_gene_cpm.pivot_table(
    #             index=['GeneID', 'GeneName'],
    #             columns='sample',
    #             values='gene_CPM',
    #             fill_value=0
    #         )
    #     }

    def polyA_len_profile(self, df: pd.DataFrame, out_dir) -> Dict[str, pd.DataFrame]:
        """Compute polyA tail-length statistics from flnc_correct.ssc reads.

        Reads each *_flnc_correct.ssc, aggregates per (TrID, GeneID, GeneName)
        and per GeneID with **median** as the gate statistic (mean as
        reference; count as data-quality flag). The full per-isoform raw
        tail-length distribution is preserved as a list column for
        downstream single-molecule analysis.

        Returns:
            (df, dict) where dict keys are:
              - 'transcript_polyA_len'  → collapsed-across-samples DataFrame
                                         (TrID, GeneID, GeneName,
                                          polyA_median, polyA_mean,
                                          polyA_count, raw_polyA_lengths)
              - 'gene_polyA_len'        → per-gene DataFrame
                                         (GeneID, polyA_median, polyA_mean,
                                          polyA_count, raw_polyA_lengths)

        Both outputs are saved as Parquet by the writer loop in
        save_results so that raw_polyA_lengths can be a native
        Arrow List<Float32>. Empty group → polyA_median = NaN
        (NOT 0; 0 means biologically "dead RNA tail").

        When polyA was auto-skipped at Stage 2.3 (no 'pt' tag in any input
        BAM), df['polyA_frac'] is all-NaN. Returning an empty dict here
        causes the writer loop in save_results to skip parquet creation
        entirely — no aidrs.{transcript,gene}_polyA_len.parquet is emitted.
        """
        # Auto-detected polyA-skip path: bail before scanning flnc_correct.ssc.
        # The flnc_correct files exist (correct_flnc_only wrote them), but every
        # row's polyA_len is 0 from the missing 'pt' tag, so aggregating would
        # produce all-zero tables that are misleading rather than informative.
        if 'polyA_frac' not in df.columns or df['polyA_frac'].isna().all():
            logger.warning(
                "[POLYA-AUTO-SKIP] No polyA_frac data on df; skipping "
                "aidrs.{transcript,gene}_polyA_len.parquet writes."
            )
            return df, {}

        pattern = os.path.join(out_dir, 'temp', '*_flnc_correct.ssc')
        files = glob.glob(pattern)
        if not files:
            logger.warning("polyA_len_profile: no *_flnc_correct.ssc found; "
                           "emitting empty polyA tables.")
            empty_tr = pd.DataFrame(
                columns=['TrID', 'GeneID', 'GeneName',
                         'polyA_median', 'polyA_mean', 'polyA_count']
            )
            empty_gn = pd.DataFrame(
                columns=['GeneID', 'polyA_median', 'polyA_mean', 'polyA_count']
            )
            return df, {
                "transcript_polyA_len": empty_tr,
                "gene_polyA_len": empty_gn,
            }
        reads = []
        for f in files:
            read_df = read_flnc(f)
            if 'TrStart_reads' in read_df.columns:
                read_df = read_df.rename(columns={'TrStart_reads': 'TrStart'})
            if 'TrEnd_reads' in read_df.columns:
                read_df = read_df.rename(columns={'TrEnd_reads': 'TrEnd'})
            reads.append(read_df)
        all_reads = pl.concat([pl.from_pandas(r) for r in reads], how='vertical_relaxed')
        # Collapse across samples: aggregate per (Chr,Strand,SSC,TrStart,TrEnd)
        # joining against the model to recover TrID/GeneID/GeneName. We keep
        # the polyA_len values as a list per isoform for downstream use.
        per_iso = (
            all_reads
            .filter(pl.col("polyA_len") > 0)
            .group_by(['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd'])
            .agg([
                pl.col("polyA_len").median().round(1).alias("polyA_median_raw"),
                pl.col("polyA_len").mean().round(1).alias("polyA_mean_raw"),
                pl.col("polyA_len").count().alias("polyA_count_raw"),
            ])
        ).to_pandas()
        # Join against model for TrID/GeneID/GeneName
        model_keys = df[['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd',
                         'TrID', 'GeneID', 'GeneName']].drop_duplicates()
        joined = model_keys.merge(
            per_iso, on=['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd'], how='left'
        )
        # Now aggregate per TrID (each TrID has unique key) — keep raw list.
        # Collect all reads (per read row, not per group already aggregated).
        # Cast Chr/Strand/SSC to Utf8 on both sides — read_flnc yields category
        # dtype for those columns, and Polars join refuses cat-vs-str mismatches
        # Polars join refuses cat-vs-str mismatches.
        reads_per_iso = (
            all_reads
            .with_columns([
                pl.col("Chr").cast(pl.Utf8),
                pl.col("Strand").cast(pl.Utf8),
                pl.col("SSC").cast(pl.Utf8),
            ])
            .filter(pl.col("polyA_len") > 0)
            .join(
                pl.from_pandas(model_keys[['Chr', 'Strand', 'SSC', 'TrStart',
                                            'TrEnd', 'TrID', 'GeneID', 'GeneName']])
                .with_columns([
                    pl.col("Chr").cast(pl.Utf8),
                    pl.col("Strand").cast(pl.Utf8),
                    pl.col("SSC").cast(pl.Utf8),
                ]),
                on=['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd'],
                how='inner',
            )
        )
        # Build per-TrID raw lists (preserving every read's tail length).
        polyA_tr = (
            reads_per_iso
            .group_by(['TrID', 'GeneID', 'GeneName'])
            .agg([
                pl.col("polyA_len").median().round(1).alias("polyA_median"),
                pl.col("polyA_len").mean().round(1).alias("polyA_mean"),
                pl.col("polyA_len").count().alias("polyA_count"),
                pl.col("polyA_len").round(1).alias("raw_polyA_lengths"),
            ])
            .sort(['GeneID', 'TrID'])
            .to_pandas()
        )
        polyA_gn = (
            reads_per_iso
            .group_by(['GeneID'])
            .agg([
                pl.col("polyA_len").median().round(1).alias("polyA_median"),
                pl.col("polyA_len").mean().round(1).alias("polyA_mean"),
                pl.col("polyA_len").count().alias("polyA_count"),
                pl.col("polyA_len").round(1).alias("raw_polyA_lengths"),
            ])
            .sort(['GeneID'])
            .to_pandas()
        )
        return df, {
            "transcript_polyA_len": polyA_tr,
            "gene_polyA_len": polyA_gn,
        }