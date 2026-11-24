import numpy as np
import pandas as pd
import os, re
import logging
import polars as pl
from .common import correct_flnc, read_flnc

logger = logging.getLogger("AIDRS")

# The keys a poly(A) statistic is defined on.
POLYA_MERGE_COLS = ['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd']

# A sample contributes to the per-key median only if it has at least
# POLYA_MIN_READS reads at that key AND at least one read with a measured
# poly(A) tail. The read-count floor drops noise-level samples; the
# "measured tail" condition drops samples that carry NO poly(A) evidence
# (a pt-less or tail-less library), which would otherwise drag the cohort
# median down (a cohort of one pt-bearing and two pt-less samples would
# otherwise take the median of [r1, 0, 0] = 0 and collapse the transcript
# to State 2).
POLYA_MIN_READS = 5

# Long-format per-sample audit trail. Written next to the delivered outputs
# (NOT under temp/, which is deleted at the end of a normal run) and named so
# that it can never be picked up by the 'temp/*_flnc_correct.ssc' glob that
# generate_reports.polyA_len_profile scans.
POLYA_PER_SAMPLE_SIDECAR = 'aidrs.transcript.polya_per_sample.tsv'


def sample_id_from_flnc(flnc_path):
    """Recover the sample id from a '{sample}_flnc.ssc' path."""
    return re.sub(r'_flnc\.ssc$', '', os.path.basename(flnc_path))


def per_sample_polya_stats(read_df: pd.DataFrame, sample: str) -> pl.DataFrame:
    """Per-key poly(A) statistics for ONE sample.

    One row per (Chr, Strand, SSC, TrStart, TrEnd) with
        polyA_freq   = number of reads with a MEASURED poly(A) tail (pt > 0)
        raw_read_freq = number of reads at that key
    """
    return (
        pl.from_pandas(read_df)
        .with_columns(
            # Only a strictly positive length is a MEASURED poly(A) tail.
            # The Dorado sentinels (-1 anchor not found, -2 tag missing)
            # and the inestimable case (0) are all "no measurement".
            polyA_freq=(pl.col("polyA_len") > 0).cast(int),
            Chr=pl.col("Chr").cast(str),
            Strand=pl.col("Strand").cast(str),
            SSC=pl.col("SSC").cast(str),
        )
        .group_by(POLYA_MERGE_COLS)
        .agg(
            polyA_freq=pl.col("polyA_freq").sum(),
            raw_read_freq=pl.col("polyA_freq").count(),
        )
        .with_columns(pl.lit(sample).alias("sample"))
    )


def aggregate_polya(per_sample: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-sample poly(A) statistics to ONE row per key.

    ``polyA_frac`` is the UNWEIGHTED MEDIAN of the per-sample ratios of the
    qualifying samples, i.e. the samples with
        raw_read_freq >= POLYA_MIN_READS  AND  polyA_freq >= 1
    A read-depth-weighted pooled ratio would let one abnormally deep library
    dominate the cohort's poly(A) evidence.

    Median convention (numpy/pandas): with an EVEN number of candidates the
    median is the MEAN of the two middle values.

    When no sample qualifies, ``polyA_frac`` falls back to today's pooled
    ratio sum(polyA_freq) / sum(raw_read_freq) (which also preserves today's
    State-2 semantics: a measured zero stays 0).

    The pooled value is kept as ``polyA_frac_pooled`` and the number of
    contributing samples as ``polyA_n_samples`` for auditing. Neither is a
    registered assessment column, so the delivered schema does not change.
    """
    ps = per_sample.copy()
    ps['polyA_frac_sample'] = ps['polyA_freq'] / ps['raw_read_freq']
    ps['_is_candidate'] = (ps['raw_read_freq'] >= POLYA_MIN_READS) & (ps['polyA_freq'] >= 1)

    pooled = (
        ps.groupby(POLYA_MERGE_COLS, as_index=False, observed=True)
        .agg(
            _polyA_sum=('polyA_freq', 'sum'),
            _raw_sum=('raw_read_freq', 'sum'),
        )
    )
    pooled['polyA_frac_pooled'] = pooled['_polyA_sum'] / pooled['_raw_sum']
    pooled = pooled[POLYA_MERGE_COLS + ['polyA_frac_pooled']]

    candidates = ps[ps['_is_candidate']]
    median = (
        candidates.groupby(POLYA_MERGE_COLS, as_index=False, observed=True)
        .agg(
            polyA_frac=('polyA_frac_sample', 'median'),
            polyA_n_samples=('polyA_frac_sample', 'size'),
        )
    )

    out = pooled.merge(median, on=POLYA_MERGE_COLS, how='left')
    # No qualifying sample => today's pooled ratio.
    out['polyA_frac'] = out['polyA_frac'].fillna(out['polyA_frac_pooled'])
    out['polyA_n_samples'] = out['polyA_n_samples'].fillna(0).astype(int)

    return out[POLYA_MERGE_COLS + ['polyA_frac', 'polyA_frac_pooled', 'polyA_n_samples']]


class polyAnnotator:
    def __init__(self, args = None):
        self.args = args

    @staticmethod
    def add_polya_frac(df: pd.DataFrame,
                       df_polya: pd.DataFrame,
                       merge_cols=None) -> pd.DataFrame:
        """
        Add the poly(A) columns to df by matching the rows of df_polya on
        Chr, Strand, SSC, TrStart, TrEnd.

        df_polya must hold ONE row per key (see aggregate_polya) with the
        columns polyA_frac, polyA_frac_pooled and polyA_n_samples.

        Parameters
        ----
        df : pd.DataFrame
            Table to which the poly(A) columns need to be added
        df_polya : pd.DataFrame
            Table containing the per-key poly(A) statistics
        merge_cols : list[str], optional
            List of column names for matching, default is
            ['Chr','Strand','SSC','TrStart','TrEnd']

        Returns
        ----
        pd.DataFrame
            df with added polyA_frac / polyA_frac_pooled / polyA_n_samples
            columns
        """

        if merge_cols is None:
            merge_cols = POLYA_MERGE_COLS

        n_input = len(df)
        out = df.merge(
            df_polya[merge_cols + ['polyA_frac', 'polyA_frac_pooled', 'polyA_n_samples']],
            on=merge_cols, how='left')

        # A merge that multiplies the model rows is a bug (a duplicated key
        # on the right-hand side), not a harmless no-op: assert it instead
        # of assuming it.
        if len(out) != n_input:
            raise RuntimeError(
                f"[POLYA-MERGE] poly(A) annotation changed the row count: "
                f"{n_input} model rows became {len(out)}. The per-key poly(A) "
                f"table must hold exactly one row per merge key."
            )

        # A key the per-key table does not cover carries NO poly(A) evidence.
        # Filling it with 0 turned "no measurement" into "measured zero" and
        # made the Stage 2.6 gate delete the model (0.73 % of the models in
        # the BC1-BC6 run). NaN is State 3 and takes the bypass.
        out['polyA_n_samples'] = out['polyA_n_samples'].fillna(0).astype(int)
        return out

    # def anno_polya(self, df, flnc):
    #     """Single sample polyA annotation (maintain backward compatibility)"""
    #     read_df = pd.read_csv(flnc, sep='\t', names=["Chr", "Strand", "TrStart", "TrEnd", "SSC", "identity", "coverage", "polyA"])
    #     df_polya = self.correct_flnc(df, read_df)
    #     df = self.add_polya_frac(df, df_polya)
    #     return df
    
    def _correct_flnc_for_files(self, df, flnc_files):
        """For each flnc_file, run correct_flnc and write {sample}_flnc_correct.ssc.

        The _flnc_correct.ssc files are required by isoform_quantification.py:39
        for transcript quantification (polyA-independent). They are also the
        input to polyA_len_profile in generate_reports.py:547.

        Returns a list of (flnc_file, read_df) for downstream consumers that
        want to do polyA aggregation. Used by both anno_polya (full path) and
        correct_flnc_only (skip-polyA path, called when bam2ssc reports no 'pt'
        tag in any input BAM).
        """
        results = []
        for flnc_file in flnc_files:
            if not os.path.exists(flnc_file):
                continue

            read_df = read_flnc(flnc_file)
            # Create a temporary args object with terminal_tolerance set to double the original value
            if self.args is not None and hasattr(self.args, 'terminal_tolerance'):
                # Create a copy of args and modify the terminal_tolerance value
                import argparse
                temp_args = argparse.Namespace(**vars(self.args))
                temp_args.terminal_tolerance = self.args.terminal_tolerance * 2
                read_df = correct_flnc(df, read_df, args=temp_args)
            else:
                # If args is not provided or does not have terminal_tolerance attribute, use default behavior
                read_df = correct_flnc(df, read_df, args=self.args)
            dir_name = os.path.dirname(flnc_file)                # Original directory
            base_name = os.path.basename(flnc_file)
            out_name = re.sub(r'_flnc\.ssc$', '_flnc_correct.ssc', base_name)
            out_file = os.path.join(dir_name, out_name)
            read_df.to_csv(out_file, sep='\t', index=True, header=False)

            results.append((flnc_file, read_df))
        return results

    def correct_flnc_only(self, df, flnc_files):
        """Run only the correct_flnc step + write _flnc_correct.ssc, no polyA aggregation.

        Called by aidrs.py when bam2ssc detects no 'pt' tag in any input BAM.
        Skipping polyA aggregation prevents all-zero polyA_frac from polluting
        the downstream Stage 2.6 3-state filter (which would otherwise classify
        every transcript as State 2 = "measured zero" instead of the correct
        State 3 = "polyA not measured"). df is returned unchanged; the caller
        is expected to add df['polyA_frac'] = NaN (State 3) itself.
        """
        self._correct_flnc_for_files(df, flnc_files)
        return df

    def _write_per_sample_sidecar(self, per_sample: pd.DataFrame,
                                  df_polya: pd.DataFrame = None):
        """Write the per-sample poly(A) statistics next to the outputs.

        The file is written to the output directory (NOT to temp/, which is
        deleted at the end of a normal run) and is named
        'aidrs.transcript.polya_per_sample.tsv', so it can never be picked up
        by the 'temp/*_flnc_correct.ssc' glob that
        generate_reports.polyA_len_profile scans.

        Long format: Chr, Strand, SSC, TrStart, TrEnd, sample, polyA_freq,
        raw_read_freq, polyA_frac (the per-sample ratio), plus the per-key
        aggregate that the model frame received:
        polyA_frac_model (the cohort value for that key), polyA_frac_pooled
        (the read-depth-weighted value), polyA_n_samples (the number of
        qualifying samples) and estimator ('median' or 'pooled_fallback').
        The per-key aggregate is carried on every row of the key so that the
        audit trail survives delivery: without it a consumer cannot tell
        whether the delivered polyA_frac came from the median or from the
        pooled fallback.

        Note that the key space of this file is the Stage 2.3 READ key space,
        which is a superset of the delivered model (a read whose ends were not
        snapped to a model keeps its own coordinates), and a model key with no
        reads at all has no row here.
        """
        output_dir = getattr(self.args, 'output', None) if self.args is not None else None
        if not output_dir:
            logger.warning(
                "[POLYA-SIDECAR] No output directory known; skipping the "
                f"{POLYA_PER_SAMPLE_SIDECAR} write."
            )
            return
        if per_sample is None or len(per_sample) == 0:
            logger.warning(
                "[POLYA-SIDECAR] No per-sample poly(A) statistics; skipping the "
                f"{POLYA_PER_SAMPLE_SIDECAR} write."
            )
            return

        out_path = os.path.join(output_dir, POLYA_PER_SAMPLE_SIDECAR)
        table = per_sample.copy()

        if df_polya is not None and len(df_polya) > 0:
            # Rename BEFORE the merge: the aggregate carries a column named
            # 'polyA_frac' (the model-level value) and the per-sample ratio is
            # computed below under the same name. Merging without renaming
            # would silently drop the aggregate.
            agg = df_polya[POLYA_MERGE_COLS + ['polyA_frac', 'polyA_frac_pooled',
                                               'polyA_n_samples']].copy()
            agg = agg.rename(columns={'polyA_frac': 'polyA_frac_model'})
            table = table.merge(agg, on=POLYA_MERGE_COLS, how='left',
                                validate='many_to_one')
            if len(table) != len(per_sample):
                raise RuntimeError(
                    f"[POLYA-SIDECAR] the per-key aggregate changed the row count: "
                    f"{len(per_sample)} rows became {len(table)}. The aggregate must "
                    f"hold exactly one row per key."
                )
            if not table['polyA_frac_model'].notna().all():
                n_missing = int(table['polyA_frac_model'].isna().sum())
                raise RuntimeError(
                    f"[POLYA-SIDECAR] {n_missing} per-sample row(s) have no per-key "
                    f"aggregate. A key that the aggregation does not cover must not "
                    f"be delivered as a pooled fallback."
                )
            table['estimator'] = np.where(table['polyA_n_samples'] > 0,
                                          'median', 'pooled_fallback')
        else:
            table['polyA_frac_model'] = np.nan
            table['polyA_frac_pooled'] = np.nan
            table['polyA_n_samples'] = 0
            table['estimator'] = 'pooled_fallback'

        table['polyA_frac'] = table['polyA_freq'] / table['raw_read_freq']
        cols = POLYA_MERGE_COLS + ['sample', 'polyA_freq', 'raw_read_freq', 'polyA_frac',
                                   'polyA_frac_model', 'polyA_frac_pooled',
                                   'polyA_n_samples', 'estimator']
        table[cols].to_csv(out_path, sep='\t', index=False)
        logger.info(
            f"[POLYA-SIDECAR] Wrote {len(table)} per-sample poly(A) rows to {out_path}."
        )

    def anno_polya(self, df, flnc_files):
        """Multi-sample polyA annotation

        The per-sample statistics are aggregated to ONE row per key before
        they are merged onto the model frame. ``polyA_frac`` is the
        unweighted median over the samples that carry measured poly(A)
        evidence (see aggregate_polya); the read-depth-weighted pooled ratio
        is kept as ``polyA_frac_pooled`` for auditing.

        Parameters:
        ----
        df : pd.DataFrame
            Dataframe to be annotated
        flnc_files : list[str]
            List of multiple flnc.ssc file paths

        Returns:
        ----
        pd.DataFrame
            Dataframe with added polyA_frac column
        """
        per_sample_frames = []
        for flnc_file, read_df in self._correct_flnc_for_files(df, flnc_files):
            sample = sample_id_from_flnc(flnc_file)
            per_sample_frames.append(per_sample_polya_stats(read_df, sample).to_pandas())

        if per_sample_frames:
            per_sample = pd.concat(per_sample_frames, ignore_index=True)
        else:
            per_sample = pd.DataFrame(
                columns=POLYA_MERGE_COLS + ['sample', 'polyA_freq', 'raw_read_freq'])

        # One row per key: median over the qualifying samples, pooled ratio
        # as the fallback and as the audit value.
        df_polya = aggregate_polya(per_sample)

        # Audit trail: the per-sample values plus the per-key aggregate,
        # written next to the outputs (the aggregate is needed to know which
        # estimator produced the delivered polyA_frac).
        self._write_per_sample_sidecar(per_sample, df_polya)

        # Perform correction and annotation
        df = self.add_polya_frac(df, df_polya)

        # Ensure TrStart and TrEnd columns are of int type
        if 'TrStart' in df.columns:
            df['TrStart'] = df['TrStart'].astype(int)
        if 'TrEnd' in df.columns:
            df['TrEnd'] = df['TrEnd'].astype(int)

        return df