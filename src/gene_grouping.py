import pandas as pd
import numpy as np
import multiprocessing as mp
import logging

logger = logging.getLogger(__name__)

# P0-7: INT_MIN Group sentinel for single-exon rows (cannot collide with
# legitimate cumsum-derived group ids, which start at 0 or 1).
SINGLE_EXON_GROUP_SENTINEL = np.iinfo(np.int32).min  # -2147483648


def compute_is_intergenic_or_antisense(df_single, df_multi=None):
    """Add is_intergenic_or_antisense column to single-exon rows.

    Pillar 1 of the Stage 2.5b 5-pillar funnel (per
    `p3_single_exon_5_pillar.md`): a single-exon row passes only if it does
    NOT overlap any same-strand multi-exon transcript on [TrStart, TrEnd].

    Args:
        df_single: single-exon DataFrame. Must contain Chr, Strand, TrStart,
            TrEnd (1-based inclusive genomic coordinates, TrStart < TrEnd).
        df_multi: optional multi-exon DataFrame from the same run. When
            provided, the overlap check is performed against it. When None,
            all rows are marked True (single-exon reads by construction are
            not in the multi-exon funnel because SSC == 'none').

    Returns:
        df_single with a new boolean column `is_intergenic_or_antisense`.
    """
    if df_single is None or len(df_single) == 0:
        if df_single is None:
            return df_single
        return df_single.assign(is_intergenic_or_antisense=pd.Series([], dtype=bool))

    df_single = df_single.copy()
    df_single['is_intergenic_or_antisense'] = True

    if df_multi is None or len(df_multi) == 0:
        return df_single

    # Vectorised per-(Chr, Strand) same-strand overlap check via cross-merge
    # + interval-overlap predicate. For C107 100k this is O(|single| x
    # |multi on same chr/strand|) but the per-key cross product is tiny in
    # practice (single-exon reads are a small minority of total).
    required = {'Chr', 'Strand', 'TrStart', 'TrEnd'}
    if not required.issubset(df_single.columns):
        logger.warning(
            "compute_is_intergenic_or_antisense: missing columns %s; "
            "defaulting to is_intergenic_or_antisense=True",
            required - set(df_single.columns),
        )
        return df_single
    if not required.issubset(df_multi.columns):
        logger.warning(
            "compute_is_intergenic_or_antisense: df_multi missing columns %s; "
            "defaulting to is_intergenic_or_antisense=True",
            required - set(df_multi.columns),
        )
        return df_single

    # Ensure TrStart <= TrEnd for both sides (defensive: pipeline usually
    # guarantees this but a swap on antisense scaffolds breaks the predicate).
    s_start = df_single['TrStart'].to_numpy()
    s_end = df_single['TrEnd'].to_numpy()
    s_start, s_end = np.minimum(s_start, s_end), np.maximum(s_start, s_end)

    # Group multi-exon by (Chr, Strand) into a dict[chr,strand] -> ndarray
    # of shape (n, 2) for start,end; then for each single row look up the
    # same-key group and check overlap.
    multi_groups = {}
    for (chr_, strand_), grp in df_multi.groupby(['Chr', 'Strand'], observed=True):
        m_start = grp['TrStart'].to_numpy()
        m_end = grp['TrEnd'].to_numpy()
        m_start, m_end = np.minimum(m_start, m_end), np.maximum(m_start, m_end)
        multi_groups[(chr_, strand_)] = np.column_stack([m_start, m_end])

    flags = np.ones(len(df_single), dtype=bool)
    df_single_indexed = df_single.reset_index(drop=True)
    s_chr = df_single_indexed['Chr'].to_numpy()
    s_strand = df_single_indexed['Strand'].to_numpy()
    for i in range(len(df_single_indexed)):
        key = (s_chr[i], s_strand[i])
        if key not in multi_groups:
            continue
        intervals = multi_groups[key]
        # Overlap predicate: s_start <= m_end AND m_start <= s_end
        overlap = (s_start[i] <= intervals[:, 1]) & (intervals[:, 0] <= s_end[i])
        if overlap.any():
            flags[i] = False

    df_single['is_intergenic_or_antisense'] = flags
    return df_single

class GeneClustering:
    """
    gene-level grouping
    """

    def __init__(self, num_processes=None):
        self.num_processes = num_processes

    @staticmethod
    def _single_strand_interval_clustering(df):
        df = df.copy()
        df['start'] = df['SSC'].str.split('-').str[0].astype(np.int32)
        df['end'] = df['SSC'].str.split('-').str[-1].astype(np.int32)

        df = df.sort_values(by='start')
        ends = df['end'].cummax().shift(1, fill_value=df['end'].iloc[0])
        group_ids = (df['start'] > ends).cumsum()
        df['Group'] = group_ids

        return df.drop(columns=['start', 'end'])

    @staticmethod
    def _cluster_for_chr(df):
        results = []
        global_group_id = 0
        for strand, strand_group in df.groupby('Strand', observed=True):
            clustered = GeneClustering._single_strand_interval_clustering(strand_group)
            # Reassign globally unique Group ID
            unique_groups = clustered['Group'].unique()
            group_mapping = {old_id: global_group_id + i for i, old_id in enumerate(unique_groups)}
            clustered['Group'] = clustered['Group'].map(group_mapping)
            global_group_id += len(unique_groups)
            results.append(clustered)

        return pd.concat(results, ignore_index=True)

    def cluster(self, df):
        if df.empty:
            return df.assign(Group=pd.Series(dtype=np.int32))

        df = df.astype({
            'Chr': 'category',
            'Strand': 'category',
            'SSC': str
        })

        grouped = [group for _, group in df.groupby('Chr', observed=True)]
        num_processes = min(self.num_processes, len(grouped))

        ctx = mp.get_context("spawn")
        with ctx.Pool(num_processes) as pool:
            results = pool.map(self._cluster_for_chr, grouped)

        clustered_df = pd.concat(results, ignore_index=True)
        
        # Ensure Group IDs are also unique across different chromosomes
        global_group_id = 0
        final_results = []
        for chr_result in results:
            if len(chr_result) > 0:
                unique_groups = chr_result['Group'].unique()
                group_mapping = {old_id: global_group_id + i for i, old_id in enumerate(unique_groups)}
                chr_result_copy = chr_result.copy()
                chr_result_copy['Group'] = chr_result_copy['Group'].map(group_mapping)
                global_group_id += len(unique_groups)
                final_results.append(chr_result_copy)
        
        if final_results:
            clustered_df = pd.concat(final_results, ignore_index=True)
        else:
            clustered_df = pd.DataFrame()
        
        return clustered_df.reset_index(drop=True)
