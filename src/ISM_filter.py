import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger("AIDRS")


def _parse_introns(tr_start: int, ssc: str, tr_end: int) -> List[Tuple[int, int]]:
    """Parse SSC into an ordered list of (donor, acceptor) intron tuples.

    SSC encodes intron boundaries as a dash-separated list of 2*(n-1)
    integers (consecutive exon_i.end - exon_{i+1}.start). A single-exon
    transcript (SSC == "NA") has no introns; returns [].

    Example:
      SSC="210231771-210232698" with TrStart=210230000 TrEnd=210235000
        -> introns = [(210231771, 210232698)]
      SSC="100-200-300-400" (3-exon) -> introns = [(100, 200), (300, 400)]
    """
    if ssc is None or (isinstance(ssc, float) and pd.isna(ssc)):
        return []
    s = str(ssc).strip()
    if not s or s.upper() == "NA":
        return []
    try:
        nums = [int(x) for x in s.split("-")]
    except ValueError:
        return []
    # Pair up [tr_start] + SSC numbers + [tr_end] to get exon boundaries,
    # then adjacent (exon[i].end, exon[i+1].start) pairs ARE the introns.
    positions = [int(tr_start)] + nums + [int(tr_end)]
    if len(positions) < 2 or len(positions) % 2 != 0:
        return []
    exons = list(zip(positions[0::2], positions[1::2]))
    introns = [(exons[i][1], exons[i + 1][0]) for i in range(len(exons) - 1)]
    return introns


def _is_contiguous_intron_subchain(
    query_introns: List[Tuple[int, int]],
    target_introns: List[Tuple[int, int]],
) -> bool:
    """Check whether query_introns is a strict ordered contiguous sub-chain of target_introns.

    Per SQANTI3 / FLAIR definition: A is an ISM (truncation candidate) of B
    iff A's intron chain is a contiguous sub-chain of B's intron chain,
    preserving order, AND A is strictly shorter than B (at least one
    fewer intron -- otherwise they're alternative splicing, not truncation).

    This replaces the legacy `_ssc_token_overlap` (which used a symmetric
    bag-of-tokens intersection and so mis-classified A3SS events where two
    isoforms share one splice site but have different acceptor ends).
    """
    n_q = len(query_introns)
    n_t = len(target_introns)
    if n_q == 0 or n_q >= n_t:
        return False
    for i in range(n_t - n_q + 1):
        if target_introns[i:i + n_q] == query_introns:
            return True
    return False


class TruncationProcessor:
    def __init__(self, threshold_truncation_source_freq=0.5, threshold_truncation_group_freq=0.5, trunc_simp_filter=False, num_processes=None, puffin_tss_rescue=0.1):
        self.threshold_truncation_source_freq = threshold_truncation_source_freq
        self.threshold_truncation_group_freq = threshold_truncation_group_freq
        self.trunc_simp_filter = trunc_simp_filter
        from .aidrs_runtime.resource_guard import ResourceGuard
        self.num_processes = ResourceGuard.get_effective_cpu_threads(num_processes)
        # Independent TSS rescue: even if a row is structurally a 5'-truncation
        # candidate of some other_row, if it carries a Puffin promoter signal
        # at its OWN 5' end, it is an independent TSS isoform (alternative
        # promoter), not a degradation artifact. Default 0.1 matches the
        # Stage 2.5b mono-exon Pillar 5 threshold.
        self.puffin_tss_rescue = puffin_tss_rescue

    def _assess_truncation_for_Chr(self, df_clustered_Chr):
        df = df_clustered_Chr.copy()
        df['sourceSSC_counts'] = 0
        df['trun_source_freq'] = 0

        # Pre-parse introns + 5' puffin signal for every row
        df['_introns'] = df.apply(
            lambda r: _parse_introns(r.get('TrStart'), r.get('SSC'), r.get('TrEnd')),
            axis=1,
        )
        # Puffin rescue: Puffin_TSS_15bp can be string ('NA', 'no') or float.
        # Coerce to float; anything non-numeric -> 0.
        def _puffin(v):
            try:
                f = float(v)
                return f if f == f else 0.0  # NaN guard
            except (TypeError, ValueError):
                return 0.0
        df['_puffin_5p'] = df['Puffin_TSS_15bp'].apply(_puffin)

        grouped_dict = {}
        for _, row in df.iterrows():
            key = (row['Chr'], row['Strand'], row['Group'])
            grouped_dict.setdefault(key, []).append(row)

        for index, row in df.iterrows():
            key = (row['Chr'], row['Strand'], row['Group'])
            sourceSSC_counts = 0
            trun_source_freq = 0
            truncation_source = []
            row_introns = row['_introns']
            row_puffin = row['_puffin_5p']
            for other_row in grouped_dict[key]:
                if row['SSC'] == other_row['SSC']:
                    continue  # identical SSCs are siblings, not truncations
                other_introns = other_row['_introns']
                if not _is_contiguous_intron_subchain(row_introns, other_introns):
                    continue  # alternative splicing / divergent isoform
                # Independent TSS rescue: if our row has its own promoter,
                # it's an independent transcription start, not a 5' degraded
                # artifact of other_row.
                if row_puffin >= self.puffin_tss_rescue:
                    continue
                sourceSSC_counts += 1
                trun_source_freq += other_row['frequency']
                truncation_source.append(other_row['SSC'])

            df.at[index, 'sourceSSC_counts'] = sourceSSC_counts
            df.at[index, 'trun_source_freq'] = trun_source_freq
            df.at[index, 'truncation_source'] = ','.join(truncation_source) if truncation_source else 'full'
        return df

    def assess_truncation(self, df_clustered, ref_anno=None):
        Chr_groups = df_clustered.groupby(['Chr','Strand'],observed=True)
        Chr_list = [group for _, group in Chr_groups]

        with Pool(self.num_processes) as pool:
            results = pool.map(self._assess_truncation_for_Chr, Chr_list)

        df = pd.concat(results)

        # Calculate ratios instead of logFC
        df['source_freq_ratio'] = np.where(
            df['truncation_source'] != 'full',
            df['frequency'] / (df['frequency'] + df['trun_source_freq']),
            np.inf
        )
        df['group_freq'] = df.groupby(['Chr', 'Strand', 'Group'],observed=True)['frequency'].transform('sum')
        # group_freq_ratio must exclude self from denominator.
        df['group_freq_ratio'] = np.where(
            df['truncation_source'] != 'full',
            df['frequency'] / df['group_freq'],
            np.inf
        )

        def truncation_classify(row):
            if row['source_freq_ratio'] >= self.threshold_truncation_source_freq and \
               row['group_freq_ratio'] >= self.threshold_truncation_group_freq:
                return 'no'
            else:
                return 'yes'

        df['truncation'] = df.apply(truncation_classify, axis=1)

        # Apply simple filter if enabled
        if self.trunc_simp_filter:
            if ref_anno is not None:
                from .isoform_classify import IsoformClassifier
                isoformclassifier = IsoformClassifier(num_processes=self.num_processes)
                df = isoformclassifier.add_category(df, ref_anno)
                original_count = len(df)
                df = df[(df['truncation'] == 'no') | (df['category'] == 'FSM')]
                filtered_count = len(df)
                logger.info(f"\tTruncation filtered: Retained {filtered_count} of {original_count} transcripts ({filtered_count/original_count*100:.2f}%).")
                if 'category' in df.columns:
                    df = df.drop(columns=['category'])
            else:
                original_count = len(df)
                df = df[df['truncation'] == 'no']
                filtered_count = len(df)
                logger.info(f"\tTruncation filtered: Retained {filtered_count} of {original_count} transcripts ({filtered_count/original_count*100:.2f}%).")

        df = df.drop(columns=['group_freq', 'sourceSSC_counts', 'trun_source_freq', 'source_freq_ratio', 'group_freq_ratio', 'truncation_source', '_introns', '_puffin_5p']).reset_index(drop=True)

        return df