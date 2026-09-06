"""Transcript consolidation / dedup logic (mid-level).

This module owns the per-group / per-gene consolidation helpers that
``src.generate_reports.annotate`` composes inside
``annotate_one_group`` and its pool-mapped static wrapper
``_annotate_one_group``. They operate on DataFrames that have already
been cluster-grouped and (optionally) reference-merged.

Functions:
- ``update_ref_with_flags``: mark TrIDs whose TSS/TES differs from the
  reference beyond ``terminal_tolerance`` as ``AlterTss`` / ``AlterTes``
  / ``AlterTssTes``; also rewrite TrStart_ref / TrEnd_ref to the query
  coordinates (so downstream stages see the query as the reference).
- ``transcript_1to1_processor``: collapse multiple reference rows for
  the same ``uniqueTr`` into a single combined ``TrID``, picking the
  best match by absolute TrStart/TrEnd distance.
- ``map_transcript_1to1``: groupby wrapper around
  ``transcript_1to1_processor``.
- ``map_query_to_ref``: assign each query row to the best reference
  gene by site-overlap count (>= 2 shared sites); fall back to
  ``NovelGene{GID}`` naming if no reference matches.
- ``novel_gene_remapping``: re-attach ``NovelGene*`` clusters to their
  closest reference gene by whole-cluster site overlap, possibly fusing
  multiple reference genes into ``NovelGeneCluster_*``.

The ``terminal_tolerance`` parameter is passed explicitly (not pulled
from a class instance) so these functions are unit-testable in
isolation.

Extracted from the original ``src/generate_reports.py`` "god class"
(IsoformAnnotator).
"""
from typing import Optional

import numpy as np
import pandas as pd

from .column_standardize import build_ref_dict


def update_ref_with_flags(ref_df: pd.DataFrame, terminal_tolerance: int) -> None:
    """Update reference DataFrame with TSS/TES alteration flags (in place).

    For each row, compute the absolute TSS/TES distance from the query
    to the reference, applying a flag suffix to ``TrID`` if either
    distance exceeds ``terminal_tolerance``. Then overwrite the
    ``TrStart_ref`` / ``TrEnd_ref`` columns with the query coordinates
    (so downstream stages see the query coordinates as the "reference"
    coordinates). Negative-strand handling swaps the TSS/TES role for
    the delta calculation but leaves the resulting flag identical.

    Args:
        ref_df: DataFrame (modified in place) with columns
                ``TrID``, ``Strand``, ``TrStart``, ``TrEnd``,
                ``TrStart_ref``, ``TrEnd_ref``.
        terminal_tolerance: Maximum allowed absolute TSS/TES distance
                            before flagging.
    """
    for idx, row in ref_df.iterrows():
        ref_trans_id = row['TrID']
        strand = row['Strand']
        q_trs, q_tre = row['TrStart'], row['TrEnd']
        r_trs, r_tre = row['TrStart_ref'], row['TrEnd_ref']

        # Calculate differences based on strand
        if strand == '+':
            d_tss, d_tes = abs(q_trs - r_trs), abs(q_tre - r_tre)
        else:
            d_tss, d_tes = abs(q_tre - r_tre), abs(q_trs - r_trs)

        # Determine flag based on differences
        flag = None
        if d_tss > terminal_tolerance and d_tes > terminal_tolerance:
            flag = 'AlterTssTes'
        elif d_tss > terminal_tolerance:
            flag = 'AlterTss'
        elif d_tes > terminal_tolerance:
            flag = 'AlterTes'

        # Apply flag to TrID if needed
        if flag:
            ref_df.at[idx, 'TrID'] = f"{ref_trans_id}_{flag}"

        # Update reference coordinates
        ref_df.at[idx, 'TrStart_ref'] = q_trs
        ref_df.at[idx, 'TrEnd_ref'] = q_tre


def transcript_1to1_processor(uni_tr_mappings: pd.DataFrame,
                              terminal_tolerance: int) -> pd.DataFrame:
    """Collapse multiple reference rows for one uniqueTr into one TrID.

    "Match" rows are those whose TrStart / TrEnd deltas to the
    reference are within ``terminal_tolerance`` on BOTH ends.

    - If any match exists: combine the matching TrIDs with ``_`` as the
      separator; set TrStart_ref / TrEnd_ref to the first match's query
      coordinates.
    - Otherwise (all miss): combine ALL TrIDs with ``_``; pick the
      nearest-miss row by absolute TrStart/TrEnd distance and use its
      TrStart_ref / TrEnd_ref; then call ``update_ref_with_flags`` to
      apply the appropriate AlterTss / AlterTes flag.

    Args:
        uni_tr_mappings: DataFrame for one uniqueTr (multiple rows
                         possible, e.g. one row per reference TrID).
        terminal_tolerance: See ``update_ref_with_flags``.

    Returns:
        DataFrame with at most one row after dedup, including a
        transient ``match_status`` column that callers must drop.
    """
    # Copy to avoid warnings
    uni_tr_mappings = uni_tr_mappings.copy()
    uni_tr_mappings['match_status'] = uni_tr_mappings.apply(
        lambda row: 'match' if (abs(row['TrStart'] - row['TrStart_ref']) <= terminal_tolerance and abs(row['TrEnd'] - row['TrEnd_ref']) <= terminal_tolerance) else 'miss',
        axis=1
    )

    # Find matching rows
    match_mask = uni_tr_mappings['match_status'] == 'match'
    matching_rows = uni_tr_mappings[match_mask]

    if not matching_rows.empty:
        working_df = matching_rows.copy()

        unique_trids = working_df['TrID'].unique()
        combined_trid = '_'.join(unique_trids)
        working_df['TrID'] = combined_trid

        trstart_ref_value = working_df['TrStart'].iloc[0]
        trend_ref_value = working_df['TrEnd'].iloc[0]
        working_df['TrStart_ref'] = trstart_ref_value
        working_df['TrEnd_ref'] = trend_ref_value

        result_row = working_df.drop_duplicates()

    else:
        working_df = uni_tr_mappings.copy()

        unique_trids = working_df['TrID'].unique()
        combined_trid = '_'.join(unique_trids)
        working_df['TrID'] = combined_trid

        working_df['TrStart_ref'] = working_df.loc[np.abs(working_df['TrStart_ref'] - working_df['TrStart'].iloc[0]).idxmin(), 'TrStart_ref']
        working_df['TrEnd_ref'] = working_df.loc[np.abs(working_df['TrEnd_ref'] - working_df['TrEnd'].iloc[0]).idxmin(), 'TrEnd_ref']
        update_ref_with_flags(working_df, terminal_tolerance)

        trstart_ref_value = working_df['TrStart'].iloc[0]
        trend_ref_value = working_df['TrEnd'].iloc[0]
        working_df['TrStart_ref'] = trstart_ref_value
        working_df['TrEnd_ref'] = trend_ref_value

        result_row = working_df.drop_duplicates()

    return result_row


def map_transcript_1to1(df: pd.DataFrame, terminal_tolerance: int) -> pd.DataFrame:
    """Groupby wrapper: collapse rows for each ``uniqueTr`` via ``transcript_1to1_processor``.

    Args:
        df: DataFrame with column ``uniqueTr``.
        terminal_tolerance: See ``update_ref_with_flags``.

    Returns:
        DataFrame with at most one row per ``uniqueTr`` and the
        transient ``match_status`` column dropped.
    """
    return (
        df.groupby(df['uniqueTr'].values, group_keys=False, as_index=False)
        .apply(lambda g: transcript_1to1_processor(g, terminal_tolerance))
        .reset_index(drop=True)
    ).drop(columns=['match_status'])


def map_query_to_ref(query_df: pd.DataFrame, ref_dict: dict) -> pd.DataFrame:
    """Map each query transcript to its best-matching reference gene (in place).

    For each query row, compute the set of sites (``TrStart``, ``TrEnd``,
    plus SSC breakpoints). Pick the reference gene with the largest
    overlap, but only when overlap is ``>= 2`` sites. If no such
    reference exists, fall back to ``NovelGene{GID}`` naming.

    After the TrID/GeneID/GeneName assignment, the row's
    ``TrStart_ref`` / ``TrEnd_ref`` columns are set to its own
    ``TrStart`` / ``TrEnd`` (self-reference) regardless of which
    branch ran.

    Args:
        query_df: DataFrame (modified in place) with columns
                  ``TrStart``, ``TrEnd``, ``SSC``, ``uniqueTr``,
                  ``Group``, plus the columns to be written
                  (``TrID``, ``GeneID``, ``GeneName``, ``TrStart_ref``,
                  ``TrEnd_ref``).
        ref_dict: Output of ``build_ref_dict`` -- ``{gene_key: [sites]}``.

    Returns:
        The same DataFrame with the per-row assignments written.
    """
    for idx, row in query_df.iterrows():
        q_sites = {int(row['TrStart']), int(row['TrEnd'])}
        ssc_str = str(row['SSC']).strip()
        if ssc_str and ssc_str != 'nan':
            q_sites |= {int(x) for x in ssc_str.split('-') if x}
        best_key, best_count = None, 0
        for ref_key, ref_sites in ref_dict.items():
            cnt = len(q_sites & set(ref_sites))
            if cnt >= 2 and cnt > best_count:
                best_count, best_key = cnt, ref_key
        trid = row['uniqueTr']
        if best_key and best_count >= 2:
            gene_id, gene_name = best_key.split('_', 1)
            query_df.at[idx, 'TrID'] = f'{gene_id}_Novel{trid}'
            query_df.at[idx, 'GeneID'] = gene_id
            query_df.at[idx, 'GeneName'] = gene_name
        else:
            gid = row['Group']
            query_df.at[idx, 'TrID'] = f'NovelGene{gid}_Novel{trid}'
            query_df.at[idx, 'GeneID'] = f'NovelGene{gid}'
            query_df.at[idx, 'GeneName'] = f'NovelGene{gid}'
        # Put ref coordinates as self
        query_df.at[idx, 'TrStart_ref'] = row['TrStart']
        query_df.at[idx, 'TrEnd_ref'] = row['TrEnd']
    return query_df


def novel_gene_remapping(df_result: pd.DataFrame,
                         ref_anno: Optional[pd.DataFrame],
                         terminal_tolerance: int = 50) -> pd.DataFrame:
    """Re-attach NovelGene clusters to closest reference gene by whole-cluster site overlap.

    For each gene group whose GeneID contains ``"NovelGene"``, collect
    all SSC breakpoint sites across the cluster. If any reference gene
    on the same ``(Chr, Strand)`` shares ``>= 2`` of those sites,
    attach the cluster to the best-matching reference (preferring
    "full coverage" matches -- i.e. all reference sites are covered).
    Multi-match clusters are fused into ``NovelGeneCluster_*`` names.

    The ``terminal_tolerance`` parameter is accepted but not used
    directly; it is here only so the signature matches the prior
    ``self._novel_gene_remapping`` signature for backwards compatibility.

    Args:
        df_result: Annotated DataFrame with columns ``GeneID``,
                   ``Chr``, ``Strand``, ``SSC``, ``TrID``.
        ref_anno: Reference annotation DataFrame; required.
        terminal_tolerance: Accepted for signature compatibility;
                            unused by this function.

    Returns:
        New DataFrame with ``GeneID``, ``GeneName``, ``TrID`` updated
        for any cluster that found a reference match.
    """
    # Build reference dictionary (grouped by Chr, Strand)
    ref_anno_dict_by_chr_strand = {}
    for (chrom, strand), group_df in ref_anno.groupby(['Chr', 'Strand']):
        inner_dict = build_ref_dict(group_df, include_term=False)
        ref_anno_dict_by_chr_strand[(chrom, strand)] = inner_dict

    # Store results for all groups
    updated_groups = []

    for gid, group_df in df_result.groupby('GeneID'):
        group_df = group_df.copy()  # Avoid modifying original data warnings
        if 'NovelGene' in gid:
            matches = []
            all_q_sites = set()
            q_chrom, q_strand = None, None

            # Collect all SSC sites for this group (ignoring TrStart/TrEnd)
            for idx, row in group_df.iterrows():
                q_chrom, q_strand = row['Chr'], row['Strand']
                ssc_str = str(row['SSC']).strip()
                if ssc_str and ssc_str not in ('nan', ''):
                    sites = {int(x) for x in ssc_str.split('-') if x.strip()}
                    all_q_sites.update(sites)

            # Check if there is corresponding reference data
            if q_chrom is not None and q_strand is not None and (q_chrom, q_strand) in ref_anno_dict_by_chr_strand:
                ref_dict = ref_anno_dict_by_chr_strand[(q_chrom, q_strand)]
                for ref_key, ref_sites in ref_dict.items():
                    cnt = len(all_q_sites & set(ref_sites))
                    if cnt >= 2:
                        ref_full_flag = (cnt == len(ref_sites))  # Whether all reference sites are covered
                        matches.append({'ref_key': ref_key, 'cnt': cnt, 'ref_full': ref_full_flag})

            if matches:
                match_df = pd.DataFrame(matches)
                if match_df.ref_full.any():
                    best_rows = match_df[match_df.ref_full]
                else:
                    max_cnt = match_df.cnt.max()
                    best_rows = match_df[match_df.cnt == max_cnt]

                # Concatenate results
                if len(best_rows) == 1:
                    gene_id, gene_name = best_rows.iloc[0].ref_key.split('_', 1)
                else:
                    parts = best_rows.ref_key.str.split('_', n=1, expand=True)
                    gene_id = 'NovelGeneCluster_' + '_'.join(parts[0])
                    gene_name = 'NovelGeneCluster_' + '_'.join(parts[1])

                # Replace current group's GeneID and GeneName
                group_df['GeneID'] = gene_id
                group_df['GeneName'] = gene_name

                def update_tr_id(tr_id):
                    parts = str(tr_id).split('_', 1)
                    if len(parts) == 2:
                        return f"{gene_id}_{parts[1]}"
                    else:
                        # If there is no underscore, replace with gene_id (or keep original? as needed)
                        return gene_id

                group_df['TrID'] = group_df['TrID'].apply(update_tr_id)

        # Add to final results regardless of modification
        updated_groups.append(group_df)

    # Merge all groups
    return pd.concat(updated_groups, ignore_index=True)