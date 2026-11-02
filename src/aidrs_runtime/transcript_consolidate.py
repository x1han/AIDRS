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
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .column_standardize import build_ref_dict
from .id_validator import MAX_TRANSCRIPT_ID_LEN, byte_len

logger = logging.getLogger(__name__)

# Maximum length (in bytes) of a GeneID / GeneName before it is replaced by a
# short aggregate id.  The hard limit for a transcript id is 254 bytes (the
# SAM v1.6 QNAME limit), and the GeneID is a prefix of the TrID, so a gene id
# must stay well under that. 200 leaves room for the "_<tail>" suffix that
# ``update_tr_id`` appends.
MAX_GENE_ID_LEN = 200

# Prefix of the aggregate id used when a gene id / gene name is too long.
LONG_FUSION_PREFIX = "NovelGeneCluster_LongFusion_"


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


def _rewrite_transcript_ids(orig_trids: pd.Series, gene_id: str) -> List[str]:
    """Rewrite a gene group's transcript ids under ``gene_id``.

    The tail (everything after the first '_') identifies the transcript only
    when it is unique inside the group: two reference transcripts flagged
    'AlterTss', or two bare reference ids, collapse onto the same candidate
    and would produce the SAME TrID. Rows whose candidate collides keep
    their full original id, which is unique within the frame.

    The length of the result is ``byte_len(gene_id) + 1 + byte_len(suffix)``;
    the suffix is unbounded, so a gene id within ``MAX_GENE_ID_LEN`` does
    not by itself guarantee a transcript id within the QNAME limit.

    Args:
        orig_trids: The group's original transcript ids.
        gene_id: The gene id the transcript ids are rewritten under.

    Returns:
        The rewritten transcript ids, in the order of ``orig_trids``.
    """
    orig_trids = orig_trids.astype(str)
    has_sep = orig_trids.str.contains('_')
    tails = orig_trids.str.split('_', n=1).str[-1]

    candidates = [
        f"{gene_id}_{tail}" if sep else gene_id
        for tail, sep in zip(tails, has_sep)
    ]
    cand_counts = pd.Series(candidates).value_counts()
    return [
        c if cand_counts[c] == 1 else f"{gene_id}_{o}"
        for c, o in zip(candidates, orig_trids)
    ]


def novel_gene_remapping(df_result: pd.DataFrame,
                         ref_anno: Optional[pd.DataFrame],
                         terminal_tolerance: int = 50,
                         long_fusion_map: Optional[List[Dict[str, object]]] = None) -> pd.DataFrame:
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
        long_fusion_map: Optional list that is APPENDED TO with one record
            per over-long gene id that was replaced by a short aggregate id
            (``NovelGeneCluster_LongFusion_<N>``). Each record keeps the
            original source gene ids / names so the association is not lost.

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
    # Deterministic counter for over-long gene ids: groups are visited in
    # groupby order (sorted by GeneID), so the same input always yields the
    # same aggregate ids.
    long_fusion_index = 0

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
                    source_gene_ids = [gene_id]
                    source_gene_names = [gene_name]
                else:
                    parts = best_rows.ref_key.str.split('_', n=1, expand=True)
                    source_gene_ids = list(parts[0])
                    source_gene_names = list(parts[1])
                    gene_id = 'NovelGeneCluster_' + '_'.join(source_gene_ids)
                    gene_name = 'NovelGeneCluster_' + '_'.join(source_gene_names)

                # The gene id and the transcript id share ONE budget: the
                # transcript id becomes the reference name (RNAME) of the
                # model FASTA / BAM, and SAM v1.6 caps QNAME at 254 bytes.
                # The gene id is the prefix of every transcript id of the
                # group and the transcript suffix is added on top of it, so
                # the two limits have to be accounted for together. (The gene
                # id itself is only a GTF attribute, but it consumes part of
                # the transcript id's budget.)
                original_gene_id, original_gene_name = gene_id, gene_name

                def _aggregate_id(reason: str) -> str:
                    """Allocate the next short aggregate id for this group.

                    The source gene ids / names are recorded in
                    ``long_fusion_map`` so the association is not lost.
                    ``reason`` says which limit forced the shortening.
                    """
                    nonlocal long_fusion_index
                    long_fusion_index += 1
                    aggregate_id = f"{LONG_FUSION_PREFIX}{long_fusion_index}"
                    if long_fusion_map is not None:
                        long_fusion_map.append({
                            'long_fusion_id': aggregate_id,
                            'source_gene_ids': ','.join(source_gene_ids),
                            'source_gene_names': ','.join(source_gene_names),
                            'n_source_genes': len(source_gene_ids),
                            'original_gene_id': original_gene_id,
                            'original_gene_name': original_gene_name,
                            'reason': reason,
                        })
                    return aggregate_id

                # An over-long gene id / gene name must not be emitted: it is
                # the prefix of every transcript id of the group, so it would
                # push them over the SAM v1.6 QNAME limit of 254 bytes.
                # Replace it with a short deterministic aggregate id and keep
                # the source list in ``long_fusion_map`` so the association
                # is not lost.
                shortened = False
                if (byte_len(gene_id) > MAX_GENE_ID_LEN
                        or byte_len(gene_name) > MAX_GENE_ID_LEN):
                    gene_id = gene_name = _aggregate_id('gene_id_too_long')
                    shortened = True

                # Replace current group's GeneID and GeneName
                group_df['GeneID'] = gene_id
                group_df['GeneName'] = gene_name

                # Rewrite the transcript ids under the (possibly shortened)
                # gene id.
                orig_trids = group_df['TrID'].astype(str)
                new_trids = _rewrite_transcript_ids(orig_trids, gene_id)

                # A gene id within its own limit does NOT guarantee a
                # transcript id within the QNAME limit: the suffix
                # ("_<tail>", or the full original id in the collision
                # fallback) is added on top of it. Shorten the gene id as
                # well and recompute. A group that was ALREADY shortened
                # cannot be shortened twice, so it keeps the aggregate id.
                if (not shortened
                        and max(byte_len(t) for t in new_trids) > MAX_TRANSCRIPT_ID_LEN):
                    gene_id = gene_name = _aggregate_id('transcript_id_too_long')
                    group_df['GeneID'] = gene_id
                    group_df['GeneName'] = gene_name
                    new_trids = _rewrite_transcript_ids(orig_trids, gene_id)

                # The part of the transcript id behind the gene id can be so
                # long that NO gene id can bring the transcript id within
                # the limit. That is not a hard error here: the transcript
                # may still be dropped by the expression filter, exactly as
                # it was before this rule existed, and what is actually
                # shipped is validated fail loud by ``validate_transcript_ids``
                # on the FILTERED model. Warn loud so it is never silent.
                over_long = [t for t in new_trids
                             if byte_len(t) > MAX_TRANSCRIPT_ID_LEN]
                if over_long:
                    longest = max(byte_len(t) for t in over_long)
                    logger.warning(
                        "[ID-BUDGET] %d transcript id(s) of gene '%s' exceed the "
                        "%d-byte QNAME limit even after shortening the gene id: "
                        "everything behind the gene id is %d bytes on its own "
                        "(longest id %d bytes), so no gene id can bring them "
                        "within the limit. The ids are kept as they are; if one "
                        "of them survives the expression filter the run aborts "
                        "in validate_transcript_ids. Offending ids: %s",
                        len(over_long), gene_id, MAX_TRANSCRIPT_ID_LEN,
                        longest - byte_len(gene_id) - 1, longest,
                        sorted(set(over_long))[:3])

                if len(set(new_trids)) != len(new_trids):
                    raise RuntimeError(
                        f"Transcript id rewrite produced duplicates inside gene "
                        f"'{gene_id}': {sorted(t for t in set(new_trids) if new_trids.count(t) > 1)}"
                    )
                group_df['TrID'] = new_trids

        # Add to final results regardless of modification
        updated_groups.append(group_df)

    # Merge all groups
    return pd.concat(updated_groups, ignore_index=True)