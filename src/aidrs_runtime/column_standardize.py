"""Column name standardization / DataFrame alignment utilities (low-level).

This module owns the smallest, pure-Python / pure-pandas utilities that
the higher-level consolidation and report-writing modules build on:

- ``reverse_complement``: pure DNA reverse-complement (no pandas).
  Used by ``to_fasta`` to flip the sense strand; pinned by
  tests/test_generate_reports_characterization.py::test_reverse_complement.
- ``build_ref_dict``: DataFrame -> ``{gene_key: [site_positions]}``;
  the per-gene site set used by ``map_query_to_ref`` and
  ``novel_gene_remapping``. Pinned by
  tests/test_generate_reports_characterization.py::test_build_ref_dict.
- ``fill_novel``: assigns NovelGene/NovelTr identifiers when no
  reference annotation is available. Used by ``annotate_one_group`` to
  handle the "all novel" branch.

Extracted from the original ``src/generate_reports.py`` "god class"
(IsoformAnnotator) so each utility is testable in isolation without
spinning up the full annotation orchestrator.
"""
from collections import defaultdict
from typing import Dict

import pandas as pd


def reverse_complement(seq: str) -> str:
    """Generate reverse complement of a DNA sequence.

    Args:
        seq: DNA sequence string. Case-insensitive -- input is uppercased
             before lookup so ``ATGC`` and ``atgc`` produce identical
             output.

    Returns:
        Reverse-complement sequence (uppercase).
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    rev_seq = seq[::-1]
    rev_comp = ''.join([complement.get(base, base) for base in rev_seq.upper()])
    return rev_comp


def build_ref_dict(ref_df: pd.DataFrame, include_term: bool = True) -> dict:
    """Build reference dictionary mapping gene identifiers to site positions.

    Args:
        ref_df: Reference annotation DataFrame.
        include_term: Whether to include terminal sites (TrStart/TrEnd)
                      in the site set. ``include_term=False`` is used by
                      ``novel_gene_remapping`` for the no-TrStart/TrEnd
                      matching path (only SSC splice sites count).

    Returns:
        Dictionary mapping ``"{GeneID}_{GeneName}"`` keys to lists of
        integer site positions. SSC string ``"200-300"`` is split into
        ``{200, 300}``; ``"nan"`` / empty strings are skipped.
    """
    ref_dict = defaultdict(set)
    for _, row in ref_df.iterrows():
        gene_id = str(row['GeneID'])
        gene_name = str(row['GeneName'])
        key = f'{gene_id}_{gene_name}'
        sites = {int(row['TrStart']), int(row['TrEnd'])} if include_term else set()
        ssc_str = str(row['SSC']).strip()
        if ssc_str and ssc_str != 'nan':
            sites |= {int(x) for x in ssc_str.split('-') if x}
        ref_dict[key].update(sites)
    return {k: list(v) for k, v in ref_dict.items()}


def fill_novel(df: pd.DataFrame) -> pd.DataFrame:
    """Assign NovelGene/NovelTr identifiers when no reference annotation is used.

    Used by ``annotate_one_group`` to assign TrID / GeneID / GeneName /
    TrStart_ref / TrEnd_ref when the entire group is novel (ref_anno is
    None, or the per-row merge produced all-NaN reference columns).

    The naming pattern is:

        GeneID      = "NovelGene{GID}"
        GeneName    = "NovelGene{GID}"      (mirrors GeneID for novel-only genes)
        TrID        = "NovelGene{GID}_Novel{uniqueTr}"
        TrStart_ref = TrStart                (self-reference)
        TrEnd_ref   = TrEnd                  (self-reference)

    Args:
        df: DataFrame with at minimum columns ``Group``, ``uniqueTr``,
            ``TrStart``, ``TrEnd``.

    Returns:
        New DataFrame with the five columns overwritten.
    """
    df = df.copy()
    for idx, row in df.iterrows():
        gid = row['Group']
        trid = row['uniqueTr']
        df.at[idx, 'GeneID'] = f'NovelGene{gid}'
        df.at[idx, 'GeneName'] = f'NovelGene{gid}'
        df.at[idx, 'TrID'] = f'NovelGene{gid}_Novel{trid}'
        df.at[idx, 'TrStart_ref'] = row['TrStart']
        df.at[idx, 'TrEnd_ref'] = row['TrEnd']
    return df