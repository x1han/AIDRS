"""Unit tests for src.gene_grouping.compute_is_intergenic_or_antisense.

Covers Pillar 1 of the Stage 2.5b 5-pillar funnel (per
p3_single_exon_5_pillar.md): a single-exon row passes only if it does NOT
overlap any same-strand multi-exon transcript on [TrStart, TrEnd].
"""
import sys
import importlib.util
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    'gene_grouping_under_test', REPO_ROOT / 'src' / 'gene_grouping.py'
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
compute = mod.compute_is_intergenic_or_antisense


def test_empty_dataframe_returns_column():
    out = compute(pd.DataFrame())
    assert 'is_intergenic_or_antisense' in out.columns
    assert len(out) == 0


def test_none_input_returns_none():
    # The function explicitly guards None input rather than crashing on
    # .assign() (which would raise AttributeError on NoneType).
    assert compute(None) is None


def test_no_df_multi_marks_all_true():
    df = pd.DataFrame({
        'Chr': ['chr1', 'chr2'],
        'Strand': ['+', '-'],
        'TrStart': [100, 200],
        'TrEnd': [200, 300],
    })
    out = compute(df)
    assert out['is_intergenic_or_antisense'].tolist() == [True, True]


def test_overlap_marks_false():
    # single row 0 (chr1+, 100-200) overlaps multi row 0 (chr1+, 50-150) → False
    # single row 1 (chr1+, 500-600) does NOT overlap either multi → True
    df_single = pd.DataFrame({
        'Chr': ['chr1', 'chr1'],
        'Strand': ['+', '+'],
        'TrStart': [100, 500],
        'TrEnd': [200, 600],
    })
    df_multi = pd.DataFrame({
        'Chr': ['chr1', 'chr1'],
        'Strand': ['+', '+'],
        'TrStart': [50, 700],
        'TrEnd': [150, 800],
    })
    out = compute(df_single, df_multi=df_multi)
    assert out['is_intergenic_or_antisense'].tolist() == [False, True]


def test_strand_isolation():
    # Overlap on opposite strand must NOT mark False (antisense is allowed)
    df_single = pd.DataFrame({
        'Chr': ['chr1'],
        'Strand': ['+'],
        'TrStart': [100],
        'TrEnd': [200],
    })
    df_multi = pd.DataFrame({
        'Chr': ['chr1'],
        'Strand': ['-'],  # opposite strand → no overlap penalty
        'TrStart': [100],
        'TrEnd': [200],
    })
    out = compute(df_single, df_multi=df_multi)
    assert out['is_intergenic_or_antisense'].tolist() == [True]


def test_chromosome_isolation():
    # Overlap on different chromosome must NOT mark False
    df_single = pd.DataFrame({
        'Chr': ['chr1'],
        'Strand': ['+'],
        'TrStart': [100],
        'TrEnd': [200],
    })
    df_multi = pd.DataFrame({
        'Chr': ['chr2'],  # different chr
        'Strand': ['+'],
        'TrStart': [100],
        'TrEnd': [200],
    })
    out = compute(df_single, df_multi=df_multi)
    assert out['is_intergenic_or_antisense'].tolist() == [True]


def test_endpoint_boundary_is_overlap():
    # Inclusive boundaries: single ending exactly at multi start is overlap.
    df_single = pd.DataFrame({
        'Chr': ['chr1'],
        'Strand': ['+'],
        'TrStart': [100],
        'TrEnd': [200],
    })
    df_multi = pd.DataFrame({
        'Chr': ['chr1'],
        'Strand': ['+'],
        'TrStart': [200],
        'TrEnd': [300],
    })
    out = compute(df_single, df_multi=df_multi)
    assert out['is_intergenic_or_antisense'].tolist() == [False]


def test_swapped_coordinates_are_normalised():
    # Pipeline usually guarantees TrStart < TrEnd; defensive swap on antisense
    # scaffolds must not break the overlap predicate.
    df_single = pd.DataFrame({
        'Chr': ['chr1'],
        'Strand': ['+'],
        'TrStart': [200],
        'TrEnd': [100],  # swapped
    })
    df_multi = pd.DataFrame({
        'Chr': ['chr1'],
        'Strand': ['+'],
        'TrStart': [50],
        'TrEnd': [150],
    })
    out = compute(df_single, df_multi=df_multi)
    # After min/max normalisation, single is [100,200]; multi is [50,150].
    # They overlap → False.
    assert out['is_intergenic_or_antisense'].tolist() == [False]


def test_missing_columns_warns_and_defaults_true():
    # If df_single is missing required columns, function logs warning and
    # returns all-True (fail-open: rather include a false positive than
    # silently drop a possibly-novel isoform).
    df_single = pd.DataFrame({'Chr': ['chr1'], 'TrStart': [100]})  # no Strand/TrEnd
    df_multi = pd.DataFrame({
        'Chr': ['chr1'],
        'Strand': ['+'],
        'TrStart': [100],
        'TrEnd': [200],
    })
    out = compute(df_single, df_multi=df_multi)
    assert out['is_intergenic_or_antisense'].tolist() == [True]


def test_input_not_mutated():
    df_single = pd.DataFrame({
        'Chr': ['chr1'],
        'Strand': ['+'],
        'TrStart': [100],
        'TrEnd': [200],
    })
    df_multi = pd.DataFrame({
        'Chr': ['chr1'],
        'Strand': ['+'],
        'TrStart': [50],
        'TrEnd': [150],
    })
    original_cols = df_single.columns.tolist()
    _ = compute(df_single, df_multi=df_multi)
    # Pillar-1 anti-pattern: callers depend on input df_single not being
    # mutated in-place (otherwise the schema-alignment block in aidrs.py
    # would see a half-mutated row).
    assert df_single.columns.tolist() == original_cols
    assert 'is_intergenic_or_antisense' not in df_single.columns


if __name__ == '__main__':
    test_empty_dataframe_returns_column()
    test_none_input_returns_none()
    test_no_df_multi_marks_all_true()
    test_overlap_marks_false()
    test_strand_isolation()
    test_chromosome_isolation()
    test_endpoint_boundary_is_overlap()
    test_swapped_coordinates_are_normalised()
    test_missing_columns_warns_and_defaults_true()
    test_input_not_mutated()
    print("All 10 tests PASSED")