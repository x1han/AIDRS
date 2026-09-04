"""Integration smoke test for P1-3 fix (gene_grouping cluster()).

Verifies that GeneClustering.cluster() no longer raises TypeError when
self.num_processes is None (the default), and no longer raises ValueError
when num_processes=0 (instead coerces to a positive int).

NOTE: This test uses the real `src.*` import path so multiprocessing.Pool
spawn workers can re-import the module. The companion
tests/test_p1_hard_crash_batch.py exercises the other 4 P1 fixes but
skips this one because importlib.util.spec_from_file_location gives the
module a synthetic name that spawn-context workers cannot re-import.
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.gene_grouping import GeneClustering


def _make_df():
    return pd.DataFrame({
        'Chr': ['chr1', 'chr1', 'chr1', 'chr2', 'chr2'],
        'Strand': ['+', '+', '-', '+', '-'],
        'SSC': ['100-200', '300-400', '500-600', '700-800', '900-1000'],
    })


def test_cluster_with_default_none_num_processes():
    """Default constructor leaves num_processes=None; cluster() must coerce
    to a positive int instead of raising TypeError on min(None, n)."""
    gc = GeneClustering()  # default num_processes=None
    df = _make_df()
    out = gc.cluster(df)
    assert 'Group' in out.columns
    assert len(out) == 5


def test_cluster_with_zero_num_processes_coerces():
    """num_processes=0 must coerce to a positive int, not Pool(0)."""
    gc = GeneClustering(num_processes=0)
    df = _make_df()
    out = gc.cluster(df)
    assert 'Group' in out.columns
    assert len(out) == 5


def test_cluster_with_negative_num_processes_coerces():
    """num_processes=-1 must coerce to a positive int."""
    gc = GeneClustering(num_processes=-1)
    df = _make_df()
    out = gc.cluster(df)
    assert 'Group' in out.columns
    assert len(out) == 5


def test_cluster_with_explicit_num_processes_uses_it():
    """Sanity: explicit positive num_processes is honoured."""
    gc = GeneClustering(num_processes=2)
    df = _make_df()
    out = gc.cluster(df)
    assert 'Group' in out.columns
    assert len(out) == 5


if __name__ == '__main__':
    test_cluster_with_default_none_num_processes()
    test_cluster_with_zero_num_processes_coerces()
    test_cluster_with_negative_num_processes_coerces()
    test_cluster_with_explicit_num_processes_uses_it()
    print("All 4 P1-3 cluster smoke tests PASSED")