"""Unit tests for the per-read poly(A) (Dorado ``pt``) driven 3'-end estimator.

The per-read ``pt`` tag influences ONLY the 3' end of the assembled
transcript; no read is ever dropped from frequency/coverage/quantification.

Pinned contracts:

- A2' (cluster gate): clusters whose pt>0 support is below the literal
  threshold 3 are dropped; a kept cluster's representative is recomputed from
  its pt>0 reads only.
- A2'-b (all-noise path): when DBSCAN finds no valid cluster, the pt>0 reads
  vote (threshold 3) and the all-read mode is the fallback.
- pt=None and an all-zero pt reproduce the pre-A2' production result exactly.
- The pt column is only applied to the 3' side: '+' changes TrEnd only,
  '-' changes TrStart only.
- The pt values follow the DBSCAN sample: they are re-indexed with the
  sampled rows, never sampled independently.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.get_terminal_sites import TerminalSitesProcessor, A2_PRIME_MIN_PT_POS


def _series(values, pt=None):
    values = list(values)
    s = pd.Series(values)
    if pt is None:
        return s, None
    return s, pd.Series(list(pt), index=s.index)


# --- (a) the six-row construction receipt (true TES = 1000) -----------------

RECEIPT_CASES = [
    # (label, [(coordinate, n_reads, pt)], production_result, a2prime_result)
    ("no-displacement",
     [(1000, 50, 1), (1000, 50, 1)], [1000], [1000]),
    ("displacement-10bp-minority-40pct",
     [(1000, 50, 1), (990, 40, 0)], [1000], [1000]),
    ("displacement-10bp-majority-60pct",
     [(1000, 40, 1), (990, 60, 0)], [990], [1000]),
    ("displacement-40bp-minority-40pct",
     [(1000, 50, 1), (960, 40, 0)], [960, 1000], [1000]),
    ("displacement-40bp-majority-60pct",
     [(1000, 40, 1), (960, 60, 0)], [960, 1000], [1000]),
    ("displacement-40bp-minority-5pct",
     [(1000, 95, 1), (960, 5, 0)], [1000], [1000]),
]


@pytest.mark.parametrize("label,pops,production,a2prime", RECEIPT_CASES)
def test_receipt_rows(label, pops, production, a2prime):
    values = [v for v, n, _ in pops for _ in range(n)]
    pts = [p for _, n, p in pops for _ in range(n)]
    s, p = _series(values, pts)
    proc = TerminalSitesProcessor()

    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max')) == production
    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max', pt=p)) == a2prime


def test_receipt_discrete_cluster():
    """40 pt>0 reads spread over 1000/1001/1002 + 60 pt=0 reads at 1005.

    A per-coordinate vote would kill the true end (no single coordinate has
    40 supporters); A2' keeps the cluster and re-estimates from the pt>0
    reads, whose mode is 1002.
    """
    values = [1000] * 13 + [1001] * 13 + [1002] * 14 + [1005] * 60
    pts = [1] * 40 + [0] * 60
    s, p = _series(values, pts)
    proc = TerminalSitesProcessor()

    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max')) == [1005]
    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max', pt=p)) == [1002]


def test_threshold_is_the_literal_constant_three():
    assert A2_PRIME_MIN_PT_POS == 3


# --- (b)/(c) pt=None and an all-zero pt reproduce production ---------------

def test_pt_none_equals_production():
    proc = TerminalSitesProcessor()
    values = [1000] * 40 + [990] * 60 + [1200] * 30
    s, _ = _series(values)
    for mode in ("min", "max"):
        expected = list(proc.get_representative_sites_DBSCAN(s.copy(), mode=mode))
        got = list(proc.get_representative_sites_DBSCAN(s.copy(), mode=mode, pt=None))
        assert got == expected


def test_all_zero_pt_equals_production():
    proc = TerminalSitesProcessor()
    values = [1000] * 40 + [990] * 60 + [1200] * 30
    s, p = _series(values, [0] * len(values))
    for mode in ("min", "max"):
        expected = list(proc.get_representative_sites_DBSCAN(s.copy(), mode=mode))
        got = list(proc.get_representative_sites_DBSCAN(s.copy(), mode=mode, pt=p))
        assert got == expected


# --- (d) strand handling: only the 3' side moves ---------------------------

def _row(strand, starts, ends, pt):
    return {
        "Chr": "chr1", "Strand": strand, "SSC": "100-200", "Group": 0,
        "TrStart_reads": np.array(starts, dtype=np.int32),
        "TrEnd_reads": np.array(ends, dtype=np.int32),
        "read_polyA_len": np.array(pt, dtype=np.int32),
        "frequency": len(starts),
    }


def test_plus_strand_only_trend_changes():
    proc = TerminalSitesProcessor()
    starts = [1000] * 100
    ends = [1000] * 40 + [990] * 60
    pt = [1] * 40 + [0] * 60
    df = pd.DataFrame([_row("+", starts, ends, pt)])

    without = proc.get_terminal_sites_forChr(df.drop(columns=["read_polyA_len"]))
    with_pt = proc.get_terminal_sites_forChr(df)

    assert without["TrStart"].iloc[0] == with_pt["TrStart"].iloc[0]
    assert without["TrEnd"].iloc[0] == 990
    assert with_pt["TrEnd"].iloc[0] == 1000


def test_minus_strand_only_trstart_changes():
    proc = TerminalSitesProcessor()
    starts = [1000] * 40 + [1010] * 60
    ends = [2000] * 100
    pt = [1] * 40 + [0] * 60
    df = pd.DataFrame([_row("-", starts, ends, pt)])

    without = proc.get_terminal_sites_forChr(df.drop(columns=["read_polyA_len"]))
    with_pt = proc.get_terminal_sites_forChr(df)

    assert without["TrEnd"].iloc[0] == with_pt["TrEnd"].iloc[0]
    assert without["TrStart"].iloc[0] == 1010
    assert with_pt["TrStart"].iloc[0] == 1000


def test_pt_never_applied_to_the_five_prime_side():
    """A pt>0 majority on the 5' side must not move the 5' end."""
    proc = TerminalSitesProcessor()
    starts = [1000] * 60 + [1010] * 40
    ends = [2000] * 100
    pt = [0] * 60 + [1] * 40
    df = pd.DataFrame([_row("+", starts, ends, pt)])

    without = proc.get_terminal_sites_forChr(df.drop(columns=["read_polyA_len"]))
    with_pt = proc.get_terminal_sites_forChr(df)

    assert with_pt["TrStart"].iloc[0] == without["TrStart"].iloc[0]
    assert with_pt["TrEnd"].iloc[0] == without["TrEnd"].iloc[0]


# --- (e) the pt column follows the DBSCAN sample ---------------------------

def test_pt_follows_the_dbscan_sample():
    """The pt>0 mode sits outside the DBSCAN sample => production behaviour.

    The pt values must be re-indexed with the sampled rows. If they were
    taken from the full set (or sampled independently), the pt>0 reads at
    9000 -- which are not part of the DBSCAN sample -- would leak into the
    estimate.
    """
    proc = TerminalSitesProcessor()
    n = 2000
    values = np.full(n, 1000, dtype=np.int64)

    series = pd.Series(values)
    sampled_index = set(series.sample(n=proc.cluster_group_size, random_state=42).index)
    outside = [i for i in range(n) if i not in sampled_index]
    assert len(outside) >= 3

    values[outside[0]] = 9000
    values[outside[1]] = 9000
    values[outside[2]] = 9000
    pts = np.zeros(n, dtype=np.int32)
    pts[outside[0]] = 1
    pts[outside[1]] = 1
    pts[outside[2]] = 1

    s, p = _series(values, pts)
    assert set(p[p > 0].index).isdisjoint(sampled_index)

    expected = list(proc.get_representative_sites_DBSCAN(series.copy(), mode="max"))
    got = list(proc.get_representative_sites_DBSCAN(s.copy(), mode="max", pt=p))
    assert expected == [1000]
    assert got == expected


def test_pt_is_not_sampled_independently():
    """A pt>0 majority inside the sample is honoured; the alignment holds."""
    proc = TerminalSitesProcessor()
    n = 2000
    values = np.full(n, 1000, dtype=np.int64)
    pts = np.zeros(n, dtype=np.int32)

    series = pd.Series(values)
    sampled_index = sorted(series.sample(n=proc.cluster_group_size, random_state=42).index)

    # 400 reads at 2000, all inside the sampled subset, all pt>0
    for i in sampled_index[:400]:
        values[i] = 2000
        pts[i] = 1

    s, p = _series(values, pts)
    got = list(proc.get_representative_sites_DBSCAN(s.copy(), mode="max", pt=p))
    assert got == [2000]


# --- the plumbing hop: the Stage-1.1 parquet carries the per-read pt -------

def test_process_data_keeps_read_polyA_len_parallel(tmp_path):
    """process_data must carry the per-read pt into the parquet, parallel to
    the read arrays (the groupby .agg keeps the within-group row order)."""
    from src.common import process_data

    flnc = tmp_path / "sample_flnc.ssc"
    rows = []
    # group (chr1,+,1-2): four reads, pt = 5, 0, 7, 0
    for i, (s, e, pt) in enumerate([(10, 20, 5), (11, 21, 0), (12, 22, 7), (13, 23, 0)]):
        rows.append(f"read{i}\tchr1\t+\t{s}\t{e}\t1-2\t0.99\t1.0\t{pt}")
    # group (chr1,+,3-4): two reads, pt = 0, 9
    for i, (s, e, pt) in enumerate([(30, 40, 0), (31, 41, 9)]):
        rows.append(f"readB{i}\tchr1\t+\t{s}\t{e}\t3-4\t0.99\t1.0\t{pt}")
    flnc.write_text("\n".join(rows) + "\n")

    count = tmp_path / "sample_ssc.count"
    count.write_text("c1\tchr1\t+\t1-2\t\nc2\tchr1\t+\t3-4\t\n")

    out = tmp_path / "sample.ssc_flnc.parquet"
    process_data(str(flnc), str(count), str(out))

    df = pd.read_parquet(out)
    assert "read_polyA_len" in df.columns
    assert "polyA_len" not in df.columns
    by_ssc = df.set_index("SSC")
    first = by_ssc.loc["1-2"]
    assert list(first["read_polyA_len"]) == [5, 0, 7, 0]
    assert len(first["read_polyA_len"]) == len(first["TrStart_reads"]) == len(first["TrEnd_reads"])
    second = by_ssc.loc["3-4"]
    assert list(second["read_polyA_len"]) == [0, 9]
    assert len(second["read_polyA_len"]) == len(second["TrStart_reads"]) == len(second["TrEnd_reads"])


# --- (f) the A2' cluster-DROP branch (a cluster with <3 pt>0 reads is dropped
#         when another cluster qualifies) -----------------------------------

def test_a2prime_drops_the_cluster_without_pt_support():
    """Two clusters: the pt>0 one qualifies, the pt=0 one does not.

    Without pt the group has two candidate ends; A2' keeps only the cluster
    that carries pt>0 support. A refactor that inverts the keep/drop rule
    would silently change this, so the branch needs its own test.
    """
    proc = TerminalSitesProcessor()
    values = [1000] * 50 + [2000] * 40
    pts = [1] * 50 + [0] * 40
    s, p = _series(values, pts)

    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max')) == [1000, 2000]
    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max', pt=p)) == [1000]


def test_a2prime_keeps_all_clusters_when_none_qualifies():
    """No cluster reaches 3 pt>0 reads => every cluster is kept, unchanged."""
    proc = TerminalSitesProcessor()
    values = [1000] * 50 + [2000] * 40
    pts = [1] * 2 + [0] * 88
    s, p = _series(values, pts)

    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max')) == [1000, 2000]
    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max', pt=p)) == [1000, 2000]


# --- (g) the 5' side never moves, on randomised groups ---------------------

def test_five_prime_side_never_moves_on_randomised_groups():
    """Randomised groups: adding pt must not move the 5'-side coordinate.

    The invariant "pt only affects the 3' side" is enforced by the call site
    in get_terminal_sites_forChr; this test exercises it on many random
    shapes so a swapped strand/mode wiring cannot pass unnoticed.
    """
    rng = np.random.default_rng(20261124)
    proc = TerminalSitesProcessor()
    for _ in range(200):
        n = int(rng.integers(20, 120))
        starts = rng.integers(1000, 2000, size=n).astype(np.int32)
        ends = starts + rng.integers(1, 500, size=n).astype(np.int32)
        pt = rng.integers(0, 2, size=n).astype(np.int32)
        strand = '+' if rng.random() < 0.5 else '-'
        df = pd.DataFrame([_row(strand, starts.tolist(), ends.tolist(), pt.tolist())])

        without = proc.get_terminal_sites_forChr(df.drop(columns=["read_polyA_len"]))
        with_pt = proc.get_terminal_sites_forChr(df)

        five = 'TrStart' if strand == '+' else 'TrEnd'
        assert with_pt[five].iloc[0] == without[five].iloc[0]


# --- (h) the fail-loud guard on the per-sample pt column --------------------

def test_missing_pt_column_fails_loud_and_names_the_sample():
    """A sample that kept its read arrays but lost its pt column must raise.

    Without this guard the whole poly(A)-aware re-estimation is silently
    disabled for that sample (no error, no log line, output unchanged).
    """
    import pytest
    from src.common import assert_pt_parallel

    df = pd.DataFrame({
        'S1_TrStart_reads': [np.array([1, 2, 3], dtype=np.int32)],
        'S1_read_polyA_len': [np.array([0, 1, 0], dtype=np.int32)],
    })
    assert assert_pt_parallel(df, ['S1_TrStart_reads']) == ['S1_read_polyA_len']

    # (1) the pt column is gone entirely
    df_missing = df.drop(columns=['S1_read_polyA_len'])
    with pytest.raises(RuntimeError, match='S1'):
        assert_pt_parallel(df_missing, ['S1_TrStart_reads'])

    # (2) the pt column has the wrong length
    df_misaligned = df.copy()
    df_misaligned['S1_read_polyA_len'] = [np.array([0, 1], dtype=np.int32)]
    with pytest.raises(RuntimeError, match='S1'):
        assert_pt_parallel(df_misaligned, ['S1_TrStart_reads'])


# --- (i) the A2'-b all-noise path ------------------------------------------
# The all-noise path (DBSCAN finds no valid cluster) is live production code:
# any group with fewer than 20 reads can never form a cluster (min_neighbors
# is max(10% of the sample, 20)), and the merged frames routinely contain such
# groups. It must be pinned by a test, not only by the docstring.

def test_a2prime_b_uses_the_pt_pos_reads_on_the_all_noise_path():
    """10 reads in two tight groups of 5 => DBSCAN calls everything noise.

    Production (no pt): the mode over all reads => [1200] (the tie rule for
    mode='max' takes the larger of the two tied modes).
    With pt: 5 of the reads at 1000 carry pt>0 => A2'-b votes with them => [1000].
    """
    proc = TerminalSitesProcessor()
    values = [1000] * 5 + [1200] * 5
    pts = [1] * 5 + [0] * 5
    s, p = _series(values, pts)

    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max')) == [1200]
    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max', pt=p)) == [1000]


def test_a2prime_b_falls_back_when_the_pt_support_is_below_the_threshold():
    """Same group, but only 2 pt>0 reads => below the constant 3 => the
    all-read mode is kept."""
    proc = TerminalSitesProcessor()
    values = [1000] * 5 + [1200] * 5
    pts = [1] * 2 + [0] * 8
    s, p = _series(values, pts)

    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max')) == [1200]
    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max', pt=p)) == [1200]


# --- (j) the threshold is behaviourally pinned to the literal constant 3 ----
# A constant that is only asserted to be 3 is not pinned: rewriting the
# comparison as max(3, int(n*0.1)) keeps the constant at 3 and changes the
# behaviour. These two cases discriminate.

def test_threshold_three_keeps_a_cluster_with_exactly_three_pt_pos_reads():
    """Exactly 3 pt>0 reads in the first cluster => the cluster qualifies.

    A threshold of 4 would drop it (and then no cluster qualifies, so both
    clusters are kept) => the result would be [1000, 2000] instead.
    """
    proc = TerminalSitesProcessor()
    values = [1000] * 50 + [2000] * 40
    pts = [1] * 3 + [0] * 87
    s, p = _series(values, pts)

    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max')) == [1000, 2000]
    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max', pt=p)) == [1000]


def test_threshold_is_not_dependent_on_the_group_size():
    """4 pt>0 reads in a 90-read group qualify => the threshold is an
    absolute count, not a fraction of the group.

    A size-dependent rule (int(90 * 0.1) = 9) would drop this cluster.
    """
    proc = TerminalSitesProcessor()
    values = [1000] * 50 + [2000] * 40
    pts = [1] * 4 + [0] * 86
    s, p = _series(values, pts)

    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max')) == [1000, 2000]
    assert list(proc.get_representative_sites_DBSCAN(s.copy(), mode='max', pt=p)) == [1000]
