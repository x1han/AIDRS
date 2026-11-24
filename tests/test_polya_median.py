"""Unit tests for the per-sample poly(A) aggregation (unweighted median).

Pinned contracts:

- ``polyA_frac`` is the UNWEIGHTED MEDIAN of the per-sample ratios of the
  samples that carry measured poly(A) evidence. A single abnormally deep
  library must not be able to drag the cohort's poly(A) evidence down.
- A sample qualifies only when it has at least POLYA_MIN_READS reads at the
  key AND at least one read with a measured poly(A) tail.
- The median convention is numpy's: with an EVEN number of candidates the
  median is the MEAN of the two middle values.
- With no qualifying sample the value falls back to today's pooled ratio
  sum(polyA_freq) / sum(raw_read_freq).
- The aggregation happens BEFORE the merge onto the model frame, and the
  merge must not change the model's row count.
- ``polyA_frac_pooled`` / ``polyA_n_samples`` stay unregistered, so the
  delivered assessment column set does not change.
- The delivered per-sample sidecar carries the per-key aggregate as well
  (``polyA_frac_model``, ``polyA_frac_pooled``, ``polyA_n_samples``,
  ``estimator``), so the audit trail survives the deletion of ``temp/`` and a
  consumer can tell which estimator produced the delivered ``polyA_frac``.
"""
import argparse
import glob
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.polya_annotation import (
    POLYA_MERGE_COLS,
    POLYA_MIN_READS,
    POLYA_PER_SAMPLE_SIDECAR,
    aggregate_polya,
    polyAnnotator,
)
from src.aidrs_runtime.column_registry import resolve_assessment_columns


KEY = dict(Chr="chr1", Strand="+", SSC="1-2", TrStart=100, TrEnd=200)


def _per_sample(rows):
    """rows: list of (sample, polyA_freq, raw_read_freq) for ONE key."""
    out = []
    for sample, polyA_freq, raw_read_freq in rows:
        out.append(dict(KEY, sample=sample, polyA_freq=polyA_freq,
                        raw_read_freq=raw_read_freq))
    return pd.DataFrame(out)


# --- (a) one deep abnormal library cannot drag the cohort down ------------

def test_deep_abnormal_library_does_not_drag_the_median_down():
    """9 normal libraries (10 reads, 90 % measured) + 1 abnormal deep
    library (1000 reads, 5 % measured)."""
    rows = [(f"S{i}", 9, 10) for i in range(9)]
    rows.append(("ABNORMAL", 50, 1000))

    out = aggregate_polya(_per_sample(rows))
    assert len(out) == 1

    polyA_frac = out["polyA_frac"].iloc[0]
    pooled = out["polyA_frac_pooled"].iloc[0]

    assert polyA_frac == pytest.approx(0.9)
    assert pooled < 0.15
    assert polyA_frac > pooled
    assert out["polyA_n_samples"].iloc[0] == 10


def test_the_pooled_ratio_would_have_been_wrong():
    """The same data under the old read-depth-weighted pooling."""
    rows = [(f"S{i}", 9, 10) for i in range(9)]
    rows.append(("ABNORMAL", 50, 1000))
    ps = _per_sample(rows)
    pooled = ps["polyA_freq"].sum() / ps["raw_read_freq"].sum()

    out = aggregate_polya(ps)
    assert out["polyA_frac"].iloc[0] != pytest.approx(pooled)
    assert out["polyA_frac_pooled"].iloc[0] == pytest.approx(pooled)


# --- (b) candidate selection ----------------------------------------------

def test_a_tail_less_sample_is_excluded_from_the_candidates():
    """A deep library with NO measured tail contributes nothing."""
    rows = [("S1", 9, 10), ("S2", 9, 10), ("NOPTA", 0, 1000)]

    out = aggregate_polya(_per_sample(rows))
    assert out["polyA_frac"].iloc[0] == pytest.approx(0.9)
    assert out["polyA_n_samples"].iloc[0] == 2


def test_a_sample_below_the_read_floor_is_excluded_from_the_candidates():
    """raw_read_freq < POLYA_MIN_READS => noise level => not a candidate."""
    rows = [("S1", 9, 10), ("S2", 9, 10), ("NOISY", 1, POLYA_MIN_READS - 1)]

    out = aggregate_polya(_per_sample(rows))
    assert out["polyA_frac"].iloc[0] == pytest.approx(0.9)
    assert out["polyA_n_samples"].iloc[0] == 2


def test_without_any_candidate_the_pooled_ratio_is_used():
    """The only sample is below the read floor => fallback to the pooled
    ratio (today's semantics)."""
    rows = [("S1", 1, POLYA_MIN_READS - 1)]

    out = aggregate_polya(_per_sample(rows))
    assert out["polyA_frac"].iloc[0] == pytest.approx(1 / (POLYA_MIN_READS - 1))
    assert out["polyA_frac"].iloc[0] == pytest.approx(out["polyA_frac_pooled"].iloc[0])
    assert out["polyA_n_samples"].iloc[0] == 0


def test_a_tail_less_cohort_stays_at_zero():
    """No sample carries a measured tail => 0 (today's State-2 semantics),
    not a crash and not a NaN."""
    rows = [("S1", 0, 100), ("S2", 0, 100)]

    out = aggregate_polya(_per_sample(rows))
    assert out["polyA_frac"].iloc[0] == pytest.approx(0.0)
    assert out["polyA_n_samples"].iloc[0] == 0


# --- (c) the even-n median convention -------------------------------------

def test_even_number_of_candidates_takes_the_mean_of_the_two_middle_values():
    """numpy convention: median([0.2, 0.8]) == 0.5."""
    rows = [("S1", 2, 10), ("S2", 8, 10)]

    out = aggregate_polya(_per_sample(rows))
    assert out["polyA_frac"].iloc[0] == pytest.approx(0.5)
    assert out["polyA_n_samples"].iloc[0] == 2


def test_odd_number_of_candidates_takes_the_middle_value():
    rows = [("S1", 1, 10), ("S2", 5, 10), ("S3", 9, 10)]

    out = aggregate_polya(_per_sample(rows))
    assert out["polyA_frac"].iloc[0] == pytest.approx(0.5)
    assert out["polyA_n_samples"].iloc[0] == 3


# --- (d) a single sample is numerically identical to today's pooling ------

def test_single_sample_median_equals_the_pooled_ratio():
    """The C107 100k golden is a single-BAM run: the median and the pooled
    ratio must be numerically identical there, so the golden cannot move."""
    rows = [("S1", 7, 11)]

    out = aggregate_polya(_per_sample(rows))
    assert out["polyA_frac"].iloc[0] == pytest.approx(out["polyA_frac_pooled"].iloc[0])
    assert out["polyA_frac"].iloc[0] == pytest.approx(7 / 11)


# --- (e) the aggregation is one row per key and never multiplies rows -----

def test_the_aggregate_frame_holds_one_row_per_key():
    rows = [("S1", 7, 11), ("S2", 3, 9)]
    ps = pd.concat([
        _per_sample(rows),
        _per_sample([("S3", 5, 10)]).assign(SSC="3-4", TrStart=300, TrEnd=400),
    ], ignore_index=True)

    out = aggregate_polya(ps)
    assert len(out) == 2
    assert not out.duplicated(subset=POLYA_MERGE_COLS).any()


def test_a_merge_that_would_multiply_rows_is_rejected():
    df = pd.DataFrame([dict(KEY, Group=0), dict(KEY, Group=1)])
    duplicated = pd.DataFrame([
        dict(KEY, polyA_frac=0.5, polyA_frac_pooled=0.5, polyA_n_samples=1),
        dict(KEY, polyA_frac=0.5, polyA_frac_pooled=0.5, polyA_n_samples=1),
    ])

    with pytest.raises(RuntimeError, match="row count"):
        polyAnnotator.add_polya_frac(df, duplicated)


def test_the_merge_preserves_the_model_row_count():
    df = pd.DataFrame([dict(KEY, Group=0), dict(KEY, Group=1)])
    stats = pd.DataFrame([dict(KEY, polyA_frac=0.5, polyA_frac_pooled=0.5,
                               polyA_n_samples=1)])

    out = polyAnnotator.add_polya_frac(df, stats)
    assert len(out) == 2
    assert list(out["polyA_frac"]) == [0.5, 0.5]


# --- (f) the delivered assessment column set does not change --------------

def test_the_new_audit_columns_are_not_registered():
    """polyA_frac is a CORE column (its VALUES may change), but
    polyA_frac_pooled / polyA_n_samples must not enter the delivered
    assessment schema."""
    base = pd.DataFrame([dict(KEY, polyA_frac=0.5, TrID="t1")])
    extended = base.assign(polyA_frac_pooled=0.4, polyA_n_samples=3)

    assert resolve_assessment_columns(base) == resolve_assessment_columns(extended)
    assert "polyA_frac" in resolve_assessment_columns(extended)
    assert "polyA_frac_pooled" not in resolve_assessment_columns(extended)
    assert "polyA_n_samples" not in resolve_assessment_columns(extended)


# --- (g) the per-sample sidecar -------------------------------------------

def _args(output_dir):
    return argparse.Namespace(
        output=str(output_dir),
        terminal_tolerance=50,
        ss_tolerance=15,
        min_aln_coverage=None,
        min_aln_identity=None,
    )


def _write_flnc(path, n_measured, n_missing=0):
    rows = []
    for i in range(n_measured):
        rows.append(f"read{i}\tchr1\t+\t100\t200\t1-2\t0.99\t1.0\t80")
    for i in range(n_missing):
        j = n_measured + i
        rows.append(f"readX{j}\tchr1\t+\t100\t200\t1-2\t0.99\t1.0\t-2")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


def _model_frame():
    return pd.DataFrame([dict(KEY, Group=0, frequency=1)])


def test_the_per_sample_sidecar_is_written_next_to_the_outputs(tmp_path):
    out_dir = tmp_path / "out"
    (out_dir / "temp").mkdir(parents=True)
    _write_flnc(out_dir / "temp" / "S1_flnc.ssc", 10)

    anno = polyAnnotator(_args(out_dir))
    df = anno.anno_polya(_model_frame(), [str(out_dir / "temp" / "S1_flnc.ssc")])

    sidecar = out_dir / POLYA_PER_SAMPLE_SIDECAR
    assert sidecar.exists()
    table = pd.read_csv(sidecar, sep="\t")
    assert list(table.columns) == POLYA_MERGE_COLS + [
        "sample", "polyA_freq", "raw_read_freq", "polyA_frac",
        "polyA_frac_model", "polyA_frac_pooled", "polyA_n_samples", "estimator"]
    assert len(table) == 1
    assert table["sample"].iloc[0] == "S1"
    assert table["polyA_freq"].iloc[0] == 10
    assert table["polyA_frac"].iloc[0] == pytest.approx(1.0)
    assert df["polyA_frac"].iloc[0] == pytest.approx(1.0)


def test_the_sidecar_carries_the_per_key_aggregate(tmp_path):
    """The audit trail must survive the delivery: the sidecar carries the
    value the model received, the pooled value, the number of qualifying
    samples and which estimator was used."""
    out_dir = tmp_path / "out"
    (out_dir / "temp").mkdir(parents=True)
    _write_flnc_mixed(out_dir / "temp" / "S1_flnc.ssc", 10)
    _write_flnc_mixed(out_dir / "temp" / "S2_flnc.ssc", 100, n_missing=900)

    anno = polyAnnotator(_args(out_dir))
    df = anno.anno_polya(
        _model_frame(),
        [str(out_dir / "temp" / "S1_flnc.ssc"), str(out_dir / "temp" / "S2_flnc.ssc")],
    )

    table = pd.read_csv(out_dir / POLYA_PER_SAMPLE_SIDECAR, sep="\t")
    # one row per (key, sample): the merge must not multiply rows
    assert len(table) == 2
    assert not table.duplicated(subset=POLYA_MERGE_COLS + ["sample"]).any()

    # the per-key aggregate is on EVERY row of the key
    assert table["polyA_frac_model"].notna().all()
    assert table["polyA_frac_model"].nunique() == 1
    assert table["polyA_frac_model"].iloc[0] == pytest.approx(df["polyA_frac"].iloc[0])
    assert table["polyA_frac_pooled"].iloc[0] == pytest.approx(
        df["polyA_frac_pooled"].iloc[0])
    assert table["polyA_n_samples"].iloc[0] == df["polyA_n_samples"].iloc[0]
    assert set(table["estimator"].unique()) == {"median"}

    # the per-sample column still holds the PER-SAMPLE ratio (1.0 and 0.1),
    # not the cohort aggregate (0.55)
    ratios = dict(zip(table["sample"], table["polyA_frac"]))
    assert ratios["S1"] == pytest.approx(1.0)
    assert ratios["S2"] == pytest.approx(0.1)
    assert df["polyA_frac"].iloc[0] == pytest.approx(0.55)


def test_the_sidecar_flags_the_pooled_fallback(tmp_path):
    """No sample qualifies (every sample is below the read floor) => the
    value is the pooled fallback, and the sidecar must say so."""
    out_dir = tmp_path / "out"
    (out_dir / "temp").mkdir(parents=True)
    _write_flnc_mixed(out_dir / "temp" / "S1_flnc.ssc", POLYA_MIN_READS - 2)

    anno = polyAnnotator(_args(out_dir))
    df = anno.anno_polya(_model_frame(), [str(out_dir / "temp" / "S1_flnc.ssc")])

    table = pd.read_csv(out_dir / POLYA_PER_SAMPLE_SIDECAR, sep="\t")
    assert table["estimator"].iloc[0] == "pooled_fallback"
    assert table["polyA_n_samples"].iloc[0] == 0
    assert table["polyA_frac_model"].iloc[0] == pytest.approx(df["polyA_frac"].iloc[0])


def test_the_sidecar_is_not_picked_up_by_the_flnc_correct_glob(tmp_path):
    out_dir = tmp_path / "out"
    (out_dir / "temp").mkdir(parents=True)
    _write_flnc(out_dir / "temp" / "S1_flnc.ssc", 10)

    anno = polyAnnotator(_args(out_dir))
    anno.anno_polya(_model_frame(), [str(out_dir / "temp" / "S1_flnc.ssc")])

    matched = glob.glob(os.path.join(str(out_dir), "temp", "*_flnc_correct.ssc"))
    assert matched
    assert not any(POLYA_PER_SAMPLE_SIDECAR in m for m in matched)
    assert not glob.glob(os.path.join(str(out_dir), "temp", f"*{POLYA_PER_SAMPLE_SIDECAR}*"))


def test_the_no_pt_auto_skip_path_writes_no_sidecar(tmp_path):
    """The auto-skip path (no 'pt' tag in any BAM) must not emit a sidecar:
    an all-zero sidecar would be indistinguishable from "measured zero"
    evidence, which is exactly the confusion the auto-skip avoids."""
    out_dir = tmp_path / "out"
    (out_dir / "temp").mkdir(parents=True)
    _write_flnc(out_dir / "temp" / "S1_flnc.ssc", 0, n_missing=10)

    anno = polyAnnotator(_args(out_dir))
    anno.correct_flnc_only(_model_frame(), [str(out_dir / "temp" / "S1_flnc.ssc")])

    assert not (out_dir / POLYA_PER_SAMPLE_SIDECAR).exists()


# --- (h) the multi-sample path through anno_polya --------------------------

def _write_flnc_mixed(path, n_measured, n_missing=0):
    """n_measured reads with a measured tail, n_missing reads without."""
    rows = []
    for i in range(n_measured):
        rows.append(f"m{i}\tchr1\t+\t100\t200\t1-2\t0.99\t1.0\t80")
    for i in range(n_missing):
        j = n_measured + i
        rows.append(f"x{j}\tchr1\t+\t100\t200\t1-2\t0.99\t1.0\t-2")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


def test_multi_sample_median_through_anno_polya(tmp_path):
    """Three libraries: two at 100 % measured tails, one deep library with
    only 10 % measured tails. The delivered polyA_frac must be the median
    (1.0), not the pooled ratio."""
    out_dir = tmp_path / "out"
    (out_dir / "temp").mkdir(parents=True)
    _write_flnc_mixed(out_dir / "temp" / "S1_flnc.ssc", 10)
    _write_flnc_mixed(out_dir / "temp" / "S2_flnc.ssc", 10)
    _write_flnc_mixed(out_dir / "temp" / "S3_flnc.ssc", 100, n_missing=900)

    anno = polyAnnotator(_args(out_dir))
    df = anno.anno_polya(
        _model_frame(),
        [str(out_dir / "temp" / f) for f in ("S1_flnc.ssc", "S2_flnc.ssc", "S3_flnc.ssc")],
    )

    assert df["polyA_frac"].iloc[0] == pytest.approx(1.0)
    assert df["polyA_frac_pooled"].iloc[0] == pytest.approx(120 / 1020)
    assert df["polyA_n_samples"].iloc[0] == 3

    table = pd.read_csv(out_dir / POLYA_PER_SAMPLE_SIDECAR, sep="\t")
    assert sorted(table["sample"].unique()) == ["S1", "S2", "S3"]
    assert len(table) == 3
