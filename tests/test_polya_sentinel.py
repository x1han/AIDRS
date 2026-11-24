"""Unit tests for the per-read poly(A) sentinel encoding (Dorado ``pt``).

Pinned contracts:

- The four Dorado states stay DISTINCT in the per-read ``polyA_len``
  column: a measured tail (> 0), ``pt == 0`` (anchor found, length
  inestimable), ``pt == -1`` (anchor not found) and "no ``pt`` tag at all".
  A collapsed encoding (everything non-positive -> 0) makes "measured
  zero" indistinguishable from "never measured".
- Only a strictly positive value counts as a MEASURED poly(A) tail. The
  ``polyA_freq`` predicate must therefore be ``> 0`` and not ``!= 0``.
- The per-BAM ``has_pt`` flag stays keyed on tag PRESENCE (not on the
  encoded value): that part of the change is not a behaviour change.
"""
import argparse
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bam2ssc import (
    POLYA_LEN_TAG_MISSING,
    POLYA_LEN_ANCHOR_NOT_FOUND,
    encode_polya_len,
)
from src.polya_annotation import polyAnnotator


# --- (a) the encoder: four distinct states --------------------------------

def test_encoder_distinguishes_the_four_states():
    measured = encode_polya_len([("NM", 0, "i"), ("pt", 87, "i")])
    inestimable = encode_polya_len([("pt", 0, "i")])
    anchor_missing = encode_polya_len([("pt", -1, "i")])
    tag_absent = encode_polya_len([("NM", 0, "i")])

    assert measured == 87
    assert inestimable == 0
    assert anchor_missing == POLYA_LEN_ANCHOR_NOT_FOUND == -1
    assert tag_absent == POLYA_LEN_TAG_MISSING == -2

    # the four states must not collapse onto each other
    assert len({measured, inestimable, anchor_missing, tag_absent}) == 4


def test_encoder_ignores_non_pt_tags():
    assert encode_polya_len([("NM", 3, "i"), ("MD", 10, "Z")]) == POLYA_LEN_TAG_MISSING


def test_encoder_uses_the_first_pt_tag_only():
    assert encode_polya_len([("pt", 5, "i"), ("pt", 9, "i")]) == 5


# --- (b) the "has a measured tail" predicate ------------------------------

def _flnc_rows(n_measured, n_inestimable, n_anchor_missing, n_tag_absent):
    """Build the rows of a `{sample}_flnc.ssc` file (read_flnc layout)."""
    rows = []
    states = (
        [(100, "i")] * n_measured
        + [(0, "i")] * n_inestimable
        + [(-1, "i")] * n_anchor_missing
        + [(None, "i")] * n_tag_absent
    )
    for i, (pt, _) in enumerate(states):
        rows.append(
            f"read{i}\tchr1\t+\t{100 + i}\t{200 + i}\t1-2\t0.99\t1.0\t"
            + ("-2" if pt is None else str(pt))
        )
    return rows


def _model_frame():
    return pd.DataFrame(
        [
            {
                "Chr": "chr1", "Strand": "+", "SSC": "1-2",
                "TrStart": 100, "TrEnd": 200, "Group": 0,
            }
        ]
    )


def _args(output_dir):
    return argparse.Namespace(
        output=str(output_dir),
        terminal_tolerance=50,
        ss_tolerance=15,
        min_aln_coverage=None,
        min_aln_identity=None,
    )


def test_only_measured_tails_count_as_polya_evidence(tmp_path):
    """10 measured + 3 inestimable + 3 anchor-not-found + 2 tag-absent reads.

    The measured fraction must be 10 / 18. With the old ``!= 0`` predicate
    the three sentinels would have been counted as evidence (13 / 18) --
    i.e. a library with no poly(A) evidence at all would look like it had
    a measured tail.
    """
    sample_dir = tmp_path / "temp"
    sample_dir.mkdir()
    flnc = sample_dir / "S1_flnc.ssc"
    flnc.write_text(
        "\n".join(_flnc_rows(10, 3, 3, 2)) + "\n"
    )

    anno = polyAnnotator(_args(tmp_path))
    df = anno.anno_polya(_model_frame(), [str(flnc)])

    assert len(df) == 1
    assert df["polyA_frac"].iloc[0] == pytest.approx(10 / 18)


def test_a_library_without_any_measured_tail_has_zero_fraction(tmp_path):
    """No measured tail at all => 0 measured reads, and the fraction is 0."""
    sample_dir = tmp_path / "temp"
    sample_dir.mkdir()
    flnc = sample_dir / "S1_flnc.ssc"
    flnc.write_text("\n".join(_flnc_rows(0, 3, 3, 2)) + "\n")

    anno = polyAnnotator(_args(tmp_path))
    df = anno.anno_polya(_model_frame(), [str(flnc)])

    assert df["polyA_frac"].iloc[0] == pytest.approx(0.0)


# --- (c) the estimator guarantee also holds for negative sentinels ---------

def test_all_negative_pt_equals_production():
    """The production no-pt case now writes -2 (tag missing) / -1 (anchor
    not found) instead of 0; the 3'-end estimator must behave exactly as
    it did for the all-zero pt case."""
    from src.get_terminal_sites import TerminalSitesProcessor

    proc = TerminalSitesProcessor()
    values = [1000] * 40 + [990] * 60 + [1200] * 30
    for negative in (POLYA_LEN_ANCHOR_NOT_FOUND, POLYA_LEN_TAG_MISSING):
        s = pd.Series(values)
        p = pd.Series([negative] * len(values))
        for mode in ("min", "max"):
            expected = list(proc.get_representative_sites_DBSCAN(s.copy(), mode=mode))
            got = list(proc.get_representative_sites_DBSCAN(s.copy(), mode=mode, pt=p))
            assert got == expected
