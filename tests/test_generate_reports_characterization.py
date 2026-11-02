"""Characterization tests for the report writers.

These pin the *current* byte-level behaviour of the output writers, so a
refactor of the id generation (issue #3: the fused-cluster gene ids) or of
the coordinate handling cannot change the delivered files silently.

The reference file lives in ``tests/fixtures/aidrs.transcript_model.gtf``.
It was captured from the implementation as of commit ``d36db49``; the
C107 100k end-to-end golden gate (``aidrs.transcript_model.gtf`` sha256
identity against the 2026-09-11 golden) is the second, independent pin.
"""
import os

import pandas as pd
import pytest

from src.aidrs_runtime.column_standardize import build_ref_dict, reverse_complement
from src.aidrs_runtime.report_writers import to_gtf

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
GTF_FIXTURE = os.path.join(FIXTURE_DIR, "aidrs.transcript_model.gtf")


def _annotated_frame() -> pd.DataFrame:
    """A small, deterministic annotated model.

    * ``G1`` (+ strand) has two transcripts; ``G1_t1`` carries a CDS
      (``Predict_NMD == 'Normal'`` with TIS/TTS locations) so the
      ribosome-walk branch is exercised, ``G1_t2`` does not.
    * ``G2`` (- strand) has one exon-only transcript.
    """
    return pd.DataFrame([
        {"Chr": "chr1", "Strand": "+", "TrStart": 1000, "TrEnd": 4000,
         "TrID": "G1_t1", "GeneID": "G1", "GeneName": "GENEA",
         "SSC": "2000-3000", "TrStart_ref": 1000, "TrEnd_ref": 4000,
         "Predict_NMD": "Normal", "TIS_related_location": 10,
         "TTS_related_location": 1500},
        {"Chr": "chr1", "Strand": "+", "TrStart": 1000, "TrEnd": 3500,
         "TrID": "G1_t2", "GeneID": "G1", "GeneName": "GENEA",
         "SSC": "2000-3000", "TrStart_ref": 1000, "TrEnd_ref": 3500,
         "Predict_NMD": "no", "TIS_related_location": None,
         "TTS_related_location": None},
        {"Chr": "chr2", "Strand": "-", "TrStart": 5000, "TrEnd": 6000,
         "TrID": "G2_t1", "GeneID": "G2", "GeneName": "GENEB",
         "SSC": "5500", "TrStart_ref": 5000, "TrEnd_ref": 6000,
         "Predict_NMD": "no", "TIS_related_location": None,
         "TTS_related_location": None},
    ])


def test_to_gtf_byte_identical(tmp_path):
    """The emitted GTF must stay byte-for-byte what it is today."""
    to_gtf(_annotated_frame(), str(tmp_path))
    produced = os.path.join(str(tmp_path), "aidrs.transcript_model.gtf")
    with open(produced, "rb") as fh:
        actual = fh.read()
    with open(GTF_FIXTURE, "rb") as fh:
        expected = fh.read()
    assert actual == expected, (
        "aidrs.transcript_model.gtf changed; if the change is intended, "
        "re-capture tests/fixtures/aidrs.transcript_model.gtf and record "
        "the cause in the SHA evolution history."
    )


def test_reverse_complement():
    assert reverse_complement("ATGC") == "GCAT"
    assert reverse_complement("atgc") == "GCAT"
    assert reverse_complement("NNNN") == "NNNN"


def test_build_ref_dict():
    df = pd.DataFrame([
        {"GeneID": "G1", "GeneName": "A", "TrStart": 100, "TrEnd": 400,
         "SSC": "200-300"},
        {"GeneID": "G1", "GeneName": "A", "TrStart": 150, "TrEnd": 450,
         "SSC": "200-300"},
        {"GeneID": "G2", "GeneName": "B", "TrStart": 10, "TrEnd": 20,
         "SSC": "nan"},
    ])
    ref = build_ref_dict(df)
    assert set(ref) == {"G1_A", "G2_B"}
    assert sorted(ref["G1_A"]) == [100, 150, 200, 300, 400, 450]
    # include_term=False drops TrStart / TrEnd and keeps only the SSC sites.
    ref_no_term = build_ref_dict(df, include_term=False)
    assert sorted(ref_no_term["G1_A"]) == [200, 300]
    assert ref_no_term["G2_B"] == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
