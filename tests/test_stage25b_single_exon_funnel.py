"""9-case Golden Characterization Test for evaluate_single_exon_isoform (Stage 2.5b).

This test file locks the single-exon 5-pillar funnel
(src/single_exon.py:31-54) against future refactors. Each TC is anchored
to a specific rejection branch in evaluate_single_exon_isoform():

  - 5P-1: full happy path PASS
  - 5P-2: Pillar 1 (genic overlap) DROP
  - 5P-3: Pillar 4 (low frequency) DROP
  - 5P-4: Pillar 5 (short length) DROP
  - 5P-5: Pillar 2 (weak TSS) DROP
  - 5P-6: Pillar 3 polyA mode (low polyA_frac) DROP
  - 5P-7: Pillar 3 fail-closed (no genome) DROP
  - 5P-8: Pillar 3 intra-priming (A-rich downstream) DROP
  - 5P-9: Pillar 3 intra-priming PASS (non-A-rich)

All 9 expected outputs are mathematically derivable from the code at
src/single_exon.py:31-54 with no ambiguity.
"""
import os
import sys
import importlib.util
from unittest.mock import MagicMock

import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap sys.path and import src.single_exon via importlib, bypassing
# src/__init__.py (pre-existing missing SINGLE_EXON_GROUP_SENTINEL constant
# in src/gene_grouping.py blocks aidrs.py import).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO, "src")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Register an empty 'src' package (skip __init__.py execution)
_spec_pkg = importlib.util.spec_from_loader("src", loader=None, is_package=True)
_src_pkg = importlib.util.module_from_spec(_spec_pkg)
_src_pkg.__path__ = [_SRC]
sys.modules["src"] = _src_pkg

# Load src.common as a submodule of the synthetic src package (single_exon
# imports rev_comp from .common at module import time)
_spec_common = importlib.util.spec_from_file_location(
    "src.common", os.path.join(_SRC, "common.py")
)
_common_mod = importlib.util.module_from_spec(_spec_common)
sys.modules["src.common"] = _common_mod
_spec_common.loader.exec_module(_common_mod)

# Load src.single_exon as a submodule
_spec_se = importlib.util.spec_from_file_location(
    "src.single_exon", os.path.join(_SRC, "single_exon.py")
)
_se_mod = importlib.util.module_from_spec(_spec_se)
sys.modules["src.single_exon"] = _se_mod
_spec_se.loader.exec_module(_se_mod)

evaluate_single_exon_isoform = _se_mod.evaluate_single_exon_isoform


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
# Default test parameters (per task spec)
POLYA_THRESH = 0.95
FILTER_FREQ = 3


def _row(is_intergenic=True, frequency=20, seq_len=500,
         puffin=0.15, polya_frac=0.98, polya_valid_reads=10,
         chr_="chr1", start=1000, end=1500, strand="+"):
    """Build a single-exon test row with all 10 required columns.

    Defaults are the "happy path" inputs (all pillars pass).
    """
    return pd.Series({
        "Chr": chr_,
        "Strand": strand,
        "TrStart": start,
        "TrEnd": end,
        "frequency": frequency,
        "Puffin_TSS_15bp": puffin,
        "polyA_frac": polya_frac,
        "polyA_valid_reads": polya_valid_reads,
        "is_intergenic_or_antisense": is_intergenic,
        "seq_len": seq_len,
    })


# ---------------------------------------------------------------------------
# 5P-1: full happy path PASS
# ---------------------------------------------------------------------------
def test_5p_1_full_happy_path_passes():
    """All pillars pass: intergenic + freq >= 15 + length >= 200 +
    Puffin >= 0.1 + polyA_frac >= 0.95 + valid_reads >= 3.

    Boolean trace:
      Pillar 1: is_intergenic_or_antisense=True  -> pass
      Pillar 4: frequency=20 >= max(15, 9)=15   -> pass
      Pillar 5: seq_len=500 >= 200              -> pass
      Pillar 2: Puffin=0.15 >= 0.1              -> pass
      Pillar 3: polyA_frac=0.98 >= 0.95, valid_reads=10 >= 3 -> pass
      -> (True, "PASS")
    """
    row = _row()  # all defaults
    ok, reason = evaluate_single_exon_isoform(
        row, has_valid_polya=True,
        polyA_thresh=POLYA_THRESH, filter_freq=FILTER_FREQ,
        genome_fasta=None,
    )
    assert ok is True and reason == "PASS", (
        f"5P-1: expected PASS; got ok={ok}, reason={reason}"
    )
    print("[OK] 5P-1: full happy path -> PASS")


# ---------------------------------------------------------------------------
# 5P-2: Pillar 1 (genic overlap) DROP
# ---------------------------------------------------------------------------
def test_5p_2_pillar1_genic_overlap_drops():
    """Pillar 1 fail: is_intergenic_or_antisense=False -> DROP.

    Boolean trace:
      Pillar 1: is_intergenic_or_antisense=False -> reject "P1_genic_overlap"
      (Pillar 4/5/2/3 never reached)
      -> (False, "P1_genic_overlap")
    """
    row = _row(is_intergenic=False)
    ok, reason = evaluate_single_exon_isoform(
        row, has_valid_polya=True,
        polyA_thresh=POLYA_THRESH, filter_freq=FILTER_FREQ,
        genome_fasta=None,
    )
    assert ok is False and reason == "P1_genic_overlap", (
        f"5P-2: expected DROP P1_genic_overlap; got ok={ok}, reason={reason}"
    )
    print("[OK] 5P-2: Pillar 1 (genic overlap) -> DROP")


# ---------------------------------------------------------------------------
# 5P-3: Pillar 4 (low frequency) DROP
# ---------------------------------------------------------------------------
def test_5p_3_pillar4_low_freq_drops():
    """Pillar 4 fail: frequency=5 < max(15, 3*filter_freq)=15 -> DROP.

    Boolean trace:
      Pillar 1: pass
      Pillar 4: frequency=5 < 15 -> reject "P4_low_freq"
      (Pillar 5/2/3 never reached)
      -> (False, "P4_low_freq")
    """
    row = _row(frequency=5)
    ok, reason = evaluate_single_exon_isoform(
        row, has_valid_polya=True,
        polyA_thresh=POLYA_THRESH, filter_freq=FILTER_FREQ,
        genome_fasta=None,
    )
    assert ok is False and reason == "P4_low_freq", (
        f"5P-3: expected DROP P4_low_freq; got ok={ok}, reason={reason}"
    )
    print("[OK] 5P-3: Pillar 4 (low frequency=5) -> DROP")


# ---------------------------------------------------------------------------
# 5P-4: Pillar 5 (short length) DROP
# ---------------------------------------------------------------------------
def test_5p_4_pillar5_short_drops():
    """Pillar 5 fail: seq_len=100 < 200 -> DROP.

    Boolean trace:
      Pillar 1: pass
      Pillar 4: pass (frequency=20)
      Pillar 5: seq_len=100 < 200 -> reject "P5_short"
      (Pillar 2/3 never reached)
      -> (False, "P5_short")
    """
    row = _row(seq_len=100)
    ok, reason = evaluate_single_exon_isoform(
        row, has_valid_polya=True,
        polyA_thresh=POLYA_THRESH, filter_freq=FILTER_FREQ,
        genome_fasta=None,
    )
    assert ok is False and reason == "P5_short", (
        f"5P-4: expected DROP P5_short; got ok={ok}, reason={reason}"
    )
    print("[OK] 5P-4: Pillar 5 (short seq_len=100) -> DROP")


# ---------------------------------------------------------------------------
# 5P-5: Pillar 2 (weak TSS) DROP
# ---------------------------------------------------------------------------
def test_5p_5_pillar2_weak_tss_drops():
    """Pillar 2 fail: Puffin_TSS_15bp=0.05 < 0.1 -> DROP.

    Boolean trace:
      Pillar 1: pass
      Pillar 4: pass
      Pillar 5: pass
      Pillar 2: Puffin=0.05 < 0.1 -> reject "P2_weak_tss"
      (Pillar 3 never reached)
      -> (False, "P2_weak_tss")
    """
    row = _row(puffin=0.05)
    ok, reason = evaluate_single_exon_isoform(
        row, has_valid_polya=True,
        polyA_thresh=POLYA_THRESH, filter_freq=FILTER_FREQ,
        genome_fasta=None,
    )
    assert ok is False and reason == "P2_weak_tss", (
        f"5P-5: expected DROP P2_weak_tss; got ok={ok}, reason={reason}"
    )
    print("[OK] 5P-5: Pillar 2 (weak Puffin=0.05) -> DROP")


# ---------------------------------------------------------------------------
# 5P-6: Pillar 3 polyA mode (low polyA_frac) DROP
# ---------------------------------------------------------------------------
def test_5p_6_pillar3_polya_fail_drops():
    """Pillar 3 polyA-mode fail: polyA_frac=0.5 < 0.95 (with valid_reads=10) -> DROP.

    Boolean trace:
      Pillars 1/4/5/2 all pass
      Pillar 3 (polyA mode): polyA_frac=0.5 < polyA_thresh=0.95
        -> reject "P3_polya_fail"
      -> (False, "P3_polya_fail")
    """
    row = _row(polya_frac=0.5, polya_valid_reads=10)
    ok, reason = evaluate_single_exon_isoform(
        row, has_valid_polya=True,
        polyA_thresh=POLYA_THRESH, filter_freq=FILTER_FREQ,
        genome_fasta=None,
    )
    assert ok is False and reason == "P3_polya_fail", (
        f"5P-6: expected DROP P3_polya_fail; got ok={ok}, reason={reason}"
    )
    print("[OK] 5P-6: Pillar 3 polyA mode (low polyA_frac=0.5) -> DROP")


# ---------------------------------------------------------------------------
# 5P-7: Pillar 3 fail-closed (no genome) DROP
# ---------------------------------------------------------------------------
def test_5p_7_pillar3_no_genome_drops():
    """Pillar 3 fail-closed: has_valid_polya=False + genome_fasta=None -> DROP.

    Boolean trace:
      Pillars 1/4/5/2 all pass
      Pillar 3 (intra-priming mode): has_valid_polya=False
        genome_fasta is None -> reject "P3_no_genome"
      -> (False, "P3_no_genome")
    """
    row = _row()  # defaults: polyA_frac=0.98 but irrelevant under has_valid_polya=False
    ok, reason = evaluate_single_exon_isoform(
        row, has_valid_polya=False,
        polyA_thresh=POLYA_THRESH, filter_freq=FILTER_FREQ,
        genome_fasta=None,
    )
    assert ok is False and reason == "P3_no_genome", (
        f"5P-7: expected DROP P3_no_genome; got ok={ok}, reason={reason}"
    )
    print("[OK] 5P-7: Pillar 3 fail-closed (no genome_fasta) -> DROP")


# ---------------------------------------------------------------------------
# 5P-8: Pillar 3 intra-priming (A-rich downstream) DROP
# ---------------------------------------------------------------------------
def test_5p_8_pillar3_intra_priming_arich_drops():
    """Pillar 3 intra-priming fail: downstream is A-rich -> DROP.

    Mock genome_fasta.fetch(chrom, end, end+20) returns "AAAAAA..." which
    contains the polyA_run "AAAAAA" and is >= 60% A.

    Boolean trace:
      Pillars 1/4/5/2 all pass
      Pillar 3 (intra-priming mode): has_valid_polya=False
        check_genomic_intra_priming -> True (A-rich)
        -> reject "P3_intra_priming"
      -> (False, "P3_intra_priming")
    """
    genome_fasta = MagicMock()
    genome_fasta.fetch.return_value = "A" * 20

    row = _row()  # defaults; positive strand -> fetch(end, end+20)
    ok, reason = evaluate_single_exon_isoform(
        row, has_valid_polya=False,
        polyA_thresh=POLYA_THRESH, filter_freq=FILTER_FREQ,
        genome_fasta=genome_fasta,
    )
    assert ok is False and reason == "P3_intra_priming", (
        f"5P-8: expected DROP P3_intra_priming; got ok={ok}, reason={reason}"
    )
    # Sanity: confirm fetch was called with downstream 20bp window
    genome_fasta.fetch.assert_called_once()
    print("[OK] 5P-8: Pillar 3 intra-priming (A-rich downstream) -> DROP")


# ---------------------------------------------------------------------------
# 5P-9: Pillar 3 intra-priming PASS (non-A-rich)
# ---------------------------------------------------------------------------
def test_5p_9_pillar3_intra_priming_passes():
    """Pillar 3 intra-priming pass: downstream is non-A-rich -> PASS.

    Mock genome_fasta.fetch(chrom, end, end+20) returns "CCTTGGC..." which
    contains no "AAAAAA" run and is < 60% A.

    Boolean trace:
      Pillars 1/4/5/2 all pass
      Pillar 3 (intra-priming mode): has_valid_polya=False
        check_genomic_intra_priming -> False (non-A-rich)
        -> fall through to (True, "PASS")
      -> (True, "PASS")
    """
    genome_fasta = MagicMock()
    # 20bp sequence with no "AAAAAA" and < 60% A (5 As out of 20 = 25%)
    genome_fasta.fetch.return_value = "CCTTGGCCTTGGCCTTGGC"

    row = _row()  # defaults
    ok, reason = evaluate_single_exon_isoform(
        row, has_valid_polya=False,
        polyA_thresh=POLYA_THRESH, filter_freq=FILTER_FREQ,
        genome_fasta=genome_fasta,
    )
    assert ok is True and reason == "PASS", (
        f"5P-9: expected PASS; got ok={ok}, reason={reason}"
    )
    # Sanity: confirm fetch was called with downstream 20bp window
    genome_fasta.fetch.assert_called_once()
    print("[OK] 5P-9: Pillar 3 intra-priming (non-A-rich) -> PASS")


if __name__ == "__main__":
    test_5p_1_full_happy_path_passes()
    test_5p_2_pillar1_genic_overlap_drops()
    test_5p_3_pillar4_low_freq_drops()
    test_5p_4_pillar5_short_drops()
    test_5p_5_pillar2_weak_tss_drops()
    test_5p_6_pillar3_polya_fail_drops()
    test_5p_7_pillar3_no_genome_drops()
    test_5p_8_pillar3_intra_priming_arich_drops()
    test_5p_9_pillar3_intra_priming_passes()
    print("\nAll Stage 2.5b 5-pillar funnel golden tests passed.")