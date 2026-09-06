"""8-case Golden Characterization Test for transcript_model_filtering (Stage 2.6).

This test file locks the multi-exon transcript_model_filtering decision tree
(src/common.py:253-511) against future refactors. Each TC is anchored to a
specific boolean branch in should_filter_row() (lines 434-474):

  - TC-N1/N2: truncation_filter (Puffin path positive + negative)
  - TC-N3/N4: nmd_filter (Puffin path positive + negative)
  - TC-N5/N6: ultra_low_quality_filter (State 1 vs State 3 polyA bypass)
  - TC-N7: FSM rescue scope (only polyA, not Puffin) -- rebuts the naive
    "FSM unconditional rescue" assumption
  - TC-N8: TranslationAI bypass + State 3 polyA bypass compound

All 8 expected outputs are mathematically derivable from the code at
src/common.py:434-474 with no ambiguity.
"""
import os
import sys
import importlib.util
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap sys.path and import src.common via importlib, bypassing
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

# Load src.common as a submodule of the synthetic src package
_spec = importlib.util.spec_from_file_location("src.common", os.path.join(_SRC, "common.py"))
_common_mod = importlib.util.module_from_spec(_spec)
sys.modules["src.common"] = _common_mod
_spec.loader.exec_module(_common_mod)

transcript_model_filtering = _common_mod.transcript_model_filtering


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def _row(puffin_50bp, polya_frac, polya_valid_reads, predict_nmd, truncation,
         category=None, chr_="test_chr_1", start=100, end=200):
    """Build a single test row for the decision-tree characterization.

    Uses a unique chromosome 'test_chr_1' to avoid collisions with real data.
    Puffin_TSS_15bp is always 1.0 so it cannot independently fire any gate.
    """
    row = {
        "Chr": chr_, "Strand": "+",
        "SSC": f"{start};{end}",
        "TrStart": start, "TrEnd": end,
        "Puffin_TSS_15bp": 1.0,
        "Puffin_TSS_50bp": puffin_50bp,
        "truncation": truncation,
        "Predict_NMD": predict_nmd,
        "polyA_frac": polya_frac,
        "polyA_valid_reads": polya_valid_reads,
        "frequency": 10,
    }
    if category is not None:
        row["category"] = category
    return row


def _df(row):
    df = pd.DataFrame([row])
    for col in ("Chr", "Strand"):
        df[col] = df[col].astype("category")
    return df


def _run(row):
    """Invoke the function under test with hard_filter=False, genome_fasta=None."""
    return transcript_model_filtering(
        _df(row), hard_filter=False, genome_fasta=None,
    )


# ---------------------------------------------------------------------------
# TC-N1: truncation_filter (Puffin bad path) -> DROP
# ---------------------------------------------------------------------------
def test_tc_n1_truncation_with_bad_puffin_drops():
    """truncation='yes' + Puffin_TSS_50bp='no' -> truncation_filter triggers
    via Puffin path. State 1 (polyA=0.98, high) means polya_frac_low=False,
    so the Puffin path is the only active lever.

    Boolean trace:
      puffin_50bp_has_no = True
      polya_frac_low = False
      is_nmd = False
      nmd_filter = False
      truncation_filter = True AND (True OR False) = True
      ultra_low_quality_filter = True AND False = False
      -> should_filter_row() = True -> DROP
    """
    row = _row(
        puffin_50bp="no",
        polya_frac=0.98, polya_valid_reads=10,
        predict_nmd="Normal", truncation="yes",
    )
    df_out = _run(row)
    assert len(df_out) == 0, (
        f"TC-N1: expected DROP (truncation_filter via Puffin path); "
        f"got kept TrStart={df_out['TrStart'].tolist()}"
    )
    print("[OK] TC-N1: truncation='yes' + Puffin='no' + polyA high -> DROP")


# ---------------------------------------------------------------------------
# TC-N2: truncation_filter negated by good Puffin -> PASS
# ---------------------------------------------------------------------------
def test_tc_n2_truncation_with_good_puffin_passes():
    """truncation='yes' + Puffin=0.05 (above 0.02 threshold) +
    polyA=0.98 (high) -> both gates of truncation_filter are False.

    Boolean trace:
      puffin_50bp_has_no = (0.05 < 0.02) = False
      polya_frac_low = False (State 1 high)
      truncation_filter = True AND (False OR False) = False
      ultra_low_quality_filter = False AND False = False
      -> should_filter_row() = False -> PASS
    """
    row = _row(
        puffin_50bp=0.05,
        polya_frac=0.98, polya_valid_reads=10,
        predict_nmd="Normal", truncation="yes",
    )
    df_out = _run(row)
    assert len(df_out) == 1 and df_out["TrStart"].iloc[0] == 100, (
        f"TC-N2: expected PASS (truncation_filter negated by good Puffin); "
        f"got kept TrStart={df_out['TrStart'].tolist()}"
    )
    print("[OK] TC-N2: truncation='yes' + Puffin=0.05 (good) -> PASS")


# ---------------------------------------------------------------------------
# TC-N3: nmd_filter (Puffin bad path) -> DROP
# ---------------------------------------------------------------------------
def test_tc_n3_nmd_with_bad_puffin_drops():
    """Predict_NMD='NMD' + Puffin_TSS_50bp='no' -> nmd_filter triggers via
    Puffin path. State 1 (polyA=0.98, high) means polya_frac_low=False.

    Boolean trace:
      has_translationai = True (Predict_NMD != 'no_orf' and not NaN)
      is_nmd = True
      puffin_50bp_has_no = True
      polya_frac_low = False
      nmd_filter = True AND (True OR False) = True
      -> DROP
    """
    row = _row(
        puffin_50bp="no",
        polya_frac=0.98, polya_valid_reads=10,
        predict_nmd="NMD", truncation="no",
    )
    df_out = _run(row)
    assert len(df_out) == 0, (
        f"TC-N3: expected DROP (nmd_filter via Puffin path); "
        f"got kept TrStart={df_out['TrStart'].tolist()}"
    )
    print("[OK] TC-N3: NMD + Puffin='no' + polyA high -> DROP")


# ---------------------------------------------------------------------------
# TC-N4: nmd_filter negated by all-good -> PASS
# ---------------------------------------------------------------------------
def test_tc_n4_nmd_with_all_good_passes():
    """Predict_NMD='NMD' + Puffin=0.05 (good) + polyA=0.98 (high) ->
    nmd_filter is False because both Puffin path and polyA path are negated.

    Boolean trace:
      is_nmd = True
      puffin_50bp_has_no = False
      polya_frac_low = False
      nmd_filter = True AND (False OR False) = False
      -> PASS
    """
    row = _row(
        puffin_50bp=0.05,
        polya_frac=0.98, polya_valid_reads=10,
        predict_nmd="NMD", truncation="no",
    )
    df_out = _run(row)
    assert len(df_out) == 1 and df_out["TrStart"].iloc[0] == 100, (
        f"TC-N4: expected PASS (nmd_filter negated by all-good); "
        f"got kept TrStart={df_out['TrStart'].tolist()}"
    )
    print("[OK] TC-N4: NMD + Puffin=0.05 (good) + polyA high -> PASS")


# ---------------------------------------------------------------------------
# TC-N5: ultra_low_quality_filter (both bad) -> DROP
# ---------------------------------------------------------------------------
def test_tc_n5_ultra_low_quality_drops():
    """Puffin_TSS_50bp='no' + polyA=0.5 (State 1, < 0.95) ->
    ultra_low_quality_filter triggers: both gates are True.

    Boolean trace:
      puffin_50bp_has_no = True
      polya_frac_low = (0.5 < 0.95) = True
      is_nmd = False (Normal)
      nmd_filter = False
      truncation_filter = False AND (...) = False
      ultra_low_quality_filter = True AND True = True
      -> DROP
    """
    row = _row(
        puffin_50bp="no",
        polya_frac=0.5, polya_valid_reads=10,
        predict_nmd="Normal", truncation="no",
    )
    df_out = _run(row)
    assert len(df_out) == 0, (
        f"TC-N5: expected DROP (ultra_low_quality_filter, both bad); "
        f"got kept TrStart={df_out['TrStart'].tolist()}"
    )
    print("[OK] TC-N5: Puffin='no' + polyA=0.5 (low) -> DROP (ultra-low)")


# ---------------------------------------------------------------------------
# TC-N6: State 3 bypass saves ultra_low -> PASS
# ---------------------------------------------------------------------------
def test_tc_n6_state3_bypass_passes():
    """Puffin_TSS_50bp='no' + polyA=NaN (State 3) -> State 3 bypass
    sets polya_frac_low=False, so ultra_low_quality_filter is False.

    Boolean trace:
      puffin_50bp_has_no = True
      polya_frac_low = False (State 3 bypass: polya_frac_low_rowwise.loc[mask_state3]
                              is never assigned, stays at the initial False)
      is_nmd = False (Normal)
      truncation_filter = False
      ultra_low_quality_filter = True AND False = False
      -> PASS
    """
    row = _row(
        puffin_50bp="no",
        polya_frac=np.nan, polya_valid_reads=0,
        predict_nmd="Normal", truncation="no",
    )
    df_out = _run(row)
    assert len(df_out) == 1 and df_out["TrStart"].iloc[0] == 100, (
        f"TC-N6: expected PASS (State 3 polyA bypass saves ultra_low); "
        f"got kept TrStart={df_out['TrStart'].tolist()}"
    )
    print("[OK] TC-N6: Puffin='no' + polyA=NaN (State 3) -> PASS (bypass)")


# ---------------------------------------------------------------------------
# TC-N7: FSM rescue scope (only polyA, not Puffin) -> DROP
# ---------------------------------------------------------------------------
def test_tc_n7_fsm_rescue_does_not_save_puffin_path_drops():
    """category='FSM' + truncation='yes' + Puffin='no' + polyA=0 (State 2).
    Naive assumption: 'FSM rescue = unconditional KEEP'. Reality: FSM rescue
    only sets polya_frac_low=False for State 2; it does NOT touch the
    Puffin gate. truncation_filter still triggers via Puffin path.

    With genome_fasta=None (legacy semantics), FSM rescue is NOT applied
    in the mask construction block, so polya_frac_low=True. Either way
    the outcome is DROP -- the gate that fires is Puffin, not polyA.

    Boolean trace (with genome_fasta=None, legacy):
      puffin_50bp_has_no = True
      polya_frac_low = True (State 2, no FSM rescue under legacy None)
      is_nmd = False
      truncation_filter = True AND (True OR True) = True (Puffin path)
      ultra_low_quality_filter = True AND True = True
      -> DROP
    """
    row = _row(
        puffin_50bp="no",
        polya_frac=0.0, polya_valid_reads=0,
        predict_nmd="Normal", truncation="yes",
        category="FSM",
    )
    df_out = _run(row)
    assert len(df_out) == 0, (
        f"TC-N7: expected DROP (FSM rescue does not save Puffin path); "
        f"got kept TrStart={df_out['TrStart'].tolist()}"
    )
    print("[OK] TC-N7: FSM + truncation='yes' + Puffin='no' -> DROP (FSM scope rebutted)")


# ---------------------------------------------------------------------------
# TC-N8: TranslationAI bypass + State 3 bypass compound -> PASS
# ---------------------------------------------------------------------------
def test_tc_n8_translationai_bypass_compound_passes():
    """Predict_NMD='no_orf' + truncation='no' + Puffin='no' + polyA=NaN.
    Two bypasses compound:
      1. TranslationAI bypass: Predict_NMD='no_orf' -> has_translationai=False
         -> is_nmd=False, so nmd_filter is always False regardless of Puffin.
      2. State 3 polyA bypass: polya_frac_low=False, so
         ultra_low_quality_filter is False.

    Boolean trace:
      puffin_50bp_has_no = True
      polya_frac_low = False (State 3)
      has_translationai = False (no_orf)
      is_nmd = False
      nmd_filter = False
      truncation_filter = False AND (...) = False
      ultra_low_quality_filter = True AND False = False
      -> PASS
    """
    row = _row(
        puffin_50bp="no",
        polya_frac=np.nan, polya_valid_reads=0,
        predict_nmd="no_orf", truncation="no",
    )
    df_out = _run(row)
    assert len(df_out) == 1 and df_out["TrStart"].iloc[0] == 100, (
        f"TC-N8: expected PASS (TranslationAI + State 3 compound bypass); "
        f"got kept TrStart={df_out['TrStart'].tolist()}"
    )
    print("[OK] TC-N8: no_orf + Puffin='no' + polyA=NaN -> PASS (compound bypass)")


if __name__ == "__main__":
    test_tc_n1_truncation_with_bad_puffin_drops()
    test_tc_n2_truncation_with_good_puffin_passes()
    test_tc_n3_nmd_with_bad_puffin_drops()
    test_tc_n4_nmd_with_all_good_passes()
    test_tc_n5_ultra_low_quality_drops()
    test_tc_n6_state3_bypass_passes()
    test_tc_n7_fsm_rescue_does_not_save_puffin_path_drops()
    test_tc_n8_translationai_bypass_compound_passes()
    print("\nAll 8 Stage 2.6 decision-tree characterization tests passed.")
