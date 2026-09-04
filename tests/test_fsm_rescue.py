"""Comprehensive FSM-only rescue matrix for transcript_model_filtering.

The FSM-only rescue (Q-A2 push-back) states:
  - If has_polya_info AND polyA_frac == 0 AND category == 'FSM'  -> KEEP (rescue)
  - If has_polya_info AND polyA_frac == 0 AND category != 'FSM'  -> DROP
  - All other paths unchanged

This test file exercises the FSM-rescue matrix across all 3 polyA states and
all 3 SQANTI3 categories (FSM / NIC / NNC):

  | category | State 1 (>thr) | State 1 (<thr) | State 2 (==0) | State 3 (NaN) |
  |----------|----------------|----------------|---------------|---------------|
  | FSM      | KEEP           | drop           | KEEP (rescue) | KEEP          |
  | NIC      | KEEP           | drop           | DROP          | KEEP          |
  | NNC      | KEEP           | drop           | DROP          | KEEP          |

Both filter paths (hard_filter=True and hard_filter=False) are exercised.
"""
import os
import sys
import logging
import importlib.util
import pandas as pd
import numpy as np

# Import common.py via importlib so its relative imports resolve, but bypass
# src/__init__.py which fails due to a pre-existing missing
# SINGLE_EXON_GROUP_SENTINEL constant in src/gene_grouping.py.
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

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def _row(category, state, chr_="chr1", start=100, end=200, valid_reads=10):
    """Build a single test row.

    state: 1 (polyA_frac > 0), 2 (polyA_frac == 0), 3 (polyA_frac NaN)
    """
    if state == 1:
        polya_frac = 0.99  # above threshold 0.95
        valid = valid_reads
    elif state == 2:
        polya_frac = 0.0
        valid = 0
    else:  # state == 3
        polya_frac = np.nan
        valid = 0
    return {
        "Chr": chr_, "Strand": "+", "SSC": f"{start};{end}",
        "TrStart": start, "TrEnd": end,
        # Puffin 15bp/50bp both strong so only polyA gate fires
        "Puffin_TSS_15bp": 1.0, "Puffin_TSS_50bp": 1.0,
        "truncation": "no", "Predict_NMD": "no_orf",
        "category": category,
        "polyA_frac": polya_frac, "polyA_valid_reads": valid,
        "frequency": 10,
    }


def _df(rows):
    df = pd.DataFrame(rows)
    for col in ("Chr", "Strand"):
        df[col] = df[col].astype("category")
    return df


class FakeFasta:
    """All-N fasta; intra-priming always False (which means non-FSM State 2
    rows still get filtered via the 3-condition framework).

    Mirrors the subset of pysam.FastaFile used by
    check_genomic_intra_priming: fetch(chrom, start, end) returns the
    half-open substring [start, end).
    """
    def __init__(self):
        self._seqs = {"chr1": "N" * 10000}
    def __getitem__(self, chrom):
        return self._seqs[chrom]
    def fetch(self, chrom, start, end):
        return self._seqs[chrom][start:end]


# -----------------------------------------------------------------------------
# Hard_filter path: explicit threshold gate
# -----------------------------------------------------------------------------
def test_fsm_hard_filter_state2_rescued():
    """hard_filter=True, FSM in State 2 -> KEEP (FSM rescue)."""
    rows = [
        _row("FSM", 2, start=100, end=200),
        _row("NIC", 2, start=300, end=400),
        _row("NNC", 2, start=500, end=600),
    ]
    df = _df(rows)
    df_out = transcript_model_filtering(
        df, hard_filter=True, genome_fasta=FakeFasta(),
    )
    kept_starts = sorted(df_out["TrStart"].tolist())
    assert kept_starts == [100], (
        f"hard_filter State 2 rescue: expected only FSM (100) kept; got {kept_starts}"
    )
    print("[OK] hard_filter: FSM in State 2 rescued; NIC + NNC in State 2 dropped")


def test_nic_nnc_hard_filter_state2_dropped():
    """hard_filter=True, NIC and NNC in State 2 -> DROP (current behavior)."""
    rows = [
        _row("FSM", 2, start=100, end=200),
        _row("NIC", 2, start=300, end=400),
        _row("NNC", 2, start=500, end=600),
    ]
    df = _df(rows)
    df_out = transcript_model_filtering(
        df, hard_filter=True, genome_fasta=FakeFasta(),
    )
    dropped = df[~df["TrStart"].isin(df_out["TrStart"])]
    assert set(dropped["category"].tolist()) == {"NIC", "NNC"}, (
        f"hard_filter State 2 drop: expected NIC + NNC dropped; got "
        f"{sorted(dropped['category'].tolist())}"
    )
    print("[OK] hard_filter: NIC + NNC in State 2 dropped (FSM-only rescue)")


def test_fsm_hard_filter_state1_and_state3_kept():
    """hard_filter=True, FSM in State 1 (high polyA) and State 3 (NaN) -> KEEP."""
    rows = [
        _row("FSM", 1, start=100, end=200),  # high polyA -> KEEP
        _row("FSM", 3, start=300, end=400),  # NaN -> KEEP (bypass)
    ]
    df = _df(rows)
    df_out = transcript_model_filtering(
        df, hard_filter=True, genome_fasta=FakeFasta(),
    )
    kept_starts = sorted(df_out["TrStart"].tolist())
    assert kept_starts == [100, 300], (
        f"hard_filter FSM State 1/3: expected [100, 300]; got {kept_starts}"
    )
    print("[OK] hard_filter: FSM in State 1 (high polyA) + State 3 (NaN) kept")


# -----------------------------------------------------------------------------
# Non-hard_filter path: 3-condition framework (NMD/truncation/ultra-low-quality)
# -----------------------------------------------------------------------------
def test_fsm_non_hard_filter_state2_rescued():
    """non-hard_filter path: FSM in State 2 -> KEEP even with Puffin_TSS_50bp='no'.

    With genome_fasta + FSM rescue, polya_frac_low=False for FSM State 2.
    ultra_low_quality_filter = Puffin_no AND polya_low -> False -> kept.

    Note: needs at least one row with polyA_valid_reads > 0 so has_polya_info
    is True (sample-level bypass guards against missing polyA data).
    """
    rows = [
        # Sample has polyA info: a State 1 row with valid polyA reads
        _row("FSM", 1, start=50, end=80),
        _row("FSM", 2, start=100, end=200),
        _row("NIC", 2, start=300, end=400),
        _row("NNC", 2, start=500, end=600),
    ]
    # Force ultra-low-quality filter to be the only lever by setting Puffin_TSS_50bp='no'
    for r in rows:
        r["Puffin_TSS_50bp"] = "no"

    df = _df(rows)
    df_out = transcript_model_filtering(
        df, hard_filter=False, genome_fasta=FakeFasta(),
    )
    kept_starts = sorted(df_out["TrStart"].tolist())
    # State 1 FSM (50) + State 2 FSM (100) survive; State 2 NIC + NNC dropped
    assert kept_starts == [50, 100], (
        f"non-hard_filter FSM State 2 rescue: expected [50, 100]; got {kept_starts}"
    )
    print("[OK] non-hard_filter: FSM in State 2 rescued; NIC + NNC dropped")


def test_nic_nnc_non_hard_filter_state2_dropped():
    """non-hard_filter path: NIC + NNC in State 2 with Puffin_TSS_50bp='no' -> DROP.

    With genome_fasta but NOT FSM, polya_frac_low=True for non-FSM State 2.
    ultra_low_quality_filter = Puffin_no AND polya_low -> True -> dropped.

    Note: needs at least one row with polyA_valid_reads > 0 so has_polya_info
    is True (sample-level bypass guards against missing polyA data).
    """
    rows = [
        # Sample has polyA info
        _row("FSM", 1, start=50, end=80),
        _row("FSM", 2, start=100, end=200),
        _row("NIC", 2, start=300, end=400),
        _row("NNC", 2, start=500, end=600),
    ]
    for r in rows:
        r["Puffin_TSS_50bp"] = "no"

    df = _df(rows)
    df_out = transcript_model_filtering(
        df, hard_filter=False, genome_fasta=FakeFasta(),
    )
    dropped = df[~df["TrStart"].isin(df_out["TrStart"])]
    dropped_categories = dropped["category"].tolist()
    # State 1 FSM (50) kept + State 2 FSM (100) kept (FSM rescue)
    # State 2 NIC (300) + NNC (500) dropped
    assert "NIC" in dropped_categories and "NNC" in dropped_categories, (
        f"non-hard_filter State 2 drop: expected NIC + NNC dropped; got "
        f"{sorted(dropped_categories)}"
    )
    assert "FSM" not in dropped["category"].tolist(), (
        f"non-hard_filter FSM should not be dropped: {sorted(dropped_categories)}"
    )
    print("[OK] non-hard_filter: NIC + NNC in State 2 dropped (FSM-only rescue)")


def test_full_matrix_non_hard_filter():
    """Full FSM/NIC/NNC x State 1/2/3 matrix in non-hard_filter path.

    Expected outcomes (Puffin_TSS_50bp='no' for State 2 rows to test the
    ultra-low-quality gate; State 1 rows have polyA high so they survive
    Puffin gate; State 3 bypasses polyA entirely):

      FSM:   State 1 (high polyA) -> KEEP (Puffin_gate=False, polya_low=False)
      FSM:   State 2              -> KEEP (FSM rescue -> polya_low=False)
      FSM:   State 3              -> KEEP (polyA bypass)

      NIC:   State 1 (high polyA) -> KEEP
      NIC:   State 2              -> DROP (no rescue)
      NIC:   State 3              -> KEEP (bypass)

      NNC:   State 1 (high polyA) -> KEEP
      NNC:   State 2              -> DROP
      NNC:   State 3              -> KEEP
    """
    rows = []
    rows.append(_row("FSM", 1, start=100, end=200))   # KEEP
    rows.append(_row("FSM", 2, start=300, end=400))   # KEEP (rescue)
    rows.append(_row("FSM", 3, start=500, end=600))   # KEEP

    rows.append(_row("NIC", 1, start=700, end=800))   # KEEP
    rows.append(_row("NIC", 2, start=900, end=1000))  # DROP
    rows.append(_row("NIC", 3, start=1100, end=1200)) # KEEP

    rows.append(_row("NNC", 1, start=1300, end=1400)) # KEEP
    rows.append(_row("NNC", 2, start=1500, end=1600)) # DROP
    rows.append(_row("NNC", 3, start=1700, end=1800)) # KEEP

    # State 2 rows: force ultra-low-quality gate via Puffin_TSS_50bp='no'
    for r in rows:
        if r["polyA_frac"] == 0.0 and r["polyA_valid_reads"] == 0:
            r["Puffin_TSS_50bp"] = "no"

    df = _df(rows)
    df_out = transcript_model_filtering(
        df, hard_filter=False, genome_fasta=FakeFasta(),
    )
    kept_starts = sorted(df_out["TrStart"].tolist())
    expected = [100, 300, 500, 700, 1100, 1300, 1700]
    assert kept_starts == expected, (
        f"non-hard_filter full matrix: expected {expected}; got {kept_starts}"
    )
    print(f"[OK] non-hard_filter full matrix: {len(df_out)}/{len(df)} kept as expected")


if __name__ == "__main__":
    test_fsm_hard_filter_state2_rescued()
    test_nic_nnc_hard_filter_state2_dropped()
    test_fsm_hard_filter_state1_and_state3_kept()
    test_fsm_non_hard_filter_state2_rescued()
    test_nic_nnc_non_hard_filter_state2_dropped()
    test_full_matrix_non_hard_filter()
    print("\nAll FSM-only rescue tests passed.")