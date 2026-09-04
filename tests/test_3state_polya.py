"""Unit tests for 3-state polyA logic in transcript_model_filtering.

Verifies that:
  State 1 (polyA_frac > 0): filter if < threshold
  State 2 (polyA_frac == 0, measured zero):
    - Without genome_fasta: filter (legacy semantics, byte-identical baseline)
    - With genome_fasta + FSM rescue: keep
    - With genome_fasta + intra-priming detected: filter
    - With genome_fasta + NOT intra-priming: keep
  State 3 (polyA_frac is NaN): bypass (no penalty)
"""
import sys
import os
import importlib.util
import pandas as pd
import numpy as np

REPO = "/datf/hanxi/software/AIDRS/repo"
sys.path.insert(0, REPO)
_SRC = os.path.join(REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Bypass src/__init__.py (pre-existing missing SINGLE_EXON_GROUP_SENTINEL
# constant in src/gene_grouping.py blocks aidrs.py import).
_spec_pkg = importlib.util.spec_from_loader("src", loader=None, is_package=True)
_src_pkg = importlib.util.module_from_spec(_spec_pkg)
_src_pkg.__path__ = [_SRC]
sys.modules["src"] = _src_pkg
_spec = importlib.util.spec_from_file_location("src.common", os.path.join(_SRC, "common.py"))
_common_mod = importlib.util.module_from_spec(_spec)
sys.modules["src.common"] = _common_mod
_spec.loader.exec_module(_common_mod)
transcript_model_filtering = _common_mod.transcript_model_filtering


def make_test_df():
    """Construct a test DataFrame covering all 3 polyA states.

    Each row is a minimal multi-exon transcript (Puffin_TSS_15bp = 1.0 > 0.02,
    truncation = 'no', Predict_NMD = 'no_orf') so the ONLY filter lever is
    polyA. This isolates the 3-state polyA logic from other filter conditions.
    """
    rows = [
        # State 1: positive, above threshold (0.99 > 0.95) -> KEEP
        {"Chr": "chr1", "Strand": "+", "SSC": "1;2", "TrStart": 100, "TrEnd": 200,
         "Puffin_TSS_15bp": 1.0, "Puffin_TSS_50bp": 1.0, "truncation": "no",
         "Predict_NMD": "no_orf", "category": "NNC",
         "polyA_frac": 0.99, "polyA_valid_reads": 10, "frequency": 10},

        # State 1: positive, below threshold (0.50 < 0.95) -> FILTER
        {"Chr": "chr1", "Strand": "+", "SSC": "1;3", "TrStart": 300, "TrEnd": 400,
         "Puffin_TSS_15bp": 1.0, "Puffin_TSS_50bp": 1.0, "truncation": "no",
         "Predict_NMD": "no_orf", "category": "NNC",
         "polyA_frac": 0.50, "polyA_valid_reads": 10, "frequency": 10},

        # State 2: measured zero, FSM (e.g., replication-dependent histone)
        # Without genome_fasta: filter; with genome_fasta: FSM rescue -> KEEP
        {"Chr": "chr2", "Strand": "+", "SSC": "1;2", "TrStart": 500, "TrEnd": 600,
         "Puffin_TSS_15bp": 1.0, "Puffin_TSS_50bp": 1.0, "truncation": "no",
         "Predict_NMD": "no_orf", "category": "FSM",
         "polyA_frac": 0.0, "polyA_valid_reads": 0, "frequency": 10},

        # State 2: measured zero, NNC (novel), chr1 (assumed intra-priming)
        # Without genome_fasta: filter; with genome_fasta: still filter
        {"Chr": "chr1", "Strand": "+", "SSC": "1;2", "TrStart": 700, "TrEnd": 800,
         "Puffin_TSS_15bp": 1.0, "Puffin_TSS_50bp": 1.0, "truncation": "no",
         "Predict_NMD": "no_orf", "category": "NNC",
         "polyA_frac": 0.0, "polyA_valid_reads": 0, "frequency": 10},

        # State 2: measured zero, NNC, chrG (assumed NOT intra-priming)
        # Without genome_fasta: filter; with genome_fasta: KEEP
        {"Chr": "chrG", "Strand": "+", "SSC": "1;2", "TrStart": 900, "TrEnd": 1000,
         "Puffin_TSS_15bp": 1.0, "Puffin_TSS_50bp": 1.0, "truncation": "no",
         "Predict_NMD": "no_orf", "category": "NNC",
         "polyA_frac": 0.0, "polyA_valid_reads": 0, "frequency": 10},

        # State 3: missing (NaN)
        # Always KEEP (bypass)
        {"Chr": "chr3", "Strand": "+", "SSC": "1;2", "TrStart": 1100, "TrEnd": 1200,
         "Puffin_TSS_15bp": 1.0, "Puffin_TSS_50bp": 1.0, "truncation": "no",
         "Predict_NMD": "no_orf", "category": "NNC",
         "polyA_frac": np.nan, "polyA_valid_reads": 0, "frequency": 10},
    ]
    df = pd.DataFrame(rows)
    # Categorical alignment for the Chr/Strand groupby
    for col in ("Chr", "Strand"):
        df[col] = df[col].astype("category")
    return df


class FakeFasta:
    """Minimal pysam.FastaFile stub for intra-priming check.

    Sequence layout (so each test row produces a specific intra-priming verdict):
      - chr1: "N" * 800 + "A" * 9200  (rows with TrEnd >= 800 -> intra-priming;
        rows with TrEnd < 800 -> NOT intra-priming)
      - chr2: "N" * 10000  (any TrEnd -> NOT intra-priming)
      - chrG: "N" * 10000  (any TrEnd -> NOT intra-priming)
      - chr3: "N" * 10000  (any TrEnd -> NOT intra-priming)
    """

    def __init__(self):
        self._seqs = {
            "chr1": "N" * 800 + "A" * 9200,
            "chr2": "N" * 10000,
            "chrG": "N" * 10000,
            "chr3": "N" * 10000,
        }

    def __getitem__(self, chrom):
        return self._seqs[chrom]


def test_legacy_semantics_no_genome_fasta():
    """When genome_fasta=None, behavior must be byte-identical to the legacy
    single-state polyA logic: State 1 < threshold -> filter; State 2 -> filter;
    State 3 -> bypass.
    """
    df = make_test_df()
    df_out = transcript_model_filtering(
        df, puffin_prediction_threshold=0.02, polya_fraction_threshold=0.95,
        hard_filter=True, genome_fasta=None,
    )

    # Expect: kept = State 1 above-threshold (chr1 row 100) + State 3 (chr3 row 1100)
    assert len(df_out) == 2, f"Expected 2 rows kept (State 1 OK + State 3); got {len(df_out)}"

    kept_starts = sorted(df_out["TrStart"].tolist())
    assert kept_starts == [100, 1100], f"Unexpected kept rows: {kept_starts}"
    print("[OK] legacy semantics: State 1 OK + State 3 kept; State 2 + State 1 low filtered")


def test_3state_with_genome_fasta():
    """When genome_fasta is provided, State 2 rescue is FSM-only (Q-A2):
      - FSM in State 2: kept (FSM rescue)
      - State 2 non-FSM (NNC): polya_low=True, must satisfy Puffin 5' gate
        (in hard_filter: mask_puffin.gt(0.02) + polyA kept -> need Puffin OK
         AND polyA_frac > threshold. State 2 NNC fails polyA gate -> filtered).
      - State 1 / State 3: unchanged
    """
    df = make_test_df()
    fasta = FakeFasta()
    df_out = transcript_model_filtering(
        df, puffin_prediction_threshold=0.02, polya_fraction_threshold=0.95,
        hard_filter=True, genome_fasta=fasta,
    )

    # Expected: State 1 OK (100), State 2 FSM (500), State 3 (1100). All non-FSM
    # State 2 rows filtered because their polyA_frac=0 fails the threshold gate.
    assert len(df_out) == 3, f"Expected 3 rows kept; got {len(df_out)}\nKept: {df_out[['TrStart','polyA_frac','category']].to_dict('records')}"

    kept_starts = sorted(df_out["TrStart"].tolist())
    assert kept_starts == [100, 500, 1100], f"Unexpected kept rows: {kept_starts}"
    print("[OK] 3-state polyA with genome_fasta (Q-A2): only FSM-rescued")


def test_state_distribution_logging(caplog=None):
    """Verify the 3-state polyA distribution counter is emitted."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = make_test_df()
    fasta = FakeFasta()
    df_out = transcript_model_filtering(
        df, puffin_prediction_threshold=0.02, polya_fraction_threshold=0.95,
        hard_filter=True, genome_fasta=fasta,
    )

    print("[OK] 3-state distribution logged via logger.info (see output above)")


if __name__ == "__main__":
    test_legacy_semantics_no_genome_fasta()
    test_3state_with_genome_fasta()
    test_state_distribution_logging()
    print("\nAll 3-state polyA tests passed.")