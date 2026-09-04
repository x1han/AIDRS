"""Quick sanity test for non-hard_filter path (default) of 3-state polyA."""
import sys
import pandas as pd
import numpy as np

REPO = "/datf/hanxi/software/AIDRS/repo"
sys.path.insert(0, REPO)

from src.common import transcript_model_filtering


def make_non_hard_filter_df():
    """Non-hard_filter only fires when polya_frac_low combines with NMD,
    truncation='yes', OR ultra-low quality (Puffin_TSS_50bp='no'). All rows
    below have Puffin_TSS_50bp='no' so ultra_low_quality_filter is the active
    gate -- the only lever on/off is polya_frac_low (state-aware).
    """
    rows = [
        # State 1: low polyA_frac + ultra-low (Puffin_TSS_50bp='no')
        # polya_frac_low=True -> filter
        {"Chr": "chr1", "Strand": "+", "SSC": "1;2", "TrStart": 100, "TrEnd": 200,
         "Puffin_TSS_15bp": 1.0, "Puffin_TSS_50bp": "no", "truncation": "no",
         "Predict_NMD": "no_orf", "category": "NNC",
         "polyA_frac": 0.50, "polyA_valid_reads": 10, "frequency": 10},

        # State 1: OK polyA_frac -> polya_frac_low=False -> keep
        {"Chr": "chr1", "Strand": "+", "SSC": "1;3", "TrStart": 300, "TrEnd": 400,
         "Puffin_TSS_15bp": 1.0, "Puffin_TSS_50bp": "no", "truncation": "no",
         "Predict_NMD": "no_orf", "category": "NNC",
         "polyA_frac": 0.99, "polyA_valid_reads": 10, "frequency": 10},

        # State 2: zero, NNC, intra-priming (chr1 TrEnd=1500 -> end+20 = A-rich)
        # Legacy: polya_frac_low=True -> filter
        # With genome + intra-priming: polya_frac_low=True -> filter (same)
        {"Chr": "chr1", "Strand": "+", "SSC": "1;4", "TrStart": 500, "TrEnd": 1500,
         "Puffin_TSS_15bp": 1.0, "Puffin_TSS_50bp": "no", "truncation": "no",
         "Predict_NMD": "no_orf", "category": "NNC",
         "polyA_frac": 0.0, "polyA_valid_reads": 0, "frequency": 10},

        # State 2: zero, FSM, NOT intra-priming (chr2 TrEnd=700 -> end+20 = N)
        # Legacy: polya_frac_low=True -> filter
        # With genome + FSM rescue: polya_frac_low=False -> keep (NEW behavior)
        {"Chr": "chr2", "Strand": "+", "SSC": "1;2", "TrStart": 600, "TrEnd": 700,
         "Puffin_TSS_15bp": 1.0, "Puffin_TSS_50bp": "no", "truncation": "no",
         "Predict_NMD": "no_orf", "category": "FSM",
         "polyA_frac": 0.0, "polyA_valid_reads": 0, "frequency": 10},

        # State 2: zero, NNC, NOT intra-priming (chr2 TrEnd=900 -> end+20 = N)
        # Legacy: polya_frac_low=True -> filter
        # With genome + NOT intra-priming: polya_frac_low=False -> keep (NEW behavior)
        {"Chr": "chr2", "Strand": "+", "SSC": "1;3", "TrStart": 800, "TrEnd": 900,
         "Puffin_TSS_15bp": 1.0, "Puffin_TSS_50bp": "no", "truncation": "no",
         "Predict_NMD": "no_orf", "category": "NNC",
         "polyA_frac": 0.0, "polyA_valid_reads": 0, "frequency": 10},

        # State 3: NaN -> bypass (always keep)
        {"Chr": "chr3", "Strand": "+", "SSC": "1;2", "TrStart": 1000, "TrEnd": 1100,
         "Puffin_TSS_15bp": 1.0, "Puffin_TSS_50bp": "no", "truncation": "no",
         "Predict_NMD": "no_orf", "category": "NNC",
         "polyA_frac": np.nan, "polyA_valid_reads": 0, "frequency": 10},
    ]
    df = pd.DataFrame(rows)
    for col in ("Chr", "Strand"):
        df[col] = df[col].astype("category")
    return df


class FakeFasta:
    """chr1: A-rich after pos 1500 (TrEnd=1500 row -> intra-priming).
    chr2: all N (NOT intra-priming for any row).
    chr3: all N.
    """
    def __init__(self):
        self._seqs = {
            "chr1": "N" * 1500 + "A" * 8500,
            "chr2": "N" * 10000,
            "chr3": "N" * 10000,
        }
    def __getitem__(self, chrom):
        return self._seqs[chrom]


def test_non_hard_filter_legacy():
    """Non-hard_filter path with genome_fasta=None: legacy semantics.
    Only kept: State 1 OK (300) + State 3 (1000). State 1 low + all State 2 are
    filtered via ultra_low_quality_filter."""
    df = make_non_hard_filter_df()
    df_out = transcript_model_filtering(
        df, hard_filter=False, genome_fasta=None,
    )
    kept_starts = sorted(df_out["TrStart"].tolist())
    assert kept_starts == [300, 1000], f"non-hard_filter legacy: expected [300, 1000]; got {kept_starts}"
    print(f"[OK] non-hard_filter legacy: {len(df_out)} rows kept: {kept_starts}")


def test_non_hard_filter_with_genome():
    """Non-hard_filter path with genome_fasta: State 2 FSM-only rescue (Q-A2).
    Non-FSM State 2 rows get polya_low=True and must survive the 3-condition
    framework. With Puffin_TSS_50bp='no', ultra_low_quality_filter triggers
    for non-FSM State 2 rows -> all filtered.
    Expected kept: State 1 OK (300) + State 2 FSM (600) + State 3 (1000).
    """
    df = make_non_hard_filter_df()
    fasta = FakeFasta()
    df_out = transcript_model_filtering(
        df, hard_filter=False, genome_fasta=fasta,
    )
    kept_starts = sorted(df_out["TrStart"].tolist())
    assert kept_starts == [300, 600, 1000], (
        f"non-hard_filter with genome (Q-A2): expected [300, 600, 1000]; got {kept_starts}"
    )
    print(f"[OK] non-hard_filter with genome_fasta (Q-A2): {len(df_out)} rows kept: {kept_starts}")


if __name__ == "__main__":
    test_non_hard_filter_legacy()
    test_non_hard_filter_with_genome()
    print("\nAll non-hard_filter 3-state polyA tests passed.")