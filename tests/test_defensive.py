"""A-class defensive unit tests for AIDRS.

Pin specific defensive code paths so a refactor that breaks byte-identity
or fail-loud semantics fails locally in <90s instead of after a 22-min
SGE rerun. Each test pins one narrow contract:

- column_registry: schema validator never auto-pads missing CORE cols.
- common: safe parsers never raise on garbage input.
- column_standardize: reverse_complement handles IUPAC + case-insensitive.
- chrom_check: dominant-style detector.
- concurrency: drain_futures_loud fail-loud + get_process_pool factory +
  worker_death_pact no-op on non-Linux.
- resource_guard: priority of explicit > NSLOTS > SLURM > host cap, and
  cgroup max-sentinel fallback to 32 GB default.
- ISM_filter: _parse_introns / _is_contiguous_intron_subchain, plus Puffin
  coercion via the public TruncationProcessor API.
- generate_reports: polyA_len_profile all-NaN skip path + empty-temp-dir
  empty-tables path.
- single_exon: 5-pillar funnel (P1/P2/P3/P4/P5).

All tests run in plain Python (aidrs env) — no torch, no real BAM, no selene.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import pytest

# repo/ on path so "from src.aidrs_runtime..." resolves (dev-mode import).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.aidrs_runtime.column_registry import (
    validate_canonical_schema,
    resolve_assessment_columns,
    CORE_ASSESSMENT_COLS,
    OPTIONAL_ASSESSMENT_COLS,
)
from src.common import safe_ssc_array, safe_int_tuple
from src.aidrs_runtime.column_standardize import reverse_complement
from src.aidrs_runtime.chrom_check import _dominant_style
from src.aidrs_runtime.concurrency import (
    drain_futures_loud,
    get_process_pool,
    _worker_death_pact,
)
from src.aidrs_runtime.resource_guard import ResourceGuard
from src.ISM_filter import (
    _parse_introns,
    _is_contiguous_intron_subchain,
    TruncationProcessor,
)
from src.single_exon import (
    evaluate_single_exon_isoform,
    apply_5_pillar_funnel,
    check_genomic_intra_priming,
)
from src.generate_reports import IsoformAnnotator


# =============================================================================
# 1-2. column_registry.validate_canonical_schema
# =============================================================================

def test_validate_canonical_schema_missing_core():
    """df missing a CORE col must raise RuntimeError, never auto-pad.

    Auto-padding would silently fabricate a downstream-looking TSV that
    no upstream stage actually produced — a stage-pipeline bug would
    ship as a "valid" assessment TSV. (column_registry.py:89)
    """
    df = pd.DataFrame({"Chr": ["chr1"], "Strand": ["+"]})  # only 2 of 19 CORE cols
    with pytest.raises(RuntimeError, match="SCHEMA DRIFT"):
        validate_canonical_schema(df)


def test_validate_canonical_schema_pass():
    """df with all CORE cols passes; returns the CORE list for chaining."""
    df = pd.DataFrame({c: [0] for c in CORE_ASSESSMENT_COLS})
    out = validate_canonical_schema(df)
    assert out == CORE_ASSESSMENT_COLS


# =============================================================================
# 3. column_registry.resolve_assessment_columns
# =============================================================================

def test_resolve_assessment_columns_appends_optional():
    """OPTIONAL cols present on df are appended in declared order (column_registry.py:74)."""
    df = pd.DataFrame(
        {c: [0] for c in CORE_ASSESSMENT_COLS + list(OPTIONAL_ASSESSMENT_COLS)}
    )
    cols = resolve_assessment_columns(df)
    # CORE in declared order first
    for i, c in enumerate(CORE_ASSESSMENT_COLS):
        assert cols[i] == c, f"CORE order violated at index {i}: {cols[i]} vs {c}"
    # Then OPTIONAL in declared order, no duplicates
    tail = cols[len(CORE_ASSESSMENT_COLS):]
    assert tail == list(OPTIONAL_ASSESSMENT_COLS)
    assert len(cols) == len(set(cols))


# =============================================================================
# 4-5. common.safe_ssc_array
# =============================================================================

def test_safe_ssc_array_garbage_inputs():
    """'NA'/'none'/None/'abc' → empty int64 array, never raise (common.py:44)."""
    for bad in ["NA", "none", "", None, "abc"]:
        out = safe_ssc_array(bad)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.int64
        assert len(out) == 0, f"expected empty for input {bad!r}, got {out}"


def test_safe_ssc_array_normal():
    """'100-200-300' → [100, 200, 300] (common.py:44)."""
    out = safe_ssc_array("100-200-300")
    assert out.tolist() == [100, 200, 300]
    assert out.dtype == np.int64


# =============================================================================
# 6-7. common.safe_int_tuple
# =============================================================================

def test_safe_int_tuple_garbage():
    """'NA'/'abc'/'1.5' → None (common.py:56)."""
    for bad in ["NA", "abc", "1.5", "", None]:
        assert safe_int_tuple(bad) is None, f"expected None for {bad!r}"


def test_safe_int_tuple_normal():
    """'1-2-3' → (1, 2, 3) (common.py:56)."""
    assert safe_int_tuple("1-2-3") == (1, 2, 3)


# =============================================================================
# 8. column_standardize.reverse_complement IUPAC
# =============================================================================

def test_reverse_complement_iupac():
    """Basic ACGT + case-insensitive + N (column_standardize.py:27).

    NOTE: column_standardize.reverse_complement is the basic 5-base version
    (A/T/C/G/N). IUPAC codes (RYSWKMBDHV) are handled by common.rev_comp,
    which is the function consumed by single_exon.check_genomic_intra_priming.
    We pin both contracts here.
    """
    # column_standardize.reverse_complement: ACGT only
    assert reverse_complement("ATGC") == "GCAT"
    # Case insensitive
    assert reverse_complement("atgc") == "GCAT"
    assert reverse_complement("AtGc") == "GCAT"
    # N is self-complement
    assert reverse_complement("NN") == "NN"
    assert reverse_complement("anN") == "NNT"
    # IUPAC unknown → unknown bases pass through (complement.get fallback)
    assert reverse_complement("R") == "R"
    assert reverse_complement("Y") == "Y"

    # common.rev_comp: IUPAC + soft-mask
    from src.common import rev_comp
    assert rev_comp("R") == "Y"
    assert rev_comp("Y") == "R"
    assert rev_comp("S") == "S"
    assert rev_comp("W") == "W"
    assert rev_comp("K") == "M"
    assert rev_comp("M") == "K"
    assert rev_comp("B") == "V"
    assert rev_comp("V") == "B"
    assert rev_comp("D") == "H"
    assert rev_comp("H") == "D"
    assert rev_comp("NN") == "NN"
    # Mixed IUPAC string round-trip
    assert rev_comp("ACGTNRYSWKMBDHV") == "BDHVKMWSRYNACGT"
    # Soft-mask characters preserved
    assert rev_comp("ACGT-N") == "N-ACGT"


# =============================================================================
# 9. chrom_check._dominant_style
# =============================================================================

def test_dominant_style_with_chr_prefix():
    """chr-prefixed majority → True; bare-number majority → False (chrom_check.py:29)."""
    assert _dominant_style({"chr1", "chr2", "chr3"}) is True
    assert _dominant_style({"chr1", "chr2"}) is True
    assert _dominant_style({"1", "2", "3"}) is False
    assert _dominant_style({"1", "2"}) is False
    # Tied: n_with == n_without → False (strict >)
    assert _dominant_style({"chr1", "1"}) is False


# =============================================================================
# 10-13. concurrency helpers
# =============================================================================

def _raise_zero_div(_):
    raise ZeroDivisionError("synthetic worker death")


def _ok_return(x):
    return x * 2


def test_drain_futures_loud_raises_on_worker_death():
    """allow_partial=False + worker raises → RuntimeError chained from original (concurrency.py:47)."""
    pool = get_process_pool(2)
    try:
        futures = [pool.submit(_raise_zero_div, i) for i in range(3)]
        with pytest.raises(RuntimeError, match="FATAL"):
            drain_futures_loud(futures, stage_name="TEST-10", allow_partial=False)
    finally:
        for f in futures:
            f.cancel() if not f.done() else None  # safe cleanup
        pool.shutdown(wait=False)


def test_drain_futures_loud_partial_returns_survivors():
    """allow_partial=True: 2 OK + 1 failing → returns 2 results (concurrency.py:47)."""
    pool = get_process_pool(2)
    try:
        futures = [
            pool.submit(_ok_return, 1),
            pool.submit(_raise_zero_div, 99),
            pool.submit(_ok_return, 3),
        ]
        results = drain_futures_loud(futures, stage_name="TEST-11", allow_partial=True)
        assert sorted(results) == [2, 6]
    finally:
        pool.shutdown(wait=False)


def test_get_process_pool_smoke():
    """get_process_pool(2) returns usable executor; tasks complete (concurrency.py:35)."""
    with get_process_pool(2) as pool:
        futs = [pool.submit(_ok_return, i) for i in range(4)]
        results = sorted(f.result() for f in futs)
    assert results == [0, 2, 4, 6]


def test_worker_death_pact_no_op_on_non_linux(monkeypatch):
    """_worker_death_pact swallows when ctypes.CDLL fails (concurrency.py:18).

    Simulates a non-Linux / missing-libc host by raising OSError from
    ctypes.CDLL. The function MUST swallow (best-effort contract) and
    not propagate the exception. If someone refactors to drop the
    `except Exception: pass`, this test will fail.
    """
    import ctypes as _ctypes

    def _raise_oserror(_name):
        raise OSError("simulated missing libc.so.6 (non-Linux / RHEL 7 alt)")

    monkeypatch.setattr(_ctypes, "CDLL", _raise_oserror)
    _worker_death_pact()  # must not raise → pass


# =============================================================================
# 14-17. ResourceGuard
# =============================================================================

@pytest.fixture
def clean_env(monkeypatch):
    """Strip AIDRS-detected env vars so tests see a deterministic baseline."""
    for var in ("NSLOTS", "SLURM_CPUS_PER_TASK", "SGE_H_VMEM"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_resource_guard_cpu_threads_explicit_wins(clean_env):
    """Explicit threads=2 beats NSLOTS=4 beats SLURM=8 (resource_guard.py:40)."""
    clean_env.setenv("NSLOTS", "4")
    clean_env.setenv("SLURM_CPUS_PER_TASK", "8")
    assert ResourceGuard.get_effective_cpu_threads(2) == 2


def test_resource_guard_cpu_threads_nslots_fallback(clean_env):
    """No explicit, NSLOTS=4, SLURM=8 → returns NSLOTS=4 (resource_guard.py:40)."""
    clean_env.setenv("NSLOTS", "4")
    clean_env.setenv("SLURM_CPUS_PER_TASK", "8")
    assert ResourceGuard.get_effective_cpu_threads(None) == 4


def test_resource_guard_cpu_threads_slurm_fallback(clean_env):
    """No explicit, no NSLOTS, SLURM=12 → returns 12 (resource_guard.py:40)."""
    clean_env.setenv("SLURM_CPUS_PER_TASK", "12")
    assert ResourceGuard.get_effective_cpu_threads(None) == 12


def test_resource_guard_memory_limit_default_when_cgroup_max(clean_env, monkeypatch):
    """No SGE env, cgroup files absent → 32.0 GB default (resource_guard.py:84).

    Force both cgroup paths to look absent via os.path.exists override so
    the test is robust against any container (Docker/K8s/podman) where
    /sys/fs/cgroup/memory/* might be mounted by default. After the cgroup
    branches fall through, SGE_H_VMEM is also unset (clean_env), so the
    function lands on the 32.0 GB default.
    """
    real_exists = os.path.exists

    def fake_exists(path):
        if path in (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory.max",
        ):
            return False
        return real_exists(path)

    monkeypatch.setattr(os.path, "exists", fake_exists)
    assert ResourceGuard.get_memory_limit_gb() == 32.0


def test_resource_guard_memory_limit_sge_env(clean_env):
    """SGE_H_VMEM=8589934592 (8 GB) → 8.0 (resource_guard.py:84)."""
    clean_env.setenv("SGE_H_VMEM", "8589934592")  # 8 GiB
    assert ResourceGuard.get_memory_limit_gb() == 8.0


# =============================================================================
# 18. ISM_filter._parse_introns safe on bad SSC
# =============================================================================

def test_parse_introns_safe_on_bad_ssc():
    """'NA'/'abc'/odd-token count → [] (ISM_filter.py:11).

    Note: positions = [start] + SSC_tokens + [end] must have EVEN total length
    to form an exon chain. So SSC must have an EVEN number of tokens (a
    transcript with N exons encodes 2*(N-1) token positions in SSC).
    """
    # 'NA'
    assert _parse_introns(100, "NA", 200) == []
    # None / NaN
    assert _parse_introns(100, None, 200) == []
    assert _parse_introns(100, float("nan"), 200) == []
    # Garbage
    assert _parse_introns(100, "abc", 200) == []
    # Empty
    assert _parse_introns(100, "", 200) == []
    # Odd token count → 0 introns (defensive guard)
    assert _parse_introns(100, "150", 200) == []
    assert _parse_introns(100, "1-2-3", 200) == []
    # Normal single-intron: SSC has 2 tokens → positions = [start, t1, t2, end]
    # → 1 intron (t1, t2).
    assert _parse_introns(100, "150-200", 300) == [(150, 200)]
    # Normal 2-intron: SSC has 4 tokens → 2 introns.
    assert _parse_introns(100, "150-200-250-300", 400) == [(150, 200), (250, 300)]


# =============================================================================
# 19. ISM_filter._is_contiguous_intron_subchain
# =============================================================================

def test_is_contiguous_intron_subchain_logic():
    """A ⊆ B iff A's intron chain is a contiguous ordered sub-chain of B (ISM_filter.py:42)."""
    target = [(1, 2), (3, 4), (5, 6)]
    # Contiguous prefix
    assert _is_contiguous_intron_subchain([(1, 2)], target) is True
    # Contiguous middle
    assert _is_contiguous_intron_subchain([(3, 4)], target) is True
    # Full chain → False (must be strictly shorter)
    assert _is_contiguous_intron_subchain(target, target) is False
    # Longer than target → False
    assert _is_contiguous_intron_subchain(target + [(7, 8)], target) is False
    # Out of order → False (A3SS-style divergent isoform)
    assert _is_contiguous_intron_subchain([(2, 1)], target) is False
    # Non-contiguous (skip one in target) → False
    assert _is_contiguous_intron_subchain([(1, 2), (5, 6)], target) is False
    # Empty query → False
    assert _is_contiguous_intron_subchain([], target) is False
    # Empty target → False
    assert _is_contiguous_intron_subchain([(1, 2)], []) is False


# =============================================================================
# 20. ISM_filter.TruncationProcessor Puffin coercion (via public API)
# =============================================================================

def test_puffin_value_coercion_via_public_api():
    """Puffin_TSS_15bp coerces via nested _puffin (ISM_filter.py:93).

    We drive the row-level _assess_truncation_for_Chr method directly
    (it's a public method on TruncationProcessor, not nested — only the
    _puffin helper inside it is nested). The diagnostic intermediate
    column 'truncation_source' is observable here without going through
    the threshold-classification layer of `assess_truncation`, so we
    pin only the Puffin-coercion contract:
      - puffin >= 0.1 → rescue fires → truncation_source = 'full'
      - puffin  < 0.1 → rescue skipped → truncation_source != 'full'
    """
    df = pd.DataFrame(
        {
            "Chr": ["chr1"] * 6,
            "Strand": ["+"] * 6,
            "Group": [1, 1, 1, 1, 1, 1],
            "uniqueTr": ["target", "low_puffin", "at_threshold",
                         "high_puffin", "just_below", "na_str"],
            "TrStart": [100, 100, 100, 100, 100, 100],
            "TrEnd": [500, 500, 500, 500, 500, 500],
            # Target: 4 tokens → 2 introns [(150,250),(350,450)].
            # Candidates: 2 tokens "150-250" → 1 intron [(150,250)],
            # contiguous sub-chain of target's introns.
            "SSC": [
                "150-250-350-450",
                "150-250",
                "150-250",
                "150-250",
                "150-250",
                "150-250",
            ],
            "Puffin_TSS_15bp": [
                0.0,    # target — self-skipped
                0.05,   # below 0.1 → NOT rescued → has source(s)
                0.10,   # AT threshold → rescued → 'full'
                0.5,    # well above → rescued → 'full'
                0.099,  # just below → NOT rescued → has source(s)
                "NA",   # coerces to 0.0 → NOT rescued → has source(s)
            ],
            "frequency": [10.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    tp = TruncationProcessor(puffin_tss_rescue=0.1)
    # Call the per-Chr helper directly to observe the diagnostic
    # 'truncation_source' column BEFORE threshold classification drops it.
    out = tp._assess_truncation_for_Chr(df)

    assert "truncation_source" in out.columns
    by_tr = {r["uniqueTr"]: r["truncation_source"] for _, r in out.iterrows()}

    # Target: same SSC against itself is sibling-skipped → no source → 'full'.
    assert by_tr["target"] == "full", (
        f"target (self-skipped) should be 'full', got {by_tr['target']!r}"
    )
    # Puffin < 0.1: rescue does NOT fire → has source(s) → truncation_source != 'full'.
    assert by_tr["low_puffin"] != "full", (
        f"puffin=0.05 (below 0.1) should have a source, got "
        f"{by_tr['low_puffin']!r}"
    )
    assert by_tr["just_below"] != "full", (
        f"puffin=0.099 (just below 0.1) should have a source, got "
        f"{by_tr['just_below']!r}"
    )
    assert by_tr["na_str"] != "full", (
        f"puffin='NA' (coerces to 0.0) should have a source, got "
        f"{by_tr['na_str']!r}"
    )
    # Puffin >= 0.1: rescue fires → 'full'.
    assert by_tr["at_threshold"] == "full", (
        f"puffin=0.10 (rescue at threshold) should be 'full', got "
        f"{by_tr['at_threshold']!r}"
    )
    assert by_tr["high_puffin"] == "full", (
        f"puffin=0.5 (well above threshold) should be 'full', got "
        f"{by_tr['high_puffin']!r}"
    )


# =============================================================================
# 21-22. generate_reports.IsoformAnnotator.polyA_len_profile
# =============================================================================

def test_polyA_len_profile_skips_all_nan_polyA(tmp_path):
    """polyA_frac all-NaN → returns (df, {}) with empty dict (generate_reports.py:556).

    This pins the polyA-auto-skip path: when no input BAM has a 'pt' tag,
    polyA_frac is all-NaN, and the writer loop MUST skip the parquet
    emission (otherwise the sidecar carries all-zero tail lengths).
    """
    annotator = IsoformAnnotator(num_processes=1)
    df = pd.DataFrame(
        {
            "Chr": ["chr1"],
            "Strand": ["+"],
            "SSC": ["100-200"],
            "TrStart": [100],
            "TrEnd": [200],
            "polyA_frac": [np.nan],
        }
    )
    out_df, sidecar_dict = annotator.polyA_len_profile(df, str(tmp_path))
    # Returns the original df unmodified.
    assert out_df is df or out_df.equals(df)
    # Sidecar dict is empty → no parquet emission.
    assert sidecar_dict == {}


def test_polyA_len_profile_emits_empty_tables_on_no_flnc_files(tmp_path):
    """Valid polyA_frac but no flnc files in temp/ → returns (df, dict with empty DataFrames)."""
    annotator = IsoformAnnotator(num_processes=1)
    df = pd.DataFrame(
        {
            "Chr": ["chr1", "chr1"],
            "Strand": ["+", "+"],
            "SSC": ["100-200", "300-400"],
            "TrStart": [100, 300],
            "TrEnd": [200, 400],
            "polyA_frac": [0.8, 0.6],
        }
    )
    # tmp_path has no temp/ subdir → glob finds nothing → empty-tables branch.
    out_df, sidecar_dict = annotator.polyA_len_profile(df, str(tmp_path))
    assert "transcript.polyA_len" in sidecar_dict
    assert "gene.polyA_len" in sidecar_dict
    # The empty-tables branch creates empty DataFrames. We pin the
    # REQUIRED schema columns (subset assertion) rather than the exact
    # full list, so a refactor that adds e.g. raw_polyA_lengths to the
    # empty-tables branch doesn't break this test.
    tr = sidecar_dict["transcript.polyA_len"]
    gn = sidecar_dict["gene.polyA_len"]
    required_tr = {"TrID", "GeneID", "GeneName",
                   "polyA_median", "polyA_mean", "polyA_count"}
    required_gn = {"GeneID", "polyA_median", "polyA_mean", "polyA_count"}
    assert required_tr.issubset(set(tr.columns)), (
        f"transcript.polyA_len missing required cols: "
        f"{required_tr - set(tr.columns)}"
    )
    assert required_gn.issubset(set(gn.columns)), (
        f"gene.polyA_len missing required cols: "
        f"{required_gn - set(gn.columns)}"
    )
    assert len(tr) == 0
    assert len(gn) == 0


# =============================================================================
# 23. single_exon.evaluate_single_exon_isoform Puffin threshold
# =============================================================================

def _make_single_exon_row(
    puffin, frequency=20, seq_len=300, is_intergenic=True,
    polyA_frac=0.8, polyA_valid_reads=5,
):
    return {
        "Chr": "chr1",
        "Strand": "+",
        "TrStart": 100,
        "TrEnd": 400,
        "Puffin_TSS_15bp": puffin,
        "frequency": frequency,
        "seq_len": seq_len,
        "is_intergenic_or_antisense": is_intergenic,
        "polyA_frac": polyA_frac,
        "polyA_valid_reads": polyA_valid_reads,
    }


def test_evaluate_single_exon_isoform_puffin_threshold():
    """Pillar 2: Puffin >= 0.1 passes; below fails; 'no' coerces to 0.0 → fails (single_exon.py:31)."""
    row = _make_single_exon_row(puffin=0.05)
    ok, reason = evaluate_single_exon_isoform(
        row, has_valid_polya=True, polyA_thresh=0.5, filter_freq=5,
    )
    assert ok is False
    assert reason == "P2_weak_tss"

    row = _make_single_exon_row(puffin=0.10)
    ok, reason = evaluate_single_exon_isoform(
        row, has_valid_polya=True, polyA_thresh=0.5, filter_freq=5,
    )
    # May still fail on P3 (polyA_frac 0.8 >= 0.5 ✓, polyA_valid_reads 5 >= 3 ✓
    # → no P3 fail). All other pillars pass → PASS.
    assert ok is True, f"expected PASS at puffin=0.10, got reason={reason}"
    assert reason == "PASS"

    # 'no' coerces to 0.0 → Pillar 2 fail.
    row = _make_single_exon_row(puffin="no")
    ok, reason = evaluate_single_exon_isoform(
        row, has_valid_polya=True, polyA_thresh=0.5, filter_freq=5,
    )
    assert ok is False
    assert reason == "P2_weak_tss"


# =============================================================================
# 24. single_exon.check_genomic_intra_priming strand + RC
# =============================================================================

class _FakeFasta:
    """Minimal mock for pysam.FastaFile.fetch — returns the same string regardless of args.

    Using a single-sequence fake (instead of a keyed dict) decouples the
    test from the `+20` window-size constant in single_exon.py: if the
    window changes from 20 to e.g. 25, this test still passes.
    """

    def __init__(self, default_seq):
        self._default = default_seq

    def fetch(self, chrom, start, end):
        return self._default


def test_intra_priming_window_rc_strand():
    """+ strand A-rich → True; - strand A-rich (after RC) → True; GCGC → False (single_exon.py:6)."""
    # + strand: A-rich downstream → flagged.
    fa = _FakeFasta("AAAAAAAAAAGGGGGGGGGG")
    assert check_genomic_intra_priming("chr1", 100, 200, "+", fa) is True

    # - strand: raw genomic fetch is non-A-rich; rev_comp makes it A-rich → flagged.
    # The raw sequence below is CCCCCCCCC TTTTTTTTTT — non-A-rich; after
    # rev_comp (T→A, C→G) it becomes AAAAAAAAAAGGGGGGGGGG — A-rich → flagged.
    fa = _FakeFasta("CCCCCCCCCCTTTTTTTTTT")
    assert check_genomic_intra_priming("chr1", 100, 200, "-", fa) is True

    # GCGC on + strand: neither A-rich nor has 6xA run → not flagged.
    fa = _FakeFasta("GCGCGCGCGCGCGCGCGCGC")
    assert check_genomic_intra_priming("chr1", 100, 200, "+", fa) is False


# =============================================================================
# 25. single_exon.apply_5_pillar_funnel integration
# =============================================================================

def test_5_pillar_funnel_integration():
    """6-row fixture: each pillar's pass/fail combination routed correctly (single_exon.py:56)."""
    rows = [
        # Row 0: all pillars pass → kept (PASS).
        _make_single_exon_row(puffin=0.5, polyA_frac=0.8, polyA_valid_reads=5),
        # Row 1: Pillar 1 fail (genic overlap).
        _make_single_exon_row(puffin=0.5, polyA_frac=0.8, polyA_valid_reads=5,
                              is_intergenic=False),
        # Row 2: Pillar 2 fail (weak TSS).
        _make_single_exon_row(puffin=0.05, polyA_frac=0.8, polyA_valid_reads=5),
        # Row 3: Pillar 3 fail (polyA below threshold).
        _make_single_exon_row(puffin=0.5, polyA_frac=0.1, polyA_valid_reads=5),
        # Row 4: Pillar 4 fail (low frequency, below max(15, 3*5)=15).
        _make_single_exon_row(puffin=0.5, polyA_frac=0.8, polyA_valid_reads=5,
                              frequency=10),
        # Row 5: Pillar 5 fail (short length, below 200).
        _make_single_exon_row(puffin=0.5, polyA_frac=0.8, polyA_valid_reads=5,
                              seq_len=150),
    ]
    df = pd.DataFrame(rows)
    kept, counts = apply_5_pillar_funnel(
        df_single=df, has_valid_polya=True, polyA_thresh=0.5, filter_freq=5,
    )
    # Row 0 is the only keeper.
    assert len(kept) == 1
    assert counts["P1_genic_overlap"] == 1
    assert counts["P2_weak_tss"] == 1
    assert counts["P3_polya_fail"] == 1
    assert counts["P4_low_freq"] == 1
    assert counts["P5_short"] == 1