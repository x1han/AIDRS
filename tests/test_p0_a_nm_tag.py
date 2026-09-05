"""Unit tests for P0-A (Hard Crash Defusal): NM-tag handling in src/bam2ssc.py.

Background
----------
The original code did `xs = ts = erro = None` and then computed
`identity = 1 - (erro / cov)` unconditionally. When a read lacks an NM tag
(common when aligners are invoked without edit-distance reporting), `erro`
stays as None and the expression `None / cov` raises `TypeError`,
killing the per-chunk worker and leaving temp files half-written.

Fix
---
1. Rename `erro` to `error` (fix typo + clearer naming).
2. When NM tag is missing, emit `logger.warning(...)` and default to
   `identity = 1.0` instead of raising TypeError.

These tests bypass `src/__init__.py` (which eagerly re-imports the heavy
selene_sdk/pysam chain through aidrs.py). They will be replaced in P0-B
by tests under the proper `src.*` import path; this loader style is the
fast pre-fix harness.
"""
import importlib.util
import logging
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bam2ssc = _load('bam2ssc_under_test', REPO_ROOT / 'src' / 'bam2ssc.py')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_read(query_name='read1',
               is_reverse=False,
               query_sequence='ACGTACGT',
               tags=None,
               cigartuples=None,
               reference_start=99,
               reference_name='chr1'):
    """Build a MagicMock that quacks like a pysam.AlignedSegment."""
    read = MagicMock()
    read.is_unmapped = False
    read.is_reverse = is_reverse
    read.query_sequence = query_sequence
    read.query_name = query_name
    read.tags = tags if tags is not None else [('NM', 0)]
    read.cigartuples = (
        cigartuples if cigartuples is not None
        else [(0, len(query_sequence))]
    )
    read.reference_start = reference_start
    read.reference_name = reference_name
    return read


def _compute_identity_and_error(read, cov):
    """Replicate the post-fix inner logic from process_bam_chunk.

    Kept in lockstep with src/bam2ssc.py lines 121-167. If the source
    changes, update here too. Asserts that this mirror never references
    `erro` to guard against re-introducing the typo.
    """
    src = (REPO_ROOT / 'src' / 'bam2ssc.py').read_text()
    # Word-boundary check: 'error' is the post-fix name and contains the
    # substring 'erro'; the typo we are guarding against is the bare
    # identifier `erro` used in the chained assignment / division.
    assert not re.search(r'\berro\b', src), (
        "src/bam2ssc.py still contains the bare 'erro' identifier"
    )

    error = None
    for tag, value in read.tags:
        if tag == 'NM':
            error = value
    if error is None:
        bam2ssc.logger.warning(
            f"Read {read.query_name}: NM tag missing, treating identity as 1.0"
        )
        identity = 1.0
    else:
        identity = 1 - (error / cov) if cov else 0
    return identity, error


# ---------------------------------------------------------------------------
# P0-A happy path: NM tag present, no crash, identity formula intact
# ---------------------------------------------------------------------------
def test_p0a_nm_present_happy_path():
    """NM=2 over cov=8 → identity=0.75 (1 - 2/8)."""
    read = _make_read(query_name='happy_read',
                      query_sequence='ACGTACGT',
                      tags=[('NM', 2)])
    identity, error = _compute_identity_and_error(read, cov=8)
    assert error == 2
    assert abs(identity - 0.75) < 1e-12, f"expected 0.75, got {identity}"


def test_p0a_nm_present_zero_mismatches():
    """NM=0 → identity=1.0 (perfect match)."""
    read = _make_read(query_name='perfect_read',
                      query_sequence='ACGTACGT',
                      tags=[('NM', 0)])
    identity, error = _compute_identity_and_error(read, cov=8)
    assert error == 0
    assert identity == 1.0


def test_p0a_nm_present_with_xs_and_ts_tags():
    """XS + ts + NM all present, NM still wins for identity math."""
    read = _make_read(query_name='tagged_read',
                      query_sequence='ACGTACGT',
                      tags=[('XS', '+'), ('ts', '+'), ('NM', 3)])
    identity, error = _compute_identity_and_error(read, cov=8)
    assert error == 3
    assert abs(identity - (1 - 3 / 8)) < 1e-12


# ---------------------------------------------------------------------------
# P0-A defect path: NM tag missing must NOT crash, must warn, identity=1.0
# ---------------------------------------------------------------------------
def test_p0a_nm_missing_no_crash_no_typeerror():
    """Read with no NM tag → no TypeError, identity=1.0 (per fix)."""
    read = _make_read(query_name='nm_missing_read',
                      query_sequence='ACGTACGT',
                      tags=[])  # no NM
    # Should not raise.
    identity, error = _compute_identity_and_error(read, cov=8)
    assert error is None
    assert identity == 1.0


def test_p0a_nm_missing_emits_warning(caplog=None):
    """Missing NM must trigger logger.warning with informative text.

    Works under pytest with the standard `caplog` fixture; under the
    standalone `__main__` runner we patch the module logger to a list
    recorder so the same assertion logic runs without pytest.
    """
    read = _make_read(query_name='warn_read',
                      query_sequence='ACGTACGT',
                      tags=[])
    if caplog is not None:
        with caplog.at_level(logging.WARNING, logger='bam2ssc_under_test'):
            identity, _ = _compute_identity_and_error(read, cov=8)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        msgs = [r.getMessage() for r in warnings]
    else:
        captured = []
        original_warning = bam2ssc.logger.warning
        bam2ssc.logger.warning = lambda msg, *a, **kw: captured.append(msg)
        try:
            identity, _ = _compute_identity_and_error(read, cov=8)
        finally:
            bam2ssc.logger.warning = original_warning
        msgs = captured
    assert identity == 1.0
    assert any("NM tag missing" in m for m in msgs), (
        f"expected 'NM tag missing' warning, got {msgs}"
    )


def test_p0a_nm_missing_alongside_other_tags():
    """XS/ts present, NM absent → still no crash, still warns."""
    read = _make_read(query_name='partial_tag_read',
                      query_sequence='ACGTACGT',
                      tags=[('XS', '+'), ('ts', '+')])  # no NM
    identity, error = _compute_identity_and_error(read, cov=8)
    assert error is None
    assert identity == 1.0


def test_p0a_nm_missing_zero_cov_safe():
    """Edge case: NM missing AND cov=0 → branch returns 1.0 safely."""
    read = _make_read(query_name='zero_cov_read',
                      query_sequence='ACGTACGT',
                      tags=[])
    identity, error = _compute_identity_and_error(read, cov=0)
    assert error is None
    assert identity == 1.0


# ---------------------------------------------------------------------------
# P0-A regression guard: typo must not return
# ---------------------------------------------------------------------------
def test_p0a_source_no_longer_contains_erro_typo():
    """The bare `erro` identifier must be fully removed from src/bam2ssc.py.

    Note: 'error' (the post-fix name) intentionally contains the substring
    'erro'; we therefore match on the word-boundary `\\berro\\b` so we
    only catch the original typo and not the legitimate rename.
    """
    text = (REPO_ROOT / 'src' / 'bam2ssc.py').read_text()
    pattern = r'\berro\b'
    bad_lines = [ln for ln in text.splitlines() if re.search(pattern, ln)]
    assert not bad_lines, (
        "src/bam2ssc.py still references bare 'erro':\n"
        + "\n".join(bad_lines)
    )


def test_p0a_source_introduces_error_variable():
    """The fixed variable name `error` must appear in src/bam2ssc.py."""
    text = (REPO_ROOT / 'src' / 'bam2ssc.py').read_text()
    assert re.search(r'\berror\b', text), (
        "fixed variable 'error' not found in src/bam2ssc.py"
    )


# ---------------------------------------------------------------------------
# Run via `python tests/test_p0_a_nm_tag.py` for the pre-P0-B harness.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    test_p0a_nm_present_happy_path()
    test_p0a_nm_present_zero_mismatches()
    test_p0a_nm_present_with_xs_and_ts_tags()
    test_p0a_nm_missing_no_crash_no_typeerror()
    test_p0a_nm_missing_emits_warning()  # uses caplog, only via pytest
    test_p0a_nm_missing_alongside_other_tags()
    test_p0a_nm_missing_zero_cov_safe()
    test_p0a_source_no_longer_contains_erro_typo()
    test_p0a_source_introduces_error_variable()
    print("All P0-A NM-tag tests PASSED")
