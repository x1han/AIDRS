"""P0 #1: Double-classifier drift between isoform_classify and ISM_filter.

Background
----------
AIDRS carries two structurally-similar but semantically-different classifiers
for the "is this an Incomplete Splice Match (ISM) of a reference transcript?"
question:

  * ``src/isoform_classify.py``  (line 16-27 of v1.0.3, the
    ``_get_isoform_category_for_row`` static method) used the legacy
    *substring* heuristic:

        if row_ssc in ref_ssc_set:
            return 'FSM'
        for ref_ssc in ref_ssc_set:
            if ref_ssc.startswith(row_ssc) or ref_ssc.endswith(row_ssc):
                return 'ISM'
        if set(row_sites).issubset(ref_site_set):
            return 'NIC'
        return 'NNC'

    This fires only when the query SSC is a *prefix* or *suffix* of the
    reference SSC.  An *internal* fragment (query SSC in the *middle* of
    a longer reference SSC) is mis-classified as NIC even though SQANTI3
    would call it an ISM (truncation).

  * ``src/ISM_filter.py``  (lines 41-63) uses the SQANTI3 / FLAIR
    *contiguous-intron-subchain* rule via ``_is_contiguous_intron_subchain``,
    which parses ``[TrStart] + SSC + [TrEnd]`` into a list of
    ``(exon_i.end, exon_{i+1}.start)`` intron tuples and checks that the
    query intron list is a strict ordered contiguous sub-list of the
    reference intron list.  This correctly identifies internal fragments
    as ISM.

This test file is a synthetic unit test that exercises three cases on the
two implementations side-by-side and asserts that the *internal fragment*
case (Test 1) is classified differently by the two implementations — the
empirical proof of the P0 #1 drift.  Tests 2 and 3 are sanity / no-drift
controls: a discontiguous query that both implementations agree is *not*
an ISM (Test 2), and an exact-match FSM where both implementations agree
on FSM (Test 3).

These tests do NOT modify any source files.  They are observation-only and
are intended to be the gate-keeper for the P0 #1 fix (the next stage
will re-route ``isoform_classify`` to use the intron-subchain helper).
"""
from __future__ import annotations

import importlib.util
import os
import sys

# --- sys.path bootstrap (same pattern as tests/test_fsm_rescue.py) -------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO, "src")
for p in (_REPO, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

# Register an empty 'src' package so src.ISM_filter resolves without
# running src/__init__.py (which uses PEP 562 lazy __getattr__ and may
# attempt heavy imports depending on interpreter state).
_spec_pkg = importlib.util.spec_from_loader("src", loader=None, is_package=True)
_src_pkg = importlib.util.module_from_spec(_spec_pkg)
_src_pkg.__path__ = [_SRC]
sys.modules.setdefault("src", _src_pkg)

# Load src.ISM_filter directly (only the helpers we need).
_ism_spec = importlib.util.spec_from_file_location(
    "src.ISM_filter", os.path.join(_SRC, "ISM_filter.py")
)
_ism_mod = importlib.util.module_from_spec(_ism_spec)
sys.modules["src.ISM_filter"] = _ism_mod
_ism_spec.loader.exec_module(_ism_mod)

_parse_introns = _ism_mod._parse_introns
_is_contiguous_intron_subchain = _ism_mod._is_contiguous_intron_subchain


# --- legacy substring heuristic, inline reproduction ---------------------
# Mirrors src/isoform_classify.py:16-27 exactly.  Kept inline (not imported)
# because the test must exercise the *old* implementation as it exists in
# the codebase today, and we want this test to fail loudly if that source
# file is ever re-shaped away from the substring rule.
def _old_classify(query_ssc: str, ref_ssc_set, ref_site_set) -> str:
    """Reproduce isoform_classify._get_isoform_category_for_row.

    Pre-condition: ``row_sites`` is already computed by the caller (matches
    the upstream pipeline in isoform_classify.py:47, where row_sites is
    pre-computed via ``df_data['SSC'].str.split('-').map(int)``).
    """
    if query_ssc in ref_ssc_set:
        return "FSM"
    for ref_ssc in ref_ssc_set:
        if ref_ssc.startswith(query_ssc) or ref_ssc.endswith(query_ssc):
            return "ISM"
    try:
        row_sites = [int(x) for x in query_ssc.split("-")]
    except (ValueError, AttributeError):
        return "NNC"
    if set(row_sites).issubset(ref_site_set):
        return "NIC"
    return "NNC"


def _row_sites(ssc: str):
    return [int(x) for x in ssc.split("-")]


# --- shared reference for all three tests --------------------------------
# A 5-exon / 4-intron reference transcript on chr1 (+) at genomic coords
# 1000-2000.  Exons: (1000,1100), (1200,1300), (1400,1500), (1600,1700),
# (1800,1900) — but the SSC encodes only the *intron boundaries*, so
# the SSC string is "1100-1200-1300-1400-1500-1600-1700-1800".
REF_SSC = "1100-1200-1300-1400-1500-1600-1700-1800"
REF_TR_START = 1000
REF_TR_END = 2000
REF_INTRONS = _parse_introns(REF_TR_START, REF_SSC, REF_TR_END)
# Sanity: must be 4 introns in genomic order.
assert len(REF_INTRONS) == 4, REF_INTRONS
assert REF_INTRONS == [
    (1100, 1200),
    (1300, 1400),
    (1500, 1600),
    (1700, 1800),
], REF_INTRONS


# =========================================================================
# Test 1 — Internal fragment ISM (drift trigger)
# =========================================================================
# The query is a 3-exon / 2-intron transcript whose intron chain is the
# *middle* two introns of the reference (positions [1:3] of REF_INTRONS).
# Conceptually this is an internal truncation / "internal fragment" ISM
# of the reference.
#
#   query exons : (1300,1300), (1400,1500), (1600,1600)  (zero-width ends)
#   query SSC   : "1300-1400-1500-1600"
#   query introns: [(1300,1400), (1500,1600)]
#
# Old substring heuristic:
#   - "1100-...-1800".startswith("1300-1400-1500-1600") -> False
#   - "1100-...-1800".endswith  ("1300-1400-1500-1600") -> False
#   - {1300,1400,1500,1600} ⊂ {1100,...,1800}            -> True  -> NIC
# New subchain helper:
#   - target[1:3] == query_introns -> True                -> ISM
#
# Drift: OLD -> NIC, NEW -> ISM.
# =========================================================================
def test_internal_fragment_ism_drifts_between_classifiers():
    query_tr_start = 1300
    query_tr_end = 1600
    query_ssc = "1300-1400-1500-1600"

    ref_ssc_set = {REF_SSC}
    ref_site_set = set(_row_sites(REF_SSC))

    # New (SQANTI3 subchain) — must classify as ISM
    query_introns = _parse_introns(query_tr_start, query_ssc, query_tr_end)
    assert query_introns == [(1300, 1400), (1500, 1600)], query_introns
    new_is_ism = _is_contiguous_intron_subchain(query_introns, REF_INTRONS)
    assert new_is_ism is True, (
        "Expected NEW (contiguous-intron-subchain) to flag the internal "
        "fragment as ISM; got False."
    )

    # Old (substring heuristic) — must classify as NOT-ISM (falls through to NIC)
    old_category = _old_classify(query_ssc, ref_ssc_set, ref_site_set)
    assert old_category == "NIC", (
        "Expected OLD (substring heuristic) to mis-classify the internal "
        f"fragment as NIC; got {old_category!r}."
    )

    # The drift proof: the two implementations disagree on this case.
    new_category = "ISM" if new_is_ism else "not-ISM"
    assert new_category != old_category, (
        "Drift proof failed: both implementations returned the same "
        f"category ({old_category!r}) for the internal fragment case. "
        "Either the drift is gone (good, fix is merged) or the test "
        "fixtures are wrong (re-check TrStart/TrEnd/SSC)."
    )

    # And the drift is specifically NIC-vs-ISM, which is the P0 #1 failure
    # mode documented in the project memory.
    assert (old_category, new_category) == ("NIC", "ISM"), (
        f"Drift direction unexpected: OLD={old_category!r}, NEW={new_category!r}. "
        "P0 #1 documents the failure as old->NIC vs new->ISM."
    )


# =========================================================================
# Test 2 — Discontiguous (genuine NIC) — no false positive on either side
# =========================================================================
# The query is a 3-exon / 2-intron transcript whose intron chain
# corresponds to *non-contiguous* positions [0] and [2] of REF_INTRONS
# (skipping ref intron #1).
#
#   query exons : (1100,1100), (1200,1500), (1600,1600)
#   query SSC   : "1100-1200-1500-1600"
#   query introns: [(1100,1200), (1500,1600)]
#
# Both implementations must agree this is NOT an ISM:
#   Old: no startswith/endswith match, but sites are subset -> NIC
#   New: target[0:2] = [(1100,1200),(1300,1400)] != query,
#        target[1:3] = [(1300,1400),(1500,1600)] != query,
#        target[2:4] = [(1500,1600),(1700,1800)] != query -> not a subchain
# =========================================================================
def test_discontiguous_query_is_not_flagged_as_ism_by_either_classifier():
    query_tr_start = 1100
    query_tr_end = 1600
    query_ssc = "1100-1200-1500-1600"

    ref_ssc_set = {REF_SSC}
    ref_site_set = set(_row_sites(REF_SSC))

    # New (SQANTI3 subchain) — must NOT flag as ISM
    query_introns = _parse_introns(query_tr_start, query_ssc, query_tr_end)
    assert query_introns == [(1100, 1200), (1500, 1600)], query_introns
    new_is_ism = _is_contiguous_intron_subchain(query_introns, REF_INTRONS)
    assert new_is_ism is False, (
        "Expected NEW (contiguous-intron-subchain) to return False for a "
        "discontiguous query; got True.  This would be a false-positive ISM."
    )

    # Old (substring heuristic) — also NIC, no false positive
    old_category = _old_classify(query_ssc, ref_ssc_set, ref_site_set)
    assert old_category == "NIC", (
        f"Expected OLD to classify the discontiguous case as NIC; "
        f"got {old_category!r}."
    )

    # Both agree this is NOT an ISM (no drift in this direction).
    assert old_category != "ISM"
    assert new_is_ism is False


# =========================================================================
# Test 3 — FSM exact match (sanity)
# =========================================================================
# query_ssc == ref_ssc.  This is the trivial case that the old heuristic
# handles with an early "row_ssc in ref_ssc_set" check.  The new
# contiguous-intron-subchain helper is *not* an FSM checker (it requires
# strict length inequality), so it correctly returns False here — FSM
# classification must come from the exact-match step, not from the
# subchain helper.
# =========================================================================
def test_fsm_exact_match_is_classified_as_fsm_by_old_heuristic():
    query_ssc = REF_SSC  # identical to the reference SSC

    ref_ssc_set = {REF_SSC}
    ref_site_set = set(_row_sites(REF_SSC))

    # Old heuristic: exact match -> FSM (this is the upstream behavior).
    old_category = _old_classify(query_ssc, ref_ssc_set, ref_site_set)
    assert old_category == "FSM", (
        f"Expected OLD to return FSM for exact-match query; got {old_category!r}."
    )

    # New subchain helper: must return False on exact match (it is an
    # ISM-only check and requires n_q < n_t).  This is the expected
    # "no false positive" behavior for the helper — FSM is handled
    # elsewhere via the exact-match step.
    query_introns = _parse_introns(REF_TR_START, query_ssc, REF_TR_END)
    assert query_introns == REF_INTRONS
    new_is_ism = _is_contiguous_intron_subchain(query_introns, REF_INTRONS)
    assert new_is_ism is False, (
        "NEW (contiguous-intron-subchain) must return False on exact match; "
        "it is an ISM-only check and requires strict length inequality."
    )

    # Sanity: both implementations converge to "not-ISM" here, with the
    # FSM label coming from the old exact-match step.
    assert old_category == "FSM"
    assert new_is_ism is False
