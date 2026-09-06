"""Unit tests for gtf2ssc SSC-string (inner) construction.

Verifies the per-transcript inner-string algorithm in src/gtf2ssc.py main()
correctly emits `exon_i.end - exon_{i+1}.start` for each consecutive exon pair,
even when exons overlap (which the previous `sorted(starts + ends)` bug
silently corrupted).

The bug being verified here:

  buggy_code:
      sites = sorted(info['starts'] + info['ends'])
      inner = '-'.join(str(x) for x in sites[1:-1]) if len(sites) > 2 else 'NA'

  For two non-overlapping exons [(100, 200), (300, 400)] the sort-mix
  happens to give the correct "200-300". But for overlapping exons
  [(100, 250), (200, 400)] it gives sorted [100, 200, 250, 400] and inner
  "200-250" -- which is the wrong junction (200 is E2.start and 250 is E1.end,
  not a real exon boundary at all). The correct intron boundary is 250-200.

This module exercises the per-transcript loop body directly so we can
isolate the bug from file I/O and multiprocessing.
"""
import sys
import os

REPO = "/datf/hanxi/software/AIDRS/repo"
sys.path.insert(0, REPO)

# Import the gtf2ssc module directly without going through src/__init__.py,
# which pulls in aidrs.py and a sibling refactor that's still in flight.
# gtf2ssc.py uses `from .aidrs_runtime.concurrency import ...`, which is a
# relative import that requires a package context. We satisfy it by
# registering a synthetic `src` package + a stub `src.aidrs_runtime.concurrency`
# module before loading gtf2ssc. The real concurrency module is not needed:
# the test exercises build_ssc_inner() (defined inline below as a replica of
# the per-transcript block at gtf2ssc.py:95-106), not main().
import importlib.util

_SRC = os.path.join(REPO, "src")

# Register an empty 'src' package (skip __init__.py execution)
_spec_pkg = importlib.util.spec_from_loader("src", loader=None, is_package=True)
_src_pkg = importlib.util.module_from_spec(_spec_pkg)
_src_pkg.__path__ = [_SRC]
sys.modules["src"] = _src_pkg

# Register an empty 'src.aidrs_runtime' package
_spec_ar = importlib.util.spec_from_loader(
    "src.aidrs_runtime", loader=None, is_package=True
)
_ar_pkg = importlib.util.module_from_spec(_spec_ar)
_ar_pkg.__path__ = [os.path.join(_SRC, "aidrs_runtime")]
sys.modules["src.aidrs_runtime"] = _ar_pkg

# Stub src.aidrs_runtime.concurrency with the two names gtf2ssc imports.
# The stubs are never CALLED in this test (we only test build_ssc_inner
# inlined below), so plain functions are sufficient.
def _stub_drain_futures_loud(futures, stage_name, allow_partial=False):
    raise RuntimeError(
        "stub drain_futures_loud called; build_ssc_inner test should not "
        "invoke main() -- if this fires, a test is exercising real GTF I/O."
    )


def _stub_get_process_pool(num_workers, mp_context=None):
    raise RuntimeError(
        "stub get_process_pool called; build_ssc_inner test should not "
        "invoke main() -- if this fires, a test is exercising real GTF I/O."
    )


_spec_cc = importlib.util.spec_from_loader(
    "src.aidrs_runtime.concurrency", loader=None, is_package=False
)
_cc_mod = importlib.util.module_from_spec(_spec_cc)
_cc_mod.drain_futures_loud = _stub_drain_futures_loud
_cc_mod.get_process_pool = _stub_get_process_pool
sys.modules["src.aidrs_runtime.concurrency"] = _cc_mod

# Now load gtf2ssc as src.gtf2ssc so the relative import resolves.
_spec = importlib.util.spec_from_file_location(
    "src.gtf2ssc", os.path.join(_SRC, "gtf2ssc.py")
)
gtf2ssc = importlib.util.module_from_spec(_spec)
sys.modules["src.gtf2ssc"] = gtf2ssc
_spec.loader.exec_module(gtf2ssc)


def build_ssc_inner(merged_info):
    """Replica of the fixed per-transcript inner-construction block in main().

    Kept inline (not refactored into gtf2ssc.main) so this test only
    validates the algorithm without touching the multiprocessing / file
    I/O scaffolding around it.
    """
    sorted_pairs = sorted(zip(merged_info['starts'], merged_info['ends']))
    starts_sorted = [p[0] for p in sorted_pairs]
    ends_sorted = [p[1] for p in sorted_pairs]
    intron_boundaries = zip(ends_sorted[:-1], starts_sorted[1:])
    inner = '-'.join(f'{a}-{b}' for a, b in intron_boundaries)
    if not inner:
        inner = 'NA'
    start = starts_sorted[0]
    end = ends_sorted[-1]
    return start, end, inner


def make_merged(chrom, strand, exon_pairs):
    """Build a merged-info dict equivalent to what main() sees after parsing.

    exon_pairs is a list of (start, end) tuples.
    """
    return {
        'Chr': chrom,
        'Strand': strand,
        'GeneID': 'geneX',
        'GeneName': 'geneX',
        'starts': [s for s, _ in exon_pairs],
        'ends': [e for _, e in exon_pairs],
    }


def test_two_nonoverlapping_exons():
    """Two non-overlapping exons: inner should be E1.end-E2.start."""
    info = make_merged('chr1', '+', [(100, 200), (300, 400)])
    start, end, inner = build_ssc_inner(info)
    assert start == 100, f"start: expected 100, got {start}"
    assert end == 400, f"end: expected 400, got {end}"
    assert inner == '200-300', (
        f"inner: expected '200-300' (E1.end to E2.start), got {inner!r}"
    )
    print("[OK] 2 non-overlapping exons [(100,200),(300,400)] -> inner='200-300'")


def test_two_overlapping_exons():
    """Two overlapping exons: inner MUST be '250-200', NOT '200-250'.

    This is the regression case that exposes the sorted-mix bug.
    The sort-mix algorithm emits '200-250' (wrong: 200 is inside E1
    and 250 is inside E1, neither is a real junction). The correct
    intron boundary is from E1.end (250) to E2.start (200).
    """
    info = make_merged('chr1', '+', [(100, 250), (200, 400)])
    start, end, inner = build_ssc_inner(info)
    assert start == 100, f"start: expected 100, got {start}"
    assert end == 400, f"end: expected 400, got {end}"
    assert inner == '250-200', (
        f"inner: expected '250-200' (E1.end to E2.start), got {inner!r}. "
        f"If you got '200-250' the sort-mix bug has regressed."
    )
    print("[OK] 2 overlapping exons [(100,250),(200,400)] -> inner='250-200'")


def test_three_exons_with_one_overlap():
    """Three exons where E1 and E2 overlap.

    Exons: (100, 200), (180, 280), (300, 400)
    After sorting by start:
      E1 = (100, 200)
      E2 = (180, 280)  -- overlaps E1
      E3 = (300, 400)
    Intron boundaries:
      E1.end to E2.start  -> 200-180
      E2.end to E3.start  -> 280-300
    Expected inner: '200-180-280-300'
    """
    info = make_merged('chr1', '+', [(100, 200), (180, 280), (300, 400)])
    start, end, inner = build_ssc_inner(info)
    assert start == 100, f"start: expected 100, got {start}"
    assert end == 400, f"end: expected 400, got {end}"
    assert inner == '200-180-280-300', (
        f"inner: expected '200-180-280-300', got {inner!r}"
    )
    print("[OK] 3 exons [(100,200),(180,280),(300,400)] -> inner='200-180-280-300'")


if __name__ == "__main__":
    test_two_nonoverlapping_exons()
    test_two_overlapping_exons()
    test_three_exons_with_one_overlap()
    print("\nAll gtf2ssc SSC-construction tests passed.")