#!/usr/bin/env python
"""Gate E.5 comparison, float-tolerant.

Byte-identity is the right gate for every file EXCEPT the assessment
tables of the TranslationAI cases: TranslationAI's float32 inference is not
bit-reproducible across runs (identical inputs -> a handful of score cells
differ by 1e-6), so a byte comparison of those two files has a false
positive mode that has nothing to do with the code under test.

This comparator therefore does:
  1. byte comparison for every file;
  2. for a file that differs, a column-wise comparison:
       - every non-numeric column must be IDENTICAL (ids, coordinates,
         TIS/TTS locations, NMD calls, ...);
       - every numeric column must agree within TOL (1e-5 ABSOLUTE, i.e.
         ~10x the observed 1e-6 inference noise).  The scores under test
         (TIS_score / TTS_score) are bounded in [-1, 1], so an absolute and a
         relative tolerance coincide there; a relative tolerance would be
         required if the tool were ever pointed at a count matrix.
A case passes only if both hold.

The gate also fails loudly on the structural failure modes: a case directory
that is missing on either side, and a file that is present in BASE but absent
from FINAL (a lost output).  Files that exist only in FINAL are reported but
do not fail the gate -- this branch legitimately ADDS output files.

Usage::

    python tools/repro/compare_factorial_semantic.py <FINAL dir> [<BASE dir>]

``FINAL`` is the output directory of the run under test (required), ``BASE``
the reference it is compared against (optional; defaults to the unpatched
chr1 factorial of the triage run, which has to exist on disk).
The exit code is 0 only if every case was compared and every file is
byte-identical or float-tolerantly equal.
"""
import os
import sys

import numpy as np
import pandas as pd

# The unpatched chr1 factorial used as the reference for this gate.  It is a
# scratch artefact of the triage run, not part of the repository, so the
# caller has to point at it explicitly if the default is not present.
DEFAULT_BASE = "/tmp/aidrs_issue_repro/factorial_unpatched"
CASES = ["case1_transAI_polyA", "case2_transAI_noPolyA",
         "case3_noTransAI_polyA", "case4_noTransAI_noPolyA"]
SKIP = {"aidrs.log"}
TOL = 1e-5


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def listfiles(d):
    out = []
    for root, _dirs, files in os.walk(d):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), d)
            if rel in SKIP or rel.startswith("metrics" + os.sep) or rel == "metrics":
                continue
            out.append(rel)
    return sorted(out)


def semantic_diff(pb, pf):
    """Column-wise comparison of two TSVs. Returns (ok, report_lines)."""
    a = pd.read_csv(pb, sep="\t", dtype=str).fillna("")
    b = pd.read_csv(pf, sep="\t", dtype=str).fillna("")
    lines = []
    if list(a.columns) != list(b.columns):
        return False, ["  column sets differ"]
    if len(a) != len(b):
        return False, [f"  row counts differ: {len(a)} vs {len(b)}"]
    for c in a.columns:
        av, bv = a[c].values, b[c].values
        if (av == bv).all():
            continue
        # numeric wherever BOTH sides parse; every other differing cell is a
        # hard difference.  Coercing per cell (instead of raising on the whole
        # column) keeps one empty cell from turning a 1e-6 float-noise column
        # into a "non-numeric values differ" failure.
        x = pd.to_numeric(pd.Series(av), errors="coerce").to_numpy(float)
        y = pd.to_numeric(pd.Series(bv), errors="coerce").to_numpy(float)
        both = ~np.isnan(x) & ~np.isnan(y)
        hard = (~both) & (av != bv)
        if hard.any():
            idx = [int(i) for i in np.flatnonzero(hard)]
            return False, ([f"  column {c}: non-numeric values differ "
                            f"({len(idx)} rows)"] +
                           [f"    row {i}: {av[i]!r} != {bv[i]!r}"
                            for i in idx[:10]])
        diff = np.abs(x - y)
        worst = float(diff[both].max()) if both.any() else 0.0
        n = int(((diff > 0) & both).sum())
        if worst > TOL:
            rows = [int(i) for i in np.flatnonzero(diff > 0)]
            return False, ([f"  column {c}: {n} cells differ, max |diff| = "
                            f"{worst:.3e} > {TOL:.0e}"] +
                           [f"    row {i}: {av[i]} != {bv[i]}"
                            for i in rows[:10]])
        lines.append(f"  column {c}: {n} cells differ, max |diff| = {worst:.3e} (<= {TOL:.0e}) -> float noise")
    if not lines:
        lines.append("  (byte difference with no differing cell: whitespace/format only)")
    return True, lines


def main():
    if len(sys.argv) < 2:
        print("usage: compare_factorial_semantic.py <FINAL dir> [<BASE dir>]",
              file=sys.stderr)
        print("FATAL: the FINAL directory is required", file=sys.stderr)
        return 2
    final = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BASE
    if not os.path.isdir(final):
        print(f"FATAL: final directory not found: {final}", file=sys.stderr)
        return 2
    if not os.path.isdir(base):
        print(f"FATAL: base directory not found: {base}\n"
              f"       pass it as the second argument", file=sys.stderr)
        return 2

    total = same = 0
    compared = 0
    rc = 0
    for case in CASES:
        fb, ff = os.path.join(base, case), os.path.join(final, case)
        print(f"\n=== {case} ===")
        if not os.path.isdir(ff):
            print("  MISSING final output dir")
            rc = 1
            continue
        if not os.path.isdir(fb):
            # a missing BASE case would otherwise be compared against an empty
            # file list and the case would be skipped silently
            print("  MISSING base output dir")
            rc = 1
            continue
        compared += 1
        fb_files, ff_files = set(listfiles(fb)), set(listfiles(ff))
        only_base = sorted(fb_files - ff_files)
        only_final = sorted(ff_files - fb_files)
        if only_base:
            # an output that BASE has and FINAL does not is a LOST output
            print(f"  only in BASE : {only_base}")
            rc = 1
        if only_final:
            print(f"  only in FINAL: {only_final}   (the new contract sidecars)")
        for rel in sorted(fb_files & ff_files):
            total += 1
            pb, pf = os.path.join(fb, rel), os.path.join(ff, rel)
            if sha(pb) == sha(pf):
                same += 1
                print(f"  IDENTICAL  {rel}")
                continue
            if rel.endswith(".tsv"):
                ok, lines = semantic_diff(pb, pf)
                if ok:
                    same += 1
                    print(f"  SEMANTICALLY EQUAL (float-tolerant)  {rel}")
                    for l in lines:
                        print(l)
                    continue
                print(f"  DIFFERS    {rel}")
                for l in lines:
                    print(l)
            else:
                print(f"  DIFFERS    {rel}")
            rc = 1

    print(f"\n==== {same}/{total} files byte-identical or float-tolerantly equal "
          f"({compared}/{len(CASES)} cases compared) ====")
    if compared != len(CASES):
        print(f"FATAL: {len(CASES) - compared} case(s) were not compared",
              file=sys.stderr)
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
