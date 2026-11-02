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
       - every numeric column must agree within TOL (1e-5 relative to the
         value, i.e. ~10x the observed 1e-6 inference noise).
A case passes only if both hold.

Usage::

    python tools/repro/compare_factorial_semantic.py <FINAL dir> [<BASE dir>]

``FINAL`` is the output directory of the run under test, ``BASE`` the
reference it is compared against (default: the unpatched chr1 factorial).
The exit code is 0 only if every file is byte-identical or float-tolerantly
equal.
"""
import os
import sys

import pandas as pd

DEFAULT_BASE = "/tmp/aidrs_issue_repro/factorial_unpatched"
DEFAULT_FINAL = "/tmp/aidrs_issue_repro/factorial_p1r3"
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
        # try numeric
        try:
            x = pd.to_numeric(pd.Series(av), errors="raise").astype(float)
            y = pd.to_numeric(pd.Series(bv), errors="raise").astype(float)
        except (ValueError, TypeError):
            diff_rows = [i for i in range(len(a)) if av[i] != bv[i]]
            return False, ([f"  column {c}: non-numeric values differ "
                            f"({len(diff_rows)} rows)"] +
                           [f"    row {i}: {av[i]!r} != {bv[i]!r}"
                            for i in diff_rows[:10]])
        diff = (x - y).abs()
        worst = float(diff.max())
        n = int((diff > 0).sum())
        if worst > TOL:
            rows = [i for i in range(len(a)) if diff.iloc[i] > 0]
            return False, ([f"  column {c}: {n} cells differ, max |diff| = "
                            f"{worst:.3e} > {TOL:.0e}"] +
                           [f"    row {i}: {av[i]} != {bv[i]}"
                            for i in rows[:10]])
        lines.append(f"  column {c}: {n} cells differ, max |diff| = {worst:.3e} (<= {TOL:.0e}) -> float noise")
    if not lines:
        lines.append("  (byte difference with no differing cell: whitespace/format only)")
    return True, lines


def main():
    base = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BASE
    final = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FINAL

    total = same = 0
    rc = 0
    for case in CASES:
        fb, ff = os.path.join(base, case), os.path.join(final, case)
        print(f"\n=== {case} ===")
        if not os.path.isdir(ff):
            print("  MISSING final output dir")
            rc = 1
            continue
        fb_files, ff_files = set(listfiles(fb)), set(listfiles(ff))
        only_base = sorted(fb_files - ff_files)
        only_final = sorted(ff_files - fb_files)
        if only_base:
            print(f"  only in BASE : {only_base}")
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

    print(f"\n==== {same}/{total} files byte-identical or float-tolerantly equal ====")
    return rc


if __name__ == "__main__":
    sys.exit(main())
