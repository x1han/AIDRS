#!/usr/bin/env python3
"""Mouse S1 chr1 2-case factorial comparison.

Specifically designed for Mouse S1 chr1 factorial:
  Case A: TranslationAI ON (default)
  Case B: TranslationAI OFF (--no_translationai)

Computes:
  - 16-col scientific SHA for each case
  - Predict_NMD distribution health check (translationai expected behavior)
  - Row-level gained/lost diff (full row data, not just samples)
"""
import argparse
import os
import sys
import hashlib

# Allow standalone invocation: tools/*.py scripts must be runnable as
# `python tools/compare_mouse_s1_factorial.py ...` without the caller
# setting PYTHONPATH.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import pandas as pd

from src.aidrs_runtime.column_registry import SCIENTIFIC_COLS


def load_isoforms(tsv_path):
    """Load assessment.tsv and return (model_keys set, full df, sha256 of intersection cols).

    Uses whatever subset of SCIENTIFIC_COLS is present in the TSV (Mouse S1 omits
    polyA_valid_reads; the SHA is computed on the actually-present subset so the
    tool works on either data shape). Emits a warning for missing canonical cols
    so the user can see when SHA-comparable subsets differ.
    """
    if not os.path.exists(tsv_path):
        return None, None, None, f"file not found: {tsv_path}"
    df = pd.read_csv(tsv_path, sep="\t")
    if df.empty:
        return set(), df, hashlib.sha256(b"").hexdigest(), None
    df["model_key"] = (
        df["Chr"].astype(str)
        + ":"
        + df["TrStart"].astype(str)
        + "_"
        + df["SSC"].astype(str)
        + "_"
        + df["TrEnd"].astype(str)
        + "("
        + df["Strand"].astype(str)
        + ")"
    )
    cols_present = [c for c in SCIENTIFIC_COLS if c in df.columns]
    cols_missing = sorted(set(SCIENTIFIC_COLS) - set(cols_present))
    if cols_missing:
        # Not fatal — just SHA on the actually-present subset.
        # Caller can decide whether the subset is informative enough.
        pass
    sha = hashlib.sha256(
        df[cols_present].to_csv(sep="\t", index=False).encode("utf-8")
    ).hexdigest()
    return set(df["model_key"]), df, sha, cols_missing


def predict_nmd_distribution(df):
    """Return dict of {Normal: N, NMD: N, no_orf: N, ...} for Predict_NMD column."""
    if "Predict_NMD" not in df.columns:
        return {}
    counts = df["Predict_NMD"].value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir",
                        default="/datf/hanxi/test/AIDRS/benchmark_mouse_s1_chr1",
                        help="Mouse S1 factorial benchmark directory")
    parser.add_argument("--full-diff-limit", type=int, default=20,
                        help="Max Gained/Lost model_keys to print (0 = unlimited)")
    parser.add_argument("--write-diff", action="store_true",
                        help="Write Gained/Lost TSVs to <bench>/diffs/")
    args = parser.parse_args()

    bench_dir = args.bench_dir
    diffs_dir = os.path.join(bench_dir, "diffs")
    case_a_tsv = os.path.join(bench_dir, "caseA_transAI_on/aidrs.transcript.assessment.tsv")
    case_b_tsv = os.path.join(bench_dir, "caseB_transAI_off/aidrs.transcript.assessment.tsv")

    print("=" * 80)
    print(" Mouse S1 chr1 2-Case Factorial Comparison (GRCm39)")
    print("=" * 80)

    cases = {
        "Case A [+TransAI]": case_a_tsv,
        "Case B [-TransAI]": case_b_tsv,
    }
    loaded = {}
    for name, path in cases.items():
        keys, df, sha, missing = load_isoforms(path)
        if keys is None:
            print(f"[WARN] {name}: {missing}")  # err message
            continue
        nmd_dist = predict_nmd_distribution(df)
        print(f"\n[{name}]")
        print(f"  path   : {path}")
        print(f"  rows   : {len(keys):,}")
        print(f"  sha    : {sha[:16]}...  (over {len([c for c in SCIENTIFIC_COLS if c in df.columns])}/{len(SCIENTIFIC_COLS)} canonical cols)")
        if missing:
            print(f"  missing cols: {missing}")
        print(f"  Predict_NMD distribution:")
        for k, v in sorted(nmd_dist.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<20s}: {v:>6,} ({v/max(len(df),1):.1%})")
        loaded[name] = {"keys": keys, "df": df, "sha": sha, "nmd": nmd_dist}

    if len(loaded) < 2:
        print("\n[FATAL] Need both cases to compare. Aborting.")
        sys.exit(1)

    a = loaded["Case A [+TransAI]"]
    b = loaded["Case B [-TransAI]"]
    shared = a["keys"] & b["keys"]
    gained = a["keys"] - b["keys"]  # In A but not in B
    lost = b["keys"] - a["keys"]    # In B but not in A

    print("\n" + "=" * 80)
    print(" Cross-Case Comparison")
    print("=" * 80)
    print(f"  Case A SHA : {a['sha'][:16]}...")
    print(f"  Case B SHA : {b['sha'][:16]}...")
    print(f"  SHA differ : {a['sha'] != b['sha']}")
    print(f"  Shared     : {len(shared):,} ({len(shared)/max(len(a['keys']),1):.1%} of A)")
    print(f"  Gained in A: {len(gained):,}")
    print(f"  Lost in A  : {len(lost):,}")

    # Per user principle: aidrs evolves. SHA divergence expected + acceptable.
    # The signal we need: Case A shows real ORF/NMD predictions; Case B shows all no_orf.
    if a["nmd"].get("no_orf", 0) == len(a["df"]) and len(a["df"]) > 0:
        print("\n[!] WARNING: Case A all rows have Predict_NMD=no_orf.")
        print("    This is the silent-worker-death failure mode.")
        print("    TranslationAI may not have run (check Stage 2.4 logs).")
    else:
        nmd_count = sum(v for k, v in a["nmd"].items() if k in ("Normal", "NMD"))
        print(f"\n[OK] Case A: {nmd_count:,} rows with non-trivial Predict_NMD "
              f"({nmd_count/max(len(a['df']),1):.1%})")

    if b["nmd"].get("no_orf", 0) != len(b["df"]):
        print("\n[!] WARNING: Case B should have all Predict_NMD=no_orf (TransAI off).")
        print("    This may indicate the --no_translationai flag was not honored.")

    # Row-level diff
    if args.write_diff:
        os.makedirs(diffs_dir, exist_ok=True)
        for label, keyset, ref_df in [
            ("gained_in_A", gained, a["df"]),
            ("lost_in_A", lost, b["df"]),
        ]:
            if not keyset:
                continue
            sub = ref_df[ref_df["model_key"].isin(keyset)].drop(columns=["model_key"])
            path = os.path.join(diffs_dir, f"A_vs_B.{label}.tsv")
            sub.to_csv(path, sep="\t", index=False)
            print(f"  Row-level diff: {path} ({len(sub):,} rows)")

    # Sample gained/lost
    limit = args.full_diff_limit
    if gained:
        gl = sorted(gained)
        shown = gl if limit == 0 else gl[:limit]
        print(f"\nGained in Case A (showing {len(shown)}/{len(gl)}):")
        for k in shown:
            print(f"  + {k}")
        if limit and len(gl) > limit:
            print(f"  ... and {len(gl) - limit} more (--full-diff-limit 0)")
    if lost:
        ll = sorted(lost)
        shown = ll if limit == 0 else ll[:limit]
        print(f"\nLost in Case A (showing {len(shown)}/{len(ll)}):")
        for k in shown:
            print(f"  - {k}")
        if limit and len(ll) > limit:
            print(f"  ... and {len(ll) - limit} more (--full-diff-limit 0)")
    print("=" * 80)


if __name__ == "__main__":
    main()
