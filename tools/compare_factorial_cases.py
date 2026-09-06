#!/usr/bin/env python3
"""Compare 5 AIDRS runs (1 v0.3 baseline + 4 new AIDRS factorial cases) on chr1.

Outputs:
  - Per-case stats: total retained, shared with baseline, gained, lost
  - Modality degradation robustness: Case 1 vs Case 2/3/4 overlap
  - Sample gained/lost model keys for biological interpretation
  - FULL Gained/Lost lists printed (with optional --full-diff-limit cap)
  - Row-level diff written to <bench>/diffs/<case>_vs_<baseline>.diff.tsv when --write-diff

The row-level diff is the killer feature for SHA-drift debugging: instead of a
black-box signal ("SHA differs"), developers see exactly which model_key
coordinates were gained or lost.
"""
import argparse
import sys
import os
import json
import hashlib
import pandas as pd


def load_isoforms(tsv_path):
    """Read assessment.tsv and return (set of model_keys, full DataFrame, sha256 of 17-col)."""
    if not os.path.exists(tsv_path):
        return None, None, None
    df = pd.read_csv(tsv_path, sep="\t")
    if df.empty:
        return set(), df, hashlib.sha256(b"").hexdigest()
    # Build a stable physical-coordinate model key
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
    # 17-col scientific SHA (skip the model_key we just added)
    cols17 = [c for c in df.columns if c != "model_key"]
    sha = hashlib.sha256(
        df[cols17].to_csv(sep="\t", index=False).encode("utf-8")
    ).hexdigest()
    return set(df["model_key"]), df, sha


def write_diff(diffs_dir, base_name, case_name, base_df, case_df, gained, lost):
    """Write row-level diff to <diffs_dir>/<case>_vs_<base>.gained.tsv / .lost.tsv."""
    os.makedirs(diffs_dir, exist_ok=True)

    def sanitize(name):
        # Strip brackets, replace problematic filename chars with safe ASCII.
        return (name.replace("[", "").replace("]", "")
                    .replace(" ", "_").replace("+", "p").replace("-", "n")
                    .replace(",", "").replace("(", "").replace(")", ""))

    safe_base = sanitize(base_name)
    safe_case = sanitize(case_name)
    prefix = os.path.join(diffs_dir, f"{safe_case}_vs_{safe_base}")

    if gained and not case_df.empty:
        gained_df = case_df[case_df["model_key"].isin(gained)].drop(columns=["model_key"])
        gained_df.to_csv(f"{prefix}.gained.tsv", sep="\t", index=False)
    if lost and not base_df.empty:
        lost_df = base_df[base_df["model_key"].isin(lost)].drop(columns=["model_key"])
        lost_df.to_csv(f"{prefix}.lost.tsv", sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bench_dir", nargs="?",
                        default="/datf/hanxi/test/AIDRS/benchmark_chr1")
    parser.add_argument("--full-diff-limit", type=int, default=50,
                        help="Max Gained/Lost model_keys to print per case "
                             "(0 = unlimited). Full list is always written to "
                             "<bench_dir>/diffs/ when --write-diff is set.")
    parser.add_argument("--write-diff", action="store_true",
                        help="Write row-level Gained/Lost TSVs to <bench_dir>/diffs/")
    args = parser.parse_args()

    bench_dir = args.bench_dir
    diffs_dir = os.path.join(bench_dir, "diffs")

    runs = {
        "v0.3 Baseline": os.path.join(
            bench_dir, "run_v03_baseline/aidrs.transcript.assessment.tsv"
        ),
        "Case 1 [+TransAI, +polyA]": os.path.join(
            bench_dir, "case1_transAI_polyA/aidrs.transcript.assessment.tsv"
        ),
        "Case 2 [+TransAI, -polyA]": os.path.join(
            bench_dir, "case2_transAI_noPolyA/aidrs.transcript.assessment.tsv"
        ),
        "Case 3 [-TransAI, +polyA]": os.path.join(
            bench_dir, "case3_noTransAI_polyA/aidrs.transcript.assessment.tsv"
        ),
        "Case 4 [-TransAI, -polyA]": os.path.join(
            bench_dir, "case4_noTransAI_noPolyA/aidrs.transcript.assessment.tsv"
        ),
    }

    models, dfs, shas = {}, {}, {}
    for name, path in runs.items():
        res = load_isoforms(path)
        if res[0] is None:
            print(f"[WARN] {name}: file not found -> {path}")
            continue
        models[name], dfs[name], shas[name] = res
        print(f"[OK]   {name}: {len(models[name])} records, sha={shas[name][:12]}...")

    if "v0.3 Baseline" not in models or "Case 1 [+TransAI, +polyA]" not in models:
        print("\n[FATAL] v0.3 baseline OR Case 1 missing -- cannot compare.")
        print("Available runs:", list(models.keys()))
        return

    base_set = models["v0.3 Baseline"]
    base_df = dfs["v0.3 Baseline"]

    print("\n" + "=" * 80)
    print(" AIDRS 2.0 Factorial Benchmark Summary (chr1)")
    print("=" * 80)
    print(f"v0.3 Baseline Total Transcripts: {len(base_set)} (sha={shas['v0.3 Baseline'][:12]}...)\n")

    for name in [
        "Case 1 [+TransAI, +polyA]",
        "Case 2 [+TransAI, -polyA]",
        "Case 3 [-TransAI, +polyA]",
        "Case 4 [-TransAI, -polyA]",
    ]:
        if name not in models:
            continue
        cur_set = models[name]
        cur_df = dfs[name]
        shared = base_set & cur_set
        gained = cur_set - base_set
        lost = base_set - cur_set

        print(f"--- {name} ---")
        print(f"  Total Retained : {len(cur_set)} (sha={shas[name][:12]}...)")
        print(f"  Shared w/ v0.3  : {len(shared)} ({len(shared)/max(len(base_set),1):.1%})")
        print(f"  Gained (+)     : {len(gained)}")
        print(f"  Lost   (-)     : {len(lost)}")

        limit = args.full_diff_limit
        if gained:
            gained_list = sorted(gained)
            shown = gained_list if limit == 0 else gained_list[:limit]
            print(f"  Gained (showing {len(shown)}/{len(gained_list)}):")
            for k in shown:
                print(f"    + {k}")
            if limit and len(gained_list) > limit:
                print(f"    ... and {len(gained_list) - limit} more (run with --full-diff-limit 0)")
        if lost:
            lost_list = sorted(lost)
            shown = lost_list if limit == 0 else lost_list[:limit]
            print(f"  Lost (showing {len(shown)}/{len(lost_list)}):")
            for k in shown:
                print(f"    - {k}")
            if limit and len(lost_list) > limit:
                print(f"    ... and {len(lost_list) - limit} more (run with --full-diff-limit 0)")

        if args.write_diff:
            write_diff(diffs_dir, "v0.3", name, base_df, cur_df, gained, lost)
            print(f"  Row-level diff written to {diffs_dir}/<safe_case>_vs_v0p3.{{gained,lost}}.tsv")
        print()

    # 4-way robustness matrix (Case 1 as the new golden)
    c1_set = models["Case 1 [+TransAI, +polyA]"]
    print("-" * 80)
    print(" Modality Degradation Robustness (Case 1 as new golden):")
    for name in [
        "Case 2 [+TransAI, -polyA]",
        "Case 3 [-TransAI, +polyA]",
        "Case 4 [-TransAI, -polyA]",
    ]:
        if name in models:
            cur = models[name]
            overlap = len(c1_set & cur)
            print(f"  {name}: Overlap = {overlap}/{len(c1_set)} "
                  f"({overlap/max(len(c1_set),1):.1%})")
    print("=" * 80)


if __name__ == "__main__":
    main()