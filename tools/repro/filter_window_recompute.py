#!/usr/bin/env python
"""Recompute the Stage 3 expression filter under both windows.

AIDRS issue #4: the ``min_expressed_samples`` filter scored
``iloc[:, 3:]`` -- i.e. it ignored the first three libraries -- and clamped
the threshold to ``n_libraries - 3``. This script recomputes the filter
under the old (buggy) window and the new (all-libraries) window on a real
delivery, so the effect of the fix can be measured without re-running the
whole pipeline.

Usage::

    python tools/repro/filter_window_recompute.py \
        --delivery /datf/hanxi/project/DRS/result_sample_col_2_minimap_aidrs_v3/AIDRS \
        --min-expressed-samples 3

It reads the pre-filter annotation (``temp/aidrs.transcript.annotated_df
.before_quantification.tsv``) and the per-sample SSC files, re-quantifies
every library, and reports how many transcripts the fix rescues.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delivery", required=True,
                    help="AIDRS output directory of a multi-library run")
    ap.add_argument("--min-expressed-samples", type=int, default=3,
                    help="value of --min_expressed_samples used for the run")
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="optional path to save the recomputed quant_results (parquet)")
    args = ap.parse_args()

    from src.isoform_quantification import IsoformQuantifier

    delivery = args.delivery
    sep = "\t"
    anno_path = os.path.join(delivery, "temp",
                             "aidrs.transcript.annotated_df.before_quantification.tsv")
    counts_path = os.path.join(delivery, "aidrs_transcript_counts.tsv")

    anno = pd.read_csv(anno_path, sep=sep, low_memory=False)
    libs = [c for c in pd.read_csv(counts_path, sep=sep, nrows=1).columns
            if c not in ("TrID", "GeneID", "GeneName")]
    print(f"annotation={len(anno)} transcripts, {len(libs)} libraries", flush=True)

    q = IsoformQuantifier(num_processes=args.num_processes,
                          min_expressed_samples=args.min_expressed_samples)
    frames = []
    t0 = time.time()
    for s in libs:
        frames.append(q.quantify_sample(s, anno, delivery))
        print(f"  {s}: {len(frames[-1])} rows  ({time.time() - t0:.0f}s)", flush=True)
    qr = pd.concat(frames, ignore_index=True)
    if args.out:
        qr.to_parquet(args.out)
        print(f"saved quant_results -> {args.out}", flush=True)

    raw = qr.pivot_table(index=["TrID", "GeneID", "GeneName"], columns="sample",
                         values="count", aggfunc="sum", fill_value=0).astype(int)
    raw = raw.reindex(columns=libs, fill_value=0)

    n, m = len(libs), args.min_expressed_samples
    keep_buggy = raw.iloc[:, 3:].gt(0).sum(axis=1) >= min(n - 3, m)
    keep_fixed = raw.gt(0).sum(axis=1) >= min(n, m)
    rescued = keep_fixed & ~keep_buggy
    lost = keep_buggy & ~keep_fixed

    print(f"\n=== filter window comparison ({n} libs, m={m}) ===")
    print(f"annotated transcripts with >=1 read : {int((raw.sum(axis=1) > 0).sum())}")
    print(f"kept by BUGGY window (libs[3:])     : {int(keep_buggy.sum())}")
    print(f"kept by FIXED window (all libs)     : {int(keep_fixed.sum())}")
    print(f"RESCUED by the fix                  : {int(rescued.sum())}")
    print(f"LOST by the fix                     : {int(lost.sum())}")

    tx_ids = raw.index.get_level_values("TrID")
    resc_ids = set(tx_ids[rescued.values])
    print("\nrescued-transcript reads per library:")
    for s in libs:
        sel = raw.index.get_level_values("TrID").isin(resc_ids)
        reads = int(raw[s][sel].sum())
        tot = int(raw[s].sum())
        print(f"  {s:34s} rescued_reads={reads:8d} "
              f"({reads / max(tot, 1) * 100:.4f}% of lib)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
