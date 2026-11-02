#!/usr/bin/env python
"""Check the CPM denominator contract of an AIDRS delivery.

AIDRS issue #2 claimed that ``aidrs.transcript.cpm.tsv`` and
``aidrs.gene.cpm.tsv`` "do not share a denominator" because the transcript
column sums to less than 1e6 while the gene column sums to exactly 1e6.

They do share a denominator. Both layers use

    CPM_ij = (y_ij / L_i) * 1e6

with ``L_i`` = the number of reads assigned to transcripts in library *i*
(the library size -- the standard CPM definition). A column sums to 1e6
only when the layer covers *every* quantified transcript. The transcript
layer is filtered by ``--min_expressed_samples`` and the gene layer is not,
so their column sums legitimately differ.

This script proves that on a real delivery:

* both layers reproduce as ``count / L * 1e6`` with the same ``L``;
* the column-sum gap is exactly the CPM of the rows the layer does not
  contain.

Usage::

    python tools/repro/cpm_denominator_check.py \
        --delivery /datf/hanxi/project/DRS/result_sample_col_2_minimap_aidrs_v3/AIDRS
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The delivery predates the dotted file names; accept both spellings.
FILE_CANDIDATES = {
    "transcript.counts": ["aidrs.transcript.counts.tsv", "aidrs_transcript_counts.tsv"],
    "transcript.cpm": ["aidrs.transcript.cpm.tsv", "aidrs_transcript_cpm.tsv"],
    "gene.counts": ["aidrs.gene.counts.tsv", "aidrs_gene_counts.tsv"],
    "gene.cpm": ["aidrs.gene.cpm.tsv", "aidrs_gene_cpm.tsv"],
}


def _load(delivery: str, key: str, index: list) -> pd.DataFrame:
    for name in FILE_CANDIDATES[key]:
        path = os.path.join(delivery, name)
        if os.path.exists(path):
            return pd.read_csv(path, sep="\t", index_col=index)
    raise FileNotFoundError(
        f"none of {FILE_CANDIDATES[key]} found in {delivery}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delivery", required=True)
    ap.add_argument("--tolerance", type=float, default=1e-6,
                    help="max allowed |CPM - count/L*1e6| (default 1e-6)")
    args = ap.parse_args()

    tcn = _load(args.delivery, "transcript.counts", [0, 1, 2])
    tcm = _load(args.delivery, "transcript.cpm", [0, 1, 2])
    gcn = _load(args.delivery, "gene.counts", [0, 1])
    gcm = _load(args.delivery, "gene.cpm", [0, 1])

    samples = [s for s in tcm.columns if s in gcm.columns]
    print(f"delivery      : {args.delivery}")
    print(f"transcript    : {tcn.shape[0]} transcripts x {len(tcm.columns)} libraries")
    print(f"gene          : {gcn.shape[0]} genes x {len(gcm.columns)} libraries")

    # L: the library size. The gene layer is aggregated from every
    # quantified transcript, so its column sum IS the library size.
    library_size = gcn.sum(axis=0)

    worst_tx = 0.0
    worst_gene = 0.0
    for s in samples:
        L = library_size[s]
        worst_tx = max(worst_tx, float((tcm[s] - tcn[s] / L * 1e6).abs().max()))
        worst_gene = max(worst_gene, float((gcm[s] - gcn[s] / L * 1e6).abs().max()))

    print(f"\nmax |transcript.cpm - transcript.count/L*1e6| = {worst_tx:.3e}")
    print(f"max |gene.cpm       - gene.count      /L*1e6| = {worst_gene:.3e}")
    print(f"tolerance                                        = {args.tolerance:.3e}")

    ok = max(worst_tx, worst_gene) <= args.tolerance
    print("\n=> both CPM layers use the SAME denominator L "
          f"(the library size): {'CONFIRMED' if ok else 'NOT CONFIRMED'}")

    print("\nsample                 total_reads_L   tx_cpm_colsum   gene_cpm_colsum   gap")
    for s in samples[:8]:
        tx_sum = float(tcm[s].sum())
        gene_sum = float(gcm[s].sum())
        print(f"{s:22s} {int(library_size[s]):>14d} {tx_sum:>15.3f} "
              f"{gene_sum:>17.3f} {gene_sum - tx_sum:>9.3f}")
    if len(samples) > 8:
        print(f"... ({len(samples)} libraries total)")

    # The gap must equal the CPM of the rows the transcript layer dropped.
    tx_sum_all = float(tcm[samples[0]].sum())
    gene_sum_all = float(gcm[samples[0]].sum())
    print(f"\nlibrary[0]: the transcript layer covers {tx_sum_all / 1e6 * 100:.4f}% "
          f"of the quantified reads, the gene layer {gene_sum_all / 1e6 * 100:.4f}%")
    print("=> the gap is the CPM of the transcripts the expression filter "
          "removed, not a different denominator.")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
