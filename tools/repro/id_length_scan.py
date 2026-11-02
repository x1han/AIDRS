#!/usr/bin/env python
"""Scan an AIDRS delivery for over-long / duplicated identifiers.

AIDRS issue #3: the transcript id becomes the reference name (RNAME) of
the model FASTA and of the per-sample BAM. SAM v1.6 limits QNAME to 254
bytes, and minimap2 stores the name length in a ``uint8_t`` inside its
``.mmi`` index -- a longer name is silently truncated, and a name of
exactly 256 bytes produces a NULL name pointer and a SIGSEGV.

This script reports, for every id column of the annotated model:

* the maximum length,
* the number of ids over the limit, broken down by id prefix,
* the number of duplicated ids,
* which source produced the over-long ids.

Usage::

    python tools/repro/id_length_scan.py \
        --delivery /datf/hanxi/project/DRS/result_sample_col_2_minimap_aidrs_v3/AIDRS
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.aidrs_runtime.transcript_consolidate import MAX_GENE_ID_LEN  # noqa: E402
from src.aidrs_runtime.id_validator import MAX_TRANSCRIPT_ID_LEN  # noqa: E402


def scan_ids(df: pd.DataFrame, column: str, limit: int,
              check_duplicates: bool = True) -> None:
    if column not in df.columns:
        return
    ids = df[column].astype(str)
    lengths = ids.str.len()
    over = ids[lengths > limit]
    print(f"\n{column}: n={len(ids)}  max_len={lengths.max()}  "
          f"n>{limit}={len(over)}")
    if len(over):
        prefixes = collections.Counter(i.split("_")[0] for i in over)
        print(f"   prefixes of the over-length ids: {dict(prefixes)}")
        longest = max(over, key=len)
        print(f"   longest ({len(longest)}): {longest[:100]}")
    if check_duplicates:
        dup = ids[ids.duplicated(keep=False)]
        print(f"   duplicated ids: {len(dup)}")
    else:
        # Gene ids and gene names repeat by design (many transcripts per
        # gene); only the transcript id must be unique.
        print(f"   distinct: {ids.nunique()} (repetition is expected here)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delivery", required=True)
    ap.add_argument("--annotation", default=None,
                    help="path to the annotated model TSV (default: "
                         "<delivery>/temp/aidrs.transcript.annotated_df."
                         "before_quantification.tsv)")
    args = ap.parse_args()

    anno = args.annotation or os.path.join(
        args.delivery, "temp",
        "aidrs.transcript.annotated_df.before_quantification.tsv")
    if not os.path.exists(anno):
        print(f"annotation not found: {anno}", file=sys.stderr)
        return 2

    df = pd.read_csv(anno, sep="\t", low_memory=False)
    print(f"=== {anno} ({len(df)} rows) ===")
    scan_ids(df, "TrID", MAX_TRANSCRIPT_ID_LEN)
    scan_ids(df, "GeneID", MAX_GENE_ID_LEN, check_duplicates=False)
    scan_ids(df, "GeneName", MAX_GENE_ID_LEN, check_duplicates=False)

    # Where does the length come from? A fused gene id joins every reference
    # gene id of the cluster, so the blow-up is in the GeneID and the TrID
    # merely inherits it.
    gene_ids = df["GeneID"].astype(str)
    long_genes = gene_ids[gene_ids.str.len() > MAX_GENE_ID_LEN].unique()
    if len(long_genes):
        print(f"\n=== source of the over-long ids ===")
        for g in sorted(long_genes, key=len):
            segments = g.split("_")
            print(f"  GeneID len={len(g):4d}  segments={len(segments):3d}  "
                  f"all ENSG: {all(s.startswith('ENSG') for s in segments[1:])}")
        print("  => the gene id is the fused reference-gene list; shortening "
              "only the transcript id would leave the GTF gene_id over-long.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
