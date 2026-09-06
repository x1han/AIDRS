#!/usr/bin/env python
"""Verify full-genome AIDRS run produces a sensible output + Stage 2.7 flag rate.

Usage:
    python tools/verify_fullgenome_sha.py /datf/hanxi/test/AIDRS/output_fullgenome_2026-09-05/

Computes the 16-col scientific SHA on the assessment.tsv file and reports:
- Total record count (Pipeline Completion Invariant: >= 1000)
- 16-col SHA
- Stage 2.7 rt_switching_flag=True count and percentage (if column present)
"""
import sys
import hashlib
import os
from pathlib import Path

# Allow standalone invocation: tools/*.py scripts must be runnable as
# `python tools/verify_fullgenome_sha.py ...` without the caller setting
# PYTHONPATH.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import pandas as pd

from src.aidrs_runtime.column_registry import SCIENTIFIC_COLS

OUTPUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/datf/hanxi/test/AIDRS/output_fullgenome_2026-09-05/")
_ASSESSMENT_CANDIDATES = [
    "aidrs.transcript.assessment.tsv",
    "assessment.tsv",
    "transcript.assessment.tsv",
]


def main():
    assessment_tsv = None
    for cand in _ASSESSMENT_CANDIDATES:
        p = OUTPUT_DIR / cand
        if p.exists():
            assessment_tsv = p
            break
    if assessment_tsv is None:
        print(f"ERROR: no assessment file found in {OUTPUT_DIR} (tried {_ASSESSMENT_CANDIDATES})", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(assessment_tsv, sep='\t', low_memory=False)
    print(f"[OK] Loaded {len(df)} records from {assessment_tsv}")

    # Pipeline Completion Invariant
    if len(df) < 1000:
        print(f"ERROR: row count {len(df)} below 1000 invariant", file=sys.stderr)
        sys.exit(2)

    # Compute 16-col SHA on columns that exist
    cols_present = [c for c in SCIENTIFIC_COLS if c in df.columns]
    cols_missing = [c for c in SCIENTIFIC_COLS if c not in df.columns]
    print(f"[INFO] Scientific columns present: {len(cols_present)}/{len(SCIENTIFIC_COLS)}")
    if cols_missing:
        print(f"[INFO] Missing (skipped in SHA): {cols_missing}")

    sub = df[cols_present].copy()
    sub_hash = hashlib.sha256(
        pd.util.hash_pandas_object(sub, index=False).values.tobytes()
    ).hexdigest()
    print(f"[RAW_SHA] {sub_hash}")
    print(f"[SHA_PREFIX] {sub_hash[:16]}...")
    print(f"[INPUT_COLS] {cols_present}")

    # Stage 2.7 rt_switching_flag stats
    if 'rt_switching_flag' in df.columns:
        flagged = df['rt_switching_flag'].astype(str).str.lower() == 'true'
        n_flagged = int(flagged.sum())
        pct = 100.0 * n_flagged / max(1, len(df))
        print(f"[STAGE_2_7] rt_switching_flag=True: {n_flagged}/{len(df)} ({pct:.2f}%)")
        if pct > 50:
            print(f"[WARN] Stage 2.7 flag rate {pct:.2f}% is high (>50%) — verify threshold")
        elif pct < 0.5:
            print(f"[WARN] Stage 2.7 flag rate {pct:.2f}% is suspiciously low (<0.5%) — verify enable")
    else:
        print(f"[INFO] rt_switching_flag column not present (Stage 2.7 disabled?)")


if __name__ == '__main__':
    main()