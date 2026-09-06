#!/usr/bin/env python
"""Verify full-genome AIDRS run produces a sensible output + Stage 2.7 flag rate.

Usage:
    python tools/verify_fullgenome_sha.py /datf/hanxi/test/AIDRS/output_fullgenome_2026-09-05/

Computes the 17-col scientific SHA on the assessment.tsv file and reports:
- Total record count (Pipeline Completion Invariant: >= 1000)
- 17-col SHA
- Stage 2.7 rt_switching_flag=True count and percentage (if column present)
"""
import sys
import hashlib
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/datf/hanxi/test/AIDRS/output_fullgenome_2026-09-05/")
_ASSESSMENT_CANDIDATES = [
    "aidrs.transcript.assessment.tsv",
    "assessment.tsv",
    "transcript.assessment.tsv",
]

# Canonical 16-col scientific SHA schema = CORE_ASSESSMENT_COLS - {TrID, GeneID, GeneName}
# Matches src/aidrs_runtime/column_registry.py CORE_ASSESSMENT_COLS exactly.
# Previously this list was a pre-F-008 v0.3 schema (predict_NMD lowercase,
# category/junction/Group SQANTI3 cols, polyA_mode, rt_switching_flag) —
# replaced 2026-09-06 to align with current 19-col CORE output.
SCIENTIFIC_COLS = [
    "Chr", "Strand", "SSC", "TrStart", "TrEnd", "frequency",
    "Puffin_TSS_15bp", "Puffin_TSS_50bp",
    "polyA_frac",
    "TIS_related_location", "TTS_related_location",
    "TIS_score", "TTS_score",
    "Predict_NMD", "truncation",
    "seq_len",
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

    # Compute 17-col SHA on columns that exist
    cols_present = [c for c in SCIENTIFIC_COLS if c in df.columns]
    cols_missing = [c for c in SCIENTIFIC_COLS if c not in df.columns]
    print(f"[INFO] Scientific columns present: {len(cols_present)}/17")
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