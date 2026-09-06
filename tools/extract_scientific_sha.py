#!/usr/bin/env python3
"""Print the 16-col scientific SHA256 of one or more AIDRS assessment TSVs.

Usage:
    python tools/extract_scientific_sha.py <tsv_path> [<tsv_path> ...]

Prints one SHA per input on its own line (same order as input). Exits 0 on
success, 2 if any file is missing columns (same convention as diff_sha.py).
"""
import hashlib
import os
import sys

# Allow standalone invocation: tools/*.py scripts must be runnable as
# `python tools/extract_scientific_sha.py ...` without the caller setting
# PYTHONPATH.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from src.aidrs_runtime.column_registry import SCIENTIFIC_COLS


def scientific_sha(tsv_path):
    import pandas as pd
    df = pd.read_csv(tsv_path, sep="\t")
    missing = [c for c in SCIENTIFIC_COLS if c not in df.columns]
    if missing:
        print(f"{tsv_path}: missing columns: {missing}", file=sys.stderr)
        return None
    text = df[SCIENTIFIC_COLS].to_csv(sep="\t", index=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    if len(sys.argv) < 2:
        print("usage: extract_scientific_sha.py <tsv_path> [<tsv_path> ...]", file=sys.stderr)
        sys.exit(1)
    exit_code = 0
    for path in sys.argv[1:]:
        sha = scientific_sha(path)
        if sha is None:
            exit_code = 2
            continue
        print(sha)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
