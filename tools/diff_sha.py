#!/usr/bin/env python3
"""Column-subset SHA256 verifier for AIDRS byte-identity regression testing."""
import argparse
import hashlib
import shutil
import sys
import tempfile

import pandas as pd

# Canonical 16-col scientific subset = CORE_ASSESSMENT_COLS - {TrID, GeneID, GeneName}
# Hardcoded to eliminate input-ambiguity and match src/aidrs_runtime/column_registry.py
# (polyA_valid_reads removed 2026-09-06: never in any actual output)
SCIENTIFIC_COLS = [
    "Chr", "Strand", "SSC", "TrStart", "TrEnd", "frequency",
    "Puffin_TSS_15bp", "Puffin_TSS_50bp",
    "polyA_frac",
    "TIS_related_location", "TTS_related_location",
    "TIS_score", "TTS_score",
    "Predict_NMD", "truncation",
    "seq_len",
]


def hash_columns(tsv_path, columns):
    df = pd.read_csv(tsv_path, sep="\t")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return None, f"missing columns: {missing}"
    subset = df[list(columns)]
    text = subset.to_csv(sep="\t", index=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), None


def self_test(baseline_tsv, columns):
    """Run a sanity check: hash the baseline, mutate one cell, hash again,
    and assert the two SHAs differ. Both hashes are computed AFTER pandas
    roundtrip so the comparison is symmetric (the test isolates mutation
    detection, not raw-vs-roundtripped precision drift).

    Returns True on success, False on failure."""
    # Round 1: read original with pandas, write to roundtrip_a. This becomes
    # the "baseline" (H0), so any raw-vs-roundtripped precision drift is
    # already baked in before we hash.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        roundtrip_a = f.name
    df_a = pd.read_csv(baseline_tsv, sep="\t")
    df_a.to_csv(roundtrip_a, sep="\t", index=False)

    h0, err0 = hash_columns(roundtrip_a, columns)
    if err0 is not None:
        print(f"self-test: baseline hash failed: {err0}", file=sys.stderr)
        return False

    # Round 2: read roundtrip_a, mutate one cell, write to roundtrip_b.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        roundtrip_b = f.name
    df_b = pd.read_csv(roundtrip_a, sep="\t")
    if len(df_b) < 5 or "frequency" not in df_b.columns:
        print(
            "self-test: cannot mutate (need >=5 rows and 'frequency' column)",
            file=sys.stderr,
        )
        return False
    df_b["frequency"] = pd.to_numeric(df_b["frequency"], errors="coerce")
    df_b.loc[4, "frequency"] = (df_b.loc[4, "frequency"] or 0) + 1
    df_b.to_csv(roundtrip_b, sep="\t", index=False)

    h1, err1 = hash_columns(roundtrip_b, columns)
    if err1 is not None:
        print(f"self-test: mutated hash failed: {err1}", file=sys.stderr)
        return False

    return h0 != h1


def main():
    parser = argparse.ArgumentParser(
        description="Column-subset SHA256 verifier for AIDRS byte-identity regression testing."
    )
    parser.add_argument("baseline_tsv")
    parser.add_argument("new_tsv")
    parser.add_argument("legacy_cols_csv")
    parser.add_argument("--self-test", action="store_true",
                        help="Hash the baseline, mutate one cell, hash again, and "
                             "assert the SHAs differ. Exits 0 on pass, 4 on fail.")
    parser.add_argument("--baseline", choices=["legacy", "current"], default="legacy",
                        help="Baseline name to label output. legacy=H_0, current=H_1. "
                             "Column-subset SHA is identical for both; this only "
                             "affects output labeling.")
    parser.add_argument('--scientific', action='store_true', help='Use 17-col scientific subset (no TrID/GeneID/GeneName)')
    parser.add_argument(
        "--use-scientific-cols",
        action="store_true",
        default=True,  # default ON for safety
        help="Use canonical 17-col SCIENTIFIC_COLS (ignores positional column arg). Default True."
    )
    parser.add_argument(
        "--no-scientific-cols",
        action="store_true",
        help="Disable SCIENTIFIC_COLS, use positional column arg instead (legacy mode)."
    )
    args = parser.parse_args()

    cols = [c.strip() for c in args.legacy_cols_csv.split(",")]

    if args.scientific:
        excluded = {'TrID', 'GeneID', 'GeneName'}
        cols = [c for c in cols if c not in excluded]
        print(f'scientific subset ({len(cols)} cols): {",".join(cols)}')

    if args.baseline == "legacy":
        baseline_label = "legacy (H_0)"
    else:
        baseline_label = "current (H_1)"
    print(f"[baseline: {baseline_label}]")

    if args.scientific:
        print("[mode: scientific (17 cols, no TrID/GeneID/GeneName)]")

    # Decide which column list to use for SHA computation.
    # Default is SCIENTIFIC_COLS (hardcoded 17-col subset).
    # --no-scientific-cols reverts to the legacy positional column argument.
    if args.use_scientific_cols and not args.no_scientific_cols:
        columns = SCIENTIFIC_COLS
        print("Using canonical 17-col SCIENTIFIC_COLS; positional column argument ignored")
    else:
        # legacy: use positional argument as before
        columns = cols
        print("Using positional column argument (legacy mode)")
    print(f"[INPUT_COLS: {','.join(columns)}]")

    if args.self_test:
        ok = self_test(args.baseline_tsv, columns)
        print(f"self-test: {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 4)

    base_hash, base_err = hash_columns(args.baseline_tsv, columns)
    new_hash, new_err = hash_columns(args.new_tsv, columns)

    if base_err:
        print(f"BASELINE ERROR: {base_err}")
        sys.exit(2)
    if new_err:
        print(f"NEW ERROR: {new_err}")
        sys.exit(2)

    print(f"baseline legacy-cols SHA: {base_hash}")
    print(f"new      legacy-cols SHA: {new_hash}")
    print(f"match: {base_hash == new_hash}")
    sys.exit(0 if base_hash == new_hash else 3)


if __name__ == "__main__":
    main()
