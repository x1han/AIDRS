#!/bin/bash
# Quick factorial gate for release candidates.
# Runs 2-case factorial on C107 chr1 (case1 +TransAI / case3 -TransAI) and
# programmatically diffs against the chr1 golden SHA prefixes in
# $GOLDEN_FILE (default: workspace/findings/chr1_factorial_golden_sha.txt).
# Exits non-zero on any case failure, comparator failure, SHA drift, or
# assessment-TSV schema drift.
#
# All hardcoded paths have env-var-with-default form so the gate runs
# in CI or on alternate hostnames without rewriting the script.
#
# To update the golden SHAs after a legitimate drift:
#   1. Run the gate; it prints the new SHAs on FAIL with [NEW_SHA] prefix.
#   2. Verify the drift cause in memory/chr1_factorial_golden.md.
#   3. Edit the golden file with the new prefixes.
set -euo pipefail

# --- Path / interpreter overrides (env-var-with-default) -----------------
AIDRS_REPO="${AIDRS_REPO:-/datf/hanxi/software/AIDRS/repo}"
BENCH_DIR="${BENCH_DIR:-/datf/hanxi/test/AIDRS/benchmark_chr1}"
REF="${REF:-/datf/hanxi/database/reference/GENCODE/GRCh38.p14/GRCh38.primary_assembly.genome.fa}"
BAM="${BAM:-/datf/hanxi/test/AIDRS/benchmark_chr1/C107_chr1_with_polyA.bam}"
GOLDEN_FILE="${GOLDEN_FILE:-/datf/hanxi/software/AIDRS/workspace/findings/chr1_factorial_golden_sha.txt}"
V03_BASELINE="${V03_BASELINE:-/datf/hanxi/test/AIDRS/benchmark_chr1/run_v03_baseline}"
PYTHON="${PYTHON:-/datf/hanxi/software/miniconda3/envs/aidrs/bin/python}"
OUT=/tmp/quick_gate_$(date +%Y%m%d_%H%M%S)

# transai env only needed for legacy compatibility; harmless when unset.
export PATH=/datf/hanxi/software/miniconda3/envs/transai/bin:$PATH

# --- Fail-fast guards on absolute paths ---------------------------------
if [ ! -d "$AIDRS_REPO" ]; then
    echo "[FAIL] AIDRS_REPO not a directory: $AIDRS_REPO" >&2
    exit 1
fi
if [ ! -d "$BENCH_DIR" ]; then
    echo "[FAIL] BENCH_DIR not a directory: $BENCH_DIR" >&2
    exit 1
fi
if [ ! -d "$V03_BASELINE" ]; then
    echo "[FAIL] v0.3 baseline missing at $V03_BASELINE" >&2
    exit 1
fi
if [ ! -f "$V03_BASELINE/aidrs.transcript.assessment.tsv" ]; then
    echo "[FAIL] v0.3 baseline missing assessment.tsv at $V03_BASELINE" >&2
    exit 1
fi
if [ ! -f "$REF" ]; then
    echo "[FAIL] reference FASTA missing: $REF" >&2
    exit 1
fi
if [ ! -f "$BAM" ]; then
    echo "[FAIL] BAM missing: $BAM" >&2
    exit 1
fi
if [ ! -f "$GOLDEN_FILE" ]; then
    echo "[FAIL] golden SHA file missing: $GOLDEN_FILE" >&2
    echo "[FAIL] See tools/run_quick_factorial_gate.sh header for how to create one." >&2
    exit 1
fi

GOLDEN_CASE1=$(grep '^case1 ' "$GOLDEN_FILE" | awk '{print $2}')
GOLDEN_CASE3=$(grep '^case3 ' "$GOLDEN_FILE" | awk '{print $2}')
if [ -z "$GOLDEN_CASE1" ] || [ -z "$GOLDEN_CASE3" ]; then
    echo "[FAIL] golden SHA file malformed (expected 'case1 <prefix>' and 'case3 <prefix>' lines)" >&2
    exit 1
fi

mkdir -p "$OUT"
ln -s "$(realpath "$V03_BASELINE")" "$OUT/run_v03_baseline"
cd "$AIDRS_REPO"

echo "[gate] case1 (+TransAI, +polyA)..."
$PYTHON -m src.aidrs --reference "$REF" --bam "$BAM" \
    --output "$OUT/case1_transAI_polyA" --threads 4 \
    > "$OUT/case1_transAI_polyA.log" 2>&1
echo "[gate] case1 completed"

echo "[gate] case3 (-TransAI, +polyA)..."
$PYTHON -m src.aidrs --reference "$REF" --bam "$BAM" \
    --output "$OUT/case3_noTransAI_polyA" --threads 4 --no_translationai \
    > "$OUT/case3_noTransAI_polyA.log" 2>&1
echo "[gate] case3 completed"

echo "[gate] running comparator..."
$PYTHON tools/compare_factorial_cases.py "$OUT" --full-diff-limit 0

# --- Schema smoke test ----------------------------------------------------
# validate_canonical_schema (added to column_registry 2026-09-06) catches
# the silent failure mode where a CORE column was renamed/dropped in a
# refactor yet the SHA happens to still match because the remaining
# 17 columns were identical. Fail-loud here means we never report a
# byte-identical SHA on a structurally-broken TSV.
echo "[gate] running validate_canonical_schema smoke test..."
SCHEMA_RC=0
$PYTHON - "$OUT/case1_transAI_polyA/aidrs.transcript.assessment.tsv" \
        "$OUT/case3_noTransAI_polyA/aidrs.transcript.assessment.tsv" <<'PYEOF' || SCHEMA_RC=$?
import sys
sys.path.insert(0, "/datf/hanxi/software/AIDRS/repo")
import pandas as pd
from src.aidrs_runtime.column_registry import (
    CORE_ASSESSMENT_COLS,
    validate_canonical_schema,
)
fail = False
for path in sys.argv[1:]:
    try:
        df = pd.read_csv(path, sep="\t", nrows=5)
    except Exception as e:
        print(f"[FAIL] schema smoke: cannot read {path}: {e}", file=sys.stderr)
        fail = True
        continue
    try:
        cols = validate_canonical_schema(df)
    except RuntimeError as e:
        print(f"[FAIL] schema smoke: {path}: {e}", file=sys.stderr)
        fail = True
        continue
    missing = [c for c in CORE_ASSESSMENT_COLS if c not in cols]
    print(f"[OK] schema smoke: {path} CORE cols ({len(cols)}) present"
          + (f"; missing={missing}" if missing else ""))
if fail:
    sys.exit(1)
PYEOF
if [ $SCHEMA_RC -ne 0 ]; then
    echo "[FAIL] schema smoke test failed -- aborting SHA compare" >&2
    exit 1
fi
echo "[gate] schema smoke test passed"

echo "[gate] comparing SHAs against golden..."
mapfile -t NEW_SHAS < <($PYTHON tools/extract_scientific_sha.py \
    "$OUT/case1_transAI_polyA/aidrs.transcript.assessment.tsv" \
    "$OUT/case3_noTransAI_polyA/aidrs.transcript.assessment.tsv" 2>/dev/null)
NEW_CASE1="${NEW_SHAS[0]:-}"
NEW_CASE3="${NEW_SHAS[1]:-}"
NEW_CASE1_PREFIX="${NEW_CASE1:0:12}"
NEW_CASE3_PREFIX="${NEW_CASE3:0:12}"

echo "[gate] case1 SHA: ${NEW_CASE1_PREFIX:-<missing>} (golden: $GOLDEN_CASE1)"
echo "[gate] case3 SHA: ${NEW_CASE3_PREFIX:-<missing>} (golden: $GOLDEN_CASE3)"

EXIT=0
if [ -z "$NEW_CASE1_PREFIX" ] || [ "$NEW_CASE1_PREFIX" != "$GOLDEN_CASE1" ]; then
    echo "[FAIL] case1 SHA drift" >&2
    echo "[FAIL] expected prefix: $GOLDEN_CASE1" >&2
    echo "[FAIL] actual prefix:   ${NEW_CASE1_PREFIX:-<missing>}" >&2
    echo "[NEW_SHA] case1 $NEW_CASE1_PREFIX" >&2
    EXIT=1
fi
if [ -z "$NEW_CASE3_PREFIX" ] || [ "$NEW_CASE3_PREFIX" != "$GOLDEN_CASE3" ]; then
    echo "[FAIL] case3 SHA drift" >&2
    echo "[FAIL] expected prefix: $GOLDEN_CASE3" >&2
    echo "[FAIL] actual prefix:   ${NEW_CASE3_PREFIX:-<missing>}" >&2
    echo "[NEW_SHA] case3 $NEW_CASE3_PREFIX" >&2
    EXIT=1
fi

# --- Auto --write-diff on FAIL ------------------------------------------
# When the SHA comparison fails, automatically produce forensic
# row-level diff TSVs in <bench>/diffs/ so the operator can see which
# rows were rescued/lost without a manual second invocation.
if [ $EXIT -ne 0 ]; then
    DIFF_DIR="$BENCH_DIR/diffs/$(basename "$OUT")"
    mkdir -p "$DIFF_DIR"
    echo "[gate] auto-running compare_factorial_cases.py --write-diff -> $DIFF_DIR"
    $PYTHON tools/compare_factorial_cases.py "$OUT" --full-diff-limit 0 \
        --write-diff "$DIFF_DIR" \
        > "$DIFF_DIR/auto_write_diff.log" 2>&1 || \
        echo "[WARN] --write-diff run failed (non-fatal); see $DIFF_DIR/auto_write_diff.log" >&2
fi

if [ $EXIT -eq 0 ]; then
    echo "[PASS] Gate clear. Output at $OUT"
fi
exit $EXIT
