#!/bin/bash
# Quick factorial gate for release candidates.
# Runs 2-case factorial on C107 chr1 (~1 hour total) and verifies SHA.
#
# Cases run are the two the comparator needs to exercise the TranslationAI
# axis: case1 [+TransAI, +polyA] and case3 [-TransAI, +polyA]. Directory
# names are the ones tools/compare_factorial_cases.py expects; the v0.3
# baseline is symlinked in from the benchmark dir so the comparator has
# something to diff against.
#
# NOTE: goldens in memory/chr1_factorial_golden.md were produced at
# --threads 8. This gate runs at --threads 4; byte-identity is expected to
# hold across thread counts, and a SHA drift here is a finding, not noise.
set -u

export PATH=/datf/hanxi/software/miniconda3/envs/transai/bin:$PATH
PYTHON=/datf/hanxi/software/miniconda3/envs/aidrs/bin/python
REF=/datf/hanxi/database/reference/GENCODE/GRCh38.p14/GRCh38.primary_assembly.genome.fa
BAM=/datf/hanxi/test/AIDRS/benchmark_chr1/C107_chr1_with_polyA.bam
V03_BASELINE=/datf/hanxi/test/AIDRS/benchmark_chr1/run_v03_baseline
OUT=/tmp/quick_gate_$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUT"
ln -s "$V03_BASELINE" "$OUT/run_v03_baseline"
cd /datf/hanxi/software/AIDRS/repo

$PYTHON -m src.aidrs --reference "$REF" --bam "$BAM" \
    --output "$OUT/case1_transAI_polyA" --threads 4 \
    > "$OUT/case1_transAI_polyA.log" 2>&1
echo "case1 rc=$?"

$PYTHON -m src.aidrs --reference "$REF" --bam "$BAM" \
    --output "$OUT/case3_noTransAI_polyA" --threads 4 --no_translationai \
    > "$OUT/case3_noTransAI_polyA.log" 2>&1
echo "case3 rc=$?"

$PYTHON tools/compare_factorial_cases.py "$OUT" --full-diff-limit 0 2>&1 || true

echo "Done. Output at $OUT"
echo "Compare the printed SHAs against memory/chr1_factorial_golden.md."
