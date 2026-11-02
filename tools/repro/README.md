# `tools/repro/` — reproduction scripts for the 2026-11 issue triage

These scripts reproduce, on real data, the defects found while triaging
AIDRS issues #2 / #3 / #4 (2026-11-02). They are read-only: they never
write into a delivery directory.

| Script | What it shows |
|---|---|
| `filter_window_recompute.py` | Issue #4 — re-quantifies every library of a delivery and compares the old `libs[3:]` filter window against the fixed all-libraries window (how many transcripts the fix rescues, and which library is most affected). |
| `id_length_scan.py` | Issue #3 — scans the annotated model for transcript / gene ids over the SAM v1.6 QNAME limit (254 bytes) and for duplicated ids, and shows which source produced the over-long ids. |
| `cpm_denominator_check.py` | Issue #2 — proves that the transcript and gene CPM matrices of a delivery use the **same** denominator (the library size), so the difference between their column sums comes from the filtered row set, not from different denominators. |
| `quant_fail_loud_check.py` | The Stage 3 fail-loud path — deletes one sample's `*_flnc_correct.ssc` just before quantification and checks that the run aborts (non-zero exit + `aidrs.quantification_failures.tsv`) unless `--allow_partial_quantification` is passed. |

## Usage

```bash
PY=/datf/hanxi/software/miniconda3/envs/aidrs/bin/python
DELIVERY=/datf/hanxi/project/DRS/result_sample_col_2_minimap_aidrs_v3/AIDRS

$PY tools/repro/id_length_scan.py --delivery $DELIVERY
$PY tools/repro/cpm_denominator_check.py --delivery $DELIVERY
$PY tools/repro/filter_window_recompute.py --delivery $DELIVERY \
        --min-expressed-samples 3 --out /tmp/copd_quant_all.parquet
```

`filter_window_recompute.py` re-quantifies every library, so it takes
roughly 15 minutes for a 30-library delivery.

## Related

- The full triage, evidence and patch design:
  `workspace/docs/aidrs_issue_final_plan_2026-11-02.md`
- The execution report and the expert-panel review:
  `workspace/docs/aidrs_batch12_execution_report_2026-11-02.md`
- The regression tests for both defects live in
  `tests/test_defensive.py` (`TestMinExpressedSamplesFilter`,
  `TestLongFusionGeneId`, `TestTranscriptIdCollision`, `TestIdValidator`,
  `TestGeneModelMap`, `TestMatrixDenominators`).
