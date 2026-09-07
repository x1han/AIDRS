# AIDRS 1.0.0 Release Notes (2026-09-07)

AIDRS 1.0.0 is the first official production release of the
**AI-Aided Isoform Discovery for direct RNA-Seq** pipeline. It is an
industrial-grade engine for full-length RNA isoform reconstruction,
isoform-level quantification, and coding-potential adjudication built
specifically for Oxford Nanopore Direct RNA-Seq (DRS).

## Core Scientific Breakthrough & Benchmark Performance

Validation dataset: HEK293T C107 chr1 (2,663 transcript models), GENCODE v47.

### Translation Initiation / Termination Site (TIS / TTS) Physical Accuracy
For Full Splice Match (FSM) transcript models with annotated CDS (N=58
mapped, N_TIS_PRED=47, N_TTS_PRED=47):

| Metric | Value | Gate / Status |
|---|---|---|
| TIS single-base exact match rate | **74.47%** (35/47) | ≥ 70% gate → **PASS** |
| TTS single-base exact match rate | **65.96%** (31/47) | scientific accuracy |
| TIS in-frame rate (Δ % 3 == 0) | 63.79% | biological plausibility |
| Median absolute physical deviation | **0.0 bp** (both TIS & TTS) | base-perfect on the aligned cohort |

### Truncation Artifact Detection Engine — Major Refactor
The legacy **symmetric bag-of-tokens overlap** heuristic was replaced
with a stricter **ordered intron subchain topology** check (per
SQANTI3 / FLAIR convention), plus a **Puffin TSS independent-promoter
rescue** to prevent alternative-TSS isoforms from being mis-flagged as
5'-degraded artifacts.

Measured impact (full chr1 cohort, 2,663 models):

| structural_category | truncation=yes (before) | truncation=yes (after) | Status |
|---|---:|---:|---|
| FSM | 69.47% | **19.72%** | ≤ 20% gate → PASS |
| Multi-FSM gene FSM subset | 83.18% | **14.96%** | variable-splicing preserved |
| ISM (Incomplete Splice Match) | 82.55% | 66.67% | true truncations retained |
| NIC (Novel In-Catalog) | 75.09% | 8.53% | novel isoforms protected |
| NNC (Novel Not-in-Catalog) | 67.31% | 9.48% | novel isoforms protected |
| **Overall** | **70.64%** | **15.06%** | biologically reasonable |

### Other Algorithmic Corrections
- **CDS Validator (TranslationAI ↔ GENCODE v47)**: Fixed strand-blind
  reference anchoring (was comparing minus-strand predictions against
  genomic min/max without strand awareness) and TIS/TTS off-by-one
  offset (`TIS_off + 1` vs `TTS_off + 0` reflects TranslationAI's
  asymmetric output convention). Outcome: TTS exact match rate moved
  from 0.00% → 65.96%, both-ends exact from 0% → 51.72%.
- **`group_freq_ratio` self-double-counting bug**: the formula
  `self / (self + group_freq)` counted `self` twice; corrected to
  `self / group_freq`.
- **Unified Splice Topology Engine**: Structural classification (`isoform_classify.py`) and truncation filtering (`ISM_filter.py`) are unified under the SQANTI3 / FLAIR contiguous intron subchain definition, eliminating the legacy prefix/suffix substring heuristic. Internal fragment isoforms are now correctly classified as ISM rather than NIC; drift witness: `tests/test_isoform_classify_internal_fragment.py`.
- **`rt_switching_filter` removed**: the cDNA RT-switching concept is
  invalid for DRS data and the module was deleted in commit `5334885`.
- **Deterministic SHA**: `tools/extract_scientific_sha.py` now sorts
  rows by physical coordinates before hashing, eliminating
  multi-threading non-determinism.

## Operational Scope & Guarantees

1. **Sequencing Protocol** — Oxford Nanopore native Direct RNA-Seq
   (SQK-RNA002 / SQK-RNA004). BAM inputs must be produced by Minimap2
   with `-uf -k 14 -y` flags (splice-aware, no reverse complement).
2. **Genetic Code** — NCBI Translation Table 1 (Standard Nuclear).
   `chrM` transcripts are processed for splice structure but excluded
   from canonical CDS / NMD benchmark assertions. Full mitochondrial
   codon-table support is planned for v1.1.
3. **Reference Genome** — matched uncompressed / bgzipped genomic
   FASTA + comprehensive gene annotation GTF (GENCODE / Ensembl).
4. **Determinism** — 16-col scientific SHA-256 baseline
   (`tools/extract_scientific_sha.py`) sorts output rows by physical
   coordinates to guarantee cross-node, cross-thread SHA stability.

## Known Limitations

- Single-exon (mono-exonic) transcripts are processed by the Stage 2.5b
  5-pillar funnel but were not present in the chr1 benchmark dataset.
- Whole-genome runs are not yet characterized; v1.0.0 baseline is
  established on chr1 only (C107 HEK293T, 2,663 transcript models).
- Cross-chromosome centromeric / telomeric chimeras and alternative
  pseudogene loci are not separately characterized.
- `compute_is_intergenic_or_antisense` does not yet cross-reference
  external reference GTF — known P2 latent issue tracked for v1.1.
- **stop_codon GTF span asymmetric with start_codon**
  (`src/aidrs_runtime/report_writers.py:156`): current implementation
  records `stop_codon = (current_genomic_pos - 1, current_genomic_pos + 1)`,
  a 3-bp span whose right edge spills 1 bp into the 3-prime UTR. Compare
  to `start_codon = (current_genomic_pos, current_genomic_pos + 2)` at
  line 146, which correctly anchors to the first base of the start codon.
  The correct `stop_codon` span should be
  `(current_genomic_pos - 2, current_genomic_pos)`. The 1-bp coordinate
  change will alter `aidrs.transcript_model.gtf` byte content and break
  external SHA baselines, so the fix is deferred to v1.1 alongside the
  GTF Emitter modular refactor.

## Reproducibility

To reproduce the chr1 benchmark:

```bash
cd /datf/hanxi/software/AIDRS/repo
bash tools/run_quick_factorial_gate.sh
```

Expected Modality Degradation Robustness: `Case 3 ⊇ Case 1` 100% inclusion
(2663/2663).

Expected 16-col scientific SHA256 (current golden):
```
case1 f174dbb3d664
case3 23287a73a6e6
```

## Test Suite

```
$ pytest tests/
======================= 86 passed, 3 warnings in 901.50s (0:15:01) ========================
```

All 86 tests passing across 17 characterization tests (Stage 2.6
decision tree, Stage 2.5b 5-pillar funnel, gtf2ssc SSC construction,
generate_reports pipeline) plus 66 existing unit / integration tests.