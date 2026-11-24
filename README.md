# AIDRS: AI-Aided Isoform Discovery for direct RNA-Seq

**AIDRS** (AI-Aided Isoform Discovery for direct RNA-Seq) is an advanced sequencing data–driven framework for full-length RNA isoform reconstruction and quantification from Oxford Nanopore Technology Direct RNA sequencing. Inspired by [ISAtools](https://github.com/shizhuoxing/ISAtools), AIDRS extends the functionality with additional capabilities for protein coding potential prediction and translation site identification.

Designed with annotation flexibility and biological fidelity in mind, AIDRS supports isoform identification with high precision and recall, and accurately resolves splice junctions and transcript boundaries directly from read evidence.

If reference annotations are available, AIDRS incorporates conserved, low-abundance isoforms through guided filtering and rescue steps, further enhancing transcriptome completeness.

## 🌟 AIDRS Enhanced Features

- **Enhanced TSS/TES Prediction**: Improved transcription start site identification with the [Puffin](https://github.com/jzhoulab/puffin) model, which mitigates the 5′-end truncation inherent in direct RNA sequencing reads.
- **Protein Coding Potential Prediction**: Using [TranslationAI](https://github.com/rnasys/TranslationAI) to assess transcript coding ability and precise identification of start and stop codons.
- **Poly(A) Length Assessment**: AIDRS consumes BAM files carrying per-read polyA tail lengths (typically emitted by Dorado basecalling) and emits per-isoform and per-gene tail-length distributions as Parquet sidecars. See [Poly(A) Length Outputs](#polya-length-outputs) below for the schema.

## 📦 Installation

AIDRS requires **Python 3.11** and several dependencies including `samtools`, PyTorch, and TensorFlow.

We recommend using [Conda](https://docs.conda.io/) to manage dependencies and environments.

### Create AIDRS environment (recommended)

```bash
git clone https://github.com/x1han/AIDRS.git
cd AIDRS
conda env create -f environment.yml
conda activate aidrs
pip install -e .
```

---

## 🧬 Example: K562 (SG-NEx) on GENCODE

This example walks through a complete run using K562 direct RNA reads from the SG-NEx consortium and the GENCODE human reference, with all AIDRS parameters at their defaults.

### Step 1 — Download references and reads

| Resource | Where to obtain |
|---|---|
| Reference genome FASTA (GRCh38) | [GENCODE Human](https://www.gencodegenes.org/human/) → *Fasta files* → GRCh38.primary_assembly.genome.fa.gz |
| Gene annotation GTF (GENCODE) | [GENCODE Human](https://www.gencodegenes.org/human/) → *GTF / GFF3 files* → gencode.primary_assembly.annotation.gtf.gz |
| K562 direct RNA reads (FASTQ) | [SG-NEx K562 direct RNA replicate 1 run 1](http://sg-nex-data.s3.amazonaws.com/data/sequencing_data_ont/fastq/SGNex_K562_directRNA_replicate1_run1/) |

### Step 2 — Align reads with minimap2

```bash
minimap2 -t 8 -ax splice -uf -k 14 -y \
    example/ref/GRCh38.primary_assembly.genome.fa.gz \
    example/fastq/SGNex_K562_directRNA_replicate1_run1.fastq.gz \
    | samtools sort -@ 4 -o example/SGNex_K562_directRNA_replicate1_run1.bam
samtools index example/SGNex_K562_directRNA_replicate1_run1.bam
```

The `-uf -k 14 -y` flags are required for forward-strand ONT reads and keep tag information e.g. RNA modification and polyA length.

### Step 3 — Run AIDRS with default parameters

```bash
aidrs \
    -r example/ref/GRCh38.primary_assembly.genome.fa.gz \
    -b example/SGNex_K562_directRNA_replicate1_run1.bam \
    -g example/ref/gencode.primary_assembly.annotation.gtf \
    --min_expressed_samples 1 \
    -o example/output
```

### Notes

- SG-NEx DRS datasets were sequenced on ONT RNA002 chemistry, which may not emit per-read polyA tags in the BAM; in those cases AIDRS skips polyA-based isoform filtering.
- Reference-guided mode (`--gtf_anno`) is **off by default**. Enable it when you want AIDRS to rescue GENCODE-annotated isoforms with low read support.
- For detailed parameter information, see the **aidrs --help**.
---

## 🧪 Regression gate

`scripts/verify_c107_golden.py` runs the C107 100k regression gate:

```bash
python scripts/verify_c107_golden.py --candidate <output dir> --sha-only
python scripts/verify_c107_golden.py --baseline <baseline output dir> --candidate <output dir>
```

The gate passes when either

- **(a)** the sha256 of `aidrs.transcript.assessment.tsv` hits the pinned golden `c289f84d1899b2dd45088200cde8334e867bae7416ece05c23283a4a6d69b440`, or
- **(b)** the change is bounded: the five quantification files are byte-identical to the baseline, the assessment schema is unchanged and only `TrStart` / `TrEnd` / `seq_len` / `TIS_related_location` / `TTS_related_location` / `TIS_score` / `TTS_score` differ, every 5' change is attributable to a new 3' variant (no model moved its 5' end while its 3' end stayed put), at most 1 % of the models moved their 3' end, and the score columns agree within 1e-5.

Mode (a) pins the assessment only — a regression confined to the counts/CPM layer would leave the assessment digest untouched — so pass `--baseline` to have the five quantification files compared in mode (a) as well. The gate prints a receipt on the comparison path (baseline and candidate sha256, matched/unmatched model counts, the 5' violation counts and the 3' change count) and fails loudly when the comparison would be vacuous. Mode (b) does not absorb changes that add or remove models — a different model count, a changed model id or a changed assessment schema is a FAIL, not a bounded change.

Byte-level comparisons need the usual preconditions: an isolated `TMPDIR` per run (a shared `TMPDIR` produces a spurious 2-row difference), the same BAM file name, `--threads` pinned, and no other AIDRS run on the same box. Produce the baseline from a pinned commit (`git archive <sha>`) and record that sha with `--baseline-label`, which is echoed into the receipt.

The pinned golden assumes that at most one library contributes reads to a key: only then are the pooled ratio and the unweighted median the same number, and only then is the `polyA_frac` column guaranteed not to move. The gate reads the library count from the quantification header and prints it in the receipt; when it is greater than one it warns, because a second library legitimately moves `polyA_frac` for the keys its reads reach and forces a re-baseline (record the new sha and its cause).

---

## ⚠️ Known Limitations

### Barcode / SeqTagger libraries and the `pt:i=0` rate

Dorado's `pt:i` tag is the measured poly(A) tail length of a read. AIDRS keeps the failure states distinct in its internal `polyA_len` column; only a strictly positive length counts as evidence:

| `pt:i` | AIDRS `polyA_len` | meaning |
| --- | --- | --- |
| `> 0` | `> 0` | a poly(A) tail was measured — this is the evidence AIDRS uses to place a 3' end |
| `0` | `0` | the poly(A) anchor was found, but the tail length could not be estimated |
| `-1` | `-1` | no poly(A) anchor was found |
| tag absent | `-2` | the read carries no `pt` tag at all (AIDRS's own sentinel; Dorado never writes `-2`) |

Barcode / SeqTagger libraries carry a **library-level** `pt:i=0` rate that differs strongly between libraries of the same cohort. Measured on the RNA-004 barcode cohort (two runs): the per-library rate ranges from 4.4–11.4 % for most libraries up to 30.5–40.9 % for one of them. A high `pt:i=0` rate means less poly(A) evidence per locus, hence fewer 3' variants AIDRS can distinguish and more coordinates that fall back to the read-level mode.

The failure mode is specific to barcoded libraries. In ten `noSeqTagger` control libraries (COPD29/38/41/45/52, HC2/3/26/30/31) the `pt:i=0` count is **exactly 0** — the only failure state present is a missing `pt` tag — while the barcoded RNA-004 cohort of the same chemistry shows 14.7–16.0 % `pt:i=0` in the merged pool. Two caveats: the control libraries were basecalled with a different Dorado release (0.8.0 vs 1.3.0), so this is not a controlled comparison; and the demultiplexing step lowers the rate AIDRS actually sees (below).

Two measurement caveats that matter when you set thresholds:

- **Measure the merged (pre-demux) BAM as well as the demuxed one.** Barcode assignment fails preferentially on `pt:i=0` reads, so demultiplexing *lowers* the measured `pt:i=0` rate: on the same cohort, 14.7 % of the merged pool became 11.6 % after demultiplexing, and 52 % of the discarded reads were `pt:i=0` reads. The post-demux number AIDRS sees is a biased-low estimate of the library's real rate.
- **`pt:i=0` reads cannot be rescued.** In Dorado's `pa` tag the tail fields are `-1` on every `pt:i=0` read — checked on 56 154 reads in one run and on a 2 % random sample of the merged pool in the other (37 261 `pt:i=0` reads, 0 exceptions; more than 90 000 `pt:i=0` reads in total) — so there is no recoverable poly(A) evidence in those reads. Only a different library preparation or a different basecalling configuration changes that.

Check both numbers for your own data (the first command gives the denominator, the second the breakdown):

```bash
samtools view -c your.bam
samtools view your.bam | grep -o 'pt:i:[-0-9]*' | sort | uniq -c | sort -rn | head
```

If `pt:i:0` dominates, the options are a library without barcoding, or accepting more read-level (rather than poly(A)-supported) 3' end placement — there is no threshold that recovers the lost evidence.

Note what `--polya_fraction_threshold` (default `0.95`) actually does today: on the **default** path the poly(A) criterion is not consulted at all — it requires a poly(A) read-count column that production does not produce — so changing that flag has no effect on a default run; the poly(A) gate is live only under `--hard_filter`. Under `--hard_filter` the threshold is applied to every row as `polyA_frac > threshold`, and it does not gate NMD/truncation rows on their own. On a `pt`-less BAM `polyA_frac` is NaN, so the poly(A) criterion is bypassed (the 5' Puffin gate still applies).

### Purely single-exon genes are not assembled

The current version does **not** assemble genes whose transcripts are all single-exon: such genes are absent from `aidrs.transcript_model.gtf` and from the quantification matrices. Measured on the annotation used for the expert review: 24,652 of 78,941 genes (31.2 %) have only single-exon transcripts, and 0 of them reach the model (exact, version-stripped and base-deduplicated matching all give 0). This is a consequence of the single-exon routing, not evidence that single-exon transcripts are handled correctly. Use a dedicated single-exon caller for all-single-exon loci.
---

## 📁 Output Files

### Main Output

- `aidrs.transcript_model.gtf`: Final filtered transcript models (including known and novel isoforms).
- `aidrs.transcript.assessment.tsv`: Transcript model assessment statistics (19-column canonical schema). The two `Puffin_TSS_*` columns are text: `no` means "no peak found", every other value is the numeric score written as a string — convert them before comparing with a threshold, otherwise a text value silently fails the numeric comparison (AIDRS's own 5' gate tests the value's type, not its text).
- `aidrs.transcript.counts.tsv` / `aidrs.transcript.cpm.tsv`: Per-sample raw counts and CPM matrices at the transcript level.
- `aidrs.gene.counts.tsv` / `aidrs.gene.cpm.tsv`: Per-sample raw counts and CPM matrices at the gene level.
- `aidrs.gene.model_map.tsv`: For every gene of the gene layer, whether it is in the transcript model and how many of its transcripts are. See [Quantification (Counts & CPM)](#quantification-counts--cpm).

See [Quantification (Counts & CPM)](#quantification-counts--cpm) for what reads contribute to these matrices and what is intentionally discarded.

### Optional Output

- `aidrs.transcript.polyA_len.parquet` / `aidrs.gene.polyA_len.parquet`: Poly(A) tail-length sidecars. See [Poly(A) Length Outputs](#polya-length-outputs).
- `aidrs.transcript.polya_per_sample.tsv`: Per-sample poly(A) statistics behind the `polyA_frac` column, one row per transcript key and sample: `polyA_freq`, `raw_read_freq`, `polyA_frac` (the per-sample ratio), plus the per-key aggregate `polyA_frac_model` (the value the model received when the key is a model key), `polyA_frac_pooled`, `polyA_n_samples` and `estimator` (`median` or `pooled_fallback`). Its key space is the Stage 2.3 *read* key space, a superset of the delivered model, and a model key with no reads has no row here. Written only when poly(A) was measured, i.e. not on a `pt`-less BAM. See [Poly(A) Length Outputs](#polya-length-outputs).
- With `--keep_temp`, the full `temp/` directory is preserved, including per-sample `*_flnc_correct.ssc` files (read-level SSC with alignment details).

---

## Poly(A) Length Outputs

When the input BAM carries per-read polyA tail lengths (typically emitted by Dorado basecalling), AIDRS writes two Parquet sidecar tables `aidrs.transcript.polyA_len.parquet` / `aidrs.gene.polyA_len.parquet` alongside `aidrs.transcript.assessment.tsv`. 

### How `polyA_frac` is computed

The `polyA_frac` column of `aidrs.transcript.assessment.tsv` is the **unweighted median of the per-sample poly(A) fractions** over the samples that carry measured poly(A) evidence for that transcript — a sample qualifies when it has at least 5 reads on the transcript **and** at least one read with a measured tail. With an even number of qualifying samples the median is the mean of the two middle values (the numpy convention). When no sample qualifies, the value falls back to the read-depth-weighted pooled ratio.

This is deliberate: a read-depth-weighted pooled ratio lets one abnormally deep library (or one library whose tails are systematically shorter) dominate the cohort's poly(A) evidence. With the median, a single abnormal library can no longer drag the whole cohort down. The per-sample numbers behind the value are delivered in `aidrs.transcript.polya_per_sample.tsv` (one row per transcript key and sample), together with the per-key aggregate, so the statistic stays auditable.

Two consequences to keep in mind: on a **single-library** run the median and the pooled ratio are the same number, so nothing changes there; and on a **multi-library** run the `polyA_frac` values change compared to earlier versions (the pooled value is no longer what the column holds).

---

## Quantification (Counts & CPM)

### The two layers have different scopes — and the same denominator

The **transcript layer** keeps only the transcripts that passed `--min_expressed_samples`;
the **gene layer** is aggregated from *all* quantified transcripts, so it is deliberately a
**superset** of the model (a gene can carry counts while none of its transcripts is in
`aidrs.transcript_model.gtf`). Both CPM tables use the **same denominator per library**
(`L` = the library size), so a CPM column sums to `1e6` only when the layer covers every
quantified transcript.

One sidecar makes that explicit:

| File | One row per | Columns |
|---|---|---|
| `aidrs.gene.model_map.tsv` | gene | `in_transcript_model`, `n_model_transcripts`, `n_total_transcripts` |

`L` itself needs no sidecar: it is the gene-counts column sum, and it is recoverable from the
transcript layer as `count / cpm * 1e6` (on the rows that carry reads) — the two agree on every
library. The count matrix and the CPM matrix of a layer describe the same rows in the same order,
so the two can be compared — or zipped — row by row.

### Reads that are discarded

AIDRS count and CPM matrices retain reads that fully match the final transcript model. If you need to retain all FLNC reads, including low-coverage reads that do not match a model, or want probabilistic assignment of ambiguous reads, use [NanoCount](https://github.com/a-slide/NanoCount) as a complementary quantifier. 

NanoCount uses an expectation-maximization (EM) algorithm to distribute multi-mapping reads probabilistically across compatible isoforms. Unlike AIDRS's default counting, NanoCount requires alignments to a **transcriptome reference** (not the genome), and reads must be aligned with `minimap2` retaining secondary alignments (`-N 10`).

AIDRS's `aidrs.transcript_model.gtf`, `aidrs.transcript_model.fasta` can feed this NanoCount workflow alongside the original BAM and reads FASTA. NanoCount is a complement to AIDRS's default counting and filtering, not a replacement for them.

---

## License

AIDRS's own source code is licensed under the **GNU General Public License v3.0 only**
(`GPL-3.0-only`); see [`LICENSE`](LICENSE). Copyright (C) 2025-2026 Xi Han.

This repository bundles third-party components that are **not** covered by the GPL and keep their
own terms; the `aidrs` wheel additionally ships the TranslationAI source and its trained weights:

| Component | Upstream | Terms |
|---|---|---|
| `aided/Puffin/` — model code and `resources/puffin.pth` | [jzhoulab/puffin](https://github.com/jzhoulab/puffin) | Academic research use only; commercial use and use by for-profit entities are expressly prohibited (BSD-3-Clause-style text carrying a non-commercial restriction; see `aided/Puffin/LICENSE`) |
| `aided/TranslationAI/` — source | [rnasys/TranslationAI](https://github.com/rnasys/TranslationAI) | GNU AGPL v3.0 (upstream's `LICENSE` file; upstream's README and `setup.py` say "GPLv3" — upstream is internally inconsistent, and so is this copy's locally authored `pyproject.toml`) |
| `aided/TranslationAI/translationai/models/*.h5` — trained weights, shipped in the wheel | same | Stated upstream as [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (academic and non-commercial use); upstream publishes no separate license text for the weights and lists Xiaojuan Fan as its author |

Notes:

- AIDRS **imports** the Puffin and TranslationAI code into the same process, so in a checkout the
  bundle is a combined work rather than a mere aggregation. GPLv3 §13 / AGPLv3 §13 permit
  combining GPLv3 code with the AGPL-3.0 TranslationAI source; if you modify that combination and
  let users interact with it remotely, AGPLv3 §13's network-source clause applies to it.
- The Puffin terms are **not** GPL-compatible (academic research use only). Consequently the
  repository distribution is **not licensed for use, modification or redistribution for commercial
  or for-profit purposes**, regardless of the GPL terms applying to AIDRS's own code; in the wheel
  the same restriction follows from the non-commercial weights.
- The trained TranslationAI weights are non-commercial (CC BY-NC 4.0) and are not conveyed under
  GPL terms.
- Both vendored copies carry local modifications (thread pinning, model-path resolution,
  documentation fixes), which is why the AGPL's source-offer obligations are live.
