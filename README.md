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

## 📁 Output Files

### Main Output

- `aidrs.transcript_model.gtf`: Final filtered transcript models (including known and novel isoforms).
- `aidrs.transcript.assessment.tsv`: Transcript model assessment statistics (19-column canonical schema). The two `Puffin_TSS_*` columns are text: `no` means "no peak found", every other value is the numeric score written as a string. AIDRS's own 5' gate reads them tolerantly (it converts a non-`no` string to a number before comparing it with the threshold), so the in-memory run and a run that re-reads this file delete exactly the same rows — but **your** analysis code must still convert them before comparing with a threshold.
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

**Contract of the three values in that column**: `> 0` = a poly(A) tail was detected on at least one read of the qualifying samples; `== 0` = the transcript was measured and no tail was detected on any of its reads; `NaN` (empty cell) = **no poly(A) evidence for this transcript** (the per-key poly(A) table does not cover it — e.g. a model with no reads at that key). `NaN` and `0` are different statements and are treated differently by the Stage 2.6 filter: `0` is a measured negative, `NaN` is a bypass.

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
