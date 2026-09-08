`# AIDRS: AI-Aided Isoform Discovery for direct RNA-Seq

**AIDRS** (AI-Aided Isoform Discovery for direct RNA-Seq) is an advanced sequencing data–driven framework for full-length RNA isoform reconstruction and quantification from Oxford Nanopore Technology Direct RNA sequencing. Inspired by [ISAtools](https://github.com/shizhuoxing/ISAtools), AIDRS extends the functionality with additional capabilities for protein coding potential prediction and translation site identification.

Designed with annotation flexibility and biological fidelity in mind, AIDRS supports isoform identification with high precision and recall, and accurately resolves splice junctions and transcript boundaries directly from read evidence.

If reference annotations are available, AIDRS incorporates conserved, low-abundance isoforms through guided filtering and rescue steps, further enhancing transcriptome completeness.

AIDRS incorporates [TranslationAI](https://github.com/rnasys/TranslationAI) for protein coding potential prediction and [Puffin](https://github.com/jzhoulab/puffin) for enhanced TSS prediction.

## 🧬 AIDRS Enhanced Features

- **Protein Coding Potential Prediction**: Using deep learning models to assess transcript coding ability and precise identification of start and stop codons.
- **Enhanced TSS/TES Prediction**: Improved transcription start site identification with the Puffin model.
- **Poly(A) Length Assessment**: AIDRS consumes BAM files carrying per-read polyA tail lengths (typically emitted by Dorado basecalling) and emits per-isoform and per-gene tail-length distributions as Parquet sidecars. See [Poly(A) Length Outputs](#polya-length-outputs) below for the schema.

## 📦 Installation

AIDRS requires **Python 3.11** and several dependencies including `samtools`, PyTorch, and TensorFlow.

We recommend using [Conda](https://docs.conda.io/) to manage dependencies and environments.

### ✅ Create AIDRS environment (recommended)

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
- `aidrs.transcript.assessment.tsv`: Transcript model assessment statistics (19-column canonical schema).
- `aidrs.transcript.counts.tsv` / `aidrs.transcript.cpm.tsv`: Per-sample raw counts and CPM matrices at the transcript level.
- `aidrs.gene.counts.tsv` / `aidrs.gene.cpm.tsv`: Per-sample raw counts and CPM matrices at the gene level.

See [Quantification (Counts & CPM)](#quantification-counts--cpm) for what reads contribute to these matrices and what is intentionally discarded.

### Optional Output

- `aidrs.transcript.polyA_len.parquet` / `aidrs.gene.polyA_len.parquet`: Poly(A) tail-length sidecars. See [Poly(A) Length Outputs](#polya-length-outputs).
- With `--keep_temp`, the full `temp/` directory is preserved, including per-sample `*_flnc_correct.ssc` files (read-level SSC with alignment details).

---

## Poly(A) Length Outputs

When the input BAM carries per-read polyA tail lengths (typically emitted by Dorado basecalling), AIDRS writes two Parquet sidecar tables `aidrs.transcript.polyA_len.parquet` / `aidrs.gene.polyA_len.parquet` alongside `aidrs.transcript.assessment.tsv`. 

---

## Quantification (Counts & CPM)

AIDRS count and CPM matrices retain reads that fully match the final transcript model. If you need to retain all FLNC reads, including low-coverage reads that do not match a model, or want probabilistic assignment of ambiguous reads, use [NanoCount](https://github.com/a-slide/NanoCount) as a complementary quantifier. 

NanoCount uses an expectation-maximization (EM) algorithm to distribute multi-mapping reads probabilistically across compatible isoforms. Unlike AIDRS's default counting, NanoCount requires alignments to a **transcriptome reference** (not the genome), and reads must be aligned with `minimap2` retaining secondary alignments (`-N 10`).

AIDRS's `aidrs.transcript_model.gtf`, `aidrs.transcript_model.fasta` can feed this NanoCount workflow alongside the original BAM and reads FASTA. NanoCount is a complement to AIDRS's default counting and filtering, not a replacement for them.
