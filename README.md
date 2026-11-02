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
- `aidrs.transcript.assessment.tsv`: Transcript model assessment statistics (19-column canonical schema).
- `aidrs.transcript.counts.tsv` / `aidrs.transcript.cpm.tsv`: Per-sample raw counts and CPM matrices at the transcript level.
- `aidrs.gene.counts.tsv` / `aidrs.gene.cpm.tsv`: Per-sample raw counts and CPM matrices at the gene level.
- `aidrs.gene.model_map.tsv`: For every gene of the gene layer, whether it is in the transcript model and how many of its transcripts are. See [Quantification (Counts & CPM)](#quantification-counts--cpm).
- `aidrs.matrix.denominators.tsv`: Per-library CPM denominator and the resulting column sums of both CPM matrices. See [Quantification (Counts & CPM)](#quantification-counts--cpm).

See [Quantification (Counts & CPM)](#quantification-counts--cpm) for what reads contribute to these matrices and what is intentionally discarded.

### Optional Output

- `aidrs.transcript.polyA_len.parquet` / `aidrs.gene.polyA_len.parquet`: Poly(A) tail-length sidecars. See [Poly(A) Length Outputs](#polya-length-outputs).
- `aidrs.novel_cluster_map.tsv`: Source gene ids of every fusion cluster whose gene id had to be shortened — either because the gene id itself exceeded 200 bytes, or because the gene id plus the transcript suffix would have exceeded the 254-byte SAM QNAME limit (the `reason` column says which). Emitted only when such a cluster exists.
- With `--keep_temp`, the full `temp/` directory is preserved, including per-sample `*_flnc_correct.ssc` files (read-level SSC with alignment details).

---

## Poly(A) Length Outputs

When the input BAM carries per-read polyA tail lengths (typically emitted by Dorado basecalling), AIDRS writes two Parquet sidecar tables `aidrs.transcript.polyA_len.parquet` / `aidrs.gene.polyA_len.parquet` alongside `aidrs.transcript.assessment.tsv`. 

---

## Quantification (Counts & CPM)

### The two layers have different scopes — and the same denominator

The **transcript layer** (`aidrs.transcript.counts.tsv`, `aidrs.transcript.cpm.tsv`) is the
final model: it contains only transcripts that passed `--min_expressed_samples`. The
**gene layer** (`aidrs.gene.counts.tsv`, `aidrs.gene.cpm.tsv`) is aggregated from *all*
quantified transcripts, so it is deliberately a **superset** of the model: a gene can
carry counts while none of its transcripts is in `aidrs.transcript_model.gtf`. The gene
counts stay the true counts of the gene (not a partial sum over surviving
transcripts), which keeps them comparable to independent gene-level counters such as
featureCounts.

`aidrs.gene.model_map.tsv` makes that scope explicit. One row per gene:

| Column | Meaning |
|---|---|
| `GeneID`, `GeneName` | the gene key used by the matrices |
| `in_transcript_model` | `True` iff the gene has ≥ 1 transcript in the delivered model |
| `n_model_transcripts` | transcripts of this gene in the delivered model |
| `n_total_transcripts` | transcripts of this gene in the *unfiltered* quantification (what the gene row was aggregated from) |

**Both CPM tables share one denominator per library.** `CPM = (y_ij / L) * 1e6`, where
`L` is the library size — the number of reads assigned to transcripts in that library.
A CPM column therefore sums to `1e6` only when the layer covers *every* quantified
transcript; otherwise it sums to less, by exactly the CPM of the rows the layer does
not contain. With the default `--min_expressed_samples=1` the transcript filter is a
no-op, so both column sums equal `1e6`. With `--min_expressed_samples >= 2` the
transcript layer is an intentional subset and its column sum is legitimately smaller
than the gene layer's.

`aidrs.matrix.denominators.tsv` records this explicitly, one row per library:

| Column | Meaning |
|---|---|
| `sample` | library name |
| `n_transcripts` | transcripts with a non-zero count in this library |
| `n_genes` | genes with a non-zero count in this library |
| `total_reads_L` | the CPM denominator `L` of this library |
| `transcript_cpm_col_sum` | sum of the delivered transcript CPM column |
| `gene_cpm_col_sum` | sum of the delivered gene CPM column |

### Reads that are discarded

AIDRS count and CPM matrices retain reads that fully match the final transcript model. If you need to retain all FLNC reads, including low-coverage reads that do not match a model, or want probabilistic assignment of ambiguous reads, use [NanoCount](https://github.com/a-slide/NanoCount) as a complementary quantifier. 

NanoCount uses an expectation-maximization (EM) algorithm to distribute multi-mapping reads probabilistically across compatible isoforms. Unlike AIDRS's default counting, NanoCount requires alignments to a **transcriptome reference** (not the genome), and reads must be aligned with `minimap2` retaining secondary alignments (`-N 10`).

AIDRS's `aidrs.transcript_model.gtf`, `aidrs.transcript_model.fasta` can feed this NanoCount workflow alongside the original BAM and reads FASTA. NanoCount is a complement to AIDRS's default counting and filtering, not a replacement for them.
