import pandas as pd
import logging
from .common import rev_comp
logger = logging.getLogger(__name__)

def check_genomic_intra_priming(chrom, start, end, strand, genome_fasta):
    try:
        if strand == "+":
            # CAVEAT: use genome_fasta.fetch() (uses .fai index for C-level
            # fseek -> ~5 microseconds), NOT genome_fasta[chrom][a:b] which
            # materialises the entire chromosome (e.g. 250 MB for chr1) as a
            # Python str per call. 5954 State-2 rows × ~300 ms each = 14+ min
            # observed in Case 2 factorial run; with .fetch() the same loop
            # is ~0.03 s (~87 000x speedup, memory delta is zero).
            seq = genome_fasta.fetch(chrom, end, end + 20).upper()
        else:
            # SQANTI3 standard: negative-strand 3-prime downstream is at smaller genomic coords.
            # Use IUPAC rev_comp from common.py (handles soft-masked bases too).
            w_start = max(0, start - 20)
            raw_seq = genome_fasta.fetch(chrom, w_start, start)
            seq = rev_comp(raw_seq.upper())
        if len(seq) < 10:
            return True  # boundary, fail-closed
        has_polyA_run = "AAAAAA" in seq
        is_A_rich = (seq.count("A") / len(seq)) >= 0.60
        return has_polyA_run or is_A_rich
    except Exception as e:
        logger.warning(f"intra-priming check failed for {chrom}:{start}-{end}: {e}")
        return True  # fail-closed

def evaluate_single_exon_isoform(row, has_valid_polya, polyA_thresh, filter_freq, genome_fasta=None):
    # Pillar 1: strict 0bp same-strand overlap (Intergenic or Antisense)
    if not row.get("is_intergenic_or_antisense", False):
        return False, "P1_genic_overlap"
    # Pillar 4: frequency >= max(15, 3*filter_freq)
    if row["frequency"] < max(15, 3 * filter_freq):
        return False, "P4_low_freq"
    # Pillar 5: length >= 200bp
    if row["seq_len"] < 200:
        return False, "P5_short"
    # Pillar 2: Puffin >= 0.1
    puffin_val = float(row["Puffin_TSS_15bp"]) if row["Puffin_TSS_15bp"] != "no" else 0.0
    if puffin_val < 0.1:
        return False, "P2_weak_tss"
    # Pillar 3: PolyA strict (Stage A R1 NaN OR semantics) OR non-A-rich downstream
    if has_valid_polya:
        if pd.isna(row["polyA_frac"]) or float(row["polyA_frac"]) < polyA_thresh or row["polyA_valid_reads"] < 3:
            return False, "P3_polya_fail"
    else:
        if genome_fasta is None:
            return False, "P3_no_genome"  # fail-closed if genome not provided
        if check_genomic_intra_priming(row["Chr"], row["TrStart"], row["TrEnd"], row["Strand"], genome_fasta):
            return False, "P3_intra_priming"
    return True, "PASS"

def apply_5_pillar_funnel(df_single, has_valid_polya, polyA_thresh, filter_freq, genome_fasta=None):
    """Apply 5-pillar funnel and return (kept_df, rejection_counts dict)."""
    rejection_counts = {"P1_genic_overlap": 0, "P4_low_freq": 0, "P5_short": 0, "P2_weak_tss": 0, "P3_polya_fail": 0, "P3_intra_priming": 0, "P3_no_genome": 0}
    keep_mask = []
    for _, row in df_single.iterrows():
        ok, reason = evaluate_single_exon_isoform(row, has_valid_polya, polyA_thresh, filter_freq, genome_fasta)
        keep_mask.append(ok)
        if not ok:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    logger.info(
        "[Single-Exon Funnel] Input Candidates: %d\n"
        "  |- Rejected by Pillar 1 (Genic Overlap): %d\n"
        "  |- Rejected by Pillar 4 (Low Frequency): %d\n"
        "  |- Rejected by Pillar 5 (Short Length): %d\n"
        "  |- Rejected by Pillar 2 (Weak TSS): %d\n"
        "  |- Rejected by Pillar 3 (PolyA/Intra-priming): %d\n"
        "  => Final Retained High-Confidence Single-Exon: %d",
        len(df_single),
        rejection_counts["P1_genic_overlap"],
        rejection_counts["P4_low_freq"],
        rejection_counts["P5_short"],
        rejection_counts["P2_weak_tss"],
        rejection_counts["P3_polya_fail"] + rejection_counts["P3_intra_priming"] + rejection_counts["P3_no_genome"],
        sum(keep_mask),
    )
    return df_single[keep_mask].copy(), rejection_counts