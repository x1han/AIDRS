"""BAM vs FASTA chromosome naming compatibility check (Fail-Fast).

Production-grade: focuses on canonical chromosomes (1-22, X, Y, M/MT)
to avoid false positives from the 3000+ decoy/patch/alt contigs typically
present in GDC/ENCODE/1000G BAM @SQ headers.

Algorithm (canonical-anchored Jaccard):
1. Detect BAM and FASTA dominant chromosome-naming style (chr-prefixed vs bare)
2. If styles differ -> systemic mismatch -> ValueError (Fail-Fast)
3. If styles match -> check coverage on canonical chromosomes only:
   - < 50% matched  -> ValueError (Fail-Fast)
   - 50%-90%        -> warning (partial mismatch)
   - >= 90%         -> info log (PASS)

Why canonical-only: GDC TCGA BAM @SQ can contain 3000+ contigs (decoys, patches,
alt loci) while FASTA is the primary assembly (~25 chrs). Dividing by len(bam_chrs)
explodes the denominator and would false-abort valid data.
"""
import logging
import pysam

logger = logging.getLogger(__name__)

# Canonical chromosome cores for human/mouse (the dominant AIDRS use cases).
# Hardcoded to primary assembly — not a species-aware design.
_CANONICAL_CORES = [str(i) for i in range(1, 23)] + ["X", "Y", "M", "MT"]


def _dominant_style(chrs: set) -> bool:
    """Return True if 'chr'-prefixed style is dominant, False if bare-number."""
    canonical_with_chr = {f"chr{c}" for c in _CANONICAL_CORES}
    canonical_without_chr = set(_CANONICAL_CORES)
    n_with = len(chrs & canonical_with_chr)
    n_without = len(chrs & canonical_without_chr)
    return n_with > n_without


def check_chromosome_naming_compatibility(
    bam_paths: list,
    fasta_path: str,
) -> None:
    """Validate BAM @SQ chromosomes against FASTA references.

    Detects systemic chr1 vs 1 mismatches; raises ValueError on failure.

    Args:
        bam_paths: list of BAM file paths (use [] to skip BAM-side check)
        fasta_path: path to reference FASTA

    Raises:
        ValueError: on systemic chromosome naming mismatch
    """
    if not fasta_path:
        return  # no FASTA provided, skip

    # FASTA chromosomes
    fasta = pysam.FastaFile(fasta_path)
    fasta_chrs = set(fasta.references)
    fasta.close()
    fasta_style = _dominant_style(fasta_chrs)

    # BAM chromosomes (union across all BAMs)
    bam_chrs_union = set()
    for bam_path in bam_paths or []:
        bam = pysam.AlignmentFile(bam_path, "rb")
        bam_chrs_union |= set(bam.references)
        bam.close()
    bam_style = _dominant_style(bam_chrs_union) if bam_chrs_union else fasta_style

    # Step 1: detect systemic style mismatch
    if bam_chrs_union and bam_style != fasta_style:
        bam_str = "'chr'-prefixed" if bam_style else "bare-number"
        fasta_str = "'chr'-prefixed" if fasta_style else "bare-number"
        raise ValueError(
            f"[FATAL INGESTION MISMATCH] Systemic chromosome naming conflict between "
            f"BAM and FASTA! BAM uses {bam_str} (e.g. {sorted(bam_chrs_union)[:3]}), "
            f"while FASTA uses {fasta_str} (e.g. {sorted(fasta_chrs)[:3]}). "
            f"Please reconcile chromosome headers before running AIDRS."
        )

    # Step 2: coverage check on canonical chromosomes (decoy-noise-free)
    canonical_with_chr = {f"chr{c}" for c in _CANONICAL_CORES}
    canonical_without_chr = set(_CANONICAL_CORES)
    active_canonical = canonical_with_chr if bam_style else canonical_without_chr
    expected_in_bam = bam_chrs_union & active_canonical
    if expected_in_bam:
        matched_ratio = len(expected_in_bam & fasta_chrs) / len(expected_in_bam)
        if matched_ratio < 0.5:
            missing = sorted(expected_in_bam - fasta_chrs)[:5]
            raise ValueError(
                f"[FATAL INGESTION MISMATCH] Only {matched_ratio:.1%} of canonical "
                f"BAM chromosomes exist in reference FASTA. Missing: {missing}... "
                f"Check species or genome build mismatch."
            )
        if matched_ratio < 0.9:
            missing = sorted(expected_in_bam - fasta_chrs)
            logger.warning(
                f"[DATA INTEGRITY] Only {matched_ratio:.1%} canonical BAM chromosomes "
                f"matched FASTA. Missing: {missing}"
            )

    logger.info(
        f"[chrom_check] PASS: {len(bam_chrs_union)} BAM chrs vs {len(fasta_chrs)} "
        f"FASTA chrs, style={'chr-prefixed' if fasta_style else 'bare-number'}"
    )
