"""Stage 2.7: Reverse-Transcriptase (RT) Switching artifact detection.

Detects junctions that are likely RT-switching artifacts based on the canonical
SQANTI3 / Cocquet et al. signature: a non-canonical splice motif flanked by an
exact direct repeat of at least RTSWITCHING_MICROHOMOLOGY_MIN_BP bp between
donor and acceptor intronic flanks.

Algorithm (per junction in a multi-exon transcript):
    1. If junction motif is canonical (GT-AG, GC-AG, AT-AC), score=0, flag=False.
    2. Else compute the donor-side intron flank (intron just upstream of acceptor
       site) and acceptor-side exon flank (exon just downstream of acceptor site)
       for the longest exact direct repeat k such that k >= 4 bp.
    3. If k >= 4, score=k, flag=True. Otherwise score=0, flag=False.

Single-exon transcripts (SSC == "NA") skip the microhomology check entirely
(score=0, flag=False).

Graceful degradation: when ``genome_fasta`` is None, we cannot compute
microhomology so we log INFO and return the input dataframe with
score=0, flag=False for every row.

Contract:
    - Adds exactly 2 new columns: ``rt_switching_score`` (int) and
      ``rt_switching_flag`` (bool).
    - Never mutates pre-existing columns -> byte-identity safe on the
      17-col scientific baseline.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger("AIDRS")

# Expert-cited threshold. NOTE: this matches the SQANTI3 literature default for
# the RT-switching artifact filter. Has NOT yet been verified vs SQANTI3's exact
# implementation in this codebase. TODO(verify): compare against SQANTI3
# rt_switching.py microhomology window logic before publishing.
RTSWITCHING_MICROHOMOLOGY_MIN_BP = 4

# Canonical splice motifs (GT-AG, GC-AG, AT-AC). Source: Burset, Seledtsov,
# Solovyev (2000) and the SQANTI3 reference implementation.
CANONICAL_JUNCTION_MOTIFS = frozenset({"GT-AG", "GC-AG", "AT-AC"})


def _extract_junctions(row: pd.Series) -> list[int]:
    """Return the ordered list of splice-site positions for a row.

    Returns an empty list for single-exon transcripts (SSC == "NA") or rows
    where the SSC string cannot be parsed. Callers should treat an empty
    return as "skip microhomology".
    """
    ssc = row.get("SSC", "NA")
    if ssc is None or (isinstance(ssc, float) and pd.isna(ssc)):
        return []
    ssc_str = str(ssc).strip()
    if ssc_str == "" or ssc_str.upper() == "NA":
        return []
    try:
        sites = [int(tok) for tok in ssc_str.split("-") if tok.strip() != ""]
    except (ValueError, AttributeError):
        return []
    return sites


def _junctions_from_sites(sites: list[int]) -> list[tuple[int, int, int, int]]:
    """Convert an ordered list of splice sites to (start, end, start, end) tuples
    representing each (exon_start, exon_end, intron_start, intron_end) junction.

    For a transcript with N>=2 splice sites, there are N-1 junctions. Each
    junction between consecutive sites i and i+1 produces:
        - exon_end   = sites[i]        (last base of upstream exon)
        - exon_start = sites[i+1]      (first base of downstream exon)
        - intron_start = sites[i] + 1  (first base of intron after exon_end)
        - intron_end = sites[i+1] - 1  (last base of intron before exon_start)
    """
    out: list[tuple[int, int, int, int]] = []
    for i in range(len(sites) - 1):
        exon_end = sites[i]
        exon_start = sites[i + 1]
        intron_start = exon_end + 1
        intron_end = exon_start - 1
        out.append((exon_end, exon_start, intron_start, intron_end))
    return out


_COMPLEMENT_TABLE = str.maketrans("ACGTN", "TGCAN")


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence.

    Uses a precomputed translation table (no per-call dict allocation). The
    motif strings we pass are always exactly 2bp or <=20bp, so this is cheap.
    """
    return seq.upper().translate(_COMPLEMENT_TABLE)[::-1]


def _junction_motif(
    genome_fasta,
    chrom: str,
    strand: str,
    intron_start: int,
    intron_end: int,
    exon_start: int,
) -> str:
    """Return the canonical 2+2 donor/acceptor motif string.

    Donor  = last 2bp of intron (intron_end-1 .. intron_end)
    Acceptor = first 2bp of downstream exon (exon_start .. exon_start+1)

    On - strand transcripts the donor/acceptor dinucleotides in transcript
    space are reverse-complements of the genomic sequence read left-to-right
    at those positions, so we reverse-complement both flanks when
    ``strand == "-"``.
    """
    donor = genome_fasta.fetch(chrom, intron_end - 1, intron_end + 1).upper()
    acceptor = genome_fasta.fetch(chrom, exon_start, exon_start + 2).upper()
    if strand == "-":
        donor = _reverse_complement(donor)
        acceptor = _reverse_complement(acceptor)
    return f"{donor}-{acceptor}"


def _longest_direct_repeat(
    genome_fasta,
    chrom: str,
    strand: str,
    intron_end: int,
    exon_start: int,
    max_window: int = 20,
) -> int:
    """Return the longest exact direct repeat k (in bp) shared between the
    intron flank immediately upstream of the acceptor site and the exon flank
    immediately downstream of the acceptor site.

    Both flanks are taken in genomic orientation (left -> right) and then
    reverse-complemented together when ``strand == "-"`` so that the repeat
    check operates in transcript space. The intron flank is
    ``intron[end-max_window : end]`` (i.e. the last ``max_window`` bp of the
    intron ending at ``intron_end``). The exon flank is
    ``exon[start : start+max_window]`` (the first ``max_window`` bp of the
    downstream exon).

    A direct repeat of length k means: the kbp suffix of the intron flank
    equals the kbp prefix of the exon flank (in transcript orientation). The
    largest such k is returned, capped at ``max_window``.
    """
    win = max_window
    # Intron tail: positions (intron_end - win) .. intron_end (end-exclusive
    # for pysam.fetch)
    intron_tail_start = max(0, intron_end - win)
    intron_tail = genome_fasta.fetch(chrom, intron_tail_start, intron_end).upper()
    # Exon head: positions exon_start .. exon_start + win
    exon_head = genome_fasta.fetch(chrom, exon_start, exon_start + win).upper()

    if strand == "-":
        intron_tail = _reverse_complement(intron_tail)
        exon_head = _reverse_complement(exon_head)

    best = 0
    for k in range(min(len(intron_tail), len(exon_head), win), 0, -1):
        if intron_tail[-k:] == exon_head[:k]:
            best = k
            break
    return best


def detect_rt_switching(
    df: pd.DataFrame,
    genome_fasta=None,
    microhomology_min_bp: int = RTSWITCHING_MICROHOMOLOGY_MIN_BP,
) -> pd.DataFrame:
    """Detect RT-switching artifact junctions.

    Parameters
    ----------
    df : pd.DataFrame
        Input transcript dataframe. Must contain columns ``Chr``, ``Strand``,
        ``SSC``, and ``TrStart``/``TrEnd`` columns used only to derive strand
        and chromosome for genome lookups.
    genome_fasta : pysam.FastaFile or None
        Reference genome. When ``None``, the function logs INFO and returns df
        with ``rt_switching_score=0`` and ``rt_switching_flag=False`` for every
        row (graceful degradation).
    microhomology_min_bp : int, default 4
        Minimum exact direct repeat (bp) between donor intron flank and
        acceptor exon flank to flag a junction as RT-switching candidate.
        Defaults to ``RTSWITCHING_MICROHOMOLOGY_MIN_BP``.

    Returns
    -------
    pd.DataFrame
        Input df with two new columns appended: ``rt_switching_score`` (int,
        maximum k across all junctions of the transcript; 0 for single-exon
        or canonical-only transcripts) and ``rt_switching_flag`` (bool, True
        iff any junction is non-canonical AND has k >= microhomology_min_bp).
        Pre-existing columns are not modified.
    """
    # Initialize output columns; byte-identity safe because these are new.
    if "rt_switching_score" not in df.columns:
        df = df.copy()
        df["rt_switching_score"] = 0
        df["rt_switching_flag"] = False

    if genome_fasta is None:
        logger.info(
            "RT-switching detection skipped: genome_fasta is None; "
            "returning score=0, flag=False for all rows."
        )
        return df

    score_col = df["rt_switching_score"].astype(int)
    flag_col = df["rt_switching_flag"].astype(bool)

    for idx, row in df.iterrows():
        ssc = row.get("SSC", "NA")
        if ssc is None or (isinstance(ssc, float) and pd.isna(ssc)) or str(ssc).strip().upper() == "NA":
            # Single-exon transcript -> skip microhomology check.
            continue
        chrom = row.get("Chr", None)
        if chrom is None:
            continue
        strand = row.get("Strand", "+")
        sites = _extract_junctions(row)
        if len(sites) < 2:
            continue

        max_score = 0
        flagged = False
        for (_exon_end, exon_start, _intron_start, intron_end) in _junctions_from_sites(sites):
            try:
                motif = _junction_motif(genome_fasta, chrom, strand, _intron_start, intron_end, exon_start)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "RT-switching motif fetch failed for %s sites=%s: %s",
                    chrom, sites, exc,
                )
                continue

            if motif in CANONICAL_JUNCTION_MOTIFS:
                continue  # Canonical -> not an RT-switching artifact.

            # Non-canonical: check microhomology.
            try:
                k = _longest_direct_repeat(genome_fasta, chrom, strand, intron_end, exon_start)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "RT-switching microhomology fetch failed for %s sites=%s: %s",
                    chrom, sites, exc,
                )
                continue

            if k > max_score:
                max_score = k
            if k >= microhomology_min_bp:
                flagged = True

        score_col.at[idx] = max_score
        flag_col.at[idx] = flagged

    df["rt_switching_score"] = score_col.astype(int)
    df["rt_switching_flag"] = flag_col.astype(bool)
    return df
