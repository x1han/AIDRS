#!/usr/bin/env python3
"""AIDRS-native structural classifier: FSM / ISM / NIC / NNC / mono-exon.

Lightweight junction-chain classifier that uses only the AIDRS SSC
(splice-site chain) and the GENCODE v47 exon table. No external
SQANTI3 is required. Built as a lookup-dict-based O(N+M) implementation
suitable for full-chromosome factorial runs in well under two minutes.

Algorithm
---------

1. Parse the GENCODE GTF, group exon rows by ``transcript_id`` (optionally
   restricted to one chromosome), and convert each transcript into a
   sorted list of genomic splice-site positions (the same representation
   AIDRS stores in its ``SSC`` column). Three lookup structures are
   built in a single pass:

   - ``by_junctions[(Chr, Strand, frozenset)] -> ENST`` for O(1) FSM hit
   - ``by_chr_strand[(Chr, Strand)] -> [(sorted_junction_list, ENST)]``
     for subset (ISM/NIC) checks
   - ``known_sites[(Chr, Strand)] -> set[int]`` for O(1) NNC detection

2. For each AIDRS transcript, parse the SSC chain (dash-separated
   integer positions, e.g. ``"100-250-400"``). If the chain is empty
   or one of the sentinels (``"NA"`` / ``"none"``), classify as
   ``mono-exon`` and emit empty ``associated_transcript``. Otherwise:

   a. ``FSM``   -- ``frozenset(SSC)`` matches a GENCODE transcript's
                  junction set exactly. Record the matching ENST.
   b. ``NNC``   -- at least one AIDRS splice site is absent from the
                  GENCODE site set for that (Chr, Strand).
   c. ``NIC``   -- AIDRS is a strict subset of some GENCODE transcript
                  whose first and last splice sites coincide with the
                  AIDRS chain (5' and 3' splice sites match but
                  GENCODE has additional internal junctions).
   d. ``ISM``   -- AIDRS is a strict subset of some GENCODE transcript
                  but the 5'/3' splice-site boundary test does not
                  match (incomplete splice match).

3. The output TSV is the input TSV plus two appended columns:
   ``structural_category`` and ``associated_transcript``. Row order is
   preserved (no row drops during classification).

Exit codes
----------

* ``0``  success
* ``1``  invocation error (bad args, missing files)
* ``2``  input TSV missing required columns
* ``3``  no GENCODE overlap with any AIDRS transcript on the chosen
         chromosome (data gap)

Usage
-----

    python tools/aidrs_native_classifier.py \\
        --input  /path/to/aidrs.transcript.assessment.tsv \\
        --gtf    /path/to/gencode.v47.primary_assembly.annotation.gtf \\
        --chromosome chr1 \\
        --output /path/to/aidrs.transcript.assessment.classified.tsv
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Standalone-invocation PYTHONPATH bootstrap (consistent with diff_sha.py,
# extract_scientific_sha.py, validate_gencode_cds_match.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# GTF parsing
# ---------------------------------------------------------------------------

_TX_ID_RE = re.compile(r'transcript_id "([^"]+)"')

# Single-exon sentinels. AIDRS writes 'NA' from bam2ssc and 'none' from
# the P3 single-exon routing stash in src/aidrs.py:91. Both are accepted.
_MONO_EXON_SENTINELS = frozenset({"NA", "none", "", "nan", "NaN"})


def _parse_gtf_exons(
    gencode_gtf: str,
    chromosome: Optional[str] = None,
) -> Tuple[
    Dict[Tuple[str, str, frozenset], str],
    Dict[Tuple[str, str], List[Tuple[List[int], str]]],
    Dict[Tuple[str, str], Set[int]],
]:
    """Parse a GENCODE GTF and return three lookup structures.

    Returns
    -------
    by_junctions : dict
        ``(Chr, Strand, frozenset_of_splice_sites) -> ENST`` for O(1)
        FSM hit lookup. When multiple GENCODE transcripts share the
        same junction set, the first one encountered wins.
    by_chr_strand : dict
        ``(Chr, Strand) -> list of (sorted_sites_list, ENST)`` for
        subset (NIC/ISM) scanning. Sorted descending by junction
        count so longer (more specific) references are tried first.
    known_sites : dict
        ``(Chr, Strand) -> set of all splice-site positions`` for
        O(1) NNC detection.
    """
    by_junctions: Dict[Tuple[str, str, frozenset], str] = {}
    by_chr_strand: Dict[Tuple[str, str], List[Tuple[List[int], str]]] = defaultdict(list)
    known_sites: Dict[Tuple[str, str], Set[int]] = defaultdict(set)

    # Group exon rows by transcript_id. ``exons`` is a dict-of-lists; we
    # only commit a transcript once we see a non-exon row or EOF.
    cur_tx: Optional[str] = None
    cur_chr: Optional[str] = None
    cur_strand: Optional[str] = None
    cur_starts: List[int] = []
    cur_ends: List[int] = []
    n_exon_rows = 0
    n_transcripts = 0
    n_skipped_no_exon = 0

    def _flush() -> None:
        nonlocal cur_tx, cur_chr, cur_strand, cur_starts, cur_ends, n_transcripts, n_skipped_no_exon
        if cur_tx is None:
            return
        if cur_chr is None or cur_strand is None or not cur_starts:
            n_skipped_no_exon += 1
        else:
            # Sort exons by start coordinate in genomic order. Exons
            # within a transcript are not always guaranteed sorted in
            # GENCODE primary_assembly output, so we defensively sort.
            sorted_pairs = sorted(zip(cur_starts, cur_ends))
            starts_sorted = [p[0] for p in sorted_pairs]
            ends_sorted = [p[1] for p in sorted_pairs]
            # Intron boundaries: (exon_i.end, exon_{i+1}.start) for
            # adjacent exons. The SSC chain is the flat list of all
            # positions in genomic order. For a single-exon transcript
            # there are no introns, so we skip it from the lookup (the
            # AIDRS-side classifier handles mono-exon explicitly).
            if len(starts_sorted) < 2:
                n_skipped_no_exon += 1
            else:
                sites: List[int] = []
                for end_i, start_next in zip(ends_sorted[:-1], starts_sorted[1:]):
                    sites.append(end_i)
                    sites.append(start_next)
                fset = frozenset(sites)
                key = (cur_chr, cur_strand, fset)
                # First-encountered ENST wins; downstream duplicates of
                # the same junction set are intentionally collapsed.
                if key not in by_junctions:
                    by_junctions[key] = cur_tx
                    by_chr_strand[(cur_chr, cur_strand)].append((sites, cur_tx))
                known_sites[(cur_chr, cur_strand)].update(sites)
                n_transcripts += 1
        cur_tx = None
        cur_chr = None
        cur_strand = None
        cur_starts = []
        cur_ends = []

    with open(gencode_gtf, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            # Find the next exon row's tx id (or any non-exon line which
            # implicitly flushes the previous transcript).
            if not line or line == "\n":
                _flush()
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                _flush()
                continue
            if cols[2] != "exon":
                # Non-exon rows (gene/transcript/CDS/UTR/...) implicitly
                # close the previous transcript. We still extract tx id
                # for symmetry but do not start a new one.
                _flush()
                # Some GTF rows (e.g. transcript) carry their own
                # transcript_id which we deliberately ignore here:
                # the canonical SSC chain is built from exon rows.
                continue
            if chromosome and cols[0] != chromosome:
                _flush()
                continue
            tx_match = _TX_ID_RE.search(cols[8])
            if not tx_match:
                _flush()
                continue
            tx_id = tx_match.group(1)
            # Same transcript id continuing across blocks -- keep
            # accumulating. Different id -- flush and start fresh.
            if cur_tx is not None and tx_id != cur_tx:
                _flush()
            if cur_tx is None:
                cur_tx = tx_id
                cur_chr = cols[0]
                cur_strand = cols[6]
                cur_starts = []
                cur_ends = []
            cur_starts.append(int(cols[3]))
            cur_ends.append(int(cols[4]))
            n_exon_rows += 1
        _flush()

    # Sort each (Chr, Strand) bucket by descending junction count so
    # that longer (more specific) GENCODE references are tried first
    # during subset scanning. This is a cheap heuristic; it does not
    # change correctness, only reduces the constant factor.
    for key in by_chr_strand:
        by_chr_strand[key].sort(key=lambda item: len(item[0]), reverse=True)

    print(
        f"[GENCODE] scanned {n_exon_rows:,} exon rows; "
        f"{n_transcripts:,} multi-exon transcripts; "
        f"{len(by_junctions):,} unique junction-sets; "
        f"{n_skipped_no_exon:,} skipped (no/multiple-exon-only)",
        file=sys.stderr,
    )
    if chromosome:
        print(f"[GENCODE] restricted to chromosome={chromosome}", file=sys.stderr)
    return by_junctions, by_chr_strand, known_sites


# ---------------------------------------------------------------------------
# AIDRS SSC parsing
# ---------------------------------------------------------------------------


def _parse_ssc(ssc: object) -> Optional[List[int]]:
    """Parse an AIDRS SSC cell into a list of integer splice sites.

    Returns
    -------
    None
        Single-exon transcript (sentinel or empty chain).
    list[int]
        Sorted-ascending list of genomic splice-site positions.
    """
    if ssc is None:
        return None
    if isinstance(ssc, float) and pd.isna(ssc):
        return None
    s = str(ssc).strip()
    if s in _MONO_EXON_SENTINELS:
        return None
    if not s:
        return None
    parts = s.split("-")
    if len(parts) < 2:
        # A single position cannot represent an intron boundary pair
        # and therefore cannot be a multi-exon transcript.
        return None
    try:
        sites = [int(p) for p in parts]
    except ValueError:
        return None
    if len(sites) < 2:
        return None
    return sites


def _classify_one(
    chr_: str,
    strand: str,
    sites: List[int],
    by_junctions: Dict[Tuple[str, str, frozenset], str],
    by_chr_strand: Dict[Tuple[str, str], List[Tuple[List[int], str]]],
    known_sites: Dict[Tuple[str, str], Set[int]],
) -> Tuple[str, str]:
    """Classify a single AIDRS transcript.

    Returns ``(category, associated_transcript_or_empty)``.
    """
    fset = frozenset(sites)
    cs_key = (chr_, strand)

    # 1. FSM -- exact junction-set match.
    fsm_key = (chr_, strand, fset)
    if fsm_key in by_junctions:
        return "FSM", by_junctions[fsm_key]

    # 2. NNC -- at least one novel splice site (not in any GENCODE
    #    transcript on the same (Chr, Strand)).
    chr_sites = known_sites.get(cs_key)
    if chr_sites is None or not fset.issubset(chr_sites):
        return "NNC", ""

    # 3. NIC / ISM -- all splice sites are catalogued; check subset
    #    against each GENCODE reference on the same (Chr, Strand).
    first, last = sites[0], sites[-1]
    bucket = by_chr_strand.get(cs_key, ())
    n_aidrs = len(fset)
    for ref_sites, ref_enst in bucket:
        ref_set = frozenset(ref_sites)
        # Strict subset: AIDRS has fewer junctions than the reference.
        if not fset.issubset(ref_set):
            continue
        if len(ref_set) <= n_aidrs:
            # Equal-length subsets are FSM (handled above) and longer
            # references fail the strict-subset test; skip.
            continue
        # NIC: 5' and 3' splice sites match the reference.
        if first == ref_sites[0] and last == ref_sites[-1]:
            return "NIC", ""
        # ISM: subset but boundaries differ.
        return "ISM", ""

    # All splice sites are known but no GENCODE reference is a strict
    # superset. By construction this is a novel combination of known
    # splice sites -- classify as NIC.
    return "NIC", ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


REQUIRED_INPUT_COLS = (
    "Chr", "Strand", "SSC", "TrID",
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "AIDRS-native junction-chain classifier (FSM/ISM/NIC/NNC/mono-exon). "
            "No external SQANTI3 required."
        ),
    )
    ap.add_argument(
        "--input",
        required=True,
        help="Path to aidrs.transcript.assessment.tsv (19-col core schema).",
    )
    ap.add_argument(
        "--gtf",
        required=True,
        help="Path to GENCODE primary_assembly.annotation.gtf.",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Path to write the classified TSV (input + 2 new columns).",
    )
    ap.add_argument(
        "--chromosome",
        default=None,
        help=(
            "Restrict GENCODE parsing to one chromosome (e.g. 'chr1'). "
            "Strongly recommended for factorial runs."
        ),
    )
    args = ap.parse_args()

    for path in (args.input, args.gtf):
        if not os.path.isfile(path):
            sys.stderr.write(f"[FATAL] file not found: {path}\n")
            return 1

    t0 = time.time()
    by_junctions, by_chr_strand, known_sites = _parse_gtf_exons(
        args.gtf, chromosome=args.chromosome
    )
    t_parse = time.time() - t0
    if not by_junctions:
        sys.stderr.write(
            "[FATAL] GENCODE GTF produced zero multi-exon transcripts. "
            "Check the file path and --chromosome filter.\n"
        )
        return 3
    print(
        f"[GENCODE] parse: {t_parse:.2f}s "
        f"({len(by_junctions):,} unique junction-sets)",
        file=sys.stderr,
    )

    df = pd.read_csv(args.input, sep="\t")
    n_in = len(df)
    print(f"[AIDRS] loaded {n_in:,} rows from {args.input}", file=sys.stderr)
    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        sys.stderr.write(
            f"[FATAL] input TSV is missing required column(s): {missing}. "
            f"Expected 19-col core schema from src/aidrs_runtime/column_registry.py.\n"
        )
        return 2

    categories: List[str] = []
    associated: List[str] = []
    n_mono = 0
    n_multi = 0
    n_fsm = 0
    n_nic = 0
    n_ism = 0
    n_nnc = 0
    t_cls0 = time.time()
    for _, row in df.iterrows():
        chr_ = str(row["Chr"])
        strand = str(row["Strand"])
        sites = _parse_ssc(row["SSC"])
        if sites is None:
            categories.append("mono-exon")
            associated.append("")
            n_mono += 1
            continue
        n_multi += 1
        cat, enst = _classify_one(chr_, strand, sites, by_junctions, by_chr_strand, known_sites)
        categories.append(cat)
        associated.append(enst)
        if cat == "FSM":
            n_fsm += 1
        elif cat == "NIC":
            n_nic += 1
        elif cat == "ISM":
            n_ism += 1
        elif cat == "NNC":
            n_nnc += 1
    t_cls = time.time() - t_cls0

    df["structural_category"] = categories
    df["associated_transcript"] = associated
    n_out = len(df)
    if n_out != n_in:
        sys.stderr.write(
            f"[FATAL] row-count drift: input={n_in:,}, output={n_out:,}. "
            f"Classifier must be 1:1.\n"
        )
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, sep="\t", index=False)

    total = time.time() - t0
    print(
        f"[AIDRS] classification: {t_cls:.2f}s for {n_in:,} rows "
        f"({(n_in / t_cls) if t_cls else float('inf'):,.0f} rows/s)",
        file=sys.stderr,
    )
    print(
        f"[AIDRS] wrote {n_out:,} rows -> {args.output} "
        f"(parse {t_parse:.2f}s + classify {t_cls:.2f}s = {total:.2f}s)",
        file=sys.stderr,
    )
    print(
        f"[AIDRS] category counts: "
        f"FSM={n_fsm:,} ISM={n_ism:,} NIC={n_nic:,} NNC={n_nnc:,} "
        f"mono-exon={n_mono:,} (multi={n_multi:,})",
        file=sys.stderr,
    )

    # Data-gap guard: if zero AIDRS transcripts mapped to any GENCODE
    # junction-set AND zero have any catalogue site, the chromosome
    # filter is wrong.
    if n_fsm == 0 and n_nic == 0 and n_ism == 0 and n_nnc == 0:
        sys.stderr.write(
            "[FATAL] zero AIDRS transcripts produced any structural_category. "
            "Likely a chromosome-name or chromosome-filter mismatch between the "
            "AIDRS input and the GENCODE GTF.\n"
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
