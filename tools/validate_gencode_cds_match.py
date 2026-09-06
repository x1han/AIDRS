#!/usr/bin/env python3
"""Validate TranslationAI TIS/TTS predictions against GENCODE v47 official CDS.

Algorithm (expert-approved, per workspace/findings/):

1. Sample scope: Case 1 output transcripts marked FSM (Full Splice Match) in a
   `category` column. FSM rows have a definitive GENCODE CDS reference --
   no splice-form inference needed. If the Case 1 TSV has no `category`
   column, the tool fails loudly (the implementer must add a SQANTI3
   classification step upstream).

2. Coordinate extraction: parse GENCODE v47 GTF, group CDS lines by
   transcript_id, and store the smallest CDS start and largest CDS end per
   transcript (1-based, half-open inclusive coordinates, matching GTF
   semantics). Strand is taken from column 6 of the GTF.

3. Mapping: the validator matches AIDRS TrID directly to GENCODE
   transcript_id. If no overlap exists between the two ID sets, the
   validator fails loudly (the TrID space is not GENCODE-compatible).

4. Physical match metrics (TIS_related_location / TTS_related_location are
   taken verbatim from the Case 1 TSV; they are 1-based CDS-relative
   offsets stored as strings -- a value of "no" indicates TranslationAI
   did not predict a CDS, and the row is excluded from TIS/TTS metrics):

   - TIS exact match: |Predicted_TIS_pos - Official_CDS_start| == 0
                     AND same chromosome and same strand
   - TTS exact match: |Predicted_TTS_pos - Official_CDS_end|   == 0
                     AND same chromosome and same strand
   - Both-ends exact: TIS exact AND TTS exact
   - TIS in-frame:   |Predicted_TIS_pos - Official_CDS_start| % 3 == 0

5. Output: stdout report + JSON dump to the path given by --json-out
   (default: tools/.out/validate_gencode_cds_match.json).

6. Pass line: TIS exact match rate >= 70%.

Usage:
    python tools/validate_gencode_cds_match.py \
        --case1-tsv /datf/hanxi/test/AIDRS/benchmark_chr1/case1_transAI_polyA/aidrs.transcript.assessment.tsv \
        --gencode-gtf /datf/hanxi/database/reference/GENCODE/GRCh38.p14/gencode.v47.primary_assembly.annotation.gtf

    # AIDRS-native classifier output (no SQANTI3 dependency):
    python tools/validate_gencode_cds_match.py \
        --case1-tsv /datf/hanxi/test/AIDRS/benchmark_chr1/case1_transAI_polyA/aidrs.transcript.classified.tsv \
        --gencode-gtf /datf/hanxi/database/reference/GENCODE/GRCh38.p14/gencode.v47.primary_assembly.annotation.gtf \
        --category-col structural_category --lookup-col associated_transcript

Exit codes:
    0  PASS (TIS exact match rate >= 70%) or no FSM rows to score
    1  invocation error (bad args, missing files)
    2  FAIL (TIS exact match rate < 70%)
    3  data gap (no FSM rows in Case 1 TSV, OR no TrID-GENCODE overlap)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Optional, Tuple

# Standalone-invocation PYTHONPATH bootstrap (consistent with diff_sha.py,
# extract_scientific_sha.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

import pandas as pd  # noqa: E402


# ---------------------------------------------------------------------------
# GENCODE GTF parsing
# ---------------------------------------------------------------------------

_TX_ID_RE = re.compile(r'transcript_id "([^"]+)"')
_CHR_RE = re.compile(r'^chr([0-9XYM]+)$', re.IGNORECASE)


def _chr_sort_key(chrom: str) -> int:
    """Numeric chromosomes first (1..22), then X=23, Y=24, M=25, others last."""
    m = _CHR_RE.match(chrom)
    if not m:
        return 99
    val = m.group(1).upper()
    table = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    return table.get(val, int(val) if val.isdigit() else 99)


def parse_gencode_cds(gencode_gtf: str, chromosome: Optional[str] = None) -> Dict[
    str, Tuple[str, int, int, str]
]:
    """Parse GENCODE GTF and return {transcript_id: (chr, cds_start, cds_end, strand)}.

    cds_start = smallest CDS start coordinate across the transcript (1-based,
                half-open -- the smallest GTF `start` value of any CDS row).
    cds_end   = largest CDS end coordinate across the transcript (the largest
                GTF `end` value of any CDS row, inclusive).
    strand    = '+' or '-' (GTF column 6).

    If `chromosome` is given (e.g. 'chr1'), only CDS rows on that chromosome
    are loaded -- a substantial speedup for chr1-only validation.
    """
    cds_by_tx: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {"chr": None, "start": 10**12, "end": -1, "strand": "."}
    )
    n_cds_total = 0
    with open(gencode_gtf, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "CDS":
                continue
            if chromosome and cols[0] != chromosome:
                continue
            n_cds_total += 1
            tx_match = _TX_ID_RE.search(cols[8])
            if not tx_match:
                continue
            tx_id = tx_match.group(1)
            start = int(cols[3])
            end = int(cols[4])
            strand = cols[6]
            entry = cds_by_tx[tx_id]
            if start < entry["start"]:
                entry["start"] = start
                entry["chr"] = cols[0]
            if end > entry["end"]:
                entry["end"] = end
            entry["strand"] = strand

    out: Dict[str, Tuple[str, int, int, str]] = {}
    for tx_id, info in cds_by_tx.items():
        if info["chr"] is None or info["end"] < 0:
            continue
        out[tx_id] = (info["chr"], int(info["start"]), int(info["end"]), info["strand"])
    print(
        f"[GENCODE] scanned {n_cds_total:,} CDS rows; {len(out):,} transcripts with CDS",
        file=sys.stderr,
    )
    return out


# ---------------------------------------------------------------------------
# Case 1 TSV loading + FSM filter
# ---------------------------------------------------------------------------

def load_fsm_rows(
    case1_tsv: str,
    category_col: str = "category",
) -> pd.DataFrame:
    """Load Case 1 TSV and return FSM rows. Fails loudly if no category col
    or no FSM rows exist.

    `category_col` accepts either the SQANTI3-style ``category`` (default) or
    the AIDRS-native ``structural_category`` produced by
    tools/aidrs_native_classifier.py.
    """
    df = pd.read_csv(case1_tsv, sep="\t")
    print(f"[Case1] loaded {len(df):,} rows from {case1_tsv}", file=sys.stderr)
    print(f"[Case1] columns: {list(df.columns)}", file=sys.stderr)
    if category_col not in df.columns:
        sys.stderr.write(
            f"[FATAL] Case 1 TSV has no '{category_col}' column. FSM filtering\n"
            f"        requires a classification column. Pass --category-col to\n"
            f"        specify 'category' (SQANTI3) or 'structural_category'\n"
            f"        (AIDRS-native).\n"
        )
        sys.exit(3)
    fsm = df[df[category_col] == "FSM"].copy()
    print(f"[Case1] FSM rows: {len(fsm):,}", file=sys.stderr)
    if len(fsm) == 0:
        sys.stderr.write(
            f"[FATAL] Case 1 TSV has zero FSM rows in '{category_col}'. Either\n"
            f"        the file was not classified, or every transcript is\n"
            f"        NIC/NNC/novel. Cannot validate TranslationAI against an\n"
            f"        empty reference set.\n"
        )
        sys.exit(3)
    return fsm


# ---------------------------------------------------------------------------
# Coordinate conversion + scoring
# ---------------------------------------------------------------------------

def _parse_tis_tts(value) -> Optional[int]:
    """Translate a TIS_related_location / TTS_related_location cell to an int.

    The Case 1 TSV stores these as CDS-relative offsets (1-based) drawn from
    the Codingblock predictor: an integer like "139", or the literal "no"
    when TranslationAI found no CDS. Any non-integer cell (NaN, "NA", etc.)
    returns None and is excluded from TIS/TTS metrics.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"no", "na", "nan", "none", "null"}:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def predict_genomic_tis_tts(row, cds_start: int, cds_end: int, strand: str) -> Tuple[
    Optional[int], Optional[int]
]:
    """Convert a row's CDS-relative TIS/TTS offsets into genomic coordinates.

    Case 1 TSV TIS_related_location / TTS_related_location are CDS-relative
    (1 = first codon of CDS). For a '+' strand transcript:
        TIS_geno = cds_start + (offset - 1)
        TTS_geno = cds_start + (offset - 1)  (also CDS-relative)
    For '-' strand, offsets are still 1-based from the 5' end of the CDS
    in transcript orientation -- which means in genomic coordinates they
    decrease as offset increases. We invert:
        TIS_geno = cds_end   - (offset - 1)
        TTS_geno = cds_end   - (offset - 1)
    """
    tis_off = _parse_tis_tts(row.get("TIS_related_location"))
    tts_off = _parse_tis_tts(row.get("TTS_related_location"))
    if strand == "+":
        tis_geno = cds_start + (tis_off - 1) if tis_off is not None else None
        tts_geno = cds_start + (tts_off - 1) if tts_off is not None else None
    else:  # '-' strand
        tis_geno = cds_end - (tis_off - 1) if tis_off is not None else None
        tts_geno = cds_end - (tts_off - 1) if tts_off is not None else None
    return tis_geno, tts_geno


def score_matches(
    fsm: pd.DataFrame,
    cds_lookup: Dict[str, Tuple[str, int, int, str]],
    lookup_col: str = "TrID",
) -> dict:
    """Compute exact-match and in-frame metrics. Returns a stats dict.

    `lookup_col` is the column whose value is matched against the GENCODE
    CDS lookup keys (ENST*). Default ``TrID`` (SQANTI3-style where the
    AIDRS TrID was joined to a reference transcript_id). For AIDRS-native
    classification output, pass ``associated_transcript`` which carries
    the GENCODE ENST matched by tools/aidrs_native_classifier.py.
    """
    n_fsm = len(fsm)
    n_mapped = 0
    n_tis_pred = 0
    n_tts_pred = 0
    n_tis_exact = 0
    n_tts_exact = 0
    n_both_exact = 0
    n_tis_inframe = 0
    delta_tis = []
    delta_tts = []
    mapping_examples = []

    for _, row in fsm.iterrows():
        tr_id = row.get(lookup_col)
        if pd.isna(tr_id):
            continue
        tr_id = str(tr_id)
        if tr_id not in cds_lookup:
            continue
        n_mapped += 1
        if len(mapping_examples) < 5:
            mapping_examples.append(tr_id)
        chr_off, cds_start, cds_end, strand = cds_lookup[tr_id]
        # Chromosome / strand sanity: Case 1 TrIDs and GENCODE IDs only
        # "match" if they coincide on the same chromosome AND strand. The
        # same-chrom check guards against chr-name aliasing (e.g. 'chr1'
        # vs '1') in case the GENCODE reference uses different prefixes.
        if str(row["Chr"]) != chr_off or str(row["Strand"]) != strand:
            continue
        tis_geno, tts_geno = predict_genomic_tis_tts(row, cds_start, cds_end, strand)
        if tis_geno is not None:
            n_tis_pred += 1
            d = abs(tis_geno - cds_start)
            delta_tis.append(d)
            if d == 0:
                n_tis_exact += 1
            if d % 3 == 0:
                n_tis_inframe += 1
        if tts_geno is not None:
            n_tts_pred += 1
            d = abs(tts_geno - cds_end)
            delta_tts.append(d)
            if d == 0:
                n_tts_exact += 1
        if tis_geno is not None and tts_geno is not None:
            if abs(tis_geno - cds_start) == 0 and abs(tts_geno - cds_end) == 0:
                n_both_exact += 1

    def _median(xs):
        if not xs:
            return None
        s = sorted(xs)
        n = len(s)
        return float(s[n // 2]) if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0

    def _rate(num, den):
        return (num / den) if den else None

    return {
        "N_FSM": int(n_fsm),
        "N_MAPPED": int(n_mapped),
        "N_TIS_PRED": int(n_tis_pred),
        "N_TTS_PRED": int(n_tts_pred),
        "TIS_exact": int(n_tis_exact),
        "TTS_exact": int(n_tts_exact),
        "both_exact": int(n_both_exact),
        "TIS_inframe": int(n_tis_inframe),
        "TIS_exact_rate": _rate(n_tis_exact, n_mapped),
        "TTS_exact_rate": _rate(n_tts_exact, n_mapped),
        "both_exact_rate": _rate(n_both_exact, n_mapped),
        "TIS_inframe_rate": _rate(n_tis_inframe, n_mapped),
        "median_abs_delta_TIS_bp": _median(delta_tis),
        "median_abs_delta_TTS_bp": _median(delta_tts),
        "mapping_examples": mapping_examples,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

PASS_THRESHOLD = 0.70


def render_report(stats: dict, case1_tsv: str, gencode_gtf: str) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("TranslationAI vs GENCODE v47 CDS exact-match validation")
    lines.append("=" * 72)
    lines.append(f"Case 1 TSV       : {case1_tsv}")
    lines.append(f"GENCODE GTF      : {gencode_gtf}")
    lines.append("")
    lines.append(f"N_FSM   (FSM rows in Case 1)         : {stats['N_FSM']:,}")
    lines.append(f"N_MAPPED (FSM with GENCODE CDS ref) : {stats['N_MAPPED']:,}")
    lines.append("")
    lines.append(f"TIS predicted (TIS_related_location != 'no') : {stats['N_TIS_PRED']:,}")
    lines.append(f"TTS predicted (TTS_related_location != 'no') : {stats['N_TTS_PRED']:,}")
    lines.append("")
    rate = lambda v: "n/a" if v is None else f"{v * 100:.2f}%"
    med = lambda v: "n/a" if v is None else f"{v:.1f} bp"
    lines.append(f"TIS exact match rate   : {rate(stats['TIS_exact_rate'])}  ({stats['TIS_exact']}/{stats['N_MAPPED']})")
    lines.append(f"TTS exact match rate   : {rate(stats['TTS_exact_rate'])}  ({stats['TTS_exact']}/{stats['N_MAPPED']})")
    lines.append(f"Both-ends exact rate   : {rate(stats['both_exact_rate'])}  ({stats['both_exact']}/{stats['N_MAPPED']})")
    lines.append(f"TIS in-frame rate      : {rate(stats['TIS_inframe_rate'])}  ({stats['TIS_inframe']}/{stats['N_MAPPED']})")
    lines.append("")
    lines.append(f"Median |delta_TIS|     : {med(stats['median_abs_delta_TIS_bp'])}")
    lines.append(f"Median |delta_TTS|     : {med(stats['median_abs_delta_TTS_bp'])}")
    lines.append("")
    if stats["TIS_exact_rate"] is None:
        verdict = "INSUFFICIENT DATA (TIS predictions are all 'no')"
        passed = False
    else:
        passed = stats["TIS_exact_rate"] >= PASS_THRESHOLD
        verdict = f"PASS (TIS exact >= {int(PASS_THRESHOLD * 100)}%)" if passed else f"FAIL (TIS exact < {int(PASS_THRESHOLD * 100)}%)"
    lines.append(f"Threshold : TIS exact match rate >= {int(PASS_THRESHOLD * 100)}%")
    lines.append(f"Verdict   : {verdict}")
    lines.append("=" * 72)
    if stats["mapping_examples"]:
        lines.append("Example TrID -> GENCODE matches:")
        for ex in stats["mapping_examples"]:
            lines.append(f"  {ex}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate TranslationAI TIS/TTS against GENCODE CDS (FSM rows only)."
    )
    ap.add_argument(
        "--case1-tsv",
        required=True,
        help="Case 1 aidrs.transcript.assessment.tsv (must contain a 'category' column).",
    )
    ap.add_argument(
        "--gencode-gtf",
        required=True,
        help="GENCODE primary_assembly.annotation.gtf (used for CDS lookup).",
    )
    ap.add_argument(
        "--chromosome",
        default=None,
        help="Restrict to one chromosome (e.g. 'chr1') for speed.",
    )
    ap.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON dump path. Default: tools/.out/validate_gencode_cds_match.json",
    )
    ap.add_argument(
        "--category-col",
        default="category",
        help=(
            "Column holding the structural classification (used to filter FSM rows). "
            "Default 'category' (SQANTI3). Use 'structural_category' for AIDRS-native "
            "output from tools/aidrs_native_classifier.py."
        ),
    )
    ap.add_argument(
        "--lookup-col",
        default="TrID",
        help=(
            "Column whose value is matched against the GENCODE CDS lookup (ENST*). "
            "Default 'TrID' (SQANTI3-style, where TrID was joined to a reference "
            "transcript_id). Use 'associated_transcript' for AIDRS-native output, "
            "which carries the GENCODE ENST matched by tools/aidrs_native_classifier.py."
        ),
    )
    args = ap.parse_args()

    for p in (args.case1_tsv, args.gencode_gtf):
        if not os.path.isfile(p):
            sys.stderr.write(f"[FATAL] file not found: {p}\n")
            return 1

    cds_lookup = parse_gencode_cds(args.gencode_gtf, chromosome=args.chromosome)
    if not cds_lookup:
        sys.stderr.write("[FATAL] GENCODE GTF produced zero CDS transcripts -- abort.\n")
        return 1

    fsm = load_fsm_rows(args.case1_tsv, category_col=args.category_col)
    stats = score_matches(fsm, cds_lookup, lookup_col=args.lookup_col)

    if stats["N_MAPPED"] == 0:
        sys.stderr.write(
            f"[FATAL] zero FSM rows matched the GENCODE CDS lookup via "
            f"'{args.lookup_col}'. Either (a) pass --lookup-col "
            f"associated_transcript for AIDRS-native output, or (b) join Case 1 "
            f"with a SQANTI3 classification that retains the reference "
            f"transcript_id, or (c) feed in a different reference set whose IDs "
            f"match Case 1.\n"
        )
        return 3

    report = render_report(stats, args.case1_tsv, args.gencode_gtf)
    print(report)

    json_out = args.json_out or os.path.join(
        _HERE, ".out", "validate_gencode_cds_match.json"
    )
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    with open(json_out, "w") as fh:
        json.dump(
            {
                "case1_tsv": args.case1_tsv,
                "gencode_gtf": args.gencode_gtf,
                "chromosome": args.chromosome,
                "pass_threshold": PASS_THRESHOLD,
                "stats": stats,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    print(f"\n[JSON] wrote {json_out}", file=sys.stderr)

    if stats["TIS_exact_rate"] is None:
        return 0  # no TIS predictions; nothing to pass/fail
    return 0 if stats["TIS_exact_rate"] >= PASS_THRESHOLD else 2


if __name__ == "__main__":
    sys.exit(main())
