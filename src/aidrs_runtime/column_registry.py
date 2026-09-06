"""Column registry for aidrs.transcript.assessment.tsv.

Centralizes the schema for the final per-transcript assessment table so
new stages can opt-in additional columns without requiring
generate_reports.py to be edited. Future stage columns MUST be added to
OPTIONAL_ASSESSMENT_COLS (or a successor list) to participate in the
assessment TSV -- they will NOT appear otherwise.

P0-C: introduced after Stage 2.7 rt_switching_flag/score were silently
dropped because generate_reports.save_results hardcoded the column
list (see _ASSESS_COLS site in src/generate_reports.py).
"""
from typing import List


# Core 17-column assessment schema (order-sensitive: matches the
# pre-Stage-2.7 layout that produced the byte-identical C107 100k
# baseline SHA a8469106...).
CORE_ASSESSMENT_COLS: List[str] = [
    "Chr",
    "Strand",
    "SSC",
    "TrStart",
    "TrEnd",
    "frequency",
    "Puffin_TSS_15bp",
    "Puffin_TSS_50bp",
    "polyA_frac",
    "TIS_related_location",
    "TTS_related_location",
    "TIS_score",
    "TTS_score",
    "Predict_NMD",
    "truncation",
    "TrID",
    "GeneID",
    "GeneName",
    "seq_len",
]


# Canonical 16-col scientific subset = CORE_ASSESSMENT_COLS - {TrID, GeneID, GeneName}.
# Single source of truth for byte-identity SHA computation across tools
# (diff_sha.py, extract_scientific_sha.py, verify_fullgenome_sha.py,
# compare_mouse_s1_factorial.py). Tools import this constant; they
# must not redefine it locally.
SCIENTIFIC_COLS: List[str] = [
    "Chr", "Strand", "SSC", "TrStart", "TrEnd", "frequency",
    "Puffin_TSS_15bp", "Puffin_TSS_50bp",
    "polyA_frac",
    "TIS_related_location", "TTS_related_location",
    "TIS_score", "TTS_score",
    "Predict_NMD", "truncation",
    "seq_len",
]


# Stage opt-in columns. Order matters: columns are appended to CORE in
# this order when they exist on the dataframe. New stage columns must
# be added here so future stages do not need to touch
# generate_reports.py.
OPTIONAL_ASSESSMENT_COLS: List[str] = [
    # Stage 2.7: RT-switching microhomology detection.
    "rt_switching_score",
    "rt_switching_flag",
    # Future examples (commented out until adopted):
    # "polyA_mode",
    # "category",
    # "junction",
]


def resolve_assessment_columns(df) -> List[str]:
    """Return CORE + OPTIONAL columns that exist on ``df``.

    CORE columns are emitted in their declared order. Each OPTIONAL
    column is appended in declared order only if it exists on ``df``
    and has not already been included. Unknown columns on ``df`` are
    intentionally ignored -- opt-in via OPTIONAL_ASSESSMENT_COLS.
    """
    cols: List[str] = list(CORE_ASSESSMENT_COLS)
    for c in OPTIONAL_ASSESSMENT_COLS:
        if c in df.columns and c not in cols:
            cols.append(c)
    return cols


def validate_canonical_schema(df) -> List[str]:
    """Fail-loud schema validator for the CORE assessment columns.

    This function intentionally raises RuntimeError on any missing
    CORE column rather than auto-padded fabrication. Auto-padding
    missing columns with empty/NA values would silently manufacture
    a downstream-looking TSV whose contents are not backed by any
    upstream stage computation. Because the assessment TSV is the
    final, user-facing artifact and the published C107 100k baseline
    SHA is anchored to its 19 CORE columns, a missing CORE column is
    a stage pipeline bug -- it must be diagnosed and fixed at its
    source, not papered over here.

    Returns the list of CORE_ASSESSMENT_COLS (in declared order)
    when validation succeeds, so callers can chain directly:
        cols = validate_canonical_schema(df)
        df[cols].to_csv(...)
    """
    missing = [c for c in CORE_ASSESSMENT_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(
            "[SCHEMA DRIFT] Missing CORE assessment columns: " + str(missing) +
            ". This indicates a stage pipeline bug. Do NOT auto-pad (would fabricate data)."
        )
    return list(CORE_ASSESSMENT_COLS)