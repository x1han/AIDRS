#!/usr/bin/env python3
"""C107 100k regression gate for AIDRS.

The gate has two accepted forms (the expert review of 2026-11-24,
workspace/docs/aidrs_pt_i0_assembly_stability_expert_review_2026-11-24T1046.md,
sections 1.3-6 and 5.2 (b)):

(a) the assessment sha256 hits the pinned golden exactly, or
(b) the change is *semantically bounded*:
      * the five quantification files are byte-identical to the baseline,
      * the assessment has the same schema and differs in at most
        TrStart / TrEnd / seq_len / TIS_related_location / TTS_related_location
        / TIS_score / TTS_score (the two related-location columns are derived
        from the transcript ends, so they move with them),
      * every 5' change is attributable to a new 3' variant and no model moved
        its 5' end while its 3' end stayed put (key-level criteria -- an
        exon-set comparison is degenerate and silently filters everything
        out, which is why the criteria are coordinate-set based at the key
        level),
      * at most 1 % of the models changed on the 3' side,
      * the score columns differ by at most 1e-5.

Mode (a) covers the assessment only: a regression confined to the
quantification layer would leave the assessment digest untouched. Pass
``--baseline`` so the five quantification files are compared in mode (a) as
well.

The 5' side is chain-aware: for a '+' transcript the 5' end is TrStart and
the 3' end is TrEnd; for a '-' transcript it is the other way round.

Operationalization of criterion (i) ("every new 5' coordinate is the partner
5' end of a new 3' variant"): for each candidate model whose 5' coordinate is
not among the baseline 5' coordinates of its key, the same model must also
carry a 3' coordinate that is not among the baseline 3' coordinates of that
key. This is a row-level operationalization of the key-level attribution
semantics; it compares coordinate *sets* per key and is therefore not
exon-set degenerate.

The gate fails loudly instead of passing vacuously: a frame whose models
cannot be matched, a coordinate column that is not integral, or a score cell
that turned into NaN is a FAIL, never a PASS.

One invariant of the pinned golden, stated per key and not per run: the
``polyA_frac`` column cannot move as long as at most ONE sample contributes
reads to a key. The pooled ratio (sum(polyA_freq)/sum(raw_read_freq)) and the
unweighted median over the qualifying samples coincide in that case, so the
P1 aggregation is a mathematical no-op for the reference run. Adding a second
BAM does not automatically move the column -- it moves it for every key that
the new BAM's reads reach, which is a legitimate re-baseline (record the new
sha and its cause). The reference sample count is read from the
quantification header and printed in the receipt, with a warning when it is
greater than one.

Usage:
    python scripts/verify_c107_golden.py --baseline DIR --candidate DIR
    python scripts/verify_c107_golden.py --candidate DIR --sha-only

Exit code 0 = gate passed, 1 = gate failed (the report says why).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

# The pinned golden of the C107 100k assessment (after the A2'/A2'-b change
# and the activation of the default path's poly(A) gate; 1 015 models).
# The previous golden c289f84d... (1 019 models) is the digest of the
# --disable_polya_filter run: the gate alone is what changed the model set.
GOLDEN_SHA256 = "004fbc40c3ecdf75c346b86250db168661ae053fedd33d5429a51eca1929d1fe"

ASSESSMENT = "aidrs.transcript.assessment.tsv"
# The five quantification files: they must be byte-identical in mode (b).
QUANT_FILES = [
    "aidrs.transcript.counts.tsv",
    "aidrs.transcript.cpm.tsv",
    "aidrs.gene.counts.tsv",
    "aidrs.gene.cpm.tsv",
    "aidrs.gene.model_map.tsv",
]
# The only columns that are allowed to differ in mode (b): the transcript
# ends, the length derived from them, the two protein anchor locations
# (derived from the ends) and the two scores.
ALLOWED_COLUMNS = {
    "TrStart", "TrEnd", "seq_len",
    "TIS_related_location", "TTS_related_location",
    "TIS_score", "TTS_score",
}
SCORE_COLUMNS = ["TIS_score", "TTS_score"]
SCORE_TOLERANCE = 1e-5
MAX_3P_CHANGED_FRACTION = 0.01

# A model is identified by its key plus its id; the id alone is not unique
# across genes.
KEY_COLUMNS = ["Chr", "Strand", "SSC"]
MODEL_COLUMNS = KEY_COLUMNS + ["TrID"]
COORD_COLUMNS = ["TrStart", "TrEnd"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_assessment(path: Path) -> pd.DataFrame:
    """Read the assessment without letting pandas turn 'NA' into NaN.

    The single-exon rows carry SSC == 'NA'; with pandas' default NA handling
    that would become NaN and every key lookup would silently miss (the
    documented 'constant 0/0' degeneracy). Everything is read as text first
    and the numeric columns are converted explicitly afterwards.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    for col in df.columns:
        if col in ALLOWED_COLUMNS or col in SCORE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def five_prime(row) -> int:
    """5' end of a transcript (chain-aware)."""
    return int(row["TrStart"]) if row["Strand"] == "+" else int(row["TrEnd"])


def three_prime(row) -> int:
    """3' end of a transcript (chain-aware)."""
    return int(row["TrEnd"]) if row["Strand"] == "+" else int(row["TrStart"])


def _key_of(row) -> tuple:
    return tuple(row[c] for c in KEY_COLUMNS)


def validate_coordinates(df: pd.DataFrame, label: str) -> list:
    """The coordinate columns must be present, numeric, non-null and integral."""
    failures = []
    for col in COORD_COLUMNS:
        if col not in df.columns:
            failures.append(f"{label}: missing column {col}")
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            failures.append(f"{label}: column {col} has {int(values.isna().sum())} "
                            "non-numeric or empty value(s)")
        elif not (values == values.round()).all():
            failures.append(f"{label}: column {col} has non-integral value(s)")
    for col in KEY_COLUMNS + ["TrID"]:
        if col not in df.columns:
            failures.append(f"{label}: missing column {col}")
    return failures


def check_five_prime_criteria(baseline: pd.DataFrame, candidate: pd.DataFrame):
    """Key-level 5' side criteria (expert review section 1.3-6).

    (i)  every 5' change is attributable to a new 3' variant: a candidate
         model whose 5' coordinate is not one of the baseline 5' coordinates
         of its key must also carry a 3' coordinate that is not one of the
         baseline 3' coordinates of that key, and
    (ii) no model moved its 5' end while its 3' end stayed put (matched by
         key + TrID).

    Returns (violations_i, violations_ii, n_matched).
    """
    base_5p = {}
    base_3p = {}
    base_models = {}
    for _, row in baseline.iterrows():
        key = _key_of(row)
        base_5p.setdefault(key, set()).add(five_prime(row))
        base_3p.setdefault(key, set()).add(three_prime(row))
        base_models[tuple(row[c] for c in MODEL_COLUMNS)] = row

    viol_i = 0
    viol_ii = 0
    n_matched = 0
    for _, row in candidate.iterrows():
        key = _key_of(row)
        p, q = five_prime(row), three_prime(row)
        old_5p = base_5p.get(key, set())
        old_3p = base_3p.get(key, set())
        if p not in old_5p and q in old_3p:
            # (i) a new 5' coordinate that does not come with a new 3'
            # variant of the same key.
            viol_i += 1

        model_key = tuple(row[c] for c in MODEL_COLUMNS)
        if model_key in base_models:
            n_matched += 1
            base_row = base_models[model_key]
            if (three_prime(base_row) == q and five_prime(base_row) != p):
                # (ii) 3' unchanged, 5' moved.
                viol_ii += 1

    return viol_i, viol_ii, n_matched


def count_three_prime_changes(baseline: pd.DataFrame, candidate: pd.DataFrame):
    """Number of matched models whose 3' end moved, and the number matched."""
    base_models = {}
    for _, row in baseline.iterrows():
        base_models[tuple(row[c] for c in MODEL_COLUMNS)] = row
    changed = 0
    matched = 0
    for _, row in candidate.iterrows():
        model_key = tuple(row[c] for c in MODEL_COLUMNS)
        if model_key not in base_models:
            continue
        matched += 1
        if three_prime(base_models[model_key]) != three_prime(row):
            changed += 1
    return changed, matched


def compare_column(base: pd.Series, cand: pd.Series) -> bool:
    """True when the two columns carry the same values.

    Numeric columns are compared numerically (so 10 and 10.0 are equal and a
    repr difference is not mistaken for a content change); everything else is
    compared as text. NaN == NaN counts as equal, NaN vs a value does not.
    """
    b_num = pd.to_numeric(base, errors="coerce")
    c_num = pd.to_numeric(cand, errors="coerce")
    if b_num.isna().equals(c_num.isna()) and b_num.notna().all():
        both_nan = b_num.isna() & c_num.isna()
        return bool(((b_num == c_num) | both_nan).all())
    return bool(base.astype(str).equals(cand.astype(str)))


def compare_scores(base: pd.Series, cand: pd.Series, label: str) -> list:
    """The score columns must agree within SCORE_TOLERANCE; NaN is not a pass."""
    failures = []
    b = pd.to_numeric(base, errors="coerce")
    c = pd.to_numeric(cand, errors="coerce")
    b_nan, c_nan = b.isna(), c.isna()
    if not b_nan.equals(c_nan):
        failures.append(f"{label}: NaN pattern differs between baseline and candidate")
        return failures
    if b_nan.any():
        both = b_nan & c_nan
        delta = (b[~both] - c[~both]).abs()
        max_delta = float(delta.max()) if len(delta) else 0.0
    else:
        delta = (b - c).abs()
        max_delta = float(delta.max()) if len(delta) else 0.0
    if max_delta > SCORE_TOLERANCE:
        failures.append(f"{label} differs by more than {SCORE_TOLERANCE} "
                        f"(max |delta| = {max_delta})")
    return failures


def check_mode_b(baseline_dir: Path, candidate_dir: Path) -> tuple:
    """Returns (failures, receipt_lines)."""
    failures = []
    receipt = []

    failures += compare_quant_files(baseline_dir, candidate_dir)

    base_assess = baseline_dir / ASSESSMENT
    cand_assess = candidate_dir / ASSESSMENT
    if not base_assess.exists() or not cand_assess.exists():
        failures.append(f"missing {ASSESSMENT} in baseline or candidate")
        return failures, receipt

    base = read_assessment(base_assess)
    cand = read_assessment(cand_assess)

    failures += validate_coordinates(base, "baseline")
    failures += validate_coordinates(cand, "candidate")
    if failures:
        return failures, receipt

    if len(base) != len(cand):
        failures.append(f"row count differs between baseline and candidate "
                        f"({len(base)} vs {len(cand)})")
        return failures, receipt

    # --- schema: same columns ------------------------------------------------
    missing = sorted(set(base.columns) - set(cand.columns))
    added = sorted(set(cand.columns) - set(base.columns))
    if missing or added:
        failures.append(f"assessment schema differs (missing={missing}, added={added})")
        return failures, receipt

    # Compare in a canonical order so that a re-ordering of the rows is not
    # mistaken for a content change.
    base = base.sort_values(MODEL_COLUMNS, kind="mergesort").reset_index(drop=True)
    cand = cand.sort_values(MODEL_COLUMNS, kind="mergesort").reset_index(drop=True)

    # --- column-level: only the allowed columns may differ ------------------
    changed_cols = []
    for col in base.columns:
        if col in ALLOWED_COLUMNS:
            continue
        if not compare_column(base[col], cand[col]):
            changed_cols.append(col)
    if changed_cols:
        failures.append(f"assessment differs in columns outside the allowed set: {changed_cols}")

    # --- score columns: bounded difference -----------------------------------
    for col in SCORE_COLUMNS:
        if col in base.columns and col in cand.columns:
            failures += compare_scores(base[col], cand[col], col)

    # --- 5' side: key-level criteria ----------------------------------------
    viol_i, viol_ii, n_matched = check_five_prime_criteria(base, cand)
    n_models = len(base)
    n_unmatched = n_models - n_matched
    if n_matched == 0:
        failures.append("no model could be matched between baseline and candidate "
                        "(the gate would be vacuous)")
    if viol_i:
        failures.append(f"5' criterion (i) violated by {viol_i} model(s): "
                        "a new 5' coordinate that is not the partner of a new 3' variant")
    if viol_ii:
        failures.append(f"5' criterion (ii) violated by {viol_ii} model(s): "
                        "3' end unchanged while the 5' end moved")

    # --- 3' side: at most 1 % of the models may move -------------------------
    changed_3p, matched_3p = count_three_prime_changes(base, cand)
    fraction = changed_3p / n_models if n_models else 0.0
    if fraction > MAX_3P_CHANGED_FRACTION:
        failures.append(
            f"3' side changed for {changed_3p}/{n_models} models "
            f"({fraction * 100:.2f} %) > {MAX_3P_CHANGED_FRACTION * 100:.1f} %"
        )

    receipt.append(f"models: baseline={n_models} candidate={len(cand)} "
                   f"matched={n_matched} unmatched={n_unmatched}")
    receipt.append(f"5' criteria: violations(i)={viol_i} violations(ii)={viol_ii}")
    receipt.append(f"3' side changed: {changed_3p}/{n_models} ({fraction * 100:.3f} %)")

    return failures, receipt


def compare_quant_files(baseline_dir: Path, candidate_dir: Path) -> list:
    """The five quantification files must be byte-identical."""
    failures = []
    for name in QUANT_FILES:
        base_path = baseline_dir / name
        cand_path = candidate_dir / name
        if not base_path.exists():
            failures.append(f"missing baseline file: {name}")
            continue
        if not cand_path.exists():
            failures.append(f"missing candidate file: {name}")
            continue
        if sha256_file(base_path) != sha256_file(cand_path):
            failures.append(f"quantification file differs (must be byte-identical): {name}")
    return failures


# The quantification header carries the id columns first, then one column per
# library.
QUANT_ID_COLUMNS = ["TrID", "GeneID", "GeneName"]


def count_reference_samples(output_dir: Path):
    """Number of libraries in the delivered counts matrix (None if unknown)."""
    counts = output_dir / "aidrs.transcript.counts.tsv"
    if not counts.exists():
        return None
    try:
        with open(counts, "r") as fh:
            header = fh.readline().rstrip("\n").split("\t")
    except OSError:
        return None
    if not header[:len(QUANT_ID_COLUMNS)] == QUANT_ID_COLUMNS:
        # An unexpected header: do not guess, report "unknown".
        return None
    n = len(header) - len(QUANT_ID_COLUMNS)
    return n if n > 0 else None


def print_sample_count(output_dir: Path, label: str = "candidate") -> None:
    """Print the library count and warn when the golden's invariant (at most
    one sample contributing reads per key) no longer holds."""
    n = count_reference_samples(output_dir)
    if n is None:
        print(f"{label} library count: unknown "
              "(aidrs.transcript.counts.tsv not readable)")
        return
    print(f"{label} library count: {n}")
    if n > 1:
        print(f"WARNING: the {label} run carries {n} libraries. The pinned "
              "golden's polyA_frac can only stay put while at most one sample "
              "contributes reads to a key; a second library legitimately moves "
              "polyA_frac for the keys it reaches and forces a re-baseline "
              "(record the new sha and its cause).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C107 100k regression gate for AIDRS")
    ap.add_argument("--baseline", help="directory with the baseline deliverables")
    ap.add_argument("--candidate", required=True, help="directory with the candidate deliverables")
    ap.add_argument("--baseline-sha", help="expected sha256 of the baseline assessment "
                                           "(the gate fails if it differs)")
    ap.add_argument("--baseline-label", default="",
                    help="free-text label for the baseline (e.g. the commit sha it was "
                         "produced from); it is echoed into the receipt")
    ap.add_argument("--sha-only", action="store_true",
                    help="only check the assessment sha256 against the pinned golden")
    args = ap.parse_args(argv)

    candidate_dir = Path(args.candidate)
    cand_assess = candidate_dir / ASSESSMENT
    if not cand_assess.exists():
        print(f"FAIL: {cand_assess} does not exist")
        return 1

    digest = sha256_file(cand_assess)
    print(f"candidate assessment sha256: {digest}")
    print_sample_count(candidate_dir, "candidate")

    baseline_dir = Path(args.baseline) if args.baseline else None
    base_digest = None
    if baseline_dir is not None:
        base_assess = baseline_dir / ASSESSMENT
        if not base_assess.exists():
            print(f"FAIL: {base_assess} does not exist")
            return 1
        base_digest = sha256_file(base_assess)
        label = f" ({args.baseline_label})" if args.baseline_label else ""
        print(f"baseline  assessment sha256: {base_digest}{label}")
        print_sample_count(baseline_dir, "baseline")
        if args.baseline_sha and args.baseline_sha.lower() != base_digest:
            print(f"FAIL: the baseline assessment sha256 is {base_digest}, "
                  f"but {args.baseline_sha.lower()} was expected")
            return 1

    if digest == GOLDEN_SHA256:
        # Mode (a). The assessment is pinned, but the quantification layer is
        # not covered by the assessment digest: a regression confined to the
        # counts/CPM layer would leave the assessment untouched. So when a
        # baseline is supplied, the five quantification files are compared
        # here as well.
        failures = []
        if baseline_dir is not None:
            failures = compare_quant_files(baseline_dir, candidate_dir)
        if failures:
            print(f"FAIL (mode a): assessment sha256 == pinned golden, but the "
                  "quantification layer differs from the baseline")
            for f in failures:
                print(f"  - {f}")
            return 1
        print(f"PASS (mode a): assessment sha256 == pinned golden {GOLDEN_SHA256}"
              + ("; the five quantification files are byte-identical to the baseline"
                 if baseline_dir is not None else ""))
        return 0

    if args.sha_only:
        print(f"FAIL: assessment sha256 {digest} != pinned golden {GOLDEN_SHA256}")
        return 1

    if baseline_dir is None:
        print(f"FAIL: assessment sha256 {digest} != pinned golden {GOLDEN_SHA256} "
              "and no --baseline was given for mode (b)")
        return 1

    failures, receipt = check_mode_b(baseline_dir, candidate_dir)
    for line in receipt:
        print(f"  {line}")
    if failures:
        print(f"FAIL (mode b): assessment sha256 {digest} != pinned golden {GOLDEN_SHA256}")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"PASS (mode b): assessment sha256 {digest} != pinned golden, but the change is bounded:")
    print("  - the five quantification files are byte-identical")
    print("  - the assessment schema is unchanged and only "
          "TrStart / TrEnd / seq_len / TIS_related_location / TTS_related_location / "
          "TIS_score / TTS_score differ")
    print("  - the 5' side satisfies the key-level criteria (i) and (ii)")
    print(f"  - the 3' side changed for at most {MAX_3P_CHANGED_FRACTION * 100:.1f} % of the models")
    print(f"  - the score columns differ by at most {SCORE_TOLERANCE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
