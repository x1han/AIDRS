"""Tests for aidrs_runtime.column_registry.resolve_assessment_columns.

Ensures the registry preserves CORE column order, appends OPTIONAL
columns in declared order when present, dedups duplicates, and ignores
unknown df columns. These are the contract that keeps generate_reports.py
free of stage-specific column lists.
"""
import sys
import os
import importlib.util
import pandas as pd

REPO = "/datf/hanxi/software/AIDRS/repo"
_SRC = os.path.join(REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Load column_registry directly (it is a leaf module with no deps).
_spec = importlib.util.spec_from_file_location(
    "aidrs_runtime.column_registry",
    os.path.join(_SRC, "aidrs_runtime", "column_registry.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
resolve_assessment_columns = _mod.resolve_assessment_columns
CORE_ASSESSMENT_COLS = _mod.CORE_ASSESSMENT_COLS
OPTIONAL_ASSESSMENT_COLS = _mod.OPTIONAL_ASSESSMENT_COLS


def test_empty_df_returns_core_only():
    """Empty df (no columns at all) returns CORE in declared order."""
    df = pd.DataFrame()
    cols = resolve_assessment_columns(df)
    assert cols == CORE_ASSESSMENT_COLS, f"expected CORE only, got {cols}"
    # CORE must be a non-empty list and preserve declaration order
    assert len(cols) >= 1
    assert cols[0] == "Chr"


def test_df_with_unknown_column_returns_just_core():
    """Unknown columns on df are ignored (opt-in via OPTIONAL_ASSESSMENT_COLS)."""
    df = pd.DataFrame(columns=CORE_ASSESSMENT_COLS + ["custom_xyz", "another_custom"])
    cols = resolve_assessment_columns(df)
    assert cols == CORE_ASSESSMENT_COLS
    assert "custom_xyz" not in cols
    assert "another_custom" not in cols


def test_df_with_duplicate_core_column_dedups():
    """Duplicate CORE column names on df dedup to unique list."""
    # Build a df where a CORE column appears twice
    df = pd.DataFrame(
        columns=[
            "Chr",
            "Strand",
            "TrID",  # appears again later in columns
            "GeneID",
            "TrID",  # duplicate
        ]
    )
    cols = resolve_assessment_columns(df)
    # No duplicates
    assert len(cols) == len(set(cols)), f"duplicates not deduped: {cols}"
    # Order stable
    assert cols == list(CORE_ASSESSMENT_COLS)


def test_optional_assessment_cols_is_empty():
    """OPTIONAL_ASSESSMENT_COLS is empty (no opt-in stage columns currently)."""
    assert OPTIONAL_ASSESSMENT_COLS == [], (
        f"expected OPTIONAL_ASSESSMENT_COLS empty, got {OPTIONAL_ASSESSMENT_COLS}"
    )


def test_registry_does_not_mutate_caller_df():
    """resolve_assessment_columns must be side-effect-free on the input df."""
    df = pd.DataFrame(columns=CORE_ASSESSMENT_COLS + ["custom_extra"])
    original_cols = list(df.columns)
    _ = resolve_assessment_columns(df)
    assert list(df.columns) == original_cols


if __name__ == "__main__":
    # Allow running directly: `python tests/test_column_registry.py`
    tests = [
        test_empty_df_returns_core_only,
        test_df_with_unknown_column_returns_just_core,
        test_df_with_duplicate_core_column_dedups,
        test_optional_assessment_cols_is_empty,
        test_registry_does_not_mutate_caller_df,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
