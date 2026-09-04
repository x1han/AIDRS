"""Stage Boundary Fail-Loud guard for AIDRS pipeline.

Purpose: detect catastrophic row loss at stage boundaries (1.2, 1.4, 1.5, 1.8,
2.5b, 2.6). The 6 guarded stages all *substantively* alter row counts via
filtering. Earlier intermediate stages (1.3, 1.6, 1.7, 2.1, 2.2, 2.3, 2.4, 2.5)
are NOT wrapped because their row loss is bounded by upstream operations and a
false positive there would block legitimate work.

Behavior:
  - Always emits an INFO log of drop_rate (informational).
  - WARN at stage-specific threshold (default 50/80 if stage not in dict).
  - FAIL (sys.exit(1)) only if strict_mode=True AND drop_rate >= fail_th.
  - Hard invariant: if n_before > 0 AND n_after == 0 -> sys.exit(1) (catastrophic).

This function is a NO-OP for the data -- it only emits logs and conditionally
exits. The wrapping pattern in aidrs.py is:

    df = stage_boundary_check("1.x", df_before, df, args.strict_stage_checks,
                             args.allow_zero_rows)

The return value is always df_after unchanged, so output bytes are preserved.
"""
import json
import logging
import os
import sys
import time
import pandas as pd

logger = logging.getLogger("AIDRS")

# (warn_th, fail_th) as drop_rate percentages.
_STAGE_THRESHOLDS = {
    "1.2":  (70.0, 95.0),  # frequency filter
    "1.4":  (15.0, 50.0),  # junction motif
    "1.5":  (30.0, 70.0),  # consensus refinement
    "1.8":  (50.0, 85.0),  # NNC/NIC graph filter
    "2.5b": (50.0, 90.0),  # single-exon 5-pillar funnel
    "2.6":  (60.0, 90.0),  # TSS+polyA correction
}

_DEFAULT_WARN = 50.0
_DEFAULT_FAIL = 80.0


def stage_boundary_check(
    stage_id: str,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    strict_mode: bool = False,
    allow_zero: bool = False,
    output_dir=None,
    metadata: dict | None = None,
) -> pd.DataFrame:
    """Boundary guard for stage transitions.

    Args:
        stage_id: stage label, e.g. "1.2", "1.4", "1.5", "1.8", "2.5b", "2.6".
        df_before: dataframe at stage entry (already filtered to stage scope).
        df_after: dataframe at stage exit (post-filter).
        strict_mode: when True, sys.exit(1) on drop_rate >= fail_th.
        allow_zero: when True, suppress the catastrophic zero-row exit (testing).
        metadata: optional caller-supplied context merged into the JSONL record
            under a "metadata" key. Omitted entirely when None or empty, so
            records from callers that pass nothing are unchanged.

    Returns:
        df_after unchanged -- function is logging-only, never modifies data.
    """
    # Treat None as zero rows (e.g., Stage 2.5b may yield df_single_kept=None).
    n_before = 0 if df_before is None else len(df_before)
    n_after = 0 if df_after is None else len(df_after)

    # Hard zero-row invariant: catastrophic loss -> fail loud.
    if not allow_zero and n_before > 0 and n_after == 0:
        logger.error(
            "[stage_check] CATASTROPHIC zero-row at stage %s: n_before=%d -> "
            "n_after=0. This is a hard failure -- pipeline aborted. "
            "Pass --allow-zero-rows to suppress this exit (testing only).",
            stage_id, n_before,
        )
        sys.exit(1)

    if n_before == 0:
        # No data in, no data out -- silently pass (boundary not meaningful).
        return df_after

    drop_rate = (1.0 - (n_after / n_before)) * 100.0
    warn_th, fail_th = _STAGE_THRESHOLDS.get(
        stage_id, (_DEFAULT_WARN, _DEFAULT_FAIL)
    )

    logger.info(
        "[stage_check] stage=%s n_before=%d n_after=%d drop_rate=%.2f%% "
        "(warn>=%.1f%%, fail>=%.1f%%)",
        stage_id, n_before, n_after, drop_rate, warn_th, fail_th,
    )

    if drop_rate >= warn_th:
        logger.warning(
            "[stage_check] stage=%s drop_rate=%.2f%% exceeds warn threshold %.1f%%",
            stage_id, drop_rate, warn_th,
        )

    if strict_mode and drop_rate >= fail_th:
        logger.error(
            "[stage_check] stage=%s drop_rate=%.2f%% exceeds fail threshold "
            "%.1f%% (strict_mode=True). Aborting.",
            stage_id, drop_rate, fail_th,
        )
        sys.exit(1)

    if output_dir is not None:
        _metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(_metrics_dir, exist_ok=True)
        if drop_rate >= fail_th:
            _level = "FAIL"
        elif drop_rate >= warn_th:
            _level = "WARNING"
        else:
            _level = "INFO"
        _record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": stage_id,
            "input": n_before,
            "output": n_after,
            "drop_rate": round(drop_rate, 2),
            "level": _level,
        }
        if metadata:
            _record["metadata"] = metadata
        with open(os.path.join(_metrics_dir, "stage_drops.jsonl"), "a") as _fh:
            _fh.write(json.dumps(_record) + "\n")

    return df_after