"""Concurrency utilities for AIDRS pipeline.

Provides canonical, fail-loud helpers for collecting ProcessPoolExecutor
futures. Use these instead of raw as_completed() loops — silent-swallow
anti-patterns (where worker exceptions are dropped on the floor) cause
silent chromosome-level data loss that no test can detect without explicit
SHA divergence checks.
"""

import logging
from concurrent.futures import as_completed

logger = logging.getLogger("AIDRS")


def drain_futures_loud(futures, stage_name, allow_partial=False):
    """Collect results from ProcessPoolExecutor futures with fail-loud semantics.

    Args:
        futures: List of concurrent.futures.Future objects (typically the
            output of executor.submit(...)). Order doesn't matter; results
            are returned in finish-order via as_completed.
        stage_name: Identifier used in log/exception messages. Should be the
            AIDRS stage number/name (e.g. "2.4 TranslationAI", "1.4 junction motif").
        allow_partial: If False (the default), any worker failure raises
            RuntimeError to prevent silent chromosome-level data loss.
            If True, partial results are returned even when some workers died
            (use only for diagnostic / non-production paths).

    Returns:
        List of worker results (in finish-order). Length equals number of
        successful workers; failed workers contribute nothing to the list
        unless allow_partial=True.

    Raises:
        RuntimeError: when at least one worker dies and allow_partial=False.
            The exception aggregates ALL worker tracebacks so the operator
            sees the full failure landscape in one go.
    """
    failed_tasks = []
    results = []

    for future in as_completed(futures):
        exc = future.exception()
        if exc is not None:
            failed_tasks.append(exc)
            logger.critical(
                f"[{stage_name} WORKER FAILED]: {exc}",
                exc_info=exc,
            )
        else:
            results.append(future.result())

    if failed_tasks and not allow_partial:
        raise RuntimeError(
            f"[{stage_name} FATAL] {len(failed_tasks)}/{len(futures)} workers "
            f"crashed! Aborting to prevent silent chromosome-level data loss. "
            f"First error: {failed_tasks[0]!r}. See logs above for full traceback list."
        )

    return results