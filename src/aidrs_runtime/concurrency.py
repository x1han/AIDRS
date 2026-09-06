"""Concurrency utilities for AIDRS pipeline.

Provides canonical, fail-loud helpers for collecting ProcessPoolExecutor
futures. Use these instead of raw as_completed() loops — silent-swallow
anti-patterns (where worker exceptions are dropped on the floor) cause
silent chromosome-level data loss that no test can detect without explicit
SHA divergence checks.
"""

import ctypes
import logging
import signal
from concurrent.futures import as_completed, ProcessPoolExecutor

logger = logging.getLogger("AIDRS")


def _worker_death_pact():
    """Initializer that wires PR_SET_PDEATHSIG=SIGKILL into each worker.

    Linux-only: requests that the kernel deliver SIGKILL to this process
    if its parent dies, so orphan workers cannot outlive the orchestrator
    and burn CPU after a crash. Failures (non-Linux, missing libc, etc.)
    are swallowed because the pact is best-effort observability, not a
    correctness invariant.
    """
    try:
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:
        pass


def get_process_pool(num_workers, mp_context=None):
    """Factory for a ProcessPoolExecutor with the death-pact initializer.

    Every worker process is set up with PR_SET_PDEATHSIG=SIGKILL so that
    orphan workers terminate immediately when the parent dies.
    """
    kwargs = {"max_workers": num_workers, "initializer": _worker_death_pact}
    if mp_context is not None:
        kwargs["mp_context"] = mp_context
    return ProcessPoolExecutor(**kwargs)


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
                "[%s WORKER FAILED]: %s",
                stage_name, exc,
                exc_info=exc,
            )
        else:
            results.append(future.result())

    if failed_tasks and not allow_partial:
        raise RuntimeError(
            f"[{stage_name} FATAL] {len(failed_tasks)}/{len(futures)} workers "
            f"crashed! Aborting to prevent silent chromosome-level data loss."
        ) from failed_tasks[0]

    return results