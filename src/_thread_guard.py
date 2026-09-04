"""Caps per-process BLAS/OMP thread count to 1 before any numpy/pandas/sklearn import.

Must be the FIRST import in any AIDRS CLI entrypoint (aidrs.py, bam2ssc.py,
gtf2ssc.py) so the cap is in effect before transitive BLAS-loaded libraries
spin up their thread pools. Uses os.environ.setdefault so a caller that
intentionally sets these vars (e.g., a CI run pre-set to the baseline value
used to generate H_P0_ALL) can still override.

Hypothesized root cause for [Aborted (core dumped)] in aidrs.py-driven flows:
fork-based multiprocessing.Pool workers collide with the parent's already-loaded
BLAS thread pool. Capping the parent's BLAS threads to 1 before the first fork
eliminates per-worker thread fan-out so forks become safe even if the worker
code itself is unchanged.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
