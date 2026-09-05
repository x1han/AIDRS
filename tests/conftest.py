"""Pytest configuration for the AIDRS test suite.

Every test module under ``tests/`` bootstraps its own ``sys.path`` from
``__file__``, so each one stays runnable as a standalone script
(``python tests/test_xxx.py``).  Pytest discovery is purely additive: this
file only adds environment-dependent deselection on top.
"""

import importlib.util
import sys

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``torch`` when torch is not importable.

    Lets torch-dependent tests be written unconditionally instead of gating
    them by hand inside the test body.  ``find_spec`` only locates the module,
    it does not import it -- important because tests/test_lazy_import.py
    asserts ``"torch" not in sys.modules``.
    """
    if importlib.util.find_spec("torch") is not None:
        return
    skip_torch = pytest.mark.skip(reason="torch not installed")
    for item in items:
        if "torch" in item.keywords:
            item.add_marker(skip_torch)


def _src_modules():
    return {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}


@pytest.fixture(autouse=True)
def _isolate_src_modules():
    """Restore ``sys.modules['src*']`` after every test.

    Only needed under pytest, where all test modules share one interpreter;
    as standalone scripts each file gets a fresh interpreter and cannot leak.

    tests/test_lazy_import.py deliberately deletes the cached ``src`` modules
    to measure cold-import cost, which replaces the ``src.gene_grouping``
    module object.  tests/test_p1_3_cluster_smoke.py binds ``GeneClustering``
    at import time and passes a bound method to ``multiprocessing.Pool``;
    pickling that method resolves the class by name and fails with
    ``PicklingError: it's not the same object`` once the module has been
    swapped underneath it.  Snapshotting and restoring keeps the two
    independent, whatever order they run in.
    """
    saved = _src_modules()
    yield
    for name in list(_src_modules()):
        if name not in saved:
            del sys.modules[name]
    sys.modules.update(saved)
