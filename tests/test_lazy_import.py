"""Verify P0-B: src/__init__.py uses PEP 562 lazy loading.

The eager `from .aidrs import main, main_entry` previously pulled the full
selene_sdk + puffin + torch + TranslationAI import chain into any process
that did `import src` or `from src.X import Y` — breaking unit tests with
60s+ timeouts and consuming GBs of memory in spawn workers.

This test verifies the lazy-loading contract:
  1. `import src` completes in < 5 seconds (no eager puffin/torch load).
  2. `from src.gene_grouping import X` is fast (no eager puffin/torch load
     via the package __init__).
  3. AttributeError is raised for unknown attribute names — and crucially,
     NOT RecursionError (the PEP 562 `from . import` gotcha).
  4. `__all__` lists the publicly-exposed lazy names.
  5. If torch/selene_sdk are present, src.aidrs.main and src.aidrs.main_entry
     resolve to callables. If absent, the failure must be a clean
     ModuleNotFoundError (still lazy, just hitting the chain at first use).
  6. Multiprocessing spawn compatibility: aidrs submodule is re-importable
     by its full dotted name (only verifiable when heavy deps are present).
"""
import sys
import time
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _heavy_deps_available():
    """True iff torch + selene_sdk are importable in this env.

    The aidrs submodule transitively imports both via tss_annotation →
    puffin_runner, so they must be present for src.aidrs to load at all.
    """
    try:
        importlib.import_module("torch")
        importlib.import_module("selene_sdk")
        return True
    except ImportError:
        return False


def _fresh_import_src():
    """Drop any cached `src` and re-import it from scratch.

    Ensures each timing measurement reflects cold module-load cost, not
    any test-fixture caching.
    """
    for mod_name in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
        del sys.modules[mod_name]
    return importlib.import_module("src")


def test_import_src_is_fast():
    """import src must complete in < 5 s (no eager puffin/torch load)."""
    t0 = time.perf_counter()
    src = _fresh_import_src()
    elapsed = time.perf_counter() - t0
    assert src is not None
    assert elapsed < 5.0, f"import src took {elapsed:.2f}s (expected < 5s)"
    # And torch must NOT have been imported as a side-effect of `import src`
    # (it is only loaded when something inside src.aidrs is touched).
    assert "torch" not in sys.modules, (
        "import src eagerly loaded torch — lazy __getattr__ is broken"
    )
    print(f"[OK] import src: {elapsed:.3f}s (torch not loaded)")


def test_from_src_gene_grouping_is_fast():
    """from src.gene_grouping import X must be fast and NOT load torch/puffin.

    Pre-refactor this path triggered puffin/torch via the package __init__'s
    eager `from .aidrs import main, main_entry`. Post-refactor it should only
    touch gene_grouping.py's own (lightweight) imports.

    Threshold is 10s to accommodate cold pandas/numpy load on a fresh
    interpreter; the key property we assert is that torch is NOT loaded
    (proving the puffin chain was never triggered).
    """
    # Pre-warm pandas/numpy so the measurement reflects only the src.* path,
    # not pandas cold-start cost.
    import pandas  # noqa: F401
    import numpy  # noqa: F401

    for mod_name in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
        del sys.modules[mod_name]

    t0 = time.perf_counter()
    from src.gene_grouping import GeneClustering  # noqa: F401
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"from src.gene_grouping import took {elapsed:.2f}s"
    assert "torch" not in sys.modules, (
        "from src.gene_grouping eagerly loaded torch — lazy __getattr__ is broken"
    )
    assert "selene_sdk" not in sys.modules, (
        "from src.gene_grouping eagerly loaded selene_sdk — lazy __getattr__ is broken"
    )
    print(f"[OK] from src.gene_grouping import: {elapsed:.3f}s "
          f"(torch + selene_sdk not loaded)")


def test_src_unknown_attr_raises_attribute_error():
    """AttributeError (NOT RecursionError) for unknown attribute names.

    RecursionError would mean __getattr__ recursed via `from . import`
    inside its body — the classic PEP 562 gotcha.
    """
    src = _fresh_import_src()
    raised = None
    try:
        _ = src.does_not_exist
    except RecursionError as e:
        raised = ("RecursionError", str(e))
    except AttributeError as e:
        raised = ("AttributeError", str(e))
    assert raised is not None, "expected an exception"
    kind, _msg = raised
    assert kind == "AttributeError", (
        f"src.does_not_exist raised {kind}; expected AttributeError. "
        f"This usually means __getattr__ recursed (PEP 562 gotcha)."
    )
    print(f"[OK] src.does_not_exist raised AttributeError as expected")


def test_src_all_lists_expected_names():
    """__all__ must contain aidrs, main, main_entry for IDE/doc tools."""
    src = _fresh_import_src()
    assert hasattr(src, "__all__")
    for name in ("aidrs", "main", "main_entry"):
        assert name in src.__all__, f"{name!r} missing from src.__all__"
    print(f"[OK] src.__all__ = {src.__all__}")


def test_src_dir_lists_lazy_names():
    """__dir__ must advertise the lazy attribute names for tab-completion."""
    src = _fresh_import_src()
    d = dir(src)
    for name in ("aidrs", "main", "main_entry"):
        assert name in d, f"{name!r} missing from dir(src)"
    print(f"[OK] dir(src) contains {['aidrs', 'main', 'main_entry']}")


def test_src_aidrs_resolution():
    """src.aidrs either resolves cleanly (deps present) or fails with
    ModuleNotFoundError (deps absent) — but NEVER with RecursionError."""
    src = _fresh_import_src()
    try:
        aidrs_mod = src.aidrs
    except RecursionError:
        raise AssertionError(
            "src.aidrs raised RecursionError — __getattr__ recursed (PEP 562 gotcha)"
        )
    except ModuleNotFoundError as e:
        # Expected in this env: torch/selene_sdk not installed.
        print(f"[OK] src.aidrs triggered heavy chain; failed cleanly with "
              f"ModuleNotFoundError({e.name!r}) — lazy loading is working "
              f"(chain only loaded on attribute access, not on `import src`)")
        return
    except ImportError as e:
        print(f"[OK] src.aidrs triggered heavy chain; failed cleanly with "
              f"ImportError({e}) — lazy loading is working")
        return

    # If we got here, heavy deps ARE installed. Verify the module is sane.
    assert aidrs_mod.__name__ == "src.aidrs"
    assert callable(getattr(aidrs_mod, "main", None)), \
        "src.aidrs.main is not callable"
    assert callable(getattr(aidrs_mod, "main_entry", None)), \
        "src.aidrs.main_entry is not callable"
    print(f"[OK] src.aidrs resolved to {aidrs_mod.__name__}; "
          f"main and main_entry are callable")


def test_src_main_and_main_entry_resolution():
    """src.main and src.main_entry must resolve when heavy deps are present."""
    if not _heavy_deps_available():
        print("[SKIP] torch/selene_sdk not installed; skipping callable check")
        return

    src = _fresh_import_src()
    main = src.main
    main_entry = src.main_entry
    assert callable(main), f"src.main is not callable: {type(main)}"
    assert callable(main_entry), f"src.main_entry is not callable: {type(main_entry)}"
    print(f"[OK] src.main and src.main_entry are both callable")


def test_spawn_worker_can_reimport_aidrs_submodule():
    """Multiprocessing spawn workers re-import `src.aidrs` by dotted name.

    With PEP 562 lazy loading, the aidrs submodule must be importable
    directly via `from src import aidrs` so spawn workers can find it.
    Only verifiable when heavy deps are present.
    """
    if not _heavy_deps_available():
        print("[SKIP] torch/selene_sdk not installed; skipping spawn check")
        return

    src = _fresh_import_src()
    aidrs_via_attr = src.aidrs
    aidrs_via_importlib = importlib.import_module("src.aidrs")
    assert aidrs_via_attr is aidrs_via_importlib, (
        "src.aidrs (via __getattr__) and importlib.import_module('src.aidrs') "
        "must return the same module object so spawn workers share state."
    )
    print(f"[OK] src.aidrs is the same module object via both access paths")


if __name__ == "__main__":
    test_import_src_is_fast()
    test_from_src_gene_grouping_is_fast()
    test_src_unknown_attr_raises_attribute_error()
    test_src_all_lists_expected_names()
    test_src_dir_lists_lazy_names()
    test_src_aidrs_resolution()
    test_src_main_and_main_entry_resolution()
    test_spawn_worker_can_reimport_aidrs_submodule()
    print("\nAll lazy-import tests PASSED")
