# AIDRS package
#
# Lazy attribute loading (PEP 562) — `import src` must NOT trigger the
# selene_sdk + puffin + torch + TranslationAI import chain. Heavy modules
# are only resolved when an attribute is actually accessed (e.g.
# `src.aidrs.main`). This keeps unit tests under their timeout and avoids
# loading GBs of torch/selene state into every spawn worker.
#
# Multiprocessing spawn compatibility: the heavy modules are still importable
# by name via `from src import aidrs` because __getattr__ resolves `aidrs`
# to the submodule on first access; spawn workers that re-import
# `src.aidrs.main` get the same module object.
#
# Implementation note: we use `importlib.import_module` rather than the
# `from . import aidrs` syntax inside __getattr__. The `from . import`
# form re-enters the package's __getattr__ via _handle_fromlist, which
# recurses infinitely when `aidrs` is not yet in sys.modules.

__all__ = ["aidrs", "main", "main_entry"]


def __getattr__(name):
    import importlib
    import sys

    # Lazy-load the aidrs submodule only when something on it is requested.
    # We intentionally do NOT eagerly import aidrs at package import time.
    if name == "aidrs":
        mod = sys.modules.get(__name__ + ".aidrs")
        if mod is None:
            mod = importlib.import_module(".aidrs", __name__)
        return mod
    if name in ("main", "main_entry"):
        mod = sys.modules.get(__name__ + ".aidrs")
        if mod is None:
            mod = importlib.import_module(".aidrs", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
