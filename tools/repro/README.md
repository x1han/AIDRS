# `tools/repro/` — gate helper

| Script | What it shows |
|---|---|
| `compare_factorial_semantic.py` | The float-tolerant comparator of the chr1 factorial gate — compares two factorial output trees file by file and, for a file that differs, column-wise (every non-numeric column must be identical, every numeric column must agree within 1e-5 absolute). Needed because TranslationAI's float32 inference is not bit-reproducible, so a byte comparison of the TranslationAI assessment tables has a false-positive mode. It fails loudly on the structural modes too: a case directory missing on either side, or an output file that is present in the base but lost from the run under test, both exit non-zero. Usage: `python tools/repro/compare_factorial_semantic.py <FINAL dir> [<BASE dir>]`. |

Not part of the pipeline: nothing in `src/` imports it, and it is not part of the
distribution (`pyproject.toml` packages only `aidrs` and `translationai`).

Regression test: `tests/test_defensive.py::TestFactorialComparatorFailsLoud`.
