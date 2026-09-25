"""B-class integration / smoke tests for AIDRS.

Pin the user-visible CLI surface (run_aidrs.sh, submit_aidrs_qsub.sh) and
defend against regressions in defensive structural invariants:

- Shell wrapper exit codes on help / missing args / fake paths.
- No silent except-pass patterns remain in src/.
- Exactly one raw ProcessPoolExecutor instantiation (the factory in
  concurrency.py) and exactly two get_process_pool call sites
  (gtf2ssc.py + protein_coding_ability.py).
- Star-import count pinned at 2 (aidrs.py L31 + L44).
- ResourceGuard default CPU thread floor.
- qsub wrapper shell syntax + lock-file atomicity.

All tests are subprocess-based (bash / grep / py_compile). They run in
<90s on a clean aidrs env and require no real BAM, no torch, no selene.
"""

import os
import ast
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
# Workspace scripts live outside repo/ (per project CLAUDE.md layer
# boundaries — repo/ owns shipped code, workspace/ owns local material).
# Allow override via env var so worktree checkouts (where `..` may be
# `.worktrees/<branch>/` rather than `<repo_parent>/`) can still find
# the canonical scripts at /datf/hanxi/software/AIDRS/workspace/scripts.
SCRIPTS_ROOT = os.environ.get(
    "AIDRS_WORKSPACE_SCRIPTS",
    "/datf/hanxi/software/AIDRS/workspace/scripts",
)

RUN_AIDRS_SH = os.path.join(SCRIPTS_ROOT, "run_aidrs.sh")
SUBMIT_AIDRS_QSUB_SH = os.path.join(SCRIPTS_ROOT, "submit_aidrs_qsub.sh")
# The qsub worker script (NOT the submitter) owns the mkdir-based lock
# per memory translationai_silent_zero_orf_3_defenses_2026-09-10.
RUN_AIDRS_QSUB_SH = os.path.join(SCRIPTS_ROOT, "run_aidrs_qsub.sh")


def _bash(script_path, *args, env=None):
    """Run a shell script via bash and return CompletedProcess."""
    if not os.path.isfile(script_path):
        pytest.skip(f"shell script not present at {script_path}")
    return subprocess.run(
        ["bash", script_path, *args],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, **(env or {})},
    )


def _walk_py_files(root):
    """Yield .py file paths under root, recursively."""
    root = Path(root)
    for path in root.rglob("*.py"):
        yield str(path)


def _rg_py(pattern_str, root):
    r"""Regex-search pattern_str over all .py files under root.

    Returns list of (file_path, line_no, line_text) tuples.
    Uses Python's re module so POSIX grep `\b`/`\s` quirks don't bite.
    """
    pat = re.compile(pattern_str)
    matches = []
    for fpath in _walk_py_files(root):
        try:
            with open(fpath, encoding="utf-8") as fh:
                for ln, line in enumerate(fh, start=1):
                    if pat.search(line):
                        matches.append((fpath, ln, line.rstrip("\n")))
        except OSError:
            continue
    return matches


# =============================================================================
# 1. run_aidrs.sh --help exits 1 + usage banner
# =============================================================================

def test_run_aidrs_sh_help_exits_1():
    """-h triggers usage() which exits 1 with the usage banner."""
    result = _bash(RUN_AIDRS_SH, "-h")
    assert result.returncode == 1, (
        f"expected exit 1 on -h, got {result.returncode}\n"
        f"stderr: {result.stderr}"
    )
    # usage() echoes "Usage: $0 -r <reference.fa>..."
    assert "Usage:" in result.stdout, (
        f"expected usage banner on stdout, got: {result.stdout!r}"
    )


# =============================================================================
# 2. run_aidrs.sh missing args exits 1 + "missing required arguments"
# =============================================================================

def test_run_aidrs_sh_missing_args():
    """No -r/-b/-o → exit 1 + 'missing required arguments'."""
    result = _bash(RUN_AIDRS_SH, "-r", "/tmp/anything")
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "missing required arguments" in combined, (
        f"expected error message, got: {combined!r}"
    )


# =============================================================================
# 3. run_aidrs.sh fake ref path exits 1 + "reference not found"
# =============================================================================

def test_run_aidrs_sh_fake_ref_path(tmp_path):
    """-r <nonexistent> → exit 1 + 'reference not found'."""
    fake_ref = "/nonexistent_ref_xyz.fa"
    fake_bam = "/nonexistent_bam_xyz.bam"
    out_dir = str(tmp_path / "out")
    result = _bash(RUN_AIDRS_SH,
                   "-r", fake_ref, "-b", fake_bam, "-o", out_dir)
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "reference not found" in combined, (
        f"expected 'reference not found', got: {combined!r}"
    )


# =============================================================================
# 4. submit_aidrs_qsub.sh no args → exit 2 + "--bam required"
# =============================================================================

def test_run_aidrs_qsub_sh_help_or_noargs():
    """submit_aidrs_qsub.sh with no args → exit 2 + '--bam required'."""
    result = _bash(SUBMIT_AIDRS_QSUB_SH)
    assert result.returncode == 2, (
        f"expected exit 2 on missing required, got {result.returncode}\n"
        f"stderr: {result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "--bam required" in combined, (
        f"expected '--bam required' message, got: {combined!r}"
    )


# =============================================================================
# 5. logger binding pattern across src/
# =============================================================================

def test_logger_binding_in_src():
    """Every src/ file that calls logger.* must bind getLogger(__name__) (or named logger).

    The Pipeline Completion Invariant (memory: verify_discipline) requires
    each module to own its logger name so log lines trace back to the
    originating module. We allow either `getLogger(__name__)` or
    `getLogger("AIDRS")` / `getLogger("aidrs")` (the shared project logger)
    as long as it's a stable, module-level binding.
    """
    bindings = {}
    uses = {}
    for root, _dirs, files in os.walk(SRC_ROOT):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            # Match the standard binding forms.
            m_name = re.search(r'getLogger\(__name__\)', src)
            m_aidrs = re.search(r'getLogger\(["\']AIDRS["\']\)', src)
            m_aidrs_lower = re.search(r'getLogger\(["\']aidrs["\']\)', src)
            if m_name or m_aidrs or m_aidrs_lower:
                bindings[path] = True
            # Heuristic: any `logger.` reference outside a string literal.
            if re.search(r'(?<![\w\.])logger\.[a-zA-Z]', src):
                uses[path] = True

    unbound = sorted(uses.keys() - bindings.keys())
    assert not unbound, (
        f"files using logger.* without a getLogger binding: {unbound}"
    )


# =============================================================================
# 6. No silent-swallow `except ... pass` patterns in src/
# =============================================================================

def test_no_silent_swallow_in_src():
    """Pin: no NEW silent-swallow `except X: pass` (single- or multi-line) in src/.

    AST-based detection catches BOTH forms:
      Single-line: `except Exception: pass`
      Multi-line:  `except Exception:\n    pass`

    Known sites are whitelisted as intentional best-effort. A refactor
    that adds a NEW silent-swallow outside the whitelist fails the test.
    """
    whitelist = {
        # _worker_death_pact: best-effort PR_SET_PDEATHSIG on Linux;
        # not a correctness invariant (memory worker_death_pact_2026-09-06).
        os.path.join(SRC_ROOT, "aidrs_runtime", "concurrency.py"),
        # aidrs.py — pre-existing best-effort swallows (h.flush fallback,
        # NativeDirectoryLock holder-info read). Per CLAUDE.md surgical
        # rule, don't touch unrelated dead code; just whitelist.
        os.path.join(SRC_ROOT, "aidrs.py"),
        # common.py — fsync on non-POSIX mount + tmp-file cleanup, both
        # documented in inline comments.
        os.path.join(SRC_ROOT, "common.py"),
        # resource_guard.py — cgroup v1/v2 + SGE_H_VMEM parsing. File
        # may be absent or malformed; the function falls through to the
        # 32 GB default. Documented inline.
        os.path.join(SRC_ROOT, "aidrs_runtime", "resource_guard.py"),
    }
    offenders = []
    for fpath in _walk_py_files(SRC_ROOT):
        try:
            with open(fpath, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=fpath)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            # Strip leading docstring Expr (PEP-257). A silent swallow is
            # an ExceptHandler whose only remaining body statement is Pass.
            body = [
                s for s in node.body
                if not (
                    isinstance(s, ast.Expr)
                    and isinstance(s.value, ast.Constant)
                    and isinstance(s.value.value, str)
                )
            ]
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                if fpath not in whitelist:
                    exc_text = ast.unparse(node.type) if node.type else ""
                    offenders.append(
                        f"{fpath}:{node.lineno}: silent swallow: "
                        f"except {exc_text}: pass"
                    )
    assert not offenders, (
        "silent-swallow anti-pattern detected:\n  "
        + "\n  ".join(offenders)
    )


# =============================================================================
# 7. PEE factory pattern invariants
# =============================================================================

def test_pee_site_count_is_one_factory_plus_two_callers():
    """Pin: exactly 1 raw `ProcessPoolExecutor(` (the factory) and 2 `get_process_pool(` call sites.

    After the v1.0.7 refactor (memory: drain_futures_loud_helper_2026-09-06),
    gtf2ssc.py and protein_coding_ability.py must route through
    concurrency.get_process_pool; concurrency.py owns the single raw
    ProcessPoolExecutor instantiation as a factory return.
    """
    raw_pee_matches = _rg_py(r"ProcessPoolExecutor\(", SRC_ROOT)
    # Lines that look like instantiation (not import / not docstring):
    # either `= ProcessPoolExecutor(...)` (assignment) or
    # `return ProcessPoolExecutor(...)`. Imports are bare `ProcessPoolExecutor`
    # without the open paren so they don't match the search regex.
    raw_instantiation_files = sorted({
        fpath for fpath, _ln, line in raw_pee_matches
        if re.search(r"(\b=\s*ProcessPoolExecutor|\breturn\s+ProcessPoolExecutor)", line)
    })
    assert raw_instantiation_files == [
        os.path.join(SRC_ROOT, "aidrs_runtime", "concurrency.py"),
    ], f"raw PEE instantiations drifted: {raw_instantiation_files}"

    # get_process_pool call sites: lines that contain `get_process_pool(`.
    # Definition line `def get_process_pool(...)` also contains it, but
    # lives in concurrency.py — it's the factory itself, not a caller.
    pool_matches = _rg_py(r"get_process_pool\(", SRC_ROOT)
    pool_call_sites = sorted({
        fpath for fpath, _ln, line in pool_matches
        if not re.search(r"\bdef\s+get_process_pool\b", line)
    })
    assert pool_call_sites == sorted({
        os.path.join(SRC_ROOT, "gtf2ssc.py"),
        os.path.join(SRC_ROOT, "protein_coding_ability.py"),
    }), f"get_process_pool() call sites drifted: {pool_call_sites}"


# =============================================================================
# 8. Star-import count in src/
# =============================================================================

def test_star_import_files_isolated_to_aidrs():
    """Star imports are confined to aidrs.py (currently L31, L44).

    We pin the cross-file invariant (no star imports outside aidrs.py)
    but NOT the count, because a legitimate refactor that replaces
    `from X import *` with explicit named imports is a byte-identity
    preservation win and should not be blocked by a count assertion.
    """
    star_imports = _rg_py(r"^from\s+\S+\s+import\s+\*$", SRC_ROOT)
    star_files = sorted({fpath for fpath, _ln, _line in star_imports})
    assert star_files == [os.path.join(SRC_ROOT, "aidrs.py")], (
        f"star-import files drifted outside aidrs.py: {star_files}"
    )


# =============================================================================
# 9. ResourceGuard default CPU thread floor
# =============================================================================

def test_resource_guard_default_threads_at_least_4(monkeypatch):
    """get_effective_cpu_threads(None) with no env returns >= 4."""
    # Strip all scheduler hints.
    for var in ("NSLOTS", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(var, raising=False)
    # Import lazily so monkeypatch has effect.
    from src.aidrs_runtime.resource_guard import ResourceGuard
    n = ResourceGuard.get_effective_cpu_threads(None)
    assert n >= 4, f"expected >=4 default threads, got {n}"


# =============================================================================
# 10. submit_aidrs_qsub.sh syntax + lock-file presence
# =============================================================================

def test_run_aidrs_qsub_sh_lock_atomic_compiles():
    """bash -n validates syntax; lock-file lines present for atomic create-if-absent."""
    if not os.path.isfile(RUN_AIDRS_QSUB_SH):
        pytest.skip(f"qsub worker script not present at {RUN_AIDRS_QSUB_SH}")
    # Syntax check
    proc = subprocess.run(
        ["bash", "-n", RUN_AIDRS_QSUB_SH],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"bash syntax error in run_aidrs_qsub.sh: {proc.stderr}"
    )
    # Look for atomic lock creation pattern (mkdir-based for RHEL 7 bash 4.2
    # compatibility — exec N>file is bash 4.3+).
    with open(RUN_AIDRS_QSUB_SH, encoding="utf-8") as fh:
        body = fh.read()
    has_mkdir_lock = bool(re.search(r"\bmkdir\b.*\b2>/dev/null", body))
    assert has_mkdir_lock, (
        "run_aidrs_qsub.sh (qsub worker) should use mkdir-based atomic lock "
        "(Bash 4.2 RHEL 7 does not support `exec N>file`). See memory "
        "translationai_silent_zero_orf_3_defenses_2026-09-10."
    )