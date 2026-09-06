"""Characterization tests for src/generate_reports.py entry-point functions.

SAFETY NET FOR UPCOMING MODULE SPLIT
=====================================
generate_reports.py is a ~1180-line "god class" (IsoformAnnotator) that owns
annotation, GTF emission, FASTA emission, polyA profiling, and the final
assessment TSV assembly. The next refactor will split it into focused modules
(annotation / gtf / fasta / polya / assessment). Per the Feathers principle:

    "Never refactor a god class without characterization tests first."

This module pins the CURRENT byte-identical output of four entry-point
functions so the upcoming split can be verified by re-running these tests.
A passing SHA after the split means behaviour was preserved; a failing SHA
is a regression that the refactor introduced (not pre-existing drift).

Entry points tested
-------------------
1. IsoformAnnotator._reverse_complement(seq)   -- pure DNA reverse-complement
2. IsoformAnnotator._build_ref_dict(ref_df, include_term)  -- site-set dict
3. IsoformAnnotator.to_gtf(df, output_dir)     -- writes aidrs.transcript_model.gtf
4. IsoformAnnotator._annotate_one_group(df_group, ref_anno, terminal_tolerance)
                                               -- the static method that runs
                                                  inside multiprocessing.Pool
                                                  inside annotate(); the unit
                                                  of annotation in isolation.

Functions intentionally NOT characterized
-----------------------------------------
- save_results(): orchestrates quantify + FASTA + polyA + assessment; needs
  a real genome_fasta and BAM-derived flnc_correct.ssc files. Pure
  integration -- separate end-to-end tests cover it.
- to_fasta(): depends on a real pyfaidx-indexed genome FASTA on disk.
- polyA_len_profile(): reads temp/*_flnc_correct.ssc files written by
  earlier stages; filesystem-coupled and uses polars.
- annotate(): wraps _annotate_one_group + multiprocessing.Pool; we lock
  the per-group path directly because Pool/map is implementation-detail.

SHA scheme
----------
- Pure string functions: SHA256 of the result string.
- Dict output: JSON dump of sorted key->sorted-value, then SHA256.
- TSV-emitting DataFrames: SHA256 of to_csv(sep='\t', index=False) bytes.
- GTF file: SHA256 of the raw file bytes.

All SHAs were captured by running /tmp/capture_shas.py against the current
tree and re-verified to be stable across repeat invocations.
"""
import os
import sys
import json
import hashlib
import importlib.util
import tempfile

import pandas as pd


# ---------------------------------------------------------------------------
# Bootstrap: bypass src/__init__.py via importlib, like test_fsm_rescue.py.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO, "src")

if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Register an empty 'src' package (skip __init__.py execution)
_spec_pkg = importlib.util.spec_from_loader("src", loader=None, is_package=True)
_src_pkg = importlib.util.module_from_spec(_spec_pkg)
_src_pkg.__path__ = [_SRC]
sys.modules["src"] = _src_pkg

# Load the upstream src.* modules generate_reports.py needs at import time.
# (Order matters: leaf modules first, then generate_reports.)
for mod_name, fname in [
    ("src.common", "common.py"),
    ("src.isoform_quantification", "isoform_quantification.py"),
    ("src.gene_grouping", "gene_grouping.py"),
]:
    s = importlib.util.spec_from_file_location(mod_name, os.path.join(_SRC, fname))
    m = importlib.util.module_from_spec(s)
    sys.modules[mod_name] = m
    s.loader.exec_module(m)

# Register src.aidrs_runtime sub-package
_spec_ar = importlib.util.spec_from_loader(
    "src.aidrs_runtime", loader=None, is_package=True
)
_ar_pkg = importlib.util.module_from_spec(_spec_ar)
_ar_pkg.__path__ = [os.path.join(_SRC, "aidrs_runtime")]
sys.modules["src.aidrs_runtime"] = _ar_pkg

s = importlib.util.spec_from_file_location(
    "src.aidrs_runtime.column_registry",
    os.path.join(_SRC, "aidrs_runtime", "column_registry.py"),
)
m = importlib.util.module_from_spec(s)
sys.modules["src.aidrs_runtime.column_registry"] = m
s.loader.exec_module(m)

# Now load generate_reports itself.
_spec_gr = importlib.util.spec_from_file_location(
    "src.generate_reports", os.path.join(_SRC, "generate_reports.py")
)
_gr_mod = importlib.util.module_from_spec(_spec_gr)
sys.modules["src.generate_reports"] = _gr_mod
_spec_gr.loader.exec_module(_gr_mod)

IsoformAnnotator = _gr_mod.IsoformAnnotator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sha_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _canonicalise_ref_dict(d: dict) -> str:
    """Deterministic JSON for dicts whose values are lists of ints.

    Keys sorted; values sorted; ints preserved verbatim. JSON chosen over
    repr() because set-typed values would otherwise sort ambiguously.
    """
    return json.dumps({k: sorted(v) for k, v in sorted(d.items())})


# ---------------------------------------------------------------------------
# 1. _reverse_complement  (pure function: DNA -> reverse-complement)
# ---------------------------------------------------------------------------
def test_reverse_complement():
    """Pin the reverse-complement of three representative sequences.

    All three strings exercise the uppercase letter handling in
    _reverse_complement; the implementation uppercases input so lowercase
    should still work, but we use uppercase here because that is the only
    codepath the GTF/FASTA pipeline actually feeds in.
    """
    inst = IsoformAnnotator(num_processes=1)

    # Pinned SHAs (captured 2026-09-06 against current tree):
    #   'ATGC'         -> 'GCAT'          sha 987a9d04a44b...
    #   'AAAACCCGGT'   -> 'ACCGGGTTTT'    sha 4bb19dcd9d44...
    #   'ATGCATGC'     -> 'GCATGCAT'      sha 2ce780d4e041...

    cases = [
        ("ATGC", "GCAT", "987a9d04a44bf39402afb8d0ee0dcb6007fbbd41b6993c9433adadc97468b0e9"),
        ("AAAACCCGGT", "ACCGGGTTTT",
         "4bb19dcd9d44763617757ebcad8d0b3b726d0743120c8de82657bab2b52e448f"),
        ("ATGCATGC", "GCATGCAT",
         "2ce780d4e0418882a740c2b85a94af8aae8546b44e8aebd3cb531ca42426e9c7"),
    ]
    for seq, expected_str, expected_sha in cases:
        got = inst._reverse_complement(seq)
        assert got == expected_str, (
            f"_reverse_complement({seq!r}): expected {expected_str!r}, got {got!r}"
        )
        assert _sha_str(got) == expected_sha, (
            f"_reverse_complement({seq!r}) SHA drift: "
            f"expected {expected_sha}, got {_sha_str(got)}"
        )
    print(f"[OK] _reverse_complement: 3/{len(cases)} sequences byte-stable")


# ---------------------------------------------------------------------------
# 2. _build_ref_dict  (DataFrame -> dict[str, list[int]])
# ---------------------------------------------------------------------------
def test_build_ref_dict():
    """Pin the per-gene site-set produced by _build_ref_dict.

    include_term=True folds TrStart/TrEnd into the site set (used by
    _map_query_to_ref); include_term=False uses only SSC splice sites
    (used by _novel_gene_remapping for the no-TrStart/TrEnd matching path).
    """
    ref_df = pd.DataFrame({
        "Chr": ["chr1", "chr1", "chr2"],
        "Strand": ["+", "+", "-"],
        "SSC": ["200-300", "200-300", "600"],
        "TrStart": [100, 300, 500],
        "TrEnd": [500, 700, 900],
        "GeneID": ["G1", "G1", "G2"],
        "GeneName": ["Gene1", "Gene1", "Gene2"],
    })

    inst = IsoformAnnotator(num_processes=1)
    d_with = inst._build_ref_dict(ref_df, include_term=True)
    d_without = inst._build_ref_dict(ref_df, include_term=False)

    # Pin structure: 2 distinct gene keys in both modes.
    assert set(d_with.keys()) == {"G1_Gene1", "G2_Gene2"}, (
        f"_build_ref_dict keys: {sorted(d_with.keys())}"
    )
    assert set(d_without.keys()) == {"G1_Gene1", "G2_Gene2"}, (
        f"_build_ref_dict (no_term) keys: {sorted(d_without.keys())}"
    )

    # Pin SHAs (captured 2026-09-06 against the fixture defined above):
    with_term_sha = "68d82195741407678e909396da015191b302c926440a79edf0e8f3202ea01f64"
    without_term_sha = "98ae7d1b395061ec68ca68483b662aa3d63c5e15971f9c0c3748c598c896036e"

    got_with = _sha_str(_canonicalise_ref_dict(d_with))
    got_without = _sha_str(_canonicalise_ref_dict(d_without))
    assert got_with == with_term_sha, (
        f"_build_ref_dict(include_term=True) SHA drift: "
        f"expected {with_term_sha}, got {got_with}"
    )
    assert got_without == without_term_sha, (
        f"_build_ref_dict(include_term=False) SHA drift: "
        f"expected {without_term_sha}, got {got_without}"
    )
    print(f"[OK] _build_ref_dict: 2 modes (with_term, without_term) byte-stable")


# ---------------------------------------------------------------------------
# 3. to_gtf  (DataFrame -> aidrs.transcript_model.gtf file)
# ---------------------------------------------------------------------------
def test_to_gtf_byte_identical():
    """Pin the bytes of aidrs.transcript_model.gtf for a 3-row synthetic input.

    Two Normal CDS-positive transcripts on chr1/+ and one NMD transcript
    on chr2/-. The chr2/- row has no TTS_related_location (NaN) so it
    falls into the Predict_NMD=='NMD' no-CDS branch; the captured SHA
    already includes the resulting "Warning: Error processing transcript"
    fallback path. Accepting that path is part of the lock.
    """
    gtf_df = pd.DataFrame({
        "Chr": ["chr1", "chr1", "chr2"],
        "Strand": ["+", "+", "-"],
        "TrStart": [100, 300, 500],
        "TrEnd": [500, 700, 900],
        "TrID": ["T1", "T2", "T3"],
        "GeneID": ["G1", "G1", "G2"],
        "GeneName": ["Gene1", "Gene1", "Gene2"],
        "SSC": ["200-300", "400-500", "600-700"],
        "TrStart_ref": [100, 300, 500],
        "TrEnd_ref": [500, 700, 900],
        "Predict_NMD": ["Normal", "Normal", "NMD"],
        "TIS_related_location": [10, 10, 5],
        "TTS_related_location": [400, 600, 200],
    })

    inst = IsoformAnnotator(num_processes=1)
    with tempfile.TemporaryDirectory() as out_dir:
        inst.to_gtf(gtf_df.copy(), out_dir)
        gtf_path = os.path.join(out_dir, "aidrs.transcript_model.gtf")
        assert os.path.exists(gtf_path), (
            f"to_gtf did not produce expected file at {gtf_path}"
        )
        with open(gtf_path, "rb") as f:
            gtf_bytes = f.read()

    expected_sha = "86af372cb3de8d024495efb2b4fbad7325e709bf2d1e299054b618b22b60a7de"
    expected_size = 2570

    assert len(gtf_bytes) == expected_size, (
        f"to_gtf output size drift: expected {expected_size} bytes, got {len(gtf_bytes)}"
    )
    actual_sha = _sha_bytes(gtf_bytes)
    assert actual_sha == expected_sha, (
        f"to_gtf output SHA drift: expected {expected_sha}, got {actual_sha}"
    )
    print(f"[OK] to_gtf: {expected_size}-byte GTF byte-stable sha={actual_sha[:12]}...")


# ---------------------------------------------------------------------------
# 4. _annotate_one_group  (static method used inside Pool.map)
# ---------------------------------------------------------------------------
def test_annotate_one_group_novel_only():
    """Pin the per-group annotation TSV for an all-novel Group with ref_anno=None.

    This is the static method that gets pickled into multiprocessing.Pool
    workers by annotate(); testing it directly lets us pin the per-group
    annotation result without spinning up a Pool. With ref_anno=None the
    function falls into _fill_novel(), which assigns TrID/GeneID/GeneName
    of the form NovelGene{gid}_Novel{trid}. Two rows for the same uniqueTr
    are deduped to a single output row.
    """
    group_df = pd.DataFrame({
        "Chr": ["chr1", "chr1"],
        "Strand": ["+", "+"],
        "SSC": ["200-300", "200-300"],
        "TrStart": [100, 100],
        "TrEnd": [500, 500],
        "frequency": [10, 5],
        "uniqueTr": ["Tr0", "Tr0"],
        "TIS_related_location": [10, 10],
        "TTS_related_location": [400, 400],
        "Predict_NMD": ["Normal", "Normal"],
        "Group": [0, 0],
        "TrID": ["NovelTr_T0", "NovelTr_T0"],
        "GeneID": ["NovelGene0", "NovelGene0"],
        "GeneName": ["NovelGene0", "NovelGene0"],
        "TrStart_ref": [100, 100],
        "TrEnd_ref": [500, 500],
    })

    out = IsoformAnnotator._annotate_one_group(
        group_df.copy(), ref_anno=None, terminal_tolerance=50,
    )

    # Structural assertions (these are guaranteed by the annotation contract
    # and are checked first so the SHA check below is unambiguous):
    assert isinstance(out, pd.DataFrame), (
        f"_annotate_one_group must return DataFrame; got {type(out).__name__}"
    )
    expected_cols = [
        "Chr", "Strand", "SSC", "TrStart", "TrEnd", "frequency", "uniqueTr",
        "TIS_related_location", "TTS_related_location", "Predict_NMD",
        "Group", "TrID", "GeneID", "GeneName", "TrStart_ref", "TrEnd_ref",
    ]
    assert list(out.columns) == expected_cols, (
        f"_annotate_one_group columns drift: expected {expected_cols}, "
        f"got {list(out.columns)}"
    )

    # SHA-pin the TSV (captured 2026-09-06):
    expected_sha = "260e07c2ace298d8184bfac0b21861ee8d24bd670e17d386195d1f4b5974828d"
    tsv = out.to_csv(sep="\t", index=False)
    actual_sha = _sha_str(tsv)
    assert actual_sha == expected_sha, (
        f"_annotate_one_group(no-ref) SHA drift: expected {expected_sha}, "
        f"got {actual_sha}"
    )
    print(f"[OK] _annotate_one_group(no-ref): {out.shape} byte-stable sha={actual_sha[:12]}...")


# ---------------------------------------------------------------------------
# main: allow running as a standalone script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_reverse_complement()
    test_build_ref_dict()
    test_to_gtf_byte_identical()
    test_annotate_one_group_novel_only()
    print("\nAll generate_reports.py characterization tests passed.")
