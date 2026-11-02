"""AIDRS Layer-1 smoke test: package-level invariants only.

Does NOT import src.aidrs (would pull selene_sdk + puffin + torch);
see src/__init__.py PEP 562 lazy-load contract.
"""
import src


def test_src_package_lazy_load_contract():
    """src/__init__.py PEP 562 lazy-load is wired correctly."""
    assert src.__all__ == ["aidrs", "main", "main_entry"]


def test_src_getattr_rejects_unknown_names():
    import pytest
    with pytest.raises(AttributeError):
        src.does_not_exist


def test_polyA_sidecar_naming_convention():
    """generate_reports polyA dict keys must use dotted form.

    Pinned by commit 7f53053: aidrs.transcript_polyA_len.parquet ->
    aidrs.transcript.polyA_len.parquet. Reverting this string would
    silently regress the on-disk filename for downstream tools.
    """
    from src import generate_reports
    src_text = open(generate_reports.__file__).read()
    assert 'aidrs.{transcript,gene}_polyA_len.parquet' not in src_text, \
        "polyA sidecar filename must use dotted form (aidrs.transcript.polyA_len.parquet)"
    assert '"transcript.polyA_len"' in src_text
    assert '"gene.polyA_len"' in src_text