"""Validation of emitted transcript / gene identifiers.

The transcript id becomes the reference name (RNAME) of the AIDRS model
FASTA and of the per-sample BAM. Two hard constraints follow from that:

* **Length** -- SAM v1.6 limits QNAME to 254 bytes
  (``hts-specs/SAMv1.tex:1493``: "QNAME limited to 254 bytes"). A longer
  name is silently truncated by some tools and crashes others
  (minimap2 stores the name length in a ``uint8_t`` inside its ``.mmi``
  index, so a name of exactly 256 bytes produces a NULL name pointer and
  a SIGSEGV).
* **Uniqueness** -- a duplicated transcript id makes the model FASTA
  ambiguous and merges two distinct transcripts into one downstream.

Both are checked here, fail loud, before anything is written.
"""
from typing import Optional

import pandas as pd

# SAM v1.6 QNAME limit, in bytes.
MAX_TRANSCRIPT_ID_LEN = 254


def byte_len(value: str) -> int:
    """Length of ``value`` in BYTES.

    The QNAME limit counts bytes, not characters: a 100-character non-ASCII
    id is 300 bytes long. Measuring characters would let such an id through.
    """
    return len(str(value).encode('utf-8'))


def byte_lengths(ids: pd.Series) -> pd.Series:
    """Length of every id of ``ids`` in BYTES (see :func:`byte_len`)."""
    return ids.astype(str).str.encode('utf-8').str.len()


def validate_transcript_ids(df: pd.DataFrame,
                            id_column: str = 'TrID',
                            max_len: int = MAX_TRANSCRIPT_ID_LEN,
                            context: str = "") -> None:
    """Fail loud when transcript ids are too long or duplicated.

    The length is measured in BYTES: the QNAME limit is a byte limit, so a
    non-ASCII id that is short in characters can still be too long.

    Args:
        df: Frame carrying the id column.
        id_column: Name of the id column to validate.
        max_len: Maximum allowed length in bytes.
        context: Optional prefix for the error message.

    Raises:
        RuntimeError: If any id exceeds ``max_len`` or any id is duplicated.
    """
    if df is None or id_column not in df.columns:
        return

    ids = df[id_column].astype(str)
    lengths = byte_lengths(ids)

    too_long_mask = lengths > max_len
    if too_long_mask.any():
        too_long = ids[too_long_mask]
        worst = lengths[too_long_mask].max()
        raise RuntimeError(
            f"[FAIL-LOUD] {context}{int(too_long_mask.sum())} transcript id(s) "
            f"exceed the {max_len}-byte limit (longest {worst} bytes): "
            f"{sorted(too_long.unique())[:3]}"
        )

    duplicated = ids[ids.duplicated(keep=False)]
    if len(duplicated):
        dupes = sorted(duplicated.unique())
        raise RuntimeError(
            f"[FAIL-LOUD] {context}{len(duplicated)} row(s) carry a duplicated "
            f"transcript id ({len(dupes)} distinct): {dupes[:3]}"
        )


def validate_gene_ids(df: pd.DataFrame,
                      id_column: str = 'GeneID',
                      max_len: Optional[int] = None,
                      context: str = "") -> None:
    """Fail loud when gene ids are too long.

    The gene id has no hard external limit (it is only a GTF attribute),
    so ``max_len`` defaults to the AIDRS cap used when generating the
    aggregate ids.

    Args:
        df: Frame carrying the id column.
        id_column: Name of the id column to validate.
        max_len: Maximum allowed length in bytes; ``None`` skips the check.
        context: Optional prefix for the error message.

    Raises:
        RuntimeError: If any id exceeds ``max_len``.
    """
    if max_len is None or df is None or id_column not in df.columns:
        return

    ids = df[id_column].astype(str)
    lengths = byte_lengths(ids)
    too_long_mask = lengths > max_len
    if too_long_mask.any():
        too_long = ids[too_long_mask]
        raise RuntimeError(
            f"[FAIL-LOUD] {context}{int(too_long_mask.sum())} gene id(s) exceed "
            f"the {max_len}-byte cap (longest {lengths[too_long_mask].max()} bytes): "
            f"{sorted(too_long.unique())[:3]}"
        )
