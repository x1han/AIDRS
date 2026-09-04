"""Unit tests for src/rt_switching_filter.py (Stage 2.7 RT-Switching detection).

These tests are isolated: no pysam, no real FASTA. A tiny in-memory
``MockFastaFile`` class mimics the pysam.FastaFile.fetch(chrom, start, end)
interface. Tests cover:
    1. Canonical motif (GT-AG) -> flag=False.
    2. Non-canonical motif with no microhomology -> flag=False.
    3. Non-canonical motif with 4bp direct repeat -> flag=True.
    4. Non-canonical motif with only 3bp direct repeat -> flag=False.
    5. genome_fasta=None (graceful degradation) -> all rows score=0 flag=False.
    6. Single-exon transcript (SSC == "NA") -> skip, score=0 flag=False.
"""
import os
import sys
import unittest

import pandas as pd

# Ensure src/ is on sys.path so we can import the new module when the tests
# directory is the cwd (the standard `python tests/test_*.py` invocation).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from rt_switching_filter import (
    CANONICAL_JUNCTION_MOTIFS,
    RTSWITCHING_MICROHOMOLOGY_MIN_BP,
    detect_rt_switching,
)


class MockFastaFile:
    """Minimal stand-in for pysam.FastaFile.

    Stores a dict of chromosome -> sequence string. fetch(chrom, start, end)
    returns the substring [start:end] (0-based half-open, matching pysam).
    Sequences may include any letters (we uppercase at use-site).
    """

    def __init__(self, contigs):
        # contigs: dict[str, str]
        self._contigs = {k: v.upper() for k, v in contigs.items()}

    def fetch(self, chrom, start, end):
        seq = self._contigs[chrom]
        # pysam fetch is half-open [start, end)
        return seq[start:end]


def _make_df(rows):
    """Helper to build a minimal transcript dataframe with the columns we read."""
    return pd.DataFrame(rows)


class TestDetectRtSwitching(unittest.TestCase):
    # ------------------------------------------------------------------
    # Case 1: canonical motif only -> flag=False, score=0
    # ------------------------------------------------------------------
    def test_canonical_only_no_flag(self):
        # Construct a region where the only junction is GT-AG with no
        # microhomology. SSC = "200-400" -> one junction between 200 and 400.
        # Layout (1-based human, 0-based for the test):
        #   ...199 200 201 | intron | 399 400 401...
        # Donor  = last 2 bp of intron = positions (intron_end-1, intron_end)
        # Acceptor = first 2 bp of downstream exon = (exon_start, exon_start+1)
        # Want donor = "GT", acceptor = "AG" -> motif = "GT-AG".
        # Intron = positions 201..399 (inclusive). exon_start = 400.
        # Set donor positions: intron_end-1=198, intron_end=199 -> GT.
        # Set acceptor positions: 400, 401 -> AG.
        # And ensure no microhomology: intron tail (180..199) != exon head (400..419).
        seq = (
            "A" * 180
            + "GT"             # donor dinuc at 180..181
            + "C" * 18         # intron filler 182..199
            + "AG"             # acceptor dinuc at 200..201 -- wait, need to be careful
            + "TTTTTTTTTT"     # exon head 202..211
            + "C" * 200
        )
        # Simpler: hand-place by index. Reconstruct cleanly.
        # Intron occupies (intron_start, intron_end) = (201, 399) 0-based half-open.
        # Donor dinuc: positions (intron_end-2, intron_end) = (397, 399) -> "GT".
        # Acceptor dinuc: positions (exon_start, exon_start+2) = (400, 402) -> "AG".
        contig = ["N"] * 500
        # Donor at 397..398 = "GT"
        contig[397] = "G"; contig[398] = "T"
        # Acceptor at 400..401 = "AG"
        contig[400] = "A"; contig[401] = "G"
        # Avoid microhomology: intron tail 380..398 should not match exon head 400..418.
        for i in range(380, 397):
            contig[i] = "C"
        for i in range(402, 420):
            contig[i] = "T"
        fasta = MockFastaFile({"chr1": "".join(contig)})

        df = _make_df([
            {"Chr": "chr1", "Strand": "+", "SSC": "200-400", "TrStart": 150, "TrEnd": 450},
        ])
        out = detect_rt_switching(df, genome_fasta=fasta)

        self.assertIn("rt_switching_score", out.columns)
        self.assertIn("rt_switching_flag", out.columns)
        self.assertEqual(int(out.iloc[0]["rt_switching_score"]), 0)
        self.assertFalse(bool(out.iloc[0]["rt_switching_flag"]))

    # ------------------------------------------------------------------
    # Case 2: non-canonical motif, NO microhomology -> flag=False
    # ------------------------------------------------------------------
    def test_noncanonical_no_homology_no_flag(self):
        # Non-canonical donor (e.g., "AT") + non-canonical acceptor (e.g., "AC").
        # Intron_end = 399, exon_start = 400. Donor positions (397, 398) = "AT".
        # Acceptor positions (400, 401) = "AC". Motif = "AT-AC" is canonical!
        # Pick a truly non-canonical: donor = "CT", acceptor = "AC" -> "CT-AC".
        contig = ["N"] * 500
        contig[397] = "C"; contig[398] = "T"     # donor "CT"
        contig[400] = "A"; contig[401] = "C"     # acceptor "AC" -> motif "CT-AC"
        # No microhomology: ensure intron tail != exon head for any k>=1.
        for i in range(380, 397):
            contig[i] = "G"
        for i in range(402, 420):
            contig[i] = "T"
        fasta = MockFastaFile({"chr1": "".join(contig)})

        df = _make_df([
            {"Chr": "chr1", "Strand": "+", "SSC": "200-400", "TrStart": 150, "TrEnd": 450},
        ])
        out = detect_rt_switching(df, genome_fasta=fasta)

        self.assertEqual(int(out.iloc[0]["rt_switching_score"]), 0)
        self.assertFalse(bool(out.iloc[0]["rt_switching_flag"]))

    # ------------------------------------------------------------------
    # Case 3: non-canonical motif WITH 4bp direct repeat -> flag=True
    # ------------------------------------------------------------------
    def test_noncanonical_with_4bp_repeat_flagged(self):
        # Intron_end = 399, exon_start = 400. Donor at (398, 399), acceptor at (400, 401).
        # Donor = "CT", acceptor = "AC" -> motif "CT-AC" (non-canonical).
        # 4bp direct repeat: intron tail last 4bp == exon head first 4bp.
        #   intron tail positions (395..398) = "ACGC"
        #   exon head positions   (400..403) = "ACGC"
        # Position 398 is shared between repeat's last char and donor's first char.
        # So we need 398="C" -> repeat = "ACGC" and donor = "CT" (398="C", 399="T").
        contig = ["N"] * 500
        contig[395] = "A"; contig[396] = "C"; contig[397] = "G"; contig[398] = "C"
        contig[399] = "T"   # donor = "CT" (positions 398, 399)
        contig[400] = "A"; contig[401] = "C"; contig[402] = "G"; contig[403] = "C"
        # Ensure no longer 5bp repeat: position 394 must differ from 404.
        contig[394] = "T"
        contig[404] = "T"

        fasta = MockFastaFile({"chr1": "".join(contig)})

        df = _make_df([
            {"Chr": "chr1", "Strand": "+", "SSC": "200-400", "TrStart": 150, "TrEnd": 450},
        ])
        out = detect_rt_switching(df, genome_fasta=fasta)

        self.assertEqual(int(out.iloc[0]["rt_switching_score"]), 4)
        self.assertTrue(bool(out.iloc[0]["rt_switching_flag"]))

    # ------------------------------------------------------------------
    # Case 4: non-canonical motif with only 3bp direct repeat -> flag=False
    # ------------------------------------------------------------------
    def test_noncanonical_3bp_repeat_not_flagged(self):
        # Donor at (398, 399) must be "CT" (non-canonical).
        # 3bp repeat at positions (396..398) == (400..402). Since position 398
        # is shared with the donor (must be "C"), position 402 must also be "C".
        # So repeat = "ACC" with exon head = "A","C","C".
        # Ensure no 4bp repeat by making position 395 != position 403.
        contig = ["N"] * 500
        contig[396] = "A"; contig[397] = "C"; contig[398] = "C"
        contig[399] = "T"   # donor = "CT" (398="C", 399="T")
        contig[400] = "A"; contig[401] = "C"; contig[402] = "C"
        contig[395] = "T"; contig[403] = "T"  # 4bp would need 395==403 -> "TACC" vs "ACCT" -- no match

        fasta = MockFastaFile({"chr1": "".join(contig)})

        df = _make_df([
            {"Chr": "chr1", "Strand": "+", "SSC": "200-400", "TrStart": 150, "TrEnd": 450},
        ])
        out = detect_rt_switching(df, genome_fasta=fasta)

        self.assertEqual(int(out.iloc[0]["rt_switching_score"]), 3)
        self.assertFalse(bool(out.iloc[0]["rt_switching_flag"]))

    # ------------------------------------------------------------------
    # Case 5: genome_fasta=None graceful degradation
    # ------------------------------------------------------------------
    def test_genome_fasta_none_returns_zero_zero(self):
        df = _make_df([
            {"Chr": "chr1", "Strand": "+", "SSC": "200-400", "TrStart": 150, "TrEnd": 450},
            {"Chr": "chr1", "Strand": "-", "SSC": "100-300-500", "TrStart": 100, "TrEnd": 500},
        ])
        out = detect_rt_switching(df, genome_fasta=None)

        self.assertEqual(int(out.iloc[0]["rt_switching_score"]), 0)
        self.assertEqual(int(out.iloc[1]["rt_switching_score"]), 0)
        self.assertFalse(bool(out.iloc[0]["rt_switching_flag"]))
        self.assertFalse(bool(out.iloc[1]["rt_switching_flag"]))

    # ------------------------------------------------------------------
    # Case 6: Single-exon (SSC == "NA") -> skip microhomology
    # ------------------------------------------------------------------
    def test_single_exon_ssc_na_skipped(self):
        # Even with a fasta that WOULD show microhomology on exon flanks, single-
        # exon transcripts are skipped (SSC == "NA").
        fasta = MockFastaFile({"chr1": "A" * 1000})
        df = _make_df([
            {"Chr": "chr1", "Strand": "+", "SSC": "NA", "TrStart": 100, "TrEnd": 900},
        ])
        out = detect_rt_switching(df, genome_fasta=fasta)

        self.assertEqual(int(out.iloc[0]["rt_switching_score"]), 0)
        self.assertFalse(bool(out.iloc[0]["rt_switching_flag"]))

    # ------------------------------------------------------------------
    # Sanity: thresholds and motifs are exposed correctly.
    # ------------------------------------------------------------------
    def test_constants(self):
        self.assertEqual(RTSWITCHING_MICROHOMOLOGY_MIN_BP, 4)
        self.assertEqual(CANONICAL_JUNCTION_MOTIFS, frozenset({"GT-AG", "GC-AG", "AT-AC"}))

    # ------------------------------------------------------------------
    # Sanity: pre-existing columns are not modified (byte-identity contract).
    # ------------------------------------------------------------------
    def test_no_pre_existing_columns_modified(self):
        fasta = MockFastaFile({"chr1": "N" * 1000})
        df = _make_df([
            {"Chr": "chr1", "Strand": "+", "SSC": "200-400",
             "TrStart": 150, "TrEnd": 450, "extra_col": 42},
        ])
        before_cols = list(df.columns)
        before_extra = int(df.iloc[0]["extra_col"])
        out = detect_rt_switching(df, genome_fasta=fasta)
        # All original columns preserved with same values.
        for col in before_cols:
            self.assertIn(col, out.columns)
        self.assertEqual(int(out.iloc[0]["extra_col"]), before_extra)

    # ------------------------------------------------------------------
    # Case 7: Negative-strand microhomology (the bug-fix scenario).
    #
    # For - strand transcripts, the donor/acceptor dinucleotides and the
    # microhomology flanks must be reverse-complemented before the
    # comparison. With the bug, microhomology was being checked against the
    # *genomic* orientation, so a true transcript-space 4bp repeat would be
    # missed.
    #
    # Transcript: Strand="-", SSC="100-300-500" (two junctions).
    # Junction 1 (100-300): canonical "GT-AG" in transcript space -> skipped.
    # Junction 2 (300-500): non-canonical motif + 4bp transcript-space repeat.
    #
    # Junction 2 details:
    #   transcript donor      = "GG",  acceptor = "GG"  -> motif "GG-GG" (non-canonical)
    #   transcript-space 4bp repeat at the junction (right-flanking): "GGGG"
    #
    # Reverse_complement reverses order, so the transcript-space 4bp pattern
    # maps to the FIRST 4bp of the genomic intron_tail (positions 479-482)
    # and the LAST 4bp of the genomic exon_head (positions 516-519). We set
    # both ranges to "CCCC" -> rc produces "GGGG" on both sides.
    # ------------------------------------------------------------------
    def test_negative_strand_with_microhomology_flagged(self):
        contig = ["N"] * 600

        # Junction 1 (100-300): canonical motif "GT-AG" in transcript space.
        # rc(genomic 298-299) = "GT"  -> set 298="A", 299="C"
        # rc(genomic 300-301) = "AG"  -> set 300="C", 301="T"
        contig[298] = "A"; contig[299] = "C"
        contig[300] = "C"; contig[301] = "T"

        # Junction 2 (300-500): non-canonical motif "GG-GG" + 4bp transcript repeat.
        # Transcript-space 4bp repeat maps to genomic 479-482 and 516-519 = "CCCC".
        contig[479] = "C"; contig[480] = "C"; contig[481] = "C"; contig[482] = "C"
        contig[516] = "C"; contig[517] = "C"; contig[518] = "C"; contig[519] = "C"
        # Donor genomic 498-499 = "CC" -> rc = "GG"
        contig[498] = "C"; contig[499] = "C"
        # Acceptor genomic 500-501 = "CC" -> rc = "GG"
        contig[500] = "C"; contig[501] = "C"
        fasta = MockFastaFile({"chr1": "".join(contig)})

        df = _make_df([
            {"Chr": "chr1", "Strand": "-", "SSC": "100-300-500",
             "TrStart": 100, "TrEnd": 500},
        ])
        out = detect_rt_switching(df, genome_fasta=fasta)

        # Junction 2 non-canonical with k=4 -> row flagged.
        self.assertEqual(int(out.iloc[0]["rt_switching_score"]), 4)
        self.assertTrue(bool(out.iloc[0]["rt_switching_flag"]))

    # ------------------------------------------------------------------
    # Case 8: Negative-strand transcript, non-canonical motif, NO microhomology.
    # Same junction 1 + junction 2 layout as Case 7, but with the 4bp pattern
    # in the exon head changed to "TTTT" so no transcript-space repeat exists.
    # ------------------------------------------------------------------
    def test_negative_strand_no_microhomology_not_flagged(self):
        contig = ["N"] * 600

        # Junction 1 (100-300): canonical motif "GT-AG" in transcript space.
        contig[298] = "A"; contig[299] = "C"
        contig[300] = "C"; contig[301] = "T"

        # Junction 2 (300-500): non-canonical motif "GG-AA", no transcript repeat.
        # Donor genomic 498-499 = "CC" -> rc "GG"; acceptor genomic 500-501 = "TT" -> rc "AA".
        # For NO repeat, set genomic 479-482 = "CCCC" and genomic 516-519 = "TTTT".
        contig[479] = "C"; contig[480] = "C"; contig[481] = "C"; contig[482] = "C"
        contig[516] = "T"; contig[517] = "T"; contig[518] = "T"; contig[519] = "T"
        contig[498] = "C"; contig[499] = "C"
        contig[500] = "T"; contig[501] = "T"
        fasta = MockFastaFile({"chr1": "".join(contig)})

        df = _make_df([
            {"Chr": "chr1", "Strand": "-", "SSC": "100-300-500",
             "TrStart": 100, "TrEnd": 500},
        ])
        out = detect_rt_switching(df, genome_fasta=fasta)

        self.assertEqual(int(out.iloc[0]["rt_switching_score"]), 0)
        self.assertFalse(bool(out.iloc[0]["rt_switching_flag"]))

    # ------------------------------------------------------------------
    # Case 9: Positive-strand behavior is byte-equivalent to the pre-fix path.
    # The fix must not regress the + strand: when strand is "+", the
    # reverse_complement branch is skipped, so microhomology is computed on
    # the raw genomic windows exactly as before.
    # ------------------------------------------------------------------
    def test_positive_strand_baseline_unchanged(self):
        # Same construction as test_noncanonical_with_4bp_repeat_flagged but
        # explicit about Strand="+"; assert score and flag are byte-equivalent
        # to the original positive-strand contract.
        contig = ["N"] * 500
        contig[395] = "A"; contig[396] = "C"; contig[397] = "G"; contig[398] = "C"
        contig[399] = "T"   # donor = "CT" (398, 399)
        contig[400] = "A"; contig[401] = "C"; contig[402] = "G"; contig[403] = "C"
        contig[394] = "T"; contig[404] = "T"

        fasta = MockFastaFile({"chr1": "".join(contig)})

        df = _make_df([
            {"Chr": "chr1", "Strand": "+", "SSC": "200-400",
             "TrStart": 150, "TrEnd": 450},
        ])
        out = detect_rt_switching(df, genome_fasta=fasta)

        # Pre-fix behavior: score=4, flag=True for + strand with this layout.
        self.assertEqual(int(out.iloc[0]["rt_switching_score"]), 4)
        self.assertTrue(bool(out.iloc[0]["rt_switching_flag"]))


if __name__ == "__main__":
    unittest.main()
