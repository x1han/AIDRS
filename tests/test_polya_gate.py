"""Unit tests for the Stage 2.6 poly(A) criterion (the P2 wrap-up).

Pinned contracts:

- The default path's poly(A) joint gate is LIVE. It used to be dead in every
  production run because the evidence test demanded the ghost column
  ``polyA_valid_reads``, which the production path never produces for
  multi-exon rows.
- The default path and ``--hard_filter`` judge the poly(A) evidence with the
  SAME three states and the SAME decision:
    State 1  polyA_frac > 0   -- a poly(A) tail was detected
    State 2  polyA_frac == 0  -- the key was measured, no tail was detected
    State 3  polyA_frac NaN   -- no poly(A) evidence for this key
  State 3 bypasses (no measurement is never a penalty); State 2 needs an FSM
  rescue; State 1 passes above ``--polya_fraction_threshold``.
- "No poly(A) evidence in the frame" bypasses the criterion entirely (a pt-less
  BAM sets polyA_frac to NaN for the whole frame in Stage 2.3).
- ``disable_polya_filter`` is an escape hatch that switches the poly(A)
  criterion off; the 5' Puffin gate still applies.
- The 5' gate is type-tolerant: the captured / delivered assessment holds the
  Puffin score as TEXT, and the in-memory run and the TSV round-trip run must
  delete exactly the same rows.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.common import polya_criterion_mask, transcript_model_filtering


PUFFIN_THRESHOLD = 0.02
POLYA_THRESHOLD = 0.95


def _frame(rows):
    """A minimal Stage 2.6 input frame.

    Columns: TrID, Chr, Strand, TrStart, TrEnd, Puffin_TSS_15bp,
    Puffin_TSS_50bp, polyA_frac, Predict_NMD, truncation, category.
    """
    df = pd.DataFrame(rows)
    df['Chr'] = df['Chr'].astype(str)
    df['Strand'] = df['Strand'].astype(str)
    return df


def _kept(df, **kwargs):
    kwargs.setdefault('genome_fasta', None)
    kept = transcript_model_filtering(
        df, PUFFIN_THRESHOLD, POLYA_THRESHOLD, **kwargs)
    return set(kept['TrID'])


def _tiny_fasta(tmp_path, length=2000):
    """A minimal FASTA so the State 2 rescue block can run."""
    import pysam
    path = str(tmp_path / 'mini.fa')
    with open(path, 'w') as fh:
        fh.write('>chr1\n' + 'N' * length + '\n')
    pysam.faidx(path)
    return pysam.FastaFile(path)


# --------------------------------------------------------------------------
# 1) the shared poly(A) decision
# --------------------------------------------------------------------------

def test_polya_criterion_mask_judges_the_three_states():
    """The one decision both filter paths use (P1-4)."""
    polyA_frac = pd.Series([0.99, 0.50, 0.0, float('nan')], index=list('abcd'))
    mask_state1 = polyA_frac.notna() & polyA_frac.gt(0.0)
    mask_state2 = polyA_frac.notna() & polyA_frac.eq(0.0)
    mask_state3 = polyA_frac.isna()
    kept = pd.Series(False, index=list('abcd'))

    ok = polya_criterion_mask(mask_state1, mask_state2, polyA_frac,
                              POLYA_THRESHOLD, True, kept)
    assert list(ok) == [True, False, False, True]  # S1 pass / S1 fail / S2 fail / S3

    # an FSM rescue saves a measured zero
    ok = polya_criterion_mask(mask_state1, mask_state2, polyA_frac,
                              POLYA_THRESHOLD, True,
                              pd.Series([False, False, True, False], index=list('abcd')))
    assert list(ok) == [True, False, True, True]

    # no poly(A) evidence in the frame -> everything bypasses
    ok = polya_criterion_mask(mask_state1, mask_state2, polyA_frac,
                              POLYA_THRESHOLD, False, kept)
    assert list(ok) == [True, True, True, True]

    # the escape hatch -> everything bypasses
    ok = polya_criterion_mask(mask_state1, mask_state2, polyA_frac,
                              POLYA_THRESHOLD, True, kept,
                              disable_polya_filter=True)
    assert list(ok) == [True, True, True, True]


# --------------------------------------------------------------------------
# 2) the default path's poly(A) joint gate is live
# --------------------------------------------------------------------------

def test_default_path_deletes_a_weak_5prime_row_with_low_polya():
    """State 1 below the threshold + a weak 5' end -> deleted.

    Before the fix the poly(A) penalty was inert, so this row survived.
    """
    df = _frame([
        # poly(A) below the threshold AND a weak 5' end -> the joint gate hits
        dict(TrID='A', Chr='chr1', Strand='+', TrStart=100, TrEnd=500,
             Puffin_TSS_15bp='no', Puffin_TSS_50bp='no', polyA_frac=0.50,
             Predict_NMD='Normal', truncation='no', category='NNC'),
        # poly(A) below the threshold but a STRONG 5' end and no NMD /
        # truncation -> the poly(A) penalty alone must not delete it
        dict(TrID='B', Chr='chr1', Strand='+', TrStart=600, TrEnd=900,
             Puffin_TSS_15bp=0.9, Puffin_TSS_50bp=0.9, polyA_frac=0.50,
             Predict_NMD='Normal', truncation='no', category='NNC'),
        # poly(A) above the threshold, weak 5' end, no NMD / truncation -> kept
        dict(TrID='C', Chr='chr1', Strand='+', TrStart=1000, TrEnd=1300,
             Puffin_TSS_15bp='no', Puffin_TSS_50bp='no', polyA_frac=0.99,
             Predict_NMD='Normal', truncation='no', category='NNC'),
    ])
    assert _kept(df) == {'B', 'C'}


def test_default_and_hard_filter_agree_on_the_polya_verdict():
    """The two paths judge the poly(A) evidence identically (P1-4).

    The default path stays a JOINT decision (the poly(A) penalty only bites
    together with the 5' gate or with NMD / truncation), while --hard_filter
    is a plain AND of the two gates. For a row with a weak 5' end both paths
    must reach the same verdict on the poly(A) evidence.
    """
    rows = [
        # State 1, above the threshold -> passes the poly(A) criterion
        dict(TrID='S1_pass', Chr='chr1', Strand='+', TrStart=100, TrEnd=400,
             Puffin_TSS_15bp=0.9, Puffin_TSS_50bp=0.9, polyA_frac=0.99,
             Predict_NMD='Normal', truncation='no', category='NNC'),
        # State 1, below the threshold, weak 5' end -> fails it
        dict(TrID='S1_fail', Chr='chr1', Strand='+', TrStart=500, TrEnd=800,
             Puffin_TSS_15bp='no', Puffin_TSS_50bp='no', polyA_frac=0.50,
             Predict_NMD='Normal', truncation='no', category='NNC'),
        # State 2 (measured zero), no FSM rescue -> fails it
        dict(TrID='S2_fail', Chr='chr1', Strand='+', TrStart=900, TrEnd=1200,
             Puffin_TSS_15bp='no', Puffin_TSS_50bp='no', polyA_frac=0.0,
             Predict_NMD='Normal', truncation='no', category='NNC'),
        # State 3 (no evidence) -> bypass
        dict(TrID='S3_bypass', Chr='chr1', Strand='+', TrStart=1300, TrEnd=1600,
             Puffin_TSS_15bp=0.9, Puffin_TSS_50bp='no', polyA_frac=float('nan'),
             Predict_NMD='Normal', truncation='no', category='NNC'),
    ]
    df = _frame(rows)
    assert _kept(df) == _kept(df, hard_filter=True) == {'S1_pass', 'S3_bypass'}


def test_fsm_rescue_keeps_a_measured_zero(tmp_path):
    """A State 2 row with an FSM rescue passes the poly(A) criterion.

    The rescue needs the reference genome (the rescue block is guarded by
    it), so this test needs a FASTA. The row is NMD, which is what makes the
    poly(A) penalty bite on the default path -- a strong 5' end alone would
    keep it anyway and the test would be vacuous.
    """
    df = _frame([
        # State 2 + FSM + NMD -> the poly(A) penalty would delete it, the
        # rescue keeps it
        dict(TrID='FSM_zero', Chr='chr1', Strand='+', TrStart=100, TrEnd=400,
             Puffin_TSS_15bp=0.9, Puffin_TSS_50bp=0.9, polyA_frac=0.0,
             Predict_NMD='NMD', truncation='no', category='FSM'),
        # State 2 without a rescue + NMD -> deleted in both cases
        dict(TrID='NNC_zero', Chr='chr1', Strand='+', TrStart=500, TrEnd=800,
             Puffin_TSS_15bp=0.9, Puffin_TSS_50bp=0.9, polyA_frac=0.0,
             Predict_NMD='NMD', truncation='no', category='NNC'),
        # a positive value, so the frame counts as "measured"
        dict(TrID='tail', Chr='chr1', Strand='+', TrStart=900, TrEnd=1200,
             Puffin_TSS_15bp=0.9, Puffin_TSS_50bp=0.9, polyA_frac=0.99,
             Predict_NMD='Normal', truncation='no', category='NNC'),
    ])
    genome = _tiny_fasta(tmp_path)
    assert _kept(df, genome_fasta=genome) == {'FSM_zero', 'tail'}
    # without the genome there is no rescue -> the State 2 NMD row is deleted
    assert _kept(df) == {'tail'}
    assert _kept(df, genome_fasta=genome, hard_filter=True) == {'FSM_zero', 'tail'}


def test_add_polya_frac_leaves_an_uncovered_key_nan():
    """A key the per-key table does not cover carries NaN, not 0 (P2-3).

    This is the fillna(0) artifact: an uncovered key used to arrive as 0,
    which the criterion read as "measured, no tail detected" (State 2) and
    therefore deleted the model.
    """
    from src.polya_annotation import polyAnnotator

    models = pd.DataFrame([
        dict(TrID='covered', Chr='chr1', Strand='+', SSC='a,1,2,', TrStart=100, TrEnd=400),
        dict(TrID='uncovered', Chr='chr1', Strand='+', SSC='b,1,2,', TrStart=500, TrEnd=800),
    ])
    per_key = pd.DataFrame([
        dict(Chr='chr1', Strand='+', SSC='a,1,2,', TrStart=100, TrEnd=400,
             polyA_frac=0.87, polyA_frac_pooled=0.87, polyA_n_samples=1),
    ])
    out = polyAnnotator().add_polya_frac(models, per_key)

    covered = out[out['TrID'] == 'covered'].iloc[0]
    uncovered = out[out['TrID'] == 'uncovered'].iloc[0]
    assert covered['polyA_frac'] == 0.87
    assert pd.isna(uncovered['polyA_frac'])
    assert pd.isna(uncovered['polyA_frac_pooled'])
    assert uncovered['polyA_n_samples'] == 0


# --------------------------------------------------------------------------
# 3) the NaN bypass
# --------------------------------------------------------------------------

def test_state_2_is_penalised_but_state_3_bypasses():
    """Measured zero (State 2) and no evidence (State 3) must not be conflated.

    This is the production fillna(0) artifact: a key the per-key poly(A)
    table does not cover used to arrive as 0 and was treated as a measured
    zero (0.73 % of the models in the BC1-BC6 run).
    """
    df = _frame([
        dict(TrID='measured_zero', Chr='chr1', Strand='+', TrStart=100, TrEnd=400,
             Puffin_TSS_15bp='no', Puffin_TSS_50bp='no', polyA_frac=0.0,
             Predict_NMD='Normal', truncation='no', category='NNC'),
        dict(TrID='no_evidence', Chr='chr1', Strand='+', TrStart=500, TrEnd=800,
             Puffin_TSS_15bp='no', Puffin_TSS_50bp='no', polyA_frac=float('nan'),
             Predict_NMD='Normal', truncation='no', category='NNC'),
        # one positive value makes the frame "measured" (see the frame-level
        # evidence test), the other two rows are then judged on their state
        dict(TrID='measured_tail', Chr='chr1', Strand='+', TrStart=900, TrEnd=1200,
             Puffin_TSS_15bp=0.9, Puffin_TSS_50bp=0.9, polyA_frac=0.99,
             Predict_NMD='Normal', truncation='no', category='NNC'),
    ])
    kept = _kept(df)
    assert 'measured_zero' not in kept
    assert 'no_evidence' in kept
    assert 'measured_tail' in kept


def test_a_frame_without_any_positive_polya_value_is_not_penalised():
    """The frame-level evidence test: no positive value -> no poly(A) evidence.

    A frame whose every key is a measured zero carries no usable poly(A)
    evidence (a library basecalled without --estimate-poly-a, or a run whose
    estimator never produced a length), so the criterion stays inert instead
    of deleting the whole frame.
    """
    df = _frame([
        dict(TrID='zero_a', Chr='chr1', Strand='+', TrStart=100, TrEnd=400,
             Puffin_TSS_15bp='no', Puffin_TSS_50bp='no', polyA_frac=0.0,
             Predict_NMD='Normal', truncation='no', category='NNC'),
        dict(TrID='zero_b', Chr='chr1', Strand='+', TrStart=500, TrEnd=800,
             Puffin_TSS_15bp='no', Puffin_TSS_50bp='no', polyA_frac=0.0,
             Predict_NMD='Normal', truncation='no', category='NNC'),
    ])
    assert _kept(df) == {'zero_a', 'zero_b'}


def test_a_frame_without_polya_evidence_is_not_penalised():
    """A pt-less BAM (whole frame NaN) must not be penalised."""
    df = _frame([
        dict(TrID='X', Chr='chr1', Strand='+', TrStart=100, TrEnd=400,
             Puffin_TSS_15bp=0.9, Puffin_TSS_50bp=0.9, polyA_frac=float('nan'),
             Predict_NMD='Normal', truncation='no', category='NNC'),
        dict(TrID='Y', Chr='chr1', Strand='+', TrStart=500, TrEnd=800,
             Puffin_TSS_15bp='no', Puffin_TSS_50bp='no', polyA_frac=float('nan'),
             Predict_NMD='Normal', truncation='no', category='NNC'),
    ])
    assert _kept(df) == {'X', 'Y'}


# --------------------------------------------------------------------------
# 4) the escape hatch
# --------------------------------------------------------------------------

def test_disable_polya_filter_switches_the_criterion_off():
    """--disable_polya_filter removes the poly(A) penalty, nothing else."""
    df = _frame([
        # NMD + a poly(A) fraction below the threshold: the poly(A) penalty is
        # what deletes it -> kept again with the escape hatch
        dict(TrID='polyA_penalty', Chr='chr1', Strand='+', TrStart=100, TrEnd=400,
             Puffin_TSS_15bp=0.9, Puffin_TSS_50bp=0.9, polyA_frac=0.50,
             Predict_NMD='NMD', truncation='no', category='NNC'),
        # a weak 5' end: deleted by the 5' gate, with or without the escape
        # hatch
        dict(TrID='five_prime_only', Chr='chr1', Strand='+',
             TrStart=500, TrEnd=800, Puffin_TSS_15bp='no', Puffin_TSS_50bp=0.01,
             polyA_frac=0.99, Predict_NMD='NMD', truncation='no', category='NNC'),
        # poly(A) above the threshold, nothing else wrong -> kept either way
        dict(TrID='clean', Chr='chr1', Strand='+', TrStart=900, TrEnd=1200,
             Puffin_TSS_15bp=0.9, Puffin_TSS_50bp=0.9, polyA_frac=0.99,
             Predict_NMD='Normal', truncation='no', category='NNC'),
    ])
    assert _kept(df) == {'clean'}
    assert _kept(df, disable_polya_filter=True) == {'polyA_penalty', 'clean'}


# --------------------------------------------------------------------------
# 5) the TSV round trip
# --------------------------------------------------------------------------

def test_tsv_round_trip_deletes_the_same_rows(tmp_path):
    """The captured assessment holds the Puffin score as TEXT.

    The 5' gate must judge the text representation exactly like the in-memory
    float, otherwise a re-read run keeps rows that the producing run deleted.
    """
    df = _frame([
        # a numeric score BELOW the threshold: the numeric arm of the gate
        dict(TrID='numeric_below', Chr='chr1', Strand='+', TrStart=100, TrEnd=400,
             Puffin_TSS_15bp=0.01, Puffin_TSS_50bp=0.01, polyA_frac=0.99,
             Predict_NMD='NMD', truncation='no', category='NNC'),
        # a numeric score above the threshold
        dict(TrID='numeric_above', Chr='chr1', Strand='+', TrStart=500, TrEnd=800,
             Puffin_TSS_15bp=0.9, Puffin_TSS_50bp=0.9, polyA_frac=0.99,
             Predict_NMD='Normal', truncation='no', category='NNC'),
        # the literal 'no'
        dict(TrID='no_peak', Chr='chr1', Strand='+', TrStart=900, TrEnd=1200,
             Puffin_TSS_15bp='no', Puffin_TSS_50bp='no', polyA_frac=0.99,
             Predict_NMD='NMD', truncation='no', category='NNC'),
        # a State 3 row
        dict(TrID='nan_row', Chr='chr1', Strand='+', TrStart=1300, TrEnd=1600,
             Puffin_TSS_15bp=0.9, Puffin_TSS_50bp=0.9, polyA_frac=float('nan'),
             Predict_NMD='Normal', truncation='no', category='NNC'),
    ])
    path = tmp_path / 'captured.tsv'
    df.to_csv(path, sep='\t', index=False)
    reread = pd.read_csv(path, sep='\t')

    assert not pd.api.types.is_numeric_dtype(reread['Puffin_TSS_50bp'])
    assert _kept(df) == _kept(reread)


def test_tsv_round_trip_is_identical_for_the_hard_filter_path(tmp_path):
    df = _frame([
        dict(TrID='a', Chr='chr1', Strand='+', TrStart=100, TrEnd=400,
             Puffin_TSS_15bp=0.01, Puffin_TSS_50bp=0.9, polyA_frac=0.99,
             Predict_NMD='Normal', truncation='no', category='NNC'),
        dict(TrID='b', Chr='chr1', Strand='+', TrStart=500, TrEnd=800,
             Puffin_TSS_15bp=0.9, Puffin_TSS_50bp=0.9, polyA_frac=0.5,
             Predict_NMD='Normal', truncation='no', category='NNC'),
    ])
    path = tmp_path / 'captured.tsv'
    df.to_csv(path, sep='\t', index=False)
    reread = pd.read_csv(path, sep='\t')
    assert _kept(df, hard_filter=True) == _kept(reread, hard_filter=True)
