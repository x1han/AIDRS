"""Unit tests for scripts/verify_c107_golden.py (the C107 100k regression gate).

The gate has two accepted forms (see the expert review, sections 1.3-6 and 5.2):
(a) the assessment sha256 hits the pinned golden, or
(b) the change is bounded: the five quantification files are byte-identical,
    the assessment differs only in {TrStart, TrEnd, seq_len, TIS_score,
    TTS_score}, every 5' change is attributable to a new 3' variant, no model
    moved its 5' end while its 3' end stayed put, at most 1 % of the models
    moved their 3' end, and the score columns agree within 1e-5.

These tests use synthetic frames; they check the *logic* of the gate, not the
C107 data itself. The frames are large enough (200 models) that a single
changed model stays below the 1 % bound, exactly like the C107 golden
(1 changed model out of 1019).
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("verify_c107_golden", TOOLS / "verify_c107_golden.py")
vgate = importlib.util.module_from_spec(spec)
sys.modules["verify_c107_golden"] = vgate
spec.loader.exec_module(vgate)

COLUMNS = [
    "Chr", "Strand", "SSC", "TrStart", "TrEnd", "frequency", "polyA_frac",
    "TIS_score", "TTS_score", "TrID", "GeneID", "GeneName", "seq_len",
]

N_MODELS = 200


def _frame(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def _write_dir(tmp_path, name, frame, quant_bytes=b"same\n"):
    d = tmp_path / name
    d.mkdir()
    frame.to_csv(d / vgate.ASSESSMENT, sep="\t", index=False)
    for fname in vgate.QUANT_FILES:
        (d / fname).write_bytes(quant_bytes)
    return d


@pytest.fixture
def base_frame():
    """200 single-model keys plus one key with two models (a 3' variant)."""
    rows = []
    for i in range(N_MODELS):
        rows.append(dict(
            Chr="chr1", Strand="+", SSC=f"{100 + i}-{200 + i}", TrStart=100 + i,
            TrEnd=1000 + i, frequency=10, polyA_frac=1.0, TIS_score=0.9,
            TTS_score=0.8, TrID=f"Gene{i}_Tr0", GeneID=f"Gene{i}",
            GeneName=f"Gene{i}", seq_len=900 + i,
        ))
    rows.append(dict(
        Chr="chr1", Strand="+", SSC="500-600", TrStart=500, TrEnd=1500,
        frequency=10, polyA_frac=1.0, TIS_score=0.9, TTS_score=0.8,
        TrID="GeneX_Tr0", GeneID="GeneX", GeneName="GeneX", seq_len=1000,
    ))
    rows.append(dict(
        Chr="chr1", Strand="+", SSC="500-600", TrStart=500, TrEnd=1600,
        frequency=10, polyA_frac=1.0, TIS_score=0.9, TTS_score=0.8,
        TrID="GeneX_Tr1", GeneID="GeneX", GeneName="GeneX", seq_len=1100,
    ))
    return _frame(rows)


def _bump_three_prime(frame, tid, delta=1):
    out = frame.copy()
    out.loc[out["TrID"] == tid, "TrEnd"] += delta
    out.loc[out["TrID"] == tid, "seq_len"] += delta
    return out


# --- mode (a): the pinned sha256 -------------------------------------------

def test_sha_only_passes_on_the_pinned_golden(tmp_path, monkeypatch, base_frame):
    d = _write_dir(tmp_path, "cand", base_frame)
    monkeypatch.setattr(vgate, "GOLDEN_SHA256", vgate.sha256_file(d / vgate.ASSESSMENT))
    assert vgate.main(["--candidate", str(d), "--sha-only"]) == 0


def test_sha_only_fails_on_a_different_sha(tmp_path, base_frame):
    d = _write_dir(tmp_path, "cand", base_frame)
    assert vgate.main(["--candidate", str(d), "--sha-only"]) == 1


def test_mode_b_needs_a_baseline_when_the_sha_misses(tmp_path, base_frame):
    d = _write_dir(tmp_path, "cand", base_frame)
    assert vgate.main(["--candidate", str(d)]) == 1


# --- mode (b): bounded changes ---------------------------------------------

def test_mode_b_accepts_a_legitimate_three_prime_only_change(tmp_path, base_frame):
    cand = _bump_three_prime(base_frame, "Gene3_Tr0")
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 0


def test_mode_b_accepts_a_moved_five_prime_that_accompanies_a_moved_three_prime(tmp_path, base_frame):
    """criterion (i): the new 5' end is the partner end of a new 3' variant."""
    cand = base_frame.copy()
    cand.loc[cand["TrID"] == "Gene3_Tr0", "TrStart"] = 333
    cand.loc[cand["TrID"] == "Gene3_Tr0", "TrEnd"] = 1004
    cand.loc[cand["TrID"] == "Gene3_Tr0", "seq_len"] = 672
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 0


def test_mode_b_rejects_a_five_prime_move_without_a_three_prime_move(tmp_path, base_frame):
    """criterion (ii): 3' unchanged, 5' moved -> must fail."""
    cand = base_frame.copy()
    cand.loc[cand["TrID"] == "Gene3_Tr0", "TrStart"] = 333
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 1


def test_mode_b_rejects_a_changed_column_outside_the_allowed_set(tmp_path, base_frame):
    cand = base_frame.copy()
    cand.loc[cand["TrID"] == "Gene3_Tr0", "polyA_frac"] = 0.5
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 1


def test_mode_b_rejects_more_than_one_percent_three_prime_changes(tmp_path, base_frame):
    cand = base_frame.copy()
    for i in range(5):  # 5 of 202 models = 2.5 % > 1 %
        cand.loc[cand["TrID"] == f"Gene{i}_Tr0", "TrEnd"] += 1
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 1


def test_mode_b_accepts_exactly_one_percent_three_prime_changes(tmp_path, base_frame):
    """2 of 202 models = 0.99 % -> at the bound, must pass."""
    cand = base_frame.copy()
    for i in range(2):
        cand.loc[cand["TrID"] == f"Gene{i}_Tr0", "TrEnd"] += 1
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 0


def test_mode_b_is_not_vacuous_when_every_model_moves_its_five_prime_end(tmp_path, base_frame):
    """The power test: all 5' ends move, all 3' ends stay -> must FAIL."""
    cand = base_frame.copy()
    cand["TrStart"] = cand["TrStart"] + 1000
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 1


def test_mode_b_is_not_vacuous_with_an_na_ssc(tmp_path, base_frame):
    """SSC == 'NA' must stay a string; with NaN keys every lookup would miss."""
    base = base_frame.copy()
    base["SSC"] = "NA"
    cand = base.copy()
    cand["TrStart"] = cand["TrStart"] + 1000
    b = _write_dir(tmp_path, "base", base)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 1


def test_mode_b_fails_when_no_model_can_be_matched(tmp_path, base_frame):
    cand = base_frame.copy()
    cand["SSC"] = "other-locus"
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 1


def test_mode_b_rejects_a_score_that_becomes_nan(tmp_path, base_frame):
    """A NaN must not slip through the 1e-5 bound."""
    cand = base_frame.copy()
    cand.loc[cand["TrID"] == "Gene3_Tr0", "TTS_score"] = float("nan")
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 1


def test_mode_b_rejects_a_dropped_assessment_column(tmp_path, base_frame):
    cand = base_frame.drop(columns=["frequency"])
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 1


def test_mode_b_rejects_an_added_assessment_column(tmp_path, base_frame):
    cand = base_frame.copy()
    cand["extra"] = 1
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 1


def test_mode_b_ignores_a_numeric_repr_difference(tmp_path, base_frame):
    """10 and 10.0 are the same value, not a content change."""
    cand = base_frame.copy()
    cand["frequency"] = cand["frequency"].astype(float)
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 0


def test_baseline_sha_mismatch_fails_loudly(tmp_path, base_frame):
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", base_frame)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c),
                        "--baseline-sha", "0" * 64]) == 1
    good = vgate.sha256_file(b / vgate.ASSESSMENT)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c),
                        "--baseline-sha", good]) == 0


def test_mode_a_also_compares_the_quantification_files(tmp_path, monkeypatch, base_frame):
    """A regression confined to the counts/CPM layer must not pass mode (a)."""
    d = _write_dir(tmp_path, "cand", base_frame)
    monkeypatch.setattr(vgate, "GOLDEN_SHA256", vgate.sha256_file(d / vgate.ASSESSMENT))
    b = _write_dir(tmp_path, "base", base_frame)
    assert vgate.main(["--baseline", str(b), "--candidate", str(d)]) == 0
    (d / "aidrs.transcript.counts.tsv").write_bytes(b"corrupted\n")
    assert vgate.main(["--baseline", str(b), "--candidate", str(d)]) == 1


def test_criterion_i_can_fire_alone(tmp_path, base_frame):
    """Criterion (i) alone: a model with a new 5' but the key's old 3'."""
    base = base_frame.copy()
    cand = base.copy()
    # a model that keeps its 3' end but gets a new 5' end
    cand.loc[cand["TrID"] == "Gene3_Tr0", "TrStart"] = 333
    viol_i, viol_ii, n_matched = vgate.check_five_prime_criteria(base, cand)
    assert viol_i == 1
    assert viol_ii == 1          # the same model also trips criterion (ii)
    # a model that is NOT matched by id (new TrID) and keeps the old 3'
    cand2 = base.copy()
    cand2.loc[cand2["TrID"] == "Gene3_Tr0", "TrID"] = "Gene3_TrX"
    cand2.loc[cand2["TrID"] == "Gene3_TrX", "TrStart"] = 333
    viol_i2, viol_ii2, n_matched2 = vgate.check_five_prime_criteria(base, cand2)
    assert viol_i2 == 1
    assert viol_ii2 == 0         # criterion (ii) needs a matched model
    assert n_matched2 == n_matched - 1


def test_receipt_reports_the_counts(tmp_path, base_frame, capsys):
    cand = base_frame.copy()
    cand.loc[cand["TrID"] == "Gene3_Tr0", "TrEnd"] += 1
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    vgate.main(["--baseline", str(b), "--candidate", str(c), "--baseline-label", "abc123"])
    out = capsys.readouterr().out
    assert "models: baseline=" in out
    assert "5' criteria: violations(i)" in out
    assert "3' side changed:" in out
    assert "abc123" in out
    assert "candidate assessment sha256:" in out
    assert "baseline  assessment sha256:" in out


def test_mode_b_rejects_a_score_difference_above_the_tolerance(tmp_path, base_frame):
    cand = base_frame.copy()
    cand.loc[cand["TrID"] == "Gene3_Tr0", "TTS_score"] = 0.8 + 1e-3
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 1


def test_mode_b_accepts_a_score_difference_below_the_tolerance(tmp_path, base_frame):
    cand = base_frame.copy()
    cand.loc[cand["TrID"] == "Gene3_Tr0", "TTS_score"] = 0.8 + 1e-6
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 0


def test_mode_b_rejects_a_differing_quantification_file(tmp_path, base_frame):
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", base_frame)
    (c / "aidrs.transcript.counts.tsv").write_bytes(b"different\n")
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 1


def test_mode_b_rejects_a_different_row_count(tmp_path, base_frame):
    cand = base_frame.iloc[:-1].copy()
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 1


def test_mode_b_is_not_fooled_by_a_row_reordering(tmp_path, base_frame):
    cand = base_frame.iloc[::-1].reset_index(drop=True)
    b = _write_dir(tmp_path, "base", base_frame)
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 0


# --- strand awareness -------------------------------------------------------

def test_minus_strand_uses_trend_as_the_five_prime_end(tmp_path):
    rows = []
    for i in range(N_MODELS):
        rows.append(dict(
            Chr="chr1", Strand="-", SSC=f"{100 + i}-{200 + i}", TrStart=100 + i,
            TrEnd=1000 + i, frequency=10, polyA_frac=1.0, TIS_score=0.9,
            TTS_score=0.8, TrID=f"GeneM{i}_Tr0", GeneID=f"GeneM{i}",
            GeneName=f"GeneM{i}", seq_len=900 + i,
        ))
    base = _frame(rows)
    b = _write_dir(tmp_path, "base", base)

    # a 3' change on the minus strand moves TrStart (the 3' end), not TrEnd
    cand = base.copy()
    cand.loc[cand["TrID"] == "GeneM3_Tr0", "TrStart"] = 104
    c = _write_dir(tmp_path, "cand", cand)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c)]) == 0

    # a 5' change on the minus strand moves TrEnd only -> criterion (ii)
    cand2 = base.copy()
    cand2.loc[cand2["TrID"] == "GeneM3_Tr0", "TrEnd"] = 1004
    c2 = _write_dir(tmp_path, "cand2", cand2)
    assert vgate.main(["--baseline", str(b), "--candidate", str(c2)]) == 1


# --- the reference sample count guard --------------------------------------
#
# The pinned golden's polyA_frac can only stay put while at most one library
# contributes reads to a key, so the gate prints the library count (read from
# the quantification header) and warns when the reference run is multi-library.

def _write_counts_with_libraries(d, n_libraries):
    header = ["TrID", "GeneID", "GeneName"] + [f"lib{i}.bam" for i in range(n_libraries)]
    (d / "aidrs.transcript.counts.tsv").write_text("\t".join(header) + "\n")


def test_the_reference_sample_count_comes_from_the_quantification_header(tmp_path):
    d = tmp_path / "cand"
    d.mkdir()
    _write_counts_with_libraries(d, 1)
    assert vgate.count_reference_samples(d) == 1
    _write_counts_with_libraries(d, 6)
    assert vgate.count_reference_samples(d) == 6


def test_a_multi_library_reference_run_is_flagged(tmp_path, capsys):
    d = tmp_path / "cand"
    d.mkdir()
    _write_counts_with_libraries(d, 6)
    vgate.print_sample_count(d)
    out = capsys.readouterr().out
    assert "reference sample count: 6" in out
    assert "WARNING" in out


def test_a_single_library_reference_run_is_not_flagged(tmp_path, capsys):
    d = tmp_path / "cand"
    d.mkdir()
    _write_counts_with_libraries(d, 1)
    vgate.print_sample_count(d)
    out = capsys.readouterr().out
    assert "reference sample count: 1" in out
    assert "WARNING" not in out


def test_an_unreadable_quantification_file_is_reported_as_unknown(tmp_path, capsys):
    d = tmp_path / "cand"
    d.mkdir()
    vgate.print_sample_count(d)
    out = capsys.readouterr().out
    assert "unknown" in out
