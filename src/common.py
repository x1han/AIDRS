#!/usr/bin/env python

import logging
import pandas as pd
import numpy as np
import multiprocessing
import re
import subprocess
import sys
import os
import gc
import glob
from collections import defaultdict
from typing import Optional, Any
from .gene_grouping import GeneClustering
from .get_terminal_sites import TerminalSitesProcessor

logger = logging.getLogger("AIDRS")

_COMPLEMENT = {
    "A": "T", "C": "G", "G": "C", "T": "A", "N": "N",
    "a": "t", "c": "g", "g": "c", "t": "a", "n": "n",
    "R": "Y", "Y": "R", "S": "S", "W": "W", "K": "M", "M": "K",
    "B": "V", "D": "H", "H": "D", "V": "B",
    "r": "y", "y": "r", "s": "s", "w": "w", "k": "m", "m": "k",
    "b": "v", "d": "h", "h": "d", "v": "b",
    "-": "-", ".": ".", "*": "*",
}
_RC_TRANS = str.maketrans("".join(_COMPLEMENT.keys()), "".join(_COMPLEMENT.values()))

def rev_comp(seq):
    """Reverse-complement a DNA/RNA sequence, including IUPAC ambiguous bases.

    Covers A/C/G/T/N + 11 IUPAC codes (RYSWKMBDHV) in upper/lower + soft-mask
    characters (-.*). Performance equivalent to plain reverse-complement via
    str.translate + slicing.
    """
    if seq is None:
        return ""
    return str(seq).translate(_RC_TRANS)[::-1]

def safe_ssc_array(s):
    """Parse SSC string to int array; safe for 'NA'/'none'/empty/None.

    Returns empty int64 array when input is unparseable, never raises.
    """
    if not isinstance(s, str) or s in ('NA', 'none', ''):
        return np.array([], dtype=np.int64)
    try:
        return np.array(list(map(int, s.split('-'))), dtype=np.int64)
    except (ValueError, AttributeError):
        return np.array([], dtype=np.int64)

def safe_int_tuple(s):
    """Parse SSC-like 'a-b-c' string to int tuple; safe for non-integer tokens.

    Returns None on parse failure (caller decides fallback).
    """
    if not isinstance(s, str) or s in ('NA', 'none', ''):
        return None
    try:
        return tuple(int(p) for p in s.split('-'))
    except (ValueError, AttributeError):
        return None

def run_bam2ssc(reference, bam, output_ssc, num_threads):
    """
    bam2ssc
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    bam2SSC_script = os.path.join(current_dir, 'bam2ssc.py')
    cmd = [sys.executable, bam2SSC_script,
        "-r", reference,
        "-b", *bam,
        "-o", output_ssc,
        "-t", str(num_threads)]
    
    subprocess.run(cmd, check=True)

def run_Ref2SSC(gtf_anno, output, num_threads):
    """
    anno2ssc
    """
    process_dir = os.path.join(output, "temp")
    os.makedirs(process_dir, exist_ok=True)

    output_SSC = os.path.join(process_dir, "anno.ssc")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    gtf2SSC_script = os.path.join(current_dir,'gtf2ssc.py')
    cmd = [sys.executable, gtf2SSC_script,
        "-i", gtf_anno,
        "-o", output_SSC,
        "-w", str(num_threads)]
    subprocess.run(cmd, check=True)

def read_flnc(flnc_path):
    dtypes_flnc = {
        1: "category",  # Chr
        2: "category",  # Strand
        3: "int32",     # TrStart_reads
        4: "int32",     # TrEnd_reads
        5: str,         # SSC
        6: "float32",   # identity
        7: "float32",   # coverage
        8: "int32"      # polyA_len
    }

    df_flnc = pd.read_csv(
        flnc_path,
        sep="\t",
        header=None,
        index_col=0,
        dtype=dtypes_flnc,
        usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8], low_memory=True
    )
    df_flnc.columns = ["Chr", "Strand", "TrStart_reads", "TrEnd_reads", "SSC", "identity", "coverage", "polyA_len"]

    return df_flnc

def process_data(flnc_path, count_path, df_raw_path, min_aln_coverage=None, min_aln_identity=None):
    """
    read preprocessing
    """
    df_flnc = read_flnc(flnc_path)

    if min_aln_coverage is not None and min_aln_identity is not None:
        df_flnc = df_flnc[(df_flnc["identity"] >= min_aln_identity) & (df_flnc["coverage"] >= min_aln_coverage)]
    
    df_flnc = df_flnc.drop(columns=["identity", "coverage"]).dropna()

    df_grouped = (
        df_flnc.groupby(["Chr", "Strand", "SSC"], observed=True)
        .agg({"TrStart_reads": list, "TrEnd_reads": list})
        .reset_index()
    )

    df_grouped["TrStart_reads"] = df_grouped["TrStart_reads"].apply(lambda x: np.array(x, dtype=np.int32))
    df_grouped["TrEnd_reads"] = df_grouped["TrEnd_reads"].apply(lambda x: np.array(x, dtype=np.int32))
    df_grouped["frequency"] = df_grouped["TrStart_reads"].apply(len)
    df_grouped.to_parquet(df_raw_path)

    del df_flnc
    gc.collect()

    df_junction = pd.read_csv(
        count_path,
        sep="\t",
        header=None,
        dtype={1: "category", 2: "category", 3: str, 4: str},
        usecols=[1, 2, 3, 4]
    )

    df_junction.columns = ["Chr", "Strand", "SSC", "junction"]
    df = df_grouped.merge(df_junction, on=["Chr", "Strand", "SSC"], how="left")
    df['junction'] = df['junction'].fillna('none')

    del df_grouped
    gc.collect()

    return df

def load_data(reference, bam, output, num_threads, min_aln_coverage, min_aln_identity):
    """
    load ssc data
    """
    process_dir = os.path.join(output, "temp")
    os.makedirs(process_dir, exist_ok=True)
    sample = os.path.splitext(os.path.basename(bam))[0] 
    output_flnc = os.path.join(process_dir, f"{sample}_flnc.ssc")
    output_count = os.path.join(process_dir, f"{sample}_ssc.count")
    df_raw_path = os.path.join(process_dir, f"{sample}.ssc_flnc.parquet")

    df = process_data(
        flnc_path=output_flnc,
        count_path=output_count,
        df_raw_path=df_raw_path,
        min_aln_coverage=min_aln_coverage,
        min_aln_identity=min_aln_identity
    )

    return df

def junction_screening(df, junction_freq_ratio, conservative_base=None):
    """
    fiter non-canonical splice motifs
    """
    if conservative_base is None:
        conservative_base = {'GT-AG', 'AT-AC', 'GC-AG'}
    else:
        conservative_base = set(conservative_base.split(','))
    
    df = df.copy()
    def contains_non_conservative(junction_str):
        junctions = {junc.upper() for junc in junction_str.split(',')}
        return not junctions.issubset(conservative_base)

    df['contains_non_conservative'] = df['junction'].apply(contains_non_conservative)
    # P1-6: SINGLE_EXON_GROUP_SENTINEL (-2147483648) is a placeholder Group id
    # shared by all single-exon rows; summing across them inflates Group_freq
    # and makes freq_ratio always tiny, which causes
    # (contains_non_conservative & freq_ratio <= threshold) to drop every
    # single-exon row with a non-conservative junction.
    # Fix: only multi-exon rows (Group >= 0) participate in the group sum;
    # single-exon rows use their own frequency as Group_freq.
    valid_group_mask = df['Group'] >= 0
    group_freq_sum = df[valid_group_mask].groupby(['Chr', 'Strand', 'Group'], observed=True)['frequency'].transform('sum')
    df.loc[valid_group_mask, 'Group_freq'] = group_freq_sum
    df.loc[~valid_group_mask, 'Group_freq'] = df.loc[~valid_group_mask, 'frequency']
    df['freq_ratio'] = df['frequency'] / df['Group_freq']

    df = df[~((df['contains_non_conservative']) & (df['freq_ratio'] <= junction_freq_ratio))]
    df.drop(columns=['contains_non_conservative', 'Group_freq', 'freq_ratio'], inplace=True)

    return df


def filter_fragmentary_transcript(df, threshold_fragmentary_transcript_bp=50):
    """
    filter fragmentary transcript
    """
    conservative_base = {'GT-AG', 'AT-AC', 'GC-AG'}

    def contains_non_conservative(junction_str):
        return not set(junction_str.split(',')).issubset(conservative_base)
    
    df['contains_non_conservative'] = df['junction'].apply(contains_non_conservative)
    total_meanfreq = df['frequency'].sum() / len(df)

    df['TrStart_mean'] = df['TrStart_reads'].apply(np.mean)
    df['TrEnd_mean'] = df['TrEnd_reads'].apply(np.mean)
    df['SSC2'] = df['SSC'].apply(lambda x: safe_ssc_array(x).tolist())
    
    df['Tr_length_min'] = df.apply(
        lambda row: np.inf if len(row['SSC2']) > 2 else min([
            # Distance from transcript 5' end to first internal splice site
            # and from last internal splice site to transcript 3' end.
            # On + strand, both quantities are positive (exon lies between
            # TrStart and TrEnd). On - strand, SSC sites ascend in genomic
            # coord as we move 5'->3' in RNA space (i.e. descend in genomic
            # coord), so the raw difference is negative; use abs() to get
            # the physical transcript length on both strands.
            abs(row['SSC2'][0] - row['TrStart_mean']),
            abs(row['TrEnd_mean'] - row['SSC2'][1])
        ]),
        axis=1
    )

    df = df[~(((df['Tr_length_min'] < threshold_fragmentary_transcript_bp) & (df['frequency'] < 0.01 * total_meanfreq)) |
              (df['contains_non_conservative']) & (df['frequency'] < 0.01 * total_meanfreq))]
    
    return df

def transcript_model_filtering(df, puffin_prediction_threshold=0.02, polya_fraction_threshold=0.95, hard_filter=False, genome_fasta=None):
    """
    Apply TSS correction and filtering to dataframe

    Requirements:
    1. After obtaining the tss_col corresponding to strand, if the tuple in the Puffin_TSS_15bp column does not contain 'no',
       then tss_col is modified to tss_col+Puffin_TSS_15bp[0]
    2. If the value in the Predict_NMD column is 'NMD' and (Puffin_TSS_50bp value has 'no' or polyA_frac value is less than polya_fraction_threshold),
       then delete this row
    3. If the value in the truncation column is 'yes' and (Puffin_TSS_50bp value has 'no' or polyA_frac value is less than polya_fraction_threshold),
       then delete this row

    Parameters
    ----
    genome_fasta : pysam.FastaFile or None
        Reference genome FASTA, required ONLY for the 3-state polyA rescue
        (Stage 2.5b). When None (default), behavior is byte-identical to the
        legacy single-state polyA logic (State 2 -> filter, State 3 -> bypass).
        When provided, State 2 rows (polyA_frac == 0, measured but all-zero)
        are subject to:
          - FSM rescue: keep when category == 'FSM' (e.g. replication-dependent
            histone mRNAs that are bona fide non-polyadenylated reference
            isoforms).
          - Genomic intra-priming check (SQANTI3 standard: +20bp downstream
            A-rich window). If intra-priming is detected, filter (oligo-dT
            primed at internal A-rich region -> artifact). If NOT detected,
            keep (genuine non-polyadenylated transcript).
    """
    # Capture input row count before any filtering for Stage 2.6 observability
    n_before = len(df)

    # ----------------------------------------------------------------------
    # Sample-level polyA modality check (Stage A R2 fix for -polyA datasets)
    # ----------------------------------------------------------------------
    # When the entire sample lacks polyA measurements (e.g., BAM without
    # pt:i tags, or polyA module never produced any signal), bypass all
    # polyA penalties rather than treating "missing data" as a death
    # penalty. This addresses the failure mode observed in Case 2 (-polyA
    # C107 chr1): polyA_valid_reads=0 for every row -> State 2 mask all
    # True -> polya_frac_low_rowwise all True -> all three gates (NMD,
    # truncation, ultra-low-quality) trigger unconditionally -> 75% row
    # loss (5954 -> 1461). With this guard, the 3-condition framework
    # collapses to the 5' Puffin-only gate, restoring the user's intent
    # that "no polyA info" is neutral, not penalising.
    has_polya_info = (
        ("polyA_valid_reads" in df.columns)
        and (df["polyA_valid_reads"].fillna(0).sum() > 0)
        and df["polyA_frac"].notna().any()
    )

    # Lazy import intra-priming checker ONLY when genome_fasta is provided.
    # Keeps the legacy byte-identical path free of new module-level imports.
    _intra_priming_check = None
    if genome_fasta is not None:
        from .single_exon import check_genomic_intra_priming as _intra_priming_check

    # 3-state polyA observability counters (cumulative across groups)
    state_counts = {'state1_pos': 0, 'state2_zero': 0, 'state2_fsm_rescue': 0,
                   'state2_intra_priming_observed': 0, 'state2_non_fsm_total': 0,
                   'state3_missing': 0}

    # Process by grouping on Chr and Strand
    df_groups = [g for _, g in df.groupby(['Chr','Strand'], observed=True)]

    processed_groups = []
    for df_group in df_groups:
        # Copy data to avoid direct modification
        df_group = df_group.copy()

        if puffin_prediction_threshold == 0:
            df_group['Puffin_TSS_15bp'] = 1
            df_group['Puffin_TSS_50bp'] = 1

        # Get strand information for the current group
        strand = df_group['Strand'].iloc[0]
        tss_col = 'TrStart' if strand == '+' else 'TrEnd'

        # ------------------------------------------------------------------
        # 3-state polyA mask construction (shared by both filter paths below)
        # ------------------------------------------------------------------
        # State 1 (positive): polyA_frac > 0 (some reads had pt:i > 0)
        # State 2 (measured zero): polyA_frac == 0 AND polyA_valid_reads == 0
        #   (all reads had pt:i = 0 -- strong experimental negative signal)
        # State 3 (missing): polyA_frac is NaN (polyA was not measured at all)
        # ------------------------------------------------------------------
        polya_frac = df_group['polyA_frac']
        polyA_valid_reads = df_group.get('polyA_valid_reads', pd.Series([0] * len(df_group), index=df_group.index))
        polyA_valid_reads = pd.to_numeric(polyA_valid_reads, errors='coerce').fillna(0).astype(int)

        mask_state1 = polya_frac.notna() & polya_frac.gt(0.0)
        mask_state2 = polya_frac.notna() & polya_frac.eq(0.0) & polyA_valid_reads.eq(0)
        mask_state3 = polya_frac.isna()

        # State 2 rescue masks (only meaningful when genome_fasta is provided)
        # Per Q-A2 expert review: ONLY FSM gets unconditional rescue. Non-FSM
        # State 2 rows (polyA_frac == 0, measured zero) are routed through the
        # 3-condition framework (NMD / truncation / ultra-low-quality) so they
        # must demonstrate a strong 5' Puffin signal to survive. Intra-priming
        # is kept as observability (logged) but NOT a rescue criterion.
        mask_state2_fsm = pd.Series(False, index=df_group.index)
        mask_state2_intra_priming = pd.Series(False, index=df_group.index)
        if genome_fasta is not None and mask_state2.any():
            # FSM rescue: known reference isoforms pass (protects histone mRNAs
            # and other genuine non-polyadenylated reference transcripts).
            if 'category' in df_group.columns:
                mask_state2_fsm = mask_state2 & df_group['category'].astype(str).eq('FSM')
            # Intra-priming check (observability only) for State 2 non-FSM rows.
            for idx in df_group.index[mask_state2 & ~mask_state2_fsm]:
                row = df_group.loc[idx]
                try:
                    mask_state2_intra_priming.loc[idx] = _intra_priming_check(
                        row['Chr'], row['TrStart'], row['TrEnd'], row['Strand'],
                        genome_fasta,
                    )
                except Exception as _exc:
                    # Fail-closed for the diagnostic only (do NOT use for filtering).
                    mask_state2_intra_priming.loc[idx] = True
                    logger.warning(
                        "State 2 intra-priming check failed for %s:%d-%d %s: %s",
                        row['Chr'], row['TrStart'], row['TrEnd'], row['Strand'], _exc,
                    )
        elif genome_fasta is None:
            # Legacy semantics: when no genome is provided, treat all State 2
            # rows as intra-priming-equivalent (recorded as True) so the
            # observability count still reflects "unknown 3' context".
            mask_state2_intra_priming = mask_state2.copy()

        # State 2 final kept: FSM-rescued ONLY (Q-A2 correction).
        # Non-FSM State 2 rows go through the standard 3-condition framework.
        mask_state2_kept = mask_state2_fsm

        # Per-row polya_frac_low (used by non-hard_filter path). Mirrors the
        # legacy polya_frac_low definition but is now state-aware:
        #   State 1 -> polya_frac < threshold
        #   State 2 -> True unless FSM-rescued (Q-A2: non-FSM must satisfy
        #             the 3-condition framework, primarily the 5' Puffin gate)
        #   State 3 -> False (bypass; no polyA evidence => no penalty)
        polya_frac_low_rowwise = pd.Series(False, index=df_group.index)
        polya_frac_low_rowwise.loc[mask_state1] = (
            df_group.loc[mask_state1, 'polyA_frac'].lt(polya_fraction_threshold).values
        )
        polya_frac_low_rowwise.loc[mask_state2] = ~mask_state2_kept.loc[mask_state2].values
        # State 3 stays False (bypass)

        # Accumulate observability counters for this group
        state_counts['state1_pos'] += int(mask_state1.sum())
        state_counts['state2_zero'] += int(mask_state2.sum())
        state_counts['state2_fsm_rescue'] += int(mask_state2_fsm.sum())
        state_counts['state2_intra_priming_observed'] += int(
            (mask_state2 & ~mask_state2_fsm & mask_state2_intra_priming).sum()
        )
        state_counts['state2_non_fsm_total'] += int((mask_state2 & ~mask_state2_fsm).sum())
        state_counts['state3_missing'] += int(mask_state3.sum())

        # If hard_filter is True, apply the specific filtering logic
        if hard_filter:
            # Avoid SettingWithCopyWarning
            df_group_sub = df_group.copy()
            df_group_sub['Puffin_TSS_15bp'] = pd.to_numeric(df_group_sub['Puffin_TSS_15bp'], errors='coerce')  # 'no' -> NaN

            # State-aware polyA keep mask:
            #   State 1 -> polyA_frac > threshold
            #   State 2 -> FSM-rescued OR NOT intra-priming
            #   State 3 -> bypass (NaN)
            mask_puffin = df_group_sub['Puffin_TSS_15bp'].gt(puffin_prediction_threshold)
            mask_polya_kept = (
                df_group_sub['polyA_frac'].gt(polya_fraction_threshold)
                | mask_state2_kept
                | df_group_sub['polyA_frac'].isna()
            )
            mask_to_keep = mask_puffin & mask_polya_kept

            if df_group_sub['polyA_frac'].isna().all():
                logger.warning(
                    "hard_filter: polyA_frac is all-NaN for Chr=%s Strand=%s (%d rows); "
                    "polyA threshold bypassed for entire group.",
                    df_group_sub['Chr'].iloc[0], df_group_sub['Strand'].iloc[0], len(df_group_sub),
                )

            df_group_filtered = df_group[mask_to_keep]
        else:
            def should_filter_row(row, _polya_low=polya_frac_low_rowwise, _has_polya_info=has_polya_info):
                # Check if Puffin_TSS_50bp is 'no' or less than puffin_prediction_threshold
                puffin_50bp_value = row['Puffin_TSS_50bp']
                puffin_50bp_has_no = (puffin_50bp_value == 'no' or
                                     (isinstance(puffin_50bp_value, (int, float)) and puffin_50bp_value < puffin_prediction_threshold))

                # State-aware polyA low flag with sample-level bypass (Stage A R2).
                # When the ENTIRE sample has no polyA measurements, polyA penalty
                # is meaningless; force polya_frac_low=False so the 3-condition
                # framework collapses to the 5' Puffin-only gate. The precomputed
                # _polya_low Series is only consulted when sample has polyA, so
                # per-row State 1/2/3 logic is preserved when applicable.
                polya_frac_low = bool(_polya_low.loc[row.name]) if _has_polya_info else False

                # TranslationAI gate: only apply NMD penalty when TranslationAI
                # actually produced a meaningful NMD verdict. Skipped/unavailable or
                # 'no_orf' (no ORF found) -> NMD is meaningless, skip the penalty.
                has_translationai = (
                    "Predict_NMD" in row
                    and pd.notna(row.get("Predict_NMD"))
                    and (row["Predict_NMD"] != "no_orf")
                )
                is_nmd = (row["Predict_NMD"] == "NMD") if has_translationai else False

                if _has_polya_info:
                    # Standard logic: polyA penalty applies alongside 5' Puffin gate.
                    nmd_filter = is_nmd and (puffin_50bp_has_no or polya_frac_low)
                    truncation_filter = (row['truncation'] == 'yes' and
                                        (puffin_50bp_has_no or polya_frac_low))
                    ultra_low_quality_filter = puffin_50bp_has_no and polya_frac_low
                else:
                    # Sample has no polyA info: only 5' Puffin gate applies.
                    # ultra_low_quality_filter is fully disabled (cannot judge
                    # without both 5' and 3' evidence); NMD/truncation survive
                    # any transcript that demonstrates a strong 5' TSS.
                    nmd_filter = is_nmd and puffin_50bp_has_no
                    truncation_filter = (row['truncation'] == 'yes' and
                                        puffin_50bp_has_no)
                    ultra_low_quality_filter = False

                return nmd_filter or truncation_filter or ultra_low_quality_filter

            # Apply filtering conditions, retain rows that don't meet filtering criteria
            mask_to_keep = ~df_group.apply(should_filter_row, axis=1)
            df_group_filtered = df_group[mask_to_keep]

        processed_groups.append(df_group_filtered)

    # Merge all processed groups
    if processed_groups:
        df_filtered = pd.concat(processed_groups, ignore_index=True)
    else:
        # If all rows were filtered out, return an empty DataFrame
        df_filtered = pd.DataFrame(columns=df.columns)

    # Stage 2.6 observability: report large drops so direct-RNA / low-polyA-yield
    # datasets are not mistaken for catastrophic regressions (70-75% drop is by design).
    n_after = len(df_filtered)
    drop_pct = (1 - n_after / n_before) * 100 if n_before > 0 else 0
    if drop_pct > 30:
        logger.info(
            "Stage 2.6 TSS+polyA correction: %d -> %d transcripts (%.1f%% drop); "
            "direct-RNA datasets may have lower yield due to lack of 3-prime polyA signal",
            n_before, n_after, drop_pct,
        )

    # 3-state polyA observability (always emitted; cheap when State 2 is empty).
    logger.info(
        "3-state polyA distribution: state1_pos=%d, state2_zero=%d "
        "(fsm_rescue=%d, non_fsm_total=%d, non_fsm_with_intra_priming=%d), "
        "state3_missing=%d; genome_fasta=%s",
        state_counts['state1_pos'], state_counts['state2_zero'],
        state_counts['state2_fsm_rescue'], state_counts['state2_non_fsm_total'],
        state_counts['state2_intra_priming_observed'], state_counts['state3_missing'],
        'provided' if genome_fasta is not None else 'None (legacy semantics)',
    )

    return df_filtered


def correct_flnc(ref_df: pd.DataFrame,
                 query_df: pd.DataFrame,
                 ss_toler: int = 15,
                 term_toler: int = 50,
                 args: Optional[Any] = None, 
                 terminal_cluster: bool = False) -> pd.DataFrame:
    """
    Correct FLNC read splice sites and transcription start/end sites
    
    Parameters:
    ref_df: DataFrame of reference transcript models
    query_df: DataFrame of read data to be corrected
    ss_toler: Splice site tolerance threshold
    term_toler: Terminal tolerance threshold
    args: Optional argument object
    terminal_cluster: Whether to use terminal clustering
    
    Returns:
    Corrected query_df
    """
    
    # If args is provided, use dynamic threshold values from args
    if args is not None:
        ss_toler = getattr(args, 'ss_tolerance', ss_toler)
        term_toler = getattr(args, 'terminal_tolerance', term_toler)
    
    def correct_read_ssc(ref_df, query_df, ss_toler=15):
        """
        Correct read splice sites (SSC) based on reference transcript models
        """
        # Parameter validation
        if not isinstance(ss_toler, (int, float)):
            ss_toler = 15  # Use default value

        # Use defaultdict to create multi-level nested dictionary to simplify code
        ref_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # Preprocess reference data
        for _, row in ref_df.iterrows():
            ssc_chrom = row['Chr']
            ssc_strand = row['Strand']
            ssc_sites = list(map(int, row.SSC.split('-')))
            ssc_len = len(ssc_sites)
            
            # Add directly to dictionary
            ref_dict[ssc_chrom][ssc_strand][ssc_len].append(ssc_sites)
        
        # Convert lists to numpy arrays to accelerate computation
        for chrom in ref_dict:
            for strand in ref_dict[chrom]:
                for length in ref_dict[chrom][strand]:
                    ref_dict[chrom][strand][length] = np.array(ref_dict[chrom][strand][length])
        
        # Process query data
        updated_ssc_sites = []
        
        for idx, row in query_df.iterrows():
            query_ssc_chrom = row['Chr']
            query_ssc_strand = row['Strand']
            query_ssc_sites = safe_ssc_array(row.SSC)
            query_ssc_len = len(query_ssc_sites)
            
            best_match_sites = query_ssc_sites  # Use original sites by default
            
            # Check if matching reference data exists
            if (query_ssc_chrom in ref_dict and 
                query_ssc_strand in ref_dict[query_ssc_chrom] and 
                query_ssc_len in ref_dict[query_ssc_chrom][query_ssc_strand]):
                
                ref_arrays = ref_dict[query_ssc_chrom][query_ssc_strand][query_ssc_len]
                
                # Use vectorized Euclidean distance calculation
                differences = ref_arrays - query_ssc_sites
                distances = np.sqrt(np.sum(differences**2, axis=1))
                
                # Find minimum distance
                min_idx = np.argmin(distances)
                min_distance = distances[min_idx]
                
                # If minimum distance is within tolerance, use reference sites
                if min_distance <= ss_toler:
                    best_match_sites = ref_arrays[min_idx]
            
            # Convert numpy array back to string format
            updated_ssc_sites.append('-'.join(map(str, best_match_sites)))
        
        # Create copy of query data to avoid modifying original data
        result_df = query_df.copy()
        result_df['SSC'] = updated_ssc_sites
        
        if 'frequency' in result_df.columns:
            group_cols = ['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd']
            agg_dict = {'frequency': 'sum'}
            
            # Retain first value of other columns
            for col in result_df.columns:
                if col not in group_cols and col != 'frequency':
                    agg_dict[col] = 'first'
            
            result_df = result_df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        return result_df

    def correct_read_terminal(ref_df, query_df, term_toler=50):
        """
        Correct read transcription start and end sites (terminal) based on reference transcript models
        """
        # Parameter validation
        if not isinstance(term_toler, (int, float)):
            term_toler = 50  # Use default value
            
        # Build reference dictionary
        ref_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for _, row in ref_df.iterrows():
            ssc_chrom = row['Chr']
            ssc_strand = row['Strand']
            ssc_str = row['SSC']
            
            # Add directly to dictionary
            ref_dict[ssc_chrom][ssc_strand][ssc_str].append([row['TrStart'], row['TrEnd']])
        
        # Convert lists to numpy arrays to accelerate computation
        for chrom in ref_dict:
            for strand in ref_dict[chrom]:
                for ssc in ref_dict[chrom][strand]:
                    ref_dict[chrom][strand][ssc] = np.array(ref_dict[chrom][strand][ssc])
        
        # Process query data
        updated_trstarts = []
        updated_trends = []
        
        for idx, row in query_df.iterrows():
            query_chrom = row['Chr']
            query_strand = row['Strand']
            query_ssc = row['SSC']
            query_trstart = row['TrStart']
            query_trend = row['TrEnd']
            
            best_match_trstart = query_trstart  # Use original site by default
            best_match_trend = query_trend      # Use original site by default
            
            # Check if matching reference data exists
            if (query_chrom in ref_dict and 
                query_strand in ref_dict[query_chrom] and 
                query_ssc in ref_dict[query_chrom][query_strand]):
                
                ref_terminals = ref_dict[query_chrom][query_strand][query_ssc]
                
                # Calculate distance between query sites and reference sites
                # Use Euclidean distance considering joint differences of TrStart and TrEnd
                distances = np.sqrt(
                    (ref_terminals[:, 0] - query_trstart)**2 + 
                    (ref_terminals[:, 1] - query_trend)**2
                )
                
                # Find minimum distance
                min_idx = np.argmin(distances)
                min_distance = distances[min_idx]
                
                # If minimum distance is within tolerance, use reference sites
                if min_distance <= term_toler:
                    best_match_trstart = ref_terminals[min_idx, 0]
                    best_match_trend = ref_terminals[min_idx, 1]
            
            updated_trstarts.append(best_match_trstart)
            updated_trends.append(best_match_trend)
        
        # Create copy of query data and update sites
        result_df = query_df.copy()
        result_df['TrStart'] = updated_trstarts
        result_df['TrEnd'] = updated_trends
        
        # If there is a frequency column, perform aggregation
        if 'frequency' in result_df.columns:
            group_cols = ['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd']
            
            agg_dict = {'frequency': 'sum'}
            # Retain first value of other columns
            for col in result_df.columns:
                if col not in group_cols and col != 'frequency':
                    agg_dict[col] = 'first'
            
            result_df = result_df.groupby(group_cols, as_index=False).agg(agg_dict)
        
        return result_df

    # Main function logic starts
    # Copy result DataFrame
    out_df = query_df.copy()

    if terminal_cluster:
        if args is None:
            raise ValueError("args parameter is required when terminal_cluster is True")
            
        # Assume GeneClustering and TerminalSitesProcessor are defined
        gene_clustering = GeneClustering(num_processes=args.threads)
        out_df = gene_clustering.cluster(out_df)
        terminalsitesprocessor = TerminalSitesProcessor(
            cluster_group_size=args.cluster_group_size,
            eps=args.eps,
            min_samples=args.min_samples,
            num_processes=args.threads
        )
        out_df = terminalsitesprocessor.get_terminal_sites(out_df)
    else:
        # Perform filtering operations on out_df
        if 'identity' in out_df.columns and 'coverage' in out_df.columns:
            # Get min_aln_coverage and min_aln_identity parameters from args
            min_aln_coverage = getattr(args, 'min_aln_coverage', None) if args else None
            min_aln_identity = getattr(args, 'min_aln_identity', None) if args else None
            
            if min_aln_coverage is not None and min_aln_identity is not None:
                out_df = out_df[
                    (out_df["identity"] >= min_aln_identity) & 
                    (out_df["coverage"] >= min_aln_coverage)
                ]
            
            # out_df = out_df.drop(columns=["identity", "coverage"]).dropna()
            out_df = out_df.dropna()
    
    # Rename TrStart_reads and TrEnd_reads columns to TrStart and TrEnd
    if 'TrStart_reads' in out_df.columns:
        out_df = out_df.rename(columns={'TrStart_reads': 'TrStart'})
    if 'TrEnd_reads' in out_df.columns:
        out_df = out_df.rename(columns={'TrEnd_reads': 'TrEnd'})
    
    # Correct splice sites and transcription start/end sites
    out_df = correct_read_ssc(ref_df, out_df, ss_toler=ss_toler)
    out_df = correct_read_terminal(ref_df, out_df, term_toler=term_toler)

    return out_df

def rescue_low_frep_reads(merged_df, df_dict, args):
    """
    Rescue low frequency reads by replacing zero frequency entries with data from flnc files.
    
    Args:
        merged_df: DataFrame containing merged data from all samples
        df_dict: Dictionary mapping sample names to their DataFrames
        args: Arguments object containing output path information
    
    Returns:
        DataFrame with rescued low frequency reads
    """
    rescued_dfs = []
    
    for sample, _ in df_dict.items():
        flnc_path = os.path.join(args.output, f"temp/{sample}.ssc_flnc.parquet")
        rescue_cols = [f'{sample}_TrStart_reads', f'{sample}_TrEnd_reads', f'{sample}_frequency']
        
        rescue_df = merged_df[['Chr', 'Strand', 'SSC'] + rescue_cols].copy()
        flnc_df = pd.read_parquet(flnc_path).dropna()
        
        zero_mask = rescue_df[f'{sample}_frequency'] == 0
        zero_rows = rescue_df[zero_mask]
        
        if not zero_rows.empty:
            # Rename flnc_df columns to match rescue_df
            flnc_renamed = flnc_df.rename(columns={
                'TrStart_reads': f'{sample}_TrStart_reads',
                'TrEnd_reads': f'{sample}_TrEnd_reads',
                'frequency': f'{sample}_frequency'
            })
            
            # Only keep flnc rows that match zero_rows keys (Chr, Strand, SSC)
            keys = ['Chr', 'Strand', 'SSC']
            flnc_rescue_candidates = flnc_renamed[flnc_renamed.set_index(keys).index.isin(zero_rows.set_index(keys).index)]
            
            # Remove zero frequency rows from rescue_df
            rescue_df_filtered = rescue_df[~zero_mask]
            
            # Add matching flnc candidates (replacing zero frequency rows)
            rescued_df = pd.concat([rescue_df_filtered, flnc_rescue_candidates], ignore_index=True)
        else:
            rescued_df = rescue_df
        
        rescued_dfs.append(rescued_df)
    
    if rescued_dfs:
        rescued_low_frep_df = rescued_dfs[0]
        for df in rescued_dfs[1:]:
            rescued_low_frep_df = rescued_low_frep_df.merge(df, on=['Chr', 'Strand', 'SSC'], how='outer')
    else:
        rescued_low_frep_df = pd.DataFrame(columns=['Chr', 'Strand', 'SSC'])
    
    return rescued_low_frep_df
