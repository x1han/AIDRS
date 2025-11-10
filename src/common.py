#!/usr/bin/env python

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
    df = df_grouped.merge(df_junction, on=["Chr", "Strand", "SSC"], how="inner").dropna()

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
    df['Group_freq'] = df.groupby(['Chr', 'Strand', 'Group'], observed=True)['frequency'].transform('sum')
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
    df['SSC2'] = df['SSC'].apply(lambda x: list(map(int, x.split('-'))))
    
    df['Tr_length_min'] = df.apply(
        lambda row: np.inf if len(row['SSC2']) > 2 else min([
            (row['SSC2'][0] - row['TrStart_mean']),
            (row['TrEnd_mean'] - row['SSC2'][1])
        ]),
        axis=1
    )

    df = df[~(((df['Tr_length_min'] < threshold_fragmentary_transcript_bp) & (df['frequency'] < 0.01 * total_meanfreq)) |
              (df['contains_non_conservative']) & (df['frequency'] < 0.01 * total_meanfreq))]
    
    return df

def transcript_model_filtering(df, puffin_prediction_threshold=0.02, polya_fraction_threshold=0.95, hard_filter=False):
    """
    Apply TSS correction and filtering to dataframe
    
    Requirements:
    1. After obtaining the tss_col corresponding to strand, if the tuple in the Puffin_TSS_15bp column does not contain 'no', 
       then tss_col is modified to tss_col+Puffin_TSS_15bp[0]
    2. If the value in the Predict_NMD column is 'NMD' and (Puffin_TSS_50bp value has 'no' or polyA_frac value is less than polya_fraction_threshold), 
       then delete this row
    3. If the value in the truncation column is 'yes' and (Puffin_TSS_50bp value has 'no' or polyA_frac value is less than polya_fraction_threshold), 
       then delete this row
    """
    # Process by grouping on Chr and Strand
    df_groups = [g for _, g in df.groupby(['Chr','Strand'], observed=True)]
    
    processed_groups = []
    for df_group in df_groups:
        # Copy data to avoid direct modification
        df_group = df_group.copy()
        
        # Get strand information for the current group
        strand = df_group['Strand'].iloc[0]
        tss_col = 'TrStart' if strand == '+' else 'TrEnd'
        
        # If hard_filter is True, apply the specific filtering logic
        if hard_filter:
            # Avoid SettingWithCopyWarning
            test = df_group.copy()
            test['Puffin_TSS_15bp'] = pd.to_numeric(test['Puffin_TSS_15bp'], errors='coerce')  # 'no' -> NaN
            
            # Filter: non-null (not originally 'no') and > 0.02,同时 polyA_frac > 0.95
            mask_to_keep = test['Puffin_TSS_15bp'].gt(0.02) & test['polyA_frac'].gt(0.95)
            df_group_filtered = df_group[mask_to_keep]
        else:
            def should_filter_row(row):
                # Check if Puffin_TSS_50bp is 'no' or less than puffin_prediction_threshold
                puffin_50bp_value = row['Puffin_TSS_50bp']
                puffin_50bp_has_no = (puffin_50bp_value == 'no' or 
                                     (isinstance(puffin_50bp_value, (int, float)) and puffin_50bp_value < puffin_prediction_threshold))
                
                # Check if polyA_frac is less than polya_fraction_threshold
                polya_frac_low = row['polyA_frac'] < polya_fraction_threshold
                
                # Condition 1: NMD and (Puffin_TSS_50bp is 'no' or < puffin_prediction_threshold or polyA_frac<polya_fraction_threshold)
                nmd_filter = (row['Predict_NMD'] == 'NMD' and 
                             (puffin_50bp_has_no or polya_frac_low))
                
                # Condition 2: truncation is 'yes' and (Puffin_TSS_50bp is 'no' or < puffin_prediction_threshold or polyA_frac<polya_fraction_threshold)
                truncation_filter = (row['truncation'] == 'yes' and 
                                    (puffin_50bp_has_no or polya_frac_low))
                
                # Condition 3: Both puffin_50bp_has_no and polya_frac_low (ultra-low quality transcript)
                ultra_low_quality_filter = puffin_50bp_has_no and polya_frac_low
                
                return nmd_filter or truncation_filter or ultra_low_quality_filter
            
            # Apply filtering conditions, retain rows that don't meet filtering criteria
            mask_to_keep = ~df_group.apply(should_filter_row, axis=1)
            df_group_filtered = df_group[mask_to_keep]
        
        processed_groups.append(df_group_filtered)
    
    # Merge all processed groups
    if processed_groups:
        return pd.concat(processed_groups, ignore_index=True)
    else:
        # If all rows were filtered out, return an empty DataFrame
        return pd.DataFrame(columns=df.columns)


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
            query_ssc_sites = np.array(list(map(int, row.SSC.split('-'))))
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
