#!/usr/bin/env python

import os
import time
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool
from typing import Optional, Dict, List, Union, Tuple
from collections import defaultdict
import glob
import polars as pl
import gffutils
import pyfaidx

from .common import read_flnc
from .gene_grouping import GeneClustering 

class IsoformAnnotator:
    """
    1. IsoformAnnotator(num_processes=20, reference)
    2. save_results(df_result, output_dir, ref_anno=None)
    """
    def __init__(self, num_processes: int = 20, terminal_tolerance: int = 50):
        self.num_processes = num_processes
        self.terminal_tolerance = terminal_tolerance

    # ------------------------------------------------------------------
    # 1. Unique external entry point (keeping signature unchanged)
    # ------------------------------------------------------------------
    def save_results(self,
                     df_result: pd.DataFrame,
                     output_dir: str,
                     reference: str, 
                     ref_anno: Optional[pd.DataFrame] = None, ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        # ---- 1. Annotation ----
        df_result.to_csv(os.path.join(output_dir,
                                         'temp/aidrs.transcript.result_df.tsv'),
                            sep='\t', index=False)
        annotated_df = self.annotate(df_result, ref_anno, self.num_processes)
        annotated_df.to_csv(os.path.join(output_dir,
                                         'temp/aidrs.transcript.annotated_df.tsv'),
                            sep='\t', index=False)
        # ---- 2. Output GTF ----
        self.to_gtf(annotated_df, output_dir)
        # ---- 2. Output FASTA ----
        self.to_fasta(os.path.join(output_dir, 'aidrs.transcript_model.gtf'), reference, output_dir)
        # ---- 3. Merge annotation information back to original data for quantification ----
        # Merge annotation information back to original data
        merge_cols = ['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd']
        df_for_quant = df_result.merge(
            annotated_df[merge_cols + ['TrID', 'GeneID', 'GeneName']],
            on=merge_cols,
            how='left'
        )
        df_for_quant['sites'] = df_for_quant.apply(
            lambda r: sorted(
                list(map(int, r['SSC'].split('-'))) +
                [int(r['TrStart']), int(r['TrEnd'])]
            ), axis=1
        )
        # ---- 4. Quantification ----
        quant_tables = self.quantify(df_for_quant)
        for name, table in quant_tables.items():
            table.to_csv(os.path.join(output_dir, f'aidrs_{name}.tsv'),
                         sep='\t')
        df_for_quant, polyA_tables = self.polyA_len_profile(df_for_quant, output_dir)
        for name, table in polyA_tables.items():
            table.to_csv(os.path.join(output_dir, f'aidrs_{name}.tsv'),
                         sep='\t')
        # ---- 5. Original assessment table ----
        df_for_quant.to_csv(os.path.join(output_dir,
                                     'aidrs.transcript.assessment.tsv'),
                        sep='\t', index=False)


    # ------------------------------------------------------------------
    # 2.1 Main Annotation Logic
    # ------------------------------------------------------------------

    def annotate(self,
                  df_result: pd.DataFrame,
                  ref_anno: Optional[pd.DataFrame],
                  num_processes: int) -> pd.DataFrame:
        # 3.1 Generate uniqueTr
        df = df_result.copy()
        df['uniqueTr'] = 'Tr' + df.groupby(
            ['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd'],
            observed=True
        ).ngroup().astype(str)
        df_unique = df[['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd', 'uniqueTr', 'TIS_related_location', 'TTS_related_location', 'Predict_NMD']].drop_duplicates()
        # 3.2 Cluster to get Group
        gene_clustering = GeneClustering(num_processes=num_processes)
        df_unique = gene_clustering.cluster(df_unique)
        # 3.3 Merge with reference annotation
        if ref_anno is not None:
            ref_anno_model = ref_anno[
                ref_anno['SSC'].isin(df_unique['SSC'].unique())
            ].copy()
            merged = df_unique.merge(
                ref_anno_model,
                on=['Chr', 'Strand', 'SSC'],
                how='left',
                suffixes=('', '_ref')
            )
        else:
            merged = df_unique.copy()
        # 3.4 Run concurrently by Group
        df_groups = [g for _, g in merged.groupby('Group', observed=True)]
        with Pool(num_processes) as pool:
            func = partial(self.annotate_one_group, ref_anno=ref_anno)
            results = pool.map(func, df_groups)
        final_df = pd.concat(results, ignore_index=True)
        # 3.5 Deduplicate key
        final_df['key'] = final_df['TrID']
        cnt = final_df.groupby('key').cumcount().add(1).astype(str)
        final_df['TrID'] = np.where(
            cnt != '1',
            final_df['TrID'] + '_' + cnt,
            final_df['TrID']
        )
        final_df = final_df.drop(columns=['key'])
        if ref_anno is not None:
            return self._novel_gene_remapping(final_df, ref_anno)
        else:
            return final_df

    def annotate_one_group(self,
                           df_group: pd.DataFrame,
                           ref_anno: Optional[pd.DataFrame]) -> pd.DataFrame:
        """With provided pure function logic completely consistent, only indentation level changes"""
        if ref_anno is not None:
            # Split data into query (novel) and reference parts
            query_df = df_group[df_group.isna().any(axis=1)].copy()
            ref_df = df_group[~df_group.isna().any(axis=1)].copy()
            
            # Process reference data if exists
            if not ref_df.empty and not (ref_df.shape[0] == len(ref_df.uniqueTr.unique()) == len(ref_df.TrID.unique())):
                ref_df = self._map_transcript_1to1(ref_df)  # Solve 1 uniqueTr vs. multiple TrIDs and multiple uniqueTrs vs. 1 TrID
            
            # Process both query and reference data
            if not query_df.empty and not ref_df.empty:
                # Update reference data with TSS/TES flags
                self._update_ref_with_flags(ref_df)
                # Build reference dictionary and map query to reference
                ref_dict = self._build_ref_dict(ref_df)
                query_df = self._map_query_to_ref(query_df, ref_dict)
                # Combine results
                result_df = pd.concat([ref_df, query_df], ignore_index=True)
                return result_df.drop_duplicates()
            elif ref_df.empty:   # All novel
                return self._fill_novel(df_group).drop_duplicates()
            else:                # All reference
                # Update reference data with TSS/TES flags
                self._update_ref_with_flags(ref_df)
                return ref_df.drop_duplicates()
        else:
            return self._fill_novel(df_group).drop_duplicates()

    # ------------------------------------------------------------------
    # 2.2 Annotation Helper Functions
    # ------------------------------------------------------------------

    def _update_ref_with_flags(self, ref_df: pd.DataFrame) -> None:
        """Update reference DataFrame with TSS/TES alteration flags"""
        for idx, row in ref_df.iterrows():
            ref_trans_id = row['TrID']
            strand = row['Strand']
            q_trs, q_tre = row['TrStart'], row['TrEnd']
            r_trs, r_tre = row['TrStart_ref'], row['TrEnd_ref']
            
            # Calculate differences based on strand
            if strand == '+':
                d_tss, d_tes = abs(q_trs - r_trs), abs(q_tre - r_tre)
            else:
                d_tss, d_tes = abs(q_tre - r_tre), abs(q_trs - r_trs)
            
            # Determine flag based on differences
            flag = None
            if d_tss > self.terminal_tolerance and d_tes > self.terminal_tolerance:
                flag = 'AlterTssTes'
            elif d_tss > self.terminal_tolerance:
                flag = 'AlterTss'
            elif d_tes > self.terminal_tolerance:
                flag = 'AlterTes'
            
            # Apply flag to TrID if needed
            if flag:
                ref_df.at[idx, 'TrID'] = f"{ref_trans_id}_{flag}"
            
            # Update reference coordinates
            ref_df.at[idx, 'TrStart_ref'] = q_trs
            ref_df.at[idx, 'TrEnd_ref'] = q_tre

    def _map_ref_term(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process transcript mapping by finding the most similar transcript based on range distance.
        Args:
            df: DataFrame containing transcript reference data with columns:
                        TrID, TrStart_ref, TrEnd_ref, uniqueTr, TrStart, TrEnd
        Returns:
            Modified DataFrame with updated TrStart_ref and TrEnd_ref values based on best match
        """
        def calculate_range_distance(ref_range, term_range):
            """Calculate the sum of absolute differences between two ranges."""
            return sum(abs(a - b) for a, b in zip(ref_range, term_range))

        def find_best_mapping(term_map_dict):
            """Find the key with minimum value in the mapping dictionary."""
            if not term_map_dict:
                return None
            min_key = min(term_map_dict, key=term_map_dict.get)
            min_value = term_map_dict[min_key]
            return min_key if min_value < (self.terminal_tolerance * 3) else None  # Using 3x terminal_tolerance instead of hardcoded 150

        # Create a copy to avoid modifying original dataframe
        result_df = df.copy()
        # Process each unique transcript ID
        for trans in result_df.TrID.unique():
            # Filter data for current transcript
            mask = result_df.TrID == trans
            current_trans_df = result_df[mask]
            # Extract reference range
            ref_start = current_trans_df.TrStart_ref.iloc[0]
            ref_end = current_trans_df.TrEnd_ref.iloc[0]
            ref_range = (ref_start, ref_end)
            # Build term dictionary mapping uniqueTr to (TrStart, TrEnd)
            term_dict = {}
            for untr in current_trans_df.uniqueTr.unique():
                untr_mask = (result_df.TrID == trans) & (result_df.uniqueTr == untr)
                start_val = int(result_df.loc[untr_mask, 'TrStart'].iloc[0])
                end_val = int(result_df.loc[untr_mask, 'TrEnd'].iloc[0])
                term_dict[untr] = (start_val, end_val)
            # Calculate mapping distances
            term_map_dict = {}
            for untr, term_range in term_dict.items():
                distance = calculate_range_distance(ref_range, term_range)
                term_map_dict[untr] = distance
            # Find best mapping if minimum distance is less than threshold
            best_match = find_best_mapping(term_map_dict)
            if best_match is not None:
                # Update map_uniqueTr column
                result_df.loc[mask, 'map_uniqueTr'] = best_match
                # Update reference range with matched transcript's start and end
                best_start, best_end = term_dict[best_match]
                result_df.loc[mask, 'TrStart_ref'] = best_start
                result_df.loc[mask, 'TrEnd_ref'] = best_end
            else:
                # Set to None if no suitable match found
                result_df.loc[mask, 'map_uniqueTr'] = None

        # Process merging of TrID for duplicate map_uniqueTr values
        def merge_duplicate_transcripts(df):
            """Merge rows with the same map_uniqueTr by combining TrID values."""
            df_result = df.copy()
            for map_tr in df_result['map_uniqueTr'].dropna().unique():
                # Find all rows with the same map_uniqueTr
                mask = df_result['map_uniqueTr'] == map_tr
                matching_rows = df_result[mask]
                if len(matching_rows) > 1:
                    # Get unique TrID values and join them with '_'
                    unique_trids = matching_rows['TrID'].unique()
                    combined_trid = '_'.join(unique_trids)
                    # Update TrID for all matching rows
                    df_result.loc[mask, 'TrID'] = combined_trid
            return df_result

        def filter_unmapped_transcripts(df):
            df_result = df.copy()

            for uni_tr in df_result['uniqueTr'].dropna().unique():
                mask = df_result['uniqueTr'] == uni_tr
                matching_rows = df_result[mask]

                if len(matching_rows) > 1:
                    # 1) Prefer mapped transcripts
                    mapped_idx = matching_rows['map_uniqueTr'].notna()
                    if mapped_idx.any():
                        keeper = matching_rows.loc[mapped_idx].index[0]
                        df_result.drop(matching_rows.index.difference([keeper]), inplace=True)
                        continue

                    # 2) All unmapped: keep first row, rebuild TrID
                    keeper = matching_rows.index[0]
                    df_result.loc[keeper, 'TrID'] = f"{matching_rows['GeneID'].iat[0]}_Novel{uni_tr}"
                    df_result.loc[keeper, 'TrStart_ref'] = matching_rows['TrStart'].iat[0]
                    df_result.loc[keeper, 'TrEnd_ref'] = matching_rows['TrEnd'].iat[0]
                    df_result.drop(matching_rows.index.difference([keeper]), inplace=True)

            return df_result

        # Apply the merging logic
        result_df = merge_duplicate_transcripts(result_df)
        result_df = filter_unmapped_transcripts(result_df)
        return result_df.drop(columns = ['map_uniqueTr']).drop_duplicates()
    
    def _transcript_1to1_processor(self, uni_tr_mappings):
        # Copy to avoid warnings
        uni_tr_mappings = uni_tr_mappings.copy()
        uni_tr_mappings['match_status'] = uni_tr_mappings.apply(
            lambda row: 'match' if (abs(row['TrStart'] - row['TrStart_ref']) <= self.terminal_tolerance and abs(row['TrEnd'] - row['TrEnd_ref']) <= self.terminal_tolerance) else 'miss',
            axis=1
        )
        
        # Find matching rows
        match_mask = uni_tr_mappings['match_status'] == 'match'
        matching_rows = uni_tr_mappings[match_mask]
        
        if not matching_rows.empty:
            working_df = matching_rows.copy()
            
            unique_trids = working_df['TrID'].unique()
            combined_trid = '_'.join(unique_trids)
            working_df['TrID'] = combined_trid
            
            trstart_ref_value = working_df['TrStart'].iloc[0]
            trend_ref_value = working_df['TrEnd'].iloc[0]
            working_df['TrStart_ref'] = trstart_ref_value
            working_df['TrEnd_ref'] = trend_ref_value
            
            result_row = working_df.drop_duplicates()
            
        else:
            unique_trids = uni_tr_mappings['TrID'].unique()
            combined_trid = '_'.join(unique_trids)
            uni_tr_mappings['TrID'] = f'{combined_trid}_AlterTssTes'
            
            trstart_ref_value = uni_tr_mappings['TrStart'].iloc[0]
            trend_ref_value = uni_tr_mappings['TrEnd'].iloc[0]
            uni_tr_mappings['TrStart_ref'] = trstart_ref_value
            uni_tr_mappings['TrEnd_ref'] = trend_ref_value
            
            result_row = uni_tr_mappings.drop_duplicates()
        
        return result_row

    def _map_transcript_1to1(self, df):
        return (
        df.groupby('uniqueTr', group_keys=False, as_index=False)
                 .apply(self._transcript_1to1_processor, include_groups=True)
                 .reset_index(drop=True)
    ).drop(columns = ['match_status'])

    def _fill_novel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle entire table as novel when no ref"""
        df = df.copy()
        for idx, row in df.iterrows():
            gid = row['Group']
            trid = row['uniqueTr']
            df.at[idx, 'GeneID'] = f'NovelGene{gid}'
            df.at[idx, 'GeneName'] = f'NovelGene{gid}'
            df.at[idx, 'TrID'] = f'NovelGene{gid}_Novel{trid}'
            df.at[idx, 'TrStart_ref'] = row['TrStart']
            df.at[idx, 'TrEnd_ref'] = row['TrEnd']
        return df

    def _build_ref_dict(self, ref_df: pd.DataFrame, include_term = True) -> dict:
        """
        Build reference dictionary mapping gene identifiers to site positions.
        
        Args:
            ref_df: Reference annotation DataFrame
            include_term: Whether to include terminal sites (TrStart/TrEnd) in site set
        
        Returns:
            Dictionary mapping gene keys to lists of site positions
        """
        ref_dict = defaultdict(set)
        for _, row in ref_df.iterrows():
            gene_id = str(row['GeneID'])
            gene_name = str(row['GeneName'])
            key = f'{gene_id}_{gene_name}'
            sites = {int(row['TrStart']), int(row['TrEnd'])} if include_term else set()
            ssc_str = str(row['SSC']).strip()
            if ssc_str and ssc_str != 'nan':
                sites |= {int(x) for x in ssc_str.split('-') if x}
            ref_dict[key].update(sites)
        return {k: list(v) for k, v in ref_dict.items()}

    def _map_query_to_ref(self, query_df: pd.DataFrame, ref_dict: dict) -> pd.DataFrame:
        for idx, row in query_df.iterrows():
            q_sites = {int(row['TrStart']), int(row['TrEnd'])}
            ssc_str = str(row['SSC']).strip()
            if ssc_str and ssc_str != 'nan':
                q_sites |= {int(x) for x in ssc_str.split('-') if x}
            best_key, best_count = None, 0
            for ref_key, ref_sites in ref_dict.items():
                cnt = len(q_sites & set(ref_sites))
                if cnt >= 2 and cnt > best_count:
                    best_count, best_key = cnt, ref_key
            trid = row['uniqueTr']
            if best_key and best_count >= 2:
                gene_id, gene_name = best_key.split('_', 1)
                query_df.at[idx, 'TrID'] = f'{gene_id}_Novel{trid}'
                query_df.at[idx, 'GeneID'] = gene_id
                query_df.at[idx, 'GeneName'] = gene_name
            else:
                gid = row['Group']
                query_df.at[idx, 'TrID'] = f'NovelGene{gid}_Novel{trid}'
                query_df.at[idx, 'GeneID'] = f'NovelGene{gid}'
                query_df.at[idx, 'GeneName'] = f'NovelGene{gid}'
            # Put ref coordinates as self
            query_df.at[idx, 'TrStart_ref'] = row['TrStart']
            query_df.at[idx, 'TrEnd_ref'] = row['TrEnd']
        return query_df


    def _novel_gene_remapping(self, df_result: pd.DataFrame, ref_anno: Optional[pd.DataFrame]) -> pd.DataFrame:
        # Build reference dictionary (grouped by Chr, Strand)
        ref_anno_dict_by_chr_strand = {}
        for (chrom, strand), group_df in ref_anno.groupby(['Chr', 'Strand']):
            inner_dict = self._build_ref_dict(group_df, include_term=False)  # Assuming you already support include_term
            ref_anno_dict_by_chr_strand[(chrom, strand)] = inner_dict

        # Store results for all groups
        updated_groups = []

        for gid, group_df in df_result.groupby('GeneID'):
            group_df = group_df.copy()  # Avoid modifying original data warnings
            if 'NovelGene' in gid:
                matches = []
                all_q_sites = set()
                q_chrom, q_strand = None, None

                # Collect all SSC sites for this group (ignoring TrStart/TrEnd)
                for idx, row in group_df.iterrows():
                    q_chrom, q_strand = row['Chr'], row['Strand']
                    ssc_str = str(row['SSC']).strip()
                    if ssc_str and ssc_str not in ('nan', ''):
                        sites = {int(x) for x in ssc_str.split('-') if x.strip()}
                        all_q_sites.update(sites)

                # Check if there is corresponding reference data
                if q_chrom is not None and q_strand is not None and (q_chrom, q_strand) in ref_anno_dict_by_chr_strand:
                    ref_dict = ref_anno_dict_by_chr_strand[(q_chrom, q_strand)]
                    for ref_key, ref_sites in ref_dict.items():
                        cnt = len(all_q_sites & set(ref_sites))
                        if cnt >= 2:
                            ref_full_flag = (cnt == len(ref_sites))  # Whether all reference sites are covered
                            matches.append({'ref_key': ref_key, 'cnt': cnt, 'ref_full': ref_full_flag})

                if matches:
                    match_df = pd.DataFrame(matches)
                    if match_df.ref_full.any():
                        best_rows = match_df[match_df.ref_full]
                    else:
                        max_cnt = match_df.cnt.max()
                        best_rows = match_df[match_df.cnt == max_cnt]

                    # Concatenate results
                    if len(best_rows) == 1:
                        gene_id, gene_name = best_rows.iloc[0].ref_key.split('_', 1)
                    else:
                        parts = best_rows.ref_key.str.split('_', n=1, expand=True)
                        gene_id = 'NovelGeneCluster_' + '_'.join(parts[0])
                        gene_name = 'NovelGeneCluster_' + '_'.join(parts[1])

                    # Replace current group's GeneID and GeneName
                    group_df['GeneID'] = gene_id
                    group_df['GeneName'] = gene_name
                    
                    def update_tr_id(tr_id):
                        parts = str(tr_id).split('_', 1)
                        if len(parts) == 2:
                            return f"{gene_id}_{parts[1]}"
                        else:
                            # If there is no underscore, replace with gene_id (or keep original? as needed)
                            return gene_id

                    group_df['TrID'] = group_df['TrID'].apply(update_tr_id)

            # Add to final results regardless of modification
            updated_groups.append(group_df)

        # Merge all groups
        return pd.concat(updated_groups, ignore_index=True)

    # ------------------------------------------------------------------
    # 3. GTF & FASTA Generation and Related Helper Functions
    # ------------------------------------------------------------------
    def to_gtf(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Aggregate by GeneID, output minimum TrStart_ref and maximum TrEnd_ref as gene range,
        including complete annotations for gene / transcript / exon / CDS / start_codon / stop_codon / UTR.
        """
        def parse_exons(tr_start, ssc_str, tr_end, strand):
            if strand == '+':
                ssc_list = np.sort([tr_start] + list(map(int, ssc_str.split('-'))) + [tr_end])
            else:
                ssc_list = np.sort(-np.array([tr_start] + list(map(int, ssc_str.split('-'))) + [tr_end]))
            exons = []
            for i in range(0, len(ssc_list)-1, 2):
                exons.append((ssc_list[i], ssc_list[i+1]))
            return exons

        def reverse_coords_and_order(data: Dict[str, Union[List[Tuple[int, int]],
                                                           Tuple[int, int]]]
                                     ) -> Dict[str, Union[List[Tuple[int, int]],
                                                          Tuple[int, int]]]:
            """
            Perform two-step processing on input dictionary:
            1. Swap coordinate order within each tuple: (a,b) -> (b,a)
            2. Reverse the entire list corresponding to each key
            For keys with single tuple values (e.g., start_codon), only perform step 1.
            """
            def swap(t: Tuple[int, int]) -> Tuple[int, int]:
                return abs(t[1]), abs(t[0])
            out = {}
            for k, v in data.items():
                if isinstance(v, list):
                    # First swap each tuple internally, then reverse the entire list
                    out[k] = [swap(t) for t in reversed(v)]
                else:
                    # Single tuple
                    out[k] = swap(v)
            out['utr5'], out['utr3'] = out.pop('utr3'), out.pop('utr5')
            return out

        def simulate_ribosome_walk(row):
            exons = parse_exons(int(row.TrStart), row.SSC, int(row.TrEnd), row.Strand)
            # Initialize variables
            current_exon_idx = 0
            current_genomic_pos = exons[0][0]  # Start from the beginning position of the first exon
            current_relative_pos = 1
            # Initialize feature lists
            features = {
                'exons': exons,
                'cds': [],
                'utr5': [],
                'utr3': [],
                'start_codon': None,
                'stop_codon': None,
                # 'bases': []  # Store base information for each step
            }
            # Current region being recorded
            current_region = 'utr5'
            region_start_genomic = current_genomic_pos
            region_start_relative = current_relative_pos
            # Calculate RNA sequence index
            # rna_index = 0
            # Iterate through each exon
            while current_exon_idx < len(exons):
                current_exon = exons[current_exon_idx]
                # Iterate through each position in the current exon
                while current_genomic_pos <= current_exon[1]:
                    # Check for region changes
                    if current_relative_pos == int(row.TIS_related_location) + 1:
                        # End 5' UTR, start CDS
                        if current_region == 'utr5':
                            # Record current UTR5 segment
                            if region_start_genomic <= current_genomic_pos - 1:
                                features['utr5'].append((region_start_genomic, current_genomic_pos - 1))
                            current_region = 'cds'
                            region_start_genomic = current_genomic_pos
                            region_start_relative = current_relative_pos
                            features['start_codon'] = (current_genomic_pos, current_genomic_pos + 2)
                    elif current_relative_pos == int(row.TTS_related_location) + 1:
                        # End CDS, start 3' UTR
                        if current_region == 'cds':
                            # Record current CDS segment
                            if region_start_genomic <= current_genomic_pos:
                                features['cds'].append((region_start_genomic, current_genomic_pos - 1))
                            current_region = 'utr3'
                            region_start_genomic = current_genomic_pos
                            region_start_relative = current_relative_pos
                            features['stop_codon'] = (current_genomic_pos, current_genomic_pos + 2)
                    # Output current position information (for debugging)
                    # print(f'step: {current_relative_pos}, position: {current_genomic_pos}, base: {base}, exon: {current_exon}, region: {current_region}')
                    # if rna_index < len(rna_sequence):
                    #     base = rna_sequence[rna_index]
                    #     features['bases'].append({
                    #         'relative_pos': current_relative_pos,
                    #         'genomic_pos': current_genomic_pos,
                    #         'base': base,
                    #         'region': current_region
                    #     })
                    #     rna_index += 1
                    # Move to next position
                    current_genomic_pos += 1
                    current_relative_pos += 1
                # Current exon has been processed, move to next exon
                current_exon_idx += 1
                if current_exon_idx < len(exons):
                    # Record current region segment in current exon
                    if current_region == 'utr5':
                        features['utr5'].append((region_start_genomic, current_exon[1]))
                    elif current_region == 'cds':
                        features['cds'].append((region_start_genomic, current_exon[1]))
                    elif current_region == 'utr3':
                        features['utr3'].append((region_start_genomic, current_exon[1]))
                    # Move to start position of next exon
                    current_genomic_pos = exons[current_exon_idx][0]
                    region_start_genomic = current_genomic_pos
            # Add last region segment of last exon
            if current_region == 'utr5':
                features['utr5'].append((region_start_genomic, exons[-1][1]))
            elif current_region == 'cds':
                features['cds'].append((region_start_genomic, exons[-1][1]))
            elif current_region == 'utr3':
                features['utr3'].append((region_start_genomic, exons[-1][1]))
            if row.Strand == '-':
                features = reverse_coords_and_order(features)
            return features

        def sort_gtf_by_hierarchy(gtf_file, output_file):
            # Create database
            db = gffutils.create_db(gtf_file, ':memory:', 
                                disable_infer_genes=True, 
                                disable_infer_transcripts=True,
                                merge_strategy='create_unique')
            # Collect transcripts by gene
            genes = {}
            for gene in db.features_of_type('gene'):
                genes[gene.id] = {
                    'feature': gene,
                    'transcripts': defaultdict(list)
                }
            # Collect transcripts for each gene
            for transcript in db.features_of_type('transcript'):
                gene_id = transcript.attributes.get('gene_id', [None])[0]
                if gene_id and gene_id in genes:
                    genes[gene_id]['transcripts'][transcript.id].append(transcript)
            # Collect sub-features for each transcript
            transcripts_features = defaultdict(list)
            for feature in db.features_of_type(['exon', 'CDS', 'UTR', 'start_codon', 'stop_codon']):
                transcript_id = feature.attributes.get('transcript_id', [None])[0]
                if transcript_id:
                    transcripts_features[transcript_id].append(feature)
            # Define priority of feature types
            feature_priority = {
                'transcript': 0,
                'exon': 1,
                'CDS': 2,
                'UTR': 3,
                'start_codon': 4,
                'stop_codon': 5
            }
            with open(output_file, 'w') as f:
                # Sort by gene start position
                sorted_genes = sorted(genes.values(), key=lambda x: (x['feature'].seqid, x['feature'].start))
                for gene_info in sorted_genes:
                    gene_feature = gene_info['feature']
                    print(gene_feature, file=f)
                    # Get all transcripts of this gene (sorted by start position)
                    gene_transcripts = []
                    for transcript_list in gene_info['transcripts'].values():
                        for transcript in transcript_list:
                            gene_transcripts.append(transcript)
                    # Sort transcripts by start position
                    sorted_transcripts = sorted(gene_transcripts, key=lambda x: x.start)
                    for transcript in sorted_transcripts:
                        print(transcript, file=f)
                        # Get all sub-features of this transcript
                        transcript_id = transcript.id
                        if transcript_id in transcripts_features:
                            sub_features = transcripts_features[transcript_id]
                            # Sort by feature type priority and position
                            sorted_sub_features = sorted(
                                sub_features, 
                                key=lambda x: (
                                    feature_priority.get(x.featuretype, 999), 
                                    x.start
                                )
                            )
                            for sub_feature in sorted_sub_features:
                                print(sub_feature, file=f)

        # 1. Only retain necessary columns
        need = ['Chr', 'Strand', 'TrStart', 'TrEnd', 'TrID', 'GeneID', 'GeneName', 
                'SSC', 'TrStart_ref', 'TrEnd_ref', 'Predict_NMD', 'TIS_related_location', 'TTS_related_location']
        df = df[need].drop_duplicates()
        # 2. First parse exons for each transcript and record gene→chrom→strand mapping
        gene_chrom = {}
        gene_strand = {}
        tx_exons = {}          # trid -> [(s1,e1), (s2,e2), ...]
        gene_txs = {}          # gene_id  -> {trid1, trid2, ...}
        tx_rows = {}           # trid -> corresponding row data
        for _, r in df.iterrows():
            chrom = r['Chr']
            strand = r['Strand']
            gid = r['GeneID']
            trid = r['TrID']
            gene_chrom[gid] = chrom
            gene_strand[gid] = strand
            gene_txs.setdefault(gid, set()).add(trid)
            tx_rows[trid] = r  # Save row data
            # Parse SSC and integrate TrStart_ref and TrEnd_ref
            tr_start_ref = int(r['TrStart_ref'])
            tr_end_ref = int(r['TrEnd_ref'])
            # Parse splice sites in SSC
            block = list(map(int, r['SSC'].split('-')))
            if len(block) % 2:
                block = block[:-1]
            # Build complete exon structure: TrStart_ref + SSC + TrEnd_ref
            all_coords = [tr_start_ref]
            for coord in block:
                all_coords.append(coord)
            all_coords.append(tr_end_ref)
            # Sort coordinates (based on strand direction)
            all_coords.sort()
            # Build exons
            exons = []
            for i in range(0, len(all_coords)-1, 2):
                s, e = all_coords[i], all_coords[i+1]
                exons.append((s, e))
            tx_exons[trid] = exons
        # 3. Calculate genomic range for each gene (using TrStart_ref and TrEnd_ref)
        gene_span = {}
        for _, r in df.iterrows():
            gid = r['GeneID']
            s0 = int(r['TrStart_ref'])
            e0 = int(r['TrEnd_ref'])
            if gid not in gene_span:
                gene_span[gid] = [s0, e0]
            else:
                gene_span[gid][0] = min(gene_span[gid][0], s0)
                gene_span[gid][1] = max(gene_span[gid][1], e0)
        # 4. Collect all unique exons under each gene
        gene_exons = {}        # gid -> {(s,e), ...}
        for gid, trset in gene_txs.items():
            exon_pool = set()
            for trid in trset:
                exon_pool.update(tx_exons[trid])
            gene_exons[gid] = sorted(exon_pool)
        # 5. Create temp directory if it doesn't exist
        temp_dir = os.path.join(output_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_output_path = os.path.join(temp_dir, 'aidrs.transcript_model.gtf')
        final_output_path = os.path.join(output_dir, 'aidrs.transcript_model.gtf')
        # 6. Write GTF to temporary file first
        with open(temp_output_path, 'w') as fo:
            for gid in sorted(gene_chrom):
                chrom = gene_chrom[gid]
                strand = gene_strand[gid]
                g_s, g_e = gene_span[gid]
                # 5.1 gene line
                fo.write(f"{chrom}\tAIDRS\tgene\t{g_s}\t{g_e}\t.\t{strand}\t.\t"
                         f"gene_id \"{gid}\"; gene_name \"{df.loc[df.GeneID==gid, 'GeneName'].iloc[0]}\";\n")
                # 5.2 transcript line (retain each transcript)
                for trid in sorted(gene_txs[gid]):
                    r = tx_rows[trid]
                    tr_s = int(r['TrStart_ref'])
                    tr_e = int(r['TrEnd_ref'])
                    # Determine if there is CDS
                    has_cds = (r['Predict_NMD'] == 'Normal' and 
                              pd.notna(r['TIS_related_location']) and 
                              pd.notna(r['TTS_related_location']))
                    fo.write(f"{chrom}\tAIDRS\ttranscript\t{tr_s}\t{tr_e}\t.\t{strand}\t.\t"
                             f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\";\n")
                    if has_cds:
                        # Call your function to get CDS, UTR, etc.
                        try:
                            ribosome_data = simulate_ribosome_walk(r)
                            # Output exon (use exons from ribosome_data to ensure consistency with CDS/UTR)
                            for i, (es, ee) in enumerate(ribosome_data['exons'], 1):
                                fo.write(f"{chrom}\tAIDRS\texon\t{es}\t{ee}\t.\t{strand}\t.\t"
                                         f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\"; exon_number \"{i}\";\n")
                            # Output CDS
                            for cds_s, cds_e in ribosome_data['cds']:
                                fo.write(f"{chrom}\tAIDRS\tCDS\t{cds_s}\t{cds_e}\t.\t{strand}\t0\t"
                                         f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\";\n")
                            # Output start_codon
                            start_s, start_e = ribosome_data['start_codon']
                            fo.write(f"{chrom}\tAIDRS\tstart_codon\t{start_s}\t{start_e}\t.\t{strand}\t.\t"
                                     f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\";\n")
                            # Output stop_codon
                            stop_s, stop_e = ribosome_data['stop_codon']
                            fo.write(f"{chrom}\tAIDRS\tstop_codon\t{stop_s}\t{stop_e}\t.\t{strand}\t.\t"
                                     f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\";\n")
                            # Output UTR (unified UTR label)
                            for utr_s, utr_e in ribosome_data['utr5']:
                                fo.write(f"{chrom}\tAIDRS\tUTR\t{utr_s}\t{utr_e}\t.\t{strand}\t.\t"
                                         f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\";\n")
                            for utr_s, utr_e in ribosome_data['utr3']:
                                fo.write(f"{chrom}\tAIDRS\tUTR\t{utr_s}\t{utr_e}\t.\t{strand}\t.\t"
                                         f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\";\n")
                        except Exception as e:
                            print(f"Warning: Error processing transcript {trid}: {e}")
                            # Fallback to output exon only on error
                            for i, (es, ee) in enumerate(tx_exons[trid], 1):
                                fo.write(f"{chrom}\tAIDRS\texon\t{es}\t{ee}\t.\t{strand}\t.\t"
                                         f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\"; exon_number \"{i}\";\n")
                    else:
                        # No CDS, output exon only
                        for i, (es, ee) in enumerate(tx_exons[trid], 1):
                            fo.write(f"{chrom}\tAIDRS\texon\t{es}\t{ee}\t.\t{strand}\t.\t"
                                     f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\"; exon_number \"{i}\";\n")
        
        # 7. Sort the temporary GTF file by hierarchy and write to final output
        sort_gtf_by_hierarchy(temp_output_path, final_output_path)
        print("=== Transcript GTF file written to: ", final_output_path)

    def to_fasta(self, gtf_file: str, genome_fasta: str, output_dir: str) -> None:
        """
        Generate transcript FASTA file from GTF and genome FASTA
        Args:
            gtf_file: Path to the GTF file
            genome_fasta: Path to the reference genome FASTA file
            output_fasta: Path to the output transcript FASTA file
        """
        output_fasta = os.path.join(output_dir, 'aidrs.transcript_model.fasta')
        # Load genome sequence
        genome = pyfaidx.Fasta(genome_fasta)
        # Parse GTF file and extract transcript sequences
        db = gffutils.create_db(gtf_file, ':memory:', 
                            disable_infer_genes=True, 
                            disable_infer_transcripts=True,
                            merge_strategy='create_unique')
        # Dictionary to store transcript sequences
        transcript_sequences = {}
        # Process each transcript
        for transcript in db.features_of_type('transcript'):
            transcript_id = transcript.id
            gene_name = transcript.attributes.get('gene_name', [f'Gene_{transcript_id}'])[0]
            # Get all exons for this transcript
            exons = []
            for exon in db.children(transcript, featuretype='exon'):
                exons.append((exon.start, exon.end))
            # Sort exons by position
            exons.sort()
            if not exons:
                continue
            # Extract sequence from each exon and concatenate
            transcript_seq = ""
            chrom = transcript.seqid
            # Check if chromosome exists in genome
            if chrom not in genome:
                print(f"Warning: Chromosome {chrom} not found in genome file")
                continue
            for start, end in exons:
                # Extract exon sequence (GTF is 1-based, pyfaidx is 0-based)
                exon_seq = str(genome[chrom][start-1:end])
                transcript_seq += exon_seq
            # Handle reverse strand
            if transcript.strand == '-':
                # Reverse complement the sequence
                transcript_seq = self._reverse_complement(transcript_seq)
            transcript_sequences[transcript_id] = {
                'sequence': transcript_seq,
                'gene_name': gene_name
            }
        # Write sequences to FASTA file
        with open(output_fasta, 'w') as f:
            for tr_id, info in transcript_sequences.items():
                seq = info['sequence']
                gene_name = info['gene_name']
                # Write header with transcript ID and gene name
                header = f">{tr_id}|{gene_name}|length={len(seq)}"
                f.write(header + '\n')
                # Write sequence in 80-character lines
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i+80] + '\n')
        print(f"Transcript FASTA file written to: {output_fasta}")

    def _reverse_complement(self, seq: str) -> str:
        """
        Generate reverse complement of DNA sequence
        Args:
            seq: DNA sequence string
        Returns:
            Reverse complement sequence
        """
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        rev_seq = seq[::-1]  # Reverse the sequence
        rev_comp = ''.join([complement.get(base, base) for base in rev_seq.upper()])
        return rev_comp


    # ------------------------------------------------------------------
    # 4. Quantification and PolyA Profiling
    # ------------------------------------------------------------------
    # def quantify(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    #     df['length'] = df['sites'].apply(
    #         lambda s: sum(s[i+1]-s[i] for i in range(0, len(s), 2))
    #     )
    #     df['rpk'] = df['quantification'] / (df['length'] / 1000)
    #     tpm_scale = 1e6 / df['rpk'].sum()
    #     df['TPM'] = (df['rpk'] * tpm_scale).round(2)
    #     # Transcript weights
    #     df['weight'] = df['length'] / df.groupby(['GeneID', 'sample'])['length'].transform('sum')
    #     df['weighted_tpm'] = df['TPM'] * df['weight']  # weighted TPM for genes
    #     return {
    #         "transcript_counts": df.pivot_table(
    #             index=['TrID', 'GeneID', 'GeneName'],
    #             columns='sample',
    #             values='quantification',
    #             fill_value=0
    #         ),
    #         "transcript_tpm": df.pivot_table(
    #             index=['TrID', 'GeneID', 'GeneName'],
    #             columns='sample',
    #             values='TPM',
    #             fill_value=0
    #         ),
    #         "gene_counts": df.groupby(['GeneID', 'GeneName', 'sample'],
    #                                  observed=True)['quantification'].sum().unstack(fill_value=0),
    #         "gene_tpm": df.groupby(['GeneID', 'GeneName', 'sample'],
    #                               observed=True)['weighted_tpm'].sum().unstack(fill_value=0)
    #     }
    def quantify(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Calculate transcript length from exon sites (sum of exon lengths)
        df['length'] = df['sites'].apply(
            lambda s: sum(s[i+1] - s[i] for i in range(0, len(s), 2))
        )
        
        transcript_tpm_list = []
        gene_tpm_list = []
        transcript_cpm_list = []
        gene_cpm_list = []
        
        # Process each sample independently
        for sample, group in df.groupby('sample'):
            group = group.copy()
            
            # --- Transcript-level metrics ---
            # 1. Calculate transcript CPM (normalize by total reads per sample)
            total_reads = group['quantification'].sum()  # Total mapped reads for sample
            group['CPM'] = (group['quantification'] / total_reads * 1e6).round(2)
            
            # 2. Calculate transcript TPM (normalize by total RPK per sample)
            group['rpk'] = group['quantification'] / (group['length'] / 1000)
            total_rpk = group['rpk'].sum()
            group['TPM'] = (group['rpk'] / total_rpk * 1e6).round(2)
            transcript_tpm_list.append(group)
            transcript_cpm_list.append(group[['TrID', 'GeneID', 'GeneName', 'CPM', 'sample']])
            
            # --- Gene-level metrics (recalculate from raw counts) ---
            # Aggregate transcript counts to gene level
            gene_agg = group.groupby(['GeneID', 'GeneName'], observed=True).agg(
                gene_count=('quantification', 'sum'),  # Sum of transcript counts
                gene_length=('length', lambda x: np.sum(x * group.loc[x.index, 'quantification']) / x.sum())  # Expression-weighted length
            ).reset_index()
            
            # Calculate gene CPM (using total sample reads)
            gene_agg['gene_CPM'] = (gene_agg['gene_count'] / total_reads * 1e6).round(2)
            
            # Calculate gene TPM (using gene-level RPK)
            gene_agg['gene_rpk'] = gene_agg['gene_count'] / (gene_agg['gene_length'] / 1000)
            total_gene_rpk = gene_agg['gene_rpk'].sum()  # Should equal total_rpk
            gene_agg['gene_TPM'] = (gene_agg['gene_rpk'] / total_gene_rpk * 1e6).round(2)
            gene_agg['sample'] = sample
            gene_tpm_list.append(gene_agg)
            gene_cpm_list.append(gene_agg[['GeneID', 'GeneName', 'gene_CPM', 'sample']])
        
        # Merge results
        df_transcript = pd.concat(transcript_tpm_list, ignore_index=True)
        df_gene = pd.concat(gene_tpm_list, ignore_index=True)
        df_transcript_cpm = pd.concat(transcript_cpm_list, ignore_index=True)
        df_gene_cpm = pd.concat(gene_cpm_list, ignore_index=True)
        
        return {
            # Transcript counts
            "transcript_counts": df.pivot_table(
                index=['TrID', 'GeneID', 'GeneName'],
                columns='sample',
                values='quantification',
                fill_value=0
            ),
            
            # Transcript TPM
            "transcript_tpm": df_transcript.pivot_table(
                index=['TrID', 'GeneID', 'GeneName'],
                columns='sample',
                values='TPM',
                fill_value=0
            ),
            
            # Transcript CPM 
            "transcript_cpm": df_transcript_cpm.pivot_table(
                index=['TrID', 'GeneID', 'GeneName'],
                columns='sample',
                values='CPM',
                fill_value=0
            ),
            
            # Gene counts
            "gene_counts": df_gene.pivot_table(
                index=['GeneID', 'GeneName'],
                columns='sample',
                values='gene_count',
                fill_value=0
            ),
            
            # Gene TPM
            "gene_tpm": df_gene.pivot_table(
                index=['GeneID', 'GeneName'],
                columns='sample',
                values='gene_TPM',
                fill_value=0
            ),
            
            # Gene CPM 
            "gene_cpm": df_gene_cpm.pivot_table(
                index=['GeneID', 'GeneName'],
                columns='sample',
                values='gene_CPM',
                fill_value=0
            )
        }

    def polyA_len_profile(self, df: pd.DataFrame, out_dir) -> Dict[str, pd.DataFrame]:
        pattern = os.path.join(out_dir, 'temp', '*_flnc_correct.ssc')
        files = glob.glob(pattern)
        samples = []
        for f in files:
            basename = os.path.basename(f)
            sample = basename.replace('_flnc_correct.ssc', '')
            samples.append(sample)
        samples = sorted(set(samples))
        df_all = pl.DataFrame()
        for sample in samples:
            flnc_file = os.path.join(out_dir, 'temp', f'{sample}_flnc_correct.ssc')
            read_df = read_flnc(flnc_file)
            if 'TrStart_reads' in read_df.columns:
                read_df = read_df.rename(columns={'TrStart_reads': 'TrStart'})
            if 'TrEnd_reads' in read_df.columns:
                read_df = read_df.rename(columns={'TrEnd_reads': 'TrEnd'})
            chunk = (
                pl.from_pandas(read_df)
                .with_columns(
                    Chr=pl.col("Chr").cast(str),
                    Strand=pl.col("Strand").cast(str),
                    SSC=pl.col("SSC").cast(str),
                    sample=pl.lit(sample),
                )
                .group_by(['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd', 'sample'])
                .agg(pl.col("polyA_len").mean().alias("polyA_len"))
            )
            df_all = pl.concat([df_all, chunk])
        df_pd = df_all.to_pandas()
        df = df.merge(df_pd, on = ['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd', 'sample'], how = 'left')
        polyA_len_mtx = df.pivot_table(
            index=['TrID', 'GeneID', 'GeneName'],
            columns='sample',
            values='polyA_len',
            fill_value=0
        )
        return df, {
                "transcript_polyA_len": polyA_len_mtx,
                "gene_polyA_len": df.groupby(['GeneID', 'GeneName', 'sample'],
                                         observed=True)['polyA_len'].mean().unstack(fill_value=0)
            }


