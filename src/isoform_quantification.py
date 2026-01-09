from collections import defaultdict
import pandas as pd
import numpy as np
import os
from .common import read_flnc
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm



class IsoformQuantifier:
    def __init__(self, include_low_quality=False, use_truncate_weight=False, num_processes=10, min_samples_expr=1):
        """
        Initialize the IsoformQuantifier.
        
        Args:
            include_low_quality (bool): Whether to include low-quality reads
            use_truncate_weight (bool): Whether to use truncate weights for quantification (only relevant when include_low_quality is True)
            num_processes (int): Number of processes for parallel processing
            min_samples_expr (int): Minimum number of samples with expression to retain transcript in count matrix
        """
        self.include_low_quality = include_low_quality
        # use_truncate_weight is only meaningful when include_low_quality is True
        self.use_truncate_weight = use_truncate_weight if include_low_quality else False
        self.num_processes = num_processes
        self.min_samples_expr = min_samples_expr

    def quantify_sample(self, sample_name: str, transcript_model_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
        """
        Quantify a single sample and return quantification results.
        
        Args:
            sample_name (str): Name of the sample
            transcript_model_df (pd.DataFrame): Transcript model dataframe
            output_dir (str): Output directory path
            
        Returns:
            pd.DataFrame: Quantification results with read counts for each transcript
        """
        # Read high-quality reads
        high_quality_path = os.path.join(output_dir, 'temp', f'{sample_name}_flnc_correct.ssc')
        if not os.path.exists(high_quality_path):
            raise FileNotFoundError(f"High-quality SSC file not found: {high_quality_path}")
        
        high_quality_reads = read_flnc(high_quality_path)
        
        # Rename columns if needed (remove _reads suffix)
        if 'TrStart_reads' in high_quality_reads.columns:
            high_quality_reads = high_quality_reads.rename(columns={'TrStart_reads': 'TrStart'})
        if 'TrEnd_reads' in high_quality_reads.columns:
            high_quality_reads = high_quality_reads.rename(columns={'TrEnd_reads': 'TrEnd'})
        
        # Match high-quality reads to transcript models
        matched_reads = self._match_reads_to_transcripts(high_quality_reads, transcript_model_df)
        
        # If including low-quality reads
        if self.include_low_quality:
            # Read low-quality reads
            low_quality_path = os.path.join(output_dir, 'temp', f'{sample_name}_flnc.ssc')
            if os.path.exists(low_quality_path):
                low_quality_reads = read_flnc(low_quality_path)
                
                # Rename columns if needed (remove _reads suffix)
                if 'TrStart_reads' in low_quality_reads.columns:
                    low_quality_reads = low_quality_reads.rename(columns={'TrStart_reads': 'TrStart'})
                if 'TrEnd_reads' in low_quality_reads.columns:
                    low_quality_reads = low_quality_reads.rename(columns={'TrEnd_reads': 'TrEnd'})
                
                # Process low-quality reads that are not in high-quality set
                low_quality_only = self._process_low_quality_reads(
                    low_quality_reads, high_quality_reads, transcript_model_df
                )
                
                # Combine matched reads with low-quality reads
                matched_reads = pd.concat([matched_reads, low_quality_only], ignore_index=True)
        
        # Calculate truncate weights if needed
        if self.use_truncate_weight:
            matched_reads = self._calculate_truncate_weights(matched_reads, transcript_model_df)
        
        # Aggregate counts by transcript
        quant_results = self._aggregate_transcript_counts(matched_reads)
        
        # Add sample name
        quant_results['sample'] = sample_name
        
        return quant_results

    def _match_reads_to_transcripts(self, reads_df: pd.DataFrame, transcript_model_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match reads to transcript models based on genomic coordinates.
        
        Args:
            reads_df (pd.DataFrame): DataFrame containing read information
            transcript_model_df (pd.DataFrame): Transcript model dataframe
            
        Returns:
            pd.DataFrame: Reads with match status labels
        """
        # Keep original read information
        reads_with_info = reads_df.copy()
        
        # Merge reads with transcript models
        merged = (reads_with_info
                  .reset_index()
                  .merge(transcript_model_df[['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd',
                                              'TrID', 'GeneID', 'GeneName']],
                         on=['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd'],
                         how='left',
                         indicator=True)
                  .drop_duplicates()
                  .set_index(0)
                  .rename_axis('read_id'))
        
        # Label matched reads
        merged['match_status'] = merged['_merge'].apply(
            lambda x: 'match' if x == 'both' else 'miss_case1'
        )
        
        # Drop the merge indicator column
        merged = merged.drop(columns=['_merge'])
        
        return merged

    def _process_low_quality_reads(self, low_quality_reads: pd.DataFrame, 
                                  high_quality_reads: pd.DataFrame, 
                                  transcript_model_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process low-quality reads that are not in the high-quality set.
        
        Args:
            low_quality_reads (pd.DataFrame): Low-quality reads
            high_quality_reads (pd.DataFrame): High-quality reads
            transcript_model_df (pd.DataFrame): Transcript model dataframe
            
        Returns:
            pd.DataFrame: Processed low-quality reads with match status
        """
        # Create a key for identifying reads
        def create_read_key(df):
            return df['Chr'].astype(str) + '_' + df['Strand'].astype(str) + '_' + \
                   df['SSC'].astype(str) + '_' + df['TrStart'].astype(str) + '_' + df['TrEnd'].astype(str)
        
        # Add keys to both dataframes
        high_quality_reads = high_quality_reads.copy()
        low_quality_reads = low_quality_reads.copy()
        high_quality_reads['read_key'] = create_read_key(high_quality_reads)
        low_quality_reads['read_key'] = create_read_key(low_quality_reads)
        
        # Identify low-quality reads that are not in high-quality set
        low_only_keys = set(low_quality_reads['read_key']) - set(high_quality_reads['read_key'])
        low_quality_only = low_quality_reads[low_quality_reads['read_key'].isin(low_only_keys)].copy()
        
        # Remove the temporary key column
        low_quality_only = low_quality_only.drop(columns=['read_key'])
        # high_quality_reads = high_quality_reads.drop(columns=['read_key'])  # Not needed here
        
        # Initially label all as miss_case3 (single exon reads)
        low_quality_only['match_status'] = 'miss_case3'
        
        # Check if any reads are fully contained within transcript ranges (truncate reads)
        # For each transcript, check if low-quality read is contained within its range
        for idx, transcript in transcript_model_df.iterrows():
            # Get reads on the same chromosome and strand
            same_chr_strand = (
                (low_quality_only['Chr'] == transcript['Chr']) &
                (low_quality_only['Strand'] == transcript['Strand'])
            )
            
            # Check if read is contained within transcript range
            contained = (
                same_chr_strand &
                (low_quality_only['TrStart'] >= transcript['TrStart']) &
                (low_quality_only['TrEnd'] <= transcript['TrEnd'])
            )
            
            # Update status for contained reads
            low_quality_only.loc[contained, 'match_status'] = 'miss_case2'
        
        # Add transcript information where possible
        low_quality_only = low_quality_only.merge(
            transcript_model_df[['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd', 'TrID', 'GeneID', 'GeneName']],
            on=['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd'],
            how='left'
        )
        
        return low_quality_only

    def _calculate_truncate_weights(self, matched_reads: pd.DataFrame, 
                                   transcript_model_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate truncate weights for miss_case2 reads.
        
        Args:
            matched_reads (pd.DataFrame): Matched reads with status labels
            transcript_model_df (pd.DataFrame): Transcript model dataframe
            
        Returns:
            pd.DataFrame: Reads with truncate weights applied
        """
        # For now, we'll implement a simple truncate weight calculation
        # In a more complex implementation, this would calculate weights based on 
        # the proportion of matched transcripts in each gene
        
        # Get matched reads (for calculating proportions)
        matched = matched_reads[matched_reads['match_status'] == 'match'].copy()
        
        # Calculate transcript counts per gene
        gene_transcript_counts = matched.groupby('GeneID').size().to_dict()
        
        # Calculate total reads per gene
        gene_read_counts = matched.groupby('GeneID').size().to_dict()
        
        # For miss_case2 reads, distribute based on truncate weights
        miss_case2_mask = matched_reads['match_status'] == 'miss_case2'
        
        # This is a simplified implementation - in practice, you would calculate
        # truncate weights based on the specific overlap and distribution
        
        return matched_reads

    def _aggregate_transcript_counts(self, matched_reads: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate read counts by transcript.
        
        Args:
            matched_reads (pd.DataFrame): Matched reads with status labels
            
        Returns:
            pd.DataFrame: Aggregated counts by transcript
        """
        # Group by transcript information and count (using more memory-efficient method)
        matched = matched_reads[matched_reads['match_status'] == 'match']
        
        if matched.empty:
            # Return empty DataFrame with correct columns if no matched reads
            return pd.DataFrame(columns=['TrID', 'GeneID', 'GeneName', 'Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd', 'count'])
        
        try:
            # Use value_counts which is more memory efficient than groupby.size()
            group_cols = ['TrID', 'GeneID', 'GeneName', 'Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd']
            transcript_counts = matched[group_cols].value_counts().reset_index(name='count')
        except MemoryError as e:
            # If still running out of memory, try chunked processing
            print(f"Warning: Memory error during aggregation, attempting chunked processing: {e}")
            transcript_counts = self._chunked_aggregate_transcript_counts(matched, group_cols)
        
        # If using truncate weights, add those counts
        if self.use_truncate_weight:
            # This would add the weighted counts from miss_case2 reads
            # For now, we'll just count the matched reads
            pass
        
        return transcript_counts

    def _chunked_aggregate_transcript_counts(self, matched: pd.DataFrame, group_cols: list, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Chunked aggregation of transcript counts to handle large datasets with limited memory.
        
        Args:
            matched (pd.DataFrame): Matched reads dataframe
            group_cols (list): List of columns to group by
            chunk_size (int): Size of each chunk for processing
            
        Returns:
            pd.DataFrame: Aggregated counts by transcript
        """
        # If dataset is small enough, process directly
        if len(matched) <= chunk_size:
            return matched[group_cols].value_counts().reset_index(name='count')
        
        # Process in chunks
        chunk_results = []
        for i in range(0, len(matched), chunk_size):
            chunk = matched.iloc[i:i+chunk_size]
            chunk_count = chunk[group_cols].value_counts()
            chunk_results.append(chunk_count)
        
        # Combine all chunk results
        if chunk_results:
            combined_counts = pd.concat(chunk_results).groupby(level=0).sum()
            result_df = combined_counts.reset_index(name='count')
            result_df.columns = group_cols + ['count']
            return result_df
        else:
            return pd.DataFrame(columns=group_cols + ['count'])

    def quantify_all_samples(self, sample_names: List[str], transcript_model_df: pd.DataFrame, 
                            output_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Quantify all samples and generate output matrices.
        
        Args:
            sample_names (List[str]): List of sample names
            transcript_model_df (pd.DataFrame): Transcript model dataframe
            output_dir (str): Output directory path
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing all quantification matrices
        """
        # Process each sample with tqdm progress bar
        sample_results = []
        for sample_name in tqdm(sample_names, desc="Quantifying samples"):
            try:
                result = self.quantify_sample(sample_name, transcript_model_df, output_dir)
                sample_results.append(result)
            except Exception as e:
                print(f"Warning: Failed to quantify sample {sample_name}: {e}")
                continue
        
        if not sample_results:
            raise ValueError("No samples were successfully quantified")
        
        # Combine all sample results
        combined_results = pd.concat(sample_results, ignore_index=True)
        
        # Generate quantification matrices
        matrices = self._generate_quantification_matrices(combined_results, transcript_model_df)
        
        # Store sample results for use by caller
        self._last_sample_results = sample_results
        
        return matrices

    def _generate_quantification_matrices(self, quant_results: pd.DataFrame, 
                                         transcript_model_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate count, CPM, and TPM matrices for genes and transcripts.
        
        Args:
            quant_results (pd.DataFrame): Combined quantification results
            transcript_model_df (pd.DataFrame): Transcript model dataframe
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing all quantification matrices
        """
        # Check if quant_results has the necessary columns
        required_columns = ['TrID', 'GeneID', 'GeneName', 'count', 'sample']
        missing_columns = [col for col in required_columns if col not in quant_results.columns]
        if missing_columns:
            print(f"Warning: quant_results is missing required columns: {missing_columns}")
            # Create empty dataframe with expected columns if critical columns are missing
            if any(col in missing_columns for col in ['TrID', 'GeneID', 'GeneName']):
                print("Critical columns missing. Creating empty dataframe.")
                empty_df = pd.DataFrame(columns=['TrID', 'GeneID', 'GeneName', 'count', 'sample'])
                return {
                    "transcript_counts": empty_df,
                    "transcript_cpm": empty_df,
                    "transcript_tpm": empty_df,
                    "gene_counts": empty_df,
                    "gene_cpm": empty_df,
                    "gene_tpm": empty_df
                }
        
        # Transcript-level matrices
        transcript_count_matrix = quant_results.pivot_table(
            index=['TrID', 'GeneID', 'GeneName'],
            columns='sample',
            values='count',
            fill_value=0
        )
        # Ensure count matrix uses integer type
        transcript_count_matrix = transcript_count_matrix.astype(int)
        # Apply min_samples_expr filter
        transcript_count_matrix = transcript_count_matrix[transcript_count_matrix.iloc[:, 3:].gt(0).sum(axis=1) >= min(transcript_count_matrix.shape[1] - 3, self.min_samples_expr)]
        
        # Calculate transcript CPM
        transcript_cpm_data = []
        for sample in quant_results['sample'].unique():
            sample_data = quant_results[quant_results['sample'] == sample].copy()
            total_reads = sample_data['count'].sum()
            sample_data['CPM'] = (sample_data['count'] / total_reads * 1e6) if total_reads > 0 else 0.0
            transcript_cpm_data.append(sample_data[['TrID', 'GeneID', 'GeneName', 'sample', 'CPM']])
        
        transcript_cpm_df = pd.concat(transcript_cpm_data, ignore_index=True)
        transcript_cpm_matrix = transcript_cpm_df.pivot_table(
            index=['TrID', 'GeneID', 'GeneName'],
            columns='sample',
            values='CPM',
            fill_value=0.0
        )
        # Ensure CPM matrix uses float type
        transcript_cpm_matrix = transcript_cpm_matrix.astype(float)
        
        # Gene-level matrices (aggregated from transcript-level)
        gene_data = quant_results.groupby(['GeneID', 'GeneName', 'sample'], observed=True)['count'].sum().reset_index()
        gene_count_matrix = gene_data.pivot_table(
            index=['GeneID', 'GeneName'],
            columns='sample',
            values='count',
            fill_value=0
        )
        # Ensure count matrix uses integer type
        gene_count_matrix = gene_count_matrix.astype(int)
        
        # Calculate gene CPM
        gene_cpm_data = []
        for sample in gene_data['sample'].unique():
            sample_data = gene_data[gene_data['sample'] == sample].copy()
            total_reads = sample_data['count'].sum()
            sample_data['CPM'] = (sample_data['count'] / total_reads * 1e6) if total_reads > 0 else 0.0
            gene_cpm_data.append(sample_data[['GeneID', 'GeneName', 'sample', 'CPM']])
        
        gene_cpm_df = pd.concat(gene_cpm_data, ignore_index=True)
        gene_cpm_matrix = gene_cpm_df.pivot_table(
            index=['GeneID', 'GeneName'],
            columns='sample',
            values='CPM',
            fill_value=0.0
        )
        # Ensure CPM matrix uses float type
        gene_cpm_matrix = gene_cpm_matrix.astype(float)
        
        return {
            "transcript_counts": transcript_count_matrix,
            "transcript_cpm": transcript_cpm_matrix,
            "gene_counts": gene_count_matrix,
            "gene_cpm": gene_cpm_matrix
        }
    
    def intersect_matrices_with_model(self, quantification_matrices: Dict[str, pd.DataFrame], 
                                     transcript_model_df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        """
        Intersect all quantification matrices with transcript_model_df based on merge_keys.
        Retains transcripts that exist in transcript_model_df, filling missing values with 0.
        
        Args:
            quantification_matrices (Dict[str, pd.DataFrame]): Dictionary of quantification matrices
            transcript_model_df (pd.DataFrame): Transcript model dataframe
            
        Returns:
            Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]: 
                - Filtered quantification matrices with missing values filled as 0
                - Filtered transcript model dataframe
                - Quantification dataframe for final merging
        """
        merge_keys = ['TrID', 'GeneID', 'GeneName']
        
        # Get the set of transcripts that exist in transcript_model_df
        df_transcript_keys = transcript_model_df[merge_keys].drop_duplicates()
        
        # Create a set of keys from transcript_model_df for intersection
        transcript_key_set = set(df_transcript_keys.itertuples(index=False, name=None))
        
        # Filter each quantification matrix to only include rows that exist in transcript_model_df
        filtered_matrices = {}
        
        # First, get all sample names from the matrices
        all_samples = set()
        for name, matrix in quantification_matrices.items():
            if hasattr(matrix, 'columns'):
                # For matrices with sample columns
                sample_cols = [col for col in matrix.columns if col not in merge_keys]
                all_samples.update(sample_cols)
        
        for name, matrix in quantification_matrices.items():
            # Convert matrix to dataframe for easier manipulation
            if isinstance(matrix.index, pd.MultiIndex):
                matrix_df = matrix.reset_index()
            else:
                matrix_df = matrix.reset_index()
                
            # Check if matrix has the merge key columns directly
            if all(key in matrix_df.columns for key in merge_keys):
                # Filter matrix to only include rows that exist in transcript_model_df
                filtered_matrix_df = matrix_df[matrix_df[merge_keys].apply(tuple, axis=1).isin(transcript_key_set)]
                
                # Reindex to include all transcripts from transcript_model_df, filling missing values with 0
                # Create a dataframe with all transcripts from transcript_model_df
                transcript_index_df = transcript_model_df[merge_keys].drop_duplicates()
                
                # Merge with filtered matrix to ensure all transcripts are included
                reindexed_matrix_df = transcript_index_df.merge(filtered_matrix_df, on=merge_keys, how='left')
                
                # Fill NaN values with 0 for numeric columns (sample columns)
                sample_columns = [col for col in reindexed_matrix_df.columns if col not in merge_keys]
                for col in sample_columns:
                    if reindexed_matrix_df[col].dtype in ['int64', 'float64']:
                        reindexed_matrix_df[col] = reindexed_matrix_df[col].fillna(0)
                
                # Set index back if it was a MultiIndex
                if isinstance(quantification_matrices[name].index, pd.MultiIndex):
                    # Try to set the same index as original
                    try:
                        filtered_matrix = reindexed_matrix_df.set_index(quantification_matrices[name].index.names)
                    except:
                        # If that fails, keep as regular index
                        filtered_matrix = reindexed_matrix_df
                else:
                    # For gene-level matrices, keep the original index structure
                    filtered_matrix = reindexed_matrix_df
                
                filtered_matrices[name] = filtered_matrix
            else:
                # If matrix doesn't have the key columns, keep as is
                filtered_matrices[name] = matrix
        
        # Create df_quant from the transcript count matrix (assuming it exists)
        # This fixes the undefined df_quant issue
        if 'transcript_counts' in filtered_matrices:
            df_quant = filtered_matrices['transcript_counts'].reset_index()
        else:
            # If transcript_counts doesn't exist, use the first available matrix
            first_matrix_name = list(filtered_matrices.keys())[0]
            df_quant = filtered_matrices[first_matrix_name].reset_index()
        
        # Add merge_keys to df_quant by joining with transcript_model_df
        # df_quant has TrID, GeneID, GeneName from index, but we need Chr, Strand, SSC, TrStart, TrEnd
        if 'TrID' in df_quant.columns:
            # Join with transcript_model_df to get merge_keys
            df_quant_with_keys = df_quant.merge(
                transcript_model_df[merge_keys].drop_duplicates(),
                on=merge_keys,
                how='left'
            )
            df_quant = df_quant_with_keys
        else:
            # If TrID is not available, we can't join with transcript_model_df
            # In this case, we'll have to work with what we have
            pass
        
        # Filter transcript_model_df to only include rows that exist in transcript_model_df
        # (which is essentially keeping it as is, but ensuring consistent structure)
        df_filtered_transcript_model = transcript_model_df.copy()
        
        # Additional filtering: Remove transcripts that have count=0 in all samples
        # Get the transcript count matrix to check counts
        transcripts_to_keep_mask = None
        if 'transcript_counts' in filtered_matrices:
            count_matrix = filtered_matrices['transcript_counts']
            # Reset index to work with columns
            if hasattr(count_matrix, 'reset_index'):
                count_df = count_matrix.reset_index() if isinstance(count_matrix.index, pd.MultiIndex) else count_matrix
            else:
                count_df = count_matrix
            
            # Identify sample columns (excluding merge keys and metadata)
            sample_cols = [col for col in count_df.columns if col not in merge_keys]
            
            # Check if there are any sample columns
            if sample_cols:
                # Find transcripts that have at least one sample with count >= 1
                # Using max(axis=1) to check the maximum count for each transcript across all samples
                max_counts_per_transcript = count_df[sample_cols].max(axis=1)
                transcripts_to_keep_mask = max_counts_per_transcript >= 1
                
                # Filter count matrix to keep only transcripts with at least one sample having count >= 1
                filtered_count_matrix = count_df[transcripts_to_keep_mask]
                
                # Get the transcript keys for the filtered transcripts
                if all(key in filtered_count_matrix.columns for key in merge_keys):
                    filtered_transcript_keys = filtered_count_matrix[merge_keys].drop_duplicates()
                    
                    # Filter df_filtered_transcript_model to only include these transcripts
                    # Create a set of tuples for faster lookup
                    filtered_key_set = set(filtered_transcript_keys.itertuples(index=False, name=None))
                    df_filtered_transcript_model = df_filtered_transcript_model[
                        df_filtered_transcript_model[merge_keys].apply(tuple, axis=1).isin(filtered_key_set)
                    ]
        
        # For df_quant, ensure all transcripts from transcript_model_df are included
        if all(key in df_quant.columns for key in merge_keys):
            # Create a dataframe with all transcripts from transcript_model_df
            transcript_index_df = df_filtered_transcript_model[merge_keys].drop_duplicates()
            
            # Merge with df_quant to ensure all transcripts are included
            df_filtered_quant = transcript_index_df.merge(df_quant, on=merge_keys, how='left')
            
            # Fill NaN values with 0 for numeric columns (sample columns)
            sample_columns = [col for col in df_filtered_quant.columns if col not in merge_keys]
            for col in sample_columns:
                if df_filtered_quant[col].dtype in ['int64', 'float64']:
                    df_filtered_quant[col] = df_filtered_quant[col].fillna(0)
            
            # Ensure proper data types based on matrix name
            # This is a simplified approach - in practice, you might need to check the original matrix type
            for col in sample_columns:
                if df_filtered_quant[col].dtype in ['int64', 'float64']:
                    # Default to int for count-like data, float for CPM-like data
                    if 'cpm' in col.lower() or 'CPM' in col:
                        df_filtered_quant[col] = df_filtered_quant[col].astype(float)
                    else:
                        df_filtered_quant[col] = df_filtered_quant[col].astype(int)
        else:
            # If df_quant doesn't have merge_keys, we can't filter it by these keys
            # Just use it as is
            df_filtered_quant = df_quant.copy()
        
        # Apply the same filtering to all quantification matrices to ensure consistency
        if transcripts_to_keep_mask is not None:
            # Update filtered_matrices to only include transcripts with at least one sample having count >= 1
            for name, matrix in filtered_matrices.items():
                # Reset index to work with columns
                if hasattr(matrix, 'reset_index'):
                    matrix_df = matrix.reset_index() if isinstance(matrix.index, pd.MultiIndex) else matrix
                else:
                    matrix_df = matrix
                
                # Check if matrix has the merge key columns and the same number of rows as count matrix
                if (all(key in matrix_df.columns for key in merge_keys) and 
                    len(matrix_df) == len(transcripts_to_keep_mask)):
                    # Apply the same filtering mask
                    filtered_matrix_df = matrix_df[transcripts_to_keep_mask].copy()
                    
                    # Set index back if it was a MultiIndex
                    if isinstance(matrix.index, pd.MultiIndex):
                        # Try to set the same index as original
                        try:
                            filtered_matrix = filtered_matrix_df.set_index(matrix.index.names)
                            filtered_matrices[name] = filtered_matrix
                        except:
                            # If that fails, keep as regular index
                            filtered_matrices[name] = filtered_matrix_df
                    else:
                        # For gene-level matrices, keep the original index structure
                        filtered_matrices[name] = filtered_matrix_df
        
        # Ensure proper data types: count matrices as int, CPM matrices as float
        for name, matrix in filtered_matrices.items():
            if name in ['transcript_counts', 'gene_counts']:
                # Convert count matrices to integer type
                for col in sample_columns:
                    if matrix[col].dtype in ['int64', 'float64']:
                        matrix[col] = matrix[col].astype(int)
            elif name in ['transcript_cpm', 'gene_cpm']:
                # Convert CPM matrices to float type
                for col in sample_columns:
                    if matrix[col].dtype in ['int64', 'float64']:
                        matrix[col] = matrix[col].astype(float)
        
        return filtered_matrices, df_filtered_transcript_model, df_filtered_quant