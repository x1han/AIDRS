import pandas as pd
import numpy as np
import multiprocessing as mp
import logging

logger = logging.getLogger(__name__)

class GeneClustering:
    """
    gene-level grouping
    """

    def __init__(self, num_processes=None):
        self.num_processes = num_processes

    @staticmethod
    def _single_strand_interval_clustering(df):
        df = df.copy()
        df['start'] = df['SSC'].str.split('-').str[0].astype(np.int32)
        df['end'] = df['SSC'].str.split('-').str[-1].astype(np.int32)

        df = df.sort_values(by='start')
        ends = df['end'].cummax().shift(1, fill_value=df['end'].iloc[0])
        group_ids = (df['start'] > ends).cumsum()
        df['Group'] = group_ids

        return df.drop(columns=['start', 'end'])

    @staticmethod
    def _cluster_for_chr(df):
        results = []
        global_group_id = 0
        for strand, strand_group in df.groupby('Strand', observed=True):
            clustered = GeneClustering._single_strand_interval_clustering(strand_group)
            # Reassign globally unique Group ID
            unique_groups = clustered['Group'].unique()
            group_mapping = {old_id: global_group_id + i for i, old_id in enumerate(unique_groups)}
            clustered['Group'] = clustered['Group'].map(group_mapping)
            global_group_id += len(unique_groups)
            results.append(clustered)

        return pd.concat(results, ignore_index=True)

    def cluster(self, df):
        if df.empty:
            return df.assign(Group=pd.Series(dtype=np.int32))

        df = df.astype({
            'Chr': 'category',
            'Strand': 'category',
            'SSC': str
        })

        grouped = [group for _, group in df.groupby('Chr', observed=True)]
        num_processes = min(self.num_processes, len(grouped))

        ctx = mp.get_context("spawn")
        with ctx.Pool(num_processes) as pool:
            results = pool.map(self._cluster_for_chr, grouped)

        clustered_df = pd.concat(results, ignore_index=True)
        
        # Ensure Group IDs are also unique across different chromosomes
        global_group_id = 0
        final_results = []
        for chr_result in results:
            if len(chr_result) > 0:
                unique_groups = chr_result['Group'].unique()
                group_mapping = {old_id: global_group_id + i for i, old_id in enumerate(unique_groups)}
                chr_result_copy = chr_result.copy()
                chr_result_copy['Group'] = chr_result_copy['Group'].map(group_mapping)
                global_group_id += len(unique_groups)
                final_results.append(chr_result_copy)
        
        if final_results:
            clustered_df = pd.concat(final_results, ignore_index=True)
        else:
            clustered_df = pd.DataFrame()
        
        return clustered_df.reset_index(drop=True)
