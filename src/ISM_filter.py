import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
import logging

logger = logging.getLogger("AIDRS")

class TruncationProcessor:
    def __init__(self, threshold_truncation_source_freq=0.5, threshold_truncation_group_freq=0.5, trunc_simp_filter=False, num_processes=10):
        self.threshold_truncation_source_freq = threshold_truncation_source_freq
        self.threshold_truncation_group_freq = threshold_truncation_group_freq
        self.trunc_simp_filter = trunc_simp_filter
        self.num_processes = num_processes

    def _assess_truncation_for_Chr(self, df_clustered_Chr):
        df = df_clustered_Chr.copy()
        df['sourceSSC_counts'] = 0
        df['trun_source_freq'] = 0

        grouped_dict = {}
        for _, row in df.iterrows():
            key = (row['Chr'], row['Strand'], row['Group'])
            grouped_dict.setdefault(key, []).append(row)

        for index, row in df.iterrows():
            key = (row['Chr'], row['Strand'], row['Group'])
            sourceSSC_counts = 0
            trun_source_freq = 0
            truncation_source = []
            for other_row in grouped_dict[key]:
                if row['SSC'] != other_row['SSC'] and row['SSC'] in other_row['SSC']:
                    sourceSSC_counts += 1
                    trun_source_freq += other_row['frequency']
                    truncation_source.append(other_row['SSC'])

            df.at[index, 'sourceSSC_counts'] = sourceSSC_counts
            df.at[index, 'trun_source_freq'] = trun_source_freq
            df.at[index, 'truncation_source'] = ','.join(truncation_source) if truncation_source else 'full'
        return df

    def assess_truncation(self, df_clustered, ref_anno=None):
        Chr_groups = df_clustered.groupby(['Chr','Strand'],observed=True)
        Chr_list = [group for _, group in Chr_groups]

        with Pool(self.num_processes) as pool:
            results = pool.map(self._assess_truncation_for_Chr, Chr_list)

        df = pd.concat(results)
        
        # Calculate ratios instead of logFC
        df['source_freq_ratio'] = np.where(
            df['truncation_source'] != 'full',
            df['frequency'] / (df['frequency'] + df['trun_source_freq']),
            np.inf
        )
        df['group_freq'] = df.groupby(['Chr', 'Strand', 'Group'],observed=True)['frequency'].transform('sum')
        df['group_freq_ratio'] = np.where(
            df['truncation_source'] != 'full',
            df['frequency'] / (df['frequency'] + df['group_freq']),
            np.inf
        )

        def truncation_classify(row):
            if row['source_freq_ratio'] >= self.threshold_truncation_source_freq and \
               row['group_freq_ratio'] >= self.threshold_truncation_group_freq:
                return 'no'
            else:
                return 'yes'

        df['truncation'] = df.apply(truncation_classify, axis=1)
        
        # Apply simple filter if enabled
        if self.trunc_simp_filter:
            if ref_anno is not None:
                # If reference annotation is provided, classify isoforms and keep 'no' or 'FSM'
                from .isoform_classify import IsoformClassifier
                isoformclassifier = IsoformClassifier(num_processes=self.num_processes)
                df = isoformclassifier.add_category(df, ref_anno)
                original_count = len(df)
                df = df[(df['truncation'] == 'no') | (df['category'] == 'FSM')]
                filtered_count = len(df)
                logger.info(f"\tTruncation filtered: Retained {filtered_count} of {original_count} transcripts ({filtered_count/original_count*100:.2f}%).")
                # Remove the category column as it's no longer needed
                if 'category' in df.columns:
                    df = df.drop(columns=['category'])
            else:
                # If no reference annotation, only keep 'no'
                original_count = len(df)
                df = df[df['truncation'] == 'no']
                filtered_count = len(df)
                logger.info(f"\tTruncation filtered: Retained {filtered_count} of {original_count} transcripts ({filtered_count/original_count*100:.2f}%).")
        
        # Drop temporary columns
        df = df.drop(columns=['group_freq', 'sourceSSC_counts', 'trun_source_freq', 'source_freq_ratio', 'group_freq_ratio', 'truncation_source']).reset_index(drop=True)
        
        return df