#!/usr/bin/env python

import multiprocessing as mp
from functools import partial
import subprocess
import os
import sys
import pandas as pd
import numpy as np

class TSS_Puffin:
    def __init__(self, genome, num_processes=8, tmp_path='temp', puffin_prediction_threshold=0.02):
        self.num_processes = num_processes
        self.tmp_path = tmp_path
        self.genome = genome
        self.puffin_prediction_threshold = puffin_prediction_threshold

    
    @staticmethod
    def run_puffin(df, genome, tmp_path):
        if df.empty:
            return

        Chrom = df['Chr'].unique()[0]
        Strand = df['Strand'].unique()[0]

        tss_col = 'TrStart' if Strand == '+' else 'TrEnd'

        tss_sites_raw = df[tss_col].unique()

        puffin_in_path = f'{tmp_path}/puffin_in'
        puffin_out_path = f'{tmp_path}/puffin_out'
        os.makedirs(puffin_in_path, exist_ok=True)
        os.makedirs(puffin_out_path, exist_ok=True)

        out_file = os.path.join(puffin_in_path, f'{Chrom}_{Strand}.tsv')

        with open(out_file, 'w') as fout:
            fout.write('chr\tstart\tend\tstrand\n')
            for tss_site in tss_sites_raw:
                Start = tss_site - 500
                End   = tss_site + 500
                fout.write(f'{Chrom}\t{Start}\t{End}\t{Strand}\n')

        current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        puffin_script = os.path.join(current_dir, 'aided', 'Puffin', 'puffin.py')
        cmd = [sys.executable, puffin_script, 
        "region", 
        "--output", puffin_out_path, 
        "--genome", genome,
        out_file]
        subprocess.run(cmd, check=True)
    
    @staticmethod
    def correct_puffin(df, puffin_prediction_threshold=0.02):

        def correct_tss(row):
            strand = row['Strand']
            tss_col = 'TrStart' if strand == '+' else 'TrEnd'
            puffin_15bp = row['Puffin_TSS_15bp']
            
            if (isinstance(puffin_15bp, tuple) and 
                len(puffin_15bp) >= 1 and 
                puffin_15bp[0] != 'no' and 
                puffin_15bp[1] > puffin_prediction_threshold ):
                row = row.copy()
                # row[tss_col] = row[tss_col] - puffin_15bp[0] + 1
                row[tss_col] = row[tss_col] + puffin_15bp[0] - 1 if strand == '+' else row[tss_col] - puffin_15bp[0] + 1
            
            return row
        
        df = df.apply(correct_tss, axis=1)

        puffin_cols = ['Puffin_TSS_15bp', 'Puffin_TSS_50bp']
        df[puffin_cols] = df[puffin_cols].apply(lambda col: col.str[1])

        group_cols = ['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd']
        
        agg_dict = {col: 'first' for col in df.columns if col != 'frequency'}
        agg_dict['frequency'] = 'sum'
        
        df = df.groupby(group_cols, as_index=False).agg(agg_dict)

        return df

        
        


    def tss_anno_by_puffin(self, df):
        df_groups = [g for _, g in df.groupby(['Chr','Strand'], observed=True)]
        with mp.Pool(self.num_processes) as pool:
            pool.map(partial(TSS_Puffin.run_puffin, genome=self.genome, tmp_path=self.tmp_path),
            df_groups
        )

    def aggr_puffin_result(self, df, puffin_out_path):

        def pick_nearest_peak(row):
            from scipy.signal import find_peaks

            x = row.values.astype(float)

            peaks, props = find_peaks(x, height=0.01)
            if len(peaks) == 0:
                return "no"

            true_idx = peaks - 50
            nearest = true_idx[np.argmin(np.abs(true_idx))]-1
            return int(nearest)
        
        def pick_highest_peak_with_value(row, search_range=15):
            from scipy.signal import find_peaks
            
            x = row.values.astype(float)
            center_idx = 50  # Center position at index 50 (corresponding to position 0)
            
            # Find all peaks
            peaks, props = find_peaks(x, height=0.01)
            if len(peaks) == 0:
                return ("no", "no")
            
            # Filter peaks within specified range
            valid_peaks = []
            peak_heights = []
            
            for peak_idx in peaks:
                # Calculate peak offset relative to center
                offset = peak_idx - center_idx
                if abs(offset) <= search_range:
                    valid_peaks.append(peak_idx)
                    peak_heights.append(x[peak_idx])
            
            if len(valid_peaks) == 0:
                return ("no", "no")
            
            # Find the highest peak
            max_height_idx = np.argmax(peak_heights)
            highest_peak_idx = valid_peaks[max_height_idx]
            highest_peak_value = peak_heights[max_height_idx]
            
            # Return position offset and peak value
            position_offset = int(highest_peak_idx - center_idx)
            return (position_offset, round(highest_peak_value, 4))
        


        rows = []

        for name in os.listdir(puffin_out_path):
            if not name.startswith("puffin_") or not name.endswith(".csv"):
                continue

            base, _ = os.path.splitext(name)
            _, Chr_, Start, End, Strand = base.split("_", 4)
            Strand = Strand.replace("minus", "-").replace("plus", "+")

            pred = pd.read_csv(os.path.join(puffin_out_path, name), index_col=0).loc["Prediction"]

            idx_names = list(range(-174, -174 + len(pred)))
            row = [Chr_, int(Start), int(End), Strand] + pred.values.tolist()
            rows.append(row)

        first_len = len(rows[0]) - 4
        cols = ["Chr", "Start", "End", "Strand"] + list(range(-174, -174 + first_len))
        merged_puffin_res = pd.DataFrame(rows, columns=cols)
        merged_puffin_res = merged_puffin_res[["Chr", "Start", "End", "Strand"] + list(range(-50, 51))]

        # merged_puffin_res = pd.concat([
        #     merged_puffin_res[["Chr", "Start", "End", "Strand"]], 
        #     pd.DataFrame(merged_puffin_res[list(range(-50, 51))].apply(
        #         pick_nearest_peak, axis=1
        #     ), columns=['Puffin_TSS'])
        # ], axis=1)
        merged_puffin_res = pd.concat([
            merged_puffin_res[["Chr", "Start", "End", "Strand"]], 
            pd.DataFrame(merged_puffin_res[list(range(-50, 51))].apply(
                lambda row: pick_highest_peak_with_value(row, 15), axis=1
            ), columns=['Puffin_TSS_15bp']), 
            pd.DataFrame(merged_puffin_res[list(range(-50, 51))].apply(
                lambda row: pick_highest_peak_with_value(row, 50), axis=1
            ), columns=['Puffin_TSS_50bp'])
        ], axis=1)
        merged_puffin_res['query_tss'] = ((merged_puffin_res['Start'] + merged_puffin_res['End']) / 2).astype(int)
        merged_puffin_res = merged_puffin_res.drop(["Start", "End"], axis=1)

        df_groups = [g for _, g in df.groupby(['Chr','Strand'], observed=True)]

        merged_df_list = []
        for df_group in df_groups:
            Chrom = df_group['Chr'].unique()[0]
            Strand = df_group['Strand'].unique()[0]

            tss_col = 'TrStart' if Strand == '+' else 'TrEnd'

            puffin_subset = merged_puffin_res[
                (merged_puffin_res['Chr'] == Chrom) & 
                (merged_puffin_res['Strand'] == Strand)
            ]

            df_merged = df_group.merge(
                puffin_subset, 
                left_on=["Chr", 'Strand', tss_col], 
                right_on=["Chr", "Strand", 'query_tss'], 
                how='outer'
            )
            
            merged_df_list.append(df_merged)

        if merged_df_list:
            df = pd.concat(merged_df_list, ignore_index=True)
            df = df.drop(['query_tss'], axis=1)

        # Delete generated temp folder
        import shutil
        if os.path.exists(puffin_out_path):
            shutil.rmtree(puffin_out_path, ignore_errors=True)
        
        # Also delete puffin_in folder
        puffin_in_path = os.path.join(os.path.dirname(puffin_out_path), 'puffin_in')
        if os.path.exists(puffin_in_path):
            shutil.rmtree(puffin_in_path, ignore_errors=True)

        return TSS_Puffin.correct_puffin(df, self.puffin_prediction_threshold)