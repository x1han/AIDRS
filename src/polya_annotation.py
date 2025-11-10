import numpy as np
import pandas as pd
import os, re
import polars as pl
from .common import correct_flnc, read_flnc

class polyAnnotator:
    def __init__(self, args = None):
        self.args = args

    @staticmethod
    def add_polya_frac(df: pd.DataFrame,
                       df_polya: pd.DataFrame,
                       merge_cols=None) -> pd.DataFrame:
        """
        Add a polyA_frac column to df: for each row, find fully matching rows in df_polya 
        based on Chr, Strand, SSC, TrStart, TrEnd, and sum their polyA values to the current row.

        Parameters
        ----
        df : pd.DataFrame
            Table to which a new column needs to be added
        df_polya : pd.DataFrame
            Table containing polyA signals
        merge_cols : list[str], optional
            List of column names for matching, default is ['Chr','Strand','SSC','TrStart','TrEnd']

        Returns
        ----
        pd.DataFrame
            df with added polyA_frac column (in-place modification + return)
        """

        if merge_cols is None:
            merge_cols = ['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd']

        df_polya['polyA_frac'] = df_polya['polyA_freq'] / df_polya['raw_read_freq']

        # First group by merge_cols in df_polya and sum the polyA values
        out = df.merge(
            df_polya[merge_cols + ['polyA_frac']], on=merge_cols, how='left')
        
        out['polyA_frac'] = out['polyA_frac'].fillna(0)
        # out['polyA_frac'] = out['polyA'] / out['frequency']
        # If you want to maintain the original column order, you can adjust the column order back
        return out

    # def anno_polya(self, df, flnc):
    #     """Single sample polyA annotation (maintain backward compatibility)"""
    #     read_df = pd.read_csv(flnc, sep='\t', names=["Chr", "Strand", "TrStart", "TrEnd", "SSC", "identity", "coverage", "polyA"])
    #     df_polya = self.correct_flnc(df, read_df)
    #     df = self.add_polya_frac(df, df_polya)
    #     return df
    
    def anno_polya(self, df, flnc_files):
        """Multi-sample polyA annotation
        
        Parameters:
        ----
        df : pd.DataFrame
            Dataframe to be annotated
        flnc_files : list[str]
            List of multiple flnc.ssc file paths
            
        Returns:
        ----
        pd.DataFrame
            Dataframe with added polyA_frac column
        """
        # Read and merge all samples' flnc.ssc files
        # First round flag
        first_round = True
        acc_df = pl.DataFrame(
            schema={
                "Chr": str,
                "Strand": str,
                "SSC": str,
                "TrStart": int,
                "TrEnd": int,
                "polyA_freq": int,
                "raw_read_freq": int,
            }
        )

        for flnc_file in flnc_files:
            if not os.path.exists(flnc_file):
                continue

            read_df = read_flnc(flnc_file)
            read_df = correct_flnc(df, read_df, args=self.args)
            dir_name = os.path.dirname(flnc_file)                # Original directory
            base_name = os.path.basename(flnc_file)
            out_name = re.sub(r'_flnc\.ssc$', '_flnc_correct.ssc', base_name)
            out_file = os.path.join(dir_name, out_name)
            read_df.to_csv(out_file, sep='\t', index=True, header=False)

            # Calculate statistics for current file
            cur = (
                pl.from_pandas(read_df)
                .with_columns(
                    polyA_freq=(pl.col("polyA_len") != 0).cast(int),
                    Chr=pl.col("Chr").cast(str),
                    Strand=pl.col("Strand").cast(str),
                    SSC=pl.col("SSC").cast(str),
                )
                .group_by(["Chr", "Strand", "SSC", "TrStart", "TrEnd"])
                .agg(
                    polyA_freq=pl.col("polyA_freq").sum(),
                    raw_read_freq=pl.col("polyA_freq").count(),
                )
            )

            if first_round:
                acc_df = cur
                first_round = False
            else:
                acc_df = (
                    pl.concat([
                        acc_df.select("Chr", "Strand", "SSC", "TrStart", "TrEnd", "polyA_freq", "raw_read_freq"),
                        cur.select("Chr", "Strand", "SSC", "TrStart", "TrEnd", "polyA_freq", "raw_read_freq"),
                    ])
                    .group_by(["Chr", "Strand", "SSC", "TrStart", "TrEnd"])
                    .agg(
                        pl.col("polyA_freq").sum(),
                        pl.col("raw_read_freq").sum(),
                    )
                )

        # Final pandas result
        df_polya = acc_df.to_pandas()
                
        # Perform correction and annotation
        df = self.add_polya_frac(df, df_polya)
        
        # Ensure TrStart and TrEnd columns are of int type
        if 'TrStart' in df.columns:
            df['TrStart'] = df['TrStart'].astype(int)
        if 'TrEnd' in df.columns:
            df['TrEnd'] = df['TrEnd'].astype(int)
        
        return df