import pandas as pd
import multiprocessing as mp
from .gene_grouping import GeneClustering
from .ISM_filter import _parse_introns, _is_contiguous_intron_subchain


class IsoformClassifier:
    def __init__(self, num_processes=10):
        self.num_processes = num_processes

    @staticmethod
    def _prepare_reference_sets(df_ref):
        ref_ssc_set = set(df_ref['SSC'].values)
        ref_site_set = set(map(int, df_ref['SSC'].str.split('-').explode().unique()))
        return ref_ssc_set, ref_site_set

    @staticmethod
    def _build_reference_intron_list(df_ref):
        """Pre-compute per-reference intron chains for the ISM sub-chain check.

        Returns a list of intron-list records, one per reference row, in the
        order they appear in ``df_ref``.  Paired with the corresponding
        ``SSC`` string so the per-row ISM check can pair them up.
        """
        ref_ssc_values = df_ref['SSC'].values
        ref_introns = [
            _parse_introns(tr_start, ssc, tr_end)
            for tr_start, ssc, tr_end in zip(
                df_ref['TrStart'].values, ref_ssc_values, df_ref['TrEnd'].values
            )
        ]
        return list(zip(ref_ssc_values, ref_introns))

    @staticmethod
    def _get_isoform_category_for_row(row_ssc, row_sites, ref_ssc_set, ref_site_set):
        if row_ssc in ref_ssc_set:
            return 'FSM'

        if set(row_sites).issubset(ref_site_set):
            return 'NIC'

        return 'NNC'

    @staticmethod
    def _get_isoform_category_for_group(df_group):
        # Reverted to v0.3 form. v1.0's earlier expansion added per-row intron
        # parsing via _parse_introns(row['TrStart'], row['SSC'], row['TrEnd']),
        # but TrStart is a numpy array at this stage (from TrStart_reads), not
        # a scalar. Skipping the intron path entirely avoids the array→scalar
        # cast bug at isoform_classify.py:87.
        results = []
        for _, group in df_group.groupby(['Chr', 'Strand', 'Group'], observed=True):
            df_ref = group[group['source'] == 'ref']
            df_data = group[group['source'] == 'data']

            if df_data.empty:
                continue

            if df_ref.empty:
                df_data['category'] = 'NNC'
                results.append(df_data)
                continue

            ref_ssc_set, ref_site_set = IsoformClassifier._prepare_reference_sets(df_ref)

            df_data = df_data.copy()
            df_data['row_sites'] = df_data['SSC'].str.split('-').map(lambda lst: list(map(int, lst)))
            df_data['category'] = df_data.apply(
                lambda row: IsoformClassifier._get_isoform_category_for_row(
                    row['SSC'], row['row_sites'], ref_ssc_set, ref_site_set
                ),
                axis=1
            )
            df_data.drop(columns='row_sites', inplace=True)
            results.append(df_data)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def get_isoform_category(self, df_combined):
        df_groups = [g for _, g in df_combined.groupby(['Chr','Strand'], observed=True)]
        with mp.Pool(self.num_processes) as pool:
            results = pool.map(IsoformClassifier._get_isoform_category_for_group, df_groups)
        return pd.concat(results, ignore_index=True).reset_index(drop=True)

    def add_category(self, df1, df_ref):
        # Reverted to v0.3 form (only Chr/Strand/SSC). v1.0's earlier expansion
        # to ['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd'] broke when TrStart was
        # still a numpy array from TrStart_reads, since _parse_introns expects
        # scalar ints. Re-running the v0.3 form avoids the array→scalar cast.
        df1 = df1.drop(columns=['category'], errors='ignore')
        df = df1[['Chr', 'Strand', 'SSC']].copy()
        df_ref = df_ref[['Chr', 'Strand', 'SSC']].copy()
        df_ref = df_ref.drop_duplicates()
        df['source'] = 'data'
        df_ref['source'] = 'ref'

        df_combined = pd.concat([df, df_ref], ignore_index=True)
        gene_clustering = GeneClustering(num_processes=self.num_processes)
        df_combined = gene_clustering.cluster(df_combined)

        df_category = self.get_isoform_category(df_combined)
        return df1.merge(
            df_category[['Chr', 'Strand', 'SSC', 'category']].drop_duplicates(),
            on=['Chr', 'Strand', 'SSC'],
            how='left'
        )