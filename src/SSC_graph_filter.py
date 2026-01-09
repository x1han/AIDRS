import pandas as pd
import networkx as nx
import numpy as np
from functools import partial
from multiprocessing import Pool
import multiprocessing as mp
import bisect
from collections import Counter
from .isoform_classify import IsoformClassifier
from .gene_grouping import GeneClustering
import logging
from typing import Tuple

logger = logging.getLogger("AIDRS")


class SSCGraphFilter:
    def __init__(self, 
                 # Parameters from NNC_NIC_filter
                 error_sites_diff_bp=10, 
                 error_sites_ratio=0.01,
                 error_sites_ratio_ref=0.02,
                 little_exon_bp=30, 
                 little_exon_mismatch_diff_bp=10,
                 Nonlittle_exon_mismatch_diff_bp=20,
                 little_exon_jump_ratio=0.05,
                 little_exon_jump_ratio_ref=0.1,
                 Nonlittle_exon_jump_ratio=0.05,
                 Nonlittle_exon_jump_ratio_ref=0.1,
                 
                 # Parameters from gtf_rescue_filtering
                 fake_exon_group_freq_ratio=0.1,
                 fake_exon_group_freq_ratio_ref=0.2,
                 fake_exon_bp=50,
                 
                 # Common parameters
                 num_processes=10):
        """
        Initialize the SSC Graph Filter with parameters from both modules.
        """
        # Parameters from NNC_NIC_filter
        self.error_sites_diff_bp = error_sites_diff_bp
        self.error_sites_ratio = error_sites_ratio
        self.error_sites_ratio_ref = error_sites_ratio_ref
        self.little_exon_bp = little_exon_bp
        self.little_exon_mismatch_diff_bp = little_exon_mismatch_diff_bp
        self.Nonlittle_exon_mismatch_diff_bp = Nonlittle_exon_mismatch_diff_bp
        self.little_exon_jump_ratio = little_exon_jump_ratio
        self.little_exon_jump_ratio_ref = little_exon_jump_ratio_ref
        self.Nonlittle_exon_jump_ratio = Nonlittle_exon_jump_ratio
        self.Nonlittle_exon_jump_ratio_ref = Nonlittle_exon_jump_ratio_ref
        
        # Parameters from gtf_rescue_filtering
        self.fake_exon_group_freq_ratio = fake_exon_group_freq_ratio
        self.fake_exon_group_freq_ratio_ref = fake_exon_group_freq_ratio_ref
        self.fake_exon_bp = fake_exon_bp
        
        # Common parameters
        self.num_processes = num_processes
    
    @staticmethod
    def rescue_fsm(df_raw, df, ref_anno):
        """
        Rescue FSM transcripts from reference annotation that have low read support.
        """
        ref_anno["key"] = ref_anno["Chr"].astype(str) + ref_anno["Strand"].astype(str) + ref_anno["SSC"].astype(str)
        df_raw["key"] = df_raw["Chr"].astype(str) + df_raw["Strand"].astype(str) + df_raw["SSC"].astype(str)
        df["key"] = df["Chr"].astype(str) + df["Strand"].astype(str) + df["SSC"].astype(str)

        df_rescue = df_raw[
            df_raw["key"].isin(ref_anno["key"]) & ~df_raw["key"].isin(df["key"])
        ].copy()
        df_rescue["category"] = "FSM"
        # Comment out debug print statement

        if 'category' not in df.columns:
            df["category"] = "Unkown"
        
        # Only perform concat operation when df_rescue is not empty to avoid FutureWarning
        if not df_rescue.empty:
            df = pd.concat([df,
                        df_rescue],ignore_index = True).reset_index(drop = True)  
        return df
    
    def filter_groups(self, df, ref_anno=None):
        """
        Filter groups based on reference annotation (if provided) and frequency.
        """
        gene_clustering = GeneClustering(num_processes=self.num_processes)
        df = gene_clustering.cluster(df)

        df['frequency'] = df['TrStart_reads'].apply(len)

        # If reference annotation is provided, use it for filtering
        if ref_anno is not None:
            df['key'] = df['Chr'].astype(str) + df['Strand'].astype(str) + df['SSC'].astype(str)
            ref_keys = set(ref_anno['key'])
            df['ref'] = df['key'].isin(ref_keys).astype(int)

            freq_threshold = df['frequency'].quantile(0.25)  # high confidence NovelGene

            group_stats = df.groupby(['Chr', 'Strand', 'Group'], observed=True).agg(
                any_ref=('ref', 'any'),
                total_freq=('frequency', 'sum')
            )

            groups_to_keep = group_stats[
                (group_stats['any_ref']) | (group_stats['total_freq'] >= freq_threshold)
            ].index

            df = df.set_index(['Chr', 'Strand', 'Group'])
            df = df.loc[groups_to_keep].reset_index()
            return df.drop(columns=['key', 'ref'])
        else:
            # Without reference annotation, just filter by frequency
            freq_threshold = df['frequency'].quantile(0.25)
            group_stats = df.groupby(['Chr', 'Strand', 'Group'], observed=True).agg(
                total_freq=('frequency', 'sum')
            )
            
            groups_to_keep = group_stats[group_stats['total_freq'] >= freq_threshold].index
            
            df = df.set_index(['Chr', 'Strand', 'Group'])
            df = df.loc[groups_to_keep].reset_index()
            return df
    

    def get_edges_weight(self, df_group):
        """
        Calculate edge weights for SSC2.
        """
        dict_net = {}

        group_freq = df_group['frequency'].sum()
        for index, row in df_group.iterrows():
            SSC2 = row['SSC2']
            freq_weight = row['frequency'] / group_freq

            for i in range(0, len(SSC2) - 1):
                node = (SSC2[i], SSC2[i + 1])
                if node not in dict_net:
                    dict_net[node] = [SSC2[i], SSC2[i + 1], freq_weight]
                else:
                    dict_net[node][2] += freq_weight
        return dict_net
    
    @staticmethod
    def get_df_SSC_StartEnd(df):
        """
        Generate SSC_extend with start and end extensions.
        """
        df = df.copy()
        df['SSC2'] = df['SSC'].apply(lambda x: list(map(int, x.split('-'))))

        extreme_values = (
            df.groupby(['Chr', 'Strand', 'Group'], observed=True)['SSC2']
            .agg(lambda x: (min([i for sublist in x for i in sublist]), 
                            max([i for sublist in x for i in sublist])))
            .apply(pd.Series)
            .rename(columns={0: 'min_value', 1: 'max_value'})
            .reset_index())

        df = df.merge(extreme_values, on=['Chr', 'Strand', 'Group'])

        df['SSC_extend'] = df.apply(
            lambda row: [row['min_value'] - 100] + row['SSC2'] + [row['max_value'] + 100], axis=1
        )
        df = df.drop(columns=['min_value', 'max_value'])

        df['SSC_extend_str'] = df['SSC_extend'].apply(lambda x: '-'.join(map(str, x)))

        return df


    @staticmethod
    def get_edges_weight_SSCextend(df_group):
        """
        Calculate edge weights for SSC_extend.
        """
        dict_net = {}

        group_freq = df_group['frequency'].sum()
        for index, row in df_group.iterrows():
            SSC2 = row['SSC_extend']
            freq_weight = row['frequency'] / group_freq

            for i in range(0, len(SSC2) - 1):
                node = (SSC2[i], SSC2[i + 1])
                if node not in dict_net:
                    dict_net[node] = [SSC2[i], SSC2[i + 1], freq_weight]
                else:
                    dict_net[node][2] += freq_weight
        return dict_net


    @staticmethod 
    def remove_misalignment(
        subgraph: nx.DiGraph,
        df_subgraph: pd.DataFrame,
        # Phase 1: Error Site Removal
        error_sites_diff_bp: int,
        error_sites_ratio: float,
        error_sites_ratio_ref: float,
        # Phase 2/3: Exon Correction
        little_exon_bp: int,
        little_exon_jump_ratio: float,
        little_exon_jump_ratio_ref: float,
        little_exon_mismatch_diff_bp: int,
        Nonlittle_exon_mismatch_diff_bp: int,
        Nonlittle_exon_jump_ratio: float,
        Nonlittle_exon_jump_ratio_ref: float,
        # Phase 3: Fake Exon Specific Parameters
        fake_exon_bp: int,
        fake_exon_group_freq_ratio: float,
        fake_exon_group_freq_ratio_ref: float
    ) -> pd.DataFrame:
        """
        Perform multi-stage correction and filtering of paths in exon subgraphs to remove erroneous splice sites and noisy paths.

        Parameters:
            subgraph (nx.DiGraph): Network graph subgraph containing path information.
            df_subgraph (pd.DataFrame): DataFrame containing SSC (Splice Site Chain) paths and their frequencies.
            
            # Phase 1: Error Site Removal
            error_sites_diff_bp (int): Distance threshold for determining whether adjacent splice sites are "close".
            error_sites_ratio (float): Frequency ratio threshold below which low-frequency sites are considered noise and removed (no ref).
            error_sites_ratio_ref (float): Frequency ratio threshold below which low-frequency sites are considered noise and removed (with ref).
            
            # Phase 2/3: Exon Correction
            little_exon_bp (int): Maximum length defining small exons (Little Exon).
            little_exon_jump_ratio (float): Phase 2: Frequency ratio threshold for little exon path jumps competing with inclusion paths (no ref).
            little_exon_jump_ratio_ref (float): Phase 2: Frequency ratio threshold for little exon path jumps competing with inclusion paths (with ref).
            little_exon_mismatch_diff_bp (int): Phase 2: Total length difference allowed in little exon N6/N4 comparison.
            Nonlittle_exon_mismatch_diff_bp (int): Phase 3: Total length difference allowed in non-little exon N6/N4 comparison.
            Nonlittle_exon_jump_ratio (float): Phase 3: Frequency ratio threshold for non-little exon path jumps competing with inclusion paths (no ref).
            Nonlittle_exon_jump_ratio_ref (float): Phase 3: Frequency ratio threshold for non-little exon path jumps competing with inclusion paths (with ref).
            
            # Phase 3: Fake Exon Specific
            fake_exon_bp (int): Maximum exon length in fake exon (Fake Exon) correction.
            fake_exon_group_freq_ratio (float): Frequency ratio threshold for Fake Exon correction (no ref).
            fake_exon_group_freq_ratio_ref (float): Frequency ratio threshold for Fake Exon correction (with ref).

        Returns:
            pd.DataFrame: Corrected and filtered SSC path DataFrame (df_subgraph_corr).
        """
        
        import pandas as pd
        import networkx as nx
        from collections import defaultdict
        import re

        df_subgraph_process = df_subgraph.copy()
        
        # Helper function: Generate reference and query site sets, and unify path category labels
        def generate_ref_query_dicts(df_subgraph_process: pd.DataFrame) -> Tuple[pd.DataFrame, set, set, bool]:
            """Classify paths as 'ref' (FSM/ISM) or 'query' (NNC/NIC) based on the 'category' column, and generate site sets."""
            ref_sites = defaultdict(int)
            query_sites = defaultdict(int)
            df_subgraph_process = df_subgraph_process.copy()

            has_ref_category = df_subgraph_process['category'].str.contains('FSM|ISM').any()
            df_subgraph_process['Type'] = 'query'

            if not has_ref_category:
                df_subgraph_process['category'] = 'query'
                return df_subgraph_process, set(), set(), False

            for index, row in df_subgraph_process.iterrows():
                category = row['category']
                sites = set(row['SSC2']) 

                is_ref_row = 'FSM' in category or 'ISM' in category
                
                if is_ref_row:
                    for site in sites:
                        ref_sites[site] += 1
                    df_subgraph_process.loc[index, 'Type'] = 'ref'
                elif 'NNC' in category or 'NIC' in category:
                    for site in sites:
                        query_sites[site] += 1
                
            ref_sites_set = set(ref_sites.keys())
            query_sites_set = set(query_sites.keys())
            
            df_subgraph_process = df_subgraph_process.drop(columns=['category'])
            df_subgraph_process = df_subgraph_process.rename(columns={'Type': 'category'})
            
            return df_subgraph_process, ref_sites_set, query_sites_set, True

        df_subgraph_process, ref_sites, query_sites, has_ref = generate_ref_query_dicts(df_subgraph_process)

        # ----------------------------------------------------
        ## Phase 1: Error Site Removal Based on Frequency
        # ----------------------------------------------------
        
        # 1. Calculate site frequencies and identify close potential error site pairs
        sites_dict = {}
        for _, row in df_subgraph_process.iterrows():
            SSC2 = row['SSC2']
            freq = row['frequency']
            for site in SSC2:
                sites_dict[site] = sites_dict.get(site, 0) + freq
                
        sites_dict_keys = sorted(list(sites_dict.keys()))
        sites_dict_keys_pairs = list(zip(sites_dict_keys, sites_dict_keys[1:]))
        potential_error_sites = [sites for sites in sites_dict_keys_pairs if abs(sites[1] - sites[0]) <= error_sites_diff_bp]

        # 2. Filter and mark low-frequency sites for removal (Ref protection logic)
        error_sites_to_remove = []

        for p_sites in potential_error_sites:
            site1, site2 = p_sites[0], p_sites[1]
            site1_freq, site2_freq = sites_dict[site1], sites_dict[site2]
            
            High_Freq_Site, Low_Freq_Site = (site1, site2) if site1_freq > site2_freq else (site2, site1)
            Low_to_High_Ratio = min(site1_freq, site2_freq) / max(site1_freq, site2_freq)
                
            threshold = error_sites_ratio_ref if has_ref else error_sites_ratio
            should_correct = False
            
            if has_ref:
                site1_is_ref, site2_is_ref = site1 in ref_sites, site2 in ref_sites
                
                if site1_is_ref and site2_is_ref:
                    # Same type sites (Ref vs Ref)
                    should_correct = False
                elif not site1_is_ref and not site2_is_ref:
                    # Same type sites (Query vs Query)
                    if Low_to_High_Ratio <= threshold: should_correct = True
                else:
                    # Different type sites (Ref vs Query): Relax threshold, only correct low-frequency Query
                    Query_Site = site2 if site1_is_ref else site1
                    if Query_Site == Low_Freq_Site and Low_to_High_Ratio <= threshold:
                        should_correct = True
            else:
                # No Ref mode
                if Low_to_High_Ratio <= threshold: should_correct = True
                    
            # 3. Mark sites to delete
            if should_correct:
                if has_ref:
                    if High_Freq_Site in ref_sites and Low_Freq_Site not in ref_sites:
                        error_sites_to_remove.append(Low_Freq_Site)
                    elif (site1_is_ref and site2_is_ref) or (not site1_is_ref and not site2_is_ref):
                        error_sites_to_remove.append(Low_Freq_Site)
                else:
                    error_sites_to_remove.append(Low_Freq_Site)

        final_error_sites = set(error_sites_to_remove)

        # 4. Filter out paths containing error sites from DataFrame
        if len(final_error_sites) > 0:
            error_sites_pattern = '|'.join([r'\b' + str(site) + r'\b' for site in final_error_sites])
            rows_to_keep = ~df_subgraph_process['SSC_extend_str'].str.contains(error_sites_pattern, regex=True)
            df_subgraph_corr = df_subgraph_process[rows_to_keep].copy()
        else:
            df_subgraph_corr = df_subgraph_process.copy()

        # ----------------------------------------------------
        ## Phase 2: Little Exon Correction Marking (N4/N6 Correction)
        # ----------------------------------------------------
        correct_paths = {}

        # 1. Path data preparation: Extract N4/N6 paths and 2-node jump frequencies
        ssc_data = []
        n4_groups = defaultdict(list)
        n6_groups = []
        jump_lookup = defaultdict(int)

        for _, row in df_subgraph_corr.iterrows():
            ssc_list = row['SSC_extend']
            freq, is_ref = row['frequency'], (row.get('category') == 'ref')
            ssc_data.append({'SSC': ssc_list, 'freq': freq, 'is_ref': is_ref})
            
            for k in range(len(ssc_list) - 1):
                jump_lookup[(ssc_list[k], ssc_list[k+1])] += freq

        for SSC_info in ssc_data:
            SSC = SSC_info['SSC']
            if len(SSC) >= 4:
                for i in range(0, len(SSC) - 2, 2): 
                    if SSC[i+1] - SSC[i] <= little_exon_bp:
                        # N4 path extraction
                        node4_index_S, node4_index_E = i - 1, i + 3
                        if node4_index_S >= 0 and node4_index_E < len(SSC): 
                            n4_path = SSC[node4_index_S:node4_index_E]
                            if len(n4_path) == 4:
                                n4_groups[(n4_path[0], n4_path[-1])].append({
                                    'path': n4_path, 'freq': SSC_info['freq'], 'is_ref': SSC_info['is_ref']
                                })
                        # N6 path extraction
                        node6_index_S, node6_index_E = i - 2, i + 4
                        if node6_index_S >= 0 and node6_index_E < len(SSC): 
                            n6_path = SSC[node6_index_S:node6_index_E]
                            if len(n6_path) == 6:
                                n6_groups.append({
                                    'path': tuple(n6_path), 'freq': SSC_info['freq'], 'is_ref': SSC_info['is_ref']
                                })

        # 2. N4 Correction Logic (Little Exon Jump Correction)
        little_exon_corrections = {}
        
        for (N1, N4), contain_paths in n4_groups.items():
            total_contain_freq = sum(p['freq'] for p in contain_paths)
            jump_freq = jump_lookup.get((N1, N4), 0)
            
            if total_contain_freq == 0 or jump_freq == 0: continue
                
            High_Freq, Low_Freq = max(total_contain_freq, jump_freq), min(total_contain_freq, jump_freq)
            High_Freq_Type = 'contain' if total_contain_freq >= jump_freq else 'jump'
            Low_Freq_Type = 'jump' if total_contain_freq >= jump_freq else 'contain'
                
            if High_Freq == 0: continue
            ratio = Low_Freq / High_Freq
            threshold = little_exon_jump_ratio_ref if has_ref else little_exon_jump_ratio
            should_correct, error_type = False, None
            is_little_exon_ref = any(p['is_ref'] for p in contain_paths)
            
            if not has_ref:
                if ratio < threshold: should_correct, error_type = True, Low_Freq_Type
            else:
                if is_little_exon_ref and High_Freq_Type == 'contain':
                    if Low_Freq_Type == 'jump' and ratio < threshold:
                        should_correct, error_type = True, 'jump'
                elif not is_little_exon_ref:
                    if ratio < threshold: should_correct, error_type = True, Low_Freq_Type
            
            if should_correct:
                best_contain_path = max(contain_paths, key=lambda p: p['freq'])['path']
                
                if error_type == 'jump':
                    little_exon_corrections[f"{N1}-{N4}"] = '-'.join(map(str, best_contain_path))
                elif error_type == 'contain':
                    worst_contain_path = min(contain_paths, key=lambda p: p['freq'])['path']
                    little_exon_corrections['-'.join(map(str, worst_contain_path))] = f"{N1}-{N4}"
        
        
        # 3. N6 Correction Logic (Little Exon Misalignment Correction)
        if len(n6_groups) > 0:
            for N6_info in n6_groups:
                N6_path = N6_info['path']
                freq_n6, is_ref_n6 = N6_info['freq'], N6_info.get('is_ref', False)
                N6_path_length = sum([N6_path[j+1] - N6_path[j] for j in range(0, len(N6_path), 2)])
                
                for SSC_info in ssc_data:
                    row_SSCextend = SSC_info['SSC']
                    freq_n4, is_ref_n4 = SSC_info['freq'], SSC_info.get('is_ref', False)
                    
                    if N6_path[0] in row_SSCextend and N6_path[-1] in row_SSCextend:
                        N6_S_idx = row_SSCextend.index(N6_path[0])
                        N6_E_idx = row_SSCextend.index(N6_path[-1])
                        row_path = row_SSCextend[N6_S_idx : N6_E_idx + 1]
                        
                        if len(row_path) == 4 and len(set(row_path) - set(N6_path)) > 0:
                            row_path_length = sum([row_path[j+1] - row_path[j] for j in range(0, len(row_path), 2)])
                            
                            if abs(N6_path_length - row_path_length) <= little_exon_mismatch_diff_bp:
                                if freq_n6 == 0 or freq_n4 == 0: continue

                                High_Freq, Low_Freq = max(freq_n4, freq_n6), min(freq_n4, freq_n6)
                                High_Type, Low_Type = ('N6', 'N4') if freq_n6 >= freq_n4 else ('N4', 'N6')
                                
                                ratio = Low_Freq / High_Freq
                                threshold = little_exon_jump_ratio_ref if has_ref else little_exon_jump_ratio
                                should_correct, correction_direction = False, None

                                if not has_ref:
                                    if ratio < threshold: should_correct, correction_direction = True, ('N4_to_N6' if Low_Type == 'N4' else 'N6_to_N4')
                                else:
                                    # Ref priority rule
                                    if is_ref_n6 and High_Type == 'N6' and Low_Type == 'N4' and ratio < threshold:
                                        should_correct, correction_direction = True, 'N4_to_N6'
                                    elif is_ref_n4 and High_Type == 'N4' and Low_Type == 'N6' and ratio < threshold:
                                        should_correct, correction_direction = True, 'N6_to_N4'

                                if should_correct:
                                    if correction_direction == 'N4_to_N6':
                                        error_key = '-'.join([str(row_path[1]), str(row_path[2])])
                                        correct_val = '-'.join([str(s) for s in N6_path[1:-1]])
                                    elif correction_direction == 'N6_to_N4':
                                        error_key = '-'.join([str(s) for s in N6_path[1:-1]])
                                        correct_val = '-'.join([str(row_path[1]), str(row_path[2])])
                                    
                                    little_exon_corrections[error_key] = correct_val

        correct_paths.update(little_exon_corrections)

        # ----------------------------------------------------
        ## Phase 3: Non-Little Exon & Fake Exon Correction
        # ----------------------------------------------------

        non_little_exon_correct = {}
        fake_exon_correct = {}
        
        # 1. Path search and aggregation
        out_nodes = [node for node, out_degree in subgraph.out_degree() if out_degree > 1]
        in_nodes = [node2 for node2, in_degree in subgraph.in_degree() if in_degree > 1]
        forked_paths = []
        for out_node in out_nodes:
            for in_node in in_nodes:
                if nx.has_path(subgraph, out_node, in_node):
                    paths = nx.all_simple_paths(subgraph, source=out_node, target=in_node, cutoff=6)
                    forked_paths.extend(paths)

        SSCs = df_subgraph_corr['SSC_extend_str'].tolist()
        forked_paths_filtered = [path for path in forked_paths if any('-'.join(map(str, path)) in chain for chain in SSCs)]
        forked_paths_4_6 = [path for path in forked_paths_filtered if len(path) in [4, 6]]

        full_ssc_data = [{'path_str': row['SSC_extend_str'], 'freq': row['frequency'], 'is_ref': (row.get('category') == 'ref')} 
                        for _, row in df_subgraph_corr.iterrows()]
        
        dict_path46 = defaultdict(lambda: {'paths': [], 'has_ref_in_group': False}) 

        for path in forked_paths_4_6:
            path_str = '-'.join(map(str, path))
            start_end = (path[0], path[-1])
            current_freq, current_has_ref = 0, False 
            
            for ssc_info in full_ssc_data:
                if path_str in ssc_info['path_str']:
                    current_freq += ssc_info['freq']
                    if ssc_info['is_ref']: current_has_ref = True
            
            if current_freq > 0:
                record = {'path': path, 'freq': current_freq, 'is_ref': current_has_ref}
                dict_path46[start_end]['paths'].append(record)
                if current_has_ref: dict_path46[start_end]['has_ref_in_group'] = True

        dict_path46 = {k: v for k, v in dict_path46.items() if len(v['paths']) > 1}

        # 2. Core correction logic A: Non Little Exon correction (Nonlittle_exon_mismatch_diff_bp & Nonlittle_exon_jump_multiple)
        for (N_start, N_end), group_info in dict_path46.items():
            group_paths = group_info['paths']
            has_n4 = any(len(p['path']) == 4 for p in group_paths)
            has_n6 = any(len(p['path']) == 6 for p in group_paths)
            
            if not (has_n4 and has_n6) or all(p['is_ref'] for p in group_paths): continue

            n4_list = [p for p in group_paths if len(p['path']) == 4]
            n6_list = [p for p in group_paths if len(p['path']) == 6]

            for n4_info in n4_list:
                for n6_info in n6_list:
                    n4_path, n6_path = n4_info['path'], n6_info['path']
                    n4_len = sum([n4_path[j+1] - n4_path[j] for j in range(0, len(n4_path), 2)])
                    n6_len = sum([n6_path[j+1] - n6_path[j] for j in range(0, len(n6_path), 2)])
                    
                    if abs(n6_len - n4_len) > Nonlittle_exon_mismatch_diff_bp: continue
                    
                    freq_n4, is_ref_n4 = n4_info['freq'], n4_info['is_ref']
                    freq_n6, is_ref_n6 = n6_info['freq'], n6_info['is_ref']
                    if freq_n6 == 0 or freq_n4 == 0: continue

                    High_Freq, Low_Freq = max(freq_n4, freq_n6), min(freq_n4, freq_n6)
                    Low_Type = 'N4' if freq_n4 < freq_n6 else 'N6'
                    ratio = Low_Freq / High_Freq
                    threshold = Nonlittle_exon_jump_ratio_ref if has_ref else Nonlittle_exon_jump_ratio
                    should_correct = False
                    
                    if not has_ref:
                        if ratio < threshold: should_correct = True
                    else:
                        if (is_ref_n6 and Low_Type == 'N4' and ratio < threshold) or \
                        (is_ref_n4 and Low_Type == 'N6' and ratio < threshold) or \
                        (not is_ref_n4 and not is_ref_n6 and ratio < threshold):
                            should_correct = True

                    if should_correct:
                        if Low_Type == 'N4':
                            error_key = '-'.join(map(str, n4_path[1:-1]))
                            correct_val = '-'.join(map(str, n6_path[1:-1]))
                        else:
                            error_key = '-'.join(map(str, n6_path[1:-1]))
                            correct_val = '-'.join(map(str, n4_path[1:-1]))

                        non_little_exon_correct[error_key] = correct_val

        # 3. Core correction logic B: Fake Exon correction
        for (N_start, N_end), group_info in dict_path46.items():
            group_paths = group_info['paths']
            has_n4 = any(len(p['path']) == 4 for p in group_paths)
            has_n6 = any(len(p['path']) == 6 for p in group_paths)
            
            if not (has_n4 and has_n6): continue
            num_refs = sum(p['is_ref'] for p in group_paths)
            if num_refs == 0 or num_refs == len(group_paths): continue # Must be Query vs Ref

            n4_list = [p for p in group_paths if len(p['path']) == 4]
            n6_list = [p for p in group_paths if len(p['path']) == 6]

            for n4_info in n4_list:
                for n6_info in n6_list:
                    n4_path, n6_path = n4_info['path'], n6_info['path']
                    is_ref_n4, is_ref_n6 = n4_info['is_ref'], n6_info['is_ref']

                    if (is_ref_n4 and is_ref_n6) or (not is_ref_n4 and not is_ref_n6): continue

                    n4_len = sum([n4_path[j+1] - n4_path[j] for j in range(0, len(n4_path), 2)])
                    n6_len = sum([n6_path[j+1] - n6_path[j] for j in range(0, len(n6_path), 2)])
                    
                    if abs(n6_len - n4_len) > fake_exon_bp: continue
                    
                    freq_n4, freq_n6 = n4_info['freq'], n6_info['freq']
                    if freq_n6 == 0 or freq_n4 == 0: continue

                    High_Freq, Low_Freq = max(freq_n4, freq_n6), min(freq_n4, freq_n6)
                    Low_Type = 'N4' if freq_n4 < freq_n6 else 'N6'
                    ratio = Low_Freq / High_Freq
                    threshold = fake_exon_group_freq_ratio_ref if has_ref else fake_exon_group_freq_ratio
                    should_correct = False
                    
                    if (is_ref_n6 and Low_Type == 'N4' and ratio < threshold) or \
                    (is_ref_n4 and Low_Type == 'N6' and ratio < threshold):
                        should_correct = True
                    
                    if should_correct:
                        if Low_Type == 'N4':
                            error_key = '-'.join(map(str, n4_path[1:-1]))
                            correct_val = '-'.join(map(str, n6_path[1:-1]))
                        else:
                            error_key = '-'.join(map(str, n6_path[1:-1]))
                            correct_val = '-'.join(map(str, n4_path[1:-1]))

                        fake_exon_correct[error_key] = correct_val

        # 4. Junction signal filtering and merging correction rules
        all_temp_corrections = {}
        all_temp_corrections.update(non_little_exon_correct)
        all_temp_corrections.update(fake_exon_correct)

        # if 'junction' in df_subgraph_corr.columns:
        #     final_validated_corrections = {}
        #     for error_key, correct_val in all_temp_corrections.items():
        #         error_rows = df_subgraph_corr[
        #             df_subgraph_corr['SSC_extend_str'].str.contains(error_key, na=False)
        #         ]
                
        #         should_correct_based_on_junction = False
        #         for _, row in error_rows.iterrows():
        #             if pd.isna(row['junction']):
        #                 continue
        #             junctions = row.get('junction', '').split(',')
        #             if not all(x.upper() == 'GT-AG' for x in junctions):
        #                 should_correct_based_on_junction = True
        #                 break
                        
        #         if should_correct_based_on_junction:
        #             final_validated_corrections[error_key] = correct_val
                    
        #     correct_paths.update(final_validated_corrections)
        # else:
        #     correct_paths.update(all_temp_corrections)

        correct_paths.update(all_temp_corrections)

        # 5. Filter paths based on final correction rules
        columns_to_keep = ['Chr', 'Strand', 'Group', 'SSC', 'TrStart_reads', 'TrEnd_reads', 'SSC2', 'frequency', 'SSC_extend_str', 'SSC_extend', 'category']
        if 'junction' in df_subgraph_corr.columns: columns_to_keep.append('junction')
        df_subgraph_corr = df_subgraph_corr[[col for col in columns_to_keep if col in df_subgraph_corr.columns]].copy()

        if len(correct_paths) > 0:
            remove_SSC = set(correct_paths.keys())
            remove_patterns = '|'.join([re.escape(k) for k in remove_SSC])
            
            # Filter out rows whose SSC paths contain any error fragments
            df_subgraph_corr = df_subgraph_corr[
                ~df_subgraph_corr['SSC'].astype(str).str.contains(remove_patterns, regex=True, na=False)
            ].copy()

        return df_subgraph_corr
    
    @staticmethod
    def nnc_nic_graph_forChr(df, little_exon_bp=30, 
                            little_exon_mismatch_diff_bp=10, Nonlittle_exon_mismatch_diff_bp=20,
                            error_sites_diff_bp=10, error_sites_ratio=0.01, error_sites_ratio_ref=0.02,
                            Nonlittle_exon_jump_ratio=0.05, Nonlittle_exon_jump_ratio_ref=0.1,
                            fake_exon_bp=50, fake_exon_group_freq_ratio=0.1, fake_exon_group_freq_ratio_ref=0.2,
                            little_exon_jump_ratio=0.05, little_exon_jump_ratio_ref=0.1):
        """
        Process graph filtering for a chromosome.
        """
        df = df.copy()
        
        df = SSCGraphFilter.get_df_SSC_StartEnd(df)

        df_list = []
        i = 0
        for _, df_group in df.groupby(['Chr', 'Strand', 'Group'], observed=True):
            dict_net = SSCGraphFilter.get_edges_weight_SSCextend(df_group)

            edges_list = list(dict_net.values())
            edges_list = [tuple(l) for l in edges_list]
            G = nx.DiGraph()
            G.add_weighted_edges_from(edges_list)

            for component in nx.weakly_connected_components(G):
                subgraph = G.subgraph(component)

                subgraph_nodes = set(list(subgraph.nodes()))
                subgraph_index = []
                for index, row in df_group.iterrows():
                    SSC2 = set(row['SSC2'])
                    if len(subgraph_nodes & SSC2) > 0:
                        subgraph_index.append(index)

                df_subgraph = df_group.loc[subgraph_index]

                df_subgraph['Group'] = i
                i += 1
                df_subgraph = SSCGraphFilter.remove_misalignment(
                    subgraph, df_subgraph,
                    # Phase 1: Error Site Removal
                    error_sites_diff_bp, error_sites_ratio, error_sites_ratio_ref,
                    # Phase 2/3: Exon Correction
                    little_exon_bp, little_exon_jump_ratio, little_exon_jump_ratio_ref, little_exon_mismatch_diff_bp, 
                    Nonlittle_exon_mismatch_diff_bp, Nonlittle_exon_jump_ratio, Nonlittle_exon_jump_ratio_ref,
                    # Phase 3: Fake Exon Specific Parameters
                    fake_exon_bp, fake_exon_group_freq_ratio, fake_exon_group_freq_ratio_ref
                )

                df_list.append(df_subgraph)

        df = pd.concat(df_list, ignore_index=True)
        return df
    
    def nnc_nic_graph(self, df, num_processes=None):
        """
        Apply graph-based filtering to NNC/NIC transcripts.
        """
        dfChr_list = [dfChr for _, dfChr in df.groupby('Chr', observed=True)]

        # Use static method instead of instance method to solve multiprocessing serialization issues
        partial_func = partial(SSCGraphFilter.nnc_nic_graph_forChr,
                              little_exon_bp=self.little_exon_bp,
                              little_exon_mismatch_diff_bp=self.little_exon_mismatch_diff_bp,
                              Nonlittle_exon_mismatch_diff_bp=self.Nonlittle_exon_mismatch_diff_bp,
                              error_sites_diff_bp=self.error_sites_diff_bp,
                              error_sites_ratio=self.error_sites_ratio,
                              error_sites_ratio_ref=self.error_sites_ratio_ref,
                              Nonlittle_exon_jump_ratio=self.Nonlittle_exon_jump_ratio,
                              Nonlittle_exon_jump_ratio_ref=self.Nonlittle_exon_jump_ratio_ref,
                              fake_exon_bp=self.fake_exon_bp,
                              fake_exon_group_freq_ratio=self.fake_exon_group_freq_ratio,
                              fake_exon_group_freq_ratio_ref=self.fake_exon_group_freq_ratio_ref,
                              little_exon_jump_ratio=self.little_exon_jump_ratio,
                              little_exon_jump_ratio_ref=self.little_exon_jump_ratio_ref)

        # Use the provided num_processes or fall back to self.num_processes
        processes = num_processes if num_processes is not None else self.num_processes

        with Pool(processes) as pool:
            results = pool.map(partial_func, dfChr_list)

        df_result = pd.concat(results, ignore_index=True).reset_index(drop=True)

        # Drop only the columns that are not needed, preserving category if it exists
        columns_to_drop = ['SSC2']
        # Only drop junction if it exists
        if 'junction' in df_result.columns:
            columns_to_drop.append('junction')
        df_result = df_result.drop(columns=[col for col in columns_to_drop if col in df_result.columns])
        return df_result
    
    def find_nearest_exon(self, ref_sites, query_sites):
        """
        Find nearest exons for site mapping.
        """
        mapped_sites = []
        for q in query_sites:
            idx = bisect.bisect_left(ref_sites, q)

            if idx == 0:
                mapped_sites.append(ref_sites[0])
            elif idx == len(ref_sites):
                mapped_sites.append(ref_sites[-1])
            else:
                before = ref_sites[idx - 1]
                after = ref_sites[idx]
                mapped_sites.append(before if abs(q - before) <= abs(q - after) else after)

        return mapped_sites

    @staticmethod
    def is_within_littleExon_range(lst, lower, upper):
        """
        Check if sites are within little exon range.
        Static version of is_within_littleExon_range.
        """
        return all(lower <= x <= upper for x in lst)
    


    def filter_with_reference(self, df_raw, df, ref_anno, num_processes=None):
        """
        Filter with reference annotation - Case 1 from requirements.
        """
        # Use the provided num_processes or fall back to self.num_processes
        processes = num_processes if num_processes is not None else self.num_processes
        
        # 1.1 GTF rescue FSM
        logger.info("\tStage 1.9: Reference provided, GTF rescue begins...")
        df = self.rescue_fsm(df_raw, df, ref_anno)
        logger.info(f"\tGTF rescue completed. Retained {len(df)} SSC records.")
        
        # 1.2 Filter groups
        df = self.filter_groups(df, ref_anno)
        
        # 1.3 Classify isoforms
        if 'category' in df.columns:
            df = df.drop(columns=['category'])
        isoformclassifier = IsoformClassifier(num_processes=processes)
        df = isoformclassifier.add_category(df, ref_anno)
        
        # 1.4 Graph-based filtering for NIC and NNC (but not FSM and ISM)
        # Include FSM/ISM in graph construction but only filter NIC/NNC
        if not df.empty:
            # Apply graph-based filtering to all transcripts to build comprehensive graph
            df = self.nnc_nic_graph(df, num_processes=processes)
        
        # Ensure consistent column order
        if 'category' in df.columns:
            df = df.drop(columns=['category'])
        df = df[['Chr', 'Strand', 'Group', 'SSC', 'TrStart_reads', 'TrEnd_reads', 'frequency']].copy()
        return df
    
    def filter_without_reference(self, df, num_processes=None):
        """
        Filter without reference annotation - Case 2 from requirements.
        """
        # Use the provided num_processes or fall back to self.num_processes
        processes = num_processes if num_processes is not None else self.num_processes
        
        # 2.1 Filter groups
        df = self.filter_groups(df, ref_anno=None)
        
        # 2.2 Mark all reads as NNC (since no reference)
        df['category'] = 'NNC'
        
        # 2.3 Graph-based filtering for NIC and NNC
        df = self.nnc_nic_graph(df, num_processes=processes)
        
        # Remove category column as it's not needed anymore
        if 'category' in df.columns:
            df = df.drop(columns=['category'])
            
        return df
    
    def filter_ssc_graph(self, df_raw, df, ref_anno=None, num_processes=None):
        """
        Main function to filter SSC using graph-based approaches.
        Handles both cases: with and without reference annotation.
        """
        if ref_anno is not None:
            # Case 1: With reference annotation
            return self.filter_with_reference(df_raw, df, ref_anno, num_processes=num_processes)
        else:
            # Case 2: Without reference annotation
            return self.filter_without_reference(df, num_processes=num_processes)