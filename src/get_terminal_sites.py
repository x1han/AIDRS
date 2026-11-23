import pandas as pd
import numpy as np
import multiprocessing as mp
from sklearn.cluster import DBSCAN
from functools import partial
from scipy.spatial.distance import cdist
from collections import Counter

# A2' gate: the minimum number of pt>0 reads a DBSCAN cluster needs before it
# is trusted (and before the all-noise path may use the pt>0 subset). Literal
# constant -- never an expression of the group size, never 1.
A2_PRIME_MIN_PT_POS = 3

class TerminalSitesProcessor:
    def __init__(self, cluster_group_size=1500, eps=15, min_neighbors=20, num_processes=None, extrem_terminal=False):
        self.cluster_group_size = cluster_group_size
        self.eps = eps
        self.min_neighbors = min_neighbors
        from .aidrs_runtime.resource_guard import ResourceGuard
        self.num_processes = ResourceGuard.get_effective_cpu_threads(num_processes)
        self.extrem_terminal = extrem_terminal

    def _sample_values_and_pt(self, values, pt):
        """Sample the read coordinates together with their per-read poly(A) tail
        lengths, so the DBSCAN clusters stay aligned with the pt column.

        ``values.sample(n=..., random_state=42)`` shuffles the order but keeps
        the original index, so the pt rows are re-indexed with the sampled
        index instead of being sampled independently.
        """
        if len(values) >= self.cluster_group_size:
            sampled = values.sample(n=self.cluster_group_size, random_state=42)
        else:
            sampled = values

        sampled_pt = None if pt is None else pt.loc[sampled.index]

        X = np.array(sampled).reshape(-1, 1)
        min_neighbors = max(int(len(X) * 0.1), self.min_neighbors)
        labels = DBSCAN(eps=self.eps, min_samples=min_neighbors).fit_predict(X)

        return sampled, sampled_pt, labels

    def _representative(self, subset, mode):
        """Representative coordinate of one subset of read coordinates.

        ``self.extrem_terminal`` (the attribute -- the ``extrem_terminal``
        argument of get_representative_sites_DBSCAN is ignored, see the quirk
        note there) switches the rule from "modal value" to "extreme value".
        The tie rule of the modal branch is the production one: the smallest
        (mode='min') or largest (mode='max') of the tied modal values.
        """
        if self.extrem_terminal:
            return subset['value'].min() if mode == 'min' else subset['value'].max()
        mode_vals = subset['value'].mode()
        return mode_vals.mode().min() if mode == 'min' else mode_vals.mode().max()

    def get_representative_sites_DBSCAN(self, values, mode='min', extrem_terminal=False, *, pt=None):
        """Representative terminal site(s) of one group of reads.

        ``pt`` (keyword-only, optional) is a pd.Series of per-read poly(A) tail
        lengths whose index equals the index of ``values``. When given, the
        3'-side representative is re-estimated from the pt>0 reads:

        * A2' (cluster gate): all reads are clustered as usual; a cluster is
          kept when it holds at least A2_PRIME_MIN_PT_POS pt>0 reads (if no
          cluster qualifies, all clusters are kept). For a kept cluster the
          representative is recomputed from that cluster's pt>0 reads only;
          a cluster with fewer pt>0 reads keeps its original representative.
        * A2'-b (all-noise path): when DBSCAN finds no valid cluster the
          production answer is the mode over all reads; with at least
          A2_PRIME_MIN_PT_POS pt>0 reads the mode of those reads is used
          instead.

        With pt=None, or with an all-zero pt, the result is exactly the
        pre-A2' production result.

        QUIRK (preserved on purpose, do not "fix"): the ``extrem_terminal``
        argument is ignored -- the branch is taken on ``self.extrem_terminal``.
        """
        sampled, sampled_pt, labels = self._sample_values_and_pt(values, pt)

        df = pd.DataFrame({'value': sampled, 'cluster': labels})
        # The pt column MUST be attached before any branch: assigning it after
        # the "no valid cluster" test would make the A2'-b path dead code.
        if sampled_pt is not None:
            df['pt'] = sampled_pt.values
        has_pt = 'pt' in df.columns
        valid_clusters = [c for c in df['cluster'].unique() if c != -1]

        if not valid_clusters:
            # Production degenerate path: the mode over ALL reads.
            subset = df
            if has_pt:
                pt_pos = df[df['pt'] > 0]
                if len(pt_pos) >= A2_PRIME_MIN_PT_POS:
                    subset = pt_pos           # A2'-b
            return [self._representative(subset, mode)]

        # Clustered path: one representative per surviving cluster.
        chosen = valid_clusters
        if has_pt:
            kept = [c for c in valid_clusters
                    if int((df[df['cluster'] == c]['pt'] > 0).sum()) >= A2_PRIME_MIN_PT_POS]
            chosen = kept if kept else valid_clusters
        results = []
        for c in chosen:
            sub = df[df['cluster'] == c]
            if has_pt:
                pt_pos = sub[sub['pt'] > 0]
                if len(pt_pos) >= A2_PRIME_MIN_PT_POS:
                    results.append(self._representative(pt_pos, mode))
                    continue
            results.append(self._representative(sub, mode))
        return sorted(set(results))

    def get_terminal_sites_forChr(self, df):
        results = []
        has_pt = 'read_polyA_len' in df.columns

        for _, row in df.iterrows():
            starts = pd.Series(row['TrStart_reads'])
            ends = pd.Series(row['TrEnd_reads'])
            strand = row['Strand']

            # Per-read poly(A) tail lengths, aligned with the read arrays.
            pt = None
            if has_pt:
                pt_values = np.asarray(row['read_polyA_len'], dtype=np.int32)
                # Explicit error, not a bare assert: asserts are stripped under
                # `python -O`, and this is the last line of defence for the
                # per-read index parallelism at this point.
                if not (len(pt_values) == len(starts) == len(ends)):
                    raise RuntimeError(
                        "read_polyA_len / TrStart_reads / TrEnd_reads length mismatch: "
                        f"{len(pt_values)} vs {len(starts)} vs {len(ends)} "
                        f"(Chr={row['Chr']}, Strand={row['Strand']}, SSC={row['SSC']})"
                    )
                pt = pd.Series(pt_values)

            if strand == '+':
                # 3' end of a '+' strand read is its HIGH end (TrEnd_reads).
                pred_starts = self.get_representative_sites_DBSCAN(starts, mode='min', extrem_terminal=self.extrem_terminal)
                pred_ends = self.get_representative_sites_DBSCAN(ends, mode='max', extrem_terminal=False, pt=pt)
            else:
                # 3' end of a '-' strand read is its LOW end (TrStart_reads).
                pred_starts = self.get_representative_sites_DBSCAN(starts, mode='min', extrem_terminal=False, pt=pt)
                pred_ends = self.get_representative_sites_DBSCAN(ends, mode='max', extrem_terminal=self.extrem_terminal)

            raw_pairs = set(zip(row['TrStart_reads'], row['TrEnd_reads']))

            if len(pred_starts) == 1 and len(pred_ends) == 1:
                s, e = pred_starts[0], pred_ends[0]
                results.append({
                    'Chr': row['Chr'],
                    'Strand': row['Strand'],
                    'SSC': row['SSC'],
                    'Group': row['Group'],
                    'TrStart': s,
                    'TrEnd': e,
                    'frequency': row['frequency']
                })
                continue

            valid_pairs = [(s, e) for s in pred_starts for e in pred_ends if (s, e) in raw_pairs]

            if len(valid_pairs) == 1:
                s, e = valid_pairs[0]
                results.append({
                    'Chr': row['Chr'],
                    'Strand': row['Strand'],
                    'SSC': row['SSC'],
                    'Group': row['Group'],
                    'TrStart': s,
                    'TrEnd': e,
                    'frequency': row['frequency']
                })
            elif len(valid_pairs) > 1:
                raw_points = np.array(list(raw_pairs))
                centers = np.array(valid_pairs)

                dist_matrix = cdist(raw_points, centers, metric='euclidean')
                assigned_centers = dist_matrix.argmin(axis=1)

                center_counter = Counter(assigned_centers)
                total = sum(center_counter.values())

                for idx, (s, e) in enumerate(valid_pairs):
                    freq = round(row['frequency'] * (center_counter[idx] / total)) if idx in center_counter else 0
                    results.append({
                        'Chr': row['Chr'],
                        'Strand': row['Strand'],
                        'SSC': row['SSC'],
                        'Group': row['Group'],
                        'TrStart': s,
                        'TrEnd': e,
                        'frequency': freq
                    })
            else:
                s = min(pred_starts)
                e = max(pred_ends)
                results.append({
                    'Chr': row['Chr'],
                    'Strand': row['Strand'],
                    'SSC': row['SSC'],
                    'Group': row['Group'],
                    'TrStart': s,
                    'TrEnd': e,
                    'frequency': row['frequency']
                })

        return pd.DataFrame(results)

    def get_terminal_sites(self, df):
        dfChr_list = [dfChr for _, dfChr in df.groupby(['Chr','Strand'], observed=True)]
        partial_func = partial(self.get_terminal_sites_forChr)

        with mp.Pool(self.num_processes) as pool:
            results = pool.map(partial_func, dfChr_list)

        df_result = pd.concat(results, ignore_index=True).reset_index(drop=True)
        
        return df_result
