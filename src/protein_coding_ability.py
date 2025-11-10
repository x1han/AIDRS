#!/usr/bin/env python

from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
from pyfaidx import Fasta
import os
import subprocess
from functools import partial
import multiprocessing as mp

class TranslationAI_ORF:
    def __init__(self, genome, tmp_path='temp', translationai_score_threshold=0.9, num_processes=8):
        self.genome = genome
        self.tmp_path = tmp_path
        self.translationai_score_threshold = translationai_score_threshold
        self.num_processes = num_processes

    @staticmethod
    def fetch_exon(row):
        chrom = str(row['Chr'])
        start = int(row['TrStart'])      # 1-based
        end   = int(row['TrEnd'])
        strand = str(row['Strand'])
        
        # Parse SSC column to get exon ranges
        ssc = str(row['SSC'])
        positions = [start] + list(map(int, ssc.split('-'))) + [end]
        
        # Convert position list to exon ranges (start, end)
        exon_ranges = []
        for i in range(0, len(positions), 2):
            if i + 1 < len(positions):
                exon_start = positions[i]
                exon_end = positions[i + 1]
                exon_ranges.append((exon_start, exon_end))
        
        # If on negative strand, reverse the entire exon list order and reverse coordinates of each exon
        if strand == '-':
            exon_ranges.reverse()
            # Reverse coordinate order within each exon
            exon_ranges = [(end, start) for start, end in exon_ranges]

        return exon_ranges

    
    @staticmethod
    def fetch_seq(row, genome):
        chrom = str(row['Chr'])
        start = int(row['TrStart'])      # 1-based
        end   = int(row['TrEnd'])
        
        # Parse SSC column to get exon ranges
        ssc = str(row['SSC'])
        positions = [start] + list(map(int, ssc.split('-'))) + [end]
        
        # Convert position list to exon ranges (start, end)
        exon_ranges = []
        for i in range(0, len(positions), 2):
            if i + 1 < len(positions):
                exon_start = positions[i]
                exon_end = positions[i + 1]
                exon_ranges.append((exon_start, exon_end))
        
        # Extract exon sequences
        exon_sequences = []
        for exon_start, exon_end in exon_ranges:
            # Ensure exon range is within transcript range
            if exon_start >= start and exon_end <= end:
                # pyfaidx slice is 0-based, left-closed, right-open
                exon_seq_str = genome[chrom][exon_start-1:exon_end].seq
                exon_sequences.append(exon_seq_str)
        
        # Concatenate all exon sequences
        seq_str = ''.join(exon_sequences)
        seq = Seq(seq_str)                         # turn into Biopython Seq

        if row['Strand'] == '-':
            seq = seq.reverse_complement()

        return str(seq)

    @staticmethod
    def run_translationai(df, genome, tmp_path):
        if df.empty:
            return
        
        df_fasta = df.copy()

        Chrom = df_fasta['Chr'].unique()[0]
        Strand = df_fasta['Strand'].unique()[0]

        genome = Fasta(genome)
        df_fasta['seq'] = df_fasta.apply(lambda row: TranslationAI_ORF.fetch_seq(row, genome), axis=1)

        fasta_out_dir = os.path.join(tmp_path, "TranslationAI_temp/")
        os.makedirs(fasta_out_dir, exist_ok=True)
        fasta_out_path = fasta_out_dir + Chrom + "_" + Strand + ".fasta"

        with open(fasta_out_path, "w") as fh:
            pass

        with open(fasta_out_path, "a") as fh:
            keys = (
                df_fasta.reset_index(drop=True)               # Row index becomes 0,1,2,...
                .pipe(lambda d:                       # Concatenate key
                    d['TrStart'].astype(str) + '~' +
                    d['SSC'].astype(str) + '~' +
                    d['TrEnd'].astype(str)
                )
            )
            for idx, key in keys.items():
                output_entries = []
                seq = df_fasta.iloc[idx]['seq']
                
                # Generate FASTA header
                try:
                    seq_name = (
                        f">{df_fasta.iloc[idx]['Chr']}:"
                        f"{int(df_fasta.iloc[idx]['TrStart'])}-"
                        f"{int(df_fasta.iloc[idx]['TrEnd'])}"
                        f"({df_fasta.iloc[idx]['Strand']})"
                        f"({key})"
                        f"({int(0)}, "
                        f"{int(0)},)"
                    )
                    output_entries.append(f"{seq_name}\n{seq}\n")
                except (KeyError, ValueError) as e:
                    # Handle missing fields or type errors
                    print(f"Skipping entry at index {idx}: {str(e)}")
                    continue

                # Batch write to file
                if output_entries:
                    fh.writelines(output_entries)
        cmd = ["translationai",
        "-I", fasta_out_path, 
        "-t", "0.5,0.5"]
        subprocess.run(cmd, check=True)

    def orf_predict_by_translationai(self, df):
        df_groups = [g for _, g in df.groupby(['Chr','Strand'], observed=True)]
        with mp.Pool(self.num_processes) as pool:
            pool.map(partial(TranslationAI_ORF.run_translationai, genome=self.genome, tmp_path=self.tmp_path),
            df_groups
        )
    
    @staticmethod
    def check_nmd(df, translationai_score_threshold=0.9):
        """
        Determine NMD (Nonsense-Mediated Decay)
        Judge whether NMD is triggered based on EJC-dependent and EJC-independent mechanisms
        """
        def determine_nmd_status(row):
            # If no ORF is predicted, return 'no_orf'
            if (row['TIS_related_location'] == 'no' or 
                row['TTS_related_location'] == 'no'):
                return 'no_orf'
            
            # If either TIS_score or TTS_score is less than translationai_score_threshold, return 'no_orf'
            try:
                tis_score = float(row['TIS_score']) if row['TIS_score'] != 'no' else 0.0
                tts_score = float(row['TTS_score']) if row['TTS_score'] != 'no' else 0.0
                if tis_score < translationai_score_threshold or tts_score < translationai_score_threshold:
                    return 'no_orf'
            except (ValueError, TypeError):
                return 'no_orf'
            
            try:
                tis_pos = int(row['TIS_related_location'])
                tts_pos = int(row['TTS_related_location'])

                
                # Get exon ranges
                exon_ranges = TranslationAI_ORF.fetch_exon(row)
                
                # EJC-dependent NMD determination
                ejc_dependent_nmd = False
                if len(exon_ranges) >= 2:
                    # Calculate cumulative exon length to find each exon-exon junction position
                    cumulative_length = 0
                    junction_positions = []
                    
                    for i, (start, end) in enumerate(exon_ranges[:-1]):  # Exclude last exon
                        cumulative_length += abs(end - start + 1)
                        junction_positions.append(cumulative_length)
                    
                    # Check if stop codon is located upstream of any exon-exon junction by ≥55nt
                    if tts_pos <= junction_positions[-1] - 55:
                        ejc_dependent_nmd = True
                
                # EJC-independent NMD determination - check 3' UTR length
                ejc_independent_nmd = False
                
                # Calculate total CDS length (from TIS to TTS)
                total_cds_length = tts_pos - tis_pos + 1
                
                # Calculate total transcript length
                total_transcript_length = sum(abs(end - start + 1) for start, end in exon_ranges)
                
                # Calculate 3' UTR length (total transcript length - TTS position)
                utr3_length = total_transcript_length - tts_pos
                
                # If 3' UTR length > 1kb (1000nt), then EJC-independent NMD may be triggered
                if utr3_length > 1000:
                    ejc_independent_nmd = True
                
                # Comprehensive NMD status determination
                if ejc_dependent_nmd and ejc_independent_nmd:
                    return 'NMD'
                elif ejc_dependent_nmd:
                    return 'NMD'
                elif ejc_independent_nmd:
                    return 'NMD'
                else:
                    return 'Normal'
                    
            except (ValueError, TypeError):
                return 'unknown'
        
        # Apply NMD determination function
        df['Predict_NMD'] = df.apply(determine_nmd_status, axis=1)
        
        return df
    
    def aggr_translationai_result(self, df, translationai_out_path):
        all_lines = []
        for name in os.listdir(translationai_out_path):
            if name.endswith("_predORFs_0.5_0.5.txt"):
                file_path = os.path.join(translationai_out_path, name)
                
                # Check if file exists and is not empty
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    continue
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        # Filter empty lines and lines with only whitespace
                        non_empty_lines = [line for line in lines if line.strip()]
                        if non_empty_lines:
                            all_lines.extend(non_empty_lines)
                        else:
                            print(f"Warning: File contains no valid data: {file_path}")
                except (IOError, OSError) as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue
        
        # Parse translationai output results and create DataFrame
        results = []
        
        # Parse data from all_lines
        for line in all_lines:
            line = line.strip()
            if line and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    # Parse header part
                    header = parts[0]
                    # Parse numerical part
                    values = parts[1].split(',')
                    
                    if len(values) >= 4:
                        # Extract chr
                        chr_part = header.split(':')[0].replace('>', '')
                        
                        # Extract strand (first parenthesis)
                        strand_start = header.find('(')
                        strand_end = header.find(')', strand_start)
                        strand = header[strand_start+1:strand_end] if strand_start != -1 and strand_end != -1 else ''
                        
                        # Extract key (second parenthesis)
                        key_start = header.find('(', strand_end+1)
                        key_end = header.find(')', key_start)
                        key = header[key_start+1:key_end] if key_start != -1 and key_end != -1 else ''
                        
                        # Split key into 3 values (format: start~ssc~end)
                        key_parts = key.split('~') if key else ['', '', '']
                        tr_start = key_parts[0] if len(key_parts) > 0 else ''
                        ssc = key_parts[1] if len(key_parts) > 1 else ''
                        tr_end = key_parts[2] if len(key_parts) > 2 else ''
                        
                        # Extract 4 numbers
                        try:
                            tis_location = int(values[0])
                            tts_location = int(values[1])
                            tis_score = float(values[2])
                            tts_score = float(values[3])
                            
                            results.append({
                                'Chr': chr_part,
                                'Strand': strand,
                                'TrStart': tr_start,
                                'TrEnd': tr_end,
                                'SSC': ssc,
                                'TIS_related_location': tis_location,
                                'TTS_related_location': tts_location,
                                'TIS_score': tis_score,
                                'TTS_score': tts_score
                            })
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing line: {line}, error: {e}")
                            continue
        
        dtypes_df = {
            "Chr": "category",
            "Strand": "category",
            "TrStart": "int32",
            "TrEnd": "int32",
            "SSC": "string",       # pandas ≥1.5 recommends using string
        }
        # Create DataFrame and return
        if results:
            meriged_translationai_res = pd.DataFrame(results)
        
            df_groups = [g for _, g in df.groupby(['Chr','Strand'], observed=True)]

            merged_df_list = []
            for df_group in df_groups:
                Chrom = df_group['Chr'].unique()[0]
                Strand = df_group['Strand'].unique()[0]

                tss_col = 'TrStart' if Strand == '+' else 'TrEnd'

                translationai_subset = meriged_translationai_res[
                    (meriged_translationai_res['Chr'] == Chrom) & 
                    (meriged_translationai_res['Strand'] == Strand)
                ]

                translationai_subset = translationai_subset.astype(dtypes_df)

                df_merged = df_group.merge(
                    translationai_subset, 
                    on=["Chr", "Strand", "TrStart", "SSC", "TrEnd"],
                    how='outer'
                )
                merged_df_list.append(df_merged)

            if merged_df_list:
                df = pd.concat(merged_df_list, ignore_index=True)
        else:
            meriged_translationai_res = pd.DataFrame()
            df_groups = [g for _, g in df.groupby(['Chr','Strand'], observed=True)]

            merged_df_list = []
            for df_group in df_groups:
                merged_df_list.append(df_group)

            if merged_df_list:
                df = pd.concat(merged_df_list, ignore_index=True)

        # Check if the required columns exist in the DataFrame before trying to fill them
        required_columns = ['TIS_related_location', 'TTS_related_location', 'TIS_score', 'TTS_score']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 'no'
        
        # Now safely fill NaN values for existing columns
        existing_required_columns = [col for col in required_columns if col in df.columns]
        if existing_required_columns:
            df[existing_required_columns] = df[existing_required_columns].fillna('no')
        
        def to_int_or_no(x):
            if pd.isna(x) or str(x).strip() == 'no':
                return 'no'
            try:
                return int(float(x))
            except (ValueError, TypeError):
                return 'no'

        cols = ['TIS_related_location', 'TTS_related_location']
        
        # Only process columns that exist in the DataFrame
        existing_cols = [col for col in cols if col in df.columns]
        for col in existing_cols:
            df[col] = df[col].apply(to_int_or_no)

        if existing_cols:
            df[existing_cols] = df[existing_cols].astype('object')

        df = TranslationAI_ORF.check_nmd(df, self.translationai_score_threshold)
        
        # Delete generated temp folder
        import shutil
        if os.path.exists(translationai_out_path):
            shutil.rmtree(translationai_out_path, ignore_errors=True)
        
        return df
