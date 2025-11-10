import sys
import logging
from traceback import print_exc
from io import StringIO
import argparse
import os
import shutil
import psutil
import re
from .common import *
from .consensus import ConsensusFilter
from .gene_grouping import GeneClustering
from .isoform_classify import IsoformClassifier
from .remove_lowConfidence_junction import SpliceConsensusFilter
from .NNC_NIC_filter import NncNicGraphProcessor
from .ISM_filter import TruncationProcessor
from .get_terminal_sites import TerminalSitesProcessor
from .gtf_rescue_filtering import GTFRescueFiltering
from .SSC_assign import SSCAssign
from .isoform_quantification import IsoformQuantifier
from .generate_reports import IsoformAnnotator
from .tss_annotation import TSS_Puffin
from .polya_annotation import polyAnnotator
from .protein_coding_ability import *


def setup_logger(output_dir):
    logger = logging.getLogger("AIDRS")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    log_file = os.path.join(output_dir, "aidrs.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def isoform_assembling(bam, args, ref_anno=None):
    """
    Isoform assembling pipeline: stages 1-8 + 13 (data preprocessing, gene grouping,
    isoform classification, filtering and quantification).
    Returns assembled and quantified isoforms ready for validation.
    """
    sample = os.path.splitext(os.path.basename(bam))[0]
    logger = logging.getLogger("AIDRS")
    logger.info(f"Processing sample: {bam}")
    
    # Stage 1.1: Data Preprocessing
    logger.info("\tStage 1.1: SSC data preprocessing...")
    df = load_data(reference = args.reference, bam = bam, output = args.output, num_threads = args.threads, min_aln_coverage = args.min_aln_coverage, min_aln_identity = args.min_aln_identity)
    len_freq1 = len(df)
    logger.info(f"\tData preprocessing completed. Loaded {len(df)} SSC records.")

    # Stage 1.2: Frequency-based SSC Filtering
    logger.info("\tStage 1.2: Filtering SSC with low read support...")
    df = df[df['frequency'] >= args.filter_freq]
    len_freq2 = len(df)
    logger.info(f"\tFrequency filtering completed. Retained {len(df)} SSC records.")
    
    # Stage 1.3: Gene-level Clustering
    logger.info("\tStage 1.3: Performing gene-level clustering based on genomic coordinates...")
    gene_clustering = GeneClustering(num_processes=args.threads)
    df = gene_clustering.cluster(df)
    logger.info(f"\tGene clustering completed. Processed {len(df)} SSC records into gene groups.")
    
    # Stage 1.4: Junction Motif Analysis
    logger.info("\tStage 1.4: Analyzing splice junction motifs (canonical vs non-canonical)...")
    df = junction_screening(df, junction_freq_ratio=args.junction_freq_ratio)
    logger.info(f"\tJunction motif analysis completed. Retained {len(df)} SSC records.")

    # Stage 1.5: Consensus-based Junction Refinement
    logger.info("\tStage 1.5: Refining splice junctions using consensus strategy...")
    consensusfilter = ConsensusFilter(consensus_bp = args.consensus_bp,
                                      consensus_multiple = args.consensus_multiple,
                                      num_processes=args.threads)
    df = consensusfilter.consensus(df)
    logger.info(f"\tConsensus refinement completed. Retained {len(df)} SSC records.")
    
    # Stage 1.6: Low-confidence Junction Pruning
    logger.info("\tStage 1.6: Removing low-confidence splice junctions...")
    if len_freq2/len_freq1 < 0.55:
        splice_consensus_filter = SpliceConsensusFilter(threshold_lowWeight_edges = args.threshold_lowWeight_edges,
                                          num_processes=args.threads)
        df = splice_consensus_filter.remove_SJ(df)
    logger.info(f"\tJunction pruning completed. Retained {len(df)} SSC records.")
    
    # Stage 1.7: Fragmentary Transcript Filtering
    logger.info("\tStage 1.7: Filtering fragmentary transcripts...")
    df = filter_fragmentary_transcript(df, threshold_fragmentary_transcript_bp = args.threshold_fragmentary_transcript_bp)
    logger.info(f"\tFragmentary transcript filtering completed. Retained {len(df)} SSC records.")

    # Stage 1.8: Novel Isoform Classification and Filtering (NNC/NIC)
    logger.info("\tStage 1.8: Classifying and filtering novel isoforms (NNC/NIC)...")
    nncnicgraphprocessor = NncNicGraphProcessor(exon_excursion_diff_bp=args.exon_excursion_diff_bp, error_sites_diff_bp=args.error_sites_diff_bp,
                                                error_sites_multiple=args.error_sites_multiple, little_exon_bp=args.little_exon_bp,
                                                little_exon_mismatch_diff_bp=args.little_exon_mismatch_diff_bp, 
                                                Nolittle_exon_mismatch_diff_bp=args.Nolittle_exon_mismatch_diff_bp,
                                                little_exon_jump_multiple=args.little_exon_jump_multiple, 
                                                num_processes = args.threads)
    df = nncnicgraphprocessor.nnc_nic_graph(df)
    
    logger.info(f"\tNovel isoform filtering completed. Retained {len(df)} SSC records.")

    # Stage 1.9: GTF Rescue FSM
    if ref_anno is not None:
        df_raw = pd.read_parquet(os.path.join(args.output, f"temp/{sample}.ssc_flnc.parquet"))
        gtf_rescue_filtering = GTFRescueFiltering(little_exon_bp=args.little_exon_bp,
                                mismatch_error_sites_bp=args.mismatch_error_sites_bp, 
                                mismatch_error_sites_groupfreq_multiple=args.mismatch_error_sites_groupfreq_multiple,
                                exon_excursion_diff_bp=args.exon_excursion_diff_bp,
                                fake_exon_group_freq_multiple=args.fake_exon_group_freq_multiple,
                                fake_exon_bp=args.fake_exon_group_freq_multiple,
                                ism_freqRatio_notrun = args.ism_freqRatio_notrun,
                                cluster_group_size = args.cluster_group_size, eps = args.eps, min_samples = args.min_samples,
                                num_processes = args.threads)
        df = gtf_rescue_filtering.anno_process(df_raw,df,ref_anno)
        logger.info(f"\tStage 1.9: GTF rescue completed. Retained {len(df)} SSC records.")
    
    df.to_parquet(os.path.join(args.output, f"temp/{sample}.ssc_flnc_correct.parquet"))

    return df, sample


def isoform_validating(df, args):
    """
    Isoform validation pipeline: stages 9-12 (terminal site prediction, 
    functional annotation, truncation analysis, and filtering).
    Takes assembled isoforms and returns validated, annotated results.
    """
    logger = logging.getLogger("AIDRS")

    gene_clustering = GeneClustering(num_processes=args.threads)
    df = gene_clustering.cluster(df)

    # --- Extract all sample names ---
    # Assume column naming format: {sample}_TrStart_reads, {sample}_TrEnd_reads, {sample}_frequency
    # Extract unique sample names through column names
    samples = set()
    for col in df.columns:
        match = re.match(r'(.+?)_(TrStart_reads|TrEnd_reads|frequency)', col)
        if match:
            samples.add(match.group(1))
    samples = sorted(samples)  # Sort to ensure reproducibility

    if not samples:
        raise ValueError("No sample-specific columns found in the dataframe (expected pattern: {sample}_TrStart_reads etc.)")

    # --- Generate all flnc.ssc paths ---
    flnc_paths = [
        os.path.join(args.output, f"temp/{sample}_flnc.ssc")
        for sample in samples
    ]

    # Stage 2.1: Transcript Start/End Site Prediction
    logger.info("Stage 2.1: Predicting transcript boundary using DBSCAN...")
    terminalsitesprocessor = TerminalSitesProcessor(
        cluster_group_size=args.cluster_group_size,
        eps=args.eps,
        min_samples=args.min_samples,
        num_processes=args.threads
    )
    df = terminalsitesprocessor.get_terminal_sites(df)
    logger.info(f"Transcript start/end prediction completed. Retained {len(df)} SSC records.")

    # Stage 2.*: Functional Annotation (TSS/polyA/ORF/NMD)
    logger.info("Stage 2.2: Performing Puffin TSS annotations...")
    puffin_tss = TSS_Puffin(genome=args.reference, num_processes=args.threads, tmp_path=f'{args.output}/temp', puffin_prediction_threshold=args.puffin_prediction_threshold)
    puffin_tss.tss_anno_by_puffin(df)
    df = puffin_tss.aggr_puffin_result(df, f'{args.output}/temp/puffin_out')

    logger.info("Stage 2.3: Performing PolyA fraction and quantification...")
    polyanno = polyAnnotator(args)
    df = polyanno.anno_polya(df, flnc_paths)

    logger.info("Stage 2.4: Performing TranslationAI ORF prediction and NMD assessment...")
    transai = TranslationAI_ORF(genome=args.reference, tmp_path=f"{args.output}/temp", translationai_score_threshold=args.translationai_score_threshold, num_processes=args.threads)
    transai.orf_predict_by_translationai(df)
    df = transai.aggr_translationai_result(df, f'{args.output}/temp/TranslationAI_temp')
    logger.info(f"Functional annotation completed. Annotated {len(df)} SSC records.")

    logger.info("Stage 2.5: Performing Truncation assessment...")
    truncationprocessor = TruncationProcessor(
        threshold_logFC_truncation_source_freq=args.threshold_logFC_truncation_source_freq,
        threshold_logFC_truncation_group_freq=args.threshold_logFC_truncation_group_freq,
        num_processes=args.threads
    )
    df = truncationprocessor.get_truncation(df)
    df = truncationprocessor.tag_truncation(df)

    # Stage 2.6: Isoform Filtering
    logger.info("Stage 2.6: Applying TSS correction and filtering...")
    df = transcript_model_filtering(df, args.puffin_prediction_threshold, args.polya_fraction_threshold, args.hard_filter)
    logger.info(f"TSS correction and filtering completed. Retained {len(df)} records.")

    return df
    


def run_pipeline(args, ref_anno=None):
    """
    Complete Isoform assembly + validation + quantification + merging pipeline
    Input: args (argument object), optional ref_anno (reference annotation GTF)
    Output: Final merged DataFrame, based on df_transcript_model, retaining Chr, Strand, SSC, TrStart, TrEnd,
          filling missing sample columns in df_result with 0 and discarding extra rows.
    """
    logger = logging.getLogger("AIDRS")
    logger.info("Starting full isoform analysis pipeline...")

    # === Step 1: Assemble each BAM ===
    df_dict = {}
    for bam in args.bam:
        df, sample = isoform_assembling(bam, args, ref_anno)
        df_dict[sample] = df

    # === Step 2: Rename columns + merge assembly results from all samples ===
    dfs_processed = []

    for sample, df in df_dict.items():
        df = df.copy()
        # Rename key columns with sample prefix
        rename_cols = {
            'TrStart_reads': f'{sample}_TrStart_reads',
            'TrEnd_reads': f'{sample}_TrEnd_reads',
            'frequency': f'{sample}_frequency'
        }
        df = df.rename(columns=rename_cols)
        # Keep key columns for merging + renamed value columns
        cols_to_keep = ['Chr', 'Strand', 'SSC'] + list(rename_cols.values())
        df = df[cols_to_keep]
        dfs_processed.append(df)

    # Merge all samples (outer join on Chr, Strand, SSC)
    merged_df = dfs_processed[0]
    for df in dfs_processed[1:]:
        merged_df = pd.merge(merged_df, df, on=['Chr', 'Strand', 'SSC'], how='outer')

    # Sort
    merged_df = merged_df.sort_values(['Chr', 'Strand', 'SSC']).reset_index(drop=True)

    # === Step 3: Fill missing values ===
    for col in merged_df.columns:
        if 'TrStart_reads' in col or 'TrEnd_reads' in col:
            merged_df[col] = merged_df[col].apply(lambda x: x if isinstance(x, np.ndarray) else np.array([]))
        elif 'frequency' in col:
            merged_df[col] = merged_df[col].fillna(0).astype(int)

    merged_df = rescue_low_frep_reads(merged_df, df_dict, args) # Rescue low frequency reads

    # === Step 4: Merge reads and frequency from all samples ===
    trstart_cols = [col for col in merged_df.columns if 'TrStart_reads' in col]
    trend_cols = [col for col in merged_df.columns if 'TrEnd_reads' in col]
    freq_cols = [col for col in merged_df.columns if 'frequency' in col]

    merged_df['TrStart_reads'] = merged_df[trstart_cols].apply(
        lambda row: np.concatenate([x for x in row.values if isinstance(x, np.ndarray)]) 
                    if any(isinstance(x, np.ndarray) for x in row.values) else np.array([]),
        axis=1
    )

    merged_df['TrEnd_reads'] = merged_df[trend_cols].apply(
        lambda row: np.concatenate([x for x in row.values if isinstance(x, np.ndarray)]) 
                    if any(isinstance(x, np.ndarray) for x in row.values) else np.array([]),
        axis=1
    )

    merged_df['frequency'] = merged_df[freq_cols].sum(axis=1).astype(int)



    # === Step 5: Validation pipeline (boundary prediction, functional annotation, truncation analysis) ===
    df_transcript_model = isoform_validating(merged_df, args)

    # === Step 6: Quantify each sample ===
    quantifier = IsoformQuantifier(num_processes=args.threads)
    quant_list = []

    for sample in df_dict.keys():
        # Fix path: using f-string formatting
        flnc_path = os.path.join(args.output, f"temp/{sample}.ssc_flnc_correct.parquet")
        read_df = pd.read_parquet(flnc_path)
        
        # Correction + quantification
        read_df = correct_flnc(df_transcript_model, read_df, args=args, terminal_cluster=True)
        read_df = quantifier.assign_quantification(read_df, flnc_path)
        read_df['sample'] = sample
        # Remove 'Group' column (if exists)
        if 'Group' in read_df.columns:
            read_df = read_df.drop(columns=['Group'])
        quant_list.append(read_df)

    # Merge all quantification results
    df_quant = pd.concat(quant_list, ignore_index=True)


    # === Step 7: Left join df_quant based on df_transcript_model as reference ===
    # Columns to retain: Chr, Strand, SSC, TrStart, TrEnd and other functional annotation columns
    model_cols = ['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd', 'Puffin_TSS_15bp', 'Puffin_TSS_50bp', 'polyA_frac', 'TIS_related_location', 'TTS_related_location', 'Predict_NMD', 'truncation']
    
    # Ensure df_transcript_model has these columns
    if not set(model_cols).issubset(df_transcript_model.columns):
        raise ValueError(f"df_transcript_model is missing required columns: {set(model_cols) - set(df_transcript_model.columns)}")

    # Key columns for joining (must exist in df_quant)
    merge_keys = ['Chr', 'Strand', 'SSC', 'TrStart', 'TrEnd']
    
    # Ensure df_quant has the key columns
    if not set(merge_keys).issubset(df_quant.columns):
        raise ValueError(f"df_quant is missing key columns: {set(merge_keys) - set(df_quant.columns)}")

    # Left join: retain all rows and model_cols columns from df_transcript_model, join with df_quant by merge_keys
    df_final = df_transcript_model[model_cols].merge(
        df_quant,
        on=merge_keys,
        how='inner'
    )

    return df_final


def parse_args(cmd_args):
    parser = argparse.ArgumentParser(description="AIDRS: AI-Aided Isoform Discovery for direct RNA-Seq")
    parser.add_argument("--reference", "-r", type=str, required=True, help="Reference genome FASTA file.")
    parser.add_argument("--bam", "-b", type=str, required=True, nargs='+', help="Input BAM file(s).")
    parser.add_argument("--output", "-o", type=str, default="aidrs_output", help="Output directory. Default: aidrs_output")
    parser.add_argument("--threads", "-t", type=int, default=4, help="Number of threads to use. Default: 4")
    parser.add_argument("--keep_temp", action="store_true", help="Keep intermediate files in the temp directory.")

    # Alignment filtering
    parser.add_argument("--min_aln_identity", type=float, default=0.95, help="Minimum alignment identity. Default: 0.95")
    parser.add_argument("--min_aln_coverage", type=float, default=0, help="Minimum alignment coverage. Default: 0")

    # SSC filtering
    parser.add_argument("--filter_freq", type=float, default=5, help="Minimum read support to retain an SSC. Default: 5")
    parser.add_argument("--min_junction_freq", type=float, default=5, help="Minimum read support for splice junctions. Default: 5")
    parser.add_argument("--junction_freq_ratio", type=float, default=0.25, help="Minimum frequency ratio for junction screening. Default: 0.25")

    # Consensus filtering
    parser.add_argument("--consensus_bp", type=int, default=20, help="Allowed deviation (bp) in consensus correction. Default: 20")
    parser.add_argument("--consensus_multiple", type=float, default=0.1, help="Supporting read ratio for consensus correction. Default: 0.1")

    # Graph edge filtering
    parser.add_argument("--threshold_lowWeight_edges", type=float, default=0.05, help="Threshold ratio for filtering weak edges. Default: 0.05")

    # NNC/NIC filtering
    parser.add_argument("--exon_excursion_diff_bp", type=int, default=20, help="Maximum exon position deviation allowed. Default: 20")
    parser.add_argument("--error_sites_diff_bp", type=int, default=10, help="Max position deviation for suspected error sites. Default: 10")
    parser.add_argument("--error_sites_multiple", type=float, default=0.01, help="Read ratio threshold for error site detection. Default: 0.01")
    parser.add_argument("--little_exon_bp", type=int, default=30, help="Maximum size for a small exon. Default: 30")
    parser.add_argument("--little_exon_mismatch_diff_bp", type=int, default=10, help="Position difference for mismatch in small exons. Default: 10")
    parser.add_argument("--Nolittle_exon_mismatch_diff_bp", type=int, default=20, help="Mismatch difference threshold for non-small exons. Default: 20")
    parser.add_argument("--little_exon_jump_multiple", type=float, default=0.1, help="Ratio threshold for exon skipping detection. Default: 0.1")

    # ISM truncation filtering
    parser.add_argument("--threshold_logFC_truncation_source_freq", type=float, default=0, help="logFC threshold of source SSCs for truncation. Default: 0")
    parser.add_argument("--threshold_logFC_truncation_group_freq", type=float, default=-100, help="logFC threshold for group truncation filtering. Default: -100")

    # Fragmentary transcript filtering
    parser.add_argument("--threshold_fragmentary_transcript_bp", type=int, default=100, help="Minimum length required to retain transcript. Default: 100")

    # Annotation-based filtering
    parser.add_argument("--gtf_anno", "-g", type=str, default=None, help="Optional GTF/GFF file for annotation-based transcript rescue and filtering.")
    parser.add_argument("--mismatch_error_sites_bp", type=int, default=20, help="Max deviation to define mismatched error sites. Default: 20")
    parser.add_argument("--mismatch_error_sites_groupfreq_multiple", type=float, default=0.25, help="Read ratio threshold for mismatch errors. Default: 0.25")
    parser.add_argument("--fake_exon_group_freq_multiple", type=float, default=0.1, help="Group ratio threshold for fake exon detection. Default: 0.1")
    parser.add_argument("--fake_exon_bp", type=int, default=50, help="Max length of a potential fake exon. Default: 50")
    parser.add_argument("--ism_freqRatio_notrun", type=float, default=0.5, help="Minimum ratio to retain non-truncated ISMs. Default: 0.5")

    # Transcription start/end prediction
    parser.add_argument("--cluster_group_size", type=int, default=1500, help="Max group size for TS clustering. Default: 1500")
    parser.add_argument("--eps", type=int, default=15, help="DBSCAN epsilon (distance threshold). Default: 15")
    parser.add_argument("--min_samples", type=int, default=20, help="Minimum samples for TS cluster. Default: 20")
    parser.add_argument("--puffin_prediction_threshold", type=float, default=0.02, help="Puffin prediction threshold for TSS annotation and filtering. Default: 0.02")
    parser.add_argument("--polya_fraction_threshold", type=float, default=0.95, help="PolyA fraction threshold for transcript filtering. Default: 0.95")
    parser.add_argument("--translationai_score_threshold", type=float, default=0.9, help="TranslationAI score threshold for ORF prediction. Default: 0.9")
    parser.add_argument("--hard_filter", action="store_true", help="Hard filtering based on Puffin_TSS_15bp and polyA_frac thresholds.")
    
    # Thresholds for splice site (SS) and transcription start/end site (TSS/TES) correction
    parser.add_argument("--ss_tolerance", type=int, default=15, help="Splice site tolerance threshold for correction. Default: 15")
    parser.add_argument("--terminal_tolerance", type=int, default=50, help="Terminal site (TSS/TES) tolerance threshold for correction. Default: 50")

    args = parser.parse_args(cmd_args)
    return args


def main(cmd_args):
    args = parse_args(cmd_args)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "temp"), exist_ok=True)
    logger = setup_logger(args.output)
    logger.info("=== AIDRS pipeline started === ")

    try:
        logger.info(f"Processing BAM files...")
        output_ssc = os.path.join(args.output, "temp")
        run_bam2ssc(args.reference, args.bam, output_ssc, args.threads)
        logger.info(f"BAM files were converted to SSC format.")

        if args.gtf_anno:
            run_Ref2SSC(args.gtf_anno, args.output, args.threads)
            ref_anno = pd.read_csv(f"{args.output}/temp/anno.ssc",sep='\t')
            ref_anno = ref_anno[ref_anno['SSC'].notna()]
        else:
            ref_anno = None

        if ref_anno is None:
            df_result = run_pipeline(args)
        else:
            df_result = run_pipeline(args,ref_anno)

        logger.info("Stage 3: Output Generating...")
        annotator = IsoformAnnotator(num_processes=args.threads, terminal_tolerance=args.terminal_tolerance)
        annotator.save_results(df_result,args.output,args.reference,ref_anno)

        if not args.keep_temp:
            temp_dir = os.path.join(args.output, "temp")
            shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info("=== AIDRS pipeline finished === ")

    except Exception as e:
        logger.error("An error occurred:")
        logger.error(str(e))
        print_exc()
        sys.exit(1)


def main_entry():
    """Entry point for the command line script"""
    try:
        main(sys.argv[1:])
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(130)


if __name__ == "__main__":
    main_entry()