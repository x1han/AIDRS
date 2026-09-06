"""GTF + TSV report writers (high-level).

This module owns the high-level report-writing logic that
``src.generate_reports.IsoformAnnotator.save_results`` orchestrates:

- ``parse_exons``: turn a ``(TrStart, SSC, TrEnd)`` triple into a list
  of exon tuples, strand-aware (negative strand is sorted on the
  negated axis then un-negated via ``abs``).
- ``reverse_coords_and_order``: strand-flip helper for negative-strand
  ribosome-walk output (swap each tuple, reverse the list).
- ``simulate_ribosome_walk``: walk exons to derive CDS / UTR5 / UTR3 /
  start_codon / stop_codon coordinates from ``TIS_related_location`` /
  ``TTS_related_location``.
- ``calculate_donor_acceptor_sites``: derive intron donor/acceptor
  coordinates from exon list.
- ``sort_gtf_by_hierarchy``: post-process a GTF file to enforce the
  ``gene -> transcript -> subfeature`` hierarchy required by the GTF
  spec; also reorders features by priority.
- ``to_gtf``: write ``aidrs.transcript_model.gtf`` from the annotated
  DataFrame. Pinned by
  tests/test_generate_reports_characterization.py::test_to_gtf_byte_identical.
- ``to_fasta``: write ``aidrs.transcript_model.fasta`` from the GTF +
  pyfaidx-indexed genome.

Functions are module-level (not class methods) so they are testable
without instantiating the full ``IsoformAnnotator`` orchestrator. The
class still keeps thin shim methods ``to_gtf`` / ``to_fasta`` /
``_reverse_complement`` for backward compatibility with ``aidrs.py``
and existing characterization tests.

Extracted from the original ``src/generate_reports.py`` "god class"
(IsoformAnnotator).
"""
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import gffutils
import numpy as np
import pandas as pd
import pyfaidx

from .column_standardize import reverse_complement


def parse_exons(tr_start, ssc_str, tr_end, strand):
    """Return exon intervals ``[(s,e), ...]`` parsed from TrStart + SSC + TrEnd.

    For ``+`` strand, sort the coordinates ascending. For ``-`` strand,
    sort on the negated axis then ``abs`` back to genomic coordinates
    (the result is the same as for ``+`` because both endpoints are
    sorted -- but the parity of which pair forms an exon differs; the
    original implementation uses this pattern verbatim and we preserve
    it byte-for-byte).
    """
    if strand == '+':
        ssc_list = np.sort([tr_start] + list(map(int, ssc_str.split('-'))) + [tr_end])
    else:
        ssc_list = np.sort(-np.array([tr_start] + list(map(int, ssc_str.split('-'))) + [tr_end]))
    exons = []
    for i in range(0, len(ssc_list)-1, 2):
        exons.append((ssc_list[i], ssc_list[i+1]))
    return exons


def reverse_coords_and_order(data: Dict[str, Union[List[Tuple[int, int]],
                                                  Tuple[int, int]]]
                             ) -> Dict[str, Union[List[Tuple[int, int]],
                                                   Tuple[int, int]]]:
    """Two-step transform for negative-strand ribosome-walk output.

    1. Swap coordinate order within each tuple: ``(a,b) -> (b,a)`` via
       ``abs`` to undo the negation done by ``parse_exons`` on
       negative strands.
    2. Reverse the entire list corresponding to each key.
    For keys with single-tuple values (e.g. ``start_codon``), only
    step 1 is performed.

    Finally swaps the ``utr5`` and ``utr3`` keys -- because the
    ribosome walked in 5'->3' transcript direction, but on the negative
    strand the genomic coordinates are 3'->5', so 5' UTR becomes 3' UTR
    in genomic-coordinate land and vice versa.
    """
    def swap(t: Tuple[int, int]) -> Tuple[int, int]:
        return abs(t[1]), abs(t[0])
    out = {}
    for k, v in data.items():
        if isinstance(v, list):
            # First swap each tuple internally, then reverse the entire list
            out[k] = [swap(t) for t in reversed(v)]
        else:
            # Single tuple
            out[k] = swap(v)
    out['utr5'], out['utr3'] = out.pop('utr3'), out.pop('utr5')
    return out


def simulate_ribosome_walk(row):
    """Walk exons and emit CDS / UTR5 / UTR3 / start_codon / stop_codon features.

    Strand-aware: on negative strand, ``reverse_coords_and_order`` is
    applied at the end to flip coordinates.

    Args:
        row: A pandas Series-like row with attributes ``TrStart``,
             ``SSC``, ``TrEnd``, ``Strand``, ``TIS_related_location``,
             ``TTS_related_location``.

    Returns:
        Dict with keys ``exons``, ``cds``, ``utr5``, ``utr3``,
        ``start_codon``, ``stop_codon``.
    """
    exons = parse_exons(int(row.TrStart), row.SSC, int(row.TrEnd), row.Strand)
    # Initialize variables
    current_exon_idx = 0
    current_genomic_pos = exons[0][0]  # Start from the beginning position of the first exon
    current_relative_pos = 1
    # Initialize feature lists
    features = {
        'exons': exons,
        'cds': [],
        'utr5': [],
        'utr3': [],
        'start_codon': None,
        'stop_codon': None,
    }
    # Current region being recorded
    current_region = 'utr5'
    region_start_genomic = current_genomic_pos
    region_start_relative = current_relative_pos
    # Iterate through each exon
    while current_exon_idx < len(exons):
        current_exon = exons[current_exon_idx]
        # Iterate through each position in the current exon
        while current_genomic_pos <= current_exon[1]:
            # Check for region changes
            if current_relative_pos == int(row.TIS_related_location) + 1:
                # End 5' UTR, start CDS
                if current_region == 'utr5':
                    # Record current UTR5 segment
                    if region_start_genomic <= current_genomic_pos - 1:
                        features['utr5'].append((region_start_genomic, current_genomic_pos - 1))
                    current_region = 'cds'
                    region_start_genomic = current_genomic_pos
                    region_start_relative = current_relative_pos
                    features['start_codon'] = (current_genomic_pos, current_genomic_pos + 2)
            elif current_relative_pos == int(row.TTS_related_location) + 1:
                # End CDS, start 3' UTR
                if current_region == 'cds':
                    # Record current CDS segment
                    if region_start_genomic <= current_genomic_pos:
                        features['cds'].append((region_start_genomic, current_genomic_pos - 1))
                    current_region = 'utr3'
                    region_start_genomic = current_genomic_pos
                    region_start_relative = current_relative_pos
                    features['stop_codon'] = (current_genomic_pos, current_genomic_pos + 2)
            # Move to next position
            current_genomic_pos += 1
            current_relative_pos += 1
        # Current exon has been processed, move to next exon
        current_exon_idx += 1
        if current_exon_idx < len(exons):
            # Record current region segment in current exon
            if current_region == 'utr5':
                features['utr5'].append((region_start_genomic, current_exon[1]))
            elif current_region == 'cds':
                features['cds'].append((region_start_genomic, current_exon[1]))
            elif current_region == 'utr3':
                features['utr3'].append((region_start_genomic, current_exon[1]))
            # Move to start position of next exon
            current_genomic_pos = exons[current_exon_idx][0]
            region_start_genomic = current_genomic_pos
    # Add last region segment of last exon
    if current_region == 'utr5':
        features['utr5'].append((region_start_genomic, exons[-1][1]))
    elif current_region == 'cds':
        features['cds'].append((region_start_genomic, exons[-1][1]))
    elif current_region == 'utr3':
        features['utr3'].append((region_start_genomic, exons[-1][1]))
    if row.Strand == '-':
        features = reverse_coords_and_order(features)
    return features


def calculate_donor_acceptor_sites(exons, strand):
    """Calculate intron donor and acceptor sites from exon list.

    For each adjacent exon pair, the intron spans ``[exon1.end+1, exon2.start-1]``.

    On ``+`` strand: donor is the first two bases at the 5' end of the
    intron, acceptor is the last two bases at the 3' end.

    On ``-`` strand: the labels swap (donor at 3' end of intron,
    acceptor at 5' end).

    Returns:
        Tuple ``(donor_sites, acceptor_sites)``, each a list of
        ``(start, end)`` tuples (each spanning exactly 2 bp).
    """
    donor_sites = []
    acceptor_sites = []

    # Iterate through adjacent exon pairs to calculate intron donor and acceptor sites
    for i in range(len(exons) - 1):
        exon1 = exons[i]
        exon2 = exons[i+1]
        intron_start = exon1[1] + 1  # Intron starts after first exon ends
        intron_end = exon2[0] - 1    # Intron ends before second exon starts

        if strand == '+':
            # Positive strand: donor_site is the first two bases at the 5' end of intron
            # acceptor_site is the last two bases at the 3' end of intron
            donor_site = (intron_start, intron_start + 1)
            acceptor_site = (intron_end - 1, intron_end)
        else:
            # Negative strand: donor_site is the last two bases at the 3' end of intron
            # acceptor_site is the first two bases at the 5' end of intron
            # Note: For negative strand, the donor and acceptor sites are reversed
            donor_site = (intron_end - 1, intron_end)      # Last two bases of intron
            acceptor_site = (intron_start, intron_start + 1)  # First two bases of intron

        donor_sites.append(donor_site)
        acceptor_sites.append(acceptor_site)

    return donor_sites, acceptor_sites


def sort_gtf_by_hierarchy(gtf_file, output_file):
    """Re-sort a GTF file by ``gene -> transcript -> subfeature`` hierarchy.

    Uses ``gffutils`` to parse the input GTF into an in-memory database,
    then re-emits features in the canonical hierarchy required by the
    GTF spec (genes sorted by ``(seqid, start)``, transcripts by
    ``start`` within their gene, subfeatures by ``(priority, start)``
    within their transcript).
    """
    # Create database
    db = gffutils.create_db(gtf_file, ':memory:',
                        disable_infer_genes=True,
                        disable_infer_transcripts=True,
                        merge_strategy='create_unique')
    # Collect transcripts by gene
    genes = {}
    for gene in db.features_of_type('gene'):
        genes[gene.id] = {
            'feature': gene,
            'transcripts': defaultdict(list)
        }
    # Collect transcripts for each gene
    for transcript in db.features_of_type('transcript'):
        gene_id = transcript.attributes.get('gene_id', [None])[0]
        if gene_id and gene_id in genes:
            genes[gene_id]['transcripts'][transcript.id].append(transcript)
    # Collect sub-features for each transcript
    transcripts_features = defaultdict(list)
    for feature in db.features_of_type(['exon', 'CDS', 'UTR5', 'UTR3', 'UTR', 'start_codon', 'stop_codon', 'donor_site', 'acceptor_site']):
        transcript_id = feature.attributes.get('transcript_id', [None])[0]
        if transcript_id:
            transcripts_features[transcript_id].append(feature)
    # Define priority of feature types
    feature_priority = {
        'transcript': 0,
        'exon': 1,
        'CDS': 2,
        'UTR5': 3,
        'UTR3': 3,
        'UTR': 3,
        'start_codon': 4,
        'stop_codon': 5,
        'donor_site': 6,
        'acceptor_site': 6
    }
    with open(output_file, 'w') as f:
        # Sort by gene start position
        sorted_genes = sorted(genes.values(), key=lambda x: (x['feature'].seqid, x['feature'].start))
        for gene_info in sorted_genes:
            gene_feature = gene_info['feature']
            print(gene_feature, file=f)
            # Get all transcripts of this gene (sorted by start position)
            gene_transcripts = []
            for transcript_list in gene_info['transcripts'].values():
                for transcript in transcript_list:
                    gene_transcripts.append(transcript)
            # Sort transcripts by start position
            sorted_transcripts = sorted(gene_transcripts, key=lambda x: x.start)
            for transcript in sorted_transcripts:
                print(transcript, file=f)
                # Get all sub-features of this transcript
                transcript_id = transcript.id
                if transcript_id in transcripts_features:
                    sub_features = transcripts_features[transcript_id]
                    # Sort by feature type priority and position
                    sorted_sub_features = sorted(
                        sub_features,
                        key=lambda x: (
                            feature_priority.get(x.featuretype, 999),
                            x.start
                        )
                    )
                    for sub_feature in sorted_sub_features:
                        print(sub_feature, file=f)


def to_gtf(df: pd.DataFrame, output_dir: str) -> None:
    """Write ``aidrs.transcript_model.gtf`` for the annotated DataFrame.

    Aggregate by GeneID: output the minimum TrStart_ref and maximum
    TrEnd_ref as the gene range, including complete annotations for
    ``gene / transcript / exon / CDS / start_codon / stop_codon / UTR /
    donor_site / acceptor_site``. Output is hierarchically sorted by
    ``sort_gtf_by_hierarchy`` into ``<output_dir>/aidrs.transcript_model.gtf``.

    CDS-bearing transcripts (``Predict_NMD == 'Normal'`` and both TIS /
    TTS locations present) go through ``simulate_ribosome_walk`` to
    derive CDS / UTR / codon coordinates; on exception, fall back to
    exon-only emission with a ``Warning: Error processing transcript``
    line to stderr.

    Pinned by
    tests/test_generate_reports_characterization.py::test_to_gtf_byte_identical.
    """
    # 1. Only retain necessary columns
    need = ['Chr', 'Strand', 'TrStart', 'TrEnd', 'TrID', 'GeneID', 'GeneName',
            'SSC', 'TrStart_ref', 'TrEnd_ref', 'Predict_NMD', 'TIS_related_location', 'TTS_related_location']
    df = df[need].drop_duplicates()
    # 2. First parse exons for each transcript and record gene->chrom->strand mapping
    gene_chrom = {}
    gene_strand = {}
    tx_exons = {}          # trid -> [(s1,e1), (s2,e2), ...]
    gene_txs = {}          # gene_id  -> {trid1, trid2, ...}
    tx_rows = {}           # trid -> corresponding row data
    for _, r in df.iterrows():
        chrom = r['Chr']
        strand = r['Strand']
        gid = r['GeneID']
        trid = r['TrID']
        gene_chrom[gid] = chrom
        gene_strand[gid] = strand
        gene_txs.setdefault(gid, set()).add(trid)
        tx_rows[trid] = r  # Save row data
        # Parse SSC and integrate TrStart_ref and TrEnd_ref
        tr_start_ref = int(r['TrStart_ref'])
        tr_end_ref = int(r['TrEnd_ref'])
        # Parse splice sites in SSC
        block = list(map(int, r['SSC'].split('-')))
        if len(block) % 2:
            block = block[:-1]
        # Build complete exon structure: TrStart_ref + SSC + TrEnd_ref
        all_coords = [tr_start_ref]
        for coord in block:
            all_coords.append(coord)
        all_coords.append(tr_end_ref)
        # Sort coordinates (based on strand direction)
        all_coords.sort()
        # Build exons
        exons = []
        for i in range(0, len(all_coords)-1, 2):
            s, e = all_coords[i], all_coords[i+1]
            exons.append((s, e))
        tx_exons[trid] = exons
    # 3. Calculate genomic range for each gene (using TrStart_ref and TrEnd_ref)
    gene_span = {}
    for _, r in df.iterrows():
        gid = r['GeneID']
        s0 = int(r['TrStart_ref'])
        e0 = int(r['TrEnd_ref'])
        if gid not in gene_span:
            gene_span[gid] = [s0, e0]
        else:
            gene_span[gid][0] = min(gene_span[gid][0], s0)
            gene_span[gid][1] = max(gene_span[gid][1], e0)
    # 4. Collect all unique exons under each gene
    gene_exons = {}        # gid -> {(s,e), ...}
    for gid, trset in gene_txs.items():
        exon_pool = set()
        for trid in trset:
            exon_pool.update(tx_exons[trid])
        gene_exons[gid] = sorted(exon_pool)
    # 5. Create temp directory if it doesn't exist
    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_output_path = os.path.join(temp_dir, 'aidrs.transcript_model.gtf')
    final_output_path = os.path.join(output_dir, 'aidrs.transcript_model.gtf')
    # 6. Write GTF to temporary file first
    with open(temp_output_path, 'w') as fo:
        for gid in sorted(gene_chrom):
            chrom = gene_chrom[gid]
            strand = gene_strand[gid]
            g_s, g_e = gene_span[gid]
            # 5.1 gene line
            fo.write(f"{chrom}\tAIDRS\tgene\t{g_s}\t{g_e}\t.\t{strand}\t.\t"
                     f"gene_id \"{gid}\"; gene_name \"{df.loc[df.GeneID==gid, 'GeneName'].iloc[0]}\";\n")
            # 5.2 transcript line (retain each transcript)
            for trid in sorted(gene_txs[gid]):
                r = tx_rows[trid]
                tr_s = int(r['TrStart_ref'])
                tr_e = int(r['TrEnd_ref'])
                # Determine if there is CDS
                has_cds = (r['Predict_NMD'] == 'Normal' and
                          pd.notna(r['TIS_related_location']) and
                          pd.notna(r['TTS_related_location']))

                # Calculate donor and acceptor sites
                donor_sites, acceptor_sites = calculate_donor_acceptor_sites(tx_exons[trid], strand)

                fo.write(f"{chrom}\tAIDRS\ttranscript\t{tr_s}\t{tr_e}\t.\t{strand}\t.\t"
                         f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\";\n")
                if has_cds:
                    # Call your function to get CDS, UTR, etc.
                    try:
                        ribosome_data = simulate_ribosome_walk(r)
                        # Output exon (use exons from ribosome_data to ensure consistency with CDS/UTR)
                        for i, (es, ee) in enumerate(ribosome_data['exons'], 1):
                            fo.write(f"{chrom}\tAIDRS\texon\t{es}\t{ee}\t.\t{strand}\t.\t"
                                     f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\"; exon_number \"{i}\";\n")
                        # Output CDS
                        for cds_s, cds_e in ribosome_data['cds']:
                            fo.write(f"{chrom}\tAIDRS\tCDS\t{cds_s}\t{cds_e}\t.\t{strand}\t0\t"
                                     f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\";\n")
                        # Output start_codon
                        start_s, start_e = ribosome_data['start_codon']
                        fo.write(f"{chrom}\tAIDRS\tstart_codon\t{start_s}\t{start_e}\t.\t{strand}\t.\t"
                                 f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\";\n")
                        # Output stop_codon
                        stop_s, stop_e = ribosome_data['stop_codon']
                        fo.write(f"{chrom}\tAIDRS\tstop_codon\t{stop_s}\t{stop_e}\t.\t{strand}\t.\t"
                                 f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\";\n")
                        # Output UTR (unified UTR label)
                        for utr_s, utr_e in ribosome_data['utr5']:
                            fo.write(f"{chrom}\tAIDRS\tUTR\t{utr_s}\t{utr_e}\t.\t{strand}\t.\t"
                                     f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\";\n")
                        for utr_s, utr_e in ribosome_data['utr3']:
                            fo.write(f"{chrom}\tAIDRS\tUTR\t{utr_s}\t{utr_e}\t.\t{strand}\t.\t"
                                     f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\";\n")
                        for utr_s, utr_e in ribosome_data['utr5']:
                            fo.write(f"{chrom}\tAIDRS\tUTR5\t{utr_s}\t{utr_e}\t.\t{strand}\t.\t"
                                     f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\";\n")
                        for utr_s, utr_e in ribosome_data['utr3']:
                            fo.write(f"{chrom}\tAIDRS\tUTR3\t{utr_s}\t{utr_e}\t.\t{strand}\t.\t"
                                     f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\";\n")

                        # Output donor sites
                        for i, (ds_start, ds_end) in enumerate(donor_sites):
                            fo.write(f"{chrom}\tAIDRS\tdonor_site\t{ds_start}\t{ds_end}\t.\t{strand}\t.\t"
                                     f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\"; junction_number \"{i+1}\";\n")

                        # Output acceptor sites
                        for i, (as_start, as_end) in enumerate(acceptor_sites):
                            fo.write(f"{chrom}\tAIDRS\tacceptor_site\t{as_start}\t{as_end}\t.\t{strand}\t.\t"
                                     f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\"; junction_number \"{i+1}\";\n")
                    except Exception as e:
                        print(f"Warning: Error processing transcript {trid}: {e}")
                        # Fallback to output exon only on error
                        for i, (es, ee) in enumerate(tx_exons[trid], 1):
                            fo.write(f"{chrom}\tAIDRS\texon\t{es}\t{ee}\t.\t{strand}\t.\t"
                                     f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\"; exon_number \"{i}\";\n")

                        # Output donor sites for fallback case
                        for i, (ds_start, ds_end) in enumerate(donor_sites):
                            fo.write(f"{chrom}\tAIDRS\tdonor_site\t{ds_start}\t{ds_end}\t.\t{strand}\t.\t"
                                     f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\"; junction_number \"{i+1}\";\n")

                        # Output acceptor sites for fallback case
                        for i, (as_start, as_end) in enumerate(acceptor_sites):
                            fo.write(f"{chrom}\tAIDRS\tacceptor_site\t{as_start}\t{as_end}\t.\t{strand}\t.\t"
                                     f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\"; junction_number \"{i+1}\";\n")
                else:
                    # No CDS, output exon only
                    for i, (es, ee) in enumerate(tx_exons[trid], 1):
                        fo.write(f"{chrom}\tAIDRS\texon\t{es}\t{ee}\t.\t{strand}\t.\t"
                                 f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\"; exon_number \"{i}\";\n")

                    # Output donor sites
                    for i, (ds_start, ds_end) in enumerate(donor_sites):
                        fo.write(f"{chrom}\tAIDRS\tdonor_site\t{ds_start}\t{ds_end}\t.\t{strand}\t.\t"
                                 f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\"; junction_number \"{i+1}\";\n")

                    # Output acceptor sites
                    for i, (as_start, as_end) in enumerate(acceptor_sites):
                        fo.write(f"{chrom}\tAIDRS\tacceptor_site\t{as_start}\t{as_end}\t.\t{strand}\t.\t"
                                 f"gene_id \"{gid}\"; transcript_id \"{trid}\"; gene_name \"{r['GeneName']}\"; junction_number \"{i+1}\";\n")

    # 7. Sort the temporary GTF file by hierarchy and write to final output
    sort_gtf_by_hierarchy(temp_output_path, final_output_path)
    print("=== Transcript GTF file written to: ", final_output_path)


def to_fasta(gtf_file: str, genome_fasta: str, output_dir: str,
             df_result_after_quant: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """Generate transcript FASTA from GTF + genome FASTA.

    Args:
        gtf_file: Path to the GTF file (typically
                  ``<output_dir>/aidrs.transcript_model.gtf``).
        genome_fasta: Path to the reference genome FASTA (must be
                      pyfaidx-indexable; ``pyfaidx.Fasta(genome_fasta)``
                      called at start).
        output_dir: Directory to write ``aidrs.transcript_model.fasta``.
        df_result_after_quant: Optional DataFrame. When provided, a
                               ``seq_len`` column is added mapped from
                               the computed transcript sequences.

    Returns:
        ``df_result_after_quant`` with ``seq_len`` column added if it
        was provided; otherwise ``None``.
    """
    output_fasta = os.path.join(output_dir, 'aidrs.transcript_model.fasta')
    # Load genome sequence
    genome = pyfaidx.Fasta(genome_fasta)
    # Parse GTF file and extract transcript sequences
    db = gffutils.create_db(gtf_file, ':memory:',
                        disable_infer_genes=True,
                        disable_infer_transcripts=True,
                        merge_strategy='create_unique')
    # Dictionary to store transcript sequences
    transcript_sequences = {}
    # Process each transcript
    for transcript in db.features_of_type('transcript'):
        transcript_id = transcript.id
        gene_name = transcript.attributes.get('gene_name', [f'Gene_{transcript_id}'])[0]
        # Get all exons for this transcript
        exons = []
        for exon in db.children(transcript, featuretype='exon'):
            exons.append((exon.start, exon.end))
        # Sort exons by position
        exons.sort()
        if not exons:
            continue
        # Extract sequence from each exon and concatenate
        transcript_seq = ""
        chrom = transcript.seqid
        # Check if chromosome exists in genome
        if chrom not in genome:
            print(f"Warning: Chromosome {chrom} not found in genome file")
            continue
        for start, end in exons:
            # Extract exon sequence (GTF is 1-based, pyfaidx is 0-based)
            exon_seq = str(genome[chrom][start-1:end])
            transcript_seq += exon_seq
        # Handle reverse strand
        if transcript.strand == '-':
            # Reverse complement the sequence
            transcript_seq = reverse_complement(transcript_seq)
        transcript_sequences[transcript_id] = {
            'sequence': transcript_seq,
            'gene_name': gene_name
        }
    # Write sequences to FASTA file
    with open(output_fasta, 'w') as f:
        for tr_id, info in transcript_sequences.items():
            seq = info['sequence']
            gene_name = info['gene_name']
            # Write header with transcript ID and gene name
            header = f">{tr_id}"
            f.write(header + '\n')
            # Write sequence in 80-character lines
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')
    print(f"Transcript FASTA file written to: {output_fasta}")

    # If df_result_after_quant is provided, add seq_len column
    if df_result_after_quant is not None:
        # Add seq_len column to df_result_after_quant
        df_result_after_quant['seq_len'] = df_result_after_quant['TrID'].map(
            {tr_id: len(info['sequence']) for tr_id, info in transcript_sequences.items()}
        )
        return df_result_after_quant

    return None