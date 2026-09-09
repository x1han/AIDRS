#!/usr/bin/env python

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
import logging
import tempfile
from functools import partial
from collections import defaultdict
from Bio.Seq import Seq
import pysam

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert BAM to SSC format")
    parser.add_argument("--reference", "-r", help="reference genome in FASTA format", type=str)
    parser.add_argument("--bam", "-b", help="input BAM file(s)", type=str, nargs='+')
    parser.add_argument("--threads", "-t", help="number of threads to use [default=1]", type=int, default=1)
    parser.add_argument("--output", "-o", help="output folder, will be created automatically [default=aidrs_output]",
                        type=str, default="aidrs_output")
    return parser.parse_args()

def count_single_bam(bam, threads_per_bam):
    bai_file = bam + '.bai'
    if not os.path.exists(bai_file):
        try:
            # T1 debug: capture stderr + 60s timeout. samtools index on a busy
            # NFS mount or under fork can silently hang; timeout raises
            # TimeoutExpired which propagates up and identifies this call
            # as the stall point. stderr=PIPE lets the parent surface the
            # actual samtools error message if index fails.
            logger.info(f'[bam2ssc] samtools index start: {bam}')
            subprocess.run(
                ['samtools', 'index', '-@', str(threads_per_bam), bam],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=60,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning(f'Failed to create index for {bam} ({type(exc).__name__}: {exc}), proceeding without index')

    try:
        logger.info(f'[bam2ssc] samtools view -c start: {bam}')
        result = subprocess.run(
            ['samtools', 'view', '-c', '-@', str(threads_per_bam), bam],
            capture_output=True, text=True, timeout=60, check=True,
        )
        count = int(result.stdout.strip())
        return bam, count
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        if not os.path.exists(bai_file):
            raise RuntimeError(
                f"Cannot count reads in {bam}: neither a .bai index nor the samtools "
                f"binary is available (samtools attempt failed: {e}). "
                f"pysam.AlignmentFile.count() requires a .bai index. "
                f"Either install samtools or create a .bai index for {bam} "
                f"(e.g. `samtools index {bam}`)."
            )
        logger.warning(f'samtools view -c failed for {bam}: {e}, falling back to pysam')
        with pysam.AlignmentFile(bam, 'rb', threads=threads_per_bam) as bf:
            count = bf.count()
        logger.info(f'BAM {bam} has {count} reads (via pysam)')
        return bam, count

def get_bam_read_counts(bam_files, threads):
    num_bams = len(bam_files)
    # P1-1: guard Pool(processes=0) crash on empty bam_files. Caller decides
    # whether empty bam_files is fatal (typically sys.exit, not our job).
    if num_bams == 0:
        return {}, 0
    threads_per_bam = max(1, threads // num_bams) if num_bams > 0 else 1

    # Cap workers at the user-supplied thread budget. Spawning num_bams
    # workers is wasteful when num_bams >> threads (e.g. 55-BAM full run on
    # an 8-core box) and can trigger resource contention / hangs. Use the
    # smaller of num_bams and threads, matching the pattern already used
    # by merge_results() and main() below.
    pool_workers = min(num_bams, threads)
    print(f'[bam2ssc][T1] Pool#1 get_bam_read_counts enter: workers={pool_workers} bams={len(bam_files)}', flush=True)
    with mp.Pool(processes=pool_workers) as pool:
        results = pool.starmap(count_single_bam, [(bam, threads_per_bam) for bam in bam_files])
    print(f'[bam2ssc][T1] Pool#1 get_bam_read_counts exit', flush=True)

    bam_lines = dict(results)
    total_lines = sum(bam_lines.values())
    if total_lines == 0:
        sys.exit('No BAM lines to process')
    return bam_lines, total_lines

def allocate_chunks(bam_files, bam_lines, total_lines, threads):
    chunk_allocations = []
    num_bams = len(bam_files)
    
    base_threads_per_bam = max(1, threads // num_bams)
    remaining_threads = threads - base_threads_per_bam * num_bams
    
    for bam in bam_files:
        bam_reads = bam_lines[bam]
        proportional_threads = int((bam_reads / total_lines) * threads) if total_lines > 0 else 1
        chunks = max(base_threads_per_bam, proportional_threads)
        chunk_allocations.append((bam, chunks))
    
    total_allocated = sum(chunks for _, chunks in chunk_allocations)
    
    if total_allocated > threads:
        scale_factor = threads / total_allocated
        chunk_allocations = [(bam, max(1, int(chunks * scale_factor))) for bam, chunks in chunk_allocations]
        total_allocated = sum(chunks for _, chunks in chunk_allocations)
    
    remaining_threads = threads - total_allocated
    if remaining_threads > 0:
        for i in range(len(chunk_allocations)):
            if remaining_threads <= 0:
                break
            chunk_allocations[i] = (chunk_allocations[i][0], chunk_allocations[i][1] + 1)
            remaining_threads -= 1
    
    total_chunks = sum(chunks for _, chunks in chunk_allocations)
    return chunk_allocations

def process_bam_chunk(bam, fasta_file, temp_dir, out_dir, threads, chunk_idx, start_read, end_read):
    bam_basename = os.path.splitext(os.path.basename(bam))[0]
    out1_tmp = os.path.join(temp_dir, f'out1_{bam_basename}_chunk_{chunk_idx}.txt')
    out2_tmp = os.path.join(temp_dir, f'out2_{bam_basename}_chunk_{chunk_idx}.txt')

    id_count = defaultdict(int)
    ec = defaultdict(int)
    seq_cache = {}
    processed_lines = 0
    written_out1 = 0
    written_out2 = 0

    with open(out1_tmp, 'w') as out1_fh, pysam.AlignmentFile(bam, 'rb', threads=threads) as bf, pysam.FastaFile(fasta_file) as fa:
        for i, read in enumerate(bf):
            if i < start_read:
                continue
            if i >= end_read:
                break
            if read.is_unmapped:
                continue
            if read.query_sequence is None:
                continue
            processed_lines += 1

            astrand = '-' if read.is_reverse else '+'
            xs = ts = None
            error = None
            for tag, value in read.tags:
                if tag == 'XS':
                    xs = value
                elif tag == 'ts':
                    ts = value
                elif tag == 'NM':
                    error = value
            if ts and not xs and ts in ('+', '-'):
                xs = ('+' if ts == '-' else '-') if read.is_reverse else ts
            strand = xs or astrand

            pos = read.reference_start + 1
            cov = clip = gap = 0
            positions = []
            for op, length in read.cigartuples:
                if op in (4, 5):
                    clip += length
                elif op == 1:
                    cov += length
                elif op == 3:
                    end = pos + gap - 1
                    positions.extend([pos, end])
                    pos = end + length + 1
                    gap = 0
                elif op == 0:
                    cov += length
                    gap += length
                else:
                    gap += length
            end = pos + gap - 1
            positions.extend([pos, end])

            seqlen = len(read.query_sequence)
            # P0-A: NM tag (edit distance) is required for identity calculation.
            # Some aligners (e.g. minimap2 without -A) omit it. When missing,
            # warn and default to identity=1.0 rather than crashing on
            # None / cov TypeError downstream.
            if error is None:
                logger.warning(
                    f"Read {read.query_name}: NM tag missing, treating identity as 1.0"
                )
                identity = 1.0
            else:
                identity = 1 - (error / cov) if cov else 0
            coverage = (seqlen - clip) / seqlen if seqlen else 0

            id_count[read.query_name] += 1
            if len(positions) == 0:
                s1 = 'NA'
                e1 = 'NA'
                str_pos = 'NA'
            elif len(positions) == 1:
                s1 = positions[0]
                e1 = positions[0]
                str_pos = 'NA'
            else:
                s1 = positions[0]
                e1 = positions[-1]
                if len(positions) > 2:
                    str_pos = '-'.join(map(str, positions[1:-1]))
                else:
                    str_pos = 'NA'

            polya_len = next((t[1] for t in read.tags if t[0] == 'pt'), None)
            # Dorado pt:i sentinel semantics:
            #   >0  = estimated poly(A/T) tail length
            #   =0  = primer anchor found, length inestimable
            #   =-1 = primer anchor NOT found (treat as missing -> 0)
            polya_len = max(0, int(polya_len)) if polya_len is not None else 0

            out1_fh.write(f'{read.query_name}.m{id_count[read.query_name]}\t'
                         f'{read.reference_name}\t{strand}\t{s1}\t{e1}\t{str_pos}\t'
                         f'{identity}\t{coverage}\t{polya_len}\n')
            written_out1 += 1

            if str_pos != 'NA':
                key = f'{read.reference_name}\t{strand}\t{str_pos}'
                if key not in seq_cache:
                    b = str_pos.split('-')
                    ds = ''
                    # P1-2: pysam raises ValueError on out-of-range fetch
                    # (k1=1 → k1-3=-2; contig-end positions exceed length).
                    # Clamp to [0, contig_length] then wrap in try/except so
                    # one edge read does not kill the whole per-chunk worker
                    # and leave the temp file half-written.
                    contig_len = fa.get_reference_length(read.reference_name) if read.reference_name else 0
                    def _safe_fetch(start, end):
                        s = max(0, min(start, contig_len))
                        e = max(0, min(end, contig_len))
                        if e <= s:
                            return ''
                        try:
                            return fa.fetch(reference=read.reference_name, start=s, end=e)
                        except (ValueError, KeyError):
                            return ''
                    if strand == '+':
                        for i, k1 in enumerate(b):
                            k1 = int(k1)
                            seq = _safe_fetch(k1, k1+2) if i % 2 == 0 else _safe_fetch(k1-3, k1-1)
                            ds += f'{seq}-' if i % 2 == 0 else f'{seq},'
                    else:
                        for i, k1 in enumerate(reversed(b)):
                            k1 = int(k1)
                            seq = _safe_fetch(k1-3, k1-1) if i % 2 == 0 else _safe_fetch(k1, k1+2)
                            seq = str(Seq(seq).reverse_complement()) if seq else ''
                            ds += f'{seq}-' if i % 2 == 0 else f'{seq},'
                    ds = ds.rstrip(',')
                    seq_cache[key] = ds
                ec[key] += 1

    with open(out2_tmp, 'w') as out2_fh:
        for k in ec:
            out2_fh.write(f'{ec[k]}\t{k}\t{seq_cache[k]}\n')
            written_out2 += 1

    return out1_tmp, out2_tmp, bam


def get_chrom_offsets(bam):
    """Pre-scan BAM once to find chromosome boundaries in iteration order.

    Returns (boundaries, total_reads):
      boundaries: list of (chrom, first_idx, last_idx) where indices are
        0-based read positions in the full BAM iteration (chrom is None
        for the trailing unmapped-tail segment, if any).
      total_reads: total number of records seen.

    For position-sorted BAMs (typical RNA-seq), this gives stable
    chromosome boundaries that map cleanly to (start_read, end_read)
    chunk ranges.
    """
    print(f'[bam2ssc][T1] get_chrom_offsets enter: {bam}', flush=True)
    boundaries = []
    current_chrom = None
    current_start = 0
    total_reads = 0
    PROGRESS_EVERY = 1_000_000
    with pysam.AlignmentFile(bam, 'rb', threads=1) as bf:
        for i, read in enumerate(bf):
            total_reads = i + 1
            chrom = read.reference_name  # None for unmapped reads
            if chrom != current_chrom:
                if current_chrom is not None:
                    boundaries.append((current_chrom, current_start, i - 1))
                current_chrom = chrom
                current_start = i
            if total_reads % PROGRESS_EVERY == 0:
                print(f'[bam2ssc][T1] get_chrom_offsets progress: {bam} reads={total_reads}', flush=True)
        if current_chrom is not None or total_reads > 0:
            # Close the trailing segment (may have chrom=None for unmapped tail).
            boundaries.append((current_chrom, current_start, total_reads - 1))
    print(f'[bam2ssc][T1] get_chrom_offsets exit: {bam} total_reads={total_reads} chroms={len(boundaries)}', flush=True)
    return boundaries, total_reads


def chunk_to_chrom_jobs(start_read, end_read, chrom_offsets):
    """Convert (start_read, end_read) read-range to per-chromosome fetch jobs.

    Returns list of (chrom, start_in_chrom, end_in_chrom) where
    start_in_chrom / end_in_chrom are half-open read offsets within
    the chromosome's bf.fetch() iteration (0-based). For a position-
    sorted BAM, processing these jobs in order produces the same
    per-chunk read sequence as the legacy O(N) skip approach.
    """
    jobs = []
    for chrom, first_idx, last_idx in chrom_offsets:
        if end_read <= first_idx:
            break
        if start_read > last_idx:
            continue
        # Map global read offsets to per-chromosome offsets (0-based, half-open).
        chrom_start = max(0, start_read - first_idx)
        chrom_end = min(last_idx + 1 - first_idx, end_read - first_idx)
        if chrom_end > chrom_start:
            jobs.append((chrom, chrom_start, chrom_end))
    return jobs


def process_bam_chunk_fast(bam, fasta_file, temp_dir, out_dir, threads, chunk_idx, chrom_jobs):
    """P1: O(work) per-chunk worker using .bai-based bf.fetch() iteration.

    Replaces the legacy O(N) skip-with-offset reader in process_bam_chunk.
    Requires that the .bai index exists for the BAM (caller must verify).

    chrom_jobs: list of (chrom, start_in_chrom, end_in_chrom) tuples
      in BAM iteration order, produced by chunk_to_chrom_jobs().

    Reads within each chromosome job are iterated in bf.fetch() order,
    which matches the legacy O(N) iteration order for position-sorted
    BAMs (the expected input). This preserves byte-identity of the
    per-chunk output for the typical RNA-seq use case.

    Unmapped reads (chrom is None) are skipped here, matching the legacy
    `if read.is_unmapped: continue` behavior in process_bam_chunk.
    """
    bam_basename = os.path.splitext(os.path.basename(bam))[0]
    out1_tmp = os.path.join(temp_dir, f'out1_{bam_basename}_chunk_{chunk_idx}.txt')
    out2_tmp = os.path.join(temp_dir, f'out2_{bam_basename}_chunk_{chunk_idx}.txt')

    id_count = defaultdict(int)
    ec = defaultdict(int)
    seq_cache = {}
    processed_lines = 0
    written_out1 = 0
    written_out2 = 0

    with open(out1_tmp, 'w') as out1_fh, pysam.AlignmentFile(bam, 'rb', threads=threads) as bf, pysam.FastaFile(fasta_file) as fa:
        for chrom, chrom_start, chrom_end in chrom_jobs:
            if chrom is None:
                # Unmapped tail -- bf.fetch() cannot return these. Skip,
                # matching the legacy `is_unmapped: continue` behavior.
                continue
            try:
                chrom_iter = bf.fetch(chrom)
            except (ValueError, KeyError):
                # Chromosome listed in offsets but absent from .bai -- skip.
                logger.warning(
                    f"Chunk {chunk_idx}: cannot fetch chrom {chrom!r} from {bam}, skipping"
                )
                continue
            for j, read in enumerate(chrom_iter):
                if j < chrom_start:
                    continue
                if j >= chrom_end:
                    break
                if read.is_unmapped:
                    continue
                if read.query_sequence is None:
                    continue
                processed_lines += 1

                astrand = '-' if read.is_reverse else '+'
                xs = ts = None
                error = None
                for tag, value in read.tags:
                    if tag == 'XS':
                        xs = value
                    elif tag == 'ts':
                        ts = value
                    elif tag == 'NM':
                        error = value
                if ts and not xs and ts in ('+', '-'):
                    xs = ('+' if ts == '-' else '-') if read.is_reverse else ts
                strand = xs or astrand

                pos = read.reference_start + 1
                cov = clip = gap = 0
                positions = []
                for op, length in read.cigartuples:
                    if op in (4, 5):
                        clip += length
                    elif op == 1:
                        cov += length
                    elif op == 3:
                        end = pos + gap - 1
                        positions.extend([pos, end])
                        pos = end + length + 1
                        gap = 0
                    elif op == 0:
                        cov += length
                        gap += length
                    else:
                        gap += length
                end = pos + gap - 1
                positions.extend([pos, end])

                seqlen = len(read.query_sequence)
                if error is None:
                    logger.warning(
                        f"Read {read.query_name}: NM tag missing, treating identity as 1.0"
                    )
                    identity = 1.0
                else:
                    identity = 1 - (error / cov) if cov else 0
                coverage = (seqlen - clip) / seqlen if seqlen else 0

                id_count[read.query_name] += 1
                if len(positions) == 0:
                    s1 = 'NA'
                    e1 = 'NA'
                    str_pos = 'NA'
                elif len(positions) == 1:
                    s1 = positions[0]
                    e1 = positions[0]
                    str_pos = 'NA'
                else:
                    s1 = positions[0]
                    e1 = positions[-1]
                    if len(positions) > 2:
                        str_pos = '-'.join(map(str, positions[1:-1]))
                    else:
                        str_pos = 'NA'

                polya_len = next((t[1] for t in read.tags if t[0] == 'pt'), None)
                polya_len = max(0, int(polya_len)) if polya_len is not None else 0

                out1_fh.write(f'{read.query_name}.m{id_count[read.query_name]}\t'
                             f'{read.reference_name}\t{strand}\t{s1}\t{e1}\t{str_pos}\t'
                             f'{identity}\t{coverage}\t{polya_len}\n')
                written_out1 += 1

                if str_pos != 'NA':
                    key = f'{read.reference_name}\t{strand}\t{str_pos}'
                    if key not in seq_cache:
                        b = str_pos.split('-')
                        ds = ''
                        contig_len = fa.get_reference_length(read.reference_name) if read.reference_name else 0
                        def _safe_fetch(start, end):
                            s = max(0, min(start, contig_len))
                            e = max(0, min(end, contig_len))
                            if e <= s:
                                return ''
                            try:
                                return fa.fetch(reference=read.reference_name, start=s, end=e)
                            except (ValueError, KeyError):
                                return ''
                        if strand == '+':
                            for i, k1 in enumerate(b):
                                k1 = int(k1)
                                seq = _safe_fetch(k1, k1+2) if i % 2 == 0 else _safe_fetch(k1-3, k1-1)
                                ds += f'{seq}-' if i % 2 == 0 else f'{seq},'
                        else:
                            for i, k1 in enumerate(reversed(b)):
                                k1 = int(k1)
                                seq = _safe_fetch(k1-3, k1-1) if i % 2 == 0 else _safe_fetch(k1, k1+2)
                                seq = str(Seq(seq).reverse_complement()) if seq else ''
                                ds += f'{seq}-' if i % 2 == 0 else f'{seq},'
                        ds = ds.rstrip(',')
                        seq_cache[key] = ds
                    ec[key] += 1

    with open(out2_tmp, 'w') as out2_fh:
        for k in ec:
            out2_fh.write(f'{ec[k]}\t{k}\t{seq_cache[k]}\n')
            written_out2 += 1

    return out1_tmp, out2_tmp, bam

def merge_single_bam(bam, files, out_dir):
    bam_basename = os.path.splitext(os.path.basename(bam))[0]
    out1 = os.path.join(out_dir, f'{bam_basename}_flnc.ssc')
    out2 = os.path.join(out_dir, f'{bam_basename}_ssc.count')

    out1_count = 0
    with open(out1, 'w') as out1_fh:
        for out1_tmp, _ in files:
            with open(out1_tmp, 'r') as in_fh:
                for line in in_fh:
                    if not line.endswith('\n'):
                        line += '\n'
                    out1_fh.write(line)
                    out1_count += 1

    global_ec = defaultdict(int)
    global_seq = {}
    for _, out2_tmp in files:
        with open(out2_tmp, 'r') as in_fh:
            for line in in_fh:
                count, ref, strand, pos_str, ds = line.strip().split('\t', 4)
                key = f'{ref}\t{strand}\t{pos_str}'
                global_ec[key] += int(count)
                global_seq[key] = ds

    with open(out2, 'w') as out2_fh:
        out2_count = 0
        for k in sorted(global_ec):
            out2_fh.write(f'{global_ec[k]}\t{k}\t{global_seq[k]}\n')
            out2_count += 1

    return bam, out1_count, out2_count

def merge_results(chunk_results, fasta_file, out_dir, threads):
    bam_groups = defaultdict(list)
    for out1_tmp, out2_tmp, bam in chunk_results:
        bam_groups[bam].append((out1_tmp, out2_tmp))

    print(f'[bam2ssc][T1] Pool#2 merge_results enter: workers={threads} groups={len(bam_groups)}', flush=True)
    with mp.Pool(processes=threads) as pool:
        results = pool.starmap(partial(merge_single_bam, out_dir=out_dir), bam_groups.items())
    print(f'[bam2ssc][T1] Pool#2 merge_results exit', flush=True)

def main():
    args = parse_args()
    print(f'[bam2ssc][T1] main() entered: bams={len(args.bam)} threads={args.threads}', flush=True)
    os.makedirs(args.output, exist_ok=True)
    bam_lines, total_lines = get_bam_read_counts(args.bam, args.threads)
    chunk_allocations = allocate_chunks(args.bam, bam_lines, total_lines, args.threads)

    # P1 perf: pre-scan BAMs with a .bai index to derive chromosome offsets
    # so per-chunk workers can iterate via bf.fetch() (O(work)) instead of
    # O(N) skip-with-offset (legacy). Falls back to legacy read-count
    # chunking if .bai is unavailable -- preserves byte-identity of the
    # legacy path.
    bam_has_index = {bam: os.path.exists(bam + '.bai') for bam in args.bam}
    bam_chrom_offsets = {}
    print(f'[bam2ssc][T1] main pre-scan loop start: bams_with_index={sum(bam_has_index.values())}/{len(args.bam)}', flush=True)
    for idx, bam in enumerate(args.bam):
        if bam_has_index[bam]:
            print(f'[bam2ssc][T1] main pre-scan {idx+1}/{len(args.bam)} start: {bam}', flush=True)
            try:
                bam_chrom_offsets[bam] = get_chrom_offsets(bam)
            except Exception as e:
                logger.warning(
                    f"Pre-scan failed for {bam}: {e}; falling back to legacy O(N) skip"
                )
                bam_has_index[bam] = False
            print(f'[bam2ssc][T1] main pre-scan {idx+1}/{len(args.bam)} end: {bam}', flush=True)

    use_fast_path = all(bam_has_index.values()) and bam_chrom_offsets

    tasks = []
    worker = process_bam_chunk_fast if use_fast_path else process_bam_chunk
    for bam, chunk_count in chunk_allocations:
        total_lines = bam_lines[bam]
        lines_per_chunk = total_lines // chunk_count + 1 if chunk_count > 1 else total_lines
        for i in range(chunk_count):
            start_read = i * lines_per_chunk
            end_read = min((i + 1) * lines_per_chunk, total_lines)
            if use_fast_path:
                chrom_offsets, _ = bam_chrom_offsets[bam]
                chrom_jobs = chunk_to_chrom_jobs(start_read, end_read, chrom_offsets)
                tasks.append((bam, args.reference, tempfile.gettempdir(), args.output,
                              args.threads, i, chrom_jobs))
            else:
                tasks.append((bam, args.reference, tempfile.gettempdir(), args.output,
                              args.threads, i, start_read, end_read))

    if use_fast_path:
        logger.info(
            f"P1 perf: using .bai-based chromosome chunking across {len(tasks)} chunks"
        )

    print(f'[bam2ssc][T1] Pool#3 main enter: workers={args.threads} tasks={len(tasks)}', flush=True)
    with mp.Pool(processes=args.threads) as pool:
        chunk_results = pool.starmap(worker, tasks)
    print(f'[bam2ssc][T1] Pool#3 main exit', flush=True)

    merge_results(chunk_results, args.reference, args.output, args.threads)

if __name__ == '__main__':
    main()
