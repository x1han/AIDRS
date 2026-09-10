"""AIDRS ResourceGuard: environment-introspective OOM defense.

Phase 1 of the v1.0.7 self-throttling redesign (see
workspace/docs/aidrs_v1_1_spec_frozen_2026-09-08.md). Detects:
  - SGE/Slurm slot counts (NSLOTS / SLURM_CPUS_PER_TASK)
  - cgroup v1 + v2 memory limits
  - SGE_H_VMEM env var (set on some SGE configs)

Returns safe defaults that match the pilot-validated constants from
job 1177481 (P0_Gonad_Male, 2.7 GB BAM, 8 worker pool, maxvmem
27.855 GB → ~3.0 GB per worker including model weights + per-call
cache).
"""
import os
import math
import logging

logger = logging.getLogger("aidrs")


class ResourceGuard:
    """AIDRS resource introspection + safe worker budget.

    All methods are static / classmethod so callers don't need to
    instantiate. Pure introspection — no side effects.
    """

    # Empirically derived from job 1177481 qacct:
    #   8 workers × ~3.0 GB each = ~24 GB + ~4 GB overhead = 27.855 GB peak
    # Each TranslationAI worker holds an independent 5-model ensemble
    # (~2.5 GB weights) plus per-shard forward cache (~0.5 GB).
    RAM_PER_WORKER_GB = 3.0

    # Conservative non-model headroom (BAM parsing, SSC graph, DBSCAN
    # coords, OS page cache, PyTorch TF runtime reservations).
    BASE_RESERVE_GB_SMALL = 4.5   # for BAMs ≤ 8 GB
    BASE_RESERVE_GB_LARGE = 6.5   # for BAMs > 8 GB

    @staticmethod
    def get_effective_cpu_threads(user_requested_threads=None):
        """Resolve effective CPU thread count from explicit > scheduler > host.

        Priority:
          1. User-specified --threads (explicit override wins)
          2. SGE $NSLOTS (set by smp parallel environment allocation)
          3. Slurm $SLURM_CPUS_PER_TASK
          4. Cap of 8 on physical cpu_count (avoid runaway on 64+ core nodes)

        Returns:
            int ≥ 1
        """
        if user_requested_threads and user_requested_threads > 0:
            return int(user_requested_threads)

        sge_slots = os.environ.get("NSLOTS")
        if sge_slots and sge_slots.isdigit() and int(sge_slots) > 0:
            slots = int(sge_slots)
            logger.info(
                f"[RESOURCE] SGE $NSLOTS={slots}, binding --threads={slots}"
            )
            return slots

        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus and slurm_cpus.isdigit() and int(slurm_cpus) > 0:
            slots = int(slurm_cpus)
            logger.info(
                f"[RESOURCE] Slurm $SLURM_CPUS_PER_TASK={slots}, "
                f"binding --threads={slots}"
            )
            return slots

        # Bare-metal fallback: cap to 8 even if the host has 64+ cores.
        # AIDRS pilot-validated max safe threads = 8 (worker pool);
        # more threads → more model copies → OOM.
        avail = os.cpu_count() or 4
        safe = min(8, avail)
        logger.info(
            f"[RESOURCE] No scheduler hint; default --threads={safe} "
            f"(host has {avail} physical cores, capped at 8)"
        )
        return safe

    @staticmethod
    def get_memory_limit_gb():
        """Detect the REAL per-job memory ceiling (not host total).

        Priority:
          1. cgroup v1 /sys/fs/cgroup/memory/memory.limit_in_bytes
          2. cgroup v2 /sys/fs/cgroup/memory.max
          3. SGE_H_VMEM env var (some SGE configs)
          4. 32.0 GB default (matches cluster's standard
             `h_vmem=4G × smp 8` allocation)

        Returns:
            float GB, ≥ 1.0
        """
        # cgroup v1 (Docker, legacy HPC containers)
        cg_v1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        if os.path.exists(cg_v1):
            try:
                with open(cg_v1, "r") as f:
                    val = int(f.read().strip())
                # 2**63-1 ≈ 9.2 EB → treat as "no limit"
                if val < (1 << 60):
                    gb = val / (1024 ** 3)
                    logger.info(f"[RESOURCE] cgroup v1 limit = {gb:.1f} GB")
                    return gb
            except (OSError, ValueError):
                pass

        # cgroup v2 (modern K8s / systemd-managed)
        cg_v2 = "/sys/fs/cgroup/memory.max"
        if os.path.exists(cg_v2):
            try:
                with open(cg_v2, "r") as f:
                    val_str = f.read().strip()
                # "max" means no limit
                if val_str != "max":
                    gb = int(val_str) / (1024 ** 3)
                    logger.info(f"[RESOURCE] cgroup v2 limit = {gb:.1f} GB")
                    return gb
            except (OSError, ValueError):
                pass

        # Some SGE configs inject SGE_H_VMEM (bytes) per slot.
        sge_vmem = os.environ.get("SGE_H_VMEM")
        if sge_vmem:
            try:
                gb = float(sge_vmem) / (1024 ** 3)
                logger.info(f"[RESOURCE] SGE_H_VMEM env = {gb:.1f} GB")
                return gb
            except ValueError:
                pass

        # Default: align with cluster's standard h_vmem=4G × smp 8 = 32 GB.
        # Conservative on bare metal; explicit on HPC.
        logger.info("[RESOURCE] No cgroup / scheduler hint; default 32.0 GB")
        return 32.0

    @classmethod
    def get_safe_translationai_workers(cls, total_threads, bam_file_size_gb=4.0):
        """Compute safe TranslationAI worker count from real memory budget.

        Each TranslationAI worker holds 5 model copies (~2.5 GB) plus
        per-shard forward cache. From pilot 1177481:
          8 workers × 3.0 GB = 27.855 GB observed peak on 32 GB allocation.

        Args:
            total_threads: requested thread/worker count (caller's arg).
            bam_file_size_gb: input BAM size; >8 GB triggers larger headroom.

        Returns:
            int ≥ 1 (safe worker count)
        """
        mem_limit = cls.get_memory_limit_gb()
        reserve = (cls.BASE_RESERVE_GB_LARGE
                   if bam_file_size_gb > 8.0
                   else cls.BASE_RESERVE_GB_SMALL)
        usable = max(3.0, mem_limit - reserve)
        max_by_mem = math.floor(usable / cls.RAM_PER_WORKER_GB)

        safe = max(1, min(total_threads, max_by_mem))

        logger.info(
            f"[RESOURCE SHIELD] TranslationAI worker budget:\n"
            f"  mem_limit={mem_limit:.1f} GB | reserve={reserve:.1f} GB | "
            f"usable={usable:.1f} GB | per_worker={cls.RAM_PER_WORKER_GB} GB\n"
            f"  bam_size={bam_file_size_gb:.1f} GB → "
            f"max_by_mem={max_by_mem} | requested={total_threads} → "
            f"safe_workers={safe}"
        )
        return safe