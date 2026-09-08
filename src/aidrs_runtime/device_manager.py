"""Device manager for opt-in GPU acceleration of Puffin and TranslationAI.

CPU is the default and the byte-identical baseline (`a8469106`). GPU is opt-in
via the `AIDRS_PUFFIN_USE_GPU` and `AIDRS_TRANSAI_USE_GPU` environment
variables, both parsed to `1` / `true` / `yes` (case-insensitive). When unset
or any other value, the corresponding runner stays on CPU and the baseline
SHA is preserved.

Singleton: `device_manager` is the single import point. Callers read
properties only after `initialize()` has been invoked once at process start;
the runners do that in their constructors.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("AIDRS")


def _truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


class DeviceManager:
    """Process-wide GPU/CPU dispatch table for the inference runners."""

    def __init__(self) -> None:
        self._initialized = False
        self._puffin_use_gpu: bool = False
        self._transai_use_gpu: bool = False
        self._puffin_device_str: str = "cpu"
        self._tf_imported: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Detect CUDA and resolve env-driven opt-in flags.

        Safe to call multiple times: subsequent calls are no-ops once
        initialized.
        """
        if self._initialized:
            return

        import torch  # local import: keep torch optional for CPU-only boxes

        cuda_available = bool(torch.cuda.is_available())
        cuda_count = torch.cuda.device_count() if cuda_available else 0

        env_puffin = os.environ.get("AIDRS_PUFFIN_USE_GPU")
        env_transai = os.environ.get("AIDRS_TRANSAI_USE_GPU")

        self._puffin_use_gpu = _truthy(env_puffin) and cuda_available
        self._transai_use_gpu = _truthy(env_transai) and cuda_available

        if self._puffin_use_gpu:
            self._puffin_device_str = "cuda:0"
        if self._transai_use_gpu:
            # Lazy import: do not force TensorFlow on boxes that never use it.
            try:
                import tensorflow as tf  # noqa: F401

                self._tf_imported = True
            except Exception as exc:  # pragma: no cover - depends on env
                logger.warning(
                    "AIDRS_TRANSAI_USE_GPU=1 but TensorFlow import failed: %s; "
                    "TranslationAI will fall back to CPU.",
                    exc,
                )
                self._transai_use_gpu = False

        logger.info(
            "DeviceManager initialized: cuda_available=%s (%d devices), "
            "puffin_use_gpu=%s, transai_use_gpu=%s",
            cuda_available,
            cuda_count,
            self._puffin_use_gpu,
            self._transai_use_gpu,
        )
        self._initialized = True

    # ------------------------------------------------------------------
    # Puffin dispatch
    # ------------------------------------------------------------------
    @property
    def puffin_use_gpu(self) -> bool:
        self.initialize()
        return self._puffin_use_gpu

    @property
    def puffin_device(self) -> str:
        self.initialize()
        return self._puffin_device_str

    # ------------------------------------------------------------------
    # TranslationAI dispatch
    # ------------------------------------------------------------------
    @property
    def transai_use_gpu(self) -> bool:
        self.initialize()
        return self._transai_use_gpu

    # ------------------------------------------------------------------
    # Cleanup hooks
    # ------------------------------------------------------------------
    def cleanup_torch_vram(self) -> None:
        """Release CUDA cache after Puffin completes.

        Called from `PuffinRunner.predict_fasta` immediately before returning,
        so a downstream TranslationAI pass can claim the same device without
        OOM. No-op on CPU.
        """
        if not self._puffin_use_gpu:
            return
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("cleanup_torch_vram no-op: %s", exc)

    def cleanup_tf_vram(self) -> None:
        """Release TensorFlow GPU memory after TranslationAI completes.

        No-op on CPU or when TensorFlow was never imported.
        """
        if not self._transai_use_gpu or not self._tf_imported:
            return
        try:
            from tensorflow.keras import backend as K

            K.clear_session()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("cleanup_tf_vram no-op: %s", exc)


device_manager = DeviceManager()
