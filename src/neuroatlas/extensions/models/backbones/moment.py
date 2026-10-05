"""MOMENT backbone for EEGBenchmarks.

Uses the native embedding head from ``momentfm.MOMENTPipeline``.
Input is padded/truncated so its length is a multiple of the patch size (8).

Embedding logic ported from TimeSeriesPhysics/benchmarks/foundation/wrappers/moment_wrapper.py.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from neuroatlas.benchmarking_helpers import CheckpointSpec

from .ts_foundation_base import UnivariateTimeSeriesBackbone, _HF_CACHE, _install_hint

# MOMENT supports up to 8192 time steps.
_MAX_SEQ_LEN = 8192
_PATCH_SIZE = 8


class MomentBackbone(UnivariateTimeSeriesBackbone):
    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        try:
            from momentfm import MOMENTPipeline
        except ImportError as exc:
            raise ImportError(
                "momentfm is required for MOMENT. Install: " + _install_hint("moment")
            ) from exc

        hf_id = spec.checkpoint_path or "AutonLab/MOMENT-1-base"
        self._embed_model = MOMENTPipeline.from_pretrained(
            hf_id,
            model_kwargs={"task_name": "embedding"},
            cache_dir=_HF_CACHE,
        )
        self._embed_model.init()
        self._embed_model.to(self.device).eval()

    # ── Univariate embedding ─────────────────────────────────────────────

    def _encode(self, signal: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` -> ``(B, d_model)`` pooled embeddings, grad-preserving."""
        B, T = signal.shape
        # Round up to next multiple of patch size, capped at max
        target_len = min(((T + _PATCH_SIZE - 1) // _PATCH_SIZE) * _PATCH_SIZE, _MAX_SEQ_LEN)
        x = self._prep_input(signal, target_len)  # (B, 1, target_len)
        mask = torch.ones(B, target_len, dtype=torch.float32, device=self.device)
        emb = self._embed_model(x_enc=x, input_mask=mask).embeddings
        return emb.mean(dim=1) if emb.dim() == 3 else emb

    @torch.no_grad()
    def _embed_univariate(self, signal: torch.Tensor) -> np.ndarray:
        return self._encode(signal).cpu().numpy().astype(np.float32)

    def _forward_univariate(self, signal: torch.Tensor) -> torch.Tensor:
        return self._encode(signal)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _prep_input(self, signal: torch.Tensor, target_len: int) -> torch.Tensor:
        """``(B, T)`` tensor -> ``(B, 1, target_len)`` on device, zero-padded or right-truncated."""
        B, T = signal.shape
        arr = signal.cpu().numpy().astype(np.float32)
        if T == target_len:
            out = arr
        elif T < target_len:
            out = np.concatenate(
                [arr, np.zeros((B, target_len - T), dtype=np.float32)], axis=-1
            )
        else:
            out = arr[:, -target_len:]
        return torch.from_numpy(out).unsqueeze(1).to(self.device)  # (B, 1, target_len)

    @torch.no_grad()
    def _embed_univariate_perpatch(self, signal: torch.Tensor) -> np.ndarray:
        B, T = signal.shape
        target_len = min(((T + _PATCH_SIZE - 1) // _PATCH_SIZE) * _PATCH_SIZE, _MAX_SEQ_LEN)
        n_valid = math.ceil(T / _PATCH_SIZE)
        x = self._prep_input(signal, target_len)  # (B, 1, target_len)
        mask = torch.ones(B, target_len, dtype=torch.float32, device=self.device)
        out = self._embed_model(x_enc=x, input_mask=mask)
        emb = out.embeddings.cpu().numpy().astype(np.float32)
        if emb.ndim == 2:
            emb = emb[:, np.newaxis, :]
        emb = emb[:, :n_valid, :]  # drop padded patches
        return emb

    def metadata(self):
        return {
            **super().metadata(),
            "device": self.device,
            "hf_id": self.spec.checkpoint_path,
        }
