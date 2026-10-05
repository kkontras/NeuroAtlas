"""Shared base class for univariate time-series foundation models.

All TS foundation models (Chronos, MOMENT, Time-MoE, ...) are univariate --
they accept ``(B, T)`` inputs.  EEG datasets provide multi-channel signals
``(B, C, T)``.  This base handles the per-channel embedding with mean-pooling
strategy: flatten ``(B, C, T)`` -> ``(B*C, T)``, embed in one (possibly
chunked) forward pass, reshape to ``(B, C, D)`` and mean-pool to ``(B, D)``.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch

from ._preproc import _ESAT_DATASETS
from .base import BenchmarkBackbone
from neuroatlas.benchmarking_helpers import CheckpointSpec
from neuroatlas._paths import models_dir

# HuggingFace cache inside the repo, not ~/.cache
_HF_CACHE = str(models_dir("foundation", "huggingface_cache"))


class UnivariateTimeSeriesBackbone(BenchmarkBackbone):
    """Base class for univariate TS foundation models applied to multi-channel EEG.

    Subclasses implement ``_embed_univariate(signal)`` for a ``(N, T)`` tensor
    and return an ``(N, D)`` numpy array.  This class handles channel flattening,
    chunked batching, and mean-pooling across channels.
    """

    _max_embed_batch: int = 128

    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        self.device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if mem_gb >= 80:
                self._max_embed_batch = 512

    # ── Public API (called by the benchmark runner) ──────────────────────

    def extract_embeddings(self, batch) -> np.ndarray:
        self.validate_batch(batch)
        x = self.require_signal(batch, "eeg", "full_signal")
        x = x.to(dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, T) -> (B, 1, T)
        B, C, T = x.shape

        # Strip all-zero channels (ESAT-only): dataio zero-pads missing
        # channel slots; pooling over them dilutes the embedding.
        meta = batch.get("meta") or [{}]
        dataset = meta[0].get("dataset", "") if meta else ""
        if dataset in _ESAT_DATASETS:
            nonzero = x.abs().sum(dim=(0, 2)) > 0
            if not nonzero.all() and nonzero.any():
                x = x[:, nonzero, :]
                B, C, T = x.shape

        # Flatten channels into the batch dimension for one efficient pass
        x_flat = x.reshape(B * C, T)  # (B*C, T)
        emb_flat = self._embed_univariate_batched(x_flat)  # (B*C, D)

        # Reshape back and mean-pool across channels
        D = emb_flat.shape[-1]
        emb_per_channel = emb_flat.reshape(B, C, D)  # (B, C, D)
        pooled = emb_per_channel.mean(axis=1)  # (B, D)
        return pooled.astype(np.float32)

    def forward_features(self, batch) -> torch.Tensor:
        """Grad-enabled twin of :meth:`extract_embeddings`.

        Same flatten → embed → mean-pool-over-channels trunk, but routed through
        :meth:`_forward_univariate` so gradients reach the encoder (and any
        the backbone).

        Chunking is deliberately **not** applied here: under autograd every chunk
        would stay in the graph anyway, so it buys no memory and only obscures the
        peak. A ``(B, C, T)`` batch becomes ``B*C`` sequences in one pass — size
        the batch accordingly (a 19-channel window at ``batch_size=16`` is 304
        sequences).
        """
        self.validate_batch(batch)
        x = self.require_signal(batch, "eeg", "full_signal")
        x = x.to(dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, T) -> (B, 1, T)
        B, C, T = x.shape

        emb_flat = self._forward_univariate(x.reshape(B * C, T))  # (B*C, D)
        D = emb_flat.shape[-1]
        return emb_flat.reshape(B, C, D).mean(dim=1)  # (B, D)

    # ── Subclass hooks ───────────────────────────────────────────────────

    @abstractmethod
    def _embed_univariate(self, signal: torch.Tensor) -> np.ndarray:
        """Embed a batch of univariate signals.

        Parameters
        ----------
        signal : torch.Tensor
            Shape ``(N, T)`` float32 tensor (may be on CPU).

        Returns
        -------
        np.ndarray of shape ``(N, D)`` float32.
        """

    def _forward_univariate(self, signal: torch.Tensor) -> torch.Tensor:
        """Grad-enabled twin of :meth:`_embed_univariate`.

        Identical maths, but returns an ``(N, D)`` tensor still attached to the
        autograd graph instead of a detached numpy array. Subclasses that support
        fine-tuning implement this; the default refuses rather than silently
        returning detached features.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _forward_univariate(); it "
            f"supports frozen-embedding probing only."
        )

    # ── Internal helpers ─────────────────────────────────────────────────

    def _embed_univariate_batched(self, signal: torch.Tensor) -> np.ndarray:
        """Run ``_embed_univariate`` in chunks to avoid OOM."""
        N = signal.shape[0]
        if N <= self._max_embed_batch:
            return self._embed_univariate(signal)
        chunks = []
        for start in range(0, N, self._max_embed_batch):
            chunk = signal[start : start + self._max_embed_batch]
            chunks.append(self._embed_univariate(chunk))
        return np.concatenate(chunks, axis=0)

    # ── Per-patch subclass hook ─────────────────────────────────────

    def _embed_univariate_perpatch(self, signal: torch.Tensor) -> np.ndarray:
        """Embed a batch of univariate signals returning per-token representations.

        Parameters
        ----------
        signal : torch.Tensor
            Shape ``(N, T)`` float32 tensor (may be on CPU).

        Returns
        -------
        np.ndarray of shape ``(N, n_tokens, D)`` float32 (valid tokens only).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support per-patch embedding extraction."
        )

    def _embed_univariate_perpatch_batched(self, signal: torch.Tensor) -> np.ndarray:
        """Run ``_embed_univariate_perpatch`` in chunks to avoid OOM."""
        N = signal.shape[0]
        if N <= self._max_embed_batch:
            return self._embed_univariate_perpatch(signal)
        chunks = []
        for start in range(0, N, self._max_embed_batch):
            chunk = signal[start : start + self._max_embed_batch]
            chunks.append(self._embed_univariate_perpatch(chunk))
        return np.concatenate(chunks, axis=0)

    def extract_embeddings_perpatch(self, batch) -> np.ndarray:
        self.validate_batch(batch)
        x = self.require_signal(batch, "eeg", "full_signal")
        x = x.to(dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, C, T = x.shape

        meta = batch.get("meta") or [{}]
        dataset = meta[0].get("dataset", "") if meta else ""
        if dataset in _ESAT_DATASETS:
            nonzero = x.abs().sum(dim=(0, 2)) > 0
            if not nonzero.all() and nonzero.any():
                x = x[:, nonzero, :]
                B, C, T = x.shape

        x_flat = x.reshape(B * C, T)
        emb_flat = self._embed_univariate_perpatch_batched(x_flat)  # (B*C, n_tok, D)
        n_tok, D = emb_flat.shape[1], emb_flat.shape[2]
        emb_per_channel = emb_flat.reshape(B, C * n_tok, D)  # (B, C*n_tok, D)
        return emb_per_channel.astype(np.float32)

    @staticmethod
    def _instance_normalize(
        x: torch.Tensor, eps: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample z-score normalisation on the last axis."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + eps
        return (x - mean) / std, mean, std


def _install_hint(family: str) -> str:
    """The install command that keeps the rest of the stack intact (neuroatlas.models.PACKAGES)."""
    from neuroatlas.models import PACKAGES

    return PACKAGES[family][2]
