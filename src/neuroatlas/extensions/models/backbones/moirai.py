"""MOIRAI backbone for EEGBenchmarks.

MOIRAI has no native embedding head — only forecasting.  We extract
embeddings by hooking the transformer encoder's final norm layer and
mean-pooling the hidden states over the patch (sequence) dimension.

The input is prepared manually: the signal is patched, scaled, and
projected through MOIRAI's ``in_proj`` before being fed to the encoder.
No GluonTS conversion is needed.

Ported from TimeSeriesPhysics/benchmarks/foundation/wrappers/moirai_wrapper.py.
"""

from __future__ import annotations

from typing import List

import math

import numpy as np
import torch

from neuroatlas.benchmarking_helpers import CheckpointSpec

from .ts_foundation_base import UnivariateTimeSeriesBackbone, _HF_CACHE

# Default patch size per model size (from MOIRAI paper).
_SIZE_PATCH = {"small": 32, "base": 32, "large": 64}


class MoiraiBackbone(UnivariateTimeSeriesBackbone):
    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        try:
            from uni2ts.model.moirai import MoiraiModule
        except ImportError as exc:
            raise ImportError(
                "uni2ts is required for MOIRAI. Install: pip install uni2ts"
            ) from exc

        hf_id = spec.checkpoint_path or "Salesforce/moirai-1.1-R-small"
        self._module = MoiraiModule.from_pretrained(hf_id, cache_dir=_HF_CACHE)
        self._module.eval()
        self._module.to(self.device)

        # Infer patch_size from variant name
        variant = spec.variant  # "small", "base", "large"
        self._patch_size = _SIZE_PATCH.get(variant, 32)

    # ── Univariate embedding via encoder hook ────────────────────────────

    def _build_module_inputs(self, signal: torch.Tensor) -> dict:
        """``(B, T)`` signal -> the kwargs ``MoiraiModule.forward`` expects."""
        B, T = signal.shape

        # Pad T to multiple of patch_size
        ps = self._patch_size
        remainder = T % ps
        if remainder != 0:
            pad_len = ps - remainder
            signal = torch.nn.functional.pad(signal, (0, pad_len))
            T = signal.shape[1]
        num_patches = T // ps

        # Build input tensors on device
        dev = self.device
        max_patch = self._module.in_proj.weight.shape[-1]  # e.g. 128

        # target: (B, num_patches, max_patch) — zero-padded from patch_size
        patches = signal.reshape(B, num_patches, ps).to(dev, dtype=torch.float32)
        target = torch.zeros(B, num_patches, max_patch, device=dev, dtype=torch.float32)
        target[:, :, :ps] = patches

        # observed_mask: True for real values, False for padding
        observed_mask = torch.zeros(B, num_patches, max_patch, device=dev, dtype=torch.bool)
        observed_mask[:, :, :ps] = True

        return {
            "target": target,
            "observed_mask": observed_mask,
            # IDs: single sample, sequential time, univariate
            "sample_id": torch.zeros(B, num_patches, dtype=torch.long, device=dev),
            "time_id": torch.arange(num_patches, device=dev).unsqueeze(0).expand(B, -1),
            "variate_id": torch.zeros(B, num_patches, dtype=torch.long, device=dev),
            # No prediction window — pure encoding
            "prediction_mask": torch.zeros(B, num_patches, dtype=torch.bool, device=dev),
            # Constant patch size
            "patch_size": torch.full((B, num_patches), ps, dtype=torch.long, device=dev),
        }

    def _encode(self, signal: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` -> ``(B, d_model)`` pooled encoder hidden states.

        MOIRAI exposes no encoder-only entry point, so the encoder output is
        captured with a forward hook while the full module runs. The captured
        tensor is *not* detached, so this is usable for both frozen extraction
        and fine-tuning; the caller decides via ``no_grad``.
        """
        captured: List[torch.Tensor] = []

        def _hook(_module, _inp, out):
            captured.append(out)

        handle = self._module.encoder.register_forward_hook(_hook)
        try:
            self._module(**self._build_module_inputs(signal))
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError(
                "MOIRAI encoder forward hook captured nothing — the module layout "
                "changed and _encode() needs updating."
            )
        # captured[0]: (B, num_patches, d_model)
        hidden = captured[0]
        return hidden.mean(dim=1) if hidden.dim() == 3 else hidden

    @torch.no_grad()
    def _embed_univariate(self, signal: torch.Tensor) -> np.ndarray:
        return self._encode(signal).cpu().numpy().astype(np.float32)

    def _forward_univariate(self, signal: torch.Tensor) -> torch.Tensor:
        return self._encode(signal)

    @torch.no_grad()
    def _embed_univariate_perpatch(self, signal: torch.Tensor) -> np.ndarray:
        B, T = signal.shape
        ps = self._patch_size
        n_valid = math.ceil(T / ps)

        remainder = T % ps
        if remainder != 0:
            pad_len = ps - remainder
            signal = torch.nn.functional.pad(signal, (0, pad_len))
            T = signal.shape[1]
        num_patches = T // ps

        dev = self.device
        max_patch = self._module.in_proj.weight.shape[-1]

        patches = signal.reshape(B, num_patches, ps).to(dev, dtype=torch.float32)
        target = torch.zeros(B, num_patches, max_patch, device=dev, dtype=torch.float32)
        target[:, :, :ps] = patches

        observed_mask = torch.zeros(B, num_patches, max_patch, device=dev, dtype=torch.bool)
        observed_mask[:, :, :ps] = True

        sample_id = torch.zeros(B, num_patches, dtype=torch.long, device=dev)
        time_id = torch.arange(num_patches, device=dev).unsqueeze(0).expand(B, -1)
        variate_id = torch.zeros(B, num_patches, dtype=torch.long, device=dev)
        prediction_mask = torch.zeros(B, num_patches, dtype=torch.bool, device=dev)
        patch_size_tensor = torch.full((B, num_patches), ps, dtype=torch.long, device=dev)

        captured: List[torch.Tensor] = []

        def _hook(_module, _inp, out):
            captured.append(out.detach())

        handle = self._module.encoder.register_forward_hook(_hook)
        try:
            self._module(
                target=target,
                observed_mask=observed_mask,
                sample_id=sample_id,
                time_id=time_id,
                variate_id=variate_id,
                prediction_mask=prediction_mask,
                patch_size=patch_size_tensor,
            )
        finally:
            handle.remove()

        hidden = captured[0]  # (B, num_patches, d_model)
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
        hidden = hidden[:, :n_valid, :]  # drop padded patches
        return hidden.cpu().numpy().astype(np.float32)

    def metadata(self):
        return {
            **super().metadata(),
            "device": self.device,
            "hf_id": self.spec.checkpoint_path,
            "patch_size": self._patch_size,
        }
