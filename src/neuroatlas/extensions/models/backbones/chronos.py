"""Chronos backbone for EEGBenchmarks.

Uses the native ``pipeline.embed()`` encoder to extract ``(B, T_tok, d_model)``
hidden states, then mean-pools over the token axis to get ``(B, d_model)``.

Embedding logic ported from TimeSeriesPhysics/benchmarks/foundation/wrappers/chronos_wrapper.py.
"""

from __future__ import annotations

import numpy as np
import torch

from neuroatlas.benchmarking_helpers import CheckpointSpec

from .ts_foundation_base import UnivariateTimeSeriesBackbone, _HF_CACHE


class ChronosBackbone(UnivariateTimeSeriesBackbone):
    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        try:
            from chronos import ChronosPipeline
        except ImportError as exc:
            raise ImportError(
                "chronos-forecasting is required for Chronos. "
                "Install: pip install chronos-forecasting"
            ) from exc

        hf_id = spec.checkpoint_path or "amazon/chronos-t5-base"
        self.pipeline = ChronosPipeline.from_pretrained(
            hf_id,
            device_map=self.device,
            torch_dtype=torch.float32,
            cache_dir=_HF_CACHE,
        )

    # ── Univariate embedding ─────────────────────────────────────────────

    @property
    def trainable_module(self) -> torch.nn.Module:
        """The inner ``nn.Module`` fine-tuning adapts.

        Chronos nests it two levels down (``ChronosPipeline.model`` is a
        ``ChronosModel`` wrapping the HuggingFace T5), so the generic attribute
        scan cannot find it — ``self.pipeline`` is a plain object, not a module.
        """
        return self.pipeline.model

    def _encode(self, signal: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` -> ``(B, d_model)`` pooled encoder states, grad-preserving.

        Inlines ``ChronosPipeline.embed`` rather than calling it, to skip that
        method's unconditional ``.cpu()`` on the result — under autograd it would
        drag the whole encoder output back to host memory and then need moving
        again for the head.

        Note that Chronos tokenises values into discrete buckets before the
        encoder, so this path is **not** differentiable with respect to the input
        signal. That is inherent to the architecture:
        the adapters live inside the T5 encoder, downstream of tokenisation.
        """
        # Chronos's tokenizer keeps its bucket boundaries on CPU while
        # device_map=cuda places the T5 weights on GPU. Passing a CUDA tensor
        # makes torch.bucketize raise a device-mismatch in
        # ChronosTokenizer._input_transform. Send the context as CPU; the
        # tokenized inputs are moved to the model's device below.
        ctx = self.pipeline._prepare_and_validate_context(
            context=signal.to(dtype=torch.float32).cpu()
        )
        token_ids, attention_mask, _ = self.pipeline.tokenizer.context_input_transform(ctx)
        model = self.pipeline.model
        emb = model.encode(
            input_ids=token_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
        )  # (B, T_tok, d_model)
        return emb.mean(dim=1) if emb.dim() == 3 else emb

    @torch.no_grad()
    def _embed_univariate(self, signal: torch.Tensor) -> np.ndarray:
        return self._encode(signal).cpu().numpy().astype(np.float32)

    def _forward_univariate(self, signal: torch.Tensor) -> torch.Tensor:
        return self._encode(signal)

    @torch.no_grad()
    def _embed_univariate_perpatch(self, signal: torch.Tensor) -> np.ndarray:
        ctx = signal.to(dtype=torch.float32).cpu()
        out = self.pipeline.embed(ctx)
        emb = out[0] if isinstance(out, tuple) else out  # (B, T_tok, d_model)
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)
        return emb.cpu().numpy().astype(np.float32)

    def metadata(self):
        return {
            **super().metadata(),
            "device": self.device,
            "hf_id": self.spec.checkpoint_path,
        }
