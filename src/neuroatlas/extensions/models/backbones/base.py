from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from neuroatlas.benchmarking_helpers import CheckpointSpec


class BenchmarkBackbone:
    """Base class for all foundation-model backbone wrappers."""
    def __init__(self, spec: CheckpointSpec):
        self.spec = spec

    def validate_batch(self, batch) -> None:
        """Validate that a batch has the expected keys."""
        if "signals" not in batch:
            raise KeyError("Batch missing 'signals' key")
        if "label" not in batch:
            raise KeyError("Batch missing 'label' key")
        if "meta" not in batch or not isinstance(batch["meta"], list):
            raise KeyError("Batch missing 'meta' key (must be a list)")

    def require_signal(self, batch, *signal_keys: str):
        """Fetch the first available signal tensor from the batch.

        Tries each key in order against ``batch["signals"]``; returns
        the first match. Raises ``KeyError`` if none found.
        """
        signals = batch.get("signals", {})
        for key in signal_keys:
            if key in signals:
                return signals[key]
        raise KeyError(
            f"None of {signal_keys!r} found in batch signals "
            f"(available: {list(signals.keys())})"
        )

    def native_head_logits(self, batch) -> np.ndarray:
        raise NotImplementedError

    def extract_embeddings(self, batch) -> np.ndarray:
        """Pooled embeddings as numpy, for the frozen probing path.

        Implementations run under ``torch.inference_mode()``. Where a wrapper
        also implements :meth:`forward_features`, this should be a thin wrapper
        over it so the two cannot diverge::

            with torch.inference_mode():
                return self.forward_features(batch).float().cpu().numpy()
        """
        raise NotImplementedError

    def forward_features(self, batch) -> "torch.Tensor":
        """Pooled embeddings as a **grad-enabled tensor**, for fine-tuning.

        Same trunk and same pooling as :meth:`extract_embeddings`, but without
        ``inference_mode``/``no_grad`` and without the numpy conversion, so
        gradients reach the backbone.

        Returns
        -------
        torch.Tensor
            Shape ``(B, embedding_dim)``, on ``self.device``, attached to the
            autograd graph.

        Notes
        -----
        Implementing this is what makes a backbone trainable by
        a finetuning caller. Wrappers that only support frozen probing
        may leave it unimplemented; the finetuning entrypoint reports the
        families that are missing it rather than silently producing detached
        features.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement forward_features(); it "
            f"supports frozen-embedding probing only. Fine-tuning requires a "
            f"grad-enabled feature path — split it out of extract_embeddings()."
        )

    def extract_embeddings_perpatch(self, batch) -> np.ndarray:
        """Return per-patch/token embeddings before pooling.

        Shape: (B, n_tokens, token_dim) -- the pre-pooling token
        representations.  Not all backbones support this (e.g. CNN-based
        models without patch tokenization).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support per-patch embedding extraction."
        )

    def metadata(self) -> Dict[str, object]:
        return {
            "checkpoint_id": self.spec.identifier,
            "model_family": self.spec.model_family,
            "embedding_key": self.spec.embedding_key,
            "embedding_dim": self.spec.embedding_dim,
        }


# Legacy alias — some backbone subclasses import this name.
BenchmarkModelWrapper = BenchmarkBackbone
