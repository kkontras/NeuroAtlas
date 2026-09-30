from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from neuroatlas.benchmarking_helpers.probes.metrics import compute_classification_metrics


def _to_device(value, device: str):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _extract_inputs(batch: Dict[str, object]) -> Dict[str, object]:
    raw_batch = batch.get("raw_batch")
    if isinstance(raw_batch, dict) and "data" in raw_batch:
        return raw_batch["data"]
    if "data" in batch:
        return batch["data"]
    if "signals" in batch:
        return batch["signals"]
    raise KeyError("Expected batch with 'data', 'signals', or benchmark 'raw_batch'.")


def _extract_labels(batch) -> np.ndarray:
    labels = batch["label"]
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels).reshape(-1)


def evaluate_core_sleep_model(model: torch.nn.Module, dataloader, device: str) -> Dict[str, float | list[float] | list[int]]:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = _to_device(_extract_inputs(batch), device)
            logits = model(inputs)["preds"]["combined"].detach().cpu().numpy()
            y_true.append(_extract_labels(batch))
            y_pred.append(np.asarray(logits).argmax(axis=1))
    return compute_classification_metrics(np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0))
