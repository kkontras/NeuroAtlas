from __future__ import annotations

from typing import Any, Dict

import numpy as np

from benchmarking_helpers import BenchmarkResult, TaskSpec
from benchmarking_helpers.probes.metrics import compute_classification_metrics


def _argmax_logits(logits: np.ndarray) -> np.ndarray:
    return np.asarray(logits).argmax(axis=1)


def _flatten_labels(batch) -> np.ndarray:
    labels = batch["label"]
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels).reshape(-1)


def evaluate_native_head(
    *,
    dataset_name: str,
    checkpoint_spec,
    datamodule,
    backbone,
    **_,
) -> BenchmarkResult:
    split_outputs: Dict[str, Any] = {}
    for split_name, dataloader in {
        "val": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }.items():
        labels = []
        preds = []
        for batch in dataloader:
            logits = backbone.native_head_logits(batch)
            preds.append(_argmax_logits(logits))
            labels.append(_flatten_labels(batch))
        split_outputs[split_name] = compute_classification_metrics(
            np.concatenate(labels, axis=0),
            np.concatenate(preds, axis=0),
        )
    return BenchmarkResult(
        checkpoint_id=checkpoint_spec.identifier,
        dataset_name=dataset_name,
        evaluation_mode="native_head_eval",
        metrics=split_outputs,
        metadata={
            **backbone.metadata(),
            **dict(getattr(datamodule, "metadata", {})),
            "task_name": "native_head_eval",
        },
    )


TASK_SPECS = [
    TaskSpec(
        slug="native_head_eval",
        description="Evaluate a checkpoint's native classification head on validation and test splits.",
        evaluator=evaluate_native_head,
    )
]
