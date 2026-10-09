from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from neuroatlas.benchmarking_helpers import BenchmarkResult, TaskSpec
from neuroatlas.benchmarking_helpers.probes.metrics import compute_classification_metrics


def _argmax_logits(logits: np.ndarray) -> np.ndarray:
    return np.asarray(logits).argmax(axis=1)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


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
    probe_dir: Optional[Path] = None,
    probe_config: Optional[Dict[str, Any]] = None,
    task_config: Optional[Dict[str, Any]] = None,
    seeds=(),
    **_,
) -> BenchmarkResult:
    split_outputs: Dict[str, Any] = {}
    test: Dict[str, Any] = {}
    for split_name, dataloader in {
        "val": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }.items():
        labels = []
        logits_all = []
        meta = []
        for batch in dataloader:
            logits = np.asarray(backbone.native_head_logits(batch))
            logits_all.append(logits)
            labels.append(_flatten_labels(batch))
            meta.extend(batch.get("meta") or [])
        y_true = np.concatenate(labels, axis=0)
        logits = np.concatenate(logits_all, axis=0)
        y_pred = _argmax_logits(logits)
        if split_name == "test":
            test = {"y_true": y_true, "y_pred": y_pred, "logits": logits, "meta": meta}
        else:
            split_outputs[split_name] = compute_classification_metrics(y_true, y_pred)

    # The test predictions, saved; the test metrics recorded are score() of
    # that file, which `neuroatlas rescore` recomputes. The validation split
    # is evaluated here and carried over as it is.
    from neuroatlas import predictions as preds

    columns: Dict[str, Any] = {
        "y_true": test["y_true"],
        "y_pred": test["y_pred"],
        "y_proba": _softmax(test["logits"]).astype(np.float32),
        "classes": np.arange(test["logits"].shape[1]),
    }
    if len(test["meta"]) == len(test["y_true"]):
        columns.update(preds.id_columns(test["meta"]))
    cache_paths: Dict[str, Any] = {}
    record = preds.new(
        "native_head_eval", dataset_name=dataset_name, checkpoint_spec=checkpoint_spec,
        datamodule=datamodule, columns=columns,
        info=preds.make_info(datamodule=datamodule, checkpoint_spec=checkpoint_spec,
                             probe_config=probe_config, task_config=task_config, seeds=seeds,
                             fit={"val": split_outputs["val"]}))
    metrics, _saved, path = preds.finalize(record, probe_dir, score)
    if path is not None:
        cache_paths["predictions"] = str(path)
    return BenchmarkResult(
        checkpoint_id=checkpoint_spec.identifier,
        dataset_name=dataset_name,
        evaluation_mode="native_head_eval",
        metrics=metrics,
        cache_paths=cache_paths,
        metadata={
            **backbone.metadata(),
            **dict(getattr(datamodule, "metadata", {})),
            "task_name": "native_head_eval",
        },
    )


def score(pred) -> Dict[str, Any]:
    """The native head's metrics from a fold's saved test predictions (the
    argmax of its logits); the validation block as evaluated."""
    cols = pred.group("")
    return {"val": pred.fit.get("val"),
            "test": compute_classification_metrics(cols["y_true"], cols["y_pred"])}


TASK_SPECS = [
    TaskSpec(
        slug="native_head_eval",
        description="Evaluate a checkpoint's native classification head on validation and test splits.",
        evaluator=evaluate_native_head,
        score=score,
    )
]
