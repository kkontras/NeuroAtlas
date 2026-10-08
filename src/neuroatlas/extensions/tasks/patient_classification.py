from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from neuroatlas.benchmarking_helpers import BenchmarkResult, TaskSpec
from neuroatlas.benchmarking_helpers.probes.probe import train_probe
from neuroatlas.extensions.tasks.linear_probe import _extract_or_load_embeddings


def _aggregate_subjects(payload, aggregation: str):
    features = np.asarray(payload.features)
    labels = np.asarray(payload.labels)
    metadata = list(payload.metadata)
    grouped: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(metadata):
        subject_id = str(item.get("subject_id"))
        grouped.setdefault(subject_id, {"features": [], "label": int(labels[idx]), "stages": []})
        grouped[subject_id]["features"].append(features[idx])
        grouped[subject_id]["stages"].append(item.get("sleep_stage"))

    rows: List[np.ndarray] = []
    y: List[int] = []
    subjects: List[str] = []
    for subject_id in sorted(grouped):
        subject = grouped[subject_id]
        subject_features = np.asarray(subject["features"], dtype=np.float32)
        if aggregation == "mean_std":
            row = np.concatenate([subject_features.mean(axis=0), subject_features.std(axis=0)], axis=0)
        else:
            row = subject_features.mean(axis=0)
        rows.append(row.astype(np.float32))
        y.append(int(subject["label"]))
        subjects.append(subject_id)
    return np.stack(rows, axis=0), np.asarray(y, dtype=np.int64), subjects


def evaluate_patient_classification(
    *,
    dataset_name: str,
    checkpoint_spec,
    datamodule,
    backbone,
    probe_config,
    task_config,
    seeds,
    cache_root,
    probe_dir: Optional[Path] = None,
    **_,
) -> BenchmarkResult:
    aggregation = str(task_config.get("aggregation", "mean"))
    train_payload, train_paths = _extract_or_load_embeddings(
        cache_root, dataset_name, "train", checkpoint_spec, backbone, datamodule.train_dataloader(), datamodule
    )
    val_payload, val_paths = _extract_or_load_embeddings(
        cache_root, dataset_name, "val", checkpoint_spec, backbone, datamodule.val_dataloader(), datamodule
    )
    test_payload, test_paths = _extract_or_load_embeddings(
        cache_root, dataset_name, "test", checkpoint_spec, backbone, datamodule.test_dataloader(), datamodule
    )

    train_x, train_y, train_subjects = _aggregate_subjects(train_payload, aggregation)
    val_x, val_y, val_subjects = _aggregate_subjects(val_payload, aggregation)
    test_x, test_y, test_subjects = _aggregate_subjects(test_payload, aggregation)

    probe_type = str(probe_config.get("type", "linear"))
    selection_metric = str(probe_config.get("selection_metric", "macro_f1"))
    probe_result = train_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        seeds=list(seeds),
        probe_type=probe_type,
        max_iter=int(probe_config.get("max_iter", 1000)),
        hidden_dims=probe_config.get("hidden_dims"),
        selection_metric=selection_metric,
    )
    cache_paths = {
        **{f"train_{key}": value for key, value in train_paths.items()},
        **{f"val_{key}": value for key, value in val_paths.items()},
        **{f"test_{key}": value for key, value in test_paths.items()},
    }
    # One row per test subject; the metrics recorded are score() of the
    # fold's saved predictions, which `neuroatlas rescore` recomputes.
    from neuroatlas import predictions as preds
    from neuroatlas.benchmarking_helpers.probes.probe import _higher_is_better

    kept = [s for s, keep in zip(test_subjects, probe_result.test_keep) if keep]
    record = preds.new(
        "patient_classification", dataset_name=dataset_name, checkpoint_spec=checkpoint_spec,
        datamodule=datamodule,
        columns={**preds.probe_columns(probe_result),
                 "subject_id": np.asarray([str(s) for s in kept], dtype=str)},
        info=preds.make_info(
            datamodule=datamodule, checkpoint_spec=checkpoint_spec, probe_config=probe_config,
            task_config=task_config, seeds=seeds, cache_paths=cache_paths,
            fit=preds.probe_fit(probe_result, selection_metric=selection_metric,
                                probe_type=probe_type,
                                higher_is_better=_higher_is_better(selection_metric),
                                hidden_dims=probe_config.get("hidden_dims")),
            score={"aggregation": aggregation, "unit": "subject_id"}))
    metrics, _saved, path = preds.finalize(record, probe_dir, score)
    if path is not None:
        cache_paths["predictions"] = str(path)
    return BenchmarkResult(
        checkpoint_id=checkpoint_spec.identifier,
        dataset_name=dataset_name,
        evaluation_mode="patient_classification_eval",
        metrics=metrics,
        cache_paths=cache_paths,
        metadata={
            **backbone.metadata(),
            **dict(getattr(datamodule, "metadata", {})),
            "task_name": "patient_classification",
            "aggregation": aggregation,
            "mode": getattr(datamodule, "metadata", {}).get("mode", ""),
            "subject_counts": {
                "train": len(train_subjects),
                "val": len(val_subjects),
                "test": len(test_subjects),
            },
        },
    )


def score(pred) -> Dict[str, Any]:
    """Subject-level classification metrics from a fold's saved predictions
    (one row per test subject)."""
    from neuroatlas import predictions as preds

    return preds.score_probe(pred.group(""), pred.fit)[0]


TASK_SPECS = [
    TaskSpec(
        slug="patient_classification",
        description="Subject-level classification from each subject's averaged embeddings "
                    "(diagnosis).",
        evaluator=evaluate_patient_classification,
        score=score,
    )
]
