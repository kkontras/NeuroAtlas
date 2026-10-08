"""Arousal detection task.

Trains a binary probe on per-epoch arousal fraction labels stored in the
embedding cache metadata. Thresholds are expressed in absolute **seconds
of arousal per epoch**; each epoch's fraction is converted via its own
``epoch_seconds`` so the comparison is invariant to epoch length.

Results are reported at multiple thresholds so downstream analysis can
choose the operating point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from neuroatlas.benchmarking_helpers import (
    BenchmarkFailure,
    BenchmarkResult,
    EmbeddingPayload,
    TaskSpec,
)
from neuroatlas.benchmarking_helpers.runtime.cache import (
    cache_exists,
    load_embedding_payload,
)
from neuroatlas.benchmarking_helpers.probes.probe import train_probe
from neuroatlas.extensions.tasks.linear_probe import (
    _dataset_context,
    _embedding_cache_dir,
    _extract_or_load_embeddings,
)
from neuroatlas.extensions.tasks._epoch_labels import (
    benchmark_datasets,
    binary_labels_from_fraction_seconds,
)

# Defaults: at the standard 30 s epoch these equal the previous fraction
# defaults [0.0, 0.1, 0.3, 0.5], preserving prior behaviour at the default
# epoch length while expressing the threshold in clinically-meaningful units.
DEFAULT_THRESHOLDS_SECONDS = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0]


def _filter_payload_by_subset(payload: EmbeddingPayload, subset: str) -> EmbeddingPayload:
    keep = [i for i, m in enumerate(payload.metadata) if m.get("subset") == subset]
    if len(keep) == len(payload.metadata):
        return payload
    return EmbeddingPayload(
        features=payload.features[keep],
        labels=payload.labels[keep],
        metadata=[payload.metadata[i] for i in keep],
    )


def evaluate_arousal_detection(
    *,
    dataset_name: str,
    checkpoint_spec,
    datamodule,
    backbone,
    probe_config: Dict[str, Any],
    task_config: Dict[str, Any],
    seeds,
    probe_dir: Path,
    cache_root: Path,
    **_,
) -> BenchmarkResult:
    """Evaluate arousal detection via binary linear probe.

    Reuses the global embedding cache from the linear probe task. Extracts
    binary labels from ``meta["arousal_fraction"]`` at configurable
    thresholds in seconds of arousal per epoch.
    """
    thresholds = task_config.get("thresholds", DEFAULT_THRESHOLDS_SECONDS)
    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]
    default_eps = datamodule.metadata.get("epoch_seconds")

    use_global_cache = (
        datamodule.supports_global_embedding_cache()
        and getattr(datamodule, "limit_windows_per_split", None) is None
    )

    cache_paths: Dict[str, Any] = {}
    if use_global_cache:
        global_cache_dir = _embedding_cache_dir(
            cache_root, dataset_name, checkpoint_spec, "all",
            datamodule, purpose="global_embeddings",
        )
        if cache_exists(global_cache_dir):
            full_payload = load_embedding_payload(global_cache_dir, mmap_mode="r")
            cache_paths["global_cache_dir"] = str(global_cache_dir)
        else:
            full_payload, paths = _extract_or_load_embeddings(
                cache_root, dataset_name, "all", checkpoint_spec, backbone,
                datamodule.full_embedding_dataloader(), datamodule,
                cache_purpose="global_embeddings",
            )
            cache_paths.update({f"global_{k}": v for k, v in paths.items()})
        split_payloads = datamodule.split_global_embedding_payload(full_payload)
        train_payload = split_payloads["train"]
        val_payload = split_payloads["val"]
        test_payload = split_payloads["test"]
    else:
        train_payload, paths = _extract_or_load_embeddings(
            cache_root, dataset_name, "train", checkpoint_spec, backbone,
            datamodule.train_dataloader(), datamodule,
        )
        cache_paths.update({f"train_{k}": v for k, v in paths.items()})
        val_payload, paths = _extract_or_load_embeddings(
            cache_root, dataset_name, "val", checkpoint_spec, backbone,
            datamodule.val_dataloader(), datamodule,
        )
        cache_paths.update({f"val_{k}": v for k, v in paths.items()})
        test_payload, paths = _extract_or_load_embeddings(
            cache_root, dataset_name, "test", checkpoint_spec, backbone,
            datamodule.test_dataloader(), datamodule,
        )
        cache_paths.update({f"test_{k}": v for k, v in paths.items()})

    subset_filter = task_config.get("subset_filter")
    if subset_filter:
        train_payload = _filter_payload_by_subset(train_payload, subset_filter)
        val_payload = _filter_payload_by_subset(val_payload, subset_filter)
        test_payload = _filter_payload_by_subset(test_payload, subset_filter)

    has_arousal = any(
        "arousal_fraction" in m
        for m in train_payload.metadata[:10]
    )
    if not has_arousal:
        return BenchmarkResult(
            checkpoint_id=checkpoint_spec.identifier,
            dataset_name=dataset_name,
            evaluation_mode="arousal_detection",
            failure=BenchmarkFailure(
                code="missing_labels",
                message=f"no arousal labels in the {dataset_name} epochs (sleep_arousal "
                        f"runs on {benchmark_datasets('sleep_arousal', 'mass and physionet2026')})"),
        )

    from neuroatlas import predictions as preds
    from neuroatlas.benchmarking_helpers.probes.probe import _higher_is_better

    probe_type = str(probe_config.get("type", "linear"))
    selection_metric = str(probe_config.get("selection_metric", "macro_f1"))
    # One probe per threshold on the same test epochs: each is a group of the
    # fold's predictions file (threshold_<s>s/...), ids at the top level.
    columns: Dict[str, Any] = dict(preds.id_columns(test_payload.metadata))
    fit: Dict[str, Any] = {}
    groups: List[str] = []
    for threshold in thresholds:
        train_labels = binary_labels_from_fraction_seconds(
            train_payload.metadata, "arousal_fraction", threshold, default_eps,
        )
        val_labels = binary_labels_from_fraction_seconds(
            val_payload.metadata, "arousal_fraction", threshold, default_eps,
        )
        test_labels = binary_labels_from_fraction_seconds(
            test_payload.metadata, "arousal_fraction", threshold, default_eps,
        )

        key = f"threshold_{threshold}s"
        groups.append(key)
        if len(np.unique(train_labels)) < 2:
            fit[key] = {
                "skipped": True,
                "reason": (f"the training epochs hold one class only at more than "
                           f"{threshold:g} s of arousal"),
            }
            continue

        probe_result = train_probe(
            train_payload.features, train_labels,
            val_payload.features, val_labels,
            test_payload.features, test_labels,
            seeds=list(seeds),
            probe_type=probe_type,
            max_iter=int(probe_config.get("max_iter", 10_000)),
            hidden_dims=probe_config.get("hidden_dims"),
            selection_metric=selection_metric,
            class_weight=probe_config.get("class_weight"),
            # --tune-c: the benchmark passes 1.0 (C.2: C = 1 for event
            # detection); a grid is chosen on validation Cohen's kappa
            c_values=probe_config.get("c_values"),
        )
        columns.update(preds.prefixed(key, preds.probe_columns(probe_result, with_row=True)))
        fit[key] = preds.probe_fit(probe_result, selection_metric=selection_metric,
                                   probe_type=probe_type,
                                   higher_is_better=_higher_is_better(selection_metric),
                                   hidden_dims=probe_config.get("hidden_dims"))

    record = preds.new(
        "arousal_detection", dataset_name=dataset_name, checkpoint_spec=checkpoint_spec,
        datamodule=datamodule, columns=columns,
        info=preds.make_info(
            datamodule=datamodule, checkpoint_spec=checkpoint_spec, probe_config=probe_config,
            task_config=task_config, seeds=seeds, cache_paths=cache_paths, fit=fit,
            groups=groups,
            score={"thresholds_seconds": list(thresholds), "default_epoch_seconds": default_eps,
                   "subset_filter": subset_filter}))
    all_threshold_results, _saved, path = preds.finalize(record, probe_dir, score)
    if path is not None:
        cache_paths["predictions"] = str(path)

    return BenchmarkResult(
        checkpoint_id=checkpoint_spec.identifier,
        dataset_name=dataset_name,
        evaluation_mode="arousal_detection",
        metrics=all_threshold_results,
        cache_paths=cache_paths,
        metadata={
            **backbone.metadata(),
            **_dataset_context(datamodule),
            "task_name": "arousal_detection",
            "thresholds_seconds": list(thresholds),
            "threshold_unit": "seconds",
        },
    )


def score(pred) -> Dict[str, Any]:
    """Arousal metrics from a fold's saved predictions: one block per
    threshold, as the probe reports them (a skipped threshold says why)."""
    from neuroatlas import predictions as preds

    out: Dict[str, Any] = {}
    for key in pred.groups:
        fit = pred.fit[key]
        if fit.get("skipped"):
            out[key] = {"skipped": True, "reason": fit.get("reason")}
            continue
        out[key] = preds.score_probe(pred.group(key), fit)[0]
    return out


TASK_SPECS = [
    TaskSpec(
        slug="arousal_detection",
        description="Arousal detection: one binary probe per threshold of scored arousal "
                    "seconds in an epoch.",
        evaluator=evaluate_arousal_detection,
        score=score,
    )
]
