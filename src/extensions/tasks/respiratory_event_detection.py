"""Respiratory-event detection task (UCDDB).

Trains one binary probe per event subtype (and per aggregate) using
``*_fraction`` meta fields. Thresholds are expressed in absolute **seconds
per epoch** — the task converts per-sample via ``epoch_seconds`` so the
comparison is invariant to epoch length.

Default event fields cover all 7 subtypes (obstructive / central / mixed
apnea and hypopnea, plus periodic breathing) and 3 aggregates (any apnea,
any hypopnea, any respiratory event). Every subtype gets its own metrics
block at each threshold; rare subtypes may be skipped when only one class
is present.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed

from benchmarking_helpers import BenchmarkResult, TaskSpec
from benchmarking_helpers.runtime.cache import (
    cache_exists,
    load_embedding_payload,
)
from benchmarking_helpers.probes.probe import train_probe
from extensions.tasks.linear_probe import (
    _dataset_context,
    _embedding_cache_dir,
    _extract_or_load_embeddings,
)
from extensions.tasks._epoch_labels import (
    binary_labels_from_fraction_seconds,
)


DEFAULT_EVENT_FIELDS: List[str] = [
    "apnea_obstructive_fraction",
    "apnea_central_fraction",
    "apnea_mixed_fraction",
    "hypopnea_obstructive_fraction",
    "hypopnea_central_fraction",
    "hypopnea_mixed_fraction",
    "periodic_breathing_fraction",
    "apnea_any_fraction",
    "hypopnea_any_fraction",
    "ahi_fraction",
    "rdi_fraction",
    "respiratory_event_any_fraction",
]
DEFAULT_THRESHOLDS_SECONDS: List[float] = [0.0, 1.0, 3.0, 5.0, 7.0, 10.0]


def _enrich_derived_fields(metadata: List[Dict[str, Any]]) -> None:
    """Derive ahi_fraction and rdi_fraction from cached subtypes at probe time."""
    for m in metadata:
        apnea_any = m.get("apnea_any_fraction")
        hypopnea = m.get("hypopnea_any_fraction") or m.get("hypopnea_fraction")
        if apnea_any is not None and hypopnea is not None:
            m.setdefault("ahi_fraction", min(float(apnea_any) + float(hypopnea), 1.0))
        ahi = m.get("ahi_fraction")
        rera = m.get("rera_fraction")
        if ahi is not None and rera is not None:
            m.setdefault("rdi_fraction", min(float(ahi) + float(rera), 1.0))


def evaluate_respiratory_event_detection(
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
    event_fields = task_config.get("event_fields", DEFAULT_EVENT_FIELDS)
    thresholds = task_config.get("thresholds", DEFAULT_THRESHOLDS_SECONDS)
    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]
    default_eps = datamodule.metadata.get("epoch_seconds")

    use_global_cache = (
        datamodule.supports_global_embedding_cache()
        and getattr(datamodule, "limit_windows_per_split", None) is None
    )

    if use_global_cache:
        global_cache_dir = _embedding_cache_dir(
            cache_root, dataset_name, checkpoint_spec, "all",
            datamodule, purpose="global_embeddings",
        )
        if cache_exists(global_cache_dir):
            full_payload = load_embedding_payload(global_cache_dir, mmap_mode="r")
        else:
            full_payload, _ = _extract_or_load_embeddings(
                cache_root, dataset_name, "all", checkpoint_spec, backbone,
                datamodule.full_embedding_dataloader(), datamodule,
                cache_purpose="global_embeddings",
            )
        split_payloads = datamodule.split_global_embedding_payload(full_payload)
        train_payload = split_payloads["train"]
        val_payload = split_payloads["val"]
        test_payload = split_payloads["test"]
    else:
        train_payload, _ = _extract_or_load_embeddings(
            cache_root, dataset_name, "train", checkpoint_spec, backbone,
            datamodule.train_dataloader(), datamodule,
        )
        val_payload, _ = _extract_or_load_embeddings(
            cache_root, dataset_name, "val", checkpoint_spec, backbone,
            datamodule.val_dataloader(), datamodule,
        )
        test_payload, _ = _extract_or_load_embeddings(
            cache_root, dataset_name, "test", checkpoint_spec, backbone,
            datamodule.test_dataloader(), datamodule,
        )

    for payload in (train_payload, val_payload, test_payload):
        _enrich_derived_fields(payload.metadata)

    # Pre-flight check: at least one configured field must be present in the cache.
    any_field_present = any(
        field in m
        for m in train_payload.metadata[:10]
        for field in event_fields
    )
    if not any_field_present:
        return BenchmarkResult(
            checkpoint_id=checkpoint_spec.identifier,
            dataset_name=dataset_name,
            evaluation_mode="respiratory_event_detection",
            failure={
                "code": "missing_labels",
                "message": "No respiratory-event fraction fields in metadata.",
                "details": {"expected_any_of": list(event_fields)},
            },
        )

    def _fit_one(
        field: str, threshold: float,
    ) -> Tuple[str, str, Dict[str, Any]]:
        train_labels = binary_labels_from_fraction_seconds(
            train_payload.metadata, field, threshold, default_eps,
        )
        val_labels = binary_labels_from_fraction_seconds(
            val_payload.metadata, field, threshold, default_eps,
        )
        test_labels = binary_labels_from_fraction_seconds(
            test_payload.metadata, field, threshold, default_eps,
        )
        key = f"threshold_{threshold}s"
        if len(np.unique(train_labels)) < 2:
            return (field, key, {"skipped": True, "reason": f"Only one class at threshold={threshold}s"})
        probe_result = train_probe(
            train_payload.features, train_labels,
            val_payload.features, val_labels,
            test_payload.features, test_labels,
            seeds=list(seeds),
            probe_type=str(probe_config.get("type", "linear")),
            max_iter=int(probe_config.get("max_iter", 10_000)),
            hidden_dims=probe_config.get("hidden_dims"),
            selection_metric=str(probe_config.get("selection_metric", "macro_f1")),
            class_weight=probe_config.get("class_weight"),
        )
        return (field, key, probe_result.metrics)

    all_results: Dict[str, Dict[str, Any]] = {}
    jobs: List[Tuple[str, float]] = []
    for field in event_fields:
        if not any(field in m for m in train_payload.metadata[:10]):
            all_results[field] = {
                "skipped": True,
                "reason": f"Field {field!r} not present in metadata",
            }
            continue
        for threshold in thresholds:
            jobs.append((field, threshold))

    n_parallel = min(len(jobs), os.cpu_count() or 1)
    results_list = Parallel(n_jobs=n_parallel, require="sharedmem")(
        delayed(_fit_one)(field, thr) for field, thr in jobs
    )
    for field, key, metrics in results_list:
        all_results.setdefault(field, {})[key] = metrics

    return BenchmarkResult(
        checkpoint_id=checkpoint_spec.identifier,
        dataset_name=dataset_name,
        evaluation_mode="respiratory_event_detection",
        metrics=all_results,
        metadata={
            **backbone.metadata(),
            **_dataset_context(datamodule),
            "task_name": "respiratory_event_detection",
            "event_fields": list(event_fields),
            "thresholds_seconds": list(thresholds),
            "threshold_unit": "seconds",
        },
    )


TASK_SPECS = [
    TaskSpec(
        slug="respiratory_event_detection",
        description="Per-subtype respiratory-event binary probes with epoch-length-aware seconds thresholds.",
        evaluator=evaluate_respiratory_event_detection,
    )
]
