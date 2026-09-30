"""Seizure detection evaluation task.

Extracts embeddings via the linear probe infrastructure, fits a LogisticRegression
with balanced class weights, and reports AUROC, AUPRC, F1, sensitivity-at-FPR,
and event-overlap metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from benchmarking_helpers import (
    BenchmarkFailure,
    BenchmarkResult,
    TaskSpec,
)
from extensions.tasks.linear_probe import _extract_or_load_embeddings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_unlabeled(features: np.ndarray, labels: np.ndarray, metadata: list):
    """Remove rows where label == -1 (age/sex/epilepsy tasks with unknown values)."""
    mask = labels != -1
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        logger.info("Filtered %d rows with label=-1 (keeping %d)", n_dropped, int(mask.sum()))
    return features[mask], labels[mask], [metadata[i] for i in range(len(metadata)) if mask[i]]


def _filter_nonfinite(features: np.ndarray, labels: np.ndarray, metadata: list, split_name: str):
    """Drop rows whose feature vector contains NaN or Inf.

    Rare single-window numerical blowups in the backbone output (seen on TUSZ
    for biot/neurolm_vq, 1 row per ~325k-row split) would otherwise trip
    sklearn's ``check_array`` in the probe. Dropping is safe because a NaN
    row carries no usable signal anyway; the count is always surfaced.
    """
    finite_mask = np.isfinite(features).all(axis=1)
    n_dropped = int((~finite_mask).sum())
    if n_dropped > 0:
        logger.warning(
            "Filtered %d non-finite feature rows from %s (keeping %d)",
            n_dropped, split_name, int(finite_mask.sum()),
        )
    return (
        features[finite_mask],
        labels[finite_mask],
        [metadata[i] for i in range(len(metadata)) if finite_mask[i]],
    )


def _sensitivity_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Compute sensitivity (recall) at a given false-positive rate."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Find the threshold where FPR is just below target
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def _fpr_per_hour(y_true: np.ndarray, y_pred: np.ndarray, window_s: float) -> float:
    """Compute false positive rate per hour."""
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    total_neg_hours = float(np.sum(y_true == 0)) * window_s / 3600.0
    if total_neg_hours <= 0:
        return 0.0
    return fp / total_neg_hours


# ---------------------------------------------------------------------------
# Event-overlap metrics
# ---------------------------------------------------------------------------


def _merge_consecutive_windows(
    predictions: np.ndarray,
    window_indices: Optional[np.ndarray] = None,
) -> List[tuple]:
    """Merge consecutive positive windows into events.

    Returns list of (start_idx, end_idx) tuples (inclusive indices).
    """
    if window_indices is None:
        window_indices = np.arange(len(predictions))
    events: List[tuple] = []
    in_event = False
    start = 0
    for i, (pred, idx) in enumerate(zip(predictions, window_indices)):
        if pred > 0:
            if not in_event:
                in_event = True
                start = i
        else:
            if in_event:
                events.append((start, i - 1))
                in_event = False
    if in_event:
        events.append((start, len(predictions) - 1))
    return events


def _event_overlap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute event-level overlap (OVLP) metrics.

    An event is a contiguous block of positive windows.  A predicted event
    matches a true event if they share any overlapping windows.
    """
    true_events = _merge_consecutive_windows(y_true)
    pred_events = _merge_consecutive_windows(y_pred)

    if len(true_events) == 0:
        return {
            "ovlp_sensitivity": 0.0,
            "ovlp_precision": 0.0 if len(pred_events) > 0 else 1.0,
            "ovlp_f1": 0.0,
            "n_true_events": 0,
            "n_pred_events": len(pred_events),
        }
    if len(pred_events) == 0:
        return {
            "ovlp_sensitivity": 0.0,
            "ovlp_precision": 0.0,
            "ovlp_f1": 0.0,
            "n_true_events": len(true_events),
            "n_pred_events": 0,
        }

    # Match by any-overlap
    matched_true = set()
    matched_pred = set()
    for ti, (ts, te) in enumerate(true_events):
        for pi, (ps, pe) in enumerate(pred_events):
            if ps <= te and pe >= ts:  # overlap
                matched_true.add(ti)
                matched_pred.add(pi)

    sensitivity = len(matched_true) / len(true_events)
    precision = len(matched_pred) / len(pred_events)
    f1 = 2 * sensitivity * precision / max(sensitivity + precision, 1e-8)

    return {
        "ovlp_sensitivity": float(sensitivity),
        "ovlp_precision": float(precision),
        "ovlp_f1": float(f1),
        "n_true_events": len(true_events),
        "n_pred_events": len(pred_events),
    }


# ---------------------------------------------------------------------------
# C-grid search on dev AUPRC
# ---------------------------------------------------------------------------


def _fit_best_lr(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> LogisticRegression:
    """Fit LogisticRegression with balanced weights and C selected by val AUPRC."""
    scaler = StandardScaler()
    train_x_s = scaler.fit_transform(train_x)
    val_x_s = scaler.transform(val_x)

    best_c = 1.0
    best_auprc = -1.0
    best_model = None

    for c_val in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(
            C=c_val,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(train_x_s, train_y)
        if hasattr(clf, "predict_proba"):
            val_proba = clf.predict_proba(val_x_s)[:, 1]
            try:
                auprc = average_precision_score(val_y, val_proba)
            except Exception:
                auprc = 0.0
        else:
            auprc = 0.0
        if auprc > best_auprc:
            best_auprc = auprc
            best_c = c_val
            best_model = clf

    logger.info("Best C=%.4f (val AUPRC=%.4f)", best_c, best_auprc)

    # Return the best model and the fitted scaler (store scaler as attribute)
    best_model._fitted_scaler = scaler
    return best_model


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def evaluate_seizure_detection(
    *,
    dataset_name: str,
    checkpoint_spec,
    datamodule,
    backbone,
    probe_config: Dict[str, Any],
    seeds,
    probe_dir: Path,
    cache_root: Path,
    task_config: Optional[Dict[str, Any]] = None,
    **_,
) -> BenchmarkResult:
    """Evaluate seizure detection performance.

    1. Extract embeddings for train / val / test
    2. Filter label=-1 rows
    3. Fit LogisticRegression(class_weight="balanced") with C-grid search on dev AUPRC
    4. Report AUROC, AUPRC, F1, precision, recall, balanced accuracy, MCC,
       sensitivity at FPR/h (1, 0.1, 0.01), event-overlap metrics
    """
    cache_paths: Dict[str, Any] = {}

    # Extract embeddings
    train_payload, train_paths = _extract_or_load_embeddings(
        cache_root, dataset_name, "train", checkpoint_spec, backbone,
        datamodule.train_dataloader(), datamodule,
    )
    val_payload, val_paths = _extract_or_load_embeddings(
        cache_root, dataset_name, "val", checkpoint_spec, backbone,
        datamodule.val_dataloader(), datamodule,
    )
    test_payload, test_paths = _extract_or_load_embeddings(
        cache_root, dataset_name, "test", checkpoint_spec, backbone,
        datamodule.test_dataloader(), datamodule,
    )
    cache_paths.update({f"train_{k}": v for k, v in train_paths.items()})
    cache_paths.update({f"val_{k}": v for k, v in val_paths.items()})
    cache_paths.update({f"test_{k}": v for k, v in test_paths.items()})

    # Filter unlabeled
    train_x, train_y, train_meta = _filter_unlabeled(
        train_payload.features, train_payload.labels, train_payload.metadata,
    )
    val_x, val_y, val_meta = _filter_unlabeled(
        val_payload.features, val_payload.labels, val_payload.metadata,
    )
    test_x, test_y, test_meta = _filter_unlabeled(
        test_payload.features, test_payload.labels, test_payload.metadata,
    )

    # Drop rows with non-finite features (rare backbone numerical blowups).
    train_x, train_y, train_meta = _filter_nonfinite(train_x, train_y, train_meta, "train")
    val_x, val_y, val_meta = _filter_nonfinite(val_x, val_y, val_meta, "val")
    test_x, test_y, test_meta = _filter_nonfinite(test_x, test_y, test_meta, "test")

    # Check we have labeled data
    for split_name, y_arr in [("train", train_y), ("val", val_y), ("test", test_y)]:
        if len(y_arr) == 0 or len(np.unique(y_arr)) < 2:
            return BenchmarkResult(
                checkpoint_id=checkpoint_spec.identifier,
                dataset_name=dataset_name,
                evaluation_mode="seizure_detection_eval",
                failure=BenchmarkFailure(
                    code="NO_LABELED_DATA",
                    message=f"Insufficient labeled data in {split_name} split "
                            f"(n={len(y_arr)}, unique={len(np.unique(y_arr)) if len(y_arr) else 0}).",
                ),
            )

    # Extract per-window recording_ids from metadata for per-recording event grouping
    def _rec_ids(meta_list):
        return np.asarray([
            m.get("recording_id", m.get("recording_idx", i))
            for i, m in enumerate(meta_list)
        ])

    window_s = float(datamodule.metadata.get("epoch_seconds", 30.0))

    # Unified probe: C-grid on val AUPRC + threshold tuning on val F1 +
    # per-recording event overlap + sens@FPR/h clinical targets
    from ._seizure_evaluation import full_binary_probe

    probe_result = full_binary_probe(
        train_x=train_x, train_y=train_y,
        val_x=val_x, val_y=val_y,
        test_x=test_x, test_y=test_y,
        test_rec_ids=_rec_ids(test_meta),
        window_s=window_s,
        seed=(seeds[0] if seeds else 0),
    )

    # Translate to the metrics schema expected by the BenchmarkRunner
    metrics: Dict[str, Any] = {
        "auroc": probe_result["test_auroc"],
        "auprc": probe_result["test_auprc"],
        "f1": probe_result["test_f1"],
        "precision": probe_result["test_precision"],
        "recall": probe_result["test_recall"],
        "balanced_accuracy": probe_result["test_bal_acc"],
        "mcc": probe_result["test_mcc"],
        "sensitivity_at_fpr_h_1_0": probe_result["test_sens_at_fpr_h_1_0"],
        "sensitivity_at_fpr_h_0_1": probe_result["test_sens_at_fpr_h_0_1"],
        "fpr_per_hour": probe_result["test_fpr_h"],
        "ovlp_sensitivity": probe_result["test_ovlp_sens"],
        "ovlp_precision": probe_result["test_ovlp_prec"],
        "ovlp_f1": probe_result["test_ovlp_f1"],
        "n_true_events": probe_result["test_n_true_events"],
        "n_pred_events": probe_result["test_n_pred_events"],
        "best_weight_decay": probe_result["best_weight_decay"],
        "tuned_threshold": probe_result["tuned_threshold"],
        "val_auprc": probe_result["val_auprc"],
        "val_f1_at_threshold": probe_result["val_f1_at_threshold"],
        "n_test": probe_result["n_test"],
        "n_test_pos": probe_result["n_test_pos"],
        "n_test_neg": probe_result["n_test_neg"],
    }

    return BenchmarkResult(
        checkpoint_id=checkpoint_spec.identifier,
        dataset_name=dataset_name,
        evaluation_mode="seizure_detection_eval",
        metrics=metrics,
        cache_paths=cache_paths,
        metadata={
            **backbone.metadata(),
            **dict(getattr(datamodule, "metadata", {})),
            "task_name": "seizure_detection",
            "embedding_split_sizes": {
                "train": len(train_y),
                "val": len(val_y),
                "test": len(test_y),
            },
        },
    )


TASK_SPECS = [
    TaskSpec(
        slug="seizure_detection",
        description=(
            "Seizure detection evaluation: extract embeddings, fit balanced "
            "LogisticRegression with C-grid on dev AUPRC, report AUROC/AUPRC/F1/"
            "sensitivity-at-FPR and event-overlap metrics."
        ),
        evaluator=evaluate_seizure_detection,
    )
]
