"""Seizure detection evaluation task.

Extracts embeddings via the linear probe infrastructure, fits a LogisticRegression
with balanced class weights, and reports the paper's event-level Sens@FA AUC
(the epilepsy headline; per fold its sensitivity-vs-FA/h curve, see
_event_sens_fa.py), AUROC, AUPRC, F1, sensitivity-at-FPR, and event-overlap
metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
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

from neuroatlas.benchmarking_helpers import (
    BenchmarkFailure,
    BenchmarkResult,
    TaskSpec,
)
from neuroatlas.extensions.tasks.linear_probe import _extract_or_load_embeddings

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
# Event-level Sens@FA (the headline)
# ---------------------------------------------------------------------------


def _event_metric_not_applicable(dataset_name: str, datamodule, test_meta: list) -> Optional[str]:
    """Why the event-level Sens@FA curve cannot be computed for this fold, or
    None if it can. It needs seizures annotated per window along continuous
    recordings (the paper's seven seizure cohorts); a cohort labelled once
    per recording (Bonn, TUAB, NMT: the paper reports AUROC and balanced
    accuracy for them, App. D.1.3) has no seizure events to count."""
    try:
        from neuroatlas.benchmarking_helpers.registry.manifest import load_manifest

        labels = load_manifest(dataset_name).get("labels") or {}
    except Exception:
        labels = {}
    mode = (getattr(datamodule, "metadata", None) or {}).get("label_mode") or labels.get("default")
    granularity = ((labels.get("modes") or {}).get(mode) or {}).get("granularity") \
        or labels.get("granularity")
    if granularity == "recording":
        return (f"{dataset_name} has one label per recording: no seizure events to score "
                f"(the paper reports AUROC and balanced accuracy for it)")
    if not test_meta or not any("recording_id" in m or "recording_idx" in m for m in test_meta):
        return "the test windows carry no recording id, so events cannot be delimited"
    return None


def _temporal_order(test_meta: list) -> np.ndarray:
    """Test windows ordered by (recording, start time), as the event metric
    reads them. The readers already emit that order (it is then the identity);
    this only guards a reader that does not."""
    n = len(test_meta)
    if not all("window_start_s" in m for m in test_meta):
        return np.arange(n)
    rec = np.asarray([str(m.get("recording_id", m.get("recording_idx", ""))) for m in test_meta])
    start = np.asarray([float(m["window_start_s"]) for m in test_meta])
    return np.lexsort((start, rec))


def _seconds_per_window(metadata: Dict[str, Any], window_s: float) -> float:
    """Signal time one window stands for: its length, or the stride when
    windows overlap. The benchmark's windows do not overlap (10 s, stride
    10 s), so this is the window length, as in the paper (10 s)."""
    stride = metadata.get("stride_s")
    try:
        stride = float(stride)
    except (TypeError, ValueError):
        return window_s
    return stride if 0 < stride < window_s else window_s


def event_sens_fa_metrics(dataset_name: str, datamodule, test_y: np.ndarray,
                          test_proba: np.ndarray, test_meta: list,
                          window_s: float) -> Dict[str, Any]:
    """This fold's ``event_sens_fa_auc``, ``event_sens_fa_curve`` (sensitivity
    on ``event_sens_fa_grid``, null where the fold cannot reach that FA/h) and
    the grid; or ``event_sens_fa_auc: None`` with the reason it does not apply.
    ``neuroatlas results`` combines the folds' curves (median, then AUC)."""
    from neuroatlas.extensions.tasks import _event_sens_fa

    reason = _event_metric_not_applicable(dataset_name, datamodule, test_meta)
    if reason is not None:
        logger.info("event-level Sens@FA not computed: %s", reason)
        return {"event_sens_fa_auc": None, "event_sens_fa_not_applicable": reason}
    order = _temporal_order(test_meta)
    rec_ids = np.asarray([str(m.get("recording_id", m.get("recording_idx"))) for m in test_meta])
    return _event_sens_fa.fold_metrics(
        np.asarray(test_y)[order], np.asarray(test_proba)[order], rec_ids[order],
        window_s=_seconds_per_window(dict(getattr(datamodule, "metadata", {}) or {}), window_s))


# ---------------------------------------------------------------------------
# Probe settings: the probe command's flags, honoured or refused
# ---------------------------------------------------------------------------

#: The C grid, class weighting and selection metric of the paper's epilepsy
#: probe (App. C.1; run/default_runs.sh section 4 passes exactly these as
#: --tune-c / --class-weight / --selection-metric). Used when a flag is absent.
PAPER_C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
PAPER_CLASS_WEIGHT = "balanced"
PAPER_SELECTION_METRIC = "auprc"

#: Selection metrics the C search can rank by, with their spellings. Both are
#: threshold-free, which is why C is chosen on one of them and the decision
#: threshold is tuned afterwards on validation F1.
_SELECTION_ALIASES = {
    "auprc": "auprc", "average_precision": "auprc", "ap": "auprc",
    "auroc": "auroc", "roc_auc": "auroc",
}
#: `probe --selection-metric` defaults to this for every task; it is not a
#: choice the user made, and seizure detection does not rank C by it.
_CLI_DEFAULT_SELECTION = "macro_f1"
_LINEAR_TYPES = ("linear", "sklearn_linear")


class ProbeSettingsError(ValueError):
    """A probe flag this task cannot honour."""


def probe_settings(probe_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The logistic-regression settings this probe runs with.

    Every probe flag that reaches the task is either honoured or refused
    with the reason -- none is dropped:

    * ``--tune-c``: the C grid (default: the paper's 0.001 ... 100).
    * ``--class-weight balanced``: balanced class weights. Absent means the
      task's own protocol, which is also balanced: the flag can only say
      ``balanced``, so there is nothing else for it to mean.
    * ``--selection-metric``: the validation metric C is ranked by --
      ``auprc`` (the paper) or ``auroc``. The probe command's default,
      ``macro_f1``, means "not given" here (it is threshold-dependent; the
      threshold is tuned after C) and is replaced by ``auprc`` with a note.
      Anything else is refused.
    * ``--max-iter``: the solver's iteration cap (default the paper's 500,
      see :data:`PAPER_MAX_ITER`). The probe command's default, 10000, means
      "not given" here, with a note; any other value is used.
    * ``--probe-type``: ``linear`` (or ``sklearn_linear``); the clinical
      metrics here need a probability from a logistic regression, so
      ``nonlinear`` is refused.
    """
    cfg = dict(probe_config or {})
    notes = []

    kind = str(cfg.get("type") or "linear")
    if kind not in _LINEAR_TYPES:
        raise ProbeSettingsError(
            f"seizure_detection fits a balanced logistic regression; --probe-type {kind} "
            f"is not supported here (use linear)")

    c_values = cfg.get("c_values")
    if c_values is None:
        c_values = list(PAPER_C_VALUES)
    else:
        try:
            c_values = [float(c) for c in c_values]
        except (TypeError, ValueError):
            raise ProbeSettingsError(f"--tune-c wants numbers, got {c_values!r}") from None
        if not c_values or any(not (c > 0) for c in c_values):
            raise ProbeSettingsError(f"--tune-c wants one or more C values > 0, got {c_values}")

    class_weight = cfg.get("class_weight")
    if class_weight is None:
        class_weight = PAPER_CLASS_WEIGHT
    elif class_weight != "balanced":
        raise ProbeSettingsError(
            f"seizure_detection weights classes 'balanced'; class_weight={class_weight!r} "
            f"is not supported")

    metric = cfg.get("selection_metric")
    if metric is None:
        metric = PAPER_SELECTION_METRIC
    elif str(metric) == _CLI_DEFAULT_SELECTION:
        notes.append(f"selection metric {metric!r} is the probe command's default; seizure "
                     f"detection ranks C by validation {PAPER_SELECTION_METRIC!r} "
                     f"(--selection-metric auprc or auroc)")
        metric = PAPER_SELECTION_METRIC
    else:
        key = str(metric).strip().lower()
        if key not in _SELECTION_ALIASES:
            raise ProbeSettingsError(
                f"seizure_detection ranks C by validation auprc or auroc; "
                f"--selection-metric {metric} is not supported")
        metric = _SELECTION_ALIASES[key]

    max_iter = cfg.get("max_iter")
    if max_iter is None:
        max_iter = PAPER_MAX_ITER
    elif int(max_iter) == _CLI_DEFAULT_MAX_ITER:
        notes.append(f"max_iter {max_iter} is the probe command's default; seizure detection "
                     f"keeps its protocol's {PAPER_MAX_ITER} (pass --max-iter N to change it)")
        max_iter = PAPER_MAX_ITER
    else:
        max_iter = int(max_iter)
    if max_iter < 1:
        raise ProbeSettingsError(f"--max-iter must be 1 or more, got {max_iter}")

    for note in notes:
        logger.warning(note)
    return {"c_values": c_values, "class_weight": class_weight,
            "selection_metric": metric, "max_iter": max_iter, "notes": notes}


#: The solver's iteration cap in the epilepsy probe the paper ran
#: (fit_lr_c_grid's 500). It is not raised by default because doing so
#: changes numbers: on Helsinki fold 0 (CBraMod) C=100 is selected and its fit
#: has not converged at 500 -- test AUROC 0.8014 at 500, 0.8094 at 2000,
#: 0.8118 at 10000 (converged) -- while on Siena fold 0 (C=0.001) all three
#: give AUROC 0.869547499. Whether to report converged fits is the authors'
#: call; `--max-iter N` asks for N.
PAPER_MAX_ITER = 500
#: `probe --max-iter` defaults to this for every task, so it is no choice.
_CLI_DEFAULT_MAX_ITER = 10_000


# ---------------------------------------------------------------------------
# Embeddings: one global cache, or one per fold and split
# ---------------------------------------------------------------------------


def _uses_global_cache(datamodule) -> bool:
    """Whether this cohort embeds once and splits per fold at probe time.

    Opt-in (``global_cache_for_seizure_detection``, set by
    ``epilepsy/_global_cache.RecordingWindowGlobalCache`` and by TUSZ's
    k-fold HDF5 module): every epilepsy adapter that can cut its folds out
    of one set of windows does. The others -- TUSZ's EDF and official-split
    readers, TUAB's H5 fast path, CHB-MIT's HDF5 backend, any cohort run with
    ``stride_s`` != ``window_s`` -- keep per-split caches, and ``neuroatlas
    run`` then extracts every fold the probe reads.
    """
    if not getattr(datamodule, "global_cache_for_seizure_detection", False):
        return False
    if getattr(datamodule, "limit_windows_per_split", None) is not None:
        return False
    try:
        return bool(datamodule.supports_global_embedding_cache())
    except Exception:
        return False


def _global_payload(cache_root, dataset_name, checkpoint_spec, backbone, datamodule, embed_chunk,
                    *, load: bool = True):
    """The global (all-windows) payload, or ``(None, paths)`` after a chunk.

    ``load=False`` (an extract-only pass): a cache that already exists is
    reported, not read -- its items.json can be hundreds of MB, and the
    pass only needs to know it is there.
    """
    from neuroatlas.benchmarking_helpers.runtime.cache import (
        cache_exists,
        load_embedding_payload,
        merge_embedding_chunks,
    )
    from neuroatlas.extensions.tasks.linear_probe import _embedding_cache_dir

    global_dir = _embedding_cache_dir(
        cache_root, dataset_name, checkpoint_spec, "all", datamodule, purpose="global_embeddings",
    )
    if cache_exists(global_dir):
        return (load_embedding_payload(global_dir, mmap_mode="r") if load else None,
                {"cache_dir": str(global_dir), "cache_hit": True})
    if embed_chunk is not None:
        chunk_idx, n_chunks = embed_chunk
        chunk_dir = global_dir / "_chunks" / f"{chunk_idx}_of_{n_chunks}"
        if not cache_exists(chunk_dir):
            _extract_or_load_embeddings(
                cache_root, dataset_name, "all", checkpoint_spec, backbone,
                datamodule.full_embedding_dataloader(), datamodule,
                cache_purpose="global_embeddings", cache_dir_override=chunk_dir,
            )
        merge_embedding_chunks(global_dir)
        return None, {"chunk_dir": str(chunk_dir)}
    if merge_embedding_chunks(global_dir):
        return (load_embedding_payload(global_dir, mmap_mode="r") if load else None,
                {"cache_dir": str(global_dir), "cache_hit": True, "merged_from_chunks": True})
    return _extract_or_load_embeddings(
        cache_root, dataset_name, "all", checkpoint_spec, backbone,
        datamodule.full_embedding_dataloader(), datamodule,
        cache_purpose="global_embeddings",
    )


def _split_payload(cache_root, dataset_name, split, checkpoint_spec, backbone, datamodule,
                   *, load: bool = True):
    """One fold's per-split payload; ``load=False`` skips reading (and even
    building the loader for) a split whose cache already exists."""
    from neuroatlas.benchmarking_helpers.runtime.cache import cache_exists
    from neuroatlas.extensions.tasks.linear_probe import _embedding_cache_dir

    if not load:
        cache_dir = _embedding_cache_dir(cache_root, dataset_name, checkpoint_spec, split, datamodule)
        if cache_exists(cache_dir):
            return None, {"cache_dir": str(cache_dir), "cache_hit": True}
    loader = getattr(datamodule, f"{split}_dataloader")()
    return _extract_or_load_embeddings(
        cache_root, dataset_name, split, checkpoint_spec, backbone, loader, datamodule,
    )


def _extract_only(dataset_name, checkpoint_spec, datamodule, backbone, cache_paths) -> BenchmarkResult:
    return BenchmarkResult(
        checkpoint_id=checkpoint_spec.identifier,
        dataset_name=dataset_name,
        evaluation_mode="seizure_detection_eval",
        metrics=None,
        cache_paths=cache_paths,
        metadata={
            **backbone.metadata(),
            **dict(getattr(datamodule, "metadata", {})),
            "task_name": "seizure_detection",
            "extract_only_status": "extracted",
            # "all": one cache for every fold -- an extraction pass over
            # several folds (embed --folds) is done after this one.
            "embedding_cache_layout": _layout(cache_paths),
        },
    )


def _layout(cache_paths: Dict[str, Any]) -> str:
    return "all" if any(k.startswith("global_") for k in cache_paths) else "per_split"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def save_predictions(probe_dir: Path, *, test_y, test_proba, test_meta: list, window_s: float,
                     threshold, dataset_name: str, checkpoint_id: str, fold) -> Path:
    """Write one fold's test predictions next to its probe outputs, so any
    metric (the event Sens@FA AUC, a new one, a fixed one) can be recomputed
    later without probing again: ``<probe dir>/predictions.npz`` with the
    window labels and scores, each window's recording and start time, the
    tuned decision threshold, and what produced them."""
    probe_dir = Path(probe_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    meta = list(test_meta or [])

    def column(*keys, default=""):
        return np.asarray([next((m[k] for k in keys if k in m), default) for m in meta])

    path = probe_dir / "predictions.npz"
    np.savez_compressed(
        path,
        y_true=np.asarray(test_y),
        y_score=np.asarray(test_proba, dtype=np.float64),
        recording_id=column("recording_id", "recording_idx").astype(str),
        subject_id=column("subject_id", "subject").astype(str),
        window_start_s=column("window_start_s", default=np.nan).astype(np.float64),
        window_s=np.float64(window_s),
        threshold=np.float64(np.nan if threshold is None else threshold),
        dataset=np.asarray(dataset_name),
        checkpoint_id=np.asarray(checkpoint_id),
        fold=np.asarray(str(fold)),
    )
    return path


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
    extract_only: bool = False,
    embed_chunk: Optional[tuple] = None,
    **_,
) -> BenchmarkResult:
    """Evaluate seizure detection performance.

    1. Extract embeddings for train / val / test
    2. Filter label=-1 rows
    3. Fit LogisticRegression(class_weight="balanced") with C-grid search on
       dev AUPRC -- the paper's settings, which --tune-c / --class-weight /
       --selection-metric / --max-iter override (see :func:`probe_settings`)
    4. Report the event-level Sens@FA curve and AUC (the headline; see
       :func:`event_sens_fa_metrics`), AUROC, AUPRC, F1, precision, recall,
       balanced accuracy, MCC, sensitivity at FPR/h (1, 0.1, 0.01),
       event-overlap metrics
    """
    cache_paths: Dict[str, Any] = {}
    # Refuse a flag this task cannot honour before any embedding is read.
    settings = None if extract_only else probe_settings(probe_config)

    if _uses_global_cache(datamodule):
        # One cache of every window, extracted once per (dataset, model);
        # this fold's train/val/test are picked from it here.
        # See extensions/datasets/epilepsy/_global_cache.py.
        full_payload, full_paths = _global_payload(
            cache_root, dataset_name, checkpoint_spec, backbone, datamodule, embed_chunk,
            load=not extract_only,
        )
        cache_paths.update({f"global_{k}": v for k, v in full_paths.items()})
        if extract_only or full_payload is None:
            return _extract_only(dataset_name, checkpoint_spec, datamodule, backbone, cache_paths)
        splits = datamodule.split_global_embedding_payload(full_payload)
        train_payload, val_payload, test_payload = splits["train"], splits["val"], splits["test"]
    else:
        if embed_chunk is not None:
            print(f"[embed-chunk] {dataset_name!r} has no global embedding cache; "
                  f"--embed-chunk ignored")
        payloads = {}
        for split in ("train", "val", "test"):
            payloads[split], paths = _split_payload(
                cache_root, dataset_name, split, checkpoint_spec, backbone, datamodule,
                load=not extract_only,
            )
            cache_paths.update({f"{split}_{k}": v for k, v in paths.items()})
        if extract_only:
            return _extract_only(dataset_name, checkpoint_spec, datamodule, backbone, cache_paths)
        train_payload, val_payload, test_payload = (
            payloads["train"], payloads["val"], payloads["test"])

    # Per-patch tokens (embed/probe --pooling per_patch) are concatenated
    # into one vector per window, as for every other probe.
    from neuroatlas.extensions.tasks.linear_probe import probe_features

    for payload in (train_payload, val_payload, test_payload):
        payload.features = probe_features(payload.features)

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
        c_values=settings["c_values"],
        class_weight=settings["class_weight"],
        selection_metric=settings["selection_metric"],
        max_iter=settings["max_iter"],
        seed=(seeds[0] if seeds else 0),
    )

    cache_paths["predictions"] = str(save_predictions(
        probe_dir, test_y=test_y, test_proba=probe_result["test_proba"], test_meta=test_meta,
        window_s=window_s, threshold=probe_result.get("tuned_threshold"),
        dataset_name=dataset_name, checkpoint_id=checkpoint_spec.identifier,
        fold=dict(getattr(datamodule, "metadata", {}) or {}).get("fold")))

    # Translate to the metrics schema expected by the BenchmarkRunner
    metrics: Dict[str, Any] = {
        # the headline (configs/benchmarks/epilepsy.yaml): this fold's curve
        # and its own AUC; `neuroatlas results` aggregates the curves
        **event_sens_fa_metrics(dataset_name, datamodule, test_y, probe_result["test_proba"],
                                test_meta, window_s),
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
        "best_c": probe_result["best_weight_decay"],
        # the validation score C was chosen by (settings["selection_metric"])
        "val_selection_score": probe_result["val_selection_score"],
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
            # "all": one cache of every window, folds assigned here;
            # "per_split": this fold's train/val/test caches.
            "embedding_cache_layout": _layout(cache_paths),
            # What the probe ran with (see probe_settings).
            "probe_c_values": list(settings["c_values"]),
            "probe_class_weight": settings["class_weight"],
            "probe_selection_metric": settings["selection_metric"],
            "probe_max_iter": settings["max_iter"],
            "probe_settings_notes": list(settings["notes"]),
            "probe_feature_dim": int(np.asarray(train_x).shape[1]),
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
            "LogisticRegression with C-grid on dev AUPRC, report the event-level "
            "Sens@FA AUC, AUROC/AUPRC/F1/sensitivity-at-FPR and event-overlap metrics."
        ),
        evaluator=evaluate_seizure_detection,
    )
]
