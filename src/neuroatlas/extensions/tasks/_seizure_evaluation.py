"""Seizure-detection evaluation: fit, choose a threshold, score events.

Lives next to seizure_detection.py, its only caller. It was in the shared
helpers because four separate epilepsy entrypoints needed identical metrics;
those four are now that one task module, so there is nothing left to share.

Not a probe variant. Epilepsy is scored clinically -- did a seizure get
caught, and how many false alarms per hour -- which needs a decision
threshold and recording boundaries. The other tasks score one label per
window and use neuroatlas.benchmarking_helpers.probes.probe instead.

Key features vs a naive sklearn probe:
- C-grid search on validation AUPRC (not val F1 — F1 depends on threshold)
- Threshold tuned on validation set (default: F1-max)
- Event-overlap metrics computed per-recording (no false merges across
  recording or subject boundaries)
- Sensitivity at clinical FPR/h targets (1/h, 0.1/h) — the SzCORE / Epilepsy
  Bench reporting standard

The benchmark's headline, the paper's event-level Sens@FA AUC, is not here:
it is in _event_sens_fa.py (numpy only, also used by ``neuroatlas results``).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level metric helpers
# ---------------------------------------------------------------------------


def _merge_consecutive(pred: np.ndarray) -> List[Tuple[int, int]]:
    """Merge consecutive positive labels into (start, end) index pairs."""
    events: List[Tuple[int, int]] = []
    in_e, start = False, 0
    for i, p in enumerate(pred):
        if p > 0 and not in_e:
            in_e, start = True, i
        elif p == 0 and in_e:
            events.append((start, i - 1))
            in_e = False
    if in_e:
        events.append((start, len(pred) - 1))
    return events


def event_overlap_metrics_grouped(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    recording_ids: Sequence[str],
) -> Dict[str, float]:
    """Event-overlap metrics, grouped by recording.

    Consecutive positive windows within the *same recording* merge into one
    event. Any-overlap matching is applied across all (recording, event)
    pairs from the truth/prediction sets.

    This prevents the common bug where two positive windows in different
    recordings get glued into one fake event by `_merge_consecutive` on the
    full concatenated array.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recording_ids = np.asarray(recording_ids)

    true_events: List[Tuple[str, int, int]] = []
    pred_events: List[Tuple[str, int, int]] = []

    for rid in np.unique(recording_ids):
        mask = recording_ids == rid
        # Indices within the recording (local, not global)
        for s, e in _merge_consecutive(y_true[mask]):
            true_events.append((rid, s, e))
        for s, e in _merge_consecutive(y_pred[mask]):
            pred_events.append((rid, s, e))

    if not true_events:
        return {
            "ovlp_sensitivity": 0.0,
            "ovlp_precision": 0.0 if pred_events else 1.0,
            "ovlp_f1": 0.0,
            "n_true_events": 0,
            "n_pred_events": len(pred_events),
        }
    if not pred_events:
        return {
            "ovlp_sensitivity": 0.0,
            "ovlp_precision": 0.0,
            "ovlp_f1": 0.0,
            "n_true_events": len(true_events),
            "n_pred_events": 0,
        }

    # Any-overlap matching, but only between events from the same recording.
    matched_true = set()
    matched_pred = set()
    # Bucket pred events by rid for efficiency
    pred_by_rid: Dict[str, List[Tuple[int, Tuple[int, int]]]] = {}
    for pi, (rid, ps, pe) in enumerate(pred_events):
        pred_by_rid.setdefault(rid, []).append((pi, (ps, pe)))
    for ti, (rid, ts, te) in enumerate(true_events):
        for pi, (ps, pe) in pred_by_rid.get(rid, []):
            if ps <= te and pe >= ts:
                matched_true.add(ti)
                matched_pred.add(pi)

    sens = len(matched_true) / len(true_events)
    prec = len(matched_pred) / len(pred_events)
    f1 = 2 * sens * prec / max(sens + prec, 1e-12)

    return {
        "ovlp_sensitivity": float(sens),
        "ovlp_precision": float(prec),
        "ovlp_f1": float(f1),
        "n_true_events": len(true_events),
        "n_pred_events": len(pred_events),
    }


def sensitivity_at_fpr_per_hour(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    window_s: float,
    targets: Sequence[float] = (1.0, 0.1),
) -> Dict[str, float]:
    """Compute window-level sensitivity at clinical FPR/h targets.

    Sweeps the probability threshold and finds the threshold where
    ``FP / (background_hours)`` is closest to each target from below.
    Returns ``{"sens_at_fpr_h_1_0": ..., "sens_at_fpr_h_0_1": ...}``.

    Note: this is still a window-level sensitivity — for event-level
    sensitivity at FPR/h, use the event-overlap metrics above combined
    with a threshold sweep.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n_neg = int((y_true == 0).sum())
    neg_hours = n_neg * window_s / 3600.0
    if neg_hours <= 0 or n_neg == 0:
        return {f"sens_at_fpr_h_{t}".replace(".", "_"): 0.0 for t in targets}

    fpr, tpr, _thr = roc_curve(y_true, y_proba)
    # Convert window-level FPR to FPR/h: FPR * n_neg / neg_hours = FPR * 3600/window_s
    fpr_h = fpr * 3600.0 / window_s  # monotonically increasing with fpr

    out: Dict[str, float] = {}
    for target in targets:
        # Largest tpr such that fpr_h <= target
        mask = fpr_h <= target
        sens = float(tpr[mask].max()) if mask.any() else 0.0
        key = f"sens_at_fpr_h_{str(target).replace('.', '_')}"
        out[key] = sens
    return out


def _fpr_per_hour(y_true: np.ndarray, y_pred: np.ndarray, window_s: float) -> float:
    """Threshold-dependent FPR/h at the chosen prediction threshold."""
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    n_neg = int((y_true == 0).sum())
    neg_hours = n_neg * window_s / 3600.0
    return float(fp / neg_hours) if neg_hours > 0 else 0.0


def au_sens_fa(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    window_s: float = 10.0,
    fa_max: float = 10.0,
) -> float:
    """Window-level area under sensitivity-vs-FA/h curve, normalised to fa_max.

    Mirrors the Siena stratification helper used for the paper figures:
    sort by descending probability, compute cumulative sensitivity and
    cumulative FA/h, trapezoid-integrate sens over fa in ``[0, fa_max]``,
    divide by ``fa_max`` so the result is in ``[0, 1]``.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_true.size == 0 or int(y_true.sum()) == 0:
        return float("nan")
    if int((y_true == 0).sum()) == 0:
        return float("nan")
    total_hours = y_true.size * window_s / 3600.0
    if total_hours <= 0:
        return float("nan")
    order = np.argsort(-y_proba, kind="stable")
    y_sorted = y_true[order].astype(np.int64)
    fa_cum = np.cumsum(y_sorted == 0) / total_hours
    sens_cum = np.cumsum(y_sorted == 1) / max(int(y_true.sum()), 1)
    fa = np.concatenate([[0.0], fa_cum])
    sens = np.concatenate([[0.0], sens_cum])
    keep = fa <= fa_max
    fa_k = fa[keep]
    sens_k = sens[keep]
    if fa_k[-1] < fa_max:
        fa_k = np.concatenate([fa_k, [fa_max]])
        sens_k = np.concatenate([sens_k, [sens_k[-1]]])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(sens_k, fa_k) / fa_max)


def event_au_sens_fa(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    recording_ids: Sequence[str],
    *,
    window_s: float = 10.0,
    fa_max: float = 10.0,
    n_thresholds: int = 50,
) -> Dict[str, float]:
    """Event-level (sensitivity, FA/h) AUC, swept over probability quantiles.

    For each threshold:
      - binarise y_proba >= thr
      - merge consecutive positive windows per recording into events
      - any-overlap-match against truth events
      - sensitivity = matched_true / n_true_events
      - FA/h = (n_pred_events − matched_pred) / total_hours
    Then sort by FA/h ascending and trapezoid-integrate sens over FA/h in
    ``[0, fa_max]``, normalised by ``fa_max``. Returns dict with the AUC,
    threshold count, and the sweep itself for diagnostics.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    rec_ids = np.asarray(recording_ids)
    if y_true.size == 0 or int(y_true.sum()) == 0:
        return {"event_au_sens_fa": float("nan"), "n_thresholds": 0}

    total_hours = y_true.size * window_s / 3600.0
    if total_hours <= 0:
        return {"event_au_sens_fa": float("nan"), "n_thresholds": 0}

    qs = np.linspace(1.0 / (n_thresholds + 1), n_thresholds / (n_thresholds + 1), n_thresholds)
    thresholds = np.unique(np.quantile(y_proba, qs))

    sens_pts: List[float] = [0.0]
    fa_pts: List[float] = [0.0]
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(np.int64)
        m = event_overlap_metrics_grouped(y_true, y_pred, rec_ids)
        n_pred = int(m["n_pred_events"])
        prec = float(m["ovlp_precision"])
        n_matched_pred = int(round(prec * n_pred))
        n_fp_events = max(n_pred - n_matched_pred, 0)
        sens_pts.append(float(m["ovlp_sensitivity"]))
        fa_pts.append(n_fp_events / total_hours)

    fa_arr = np.asarray(fa_pts)
    sens_arr = np.asarray(sens_pts)
    order = np.argsort(fa_arr, kind="stable")
    fa_arr = fa_arr[order]
    sens_arr = sens_arr[order]

    # Make sens monotone non-decreasing along sorted FA — at the same FA the
    # higher sensitivity dominates (operating point envelope).
    sens_envelope = np.maximum.accumulate(sens_arr)

    keep = fa_arr <= fa_max
    fa_k = fa_arr[keep]
    sens_k = sens_envelope[keep]
    if fa_k.size == 0 or fa_k[-1] < fa_max:
        fa_k = np.concatenate([fa_k, [fa_max]])
        sens_k = np.concatenate([sens_k, [sens_k[-1] if sens_k.size else 0.0]])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trapz(sens_k, fa_k) / fa_max)
    return {
        "event_au_sens_fa": auc,
        "n_thresholds": int(thresholds.size),
    }


# ---------------------------------------------------------------------------
# Core probe steps
# ---------------------------------------------------------------------------


def fit_lr_c_grid(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    *,
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    class_weight: str = "balanced",
    max_iter: int = 500,
    seed: int = 0,
    selection_metric: str = "auprc",
    val_rec_ids: Optional[Sequence[str]] = None,
    window_s: float = 10.0,
) -> Tuple[LogisticRegression, StandardScaler, float, float]:
    """Fit balanced LogisticRegression with C-grid search on val AUPRC (binary;
    ``selection_metric="auroc"`` ranks by val AUROC, ``"event_sens_fa_auc"``
    by the validation fold's event-level Sens@FA AUC -- the epilepsy headline;
    it needs ``val_rec_ids``, windows in recording/time order) or val macro-F1
    (multiclass).

    Returns (best_clf, fitted_scaler, best_C, best_val_score).
    """
    if selection_metric not in ("auprc", "auroc", "event_sens_fa_auc"):
        raise ValueError(f"selection_metric must be 'auprc', 'auroc' or 'event_sens_fa_auc', "
                         f"got {selection_metric!r}")
    if selection_metric == "event_sens_fa_auc" and val_rec_ids is None:
        raise ValueError("selection_metric='event_sens_fa_auc' needs the validation windows' "
                         "recording ids (val_rec_ids)")
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xval_s = scaler.transform(Xval)

    is_binary = len(np.unique(np.concatenate([ytr, yval]))) <= 2
    if selection_metric == "event_sens_fa_auc":
        from ._event_sens_fa import curve_auc, fold_curve

        def rank(y, p):
            return float(curve_auc(fold_curve(y, p, np.asarray(val_rec_ids), window_s=window_s)))
    else:
        rank = average_precision_score if selection_metric == "auprc" else roc_auc_score

    best_clf, best_C, best_score = None, None, -np.inf
    # For the event metric: a C whose validation curve never reaches the
    # 0.1-100 FA/h range scores NaN; if every C does, rank by AUPRC instead.
    fallback = []
    for C in c_values:
        clf = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=seed,
        )
        try:
            clf.fit(Xtr_s, ytr)
        except Exception as e:
            logger.warning("LR fit failed for C=%s: %s", C, e)
            continue
        if is_binary and hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(Xval_s)[:, 1]
            try:
                score = float(rank(yval, proba))
            except Exception:
                score = 0.0
            if selection_metric == "event_sens_fa_auc":
                try:
                    fallback.append((float(average_precision_score(yval, proba)), C, clf))
                except Exception:
                    fallback.append((0.0, C, clf))
                if not np.isfinite(score):
                    continue
        else:
            pred = clf.predict(Xval_s)
            score = float(f1_score(yval, pred, average="macro", zero_division=0))

        if score > best_score:
            best_score = score
            best_clf = clf
            best_C = C

    if best_clf is None and fallback:
        best_score, best_C, best_clf = max(fallback, key=lambda t: t[0])
        logger.warning("the validation fold's event Sens@FA curve never reaches 0.1-100 FA/h "
                       "for any C; C=%s chosen by validation AUPRC instead", best_C)

    if best_clf is None:
        # Last-resort fallback: fit at C=1.0 without C-grid
        best_clf = LogisticRegression(
            C=1.0, class_weight=class_weight, max_iter=max_iter,
            solver="lbfgs", random_state=seed,
        )
        best_clf.fit(Xtr_s, ytr)
        best_C = 1.0
        best_score = 0.0

    return best_clf, scaler, best_C, best_score


def tune_threshold(
    clf: LogisticRegression,
    scaler: StandardScaler,
    Xval: np.ndarray,
    yval: np.ndarray,
    *,
    objective: str = "f1",
    n_thresholds: int = 101,
) -> Tuple[float, float]:
    """Sweep thresholds on val proba, pick threshold that maximizes the
    chosen objective.

    objective: one of {"f1", "mcc", "youden"}.
    Returns (best_threshold, best_score).
    """
    proba = clf.predict_proba(scaler.transform(Xval))[:, 1]
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_thr, best_score = 0.5, -np.inf
    for t in thresholds:
        pred = (proba >= t).astype(np.int64)
        if objective == "f1":
            s = f1_score(yval, pred, zero_division=0)
        elif objective == "mcc":
            s = matthews_corrcoef(yval, pred)
        elif objective == "youden":
            tn = int(((pred == 0) & (yval == 0)).sum())
            fp = int(((pred == 1) & (yval == 0)).sum())
            fn = int(((pred == 0) & (yval == 1)).sum())
            tp = int(((pred == 1) & (yval == 1)).sum())
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            s = sens + spec - 1
        else:
            raise ValueError(f"unknown objective {objective!r}")
        if s > best_score:
            best_score = float(s)
            best_thr = float(t)
    return best_thr, best_score


# ---------------------------------------------------------------------------
# High-level probe pipelines
# ---------------------------------------------------------------------------


def full_binary_probe(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_rec_ids: Optional[Sequence[str]] = None,
    val_rec_ids: Optional[Sequence[str]] = None,
    window_s: float = 30.0,
    event_window_s: Optional[float] = None,     # seconds a window stands for (the event metric)
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    class_weight: str = "balanced",
    selection_metric: str = "auprc",
    max_iter: int = 500,
    threshold_objective: str = "f1",
    seed: int = 0,
    # When set, dumps test_y/test_proba/test_rec_ids/threshold to this path so
    # ROC/Sens-FA/h plots can be regenerated without re-fitting. Filename is
    # interpreted by save_probe_predictions; common pattern is
    # ``out_dir / f"preds_fold{F}.npz"``.
    predictions_path: Optional[Path] = None,
    predictions_model_id: Optional[str] = None,
    predictions_fold: Optional[int] = None,
) -> Dict[str, Any]:
    """End-to-end binary seizure-detection linear probe.

    C is chosen from ``c_values`` by validation ``selection_metric`` (AUPRC,
    the paper's, or AUROC), with ``class_weight`` and ``max_iter`` passed to
    the solver; the decision threshold is then tuned on validation F1.

    Returns a flat dict with all standard metrics. Use ``test_rec_ids`` to
    enable per-recording event grouping; if ``None``, the event metrics
    fall back to merging the whole test array as one sequence (legacy
    behavior, not recommended).
    """
    model, scaler, best_c, val_score = fit_lr_c_grid(
        train_x, train_y, val_x, val_y,
        c_values=c_values, class_weight=class_weight, max_iter=max_iter, seed=seed,
        selection_metric=selection_metric, val_rec_ids=val_rec_ids,
        window_s=event_window_s if event_window_s is not None else window_s,
    )
    val_proba = model.predict_proba(scaler.transform(val_x))[:, 1]
    try:
        val_auprc = float(average_precision_score(val_y, val_proba))
    except Exception:
        val_auprc = 0.0
    val_event = None
    if val_rec_ids is not None:
        from ._event_sens_fa import curve_auc, fold_curve

        v = float(curve_auc(fold_curve(val_y, val_proba, np.asarray(val_rec_ids),
                                       window_s=event_window_s if event_window_s is not None
                                       else window_s)))
        val_event = v if np.isfinite(v) else None
    thr, val_f1 = tune_threshold(model, scaler, val_x, val_y, objective=threshold_objective)

    test_proba = model.predict_proba(scaler.transform(test_x))[:, 1]
    test_pred = (test_proba >= thr).astype(np.int64)

    # Threshold-free
    auroc = float(roc_auc_score(test_y, test_proba))
    auprc = float(average_precision_score(test_y, test_proba))
    sens_fpr = sensitivity_at_fpr_per_hour(test_y, test_proba, window_s)

    # Threshold-tuned
    f1 = float(f1_score(test_y, test_pred, zero_division=0))
    prec = float(precision_score(test_y, test_pred, zero_division=0))
    rec = float(recall_score(test_y, test_pred, zero_division=0))
    bal = float(balanced_accuracy_score(test_y, test_pred))
    mcc = float(matthews_corrcoef(test_y, test_pred))
    acc = float(accuracy_score(test_y, test_pred))
    fpr_h = _fpr_per_hour(test_y, test_pred, window_s)

    # Event-level
    if test_rec_ids is not None:
        ovlp = event_overlap_metrics_grouped(test_y, test_pred, test_rec_ids)
    else:
        # Fallback: single-sequence grouping (legacy)
        ovlp = event_overlap_metrics_grouped(
            test_y, test_pred, np.zeros(len(test_y), dtype=object),
        )

    out: Dict[str, Any] = {
        # Threshold-free
        "test_auroc": auroc,
        "test_auprc": auprc,
        **{f"test_{k}": v for k, v in sens_fpr.items()},
        # Threshold-tuned
        "test_f1": f1,
        "test_precision": prec,
        "test_recall": rec,
        "test_bal_acc": bal,
        "test_mcc": mcc,
        "test_acc": acc,
        "test_fpr_h": fpr_h,
        # Event-level
        "test_ovlp_sens": ovlp["ovlp_sensitivity"],
        "test_ovlp_prec": ovlp["ovlp_precision"],
        "test_ovlp_f1": ovlp["ovlp_f1"],
        "test_n_true_events": ovlp["n_true_events"],
        "test_n_pred_events": ovlp["n_pred_events"],
        # Diagnostic
        "best_weight_decay": float(best_c),
        "tuned_threshold": float(thr),
        "val_auprc": float(val_auprc),
        "val_event_sens_fa_auc": val_event,
        "selection_metric": selection_metric,
        "val_selection_score": float(val_score),
        "val_f1_at_threshold": float(val_f1),
        "n_train": int(len(train_y)),
        "n_val": int(len(val_y)),
        "n_test": int(len(test_y)),
        "n_test_pos": int((test_y == 1).sum()),
        "n_test_neg": int((test_y == 0).sum()),
        # The window probabilities themselves (an array, not a metric): the
        # caller sweeps them for the event-level Sens@FA curve.
        "test_proba": test_proba,
    }

    if predictions_path is not None:
        from .predictions import save_probe_predictions
        save_probe_predictions(
            out_dir=Path(predictions_path).parent,
            fold=predictions_fold if predictions_fold is not None else 0,
            test_y=test_y,
            test_proba=test_proba,
            test_rec_ids=np.asarray(test_rec_ids, dtype=object) if test_rec_ids is not None else None,
            model_id=predictions_model_id or "",
            threshold=float(thr),
            n_train=int(len(train_y)),
            n_val=int(len(val_y)),
            filename=Path(predictions_path).name,
            best_weight_decay=float(best_c),
            seed=int(seed),
        )

    return out


def full_multiclass_probe(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    class_labels: Optional[Sequence[int]] = None,
    class_names: Optional[Dict[int, str]] = None,
    c_values: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    seed: int = 0,
    test_rec_ids: Optional[Sequence[str]] = None,
    predictions_path: Optional[Path] = None,
    predictions_model_id: Optional[str] = None,
    predictions_fold: Optional[int] = None,
) -> Dict[str, Any]:
    """End-to-end multiclass probe with C-grid on val macro-F1 and balanced
    class weights.

    ``class_labels`` is the sorted list of class codes to report per-class
    F1 on; falls back to unique labels across all splits if not provided.
    """
    model, scaler, best_c, val_macro_f1 = fit_lr_c_grid(
        train_x, train_y, val_x, val_y,
        c_values=c_values, class_weight="balanced", seed=seed,
    )
    test_pred = model.predict(scaler.transform(test_x))
    # For ROC reconstruction we need the per-class scores too.
    test_proba_full = None
    if predictions_path is not None:
        try:
            test_proba_full = model.predict_proba(scaler.transform(test_x))
        except Exception:
            test_proba_full = None

    all_classes = class_labels
    if all_classes is None:
        all_classes = sorted(np.unique(np.concatenate([train_y, val_y, test_y])).tolist())

    macro_f1 = float(f1_score(test_y, test_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(test_y, test_pred, average="weighted", zero_division=0))
    bal = float(balanced_accuracy_score(test_y, test_pred))
    mcc = float(matthews_corrcoef(test_y, test_pred))
    acc = float(accuracy_score(test_y, test_pred))

    per_class = f1_score(
        test_y, test_pred, average=None, labels=list(all_classes), zero_division=0,
    )
    if class_names:
        per_class_dict = {class_names.get(c, str(c)): float(f) for c, f in zip(all_classes, per_class)}
    else:
        per_class_dict = {str(c): float(f) for c, f in zip(all_classes, per_class)}

    if predictions_path is not None:
        from .predictions import save_probe_predictions
        # Multiclass: save the per-class soft scores under test_proba (can be 2D).
        proba_arr = test_proba_full if test_proba_full is not None else test_pred.astype(np.float64)
        save_probe_predictions(
            out_dir=Path(predictions_path).parent,
            fold=predictions_fold if predictions_fold is not None else 0,
            test_y=test_y,
            test_proba=proba_arr,
            test_rec_ids=np.asarray(test_rec_ids, dtype=object) if test_rec_ids is not None else None,
            model_id=predictions_model_id or "",
            threshold=None,
            n_train=int(len(train_y)),
            n_val=int(len(val_y)),
            filename=Path(predictions_path).name,
            test_pred=test_pred.astype(np.int64),
            best_weight_decay=float(best_c),
            seed=int(seed),
        )

    return {
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "test_bal_acc": bal,
        "test_mcc": mcc,
        "test_acc": acc,
        "test_f1_per_class": per_class_dict,
        "best_weight_decay": float(best_c),
        "val_macro_f1": float(val_macro_f1),
        "n_train": int(len(train_y)),
        "n_val": int(len(val_y)),
        "n_test": int(len(test_y)),
    }
