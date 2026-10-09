from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true
    if len(y_true) >= 2:
        r, p = pearsonr(y_true, y_pred)
        pr, pp = float(r), float(p)
    else:
        pr, pp = float("nan"), float("nan")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": pr,
        "pearson_p": pp,
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals, ddof=0)),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true.reshape(-1), y_pred.reshape(-1)]))
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_per_class": f1_score(y_true, y_pred, labels=labels, average=None).astype(float).tolist(),
        "labels": labels.astype(int).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    # Binary-task extras: rely on confusion matrix and (optionally) scores.
    true_labels = np.unique(y_true)
    is_binary = true_labels.size == 2 and set(true_labels.tolist()) <= {0, 1}
    if is_binary:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        fn, tp = int(cm[1, 0]), int(cm[1, 1])
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec_pos = tp / (tp + fp) if (tp + fp) else 0.0
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        out["sensitivity"] = float(sens)
        out["specificity"] = float(spec)
        out["precision_pos"] = float(prec_pos)
        if y_score is not None:
            y_score = np.asarray(y_score).reshape(-1)
            if y_score.shape[0] == y_true.shape[0]:
                try:
                    out["auroc"] = float(roc_auc_score(y_true, y_score))
                except ValueError:
                    out["auroc"] = float("nan")
                try:
                    out["auprc"] = float(average_precision_score(y_true, y_score))
                except ValueError:
                    out["auprc"] = float("nan")
    return out
