from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .metrics import compute_classification_metrics, compute_regression_metrics

logger = logging.getLogger(__name__)

# --- Regression probe support (EEGBenchmarks parity) ---
# Set to False to disable regression mode and revert to classification-only.
_REGRESSION_PROBE_FIX = True

_HIGHER_IS_BETTER = {
    "accuracy": True, "macro_f1": True, "weighted_f1": True, "cohen_kappa": True,
    "r2": True, "pearson_r": True,
    "mae": False, "rmse": False,
}


def _higher_is_better(metric_name: str) -> bool:
    return _HIGHER_IS_BETTER.get(metric_name, True)


# The metrics a regression probe reports and can therefore select on.
# compute_regression_metrics also returns pearson_p and the residual moments,
# which are diagnostics, not selection criteria.
REGRESSION_SELECTION_METRICS = ("mae", "rmse", "r2", "pearson_r")


def _check_selection_metric(selection_metric: str, is_regression: bool) -> None:
    """Refuse a selection metric the probe will not compute, before fitting.

    A classification metric handed to a regression probe used to surface as
    ``KeyError: 'macro_f1'`` after the first seed had been fitted (F-072).
    """
    if is_regression and selection_metric not in REGRESSION_SELECTION_METRICS:
        raise ValueError(
            f"selection_metric={selection_metric!r} is not a regression metric; a "
            f"regression probe reports {', '.join(REGRESSION_SELECTION_METRICS)} "
            f"(use 'mae', lower is better)."
        )


def _selection_value(metrics: Dict[str, Any], selection_metric: str, split: str) -> float:
    try:
        return float(metrics[selection_metric])
    except KeyError:
        available = sorted(k for k, v in metrics.items() if isinstance(v, (int, float)))
        raise ValueError(
            f"selection_metric={selection_metric!r} is not among the {split} metrics "
            f"this probe computed ({', '.join(available)})."
        ) from None


@dataclass
class ProbeResult:
    metrics: Dict[str, Any]
    per_seed: List[Dict[str, Any]]
    best_seed: int
    best_val_metric: float
    estimator: Any
    # What the test metrics were computed from, for the fold's saved
    # predictions (neuroatlas.predictions): the test labels scored, which of
    # the test rows given they are (non-finite rows are dropped), and each
    # seed's outputs in per_seed order -- y_pred, and y_score (binary) /
    # y_proba (classification) where the estimator gives probabilities.
    test_y: Optional[np.ndarray] = None
    test_keep: Optional[np.ndarray] = None
    test_outputs: Optional[List[Dict[str, Any]]] = None
    classes: Optional[np.ndarray] = None


def _build_probe_classifier(
    probe_type: str,
    seed: int,
    max_iter: int,
    hidden_dims: Optional[List[int]] = None,
    early_stopping: bool = True,
    class_weight: Optional[str] = None,
):
    probe_type = probe_type.lower()
    if probe_type in ("linear", "sklearn_linear"):
        classifier = LogisticRegression(
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
            class_weight=class_weight,
        )
    elif probe_type == "nonlinear":
        classifier = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_dims or [256, 128]),
            max_iter=max_iter,
            random_state=seed,
            early_stopping=early_stopping,
            n_iter_no_change=20,
        )
    else:
        raise ValueError(
            f"Unsupported probe type {probe_type!r}. "
            f"Expected 'linear', 'sklearn_linear', or 'nonlinear'."
        )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])


def _build_probe_regressor(
    probe_type: str,
    seed: int,
    max_iter: int,
    hidden_dims: Optional[List[int]] = None,
    early_stopping: bool = True,
    ridge_alpha: float = 1.0,
):
    probe_type = probe_type.lower()
    if probe_type == "linear":
        regressor = Ridge(alpha=ridge_alpha, random_state=seed)
    elif probe_type == "nonlinear":
        regressor = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_dims or [256, 128]),
            max_iter=max_iter,
            random_state=seed,
            early_stopping=early_stopping,
            n_iter_no_change=20,
        )
    else:
        raise ValueError(f"Unsupported regression probe type {probe_type!r}. Expected 'linear' or 'nonlinear'.")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ])


def train_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    seeds: List[int],
    probe_type: str = "linear",
    max_iter: int = 10_000,
    hidden_dims: Optional[List[int]] = None,
    selection_metric: str = "macro_f1",
    class_weight: Optional[str] = None,
    c_values: Optional[List[float]] = None,
    mode: str = "classification",
    ridge_alpha: float = 1.0,
) -> ProbeResult:
    is_regression = _REGRESSION_PROBE_FIX and mode == "regression"
    _check_selection_metric(selection_metric, is_regression)

    # Filter out non-finite embedding rows (e.g. GPU-corrupted cache entries).
    def _drop_nonfinite(feats, labs, name):
        finite = np.isfinite(feats).all(axis=1)
        n_bad = int((~finite).sum())
        if n_bad:
            logger.warning("probe: dropped %d of %d %s rows with non-finite features",
                           n_bad, len(feats), name)
            return feats[finite], labs[finite], finite
        return feats, labs, finite
    train_features, train_labels, _ = _drop_nonfinite(train_features, train_labels, "train")
    val_features, val_labels, _ = _drop_nonfinite(val_features, val_labels, "val")
    test_features, test_labels, test_keep = _drop_nonfinite(test_features, test_labels, "test")

    per_seed: List[Dict[str, Any]] = []
    estimators: Dict[int, Any] = {}
    # each seed's test outputs, kept for the fold's predictions file
    test_outputs: List[Dict[str, Any]] = []
    classes = None

    if is_regression:
        nonlinear_early_stopping = bool(train_features.shape[0] >= 10)
        is_binary = False
        probe_class_weight = None
    else:
        unique_labels, label_counts = np.unique(train_labels, return_counts=True)
        nonlinear_early_stopping = bool(
            train_features.shape[0] >= 10 and unique_labels.size >= 2 and label_counts.min() >= 2
        )
        is_binary = unique_labels.size == 2 and set(unique_labels.tolist()) <= {0, 1}

        # Log class distribution and weighting decision.
        dist = dict(zip(unique_labels.tolist(), label_counts.tolist()))
        logger.info("[probe] class distribution: %s", dist)
        if class_weight == "balanced":
            n_samples = int(label_counts.sum())
            n_classes = len(unique_labels)
            effective = {
                int(lab): round(n_samples / (n_classes * cnt), 4)
                for lab, cnt in zip(unique_labels.tolist(), label_counts.tolist())
            }
            logger.info("[probe] class_weight=balanced, effective weights: %s", effective)
        else:
            logger.info("[probe] class_weight=None, no reweighting")
        probe_class_weight = class_weight

    print(  # progress: one line per fit, ordinary output
        f"[probe] fitting {probe_type} {mode} probe on "
        f"train={len(train_labels)} val={len(val_labels)} test={len(test_labels)} "
        f"with {len(seeds)} seed(s)"
    )

    for seed in tqdm(seeds, desc=f"Probe seeds ({probe_type})", leave=True):
        if is_regression:
            estimator = _build_probe_regressor(
                probe_type=probe_type,
                seed=seed,
                max_iter=max_iter,
                hidden_dims=hidden_dims,
                early_stopping=nonlinear_early_stopping,
                ridge_alpha=ridge_alpha,
            )
            estimator.fit(train_features, train_labels)
            val_pred = estimator.predict(val_features)
            test_pred = estimator.predict(test_features)
            val_metrics = compute_regression_metrics(val_labels, val_pred)
            test_metrics = compute_regression_metrics(test_labels, test_pred)
            estimators[seed] = estimator
            test_outputs.append({"y_pred": np.asarray(test_pred)})
        else:
            if c_values and probe_type in ("linear", "sklearn_linear"):
                best_clf, best_c, best_val_score = None, None, -np.inf
                unconverged = set()
                for C in c_values:
                    clf = Pipeline([
                        ("scaler", StandardScaler()),
                        ("classifier", LogisticRegression(
                            C=C, max_iter=max_iter, random_state=seed,
                            solver="lbfgs", class_weight=probe_class_weight,
                        )),
                    ])
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        clf.fit(train_features, train_labels)
                    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                        unconverged.add(C)
                        logger.info("[probe] C=%.4g seed=%d did not converge within max_iter=%d",
                                    C, seed, max_iter)
                    val_pred = clf.predict(val_features)
                    val_m = compute_classification_metrics(val_labels, val_pred)
                    score = float(val_m["cohen_kappa"])
                    if score > best_val_score:
                        best_val_score, best_clf, best_c = score, clf, C
                estimator = best_clf
                if best_c in unconverged:
                    # only the C that is kept can change the result
                    logger.warning("probe: the selected C=%.4g (seed %d) did not converge within "
                                   "max_iter=%d\nfix: --max-iter N, larger", best_c, seed, max_iter)
                logger.info("[probe] seed=%d best C=%s val_cohen_kappa=%.6f",
                            seed, best_c, best_val_score)
            else:
                estimator = _build_probe_classifier(
                    probe_type=probe_type,
                    seed=seed,
                    max_iter=max_iter,
                    hidden_dims=hidden_dims,
                    early_stopping=nonlinear_early_stopping,
                    class_weight=probe_class_weight,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    estimator.fit(train_features, train_labels)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    logger.warning(
                        "probe: the solver did not converge within max_iter=%d\n"
                        "fix: --max-iter N, larger", max_iter)
            val_pred = estimator.predict(val_features)
            test_pred = estimator.predict(test_features)
            val_score = test_score = None
            test_proba = None
            if hasattr(estimator, "predict_proba"):
                try:
                    test_proba = np.asarray(estimator.predict_proba(test_features))
                except Exception:
                    test_proba = None
            if is_binary and hasattr(estimator, "predict_proba"):
                try:
                    val_score = np.asarray(estimator.predict_proba(val_features))[:, 1]
                    test_score = test_proba[:, 1] if test_proba is not None else None
                except Exception:
                    val_score = test_score = None
                if test_score is None:
                    val_score = None
            val_metrics = compute_classification_metrics(val_labels, val_pred, y_score=val_score)
            test_metrics = compute_classification_metrics(test_labels, test_pred, y_score=test_score)
            estimators[seed] = estimator
            if classes is None:
                classes = getattr(estimator, "classes_", None)
            test_outputs.append({"y_pred": np.asarray(test_pred), "y_score": test_score,
                                 "y_proba": test_proba})

        per_seed.append({
            "seed": seed,
            "val": val_metrics,
            "test": test_metrics,
        })
        primary_test_metric = "test_mae" if is_regression else "test_accuracy"
        primary_test_key = "mae" if is_regression else "accuracy"
        print(  # progress: the seed's validation and test score
            f"[probe] seed={seed} "
            f"val_{selection_metric}={_selection_value(val_metrics, selection_metric, 'val'):.6f} "
            f"{primary_test_metric}={float(test_metrics[primary_test_key]):.6f}"
        )

    comparator = max if _higher_is_better(selection_metric) else min
    best_row = comparator(per_seed, key=lambda row: float(row["val"][selection_metric]))
    best_seed = int(best_row["seed"])
    logger.info(
        f"[probe] selected best_seed={best_seed} "
        f"best_val_{selection_metric}={float(best_row['val'][selection_metric]):.6f}"
    )
    if is_regression:
        metric_names = ["mae", "rmse", "r2", "pearson_r"]
    else:
        metric_names = ["accuracy", "macro_f1", "weighted_f1", "cohen_kappa"]
    summary: Dict[str, Any] = {}
    for metric_name in metric_names:
        values = np.asarray([row["test"][metric_name] for row in per_seed], dtype=float)
        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    if not is_regression:
        summary["confusion_matrix"] = per_seed[0]["test"]["confusion_matrix"]
    summary["best_seed"] = best_seed
    summary["best_val_metric"] = float(best_row["val"][selection_metric])
    summary["best_val"] = best_row["val"]
    summary["best_test"] = best_row["test"]
    summary["probe_type"] = probe_type
    if hidden_dims:
        summary["hidden_dims"] = list(hidden_dims)
    return ProbeResult(
        metrics=summary,
        per_seed=per_seed,
        best_seed=best_seed,
        best_val_metric=float(best_row["val"][selection_metric]),
        estimator=estimators[best_seed],
        test_y=np.asarray(test_labels),
        test_keep=np.asarray(test_keep, dtype=bool),
        test_outputs=test_outputs,
        classes=None if classes is None else np.asarray(classes),
    )


def train_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    seeds: List[int],
    max_iter: int = 10_000,
    class_weight: Optional[str] = None,
    mode: str = "classification",
) -> ProbeResult:
    return train_probe(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
        seeds=seeds,
        probe_type="linear",
        max_iter=max_iter,
        class_weight=class_weight,
        mode=mode,
    )
