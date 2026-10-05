from __future__ import annotations

import copy
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import trim_mean as _scipy_trim_mean

from neuroatlas.benchmarking_helpers import BenchmarkResult, TaskSpec
from neuroatlas.benchmarking_helpers.probes.metrics import compute_regression_metrics
from neuroatlas.benchmarking_helpers.probes.probe import train_probe
from neuroatlas.extensions.tasks.linear_probe import _extract_or_load_embeddings


def _log(msg: str) -> None:
    print(f"[brain_age] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Small MLP probe (PyTorch, 2-layer: input→64→1)
# ---------------------------------------------------------------------------

class _MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_mlp_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    n_epochs: int = 100,
    batch_size: int = 256,
    device: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a 2-layer MLP (input→hidden_dim→1) with Adam / MSE.

    Returns metrics dict with per-epoch training log, best-val results,
    and final test predictions (no model saved).
    """
    # Reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _log(f"  MLP probe: {train_x.shape[0]} train, {val_x.shape[0]} val, {test_x.shape[0]} test  "
         f"dim={train_x.shape[1]}  hidden={hidden_dim}  lr={lr}  epochs={n_epochs}  "
         f"batch={batch_size}  device={device}  seed={seed}")

    input_dim = train_x.shape[1]
    model = _MLPProbe(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Normalise features (z-score on train)
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0) + 1e-8
    train_xn = torch.as_tensor((train_x - mu) / sigma, dtype=torch.float32, device=device)
    val_xn = torch.as_tensor((val_x - mu) / sigma, dtype=torch.float32, device=device)
    test_xn = torch.as_tensor((test_x - mu) / sigma, dtype=torch.float32, device=device)
    train_yt = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    val_yt = torch.as_tensor(val_y, dtype=torch.float32, device=device)
    test_yt = torch.as_tensor(test_y, dtype=torch.float32, device=device)

    n_train = len(train_yt)
    epoch_log: List[Dict[str, float]] = []
    best_val_mae = float("inf")
    best_state: Dict[str, Any] = {}
    best_epoch = 0

    from tqdm import tqdm as _tqdm
    epoch_iter = _tqdm(range(1, n_epochs + 1), desc="MLP epochs", file=sys.stdout,
                       leave=False, ncols=100, dynamic_ncols=False)
    for epoch in epoch_iter:
        # --- train ---
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(train_xn[idx])
            loss = criterion(pred, train_yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # --- val ---
        model.eval()
        with torch.no_grad():
            val_pred = model(val_xn)
            val_loss = criterion(val_pred, val_yt).item()
            val_mae = float((val_pred - val_yt).abs().mean())

        epoch_log.append({
            "epoch": epoch,
            "train_mse": round(avg_train_loss, 6),
            "val_mse": round(val_loss, 6),
            "val_mae": round(val_mae, 4),
        })

        epoch_iter.set_postfix(val_mae=f"{val_mae:.4f}", best=f"{best_val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    epoch_iter.close()

    # --- evaluate best model ---
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred_np = model(val_xn).cpu().numpy()
        test_pred_np = model(test_xn).cpu().numpy()

    val_metrics = compute_regression_metrics(val_y, val_pred_np)
    test_metrics = compute_regression_metrics(test_y, test_pred_np)

    _log(f"  MLP done: best_epoch={best_epoch}  val_mae={val_metrics['mae']:.4f}  test_mae={test_metrics['mae']:.4f}")

    return {
        "best_epoch": best_epoch,
        "n_epochs": n_epochs,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "best_val": val_metrics,
        "best_test": test_metrics,
        "training_log": epoch_log,
    }


# ---------------------------------------------------------------------------
# Subject-level embedding aggregation (existing approach)
# ---------------------------------------------------------------------------

def _aggregate_subjects_regression(payload, aggregation: str):
    features = np.asarray(payload.features)
    labels = np.asarray(payload.labels)
    metadata = list(payload.metadata)
    grouped: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(metadata):
        subject_id = str(item.get("subject_id"))
        grouped.setdefault(subject_id, {"features": [], "label": float(labels[idx]), "stages": []})
        grouped[subject_id]["features"].append(features[idx])
        grouped[subject_id]["stages"].append(item.get("sleep_stage"))

    rows: List[np.ndarray] = []
    y: List[float] = []
    subjects: List[str] = []
    for subject_id in sorted(grouped):
        subject = grouped[subject_id]
        subject_features = np.asarray(subject["features"], dtype=np.float32)
        if aggregation == "mean_std":
            row = np.concatenate([subject_features.mean(axis=0), subject_features.std(axis=0)], axis=0)
        else:
            row = subject_features.mean(axis=0)
        rows.append(row.astype(np.float32))
        y.append(float(subject["label"]))
        subjects.append(subject_id)
    return np.stack(rows, axis=0), np.asarray(y, dtype=np.float64), subjects


# ---------------------------------------------------------------------------
# Per-subject predictions helper
# ---------------------------------------------------------------------------

def _get_subject_predictions(
    estimator,
    features: np.ndarray,
    labels: np.ndarray,
    subjects: List[str],
) -> List[Dict[str, Any]]:
    """Generate per-subject prediction records from a fitted estimator."""
    predictions = estimator.predict(features)
    return [
        {"subject_id": s, "true_age": float(y), "predicted_age": float(p)}
        for s, y, p in zip(subjects, labels, predictions)
    ]


# ---------------------------------------------------------------------------
# Epoch-level predict-then-aggregate helpers
# ---------------------------------------------------------------------------

def _aggregate_epoch_predictions(
    epoch_predictions: np.ndarray,
    epoch_labels: np.ndarray,
    metadata: List[Dict[str, Any]],
    trim_proportion: float = 0.1,
) -> Dict[str, Any]:
    """Group epoch-level predictions by subject, return mean + trimmed-mean per subject."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(metadata):
        subject_id = str(item.get("subject_id"))
        if subject_id not in grouped:
            grouped[subject_id] = {"predictions": [], "label": float(epoch_labels[idx])}
        grouped[subject_id]["predictions"].append(float(epoch_predictions[idx]))

    subjects: List[str] = []
    true_ages: List[float] = []
    mean_preds: List[float] = []
    trimmed_preds: List[float] = []
    for subject_id in sorted(grouped):
        preds = np.array(grouped[subject_id]["predictions"])
        subjects.append(subject_id)
        true_ages.append(grouped[subject_id]["label"])
        mean_preds.append(float(np.mean(preds)))
        if len(preds) >= 4:
            trimmed_preds.append(float(_scipy_trim_mean(preds, proportiontocut=trim_proportion)))
        else:
            # too few epochs to trim — fall back to plain mean
            trimmed_preds.append(float(np.mean(preds)))

    return {
        "subjects": subjects,
        "true_ages": np.array(true_ages),
        "mean_predictions": np.array(mean_preds),
        "trimmed_mean_predictions": np.array(trimmed_preds),
    }


def _run_epoch_regression(
    train_payload,
    val_payload,
    test_payload,
    seeds: List[int],
    probe_config: Dict[str, Any],
    trim_proportion: float = 0.1,
    mlp_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train a probe on epoch-level embeddings, then aggregate predictions per subject."""
    _log(f"Epoch-level regression: {len(train_payload.labels):,d} train, "
         f"{len(val_payload.labels):,d} val, {len(test_payload.labels):,d} test epochs")

    epoch_probe = train_probe(
        np.asarray(train_payload.features),
        np.asarray(train_payload.labels, dtype=np.float64),
        np.asarray(val_payload.features),
        np.asarray(val_payload.labels, dtype=np.float64),
        np.asarray(test_payload.features),
        np.asarray(test_payload.labels, dtype=np.float64),
        seeds=seeds,
        probe_type=str(probe_config.get("type", "linear")),
        max_iter=int(probe_config.get("max_iter", 1000)),
        hidden_dims=probe_config.get("hidden_dims"),
        selection_metric=_regression_selection_metric(probe_config),
        mode="regression",
        ridge_alpha=float(probe_config.get("ridge_alpha", 1.0)),
    )

    estimator = epoch_probe.estimator
    val_epoch_pred = estimator.predict(np.asarray(val_payload.features))
    test_epoch_pred = estimator.predict(np.asarray(test_payload.features))

    val_agg = _aggregate_epoch_predictions(
        val_epoch_pred, val_payload.labels, val_payload.metadata, trim_proportion,
    )
    test_agg = _aggregate_epoch_predictions(
        test_epoch_pred, test_payload.labels, test_payload.metadata, trim_proportion,
    )

    val_mean_metrics = compute_regression_metrics(val_agg["true_ages"], val_agg["mean_predictions"])
    val_trimmed_metrics = compute_regression_metrics(val_agg["true_ages"], val_agg["trimmed_mean_predictions"])
    test_mean_metrics = compute_regression_metrics(test_agg["true_ages"], test_agg["mean_predictions"])
    test_trimmed_metrics = compute_regression_metrics(test_agg["true_ages"], test_agg["trimmed_mean_predictions"])

    subject_predictions = {
        "val": [
            {
                "subject_id": s,
                "true_age": float(y),
                "predicted_age_mean": float(m),
                "predicted_age_trimmed_mean": float(t),
            }
            for s, y, m, t in zip(
                val_agg["subjects"], val_agg["true_ages"],
                val_agg["mean_predictions"], val_agg["trimmed_mean_predictions"],
            )
        ],
        "test": [
            {
                "subject_id": s,
                "true_age": float(y),
                "predicted_age_mean": float(m),
                "predicted_age_trimmed_mean": float(t),
            }
            for s, y, m, t in zip(
                test_agg["subjects"], test_agg["true_ages"],
                test_agg["mean_predictions"], test_agg["trimmed_mean_predictions"],
            )
        ],
    }

    result: Dict[str, Any] = {
        "epoch_level_best_test": epoch_probe.metrics.get("best_test", {}),
        "epoch_level_best_val": epoch_probe.metrics.get("best_val", {}),
        "subject_mean": {"val": val_mean_metrics, "test": test_mean_metrics},
        "subject_trimmed_mean": {
            "val": val_trimmed_metrics,
            "test": test_trimmed_metrics,
            "trim_proportion": trim_proportion,
        },
        "subject_predictions": subject_predictions,
        "best_seed": epoch_probe.best_seed,
    }

    # --- Optional MLP on epoch-level embeddings ---
    if mlp_config is not None:
        _log("Training MLP on epoch-level embeddings")
        mlp_result = _train_mlp_probe(
            np.asarray(train_payload.features, dtype=np.float32),
            np.asarray(train_payload.labels, dtype=np.float64),
            np.asarray(val_payload.features, dtype=np.float32),
            np.asarray(val_payload.labels, dtype=np.float64),
            np.asarray(test_payload.features, dtype=np.float32),
            np.asarray(test_payload.labels, dtype=np.float64),
            **mlp_config,
        )
        result["mlp"] = mlp_result

    return result


# ---------------------------------------------------------------------------
# Nested-CV alpha selection (subject-level)
# ---------------------------------------------------------------------------
#
# Ported from the pre-merge EEGBenchmarks implementation, which is what produced
# the published brain-age numbers. The distinction from the val-split sweep below
# matters: this selects alpha by inner CV over the *merged* train+val set and
# reports under ``metrics["ridge_subject_nested_cv"]``, which is the key the
# paper's figures and LaTeX tables read
#
# Enabled per run with ``task.nested_cv = true``; defaults off so every existing
# probe path keeps its behaviour.

_AGE_BINS = np.array([0, 35, 50, 65, 80, 200], dtype=float)


def _inner_stratified_splits(
    labels: np.ndarray, n_inner_folds: int, seed: int,
) -> List[tuple]:
    """StratifiedKFold on age bins, with plain-KFold fallback if a bin is empty."""
    from sklearn.model_selection import KFold, StratifiedKFold
    bins = np.digitize(labels, _AGE_BINS) - 1
    idx = np.arange(len(labels))
    try:
        skf = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
        return list(skf.split(idx, bins))
    except ValueError:
        _log("  nested-CV: StratifiedKFold on age bins failed — falling back to plain KFold")
        kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
        return list(kf.split(idx))


def _build_ridge_pipeline(alpha: float, use_standard_scaler: bool = True):
    """StandardScaler + Ridge, matching the upstream nested-CV pipeline.

    Kept separate from the inline pipeline in ``_run_ridge_alpha_sweep`` so that
    adding the nested-CV path cannot perturb results already produced by the
    val-split sweep.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    steps = []
    if use_standard_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("ridge", Ridge(alpha=max(alpha, 1e-10))))
    return Pipeline(steps)


def _run_ridge_nested_cv_subject(
    outer_train_x: np.ndarray,
    outer_train_y: np.ndarray,
    outer_train_subjects: List[str],
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_subjects: List[str],
    alpha_values: List[float],
    n_inner_folds: int = 4,
    seed: int = 43,
    use_standard_scaler: bool = True,
) -> Dict[str, Any]:
    """Nested-CV alpha selection for the subject-level Ridge probe.

    Uses the merged (train + val) outer-train set: splits it with StratifiedKFold
    on age bins, picks alpha by minimum mean inner-val MAE, refits Ridge(alpha*)
    on the full outer-train set, and evaluates on the held-out test fold.

    Also returns inner-val out-of-fold predictions at alpha* (every outer-train
    subject appears in exactly one inner-val fold), which is what a downstream
    Cole-style bias correction would need to fit on out-of-fold rather than
    in-sample predictions.
    """
    _log(f"  nested-CV(subject): inner={n_inner_folds}-fold on "
         f"{len(outer_train_subjects)} subjects, alpha sweep {alpha_values}  "
         f"(StandardScaler={use_standard_scaler})")
    inner_splits = _inner_stratified_splits(outer_train_y, n_inner_folds, seed)

    per_alpha_mae: Dict[float, List[float]] = {a: [] for a in alpha_values}
    per_alpha_oof: Dict[float, Dict[int, float]] = {a: {} for a in alpha_values}
    per_alpha_failed: Dict[float, int] = {a: 0 for a in alpha_values}
    for inner_tr_idx, inner_va_idx in inner_splits:
        tr_x, tr_y = outer_train_x[inner_tr_idx], outer_train_y[inner_tr_idx]
        va_x, va_y = outer_train_x[inner_va_idx], outer_train_y[inner_va_idx]
        for alpha in alpha_values:
            try:
                pipe = _build_ridge_pipeline(alpha, use_standard_scaler)
                pipe.fit(tr_x, tr_y)
                va_pred = pipe.predict(va_x)
                if not np.isfinite(va_pred).all():
                    raise ValueError("non-finite inner-val predictions (overflow)")
                per_alpha_mae[alpha].append(float(np.mean(np.abs(va_pred - va_y))))
                for idx, p in zip(inner_va_idx, va_pred):
                    per_alpha_oof[alpha][int(idx)] = float(p)
            except Exception as exc:
                per_alpha_failed[alpha] += 1
                _log(f"  nested-CV inner: alpha={alpha:<6g} skipped on inner fold "
                     f"({type(exc).__name__}: {exc})")

    valid_alphas = [a for a in alpha_values if per_alpha_mae[a]]
    if not valid_alphas:
        raise RuntimeError(
            f"All alpha values failed the inner CV (overflow / non-finite "
            f"predictions). alpha grid={alpha_values}."
        )
    inner_curve = {
        a: {
            "mean_mae": float(np.mean(per_alpha_mae[a])) if per_alpha_mae[a] else float("inf"),
            "std_mae": float(np.std(per_alpha_mae[a], ddof=0)) if per_alpha_mae[a] else float("nan"),
            "n_inner_folds_ok": len(per_alpha_mae[a]),
            "n_inner_folds_failed": per_alpha_failed[a],
        }
        for a in alpha_values
    }
    selected_alpha = min(valid_alphas, key=lambda a: inner_curve[a]["mean_mae"])

    alphas_list: List[Dict[str, Any]] = []
    best_pipe = None
    best_test_metrics: Dict[str, Any] = {}
    best_test_pred: Optional[np.ndarray] = None
    for alpha in alpha_values:
        try:
            pipe = _build_ridge_pipeline(alpha, use_standard_scaler)
            pipe.fit(outer_train_x, outer_train_y)
            t_pred = pipe.predict(test_x)
            if not np.isfinite(t_pred).all():
                raise ValueError("non-finite outer-train test predictions (overflow)")
            t_metrics = compute_regression_metrics(test_y, t_pred)
        except Exception as exc:
            alphas_list.append({"alpha": alpha, "skipped": True,
                                "reason": f"{type(exc).__name__}: {exc}"})
            _log(f"  nested-CV outer-test: alpha={alpha:<6g} skipped "
                 f"({type(exc).__name__}: {exc})")
            continue
        alphas_list.append({
            "alpha": alpha,
            "val": {"mae": inner_curve[alpha]["mean_mae"],
                    "mae_std": inner_curve[alpha]["std_mae"]},
            "test": t_metrics,
        })
        if alpha == selected_alpha:
            best_pipe = pipe
            best_test_metrics = t_metrics
            best_test_pred = t_pred
    if best_pipe is None:
        raise RuntimeError(
            f"Selected alpha* = {selected_alpha} failed during outer-train test "
            f"refit; every alpha in {alpha_values} failed. Cannot return a valid "
            f"nested-CV result."
        )

    subject_predictions: List[Dict[str, Any]] = []
    if best_test_pred is not None:
        subject_predictions = [
            {"subject_id": str(s), "true_age": float(y), "predicted_age": float(p)}
            for s, y, p in zip(test_subjects, test_y, best_test_pred)
        ]

    oof_at_star = per_alpha_oof.get(selected_alpha, {})
    oof_predictions: List[Dict[str, Any]] = [
        {
            "subject_id": str(outer_train_subjects[idx]),
            "true_age": float(outer_train_y[idx]),
            "predicted_age": float(oof_at_star[idx]),
        }
        for idx in range(len(outer_train_subjects)) if idx in oof_at_star
    ]

    return {
        "alphas": alphas_list,
        "inner_val_curve": inner_curve,
        "selected_alpha": selected_alpha,
        "best_val_mae": inner_curve[selected_alpha]["mean_mae"],
        "best_val": {
            "mae": inner_curve[selected_alpha]["mean_mae"],
            "mae_std": inner_curve[selected_alpha]["std_mae"],
        },
        "best_test": best_test_metrics,
        "best_estimator": best_pipe,
        "subject_predictions": subject_predictions,
        "oof_predictions": oof_predictions,
        "use_standard_scaler": bool(use_standard_scaler),
        "n_inner_folds": int(n_inner_folds),
        "nested_cv_seed": int(seed),
        "n_outer_train_subjects": int(len(outer_train_subjects)),
        "n_test_subjects": int(len(test_subjects)),
    }


# ---------------------------------------------------------------------------
# Ridge alpha sweep (optional ablation)
# ---------------------------------------------------------------------------

def _run_ridge_alpha_sweep(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    alpha_values: List[float],
) -> Dict[str, Any]:
    """Sweep Ridge alpha values and report val/test metrics for each.

    Configured via ``probe.ridge_alpha_sweep`` in the benchmark JSON, e.g.::

        "probe": {"ridge_alpha_sweep": [0.1, 1, 5, 10, 20, 30, 50, 100]}
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    _log(f"Ridge alpha sweep: {alpha_values}")
    t0 = time.time()
    results_per_alpha: List[Dict[str, Any]] = []
    best_alpha = None
    best_val_mae = float("inf")
    best_val_metrics: Dict[str, Any] = {}
    best_test_metrics: Dict[str, Any] = {}
    best_pipe = None

    for alpha in alpha_values:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=max(alpha, 1e-10))),
        ])
        pipe.fit(train_x, train_y)
        val_pred = pipe.predict(val_x)
        test_pred = pipe.predict(test_x)
        val_metrics = compute_regression_metrics(val_y, val_pred)
        test_metrics = compute_regression_metrics(test_y, test_pred)
        results_per_alpha.append({
            "alpha": alpha,
            "val": val_metrics,
            "test": test_metrics,
        })
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_alpha = alpha
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics
            best_pipe = pipe
        _log(f"  alpha={alpha:<6g}  val_mae={val_metrics['mae']:.4f}  test_mae={test_metrics['mae']:.4f}  "
             f"test_r2={test_metrics['r2']:.4f}  test_r={test_metrics['pearson_r']:.4f}")

    _log(f"  Best alpha (by val MAE): {best_alpha}  ({time.time()-t0:.1f}s)")
    return {
        "alphas": results_per_alpha,
        "best_alpha": best_alpha,
        "best_val_mae": best_val_mae,
        "best_val": best_val_metrics,
        "best_test": best_test_metrics,
        "best_estimator": best_pipe,
    }


# ---------------------------------------------------------------------------
# Labels: ages must be present
# ---------------------------------------------------------------------------

class AgesMissingError(ValueError):
    """Every row of the embedding payload has a null / non-finite age.

    Raised instead of letting Ridge fail with "Input y contains NaN": the usual
    cause is a cache written while the dataset's age source could not be read
    (Sleep-EDF's SC-subjects.xls without ``xlrd``), and the cache keeps those
    null ages after the dependency is installed (F-069).
    """


def _cache_dirs(cache_paths: Dict[str, Any]) -> List[str]:
    return sorted({str(v) for k, v in cache_paths.items() if k.endswith("cache_dir") and v})


def _ages_missing(dataset_name: str, checkpoint_id: str, cache_paths: Dict[str, Any],
                  n_rows: int) -> AgesMissingError:
    dirs = _cache_dirs(cache_paths)
    where = "\n".join(f"    {d}" for d in dirs) or "    (no cache directory recorded)"
    return AgesMissingError(
        f"ages are missing in the embedding cache for {dataset_name}/{checkpoint_id}: "
        f"all {n_rows:,d} rows have age = null, so there is nothing to regress on.\n"
        f"  cache:\n{where}\n"
        f"  The cache was written when the dataset's age source could not be read "
        f"(Sleep-EDF: SC-subjects.xls needs `pip install xlrd`; ISRUC: the "
        f"Details_subgroup_*.xlsx tables need `pip install openpyxl`; WSC: the "
        f"csv_path demographics CSV). Installing the dependency does not repair "
        f"an existing cache: fix the age source, delete the directory above and "
        f"re-run (`neuroatlas run brain_age ...` re-extracts it; so does "
        f"`neuroatlas embed --dataset {dataset_name} --models {checkpoint_id}`)."
    )


def _cohort_protocol(task_config: Dict[str, Any], dataset_name: str) -> Optional[Dict[str, Any]]:
    """The published split protocol for this cohort, if the task config has one.

    ``task.cohort_protocols.<dataset>`` (set by configs/tasks/brain_age.json)
    replaces the cohort's own sleep-staging folds with the brain-age split the
    paper used: cohort filter, age-stratified seeded subject folds, the unit the
    ridge sees. A cohort without an entry keeps its datamodule's split.
    """
    protocols = task_config.get("cohort_protocols") or {}
    protocol = protocols.get(dataset_name)
    return dict(protocol) if protocol is not None else None


def _split_by_protocol(full_payload, *, protocol, dataset_name, checkpoint_id,
                       datamodule, cache_paths):
    """Apply the cohort's brain-age protocol to its full ``all`` payload."""
    from neuroatlas.extensions.datasets.adapters.precomputed_embeddings import (
        split_age_cohort,
    )

    meta = dict(getattr(datamodule, "metadata", {}) or {})
    fold = meta.get("fold")
    if fold is None:
        raise ValueError(f"brain_age: the {dataset_name} datamodule reports no fold")
    n_folds = int(protocol.get("n_folds", 5))
    dm_folds = meta.get("n_folds", meta.get("num_folds"))
    if dm_folds is not None and int(dm_folds) != n_folds:
        raise ValueError(
            f"brain_age: the published protocol for {dataset_name} is {n_folds}-fold, "
            f"but this run asked for {dm_folds} folds (--set n_folds). Its folds are "
            f"drawn by the protocol, not the cohort's fold list; drop --set n_folds."
        )
    label_field = str(protocol.get("label_field", "age"))
    n_labelled = 0
    for row in full_payload.metadata:
        value = row.get(label_field)
        if value is not None and np.isfinite(float(value)):
            n_labelled += 1
            break
    if n_labelled == 0:
        raise _ages_missing(dataset_name, checkpoint_id, cache_paths, len(full_payload.metadata))
    shown = {k: v for k, v in protocol.items() if not str(k).startswith("_")}
    _log(f"Split: the published brain-age protocol for {dataset_name}, not the cohort's "
         f"sleep-staging folds: {shown}")
    splits, split_meta = split_age_cohort(
        full_payload, dataset_name=dataset_name, fold=int(fold), protocol=protocol,
    )
    return splits, split_meta


def _require_ages(
    payloads: Dict[str, Any],
    *,
    dataset_name: str,
    checkpoint_id: str,
    cache_paths: Dict[str, Any],
) -> Dict[str, Any]:
    """Return the payloads with unlabelled rows removed; refuse if none has an age.

    A subject without an age (ISRUC's I_086, say) is genuinely unknown and is
    dropped, and the count is logged. No age at all is never a property of a
    cohort: it means the age source was not read when the embeddings were
    cached, and the message names the cache and how to rebuild it.
    """
    from neuroatlas.benchmarking_helpers import EmbeddingPayload

    finite = {
        split: np.isfinite(np.asarray(p.labels, dtype=np.float64))
        for split, p in payloads.items()
    }
    n_rows = sum(int(m.size) for m in finite.values())
    n_finite = sum(int(m.sum()) for m in finite.values())
    if n_finite == 0:
        raise _ages_missing(dataset_name, checkpoint_id, cache_paths, n_rows)
    out: Dict[str, Any] = {}
    for split, payload in payloads.items():
        mask = finite[split]
        if mask.all():
            out[split] = payload
            continue
        dropped = sorted({
            str(payload.metadata[i].get("subject_id"))
            for i in np.flatnonzero(~mask)
        })
        _log(f"{split}: dropping {int((~mask).sum()):,d} rows of {len(dropped)} subject(s) "
             f"with no age: {', '.join(dropped[:10])}{' ...' if len(dropped) > 10 else ''}")
        idx = np.flatnonzero(mask)
        out[split] = EmbeddingPayload(
            features=np.asarray(payload.features)[idx],
            labels=np.asarray(payload.labels, dtype=np.float64)[idx],
            metadata=[payload.metadata[i] for i in idx],
        )
    return out


def _regression_selection_metric(probe_config: Dict[str, Any]) -> str:
    """The metric the epoch-level ridge picks its seed on.

    ``probe --selection-metric`` defaults to the classification ``macro_f1``,
    which a regression probe does not compute; brain age then selects on MAE
    (lower is better), the metric the benchmark reports. An explicit regression
    metric is honoured.
    """
    from neuroatlas.benchmarking_helpers.probes.probe import REGRESSION_SELECTION_METRICS

    configured = probe_config.get("selection_metric")
    if configured in REGRESSION_SELECTION_METRICS:
        return str(configured)
    if configured:
        _log(f"selection metric {configured!r} is a classification metric; "
             f"brain age selects on 'mae' (lower is better)")
    return "mae"


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def _run_isolated(name: str, fn, failures: Dict[str, str]):
    """Run one secondary strategy; a failure is logged and recorded, not raised.

    A strategy that is not the headline must not take the fold down with it:
    F-072's epoch-level KeyError used to discard the subject-level ridge that
    had just succeeded.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 -- recorded in metrics["strategy_failures"]
        failures[name] = f"{type(exc).__name__}: {exc}"
        _log(f"  {name} FAILED, continuing without it: {type(exc).__name__}: {exc}")
        _log("  " + traceback.format_exc().rstrip().replace("\n", "\n[brain_age]   "))
        return None


def evaluate_brain_age(
    *,
    dataset_name: str,
    checkpoint_spec,
    datamodule,
    backbone,
    probe_config,
    task_config,
    seeds,
    cache_root,
    **_,
) -> BenchmarkResult:
    aggregation = str(task_config.get("aggregation", "mean"))
    trim_proportion = float(task_config.get("trim_proportion", 0.1))
    run_epoch_regression = bool(task_config.get("epoch_regression", True))
    run_mlp = bool(task_config.get("mlp", False))
    # Nested-CV alpha selection — the protocol behind the published brain-age
    # numbers, switched on by the brain_age task preset
    # (src/neuroatlas/configs/tasks/brain_age.json). When it runs it is the
    # headline: metrics.best_test is its test block.
    run_nested_cv = bool(task_config.get("nested_cv", False))
    n_inner_folds = int(task_config.get("n_inner_folds", 4))
    nested_cv_seed = int(task_config.get("nested_cv_seed", 43))
    use_standard_scaler = bool(task_config.get("use_standard_scaler", True))
    mlp_cfg: Optional[Dict[str, Any]] = None
    if run_mlp:
        mlp_raw = task_config.get("mlp_config", {})
        mlp_cfg = {
            "hidden_dim": int(mlp_raw.get("hidden_dim", 64)),
            "lr": float(mlp_raw.get("lr", 1e-3)),
            "n_epochs": int(mlp_raw.get("n_epochs", 100)),
            "batch_size": int(mlp_raw.get("batch_size", 256)),
        }
    headline = "ridge_subject_nested_cv" if run_nested_cv else "ridge_subject"

    fold = getattr(datamodule, "metadata", {}).get("fold", "?")
    _log("=" * 60)
    _log(f"Fold {fold}  |  model={checkpoint_spec.identifier}  |  dataset={dataset_name}")
    device_str = "cpu"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        device_str = f"cuda ({gpu}, {mem:.1f} GB)"
    _log(f"Device: {device_str}")
    # Only what will run, in the order it runs.
    strategies = ["Ridge(subject, val-selected alpha)"]
    if run_mlp:
        strategies.append("MLP(subject)")
    if run_epoch_regression:
        strategies.append("Ridge(epoch→agg)")
        if run_mlp:
            strategies.append("MLP(epoch)")
    if run_nested_cv:
        strategies.append("Ridge(subject, nested-CV alpha)")
    _log(f"Strategies: {', '.join(strategies)}  |  headline: {headline}")
    _log(f"Seeds: {list(seeds)}  |  aggregation={aggregation}")
    fold_t0 = time.time()

    # Use global embedding cache when supported: embed ALL epochs once,
    # then slice into train/val/test per fold.  This avoids re-embedding
    # the same samples across different folds (5× saving).
    cache_paths: Dict[str, Any] = {}
    # Optional age-matched holdout cohort (brain-age-gap study). Only a
    # datamodule configured with holdout_eval_groups returns it; otherwise it
    # stays None and nothing below changes.
    holdout_payload = None
    # The paper's per-cohort brain-age split (cohort filter, age-stratified
    # seeded folds), applied to the cohort's own cached embeddings.
    protocol = _cohort_protocol(task_config, dataset_name)
    split_protocol_meta: Optional[Dict[str, Any]] = None
    use_global = datamodule.supports_global_embedding_cache()
    if protocol is not None and not use_global:
        raise ValueError(
            f"brain_age: {dataset_name} has a published split protocol, which re-splits "
            f"the cohort's full embedding cache, but its datamodule does not serve one."
        )
    if use_global:
        full_payload, full_paths = _extract_or_load_embeddings(
            cache_root, dataset_name, "all", checkpoint_spec, backbone,
            datamodule.full_embedding_dataloader(), datamodule,
            cache_purpose="global_embeddings",
        )
        cache_paths.update({f"global_{k}": v for k, v in full_paths.items()})
        if protocol is not None:
            split_payloads, split_protocol_meta = _split_by_protocol(
                full_payload, protocol=protocol, dataset_name=dataset_name,
                checkpoint_id=checkpoint_spec.identifier, datamodule=datamodule,
                cache_paths=cache_paths,
            )
        else:
            split_payloads = datamodule.split_global_embedding_payload(full_payload)
        train_payload = split_payloads["train"]
        val_payload = split_payloads["val"]
        test_payload = split_payloads["test"]
        holdout_payload = split_payloads.get("holdout")
    else:
        train_payload, train_paths = _extract_or_load_embeddings(
            cache_root, dataset_name, "train", checkpoint_spec, backbone, datamodule.train_dataloader(), datamodule
        )
        val_payload, val_paths = _extract_or_load_embeddings(
            cache_root, dataset_name, "val", checkpoint_spec, backbone, datamodule.val_dataloader(), datamodule
        )
        test_payload, test_paths = _extract_or_load_embeddings(
            cache_root, dataset_name, "test", checkpoint_spec, backbone, datamodule.test_dataloader(), datamodule
        )
        cache_paths.update({f"train_{k}": v for k, v in train_paths.items()})
        cache_paths.update({f"val_{k}": v for k, v in val_paths.items()})
        cache_paths.update({f"test_{k}": v for k, v in test_paths.items()})

    # Refuse to probe an empty or malformed payload. Without this the failure
    # surfaces much later as "axis 1 is out of bounds for array of dimension 1"
    # from inside numpy, which says nothing about the cause. The usual cause is
    # a precomputed datamodule whose embedding cache is absent or still being
    # written — its full_embedding_dataloader is a no-op stub, so the fallback
    # extraction path yields nothing.
    for split_name, payload in (("train", train_payload), ("val", val_payload),
                                ("test", test_payload)):
        feats = np.asarray(payload.features)
        if feats.size == 0 or feats.ndim != 2:
            raise ValueError(
                f"brain_age: {split_name} payload for {dataset_name}/"
                f"{checkpoint_spec.identifier} has features with shape "
                f"{feats.shape} — expected 2-D (n_rows, dim) with at least one "
                f"row. If this dataset reads a precomputed embedding cache, the "
                f"cache is probably missing or not yet finalized; check the "
                f"[precomputed] log line above for the path it looked at."
            )

    # Ages: refuse a cache without any, drop the subjects that have none.
    labelled = _require_ages(
        {"train": train_payload, "val": val_payload, "test": test_payload},
        dataset_name=dataset_name, checkpoint_id=checkpoint_spec.identifier,
        cache_paths=cache_paths,
    )
    train_payload, val_payload, test_payload = labelled["train"], labelled["val"], labelled["test"]
    if holdout_payload is not None and len(holdout_payload.labels) > 0:
        holdout_payload = _require_ages(
            {"holdout": holdout_payload}, dataset_name=dataset_name,
            checkpoint_id=checkpoint_spec.identifier, cache_paths=cache_paths,
        )["holdout"]

    # --- Subject-level features: aggregate embeddings per subject -----------
    _log(f"Splits: train={len(train_payload.labels):,d}  val={len(val_payload.labels):,d}  test={len(test_payload.labels):,d} epochs")
    t0 = time.time()
    train_x, train_y, train_subjects = _aggregate_subjects_regression(train_payload, aggregation)
    val_x, val_y, val_subjects = _aggregate_subjects_regression(val_payload, aggregation)
    test_x, test_y, test_subjects = _aggregate_subjects_regression(test_payload, aggregation)
    _log(f"Subject aggregation ({aggregation}): {len(train_subjects)}+{len(val_subjects)}+{len(test_subjects)} subjects  "
         f"features={train_x.shape[1]}  ({time.time()-t0:.1f}s)")

    # Alpha grid: probe.ridge_alpha_sweep (the preset carries the paper's
    # grid), else the single probe.ridge_alpha.
    alpha_sweep_values = probe_config.get("ridge_alpha_sweep")
    if alpha_sweep_values is None:
        alpha_sweep_values = [float(probe_config.get("ridge_alpha", 1.0))]
    alpha_sweep_values = [float(a) for a in alpha_sweep_values]

    # Every strategy but the headline runs isolated: a failure is logged and
    # recorded under metrics["strategy_failures"], and the others still report.
    failures: Dict[str, str] = {}

    def _strategy(name: str, fn):
        if name == headline:
            return fn()  # no headline, no result: let the fold fail
        return _run_isolated(name, fn, failures)

    step = iter(range(1, len(strategies) + 1))
    n_steps = len(strategies)

    # --- Ridge on subject-level embeddings, alpha chosen on the val split ---
    def _ridge_subject():
        _log(f"Strategy {next(step)}/{n_steps}: Ridge on subject-level embeddings "
             f"(val-based alpha selection)")
        t0 = time.time()
        out = _run_ridge_alpha_sweep(
            train_x, train_y, val_x, val_y, test_x, test_y, alpha_sweep_values,
        )
        rt = out["best_test"]
        _log(f"  Ridge(subject): alpha={out['best_alpha']}  MAE={rt['mae']:.4f}  "
             f"R2={rt['r2']:.4f}  r={rt['pearson_r']:.4f}  ({time.time()-t0:.1f}s)")
        return out

    sweep_result = _strategy("ridge_subject", _ridge_subject)
    ridge_estimator = sweep_result["best_estimator"] if sweep_result else None

    # --- MLP on subject-level embeddings (aggregate-then-MLP) ---------------
    mlp_subject = None
    if mlp_cfg is not None:
        def _mlp_subject():
            _log(f"Strategy {next(step)}/{n_steps}: MLP on subject-level embeddings")
            t0 = time.time()
            out = _train_mlp_probe(train_x, train_y, val_x, val_y, test_x, test_y, **mlp_cfg)
            _log(f"  ({time.time()-t0:.1f}s)")
            return out

        mlp_subject = _strategy("mlp_subject", _mlp_subject)

    # --- Regress per epoch, then aggregate predictions per subject ----------
    epoch_regression = None
    if run_epoch_regression:
        def _epoch():
            _log(f"Strategy {next(step)}/{n_steps}: Ridge on epoch-level (predict-then-aggregate)")
            t0 = time.time()
            out = _run_epoch_regression(
                train_payload, val_payload, test_payload,
                seeds=list(seeds),
                probe_config=dict(probe_config),
                trim_proportion=trim_proportion,
                mlp_config=mlp_cfg,
            )
            er_test = out.get("subject_mean", {}).get("test", {})
            _log(f"  Epoch→Mean: MAE={er_test.get('mae', float('nan')):.4f}  "
                 f"R2={er_test.get('r2', float('nan')):.4f}")
            if "mlp" in out:
                mlp_test = out["mlp"].get("best_test", {})
                _log(f"  Epoch MLP:  MAE={mlp_test.get('mae', float('nan')):.4f}  "
                     f"R2={mlp_test.get('r2', float('nan')):.4f}")
            _log(f"  ({time.time()-t0:.1f}s)")
            return out

        epoch_regression = _strategy("epoch_regression", _epoch)

    # --- Nested CV on outer-train (train+val) for alpha selection ------------
    # The block whose output the paper's figures/tables read
    # (metrics["ridge_subject_nested_cv"].best_test.mae).
    nested_subject = None
    if run_nested_cv:
        def _nested():
            _log(f"Strategy {next(step)}/{n_steps}: Nested CV on outer-train (train+val) "
                 f"for alpha selection")
            t0 = time.time()
            outer_train_x = np.concatenate([train_x, val_x], axis=0)
            outer_train_y = np.concatenate([train_y, val_y], axis=0)
            outer_train_subjects = list(train_subjects) + list(val_subjects)
            out = _run_ridge_nested_cv_subject(
                outer_train_x, outer_train_y, outer_train_subjects,
                test_x, test_y, test_subjects,
                list(alpha_sweep_values),
                n_inner_folds=n_inner_folds,
                seed=nested_cv_seed,
                use_standard_scaler=use_standard_scaler,
            )
            ns_test = out["best_test"]
            _log(f"  Ridge(subject, nested-CV): alpha*={out['selected_alpha']}  "
                 f"MAE={ns_test['mae']:.4f}  R2={ns_test['r2']:.4f}  "
                 f"r={ns_test['pearson_r']:.4f}  ({time.time()-t0:.1f}s)")
            return out

        nested_subject = _strategy("ridge_subject_nested_cv", _nested)
    ridge_subject_nested_cv_metrics = (
        {k: v for k, v in nested_subject.items() if k != "best_estimator"}
        if nested_subject is not None else None
    )

    # --- Brain-age gap on the held-out cohort --------------------------------
    # Train on one group (the datamodule's cv_filter), then score this fold's
    # model on an age-matched cohort spanning two groups and compare their
    # brain-age gaps: BAG = predicted - true age per subject, and
    # delta_bag_mean = mean BAG(second group) - mean BAG(first group), groups in
    # sorted label order (e.g. "ci_label=1" - "ci_label=0"). Uses the headline
    # estimator: the nested-CV refit when enabled, else the validation-selected
    # ridge. Ported from the pre-merge EEGBenchmarks task; the Cole
    # bias-corrected variant is not.
    holdout_metrics: Optional[Dict[str, Any]] = None
    holdout_predictions: Optional[List[Dict[str, Any]]] = None
    if holdout_payload is not None and len(holdout_payload.labels) > 0:
        holdout_estimator = (
            nested_subject["best_estimator"] if run_nested_cv else ridge_estimator
        )
        t0 = time.time()
        holdout_x, holdout_y, holdout_subjects = _aggregate_subjects_regression(
            holdout_payload, aggregation
        )
        holdout_pred = holdout_estimator.predict(holdout_x)
        subj_to_group: Dict[str, str] = {}
        for row in holdout_payload.metadata:
            sid = str(row.get("subject_id"))
            grp = row.get("holdout_group", "")
            if grp and sid not in subj_to_group:
                subj_to_group[sid] = str(grp)
        holdout_predictions = [
            {
                "subject_id": s_,
                "true_age": float(y),
                "predicted_age": float(p_),
                "holdout_group": subj_to_group.get(s_, ""),
            }
            for s_, y, p_ in zip(holdout_subjects, holdout_y, holdout_pred)
        ]
        group_rows: Dict[str, List[Dict[str, Any]]] = {}
        for r in holdout_predictions:
            group_rows.setdefault(r["holdout_group"], []).append(r)
        by_group: Dict[str, Dict[str, Any]] = {}
        for grp, rows in group_rows.items():
            ys = np.array([r["true_age"] for r in rows], dtype=float)
            ps = np.array([r["predicted_age"] for r in rows], dtype=float)
            bag = ps - ys
            by_group[grp] = {
                "n": int(len(rows)),
                "mae": float(np.mean(np.abs(bag))),
                "mean_bag": float(np.mean(bag)),
                "std_bag": float(np.std(bag, ddof=1)) if len(rows) > 1 else 0.0,
                "pearson_r": (
                    float(np.corrcoef(ys, ps)[0, 1])
                    if len(rows) > 1 and np.std(ys) > 0 and np.std(ps) > 0
                    else float("nan")
                ),
            }
        groups = sorted(by_group)
        delta_bag = (
            by_group[groups[1]]["mean_bag"] - by_group[groups[0]]["mean_bag"]
            if len(groups) == 2 else float("nan")
        )
        holdout_metrics = {
            "by_group": by_group,
            "delta_bag_groups": f"{groups[1]} - {groups[0]}" if len(groups) == 2 else None,
            "delta_bag_mean": float(delta_bag),
            "selected_alpha": (
                nested_subject["selected_alpha"] if run_nested_cv else sweep_result["best_alpha"]
            ),
            "estimator_source": "ridge_subject_nested_cv" if run_nested_cv else "best_test",
            "n_total": int(len(holdout_predictions)),
        }
        for grp in groups:
            m = by_group[grp]
            _log(f"  Holdout[{grp}]: n={m['n']}  MAE={m['mae']:.4f}  "
                 f"BAG={m['mean_bag']:+.4f}+/-{m['std_bag']:.4f}  r={m['pearson_r']:.4f}")
        _log(f"  Holdout delta BAG ({holdout_metrics['delta_bag_groups']}) = "
             f"{delta_bag:+.4f}  ({time.time()-t0:.1f}s)")

    # --- Dataset age statistics ---------------------------------------------
    all_ages = np.concatenate([train_y, val_y, test_y])
    age_stats = {
        "mean": float(all_ages.mean()),
        "std": float(all_ages.std(ddof=0)),
        "min": float(all_ages.min()),
        "max": float(all_ages.max()),
        "median": float(np.median(all_ages)),
        "n_subjects": int(len(all_ages)),
    }

    # --- Assemble result ----------------------------------------------------
    # best_test / best_val / selected_alpha are the headline strategy's, which
    # is what `neuroatlas results` reads.
    if run_nested_cv:
        head_test = nested_subject["best_test"]
        head_val = nested_subject["best_val"]
        head_val_mae = nested_subject["best_val_mae"]
        head_alpha = nested_subject["selected_alpha"]
        subject_predictions = {"test": nested_subject["subject_predictions"]}
    else:
        head_test = sweep_result["best_test"]
        head_val = sweep_result["best_val"]
        head_val_mae = sweep_result["best_val_mae"]
        head_alpha = sweep_result["best_alpha"]
        subject_predictions = {
            "val": _get_subject_predictions(ridge_estimator, val_x, val_y, val_subjects),
            "test": _get_subject_predictions(ridge_estimator, test_x, test_y, test_subjects),
        }
    _log(f"Headline ({headline}): alpha={head_alpha}  MAE={head_test['mae']:.4f}  "
         f"r={head_test['pearson_r']:.4f}  R2={head_test['r2']:.4f}")
    if failures:
        _log(f"Strategies that failed (recorded in metrics.strategy_failures): "
             f"{', '.join(sorted(failures))}")
    _log(f"Fold {fold} complete in {time.time()-fold_t0:.1f}s")
    _log("=" * 60)
    metrics: Dict[str, Any] = {
        "best_test": head_test,
        "best_val": head_val,
        "best_val_metric": head_val_mae,
        "selection_metric": "mae",
        "headline_strategy": headline,
        "best_seed": 0,
        "probe_type": "linear",
        "selected_alpha": head_alpha,
        "alpha_grid": list(alpha_sweep_values),
        "subject_predictions": subject_predictions,
    }
    if sweep_result is not None:
        metrics["ridge_alpha_sweep"] = {
            k: v for k, v in sweep_result.items() if k != "best_estimator"
        }
    if ridge_subject_nested_cv_metrics is not None:
        # Key name is load-bearing: the paper's figures and LaTeX tables read
        # metrics.ridge_subject_nested_cv.best_test.mae — do not rename.
        metrics["ridge_subject_nested_cv"] = ridge_subject_nested_cv_metrics
    if mlp_subject is not None:
        metrics["mlp_subject"] = mlp_subject
    if epoch_regression is not None:
        metrics["epoch_regression"] = epoch_regression
    if failures:
        metrics["strategy_failures"] = failures
    if holdout_metrics is not None:
        metrics["holdout_metrics"] = holdout_metrics
        metrics["holdout_predictions"] = holdout_predictions

    return BenchmarkResult(
        checkpoint_id=checkpoint_spec.identifier,
        dataset_name=dataset_name,
        evaluation_mode="brain_age_eval",
        metrics=metrics,
        cache_paths=cache_paths,
        metadata={
            **backbone.metadata(),
            **dict(getattr(datamodule, "metadata", {})),
            "task_name": "brain_age",
            "aggregation": aggregation,
            "mode": getattr(datamodule, "metadata", {}).get("mode", ""),
            "age_stats": age_stats,
            "subject_counts": {
                "train": len(train_subjects),
                "val": len(val_subjects),
                "test": len(test_subjects),
            },
            **(
                {
                    # The folds came from the protocol, not from the cohort's
                    # sleep-staging fold manifest.
                    "folds_manifest": None,
                    "split_protocol": split_protocol_meta,
                    "unit": split_protocol_meta.get("aggregation_group"),
                }
                if split_protocol_meta is not None else {}
            ),
        },
    )


TASK_SPECS = [
    TaskSpec(
        slug="brain_age",
        description="Aggregate epoch embeddings per subject and train a subject-level age regressor.",
        evaluator=evaluate_brain_age,
    )
]
