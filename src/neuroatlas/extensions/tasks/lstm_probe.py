"""Bidirectional LSTM probe for per-epoch sleep-staging classification.

Builds a temporal context window around each epoch (default 21 epochs =
10 before + center + 10 after) and classifies the center epoch using a
bidirectional LSTM.  Reuses pre-extracted embedding caches and the same
windowing infrastructure as the attention probe.

Auto-registers via TASK_SPECS.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from neuroatlas.benchmarking_helpers import BenchmarkResult, EmbeddingPayload, TaskSpec
from neuroatlas.benchmarking_helpers.probes.metrics import compute_classification_metrics
from neuroatlas.benchmarking_helpers.probes.probe import ProbeResult
from neuroatlas.extensions.tasks.attention_probe import (
    _build_window_index,
    _drop_nonfinite,
    _load_split_payloads,
)
from neuroatlas.extensions.tasks.linear_probe import _dataset_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------------------

class _LSTMProbeHead(nn.Module):
    """BiLSTM encoder that classifies the center epoch of a context window."""

    def __init__(self, dim: int, n_classes: int, hidden_size: int = 128,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bn = nn.BatchNorm1d(hidden_size * 2, affine=False, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, W, D) windowed embeddings.
            mask: (B, W) boolean — True for genuine positions (unused by LSTM
                  but accepted for interface compatibility with attention probe).
        Returns:
            (B, n_classes) logits for the center epoch.
        """
        # x: (B, W, D) -> lstm_out: (B, W, 2*hidden)
        lstm_out, _ = self.lstm(x)
        center = x.shape[1] // 2
        h = lstm_out[:, center, :]  # (B, 2*hidden)
        h = self.bn(h)
        h = self.drop(h)
        return self.classifier(h)


# ---------------------------------------------------------------------------
# Torch LSTM probe (sklearn-compatible fit/predict interface)
# ---------------------------------------------------------------------------

class _TorchLSTMProbe:
    """LSTM probe trained on GPU with mini-batch SGD."""

    def __init__(
        self,
        dim: int,
        n_classes: int,
        seed: int = 0,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        max_epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        device: str = "cuda",
        class_weight: Optional[str] = None,
    ):
        self.dim = dim
        self.n_classes = n_classes
        self.seed = seed
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self.class_weight = class_weight
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        self._model: Optional[_LSTMProbeHead] = None

    # --- windowed interface ---------------------------------------------------

    def fit_windowed(
        self,
        features: np.ndarray, win_idx: np.ndarray, y: np.ndarray, masks: np.ndarray,
        features_val: np.ndarray, win_idx_val: np.ndarray, y_val: np.ndarray, masks_val: np.ndarray,
    ):
        torch.manual_seed(self.seed)

        self._mean = torch.as_tensor(
            features.mean(axis=0), dtype=torch.float32, device=self.device,
        )
        self._std = torch.as_tensor(
            features.std(axis=0), dtype=torch.float32, device=self.device,
        ).clamp(min=1e-8)
        self._mean_cpu = self._mean.cpu()
        self._std_cpu = self._std.cpu()

        N = len(y)
        y_t = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.long)
        m_t = torch.as_tensor(np.ascontiguousarray(masks), dtype=torch.bool)
        y_v = torch.as_tensor(np.ascontiguousarray(y_val), dtype=torch.long)
        m_v = torch.as_tensor(np.ascontiguousarray(masks_val), dtype=torch.bool)

        self._train_loop(
            features, win_idx, y_t, m_t,
            features_val, win_idx_val, y_v, m_v, N,
        )

    def predict_windowed(self, features: np.ndarray, win_idx: np.ndarray, masks: np.ndarray) -> np.ndarray:
        self._model.eval()
        preds = []
        masks_t = torch.as_tensor(np.ascontiguousarray(masks), dtype=torch.bool)
        with torch.no_grad():
            for start in range(0, len(win_idx), self.batch_size):
                end = min(start + self.batch_size, len(win_idx))
                xb = self._gather_batch(features, win_idx[start:end])
                mb = masks_t[start:end].to(self.device)
                preds.append(self._model(xb, mask=mb).argmax(dim=1).cpu())
        return torch.cat(preds).numpy()

    def predict_proba_windowed(self, features: np.ndarray, win_idx: np.ndarray, masks: np.ndarray) -> np.ndarray:
        self._model.eval()
        probs = []
        masks_t = torch.as_tensor(np.ascontiguousarray(masks), dtype=torch.bool)
        with torch.no_grad():
            for start in range(0, len(win_idx), self.batch_size):
                end = min(start + self.batch_size, len(win_idx))
                xb = self._gather_batch(features, win_idx[start:end])
                mb = masks_t[start:end].to(self.device)
                probs.append(torch.softmax(self._model(xb, mask=mb), dim=1).cpu())
        return torch.cat(probs).numpy()

    # --- internal -------------------------------------------------------------

    def _gather_batch(self, features: np.ndarray, batch_win_idx: np.ndarray) -> torch.Tensor:
        xb = torch.as_tensor(
            np.ascontiguousarray(features[batch_win_idx]),
            dtype=torch.float32,
        )
        return ((xb - self._mean_cpu) / self._std_cpu).to(self.device)

    def _build_model(self):
        self._model = _LSTMProbeHead(
            self.dim, self.n_classes,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _class_weights(self, y: np.ndarray) -> Optional[torch.Tensor]:
        if self.class_weight != "balanced":
            return None
        y_t = torch.as_tensor(y, dtype=torch.long, device=self.device)
        counts = torch.bincount(y_t, minlength=self.n_classes).float()
        return (counts.sum() / (self.n_classes * counts.clamp(min=1))).to(self.device)

    def _train_loop(
        self,
        features: np.ndarray, win_idx: np.ndarray,
        y: torch.Tensor, masks: torch.Tensor,
        features_val: np.ndarray, win_idx_val: np.ndarray,
        y_val: torch.Tensor, masks_val: torch.Tensor,
        N: int,
    ):
        self._build_model()
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )
        weight_tensor = self._class_weights(y.numpy())
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.max_epochs):
            self._model.train()
            perm = torch.randperm(N).numpy()
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                idx = perm[start:end]
                if len(idx) < 2:
                    continue
                xb = self._gather_batch(features, win_idx[idx])
                yb = y[idx].to(self.device)
                mb = masks[idx].to(self.device)
                optimizer.zero_grad()
                criterion(self._model(xb, mask=mb), yb).backward()
                optimizer.step()

            val_loss = self._eval_loss(
                features_val, win_idx_val, y_val, masks_val, criterion,
            )
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            scheduler.step(val_loss)
            cur_lr = optimizer.param_groups[0]["lr"]

            if epoch % 10 == 0 or improved or wait >= self.patience:
                print(f"  epoch {epoch:3d}  val_loss={val_loss:.4f}  best={best_val_loss:.4f}  wait={wait}  lr={cur_lr:.1e}")

            if wait >= self.patience:
                break

        print(f"  stopped at epoch {epoch}, best_val_loss={best_val_loss:.4f}")
        if best_state is not None:
            self._model.load_state_dict(best_state)

    def _eval_loss(self, features, win_idx, y, masks, criterion) -> float:
        self._model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(y), self.batch_size):
                end = min(start + self.batch_size, len(y))
                xb = self._gather_batch(features, win_idx[start:end])
                yb = y[start:end].to(self.device)
                mb = masks[start:end].to(self.device)
                total += criterion(self._model(xb, mask=mb), yb).item() * (end - start)
                count += end - start
        return total / max(count, 1)


# ---------------------------------------------------------------------------
# Multi-seed training
# ---------------------------------------------------------------------------

def _read_lstm_kwargs(task_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hidden_size": int(task_config.get("lstm_hidden_size", 128)),
        "num_layers": int(task_config.get("lstm_num_layers", 1)),
        "dropout": float(task_config.get("lstm_dropout", 0.3)),
        "max_epochs": int(task_config.get("lstm_max_train_epochs", 100)),
        "batch_size": int(task_config.get("lstm_batch_size", 64)),
        "lr": float(task_config.get("lstm_lr", 1e-3)),
        "weight_decay": float(task_config.get("lstm_weight_decay", 1e-4)),
        "patience": int(task_config.get("lstm_patience", 15)),
    }


def _aggregate_seeds(per_seed, estimators, selection_metric):
    best_row = max(per_seed, key=lambda r: float(r["val"][selection_metric]))
    best_seed = int(best_row["seed"])
    best_val_sm = float(best_row["val"][selection_metric])
    print(
        f"[lstm_probe] selected best_seed={best_seed} "
        f"best_val_{selection_metric}={best_val_sm:.6f}"
    )
    metric_names = ["accuracy", "macro_f1", "weighted_f1", "cohen_kappa"]
    summary: Dict[str, Any] = {}
    for name in metric_names:
        values = np.asarray([r["test"][name] for r in per_seed], dtype=float)
        summary[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    summary["confusion_matrix"] = per_seed[0]["test"]["confusion_matrix"]
    summary["best_seed"] = best_seed
    summary["best_val_metric"] = float(best_row["val"][selection_metric])
    summary["best_val"] = best_row["val"]
    summary["best_test"] = best_row["test"]
    summary["probe_type"] = "lstm"
    return ProbeResult(
        metrics=summary,
        per_seed=per_seed,
        best_seed=best_seed,
        best_val_metric=float(best_row["val"][selection_metric]),
        estimator=estimators[best_seed],
    )


def _train_lstm_probe_windowed(
    train_features, train_idx, train_y, train_masks,
    val_features, val_idx, val_y, val_masks,
    test_features, test_idx, test_y, test_masks,
    *,
    seeds: List[int],
    dim: int,
    n_classes: int,
    window_size: int = 21,
    selection_metric: str = "macro_f1",
    class_weight: Optional[str] = None,
    **lstm_kwargs,
) -> ProbeResult:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_binary = n_classes == 2

    per_seed: List[Dict[str, Any]] = []
    estimators: Dict[int, _TorchLSTMProbe] = {}

    print(
        f"[lstm_probe] fitting windowed probe on "
        f"train={len(train_y)} val={len(val_y)} test={len(test_y)} "
        f"with {len(seeds)} seed(s), dim={dim}, window={window_size}"
    )

    for seed in tqdm(seeds, desc="LSTM probe seeds", leave=True):
        probe = _TorchLSTMProbe(
            dim=dim, n_classes=n_classes, seed=seed,
            device=device, class_weight=class_weight, **lstm_kwargs,
        )
        probe.fit_windowed(
            train_features, train_idx, train_y, train_masks,
            val_features, val_idx, val_y, val_masks,
        )

        val_pred = probe.predict_windowed(val_features, val_idx, val_masks)
        test_pred = probe.predict_windowed(test_features, test_idx, test_masks)

        val_score = test_score = None
        if is_binary:
            try:
                val_score = probe.predict_proba_windowed(val_features, val_idx, val_masks)[:, 1]
                test_score = probe.predict_proba_windowed(test_features, test_idx, test_masks)[:, 1]
            except Exception:
                pass

        val_metrics = compute_classification_metrics(val_y, val_pred, y_score=val_score)
        test_metrics = compute_classification_metrics(test_y, test_pred, y_score=test_score)

        estimators[seed] = probe
        per_seed.append({"seed": seed, "val": val_metrics, "test": test_metrics})
        print(
            f"[lstm_probe] seed={seed} "
            f"val_{selection_metric}={float(val_metrics[selection_metric]):.6f} "
            f"test_accuracy={float(test_metrics['accuracy']):.6f}"
        )

    return _aggregate_seeds(per_seed, estimators, selection_metric)


# ---------------------------------------------------------------------------
# Task evaluator
# ---------------------------------------------------------------------------

def evaluate_lstm_probe(
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
    """Windowed bidirectional LSTM probe: one prediction per epoch."""

    window_size = int(task_config.get("window_size", 21))
    selection_metric = str(probe_config.get("selection_metric", "macro_f1"))
    class_weight = probe_config.get("class_weight")
    lstm_kwargs = _read_lstm_kwargs(task_config)

    train_p, val_p, test_p, cache_paths = _load_split_payloads(
        dataset_name, checkpoint_spec, datamodule, backbone, Path(cache_root),
    )

    label_mode = datamodule.metadata.get("label_mode", "sleep_stage")
    if label_mode == "sleep_stage" or label_mode.startswith("scorer_"):
        for p in (train_p, val_p, test_p):
            mask = np.asarray(p.labels) >= 0
            p.features = np.asarray(p.features)[mask]
            p.labels = np.asarray(p.labels)[mask]
            p.metadata = [m for m, keep in zip(p.metadata, mask) if keep]

    train_feats, train_labels, train_finite = _drop_nonfinite(
        np.asarray(train_p.features), np.asarray(train_p.labels), "train")
    val_feats, val_labels, val_finite = _drop_nonfinite(
        np.asarray(val_p.features), np.asarray(val_p.labels), "val")
    test_feats, test_labels, test_finite = _drop_nonfinite(
        np.asarray(test_p.features), np.asarray(test_p.labels), "test")

    train_meta = [m for m, keep in zip(train_p.metadata, train_finite) if keep]
    val_meta = [m for m, keep in zip(val_p.metadata, val_finite) if keep]
    test_meta = [m for m, keep in zip(test_p.metadata, test_finite) if keep]

    train_p_clean = EmbeddingPayload(train_feats, train_labels, train_meta)
    val_p_clean = EmbeddingPayload(val_feats, val_labels, val_meta)
    test_p_clean = EmbeddingPayload(test_feats, test_labels, test_meta)

    train_idx, train_masks, train_y = _build_window_index(train_p_clean, window_size)
    val_idx, val_masks, val_y = _build_window_index(val_p_clean, window_size)
    test_idx, test_masks, test_y = _build_window_index(test_p_clean, window_size)

    dim = train_feats.shape[1]
    n_classes = int(max(train_y.max(), val_y.max(), test_y.max())) + 1

    probe_result = _train_lstm_probe_windowed(
        train_feats, train_idx, train_y, train_masks,
        val_feats, val_idx, val_y, val_masks,
        test_feats, test_idx, test_y, test_masks,
        seeds=list(seeds),
        dim=dim,
        n_classes=n_classes,
        window_size=window_size,
        selection_metric=selection_metric,
        class_weight=class_weight,
        **lstm_kwargs,
    )

    return BenchmarkResult(
        checkpoint_id=checkpoint_spec.identifier,
        dataset_name=dataset_name,
        evaluation_mode="lstm_probe_eval",
        metrics=probe_result.metrics,
        cache_paths=cache_paths,
        metadata={
            **backbone.metadata(),
            **_dataset_context(datamodule),
            "task_name": "lstm_probe",
            "window_size": window_size,
            "hidden_size": lstm_kwargs["hidden_size"],
            "num_layers": lstm_kwargs["num_layers"],
            "probe_seeds": list(seeds),
            "per_seed": probe_result.per_seed,
        },
    )


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

TASK_SPECS = [
    TaskSpec(
        slug="lstm_probe",
        description="Bidirectional LSTM probe for per-epoch classification with temporal context.",
        evaluator=evaluate_lstm_probe,
    ),
]
