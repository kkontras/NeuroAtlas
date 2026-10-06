"""Attention probe tasks based on Efficient Probing (Psomas et al., ICLR 2026).

Two evaluation modes:

* ``attention_probe`` — non-aggregating (sleep staging): builds a temporal
  context window around each epoch and classifies per epoch.
* ``attention_probe_patient`` — aggregating (diagnosis): attends over all
  epoch embeddings per subject to produce a single subject-level prediction.

Both reuse pre-extracted embedding caches and auto-register via TASK_SPECS.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from neuroatlas.benchmarking_helpers import BenchmarkResult, EmbeddingPayload, TaskSpec
from neuroatlas.benchmarking_helpers.runtime.cache import (
    cache_exists,
    load_embedding_payload,
)
from neuroatlas.benchmarking_helpers.probes.metrics import compute_classification_metrics
from neuroatlas.benchmarking_helpers.probes.probe import ProbeResult
from neuroatlas.extensions.tasks.linear_probe import (
    _dataset_context,
    _embedding_cache_dir,
    _extract_or_load_embeddings,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EP cross-attention module (adapted from billpsomas/efficient-probing)
# ---------------------------------------------------------------------------

class EfficientProbe(nn.Module):
    """Lightweight cross-attention pooling following Psomas et al.

    Learnable query tokens attend over input tokens using identity keys
    and a single value projection.  Each query produces a disjoint slice
    of the output, which are concatenated to form the final descriptor.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        num_queries: int = 32,
        d_out: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.d_out = d_out
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.d_per_query = max(1, (dim // d_out) // num_queries)
        self.d_v = self.d_per_query * num_queries

        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.v = nn.Linear(dim, self.d_v, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input token embeddings.
            mask: (B, N) boolean — True for valid positions.

        Returns:
            (B, d_v) pooled descriptor.
        """
        B, N, C = x.shape

        q = self.cls_token.expand(B, -1, -1)
        q = q.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)  # (B, H, num_queries, N)

        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        v = self.v(x)  # (B, N, d_v)
        v = v.reshape(B, N, self.num_queries, self.d_per_query).permute(0, 2, 1, 3)
        # v: (B, num_queries, N, d_per_query)

        # attn: (B, H, num_queries, N) — with H=1 default
        out = torch.matmul(attn.squeeze(1).unsqueeze(2), v)
        # out: (B, num_queries, 1, d_per_query)
        out = out.reshape(B, self.d_v)
        return out


class _AttentionProbeHead(nn.Module):
    """EP pooling + BatchNorm + Linear classifier."""

    def __init__(self, dim: int, n_classes: int, num_heads: int = 1,
                 num_queries: int = 32, d_out: int = 1):
        super().__init__()
        self.ep = EfficientProbe(dim, num_heads=num_heads,
                                 num_queries=num_queries, d_out=d_out)
        d_v = self.ep.d_v
        self.bn = nn.BatchNorm1d(d_v, affine=False, eps=1e-6)
        self.classifier = nn.Linear(d_v, n_classes)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.ep(x, mask=mask)
        out = self.bn(out)
        return self.classifier(out)


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def _recording_key(m: dict) -> str:
    return str(m.get("recording_id") or m.get("subject_id", "unknown"))


def _drop_nonfinite(feats: np.ndarray, labs: np.ndarray, name: str):
    finite = np.isfinite(feats).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        print(f"[attention_probe] dropping {n_bad}/{len(feats)} non-finite {name} rows")
        return feats[finite], labs[finite], finite
    return feats, labs, finite


def _build_window_index(
    payload: EmbeddingPayload, window_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build index arrays for lazy windowed feature construction.

    Instead of materialising the full (N, W, D) array, returns lightweight
    index arrays.  Windows are gathered on-the-fly per mini-batch during
    training, reducing memory from O(N*W*D) to O(N*D + N*W).

    Returns:
        win_idx: (N, window_size) int32 — indices into payload.features
        masks:   (N, window_size) bool — True for genuine positions
        labels:  (N,) int64
    """
    labels = np.asarray(payload.labels)
    metadata = payload.metadata
    half = window_size // 2

    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, m in enumerate(metadata):
        groups[_recording_key(m)].append(idx)

    for key in groups:
        groups[key].sort(key=lambda i: metadata[i].get("epoch_index", i))

    N = len(metadata)
    win_idx = np.empty((N, window_size), dtype=np.int32)
    masks = np.zeros((N, window_size), dtype=bool)
    out_labels = np.empty(N, dtype=np.int64)

    for rec_indices in groups.values():
        L = len(rec_indices)

        for pos, global_idx in enumerate(rec_indices):
            start = pos - half

            for w_pos in range(window_size):
                src = start + w_pos
                clamped = max(0, min(L - 1, src))
                win_idx[global_idx, w_pos] = rec_indices[clamped]
                masks[global_idx, w_pos] = 0 <= src < L

            out_labels[global_idx] = labels[global_idx]

    return win_idx, masks, out_labels


def _build_subject_sequences(
    payload: EmbeddingPayload, max_epochs: Optional[int] = None,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """Group epoch embeddings by subject for aggregating attention probe.

    Returns:
        sequences: list of (N_i, D) arrays, one per subject
        labels:    (N_subjects,) int64
        subjects:  list of subject_id strings
    """
    features = np.asarray(payload.features)
    labels = np.asarray(payload.labels)
    metadata = payload.metadata

    grouped: Dict[str, Dict[str, Any]] = {}
    for idx, m in enumerate(metadata):
        sid = str(m.get("subject_id", "unknown"))
        if sid not in grouped:
            grouped[sid] = {"indices": [], "label": int(labels[idx])}
        grouped[sid]["indices"].append(idx)

    sequences: List[np.ndarray] = []
    out_labels: List[int] = []
    subjects: List[str] = []

    for sid in sorted(grouped):
        info = grouped[sid]
        indices = sorted(info["indices"], key=lambda i: metadata[i].get("epoch_index", i))
        feats = features[indices].astype(np.float32)

        if max_epochs is not None and len(feats) > max_epochs:
            step = len(feats) / max_epochs
            selected = [int(i * step) for i in range(max_epochs)]
            feats = feats[selected]

        sequences.append(feats)
        out_labels.append(info["label"])
        subjects.append(sid)

    return sequences, np.asarray(out_labels, dtype=np.int64), subjects


def _collate_subject_batch(
    batch_features: List[np.ndarray], batch_labels: List[int], device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length subject sequences to uniform length within batch."""
    max_len = max(f.shape[0] for f in batch_features)
    D = batch_features[0].shape[1]
    B = len(batch_features)
    padded = torch.zeros(B, max_len, D, device=device)
    masks = torch.zeros(B, max_len, dtype=torch.bool, device=device)
    for i, f in enumerate(batch_features):
        n = f.shape[0]
        padded[i, :n] = torch.as_tensor(f, device=device)
        masks[i, :n] = True
    labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return padded, masks, labels


# ---------------------------------------------------------------------------
# Torch attention probe (sklearn-compatible fit/predict interface)
# ---------------------------------------------------------------------------

class _TorchAttentionProbe:
    """Attention probe trained on GPU with mini-batch SGD."""

    def __init__(
        self,
        dim: int,
        n_classes: int,
        seed: int = 0,
        max_epochs: int = 300,
        batch_size: int = 64,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        patience: int = 50,
        device: str = "cuda",
        class_weight: Optional[str] = None,
        num_heads: int = 1,
        num_queries: int = 32,
        d_out: int = 1,
    ):
        self.dim = dim
        self.n_classes = n_classes
        self.seed = seed
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self.class_weight = class_weight
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.d_out = d_out
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        self._model: Optional[_AttentionProbeHead] = None

    # --- windowed (non-aggregating) -----------------------------------------

    def fit_windowed(
        self,
        features: np.ndarray, win_idx: np.ndarray, y: np.ndarray, masks: np.ndarray,
        features_val: np.ndarray, win_idx_val: np.ndarray, y_val: np.ndarray, masks_val: np.ndarray,
    ):
        """Train with lazy window construction from (N, D) features + (N, W) index.

        Windows are gathered on-the-fly per mini-batch, avoiding the
        O(N*W*D) memory cost of materialising the full windowed array.
        """
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
        return self._predict_impl(features, win_idx, masks)

    def predict_proba_windowed(self, features: np.ndarray, win_idx: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return self._predict_proba_impl(features, win_idx, masks)

    # --- variable-length (aggregating) --------------------------------------

    def fit_variable(
        self,
        sequences: List[np.ndarray], y: np.ndarray,
        sequences_val: List[np.ndarray], y_val: np.ndarray,
    ):
        """Train on variable-length per-subject sequences."""
        torch.manual_seed(self.seed)

        all_feats = np.concatenate(sequences, axis=0)
        self._mean = torch.as_tensor(
            all_feats.mean(axis=0), dtype=torch.float32, device=self.device,
        )
        self._std = torch.as_tensor(
            all_feats.std(axis=0), dtype=torch.float32, device=self.device,
        ).clamp(min=1e-8)
        del all_feats

        N = len(sequences)
        self._build_model()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        weight_tensor = self._class_weights(y)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.max_epochs):
            self._model.train()
            perm = torch.randperm(N)
            for start in range(0, N, self.batch_size):
                idx = perm[start:start + self.batch_size].tolist()
                if len(idx) < 2:
                    continue
                batch_seqs = [sequences[i] for i in idx]
                batch_labels = [int(y[i]) for i in idx]
                xb, mb, yb = self._collate_and_normalise(batch_seqs, batch_labels)
                optimizer.zero_grad()
                criterion(self._model(xb, mask=mb), yb).backward()
                optimizer.step()

            val_loss = self._eval_loss_variable(sequences_val, y_val, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

    def predict_variable(self, sequences: List[np.ndarray]) -> np.ndarray:
        self._model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                batch = sequences[start:start + self.batch_size]
                labels_dummy = [0] * len(batch)
                xb, mb, _ = self._collate_and_normalise(batch, labels_dummy)
                preds.append(self._model(xb, mask=mb).argmax(dim=1).cpu())
        return torch.cat(preds).numpy()

    def predict_proba_variable(self, sequences: List[np.ndarray]) -> np.ndarray:
        self._model.eval()
        probs = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                batch = sequences[start:start + self.batch_size]
                labels_dummy = [0] * len(batch)
                xb, mb, _ = self._collate_and_normalise(batch, labels_dummy)
                probs.append(torch.softmax(self._model(xb, mask=mb), dim=1).cpu())
        return torch.cat(probs).numpy()

    # --- internal -----------------------------------------------------------

    def _gather_batch(self, features: np.ndarray,
                      batch_win_idx: np.ndarray) -> torch.Tensor:
        """Gather a batch of windows from features and normalise.

        Args:
            features: (N, D) embedding array.
            batch_win_idx: (B, W) int32 indices into features.
        Returns:
            (B, W, D) normalised tensor on self.device.
        """
        xb = torch.as_tensor(
            np.ascontiguousarray(features[batch_win_idx]),
            dtype=torch.float32,
        )
        return ((xb - self._mean_cpu) / self._std_cpu).to(self.device)

    def _collate_and_normalise(
        self, sequences: List[np.ndarray], labels: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded, masks, y = _collate_subject_batch(sequences, labels, torch.device(self.device))
        padded = (padded - self._mean) / self._std
        padded = padded * masks.unsqueeze(-1)
        return padded, masks, y

    def _build_model(self):
        self._model = _AttentionProbeHead(
            self.dim, self.n_classes,
            num_heads=self.num_heads, num_queries=self.num_queries, d_out=self.d_out,
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
        """Windows gathered on-the-fly from features + win_idx per batch."""
        self._build_model()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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

            val_loss = self._eval_loss_windowed(
                features_val, win_idx_val, y_val, masks_val, criterion,
            )
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if epoch % 10 == 0 or improved or wait >= self.patience:
                print(f"  epoch {epoch:3d}  val_loss={val_loss:.4f}  best={best_val_loss:.4f}  wait={wait}")

            if wait >= self.patience:
                break

        print(f"  stopped at epoch {epoch}, best_val_loss={best_val_loss:.4f}")
        if best_state is not None:
            self._model.load_state_dict(best_state)

    def _eval_loss_windowed(self, features, win_idx, y, masks, criterion) -> float:
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

    def _eval_loss_variable(self, sequences, y, criterion) -> float:
        self._model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                batch = sequences[start:start + self.batch_size]
                labels = [int(y[i]) for i in range(start, min(start + self.batch_size, len(sequences)))]
                xb, mb, yb = self._collate_and_normalise(batch, labels)
                total += criterion(self._model(xb, mask=mb), yb).item() * len(batch)
                count += len(batch)
        return total / max(count, 1)

    def _predict_impl(self, features: np.ndarray, win_idx: np.ndarray,
                      masks: np.ndarray) -> np.ndarray:
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

    def _predict_proba_impl(self, features: np.ndarray, win_idx: np.ndarray,
                            masks: np.ndarray) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# Multi-seed training (mirrors train_probe() output contract)
# ---------------------------------------------------------------------------

def _read_ep_kwargs(task_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_heads": int(task_config.get("num_heads", 1)),
        "num_queries": int(task_config.get("num_queries", 32)),
        "d_out": int(task_config.get("d_out", 1)),
        "max_epochs": int(task_config.get("max_train_epochs", 300)),
        "batch_size": int(task_config.get("attention_batch_size", 64)),
        "lr": float(task_config.get("attention_lr", 3e-4)),
        "weight_decay": float(task_config.get("attention_weight_decay", 1e-4)),
        "patience": int(task_config.get("attention_patience", 50)),
    }


def _train_attention_probe_windowed(
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
    **ep_kwargs,
) -> ProbeResult:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_binary = n_classes == 2

    per_seed: List[Dict[str, Any]] = []
    estimators: Dict[int, _TorchAttentionProbe] = {}
    test_outputs: List[Dict[str, Any]] = []

    print(
        f"[attention_probe] fitting windowed probe on "
        f"train={len(train_y)} val={len(val_y)} test={len(test_y)} "
        f"with {len(seeds)} seed(s), dim={dim}, window={window_size}"
    )

    for seed in tqdm(seeds, desc="Attention probe seeds", leave=True):
        probe = _TorchAttentionProbe(
            dim=dim, n_classes=n_classes, seed=seed,
            device=device, class_weight=class_weight, **ep_kwargs,
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
        try:
            test_proba = probe.predict_proba_windowed(test_features, test_idx, test_masks)
        except Exception:
            test_proba = None

        val_metrics = compute_classification_metrics(val_y, val_pred, y_score=val_score)
        test_metrics = compute_classification_metrics(test_y, test_pred, y_score=test_score)

        estimators[seed] = probe
        per_seed.append({"seed": seed, "val": val_metrics, "test": test_metrics})
        test_outputs.append({"y_pred": np.asarray(test_pred), "y_score": test_score,
                             "y_proba": test_proba})
        print(
            f"[attention_probe] seed={seed} "
            f"val_{selection_metric}={float(val_metrics[selection_metric]):.6f} "
            f"test_accuracy={float(test_metrics['accuracy']):.6f}"
        )

    return _aggregate_seeds(per_seed, estimators, selection_metric, test_y=test_y,
                            test_outputs=test_outputs, n_classes=n_classes)


def _train_attention_probe_variable(
    train_seqs, train_y,
    val_seqs, val_y,
    test_seqs, test_y,
    *,
    seeds: List[int],
    dim: int,
    n_classes: int,
    selection_metric: str = "macro_f1",
    class_weight: Optional[str] = None,
    **ep_kwargs,
) -> ProbeResult:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_binary = n_classes == 2

    per_seed: List[Dict[str, Any]] = []
    estimators: Dict[int, _TorchAttentionProbe] = {}
    test_outputs: List[Dict[str, Any]] = []

    print(
        f"[attention_probe] fitting variable-length probe on "
        f"train={len(train_y)} val={len(val_y)} test={len(test_y)} subjects "
        f"with {len(seeds)} seed(s), dim={dim}"
    )

    for seed in tqdm(seeds, desc="Attention probe seeds", leave=True):
        probe = _TorchAttentionProbe(
            dim=dim, n_classes=n_classes, seed=seed,
            device=device, class_weight=class_weight, **ep_kwargs,
        )
        probe.fit_variable(train_seqs, train_y, val_seqs, val_y)

        val_pred = probe.predict_variable(val_seqs)
        test_pred = probe.predict_variable(test_seqs)

        val_score = test_score = None
        if is_binary:
            try:
                val_score = probe.predict_proba_variable(val_seqs)[:, 1]
                test_score = probe.predict_proba_variable(test_seqs)[:, 1]
            except Exception:
                pass
        try:
            test_proba = probe.predict_proba_variable(test_seqs)
        except Exception:
            test_proba = None

        val_metrics = compute_classification_metrics(val_y, val_pred, y_score=val_score)
        test_metrics = compute_classification_metrics(test_y, test_pred, y_score=test_score)

        estimators[seed] = probe
        per_seed.append({"seed": seed, "val": val_metrics, "test": test_metrics})
        test_outputs.append({"y_pred": np.asarray(test_pred), "y_score": test_score,
                             "y_proba": test_proba})
        print(
            f"[attention_probe] seed={seed} "
            f"val_{selection_metric}={float(val_metrics[selection_metric]):.6f} "
            f"test_accuracy={float(test_metrics['accuracy']):.6f}"
        )

    return _aggregate_seeds(per_seed, estimators, selection_metric, test_y=test_y,
                            test_outputs=test_outputs, n_classes=n_classes)


def _aggregate_seeds(per_seed, estimators, selection_metric, test_y=None, test_outputs=None,
                     n_classes=None):
    best_row = max(per_seed, key=lambda r: float(r["val"][selection_metric]))
    best_seed = int(best_row["seed"])
    print(
        f"[attention_probe] selected best_seed={best_seed} "
        f"best_val_{selection_metric}={float(best_row['val'][selection_metric]):.6f}"
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
    summary["probe_type"] = "attention"
    return ProbeResult(
        metrics=summary,
        per_seed=per_seed,
        best_seed=best_seed,
        best_val_metric=float(best_row["val"][selection_metric]),
        estimator=estimators[best_seed],
        test_y=None if test_y is None else np.asarray(test_y),
        test_outputs=test_outputs,
        classes=None if n_classes is None else np.arange(int(n_classes)),
    )


# ---------------------------------------------------------------------------
# Embedding loading (shared by both evaluators)
# ---------------------------------------------------------------------------

def _load_split_payloads(
    dataset_name, checkpoint_spec, datamodule, backbone, cache_root,
):
    """Load train/val/test embedding payloads, reusing the global cache."""
    use_global = (
        datamodule.supports_global_embedding_cache()
        and getattr(datamodule, "limit_windows_per_split", None) is None
    )
    cache_paths: Dict[str, str] = {}

    if use_global:
        global_dir = _embedding_cache_dir(
            cache_root, dataset_name, checkpoint_spec, "all",
            datamodule, purpose="global_embeddings",
        )
        if cache_exists(global_dir):
            full = load_embedding_payload(global_dir, mmap_mode="r")
            cache_paths["global_cache_dir"] = str(global_dir)
        else:
            full, paths = _extract_or_load_embeddings(
                cache_root, dataset_name, "all", checkpoint_spec, backbone,
                datamodule.full_embedding_dataloader(), datamodule,
                cache_purpose="global_embeddings",
            )
            cache_paths.update(paths)
        splits = datamodule.split_global_embedding_payload(full)
        return splits["train"], splits["val"], splits["test"], cache_paths

    train, tp = _extract_or_load_embeddings(
        cache_root, dataset_name, "train", checkpoint_spec, backbone,
        datamodule.train_dataloader(), datamodule,
    )
    val, vp = _extract_or_load_embeddings(
        cache_root, dataset_name, "val", checkpoint_spec, backbone,
        datamodule.val_dataloader(), datamodule,
    )
    test, xp = _extract_or_load_embeddings(
        cache_root, dataset_name, "test", checkpoint_spec, backbone,
        datamodule.test_dataloader(), datamodule,
    )
    for prefix, paths in [("train", tp), ("val", vp), ("test", xp)]:
        cache_paths.update({f"{prefix}_{k}": v for k, v in paths.items()})
    return train, val, test, cache_paths


# ---------------------------------------------------------------------------
# Task evaluators
# ---------------------------------------------------------------------------

def evaluate_attention_probe(
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
    """Non-aggregating attention probe: one prediction per epoch."""

    window_size = int(task_config.get("window_size", 21))
    selection_metric = str(probe_config.get("selection_metric", "macro_f1"))
    class_weight = probe_config.get("class_weight")
    ep_kwargs = _read_ep_kwargs(task_config)

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

    probe_result = _train_attention_probe_windowed(
        train_feats, train_idx, train_y, train_masks,
        val_feats, val_idx, val_y, val_masks,
        test_feats, test_idx, test_y, test_masks,
        seeds=list(seeds),
        dim=dim,
        n_classes=n_classes,
        window_size=window_size,
        selection_metric=selection_metric,
        class_weight=class_weight,
        **ep_kwargs,
    )

    # The fold's test predictions (one row per test epoch), saved; the metrics
    # recorded are score() of that file, which `neuroatlas rescore` recomputes.
    from neuroatlas import predictions as preds

    record = preds.new(
        "attention_probe", dataset_name=dataset_name, checkpoint_spec=checkpoint_spec,
        datamodule=datamodule,
        columns={**preds.probe_columns(probe_result), **preds.id_columns(test_meta)},
        info=preds.make_info(
            datamodule=datamodule, checkpoint_spec=checkpoint_spec, probe_config=probe_config,
            task_config=task_config, seeds=seeds, cache_paths=cache_paths,
            fit=preds.probe_fit(probe_result, selection_metric=selection_metric,
                                probe_type="attention")))
    metrics, saved, path = preds.finalize(record, probe_dir, score)
    if path is not None:
        cache_paths["predictions"] = str(path)
    return BenchmarkResult(
        checkpoint_id=checkpoint_spec.identifier,
        dataset_name=dataset_name,
        evaluation_mode="attention_probe_eval",
        metrics=metrics,
        cache_paths=cache_paths,
        metadata={
            **backbone.metadata(),
            **_dataset_context(datamodule),
            "task_name": "attention_probe",
            "window_size": window_size,
            "num_queries": ep_kwargs["num_queries"],
            "probe_seeds": list(seeds),
            "per_seed": preds.per_seed(saved),
        },
    )


def evaluate_attention_probe_patient(
    *,
    dataset_name: str,
    checkpoint_spec,
    datamodule,
    backbone,
    probe_config: Dict[str, Any],
    task_config: Dict[str, Any],
    seeds,
    cache_root: Path,
    probe_dir: Optional[Path] = None,
    **_,
) -> BenchmarkResult:
    """Aggregating attention probe: one prediction per subject."""

    max_epochs_per_subject = task_config.get("max_epochs_per_subject")
    if max_epochs_per_subject is not None:
        max_epochs_per_subject = int(max_epochs_per_subject)
    selection_metric = str(probe_config.get("selection_metric", "macro_f1"))
    class_weight = probe_config.get("class_weight")
    ep_kwargs = _read_ep_kwargs(task_config)

    train_p, val_p, test_p, cache_paths = _load_split_payloads(
        dataset_name, checkpoint_spec, datamodule, backbone, Path(cache_root),
    )

    train_seqs, train_y, train_subjects = _build_subject_sequences(train_p, max_epochs_per_subject)
    val_seqs, val_y, val_subjects = _build_subject_sequences(val_p, max_epochs_per_subject)
    test_seqs, test_y, test_subjects = _build_subject_sequences(test_p, max_epochs_per_subject)

    dim = train_seqs[0].shape[1]
    n_classes = int(max(train_y.max(), val_y.max(), test_y.max())) + 1

    probe_result = _train_attention_probe_variable(
        train_seqs, train_y,
        val_seqs, val_y,
        test_seqs, test_y,
        seeds=list(seeds),
        dim=dim,
        n_classes=n_classes,
        selection_metric=selection_metric,
        class_weight=class_weight,
        **ep_kwargs,
    )

    # One row per test subject, saved; the metrics recorded are score() of
    # that file, which `neuroatlas rescore` recomputes.
    from neuroatlas import predictions as preds

    record = preds.new(
        "attention_probe_patient", dataset_name=dataset_name, checkpoint_spec=checkpoint_spec,
        datamodule=datamodule,
        columns={**preds.probe_columns(probe_result),
                 "subject_id": np.asarray([str(s) for s in test_subjects], dtype=str)},
        info=preds.make_info(
            datamodule=datamodule, checkpoint_spec=checkpoint_spec, probe_config=probe_config,
            task_config=task_config, seeds=seeds, cache_paths=cache_paths,
            fit=preds.probe_fit(probe_result, selection_metric=selection_metric,
                                probe_type="attention")))
    metrics, _saved, path = preds.finalize(record, probe_dir, score)
    if path is not None:
        cache_paths["predictions"] = str(path)
    return BenchmarkResult(
        checkpoint_id=checkpoint_spec.identifier,
        dataset_name=dataset_name,
        evaluation_mode="attention_probe_patient_eval",
        metrics=metrics,
        cache_paths=cache_paths,
        metadata={
            **backbone.metadata(),
            **dict(getattr(datamodule, "metadata", {})),
            "task_name": "attention_probe_patient",
            "num_queries": ep_kwargs["num_queries"],
            "max_epochs_per_subject": max_epochs_per_subject,
            "subject_counts": {
                "train": len(train_subjects),
                "val": len(val_subjects),
                "test": len(test_subjects),
            },
        },
    )


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def score(pred) -> Dict[str, Any]:
    """The attention probe's metrics (per epoch or per subject) from a fold's
    saved predictions."""
    from neuroatlas import predictions as preds

    return preds.score_probe(pred.group(""), pred.fit)[0]


TASK_SPECS = [
    TaskSpec(
        slug="attention_probe",
        description="Context-window attention probe for per-epoch classification (sleep staging).",
        evaluator=evaluate_attention_probe,
        score=score,
    ),
    TaskSpec(
        slug="attention_probe_patient",
        description="Attention-pooling probe for subject-level classification (diagnosis).",
        evaluator=evaluate_attention_probe_patient,
        score=score,
    ),
]
