"""On-the-fly windowed Dataset for SeizeIt1 continuous HDF5 caches.

Reads the pre-built continuous HDF5 (19 unipolar channels @ 256 Hz) and
materialises fixed-length windows at ``__getitem__`` time.  Same schema as
the CHB-MIT / TUSZ / EPILEPSIAE / Siena / SZ2 dataio modules.

Adds explicit ``meta["unit"] = "uV"`` and ``meta["sampling_rate"] = 256``
per MODEL_CONTRACTS §0.  SZ1 is the first seizure dataset to declare the
unit proactively; SZ2 and earlier datasets rely on the wrapper fallback.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from extensions.datasets._recording_stats import compute_recording_stats
from extensions.datasets.epilepsy.sz1_preprocessor import (
    BIPOLAR_NAMES,
    CANONICAL_19,
    CANONICAL_IDX,
    GAP_LABEL,
    TARGET_FS,
    bipolar_from_unipolar,
)

logger = logging.getLogger(__name__)


class SeizeIt1ContinuousDataset(Dataset):
    """Windowed view over a SeizeIt1 continuous HDF5 cache.

    Args:
        h5_path: Path to the continuous HDF5 cache file.
        window_s: Window duration in seconds.
        stride_s: Stride in seconds.  ``None`` -> non-overlapping (= window_s).
        recording_indices: If given, only include windows from these recording
            indices (for k-fold filtering).
        label_mode: ``"binary"`` (seizure vs background).
        normalize: ``"none"`` or ``"per_window_zscore"``.
        montage: ``"unipolar"`` (19 ch) or ``"bipolar"`` (20 ch TCP).
        overlap_threshold: Seizure fraction above which the binary label is 1.
    """

    def __init__(
        self,
        h5_path: str,
        window_s: float = 10.0,
        stride_s: Optional[float] = None,
        recording_indices: Optional[Sequence[int]] = None,
        label_mode: str = "binary",
        normalize: str = "none",
        montage: str = "unipolar",
        overlap_threshold: float = 0.0,
    ) -> None:
        self._h5_path = str(h5_path)
        self._window_s = window_s
        self._stride_s = stride_s if stride_s is not None else window_s
        self._label_mode = label_mode
        self._normalize = normalize
        self._montage = montage
        self._overlap_threshold = overlap_threshold

        self._window_samples = int(round(window_s * TARGET_FS))
        self._stride_samples = int(round(self._stride_s * TARGET_FS))

        self._h5: Optional[h5py.File] = None

        with h5py.File(self._h5_path, "r") as f:
            self._offsets = f["recording_offsets"][:].astype(np.int64)
            self._recording_ids = _decode_vlen(f["recording_ids"][:])
            self._subject_ids = _decode_vlen(f["subject_ids"][:])
            self._durations_s = f["durations_s"][:].astype(np.float32)
            self._n_missing = f["n_missing_channels"][:].astype(np.int16)
            self._channel_mask = f["channel_mask"][:].astype(bool)
            self._samplewise_label = f["samplewise_label"][:]

        n_recordings = len(self._recording_ids)

        rec_set = (
            set(recording_indices)
            if recording_indices is not None
            else set(range(n_recordings))
        )

        index: List[Tuple[int, int]] = []
        for rec_i in range(n_recordings):
            if rec_i not in rec_set:
                continue
            rec_start = int(self._offsets[rec_i])
            rec_end = int(self._offsets[rec_i + 1])
            if rec_end - rec_start < self._window_samples:
                continue

            pos = rec_start
            while pos + self._window_samples <= rec_end:
                wl = self._samplewise_label[pos: pos + self._window_samples]
                if not np.all(wl == GAP_LABEL):
                    index.append((rec_i, pos))
                pos += self._stride_samples

        self._index = (
            np.array(index, dtype=np.int64)
            if index
            else np.zeros((0, 2), dtype=np.int64)
        )
        self._active_rec_indices = sorted(rec_set & set(range(n_recordings)))

        # Lazy per-recording amplitude-stats cache (MODEL_CONTRACTS §3).
        # Computed on the unipolar source before any bipolar derivation.
        self._rec_stats_cache: Dict[int, Dict[str, Any]] = {}

        logger.info(
            "SeizeIt1ContinuousDataset: %s — %d recordings, %d windows "
            "(%.1fs @ %.1fs stride)",
            self._h5_path,
            len(self._active_rec_indices),
            len(self._index),
            self._window_s,
            self._stride_s,
        )

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None or not self._h5.id.valid:
            self._h5 = h5py.File(self._h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_i, abs_start = int(self._index[idx, 0]), int(self._index[idx, 1])
        abs_end = abs_start + self._window_samples

        signals = self.h5["signals"][abs_start:abs_end].T.astype(np.float32)  # (19, T)

        # Lazily compute and cache per-recording stats on the unipolar source
        # (before any bipolar derivation) so BIOT/REVE/SleepFM wrappers have
        # recording_mean, recording_std, recording_q95 (MODEL_CONTRACTS §3).
        if rec_i not in self._rec_stats_cache:
            rec_start_sample = int(self._offsets[rec_i])
            rec_end_sample = int(self._offsets[rec_i + 1])
            rec_full = self.h5["signals"][rec_start_sample:rec_end_sample].T.astype(np.float32)
            self._rec_stats_cache[rec_i] = compute_recording_stats(rec_full)

        if self._montage == "bipolar":
            signals = bipolar_from_unipolar(signals)  # (20, T)

        if self._normalize == "per_window_zscore":
            mean = signals.mean(axis=1, keepdims=True)
            std = signals.std(axis=1, keepdims=True)
            std[std < 1e-8] = 1.0
            signals = (signals - mean) / std

        window_labels = self._samplewise_label[abs_start:abs_end]
        valid_mask = window_labels != GAP_LABEL
        n_valid = valid_mask.sum()
        seizure_count = ((window_labels == 1) & valid_mask).sum()
        seizure_fraction = float(seizure_count / max(n_valid, 1))
        binary_label = int(seizure_fraction > self._overlap_threshold)

        rec_start_sample = int(self._offsets[rec_i])
        window_start_in_rec = abs_start - rec_start_sample
        window_start_s = window_start_in_rec / TARGET_FS
        window_end_s = window_start_s + self._window_s

        ch_names = list(BIPOLAR_NAMES if self._montage == "bipolar" else CANONICAL_19)
        meta: Dict[str, Any] = {
            "dataset": "sz1",
            "split": "",
            "subject_id": self._subject_ids[rec_i],
            "recording_id": self._recording_ids[rec_i],
            "recording_idx": rec_i,
            "window_start_s": float(window_start_s),
            "window_end_s": float(window_end_s),
            "recording_duration_s": float(self._durations_s[rec_i]),
            "seizure_fraction": seizure_fraction,
            "has_seizure": bool(binary_label),
            "binary_label": binary_label,
            "n_missing_channels": int(self._n_missing[rec_i]),
            "channels": ch_names,
            "montage": self._montage,
            "normalize": self._normalize,
            # MODEL_CONTRACTS §0: explicit unit declaration
            "unit": "uV",
            "sampling_rate": TARGET_FS,
            "fs": TARGET_FS,
            "window_s": self._window_s,
            "stride_s": self._stride_s,
        }
        meta.update(self._rec_stats_cache[rec_i])

        return {
            "eeg": torch.from_numpy(signals),
            "label": binary_label,
            "meta": meta,
        }

    @property
    def subject_ids(self) -> List[str]:
        return list({self._subject_ids[i] for i in self._active_rec_indices})

    @property
    def n_recordings(self) -> int:
        return len(self._recording_ids)

    def binary_labels(self) -> np.ndarray:
        labels = np.zeros(len(self._index), dtype=np.int64)
        for i in range(len(self._index)):
            abs_start = int(self._index[i, 1])
            window = self._samplewise_label[
                abs_start: abs_start + self._window_samples
            ]
            valid = window[window != GAP_LABEL]
            if len(valid) > 0:
                frac = (valid == 1).sum() / len(valid)
                labels[i] = int(frac > self._overlap_threshold)
        return labels

    def __del__(self) -> None:
        if self._h5 is not None and self._h5.id.valid:
            self._h5.close()


def _decode_vlen(arr: np.ndarray) -> List[str]:
    return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in arr]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def _collate_sz1(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for SeizeIt1ContinuousDataset batches."""
    eeg = torch.stack([item["eeg"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "eeg": eeg,
        "labels": labels,
        "meta": [item["meta"] for item in batch],
    }
