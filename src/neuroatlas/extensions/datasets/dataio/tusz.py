"""On-the-fly windowed Dataset for TUSZ continuous HDF5 caches.

Reads the pre-built continuous HDF5 (19 unipolar channels @ 256 Hz) and
materialises fixed-length windows at ``__getitem__`` time.  Supports multiple
``label_mode`` values for seizure detection, onset, demographics, etc.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

# Re-export key constants so downstream code can import from one place.
from neuroatlas.extensions.datasets.epilepsy.tusz_preprocessor import (
    BIPOLAR_MONTAGE,
    SEIZURE_TYPE_CODES,
    SEIZURE_TYPE_NAMES,
    TARGET_FS,
    TCP_MONTAGE_CHANNELS,
    UNIPOLAR_ELECTRODES,
    bipolar_from_unipolar,
)

_SEX_CODE: Dict[str, int] = {"M": 0, "F": 1}
_EPILEPSY_CODE: Dict[str, int] = {"epilepsy": 1, "no_epilepsy": 0}

_VALID_LABEL_MODES = (
    "binary", "multiclass", "onset", "missing_channels", "age", "sex", "epilepsy",
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _has_onset_transition(samplewise_label: np.ndarray) -> bool:
    """Return True if there is a 0->1 transition in the binary label array."""
    if len(samplewise_label) < 2:
        return False
    diff = np.diff(samplewise_label.astype(np.int8))
    return bool(np.any(diff > 0))


def _majority_nonbckg_type(samplewise_type: np.ndarray) -> int:
    """Return the most common non-background seizure type code in the window.

    Returns 0 (background) if no seizure samples exist.
    """
    nonzero = samplewise_type[samplewise_type > 0]
    if len(nonzero) == 0:
        return 0
    counts = Counter(nonzero.tolist())
    return counts.most_common(1)[0][0]


def _multiclass_fractions(samplewise_type: np.ndarray) -> Dict[str, float]:
    """Return a dict mapping seizure type names to their fractional presence in the window."""
    n = len(samplewise_type)
    if n == 0:
        return {}
    unique, counts = np.unique(samplewise_type, return_counts=True)
    result: Dict[str, float] = {}
    for code, count in zip(unique.tolist(), counts.tolist()):
        if code < len(SEIZURE_TYPE_NAMES):
            result[SEIZURE_TYPE_NAMES[code]] = count / n
    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TUSZContinuousDataset(Dataset):
    """Windowed view over a TUSZ continuous HDF5 cache.

    Windows are computed on the fly from ``recording_offsets`` + ``window_s`` /
    ``stride_s``.  Each window stays within a single recording boundary.

    Args:
        h5_path: Path to a continuous HDF5 cache file.
        window_s: Window duration in seconds.
        stride_s: Stride in seconds.  Defaults to ``window_s`` (no overlap).
        recording_indices: If given, only include windows from these recording indices
            (for k-fold filtering).
        label_mode: One of ``"binary"``, ``"multiclass"``, ``"onset"``,
            ``"missing_channels"``, ``"age"``, ``"sex"``, ``"epilepsy"``.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        montage: ``"unipolar"`` or ``"bipolar"``.
    """

    def __init__(
        self,
        h5_path: str,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        recording_indices: Optional[Sequence[int]] = None,
        label_mode: str = "binary",
        normalize: str = "none",
        montage: str = "unipolar",
    ) -> None:
        if label_mode not in _VALID_LABEL_MODES:
            raise ValueError(f"Unsupported label_mode {label_mode!r}. Expected one of {_VALID_LABEL_MODES}")

        self.h5_path = str(h5_path)
        self.window_s = float(window_s)
        self.stride_s = float(stride_s) if stride_s is not None else self.window_s
        self.label_mode = label_mode
        self.normalize = normalize
        self.montage = montage
        self.window_samples = int(round(self.window_s * TARGET_FS))
        self.stride_samples = int(round(self.stride_s * TARGET_FS))

        # Open HDF5 and read lightweight metadata
        self._h5 = h5py.File(self.h5_path, "r")
        self._recording_offsets = self._h5["recording_offsets"][:]
        n_total_rec = len(self._recording_offsets) - 1

        # Per-recording metadata (load once, relatively small)
        self._recording_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in self._h5["recording_ids"][:]]
        self._subject_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in self._h5["subject_ids"][:]]
        self._session_ids = (
            [s.decode() if isinstance(s, bytes) else str(s) for s in self._h5["session_ids"][:]]
            if "session_ids" in self._h5 else [""] * n_total_rec
        )
        self._durations_s = self._h5["durations_s"][:]
        self._n_missing = self._h5["n_missing_electrodes"][:]
        self._native_fs = (
            self._h5["native_fs"][:] if "native_fs" in self._h5
            else np.full(n_total_rec, TARGET_FS, dtype=np.float32)
        )
        self._montage_types = (
            [s.decode() if isinstance(s, bytes) else str(s) for s in self._h5["montage_types"][:]]
            if "montage_types" in self._h5 else [""] * n_total_rec
        )
        self._ages = self._h5["ages"][:]
        self._sexes = [s.decode() if isinstance(s, bytes) else str(s) for s in self._h5["sexes"][:]]
        self._epilepsy_diag = [s.decode() if isinstance(s, bytes) else str(s) for s in self._h5["epilepsy_diagnosis"][:]]
        # Per-channel EDF physical dimension, shape (n_rec, 19). Legacy caches
        # (pre physical_units) default to all "uV" — the implicit assumption
        # before the field existed.
        if "physical_units" in self._h5:
            self._physical_units = self._h5["physical_units"][()]
        else:
            self._physical_units = np.full(
                (n_total_rec, len(UNIPOLAR_ELECTRODES)),
                b"uV", dtype="S8",
            )

        # Filter recordings
        if recording_indices is not None:
            valid_indices = [i for i in recording_indices if 0 <= i < n_total_rec]
        else:
            valid_indices = list(range(n_total_rec))

        # Drop recordings that are too short for one window
        dropped = 0
        kept_indices: List[int] = []
        for idx in valid_indices:
            rec_len = int(self._recording_offsets[idx + 1] - self._recording_offsets[idx])
            if rec_len >= self.window_samples:
                kept_indices.append(idx)
            else:
                dropped += 1
        if dropped > 0:
            logger.info(
                "Dropped %d recordings shorter than %.1fs window (kept %d)",
                dropped, self.window_s, len(kept_indices),
            )

        # Store active recording indices for subject_ids property
        self._active_rec_indices = kept_indices

        # Build window index: list of (recording_idx, sample_start_within_recording)
        self._windows: List[Tuple[int, int]] = []
        for rec_idx in kept_indices:
            rec_start = int(self._recording_offsets[rec_idx])
            rec_end = int(self._recording_offsets[rec_idx + 1])
            rec_len = rec_end - rec_start
            n_windows = max(1, (rec_len - self.window_samples) // self.stride_samples + 1)
            for w in range(n_windows):
                offset = w * self.stride_samples
                if offset + self.window_samples <= rec_len:
                    self._windows.append((rec_idx, offset))

        logger.info(
            "TUSZContinuousDataset: %s — %d recordings, %d windows (%.1fs @ %.1fs stride, label_mode=%s)",
            self.h5_path, len(kept_indices), len(self._windows),
            self.window_s, self.stride_s, self.label_mode,
        )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec_idx, local_offset = self._windows[index]
        global_start = int(self._recording_offsets[rec_idx]) + local_offset
        global_end = global_start + self.window_samples

        # Read signals
        signals = self._h5["signals"][global_start:global_end]  # (window_samples, 19)

        # Montage
        if self.montage == "bipolar":
            signals = bipolar_from_unipolar(signals)

        # Normalize
        if self.normalize == "per_window_zscore":
            mean = signals.mean(axis=0, keepdims=True)
            std = signals.std(axis=0, keepdims=True)
            std[std < 1e-8] = 1.0
            signals = (signals - mean) / std

        # Transpose to (channels, time) for PyTorch convention
        eeg = torch.as_tensor(signals.T, dtype=torch.float32)  # (C, T)

        # Read samplewise labels for this window
        sw_label = self._h5["samplewise_label"][global_start:global_end]
        sw_type = self._h5["samplewise_type"][global_start:global_end]

        # Compute window-level statistics
        seizure_fraction = float(np.mean(sw_label > 0))
        has_seizure = seizure_fraction > 0
        binary_label = int(has_seizure)
        multiclass_label = _majority_nonbckg_type(sw_type)
        seizure_onset = _has_onset_transition(sw_label)
        mc_fractions = _multiclass_fractions(sw_type)

        # Determine label based on mode
        if self.label_mode == "binary":
            label = binary_label
        elif self.label_mode == "multiclass":
            label = multiclass_label
        elif self.label_mode == "onset":
            label = int(seizure_onset)
        elif self.label_mode == "missing_channels":
            label = int(self._n_missing[rec_idx])
        elif self.label_mode == "age":
            age = int(self._ages[rec_idx])
            label = age if age >= 0 else -1
        elif self.label_mode == "sex":
            sex_str = self._sexes[rec_idx]
            label = _SEX_CODE.get(sex_str, -1)
        elif self.label_mode == "epilepsy":
            diag = self._epilepsy_diag[rec_idx]
            label = _EPILEPSY_CODE.get(diag, -1)
        else:
            label = binary_label

        # Per-recording metadata
        window_end_s = (local_offset + self.window_samples) / TARGET_FS
        recording_duration_s = float(self._durations_s[rec_idx])

        # Channel names — needed by some backbones (e.g. BENDR, REVE)
        # to route per-channel inputs into their pretrained layout.
        if self.montage == "bipolar":
            channel_names = list(BIPOLAR_MONTAGE)
        else:
            channel_names = list(UNIPOLAR_ELECTRODES)

        meta: Dict[str, Any] = {
            "dataset": "tusz",
            "recording_id": self._recording_ids[rec_idx],
            "subject_id": self._subject_ids[rec_idx],
            "session_id": self._session_ids[rec_idx],
            "recording_index": rec_idx,
            "window_index": index,
            "channels": channel_names,
            "sfreq": float(TARGET_FS),
            "sampling_rate": float(TARGET_FS),
            "unit": "uV",
            "native_fs": float(self._native_fs[rec_idx]),
            "montage_type": self._montage_types[rec_idx],
            "seizure_fraction": seizure_fraction,
            "has_seizure": has_seizure,
            "binary_label": binary_label,
            "multiclass_label": multiclass_label,
            "seizure_onset_in_window": seizure_onset,
            "multiclass_fractions": mc_fractions,
            "recording_duration_s": recording_duration_s,
            "window_end_s": window_end_s,
            "n_missing_electrodes": int(self._n_missing[rec_idx]),
            "age": int(self._ages[rec_idx]),
            "sex": self._sexes[rec_idx],
            "epilepsy_diagnosis": self._epilepsy_diag[rec_idx],
            "physical_units": [
                u.decode() if isinstance(u, (bytes, bytearray)) else str(u)
                for u in self._physical_units[rec_idx]
            ],
        }

        return {
            "signals": {"eeg": eeg},
            "label": torch.tensor(label, dtype=torch.long),
            "meta": meta,
        }

    def close(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5 = None

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def subject_ids(self) -> List[str]:
        """Subject IDs for the active recordings (respects recording_indices)."""
        return list({self._subject_ids[i] for i in self._active_rec_indices})

    @property
    def n_recordings(self) -> int:
        return int(self._recording_offsets.shape[0] - 1)

    # ------------------------------------------------------------------
    # Vectorised label access (no windowed I/O needed)
    # ------------------------------------------------------------------

    def binary_labels(self) -> np.ndarray:
        """Return binary labels for all windows via cumulative sum trick."""
        labels = np.zeros(len(self._windows), dtype=np.int64)
        # Read the full samplewise_label cumsum approach
        cum = None
        for i, (rec_idx, local_offset) in enumerate(self._windows):
            gs = int(self._recording_offsets[rec_idx]) + local_offset
            ge = gs + self.window_samples
            chunk = self._h5["samplewise_label"][gs:ge]
            labels[i] = 1 if np.any(chunk > 0) else 0
        return labels

    def all_labels(self) -> np.ndarray:
        """Return labels for all windows according to the current label_mode."""
        if self.label_mode == "binary":
            return self.binary_labels()

        labels = np.zeros(len(self._windows), dtype=np.int64)
        for i, (rec_idx, local_offset) in enumerate(self._windows):
            if self.label_mode in ("age", "sex", "epilepsy", "missing_channels"):
                # Per-recording label — no I/O needed
                if self.label_mode == "age":
                    age = int(self._ages[rec_idx])
                    labels[i] = age if age >= 0 else -1
                elif self.label_mode == "sex":
                    labels[i] = _SEX_CODE.get(self._sexes[rec_idx], -1)
                elif self.label_mode == "epilepsy":
                    labels[i] = _EPILEPSY_CODE.get(self._epilepsy_diag[rec_idx], -1)
                elif self.label_mode == "missing_channels":
                    labels[i] = int(self._n_missing[rec_idx])
            elif self.label_mode == "multiclass":
                gs = int(self._recording_offsets[rec_idx]) + local_offset
                ge = gs + self.window_samples
                sw_type = self._h5["samplewise_type"][gs:ge]
                labels[i] = _majority_nonbckg_type(sw_type)
            elif self.label_mode == "onset":
                gs = int(self._recording_offsets[rec_idx]) + local_offset
                ge = gs + self.window_samples
                sw_label = self._h5["samplewise_label"][gs:ge]
                labels[i] = int(_has_onset_transition(sw_label))
        return labels


# Alias
TUSZWindowedDataset = TUSZContinuousDataset


# ---------------------------------------------------------------------------
# Collate + DataLoader factory
# ---------------------------------------------------------------------------


def _collate_tusz(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for TUSZContinuousDataset batches."""
    eeg = torch.stack([item["signals"]["eeg"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    return {
        "signals": {"eeg": eeg},
        "label": labels,
        "meta": [item["meta"] for item in batch],
        "raw_batch": batch,
    }


def build_loaders(
    h5_path: str,
    window_s: float = 30.0,
    stride_s: Optional[float] = None,
    label_mode: str = "binary",
    normalize: str = "none",
    montage: str = "unipolar",
    batch_size: int = 64,
    num_workers: int = 4,
    recording_indices_train: Optional[Sequence[int]] = None,
    recording_indices_val: Optional[Sequence[int]] = None,
    recording_indices_test: Optional[Sequence[int]] = None,
    balance: str = "none",
) -> Dict[str, DataLoader]:
    """Build train/val/test DataLoaders for a TUSZ cache.

    Args:
        balance: ``"none"`` or ``"weighted_sampler"`` to oversample minority class.
    """
    loaders: Dict[str, DataLoader] = {}
    for split_name, rec_indices, shuffle in [
        ("train", recording_indices_train, True),
        ("val", recording_indices_val, False),
        ("test", recording_indices_test, False),
    ]:
        ds = TUSZContinuousDataset(
            h5_path=h5_path,
            window_s=window_s,
            stride_s=stride_s,
            recording_indices=rec_indices,
            label_mode=label_mode,
            normalize=normalize,
            montage=montage,
        )
        sampler = None
        if split_name == "train" and balance == "weighted_sampler":
            labels = ds.binary_labels()
            counts = np.bincount(labels)
            if len(counts) >= 2 and counts.min() > 0:
                weights = 1.0 / counts[labels]
                sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
                shuffle = False

        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=_collate_tusz,
            pin_memory=False,
        )
    return loaders
