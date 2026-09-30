"""
DCSM (Danish Center for Sleep Medicine) dataset loader for EEGBenchmarks.

Loads HDF5 PSG files and text hypnogram files directly at runtime.
Each subject directory contains psg.h5 (channels under /channels/<name>)
and hypnogram.ids (init,duration,stage CSV).

Provides DCSMDataset with lazy per-subject HDF5 caching, following the
same pattern as dataio/mass.py.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from scipy.signal import butter, filtfilt, firwin, iirnotch, sosfiltfilt
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DCSM_STAGES_MAP: Dict[str, int] = {
    "W": 0,
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
}

LABEL_NAMES = ["W", "N1", "N2", "N3", "REM"]

DEFAULT_RAW_ROOT = "${EEG_DATA_ROOT}/data/raw/dcsm/data/sleep/DCSM"

DEFAULT_CHANNELS: List[str] = [
    "C3-M2", "C4-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    h5_path: str
    hypnogram_path: str


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def scan_dcsm_subjects(
    data_root: str,
) -> List[SubjectRecord]:
    """Scan DCSM directory for subject directories with psg.h5 + hypnogram.ids."""
    records: List[SubjectRecord] = []
    root = Path(data_root)
    for entry in sorted(os.listdir(root)):
        subject_dir = root / entry
        if not subject_dir.is_dir():
            continue
        h5_path = subject_dir / "psg.h5"
        hyp_path = subject_dir / "hypnogram.ids"
        if h5_path.exists() and hyp_path.exists():
            records.append(SubjectRecord(
                subject_id=entry,
                h5_path=str(h5_path),
                hypnogram_path=str(hyp_path),
            ))
    return records


# ---------------------------------------------------------------------------
# Hypnogram reading
# ---------------------------------------------------------------------------

def read_dcsm_hypnogram(
    hypnogram_path: str,
    epoch_seconds: float = 30,
) -> Tuple[np.ndarray, float, float]:
    """Read sleep stage annotations from a DCSM hypnogram.ids file.

    Returns
    -------
    stages : ndarray of int16
        Per-epoch labels. Unscored epochs are -1.
    scores_start : float
        Onset of the first scored segment (seconds).
    scores_end : float
        End of the last scored segment (seconds).
    """
    segments: List[Tuple[float, float, int]] = []
    with open(hypnogram_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            init = float(parts[0])
            duration = float(parts[1])
            stage_str = parts[2].strip()
            stage = DCSM_STAGES_MAP.get(stage_str, -1)
            segments.append((init, duration, stage))

    if not segments:
        return np.array([], dtype=np.int16), 0.0, 0.0

    scores_start = segments[0][0]
    scores_end = max(init + duration for init, duration, _ in segments)

    # Build per-second label array relative to scores_start
    total_seconds = int(scores_end - scores_start)
    stages_per_second = np.full(total_seconds, -1, dtype=np.int16)
    for init, duration, stage in segments:
        s = int(init - scores_start)
        e = s + int(duration)
        stages_per_second[s:e] = stage

    # Collapse to per-epoch labels
    epoch_sec = int(epoch_seconds)
    n_epochs = len(stages_per_second) // epoch_sec
    stages: List[int] = []
    for i in range(n_epochs):
        epoch_labels = stages_per_second[i * epoch_sec : (i + 1) * epoch_sec]
        unique = set(epoch_labels.tolist())
        if len(unique) == 1:
            stages.append(unique.pop())
        else:
            stages.append(-1)

    return np.array(stages, dtype=np.int16), scores_start, scores_end


# ---------------------------------------------------------------------------
# Channel reading (HDF5)
# ---------------------------------------------------------------------------

def read_h5_channels(
    h5_path: str,
    channel_names: List[str],
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Read named channels from a DCSM psg.h5 file.

    Returns
    -------
    signals : ndarray, shape (n_channels, n_samples)
    sfreq : float
    """
    with h5py.File(h5_path, "r") as f:
        sfreq = float(f.attrs["sample_rate"])
        s_start = int(start_sec * sfreq)
        if duration_sec is not None:
            s_end = s_start + int(duration_sec * sfreq)
        else:
            s_end = None

        channel_data: List[np.ndarray] = []
        for name in channel_names:
            key = f"channels/{name}"
            if key not in f:
                # Missing channel — will be zero-filled
                if s_end is not None:
                    channel_data.append(np.zeros(s_end - s_start, dtype=np.float32))
                else:
                    channel_data.append(None)
                continue
            data = f[key][s_start:s_end]
            channel_data.append(data.astype(np.float32))

    # Zero-fill any None channels
    resolved = [c for c in channel_data if c is not None]
    if not resolved:
        raise ValueError(f"No channels could be resolved from {h5_path}.")
    min_samples = min(len(c) for c in resolved)
    for i, c in enumerate(channel_data):
        if c is None:
            channel_data[i] = np.zeros(min_samples, dtype=np.float32)

    stacked = np.stack([c[:min_samples] for c in channel_data], axis=0)
    return stacked, sfreq


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DCSMDataset(Dataset):
    """Epoch-level PyTorch dataset for the DCSM collection.

    Loads HDF5 PSG files lazily per subject and caches the processed
    signals in memory.

    Parameters
    ----------
    subject_records : list of SubjectRecord
        Subjects to include in this dataset split.
    channel_specs : list of str
        List of HDF5 channel names to load.
    epoch_seconds : float
        Duration of one epoch in seconds.
    bandpass : tuple of (low, high) or None
        Bandpass filter cutoffs in Hz.
    notch : float or None
        Notch filter frequency in Hz.
    fold_assignments : dict or None
        Per-subject fold assignments for global embedding cache.
    compute_recording_stats : bool
        If True, compute per-recording mean/std/q95 after filtering and
        cache them for downstream use (e.g. BENDR SCALE channel).
    """

    def __init__(
        self,
        subject_records: Sequence[SubjectRecord],
        channel_specs: List[str] = DEFAULT_CHANNELS,
        epoch_seconds: float = 30,
        bandpass: Optional[Tuple[float, float]] = None,
        notch: Optional[float] = None,
        highpass: Optional[float] = None,
        fold_assignments: Optional[Dict[str, Dict[str, str]]] = None,
        compute_recording_stats: bool = False,
        use_all_eeg_channels: bool = False,
    ) -> None:
        self._montage_channels = list(channel_specs)
        self._montage_names = list(channel_specs)
        self.channel_specs = channel_specs
        self.epoch_seconds = epoch_seconds
        self.bandpass = bandpass
        self.notch = notch
        self.highpass = highpass
        self._fold_assignments = fold_assignments or {}
        self._compute_recording_stats = compute_recording_stats
        self._use_all_eeg_channels = use_all_eeg_channels
        self._discovered_channels: Dict[str, List[str]] = {}

        self._subject_records = list(subject_records)
        self._records_by_id: Dict[str, SubjectRecord] = {}
        self._scores_range: Dict[str, Tuple[float, float]] = {}
        self._index: List[Tuple[str, int, int]] = []
        self._index_built = False
        self._h5_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._stats_cache: Dict[str, Optional[Dict[str, np.ndarray]]] = {}

    def _ensure_index_built(self) -> None:
        """Build epoch index lazily on first access (reads hypnogram files)."""
        if self._index_built:
            return
        for record in self._subject_records:
            self._records_by_id[record.subject_id] = record
            stages, scores_start, scores_end = read_dcsm_hypnogram(
                record.hypnogram_path, epoch_seconds=self.epoch_seconds,
            )
            self._scores_range[record.subject_id] = (scores_start, scores_end)
            for ep_idx, stage in enumerate(stages):
                self._index.append(
                    (record.subject_id, ep_idx, int(stage))
                )
        self._index_built = True

    def _load_subject_h5(
        self, subject_id: str,
    ) -> Tuple[np.ndarray, float]:
        """Load, filter, and cache a subject's PSG signals."""
        if subject_id in self._h5_cache:
            return self._h5_cache[subject_id]

        record = self._records_by_id[subject_id]
        scores_start, scores_end = self._scores_range[subject_id]
        duration = scores_end - scores_start

        if self._use_all_eeg_channels:
            from extensions.datasets.dataio._eeg_channel_discovery import (
                is_eeg_label,
            )
            with h5py.File(record.h5_path, "r") as f:
                all_keys = list(f["channels"].keys()) if "channels" in f else []
                eeg_keys = [k for k in all_keys if is_eeg_label(k)]
            if eeg_keys:
                self._discovered_channels[subject_id] = eeg_keys
                signals, sfreq = read_h5_channels(
                    record.h5_path, eeg_keys,
                    start_sec=scores_start, duration_sec=duration,
                )
            else:
                signals, sfreq = read_h5_channels(
                    record.h5_path, self._montage_channels,
                    start_sec=scores_start, duration_sec=duration,
                )
        else:
            signals, sfreq = read_h5_channels(
                record.h5_path, self._montage_channels,
                start_sec=scores_start, duration_sec=duration,
            )

        if self.highpass is not None:
            nyq = sfreq / 2.0
            sos = butter(5, self.highpass / nyq, btype="high", output="sos")
            signals = sosfiltfilt(sos, signals, axis=-1).astype(np.float32)

        if self.bandpass is not None:
            lo, hi = self.bandpass
            n_taps = 101
            b = firwin(n_taps, [lo, hi], pass_zero=False, fs=sfreq)
            signals = filtfilt(b, 1, signals, axis=-1).astype(np.float32)

        if self.notch is not None and self.notch < sfreq / 2.0:
            b, a = iirnotch(self.notch, Q=30.0, fs=sfreq)
            sos = np.array([[b[0], b[1], b[2], a[0], a[1], a[2]]])
            signals = sosfiltfilt(sos, signals, axis=-1).astype(np.float32)

        rec_stats: Optional[Dict[str, np.ndarray]] = None
        if self._compute_recording_stats and signals.size > 0:
            rec_stats = {
                "recording_mean": signals.mean(axis=1),
                "recording_std": signals.std(axis=1),
                "recording_q95": np.quantile(np.abs(signals), 0.95, axis=1),
            }
            n_ch = signals.shape[0]
            if n_ch > 1:
                bq = {}
                for i in range(n_ch):
                    for j in range(i + 1, n_ch):
                        bq[f"{i},{j}"] = float(
                            np.quantile(np.abs(signals[i] - signals[j]), 0.95)
                        )
                rec_stats["recording_q95_bipolar"] = bq

        self._h5_cache.clear()
        self._stats_cache.clear()
        self._h5_cache[subject_id] = (signals, sfreq)
        self._stats_cache[subject_id] = rec_stats
        return signals, sfreq

    def evict_subject(self, subject_id: str) -> None:
        self._h5_cache.pop(subject_id, None)
        self._stats_cache.pop(subject_id, None)

    @staticmethod
    def eviction_key(meta: Dict[str, object]) -> Optional[str]:
        sid = meta.get("subject_id")
        return str(sid) if sid is not None else None

    def __len__(self) -> int:
        self._ensure_index_built()
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        self._ensure_index_built()
        subject_id, ep_idx, sleep_stage = self._index[idx]
        signals, sfreq = self._load_subject_h5(subject_id)

        n_samples = int(round(self.epoch_seconds * sfreq))
        start_sample = ep_idx * n_samples
        epoch = signals[:, start_sample : start_sample + n_samples]
        if epoch.shape[1] < n_samples:
            epoch = np.pad(epoch, ((0, 0), (0, n_samples - epoch.shape[1])))

        sample: Dict[str, object] = {
            "eeg": torch.tensor(epoch, dtype=torch.float32),
            "sleep_stage": sleep_stage,
            "subject_id": subject_id,
            "epoch_idx": ep_idx,
            "channels": self._discovered_channels.get(subject_id, self._montage_names),
            "unit": "uV",
            "location": "Copenhagen, Denmark",
        }
        rec_stats = self._stats_cache.get(subject_id)
        if rec_stats is not None:
            for k, v in rec_stats.items():
                sample[k] = v
        if subject_id in self._fold_assignments:
            sample["fold_assignments"] = self._fold_assignments[subject_id]
        return sample

    @property
    def subjects(self) -> List[str]:
        return [r.subject_id for r in self._subject_records]
