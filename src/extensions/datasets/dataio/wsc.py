"""
WSC (Wisconsin Sleep Cohort) dataset loader for EEGBenchmarks.

Loads raw EDF files via edfio at runtime. Annotations from .stg.txt
(epoch-per-line) or .allscore.txt (timestamp-based) files. Age metadata
from the WSC CSV dataset file.

Provides WSCDataset with lazy per-subject EDF caching.
"""

from __future__ import annotations

import datetime
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import edfio
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, firwin, iirnotch, sosfiltfilt
from torch.utils.data import Dataset

from extensions.datasets.dataio._stage_resample import (
    resample_native_epoch_labels,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stage mapping for .stg.txt files
WSC_STG_MAP: Dict[int, int] = {
    0: 0,   # Wake
    1: 1,   # N1
    2: 2,   # N2
    3: 3,   # N3
    4: 3,   # N3 (old S4)
    5: 4,   # REM
    6: -1,  # Movement
    7: -1,  # Unscored
}

# Stage mapping for .allscore.txt files
WSC_ALLSCORE_MAP: Dict[str, int] = {
    "W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4,
    "NO STAGE": -1, "MVT": -1,
}

LABEL_NAMES = ["W", "N1", "N2", "N3", "REM"]

DEFAULT_RAW_ROOT = "${EEG_DATA_ROOT}/data/raw/wsc/polysomnography"
DEFAULT_CSV_PATH = "${EEG_DATA_ROOT}/data/raw/wsc/datasets/wsc-dataset-0.8.0.csv"

DEFAULT_CHANNELS: List[str] = [
    "C3_M2", "C4_M1", "Cz_M2", "F3_M2", "F4_M1", "O1_M2", "O2_M1", "Pz_M2",
]

_WSC_REFERENCE_ALIASES: Dict[str, List[str]] = {
    "C3_M2": ["C3_M1", "C3_AVG"],
    "C4_M1": ["C4_M2", "C4_AVG"],
    "Cz_M2": ["Cz_M1", "Cz_AVG"],
    "F3_M2": ["F3_M1", "F3_AVG"],
    "F4_M1": ["F4_M2", "F4_AVG"],
    "O1_M2": ["O1_M1", "O1_AVG"],
    "O2_M1": ["O2_M2", "O2_AVG"],
    "Pz_M2": ["Pz_M1", "Pz_AVG"],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str       # wsc_id (e.g. "10119")
    visit: int            # visit number (1-5)
    recording_id: str     # unique: "{subject_id}_visit{visit}"
    edf_path: str
    annotation_path: str  # .stg.txt or .allscore.txt
    age: Optional[float]
    sex: Optional[str] = None


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def scan_wsc_subjects(
    data_root: str,
    csv_path: str = DEFAULT_CSV_PATH,
    visits: Optional[Sequence[int]] = None,
) -> List[SubjectRecord]:
    """Scan WSC directory for EDF + annotation file pairs.

    Parses filenames like ``wsc-visit1-10119-nsrr.edf`` to extract
    subject_id and visit. Joins with the CSV for age metadata.
    """
    root = Path(data_root)
    edf_files = sorted(f for f in os.listdir(root) if f.endswith(".edf"))

    # Load demographic metadata
    age_lookup: Dict[Tuple[str, int], float] = {}
    sex_lookup: Dict[Tuple[str, int], str] = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, usecols=["wsc_id", "wsc_vst", "age", "sex"], low_memory=False)
        for _, row in df.iterrows():
            key = (str(int(row["wsc_id"])), int(row["wsc_vst"]))
            age_lookup[key] = float(row["age"])
            raw_sex = row["sex"]
            if pd.notna(raw_sex):
                sex_lookup[key] = str(raw_sex).strip()

    records: List[SubjectRecord] = []
    pattern = re.compile(r"wsc-visit(\d+)-(\d+)-nsrr\.edf")

    for edf_name in edf_files:
        match = pattern.match(edf_name)
        if not match:
            continue
        visit = int(match.group(1))
        subject_id = match.group(2)

        if visits is not None and visit not in visits:
            continue

        base = edf_name.replace(".edf", "")
        stg_path = root / f"{base}.stg.txt"
        allscore_path = root / f"{base}.allscore.txt"

        if allscore_path.exists():
            ann_path = str(allscore_path)
        elif stg_path.exists():
            ann_path = str(stg_path)
        else:
            continue

        key = (subject_id, visit)
        age = age_lookup.get(key)
        sex = sex_lookup.get(key)

        records.append(SubjectRecord(
            subject_id=subject_id,
            visit=visit,
            recording_id=f"{subject_id}_visit{visit}",
            edf_path=str(root / edf_name),
            annotation_path=ann_path,
            age=age,
            sex=sex,
        ))

    return records


# ---------------------------------------------------------------------------
# Annotation reading
# ---------------------------------------------------------------------------

_STG_NATIVE_EPOCH = 30  # .stg.txt files always use 30-second epochs


def _read_stg_annotations(
    stg_path: str,
    epoch_seconds: float = 30,
) -> Tuple[np.ndarray, float, float]:
    """Parse a .stg.txt file (epoch-per-line with stage codes).

    The file always stores one label per 30-second epoch.  When
    ``epoch_seconds`` differs, labels are duplicated (shorter epochs)
    or merged (longer epochs).
    """
    with open(stg_path) as f:
        lines = f.readlines()

    # Skip header if present
    if lines and lines[0].startswith("Epoch"):
        lines = lines[1:]

    stages_30s: List[int] = []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            raw_stage = int(parts[1])
            stages_30s.append(WSC_STG_MAP.get(raw_stage, -1))

    stages = resample_native_epoch_labels(
        stages_30s, int(epoch_seconds), native_epoch=_STG_NATIVE_EPOCH,
    )

    scores_start = 0.0
    scores_end = len(stages) * epoch_seconds
    return np.array(stages, dtype=np.int16), scores_start, scores_end


def _read_allscore_annotations(
    allscore_path: str,
    epoch_seconds: float = 30,
) -> Tuple[np.ndarray, float, float]:
    """Parse a .allscore.txt file (timestamp-based stage markers).

    Always parses at 30-second resolution first, then duplicates or
    merges to match ``epoch_seconds``.
    """
    with open(allscore_path, encoding="latin1") as f:
        lines = f.readlines()

    # Find START RECORDING
    start_time = None
    startline = 0
    for i, line in enumerate(lines):
        if "START RECORDING" in line:
            time_str = line.split("\t")[0]
            start_time = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
            startline = i
            break

    if start_time is None:
        return np.array([], dtype=np.int16), 0.0, 0.0

    # Parse stage transitions at native 30s resolution
    stages_30s: List[int] = []
    current_time = start_time
    stage = -1  # unscored at beginning

    for line in lines[startline:]:
        if "STAGE - " in line:
            new_stage_str = line.split("STAGE - ")[-1].replace("\n", "")
            new_stage = WSC_ALLSCORE_MAP.get(new_stage_str, -1)

            stage_time_str = line.split("\t")[0]
            stage_time = datetime.datetime.strptime(stage_time_str, "%H:%M:%S.%f")
            # Handle day boundary (recordings cross midnight)
            if stage_time.hour <= 12:
                stage_time += datetime.timedelta(days=1)

            time_passed = stage_time - current_time
            passed_epochs = int(time_passed.total_seconds() / _STG_NATIVE_EPOCH)
            stages_30s.extend([stage] * passed_epochs)

            stage = new_stage
            current_time = stage_time

    # Handle final segment
    end_time_str = lines[-1].split("\t")[0]
    try:
        end_time = datetime.datetime.strptime(end_time_str, "%H:%M:%S.%f")
        end_time += datetime.timedelta(days=1)
        time_passed = end_time - current_time
        passed_epochs = int(time_passed.total_seconds() / _STG_NATIVE_EPOCH)
        stages_30s.extend([stage] * passed_epochs)
    except ValueError:
        pass

    stages = resample_native_epoch_labels(
        stages_30s, int(epoch_seconds), native_epoch=_STG_NATIVE_EPOCH,
    )

    scores_start = 0.0
    scores_end = len(stages) * epoch_seconds
    return np.array(stages, dtype=np.int16), scores_start, scores_end


def read_wsc_annotations(
    annotation_path: str,
    epoch_seconds: float = 30,
) -> Tuple[np.ndarray, float, float]:
    """Read WSC annotations, dispatching by file type."""
    if annotation_path.endswith(".stg.txt"):
        return _read_stg_annotations(annotation_path, epoch_seconds)
    elif annotation_path.endswith(".allscore.txt"):
        return _read_allscore_annotations(annotation_path, epoch_seconds)
    else:
        raise ValueError(f"Unknown annotation format: {annotation_path}")


# ---------------------------------------------------------------------------
# Channel reading (edfio)
# ---------------------------------------------------------------------------

def _resolve_channel_key(
    name: str, label_to_signal: Dict[str, object],
) -> Optional[str]:
    """Find a channel key by primary name or reference alias."""
    key = name.upper()
    if key in label_to_signal:
        return key
    for alias in _WSC_REFERENCE_ALIASES.get(name, []):
        akey = alias.upper()
        if akey in label_to_signal:
            return akey
    return None


def check_edf_channels(edf_path: str, channel_names: List[str]) -> bool:
    """Return True if the EDF has at least one of the requested channels with sfreq > 0.

    Uses edfio header parsing only — does not read signal data.
    Tries reference aliases when the primary channel name is missing.
    """
    try:
        edf = edfio.read_edf(edf_path)
        label_to_signal = {s.label.upper(): s for s in edf.signals}
        for name in channel_names:
            resolved = _resolve_channel_key(name, label_to_signal)
            if resolved is not None:
                sig = label_to_signal[resolved]
                if float(sig.sampling_frequency) > 0:
                    return True
    except Exception:
        pass
    return False


def read_edf_channels(
    edf_path: str,
    channel_names: List[str],
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Read named channels from a WSC EDF file using edfio.

    Tries reference aliases when the primary channel name is missing.

    Returns
    -------
    signals : ndarray, shape (n_channels, n_samples)
    sfreq : float
    """
    edf = edfio.read_edf(edf_path)
    label_to_signal = {s.label.upper(): s for s in edf.signals}

    channel_data: List[np.ndarray] = []
    ref_sfreq: Optional[float] = None

    for name in channel_names:
        resolved = _resolve_channel_key(name, label_to_signal)
        if resolved is not None:
            sig = label_to_signal[resolved]
            sfreq = float(sig.sampling_frequency)
            s_start = int(start_sec * sfreq)
            if duration_sec is not None:
                data = sig.data[s_start:s_start + int(duration_sec * sfreq)]
            elif s_start > 0:
                data = sig.data[s_start:]
            else:
                data = sig.data
            channel_data.append(data.astype(np.float32))
            if ref_sfreq is None:
                ref_sfreq = sfreq
        else:
            channel_data.append(None)

    # Zero-fill missing channels
    resolved_data = [c for c in channel_data if c is not None]
    if not resolved_data:
        return np.array([]), 0.0
    min_samples = min(len(c) for c in resolved_data)
    for i, c in enumerate(channel_data):
        if c is None:
            channel_data[i] = np.zeros(min_samples, dtype=np.float32)

    stacked = np.stack([c[:min_samples] for c in channel_data], axis=0)
    return stacked, float(ref_sfreq)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WSCDataset(Dataset):
    """Epoch-level PyTorch dataset for the WSC collection.

    Parameters
    ----------
    subject_records : list of SubjectRecord
        Recordings to include in this dataset split.
    channel_specs : list of str
        List of channel names to load from the EDF.
    epoch_seconds : float
        Duration of one epoch in seconds.
    bandpass : tuple of (low, high) or None
        Bandpass filter cutoffs in Hz.
    notch : float or None
        Notch filter frequency in Hz.
    fold_assignments : dict or None
        Per-subject fold assignments for global embedding cache.
    compute_recording_stats : bool
        When *True*, compute per-recording mean / std / q95 and expose
        them in every sample returned by ``__getitem__``.
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

        self._subject_records = list(subject_records)
        self._records_by_id: Dict[str, SubjectRecord] = {}
        self._scores_range: Dict[str, Tuple[float, float]] = {}
        self._index: List[Tuple[str, int, int]] = []  # (recording_id, epoch_idx, stage)
        self._index_built = False
        self._dropped_records: List[Tuple[str, str]] = []
        self._edf_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._stats_cache: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
        self._discovered_channels: Dict[str, List[str]] = {}

    def _ensure_index_built(self) -> None:
        """Build epoch index lazily on first access."""
        if self._index_built:
            return
        skipped = 0
        for record in self._subject_records:
            if not self._use_all_eeg_channels and not check_edf_channels(record.edf_path, self._montage_channels):
                self._dropped_records.append((
                    record.recording_id,
                    "no montage channel found by check_edf_channels",
                ))
                skipped += 1
                continue
            self._records_by_id[record.recording_id] = record
            stages, scores_start, scores_end = read_wsc_annotations(
                record.annotation_path, epoch_seconds=self.epoch_seconds,
            )
            self._scores_range[record.recording_id] = (scores_start, scores_end)
            for ep_idx, stage in enumerate(stages):
                self._index.append(
                    (record.recording_id, ep_idx, int(stage))
                )
        if skipped:
            import warnings
            warnings.warn(f"WSCDataset: skipped {skipped} recording(s) with missing/unreadable EEG channels.")
        self._index_built = True

    def _load_subject_edf(
        self, recording_id: str,
    ) -> Tuple[np.ndarray, float]:
        """Load, filter, and cache a recording's PSG signals."""
        if recording_id in self._edf_cache:
            return self._edf_cache[recording_id]

        record = self._records_by_id[recording_id]
        scores_start, scores_end = self._scores_range[recording_id]
        duration = scores_end - scores_start

        if self._use_all_eeg_channels:
            from ._eeg_channel_discovery import read_all_eeg_channels
            signals, ch_names, sfreq = read_all_eeg_channels(
                record.edf_path,
                start_sec=scores_start, duration_sec=duration,
            )
            self._discovered_channels[recording_id] = ch_names
        else:
            signals, sfreq = read_edf_channels(
                record.edf_path, self._montage_channels,
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
        self._edf_cache.clear()
        self._stats_cache.clear()
        self._stats_cache[recording_id] = rec_stats
        self._edf_cache[recording_id] = (signals, sfreq)
        return signals, sfreq

    def evict_subject(self, recording_id: str):
        self._edf_cache.pop(recording_id, None)
        self._stats_cache.pop(recording_id, None)

    @staticmethod
    def eviction_key(meta: Dict[str, object]) -> Optional[str]:
        """WSC keys its signal cache by ``recording_id`` (not ``subject_id``).

        A single subject can have multiple visits with distinct
        ``recording_id``s; passing ``subject_id`` to ``evict_subject`` would
        silently miss every pop and leak the full dataset into RAM.
        """
        rid = meta.get("recording_id")
        return str(rid) if rid is not None else None

    def __len__(self) -> int:
        self._ensure_index_built()
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        self._ensure_index_built()
        recording_id, ep_idx, sleep_stage = self._index[idx]
        signals, sfreq = self._load_subject_edf(recording_id)
        record = self._records_by_id[recording_id]

        n_samples = int(round(self.epoch_seconds * sfreq))
        start_sample = ep_idx * n_samples
        epoch = signals[:, start_sample : start_sample + n_samples]
        if epoch.shape[1] < n_samples:
            epoch = np.pad(epoch, ((0, 0), (0, n_samples - epoch.shape[1])))

        sample = {
            "eeg": torch.tensor(epoch, dtype=torch.float32),
            "sleep_stage": sleep_stage,
            "subject_id": record.subject_id,
            "recording_id": recording_id,
            "visit": record.visit,
            "epoch_idx": ep_idx,
            "channels": (self._discovered_channels.get(recording_id, self._montage_names)),
            "sampling_rate": sfreq,
            "unit": "uV",
            "age": record.age,
            "sex": record.sex,
            "location": "Madison, Wisconsin, US",
        }
        rec_stats = self._stats_cache.get(recording_id)
        if rec_stats is not None:
            for k, v in rec_stats.items():
                sample[k] = v
        if record.subject_id in self._fold_assignments:
            sample["fold_assignments"] = self._fold_assignments[record.subject_id]
        return sample

    @property
    def subjects(self) -> List[str]:
        return list({r.subject_id for r in self._subject_records})
