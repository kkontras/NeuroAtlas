"""
MASS (Montreal Archive of Sleep Studies) dataset loader for EEGBenchmarks.

Loads raw EDF files directly at runtime — no preprocessing step, no stored
output.  Signal processing (filtering, resampling, epoching) happens on the
fly with all parameters configurable.

Provides MASSDataset with lazy per-subject EDF caching, following the same
pattern as dataio/parkinson.py.
"""

from __future__ import annotations

import gzip
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import edfio
import mne
import numpy as np
import torch
from scipy.signal import butter, filtfilt, firwin, iirnotch, sosfiltfilt
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASS_STAGES_MAP: Dict[str, int] = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": -1,
}

LABEL_NAMES = ["W", "N1", "N2", "N3", "REM"]

DEFAULT_RAW_ROOT = "${EEG_DATA_ROOT}/mass"

SUBSETS = [1, 2, 3, 4, 5]

DEFAULT_CHANNELS: List[str] = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]

# Reference suffixes tried in priority order per recording
_REFERENCE_SUFFIXES_PRIORITY = ("-CLE", "-LER")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    subset: str
    edf_path: str
    annotation_path: str


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

@contextmanager
def _open_annotation_edf(path: str) -> Iterator[str]:
    """Yield a readable EDF path, decompressing .edf.gz to a temp file if needed."""
    if path.endswith(".gz"):
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp_path = tmp.name
            with gzip.open(path, "rb") as gz:
                shutil.copyfileobj(gz, tmp)
        try:
            yield tmp_path
        finally:
            os.unlink(tmp_path)
    else:
        yield path


def scan_mass_subjects(
    data_root: str,
    subsets: Sequence[int] = SUBSETS,
) -> List[SubjectRecord]:
    """Scan MASS subset directories for PSG + annotation EDF pairs."""
    records: List[SubjectRecord] = []
    root = Path(data_root)
    for subset_num in subsets:
        subset_dir = root / f"SS0{subset_num}"
        if not subset_dir.is_dir():
            continue
        annotations_dir = subset_dir / "annotations"
        edf_files = sorted(
            f for f in os.listdir(subset_dir)
            if f.endswith(".edf") and not f.startswith(".")
        )
        for edf_name in edf_files:
            subject_id = edf_name.split(" ")[0]  # e.g. "01-01-0001"
            # Find matching annotation EDF
            if annotations_dir.is_dir():
                ann_matches = [
                    a for a in os.listdir(annotations_dir)
                    if subject_id in a
                    and (a.endswith(".edf") or a.endswith(".edf.gz"))
                    and not a.startswith(".")
                ]
            else:
                ann_matches = []
            if not ann_matches:
                continue
            records.append(SubjectRecord(
                subject_id=subject_id,
                subset=f"SS0{subset_num}",
                edf_path=str(subset_dir / edf_name),
                annotation_path=str(annotations_dir / sorted(ann_matches)[0]),
            ))
    return records


# ---------------------------------------------------------------------------
# Channel reading (edfio — single open for resolve + read)
# ---------------------------------------------------------------------------

def resolve_and_read_channels(
    edf_path: str,
    montage_electrodes: Sequence[str],
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
) -> Tuple[np.ndarray, List[str], float, bool]:
    """Resolve montage and read channels in a single EDF open.

    Returns
    -------
    signals : ndarray, shape (n_channels, n_samples)
    channel_names : list of str
    sfreq : float
    any_resolved : bool
        True if at least one channel was resolved (False = all missing).
    """
    edf = edfio.read_edf(edf_path)

    # Build label lookup: upper-case label → signal object
    label_to_signal: Dict[str, edfio.EdfSignal] = {}
    for sig in edf.signals:
        label_to_signal[sig.label.upper()] = sig

    # Resolve montage + read in one pass
    channel_data: List[Optional[np.ndarray]] = []
    names: List[str] = []
    ref_sfreq: Optional[float] = None

    for spec in montage_electrodes:
        found = False
        for suffix in _REFERENCE_SUFFIXES_PRIORITY:
            key = f"EEG {spec}{suffix}".upper()
            if key in label_to_signal:
                sig = label_to_signal[key]
                sfreq = float(sig.sampling_frequency)
                s_start = int(start_sec * sfreq)
                if duration_sec is not None:
                    data = sig.data[s_start:s_start + int(duration_sec * sfreq)]
                elif s_start > 0:
                    data = sig.data[s_start:]
                else:
                    data = sig.data
                channel_data.append(data.astype(np.float32))
                names.append(sig.label)
                if ref_sfreq is None:
                    ref_sfreq = sfreq
                found = True
                break
        if not found:
            channel_data.append(None)
            names.append("MISSING")

    # Zero-fill missing channels
    resolved = [c for c in channel_data if c is not None]
    if not resolved:
        return np.array([]), names, 0.0, False

    min_samples = min(len(c) for c in resolved)
    for i, c in enumerate(channel_data):
        if c is None:
            channel_data[i] = np.zeros(min_samples, dtype=np.float32)

    stacked = np.stack([c[:min_samples] for c in channel_data], axis=0)
    return stacked, names, float(ref_sfreq), True


# ---------------------------------------------------------------------------
# Annotation reading (MNE)
# ---------------------------------------------------------------------------

def read_mass_annotations(
    annotation_edf_path: str,
    epoch_seconds: float = 30,
) -> Tuple[np.ndarray, float, float]:
    """Read sleep stage annotations from a MASS annotation EDF.

    Returns
    -------
    stages : ndarray of int16
        Per-epoch labels.  Unscored epochs are -1.
    scores_start : float
        Onset of the first scored annotation (seconds from recording start).
    scores_end : float
        End of the last scored epoch (seconds from recording start).
    """
    mne.set_log_level("ERROR")
    with _open_annotation_edf(annotation_edf_path) as edf_path:
        events_data = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

        annotations = [
            (MASS_STAGES_MAP[ann["description"]], ann["onset"], ann["duration"])
            for ann in events_data.annotations
            if ann["description"] in MASS_STAGES_MAP
        ]

    if not annotations:
        return np.array([], dtype=np.int16), 0.0, 0.0

    start = annotations[0][1]

    # Build per-second label array
    stages_per_second: List[int] = []
    cursor = start
    for stage, onset, duration in annotations:
        if onset > cursor:
            gap = int(onset - cursor)
            stages_per_second.extend([-1] * gap)
        stages_per_second.extend([stage] * int(duration))
        cursor = onset + duration

    # Collapse to per-epoch labels
    epoch_sec = int(epoch_seconds)
    n_epochs = len(stages_per_second) // epoch_sec
    stages: List[int] = []
    for i in range(n_epochs):
        epoch_labels = stages_per_second[i * epoch_sec : (i + 1) * epoch_sec]
        unique = set(epoch_labels)
        if len(unique) == 1:
            stages.append(unique.pop())
        else:
            stages.append(-1)

    scores_start = start
    scores_end = start + len(stages) * epoch_seconds
    return np.array(stages, dtype=np.int16), scores_start, scores_end


def read_mass_arousal_labels(
    annotation_edf_path: str,
    epoch_seconds: float,
    scores_start: float,
    scores_end: float,
) -> Optional[np.ndarray]:
    """Compute per-epoch arousal fraction from MicroArousal annotations.

    Returns an array of floats (0.0–1.0) representing the fraction of
    each epoch spent in arousal, or ``None`` if no MicroArousal events
    are found (e.g. non-SS01 subjects).
    """
    mne.set_log_level("ERROR")
    with _open_annotation_edf(annotation_edf_path) as edf_path:
        events_data = mne.io.read_raw_edf(
            edf_path, preload=False, verbose=False,
        )

        arousals = [
            (ann["onset"], ann["duration"])
            for ann in events_data.annotations
            if "MicroArousal" in ann["description"]
        ]

    if not arousals:
        return None

    epoch_sec = int(epoch_seconds)
    n_epochs = int((scores_end - scores_start) / epoch_seconds)
    fractions = np.zeros(n_epochs, dtype=np.float32)

    for onset, duration in arousals:
        # Arousal time range relative to scores_start
        a_start = onset - scores_start
        a_end = a_start + duration
        if a_end <= 0 or a_start >= scores_end - scores_start:
            continue
        a_start = max(a_start, 0)
        a_end = min(a_end, scores_end - scores_start)

        # Find which epochs this arousal overlaps
        ep_start = int(a_start // epoch_sec)
        ep_end = min(int(a_end // epoch_sec) + 1, n_epochs)
        for ep in range(ep_start, ep_end):
            ep_begin = ep * epoch_sec
            ep_finish = ep_begin + epoch_sec
            overlap_start = max(a_start, ep_begin)
            overlap_end = min(a_end, ep_finish)
            if overlap_end > overlap_start:
                fractions[ep] += (overlap_end - overlap_start) / epoch_sec

    # Clamp to [0, 1]
    np.clip(fractions, 0.0, 1.0, out=fractions)
    return fractions


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MASSDataset(Dataset):
    """Epoch-level PyTorch dataset for the MASS collection.

    Loads raw EDF files lazily per subject and caches the processed signals
    in memory (same pattern as :class:`ParkinsonDataset`).

    Parameters
    ----------
    subject_records : list of SubjectRecord
        Subjects to include in this dataset split.
    channel_specs : list of str
        List of electrode names to load from the EDF.
    epoch_seconds : float
        Duration of one epoch in seconds.
    bandpass : tuple of (low, high) or None
        Bandpass filter cutoffs in Hz.
    notch : float or None
        Notch filter frequency in Hz (e.g. 50 or 60).
    fold_assignments : dict or None
        Per-subject fold assignments for global embedding cache.
        Maps ``subject_id → {"fold_0": "train"|"valid"|"test", ...}``.
        When set, included in each sample's output for cache splitting.
    compute_recording_stats : bool
        If True, compute per-recording mean/std/q95 after filtering.
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
        self._montage_electrodes = list(channel_specs)
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
        self._arousal_fractions: Dict[str, np.ndarray] = {}  # subject_id -> per-epoch fractions
        self._index: List[Tuple[str, int, int, str]] = []
        self._index_built = False
        self._dropped_records: List[Tuple[str, str]] = []
        self._edf_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._stats_cache: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
        self._discovered_channels: Dict[str, List[str]] = {}

    def _ensure_index_built(self) -> None:
        """Build epoch index lazily on first access (reads annotation EDFs)."""
        if self._index_built:
            return
        for record in self._subject_records:
            if self._use_all_eeg_channels:
                has_any = True
            else:
                # Quick channel check via edfio (fast header read)
                edf = edfio.read_edf(record.edf_path)
                upper_labels = {sig.label.upper() for sig in edf.signals}
                has_any = False
                for spec in self._montage_electrodes:
                    for suffix in _REFERENCE_SUFFIXES_PRIORITY:
                        if f"EEG {spec}{suffix}".upper() in upper_labels:
                            has_any = True
                            break
                    if has_any:
                        break
            if not has_any:
                self._dropped_records.append((
                    record.subject_id,
                    f"no montage channel in EDF header (labels: {sorted(upper_labels)})",
                ))
                continue

            self._records_by_id[record.subject_id] = record
            stages, scores_start, scores_end = read_mass_annotations(
                record.annotation_path, epoch_seconds=self.epoch_seconds,
            )
            self._scores_range[record.subject_id] = (scores_start, scores_end)

            # Arousal labels. Required for SS01 (MicroArousal annotations are
            # part of that subset's ground truth); optional elsewhere.
            arousal = read_mass_arousal_labels(
                record.annotation_path, self.epoch_seconds,
                scores_start, scores_end,
            )
            if arousal is not None:
                self._arousal_fractions[record.subject_id] = arousal
            elif record.subset == "SS01":
                raise RuntimeError(
                    f"MASS SS01 subject {record.subject_id!r} has no MicroArousal "
                    f"annotations in {record.annotation_path!r}. SS01 arousal labels "
                    f"are required for the arousal_detection task."
                )

            for ep_idx, stage in enumerate(stages):
                self._index.append(
                    (record.subject_id, ep_idx, int(stage), record.subset)
                )
        self._index_built = True

    def _load_subject_edf(
        self, subject_id: str,
    ) -> Tuple[np.ndarray, float]:
        """Load, filter, resample, and cache a subject's PSG signals."""
        if subject_id in self._edf_cache:
            return self._edf_cache[subject_id]

        record = self._records_by_id[subject_id]
        scores_start, scores_end = self._scores_range[subject_id]
        duration = scores_end - scores_start

        if self._use_all_eeg_channels:
            from ._eeg_channel_discovery import read_all_eeg_channels

            def _mass_eeg_filter(label: str) -> bool:
                return label.strip().upper().startswith("EEG ")

            signals, ch_names, sfreq = read_all_eeg_channels(
                record.edf_path,
                start_sec=scores_start, duration_sec=duration,
                label_filter=_mass_eeg_filter,
            )
            self._discovered_channels[subject_id] = ch_names
        else:
            signals, _, sfreq, _ = resolve_and_read_channels(
                record.edf_path, self._montage_electrodes,
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

        # Recording-level statistics (gated)
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
        self._edf_cache[subject_id] = (signals, sfreq)
        self._stats_cache[subject_id] = rec_stats
        return signals, sfreq

    def evict_subject(self, subject_id: str) -> None:
        self._edf_cache.pop(subject_id, None)
        self._stats_cache.pop(subject_id, None)

    @staticmethod
    def eviction_key(meta: Dict[str, object]) -> Optional[str]:
        """Identity under which ``evict_subject`` should be called for this meta."""
        sid = meta.get("subject_id")
        return str(sid) if sid is not None else None

    def __len__(self) -> int:
        self._ensure_index_built()
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        self._ensure_index_built()
        subject_id, ep_idx, sleep_stage, subset = self._index[idx]
        signals, sfreq = self._load_subject_edf(subject_id)

        n_samples = int(round(self.epoch_seconds * sfreq))
        start_sample = ep_idx * n_samples
        epoch = signals[:, start_sample : start_sample + n_samples]
        if epoch.shape[1] < n_samples:
            epoch = np.pad(epoch, ((0, 0), (0, n_samples - epoch.shape[1])))

        sample: Dict[str, object] = {
            "eeg": torch.tensor(epoch, dtype=torch.float32),
            "sleep_stage": sleep_stage,
            "subject_id": subject_id,
            "subset": subset,
            "epoch_idx": ep_idx,
            "channels": (self._discovered_channels.get(subject_id, self._montage_names)),
            "sampling_rate": sfreq,
            "unit": "uV",
            "epoch_seconds": float(self.epoch_seconds),
            "location": "Montreal, Canada",
        }
        rec_stats = self._stats_cache.get(subject_id)
        if rec_stats is not None:
            for k, v in rec_stats.items():
                sample[k] = v
        if subject_id in self._fold_assignments:
            sample["fold_assignments"] = self._fold_assignments[subject_id]
        if subject_id in self._arousal_fractions:
            fracs = self._arousal_fractions[subject_id]
            sample["arousal_fraction"] = float(fracs[ep_idx]) if ep_idx < len(fracs) else 0.0
        return sample

    @property
    def subjects(self) -> List[str]:
        return [r.subject_id for r in self._subject_records]
