"""Low-level Parkinson PSG dataset loader.

Amplitude convention: ``mne.io.read_raw_edf(...).get_data()`` returns
values in **Volts**. This loader does not rescale — the tensor delivered
to :class:`ParkinsonDataset.__getitem__` is in Volts. The benchmark
adapter at ``adapters/parkinson.py`` declares ``unit="V"`` in each
``meta[i]`` so wrappers convert via ``unit_to_uv`` in ``_preproc.py``.
"""
from __future__ import annotations

import csv
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

# Sleep stage label mapping (from TSV annotation files)
STAGE_MAP: Dict[str, int] = {
    "Wake": 0,
    "S1": 1,
    "S2": 2,
    "S3": 3,
    "REM": 4,
}

# Stages to exclude from epoch table
_EXCLUDED_STAGES = {"Unscorable", "LIGHTS_OFF"}

EPOCH_SECONDS = 30

# Default channel sets
SINGLE_CHANNEL = ["C3-A2"]
MULTI_CHANNELS = ["C3-A2", "C4-A1", "O1-A2", "O2-A1", "F3-A2", "F4-A1"]

# Bipolar PSG name → standard 10-20 name (used by models with spatial position lookups)
BIPOLAR_TO_STANDARD: Dict[str, str] = {
    "C3-A2": "C3",
    "C4-A1": "C4",
    "O1-A2": "O1",
    "O2-A1": "O2",
    "F3-A2": "F3",
    "F4-A1": "F4",
}


def load_subject_list(datasets_dir: str, filename: str = "ds_all_86.tsv") -> List[str]:
    """Return list of subject IDs from a dataset TSV file (e.g. ds_all_86.tsv).

    Lines starting with '#' are treated as comments and skipped.
    """
    path = Path(datasets_dir) / filename
    subjects: List[str] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            subjects.append(line)
    return subjects


def load_pd_labels(csv_path: str) -> Dict[str, int]:
    """Return mapping from subject record_id → 0 (HOA) or 1 (PD).

    Reads the Target_sleep_demographic.csv file.  The 'group' column contains
    either 'HOA' (Healthy Older Adults) or 'PD' (Parkinson's Disease).
    """
    mapping: Dict[str, int] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            record_id = row["record_id"].strip()
            group = row["group"].strip().upper()
            if group == "HOA":
                mapping[record_id] = 0
            elif group == "PD":
                mapping[record_id] = 1
    return mapping


def _find_edf_path(data_root: Path, subject_id: str) -> Path:
    """Locate the primary EDF file for a subject (rX suffix = r1 by default)."""
    subject_dir = data_root / "Data" / subject_id
    preferred = subject_dir / f"{subject_id}_r1.edf"
    if preferred.exists():
        return preferred
    candidates = sorted(subject_dir.glob("*.edf"))
    if not candidates:
        raise FileNotFoundError(f"No EDF file found for {subject_id} in {subject_dir}")
    return candidates[0]


def _find_tsv_path(data_root: Path, subject_id: str) -> Path:
    """Locate the sleep-stage annotation TSV for a subject."""
    subject_dir = data_root / "Data" / subject_id
    preferred = subject_dir / f"{subject_id}_r1_a1.tsv"
    if preferred.exists():
        return preferred
    candidates = sorted(subject_dir.glob("*.tsv"))
    if not candidates:
        raise FileNotFoundError(f"No annotation TSV found for {subject_id} in {subject_dir}")
    return candidates[0]


def _parse_annotation_tsv(tsv_path: Path) -> List[Tuple[float, float, str]]:
    """Return list of (start_sec, end_sec, stage_label) from annotation TSV."""
    epochs = []
    with open(tsv_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
                stage = parts[2].strip()
            except ValueError:
                continue
            epochs.append((start, end, stage))
    return epochs


def _build_epoch_table(tsv_path: Path) -> List[Tuple[float, int]]:
    """Return list of (start_sec, stage_int) for scorable 30-second epochs only."""
    annotations = _parse_annotation_tsv(tsv_path)
    table: List[Tuple[float, int]] = []
    for start, end, stage in annotations:
        if stage in _EXCLUDED_STAGES:
            continue
        if stage not in STAGE_MAP:
            continue
        if abs((end - start) - EPOCH_SECONDS) > 1.0:
            continue
        table.append((start, STAGE_MAP[stage]))
    return table


class ParkinsonDataset(Dataset):
    """Epoch-level PyTorch dataset for the Parkinson polysomnography collection.

    One sample corresponds to one 30-second EEG epoch.  The patient-level
    Parkinson/HOA binary label is attached to every epoch.  Sleep-stage
    information is kept in the returned dict as well.

    Supports single-channel and multi-channel operation via the ``channels``
    parameter.  Pass ``SINGLE_CHANNEL`` (default) for a single C3-A2 trace, or
    ``MULTI_CHANNELS`` for all six EEG derivations.

    EDF files are loaded lazily per subject on first access and cached in
    memory for the lifetime of the dataset object.

    Args:
        data_root: Root directory of the Parkinson dataset.
        subject_ids: Subjects to include.
        pd_labels: Mapping subject_id → 0 (HOA) / 1 (PD).
        channels: List of EDF channel names to extract (e.g. ["C3-A2"] or
            MULTI_CHANNELS).  The returned ``eeg`` tensor has shape
            ``(len(channels), n_samples)``.
        target_sfreq: Resample to this frequency; None keeps the native rate.
    """

    def __init__(
        self,
        data_root: str,
        subject_ids: Sequence[str],
        pd_labels: Dict[str, int],
        channels: List[str] = SINGLE_CHANNEL,
        target_sfreq: Optional[float] = 256.0,
    ) -> None:
        self.data_root = Path(data_root)
        self.channels = list(channels)
        self._display_names = [BIPOLAR_TO_STANDARD.get(c, c) for c in self.channels]
        self.target_sfreq = target_sfreq

        self._index: List[Tuple[str, int, float, int, int]] = []
        self._subjects: List[str] = []

        for subj in subject_ids:
            if subj not in pd_labels:
                continue
            try:
                tsv_path = _find_tsv_path(self.data_root, subj)
            except FileNotFoundError:
                continue
            epoch_table = _build_epoch_table(tsv_path)
            pd_label = pd_labels[subj]
            self._subjects.append(subj)
            for ep_idx, (start_sec, sleep_stage) in enumerate(epoch_table):
                self._index.append((subj, ep_idx, start_sec, sleep_stage, pd_label))

        # Cache: subject_id → (signals_2d: (N_ch, n_samples), sfreq)
        self._edf_cache: Dict[str, Tuple[np.ndarray, float]] = {}

    def _load_subject_edf(self, subject_id: str) -> Tuple[np.ndarray, float]:
        """Load and cache all requested channels for a subject.

        Channel matching strips common EEG prefixes (e.g. "EEG C3-A2" → "C3-A2")
        so that bipolar PSG names work regardless of whether the EDF stores them
        with or without a leading modality tag.

        Uses a single sequential preload for the full EDF (fast on NFS), then
        picks only the needed channels in memory before resampling.
        """
        if subject_id in self._edf_cache:
            return self._edf_cache[subject_id]

        import warnings
        import mne

        _MODALITY_PREFIXES = ("EEG ", "EOG ", "EMG ", "ECG ", "REF ")

        def _strip_prefix(name: str) -> str:
            for pfx in _MODALITY_PREFIXES:
                if name.startswith(pfx):
                    return name[len(pfx):]
            return name

        edf_path = _find_edf_path(self.data_root, subject_id)

        # Sequential preload — one contiguous NFS read, fast on network filesystems.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

        # Build stripped-name → original-name map and pick only needed channels.
        stripped_to_orig: Dict[str, str] = {_strip_prefix(ch): ch for ch in raw.ch_names}
        req_to_orig: Dict[str, str] = {}
        include_orig: List[str] = []
        for ch in self.channels:
            orig = stripped_to_orig.get(ch)
            if orig is not None:
                req_to_orig[ch] = orig
                include_orig.append(orig)

        if include_orig:
            raw.pick(include_orig)

        sfreq = raw.info["sfreq"]
        raw_data = raw.get_data()  # (N_present, n_samples) — float64

        # Resample using polyphase filtering (O(N), ~3× faster than FFT-based).
        if self.target_sfreq is not None and abs(sfreq - self.target_sfreq) > 0.5:
            up = int(self.target_sfreq)
            down = int(sfreq)
            g = gcd(up, down)
            raw_data = resample_poly(raw_data, up // g, down // g, axis=1)
            sfreq = self.target_sfreq

        n_samples = raw_data.shape[1]

        # Build output in the requested channel order; zero-fill missing channels.
        orig_to_row: Dict[str, int] = {ch: i for i, ch in enumerate(raw.ch_names)}
        signals = np.zeros((len(self.channels), n_samples), dtype=np.float32)
        for out_i, ch in enumerate(self.channels):
            orig = req_to_orig.get(ch)
            if orig is not None and orig in orig_to_row:
                signals[out_i] = raw_data[orig_to_row[orig]]

        self._edf_cache.clear()
        self._edf_cache[subject_id] = (signals, sfreq)
        return signals, sfreq

    def evict_subject(self, subject_id: str):
        self._edf_cache.pop(subject_id, None)

    @staticmethod
    def eviction_key(meta):
        sid = meta.get("subject_id")
        return str(sid) if sid is not None else None

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        subject_id, ep_within_subj, start_sec, sleep_stage, pd_label = self._index[idx]
        signals, sfreq = self._load_subject_edf(subject_id)

        start_sample = int(round(start_sec * sfreq))
        n_samples = int(round(EPOCH_SECONDS * sfreq))

        epoch = signals[:, start_sample : start_sample + n_samples]

        # Pad last epoch if recording ends early
        if epoch.shape[1] < n_samples:
            epoch = np.pad(epoch, ((0, 0), (0, n_samples - epoch.shape[1])))

        # Mean-center each channel independently
        epoch = epoch - epoch.mean(axis=1, keepdims=True)

        eeg_tensor = torch.tensor(epoch, dtype=torch.float32)  # (N_ch, n_samples)

        return {
            "eeg": eeg_tensor,
            "channels": self._display_names,
            "sleep_stage": sleep_stage,
            "pd_label": pd_label,
            "subject_id": subject_id,
            "epoch_idx": ep_within_subj,
        }

    @property
    def subjects(self) -> List[str]:
        return list(self._subjects)
