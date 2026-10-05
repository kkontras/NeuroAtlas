"""DOD (Dreem Open Datasets) loader for EEGBenchmarks.

Reads per-subject HDF5 files directly at runtime. Each h5 file has the
Dreem layout:

    hypnogram               -- int64 array, one entry per 30 s epoch,
                               codes {-1: NOT SCORED, 0: W, 1: N1, 2: N2,
                               3: N3, 4: REM}
    signals/<domain>/<ch>   -- float32 array per channel, with per-group
                               attribute ``fs`` (Hz). Domains are ``eeg``,
                               ``eog``, ``emg``.

Groups are inferred from the containing directory name under the data
root (``dodh`` = healthy, ``dodo`` = obstructive apnea).

Optional per-expert sleep-stage annotations live in a clone of
``Dreem-Organization/dreem-learning-evaluation`` at
``<scorers_root>/<group>/scorer_{1..5}/<uuid>.json``. The Zenodo h5
hypnogram is already trimmed to lights-off/lights-on, while the scorer
JSONs span the entire raw recording. This module replicates the exact
trim logic from ``evaluation.py:244-265`` of that repo to align the 5
expert arrays to the h5 grid:

    index_min = max_across_scorers(first non-(-1) epoch),
                 clamped up by an optional per-record ``lights_off``
    index_max = len - max_across_scorers(trailing (-1) count),
                 clamped down by an optional per-record ``lights_on``
    scorer[index_min:index_max] is then the aligned array.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from scipy.signal import butter, filtfilt, firwin, iirnotch, sosfiltfilt
from torch.utils.data import Dataset

from neuroatlas.extensions.datasets.dataio._stage_resample import (
    resample_native_epoch_labels,
)


LABEL_NAMES = ["W", "N1", "N2", "N3", "REM"]

DEFAULT_RAW_ROOT = "${EEG_DATA_ROOT}/data/DOD"
DOD_GROUPS: Tuple[str, str] = ("dodh", "dodo")
DOD_GROUP_TO_OSA_LABEL: Dict[str, int] = {"dodh": 0, "dodo": 1}
DOD_NATIVE_EPOCH = 30
DOD_SCORER_COLUMNS: Tuple[str, ...] = (
    "scorer_1", "scorer_2", "scorer_3", "scorer_4", "scorer_5",
)

# Hand-coded overrides copied from Dreem-Organization/dreem-learning-evaluation
# README, applied on top of the automatic first/last-non-(-1) trim.
DOD_LIGHTS_OFF_OVERRIDES: Dict[str, int] = {
    "63b799f6-8a4f-4224-8797-ea971f78fb53": 60,
    "de3af7b1-ab6f-43fd-96f0-6fc64e8d2ed4": 60,
}
DOD_LIGHTS_ON_OVERRIDES: Dict[str, int] = {
    "a14f8058-f636-4be7-a67a-8f7f91a419e7": 620,
}

# Channel spec: (domain, dataset_name) pointing at signals/<domain>/<name>.
ChannelSpec = Tuple[str, str]

DEFAULT_CHANNELS: List[str] = [
    "C3", "C4", "F3-F4", "F3", "F3-O1", "F4", "F4-O2",
    "FP1-F3", "FP1", "FP1-O1", "FP2-F4", "FP2", "FP2-O2",
    "O1", "O2",
]

_DOD_DISPLAY_TO_H5: Dict[str, ChannelSpec] = {
    "C3": ("eeg", "C3_M2"), "C4": ("eeg", "C4_M1"),
    "F3-F4": ("eeg", "F3_F4"), "F3": ("eeg", "F3_M2"), "F4": ("eeg", "F4_M1"),
    "F3-O1": ("eeg", "F3_O1"), "F4-O2": ("eeg", "F4_O2"),
    "FP1-F3": ("eeg", "FP1_F3"), "FP1": ("eeg", "FP1_M2"), "FP1-O1": ("eeg", "FP1_O1"),
    "FP2-F4": ("eeg", "FP2_F4"), "FP2": ("eeg", "FP2_M1"), "FP2-O2": ("eeg", "FP2_O2"),
    "O1": ("eeg", "O1_M2"), "O2": ("eeg", "O2_M1"),
}


def _resolve_display_names_to_h5(display_names: List[str]) -> Tuple[List[ChannelSpec], List[str]]:
    """Convert a list of DOD display names to (ChannelSpec list, name list)."""
    specs: List[ChannelSpec] = []
    names: List[str] = []
    for name in display_names:
        spec = _DOD_DISPLAY_TO_H5.get(name)
        if spec is None:
            raise ValueError(
                f"Unknown DOD display name {name!r}. "
                f"Available: {sorted(_DOD_DISPLAY_TO_H5)}"
            )
        specs.append(spec)
        names.append(name)
    return specs, names


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    h5_path: str
    group: str


def scan_dod_subjects(data_root: str) -> List[SubjectRecord]:
    """Scan ``data_root/{dodh,dodo}/*.h5`` and build SubjectRecord list."""
    root = Path(data_root)
    records: List[SubjectRecord] = []
    for group in DOD_GROUPS:
        group_dir = root / group
        if not group_dir.exists():
            continue
        for h5_path in sorted(group_dir.glob("*.h5")):
            uuid_stem = h5_path.stem
            records.append(SubjectRecord(
                subject_id=f"{group}__{uuid_stem}",
                h5_path=str(h5_path),
                group=group,
            ))
    return records


def _validate_epoch_seconds(epoch_seconds: float) -> int:
    if int(epoch_seconds) != epoch_seconds or epoch_seconds <= 0:
        raise ValueError(
            f"epoch_seconds must be a positive integer for DOD, got {epoch_seconds!r}"
        )
    return int(epoch_seconds)


def _clamp_stage(x: int) -> int:
    return x if 0 <= x <= 4 else -1


def read_dod_stages(
    h5_path: str,
    epoch_seconds: float = DOD_NATIVE_EPOCH,
) -> Tuple[np.ndarray, float, float]:
    """Read the consensus hypnogram and resample to ``epoch_seconds``.

    Returns (stages, 0.0, scores_end). Stage codes outside {-1..4} are
    clamped to -1. At non-divisor targets the
    ``resample_native_epoch_labels`` "windows entirely inside a single-
    label run keep that label; genuinely mixed windows become -1" rule
    applies.
    """
    eps = _validate_epoch_seconds(epoch_seconds)
    with h5py.File(h5_path, "r") as h:
        native = [_clamp_stage(int(v)) for v in h["hypnogram"][:]]
    stages = resample_native_epoch_labels(native, eps, native_epoch=DOD_NATIVE_EPOCH)
    arr = np.asarray(stages, dtype=np.int16)
    return arr, 0.0, float(len(arr) * eps)


def read_dod_expert_scorings(
    scorers_root: Optional[str],
    group: str,
    uuid: str,
    expected_native_n_epochs: int,
    lights_off_overrides: Optional[Dict[str, int]] = None,
    lights_on_overrides: Optional[Dict[str, int]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Load + align per-expert stages at the native 30 s grid.

    Replicates the lights-off / lights-on trim from
    ``dreem-learning-evaluation/evaluation.py:244-265``:

    * Read ``scorers/<group>/scorer_{1..5}/<uuid>.json``.
    * Skip scorers whose JSON is missing or malformed.
    * Require at least 2 scorers present and all present scorers to share
      the same raw length ``N_raw``.
    * ``index_min = max(first non-(-1) per scorer)``, clamped up by
      ``lights_off_overrides.get(uuid, 0)``.
    * ``index_max = N_raw - max(trailing (-1) per scorer)``, clamped down
      by ``lights_on_overrides.get(uuid, +inf)``.
    * Only succeeds when ``index_max - index_min == expected_native_n_epochs``
      (i.e. the trim gives the exact h5 hypnogram length); else returns
      ``None`` so the caller falls back to consensus-only labels.
    * Returns a dict keyed by ``scorer_1..scorer_5`` whose values are
      ``np.ndarray[int16]`` of length ``expected_native_n_epochs``,
      with codes clamped to ``{-1..4}``.
    """
    if not scorers_root:
        return None
    lights_off = lights_off_overrides or DOD_LIGHTS_OFF_OVERRIDES
    lights_on = lights_on_overrides or DOD_LIGHTS_ON_OVERRIDES

    raw: Dict[str, np.ndarray] = {}
    for scorer in DOD_SCORER_COLUMNS:
        jp = Path(scorers_root) / group / scorer / f"{uuid}.json"
        if not jp.exists():
            continue
        try:
            with open(jp) as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                continue
            raw[scorer] = np.asarray(data, dtype=np.int64)
        except (OSError, json.JSONDecodeError, ValueError):
            continue

    if len(raw) < 2:
        return None
    raw_lens = {len(v) for v in raw.values()}
    if len(raw_lens) != 1:
        return None
    n_raw = raw_lens.pop()

    first_pos = [int(np.where(arr >= 0)[0][0]) for arr in raw.values()]
    trailing = [int(np.where(arr[::-1] >= 0)[0][0]) for arr in raw.values()]
    index_min = max(max(first_pos), lights_off.get(uuid, 0))
    index_max = min(n_raw - max(trailing), lights_on.get(uuid, n_raw))

    if index_max - index_min != expected_native_n_epochs:
        return None

    out: Dict[str, np.ndarray] = {}
    for scorer, arr in raw.items():
        trimmed = arr[index_min:index_max]
        out[scorer] = np.asarray(
            [_clamp_stage(int(v)) for v in trimmed], dtype=np.int16,
        )
    return out


def read_dod_expert_stages_for_grid(
    raw_expert: Optional[Dict[str, np.ndarray]],
    epoch_seconds: float,
) -> Optional[Dict[str, np.ndarray]]:
    """Resample each per-scorer native-grid array onto ``epoch_seconds``.

    Reuses ``resample_native_epoch_labels`` with ``native_epoch=30`` so
    consensus and expert labels share the same alignment rule. Returns
    ``None`` when the input is ``None``.
    """
    if raw_expert is None:
        return None
    eps = _validate_epoch_seconds(epoch_seconds)
    out: Dict[str, np.ndarray] = {}
    for scorer, arr in raw_expert.items():
        resampled = resample_native_epoch_labels(
            arr.tolist(), eps, native_epoch=DOD_NATIVE_EPOCH,
        )
        out[scorer] = np.asarray(resampled, dtype=np.int16)
    return out


def read_dod_channels(
    h5_path: str,
    channel_specs: Sequence[ChannelSpec],
    duration_sec: Optional[float] = None,
) -> Tuple[np.ndarray, List[str], float]:
    """Read stacked channels and the reference sampling rate.

    Missing channels are zero-filled so model input dimensionality stays
    fixed. Sampling rate is taken from the first resolved channel's
    domain-group ``fs`` attribute; all DOD files are uniform at 250 Hz.
    """
    data: List[Optional[np.ndarray]] = []
    ref_sfreq: Optional[float] = None
    with h5py.File(h5_path, "r") as h:
        for domain, ch in channel_specs:
            dset_path = f"signals/{domain}/{ch}"
            if dset_path not in h:
                data.append(None)
                continue
            group = h[f"signals/{domain}"]
            sfreq = float(group.attrs["fs"])
            dset = h[dset_path]
            if duration_sec is not None:
                n = int(duration_sec * sfreq)
                arr = dset[:n].astype(np.float32)
            else:
                arr = dset[:].astype(np.float32)
            data.append(arr)
            if ref_sfreq is None:
                ref_sfreq = sfreq
    resolved = [c for c in data if c is not None]
    if not resolved:
        return np.array([]), ["MISSING"] * len(channel_specs), 0.0
    min_samples = min(len(c) for c in resolved)
    for i, c in enumerate(data):
        if c is None:
            data[i] = np.zeros(min_samples, dtype=np.float32)
    stacked = np.stack([c[:min_samples] for c in data], axis=0)
    return stacked, [f"{d}/{c}" for d, c in channel_specs], float(ref_sfreq or 0.0)


class DODDataset(Dataset):
    """Epoch-level PyTorch dataset for DOD."""

    _DOD_LOCATION = {"dodh": "France", "dodo": "Stanford, California, US"}

    def __init__(
        self,
        subject_records: Sequence[SubjectRecord],
        channel_specs: List[str] = DEFAULT_CHANNELS,
        epoch_seconds: float = DOD_NATIVE_EPOCH,
        bandpass: Optional[Tuple[float, float]] = None,
        notch: Optional[float] = None,
        highpass: Optional[float] = None,
        notch_map: Optional[Dict[str, float]] = None,
        fold_assignments: Optional[Dict[str, Dict[str, str]]] = None,
        expert_scorers_root: Optional[str] = None,
        lights_off_overrides: Optional[Dict[str, int]] = None,
        lights_on_overrides: Optional[Dict[str, int]] = None,
        compute_recording_stats: bool = False,
        use_all_eeg_channels: bool = False,
    ) -> None:
        self._channel_list, self._montage_names = _resolve_display_names_to_h5(list(channel_specs))
        self.channel_specs = channel_specs
        self.epoch_seconds = float(_validate_epoch_seconds(epoch_seconds))
        self.bandpass = bandpass
        self.notch = notch
        self.highpass = highpass
        self.notch_map = notch_map
        self._fold_assignments = fold_assignments or {}
        self._expert_scorers_root = expert_scorers_root
        self._lights_off_overrides = lights_off_overrides
        self._lights_on_overrides = lights_on_overrides
        self._compute_recording_stats = compute_recording_stats
        self._use_all_eeg_channels = use_all_eeg_channels
        self._discovered_channels: Dict[str, List[str]] = {}
        self._subject_records = list(subject_records)
        self._records_by_id: Dict[str, SubjectRecord] = {}
        self._scores_end: Dict[str, float] = {}
        self._expert_stages: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
        self._index: List[Tuple[str, int, int]] = []
        self._index_built = False
        self._signal_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._stats_cache: Dict[str, Optional[Dict[str, np.ndarray]]] = {}

    def _ensure_index_built(self) -> None:
        if self._index_built:
            return
        for record in self._subject_records:
            stages, _, scores_end = read_dod_stages(
                record.h5_path, epoch_seconds=self.epoch_seconds,
            )
            with h5py.File(record.h5_path, "r") as h:
                native_n_epochs = int(h["hypnogram"].shape[0])

            raw_expert = read_dod_expert_scorings(
                self._expert_scorers_root,
                group=record.group,
                uuid=Path(record.h5_path).stem,
                expected_native_n_epochs=native_n_epochs,
                lights_off_overrides=self._lights_off_overrides,
                lights_on_overrides=self._lights_on_overrides,
            )
            experts = read_dod_expert_stages_for_grid(
                raw_expert, epoch_seconds=self.epoch_seconds,
            )

            self._records_by_id[record.subject_id] = record
            self._scores_end[record.subject_id] = scores_end
            self._expert_stages[record.subject_id] = experts

            for ep_idx, stage in enumerate(stages):
                self._index.append((record.subject_id, ep_idx, int(stage)))
        self._index_built = True

    def _load_subject_signals(self, subject_id: str) -> Tuple[np.ndarray, float]:
        if subject_id in self._signal_cache:
            return self._signal_cache[subject_id]

        record = self._records_by_id[subject_id]
        scores_end = self._scores_end[subject_id]

        if self._use_all_eeg_channels:
            with h5py.File(record.h5_path, "r") as h:
                eeg_group = h.get("signals/eeg")
                if eeg_group is not None:
                    eeg_keys = sorted(eeg_group.keys())
                else:
                    eeg_keys = []
            if eeg_keys:
                eeg_specs = [("eeg", k) for k in eeg_keys]
                display_names = [k.replace("_", "-") for k in eeg_keys]
                self._discovered_channels[subject_id] = display_names
                signals, _, sfreq = read_dod_channels(
                    record.h5_path, eeg_specs, duration_sec=scores_end,
                )
            else:
                signals, _, sfreq = read_dod_channels(
                    record.h5_path, self._channel_list, duration_sec=scores_end,
                )
        else:
            signals, _, sfreq = read_dod_channels(
                record.h5_path, self._channel_list, duration_sec=scores_end,
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

        notch_freq = self.notch
        if self.notch_map is not None:
            notch_freq = self.notch_map.get(record.group, self.notch)
        if notch_freq is not None and notch_freq < sfreq / 2.0:
            b, a = iirnotch(notch_freq, Q=30.0, fs=sfreq)
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

        self._signal_cache.clear()
        self._stats_cache.clear()
        self._signal_cache[subject_id] = (signals, sfreq)
        self._stats_cache[subject_id] = rec_stats
        return signals, sfreq

    def evict_subject(self, subject_id: str) -> None:
        self._signal_cache.pop(subject_id, None)
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
        signals, sfreq = self._load_subject_signals(subject_id)

        n_samples = int(round(self.epoch_seconds * sfreq))
        start = ep_idx * n_samples
        epoch = signals[:, start : start + n_samples]
        if epoch.shape[1] < n_samples:
            epoch = np.pad(epoch, ((0, 0), (0, n_samples - epoch.shape[1])))

        record = self._records_by_id[subject_id]
        experts = self._expert_stages.get(subject_id)

        sample: Dict[str, object] = {
            "eeg": torch.tensor(epoch, dtype=torch.float32),
            "sleep_stage": sleep_stage,
            "subject_id": subject_id,
            "group": record.group,
            "epoch_idx": ep_idx,
            "channels": self._discovered_channels.get(subject_id, self._montage_names),
            "epoch_seconds": self.epoch_seconds,
            "unit": "uV",
            "location": self._DOD_LOCATION.get(record.group, ""),
        }
        rec_stats = self._stats_cache.get(subject_id)
        if rec_stats is not None:
            for k, v in rec_stats.items():
                sample[k] = v
        if experts is not None:
            sample["expert_stages"] = {
                scorer: int(arr[ep_idx]) if ep_idx < len(arr) else -1
                for scorer, arr in experts.items()
            }
        if subject_id in self._fold_assignments:
            sample["fold_assignments"] = self._fold_assignments[subject_id]
        return sample

    @property
    def subjects(self) -> List[str]:
        return [r.subject_id for r in self._subject_records]
