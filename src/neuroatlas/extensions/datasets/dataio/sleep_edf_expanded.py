"""Sleep-EDF Expanded (Cassette + Telemetry) loader for EEGBenchmarks.

Reads raw PSG ``.edf`` files directly at runtime from the dataset's folder
(what PhysioNet's ``sleep-edfx/1.0.0/`` holds: ``sleep-cassette/``,
``sleep-telemetry/`` and the two subject tables). Hypnograms come from
EDF+ annotation files. Subject demographics come from the bundled XLS
spreadsheets (``SC-subjects.xls``, ``ST-subjects.xls``).

Both cassette and telemetry subsets share the same two EEG channels
(``EEG Fpz-Cz``, ``EEG Pz-Oz``) sampled at 100 Hz in µV.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import edfio
import numpy as np
import torch
from scipy.signal import butter, iirnotch, sosfiltfilt
from torch.utils.data import Dataset

from neuroatlas.extensions.datasets.dataio._stage_resample import (
    resample_native_epoch_labels,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_TEXT_MAP: Dict[str, int] = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": -1,
    "Movement time": -1,
}

LABEL_NAMES = ["W", "N1", "N2", "N3", "REM"]

DEFAULT_CHANNELS: List[str] = ["EEG Fpz-Cz", "EEG Pz-Oz"]

_SC_LOCATION = "Leiden University, Netherlands"
_ST_LOCATION = "Westeinde Hospital, The Hague, Netherlands"

_NATIVE_EPOCH = 30

_SC_PSG_RE = re.compile(r"^SC4(\d{2})(\d)\w0-PSG\.edf$")
_ST_PSG_RE = re.compile(r"^ST7(\d{2})(\d)\w0-PSG\.edf$")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    recording_id: str
    subset: str
    night: int
    psg_path: str
    hypnogram_path: str
    age: Optional[float] = None
    sex: Optional[str] = None
    condition: str = "healthy"
    location: str = _SC_LOCATION


# ---------------------------------------------------------------------------
# Demographics loading
# ---------------------------------------------------------------------------

@dataclass
class _SCDemo:
    age: float
    sex: str


@dataclass
class _STDemo:
    age: float
    sex: str
    placebo_night: int
    temazepam_night: int


class DemographicsUnavailableError(OSError):
    """The subject spreadsheet exists in the download but cannot be read
    (an OSError: its text reaches the user as it is, with no class name)."""


def _read_demographics_xls(xls_path: str):
    """Read one of the bundled ``*-subjects.xls`` tables, or say why not.

    These are the only source of age and sex. They used to be skipped on any
    error, which wrote ``"age": null`` into every cached embedding row when
    ``xlrd`` was missing; the cache stayed valid for sleep staging and
    poisoned brain age for good (F-069). Both failures are now loud.
    """
    if not os.path.exists(xls_path):
        # It ships with the PhysioNet download and is the only source of
        # age and sex.
        from neuroatlas.extensions.datasets._missing import no_data

        raise FileNotFoundError(no_data(
            "sleep_edf_expanded", xls_path, "data_root", what="no subject table at",
            detail="the PhysioNet download puts it at the root of the dataset"))
    import pandas as pd
    try:
        return pd.read_excel(xls_path)
    except ImportError as exc:
        # without it every row would carry age = null
        raise DemographicsUnavailableError(
            f"sleep_edf_expanded: cannot read {xls_path}: reading an .xls file needs "
            f"the xlrd package\nfix: pip install xlrd"
        ) from exc
    except Exception as exc:
        from neuroatlas.cli._msg import exception_text

        raise DemographicsUnavailableError(
            f"sleep_edf_expanded: cannot read {xls_path} ({exception_text(exc)})"
        ) from exc


def _load_sc_demographics(xls_path: str) -> Dict[int, _SCDemo]:
    df = _read_demographics_xls(xls_path)
    result: Dict[int, _SCDemo] = {}
    for _, row in df.iterrows():
        try:
            subj = int(row["subject"])
            age = float(row["age"])
            sex_code = int(row["sex (F=1)"])
            sex = "F" if sex_code == 1 else "M"
        except (KeyError, TypeError, ValueError):
            continue
        if subj not in result:
            result[subj] = _SCDemo(age=age, sex=sex)
    return result


def _load_st_demographics(xls_path: str) -> Dict[int, _STDemo]:
    df = _read_demographics_xls(xls_path)
    result: Dict[int, _STDemo] = {}
    for _, row in df.iterrows():
        try:
            subj_raw = row.iloc[0]
            if str(subj_raw).strip().lower() == "nr":
                continue
            subj = int(subj_raw)
            age = float(row.iloc[1])
            sex_code = int(row.iloc[2])
            sex = "M" if sex_code == 1 else "F"
            placebo_night = int(row.iloc[3])
            temazepam_night = int(row.iloc[5])
        except (TypeError, ValueError, IndexError):
            continue
        result[subj] = _STDemo(
            age=age, sex=sex,
            placebo_night=placebo_night, temazepam_night=temazepam_night,
        )
    return result


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def scan_sleep_edf_expanded_subjects(
    data_root: str,
) -> List[SubjectRecord]:
    root = Path(data_root)
    sc_dir = root / "sleep-cassette"
    st_dir = root / "sleep-telemetry"
    sc_xls = str(root / "SC-subjects.xls")
    st_xls = str(root / "ST-subjects.xls")

    # Each table is required only for the subset that is actually present.
    sc_demos = (_load_sc_demographics(sc_xls)
                if any(sc_dir.glob("SC*-PSG.edf")) else {})
    st_demos = (_load_st_demographics(st_xls)
                if any(st_dir.glob("ST*-PSG.edf")) else {})

    records: List[SubjectRecord] = []

    # --- Cassette ---
    if sc_dir.is_dir():
        hyp_lookup: Dict[str, str] = {}
        for p in sorted(sc_dir.glob("*-Hypnogram.edf")):
            hyp_lookup[p.name[:7]] = str(p)

        for psg in sorted(sc_dir.glob("SC*-PSG.edf")):
            m = _SC_PSG_RE.match(psg.name)
            if m is None:
                continue
            subj_num = int(m.group(1))
            night = int(m.group(2))
            prefix = psg.name[:7]
            hyp_path = hyp_lookup.get(prefix)
            if hyp_path is None:
                continue

            demo = sc_demos.get(subj_num)
            recording_id = prefix.rstrip("EFGH")[:6]  # SC4XXN
            records.append(SubjectRecord(
                subject_id=f"SC_{subj_num:02d}",
                recording_id=psg.name[:7],
                subset="cassette",
                night=night,
                psg_path=str(psg),
                hypnogram_path=hyp_path,
                age=demo.age if demo else None,
                sex=demo.sex if demo else None,
                condition="healthy",
                location=_SC_LOCATION,
            ))

    # --- Telemetry ---
    if st_dir.is_dir():
        hyp_lookup_st: Dict[str, str] = {}
        for p in sorted(st_dir.glob("*-Hypnogram.edf")):
            hyp_lookup_st[p.name[:7]] = str(p)

        for psg in sorted(st_dir.glob("ST*-PSG.edf")):
            m = _ST_PSG_RE.match(psg.name)
            if m is None:
                continue
            subj_num = int(m.group(1))
            night = int(m.group(2))
            prefix = psg.name[:7]
            hyp_path = hyp_lookup_st.get(prefix)
            if hyp_path is None:
                continue

            demo = st_demos.get(subj_num)
            condition = "unknown"
            if demo is not None:
                if night == demo.placebo_night:
                    condition = "placebo"
                elif night == demo.temazepam_night:
                    condition = "temazepam"

            records.append(SubjectRecord(
                subject_id=f"ST_{subj_num:02d}",
                recording_id=psg.name[:7],
                subset="telemetry",
                night=night,
                psg_path=str(psg),
                hypnogram_path=hyp_path,
                age=demo.age if demo else None,
                sex=demo.sex if demo else None,
                condition=condition,
                location=_ST_LOCATION,
            ))

    # A table that was read but matched no subject (renamed columns, a
    # different file) would be the same silent null as an unreadable one.
    for subset, xls in (("cassette", sc_xls), ("telemetry", st_xls)):
        rows = [r for r in records if r.subset == subset]
        missing = sorted({r.subject_id for r in rows if r.age is None})
        if rows and len(missing) == len({r.subject_id for r in rows}):
            raise DemographicsUnavailableError(
                f"sleep_edf_expanded: {xls} gives an age for none of the "
                f"{len(missing)} {subset} subjects: it does not have the layout of "
                f"PhysioNet's {subset} subject table"
            )
        if missing:
            from neuroatlas import quiet

            quiet.warn_once(
                logging.getLogger(__name__), f"sleep_edf_expanded no age:{subset}",
                "Sleep-EDF: %s gives no age or sex for %d %s subject(s): %s",
                os.path.basename(xls), len(missing), subset, ", ".join(missing))

    records.sort(key=lambda r: r.recording_id)
    return records


# ---------------------------------------------------------------------------
# Annotation reading — sleep stages
# ---------------------------------------------------------------------------

def read_sleep_edf_expanded_stages(
    hypnogram_path: str,
    psg_duration_sec: float,
    epoch_seconds: float = 30,
) -> Tuple[np.ndarray, float]:
    """Parse EDF+ hypnogram annotations and align to PSG duration.

    Annotations use ``onset`` (seconds from shared PSG/hypnogram start time)
    to place stages. Epochs not covered by any annotation get -1 (unscored).

    Returns ``(stages, scores_end_sec)`` where ``scores_end_sec`` is the
    effective scored duration used for signal loading.
    """
    hyp = edfio.read_edf(hypnogram_path)
    anns = hyp.annotations

    if not anns:
        return np.array([], dtype=np.int16), 0.0

    last_ann = anns[-1]
    annotation_end = last_ann.onset + last_ann.duration
    total_duration = min(psg_duration_sec, annotation_end)
    total_duration = int(total_duration // _NATIVE_EPOCH) * _NATIVE_EPOCH
    n_epochs = int(total_duration // _NATIVE_EPOCH)

    if n_epochs <= 0:
        return np.array([], dtype=np.int16), 0.0

    stages = np.full(n_epochs, -1, dtype=np.int16)

    for ann in anns:
        stage = STAGE_TEXT_MAP.get(ann.text)
        if stage is None:
            continue
        start_epoch = int(ann.onset // _NATIVE_EPOCH)
        n_ann_epochs = int(ann.duration // _NATIVE_EPOCH)
        end_epoch = min(start_epoch + n_ann_epochs, n_epochs)
        if start_epoch < n_epochs:
            stages[max(0, start_epoch):end_epoch] = stage

    scores_end = float(total_duration)

    if int(epoch_seconds) != _NATIVE_EPOCH:
        stages_list = resample_native_epoch_labels(
            stages.tolist(), int(epoch_seconds), native_epoch=_NATIVE_EPOCH,
        )
        stages = np.asarray(stages_list, dtype=np.int16)
        scores_end = float(len(stages) * epoch_seconds)

    return stages, scores_end


# ---------------------------------------------------------------------------
# Channel reading
# ---------------------------------------------------------------------------

def read_sleep_edf_expanded_channels(
    psg_path: str,
    channel_labels: Sequence[str],
    duration_sec: Optional[float] = None,
) -> Tuple[np.ndarray, List[str], float]:
    """Read named channels from a SleepEDF PSG EDF.

    Returns ``(signals, names, sfreq)``. Missing channels are zero-filled.
    """
    edf = edfio.read_edf(psg_path)
    label_to_signal = {sig.label.upper(): sig for sig in edf.signals}

    data: List[Optional[np.ndarray]] = []
    names: List[str] = []
    ref_sfreq: Optional[float] = None
    for label in channel_labels:
        sig = label_to_signal.get(label.upper())
        if sig is None:
            data.append(None)
            names.append("MISSING")
            continue
        sfreq = float(sig.sampling_frequency)
        if duration_sec is not None:
            n = int(duration_sec * sfreq)
            arr = sig.data[:n].astype(np.float32)
        else:
            arr = sig.data.astype(np.float32)
        data.append(arr)
        names.append(sig.label)
        if ref_sfreq is None:
            ref_sfreq = sfreq

    resolved = [c for c in data if c is not None]
    if not resolved:
        return np.array([]), names, 0.0
    min_samples = min(len(c) for c in resolved)
    for i, c in enumerate(data):
        if c is None:
            data[i] = np.zeros(min_samples, dtype=np.float32)
    stacked = np.stack([c[:min_samples] for c in data], axis=0)
    return stacked, names, float(ref_sfreq)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SleepEDFExpandedDataset(Dataset):
    """Epoch-level PyTorch dataset for Sleep-EDF Expanded."""

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
        self._channel_labels = list(channel_specs)
        self._montage_names = list(channel_specs)
        self.channel_specs = channel_specs
        self.epoch_seconds = float(epoch_seconds)
        self.bandpass = bandpass
        self.notch = notch
        self.highpass = highpass
        self._fold_assignments = fold_assignments or {}
        self._compute_recording_stats = compute_recording_stats
        self._use_all_eeg_channels = use_all_eeg_channels
        self._subject_records = list(subject_records)
        self._records_by_id: Dict[str, SubjectRecord] = {}
        self._scores_end: Dict[str, float] = {}
        self._index: List[Tuple[str, int, int]] = []
        self._index_built = False
        self._dropped_records: List[Tuple[str, str]] = []
        self._edf_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._stats_cache: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
        self._discovered_channels: Dict[str, List[str]] = {}

    def _ensure_index_built(self) -> None:
        if self._index_built:
            return
        for record in self._subject_records:
            edf = edfio.read_edf(record.psg_path)
            psg_duration = edf.duration

            if not self._use_all_eeg_channels:
                upper_labels = {sig.label.upper() for sig in edf.signals}
                if not any(label.upper() in upper_labels for label in self._channel_labels):
                    self._dropped_records.append((
                        record.recording_id,
                        f"no channel in EDF header (labels: {sorted(upper_labels)})",
                    ))
                    continue

            stages, scores_end = read_sleep_edf_expanded_stages(
                record.hypnogram_path,
                psg_duration,
                epoch_seconds=self.epoch_seconds,
            )

            self._records_by_id[record.recording_id] = record
            self._scores_end[record.recording_id] = scores_end

            for ep_idx, stage in enumerate(stages):
                self._index.append((record.recording_id, ep_idx, int(stage)))
        self._index_built = True

    def _load_recording_edf(self, recording_id: str) -> Tuple[np.ndarray, float]:
        if recording_id in self._edf_cache:
            return self._edf_cache[recording_id]

        record = self._records_by_id[recording_id]
        scores_end = self._scores_end[recording_id]

        if self._use_all_eeg_channels:
            from ._eeg_channel_discovery import read_all_eeg_channels
            signals, ch_names, sfreq = read_all_eeg_channels(
                record.psg_path, duration_sec=scores_end,
            )
            self._discovered_channels[recording_id] = ch_names
        else:
            signals, _, sfreq = read_sleep_edf_expanded_channels(
                record.psg_path, self._channel_labels, duration_sec=scores_end,
            )

        if self.highpass is not None:
            nyq = sfreq / 2.0
            sos = butter(5, self.highpass / nyq, btype="high", output="sos")
            signals = sosfiltfilt(sos, signals, axis=-1).astype(np.float32)

        if self.bandpass is not None:
            lo, hi = self.bandpass
            nyq = sfreq / 2.0
            sos = butter(5, [lo / nyq, hi / nyq], btype="band", output="sos")
            signals = sosfiltfilt(sos, signals, axis=-1).astype(np.float32)

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
        self._edf_cache[recording_id] = (signals, sfreq)
        self._stats_cache[recording_id] = rec_stats
        return signals, sfreq

    def evict_subject(self, recording_id: str) -> None:
        self._edf_cache.pop(recording_id, None)
        self._stats_cache.pop(recording_id, None)

    @staticmethod
    def eviction_key(meta: Dict[str, object]) -> Optional[str]:
        rid = meta.get("recording_id")
        return str(rid) if rid is not None else None

    def __len__(self) -> int:
        self._ensure_index_built()
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        self._ensure_index_built()
        recording_id, ep_idx, sleep_stage = self._index[idx]
        signals, sfreq = self._load_recording_edf(recording_id)

        n_samples = int(round(self.epoch_seconds * sfreq))
        start = ep_idx * n_samples
        epoch = signals[:, start : start + n_samples]
        if epoch.shape[1] < n_samples:
            epoch = np.pad(epoch, ((0, 0), (0, n_samples - epoch.shape[1])))

        record = self._records_by_id[recording_id]

        sample: Dict[str, object] = {
            "eeg": torch.tensor(epoch, dtype=torch.float32),
            "sleep_stage": sleep_stage,
            "subject_id": record.subject_id,
            "recording_id": recording_id,
            "epoch_idx": ep_idx,
            "channels": self._discovered_channels.get(recording_id, self._montage_names),
            "epoch_seconds": self.epoch_seconds,
            "unit": "uV",
            "age": record.age,
            "sex": record.sex,
            "subset": record.subset,
            "night": record.night,
            "condition": record.condition,
            "location": record.location,
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
        return [r.subject_id for r in self._subject_records]
