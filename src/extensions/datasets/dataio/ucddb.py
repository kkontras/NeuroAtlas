"""UCDDB (St Vincent's / UCD Sleep Apnea Database) loader for EEGBenchmarks.

Reads raw ``.rec`` EDF files directly at runtime. Sleep stages come from
``_stage.txt`` (one integer per line, 30 s epochs). Respiratory events come
from ``_respevt.txt`` (fixed-width, time-of-day + duration) and are
converted to per-epoch fractions per subtype. Subject age is loaded from
``SubjectDetails`` (CSV; fallback to XLS via xlrd when available).

Only the EEG ``.rec`` file is used; the accompanying ``_lifecard.edf``
ambulatory ECG file is ignored.
"""

from __future__ import annotations

import datetime
import os
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

# 0=Wake, 1=REM, 2=Stage1, 3=Stage2, 4=Stage3, 5=Stage4, 6=Artifact, 7=Indeterminate
UCDDB_STAGE_MAP: Dict[int, int] = {
    0: 0,
    1: 4,
    2: 1,
    3: 2,
    4: 3,
    5: 3,
    6: -1,
    7: -1,
}

LABEL_NAMES = ["W", "N1", "N2", "N3", "REM"]

DEFAULT_RAW_ROOT = "${EEG_DATA_ROOT}/data/stvincent_ucddb/files/ucddb/1.0.0"
DEFAULT_AGE_CSV = f"{DEFAULT_RAW_ROOT}/SubjectDetails.csv"
DEFAULT_AGE_XLS = f"{DEFAULT_RAW_ROOT}/SubjectDetails.xls"

DEFAULT_CHANNELS: List[str] = ["C3A2", "C4A1"]

RESP_EVENT_SUBTYPES: List[str] = [
    "apnea_obstructive",
    "apnea_central",
    "apnea_mixed",
    "hypopnea_obstructive",
    "hypopnea_central",
    "hypopnea_mixed",
    "periodic_breathing",
]
RESP_EVENT_AGGREGATES: List[str] = [
    "apnea_any",
    "hypopnea_any",
    "ahi",
    "respiratory_event_any",
]
RESP_EVENT_NAMES: List[str] = RESP_EVENT_SUBTYPES + RESP_EVENT_AGGREGATES
RESP_EVENT_FRACTION_FIELDS: List[str] = [f"{n}_fraction" for n in RESP_EVENT_NAMES]

_RESPEVT_HEADER_LINES = 3
_RESPEVT_TYPE_TO_SUBTYPE: Dict[str, str] = {
    "APNEA-O": "apnea_obstructive",
    "APNEA-C": "apnea_central",
    "APNEA-M": "apnea_mixed",
    "HYP-O": "hypopnea_obstructive",
    "HYP-C": "hypopnea_central",
    "HYP-M": "hypopnea_mixed",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    edf_path: str
    stage_path: str
    respevt_path: str
    age: Optional[float]
    sex: Optional[str] = None


# ---------------------------------------------------------------------------
# Demographics lookup
# ---------------------------------------------------------------------------

@dataclass
class _Demographics:
    age: Optional[float] = None
    sex: Optional[str] = None


def _load_demographics_lookup(
    csv_path: Optional[str],
    xls_path: Optional[str] = None,
) -> Dict[str, _Demographics]:
    """Return ``{subject_id_lower: _Demographics}``.

    Tries ``csv_path`` first (if set and present), then falls back to
    ``xls_path`` via pandas + xlrd. Returns empty dict if neither is usable.
    Any parse failure is swallowed so the adapter stays usable without
    metadata.
    """
    result: Dict[str, _Demographics] = {}
    paths: List[Tuple[str, str]] = []
    if csv_path:
        paths.append(("csv", csv_path))
    if xls_path:
        paths.append(("xls", xls_path))

    for kind, path in paths:
        if not os.path.exists(path):
            continue
        try:
            if kind == "csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
        except Exception:
            continue

        id_col = _find_column(df, ("study number", "subject", "subject id", "study"))
        if id_col is None:
            continue
        age_col = _find_column(df, ("age",))
        sex_col = _find_column(df, ("gender", "sex"))

        for _, row in df.iterrows():
            raw_id = row[id_col]
            if pd.isna(raw_id):
                continue
            sid = str(raw_id).strip().lower()
            if not sid.startswith("ucddb"):
                try:
                    sid = f"ucddb{int(float(sid)):03d}"
                except ValueError:
                    continue

            age: Optional[float] = None
            if age_col is not None and not pd.isna(row[age_col]):
                try:
                    age = float(row[age_col])
                except (TypeError, ValueError):
                    pass

            sex: Optional[str] = None
            if sex_col is not None and not pd.isna(row[sex_col]):
                sex = str(row[sex_col]).strip()

            result[sid] = _Demographics(age=age, sex=sex)
        if result:
            return result
    return result


def _find_column(df: "pd.DataFrame", needles: Sequence[str]) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for n in needles:
        if n in lowered:
            return lowered[n]
    for n in needles:
        for low, orig in lowered.items():
            if n in low:
                return orig
    return None


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def scan_ucddb_subjects(
    data_root: str,
    age_csv: Optional[str] = None,
    age_xls: Optional[str] = None,
) -> List[SubjectRecord]:
    root = Path(data_root)
    records_file = root / "RECORDS"
    if not records_file.exists():
        return []

    ids: List[str] = []
    seen: set = set()
    with open(records_file) as fh:
        for line in fh:
            entry = line.strip()
            if not entry or not entry.endswith(".rec"):
                # Skip *_lifecard.edf and any stray entries.
                continue
            subject_id = entry[:-4]
            if subject_id in seen:
                continue
            seen.add(subject_id)
            ids.append(subject_id)

    demographics = _load_demographics_lookup(age_csv, age_xls)

    records: List[SubjectRecord] = []
    for subject_id in ids:
        edf_path = root / f"{subject_id}.rec"
        stage_path = root / f"{subject_id}_stage.txt"
        respevt_path = root / f"{subject_id}_respevt.txt"
        if not (edf_path.exists() and stage_path.exists() and respevt_path.exists()):
            continue
        demo = demographics.get(subject_id.lower())
        records.append(SubjectRecord(
            subject_id=subject_id,
            edf_path=str(edf_path),
            stage_path=str(stage_path),
            respevt_path=str(respevt_path),
            age=demo.age if demo else None,
            sex=demo.sex if demo else None,
        ))
    return records


# ---------------------------------------------------------------------------
# Channel reading
# ---------------------------------------------------------------------------

def read_ucddb_channels(
    edf_path: str,
    channel_labels: Sequence[str],
    duration_sec: Optional[float] = None,
) -> Tuple[np.ndarray, List[str], float]:
    """Read named channels from a UCDDB .rec EDF.

    Channels are pre-referenced bipolar (``C3A2``, ``C4A1``) and are read
    directly by EDF label (case-insensitive). Returns ``(signals, names, sfreq)``.
    Missing channels are zero-filled so model input dimensionality stays fixed.
    """
    edf = edfio.read_edf(edf_path)
    label_to_signal = {sig.label.upper(): sig for sig in edf.signals}

    data: List[np.ndarray] = []
    names: List[str] = []
    ref_sfreq: Optional[float] = None
    for label in channel_labels:
        sig = label_to_signal.get(label.upper())
        if sig is None:
            data.append(None)  # filled after min-length resolved
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
# Annotation reading — sleep stages
# ---------------------------------------------------------------------------

_UCDDB_NATIVE_EPOCH = 30


def read_ucddb_stages(
    stage_path: str,
    epoch_seconds: float = 30,
) -> Tuple[np.ndarray, float, float]:
    """Parse ``<id>_stage.txt`` at the 30 s native grid and resample.

    One integer per line (CRLF-safe). The file is always per-30 s native
    epoch, anchored to EDF second 0. Target-epoch resampling goes through
    ``resample_native_epoch_labels``, which handles divisor, multiple, and
    non-divisor epoch lengths under the single rule "all underlying native
    labels agree -> that label, else -1".
    """
    stages_native: List[int] = []
    with open(stage_path) as fh:
        for line in fh:
            token = line.strip()
            if not token:
                continue
            try:
                raw = int(token)
            except ValueError:
                continue
            stages_native.append(UCDDB_STAGE_MAP.get(raw, -1))

    stages = resample_native_epoch_labels(
        stages_native, int(epoch_seconds), native_epoch=_UCDDB_NATIVE_EPOCH,
    )
    arr = np.asarray(stages, dtype=np.int16)
    scores_end = float(len(arr) * epoch_seconds)
    return arr, 0.0, scores_end


# ---------------------------------------------------------------------------
# Annotation reading — respiratory events
# ---------------------------------------------------------------------------

def _edf_starttime_seconds(start_time: datetime.time) -> float:
    return start_time.hour * 3600.0 + start_time.minute * 60.0 + start_time.second + start_time.microsecond / 1e6


def _parse_time_of_day(token: str) -> Optional[float]:
    try:
        hh, mm, ss = token.split(":")
        return int(hh) * 3600.0 + int(mm) * 60.0 + int(ss)
    except (ValueError, AttributeError):
        return None


def read_ucddb_respiratory_events(
    respevt_path: str,
    epoch_seconds: float,
    scores_end: float,
    edf_start_time: datetime.time,
) -> Dict[str, np.ndarray]:
    """Compute per-epoch respiratory-event fractions for UCDDB.

    Returns one ``float32`` array per subtype (obstructive / central / mixed
    apnea & hypopnea, plus periodic breathing) and per aggregate
    (``apnea_any``, ``hypopnea_any``, ``respiratory_event_any``), each of
    shape ``(n_epochs,)`` with values in ``[0, 1]``.
    """
    epoch_sec = float(epoch_seconds)
    n_epochs = max(0, int(scores_end // epoch_sec))
    arrays: Dict[str, np.ndarray] = {
        name: np.zeros(n_epochs, dtype=np.float32)
        for name in RESP_EVENT_NAMES
    }
    if n_epochs == 0 or not os.path.exists(respevt_path):
        return arrays

    edf_start_sec = _edf_starttime_seconds(edf_start_time)

    def _accumulate(name: str, onset_sec: float, duration_sec: float) -> None:
        arr = arrays[name]
        a_start = onset_sec
        a_end = onset_sec + duration_sec
        if a_end <= 0 or a_start >= scores_end:
            return
        a_start = max(a_start, 0.0)
        a_end = min(a_end, scores_end)
        ep_start = int(a_start // epoch_sec)
        ep_end = min(int(a_end // epoch_sec) + 1, n_epochs)
        for ep in range(ep_start, ep_end):
            ep_begin = ep * epoch_sec
            ep_finish = ep_begin + epoch_sec
            overlap = min(a_end, ep_finish) - max(a_start, ep_begin)
            if overlap > 0:
                arr[ep] += overlap / epoch_sec

    with open(respevt_path, errors="replace") as fh:
        lines = fh.readlines()

    for raw in lines[_RESPEVT_HEADER_LINES:]:
        line = raw.rstrip("\r\n")
        if len(line) < 9 or not line.strip():
            continue
        time_token = line[:8].strip()
        onset_abs = _parse_time_of_day(time_token)
        if onset_abs is None:
            continue
        rest_tokens = line[8:].split()
        if not rest_tokens:
            continue
        type_token = rest_tokens[0]

        duration_sec: Optional[int] = None
        for tok in rest_tokens[1:]:
            if tok.isdigit():
                val = int(tok)
                if 1 <= val <= 600:
                    duration_sec = val
                    break
        if duration_sec is None:
            continue

        onset_rel = onset_abs - edf_start_sec
        if onset_rel < 0:
            onset_rel += 86400.0  # midnight rollover

        subtype = _RESPEVT_TYPE_TO_SUBTYPE.get(type_token)
        if subtype is not None:
            _accumulate(subtype, onset_rel, float(duration_sec))

        is_pb = type_token in {"PB", "CS"} or any(
            tok in {"PB", "CS"} for tok in rest_tokens[1:3]
        )
        if is_pb:
            _accumulate("periodic_breathing", onset_rel, float(duration_sec))

    arrays["apnea_any"] = np.clip(
        arrays["apnea_obstructive"]
        + arrays["apnea_central"]
        + arrays["apnea_mixed"],
        0.0, 1.0,
    )
    arrays["hypopnea_any"] = np.clip(
        arrays["hypopnea_obstructive"]
        + arrays["hypopnea_central"]
        + arrays["hypopnea_mixed"],
        0.0, 1.0,
    )
    arrays["ahi"] = np.clip(
        arrays["apnea_any"] + arrays["hypopnea_any"],
        0.0, 1.0,
    )
    arrays["respiratory_event_any"] = arrays["ahi"].copy()
    for name in RESP_EVENT_SUBTYPES:
        np.clip(arrays[name], 0.0, 1.0, out=arrays[name])

    return arrays


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UCDDBDataset(Dataset):
    """Epoch-level PyTorch dataset for UCDDB."""

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
        self._resp_fractions: Dict[str, Dict[str, np.ndarray]] = {}
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
            edf = edfio.read_edf(record.edf_path)
            if not self._use_all_eeg_channels:
                upper_labels = {sig.label.upper() for sig in edf.signals}
                if not any(label.upper() in upper_labels for label in self._channel_labels):
                    self._dropped_records.append((
                        record.subject_id,
                        f"no channel in EDF header (labels: {sorted(upper_labels)})",
                    ))
                    continue

            stages, _, scores_end = read_ucddb_stages(
                record.stage_path, epoch_seconds=self.epoch_seconds,
            )
            resp_fractions = read_ucddb_respiratory_events(
                record.respevt_path,
                self.epoch_seconds,
                scores_end,
                edf.starttime,
            )

            self._records_by_id[record.subject_id] = record
            self._scores_end[record.subject_id] = scores_end
            self._resp_fractions[record.subject_id] = resp_fractions

            for ep_idx, stage in enumerate(stages):
                self._index.append((record.subject_id, ep_idx, int(stage)))
        self._index_built = True

    def _load_subject_edf(self, subject_id: str) -> Tuple[np.ndarray, float]:
        if subject_id in self._edf_cache:
            return self._edf_cache[subject_id]

        record = self._records_by_id[subject_id]
        scores_end = self._scores_end[subject_id]

        if self._use_all_eeg_channels:
            from ._eeg_channel_discovery import read_all_eeg_channels
            signals, ch_names, sfreq = read_all_eeg_channels(
                record.edf_path, duration_sec=scores_end,
            )
            self._discovered_channels[subject_id] = ch_names
        else:
            signals, _, sfreq = read_ucddb_channels(
                record.edf_path, self._channel_labels, duration_sec=scores_end,
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

        if self.notch is not None:
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
        self._edf_cache[subject_id] = (signals, sfreq)
        self._stats_cache[subject_id] = rec_stats
        return signals, sfreq

    def evict_subject(self, subject_id: str) -> None:
        self._edf_cache.pop(subject_id, None)
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
        signals, sfreq = self._load_subject_edf(subject_id)

        n_samples = int(round(self.epoch_seconds * sfreq))
        start = ep_idx * n_samples
        epoch = signals[:, start : start + n_samples]
        if epoch.shape[1] < n_samples:
            epoch = np.pad(epoch, ((0, 0), (0, n_samples - epoch.shape[1])))

        record = self._records_by_id[subject_id]
        resp = self._resp_fractions[subject_id]

        sample: Dict[str, object] = {
            "eeg": torch.tensor(epoch, dtype=torch.float32),
            "sleep_stage": sleep_stage,
            "subject_id": subject_id,
            "epoch_idx": ep_idx,
            "channels": (self._discovered_channels.get(subject_id, self._montage_names)),
            "epoch_seconds": self.epoch_seconds,
            "unit": "uV",
            "age": record.age,
            "sex": record.sex,
            "location": "Dublin, Ireland",
        }
        rec_stats = self._stats_cache.get(subject_id)
        if rec_stats is not None:
            for k, v in rec_stats.items():
                sample[k] = v
        for name in RESP_EVENT_NAMES:
            arr = resp[name]
            sample[f"{name}_fraction"] = float(arr[ep_idx]) if ep_idx < len(arr) else 0.0
        if subject_id in self._fold_assignments:
            sample["fold_assignments"] = self._fold_assignments[subject_id]
        return sample

    @property
    def subjects(self) -> List[str]:
        return [r.subject_id for r in self._subject_records]
