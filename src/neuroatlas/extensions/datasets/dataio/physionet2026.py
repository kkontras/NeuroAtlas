"""PhysioNet Challenge 2026 loader for EEGBenchmarks.

Reads per-subject EDF files directly at runtime from the training set.
Each subject has:

    physiological_data/<site>/<bids>_ses-<n>.edf       -- PSG signals
    human_annotations/<site>/<bids>_ses-<n>_expert_annotations.edf
    algorithmic_annotations/<site>/<bids>_ses-<n>_caisr_annotations.edf

Human annotations contain:
    stage_expert    (1/30 Hz)  -- sleep stage per 30 s epoch
    arousal_expert  (2 Hz)     -- binary arousal at 0.5 s resolution
    resp_expert     (1 Hz)     -- respiratory event code per second
    limb_expert     (1 Hz)     -- limb movement code per second

Stage codes (physionet): 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 0=Wake, 9=Unavailable.
Mapped to canonical: 0=W, 1=N1, 2=N2, 3=N3, 4=REM, -1=unscored.

Sites: S0001 (BIDMC), I0002 (Emory), I0006 (MGB).
S0001/I0002 have pre-referenced bipolar channels (C3-M2, etc.).
I0006 has unreferenced monopolar channels requiring runtime subtraction.
"""

from __future__ import annotations

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

from neuroatlas.extensions.datasets.dataio._stage_resample import (
    resample_native_epoch_labels,
)


LABEL_NAMES = ["W", "N1", "N2", "N3", "REM"]
NATIVE_EPOCH = 30

SITE_LOCATION = {
    "S0001": "Boston, Massachusetts, US",
    "I0002": "Atlanta, Georgia, US",
    "I0006": "Boston, Massachusetts, US",
}

# Physionet stage code -> canonical code
_STAGE_MAP = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 0, 9: -1}

# Respiratory event codes (human annotation)
RESP_SUBTYPES = [
    "obstructive_apnea",
    "central_apnea",
    "mixed_apnea",
    "hypopnea",
    "rera",
]
RESP_AGGREGATES = [
    "apnea_any",
    "ahi",
    "rdi",
    "respiratory_event_any",
]
RESP_EVENT_NAMES = RESP_SUBTYPES + RESP_AGGREGATES
RESP_EVENT_FRACTION_FIELDS = [f"{n}_fraction" for n in RESP_EVENT_NAMES]

# Code -> subtype mapping for resp_expert
_RESP_CODE_TO_SUBTYPE = {
    1: "obstructive_apnea",
    2: "central_apnea",
    3: "mixed_apnea",
    4: "hypopnea",
    5: "hypopnea",
    6: "hypopnea",
    7: "rera",
    8: "obstructive_apnea",  # unspecified apnea -> fold into OA
    9: "hypopnea",           # unspecified hypopnea
}

LIMB_EVENT_NAMES = ["limb_movement_any", "limb_movement_plm"]
LIMB_EVENT_FRACTION_FIELDS = [f"{n}_fraction" for n in LIMB_EVENT_NAMES]

ALL_EVENT_FRACTION_FIELDS = RESP_EVENT_FRACTION_FIELDS + LIMB_EVENT_FRACTION_FIELDS

# CAISR metadata field names (per-epoch values stored in meta)
CAISR_FIELDS = [
    "caisr_stage",
    "caisr_arousal_fraction",
    "caisr_obstructive_apnea_fraction",
    "caisr_central_apnea_fraction",
    "caisr_mixed_apnea_fraction",
    "caisr_hypopnea_fraction",
    "caisr_rera_fraction",
    "caisr_apnea_any_fraction",
    "caisr_respiratory_event_any_fraction",
    "caisr_limb_movement_any_fraction",
    "caisr_limb_movement_plm_fraction",
    "caisr_prob_n3",
    "caisr_prob_n2",
    "caisr_prob_n1",
    "caisr_prob_r",
    "caisr_prob_w",
    "caisr_prob_arousal",
]

# Channel spec: (bipolar_label, monopolar_a, monopolar_b)
ChannelSpec = Tuple[str, str, str]

DEFAULT_CHANNELS: List[str] = ["C3-M2", "C4-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"]

DEFAULT_CHANNEL_SPECS: List[ChannelSpec] = [
    ("C3-M2", "C3", "M2"),
    ("C4-M1", "C4", "M1"),
    ("F3-M2", "F3", "M2"),
    ("F4-M1", "F4", "M1"),
    ("O1-M2", "O1", "M2"),
    ("O2-M1", "O2", "M1"),
]


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    site_id: str
    psg_path: str
    annotation_path: Optional[str]
    caisr_annotation_path: Optional[str]
    age: Optional[float]
    sex: Optional[str]
    cognitive_impairment: Optional[bool]
    time_to_event: Optional[float]
    time_to_last_visit: Optional[float]
    bmi: Optional[float]

    @property
    def has_human_annotations(self) -> bool:
        return self.annotation_path is not None


def scan_physionet2026_subjects(data_root: str) -> List[SubjectRecord]:
    """Build SubjectRecord list from demographics.csv and directory scan."""
    root = Path(data_root)
    csv_path = root / "demographics.csv"
    df = pd.read_csv(csv_path)

    records: List[SubjectRecord] = []
    for _, row in df.iterrows():
        site = str(row["SiteID"])
        bids = str(row["BidsFolder"])
        session = int(row["SessionID"])
        stem = f"{bids}_ses-{session}"

        psg_path = root / "physiological_data" / site / f"{stem}.edf"
        if not psg_path.exists():
            continue

        ann_path = root / "human_annotations" / site / f"{stem}_expert_annotations.edf"
        caisr_path = root / "algorithmic_annotations" / site / f"{stem}_caisr_annotations.edf"

        sex_raw = row.get("Sex")
        sex = None
        if sex_raw == "Male":
            sex = "M"
        elif sex_raw == "Female":
            sex = "F"

        ci_raw = row.get("Cognitive_Impairment")
        ci = None
        if ci_raw is True or ci_raw == "True" or ci_raw == True:
            ci = True
        elif ci_raw is False or ci_raw == "False" or ci_raw == False:
            ci = False

        age = float(row["Age"]) if pd.notna(row.get("Age")) else None
        tte = float(row["Time_to_Event"]) if pd.notna(row.get("Time_to_Event")) else None
        ttlv = float(row["Time_to_Last_Visit"]) if pd.notna(row.get("Time_to_Last_Visit")) else None
        bmi = float(row["BMI"]) if pd.notna(row.get("BMI")) else None

        records.append(SubjectRecord(
            subject_id=str(row["BDSPPatientID"]),
            site_id=site,
            psg_path=str(psg_path),
            annotation_path=str(ann_path) if ann_path.exists() else None,
            caisr_annotation_path=str(caisr_path) if caisr_path.exists() else None,
            age=age,
            sex=sex,
            cognitive_impairment=ci,
            time_to_event=tte,
            time_to_last_visit=ttlv,
            bmi=bmi,
        ))
    return records


def _validate_epoch_seconds(epoch_seconds: float) -> int:
    if int(epoch_seconds) != epoch_seconds or epoch_seconds <= 0:
        raise ValueError(
            f"epoch_seconds must be a positive integer, got {epoch_seconds!r}"
        )
    return int(epoch_seconds)


def _map_stage(x: int) -> int:
    return _STAGE_MAP.get(x, -1)


def _get_edf_signal(edf: edfio.Edf, label: str) -> Optional[edfio.EdfSignal]:
    for s in edf.signals:
        if s.label.strip() == label:
            return s
    return None


def _get_psg_duration_epochs(psg_path: str) -> int:
    """Get number of 30s epochs from PSG file duration."""
    edf = edfio.read_edf(psg_path)
    ch = edf.signals[0]
    dur_sec = len(ch.data) / ch.sampling_frequency
    return int(dur_sec // NATIVE_EPOCH)


def read_physionet2026_stages(
    annotation_path: Optional[str],
    epoch_seconds: float = NATIVE_EPOCH,
    fallback_n_epochs: Optional[int] = None,
) -> Tuple[np.ndarray, float, float]:
    """Read human sleep stages and resample to epoch_seconds.

    Returns (stages, 0.0, scores_end). If annotation_path is None,
    returns all-invalid (-1) labels using fallback_n_epochs.
    """
    eps = _validate_epoch_seconds(epoch_seconds)

    if annotation_path is None:
        n = fallback_n_epochs or 0
        stages = resample_native_epoch_labels(
            [-1] * n, eps, native_epoch=NATIVE_EPOCH,
        )
        arr = np.asarray(stages, dtype=np.int16)
        return arr, 0.0, float(len(arr) * eps)

    edf = edfio.read_edf(annotation_path)
    sig = _get_edf_signal(edf, "stage_expert")
    if sig is None:
        raise ValueError(f"No stage_expert signal in {annotation_path}")

    native = [_map_stage(int(v)) for v in sig.data]
    stages = resample_native_epoch_labels(native, eps, native_epoch=NATIVE_EPOCH)
    arr = np.asarray(stages, dtype=np.int16)
    return arr, 0.0, float(len(arr) * eps)


def _signal_to_epoch_fractions(
    data: np.ndarray,
    signal_fs: float,
    epoch_seconds: float,
    n_epochs: int,
    positive_mask: np.ndarray,
) -> np.ndarray:
    """Convert a binary mask at signal_fs to per-epoch fractions."""
    samples_per_epoch = int(round(signal_fs * epoch_seconds))
    out = np.zeros(n_epochs, dtype=np.float32)
    for ep in range(n_epochs):
        start = ep * samples_per_epoch
        end = start + samples_per_epoch
        chunk = positive_mask[start:end]
        if len(chunk) > 0:
            out[ep] = float(chunk.sum()) / len(chunk)
    return np.clip(out, 0.0, 1.0)


def read_physionet2026_arousals(
    annotation_path: Optional[str],
    epoch_seconds: float,
    n_epochs: int,
) -> Optional[np.ndarray]:
    """Read arousal_expert -> per-epoch arousal fraction."""
    if annotation_path is None:
        return None

    edf = edfio.read_edf(annotation_path)
    sig = _get_edf_signal(edf, "arousal_expert")
    if sig is None:
        return None

    data = sig.data.astype(np.int8)
    mask = (data == 1)
    return _signal_to_epoch_fractions(
        data, sig.sampling_frequency, epoch_seconds, n_epochs, mask,
    )


def read_physionet2026_respiratory(
    annotation_path: Optional[str],
    epoch_seconds: float,
    n_epochs: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Read resp_expert -> per-epoch respiratory event fractions."""
    if annotation_path is None:
        return None

    edf = edfio.read_edf(annotation_path)
    sig = _get_edf_signal(edf, "resp_expert")
    if sig is None:
        return None

    data = sig.data.astype(np.int8)
    fs = sig.sampling_frequency
    result: Dict[str, np.ndarray] = {}

    for name in RESP_SUBTYPES:
        codes = [c for c, s in _RESP_CODE_TO_SUBTYPE.items() if s == name]
        mask = np.isin(data, codes)
        result[name] = _signal_to_epoch_fractions(data, fs, epoch_seconds, n_epochs, mask)

    result["apnea_any"] = np.clip(
        result["obstructive_apnea"] + result["central_apnea"] + result["mixed_apnea"],
        0.0, 1.0,
    )
    result["ahi"] = np.clip(
        result["apnea_any"] + result["hypopnea"],
        0.0, 1.0,
    )
    result["rdi"] = np.clip(
        result["ahi"] + result["rera"],
        0.0, 1.0,
    )
    result["respiratory_event_any"] = result["rdi"].copy()
    return result


def read_physionet2026_limb_movements(
    annotation_path: Optional[str],
    epoch_seconds: float,
    n_epochs: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Read limb_expert -> per-epoch limb movement fractions."""
    if annotation_path is None:
        return None

    edf = edfio.read_edf(annotation_path)
    sig = _get_edf_signal(edf, "limb_expert")
    if sig is None:
        return None

    data = sig.data.astype(np.int8)
    fs = sig.sampling_frequency

    any_mask = (data != 0)
    plm_mask = (data == 2)

    return {
        "limb_movement_any": _signal_to_epoch_fractions(data, fs, epoch_seconds, n_epochs, any_mask),
        "limb_movement_plm": _signal_to_epoch_fractions(data, fs, epoch_seconds, n_epochs, plm_mask),
    }


def read_physionet2026_caisr(
    caisr_annotation_path: Optional[str],
    epoch_seconds: float,
    n_epochs: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Read CAISR algorithmic annotations into per-epoch metadata arrays."""
    if caisr_annotation_path is None:
        return None

    edf = edfio.read_edf(caisr_annotation_path)
    result: Dict[str, np.ndarray] = {}
    eps = _validate_epoch_seconds(epoch_seconds)

    # Stage
    sig = _get_edf_signal(edf, "stage_caisr")
    if sig is not None:
        native = [_map_stage(int(v)) for v in sig.data]
        stages = resample_native_epoch_labels(native, eps, native_epoch=NATIVE_EPOCH)
        result["caisr_stage"] = np.asarray(stages[:n_epochs], dtype=np.int16)

    # Arousal fraction
    sig = _get_edf_signal(edf, "arousal_caisr")
    if sig is not None:
        data = sig.data.astype(np.int8)
        mask = (data == 1)
        result["caisr_arousal_fraction"] = _signal_to_epoch_fractions(
            data, sig.sampling_frequency, epoch_seconds, n_epochs, mask,
        )

    # Respiratory
    sig = _get_edf_signal(edf, "resp_caisr")
    if sig is not None:
        data = sig.data.astype(np.int8)
        fs = sig.sampling_frequency
        for name in RESP_SUBTYPES:
            codes = [c for c, s in _RESP_CODE_TO_SUBTYPE.items() if s == name]
            mask = np.isin(data, codes)
            result[f"caisr_{name}_fraction"] = _signal_to_epoch_fractions(
                data, fs, epoch_seconds, n_epochs, mask,
            )
        result["caisr_apnea_any_fraction"] = np.clip(
            result.get("caisr_obstructive_apnea_fraction", np.zeros(n_epochs))
            + result.get("caisr_central_apnea_fraction", np.zeros(n_epochs))
            + result.get("caisr_mixed_apnea_fraction", np.zeros(n_epochs)),
            0.0, 1.0,
        )
        result["caisr_respiratory_event_any_fraction"] = np.clip(
            result.get("caisr_apnea_any_fraction", np.zeros(n_epochs))
            + result.get("caisr_hypopnea_fraction", np.zeros(n_epochs))
            + result.get("caisr_rera_fraction", np.zeros(n_epochs)),
            0.0, 1.0,
        )

    # Limb
    sig = _get_edf_signal(edf, "limb_caisr")
    if sig is not None:
        data = sig.data.astype(np.int8)
        fs = sig.sampling_frequency
        result["caisr_limb_movement_any_fraction"] = _signal_to_epoch_fractions(
            data, fs, epoch_seconds, n_epochs, (data != 0),
        )
        result["caisr_limb_movement_plm_fraction"] = _signal_to_epoch_fractions(
            data, fs, epoch_seconds, n_epochs, (data == 2),
        )

    # Stage probabilities
    for prob_label, field in [
        ("caisr_prob_n3", "caisr_prob_n3"),
        ("caisr_prob_n2", "caisr_prob_n2"),
        ("caisr_prob_n1", "caisr_prob_n1"),
        ("caisr_prob_r", "caisr_prob_r"),
        ("caisr_prob_w", "caisr_prob_w"),
    ]:
        sig = _get_edf_signal(edf, prob_label)
        if sig is not None:
            arr = sig.data.astype(np.float32)[:n_epochs]
            if len(arr) < n_epochs:
                arr = np.pad(arr, (0, n_epochs - len(arr)))
            result[field] = arr

    # Arousal probability
    sig = _get_edf_signal(edf, "caisr_prob_arous")
    if sig is not None:
        data = sig.data.astype(np.float32)
        samples_per_epoch = int(round(sig.sampling_frequency * epoch_seconds))
        probs = np.zeros(n_epochs, dtype=np.float32)
        for ep in range(n_epochs):
            start = ep * samples_per_epoch
            end = start + samples_per_epoch
            chunk = data[start:end]
            if len(chunk) > 0:
                probs[ep] = float(chunk.mean())
        result["caisr_prob_arousal"] = probs

    return result


def read_physionet2026_channels(
    psg_path: str,
    channel_specs: Sequence[ChannelSpec],
    duration_sec: Optional[float] = None,
) -> Tuple[np.ndarray, List[str], float]:
    """Read EEG channels, handling both bipolar and monopolar sites.

    For each channel spec (bipolar_label, mono_a, mono_b):
      - Try reading bipolar_label directly
      - If not found, read mono_a and mono_b and subtract (mono_a - mono_b)
      - If neither works, zero-fill

    Returns (stacked_channels (C, T), display_names, sfreq).
    """
    edf = edfio.read_edf(psg_path)

    signal_map: Dict[str, edfio.EdfSignal] = {}
    for s in edf.signals:
        signal_map[s.label.strip()] = s

    data: List[Optional[np.ndarray]] = []
    ref_sfreq: Optional[float] = None

    for bipolar, mono_a, mono_b in channel_specs:
        if bipolar in signal_map:
            sig = signal_map[bipolar]
            sfreq = sig.sampling_frequency
            arr = sig.data.astype(np.float32)
            if duration_sec is not None:
                n = int(duration_sec * sfreq)
                arr = arr[:n]
            data.append(arr)
            if ref_sfreq is None:
                ref_sfreq = sfreq
        elif mono_a in signal_map and mono_b in signal_map:
            sig_a = signal_map[mono_a]
            sig_b = signal_map[mono_b]
            sfreq = sig_a.sampling_frequency
            arr_a = sig_a.data.astype(np.float32)
            arr_b = sig_b.data.astype(np.float32)
            min_len = min(len(arr_a), len(arr_b))
            if duration_sec is not None:
                min_len = min(min_len, int(duration_sec * sfreq))
            arr = arr_a[:min_len] - arr_b[:min_len]
            data.append(arr)
            if ref_sfreq is None:
                ref_sfreq = sfreq
        elif f"{mono_a}-AVG" in signal_map:
            sig = signal_map[f"{mono_a}-AVG"]
            sfreq = sig.sampling_frequency
            arr = sig.data.astype(np.float32)
            if duration_sec is not None:
                n = int(duration_sec * sfreq)
                arr = arr[:n]
            data.append(arr)
            if ref_sfreq is None:
                ref_sfreq = sfreq
        else:
            data.append(None)

    resolved = [c for c in data if c is not None]
    if not resolved:
        return np.array([]), ["MISSING"] * len(channel_specs), 0.0

    min_samples = min(len(c) for c in resolved)
    for i, c in enumerate(data):
        if c is None:
            data[i] = np.zeros(min_samples, dtype=np.float32)

    stacked = np.stack([c[:min_samples] for c in data], axis=0)
    return stacked, [bipolar for bipolar, _, _ in channel_specs], float(ref_sfreq or 0.0)


def _display_names_to_channel_specs(names: List[str]) -> List[ChannelSpec]:
    """Convert bipolar display names like 'C3-M2' to (bipolar, mono_a, mono_b) tuples."""
    specs: List[ChannelSpec] = []
    for name in names:
        if "-" in name:
            parts = name.split("-", 1)
            specs.append((name, parts[0], parts[1]))
        else:
            specs.append((name, name, ""))
    return specs


class PhysioNet2026Dataset(Dataset):
    """Epoch-level PyTorch dataset for PhysioNet Challenge 2026."""

    def __init__(
        self,
        subject_records: Sequence[SubjectRecord],
        channel_specs: List[str] = DEFAULT_CHANNELS,
        epoch_seconds: float = NATIVE_EPOCH,
        bandpass: Optional[Tuple[float, float]] = None,
        notch: Optional[float] = None,
        highpass: Optional[float] = None,
        fold_assignments: Optional[Dict[str, Dict[str, str]]] = None,
        compute_recording_stats: bool = False,
        use_all_eeg_channels: bool = False,
    ) -> None:
        self._channel_list = _display_names_to_channel_specs(list(channel_specs))
        self._montage_names = list(channel_specs)
        self.channel_specs = channel_specs
        self.epoch_seconds = float(_validate_epoch_seconds(epoch_seconds))
        self.bandpass = bandpass
        self.notch = notch
        self.highpass = highpass
        self._fold_assignments = fold_assignments or {}
        self._compute_recording_stats = compute_recording_stats
        self._use_all_eeg_channels = use_all_eeg_channels
        self._discovered_channels: Dict[str, List[str]] = {}
        self._subject_records = list(subject_records)
        self._records_by_id: Dict[str, SubjectRecord] = {}
        self._scores_end: Dict[str, float] = {}
        self._arousals: Dict[str, Optional[np.ndarray]] = {}
        self._respiratory: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
        self._limb: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
        self._caisr: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
        self._index: List[Tuple[str, int, int]] = []
        self._index_built = False
        self._signal_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._stats_cache: Dict[str, Optional[Dict[str, np.ndarray]]] = {}

    def _ensure_index_built(self) -> None:
        if self._index_built:
            return
        for record in self._subject_records:
            # Determine fallback epoch count from CAISR or PSG
            fallback_n = None
            if record.annotation_path is None:
                if record.caisr_annotation_path is not None:
                    try:
                        cedf = edfio.read_edf(record.caisr_annotation_path)
                        for s in cedf.signals:
                            if s.label.strip() == "stage_caisr":
                                fallback_n = len(s.data)
                                break
                    except Exception:
                        pass
                if fallback_n is None:
                    fallback_n = _get_psg_duration_epochs(record.psg_path)

            stages, _, scores_end = read_physionet2026_stages(
                record.annotation_path,
                epoch_seconds=self.epoch_seconds,
                fallback_n_epochs=fallback_n,
            )
            n_epochs = len(stages)

            arousals = read_physionet2026_arousals(
                record.annotation_path, self.epoch_seconds, n_epochs,
            )
            respiratory = read_physionet2026_respiratory(
                record.annotation_path, self.epoch_seconds, n_epochs,
            )
            limb = read_physionet2026_limb_movements(
                record.annotation_path, self.epoch_seconds, n_epochs,
            )
            caisr = read_physionet2026_caisr(
                record.caisr_annotation_path, self.epoch_seconds, n_epochs,
            )

            self._records_by_id[record.subject_id] = record
            self._scores_end[record.subject_id] = scores_end
            self._arousals[record.subject_id] = arousals
            self._respiratory[record.subject_id] = respiratory
            self._limb[record.subject_id] = limb
            self._caisr[record.subject_id] = caisr

            for ep_idx, stage in enumerate(stages):
                self._index.append((record.subject_id, ep_idx, int(stage)))
        self._index_built = True

    def _load_subject_signals(self, subject_id: str) -> Tuple[np.ndarray, float]:
        if subject_id in self._signal_cache:
            return self._signal_cache[subject_id]

        record = self._records_by_id[subject_id]
        scores_end = self._scores_end[subject_id]

        if self._use_all_eeg_channels:
            from neuroatlas.extensions.datasets.dataio._eeg_channel_discovery import (
                read_all_eeg_channels,
            )
            signals, names, sfreq = read_all_eeg_channels(
                record.psg_path, duration_sec=scores_end,
            )
            if names:
                self._discovered_channels[subject_id] = names
        else:
            signals, _, sfreq = read_physionet2026_channels(
                record.psg_path, self._channel_list, duration_sec=scores_end,
            )

        if signals.size == 0:
            n_samples = int(round(scores_end * 200.0))
            signals = np.zeros((len(self._channel_list), n_samples), dtype=np.float32)
            sfreq = 200.0

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

        sample: Dict[str, object] = {
            "eeg": torch.tensor(epoch, dtype=torch.float32),
            "sleep_stage": sleep_stage,
            "subject_id": subject_id,
            "site_id": record.site_id,
            "epoch_idx": ep_idx,
            "channels": self._discovered_channels.get(subject_id, self._montage_names),
            "sampling_rate": sfreq,
            "unit": "uV",
            "epoch_seconds": self.epoch_seconds,
            "location": SITE_LOCATION.get(record.site_id, ""),
            "has_human_annotations": record.has_human_annotations,
            "age": record.age,
            "sex": record.sex,
            "cognitive_impairment": record.cognitive_impairment,
            "time_to_event": record.time_to_event,
            "time_to_last_visit": record.time_to_last_visit,
            "bmi": record.bmi,
        }

        rec_stats = self._stats_cache.get(subject_id)
        if rec_stats is not None:
            for k, v in rec_stats.items():
                sample[k] = v

        # Human annotation fractions
        arousals = self._arousals.get(subject_id)
        if arousals is not None and ep_idx < len(arousals):
            sample["arousal_fraction"] = float(arousals[ep_idx])
        else:
            sample["arousal_fraction"] = None

        respiratory = self._respiratory.get(subject_id)
        if respiratory is not None:
            for name in RESP_EVENT_NAMES:
                arr = respiratory.get(name)
                sample[f"{name}_fraction"] = float(arr[ep_idx]) if arr is not None and ep_idx < len(arr) else None
        else:
            for name in RESP_EVENT_NAMES:
                sample[f"{name}_fraction"] = None

        limb = self._limb.get(subject_id)
        if limb is not None:
            for name in LIMB_EVENT_NAMES:
                arr = limb.get(name)
                sample[f"{name}_fraction"] = float(arr[ep_idx]) if arr is not None and ep_idx < len(arr) else None
        else:
            for name in LIMB_EVENT_NAMES:
                sample[f"{name}_fraction"] = None

        # CAISR metadata
        caisr = self._caisr.get(subject_id)
        if caisr is not None:
            for field in CAISR_FIELDS:
                arr = caisr.get(field)
                if arr is not None and ep_idx < len(arr):
                    sample[field] = float(arr[ep_idx]) if arr.dtype != np.int16 else int(arr[ep_idx])
                else:
                    sample[field] = None
        else:
            for field in CAISR_FIELDS:
                sample[field] = None

        if subject_id in self._fold_assignments:
            sample["fold_assignments"] = self._fold_assignments[subject_id]
        return sample

    @property
    def subjects(self) -> List[str]:
        return [r.subject_id for r in self._subject_records]
