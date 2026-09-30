"""Online CFS (Cleveland Family Study, NSRR visit5) raw-EDF adapter.

Reads raw EDFs + NSRR XML annotations directly. Derives bipolar EEG
montages on-the-fly from the referential channels (C3, C4, M1, M2),
resamples 128 → 256 Hz, applies 0.5 Hz Butterworth-SOS highpass + 60 Hz
notch, converts native mV signals to µV based on EDF physical_dimension,
and publishes recording-level stats (mean/std/q95 + bipolar variant) per
MODEL_CONTRACTS.md §2.

Dataset location::

    ${EEG_DATA_ROOT}/raw-sleep/cfs/
    ├── polysomnography/edfs/cfs-visit5-<nsrrid>.edf
    ├── polysomnography/annotations-events-nsrr/cfs-visit5-<nsrrid>-nsrr.xml
    └── datasets/cfs-visit5-harmonized-dataset-0.7.0.csv  (nsrr_age, nsrr_sex)

Channel naming (CFS visit5):
    C3, C4, M1, M2  (mastoid references; no A1/A2 alternative)
    fs = 128 Hz native, resampled to 256 Hz target.
    physical_dimension = 'mV' (verified across 5 sample EDFs).

Bipolar derivations computed by this adapter:
    C3-M2 = C3 - M2
    C4-M1 = C4 - M1

CFS spans ages 6.85 – 90+ (children + parents from a family-based cohort) —
wider than typical brain-age cohorts. Splits stratify by an extended
age-bin set; users wanting an adult-only subset can filter via task-side
``train_filter`` à la PhysioNet2026.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .base import BenchmarkDataModule


# ---------------------------------------------------------------------------
# Stage mapping: NSRR XML EventConcept → integer label
# (mirrors mros_raw_brain_age._STAGE_MAP — same NSRR scoring vocabulary)
# ---------------------------------------------------------------------------

_STAGE_MAP = {
    "Wake|0": 0,
    "Stage 1 sleep|1": 1,
    "Stage 2 sleep|2": 2,
    "Stage 3 sleep|3": 3,
    "Stage 4 sleep|4": 3,  # R&K N4 → AASM N3
    "REM sleep|5": 4,
}

_STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

# ---------------------------------------------------------------------------
# Bipolar derivation recipes
# ---------------------------------------------------------------------------

_BIPOLAR_RECIPES: Dict[str, List[Tuple[str, str]]] = {
    "C3-M2": [("C3", "M2")],
    "C4-M1": [("C4", "M1")],
}

# CFS visit5 native EEG fs is 128 Hz; we resample to 256 Hz so the cache
# matches MrOS / SleepEDF conventions used by the FM wrappers.
_NATIVE_SFREQ = 128.0
_SFREQ = 256.0
_EPOCH_SECONDS = 30.0
_EPOCH_SAMPLES = int(_SFREQ * _EPOCH_SECONDS)  # 7680

# EDF physical_dimension → µV multiplicative factor.
_UNIT_TO_UV: Dict[str, float] = {
    "uV": 1.0,
    "µV": 1.0,
    "μV": 1.0,
    "mV": 1e3,
    "V": 1e6,
}


# ---------------------------------------------------------------------------
# Filters (per checklist §0/§2 — Butterworth SOS HP + iirnotch SOS)
# ---------------------------------------------------------------------------

def _highpass_sos(x: np.ndarray, sfreq: float, cutoff_hz: float, order: int = 5) -> np.ndarray:
    """Order-5 Butterworth SOS highpass, zero-phase via sosfiltfilt."""
    from scipy.signal import butter, sosfiltfilt

    nyq = 0.5 * sfreq
    sos = butter(order, cutoff_hz / nyq, btype="high", output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32, copy=False)


def _notch_sos(x: np.ndarray, sfreq: float, freq_hz: float, quality: float = 30.0) -> np.ndarray:
    """Single-frequency IIR notch via iirnotch → SOS, zero-phase via sosfiltfilt."""
    from scipy.signal import iirnotch, sosfiltfilt

    b, a = iirnotch(freq_hz, quality, sfreq)
    sos = np.array([[b[0], b[1], b[2], a[0], a[1], a[2]]])
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32, copy=False)


def _resample_native_to_target(x: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    """Polyphase resample along last axis (anti-aliased)."""
    if int(src_fs) == int(dst_fs):
        return x.astype(np.float32, copy=False)
    from math import gcd
    from scipy.signal import resample_poly

    s = int(src_fs)
    d = int(dst_fs)
    g = gcd(s, d)
    up = d // g
    down = s // g
    return resample_poly(x, up, down, axis=-1).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# EDF reader — pyedflib so we can inspect physical_dimension per channel
# ---------------------------------------------------------------------------

def _read_edf_referential(
    edf_path: Path,
    referential_labels: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Read needed referential channels from the EDF.

    Returns:
      signals: {label: float32 array @ native sample rate (after fs-homogeneity check)}
      units: {label: physical_dimension as reported by EDF header}

    Raises:
      ValueError if requested labels mismatch sample rates (CFS EEG is 128 Hz
      across C3/C4/M1/M2 — fail loudly if a recording violates that).
    """
    import pyedflib

    r = pyedflib.EdfReader(str(edf_path))
    try:
        labels = r.getSignalLabels()
        idx_by_label = {lbl: i for i, lbl in enumerate(labels)}
        signals: Dict[str, np.ndarray] = {}
        units: Dict[str, str] = {}
        observed_fs: Optional[float] = None
        for lbl in referential_labels:
            i = idx_by_label.get(lbl)
            if i is None:
                continue  # missing — caller skips the recording
            fs = float(r.getSampleFrequency(i))
            if observed_fs is None:
                observed_fs = fs
            elif abs(fs - observed_fs) > 1e-6:
                raise ValueError(
                    f"{edf_path.name}: referential channel {lbl!r} fs={fs} "
                    f"differs from observed fs={observed_fs}; CFS EEG is "
                    f"expected to be 128 Hz across C3/C4/M1/M2."
                )
            signals[lbl] = r.readSignal(i).astype(np.float32, copy=False)
            units[lbl] = str(r.getPhysicalDimension(i))
        return signals, units
    finally:
        r._close()


def _to_uv_inplace(signals: Dict[str, np.ndarray], units: Dict[str, str]) -> str:
    """Scale every signal to µV based on its EDF-declared unit.

    Returns the (single) original unit string for logging — raises if the
    bipolar source channels carry mixed units (would corrupt subtraction).
    """
    seen: set[str] = set()
    for lbl, sig in signals.items():
        u = units.get(lbl, "").strip()
        seen.add(u)
        factor = _UNIT_TO_UV.get(u)
        if factor is None:
            raise ValueError(
                f"Unknown EDF physical_dimension {u!r} for channel {lbl!r}; "
                f"expected one of {sorted(_UNIT_TO_UV)}."
            )
        if factor != 1.0:
            signals[lbl] = (sig * factor).astype(np.float32, copy=False)
    if len(seen) > 1:
        raise ValueError(
            f"Mixed units across bipolar source channels: {seen}. "
            f"Subtraction in mismatched units would silently corrupt the bipolar."
        )
    return next(iter(seen)) if seen else ""


# ---------------------------------------------------------------------------
# Recording discovery
# ---------------------------------------------------------------------------

def _discover_recordings(data_root: Path) -> List[Dict[str, Any]]:
    """Scan polysomnography/edfs/ for paired EDF + NSRR XML files.

    Returns list of dicts with: edf_path, xml_path, subject_id (int), visit (5).
    """
    records: List[Dict[str, Any]] = []
    edf_dir = data_root / "polysomnography" / "edfs"
    xml_dir = data_root / "polysomnography" / "annotations-events-nsrr"

    if not edf_dir.is_dir() or not xml_dir.is_dir():
        return records

    for edf_path in sorted(edf_dir.glob("cfs-visit*-*.edf")):
        m = re.match(r"cfs-visit(\d+)-(\d+)\.edf$", edf_path.name)
        if not m:
            continue
        visit = int(m.group(1))
        subject_id = int(m.group(2))
        xml_path = xml_dir / f"{edf_path.stem}-nsrr.xml"
        if not xml_path.exists():
            continue
        records.append({
            "edf_path": edf_path,
            "xml_path": xml_path,
            "subject_id": subject_id,
            "visit": visit,
        })
    return records


def _load_metadata(data_root: Path) -> Dict[int, Dict[str, Any]]:
    """Load CFS harmonized demographics → {subject_id: {age, sex, age_gt89}}."""
    meta: Dict[int, Dict[str, Any]] = {}
    datasets_dir = data_root / "datasets"
    csvs = sorted(datasets_dir.glob("cfs-visit*-harmonized-*.csv"))
    if not csvs:
        # Fall back to the unharmonized CSV.
        csvs = sorted(datasets_dir.glob("cfs-visit*-dataset-*.csv"))
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            try:
                sid = int(row["nsrrid"])
            except (KeyError, ValueError):
                continue
            age = float(row.get("nsrr_age")) if not pd.isna(row.get("nsrr_age", np.nan)) else None
            if age is None:
                continue
            sex = str(row.get("nsrr_sex", "unknown"))
            age_gt89 = str(row.get("nsrr_age_gt89", "no")).strip().lower() == "yes"
            meta[sid] = {"age": age, "sex": sex, "age_gt89": age_gt89}
    return meta


# ---------------------------------------------------------------------------
# Bipolar derivation
# ---------------------------------------------------------------------------

def _derive_bipolar(
    edf_signals: Dict[str, np.ndarray],
    channels: Sequence[str],
) -> Optional[np.ndarray]:
    derived: List[np.ndarray] = []
    for ch_name in channels:
        recipes = _BIPOLAR_RECIPES.get(ch_name)
        if recipes is None:
            raise ValueError(f"Unknown bipolar channel {ch_name!r}; known: {list(_BIPOLAR_RECIPES)}")
        found = False
        for pos_label, neg_label in recipes:
            if pos_label in edf_signals and neg_label in edf_signals:
                derived.append(edf_signals[pos_label] - edf_signals[neg_label])
                found = True
                break
        if not found:
            return None
    return np.stack(derived, axis=0)


# ---------------------------------------------------------------------------
# XML annotation parsing
# ---------------------------------------------------------------------------

def _parse_nsrr_stages(xml_path: Path, sfreq: float) -> List[Tuple[int, int]]:
    """Parse NSRR XML → list of (onset_sample @ sfreq, stage_int) for each 30-s epoch."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    epoch_samples = int(sfreq * 30)
    epochs: List[Tuple[int, int]] = []
    for event in root.iter("ScoredEvent"):
        etype = event.find("EventType")
        concept = event.find("EventConcept")
        if etype is None or concept is None:
            continue
        if "Stages" not in (etype.text or ""):
            continue
        stage = _STAGE_MAP.get(concept.text)
        if stage is None:
            continue
        onset_s = float(event.find("Start").text)
        duration_s = float(event.find("Duration").text)
        onset_sample = int(onset_s * sfreq)
        duration_samples = int(duration_s * sfreq)
        for offset in range(0, duration_samples, epoch_samples):
            epochs.append((onset_sample + offset, stage))
    return epochs


def _count_recording_epochs(xml_path: Path, crop_wake_mins: Optional[int]) -> List[Tuple[int, int]]:
    """Fast epoch counting — XML only, no EDF I/O. Onsets in target-fs samples."""
    epoch_list = _parse_nsrr_stages(xml_path, _SFREQ)
    if not epoch_list:
        return []
    if crop_wake_mins is not None:
        sleep_indices = [i for i, (_, s) in enumerate(epoch_list) if s != 0]
        if sleep_indices:
            margin_epochs = (crop_wake_mins * 60) // 30
            start_idx = max(0, sleep_indices[0] - margin_epochs)
            end_idx = min(len(epoch_list), sleep_indices[-1] + margin_epochs + 1)
            epoch_list = epoch_list[start_idx:end_idx]
    return epoch_list


# ---------------------------------------------------------------------------
# Per-recording loader (used by both DataLoader Dataset and shard extractor)
# ---------------------------------------------------------------------------

def _load_recording_signals(
    edf_path: Path,
    channels: Sequence[str],
    *,
    bandpass_hz: Optional[Tuple[Optional[float], Optional[float]]] = None,
    notch_hz: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, str]]:
    """Read EDF, convert to µV, resample to target fs, filter, derive bipolar.

    Returns (bipolar [n_channels, n_samples_at_target_fs] in µV, original_unit_str)
    or None if any required referential channel is missing.
    """
    referential_needed: List[str] = []
    for ch in channels:
        recipes = _BIPOLAR_RECIPES.get(ch)
        if recipes is None:
            raise ValueError(f"Unknown bipolar channel {ch!r}")
        for pos, neg in recipes:
            if pos not in referential_needed:
                referential_needed.append(pos)
            if neg not in referential_needed:
                referential_needed.append(neg)

    signals, units = _read_edf_referential(edf_path, referential_needed)
    if not signals:
        return None
    original_unit = _to_uv_inplace(signals, units)

    # Resample referential channels to target fs BEFORE bipolar derivation
    # (resampling is linear, so resample-then-subtract == subtract-then-resample
    # for ideal polyphase; we do it here so filters run at the canonical fs).
    if int(_NATIVE_SFREQ) != int(_SFREQ):
        for lbl, sig in signals.items():
            signals[lbl] = _resample_native_to_target(sig, _NATIVE_SFREQ, _SFREQ)

    # Filters per checklist §0: Butterworth-SOS HP (order=5, cutoff=0.5 Hz) +
    # iirnotch (Q=30, freq=60 Hz US powerline). Applied to each referential
    # channel before bipolar derivation — linear filters, so equivalent to
    # filtering after derivation; we do it here so per-recording stats below
    # are computed on the filtered bipolar that the FM actually sees.
    if bandpass_hz is not None:
        l, h = bandpass_hz
        if l is not None and l > 0:
            for lbl, sig in signals.items():
                signals[lbl] = _highpass_sos(sig, _SFREQ, float(l))
        if h is not None and h > 0:
            # not used for current contract; keep stub for future extensions
            raise NotImplementedError(
                "Lowpass via bandpass_hz[1] not implemented in CFS adapter; "
                "use notch_hz only for now (matches MrOS contract)."
            )
    if notch_hz is not None and notch_hz > 0:
        for lbl, sig in signals.items():
            signals[lbl] = _notch_sos(sig, _SFREQ, float(notch_hz))

    bipolar = _derive_bipolar(signals, channels)
    if bipolar is None:
        return None
    return bipolar, original_unit


def _compute_recording_stats(bipolar: np.ndarray) -> Dict[str, object]:
    """Per-channel + bipolar-pairwise q95 stats, computed on filtered bipolar.

    Mirrors the helper in dataio/wsc.py and sleepedf_raw_bp_test._SleepEDFRawBPDataset.
    """
    sigs = np.ascontiguousarray(bipolar)
    n_ch = sigs.shape[0]
    out: Dict[str, object] = {
        "recording_mean": sigs.mean(axis=1).astype(np.float32),
        "recording_std": sigs.std(axis=1).astype(np.float32),
        "recording_q95": np.quantile(np.abs(sigs), 0.95, axis=1).astype(np.float32),
    }
    if n_ch > 1:
        bq: Dict[str, float] = {}
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                bq[f"{i},{j}"] = float(np.quantile(np.abs(sigs[i] - sigs[j]), 0.95))
        out["recording_q95_bipolar"] = bq
    return out


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

RecordInfo = Tuple[Path, Path, int, int, float, str]  # edf, xml, subject, visit, age, sex


class _CFSRawDataset(Dataset):
    """Lazily loads CFS recordings and serves individual 30-s epochs.

    XML annotations are parsed at init (fast). EDF data is loaded on demand
    with a small LRU cache; per-recording stats are cached separately so
    they survive eviction.
    """

    def __init__(
        self,
        records: List[RecordInfo],
        channels: Sequence[str],
        *,
        crop_wake_mins: Optional[int] = None,
        bandpass_hz: Optional[Tuple[Optional[float], Optional[float]]] = None,
        notch_hz: Optional[float] = None,
        compute_recording_stats: bool = False,
    ) -> None:
        self._channels = list(channels)
        self._crop_wake_mins = crop_wake_mins
        self._bandpass_hz = bandpass_hz
        self._notch_hz = notch_hz
        self._compute_recording_stats = bool(compute_recording_stats)
        self._records = list(records)
        self._rec_cache: Dict[int, Optional[np.ndarray]] = {}
        self._cache_maxsize = 4
        self._stats_cache: Dict[int, Dict[str, object]] = {}
        self._index: List[Tuple[int, int, int, int, int, float, str]] = []
        for rec_idx, (_, xml_path, subject_id, visit, age, sex) in enumerate(records):
            for onset_sample, stage in _count_recording_epochs(xml_path, crop_wake_mins):
                self._index.append((rec_idx, onset_sample, stage, subject_id, visit, age, sex))

    def __len__(self) -> int:
        return len(self._index)

    def _load_recording(self, rec_idx: int) -> Optional[np.ndarray]:
        if rec_idx in self._rec_cache:
            return self._rec_cache[rec_idx]
        edf_path = self._records[rec_idx][0]
        result = _load_recording_signals(
            edf_path, self._channels,
            bandpass_hz=self._bandpass_hz, notch_hz=self._notch_hz,
        )
        bipolar = result[0] if result is not None else None
        if (
            self._compute_recording_stats
            and bipolar is not None
            and bipolar.size > 0
            and rec_idx not in self._stats_cache
        ):
            self._stats_cache[rec_idx] = _compute_recording_stats(bipolar)
        if len(self._rec_cache) >= self._cache_maxsize:
            self._rec_cache.pop(next(iter(self._rec_cache)))
        self._rec_cache[rec_idx] = bipolar
        return bipolar

    def __getitem__(self, idx: int):
        rec_idx, onset_sample, stage, subject_id, visit, age, sex = self._index[idx]
        bipolar = self._load_recording(rec_idx)
        if bipolar is None:
            epoch = np.zeros((len(self._channels), _EPOCH_SAMPLES), dtype=np.float32)
        else:
            ep_end = onset_sample + _EPOCH_SAMPLES
            if ep_end <= bipolar.shape[1]:
                epoch = bipolar[:, onset_sample:ep_end].astype(np.float32)
            else:
                epoch = np.zeros((len(self._channels), _EPOCH_SAMPLES), dtype=np.float32)
        sample: Dict[str, object] = {
            "eeg": torch.from_numpy(epoch.copy()),
            "age": age,
            "subject_id": str(subject_id),
            "visit": int(visit),
            "sleep_stage": int(stage),
            "sex": sex,
            "epoch_index": onset_sample // _EPOCH_SAMPLES,
        }
        rec_stats = self._stats_cache.get(rec_idx)
        if rec_stats is not None:
            for k, v in rec_stats.items():
                sample[k] = v
        return sample


# ---------------------------------------------------------------------------
# Collate + LoaderAdapter
# ---------------------------------------------------------------------------

def _collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    eeg = torch.stack([item["eeg"] for item in batch])
    age = torch.tensor([item["age"] for item in batch], dtype=torch.float32)
    out: Dict[str, object] = {
        "eeg": eeg,
        "age": age,
        "subject_ids": [item["subject_id"] for item in batch],
        "visits": [item["visit"] for item in batch],
        "sleep_stages": [item["sleep_stage"] for item in batch],
        "sexes": [item["sex"] for item in batch],
        "epoch_indices": [item["epoch_index"] for item in batch],
    }
    for key in ("recording_mean", "recording_std", "recording_q95", "recording_q95_bipolar"):
        if all(key in item for item in batch):
            out[key] = [item[key] for item in batch]
    return out


class _LoaderAdapter:
    """Emit standard benchmark batches with MODEL_CONTRACTS §0/§3 meta."""

    def __init__(
        self,
        loader: DataLoader,
        *,
        sampling_rate: float,
        unit: str,
        channels: List[str],
        dataset_name: str,
    ) -> None:
        self._loader = loader
        self._sampling_rate = float(sampling_rate)
        self._unit = unit
        self._channels = list(channels)
        self._dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        _STATS_KEYS = ("recording_mean", "recording_std", "recording_q95", "recording_q95_bipolar")
        for batch in self._loader:
            meta: List[Dict[str, object]] = []
            for i in range(len(batch["subject_ids"])):
                m: Dict[str, object] = {
                    "dataset": self._dataset_name,
                    "subject_id": batch["subject_ids"][i],
                    "age": float(batch["age"][i]),
                    "visit": int(batch["visits"][i]),
                    "sleep_stage": int(batch["sleep_stages"][i]),
                    "sex": batch["sexes"][i],
                    "epoch_index": batch["epoch_indices"][i],
                    "sampling_rate": self._sampling_rate,
                    "unit": self._unit,
                    "channels": list(self._channels),
                }
                for key in _STATS_KEYS:
                    if key in batch:
                        m[key] = batch[key][i]
                meta.append(m)
            yield {
                "signals": {"eeg": batch["eeg"]},
                "label": batch["age"],
                "meta": meta,
            }


# ---------------------------------------------------------------------------
# Subject-level splitting (stratified by extended age bins)
# ---------------------------------------------------------------------------

def _subject_splits(
    subjects: List[int],
    ages: Dict[int, float],
    fold: int,
    n_folds: int,
    seed: int = 42,
    age_bins: Sequence[float] = (0, 18, 35, 50, 65, 200),
) -> Tuple[List[int], List[int], List[int]]:
    """Stratified k-fold split with an extra child bin (CFS includes <18)."""
    from neuroatlas.benchmarking_helpers.registry.splits import (
        make_subject_kfold,
    )

    subject_ages = np.array([ages[s] for s in subjects])
    bins = np.array(list(age_bins), dtype=float)
    age_bin_ids = np.digitize(subject_ages, bins) - 1

    # Shared rule: benchmarking_helpers/splits. Age bins are a harsher
    # stratification than a binary label -- one sparse decade can hold
    # fewer subjects than folds -- so passing n_folds straight to
    # StratifiedKFold raised on exactly the cohorts it mattered for.
    skf, _stratified, n_folds = make_subject_kfold(
        n_folds, age_bin_ids, seed=seed)
    splits = list(skf.split(subjects, age_bin_ids))
    train_val_idx, test_idx = splits[fold % n_folds]
    test_subjects = [subjects[i] for i in test_idx]

    train_val_subjects = [subjects[i] for i in train_val_idx]
    train_val_bins = age_bin_ids[train_val_idx]
    inner_skf, _inner_strat, inner_folds = make_subject_kfold(
        n_folds, train_val_bins, seed=seed)
    val_fold = (fold + 1) % inner_folds
    inner_splits = list(inner_skf.split(train_val_subjects, train_val_bins))
    inner_train_idx, inner_val_idx = inner_splits[val_fold % len(inner_splits)]
    train_subjects = [train_val_subjects[i] for i in inner_train_idx]
    val_subjects = [train_val_subjects[i] for i in inner_val_idx]
    return train_subjects, val_subjects, test_subjects


# ---------------------------------------------------------------------------
# BenchmarkDataModule
# ---------------------------------------------------------------------------

class CFSRawDataModule(BenchmarkDataModule):
    """BenchmarkDataModule reading raw CFS visit5 EDF files on the fly.

    Args:
        data_root: CFS dataset root (containing polysomnography/, datasets/).
        channels: Bipolar derivation names, e.g., ["C3-M2", "C4-M1"].
        bandpass_hz: (l_freq, h_freq); only highpass implemented (h_freq must be 0/None).
        notch_hz: Notch frequency (60 for US powerline; CFS is from Cleveland).
        fold/n_folds: subject-level stratified-by-age k-fold split.
        crop_wake_mins: optional wake cropping around the sleep period.
    """

    def __init__(
        self,
        data_root: str,
        channels: Optional[List[str]] = None,
        batch_size: int = 256,
        num_workers: int = 0,
        fold: int = 0,
        n_folds: int = 5,
        crop_wake_mins: Optional[int] = None,
        bandpass_hz: Optional[Sequence[Optional[float]]] = (0.5, 0),
        notch_hz: Optional[float] = 60.0,
        compute_recording_stats: bool = False,
        age_bins: Optional[Sequence[float]] = None,
        name: str = "cfs_raw_2ch",
        **kwargs: object,
    ) -> None:
        if channels is None:
            channels = ["C3-M2", "C4-M1"]
        if age_bins is None:
            age_bins = (0, 18, 35, 50, 65, 200)
        bp: Optional[Tuple[Optional[float], Optional[float]]] = None
        if bandpass_hz is not None:
            l = bandpass_hz[0] if bandpass_hz[0] and bandpass_hz[0] > 0 else None
            h = bandpass_hz[1] if bandpass_hz[1] and bandpass_hz[1] > 0 else None
            bp = (l, h)
        nh = float(notch_hz) if notch_hz and notch_hz > 0 else None

        self._compute_recording_stats = bool(compute_recording_stats)
        self._bandpass_hz = bp
        self._notch_hz = nh

        self._raw_sampling_rate = float(_SFREQ)
        # Data is converted to µV at extraction time (see _to_uv_inplace),
        # so meta["unit"] is the canonical "uV" — wrappers' unit_to_uv is a no-op.
        self._raw_unit = "uV"
        self._raw_channels_meta = list(channels)
        self._dataset_name = name

        meta: Dict[str, object] = {
            "fold": fold,
            "n_folds": n_folds,
            "channels": list(channels),
            "sampling_rate": _SFREQ,
            "epoch_seconds": _EPOCH_SECONDS,
            "signal_kind": "raw",
            "crop_wake_mins": crop_wake_mins,
            "bandpass_hz": [bp[0], bp[1]] if bp is not None else None,
            "notch_hz": nh,
        }
        super().__init__(name=name, metadata=meta)

        root = Path(data_root)
        self._data_root = root
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._channels = list(channels)
        self._crop_wake_mins = crop_wake_mins
        self._fold = fold
        self._n_folds = n_folds
        self._age_bins = tuple(age_bins)

        all_records = _discover_recordings(root)
        subject_meta = _load_metadata(root)

        subject_set = sorted({r["subject_id"] for r in all_records if r["subject_id"] in subject_meta})
        # Say what actually went wrong, here. An empty cohort used to travel
        # all the way to the splitter and surface as "cross-validation needs at
        # least 2 subjects, got 0" -- a message about folds, for a problem
        # about a path.
        if not subject_set:
            raise FileNotFoundError(
                f"No CFS recordings found under data_root={root}. "
                + (f"{len(all_records)} signal file(s) were discovered but none "
                   "matched a subject in the metadata table; check that the "
                   "dataset CSV is alongside them."
                   if all_records else
                   "The directory is missing, empty, or not the CFS root — it "
                   "should contain the polysomnography EDFs and the CFS "
                   "dataset CSV.")
            )
        ages = {s: subject_meta[s]["age"] for s in subject_set}

        train_subs, val_subs, test_subs = _subject_splits(
            subject_set, ages, fold, n_folds, age_bins=self._age_bins,
        )
        self._subject_to_split: Dict[int, str] = {}
        for s in train_subs:
            self._subject_to_split[s] = "train"
        for s in val_subs:
            self._subject_to_split[s] = "val"
        for s in test_subs:
            self._subject_to_split[s] = "test"

        def _records_for(sub_set):
            result = []
            for r in all_records:
                sid = r["subject_id"]
                if sid not in sub_set or sid not in subject_meta:
                    continue
                sm = subject_meta[sid]
                result.append((r["edf_path"], r["xml_path"], sid, r["visit"], sm["age"], sm["sex"]))
            return result

        self._all_records_info = _records_for(set(subject_set))

        ds_kwargs = dict(
            crop_wake_mins=crop_wake_mins,
            bandpass_hz=bp, notch_hz=nh,
            compute_recording_stats=self._compute_recording_stats,
        )
        print(f"[cfs] Indexing train split ({len(train_subs)} subjects)...", flush=True)
        self._train_ds = _CFSRawDataset(_records_for(set(train_subs)), channels, **ds_kwargs)
        print(f"[cfs] Indexing val split ({len(val_subs)} subjects)...", flush=True)
        self._val_ds = _CFSRawDataset(_records_for(set(val_subs)), channels, **ds_kwargs)
        print(f"[cfs] Indexing test split ({len(test_subs)} subjects)...", flush=True)
        self._test_ds = _CFSRawDataset(_records_for(set(test_subs)), channels, **ds_kwargs)
        print(
            f"[cfs] Indexed: train={len(self._train_ds)} val={len(self._val_ds)} "
            f"test={len(self._test_ds)} epochs ({len(all_records)} recordings, "
            f"{len(subject_set)} subjects). Filter: HP={bp} notch={nh}Hz fs={_SFREQ}Hz",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Global embedding cache support
    # ------------------------------------------------------------------

    def supports_global_embedding_cache(self) -> bool:
        return True

    def cache_context(self, purpose: str = "default") -> Dict[str, object]:
        bp = self.metadata.get("bandpass_hz")
        if purpose == "global_embeddings":
            return {
                "channels": self.metadata.get("channels"),
                "sampling_rate": self.metadata.get("sampling_rate"),
                "crop_wake_mins": self.metadata.get("crop_wake_mins"),
                "data_root": str(self._data_root),
                "bandpass_hz": bp,
                "notch_hz": self.metadata.get("notch_hz"),
            }
        return {
            "fold": self.metadata.get("fold"),
            "channels": self.metadata.get("channels"),
            "sampling_rate": self.metadata.get("sampling_rate"),
            "crop_wake_mins": self.metadata.get("crop_wake_mins"),
            "data_root": str(self._data_root),
            "bandpass_hz": bp,
            "notch_hz": self.metadata.get("notch_hz"),
        }

    def full_embedding_dataloader(self) -> _LoaderAdapter:
        full_ds = _CFSRawDataset(
            self._all_records_info, self._channels,
            crop_wake_mins=self._crop_wake_mins,
            bandpass_hz=self._bandpass_hz, notch_hz=self._notch_hz,
            compute_recording_stats=self._compute_recording_stats,
        )
        loader = DataLoader(
            full_ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=1,
            prefetch_factor=2,
            collate_fn=_collate_fn,
            pin_memory=False,
        )
        return _LoaderAdapter(
            loader,
            sampling_rate=self._raw_sampling_rate,
            unit=self._raw_unit,
            channels=self._raw_channels_meta,
            dataset_name=self._dataset_name,
        )

    def split_global_embedding_payload(self, payload) -> Dict[str, "EmbeddingPayload"]:
        from neuroatlas.benchmarking_helpers.registry.contracts import EmbeddingPayload

        features = np.asarray(payload.features)
        labels = np.asarray(payload.labels)
        metadata_list = list(payload.metadata)
        result: Dict[str, EmbeddingPayload] = {}
        for split_name in ("train", "val", "test"):
            mask = [
                self._subject_to_split.get(int(m.get("subject_id", -1)), "") == split_name
                for m in metadata_list
            ]
            mask_arr = np.array(mask)
            result[split_name] = EmbeddingPayload(
                features=features[mask_arr],
                labels=labels[mask_arr],
                metadata=[m for m, keep in zip(metadata_list, mask) if keep],
            )
        return result

    # ------------------------------------------------------------------

    def _make_loader(self, dataset: _CFSRawDataset, shuffle: bool) -> _LoaderAdapter:
        workers = min(self._num_workers, 1) if not shuffle else 0
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=workers,
            prefetch_factor=2 if workers > 0 else None,
            collate_fn=_collate_fn,
            pin_memory=False,
        )
        return _LoaderAdapter(
            loader,
            sampling_rate=self._raw_sampling_rate,
            unit=self._raw_unit,
            channels=self._raw_channels_meta,
            dataset_name=self._dataset_name,
        )

    def train_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._test_ds, shuffle=False)
