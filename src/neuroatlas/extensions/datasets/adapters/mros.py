"""Online MrOS brain age adapter — reads raw EDF files + NSRR XML annotations directly.

Uses ``edfio`` for fast EDF I/O and derives bipolar EEG montages on-the-fly from
the referential channels (C3, C4, A1/M1, A2/M2).

Dataset location::

    ${EEG_DATA_ROOT}/raw-sleep/mros/
    ├── polysomnography/edfs/{visit1,visit2}/mros-visit{1,2}-aa{id}.edf
    ├── polysomnography/annotations-events-nsrr/{visit1,visit2}/mros-visit{1,2}-aa{id}-nsrr.xml
    └── datasets/mros-visit{1,2}-harmonized-0.6.0.csv  (age, sex)

Channel naming differs between visits:
    Visit 1: C3, C4, A1, A2        (mastoid references)
    Visit 2: C3, C4, M1, M2        (same electrodes, different labels)

Bipolar derivations computed by this adapter:
    C3-A2 = C3 - A2 (visit1)  or  C3 - M2 (visit2)
    C4-A1 = C4 - A1 (visit1)  or  C4 - M1 (visit2)
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
# ---------------------------------------------------------------------------

_STAGE_MAP = {
    "Wake|0": 0,
    "Stage 1 sleep|1": 1,
    "Stage 2 sleep|2": 2,
    "Stage 3 sleep|3": 3,
    "Stage 4 sleep|4": 3,  # R&K N4 → AASM N3 (matches teamSleep_TO sleep_edf.py)
    "REM sleep|5": 4,
}

_STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

# ---------------------------------------------------------------------------
# Bipolar derivation recipes
# ---------------------------------------------------------------------------

_BIPOLAR_RECIPES: Dict[str, List[Tuple[str, str]]] = {
    "C3-A2": [("C3", "A2"), ("C3", "M2")],  # visit1 or visit2 naming
    "C4-A1": [("C4", "A1"), ("C4", "M1")],
}

_SFREQ = 256.0  # All EEG channels in MrOS are 256 Hz


def _read_edf_channels(edf_path: Path, channels: Sequence[str]) -> Dict[str, np.ndarray]:
    """Read EDF and return only the channels needed for bipolar derivation.

    Uses full bulk read (not lazy) — single sequential NFS transfer is faster
    than many small seeks on high-latency network filesystems.
    """
    import edfio

    edf = edfio.read_edf(str(edf_path), lazy_load_data=False)
    # Build lookup of available signals
    edf_signals: Dict[str, np.ndarray] = {}
    for s in edf.signals:
        edf_signals[s.label] = s.data
    return edf_signals
_EPOCH_SECONDS = 30.0
_EPOCH_SAMPLES = int(_SFREQ * _EPOCH_SECONDS)  # 7680


# ---------------------------------------------------------------------------
# Recording discovery
# ---------------------------------------------------------------------------

def _discover_recordings(data_root: Path) -> List[Dict[str, Any]]:
    """Scan polysomnography/edfs/ for paired EDF + NSRR XML files.

    Returns list of dicts with keys: edf_path, xml_path, subject_id, visit.
    """
    records = []
    edf_base = data_root / "polysomnography" / "edfs"
    xml_base = data_root / "polysomnography" / "annotations-events-nsrr"

    for visit_dir in sorted(edf_base.iterdir()):
        if not visit_dir.is_dir():
            continue
        visit_match = re.match(r"visit(\d+)", visit_dir.name)
        if not visit_match:
            continue
        visit = int(visit_match.group(1))

        for edf_path in sorted(visit_dir.glob("mros-visit*-aa*.edf")):
            # Extract subject ID: mros-visit1-aa5050.edf → 5050
            match = re.search(r"aa(\d+)", edf_path.stem)
            if not match:
                continue
            subject_id = int(match.group(1))

            # Find matching NSRR XML
            xml_name = edf_path.stem + "-nsrr.xml"
            xml_path = xml_base / visit_dir.name / xml_name
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
    """Load harmonized CSVs for all visits → {subject_id: {"age": float, "sex": str, "age_gt89": bool}}.

    Uses the per-visit age so each visit gets the correct age at that time.
    Returns one entry per subject using the visit-1 age (primary), with visit-2
    age stored separately for per-recording age lookup.
    """
    meta: Dict[int, Dict[str, Any]] = {}
    datasets_dir = data_root / "datasets"

    csvs = sorted(datasets_dir.glob("mros-visit*-harmonized-*.csv"))
    # The harmonized CSVs are the only source of age; without them every
    # recording is dropped as subject-less and the failure surfaced later as a
    # cross-validation error about folds.
    if not csvs:
        raise FileNotFoundError(
            f"MrOS demographics not found: no mros-visit*-harmonized-*.csv under "
            f"{datasets_dir}. They ship with the NSRR download (datasets/) and are "
            f"the only source of age and sex. "
            f"Check that data_root={data_root} is the MrOS root "
            f"(the polysomnography/ EDFs next to datasets/)."
        )
    for csv_path in csvs:
        visit_match = re.search(r"visit(\d+)", csv_path.name)
        visit = int(visit_match.group(1)) if visit_match else 0
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            nsrrid = str(row["nsrrid"])  # e.g., "AA5050"
            sid = int(nsrrid.replace("AA", ""))
            age = float(row["nsrr_age"])
            sex = str(row.get("nsrr_sex", "unknown"))
            age_gt89 = bool(row.get("nsrr_age_gt89", "no") == "yes") if "nsrr_age_gt89" in row else False

            if sid not in meta:
                meta[sid] = {"age": age, "sex": sex, "age_gt89": age_gt89}
            # Store per-visit age
            meta[sid][f"age_visit{visit}"] = age
    return meta


# ---------------------------------------------------------------------------
# Bipolar derivation from referential channels
# ---------------------------------------------------------------------------

def _derive_bipolar(
    edf_signals: Dict[str, np.ndarray],
    channels: Sequence[str],
) -> Optional[np.ndarray]:
    """Derive bipolar channels from referential EDF signals.

    Args:
        edf_signals: {label: data_array} for all EDF signals.
        channels: Requested bipolar channels, e.g., ["C3-A2", "C4-A1"].

    Returns:
        ndarray [n_channels, n_samples] or None if a required channel is missing.
    """
    derived = []
    for ch_name in channels:
        recipes = _BIPOLAR_RECIPES.get(ch_name)
        if recipes is None:
            raise ValueError(f"Unknown bipolar channel {ch_name!r}. Known: {list(_BIPOLAR_RECIPES.keys())}")

        found = False
        for pos_label, neg_label in recipes:
            if pos_label in edf_signals and neg_label in edf_signals:
                pos = edf_signals[pos_label].astype(np.float32)
                neg = edf_signals[neg_label].astype(np.float32)
                derived.append(pos - neg)
                found = True
                break

        if not found:
            return None  # Missing channel — caller should skip this recording

    return np.stack(derived, axis=0)  # [n_channels, n_samples]


# ---------------------------------------------------------------------------
# XML annotation parsing
# ---------------------------------------------------------------------------

def _parse_nsrr_stages(xml_path: Path, sfreq: float) -> List[Tuple[int, int]]:
    """Parse NSRR XML annotations into a list of (onset_sample, stage_int) per 30s epoch.

    Returns list of (onset_sample, stage) for each valid 30s epoch.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    epoch_samples = int(sfreq * 30)

    epochs = []
    for event in root.iter("ScoredEvent"):
        etype = event.find("EventType")
        concept = event.find("EventConcept")
        if etype is None or concept is None:
            continue
        if "Stages" not in (etype.text or ""):
            continue

        stage = _STAGE_MAP.get(concept.text)
        if stage is None:
            continue  # skip unknown stages

        onset_s = float(event.find("Start").text)
        duration_s = float(event.find("Duration").text)
        onset_sample = int(onset_s * sfreq)
        duration_samples = int(duration_s * sfreq)

        # Expand annotation into individual 30s epochs
        for offset in range(0, duration_samples, epoch_samples):
            ep_start = onset_sample + offset
            epochs.append((ep_start, stage))

    return epochs


# ---------------------------------------------------------------------------
# Epoch extraction
# ---------------------------------------------------------------------------

def _extract_epochs(
    edf_path: Path,
    xml_path: Path,
    channels: Sequence[str],
    crop_wake_mins: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read raw EDF, derive bipolar channels, extract 30s epochs aligned to XML annotations.

    Args:
        edf_path: Path to MrOS EDF file.
        xml_path: Path to NSRR XML annotation file.
        channels: Bipolar channel names to derive, e.g., ["C3-A2", "C4-A1"].
        crop_wake_mins: Optional wake cropping (None = keep all epochs).

    Returns:
        epochs: [n_epochs, n_channels, 7680] float32
        stages: [n_epochs] int
        Returns empty arrays if bipolar derivation fails (missing channels).
    """
    import edfio

    edf_signals = _read_edf_channels(edf_path, channels)

    # Derive bipolar channels
    bipolar = _derive_bipolar(edf_signals, channels)
    if bipolar is None:
        return np.empty((0, len(channels), _EPOCH_SAMPLES), dtype=np.float32), np.empty(0, dtype=np.int64)

    n_samples = bipolar.shape[1]

    # Parse stage annotations
    epoch_list = _parse_nsrr_stages(xml_path, _SFREQ)

    if not epoch_list:
        return np.empty((0, len(channels), _EPOCH_SAMPLES), dtype=np.float32), np.empty(0, dtype=np.int64)

    # Optional wake cropping
    if crop_wake_mins is not None:
        sleep_indices = [i for i, (_, s) in enumerate(epoch_list) if s != 0]
        if sleep_indices:
            margin_epochs = (crop_wake_mins * 60) // 30
            start_idx = max(0, sleep_indices[0] - margin_epochs)
            end_idx = min(len(epoch_list), sleep_indices[-1] + margin_epochs + 1)
            epoch_list = epoch_list[start_idx:end_idx]

    # Extract epoch arrays
    valid_epochs = []
    valid_stages = []
    for onset_sample, stage in epoch_list:
        ep_end = onset_sample + _EPOCH_SAMPLES
        if ep_end <= n_samples:
            valid_epochs.append(bipolar[:, onset_sample:ep_end].copy())
            valid_stages.append(stage)

    if not valid_epochs:
        return np.empty((0, len(channels), _EPOCH_SAMPLES), dtype=np.float32), np.empty(0, dtype=np.int64)

    return (
        np.stack(valid_epochs, axis=0).astype(np.float32),
        np.array(valid_stages, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

RecordInfo = Tuple[Path, Path, int, int, float, str]  # edf, xml, subject_id, visit, age, sex


def _count_recording_epochs(xml_path: Path, crop_wake_mins: Optional[int]) -> List[Tuple[int, int]]:
    """Fast epoch counting — only parses XML, no EDF reading.

    Returns list of (onset_sample, stage) for valid epochs.
    """
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


class _MrOSRawDataset(Dataset):
    """Lazily loads raw MrOS EDF recordings and serves individual 30s epochs.

    Only the XML annotations are parsed at init time (fast).
    EDF signal data is loaded on-demand with an LRU cache.
    """

    def __init__(
        self,
        records: List[RecordInfo],
        channels: Sequence[str],
        crop_wake_mins: Optional[int] = None,
        compute_recording_stats: bool = False,
    ) -> None:
        self._channels = list(channels)
        self._crop_wake_mins = crop_wake_mins
        self._compute_recording_stats = bool(compute_recording_stats)
        self._records = list(records)
        self._rec_cache: Dict[int, Optional[np.ndarray]] = {}
        self._cache_maxsize = 4  # Keep small — MrOS has ~3,900 recordings, each ~20-50MB bipolar
        # Per-recording stats cache survives _rec_cache eviction; keyed by rec_idx.
        self._stats_cache: Dict[int, Dict[str, object]] = {}
        # Build flat index by parsing only XML annotations (fast, no EDF I/O)
        # index entry: (record_idx, epoch_onset_sample, stage, subject_id, visit, age, sex)
        self._index: List[Tuple[int, int, int, int, int, float, str]] = []

        for rec_idx, (edf_path, xml_path, subject_id, visit, age, sex) in enumerate(records):
            epoch_list = _count_recording_epochs(xml_path, crop_wake_mins)
            for onset_sample, stage in epoch_list:
                self._index.append((rec_idx, onset_sample, stage, subject_id, visit, age, sex))

    def __len__(self) -> int:
        return len(self._index)

    def _load_recording(self, rec_idx: int) -> Optional[np.ndarray]:
        """Load and derive bipolar channels for one recording. Per-instance cached."""
        if rec_idx in self._rec_cache:
            return self._rec_cache[rec_idx]
        edf_path = self._records[rec_idx][0]
        edf_signals = _read_edf_channels(edf_path, self._channels)
        result = _derive_bipolar(edf_signals, self._channels)
        # Compute per-recording stats BEFORE eviction so they survive _rec_cache.
        # MrOS source rows are already bipolar derivations (e.g. C3-A2), so from
        # BIOT's perspective these are "direct" 16-slot fills — no q95_bipolar needed.
        if (
            self._compute_recording_stats
            and result is not None
            and result.size > 0
            and rec_idx not in self._stats_cache
        ):
            signals = np.ascontiguousarray(result)
            self._stats_cache[rec_idx] = {
                "recording_mean": signals.mean(axis=1),
                "recording_std": signals.std(axis=1),
                "recording_q95": np.quantile(np.abs(signals), 0.95, axis=1),
            }
        # Simple LRU: evict oldest if cache is full
        if len(self._rec_cache) >= self._cache_maxsize:
            oldest = next(iter(self._rec_cache))
            del self._rec_cache[oldest]
        self._rec_cache[rec_idx] = result
        return result

    def __getitem__(self, idx: int):
        rec_idx, onset_sample, stage, subject_id, visit, age, sex = self._index[idx]
        bipolar = self._load_recording(rec_idx)
        if bipolar is None:
            # Missing channels — return zeros (shouldn't happen if index was built correctly)
            epoch = np.zeros((len(self._channels), _EPOCH_SAMPLES), dtype=np.float32)
        else:
            ep_end = onset_sample + _EPOCH_SAMPLES
            if ep_end <= bipolar.shape[1]:
                epoch = bipolar[:, onset_sample:ep_end].astype(np.float32)
            else:
                epoch = np.zeros((len(self._channels), _EPOCH_SAMPLES), dtype=np.float32)
        eeg = torch.from_numpy(epoch.copy())
        sample: Dict[str, object] = {
            "eeg": eeg,
            "age": age,
            "subject_id": str(subject_id),
            "visit": visit,
            "sleep_stage": stage,
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

def _collate_fn(batch):
    eeg = torch.stack([item["eeg"] for item in batch])
    age = torch.tensor([item["age"] for item in batch], dtype=torch.float32)
    subject_ids = [item["subject_id"] for item in batch]
    visits = [item["visit"] for item in batch]
    sleep_stages = [item["sleep_stage"] for item in batch]
    sexes = [item["sex"] for item in batch]
    epoch_indices = [item["epoch_index"] for item in batch]
    out = {
        "eeg": eeg,
        "age": age,
        "subject_ids": subject_ids,
        "visits": visits,
        "sleep_stages": sleep_stages,
        "sexes": sexes,
        "epoch_indices": epoch_indices,
    }
    # Per-recording stats — emit only when every sample carries the key.
    for key in ("recording_mean", "recording_std", "recording_q95"):
        if all(key in item for item in batch):
            out[key] = [item[key] for item in batch]
    return out


class _LoaderAdapter:
    """Wrap a DataLoader to emit standard benchmark batches.

    Per MODEL_CONTRACTS.md §0/§3: publishes the canonical per-sample
    meta[i] schema (``sampling_rate``, ``unit``, ``channels``). No legacy ``signal_description`` carrier.
    """

    def __init__(
        self,
        loader: DataLoader,
        *,
        sampling_rate: float,
        unit: str,
        channels: List[str],
    ) -> None:
        self._loader = loader
        self._sampling_rate = float(sampling_rate)
        self._unit = unit
        self._channels = list(channels)

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        _STATS_KEYS = ("recording_mean", "recording_std", "recording_q95")
        for batch in self._loader:
            meta = []
            for i in range(len(batch["subject_ids"])):
                m: Dict[str, object] = {
                    "dataset": "mros_raw_brain_age",
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
# Subject-level splitting
# ---------------------------------------------------------------------------

def _subject_splits(
    subjects: List[int],
    ages: Dict[int, float],
    fold: int,
    n_folds: int,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Stratified k-fold split at the subject level, using age bins."""
    from neuroatlas.benchmarking_helpers.registry.splits import (
        make_subject_kfold,
    )

    subject_ages = np.array([ages[s] for s in subjects])
    # Finer bins for MrOS narrow age range (67-90)
    bins = np.array([0, 72, 76, 80, 85, 200], dtype=float)
    age_bins = np.digitize(subject_ages, bins) - 1

    # Shared rule: benchmarking_helpers/splits. Age bins are a harsher
    # stratification than a binary label -- one sparse decade can hold
    # fewer subjects than folds -- so passing n_folds straight to
    # StratifiedKFold raised on exactly the cohorts it mattered for.
    skf, _stratified, n_folds = make_subject_kfold(
        n_folds, age_bins, seed=seed)
    splits = list(skf.split(subjects, age_bins))

    train_val_idx, test_idx = splits[fold % n_folds]
    test_subjects = [subjects[i] for i in test_idx]

    train_val_subjects = [subjects[i] for i in train_val_idx]
    train_val_bins = age_bins[train_val_idx]
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

class MrOSRawBrainAgeDataModule(BenchmarkDataModule):
    """BenchmarkDataModule that reads raw MrOS EDF files on-the-fly.

    No preprocessing required — reads directly from the MrOS distribution.
    Derives bipolar EEG montages (C3-A2, C4-A1) from referential channels,
    handles visit-dependent channel naming (A1/A2 vs M1/M2), and extracts
    30s epochs aligned to NSRR XML sleep stage annotations.

    Args:
        data_root: Path to the MrOS dataset root (containing polysomnography/ and datasets/).
        channels: Bipolar derivation names to compute, e.g., ["C3-A2", "C4-A1"].
        batch_size: Batch size for data loaders.
        num_workers: DataLoader worker count.
        fold: Zero-based fold index for k-fold CV.
        n_folds: Total number of CV folds.
        crop_wake_mins: Crop wake epochs beyond this many minutes from first/last
            sleep epoch. None disables cropping (default: keep all epochs).
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
        compute_recording_stats: bool = False,
        **kwargs: object,
    ) -> None:
        if channels is None:
            channels = ["C3-A2", "C4-A1"]
        self._compute_recording_stats = bool(compute_recording_stats)

        # Signal properties used to build per-sample meta[i] — see
        # MODEL_CONTRACTS.md §0/§3. MrOS EDF reader path in this
        # adapter preserves the µV-scaled online-resampled signal;
        # 500 µV is the scalp-EEG ptp constant consumed by BENDR's
        # 20th SCALE channel.
        self._raw_sampling_rate = float(_SFREQ)
        self._raw_unit = "uV"
        self._raw_channels_meta = list(channels)

        meta: Dict[str, object] = {
            "fold": fold,
            "n_folds": n_folds,
            "channels": list(channels),
            "sampling_rate": _SFREQ,
            "epoch_seconds": _EPOCH_SECONDS,
            "signal_kind": "raw",
            "crop_wake_mins": crop_wake_mins,
        }
        super().__init__(name="mros_raw_brain_age", metadata=meta)

        root = Path(data_root)
        self._data_root = root
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._channels = list(channels)
        self._crop_wake_mins = crop_wake_mins
        self._fold = fold
        self._n_folds = n_folds

        # Discover recordings and load metadata
        all_records = _discover_recordings(root)
        subject_meta = _load_metadata(root)

        # Build subject list (unique subjects across both visits)
        subject_set = sorted({r["subject_id"] for r in all_records if r["subject_id"] in subject_meta})

        # Use visit-1 age for splitting (primary visit)
        ages = {s: subject_meta[s]["age"] for s in subject_set}

        # Split subjects
        train_subs, val_subs, test_subs = _subject_splits(subject_set, ages, fold, n_folds)
        train_set = set(train_subs)
        val_set = set(val_subs)
        test_set = set(test_subs)

        self._subject_to_split = {}
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
                # Use per-visit age if available
                visit = r["visit"]
                age = sm.get(f"age_visit{visit}", sm["age"])
                result.append((r["edf_path"], r["xml_path"], sid, visit, age, sm["sex"]))
            return result

        self._all_records_info = _records_for(set(subject_set))

        print(f"[mros] Indexing train split ({len(train_subs)} subjects)...", flush=True)
        self._train_ds = _MrOSRawDataset(
            _records_for(train_set), channels, crop_wake_mins,
            compute_recording_stats=self._compute_recording_stats,
        )
        print(f"[mros] Indexing val split ({len(val_subs)} subjects)...", flush=True)
        self._val_ds = _MrOSRawDataset(
            _records_for(val_set), channels, crop_wake_mins,
            compute_recording_stats=self._compute_recording_stats,
        )
        print(f"[mros] Indexing test split ({len(test_subs)} subjects)...", flush=True)
        self._test_ds = _MrOSRawDataset(
            _records_for(test_set), channels, crop_wake_mins,
            compute_recording_stats=self._compute_recording_stats,
        )
        print(f"[mros] Indexed: train={len(self._train_ds)} val={len(self._val_ds)} test={len(self._test_ds)} epochs "
              f"(EDF loading is lazy, {len(all_records)} recordings across {len(subject_set)} subjects)", flush=True)

    # ------------------------------------------------------------------
    # Global embedding cache support
    # ------------------------------------------------------------------

    def supports_global_embedding_cache(self) -> bool:
        return True

    def cache_context(self, purpose: str = "default") -> Dict[str, object]:
        if purpose == "global_embeddings":
            return {
                "channels": self.metadata.get("channels"),
                "sampling_rate": self.metadata.get("sampling_rate"),
                "crop_wake_mins": self.metadata.get("crop_wake_mins"),
                "data_root": str(self._data_root),
            }
        return {
            "fold": self.metadata.get("fold"),
            "channels": self.metadata.get("channels"),
            "sampling_rate": self.metadata.get("sampling_rate"),
            "crop_wake_mins": self.metadata.get("crop_wake_mins"),
            "data_root": str(self._data_root),
        }

    def full_embedding_dataloader(self) -> _LoaderAdapter:
        full_ds = _MrOSRawDataset(
            self._all_records_info, self._channels, self._crop_wake_mins,
            compute_recording_stats=self._compute_recording_stats,
        )
        # 1 worker for double-buffering: prefetches next recording from NFS while
        # GPU processes current one. Index is sorted by recording, so access is sequential.
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
        )

    def split_global_embedding_payload(self, payload) -> Dict[str, "EmbeddingPayload"]:
        from neuroatlas.benchmarking_helpers.registry.contracts import EmbeddingPayload

        features = np.asarray(payload.features)
        labels = np.asarray(payload.labels)
        metadata_list = list(payload.metadata)

        result = {}
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

    def _make_loader(self, dataset: _MrOSRawDataset, shuffle: bool) -> _LoaderAdapter:
        # Sequential extraction (shuffle=False): 1 worker for double-buffering.
        # Shuffled mode (not used for embedding extraction): no workers needed.
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
        )

    def train_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._test_ds, shuffle=False)
