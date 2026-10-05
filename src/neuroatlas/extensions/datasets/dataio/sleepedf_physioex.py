from __future__ import annotations

import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import mne
import numpy as np
import pandas as pd
import torch
from braindecode.datasets.base import BaseConcatDataset, BaseDataset
from braindecode.preprocessing import Preprocessor as BDPreprocessor
from braindecode.preprocessing import create_windows_from_events, preprocess
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from physioex.preprocess.utils.signal import xsleepnet_preprocessing
from physioex.preprocess.preprocessor import Preprocessor as PhysioExPreprocessor

from neuroatlas.extensions.models.backbones.core_sleep import load_core_sleep_model
from neuroatlas.extensions.datasets.dataio.core_sleep_eval import evaluate_core_sleep_model

_SLEEPEDF_MAPPING = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

_CORE_SLEEP_PICKS = ["Fpz-Cz", "EOG horizontal", "EMG submental"]
_CORE_SLEEP_SIGNAL_SHAPE = [3, 3000]
_CORE_SLEEP_XSLEEPNET_SHAPE = [3, 29, 129]
_CHANNEL_TYPE_MAP = {
    "EOG horizontal": "eog",
    "Resp oro-nasal": "misc",
    "EMG submental": "misc",
    "Temp rectal": "misc",
    "Event marker": "misc",
}


@dataclass(frozen=True)
class SleepEDFRecord:
    subject_orig: int
    recording_orig: int
    psg_path: str
    hypnogram_path: str


def _build_subject_folds(subject_values: np.ndarray, num_folds: int = 5, seed: int = 42) -> list[np.ndarray]:
    if len(subject_values) < num_folds:
        raise ValueError(f"Need at least {num_folds} unique subjects for {num_folds}-fold splitting.")
    rng = np.random.RandomState(seed)
    shuffled = np.asarray(subject_values).copy()
    rng.shuffle(shuffled)
    return [np.asarray(chunk, dtype=int) for chunk in np.array_split(shuffled, num_folds)]


def _crop_wake_region(raw: mne.io.BaseRaw, annotations: mne.Annotations, crop_wake_mins: int) -> None:
    if crop_wake_mins <= 0:
        return
    mask = [desc[-1] in ["1", "2", "3", "4", "R"] for desc in annotations.description]
    sleep_event_inds = np.where(mask)[0]
    if sleep_event_inds.size == 0:
        return
    tmin = annotations[int(sleep_event_inds[0])]["onset"] - crop_wake_mins * 60
    tmax = annotations[int(sleep_event_inds[-1])]["onset"] + crop_wake_mins * 60
    raw.crop(tmin=max(tmin, raw.times[0]), tmax=min(tmax, raw.times[-1]))


class LocalSleepEDFPhysioExPreprocessor(PhysioExPreprocessor):
    """Use PhysioEx preprocessing utilities against an existing local Sleep-EDF root."""

    def __init__(
        self,
        raw_root: str,
        data_folder: str,
        dataset_name: str = "sleepedf_physioex",
        crop_wake_mins: int = 30,
        max_records: Optional[int] = None,
    ):
        self.raw_root = Path(raw_root)
        self.dataset_name = dataset_name
        self.crop_wake_mins = crop_wake_mins
        self.max_records = max_records
        self._records: List[SleepEDFRecord] = []
        self.age_dict: Dict[int, Optional[float]] = {}
        self.sex_dict: Dict[int, Optional[str]] = {}
        super().__init__(
            dataset_name=dataset_name,
            signal_shape=_CORE_SLEEP_SIGNAL_SHAPE,
            preprocessors_name=["xsleepnet"],
            preprocessors=[xsleepnet_preprocessing],
            preprocessors_shape=[_CORE_SLEEP_XSLEEPNET_SHAPE],
            data_folder=data_folder,
        )

    def download_dataset(self) -> None:
        if not self.raw_root.exists():
            raise FileNotFoundError(f"Sleep-EDF raw root not found: {self.raw_root}")

    def get_subjects_records(self) -> List[SleepEDFRecord]:
        records: List[SleepEDFRecord] = []
        for patient_dir in sorted(path for path in self.raw_root.iterdir() if path.is_dir()):
            psg_files = sorted(patient_dir.glob("*-PSG.edf"))
            for psg_path in psg_files:
                prefix = psg_path.name.split("-PSG.edf")[0]
                hyp_matches = sorted(patient_dir.glob(f"{prefix[:6]}*-Hypnogram.edf"))
                if not hyp_matches:
                    logger.warning(f"Missing hypnogram for {psg_path}")
                    continue
                subject_orig = int(patient_dir.name.split("_")[-1])
                recording_orig = int(prefix[5])
                records.append(
                    SleepEDFRecord(
                        subject_orig=subject_orig,
                        recording_orig=recording_orig,
                        psg_path=str(psg_path),
                        hypnogram_path=str(hyp_matches[0]),
                    )
                )
        if self.max_records is not None:
            records = records[: self.max_records]
        self._records = records
        return records

    def read_subject_record(self, record: SleepEDFRecord):
        raw = mne.io.read_raw_edf(record.psg_path, preload=True, exclude=(), verbose="ERROR")
        annotations = mne.read_annotations(record.hypnogram_path)
        raw.set_annotations(annotations, emit_warning=False)
        _crop_wake_region(raw, annotations, self.crop_wake_mins)
        eeg_renames = {name: name.replace("EEG ", "") for name in raw.ch_names if name.startswith("EEG ")}
        raw.rename_channels(eeg_renames)
        raw.set_channel_types(_CHANNEL_TYPE_MAP)

        subject_info = raw.info.get("subject_info", {}) or {}
        self.age_dict[record.subject_orig] = subject_info.get("his_id")
        self.sex_dict[record.subject_orig] = subject_info.get("sex")

        dataset = BaseConcatDataset(
            [BaseDataset(raw, pd.Series({"subject": record.subject_orig, "recording": record.recording_orig}, name=""))]
        )
        preprocessors = [
            BDPreprocessor(lambda data: np.multiply(data, 1e6), apply_on_array=True),
            BDPreprocessor("filter", l_freq=0.3, h_freq=40),
        ]
        preprocess(dataset, preprocessors, n_jobs=1)
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            window_size_samples=30 * 100,
            window_stride_samples=30 * 100,
            preload=True,
            mapping=_SLEEPEDF_MAPPING,
            picks=_CORE_SLEEP_PICKS,
            n_jobs=1,
        )

        signals, labels = [], []
        for index in range(len(windows_dataset)):
            sig, label, _ = windows_dataset[index]
            signals.append(sig)
            labels.append(label)

        if not signals:
            return None, None
        return np.asarray(signals, dtype=np.float32), np.asarray(labels, dtype=np.int16)

    def customize_table(self, table: pd.DataFrame) -> pd.DataFrame:
        subject_orig, recording_orig, psg_paths, hyp_paths, ages, sexes = [], [], [], [], [], []
        for record_idx in table["subject_id"]:
            record = self._records[int(record_idx)]
            subject_orig.append(record.subject_orig)
            recording_orig.append(record.recording_orig)
            psg_paths.append(record.psg_path)
            hyp_paths.append(record.hypnogram_path)
            ages.append(self.age_dict.get(record.subject_orig))
            sexes.append(self.sex_dict.get(record.subject_orig))
        table["subject_orig"] = subject_orig
        table["recording_orig"] = recording_orig
        table["psg_path"] = psg_paths
        table["hypnogram_path"] = hyp_paths
        table["nsrr_age"] = ages
        table["sex"] = sexes
        return table

    def get_sets(self):
        subject_values = np.array(sorted(self.table["subject_orig"].unique()))
        subject_folds = _build_subject_folds(subject_values, num_folds=5, seed=42)

        def _record_ids(subject_subset: Sequence[int]) -> np.ndarray:
            record_ids = self.table.loc[self.table["subject_orig"].isin(subject_subset), "subject_id"].to_numpy(dtype=int)
            return record_ids.reshape(1, -1)

        train_subjects, valid_subjects, test_subjects = [], [], []
        for fold_idx in range(len(subject_folds)):
            test_subjects_fold = subject_folds[fold_idx]
            valid_subjects_fold = subject_folds[(fold_idx + 1) % len(subject_folds)]
            train_subjects_fold = np.concatenate([
                subject_folds[index]
                for index in range(len(subject_folds))
                if index not in {fold_idx, (fold_idx + 1) % len(subject_folds)}
            ])
            train_subjects.append(_record_ids(train_subjects_fold))
            valid_subjects.append(_record_ids(valid_subjects_fold))
            test_subjects.append(_record_ids(test_subjects_fold))

        return train_subjects, valid_subjects, test_subjects


class PhysioExSleepEDFCoreDataset(Dataset):
    def __init__(
        self,
        processed_root: str,
        split: str | None,
        fold: int = 0,
        norm_path: str | None = None,
        limit_windows: Optional[int] = None,
    ):
        self.processed_root = Path(processed_root)
        self.table = pd.read_csv(self.processed_root / "table.csv")
        self.fold_columns = [column for column in self.table.columns if column.startswith("fold_")]
        fold_key = f"fold_{fold}"
        if fold_key not in self.table.columns:
            raise KeyError(f"Split column {fold_key!r} not found in {self.processed_root / 'table.csv'}")
        self.rows = self.table if split is None else self.table[self.table[fold_key] == split]
        self.rows = self.rows.reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows found for split={split!r} in {self.processed_root}")
        if norm_path is None:
            norm_path = "src/neuroatlas/extensions/datasets/dataio/SHHS/assets/metrics_eeg_eog_emg_stft.pkl"
        with open(norm_path, "rb") as handle:
            self.norm_stats = pickle.load(handle)
        self._xsleep_cache: Dict[str, np.memmap] = {}
        self._label_cache: Dict[str, np.memmap] = {}
        self.index: List[tuple[int, int]] = []
        for row_idx, row in self.rows.iterrows():
            num_windows = int(row["num_windows"])
            self.index.extend((row_idx, window_idx) for window_idx in range(num_windows))
        if limit_windows is not None:
            self.index = self.index[:limit_windows]

    def __len__(self) -> int:
        return len(self.index)

    def _load_xsleep(self, row: pd.Series) -> np.memmap:
        path = str(row["xsleepnet"])
        if path not in self._xsleep_cache:
            shape = (int(row["num_windows"]),) + tuple(_CORE_SLEEP_XSLEEPNET_SHAPE)
            self._xsleep_cache[path] = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
        return self._xsleep_cache[path]

    def _load_labels(self, row: pd.Series) -> np.memmap:
        path = str(row["labels"])
        if path not in self._label_cache:
            self._label_cache[path] = np.memmap(path, dtype=np.int16, mode="r", shape=(int(row["num_windows"]),))
        return self._label_cache[path]

    def _normalize_channel(self, tf: np.ndarray, stat_key: str) -> torch.Tensor:
        tf = (tf - self.norm_stats["mean"][stat_key]["ch_0"][None, :]) / self.norm_stats["std"][stat_key]["ch_0"][None, :]
        tf = np.transpose(tf, (1, 0)).astype(np.float32)
        return torch.from_numpy(tf).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    def __getitem__(self, idx: int):
        row_idx, window_idx = self.index[idx]
        row = self.rows.iloc[row_idx]
        xsleep = self._load_xsleep(row)[window_idx]
        label = int(self._load_labels(row)[window_idx])
        return {
            "data": {
                "stft_eeg": self._normalize_channel(xsleep[0], "stft_eeg"),
                "stft_eog": self._normalize_channel(xsleep[1], "stft_eog"),
            },
            "label": torch.tensor(label, dtype=torch.long),
            "meta": {
                "subject_orig": int(row["subject_orig"]),
                "recording_orig": int(row["recording_orig"]),
                "window_idx": window_idx,
                "psg_path": row["psg_path"],
                "hypnogram_path": row["hypnogram_path"],
                "fold_assignments": {column: row[column] for column in self.fold_columns},
                # Underlying EEG context (STFT was computed from µV-scaled
                # Sleep-EDF signal at 100 Hz); informational for models
                # that consume the time-domain companion.
                "sampling_rate": 100.0,
                "unit": "uV",
                # Dataset-level ptp in µV, consumed by BENDR's 20th
                # SCALE channel (MODEL_CONTRACTS.md §2).
            },
        }


class PhysioExSleepEDFSignalDataset(Dataset):
    def __init__(
        self,
        processed_root: str,
        split: str | None,
        fold: int = 0,
        signal_kind: str = "raw",
        limit_windows: Optional[int] = None,
    ):
        self.processed_root = Path(processed_root)
        self.table = pd.read_csv(self.processed_root / "table.csv")
        self.fold_columns = [column for column in self.table.columns if column.startswith("fold_")]
        fold_key = f"fold_{fold}"
        if fold_key not in self.table.columns:
            raise KeyError(f"Split column {fold_key!r} not found in {self.processed_root / 'table.csv'}")
        self.rows = self.table if split is None else self.table[self.table[fold_key] == split]
        self.rows = self.rows.reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows found for split={split!r} in {self.processed_root}")
        self.signal_kind = signal_kind
        self._signal_cache: Dict[str, np.memmap] = {}
        self._label_cache: Dict[str, np.memmap] = {}
        self.index: List[tuple[int, int]] = []
        for row_idx, row in self.rows.iterrows():
            num_windows = int(row["num_windows"])
            self.index.extend((row_idx, window_idx) for window_idx in range(num_windows))
        if limit_windows is not None:
            self.index = self.index[:limit_windows]

    def __len__(self) -> int:
        return len(self.index)

    def _load_signal(self, row: pd.Series) -> np.memmap:
        path = str(row[self.signal_kind])
        if path not in self._signal_cache:
            shape = (int(row["num_windows"]),) + tuple(_CORE_SLEEP_SIGNAL_SHAPE)
            self._signal_cache[path] = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
        return self._signal_cache[path]

    def _load_labels(self, row: pd.Series) -> np.memmap:
        path = str(row["labels"])
        if path not in self._label_cache:
            self._label_cache[path] = np.memmap(path, dtype=np.int16, mode="r", shape=(int(row["num_windows"]),))
        return self._label_cache[path]

    def __getitem__(self, idx: int):
        row_idx, window_idx = self.index[idx]
        row = self.rows.iloc[row_idx]
        signal = self._load_signal(row)[window_idx]
        label = int(self._load_labels(row)[window_idx])
        return {
            "signals": {
                "eeg": torch.from_numpy(signal[:1].copy()),
                "full_signal": torch.from_numpy(signal.copy()),
            },
            "label": torch.tensor(label, dtype=torch.long),
            "meta": {
                "subject_orig": int(row["subject_orig"]),
                "recording_orig": int(row["recording_orig"]),
                "window_idx": window_idx,
                "psg_path": row["psg_path"],
                "hypnogram_path": row["hypnogram_path"],
                "signal_kind": self.signal_kind,
                "fold_assignments": {column: row[column] for column in self.fold_columns},
                # Sleep-EDF raw signals are preserved in µV at 100 Hz by
                # the PhysioEx preprocessor.
                "sampling_rate": 100.0,
                "unit": "uV",
                # Dataset-level ptp in µV for BENDR's 20th SCALE
                # channel (MODEL_CONTRACTS.md §2).
            },
        }


def preprocess_sleepedf_physioex(
    raw_root: str,
    data_root: str,
    dataset_name: str = "sleepedf_physioex",
    max_records: Optional[int] = None,
    force: bool = False,
) -> Path:
    dataset_root = Path(data_root) / dataset_name
    if dataset_root.exists() and (dataset_root / "table.csv").exists() and not force:
        logger.info(f"Reusing existing processed Sleep-EDF data at {dataset_root}")
        return dataset_root
    if force and dataset_root.exists():
        shutil.rmtree(dataset_root)
    preprocessor = LocalSleepEDFPhysioExPreprocessor(
        raw_root=raw_root,
        data_folder=data_root,
        dataset_name=dataset_name,
        max_records=max_records,
    )
    preprocessor.run()
    return dataset_root


def run_core_sleep_validation_on_sleepedf(
    processed_root: str,
    checkpoint_path: str,
    fold: int = 0,
    batch_size: int = 32,
    max_batches: Optional[int] = None,
    limit_windows_per_split: Optional[int] = None,
    num_workers: int = 0,
    device: Optional[str] = None,
    norm_path: Optional[str] = None,
):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    valid_ds = PhysioExSleepEDFCoreDataset(
        processed_root=processed_root,
        split="valid",
        fold=fold,
        norm_path=norm_path,
        limit_windows=limit_windows_per_split,
    )
    test_ds = PhysioExSleepEDFCoreDataset(
        processed_root=processed_root,
        split="test",
        fold=fold,
        norm_path=norm_path,
        limit_windows=limit_windows_per_split,
    )
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if max_batches is not None:
        class LimitedLoader:
            def __init__(self, loader, limit):
                self.loader = loader
                self.limit = limit

            def __iter__(self):
                count = 0
                for batch in self.loader:
                    if count >= self.limit:
                        break
                    yield batch
                    count += 1

            def __len__(self):
                return min(len(self.loader), self.limit)

        valid_loader = LimitedLoader(valid_loader, max_batches)
        test_loader = LimitedLoader(test_loader, max_batches)

    model = load_core_sleep_model(Path(checkpoint_path), device=device)
    val_metrics = evaluate_core_sleep_model(model, valid_loader, device=device)
    test_metrics = evaluate_core_sleep_model(model, test_loader, device=device)
    return val_metrics, test_metrics
