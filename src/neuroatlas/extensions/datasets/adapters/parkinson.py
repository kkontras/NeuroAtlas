from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from neuroatlas.benchmarking_helpers.registry.splits import make_subject_kfold

from neuroatlas.extensions.datasets.dataio.parkinson import (
    MULTI_CHANNELS,
    SINGLE_CHANNEL,
    ParkinsonDataset,
    load_pd_labels,
    load_subject_list,
)

from ._runtime_keys import RAW_ONLY
from .base import BenchmarkDataModule
from neuroatlas.benchmarking_helpers import dataloader_worker_init_fn

_DATA_ROOT_DEFAULT = "${EEG_DATA_ROOT}/raw-sleep/Parkinson_data"


def _collate_fn(batch):
    """Custom collate that handles variable-length signals by keeping them stacked."""
    eeg = torch.stack([item["eeg"] for item in batch])  # (B, 1, samples)
    sleep_stage = torch.tensor([item["sleep_stage"] for item in batch], dtype=torch.long)
    pd_label = torch.tensor([item["pd_label"] for item in batch], dtype=torch.long)
    subject_ids = [item["subject_id"] for item in batch]
    epoch_idxs = [item["epoch_idx"] for item in batch]
    out = {
        "eeg": eeg,
        "sleep_stage": sleep_stage,
        "pd_label": pd_label,
        "subject_ids": subject_ids,
        "epoch_idxs": epoch_idxs,
    }
    # Per-sample recording statistics, kept as lists (one dict/array per
    # sample), present only when the dataset was asked to compute them.
    for key in _RECORDING_STAT_KEYS:
        if all(key in item for item in batch):
            out[key] = [item[key] for item in batch]
    return out


_RECORDING_STAT_KEYS = ("recording_mean", "recording_std", "recording_q95", "recording_q95_bipolar")


class _LoaderAdapter:
    """Wrap a DataLoader to emit batches in the standard benchmark format.

    Standard format::

        {
            "signals": {"eeg": Tensor(B, 1, samples)},
            "label":   Tensor(B,)  — PD/HOA binary label,
            "meta":    List[dict]  — per-sample metadata,
            "raw_batch": original raw batch dict,
        }

    ``meta[i]`` publishes ``sampling_rate``, ``unit``, and ``channels``
    per ``MODEL_CONTRACTS.md`` §3. The Parkinson pipeline resamples via
    ``dataio/parkinson.py`` to ``target_sfreq`` (default 256 Hz); MNE's
    ``raw.get_data()`` returns **Volts** and the dataio does not convert
    — hence ``unit="V"``. Wrappers apply the µV conversion via
    ``unit_to_uv`` in their ``_prepare_input``.
    """

    def __init__(
        self,
        loader: DataLoader,
        *,
        sampling_rate: Optional[float],
        unit: str,
        channels: List[str],
    ) -> None:
        self._loader = loader
        self._sampling_rate = float(sampling_rate) if sampling_rate is not None else None
        self._unit = unit
        self._channels = list(channels)

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        for batch in self._loader:
            per_sample: Dict[str, object] = {
                "unit": self._unit,
                "channels": list(self._channels),
            }
            if self._sampling_rate is not None:
                per_sample["sampling_rate"] = self._sampling_rate
            meta = [
                {
                    "dataset": "parkinson",
                    "subject_id": batch["subject_ids"][i],
                    "epoch_index": batch["epoch_idxs"][i],
                    "sleep_stage": int(batch["sleep_stage"][i]),
                    **per_sample,
                    **{key: batch[key][i] for key in _RECORDING_STAT_KEYS if key in batch},
                }
                for i in range(len(batch["subject_ids"]))
            ]
            yield {
                "signals": {"eeg": batch["eeg"]},
                "label": batch["pd_label"],
                "meta": meta,
                "raw_batch": batch,
            }


def _patient_splits(
    all_subjects: List[str],
    pd_labels: Dict[str, int],
    fold: int,
    n_folds: int,
) -> tuple[List[str], List[str], List[str]]:
    """Return (train_subjects, val_subjects, test_subjects) for the given fold.

    Uses StratifiedKFold at the patient level.  Each outer fold defines a test
    set (~1/n_folds of patients).  The remaining patients are further split 90/10
    into train/val by taking the *next* fold index as the validation set.
    """
    # Keep only subjects that have labels
    labeled = [s for s in all_subjects if s in pd_labels]
    y = np.array([pd_labels[s] for s in labeled])

    # Shared rule: benchmarking_helpers/splits. Previously this passed
    # n_folds straight to StratifiedKFold, which raises as soon as a class
    # has fewer members than folds -- so n_folds above the minority-class
    # size was an error rather than a split, and LOSO was unreachable.
    skf, _stratified, n_folds = make_subject_kfold(n_folds, y, seed=42)
    splits = list(skf.split(labeled, y))

    # Test fold
    train_val_idx, test_idx = splits[fold % n_folds]
    test_subjects = [labeled[i] for i in test_idx]

    # Val fold: use the next fold's test indices from the train_val pool
    train_val_subjects = [labeled[i] for i in train_val_idx]
    train_val_y = y[train_val_idx]
    inner_skf, _inner_strat, inner_folds = make_subject_kfold(
        n_folds, train_val_y, seed=42)
    val_fold = (fold + 1) % inner_folds
    inner_splits = list(inner_skf.split(train_val_subjects, train_val_y))
    inner_train_idx, inner_val_idx = inner_splits[val_fold % len(inner_splits)]
    train_subjects = [train_val_subjects[i] for i in inner_train_idx]
    val_subjects = [train_val_subjects[i] for i in inner_val_idx]

    return train_subjects, val_subjects, test_subjects


class ParkinsonBenchmarkDataModule(BenchmarkDataModule):
    """BenchmarkDataModule for the Parkinson PSG dataset.

    Performs patient-level stratified k-fold cross-validation.  Each sample is
    a 30-second EEG epoch; the label is the binary patient-level PD/HOA label.

    Args:
        data_root: Path to the Parkinson_data directory.
        fold: Zero-based fold index for CV.
        n_folds: Total number of CV folds (default 5).
        batch_size: Batch size for data loaders.
        channel: EEG channel to extract from EDF files.
        target_sfreq: Resample to this frequency (Hz); None keeps native rate.
        num_workers: DataLoader worker count.
        subject_list_file: Filename inside Datasets/ subdirectory listing subjects.
        compute_recording_stats: Publish each night's amplitude statistics
            in ``meta`` for the models that normalise per recording.
    """

    RUNTIME_KEYS_FIXED = RAW_ONLY
    RUNTIME_KEYS_IGNORED = {
        "epoch_seconds": (
            "the samples are the hypnogram's scored 30-s epochs "
            "(dataio/parkinson.py EPOCH_SECONDS); a model's own window is met by its wrapper"
        ),
    }

    def __init__(
        self,
        data_root: str = _DATA_ROOT_DEFAULT,
        fold: int = 0,
        n_folds: int = 5,
        batch_size: int = 32,
        mode: str = "single",
        target_sfreq: Optional[float] = 256.0,
        num_workers: int = 0,
        subject_list_file: str = "ds_all_86.tsv",
        channel_specs=None,
        use_all_eeg_channels: bool = False,
        compute_recording_stats: bool = False,
    ) -> None:
        mode = str(mode).lower()
        if use_all_eeg_channels:
            mode = "multi"
        if mode not in {"single", "multi"}:
            raise ValueError(f"Unsupported Parkinson mode {mode!r}. Expected 'single' or 'multi'.")
        channels = SINGLE_CHANNEL if mode == "single" else MULTI_CHANNELS
        if isinstance(channel_specs, list):
            self._channel_names_override = list(channel_specs)
        else:
            self._channel_names_override = None
        meta: Dict[str, object] = {
            "canonical_label_space": ["HOA", "PD"],
            "epoch_seconds": 30,
            "channel_policy": ["eeg"],
            "signal_kind": "raw",
            "fold": fold,
            "n_folds": n_folds,
            "mode": mode,
            "channels": list(channels),
        }
        super().__init__(name="parkinson", metadata=meta)

        self._data_root = Path(data_root)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._mode = mode
        self._channels = list(channels)
        self._target_sfreq = target_sfreq

        csv_path = self._data_root / "Target_sleep_demographic.csv"
        pd_labels = load_pd_labels(str(csv_path))

        all_subjects = load_subject_list(
            str(self._data_root / "Datasets"), filename=subject_list_file
        )

        train_subjects, val_subjects, test_subjects = _patient_splits(
            all_subjects, pd_labels, fold=fold, n_folds=n_folds
        )

        self._train_ds = ParkinsonDataset(
            data_root=str(self._data_root),
            subject_ids=train_subjects,
            pd_labels=pd_labels,
            channels=self._channels,
            target_sfreq=target_sfreq,
            compute_recording_stats=compute_recording_stats,
        )
        self._val_ds = ParkinsonDataset(
            data_root=str(self._data_root),
            subject_ids=val_subjects,
            pd_labels=pd_labels,
            channels=self._channels,
            target_sfreq=target_sfreq,
            compute_recording_stats=compute_recording_stats,
        )
        self._test_ds = ParkinsonDataset(
            data_root=str(self._data_root),
            subject_ids=test_subjects,
            pd_labels=pd_labels,
            channels=self._channels,
            target_sfreq=target_sfreq,
            compute_recording_stats=compute_recording_stats,
        )

    def _make_loader(self, dataset: ParkinsonDataset, shuffle: bool) -> _LoaderAdapter:
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
            collate_fn=_collate_fn,
            pin_memory=False,
            worker_init_fn=dataloader_worker_init_fn,
        )
        # MNE's raw.get_data() returns Volts; Parkinson dataio preserves
        # that scale end-to-end (see dataio/parkinson.py). Adapter declares
        # unit="V" so wrappers' unit_to_uv step converts to µV.
        # target_sfreq can be None (keep-native). In that case recordings
        # retain their per-file fs and the adapter cannot publish a single
        # rate — wrappers fall back to inferring from tensor length.
        return _LoaderAdapter(
            loader,
            sampling_rate=self._target_sfreq,
            unit="V",
            channels=self._channel_names_override or self._channels,
        )

    def train_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._test_ds, shuffle=False)
