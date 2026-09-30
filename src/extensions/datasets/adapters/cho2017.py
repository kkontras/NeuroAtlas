"""BenchmarkDataModule adapter for the Cho2017 motor-imagery dataset."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from extensions.datasets.dataio.bci import (
    DATASET_CONFIGS,
    get_subject_split,
    load_and_preprocess,
    load_preprocessed_dataset,
)

from .base import BenchmarkDataModule

_CFG = DATASET_CONFIGS["cho2017"]
_SLUG = "cho2017"


class _PreprocessedDataset(Dataset):
    def __init__(self, mat: Dict[str, Any], subject_indices: Sequence[int]) -> None:
        self._items: list = []
        data = np.asarray(mat["data_raw"]).squeeze()
        conditions = np.asarray(mat["condition"]).squeeze()
        subjects = np.asarray(mat["subject_name"]).squeeze()
        for idx in subject_indices:
            trials = np.asarray(data[idx], dtype=np.float32)
            labels = np.asarray(conditions[idx]).flatten().astype(int)
            subj_id = int(np.asarray(subjects[idx]).flat[0])
            for t in range(len(trials)):
                self._items.append((trials[t], int(labels[t]), subj_id, t))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        x, y, subj, trial = self._items[idx]
        return {
            "eeg": torch.as_tensor(x, dtype=torch.float32),
            "label": y,
            "subject_id": subj,
            "trial_idx": trial,
        }


class _EpochDataset(Dataset):
    def __init__(self, windows_dataset, subject_ids: Sequence[int]) -> None:
        self._items: list = []
        subject_set = set(subject_ids)
        for ds in windows_dataset.datasets:
            subj = ds.description["subject"]
            if subj not in subject_set:
                continue
            for j in range(len(ds)):
                x, y, _ = ds[j]
                self._items.append((x, int(y), int(subj), j))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        x, y, subj, trial = self._items[idx]
        return {
            "eeg": torch.as_tensor(x, dtype=torch.float32),
            "label": y,
            "subject_id": subj,
            "trial_idx": trial,
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "eeg": torch.stack([b["eeg"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "subject_ids": [b["subject_id"] for b in batch],
        "trial_idxs": [b["trial_idx"] for b in batch],
    }


class _LoaderAdapter:
    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        for batch in self._loader:
            meta = [
                {
                    "dataset": _SLUG,
                    "subject_id": batch["subject_ids"][i],
                    "trial_idx": batch["trial_idxs"][i],
                    "sampling_rate": float(_CFG.resample_sfreq),
                    "unit": "uV",
                    "channels": list(_CFG.channels),
                }
                for i in range(len(batch["subject_ids"]))
            ]
            yield {"signals": {"eeg": batch["eeg"]}, "label": batch["label"], "meta": meta, "raw_batch": batch}


class Cho2017BenchmarkDataModule(BenchmarkDataModule):
    """BenchmarkDataModule for Cho2017 (motor imagery, 2-class).

    If a preprocessed .mat exists, loads from it. Otherwise falls back to MOABB.
    """

    def __init__(
        self,
        fold: int = 0,
        n_folds: int = 5,
        batch_size: int = 64,
        num_workers: int = 0,
        n_jobs: int = 1,
        subject_ids: Optional[Sequence[int]] = None,
        preprocessed_path: Optional[str] = None,
    ) -> None:
        subjects = list(subject_ids) if subject_ids is not None else list(_CFG.subjects)
        train_subj, val_subj, test_subj = get_subject_split(subjects, fold=fold, n_folds=n_folds)
        meta: Dict[str, object] = {
            "canonical_label_space": list(_CFG.targets),
            "epoch_seconds": _CFG.tmax - _CFG.tmin,
            "channel_policy": ["eeg"],
            "signal_kind": "raw",
            "fold": fold,
            "n_folds": n_folds,
            "n_channels": len(_CFG.channels),
            "sfreq": _CFG.resample_sfreq,
            "sampling_rate": float(_CFG.resample_sfreq),
        }
        super().__init__(name=_SLUG, metadata=meta)

        mat = load_preprocessed_dataset(_SLUG, preprocessed_path=preprocessed_path)
        if mat is not None:
            subj_names = np.asarray(mat["subject_name"]).squeeze()
            n_subjects = subj_names.shape[0]
            subj_to_idx = {int(np.asarray(subj_names[i]).flat[0]): i for i in range(n_subjects)}
            train_idx = [subj_to_idx[s] for s in train_subj if s in subj_to_idx]
            val_idx = [subj_to_idx[s] for s in val_subj if s in subj_to_idx]
            test_idx = [subj_to_idx[s] for s in test_subj if s in subj_to_idx]
            self._train_ds = _PreprocessedDataset(mat, train_idx)
            self._val_ds = _PreprocessedDataset(mat, val_idx)
            self._test_ds = _PreprocessedDataset(mat, test_idx)
            meta["source"] = "preprocessed"
        else:
            windows = load_and_preprocess(_CFG, subject_ids=subjects, n_jobs=n_jobs)
            self._train_ds = _EpochDataset(windows, train_subj)
            self._val_ds = _EpochDataset(windows, val_subj)
            self._test_ds = _EpochDataset(windows, test_subj)
            meta["source"] = "moabb"

        self._batch_size = batch_size
        self._num_workers = num_workers

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> _LoaderAdapter:
        return _LoaderAdapter(DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle, num_workers=self._num_workers, collate_fn=_collate, pin_memory=False))

    def train_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._test_ds, shuffle=False)
