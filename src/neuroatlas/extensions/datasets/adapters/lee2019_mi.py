"""BenchmarkDataModule adapter for the Lee2019_MI motor-imagery dataset."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from neuroatlas.extensions.datasets.dataio.bci import (
    DATASET_CONFIGS,
    get_subject_split,
    load_and_preprocess,
    load_preprocessed_dataset,
)

from neuroatlas.extensions.models.backbones._preproc import BCI_DOMAIN
from ._runtime_keys import BCI_TRIALS, RAW_ONLY
from .base import BenchmarkDataModule

_CFG = DATASET_CONFIGS["lee2019_mi"]
_SLUG = "lee2019_mi"


class _PreprocessedDataset(Dataset):
    """Lee2019_MI preprocessed .mat structure:

    - ``data_raw``: ``(N_subj*N_sess, n_trials, n_ch, n_t)`` — e.g. (108, 100, 20, 300)
    - ``condition``: ``(N_subj, n_trials*N_sess)`` — e.g. (54, 200)  (object array of int arrays)
    - ``subject_name``: ``(1, N_subj)`` — e.g. (1, 54)

    54 subjects × 2 sessions = 108 data entries.
    """

    def __init__(
        self,
        mat: Dict[str, Any],
        subject_indices: Sequence[int],
        session_mode: str = "stack",
        session_id: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        session_mode : ``"stack"`` (default) concatenates all sessions per
            subject; ``"single"`` uses only the session given by *session_id*.
        session_id : 0-based session index, required when ``session_mode="single"``.
        """
        self._items: list = []
        data = np.asarray(mat["data_raw"], dtype=np.float64)
        conditions = np.asarray(mat["condition"]).squeeze()
        subjects = np.asarray(mat["subject_name"]).squeeze()

        n_unique = len(subjects)
        total_entries = data.shape[0]
        n_sessions = total_entries // n_unique if n_unique > 0 else 1
        n_trials_per_session = data.shape[1]

        sessions_to_use = list(range(n_sessions))
        if session_mode == "single":
            if session_id is None:
                raise ValueError("session_id is required when session_mode='single'")
            sessions_to_use = [session_id]

        for idx in subject_indices:
            subj_id = int(np.asarray(subjects[idx]).flat[0])
            subj_labels = np.asarray(conditions[idx]).flatten().astype(int)
            for sess in sessions_to_use:
                flat_idx = idx * n_sessions + sess
                if flat_idx >= total_entries:
                    continue
                trials = np.asarray(data[flat_idx], dtype=np.float32)
                label_offset = sess * n_trials_per_session
                for t in range(n_trials_per_session):
                    label = int(subj_labels[label_offset + t]) if (label_offset + t) < len(subj_labels) else 0
                    self._items.append((trials[t], label, subj_id, t, sess))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        x, y, subj, trial, sess = self._items[idx]
        return {
            "eeg": torch.as_tensor(x, dtype=torch.float32),
            "label": y,
            "subject_id": subj,
            "trial_idx": trial,
            "session_id": sess,
        }


class _EpochDataset(Dataset):
    def __init__(self, windows_dataset, subject_ids: Sequence[int]) -> None:
        self._items: list = []
        subject_set = set(subject_ids)
        for ds in windows_dataset.datasets:
            subj = ds.description["subject"]
            if subj not in subject_set:
                continue
            sess = ds.description.get("session", 0)
            for j in range(len(ds)):
                x, y, _ = ds[j]
                self._items.append((x, int(y), int(subj), j, sess))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        x, y, subj, trial, sess = self._items[idx]
        return {
            "eeg": torch.as_tensor(x, dtype=torch.float32),
            "label": y,
            "subject_id": subj,
            "trial_idx": trial,
            "session_id": sess,
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "eeg": torch.stack([b["eeg"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "subject_ids": [b["subject_id"] for b in batch],
        "trial_idxs": [b["trial_idx"] for b in batch],
        "session_ids": [b["session_id"] for b in batch],
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
                    "domain": BCI_DOMAIN,
                    "subject_id": batch["subject_ids"][i],
                    "trial_idx": batch["trial_idxs"][i],
                    "session_id": batch["session_ids"][i],
                    "sampling_rate": float(_CFG.resample_sfreq),
                    "unit": "uV",
                    "channels": list(_CFG.channels),
                }
                for i in range(len(batch["subject_ids"]))
            ]
            yield {"signals": {"eeg": batch["eeg"]}, "label": batch["label"], "meta": meta, "raw_batch": batch}


class Lee2019MIBenchmarkDataModule(BenchmarkDataModule):
    """BenchmarkDataModule for Lee2019_MI (motor imagery, 2-class).

    If a preprocessed .mat exists, loads from it. Otherwise falls back to MOABB.
    """

    RUNTIME_KEYS_FIXED = RAW_ONLY
    RUNTIME_KEYS_IGNORED = BCI_TRIALS
    FIXED_WINDOW = True

    def __init__(
        self,
        fold: int = 0,
        n_folds: int = 5,
        batch_size: int = 64,
        num_workers: int = 0,
        n_jobs: int = 1,
        subject_ids: Optional[Sequence[int]] = None,
        preprocessed_path: Optional[str] = None,
        session_mode: str = "stack",
        session_id: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        session_mode : ``"stack"`` concatenates both sessions per subject
            (default); ``"single"`` uses only the session at *session_id*.
        session_id : 0-based session index, required when ``session_mode="single"``.
        """
        subjects = list(subject_ids) if subject_ids is not None else list(_CFG.subjects)
        train_subj, val_subj, test_subj = get_subject_split(subjects, fold=fold, n_folds=n_folds)
        meta: Dict[str, object] = {
            "domain": BCI_DOMAIN,
            "canonical_label_space": list(_CFG.targets),
            "epoch_seconds": _CFG.tmax - _CFG.tmin,
            "channel_policy": ["eeg"],
            "signal_kind": "raw",
            "fold": fold,
            "n_folds": n_folds,
            "n_channels": len(_CFG.channels),
            "sfreq": _CFG.resample_sfreq,
            "sampling_rate": float(_CFG.resample_sfreq),
            "session_mode": session_mode,
        }
        super().__init__(name=_SLUG, metadata=meta)

        mat = load_preprocessed_dataset(_SLUG, preprocessed_path=preprocessed_path)
        if mat is not None:
            subj_names = np.asarray(mat["subject_name"]).squeeze()
            unique_subjs = np.unique(subj_names)
            subj_to_idx = {int(s): i for i, s in enumerate(unique_subjs)}
            train_idx = [subj_to_idx[s] for s in train_subj if s in subj_to_idx]
            val_idx = [subj_to_idx[s] for s in val_subj if s in subj_to_idx]
            test_idx = [subj_to_idx[s] for s in test_subj if s in subj_to_idx]
            self._train_ds = _PreprocessedDataset(mat, train_idx, session_mode=session_mode, session_id=session_id)
            self._val_ds = _PreprocessedDataset(mat, val_idx, session_mode=session_mode, session_id=session_id)
            self._test_ds = _PreprocessedDataset(mat, test_idx, session_mode=session_mode, session_id=session_id)
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
