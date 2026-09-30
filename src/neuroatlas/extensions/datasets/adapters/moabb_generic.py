"""Generic BenchmarkDataModule adapter for any MOABB dataset.

Works with any dataset registered in ``dataio/moabb_loader.MOABB_DATASETS``.
The adapter loads data entirely from MOABB (no preprocessed-file fast path).
Subject splitting reuses the deterministic ``get_subject_split`` from
``dataio/bci``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .base import BenchmarkDataModule


# ---------------------------------------------------------------------------
# Torch Dataset over a braindecode windowed dataset
# ---------------------------------------------------------------------------

class _EpochDataset(Dataset):
    """Flat torch Dataset over a braindecode windowed dataset."""

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


# ---------------------------------------------------------------------------
# Collate + loader adapter
# ---------------------------------------------------------------------------

def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "eeg": torch.stack([b["eeg"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "subject_ids": [b["subject_id"] for b in batch],
        "trial_idxs": [b["trial_idx"] for b in batch],
        "session_ids": [b["session_id"] for b in batch],
    }


class _LoaderAdapter:
    """Wrap a DataLoader to emit the standard benchmark batch format."""

    def __init__(
        self,
        slug: str,
        loader: DataLoader,
        sampling_rate: float,
        channels: List[str],
    ) -> None:
        self._slug = slug
        self._loader = loader
        self._sampling_rate = sampling_rate
        self._channels = channels

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        for batch in self._loader:
            meta = [
                {
                    "dataset": self._slug,
                    "subject_id": batch["subject_ids"][i],
                    "trial_idx": batch["trial_idxs"][i],
                    "session_id": batch["session_ids"][i],
                    "sampling_rate": self._sampling_rate,
                    "unit": "uV",
                    "channels": list(self._channels),
                }
                for i in range(len(batch["subject_ids"]))
            ]
            yield {
                "signals": {"eeg": batch["eeg"]},
                "label": batch["label"],
                "meta": meta,
                "raw_batch": batch,
            }


# ---------------------------------------------------------------------------
# Generic BenchmarkDataModule
# ---------------------------------------------------------------------------

class MOABBBenchmarkDataModule(BenchmarkDataModule):
    """BenchmarkDataModule for any dataset registered in MOABB_DATASETS.

    Data is loaded entirely from MOABB (download + preprocess + window).
    """

    def __init__(
        self,
        slug: str,
        fold: int = 0,
        n_folds: int = 5,
        batch_size: int = 64,
        num_workers: int = 0,
        n_jobs: int = 1,
        subject_ids: Optional[Sequence[int]] = None,
        confound_control: bool = False,
    ) -> None:
        from neuroatlas.extensions.datasets.dataio.moabb_loader import (
            MOABB_DATASETS,
            get_moabb_subjects,
            load_and_preprocess_moabb,
        )
        from neuroatlas.extensions.datasets.dataio.bci import get_subject_split

        cfg = MOABB_DATASETS[slug]
        subjects = list(subject_ids) if subject_ids is not None else get_moabb_subjects(cfg)
        train_subj, val_subj, test_subj = get_subject_split(
            subjects, fold=fold, n_folds=n_folds,
        )

        meta: Dict[str, object] = {
            "paradigm": cfg.paradigm,
            "n_classes": cfg.n_classes,
            "epoch_seconds": cfg.trial_duration,
            "channel_policy": ["eeg"],
            "signal_kind": "raw",
            "fold": fold,
            "n_folds": n_folds,
            "n_channels": cfg.n_channels,
            "sfreq": cfg.resample_sfreq,
            "sampling_rate": float(cfg.resample_sfreq),
            "source": "moabb",
            # Part of the cache context, so confound-controlled embeddings
            # land in their own cell instead of overwriting the others.
            "confound_control": bool(confound_control),
        }
        super().__init__(name=slug, metadata=meta)

        windows = load_and_preprocess_moabb(
            cfg, subject_ids=subjects, n_jobs=n_jobs,
            confound_control=confound_control,
        )
        self._train_ds = _EpochDataset(windows, train_subj)
        self._val_ds = _EpochDataset(windows, val_subj)
        self._test_ds = _EpochDataset(windows, test_subj)

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._slug = slug
        self._sampling_rate = float(cfg.resample_sfreq)
        self._channels: List[str] = []
        if windows.datasets:
            raw = windows.datasets[0].raw
            if raw is not None:
                self._channels = list(raw.ch_names)
            else:
                self._channels = [f"EEG{i}" for i in range(cfg.n_channels)]

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> _LoaderAdapter:
        return _LoaderAdapter(
            self._slug,
            DataLoader(
                dataset,
                batch_size=self._batch_size,
                shuffle=shuffle,
                num_workers=self._num_workers,
                collate_fn=_collate,
                pin_memory=False,
            ),
            sampling_rate=self._sampling_rate,
            channels=self._channels,
        )

    def train_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._test_ds, shuffle=False)
