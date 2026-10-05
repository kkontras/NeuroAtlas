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

from neuroatlas.extensions.models.backbones._preproc import BCI_DOMAIN

from ._runtime_keys import BCI_TRIALS, RAW_ONLY
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
                    # What the wrappers' BCI branches key on (is_bci_batch);
                    # the slug alone never matched them.
                    "domain": BCI_DOMAIN,
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

    RUNTIME_KEYS_FIXED = RAW_ONLY
    RUNTIME_KEYS_IGNORED = BCI_TRIALS
    FIXED_WINDOW = True

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
            # In the cache context too: embeddings made before the BCI
            # branches fired are a different input, not a cache hit.
            "domain": BCI_DOMAIN,
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
        self._windows = windows
        self._subjects = list(subjects)
        self._split_subjects = {"train": list(train_subj), "val": list(val_subj),
                                "test": list(test_subj)}
        self._train_ds = _EpochDataset(windows, train_subj)
        self._val_ds = _EpochDataset(windows, val_subj)
        self._test_ds = _EpochDataset(windows, test_subj)

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._slug = slug
        self._sampling_rate = float(cfg.resample_sfreq)
        # The length of the windows actually cut, which is what FIXED_WINDOW
        # hands the backbone: the trial, or the trial less the first second
        # under confound control (resolve_confound_control's start offset).
        for ds in windows.datasets:
            if len(ds):
                n_times = int(ds[0][0].shape[-1])
                self.metadata["epoch_seconds"] = n_times / float(cfg.resample_sfreq)
                break
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

    # -- global embedding cache ----------------------------------------------
    #
    # A trial's embedding does not depend on the fold, so the cohort is
    # embedded once and each fold's split is cut from it by subject. Without
    # this the embeddings were cached per (fold, n_folds): `embed` (5-fold
    # default) and `probe --set n_folds=loso` looked in different cells, so
    # `run bci_motor_imagery` failed with "No embeddings ... split='train'"
    # right after its own embed step, and LOSO would have embedded the cohort
    # once per subject.

    def cache_context(self, purpose: str = "default") -> Dict[str, object]:
        context = dict(self.metadata)
        if purpose == "global_embeddings":
            context.pop("fold", None)
            context.pop("n_folds", None)
        return context

    def supports_global_embedding_cache(self) -> bool:
        return True

    def full_embedding_dataloader(self) -> _LoaderAdapter:
        subjects = self._subjects
        if self.embed_chunk is not None:
            chunk_idx, n_chunks = self.embed_chunk
            subjects = [s for i, s in enumerate(sorted(subjects)) if i % n_chunks == chunk_idx]
        return self._make_loader(_EpochDataset(self._windows, subjects), shuffle=False)

    def split_global_embedding_payload(self, payload) -> Dict[str, object]:
        from neuroatlas.benchmarking_helpers import EmbeddingPayload
        import numpy as np

        subject = np.asarray([int(m["subject_id"]) for m in payload.metadata])
        out: Dict[str, object] = {}
        for split, members in self._split_subjects.items():
            keep = np.flatnonzero(np.isin(subject, members))
            if keep.size == 0:
                raise ValueError(
                    f"{self._slug}: no embedding rows for the {split} subjects {members} "
                    f"(fold {self.metadata.get('fold')}, n_folds {self.metadata.get('n_folds')})."
                )
            out[split] = EmbeddingPayload(
                features=payload.features[keep],
                labels=np.asarray(payload.labels)[keep],
                metadata=[payload.metadata[i] for i in keep],
            )
        return out
