"""BenchmarkDataModule adapter for the Siena Scalp EEG dataset.

Reads directly from BIDS EDF files (``backend="bids"``); the legacy
HDF5-cache backend has been removed (its dataio module
``dataio/siena.py`` was deleted in favour of the BIDS-direct path).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from benchmarking_helpers.registry.splits import make_subject_kfold
from torch.utils.data import DataLoader, WeightedRandomSampler

from .base import BenchmarkDataModule

_BIDS_ROOT_DEFAULT = "${REPO_ROOT}/siena_cache/raw/BIDS_Siena"


# ---------------------------------------------------------------------------
# Loader adapter
# ---------------------------------------------------------------------------


# The shared wrapper; the local name is kept so call sites are unchanged.
from ._loader_adapters import SplitTaggingLoader as _LoaderAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Generic patient-level split
# ---------------------------------------------------------------------------


def _patient_splits_generic(
    subject_ids_per_rec: List[str],
    has_seizure_per_rec: List[bool],
    fold: int,
    n_folds: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Patient-disjoint stratified k-fold splits."""
    n_rec = len(subject_ids_per_rec)

    subject_to_recs: Dict[str, List[int]] = {}
    subject_has_seizure: Dict[str, bool] = {}
    for i in range(n_rec):
        sid = subject_ids_per_rec[i]
        subject_to_recs.setdefault(sid, []).append(i)
        if has_seizure_per_rec[i]:
            subject_has_seizure[sid] = True
        subject_has_seizure.setdefault(sid, False)

    all_subjects = sorted(subject_to_recs.keys())
    strat_labels = [
        "seiz" if subject_has_seizure.get(s, False) else "none"
        for s in all_subjects
    ]

    # Siena has only 14 subjects — cap folds at subject count
    # One rule for every dataset: benchmarking_helpers/splits. Capping at the
    # subject count was not enough -- StratifiedKFold additionally needs every
    # class to have n_splits members, so `n_folds` above the minority-class
    # size raised instead of splitting. The helper keeps the requested folds
    # and drops stratification when the labels cannot carry it, which is what
    # makes leave-one-subject-out (n_folds = n_subjects) expressible here.
    skf, _stratified, effective_folds = make_subject_kfold(
        n_folds, strat_labels, seed=42)
    splits = list(skf.split(all_subjects, strat_labels))
    train_val_idx, test_idx = splits[fold % effective_folds]

    test_subjects = {all_subjects[i] for i in test_idx}

    tv_subjects = [all_subjects[i] for i in train_val_idx]
    tv_strat = [strat_labels[i] for i in train_val_idx]

    inner_skf, _inner_strat, inner_folds = make_subject_kfold(
        effective_folds, tv_strat, seed=42)
    inner_splits = list(inner_skf.split(tv_subjects, tv_strat))
    val_fold = (fold + 1) % inner_folds
    inner_train_idx, inner_val_idx = inner_splits[val_fold % len(inner_splits)]

    train_subjects = {tv_subjects[i] for i in inner_train_idx}
    val_subjects = {tv_subjects[i] for i in inner_val_idx}

    train_recs = [i for i in range(n_rec) if subject_ids_per_rec[i] in train_subjects]
    val_recs = [i for i in range(n_rec) if subject_ids_per_rec[i] in val_subjects]
    test_recs = [i for i in range(n_rec) if subject_ids_per_rec[i] in test_subjects]

    return train_recs, val_recs, test_recs


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class SienaBenchmarkDataModule(BenchmarkDataModule):
    """BenchmarkDataModule for the Siena Scalp EEG seizure detection dataset.

    Args:
        backend: ``"bids"`` (default) or ``"hdf5"``.
        bids_root: Path to BIDS root (for ``backend="bids"``).
        cache_root: Path to HDF5 cache dir (for ``backend="hdf5"``).
        fold: Current fold index (0-based).
        n_folds: Total number of folds.
        batch_size: Batch size for dataloaders.
        num_workers: DataLoader workers.
        window_s: Window duration in seconds.
        stride_s: Window stride in seconds.
        montage: ``"unipolar"`` (19 ch) or ``"bipolar"`` (18 ch).
        label_mode: ``"binary"`` for seizure detection.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        balance: ``"weighted_sampler"`` or ``"none"``.
        overlap_threshold: Seizure fraction threshold for binary labels.
    """

    def __init__(
        self,
        backend: str = "bids",
        bids_root: Optional[str] = _BIDS_ROOT_DEFAULT,
        fold: int = 0,
        n_folds: int = 5,
        batch_size: int = 64,
        num_workers: int = 4,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        montage: str = "bipolar",
        label_mode: str = "binary",
        normalize: str = "none",
        balance: str = "weighted_sampler",
        overlap_threshold: float = 0.0,
        signal_kind: str = "raw",
        **kwargs,
    ) -> None:
        metadata = {
            "canonical_label_space": ["bckg", "seiz"],
            "epoch_seconds": window_s,
            "channel_policy": ["eeg"],
            "signal_kind": signal_kind,
            "fold": fold,
            "n_folds": n_folds,
            "backend": backend,
        }
        super().__init__(name="siena", metadata=metadata)

        self._backend = backend
        self._fold = fold
        self._n_folds = n_folds
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._window_s = window_s
        self._stride_s = stride_s
        self._montage = montage
        self._label_mode = label_mode
        self._normalize = normalize
        self._balance = balance
        self._overlap_threshold = overlap_threshold

        if backend != "bids":
            raise ValueError(
                f"Unknown backend {backend!r}. The legacy 'hdf5' backend has "
                f"been removed; only 'bids' is supported."
            )
        from extensions.datasets.epilepsy.bids_index import (
            BIDSRecordingIndex,
            find_bids_root,
        )
        from extensions.datasets.dataio.siena_bids import (
            SienaBIDSDataset,
            _collate_siena,
        )

        root = find_bids_root(Path(bids_root)) if bids_root else None
        if root is None:
            raise FileNotFoundError(
                f"No BIDS root found at {bids_root}. Download with: "
                "python -m entrypoints.fetch --dataset siena --download"
            )
        self._bids_root = str(root)
        self._bids_index = BIDSRecordingIndex.from_bids_root(root)
        self._DatasetCls = SienaBIDSDataset
        self._collate_fn = _collate_siena

        self._train_recs, self._val_recs, self._test_recs = _patient_splits_generic(
            self._bids_index.subject_ids_per_recording(),
            self._bids_index.seizure_presence_per_recording(),
            fold, n_folds,
        )

        self._datasets: Dict[str, Any] = {}

    def _get_dataset(self, split: str):
        if split not in self._datasets:
            rec_indices = {
                "train": self._train_recs,
                "val": self._val_recs,
                "test": self._test_recs,
            }[split]

            stride = self._stride_s if split == "train" else self._window_s

            self._datasets[split] = self._DatasetCls(
                bids_root=self._bids_root,
                window_s=self._window_s,
                stride_s=stride,
                overlap_threshold=self._overlap_threshold,
                montage=self._montage,
                label_mode=self._label_mode,
                normalize=self._normalize,
                recording_indices=rec_indices,
            )
        return self._datasets[split]

    def _make_loader(self, split: str) -> _LoaderAdapter:
        ds = self._get_dataset(split)
        # Embed-only path doesn't benefit from shuffle and it kills LRU
        # locality on lazy EDF dataios. Probes run on cached embeddings,
        # not on this loader.
        shuffle = False
        sampler = None

        if split == "train" and self._balance == "weighted_sampler" and len(ds) > 0:
            labels = ds.binary_labels()
            n_pos = max(int(labels.sum()), 1)
            n_neg = max(len(labels) - n_pos, 1)
            weights = np.where(labels == 1, 1.0 / n_pos, 1.0 / n_neg)
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(weights).double(),
                num_samples=len(ds),
                replacement=True,
            )
            shuffle = False

        loader = DataLoader(
            ds,
            batch_size=self._batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=False,
        )
        return _LoaderAdapter(loader, split_name=split)

    def train_dataloader(self):
        return self._make_loader("train")

    def val_dataloader(self):
        return self._make_loader("val")

    def test_dataloader(self):
        return self._make_loader("test")

    def datasets(self) -> Dict[str, Any]:
        for split in ("train", "val", "test"):
            self._get_dataset(split)
        return dict(self._datasets)
