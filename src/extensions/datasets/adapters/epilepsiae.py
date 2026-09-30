"""BenchmarkDataModule adapter for the EPILEPSIAE dataset.

Reads raw .data/.head files on the fly — no HDF5 cache needed.
Wraps :class:`EpilepsiAEContinuousDataset` with patient-disjoint
stratified k-fold splits, class-balanced sampling, and the standard
``{signals, label, meta, raw_batch}`` batch format.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from benchmarking_helpers.registry.splits import make_subject_kfold
from torch.utils.data import DataLoader, WeightedRandomSampler

from extensions.datasets.epilepsy.epilepsiae_preprocessor import (
    DATA_ROOT,
    EpilepsiAEPreprocessor,
    PatientMeta,
)
from extensions.datasets.dataio.epilepsiae import (
    EpilepsiAEContinuousDataset,
)

from .base import BenchmarkDataModule

logger = logging.getLogger(__name__)

_DATA_ROOT_DEFAULT = str(DATA_ROOT)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def _epilepsiae_collate(batch):
    """Stack signals and labels; keep per-sample metadata as a list."""
    eeg = torch.stack([item["eeg"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    meta = [item["meta"] for item in batch]
    return {"eeg": eeg, "labels": labels, "meta": meta}


# ---------------------------------------------------------------------------
# Loader adapter
# ---------------------------------------------------------------------------


# The shared wrapper; the local name is kept so call sites are unchanged.
from ._loader_adapters import SplitTaggingLoader as _LoaderAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Split logic (operates on in-memory recording list, no HDF5)
# ---------------------------------------------------------------------------


def _patient_splits_from_recordings(
    recordings: List[Any],
    fold: int,
    n_folds: int,
) -> tuple[List[int], List[int], List[int]]:
    """Patient-disjoint stratified k-fold splits from recording metadata.

    ``recordings`` is the ``_recordings`` list from the dataset — each
    element has ``.subject_id``, ``.variant``, and ``.samplewise_label``.

    Returns (train_rec_indices, val_rec_indices, test_rec_indices).
    """
    n_rec = len(recordings)

    # Build per-subject info
    subject_to_recs: Dict[str, List[int]] = {}
    subject_to_variant: Dict[str, str] = {}
    subject_seizure_samples: Dict[str, int] = {}

    for i, rec in enumerate(recordings):
        sid = rec.subject_id
        subject_to_recs.setdefault(sid, []).append(i)
        subject_to_variant[sid] = rec.variant
        n_seiz = int((rec.samplewise_label == 1).sum())
        subject_seizure_samples[sid] = subject_seizure_samples.get(sid, 0) + n_seiz

    all_subjects = sorted(subject_to_recs.keys())

    def _strat_key(sid: str) -> str:
        v = subject_to_variant[sid]
        n = subject_seizure_samples.get(sid, 0)
        if n == 0:
            bin_label = "0"
        elif n < 1000:
            bin_label = "low"
        elif n < 10000:
            bin_label = "mid"
        else:
            bin_label = "high"
        return f"{v}_{bin_label}"

    strat_labels = [_strat_key(s) for s in all_subjects]

    # Merge rare strata
    counts = Counter(strat_labels)
    for i, s in enumerate(strat_labels):
        if counts[s] < n_folds:
            strat_labels[i] = subject_to_variant[all_subjects[i]]

    # Outer fold -> test set
    # Shared rule: benchmarking_helpers/splits. Previously this passed
    # n_folds straight to StratifiedKFold, which raises as soon as a class
    # has fewer members than folds -- so n_folds above the minority-class
    # size was an error rather than a split, and LOSO was unreachable.
    skf, _stratified, n_folds = make_subject_kfold(n_folds, strat_labels, seed=42)
    splits = list(skf.split(all_subjects, strat_labels))
    train_val_idx, test_idx = splits[fold % n_folds]

    test_subjects = {all_subjects[i] for i in test_idx}

    # Inner fold -> val set
    tv_subjects = [all_subjects[i] for i in train_val_idx]
    tv_strat = [strat_labels[i] for i in train_val_idx]

    inner_counts = Counter(tv_strat)
    for i, s in enumerate(tv_strat):
        if inner_counts[s] < n_folds:
            tv_strat[i] = subject_to_variant[tv_subjects[i]]

    inner_skf, _inner_strat, inner_folds = make_subject_kfold(
        n_folds, tv_strat, seed=42)
    inner_splits = list(inner_skf.split(tv_subjects, tv_strat))
    val_fold = (fold + 1) % inner_folds
    inner_train_idx, inner_val_idx = inner_splits[val_fold % len(inner_splits)]

    train_subjects = {tv_subjects[i] for i in inner_train_idx}
    val_subjects = {tv_subjects[i] for i in inner_val_idx}

    # Map subjects -> recording indices
    subject_ids = [rec.subject_id for rec in recordings]
    train_recs = [i for i in range(n_rec) if subject_ids[i] in train_subjects]
    val_recs = [i for i in range(n_rec) if subject_ids[i] in val_subjects]
    test_recs = [i for i in range(n_rec) if subject_ids[i] in test_subjects]

    return train_recs, val_recs, test_recs


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class EpilepsiAEBenchmarkDataModule(BenchmarkDataModule):
    """BenchmarkDataModule for the EPILEPSIAE multi-center seizure dataset.

    Reads raw .data/.head files on the fly — no preprocessing or HDF5
    cache required.  Patient-level stratified k-fold cross-validation.

    Args:
        data_root: Root of the raw EPILEPSIAE corpus.
        fold: Current fold index (0-based).
        n_folds: Total number of folds (default 5).
        batch_size: Batch size for dataloaders.
        num_workers: DataLoader workers.
        window_s: Window duration in seconds.
        stride_s: Window stride in seconds (default: same as window_s).
        montage: ``"unipolar"`` (19 ch) or ``"bipolar"`` (18 ch).
        label_mode: ``"binary"``, ``"multiclass"``, or ``"pattern"``.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        balance: ``"weighted_sampler"`` or ``"none"``.
        overlap_threshold: Seizure fraction threshold for binary labels.
        signal_kind: Ignored (present for config compatibility).
    """

    def __init__(
        self,
        data_root: str = _DATA_ROOT_DEFAULT,
        fold: int = 0,
        n_folds: int = 5,
        batch_size: int = 64,
        num_workers: int = 0,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        montage: str = "bipolar",
        label_mode: str = "binary",
        normalize: str = "none",
        balance: str = "weighted_sampler",
        overlap_threshold: float = 0.0,
        signal_kind: str = "raw",
        variants: Sequence[str] = ("surf30", "surfPA", "surfCO"),
        **kwargs,
    ) -> None:
        metadata = {
            "canonical_label_space": "seizure_detection" if label_mode == "binary" else label_mode,
            "epoch_seconds": window_s,
            "channel_policy": montage,
            "signal_kind": signal_kind,
            "fold": fold,
            "n_folds": n_folds,
        }
        super().__init__(name="epilepsiae", metadata=metadata)

        self._data_root = data_root
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
        self._variants = tuple(variants)

        # Discover patients once — shared across all splits
        logger.info("Discovering EPILEPSIAE patients from %s ...", data_root)
        preprocessor = EpilepsiAEPreprocessor(data_root=Path(data_root))
        self._patients = preprocessor.discover_patients(variants=variants)
        self._annotations = preprocessor.annotations
        self._origin_map = preprocessor.origin_map
        logger.info("Found %d patients", len(self._patients))

        # Build a full dataset (all recordings) to compute splits from
        # its recording metadata.  No signal data is loaded here.
        self._full_dataset = EpilepsiAEContinuousDataset(
            data_root=data_root,
            patients=self._patients,
            annotations=self._annotations,
            origin_map=self._origin_map,
            window_s=window_s,
            stride_s=stride_s,
            montage=montage,
            label_mode=label_mode,
            normalize=normalize,
            overlap_threshold=overlap_threshold,
            variants=variants,
        )

        # Compute splits from the recording list
        self._train_recs, self._val_recs, self._test_recs = (
            _patient_splits_from_recordings(
                self._full_dataset._recordings, fold, n_folds,
            )
        )

        self._datasets: Dict[str, EpilepsiAEContinuousDataset] = {}

    def _get_dataset(self, split: str) -> EpilepsiAEContinuousDataset:
        if split not in self._datasets:
            rec_indices = {
                "train": self._train_recs,
                "val": self._val_recs,
                "test": self._test_recs,
            }[split]

            stride = self._stride_s if split == "train" else self._window_s

            self._datasets[split] = EpilepsiAEContinuousDataset(
                data_root=self._data_root,
                patients=self._patients,
                annotations=self._annotations,
                origin_map=self._origin_map,
                window_s=self._window_s,
                stride_s=stride,
                overlap_threshold=self._overlap_threshold,
                montage=self._montage,
                label_mode=self._label_mode,
                normalize=self._normalize,
                recording_indices=rec_indices,
                variants=self._variants,
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
            collate_fn=_epilepsiae_collate,
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

    def datasets(self) -> Dict[str, EpilepsiAEContinuousDataset]:
        """Return all three split datasets (creating them if needed)."""
        for split in ("train", "val", "test"):
            self._get_dataset(split)
        return dict(self._datasets)
