"""BenchmarkDataModule adapter for the NMT Scalp EEG Dataset.

End-to-end direct-EDF pipeline — no preprocessing step, no HDF5 cache.
A fresh checkout that has only fetched the corpus can use this
dataset out of the box.

Two split modes:

- ``"official"`` — NMT ships its own train/eval split via the
  ``{abnormal,normal}/{train,eval}/`` layout.  We carve a subject-
  disjoint val fold out of train and use eval as the test set.
- ``"kfold"`` — pool both splits and run subject-stratified k-fold on
  the ``is_abnormal`` label.

Every NMT EDF is one recording per unique participant, so subject-
disjoint splits are also recording-disjoint.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from neuroatlas.extensions.datasets.adapters.tuab import (
    _stratified_subject_split,
    _subject_label_table,
    _train_val_subject_split,
)
from neuroatlas.extensions.datasets.dataio.nmt import (
    NMTEdfDataset,
    collate_nmt,
    discover_nmt_recordings,
)

from .base import BenchmarkDataModule
from neuroatlas.extensions.datasets.epilepsy._global_cache import RecordingWindowGlobalCache

logger = logging.getLogger(__name__)


_DEFAULT_RAW_ROOT = "${EEG_DATA_ROOT}/nmt"


# ---------------------------------------------------------------------------
# Loader adapter — wraps DataLoader batches into the benchmark contract.
# ---------------------------------------------------------------------------


# The shared wrapper; the local name is kept so call sites are unchanged.
from ._loader_adapters import SplitTaggingLoader as _LoaderAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class NMTBenchmarkDataModule(RecordingWindowGlobalCache, BenchmarkDataModule):
    """BenchmarkDataModule for the NMT Scalp EEG Dataset.

    Args:
        raw_root: Directory containing ``{abnormal,normal}/{train,eval}/*.edf``
            and (optionally) ``Labels.csv``.
        split_mode: ``"official"`` or ``"kfold"`` (default ``"official"``).
        fold: Current fold index — seeds the inner val carve-out in
            ``official`` mode, selects the test fold in ``kfold`` mode.
        n_folds: Total folds (default 5).
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        window_s: Window length in seconds (default 30).
        stride_s: Stride; ``None`` = non-overlapping (= ``window_s``).
        label_mode: Only ``"binary"`` supported.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        balance: ``"weighted_sampler"`` for class-balanced train loader
            (default), or ``"none"``.
        cache_max_recordings: LRU size for the EDF dataset.
        seed: RNG seed for subject-stratified splits (default 42).
    """

    def __init__(
        self,
        raw_root: str = _DEFAULT_RAW_ROOT,
        split_mode: str = "official",
        fold: int = 0,
        n_folds: int = 5,
        batch_size: int = 64,
        num_workers: int = 4,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        label_mode: str = "binary",
        normalize: str = "none",
        balance: str = "weighted_sampler",
        cache_max_recordings: int = 16,
        seed: int = 42,
        signal_kind: str = "raw",
        montage: str = "unipolar",
        **_unused: Any,
    ) -> None:
        if split_mode not in ("official", "kfold"):
            raise ValueError(
                f"split_mode must be 'official' or 'kfold', got {split_mode!r}"
            )
        if label_mode != "binary":
            raise ValueError(
                f"NMT only supports label_mode='binary', got {label_mode!r}"
            )

        metadata = {
            "canonical_label_space": "binary",
            "epoch_seconds": float(window_s),
            "channel_policy": ["eeg"],
            "signal_kind": signal_kind,
            "fold": fold,
            "n_folds": n_folds,
        }
        super().__init__(name="nmt", metadata=metadata)

        self._raw_root = raw_root
        self._split_mode = split_mode
        self._fold = fold
        self._n_folds = n_folds
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._window_s = float(window_s)
        self._stride_s = float(stride_s) if stride_s is not None else float(window_s)
        self._normalize = normalize
        self._balance = balance
        self._cache_max = int(cache_max_recordings)
        self._seed = seed
        if montage not in ("unipolar", "bipolar"):
            raise ValueError(
                f"montage must be 'unipolar' or 'bipolar', got {montage!r}"
            )
        self._montage = montage

        if not Path(raw_root).exists():
            raise FileNotFoundError(
                f"NMT raw_root not found: {raw_root}\n"
                f"Run `python -m neuroatlas.entrypoints.fetch --dataset nmt --download` (form-gated — see the "
                f"script's manual-action block) to stage the data, or point "
                f"`raw_root` at an existing copy of the NMT layout."
            )

        self._all_recordings: List[Dict[str, Any]] = discover_nmt_recordings(
            raw_root, splits=("train", "eval"),
        )
        if not self._all_recordings:
            raise RuntimeError(
                f"NMT raw_root {raw_root} is present but discovery found no EDFs. "
                f"Expected layout: {{abnormal,normal}}/{{train,eval}}/*.edf"
            )

        self._subject_ids: List[str] = [r["subject_id"] for r in self._all_recordings]
        self._is_abnormal: List[int] = [int(r["is_abnormal"]) for r in self._all_recordings]
        self._splits_per_rec: List[str] = [r["split"] for r in self._all_recordings]

        self._train_idx, self._val_idx, self._test_idx = self._compute_splits()
        # The names the global-cache mixin reads.
        self._train_recs, self._val_recs, self._test_recs = (
            self._train_idx, self._val_idx, self._test_idx)
        self._collate_fn = collate_nmt
        logger.info(
            "NMT splits (mode=%s, fold=%d/%d): train=%d val=%d test=%d recordings",
            split_mode, fold, n_folds,
            len(self._train_idx), len(self._val_idx), len(self._test_idx),
        )

        self._datasets: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def _compute_splits(self) -> Tuple[List[int], List[int], List[int]]:
        if self._split_mode == "official":
            train_recs = [i for i, s in enumerate(self._splits_per_rec) if s == "train"]
            eval_recs = [i for i, s in enumerate(self._splits_per_rec) if s == "eval"]
            if not train_recs:
                raise RuntimeError("NMT 'official' split: no recordings in train/.")
            if not eval_recs:
                logger.warning("NMT 'official' split: no recordings in eval/ — test set empty.")

            train_subj_ids = [self._subject_ids[i] for i in train_recs]
            train_abn = [self._is_abnormal[i] for i in train_recs]
            subjects, y = _subject_label_table(train_subj_ids, train_abn)
            tr_subj, va_subj = _train_val_subject_split(
                subjects, y,
                fold=self._fold, n_folds=self._n_folds, seed=self._seed,
            )
            tr_set, va_set = set(tr_subj), set(va_subj)
            train_idx = [i for i in train_recs if self._subject_ids[i] in tr_set]
            val_idx = [i for i in train_recs if self._subject_ids[i] in va_set]
            test_idx = eval_recs
            return train_idx, val_idx, test_idx

        # kfold
        subjects, y = _subject_label_table(self._subject_ids, self._is_abnormal)
        tr_subj, va_subj, te_subj = _stratified_subject_split(
            subjects, y,
            fold=self._fold, n_folds=self._n_folds, seed=self._seed,
        )
        tr_set, va_set, te_set = set(tr_subj), set(va_subj), set(te_subj)
        train_idx = [i for i, s in enumerate(self._subject_ids) if s in tr_set]
        val_idx = [i for i, s in enumerate(self._subject_ids) if s in va_set]
        test_idx = [i for i, s in enumerate(self._subject_ids) if s in te_set]
        return train_idx, val_idx, test_idx

    # ------------------------------------------------------------------
    # Dataset construction (per split)
    # ------------------------------------------------------------------

    def _get_dataset(self, split: str) -> Any:
        if split in self._datasets:
            return self._datasets[split]

        rec_indices = {
            "train": self._train_idx,
            "val": self._val_idx,
            "test": self._test_idx,
        }[split]

        if not rec_indices:
            raise RuntimeError(
                f"NMT split {split!r} has 0 recordings — check split_mode and fold index."
            )

        stride_s = self._stride_s if split == "train" else self._window_s
        ds = self._window_dataset(rec_indices, stride_s)
        self._datasets[split] = ds
        return ds

    def _window_dataset(self, rec_indices, stride_s, signal_cache_size: Optional[int] = None):
        return NMTEdfDataset(
            recordings=self._all_recordings,
            window_s=self._window_s,
            stride_s=stride_s,
            normalize=self._normalize,
            recording_indices=rec_indices,
            cache_max_recordings=self._cache_max if signal_cache_size is None else signal_cache_size,
            montage=self._montage,
        )

    # ------------------------------------------------------------------
    # Loader construction
    # ------------------------------------------------------------------

    def _make_loader(self, split: str) -> _LoaderAdapter:
        ds = self._get_dataset(split)
        # Embed-only path doesn't benefit from shuffle and it kills LRU
        # locality on lazy EDF dataios. Probes run on cached embeddings,
        # not on this loader.
        shuffle = False
        sampler = None

        if (
            split == "train"
            and self._balance == "weighted_sampler"
            and len(ds) > 0
        ):
            targets = ds.targets()
            unique, counts = np.unique(targets, return_counts=True)
            class_weight = {int(c): 1.0 / float(n) for c, n in zip(unique, counts)}
            weights = np.array(
                [class_weight[int(t)] for t in targets], dtype=np.float64,
            )
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
            collate_fn=collate_nmt,
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

    # ------------------------------------------------------------------
    # Global embedding cache (epilepsy/_global_cache.py): every window of
    # every recording once, folds assigned when probing. Seizure detection
    # used to cut this fold's train split instead, so fold 1 found nothing.
    # ------------------------------------------------------------------

    def _n_recordings(self) -> int:
        return len(self._all_recordings)

    def _subject_of(self, rec_index: int) -> str:
        return self._subject_ids[rec_index]
