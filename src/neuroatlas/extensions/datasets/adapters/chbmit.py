"""BenchmarkDataModule adapter for the CHB-MIT Scalp EEG dataset.

Supports two backends:
- ``"bids"`` (default) — reads directly from BIDS EDF files, no preprocessing
- ``"hdf5"`` — reads from a pre-built continuous HDF5 cache

Both backends produce identical batch shapes and metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from neuroatlas.benchmarking_helpers.registry.splits import make_subject_kfold
from torch.utils.data import DataLoader, WeightedRandomSampler

from .base import BenchmarkDataModule
from neuroatlas.extensions.datasets.epilepsy._global_cache import RecordingWindowGlobalCache

_CACHE_ROOT_DEFAULT = "${REPO_ROOT}/chbmit_cache/hdf5"
_CACHE_FILENAME = "chbmit_256hz_continuous_bipolar18.h5"
_BIDS_ROOT_DEFAULT = "${REPO_ROOT}/chbmit_cache/raw/BIDS_CHB-MIT"


# ---------------------------------------------------------------------------
# Loader adapter
# ---------------------------------------------------------------------------


# The shared wrapper; the local name is kept so call sites are unchanged.
from ._loader_adapters import SplitTaggingLoader as _LoaderAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Generic patient-level split (works with both backends)
# ---------------------------------------------------------------------------


def _patient_splits_generic(
    subject_ids_per_rec: List[str],
    has_seizure_per_rec: List[bool],
    fold: int,
    n_folds: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Patient-disjoint stratified k-fold splits.

    Stratifies on per-subject seizure presence.

    Returns (train_rec_indices, val_rec_indices, test_rec_indices).
    """
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

    # Shared rule: benchmarking_helpers/splits. Previously this passed
    # n_folds straight to StratifiedKFold, which raises as soon as a class
    # has fewer members than folds -- so n_folds above the minority-class
    # size was an error rather than a split, and LOSO was unreachable.
    skf, _stratified, n_folds = make_subject_kfold(n_folds, strat_labels, seed=42)
    splits = list(skf.split(all_subjects, strat_labels))
    train_val_idx, test_idx = splits[fold % n_folds]

    test_subjects = {all_subjects[i] for i in test_idx}

    tv_subjects = [all_subjects[i] for i in train_val_idx]
    tv_strat = [strat_labels[i] for i in train_val_idx]

    inner_skf, _inner_strat, inner_folds = make_subject_kfold(
        n_folds, tv_strat, seed=42)
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


class CHBMITBenchmarkDataModule(RecordingWindowGlobalCache, BenchmarkDataModule):
    """BenchmarkDataModule for the CHB-MIT seizure detection dataset.

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
        montage: Montage type (CHB-MIT BIDS is already bipolar).
        label_mode: ``"binary"`` for seizure detection.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        balance: ``"weighted_sampler"`` or ``"none"``.
        overlap_threshold: Seizure fraction threshold for binary labels.
    """

    def __init__(
        self,
        backend: str = "bids",
        bids_root: Optional[str] = _BIDS_ROOT_DEFAULT,
        cache_root: str = _CACHE_ROOT_DEFAULT,
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
        folds_manifest: Optional[str] = "chbmit",
        strict_folds: bool = True,
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
            "folds_manifest": folds_manifest,
        }
        super().__init__(name="chbmit", metadata=metadata)

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
        self._folds_manifest = folds_manifest
        self._strict_folds = strict_folds

        # Backend-specific init
        if backend == "bids":
            from neuroatlas.extensions.datasets.epilepsy.bids_index import (
                BIDSRecordingIndex,
                find_bids_root,
            )
            from neuroatlas.extensions.datasets.dataio.chbmit_bids import (
                CHBMITBIDSDataset,
                _collate_chbmit,
            )

            root = find_bids_root(Path(bids_root)) if bids_root else None
            if root is None:
                raise FileNotFoundError(
                    f"No BIDS root found at {bids_root}. Download with: "
                    "`neuroatlas data download chbmit`; if it is elsewhere, `neuroatlas config set chbmit.bids_root <path>`"
                )
            self._bids_root = str(root)
            self._bids_index = BIDSRecordingIndex.from_bids_root(root)
            self._DatasetCls = CHBMITBIDSDataset
            self._collate_fn = _collate_chbmit

            # Compute splits from BIDS index
            self._train_recs, self._val_recs, self._test_recs = self._resolve_splits(
                self._bids_index.subject_ids_per_recording(),
                self._bids_index.seizure_presence_per_recording(),
                fold, n_folds,
            )

        elif backend == "hdf5":
            from neuroatlas.extensions.datasets.dataio.chbmit import (
                CHBMITContinuousDataset,
                _collate_chbmit,
            )

            self._h5_path = str(Path(cache_root) / _CACHE_FILENAME)
            self._DatasetCls = CHBMITContinuousDataset
            self._collate_fn = _collate_chbmit

            # Compute splits from HDF5
            import h5py
            with h5py.File(self._h5_path, "r") as f:
                subj_ids = [
                    s.decode() if isinstance(s, bytes) else str(s)
                    for s in f["subject_ids"][:]
                ]
                samplewise_label = f["samplewise_label"][:]
                offsets = f["recording_offsets"][:].astype(np.int64)

            has_seizure = []
            for i in range(len(subj_ids)):
                chunk = samplewise_label[offsets[i]:offsets[i + 1]]
                has_seizure.append(bool(np.any(chunk == 1)))

            self._train_recs, self._val_recs, self._test_recs = self._resolve_splits(
                subj_ids, has_seizure, fold, n_folds,
            )
        else:
            raise ValueError(f"Unknown backend {backend!r}. Expected 'bids' or 'hdf5'.")

        self._datasets: Dict[str, Any] = {}

    def _resolve_splits(
        self,
        subject_ids_per_rec: List[str],
        has_seizure_per_rec: List[bool],
        fold: int,
        n_folds: int,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Recording-index splits, from the frozen manifest when one is named.

        Falls back to the runtime ``StratifiedKFold`` in ``_patient_splits_generic``
        when ``folds_manifest`` is None, preserving the historical behaviour.
        """
        if self._folds_manifest is None:
            return _patient_splits_generic(
                subject_ids_per_rec, has_seizure_per_rec, fold, n_folds,
            )

        from neuroatlas.benchmarking_helpers.registry.fold_manifest import (
            load_fold_split,
            recording_indices_for_split,
        )

        split = load_fold_split(self._folds_manifest, fold)
        if split.n_folds != n_folds:
            raise ValueError(
                f"manifest {self._folds_manifest} has n_folds={split.n_folds} but the adapter "
                f"was given n_folds={n_folds}"
            )
        recs = recording_indices_for_split(
            subject_ids_per_rec, split, strict=self._strict_folds,
        )
        self.metadata["fold_source"] = str(split.source_path)
        self.metadata["fold_stats"] = split.stats
        return recs["train"], recs["val"], recs["test"]

    def _get_dataset(self, split: str):
        if split not in self._datasets:
            rec_indices = {
                "train": self._train_recs,
                "val": self._val_recs,
                "test": self._test_recs,
            }[split]

            stride = self._stride_s if split == "train" else self._window_s
            self._datasets[split] = self._window_dataset(rec_indices, stride)
        return self._datasets[split]

    def _window_dataset(self, rec_indices, stride_s, signal_cache_size: int = 8):
        if self._backend == "bids":
            return self._DatasetCls(
                bids_root=self._bids_root,
                window_s=self._window_s,
                stride_s=stride_s,
                overlap_threshold=self._overlap_threshold,
                montage=self._montage,
                label_mode=self._label_mode,
                normalize=self._normalize,
                recording_indices=rec_indices,
                signal_cache_size=signal_cache_size,
            )
        return self._DatasetCls(  # hdf5
            h5_path=self._h5_path,
            window_s=self._window_s,
            stride_s=stride_s,
            overlap_threshold=self._overlap_threshold,
            montage=self._montage,
            label_mode=self._label_mode,
            normalize=self._normalize,
            recording_indices=rec_indices,
        )

    # -- global embedding cache (epilepsy/_global_cache.py) ----------------

    def supports_global_embedding_cache(self) -> bool:
        # The hdf5 reader's per-window metadata has not been checked against
        # the fold split; that backend keeps per-split extraction.
        return self._backend == "bids" and super().supports_global_embedding_cache()

    def _n_recordings(self) -> int:
        return len(self._bids_index.recordings)

    def _subject_of(self, rec_index: int) -> str:
        return self._bids_index.recordings[rec_index].subject_id

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
