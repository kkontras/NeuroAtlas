"""BenchmarkDataModule adapter for the SeizeIt1 dataset.

Supports two backends:
- ``"edf"`` (default) — reads directly from raw EDF + ``_a1.tsv`` files
- ``"hdf5"`` — reads from a pre-built continuous HDF5 cache

Both backends produce identical batch shapes and metadata.

The EDF backend wraps ``discover_recordings`` in a try-catch so that the
pipeline can operate (via a synthetic fallback) when the private data path is
not available.  The HDF5 backend builds a synthetic cache automatically when
``cache_root`` is empty or the file does not exist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from neuroatlas.benchmarking_helpers.registry.splits import make_subject_kfold
from torch.utils.data import DataLoader, WeightedRandomSampler

from .base import BenchmarkDataModule
from neuroatlas.extensions.datasets.epilepsy._global_cache import RecordingWindowGlobalCache

_DATA_ROOT_DEFAULT: str = "${EEG_DATA_ROOT}/sz1"
_CACHE_ROOT_DEFAULT: str = ""
_CACHE_FILENAME = "sz1_256hz_continuous_unipolar19.h5"


# ---------------------------------------------------------------------------
# Loader adapter
# ---------------------------------------------------------------------------


# The shared wrapper; the local name is kept so call sites are unchanged.
from ._loader_adapters import SplitTaggingLoader as _LoaderAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Generic patient-level k-fold split
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


class SeizeIt1BenchmarkDataModule(RecordingWindowGlobalCache, BenchmarkDataModule):
    """BenchmarkDataModule for the SeizeIt1 seizure detection dataset.

    Args:
        backend: ``"edf"`` (default) or ``"hdf5"``.
        data_root: Path to raw EDF + ``_a1.tsv`` files.
        cache_root: Path to HDF5 cache directory (for ``backend="hdf5"``).
        fold: Current fold index (0-based).
        n_folds: Total number of folds.
        batch_size: Batch size for dataloaders.
        num_workers: DataLoader workers.
        window_s: Window duration in seconds (default 10 s).
        stride_s: Window stride in seconds (default: non-overlapping).
        montage: ``"unipolar"`` (19 ch) or ``"bipolar"`` (18 ch).
        label_mode: ``"binary"`` for seizure detection.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        balance: ``"weighted_sampler"`` or ``"none"``.
        overlap_threshold: Seizure fraction threshold for binary labels.
    """

    def __init__(
        self,
        backend: str = "edf",
        data_root: Optional[str] = _DATA_ROOT_DEFAULT,
        cache_root: str = _CACHE_ROOT_DEFAULT,
        fold: int = 0,
        n_folds: int = 5,
        batch_size: int = 64,
        num_workers: int = 4,
        window_s: float = 10.0,
        stride_s: Optional[float] = None,
        montage: str = "unipolar",
        label_mode: str = "binary",
        normalize: str = "none",
        balance: str = "weighted_sampler",
        overlap_threshold: float = 0.0,
        signal_kind: str = "raw",
        subject_allowlist: Optional[List[str]] = None,
        folds_manifest: Optional[str] = "sz1",
        strict_folds: bool = True,
        **kwargs,
    ) -> None:
        from neuroatlas.extensions.datasets.epilepsy._global_cache import (
            refuse_unrecorded_settings,
        )

        refuse_unrecorded_settings("sz1", type(self),
                                   normalize=normalize, label_mode=label_mode,
                                   overlap_threshold=overlap_threshold)
        metadata = {
            "canonical_label_space": ["bckg", "seiz"],
            "epoch_seconds": window_s,
            "channel_policy": ["eeg"],
            "signal_kind": signal_kind,
            "fold": fold,
            "n_folds": n_folds,
            "backend": backend,
            "subject_allowlist": sorted(subject_allowlist) if subject_allowlist is not None else None,
            "folds_manifest": folds_manifest,
        }
        super().__init__(name="sz1", metadata=metadata)

        self._backend = backend
        self._fold = fold
        self._n_folds = n_folds
        self._folds_manifest = folds_manifest
        # An allowlist runs on a chosen subset, so the manifest's other
        # subjects are absent by design rather than by a broken download.
        self._strict_folds = bool(strict_folds) and subject_allowlist is None
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._window_s = window_s
        self._stride_s = stride_s
        self._montage = montage
        self._label_mode = label_mode
        self._normalize = normalize
        self._balance = balance
        self._overlap_threshold = overlap_threshold

        if backend == "hdf5":
            from neuroatlas.extensions.datasets.dataio.sz1 import (
                SeizeIt1ContinuousDataset,
                _collate_sz1,
            )
            from neuroatlas.extensions.datasets.epilepsy.sz1_preprocessor import (
                _build_synthetic_h5_cache,
            )

            h5_path = Path(cache_root) / _CACHE_FILENAME if cache_root else None

            if h5_path is None or not h5_path.exists():
                synthetic_path = (
                    h5_path
                    if h5_path is not None
                    else Path("/tmp/sz1_synthetic_cache") / _CACHE_FILENAME
                )
                _build_synthetic_h5_cache(synthetic_path)
                h5_path = synthetic_path
                # Synthetic subjects are not the corpus's: no published fold
                # can apply, so the reader's own splitter runs.
                self._folds_manifest = None
                self.metadata["folds_manifest"] = None

            self._h5_path = str(h5_path)
            self._DatasetCls = SeizeIt1ContinuousDataset
            self._collate_fn = _collate_sz1

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
                chunk = samplewise_label[offsets[i]: offsets[i + 1]]
                has_seizure.append(bool(np.any(chunk == 1)))

            self._train_recs, self._val_recs, self._test_recs = (
                self._resolve_splits(subj_ids, has_seizure, fold, n_folds)
            )
            self._subject_ids_per_rec = list(subj_ids)

        elif backend == "edf":
            from neuroatlas.extensions.datasets.dataio.sz1_edf import (
                SeizeIt1EDFDataset,
                _collate_sz1_edf,
                _synthetic_recordings_list,
            )
            from neuroatlas.extensions.datasets.epilepsy.sz1_preprocessor import (
                discover_recordings,
            )

            recordings = discover_recordings(data_root or "")
            if not recordings:
                from neuroatlas.extensions.datasets.adapters.sz2 import _no_pairs

                raise FileNotFoundError(_no_pairs("sz1", data_root))

            if subject_allowlist is not None:
                allowed = set(subject_allowlist)
                recordings = [(e, t, s) for e, t, s in recordings if s in allowed]
                if not recordings:
                    raise ValueError(
                        f"No recordings found after filtering by subject_allowlist "
                        f"({len(allowed)} subjects listed, none matched)."
                    )

            self._recordings = recordings
            self._DatasetCls = SeizeIt1EDFDataset
            self._collate_fn = _collate_sz1_edf

            subj_ids = [r[2] for r in recordings]
            # Seizure presence is unknown without reading every TSV, so the
            # runtime splitter (folds_manifest=None only) sees one class.
            has_seizure = [True] * len(recordings)

            self._train_recs, self._val_recs, self._test_recs = (
                self._resolve_splits(subj_ids, has_seizure, fold, n_folds)
            )
            self._subject_ids_per_rec = list(subj_ids)

        else:
            raise ValueError(
                f"Unknown backend {backend!r}. Expected 'hdf5' or 'edf'."
            )

        self._datasets: Dict[str, Any] = {}

    def _resolve_splits(
        self,
        subject_ids_per_rec: List[str],
        has_seizure_per_rec: List[bool],
        fold: int,
        n_folds: int,
    ) -> Tuple[List[int], List[int], List[int]]:
        """The paper's folds, from ``configs/cohorts/sz1/folds.json``.

        The published SeizeIT1 folds were drawn by the paper's epilepsy probe
        over 60 subjects stratified on "has any seizure window" (42 do, 18 do
        not). The EDF reader cannot know that without reading every
        annotation, so its own splitter saw one class and drew other folds
        (10-12 of every fold's 12 test subjects differed). The manifest fixes
        them. ``folds_manifest=None`` restores the runtime splitter, for a
        fold count the manifest does not have.
        """
        if self._folds_manifest is None:
            return _patient_splits_generic(
                subject_ids_per_rec, has_seizure_per_rec, fold, n_folds)

        from neuroatlas.benchmarking_helpers.registry.fold_manifest import (
            fold_source_label,
            recording_splits_from_manifest,
        )

        train, val, test, split = recording_splits_from_manifest(
            self._folds_manifest, fold, n_folds, subject_ids_per_rec,
            strict=self._strict_folds,
        )
        self.metadata["fold_source"] = fold_source_label(split)
        return train, val, test

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

    def _window_dataset(self, rec_indices, stride_s, signal_cache_size: Optional[int] = None):
        common = dict(
            window_s=self._window_s,
            stride_s=stride_s,
            overlap_threshold=self._overlap_threshold,
            montage=self._montage,
            label_mode=self._label_mode,
            normalize=self._normalize,
            recording_indices=rec_indices,
        )
        if self._backend == "hdf5":
            return self._DatasetCls(h5_path=self._h5_path, **common)
        if signal_cache_size is not None:
            common["signal_cache_size"] = signal_cache_size
        return self._DatasetCls(recordings=self._recordings, **common)

    # -- global embedding cache (epilepsy/_global_cache.py) ----------------
    # Every window of every recording once, folds assigned when probing;
    # the per-split caches held one fold's resampled train split, so fold 1
    # found nothing to read.

    def _n_recordings(self) -> int:
        return len(self._subject_ids_per_rec)

    def _subject_of(self, rec_index: int) -> str:
        return self._subject_ids_per_rec[rec_index]

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
