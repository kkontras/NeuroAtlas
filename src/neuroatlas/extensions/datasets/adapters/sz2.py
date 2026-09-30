"""BenchmarkDataModule adapter for the SeizeIt2 dataset.

Supports two backends:
- ``"hdf5"`` (default) — reads from a pre-built continuous HDF5 cache
- ``"edf"`` — reads directly from raw EDF + TSV files, no preprocessing

Both backends produce identical batch shapes and metadata.

Because the data path is private and may not yet be set, the ``"edf"``
backend wraps ``discover_recordings`` in a try-catch (row 3 in
``POINTER_SZ2_TRYCATCH.md``).  The ``"hdf5"`` backend falls back to a
synthetic HDF5 when ``cache_root`` is empty or the file does not exist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from neuroatlas.benchmarking_helpers.registry.splits import make_subject_kfold
from torch.utils.data import DataLoader, WeightedRandomSampler

from .base import BenchmarkDataModule

# ---------------------------------------------------------------------------
# POINTER_SZ2_PATH.md rows 4-5: path placeholders — fill in before use
# ---------------------------------------------------------------------------

_DATA_ROOT_DEFAULT: str = "${EEG_DATA_ROOT}/SeizeIT2/Anonymous_Hospital_Adult/"    # fill in SeizeIt2 EDF root directory
_CACHE_ROOT_DEFAULT: str = ""   # fill in HDF5 cache directory
_CACHE_FILENAME = "sz2_256hz_continuous_unipolar19.h5"


# ---------------------------------------------------------------------------
# Loader adapter
# ---------------------------------------------------------------------------


# The shared wrapper; the local name is kept so call sites are unchanged.
from ._loader_adapters import SplitTaggingLoader as _LoaderAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Generic patient-level k-fold split (copied from adapters/siena.py)
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


class SeizeIt2BenchmarkDataModule(BenchmarkDataModule):
    """BenchmarkDataModule for the SeizeIt2 seizure detection dataset.

    Args:
        backend: ``"hdf5"`` (default) or ``"edf"``.
        data_root: Path to raw EDF + TSV files (for ``backend="edf"`` and for
            building the HDF5 cache with the preprocessor).
        cache_root: Path to HDF5 cache directory (for ``backend="hdf5"``).
        fold: Current fold index (0-based).
        n_folds: Total number of folds.
        batch_size: Batch size for dataloaders.
        num_workers: DataLoader workers.
        window_s: Window duration in seconds.
        stride_s: Window stride in seconds (default: non-overlapping).
        montage: ``"unipolar"`` (19 ch) or ``"bipolar"`` (18 ch).
        label_mode: ``"binary"`` for seizure detection.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        balance: ``"weighted_sampler"`` or ``"none"``.
        overlap_threshold: Seizure fraction threshold for binary labels.
    """

    def __init__(
        self,
        backend: str = "hdf5",
        data_root: Optional[str] = _DATA_ROOT_DEFAULT,
        cache_root: str = _CACHE_ROOT_DEFAULT,
        fold: int = 0,
        n_folds: int = 5,
        batch_size: int = 64,
        num_workers: int = 4,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        montage: str = "unipolar",
        label_mode: str = "binary",
        normalize: str = "none",
        balance: str = "weighted_sampler",
        overlap_threshold: float = 0.0,
        signal_kind: str = "raw",
        subject_allowlist: Optional[List[str]] = None,
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
            "subject_allowlist": sorted(subject_allowlist) if subject_allowlist is not None else None,
        }
        super().__init__(name="sz2", metadata=metadata)

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

        if backend == "hdf5":
            from neuroatlas.extensions.datasets.dataio.sz2 import (
                SeizeIt2ContinuousDataset,
                _collate_sz2,
            )
            from neuroatlas.extensions.datasets.epilepsy.sz2_preprocessor import (
                _build_synthetic_h5_cache,
            )

            h5_path = Path(cache_root) / _CACHE_FILENAME if cache_root else None

            if h5_path is None or not h5_path.exists():
                # Build a synthetic cache so the rest of the pipeline works
                synthetic_path = (
                    h5_path
                    if h5_path is not None
                    else Path("/tmp/sz2_synthetic_cache") / _CACHE_FILENAME
                )
                _build_synthetic_h5_cache(synthetic_path)
                h5_path = synthetic_path

            self._h5_path = str(h5_path)
            self._DatasetCls = SeizeIt2ContinuousDataset
            self._collate_fn = _collate_sz2

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
                chunk = samplewise_label[offsets[i] : offsets[i + 1]]
                has_seizure.append(bool(np.any(chunk == 1)))

            self._train_recs, self._val_recs, self._test_recs = (
                _patient_splits_generic(subj_ids, has_seizure, fold, n_folds)
            )

        elif backend == "edf":
            from neuroatlas.extensions.datasets.dataio.sz2_edf import (
                SeizeIt2EDFDataset,
                _collate_sz2_edf,
                _synthetic_recordings_list,
            )
            from neuroatlas.extensions.datasets.epilepsy.sz2_preprocessor import (
                discover_recordings,
            )

            # POINTER_SZ2_TRYCATCH.md row 3: try-catch around discover_recordings
            try:
                recordings = discover_recordings(data_root or "")
                if not recordings:
                    raise FileNotFoundError(
                        f"No EDF+TSV pairs found under {data_root!r}. "
                        "Set _DATA_ROOT_DEFAULT in adapters/sz2.py."
                    )
            except Exception as exc:
                raise Exception("Data path not set or unreadable") from exc

            if subject_allowlist is not None:
                allowed = set(subject_allowlist)
                recordings = [(e, t, s) for e, t, s in recordings if s in allowed]
                if not recordings:
                    raise ValueError(
                        f"No recordings found after filtering by subject_allowlist "
                        f"({len(allowed)} subjects listed, none matched)."
                    )

            self._recordings = recordings
            self._DatasetCls = SeizeIt2EDFDataset
            self._collate_fn = _collate_sz2_edf

            subj_ids = [r[2] for r in recordings]
            # We cannot know seizure presence without reading files, so assume all have seizures
            has_seizure = [True] * len(recordings)

            self._train_recs, self._val_recs, self._test_recs = (
                _patient_splits_generic(subj_ids, has_seizure, fold, n_folds)
            )

        else:
            raise ValueError(
                f"Unknown backend {backend!r}. Expected 'hdf5' or 'edf'."
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

            if self._backend == "hdf5":
                self._datasets[split] = self._DatasetCls(
                    h5_path=self._h5_path,
                    window_s=self._window_s,
                    stride_s=stride,
                    overlap_threshold=self._overlap_threshold,
                    montage=self._montage,
                    label_mode=self._label_mode,
                    normalize=self._normalize,
                    recording_indices=rec_indices,
                )
            else:
                self._datasets[split] = self._DatasetCls(
                    recordings=self._recordings,
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
