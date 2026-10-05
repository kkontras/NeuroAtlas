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
from neuroatlas.benchmarking_helpers.registry.splits import make_subject_kfold
from torch.utils.data import DataLoader, WeightedRandomSampler

from .base import BenchmarkDataModule
from neuroatlas.extensions.datasets.epilepsy._global_cache import RecordingWindowGlobalCache

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
# The paper's actual folds (split_mode="paper_recordings")
# ---------------------------------------------------------------------------

#: ``patient`` -- the default, and what the paper states (App. B.1, C.1): patient-disjoint folds.
#: ``paper_recordings`` -- the folds the published Siena numbers were actually drawn on, frozen in
#: ``configs/cohorts/siena/folds_paper_recordings.json``. The paper's probe saw subject id "raw"
#: for every window, fell back to grouping by recording, and split 40 recordings, so most test
#: patients also have recordings in train or val. Offered only to reproduce the published numbers.
SPLIT_MODES = ("patient", "paper_recordings")


def paper_recordings_manifest_path() -> Path:
    from neuroatlas._paths import configs_dir

    return configs_dir("cohorts", "siena", "folds_paper_recordings.json")


def _paper_recording_splits(
    recording_ids: Sequence[str],
    fold: int,
    n_folds: int,
    manifest_path: Optional[Path] = None,
) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    """Recording-index splits of the paper's run, from the frozen manifest.

    Recordings the manifest does not list (the BIDS release has one the paper's
    embeddings lack) get no role, so they are in no split. A manifest recording
    missing from *recording_ids* is an error: the folds could not be the paper's.
    """
    import json

    path = Path(manifest_path) if manifest_path is not None else paper_recordings_manifest_path()
    manifest = json.loads(path.read_text())
    if int(manifest["n_folds"]) != int(n_folds):
        raise ValueError(
            f"split_mode=paper_recordings is the paper's {manifest['n_folds']}-fold split; "
            f"n_folds={n_folds} has no paper counterpart")
    folds = manifest["folds"][str(int(fold) % int(n_folds))]
    position = {str(r): i for i, r in enumerate(recording_ids)}
    listed = [r for s in ("train", "val", "test") for r in folds[s]]
    missing = sorted(r for r in listed if r not in position)
    if missing:
        raise FileNotFoundError(
            f"split_mode=paper_recordings: {len(missing)} recording(s) of the paper's folds are not "
            f"in this BIDS root (first: {missing[0]}); the paper's split cannot be rebuilt here")
    excluded = sorted(set(position) - set(listed))
    splits = [sorted(position[r] for r in folds[s]) for s in ("train", "val", "test")]
    # Under keys the embedding caches already treat as fold bookkeeping
    # (epilepsy/_global_cache._FOLD_KEYS): recorded with every result, never
    # part of the global cache key -- the split does not change which windows exist.
    info = {"fold_source": str(path),
            "fold_stats": {"split_mode": "paper_recordings", "excluded_recordings": excluded,
                           "test_patients": folds.get("test_patients")}}
    return splits[0], splits[1], splits[2], info


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class SienaBenchmarkDataModule(RecordingWindowGlobalCache, BenchmarkDataModule):
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
        montage: ``"unipolar"`` (19 electrodes, the default) or ``"bipolar"`` (the 20-pair TCP montage).
        label_mode: ``"binary"`` for seizure detection.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        balance: ``"weighted_sampler"`` or ``"none"``.
        overlap_threshold: Seizure fraction threshold for binary labels.
        split_mode: ``"patient"`` (default; patient-disjoint folds, as the
            paper states) or ``"paper_recordings"`` (the recording-level folds
            the published numbers were drawn on; see ``SPLIT_MODES``).
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
        montage: str = "unipolar",
        label_mode: str = "binary",
        normalize: str = "none",
        balance: str = "weighted_sampler",
        overlap_threshold: float = 0.0,
        signal_kind: str = "raw",
        split_mode: str = "patient",
        **kwargs,
    ) -> None:
        if split_mode not in SPLIT_MODES:
            raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {split_mode!r}")
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
        from neuroatlas.extensions.datasets.epilepsy.bids_index import (
            BIDSRecordingIndex,
            find_bids_root,
        )
        from neuroatlas.extensions.datasets.dataio.siena_bids import (
            SienaBIDSDataset,
            _collate_siena,
        )

        root = find_bids_root(Path(bids_root)) if bids_root else None
        if root is None:
            raise FileNotFoundError(
                f"No BIDS root found at {bids_root}. Download with: "
                "`neuroatlas data download siena`; if it is elsewhere, `neuroatlas config set siena.bids_root <path>`"
            )
        self._bids_root = str(root)
        self._bids_index = BIDSRecordingIndex.from_bids_root(root)
        self._DatasetCls = SienaBIDSDataset
        self._collate_fn = _collate_siena

        self._split_mode = split_mode
        if split_mode == "paper_recordings":
            (self._train_recs, self._val_recs, self._test_recs,
             fold_info) = _paper_recording_splits(
                [r.recording_id for r in self._bids_index.recordings], fold, n_folds)
            self.metadata.update(fold_info)
        else:
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
            self._datasets[split] = self._window_dataset(rec_indices, stride)
        return self._datasets[split]

    def _window_dataset(self, rec_indices, stride_s, signal_cache_size: int = 8):
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

    # -- global embedding cache (epilepsy/_global_cache.py) ----------------

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
