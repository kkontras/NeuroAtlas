"""BenchmarkDataModule adapter for the Helsinki Neonatal EEG Dataset.

Lazy EDF reader (``backend="edf"``) + per-expert annotation metadata.
A fresh checkout that only fetched the corpus can
use this backend directly. Per-expert per-window binary labels + seizure
fractions are always exposed in ``meta`` for inter-rater studies.

The legacy HDF5-cache backend has been removed (its dataio module
``dataio/helsinki_neonatal_h5.py`` was deleted).

Patient-level stratified k-fold on seizure presence. One EDF per neonate,
so subject-disjoint == recording-disjoint.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from neuroatlas.benchmarking_helpers.registry.splits import make_subject_kfold
from torch.utils.data import DataLoader, WeightedRandomSampler

from .base import BenchmarkDataModule
from neuroatlas.extensions.datasets.epilepsy._global_cache import RecordingWindowGlobalCache

logger = logging.getLogger(__name__)

_DEFAULT_RAW_ROOT = "${EEG_DATA_ROOT}/helsinki_neonatal"


# ---------------------------------------------------------------------------
# Loader adapter
# ---------------------------------------------------------------------------


# The shared wrapper; the local name is kept so call sites are unchanged.
from ._loader_adapters import SplitTaggingLoader as _LoaderAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Patient-level stratified k-fold
# ---------------------------------------------------------------------------


def _patient_splits(
    subject_ids_per_rec: Sequence[str],
    has_seizure_per_rec: Sequence[bool],
    fold: int,
    n_folds: int,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Patient-disjoint stratified k-fold splits on per-subject seizure presence."""
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
    strat = ["seiz" if subject_has_seizure[s] else "none" for s in all_subjects]

    # Shared rule: benchmarking_helpers/splits. The local _safe_n_splits it
    # replaces reduced the fold count instead, which silently ran fewer folds
    # than asked for; this keeps them and drops stratification instead.
    skf, _stratified, n_eff = make_subject_kfold(n_folds, strat, seed=seed)
    splits = list(skf.split(all_subjects, strat))
    tv_idx, te_idx = splits[fold % n_eff]
    test_subjects = {all_subjects[i] for i in te_idx}

    tv_subjects = [all_subjects[i] for i in tv_idx]
    tv_strat = [strat[i] for i in tv_idx]

    inner, _inner_strat, inner_n = make_subject_kfold(n_folds, tv_strat, seed=seed)
    inner_splits = list(inner.split(tv_subjects, tv_strat))
    val_fold = (fold + 1) % len(inner_splits)
    inner_tr, inner_va = inner_splits[val_fold]
    train_subjects = {tv_subjects[i] for i in inner_tr}
    val_subjects = {tv_subjects[i] for i in inner_va}

    train_recs = [i for i in range(n_rec) if subject_ids_per_rec[i] in train_subjects]
    val_recs = [i for i in range(n_rec) if subject_ids_per_rec[i] in val_subjects]
    test_recs = [i for i in range(n_rec) if subject_ids_per_rec[i] in test_subjects]
    return train_recs, val_recs, test_recs


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class HelsinkiNeonatalBenchmarkDataModule(RecordingWindowGlobalCache, BenchmarkDataModule):
    """BenchmarkDataModule for the Helsinki Neonatal Seizure Dataset.

    Args:
        backend: ``"edf"`` (default) or ``"hdf5"``.
        raw_root: Directory containing ``eegN.edf`` + ``annotations_2017_{A,B,C}.mat``
            (for ``backend="edf"``).
        fold, n_folds: Patient-level stratified k-fold index + count.
        batch_size, num_workers: DataLoader settings.
        window_s, stride_s: Window length / stride (seconds).
        montage: ``"unipolar"`` (19 ch, default) or ``"bipolar"`` (18 ch TCP).
        consensus: Inter-rater collapse mode — ``"majority"`` (≥2/3, default),
            ``"unanimous"`` (3/3), ``"any"`` (≥1/3).  Applies to both backends,
            but the HDF5 backend can only apply the mode that was baked at
            cache-build time (checked via the ``consensus`` HDF5 attr).
        label_mode: Only ``"binary"`` supported.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        balance: ``"weighted_sampler"`` (default) or ``"none"``.
        overlap_threshold: Seizure fraction above which binary label is 1.
        include_partial_subjects: Keep Stevenson's 18 mixed-agreement subjects.
            Default ``True`` (use all 79).  Set ``False`` for the classic
            39 + 22 = 61-subject unanimous subset.
        signal_kind: Echoed into metadata (default ``"raw"``).
        seed: RNG seed for the stratified split (default 42).
    """

    def __init__(
        self,
        backend: str = "edf",
        raw_root: str = _DEFAULT_RAW_ROOT,
        fold: int = 0,
        n_folds: int = 5,
        batch_size: int = 64,
        num_workers: int = 4,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        montage: str = "unipolar",
        consensus: str = "majority",
        label_mode: str = "binary",
        normalize: str = "none",
        balance: str = "weighted_sampler",
        overlap_threshold: float = 0.0,
        include_partial_subjects: bool = True,
        signal_kind: str = "raw",
        seed: int = 42,
        **_unused: Any,
    ) -> None:
        from neuroatlas.extensions.datasets.epilepsy._global_cache import (
            refuse_unrecorded_settings,
        )

        refuse_unrecorded_settings("helsinki_neonatal", type(self),
                                   normalize=normalize, label_mode=label_mode,
                                   overlap_threshold=overlap_threshold)
        if backend != "edf":
            raise ValueError(
                f"helsinki_neonatal reads only its EDF files (backend=edf), not "
                f"backend={backend!r}"
            )
        if label_mode != "binary":
            raise ValueError(
                f"HelsinkiNeonatal only supports label_mode='binary', got {label_mode!r}"
            )

        metadata = {
            "canonical_label_space": ["bckg", "seiz"],
            "epoch_seconds": float(window_s),
            "channel_policy": ["eeg"],
            "signal_kind": signal_kind,
            "fold": fold,
            "n_folds": n_folds,
            "consensus": consensus,
            "backend": backend,
        }
        super().__init__(name="helsinki_neonatal", metadata=metadata)

        self._backend = backend
        self._fold = fold
        self._n_folds = n_folds
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._window_s = float(window_s)
        self._stride_s = float(stride_s) if stride_s is not None else float(window_s)
        self._montage = montage
        self._consensus = consensus
        self._normalize = normalize
        self._balance = balance
        self._overlap_threshold = float(overlap_threshold)
        self._include_partial = bool(include_partial_subjects)
        self._raw_root = raw_root
        self._seed = int(seed)

        if not Path(raw_root).exists():
            from neuroatlas.extensions.datasets._missing import no_data

            raise FileNotFoundError(no_data("helsinki_neonatal", raw_root, "raw_root"))
        self._prepare_edf_backend()

        self._datasets: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Backend preparation — builds the template Dataset once to extract
    # subject ids and seizure presence, so we can k-fold without re-reading.
    # ------------------------------------------------------------------

    def _prepare_edf_backend(self) -> None:
        from neuroatlas.extensions.datasets.dataio.helsinki_neonatal import (
            HelsinkiNeonatalEdfDataset,
            _collate_helsinki,
        )

        template = HelsinkiNeonatalEdfDataset(
            raw_root=self._raw_root,
            window_s=self._window_s,
            stride_s=self._window_s,  # stride=window for a cheap probe only
            montage=self._montage,
            normalize=self._normalize,
            consensus=self._consensus,
            overlap_threshold=self._overlap_threshold,
            signal_cache_size=1,
            include_partial_subjects=self._include_partial,
        )
        self._template_dataset = template
        self._subject_ids_per_rec = template.subject_ids_per_recording()
        self._has_seizure_per_rec = template.seizure_presence_per_recording()
        self._DatasetCls = HelsinkiNeonatalEdfDataset
        self._collate_fn = _collate_helsinki

        self._train_recs, self._val_recs, self._test_recs = _patient_splits(
            self._subject_ids_per_rec, self._has_seizure_per_rec,
            self._fold, self._n_folds, seed=self._seed,
        )
        logger.info(
            "Helsinki splits (edf backend, fold=%d/%d): train=%d val=%d test=%d recordings",
            self._fold, self._n_folds,
            len(self._train_recs), len(self._val_recs), len(self._test_recs),
        )

    # ------------------------------------------------------------------
    # Per-split dataset construction
    # ------------------------------------------------------------------

    def _get_dataset(self, split: str) -> Any:
        if split in self._datasets:
            return self._datasets[split]

        rec_indices = {
            "train": self._train_recs,
            "val": self._val_recs,
            "test": self._test_recs,
        }[split]

        if not rec_indices:
            raise RuntimeError(
                f"helsinki_neonatal fold {self._fold}: its {split} split has no "
                f"recordings (n_folds={self._n_folds})"
            )

        stride_s = self._stride_s if split == "train" else self._window_s
        ds = self._window_dataset(rec_indices, stride_s)
        self._datasets[split] = ds
        return ds

    def _window_dataset(self, rec_indices, stride_s, signal_cache_size: int = 8):
        return self._DatasetCls(
            raw_root=self._raw_root,
            window_s=self._window_s,
            stride_s=stride_s,
            recording_indices=rec_indices,
            montage=self._montage,
            normalize=self._normalize,
            consensus=self._consensus,
            overlap_threshold=self._overlap_threshold,
            include_partial_subjects=self._include_partial,
            signal_cache_size=signal_cache_size,
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

    # ------------------------------------------------------------------
    # Global embedding cache (epilepsy/_global_cache.py): every window of
    # every recording once, folds assigned when probing. The per-split
    # caches it replaces for seizure detection held a weighted resample of
    # one fold's train split, so fold 1 found nothing to read.
    # ------------------------------------------------------------------

    def _n_recordings(self) -> int:
        return len(self._subject_ids_per_rec)

    def _subject_of(self, rec_index: int) -> str:
        return self._subject_ids_per_rec[rec_index]
