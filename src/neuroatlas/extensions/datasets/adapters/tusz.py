"""BenchmarkDataModule adapter for the TUSZ (TUH EEG Seizure) dataset.

Supports two split modes:

- ``"official"`` — uses three per-split HDF5 caches (train / dev / eval).
- ``"kfold"``    — uses one merged all-splits HDF5; each patient takes its
  role from the paper's frozen folds (``configs/cohorts/tusz/folds.json``),
  or, with ``folds_manifest=None``, from patient-level StratifiedKFold.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from neuroatlas.benchmarking_helpers.registry.splits import make_subject_kfold

from .base import BenchmarkDataModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patient-level k-fold helpers
# ---------------------------------------------------------------------------


def _all_fold_assignments(
    subject_ids: List[str],
    recording_has_seizure: np.ndarray,
    n_folds: int = 5,
) -> Dict[str, Dict[int, str]]:
    """Compute patient-level StratifiedKFold assignments for all folds.

    Returns ``{subject_id: {fold_idx: "train"|"val"|"test"}}``.
    """
    # Aggregate seizure presence to patient level
    patient_seizure: Dict[str, int] = {}
    for subj, has_sz in zip(subject_ids, recording_has_seizure):
        patient_seizure.setdefault(subj, 0)
        if has_sz:
            patient_seizure[subj] = 1

    patients = sorted(patient_seizure.keys())
    y_patient = np.array([patient_seizure[p] for p in patients])

    assignments: Dict[str, Dict[int, str]] = {p: {} for p in patients}

    for fold in range(n_folds):
        train_p, val_p, test_p = _patient_splits(patients, y_patient, fold, n_folds)
        for p in train_p:
            assignments[p][fold] = "train"
        for p in val_p:
            assignments[p][fold] = "val"
        for p in test_p:
            assignments[p][fold] = "test"

    return assignments


def _patient_splits(
    patients: List[str],
    y_patient: np.ndarray,
    fold: int,
    n_folds: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Return ``(train, val, test)`` patient lists for one fold.

    Mirrors the Parkinson adapter pattern:
    - outer fold defines the test set
    - inner split of the remainder gives train / val
    """
    # Shared rule: benchmarking_helpers/splits. Previously this passed
    # n_folds straight to StratifiedKFold, which raises as soon as a class
    # has fewer members than folds -- so n_folds above the minority-class
    # size was an error rather than a split, and LOSO was unreachable.
    skf, _stratified, n_folds = make_subject_kfold(n_folds, y_patient, seed=42)
    splits = list(skf.split(patients, y_patient))

    train_val_idx, test_idx = splits[fold % n_folds]
    test_patients = [patients[i] for i in test_idx]

    train_val_patients = [patients[i] for i in train_val_idx]
    train_val_y = y_patient[train_val_idx]

    inner_skf, _inner_strat, inner_folds = make_subject_kfold(
        n_folds, train_val_y, seed=42)
    val_fold = (fold + 1) % inner_folds
    inner_splits = list(inner_skf.split(train_val_patients, train_val_y))
    inner_train_idx, inner_val_idx = inner_splits[val_fold % len(inner_splits)]
    train_patients = [train_val_patients[i] for i in inner_train_idx]
    val_patients = [train_val_patients[i] for i in inner_val_idx]

    return train_patients, val_patients, test_patients


# ---------------------------------------------------------------------------
# Loader adapter
# ---------------------------------------------------------------------------


class _LoaderAdapter:
    """Wraps DataLoader output into the standard benchmark batch shape."""

    def __init__(self, loader, dataset_name: str = "tusz") -> None:
        self._loader = loader
        self._dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for batch in self._loader:
            yield batch  # already in standard format from _collate_tusz


# ---------------------------------------------------------------------------
# BenchmarkDataModule
# ---------------------------------------------------------------------------


class TUSZDataModule(BenchmarkDataModule):
    """BenchmarkDataModule for TUSZ seizure detection.

    Args:
        cache_root: Directory containing TUSZ HDF5 caches.
        split_mode: ``"official"`` or ``"kfold"``.
        fold: Fold index for k-fold mode.
        n_folds: Total folds for k-fold mode.
        window_s: Window size in seconds.
        stride_s: Stride in seconds (defaults to ``window_s``).
        montage: ``"unipolar"`` or ``"bipolar"``.
        label_mode: Label mode for the dataset.
        normalize: Normalization mode.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        balance: ``"none"`` or ``"weighted_sampler"``.
        corpus_version: TUSZ version string.
        overlap_threshold: (unused, kept for config compat).
    """

    def __init__(
        self,
        cache_root: str = "data/tusz",
        split_mode: str = "official",
        fold: int = 0,
        n_folds: int = 5,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        montage: str = "bipolar",
        label_mode: str = "binary",
        normalize: str = "none",
        batch_size: int = 64,
        num_workers: int = 4,
        balance: str = "weighted_sampler",
        corpus_version: str = "v2.0.3",
        overlap_threshold: float = 0.0,
        folds_manifest: Optional[str] = "tusz",
        strict_folds: bool = True,
        **kwargs: Any,
    ) -> None:
        n_channels = 18 if montage == "bipolar" else 19
        meta: Dict[str, Any] = {
            "canonical_label_space": ["bckg", "seiz"],
            "epoch_seconds": window_s,
            "channel_policy": ["eeg"],
            "signal_kind": "raw",
            "split_mode": split_mode,
            "fold": fold,
            "n_folds": n_folds,
            "montage": montage,
            "label_mode": label_mode,
            "n_channels": n_channels,
        }
        if split_mode == "kfold":
            meta["folds_manifest"] = folds_manifest
        super().__init__(name="tusz", metadata=meta)

        self._cache_root = Path(cache_root)
        self._split_mode = split_mode
        self._fold = fold
        self._n_folds = n_folds
        self._folds_manifest = folds_manifest
        self._strict_folds = bool(strict_folds)
        self._manifest_roles: Optional[Dict[str, str]] = None
        self._window_s = window_s
        self._stride_s = stride_s
        self._montage = montage
        self._label_mode = label_mode
        self._normalize = normalize
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._balance = balance
        self._corpus_version = corpus_version

        from neuroatlas.extensions.datasets.dataio.tusz import (
            TUSZContinuousDataset,
            _collate_tusz,
        )
        from neuroatlas.extensions.datasets.epilepsy.tusz_preprocessor import (
            CACHE_SCHEMA_TAG,
        )

        self._collate_fn = _collate_tusz

        if split_mode == "official":
            self._init_official(TUSZContinuousDataset, CACHE_SCHEMA_TAG)
        elif split_mode == "kfold":
            self._init_kfold(TUSZContinuousDataset, CACHE_SCHEMA_TAG)
        else:
            raise ValueError(f"Unsupported split_mode {split_mode!r}. Expected 'official' or 'kfold'.")

    def _h5_path(self, split: str, schema_tag: str) -> str:
        fname = f"tusz_{self._corpus_version}_{split}_{schema_tag}.h5"
        return str(self._cache_root / fname)

    def _init_official(self, DatasetCls, schema_tag: str) -> None:
        common = dict(
            window_s=self._window_s,
            stride_s=self._stride_s,
            label_mode=self._label_mode,
            normalize=self._normalize,
            montage=self._montage,
        )
        self._train_ds = DatasetCls(h5_path=self._h5_path("train", schema_tag), **common)
        self._val_ds = DatasetCls(h5_path=self._h5_path("dev", schema_tag), **common)
        self._test_ds = DatasetCls(h5_path=self._h5_path("eval", schema_tag), **common)

    def _init_kfold(self, DatasetCls, schema_tag: str) -> None:
        import h5py as _h5py

        h5_path = self._h5_path("all", schema_tag)
        with _h5py.File(h5_path, "r") as h5:
            subject_ids = [s.decode() if isinstance(s, bytes) else str(s)
                           for s in h5["subject_ids"][:]]
            samplewise_label_exists = "samplewise_label" in h5
            offsets = h5["recording_offsets"][:]

        if self._folds_manifest is not None:
            # The paper's folds (configs/cohorts/tusz/folds.json), not a
            # re-derivation: see adapters/tusz_edf.py.
            from neuroatlas.benchmarking_helpers.registry.fold_manifest import (
                fold_source_label,
                recording_splits_from_manifest,
            )

            train_recs, val_recs, test_recs, split = recording_splits_from_manifest(
                self._folds_manifest, self._fold, self._n_folds, subject_ids,
                strict=self._strict_folds,
            )
            self.metadata["fold_source"] = fold_source_label(split)
            self._manifest_roles = {}
            for name, recs in (("train", train_recs), ("val", val_recs), ("test", test_recs)):
                for r in recs:
                    self._manifest_roles[subject_ids[r]] = name
            common = dict(
                h5_path=h5_path,
                window_s=self._window_s,
                stride_s=self._stride_s,
                label_mode=self._label_mode,
                normalize=self._normalize,
                montage=self._montage,
            )
            self._train_ds = DatasetCls(recording_indices=train_recs, **common)
            self._val_ds = DatasetCls(recording_indices=val_recs, **common)
            self._test_ds = DatasetCls(recording_indices=test_recs, **common)
            return

        # Determine per-recording seizure presence
        n_rec = len(subject_ids)
        rec_has_seizure = np.zeros(n_rec, dtype=np.int64)
        if samplewise_label_exists:
            with _h5py.File(h5_path, "r") as h5:
                for i in range(n_rec):
                    s = int(offsets[i])
                    e = int(offsets[i + 1])
                    chunk_size = min(e - s, 5_000_000)
                    for cs in range(s, e, chunk_size):
                        ce = min(cs + chunk_size, e)
                        if np.any(h5["samplewise_label"][cs:ce] > 0):
                            rec_has_seizure[i] = 1
                            break

        assignments = _all_fold_assignments(subject_ids, rec_has_seizure, self._n_folds)

        train_recs: List[int] = []
        val_recs: List[int] = []
        test_recs: List[int] = []
        for i, subj in enumerate(subject_ids):
            role = assignments.get(subj, {}).get(self._fold, "train")
            if role == "train":
                train_recs.append(i)
            elif role == "val":
                val_recs.append(i)
            elif role == "test":
                test_recs.append(i)

        logger.info(
            "k-fold %d/%d: train=%d val=%d test=%d recordings",
            self._fold, self._n_folds, len(train_recs), len(val_recs), len(test_recs),
        )

        common = dict(
            h5_path=h5_path,
            window_s=self._window_s,
            stride_s=self._stride_s,
            label_mode=self._label_mode,
            normalize=self._normalize,
            montage=self._montage,
        )
        self._train_ds = DatasetCls(recording_indices=train_recs, **common)
        self._val_ds = DatasetCls(recording_indices=val_recs, **common)
        self._test_ds = DatasetCls(recording_indices=test_recs, **common)

    def _make_loader(self, dataset, shuffle: bool) -> _LoaderAdapter:
        from torch.utils.data import DataLoader, WeightedRandomSampler

        sampler = None
        if shuffle and self._balance == "weighted_sampler":
            labels = dataset.binary_labels()
            counts = np.bincount(labels.astype(np.int64))
            if len(counts) >= 2 and counts.min() > 0:
                weights = 1.0 / counts[labels]
                sampler = WeightedRandomSampler(
                    weights.tolist(), num_samples=len(weights), replacement=True
                )
                shuffle = False

        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=False,
        )
        return _LoaderAdapter(loader, "tusz")

    def train_dataloader(self) -> _LoaderAdapter:
        # Embed-only path doesn't benefit from shuffle and it kills LRU
        # locality on lazy EDF dataios. Probes run on cached embeddings.
        return self._make_loader(self._train_ds, shuffle=False)

    def val_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._test_ds, shuffle=False)

    def datasets(self) -> Dict[str, "TUSZContinuousDataset"]:
        return {"train": self._train_ds, "val": self._val_ds, "test": self._test_ds}

    # -- Global embedding cache (embed once, split by fold) ------------------

    #: k-fold mode embeds every recording once and assigns folds when
    #: probing, for seizure detection as for the linear probe.
    global_cache_for_seizure_detection = True

    def cache_context(self, purpose: str = "default") -> Dict[str, Any]:
        """What the embedding-cache key hashes.

        The global (k-fold) cache holds every window of every recording, so
        nothing about the fold belongs in its key; it used to inherit the
        whole metadata, fold included, so each fold looked for its own
        "global" cache and fold 1 found nothing. In ``official`` mode the
        corpus's own train/dev/eval files are the splits whatever the fold,
        so the fold is not part of the per-split key either.
        """
        from neuroatlas.extensions.datasets.epilepsy._global_cache import _FOLD_KEYS

        context = dict(self.metadata)
        if purpose == "global_embeddings":
            for key in _FOLD_KEYS:
                context.pop(key, None)
            context["epoch_seconds"] = float(self._window_s)
            context["stride_s"] = float(self._window_s)
            context["global_layout"] = "all_recordings_v1"
        elif self._split_mode == "official":
            for key in _FOLD_KEYS:
                context.pop(key, None)
        return context

    def supports_global_embedding_cache(self) -> bool:
        # The global cache holds windows cut at window_s; it can stand in for
        # the train split only when train windows are cut the same way.
        stride = self._stride_s
        return self._split_mode == "kfold" and (
            stride is None or float(stride) == float(self._window_s))

    def full_embedding_dataloader(self) -> _LoaderAdapter:
        """Return a loader over ALL recordings (no fold filtering).

        Used to embed the entire dataset once; the result is then split
        per-fold via :meth:`split_global_embedding_payload`.
        """
        from neuroatlas.extensions.datasets.dataio.tusz import TUSZContinuousDataset
        from neuroatlas.extensions.datasets.epilepsy.tusz_preprocessor import CACHE_SCHEMA_TAG
        from torch.utils.data import DataLoader

        h5_path = self._h5_path("all", CACHE_SCHEMA_TAG)
        ds = TUSZContinuousDataset(
            h5_path=h5_path,
            window_s=self._window_s,
            stride_s=self._stride_s,
            label_mode=self._label_mode,
            normalize=self._normalize,
            montage=self._montage,
            # No recording_indices — uses ALL recordings.
        )
        loader = DataLoader(
            ds,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=False,
        )
        return _LoaderAdapter(loader, "tusz")

    def split_global_embedding_payload(self, payload) -> Dict[str, Any]:
        """Partition a global embedding payload into train/val/test by fold.

        ``payload`` is an :class:`EmbeddingPayload` (features, labels,
        metadata) produced from :meth:`full_embedding_dataloader`. Each
        sample's ``meta["subject_id"]`` is looked up in the fold
        assignments to determine its role.
        """
        from neuroatlas.benchmarking_helpers import EmbeddingPayload
        import h5py as _h5py
        from neuroatlas.extensions.datasets.epilepsy.tusz_preprocessor import CACHE_SCHEMA_TAG

        if self._manifest_roles is not None:
            groups: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
            for i, meta in enumerate(payload.metadata):
                role = self._manifest_roles.get(str(meta.get("subject_id", "")))
                if role is not None:
                    groups[role].append(i)
            feats = np.asarray(payload.features)
            labs = np.asarray(payload.labels)
            out: Dict[str, EmbeddingPayload] = {}
            for name, indices in groups.items():
                idx = np.asarray(indices, dtype=np.int64)
                out[name] = EmbeddingPayload(
                    features=feats[idx] if idx.size else np.zeros((0, feats.shape[1]), dtype=feats.dtype),
                    labels=labs[idx] if idx.size else np.zeros((0,), dtype=labs.dtype),
                    metadata=[payload.metadata[i] for i in indices],
                )
            return out

        # folds_manifest=None: re-derive fold assignments (same deterministic
        # computation as _init_kfold).
        h5_path = self._h5_path("all", CACHE_SCHEMA_TAG)
        with _h5py.File(h5_path, "r") as h5:
            subject_ids = [s.decode() if isinstance(s, bytes) else str(s)
                           for s in h5["subject_ids"][:]]
            offsets = h5["recording_offsets"][:]
        n_rec = len(subject_ids)
        rec_has_seizure = np.zeros(n_rec, dtype=np.int64)
        with _h5py.File(h5_path, "r") as h5:
            for i in range(n_rec):
                s, e = int(offsets[i]), int(offsets[i + 1])
                for cs in range(s, e, 5_000_000):
                    ce = min(cs + 5_000_000, e)
                    if np.any(h5["samplewise_label"][cs:ce] > 0):
                        rec_has_seizure[i] = 1
                        break
        assignments = _all_fold_assignments(subject_ids, rec_has_seizure, self._n_folds)

        # Partition the payload by subject → fold role.
        features = np.asarray(payload.features)
        labels = np.asarray(payload.labels)
        metadata = list(payload.metadata)

        split_idx: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        for i, meta in enumerate(metadata):
            subj = str(meta.get("subject_id", ""))
            role = assignments.get(subj, {}).get(self._fold, "train")
            split_idx[role].append(i)

        result: Dict[str, EmbeddingPayload] = {}
        for split_name, indices in split_idx.items():
            idx = np.asarray(indices, dtype=np.int64)
            result[split_name] = EmbeddingPayload(
                features=features[idx] if idx.size else np.zeros((0, features.shape[1]), dtype=features.dtype),
                labels=labels[idx] if idx.size else np.zeros((0,), dtype=labels.dtype),
                metadata=[metadata[i] for i in indices],
            )
        return result
