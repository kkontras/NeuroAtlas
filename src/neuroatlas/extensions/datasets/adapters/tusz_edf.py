"""BenchmarkDataModule adapter that streams TUSZ EDFs directly (no HDF5).

Thin wrapper around :class:`TUSZEDFDirectDataset`. Exposes the same
``train/val/test_dataloader`` contract as :class:`TUSZDataModule` so the
benchmark runner can drive it without modification.

Two split modes:

* ``"kfold"`` -- the paper's protocol (App. C.1, "folds are defined at the
  patient level"; the TUSZ results are 5-fold means). The recordings of the
  corpus's train/, dev/ and eval/ trees are pooled (675 patients in v2.0.3)
  and each patient takes the role ``configs/cohorts/tusz/folds.json`` gives it
  in the requested fold: the folds the paper's epilepsy probe drew. The whole
  corpus is embedded once and every fold is cut from that cache at probe time.
* ``"official"`` -- the corpus's own train/dev/eval partition, the same for
  every fold.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .base import BenchmarkDataModule

logger = logging.getLogger(__name__)


class _LoaderAdapter:
    def __init__(self, loader, dataset_name: str = "tusz") -> None:
        self._loader = loader
        self._dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for batch in self._loader:
            yield batch


class TUSZEDFDirectDataModule(BenchmarkDataModule):
    """TUSZ DataModule that reads EDFs on the fly, no HDF5 cache required.

    Args:
        raw_root: Path to the TUSZ ``.../edf`` directory containing
            ``train/``, ``dev/``, ``eval/`` sub-trees.
        split_mode: Must be ``"official"``.
        window_s: Window duration in seconds.
        stride_s: Window stride in seconds (defaults to ``window_s``).
        montage: ``"unipolar"`` or ``"bipolar"``.
        label_mode: Only ``"binary"`` is supported in v1.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.
        balance: ``"none"`` or ``"weighted_sampler"`` (train split only).
        lru_recordings: Per-worker LRU size for filtered recordings.
    """

    def __init__(
        self,
        raw_root: str = "${EEG_DATA_ROOT}/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf",
        split_mode: str = "official",
        window_s: float = 10.0,
        stride_s: Optional[float] = None,
        montage: str = "bipolar",
        label_mode: str = "binary",
        normalize: str = "none",
        batch_size: int = 64,
        num_workers: int = 4,
        balance: str = "none",
        lru_recordings: int = 2,
        fold: int = 0,
        n_folds: int = 5,
        folds_manifest: Optional[str] = "tusz",
        strict_folds: bool = True,
        **kwargs: Any,
    ) -> None:
        if split_mode not in ("official", "kfold"):
            raise ValueError(
                f"TUSZEDFDirectDataModule split_mode must be 'kfold' (the paper's "
                f"patient-level folds) or 'official' (train/dev/eval), got {split_mode!r}."
            )
        if normalize != "none":
            raise ValueError(
                f"TUSZEDFDirectDataModule does not implement normalize={normalize!r}; "
                f"use TUSZDataModule (HDF5) if per-window z-score is required."
            )

        n_channels = 20 if montage == "bipolar" else 19
        meta: Dict[str, Any] = {
            "canonical_label_space": ["bckg", "seiz"],
            "epoch_seconds": window_s,
            "channel_policy": ["eeg"],
            "signal_kind": "raw",
            "split_mode": split_mode,
            "montage": montage,
            "label_mode": label_mode,
            "n_channels": n_channels,
            "source": "edf_direct",
        }
        if split_mode == "kfold":
            # Per-split caches (stride != window) differ per fold; the global
            # cache drops these keys again (cache_context).
            meta.update({"fold": int(fold), "n_folds": int(n_folds),
                         "folds_manifest": folds_manifest})
        super().__init__(name="tusz", metadata=meta)
        self._split_mode = split_mode

        self._raw_root = str(raw_root)
        self._window_s = float(window_s)
        self._stride_s = float(stride_s) if stride_s is not None else self._window_s
        self._montage = montage
        self._label_mode = label_mode
        self._batch_size = int(batch_size)
        self._num_workers = int(num_workers)
        self._balance = balance
        self._lru_recordings = int(lru_recordings)

        from neuroatlas.extensions.datasets.dataio.tusz_edf import (
            TUSZEDFDirectDataset,
        )
        from neuroatlas.extensions.datasets.dataio.tusz import _collate_tusz

        self._collate_fn = _collate_tusz

        common = dict(
            raw_root=self._raw_root,
            window_s=self._window_s,
            stride_s=self._stride_s,
            montage=self._montage,
            label_mode=self._label_mode,
            lru_recordings=self._lru_recordings,
        )
        self._common = common
        self._all_ds = None
        if split_mode == "official":
            self._train_ds = TUSZEDFDirectDataset(split="train", **common)
            self._val_ds = TUSZEDFDirectDataset(split="dev", **common)
            self._test_ds = TUSZEDFDirectDataset(split="eval", **common)
        else:
            self._init_kfold(int(fold), int(n_folds), folds_manifest, bool(strict_folds))

    # -- k-fold: the paper's patient-level folds ------------------------------

    def _init_kfold(self, fold: int, n_folds: int, folds_manifest: Optional[str],
                    strict: bool) -> None:
        from neuroatlas.extensions.datasets.dataio.tusz_edf import (
            TUSZEDFDirectDataset,
            _discover_edf_manifest,
        )

        # train/, dev/ and eval/ pooled, in that order: the corpus the paper's
        # folds were drawn over. A recording keeps its official split name in
        # its own meta; its fold role comes from the manifest.
        pooled: List[Dict[str, Any]] = []
        for official in ("train", "dev", "eval"):
            pooled.extend(_discover_edf_manifest(self._raw_root, official))
        self._pooled = pooled
        subjects = [str(rec["subject_id"]) for rec in pooled]

        if folds_manifest is None:
            train, val, test = _runtime_kfold(pooled, subjects, fold, n_folds)
        else:
            from neuroatlas.benchmarking_helpers.registry.fold_manifest import (
                fold_source_label,
                recording_splits_from_manifest,
            )

            train, val, test, split = recording_splits_from_manifest(
                folds_manifest, fold, n_folds, subjects, strict=strict,
            )
            self.metadata["fold_source"] = fold_source_label(split)

        self._role_of_subject: Dict[str, str] = {}
        for name, recs in (("train", train), ("val", val), ("test", test)):
            for r in recs:
                self._role_of_subject[subjects[r]] = name
        logger.info(
            "TUSZ EDF k-fold %d/%d: train=%d val=%d test=%d recordings (%d patients)",
            fold, n_folds, len(train), len(val), len(test), len(self._role_of_subject),
        )
        self._train_ds = TUSZEDFDirectDataset(
            split="train", manifest=[pooled[i] for i in train], **self._common)
        self._val_ds = TUSZEDFDirectDataset(
            split="dev", manifest=[pooled[i] for i in val], **self._common)
        self._test_ds = TUSZEDFDirectDataset(
            split="eval", manifest=[pooled[i] for i in test], **self._common)

    # -- DataLoader factory --------------------------------------------------

    def _make_loader(self, dataset, shuffle: bool) -> _LoaderAdapter:
        from torch.utils.data import DataLoader, WeightedRandomSampler

        sampler = None
        if shuffle and self._balance == "weighted_sampler":
            labels = dataset.binary_labels()
            counts = np.bincount(labels.astype(np.int64))
            if len(counts) >= 2 and counts.min() > 0:
                weights = 1.0 / counts[labels]
                sampler = WeightedRandomSampler(
                    weights.tolist(), num_samples=len(weights), replacement=True,
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
        # When balance="none" the train loader is used for embed-only (or a
        # downstream probe that doesn't depend on window order). Streaming the
        # train split sequentially lets the per-recording LRU absorb the
        # EDF-open + filtfilt cost; shuffling defeats the LRU entirely.
        shuffle = self._balance != "none"
        return self._make_loader(self._train_ds, shuffle=shuffle)

    def val_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._test_ds, shuffle=False)

    def datasets(self) -> Dict[str, Any]:
        return {"train": self._train_ds, "val": self._val_ds, "test": self._test_ds}

    # -- global embedding cache (k-fold): embed once, cut folds when probing --

    #: ``seizure_detection`` takes the global cache only from adapters that
    #: opt in; ``supports_global_embedding_cache`` still decides per run.
    global_cache_for_seizure_detection = True

    def supports_global_embedding_cache(self) -> bool:
        # One window grid serves every role only when train windows are cut
        # like val/test ones (stride == window, the benchmark's 10 s / 10 s).
        return self._split_mode == "kfold" and self._stride_s == self._window_s

    def cache_context(self, purpose: str = "default") -> Dict[str, Any]:
        """What the embedding-cache key hashes. The global cache holds every
        window of the pooled corpus, so nothing about the fold is in its key."""
        from neuroatlas.extensions.datasets.epilepsy._global_cache import _FOLD_KEYS

        context = dict(self.metadata)
        if purpose == "global_embeddings":
            for key in _FOLD_KEYS:
                context.pop(key, None)
            context["epoch_seconds"] = float(self._window_s)
            context["stride_s"] = float(self._window_s)
            context["global_layout"] = "pooled_train_dev_eval_v1"
        return context

    def full_embedding_dataloader(self) -> _LoaderAdapter:
        """Every window of every pooled recording, in (recording, start) order."""
        from neuroatlas.extensions.datasets.dataio.tusz_edf import TUSZEDFDirectDataset

        if self._split_mode != "kfold":
            raise RuntimeError("the TUSZ global cache exists in split_mode='kfold' only")
        if self._all_ds is None:
            self._all_ds = TUSZEDFDirectDataset(
                split="train", manifest=list(self._pooled), **self._common)
        return self._make_loader(self._all_ds, shuffle=False)

    def split_global_embedding_payload(self, payload) -> Dict[str, Any]:
        """This fold's train/val/test rows of the global payload, by patient.

        Rows keep the cache's (recording, window) order, which is the order the
        per-split readers emit. A patient with no role (``strict_folds=false``
        and not in the manifest) is in no split.
        """
        from neuroatlas.benchmarking_helpers import EmbeddingPayload

        groups: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        for i, m in enumerate(payload.metadata):
            role = self._role_of_subject.get(str(m["subject_id"]))
            if role is not None:
                groups[role].append(i)
        if not any(groups.values()):
            raise ValueError(
                "tusz: none of the cached windows' patients has a role in this "
                "fold -- the cache was not made from this corpus."
            )
        features = payload.features
        labels = np.asarray(payload.labels)
        out: Dict[str, Any] = {}
        for name, idx in groups.items():
            arr = np.asarray(idx, dtype=np.int64)
            items = []
            for i in idx:
                m = dict(payload.metadata[i])
                m["split"] = name
                items.append(m)
            out[name] = EmbeddingPayload(
                features=np.asarray(features[arr]) if arr.size else
                np.zeros((0,) + tuple(features.shape[1:]), dtype=features.dtype),
                labels=labels[arr] if arr.size else np.zeros((0,), dtype=labels.dtype),
                metadata=items,
            )
        return out


def _runtime_kfold(pooled, subjects, fold: int, n_folds: int):
    """``folds_manifest=None``: derive the folds with the shared patient
    splitter, stratified on "has any seizure" read from the ``.csv_bi``
    annotations -- the paper's rule, for a fold count it never ran."""
    from neuroatlas.extensions.datasets.adapters.tusz import _patient_splits
    from neuroatlas.extensions.datasets.epilepsy.tusz.annotations import parse_csv_bi

    has_seizure: Dict[str, int] = {}
    for rec, sid in zip(pooled, subjects):
        seiz = bool(rec.get("csv_bi_path")) and any(
            ev["label"] == "seiz" for ev in parse_csv_bi(rec["csv_bi_path"]))
        has_seizure[sid] = max(has_seizure.get(sid, 0), int(seiz))
    patients = sorted(has_seizure)
    y = np.asarray([has_seizure[p] for p in patients])
    tr, va, te = _patient_splits(patients, y, fold, n_folds)
    role = {**{p: "train" for p in tr}, **{p: "val" for p in va}, **{p: "test" for p in te}}
    out: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for i, sid in enumerate(subjects):
        out[role[sid]].append(i)
    return out["train"], out["val"], out["test"]
