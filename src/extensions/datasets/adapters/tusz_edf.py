"""BenchmarkDataModule adapter that streams TUSZ EDFs directly (no HDF5).

Thin wrapper around :class:`TUSZEDFDirectDataset`. Exposes the same
``train/val/test_dataloader`` contract as :class:`TUSZDataModule` so the
benchmark runner can drive it without modification. ``split_mode="official"``
only — k-fold over EDFs is not supported in v1.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional

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
        **kwargs: Any,
    ) -> None:
        if split_mode != "official":
            raise ValueError(
                f"TUSZEDFDirectDataModule supports split_mode='official' only "
                f"(got {split_mode!r}). Use the HDF5-backed TUSZDataModule for k-fold."
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
        super().__init__(name="tusz", metadata=meta)

        self._raw_root = str(raw_root)
        self._window_s = float(window_s)
        self._stride_s = float(stride_s) if stride_s is not None else self._window_s
        self._montage = montage
        self._label_mode = label_mode
        self._batch_size = int(batch_size)
        self._num_workers = int(num_workers)
        self._balance = balance
        self._lru_recordings = int(lru_recordings)

        from extensions.datasets.dataio.tusz_edf import (
            TUSZEDFDirectDataset,
        )
        from extensions.datasets.dataio.tusz import _collate_tusz

        self._collate_fn = _collate_tusz

        common = dict(
            raw_root=self._raw_root,
            window_s=self._window_s,
            stride_s=self._stride_s,
            montage=self._montage,
            label_mode=self._label_mode,
            lru_recordings=self._lru_recordings,
        )
        self._train_ds = TUSZEDFDirectDataset(split="train", **common)
        self._val_ds = TUSZEDFDirectDataset(split="dev", **common)
        self._test_ds = TUSZEDFDirectDataset(split="eval", **common)

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

    def supports_global_embedding_cache(self) -> bool:
        return False
