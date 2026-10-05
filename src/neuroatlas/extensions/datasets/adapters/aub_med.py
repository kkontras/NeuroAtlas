"""BenchmarkDataModule adapter for the AUB-MED / Nasreddine Epileptic EEG dataset.

End-to-end direct-EDF pipeline — no preprocessing step, no HDF5 cache.
A fresh checkout that has only fetched the corpus can use this
dataset out of the box.

With only 6 patients (p10-p15), ``n_folds`` is capped at the number of
subjects; any larger request is silently reduced to
``len(unique_subjects)`` by the shared patient-level splitter.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from neuroatlas.extensions.datasets.adapters.siena import _patient_splits_generic
from neuroatlas.extensions.datasets.dataio.aub_med import (
    AUBMedDataset,
    collate_aub_med,
    discover_aub_med_recordings,
)

from .base import BenchmarkDataModule

logger = logging.getLogger(__name__)


_DEFAULT_RAW_DIR = "${EEG_DATA_ROOT}/aub_med/raw"


# ---------------------------------------------------------------------------
# Loader adapter
# ---------------------------------------------------------------------------


# The shared wrapper; the local name is kept so call sites are unchanged.
from ._loader_adapters import SplitTaggingLoader as _LoaderAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class AUBMedBenchmarkDataModule(BenchmarkDataModule):
    """BenchmarkDataModule for AUB-MED seizure detection.

    Args:
        raw_dir: Directory containing ``pN_Record{K}.edf`` files.
        annotations_dir: Directory with ``Seizure_times.py``. Defaults to
            ``<raw_dir>/../annotations``.
        fold: Current fold index (0-based).
        n_folds: Total number of folds.  Capped at the number of unique
            subjects (6 in the shipped release) by the splitter.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        window_s: Window duration in seconds.
        stride_s: Window stride in seconds.
        montage: ``"unipolar"`` (19 ch) or ``"bipolar"`` (18 ch).
        label_mode: ``"binary"`` for seizure detection.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        balance: ``"weighted_sampler"`` or ``"none"``.
        overlap_threshold: Seizure fraction threshold for binary labels.
        signal_cache_size: LRU size for decoded recordings.
    """

    def __init__(
        self,
        raw_dir: str = _DEFAULT_RAW_DIR,
        annotations_dir: Optional[str] = None,
        fold: int = 0,
        n_folds: int = 6,
        batch_size: int = 64,
        num_workers: int = 4,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        montage: str = "bipolar",
        label_mode: str = "binary",
        normalize: str = "none",
        balance: str = "weighted_sampler",
        overlap_threshold: float = 0.0,
        signal_cache_size: int = 8,
        signal_kind: str = "raw",
        **_unused: Any,
    ) -> None:
        metadata = {
            "canonical_label_space": ["bckg", "seiz"],
            "epoch_seconds": float(window_s),
            "channel_policy": ["eeg"],
            "signal_kind": signal_kind,
            "fold": fold,
            "n_folds": n_folds,
        }
        super().__init__(name="aub_med", metadata=metadata)

        self._raw_dir = raw_dir
        self._annotations_dir = (
            annotations_dir if annotations_dir is not None
            else str(Path(raw_dir).parent / "annotations")
        )
        self._fold = fold
        self._n_folds = n_folds
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._window_s = float(window_s)
        self._stride_s = float(stride_s) if stride_s is not None else float(window_s)
        self._montage = montage
        self._label_mode = label_mode
        self._normalize = normalize
        self._balance = balance
        self._overlap_threshold = float(overlap_threshold)
        self._signal_cache_size = int(signal_cache_size)

        if not Path(raw_dir).exists():
            raise FileNotFoundError(
                f"AUB-MED raw_dir not found: {raw_dir}\n"
                f"Run `python -m neuroatlas.entrypoints.fetch --dataset aub_med --download` to stage the data, "
                f"or point `raw_dir` at an existing copy."
            )

        self._recordings = discover_aub_med_recordings(
            self._raw_dir, self._annotations_dir,
        )
        if not self._recordings:
            raise RuntimeError(
                f"AUB-MED: no EDFs discovered under {self._raw_dir}. "
                f"Expected pN_Record{{K}}.edf files."
            )

        subject_ids_per_rec = [r.subject_id for r in self._recordings]
        has_seizure_per_rec = [bool(r.seizure_intervals_s) for r in self._recordings]

        self._train_recs, self._val_recs, self._test_recs = _patient_splits_generic(
            subject_ids_per_rec, has_seizure_per_rec, fold, n_folds,
        )
        logger.info(
            "AUB-MED splits (fold=%d/%d): train=%d val=%d test=%d recordings",
            fold, n_folds,
            len(self._train_recs), len(self._val_recs), len(self._test_recs),
        )

        self._datasets: Dict[str, Any] = {}

    def cache_context(self, purpose: str = "default") -> Dict[str, Any]:
        # a montage other than the manifest's is its own cache (same rule as
        # the other epilepsy readers, epilepsy/_global_cache.montage_context)
        from neuroatlas.extensions.datasets.epilepsy._global_cache import montage_context

        return montage_context(self, dict(self.metadata))

    # ------------------------------------------------------------------
    # Per-split dataset construction
    # ------------------------------------------------------------------

    def _get_dataset(self, split: str) -> AUBMedDataset:
        if split in self._datasets:
            return self._datasets[split]

        rec_indices = {
            "train": self._train_recs,
            "val": self._val_recs,
            "test": self._test_recs,
        }[split]

        if not rec_indices:
            raise RuntimeError(
                f"AUB-MED split {split!r} has 0 recordings — check fold index."
            )

        stride_s = self._stride_s if split == "train" else self._window_s
        ds = AUBMedDataset(
            raw_dir=self._raw_dir,
            annotations_dir=self._annotations_dir,
            recordings=self._recordings,
            recording_indices=rec_indices,
            window_s=self._window_s,
            stride_s=stride_s,
            label_mode=self._label_mode,
            normalize=self._normalize,
            montage=self._montage,
            overlap_threshold=self._overlap_threshold,
            signal_cache_size=self._signal_cache_size,
        )
        self._datasets[split] = ds
        return ds

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
            collate_fn=collate_aub_med,
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
