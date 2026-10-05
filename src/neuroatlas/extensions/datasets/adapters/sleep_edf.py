from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .base import BenchmarkDataModule
from neuroatlas.benchmarking_helpers import EmbeddingPayload, dataloader_worker_init_fn

_SLEEP_EDF_LABEL_MAP = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "N4": 3,
    "R": 4,
    "REM": 4,
}


_SLEEP_EDF_DEFAULT_SAMPLING_RATE = 100.0
_SLEEP_EDF_DEFAULT_UNIT = "uV"


class SleepEDFNPZDataset(Dataset):
    """Expected format for each sample .npz:

    - signal: ndarray [channels, time] or [time]
    - label: int or stage string
    - subject_id: optional scalar
    - epoch_index: optional scalar
    - sampling_rate: optional scalar (Hz). Defaults to 100 Hz
      (Sleep-EDF canonical rate) if absent.
    - unit: optional str in {"uV", "mV", "V"}. Defaults to "uV"
      (Sleep-EDF canonical unit; NPZ exports produced by this benchmark
      are in µV) if absent.
    - channels: optional array/list of channel names
    """

    def __init__(self, root: Path):
        self.root = root
        self.files = sorted(root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No .npz files found in {root}. Export Sleep-EDF epochs into split directories first."
            )

    def __len__(self) -> int:
        return len(self.files)

    def _normalize_label(self, value) -> int:
        if isinstance(value, np.ndarray):
            value = value.item()
        if isinstance(value, (np.integer, int)):
            return int(value)
        value = str(value)
        if value not in _SLEEP_EDF_LABEL_MAP:
            raise ValueError(f"Unsupported Sleep-EDF label {value!r}.")
        return _SLEEP_EDF_LABEL_MAP[value]

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = np.load(self.files[index], allow_pickle=True)
        signal = sample["signal"]
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]
        label = self._normalize_label(sample["label"])
        subject_id = int(sample["subject_id"]) if "subject_id" in sample else -1
        epoch_index = int(sample["epoch_index"]) if "epoch_index" in sample else index
        sampling_rate = (
            float(sample["sampling_rate"])
            if "sampling_rate" in sample
            else _SLEEP_EDF_DEFAULT_SAMPLING_RATE
        )
        unit = (
            str(sample["unit"].item() if hasattr(sample["unit"], "item") else sample["unit"])
            if "unit" in sample
            else _SLEEP_EDF_DEFAULT_UNIT
        )
        channels: Optional[List[str]] = None
        if "channels" in sample:
            channels = [str(value) for value in sample["channels"].tolist()]
        return {
            "signals": {"eeg": torch.as_tensor(signal, dtype=torch.float32)},
            "label": torch.tensor(label, dtype=torch.long),
            "meta": {
                "dataset": "sleep_edf",
                "subject_id": subject_id,
                "epoch_index": epoch_index,
                "sampling_rate": sampling_rate,
                "unit": unit,
                "channels": channels or ["eeg"],
                "source_file": str(self.files[index]),
            },
        }


def _collate_npz(batch: List[Dict[str, object]]) -> Dict[str, object]:
    signals = torch.stack([item["signals"]["eeg"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    return {
        "signals": {"eeg": signals},
        "label": labels,
        "meta": [item["meta"] for item in batch],
        "raw_batch": batch,
    }


def _collate_physioex(batch: List[Dict[str, object]]) -> Dict[str, object]:
    stft_eeg = torch.stack([item["data"]["stft_eeg"] for item in batch], dim=0)
    stft_eog = torch.stack([item["data"]["stft_eog"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    return {
        "signals": {
            "stft_eeg": stft_eeg,
            "stft_eog": stft_eog,
        },
        "label": labels,
        "meta": [item["meta"] for item in batch],
        "raw_batch": batch,
    }


def _collate_physioex_raw(batch: List[Dict[str, object]]) -> Dict[str, object]:
    eeg = torch.stack([item["signals"]["eeg"] for item in batch], dim=0)
    full_signal = torch.stack([item["signals"]["full_signal"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    return {
        "signals": {
            "eeg": eeg,
            "full_signal": full_signal,
        },
        "label": labels,
        "meta": [item["meta"] for item in batch],
        "raw_batch": batch,
    }


class SleepEDFBenchmarkDataModule(BenchmarkDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 8,
        data_format: str = "npz",
        fold: Optional[int] = None,
        norm_path: Optional[str] = None,
        limit_windows_per_split: Optional[int] = None,
        num_workers: int = 0,
        signal_kind: Optional[str] = None,
        channel_specs=None,
        use_all_eeg_channels: bool = False,
    ):
        root = Path(data_root)
        metadata = {
            "canonical_label_space": ["W", "N1", "N2", "N3", "REM"],
            "epoch_seconds": 30,
            "data_format": data_format,
        }
        if fold is not None:
            metadata["fold"] = int(fold)
        super().__init__(
            name="sleep_edf",
            metadata=metadata,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_format = data_format.lower()
        self.signal_kind = signal_kind
        self.fold = fold
        self.limit_windows_per_split = limit_windows_per_split

        if self.data_format == "npz":
            self.metadata.update({
                "channel_policy": ["eeg"],
                "expected_layout": {
                    "train": "*.npz",
                    "val": "*.npz",
                    "test": "*.npz",
                },
                "label_map": _SLEEP_EDF_LABEL_MAP,
            })
            self._collate_fn = _collate_npz
            self.train_ds = SleepEDFNPZDataset(root / "train")
            self.val_ds = SleepEDFNPZDataset(root / "val")
            self.test_ds = SleepEDFNPZDataset(root / "test")
            return

        if self.data_format == "physioex":
            from neuroatlas.extensions.datasets.dataio.sleep_edf import (
                PhysioExSleepEDFCoreDataset,
                PhysioExSleepEDFSignalDataset,
            )

            if fold is None:
                raise ValueError("Sleep-EDF PhysioEx format requires an explicit fold.")
            active_signal_kind = str(signal_kind or "xsleepnet").lower()
            self.metadata.update({
                "signal_kind": active_signal_kind,
                "expected_layout": {
                    "table": "table.csv with fold columns",
                    "features": "*.dat memmaps",
                },
                "label_map": _SLEEP_EDF_LABEL_MAP,
            })
            if active_signal_kind == "xsleepnet":
                self.metadata["channel_policy"] = ["stft_eeg", "stft_eog"]
                self._collate_fn = _collate_physioex
                self.all_ds = PhysioExSleepEDFCoreDataset(
                    processed_root=str(root),
                    split=None,
                    fold=fold,
                    norm_path=norm_path,
                    limit_windows=None,
                )
                self.metadata["available_folds"] = list(self.all_ds.fold_columns)
                self.train_ds = PhysioExSleepEDFCoreDataset(
                    processed_root=str(root),
                    split="train",
                    fold=fold,
                    norm_path=norm_path,
                    limit_windows=limit_windows_per_split,
                )
                self.val_ds = PhysioExSleepEDFCoreDataset(
                    processed_root=str(root),
                    split="valid",
                    fold=fold,
                    norm_path=norm_path,
                    limit_windows=limit_windows_per_split,
                )
                self.test_ds = PhysioExSleepEDFCoreDataset(
                    processed_root=str(root),
                    split="test",
                    fold=fold,
                    norm_path=norm_path,
                    limit_windows=limit_windows_per_split,
                )
                return
            if active_signal_kind == "raw":
                self.metadata["channel_policy"] = ["eeg"]
                self._collate_fn = _collate_physioex_raw
                self.all_ds = PhysioExSleepEDFSignalDataset(
                    processed_root=str(root),
                    split=None,
                    fold=fold,
                    signal_kind="raw",
                    limit_windows=None,
                )
                self.metadata["available_folds"] = list(self.all_ds.fold_columns)
                self.train_ds = PhysioExSleepEDFSignalDataset(
                    processed_root=str(root),
                    split="train",
                    fold=fold,
                    signal_kind="raw",
                    limit_windows=limit_windows_per_split,
                )
                self.val_ds = PhysioExSleepEDFSignalDataset(
                    processed_root=str(root),
                    split="valid",
                    fold=fold,
                    signal_kind="raw",
                    limit_windows=limit_windows_per_split,
                )
                self.test_ds = PhysioExSleepEDFSignalDataset(
                    processed_root=str(root),
                    split="test",
                    fold=fold,
                    signal_kind="raw",
                    limit_windows=limit_windows_per_split,
                )
                return
            raise ValueError(f"Unsupported PhysioEx signal kind {active_signal_kind!r}.")
            return

        raise ValueError(f"Unsupported Sleep-EDF data format {data_format!r}.")

    def cache_context(self, purpose: str = "default") -> Dict[str, object]:
        context = dict(self.metadata)
        if purpose == "global_embeddings":
            context.pop("fold", None)
        return context

    def supports_global_embedding_cache(self) -> bool:
        return self.data_format == "physioex"

    def full_embedding_dataloader(self):
        if not self.supports_global_embedding_cache():
            raise NotImplementedError("Global embedding cache is only supported for Sleep-EDF PhysioEx datasets.")
        return DataLoader(
            self.all_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def split_global_embedding_payload(self, payload: EmbeddingPayload) -> Dict[str, EmbeddingPayload]:
        if self.fold is None:
            raise ValueError("Cannot split global embeddings without a selected fold.")
        fold_key = f"fold_{int(self.fold)}"

        def _subset(split_name: str) -> EmbeddingPayload:
            keep_indices = [
                index
                for index, item in enumerate(payload.metadata)
                if item.get("fold_assignments", {}).get(fold_key) == split_name
            ]
            if self.limit_windows_per_split is not None:
                keep_indices = keep_indices[: int(self.limit_windows_per_split)]
            if not keep_indices:
                raise ValueError(f"No embedding rows found for {split_name!r} in {fold_key}.")
            indices = np.asarray(keep_indices, dtype=int)
            return EmbeddingPayload(
                features=payload.features[indices],
                labels=payload.labels[indices],
                metadata=[payload.metadata[index] for index in keep_indices],
            )

        return {
            "train": _subset("train"),
            "val": _subset("valid"),
            "test": _subset("test"),
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            worker_init_fn=dataloader_worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            worker_init_fn=dataloader_worker_init_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            worker_init_fn=dataloader_worker_init_fn,
        )
