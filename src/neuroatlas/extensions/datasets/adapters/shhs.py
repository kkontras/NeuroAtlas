from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import easydict
import torch

from neuroatlas.extensions.datasets.dataio.shhs import SleepDataLoader

from .base import BenchmarkDataModule

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "datasets" / "SHHS" / "assets"


_NORM_STFT = str(_ASSETS_DIR / "metrics_eeg_eog_emg_stft.pkl")
_NORM_STFT_TIME = str(_ASSETS_DIR / "metrics_eeg_eog_emg_stft_time.pkl")

# views per signal_kind — data_type + "_" + mod becomes the view key
_VIEWS_BY_KIND = {
    "stft": [
        {"list_dir": "patient_mat_list.txt",     "data_type": "stft", "mod": "eeg", "num_ch": 1},
        {"list_dir": "patient_eog_mat_list.txt", "data_type": "stft", "mod": "eog", "num_ch": 1},
    ],
    "raw": [
        {"list_dir": "patient_mat_list.txt", "data_type": "time", "mod": "eeg", "num_ch": 1},
    ],
    "both": [
        {"list_dir": "patient_mat_list.txt",     "data_type": "stft", "mod": "eeg", "num_ch": 1},
        {"list_dir": "patient_eog_mat_list.txt", "data_type": "stft", "mod": "eog", "num_ch": 1},
        {"list_dir": "patient_mat_list.txt",     "data_type": "time", "mod": "eeg", "num_ch": 1},
    ],
}
_NORM_BY_KIND = {
    "stft": _NORM_STFT,
    "raw":  _NORM_STFT_TIME,
    "both": _NORM_STFT_TIME,  # has both stft_eeg and time_eeg keys
}


def _default_shhs_config(
    data_root: str,
    fold: Optional[int] = None,
    batch_size: int = 2,
    signal_kind: str = "stft",
) -> easydict.EasyDict:
    if fold is not None:
        data_split = {
            "split_method": "patients_test",
            "fold": fold,
            "trainvaltest_splits_file": str(_ASSETS_DIR / "trainvaltest_splits.pkl"),
            "val_split_rate": 0.1,
            "test_split_rate": 0.1,
        }
    else:
        data_split = {
            "split_method": "patients_sleeptransformer",
            "folds_file": str(_ASSETS_DIR / "data_split_eval.mat"),
        }
    views = _VIEWS_BY_KIND.get(signal_kind, _VIEWS_BY_KIND["stft"])
    norm_dir = _NORM_BY_KIND.get(signal_kind, _NORM_STFT)
    return easydict.EasyDict({
        "training_params": {
            "batch_size": batch_size,
            "test_batch_size": batch_size,
            "data_loader_workers": 0,
            "pin_memory": False,
        },
        "dataset": {
            "dataloader_class": "SleepDataLoader",
            "data_roots": data_root,
            "outer_seq_length": 21,
            "norm_dir": norm_dir,
            "broken_patients_filepath": str(_ASSETS_DIR / "noisy_patients_trial2_pp.pkl"),
            "fold": fold,
            "data_view_dir": views,
            "data_split": data_split,
            "filter_patients": {
                "train": {"use_type": False},
                "val": {"use_type": False},
                "test": {"use_type": False},
                "total": {"use_type": False},
            },
        },
        "model": {"args": {"num_classes": 5}},
        "statistics": {"print": False},
    })


def _flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)


# SHHS is standardly resampled to 100 Hz per 30 s epoch (3000 samples)
# and amplitude-normalized to µV by the upstream SleepDataLoader.
_SHHS_SAMPLING_RATE = 100.0
_SHHS_UNIT = "uV"
# (MODEL_CONTRACTS.md §2). 500 µV matches the DN3 default for
# scalp-EEG corpora and aligns with sleep_edf / parkinson.
# SHHS's standard EEG lead for sleep scoring (AASM): central
# derivation referenced against the contralateral mastoid. The
# SleepDataLoader exposes a single time-domain EEG channel; the
# adapter publishes that identity so the channel-map layer can
# resolve it per model (src/neuroatlas/configs/channel_maps/shhs.yaml).
_SHHS_EEG_CHANNELS = ("C4-A1",)


def _build_meta(idx_tensor: torch.Tensor, dataset_name: str) -> List[Dict[str, object]]:
    idx_tensor = idx_tensor.reshape(-1, idx_tensor.shape[-1]).cpu()
    output: List[Dict[str, object]] = []
    for row in idx_tensor.tolist():
        output.append({
            "dataset": dataset_name,
            "subject_id": int(row[0]),
            "epoch_index": int(row[1]),
            "sampling_rate": _SHHS_SAMPLING_RATE,
            "unit": _SHHS_UNIT,
            "channels": list(_SHHS_EEG_CHANNELS),
        })
    return output


class _LoaderAdapter:
    def __init__(self, loader: Iterable, dataset_name: str, signal_kind: str = "stft"):
        self.loader = loader
        self.dataset_name = dataset_name
        self.signal_kind = signal_kind

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        for batch in self.loader:
            labels = _flatten_tensor(batch["label"])
            data = batch["data"]
            signals: Dict[str, object] = {}

            if self.signal_kind in ("stft", "both"):
                for key in ("stft_eeg", "stft_eog"):
                    if key in data:
                        signals[key] = data[key]

            if self.signal_kind in ("raw", "both"):
                raw = data.get("time_eeg")
                if raw is not None:
                    # raw shape from loader: (batch, seq_len, 1, 3000)
                    # flatten seq into batch dim so per-epoch models see (batch*seq, 1, 3000)
                    b = raw.shape[0]
                    signals["eeg"] = raw.reshape(b * raw.shape[1], *raw.shape[2:])

            yield {
                "signals": signals,
                "label": labels,
                "meta": _build_meta(batch["idx"], self.dataset_name),
                "raw_batch": batch,
            }


class SHHSBenchmarkDataModule(BenchmarkDataModule):
    def __init__(
        self,
        data_root: str,
        fold: Optional[int] = None,
        batch_size: int = 2,
        signal_kind: str = "stft",
        channel_specs=None,
        use_all_eeg_channels: bool = False,
    ):
        channel_policies = {
            "stft": ["stft_eeg", "stft_eog"],
            "raw":  ["eeg"],
            "both": ["stft_eeg", "stft_eog", "eeg"],
        }
        meta: Dict[str, object] = {
            "canonical_label_space": ["W", "N1", "N2", "N3", "REM"],
            "epoch_seconds": 30,
            "channel_policy": channel_policies.get(signal_kind, ["stft_eeg", "stft_eog"]),
            "signal_kind": signal_kind,
        }
        if fold is not None:
            meta["fold"] = fold
        super().__init__(name="shhs", metadata=meta)
        self.signal_kind = signal_kind
        self.config = _default_shhs_config(data_root, fold=fold, batch_size=batch_size, signal_kind=signal_kind)
        self.loader = SleepDataLoader(self.config)

    def train_dataloader(self):
        return _LoaderAdapter(self.loader.train_loader, self.name, self.signal_kind)

    def val_dataloader(self):
        return _LoaderAdapter(self.loader.valid_loader, self.name, self.signal_kind)

    def test_dataloader(self):
        return _LoaderAdapter(self.loader.test_loader, self.name, self.signal_kind)
