from __future__ import annotations

import logging
from math import ceil
from pathlib import Path

import numpy as np
import torch

from neuroatlas.benchmarking_helpers import CheckpointSpec

from ._preproc import _PAPER_PREPROC_DATASETS, resample_poly_with_fallback
from .base import BenchmarkBackbone
from .core_sleep import _best_channel_index
from .sleepyco_arch import MainModel

_LOG = logging.getLogger(__name__)

# Pretraining recipe (SHHS finetune, configs/SleePyCo-Transformer_SL-10_numScales-3_SHHS_freezefinetune.json)
_TARGET_SFREQ = 100.0
_EPOCH_SECONDS = 30.0
_SEQUENCE_LENGTH = 10
_EPOCH_SAMPLES = int(_TARGET_SFREQ * _EPOCH_SECONDS)        # 3000
_TARGET_LEN = _EPOCH_SAMPLES * _SEQUENCE_LENGTH              # 30000


def _shhs_finetune_config() -> dict:
    """Mirror of the upstream SHHS freezefinetune JSON."""
    return {
        "dataset": {
            "name": "SHHS",
            "eeg_channel": "C4-A1",
            "num_splits": 1,
            "seq_len": _SEQUENCE_LENGTH,
            "target_idx": -1,
            "root_dir": "./",
        },
        "backbone": {"name": "SleePyCo", "init_weights": False, "dropout": False},
        "feature_pyramid": {"dim": 128, "num_scales": 3},
        "classifier": {
            "name": "Transformer",
            "model_dim": 128,
            "feedforward_dim": 128,
            "pool": "attn",
            "dropout": False,
            "num_classes": 5,
            "pos_enc": {"dropout": False},
        },
        "training_params": {"mode": "freezefinetune"},
    }


def _strip_module_prefix(state_dict: dict) -> dict:
    """The official SHHS checkpoint was saved through DataParallel — strip ``module.``."""
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_sleepyco_model(checkpoint_path: str | Path, device: str | None = None) -> torch.nn.Module:
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MainModel(_shhs_finetune_config())
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"no SleePyCo weights at {checkpoint_path}\n"
            "fix: neuroatlas models download sleepyco_shhs_fold0"
        )
    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


class SleePyCoBackbone(BenchmarkBackbone):
    """SHHS-pretrained SleePyCo (feature-pyramid CNN + Transformer head).

    Window contract: 10 × 30 s sleep epochs (= 300 s) of single-channel EEG at
    100 Hz, fed as a flat tensor of shape ``(B, 1, 30000)``. The CNN reads the
    sequence as one continuous signal — that is the recipe used during the
    upstream finetune. When the source sampling rate differs from 100 Hz the
    wrapper resamples via polyphase FIR (``resample_poly_with_fallback``).

    Native head emits 5-class logits (W, N1, N2, N3, REM). Embedding is the
    attention-pooled Transformer output averaged across the 3 feature-pyramid
    scales.
    """

    _logged = False

    def __init__(self, spec: CheckpointSpec) -> None:
        super().__init__(spec)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if abs(float(spec.expected_epoch_seconds) - _EPOCH_SECONDS) > 1e-6:
            raise ValueError(
                f"SleePyCo: expected_epoch_seconds={spec.expected_epoch_seconds:g} "
                f"unsupported; SHHS pretraining used {_EPOCH_SECONDS:g} s sleep epochs."
            )
        if int(spec.expected_sequence_length) not in (1, _SEQUENCE_LENGTH):
            raise ValueError(
                f"SleePyCo: expected_sequence_length={spec.expected_sequence_length} "
                f"unsupported; only 1 (independent epochs) and {_SEQUENCE_LENGTH} are accepted."
            )
        self.target_sfreq = _TARGET_SFREQ
        self.epoch_seconds = _EPOCH_SECONDS
        self.sequence_length = int(spec.expected_sequence_length)
        self.target_len = _EPOCH_SAMPLES * self.sequence_length
        self.model = load_sleepyco_model(spec.checkpoint_path or "", device=self.device)
        if not SleePyCoBackbone._logged:
            _LOG.info(
                "SleePyCo wrapper: SHHS-finetuned, 10 × 30 s @ 100 Hz "
                "(input shape (B, 1, 30000)); strict load."
            )
            SleePyCoBackbone._logged = True

    def _coerce_input(self, batch) -> torch.Tensor:
        signals = batch["signals"]
        if "eeg" in signals:
            eeg = signals["eeg"].to(dtype=torch.float32)
        else:
            raise KeyError(
                "SleePyCo expects batch['signals']['eeg'] (raw 1-channel EEG)."
            )

        meta = batch.get("meta") or [{}]
        ch_idx = 0
        ch_list = (meta[0] if isinstance(meta, list) else meta).get("channels")
        if ch_list:
            ch_idx = _best_channel_index(ch_list, signal=eeg)

        # Accept (B, T), (B, 1, T), or (B, S, C, T) where S=sequence_length
        if eeg.ndim == 2:
            eeg = eeg.unsqueeze(1)
        elif eeg.ndim == 4:
            B, S, C, T = eeg.shape
            if S != self.sequence_length:
                raise ValueError(
                    f"SleePyCo: sequence dim {S} != expected {self.sequence_length}."
                )
            eeg = eeg[:, :, ch_idx : ch_idx + 1, :]
            eeg = eeg.reshape(B, 1, S * T)
        if eeg.ndim == 3 and eeg.shape[1] > 1:
            eeg = eeg[:, ch_idx : ch_idx + 1, :]

        # Resample to 100 Hz when source rate differs.
        m0 = meta[0] if isinstance(meta, list) else meta
        src_fs = m0.get("sampling_rate") or m0.get("sfreq")
        if src_fs is None:
            src_fs = eeg.shape[-1] / (self.sequence_length * self.epoch_seconds)
        src_fs = float(src_fs)
        dataset = m0.get("dataset", "")
        _backend = "scipy" if dataset in _PAPER_PREPROC_DATASETS else "auto"
        if abs(src_fs - self.target_sfreq) > 1e-6:
            eeg, _ = resample_poly_with_fallback(eeg, src_fs, self.target_sfreq, backend=_backend)
            # Snap to exact target length within 0.3 % tolerance.
            diff = eeg.shape[-1] - self.target_len
            max_snap = max(2, ceil(self.target_len * 0.003))
            if 0 < diff <= max_snap:
                eeg = eeg[..., : self.target_len]
            elif -max_snap <= diff < 0:
                eeg = torch.nn.functional.pad(eeg, (0, -diff))

        if eeg.shape[-1] != self.target_len:
            raise ValueError(
                f"SleePyCo: window length mismatch — expected {self.target_len} samples "
                f"({self.sequence_length} epochs × {self.epoch_seconds:g} s × {self.target_sfreq:g} Hz), "
                f"got {eeg.shape[-1]} (src_fs={src_fs:.1f} Hz)."
            )
        return eeg.to(self.device, non_blocking=True)

    def _forward(self, batch):
        x = self._coerce_input(batch)
        with torch.inference_mode():
            return self.model(x)

    def native_head_logits(self, batch) -> np.ndarray:
        logits, _ = self._forward(batch)
        return logits.detach().cpu().numpy()

    def extract_embeddings(self, batch) -> np.ndarray:
        _, pooled = self._forward(batch)
        return pooled.detach().cpu().numpy()

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            "device": self.device,
            "target_sfreq": self.target_sfreq,
            "epoch_seconds": self.epoch_seconds,
            "sequence_length": self.sequence_length,
            "target_len": self.target_len,
        }
