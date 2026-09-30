from __future__ import annotations

import logging
from pathlib import Path

import easydict
import numpy as np
import torch

from .base import BenchmarkBackbone
from .core_sleep import (
    _REQUIRED_EPOCH_SECONDS,
    _TARGET_LEN,
    _TARGET_SFREQ,
    _best_channel_index,
    CoreSleepBackbone,
)
from .core_sleep_model import SleepEnc
from neuroatlas.benchmarking_helpers import CheckpointSpec

_LOG = logging.getLogger(__name__)


def _encoder_args() -> easydict.EasyDict:
    # Mirrors Trui/eeg_eoe.json model.args (the unimodal_eeg checkpoint).
    return easydict.EasyDict({
        "dmodel": 128,
        "modality": "eeg",
        "dim_proj": 128,
        "dim_feedforward": 1024,
        "dropout": 0.3,
        "rpos": True,
        "pos": False,
    })


def load_sleep_transformer_model(checkpoint_path: str | Path, device: str | None = None) -> torch.nn.Module:
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SleepEnc(args=_encoder_args())
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SleepTransformer checkpoint not found at {checkpoint_path}. "
            "Keep the local artifact in place or update the registry."
        )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.to(device)
    return model


class SleepTransformerBackbone(BenchmarkBackbone):
    """SHHS-pretrained EEG-only SleepEnc (same trunk as CoRe-Sleep, single stream).

    Window contract: 30 s @ 100 Hz, STFT (n_fft=256, hop=103, 29 frames). Mirrors
    CoreSleepBackbone — the only differences are (a) `SleepEnc(modality="eeg")`
    in place of the bimodal `Sleep_CoRe`, and (b) we feed a zero `stft_eog`
    alongside the real `stft_eeg` so SleepEnc takes the same axis-ordering
    branch (`b outer mod ch f inner -> ...`) as during pretraining; the
    EOG tensor is unused inside `SleepEnc.forward` when modality="eeg".
    """

    _logged = False

    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.epoch_seconds = float(spec.expected_epoch_seconds)
        if abs(self.epoch_seconds - _REQUIRED_EPOCH_SECONDS) > 1e-6:
            raise ValueError(
                f"SleepTransformer: epoch_seconds={self.epoch_seconds:g} not supported. "
                f"Pretrained on SHHS 30 s epochs; only epoch_seconds=30 is accepted."
            )
        self.target_sfreq = _TARGET_SFREQ
        self.target_len = _TARGET_LEN
        self.model = load_sleep_transformer_model(spec.checkpoint_path or "", device=self.device)
        from .core_sleep import _STFT_NORM_PATH
        nf = np.load(str(_STFT_NORM_PATH))
        self._stft_log_norm = (
            torch.from_numpy(nf["mean"]).float().to(self.device),
            torch.from_numpy(nf["std"]).float().to(self.device),
        )
        if not SleepTransformerBackbone._logged:
            _LOG.info(
                "SleepTransformer wrapper: SHHS-pretrained SleepEnc (eeg) — 100 Hz, "
                "30 s epochs, STFT 256/103/29; strict load."
            )
            SleepTransformerBackbone._logged = True

    def _to_device(self, value):
        return CoreSleepBackbone._to_device(self, value)

    def _forward(self, batch):
        signals = batch["signals"]
        if "stft_eeg" in signals:
            stft_eeg = signals["stft_eeg"]
        elif "eeg" in signals:
            eeg = signals["eeg"].to(dtype=torch.float32)
            meta = batch.get("meta") or [{}]

            ch_idx = 0
            ch_list = (meta[0] if isinstance(meta, list) else meta).get("channels")
            if ch_list:
                ch_idx = _best_channel_index(ch_list, signal=eeg)

            if eeg.ndim == 4:
                # Multi-epoch sequential input: (B, N, C, T)
                B, N, C, T = eeg.shape
                src_fs = CoreSleepBackbone._resolve_src_fs(meta, T, self.epoch_seconds)
                flat = eeg.reshape(B * N, C, T)
                flat_stft = CoreSleepBackbone._raw_to_stft(flat, src_fs=src_fs, channel_idx=ch_idx, meta=meta, log_norm=self._stft_log_norm)
                stft_eeg = flat_stft.view(B, N, *flat_stft.shape[2:])
            else:
                if eeg.ndim == 2:
                    eeg = eeg.unsqueeze(1)
                T = eeg.shape[-1]
                src_fs = CoreSleepBackbone._resolve_src_fs(meta, T, self.epoch_seconds)
                stft_eeg = CoreSleepBackbone._raw_to_stft(eeg, src_fs=src_fs, channel_idx=ch_idx, meta=meta, log_norm=self._stft_log_norm)
        else:
            raise KeyError(
                "SleepTransformer expects batch['signals']['stft_eeg'] or "
                "batch['signals']['eeg'] (raw, to compute STFT on the fly)."
            )
        model_inputs = {"stft_eeg": stft_eeg, "stft_eog": torch.zeros_like(stft_eeg)}
        model_inputs = self._to_device(model_inputs)
        with torch.inference_mode():
            return self.model(model_inputs)

    def native_head_logits(self, batch) -> np.ndarray:
        output = self._forward(batch)
        return output["preds"]["combined"].detach().cpu().numpy()

    def extract_embeddings(self, batch) -> np.ndarray:
        output = self._forward(batch)
        features = output["features"][self.spec.embedding_key]

        target = batch.get("_seq_target_idx")
        if target is not None and target != "all":
            eeg = batch["signals"]["eeg"]
            B, W = eeg.shape[0], eeg.shape[1]
            idx = int(target) % W
            features = features.view(B, W, -1)[:, idx, :]

        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        return features.detach().cpu().numpy()

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            "device": self.device,
            "target_sfreq": self.target_sfreq,
            "epoch_seconds": self.epoch_seconds,
            "target_len": self.target_len,
        }
