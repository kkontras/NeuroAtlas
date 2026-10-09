from __future__ import annotations

import logging
import re
from pathlib import Path

import easydict
import numpy as np
import torch

from .core_sleep_model import Sleep_CoRe, SleepEnc

from .base import BenchmarkBackbone
from ._preproc import _PAPER_PREPROC_DATASETS, resample_poly_with_fallback, snap_to_epoch_length
from neuroatlas.benchmarking_helpers import CheckpointSpec
from neuroatlas.benchmarking_helpers.runtime.sequential_epochs import rows_per_labelled_epoch

_LOG = logging.getLogger(__name__)

_TARGET_SFREQ = 100.0
_REQUIRED_EPOCH_SECONDS = 30.0
_TARGET_LEN = int(_TARGET_SFREQ * _REQUIRED_EPOCH_SECONDS)  # 3000 samples
_STFT_N_FFT = 256
_STFT_HOP = 103
_STFT_FRAMES = 29  # SHHS pretraining format

_SLEEP_CHANNEL_PREFERENCE = [
    "C4-A1", "C4-M1", "C4_M1", "C4A1", "C4",
    "C3-A2", "C3-M2", "C3_M2", "C3A2", "C3",
    "EEG Fpz-Cz", "Fpz-Cz", "FPZ", "FZ",
    "EEG Pz-Oz", "Pz-Oz", "PZ", "CZ",
]


def _best_channel_index(channels, preference=_SLEEP_CHANNEL_PREFERENCE, signal=None):
    """Select the best EEG channel from *channels* by preference order.

    When *signal* is provided (shape ``(B, C, T)`` or ``(B, N, C, T)``),
    channels whose signal is identically zero across the batch are skipped.
    """
    norm = {ch.upper().replace(" ", "").replace("_", "-"): i for i, ch in enumerate(channels)}
    if signal is not None:
        if signal.ndim == 4:
            live = signal.abs().sum(dim=(0, 1, 3)) > 0
        else:
            live = signal.abs().sum(dim=(0, 2)) > 0
    else:
        live = None
    for pref in preference:
        key = pref.upper().replace(" ", "").replace("_", "-")
        if key in norm:
            idx = norm[key]
            if live is None or (idx < len(live) and live[idx]):
                return idx
    if live is not None:
        for i in range(len(live)):
            if live[i]:
                return i
    return 0


def _encoder_args(modality: str) -> easydict.EasyDict:
    return easydict.EasyDict({
        "dmodel": 128,
        "modality": modality,
        "CA_flag": True,
        "dim_proj": 128,
        "dim_feedforward": 1024,
        "dropout": 0.3,
        "rpos": True,
        "pos": False,
    })


def _model_args() -> easydict.EasyDict:
    return easydict.EasyDict({
        "d_model": 128,
        "num_classes": 5,
        "fc_inner": 64,
        "dim_proj": 128,
        "dim_feedforward": 1024,
        "dropout": 0.3,
        "rpos": True,
        "pos": False,
        "multi_loss": {
            "multi_supervised_w": {"combined": 1, "c": 1, "g": 1, "align": 0.1}
        },
    })


def load_core_sleep_model(checkpoint_path: str | Path, device: str | None = None,
                          identifier: str | None = None) -> torch.nn.Module:
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    enc_eeg = SleepEnc(args=_encoder_args("eeg"))
    enc_eog = SleepEnc(args=_encoder_args("eog"))
    model = Sleep_CoRe(args=_model_args(), encs=[enc_eeg, enc_eog])
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        command = (f"neuroatlas models download {identifier}" if identifier
                   else "neuroatlas models status (each checkpoint's weights, and the "
                        "command that fetches them)")
        raise FileNotFoundError(
            f"no weights for {identifier or 'CoRe-Sleep'} at {checkpoint_path}\n"
            f"fix: {command}"
        )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.to(device)
    return model


def _resolve_feature_tensor(output, embedding_key: str):
    candidates = [
        embedding_key,
        f"features.{embedding_key}",
        "features.combined",
        "embeddings",
        "embedding",
        "aggregated_token",
    ]
    for candidate in candidates:
        current = output
        found = True
        for part in candidate.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break
        if found:
            return current
    raise KeyError(
        f"Could not resolve embedding key {embedding_key!r} from model output. "
        f"Available top-level keys: {sorted(output.keys())}"
    )


# The per-frequency mean/std of the log STFT on SHHS that CoRe-Sleep and
# SleepTransformer were trained with (2.5 KB). It ships with the package: it
# used to be read from <models root>/shhs/, where nothing put it, so a run
# with the weights in place failed on it after extraction had started.
_STFT_NORM_PATH = Path(__file__).with_name("data") / "stft_norm_eeg.npz"


class CoreSleepBackbone(BenchmarkBackbone):
    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.epoch_seconds = float(spec.expected_epoch_seconds)
        if abs(self.epoch_seconds - _REQUIRED_EPOCH_SECONDS) > 1e-6:
            raise ValueError(
                f"CoRe-Sleep: epoch_seconds={self.epoch_seconds:g} not supported. "
                f"Pretrained on SHHS 30 s epochs; only epoch_seconds=30 is accepted."
            )
        self.target_sfreq = _TARGET_SFREQ
        self.target_len = _TARGET_LEN
        from ._checkpoint_download import ensure_checkpoint
        checkpoint_path = ensure_checkpoint(
            spec.checkpoint_path or "",
            source_type=spec.source_type,
            source_reference=spec.source_reference,
            identifier=spec.identifier,
        )
        self.model = load_core_sleep_model(checkpoint_path, device=self.device,
                                           identifier=spec.identifier)
        nf = np.load(str(_STFT_NORM_PATH))
        self._stft_log_norm = (
            torch.from_numpy(nf["mean"]).float().to(self.device),
            torch.from_numpy(nf["std"]).float().to(self.device),
        )

    def _to_device(self, value):
        if torch.is_tensor(value):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, dict):
            return {key: self._to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._to_device(item) for item in value)
        return value

    @staticmethod
    def _raw_to_stft(
        eeg_tensor: torch.Tensor,
        src_fs: float,
        channel_idx: int = 0,
        meta: list | None = None,
        log_norm: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute STFT from raw EEG to match the SHHS pretraining format.

        Input:  (B, C, T) raw EEG at ``src_fs`` Hz (must cover 30 s exactly).
        Output: (B, 1, 1, 1, 129, 29) — the 6D format Sleep_CoRe expects.

        When *log_norm* is provided as ``(mean, std)`` tensors of shape
        ``(129,)``, applies the SHHS pretraining normalization:
        ``20 * log10(|STFT|)`` followed by per-frequency-bin z-score.
        """
        x = eeg_tensor[:, channel_idx : channel_idx + 1, :]  # (B, 1, T)
        T = x.shape[-1]
        dataset = meta[0].get("dataset", "") if meta else ""
        _backend = "scipy" if dataset in _PAPER_PREPROC_DATASETS else "auto"
        x, _ = resample_poly_with_fallback(x, float(src_fs), _TARGET_SFREQ, backend=_backend)
        x = snap_to_epoch_length(x, _TARGET_SFREQ, meta)
        if x.shape[-1] != _TARGET_LEN:
            raise ValueError(
                f"CoRe-Sleep: window length mismatch — expected {_TARGET_LEN} samples "
                f"(30 s × {_TARGET_SFREQ:g} Hz), got {x.shape[-1]}. "
                f"Input was {T} samples at fs {src_fs:.3f} Hz."
            )
        x = x.squeeze(1)  # (B, _TARGET_LEN)
        window = torch.hann_window(_STFT_N_FFT, device=x.device)
        stft = torch.stft(
            x, n_fft=_STFT_N_FFT, hop_length=_STFT_HOP,
            win_length=_STFT_N_FFT, window=window, return_complex=True,
        )
        mag = stft.abs()  # (B, 129, n_frames)
        if log_norm is not None:
            norm_mean, norm_std = log_norm[0].to(mag.device), log_norm[1].to(mag.device)
            mag = 20.0 * torch.log10(mag.clamp(min=1e-10))
            mag = (mag - norm_mean[:, None]) / norm_std[:, None]
        # STFT of 3000 samples with center=True yields 30 frames; SHHS
        # pretraining format uses exactly 29 — drop the trailing frame.
        if mag.shape[-1] != _STFT_FRAMES + 1:
            raise ValueError(
                f"CoRe-Sleep: unexpected STFT frame count {mag.shape[-1]} "
                f"(expected {_STFT_FRAMES + 1} for a 30 s input at 100 Hz)."
            )
        mag = mag[:, :, :_STFT_FRAMES]
        # The model slices freq as [:, :, :, :, 1:, :] (removes DC bin).
        return mag.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 1, 129, 29)

    @staticmethod
    def _resolve_src_fs(meta, T, epoch_seconds):
        """Read sampling rate from meta, falling back to T / epoch_seconds."""
        if meta:
            m0 = meta[0] if isinstance(meta, list) else meta
            sr = m0.get("sampling_rate") or m0.get("sfreq")
            if sr is not None:
                return float(sr)
        return T / epoch_seconds

    def _forward(self, batch):
        """Run the bimodal model the way the paper's embeddings were made.

        CoRe-Sleep is an EEG+EOG model; the benchmark supplies EEG only, and
        the paper ran it with an all-zero EOG spectrogram through the full
        bimodal forward, reading the fused ``combined`` features and head
        (EEGBenchmarks @ fe49e60, which wrote the published Sleep-EDF cache).
        That is reproduced here: over the 457,947 Sleep-EDF rows the fresh
        embeddings differ from the paper's by 1e-6 on average (correlation
        1.0). The model's EEG-only path (``skip_view="eog"``, EOG branch never
        built, ``features["eeg"]``) gives different embeddings (correlation
        0.56 with the paper's) and is not what the paper reports.
        """
        raw_batch = batch.get("raw_batch")
        if isinstance(raw_batch, dict) and "data" in raw_batch:
            model_inputs = raw_batch["data"]
        else:
            signals = batch["signals"]
            if "stft_eeg" in signals:
                stft_eeg = signals["stft_eeg"]
                stft_eog = signals.get("stft_eog")
                if stft_eog is None:
                    stft_eog = torch.zeros_like(stft_eeg)
                model_inputs = {"stft_eeg": stft_eeg, "stft_eog": stft_eog}
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
                    src_fs = self._resolve_src_fs(meta, T, self.epoch_seconds)
                    flat = eeg.reshape(B * N, C, T)
                    flat_stft = self._raw_to_stft(flat, src_fs=src_fs, channel_idx=ch_idx, meta=meta, log_norm=self._stft_log_norm)
                    # flat_stft: (B*N, 1, 1, 1, 129, 29) → (B, N, 1, 1, 129, 29)
                    stft_eeg = flat_stft.view(B, N, *flat_stft.shape[2:])
                else:
                    if eeg.ndim == 2:
                        eeg = eeg.unsqueeze(1)
                    T = eeg.shape[-1]
                    src_fs = self._resolve_src_fs(meta, T, self.epoch_seconds)
                    stft_eeg = self._raw_to_stft(eeg, src_fs=src_fs, channel_idx=ch_idx, meta=meta, log_norm=self._stft_log_norm)

                # The paper's EEG-only protocol: a zero EOG spectrogram through
                # the bimodal model (see the docstring).
                model_inputs = {"stft_eeg": stft_eeg, "stft_eog": torch.zeros_like(stft_eeg)}
            else:
                raise KeyError(
                    "CoreSleep expects batch['signals']['stft_eeg'] or "
                    "batch['signals']['eeg'] (raw, to compute STFT on the fly)."
                )
        model_inputs = self._to_device(model_inputs)
        with torch.inference_mode():
            return self.model(model_inputs)

    def native_head_logits(self, batch) -> np.ndarray:
        """The fused head's logits, one row per labelled epoch (``(B * W, 5)``
        for a window batch with every epoch labelled)."""
        output = self._forward(batch)
        logits = output["preds"]["combined"]
        return rows_per_labelled_epoch(logits, batch).detach().cpu().numpy()

    def extract_embeddings(self, batch) -> np.ndarray:
        """``combined`` features, one row per labelled epoch: ``(B * W, 128)``
        for a batch of ``B`` windows of ``W`` epochs, the same rows as the
        batch's labels and meta."""
        output = self._forward(batch)
        features = _resolve_feature_tensor(output, self.spec.embedding_key)
        return rows_per_labelled_epoch(features, batch).detach().cpu().numpy()

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            "device": self.device,
            "target_sfreq": self.target_sfreq,
            "epoch_seconds": self.epoch_seconds,
            "target_len": self.target_len,
        }
