"""EEGNetv4 backbone wrapper for the PierreGtch/EEGNetv4 pretrained checkpoints.

Each checkpoint is a small `braindecode.models.EEGNetv4` trained on one MOABB
motor-imagery dataset with the following pretraining contract (taken from the
model card + notebook at neurotechlab.socsci.ru.nl/resources/pretrained_imagery_models):

- 3 central motor channels: C3, Cz, C4
- Resample to 128 Hz
- 3 s windows (385 samples — the model ships with `input_window_samples=385`)
- Bandpass 0.5 - 40 Hz

The HF checkpoints pickle two files per variant:

- `kwargs.pkl`: `{module_cls, module_kwargs}` where `module_cls` is the legacy
  `braindecode.models.eegnet.EEGNetv4` and `module_kwargs` uses the legacy arg
  names `in_chans`, `n_classes`, `input_window_samples`.
- `model-params.pkl`: a plain `state_dict`.

Current braindecode (>= 1.4) renamed the module to `EEGNet`, added a
`max_norm` parametrization on `conv_spatial.weight`, and moved the classifier
under a `final_layer` submodule. We translate both the kwargs and the
state-dict key names on load so the pretrained weights land in the right
tensors.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt

from neuroatlas.benchmarking_helpers import BenchmarkBatch, CheckpointSpec

from .base import BenchmarkModelWrapper

_TARGET_CHANNELS: tuple[str, ...] = ("C3", "CZ", "C4")
_TARGET_SFREQ = 128.0
_BANDPASS = (0.5, 40.0)

_LEGACY_KW_ALIAS = {
    "in_chans": "n_chans",
    "n_classes": "n_outputs",
    "input_window_samples": "n_times",
}

_CHANNEL_ALIASES = {
    "C3": "C3",
    "CZ": "CZ",
    "C4": "C4",
    "C3-A2": "C3",
    "C4-A1": "C4",
    "EEG C3": "C3",
    "EEG CZ": "CZ",
    "EEG C4": "C4",
}


def _normalize_channel_name(name: str | None) -> str:
    if name is None:
        return ""
    candidate = str(name).strip().upper().replace("_", "-")
    candidate = " ".join(candidate.split())
    if candidate.startswith("EEG "):
        candidate = candidate[4:]
    candidate = candidate.replace(" ", "")
    return _CHANNEL_ALIASES.get(candidate, candidate)


def _infer_batch_channels(batch: BenchmarkBatch, n_chans: int) -> List[str]:
    meta = batch.get("meta", [])
    if meta and isinstance(meta[0], Mapping):
        channels = meta[0].get("channels")
        if isinstance(channels, Sequence) and not isinstance(channels, (str, bytes)):
            return [str(name) for name in list(channels)[:n_chans]]
    return []


def _translate_state_dict(sd: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename legacy EEGNetv4 state-dict keys to current braindecode layout.

    - `conv_spatial.weight` → `conv_spatial.parametrizations.weight.original`
      (current braindecode wraps the depthwise spatial conv in a max-norm
      parametrization; the underlying weight lives under `.original`).
    - `conv_classifier.*` → `final_layer.conv_classifier.*`
      (the classifier was moved under a `final_layer` submodule).
    """
    translated: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        new_key = key
        if new_key == "conv_spatial.weight":
            new_key = "conv_spatial.parametrizations.weight.original"
        elif new_key.startswith("conv_classifier."):
            new_key = "final_layer." + new_key
        translated[new_key] = value
    return translated


def _bandpass_filter(x: torch.Tensor, sfreq: float) -> torch.Tensor:
    nyq = 0.5 * sfreq
    low = _BANDPASS[0] / nyq
    high = min(_BANDPASS[1] / nyq, 0.999)
    if low >= high:
        return x
    sos = butter(4, [low, high], btype="band", output="sos")
    x_np = x.detach().cpu().numpy()
    filtered = sosfiltfilt(sos, x_np, axis=-1)
    return torch.from_numpy(filtered.astype(x_np.dtype, copy=False)).to(x.device)


def _resample_by_sfreq(x: torch.Tensor, input_sfreq: float, target_sfreq: float) -> torch.Tensor:
    if abs(input_sfreq - target_sfreq) < 0.5:
        return x
    from math import gcd
    from scipy.signal import resample_poly

    up = int(round(target_sfreq))
    down = int(round(input_sfreq))
    g = gcd(up, down)
    x_np = x.detach().cpu().numpy()
    y_np = resample_poly(x_np, up // g, down // g, axis=-1)
    return torch.from_numpy(y_np.astype(x_np.dtype, copy=False)).to(x.device)


def _center_crop_or_pad(x: torch.Tensor, target_len: int) -> torch.Tensor:
    cur = x.shape[-1]
    if cur == target_len:
        return x
    if cur > target_len:
        start = (cur - target_len) // 2
        return x[..., start : start + target_len]
    pad = target_len - cur
    left = pad // 2
    right = pad - left
    return F.pad(x, (left, right))


def _map_to_motor_channels(
    x: torch.Tensor, channels: Sequence[str] | None
) -> tuple[torch.Tensor, Dict[str, object]]:
    n_source = x.shape[1]
    source_channels = list(channels or [])
    if len(source_channels) < n_source:
        source_channels.extend([""] * (n_source - len(source_channels)))
    else:
        source_channels = source_channels[:n_source]
    normalized = [_normalize_channel_name(name) for name in source_channels]

    source_index = {name: idx for idx, name in enumerate(normalized) if name}
    mapped = torch.zeros(
        (x.shape[0], len(_TARGET_CHANNELS), x.shape[-1]), dtype=x.dtype, device=x.device
    )
    matched: List[str] = []
    missing: List[str] = []
    for out_idx, target in enumerate(_TARGET_CHANNELS):
        src_idx = source_index.get(target)
        if src_idx is None:
            missing.append(target)
            continue
        mapped[:, out_idx] = x[:, src_idx]
        matched.append(target)
    report: Dict[str, object] = {
        "source_channels": [str(name) for name in source_channels],
        "matched_channels": matched,
        "missing_channels": missing,
        "target_channels": list(_TARGET_CHANNELS),
    }
    return mapped, report


def _load_kwargs(checkpoint_dir: Path) -> Dict[str, object]:
    kwargs_path = checkpoint_dir / "kwargs.pkl"
    if not kwargs_path.exists():
        raise FileNotFoundError(f"Missing kwargs.pkl at {kwargs_path}.")
    with kwargs_path.open("rb") as f:
        return pickle.load(f)


def _translate_kwargs(raw: Mapping[str, object]) -> Dict[str, object]:
    return {_LEGACY_KW_ALIAS.get(key, key): value for key, value in raw.items()}


def _build_model(module_kwargs: Mapping[str, object]):
    try:
        from braindecode.models import EEGNetv4
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "braindecode is required to load the PierreGtch/EEGNetv4 checkpoints."
        ) from exc
    return EEGNetv4(**dict(module_kwargs))


class EEGNetv4Backbone(BenchmarkModelWrapper):
    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.target_sfreq = float(spec.expected_sampling_rate or _TARGET_SFREQ)
        self.model, self.module_kwargs, self._load_report = self._load_model()
        self.target_len = int(self.module_kwargs["n_times"])
        self.feature_model = self._build_feature_model(self.model)
        self._last_mapping: Dict[str, object] = {}

    def _resolve_checkpoint_dir(self) -> Path:
        path = Path(self.spec.checkpoint_path or "")
        if not path.is_dir():
            raise FileNotFoundError(
                f"EEGNetv4 checkpoint directory not found: {path!r}. "
                "Download PierreGtch/EEGNetv4 snapshot under "
                "artifacts/models/foundation/eegnetv4/ first."
            )
        return path

    def _load_model(self):
        checkpoint_dir = self._resolve_checkpoint_dir()
        raw_kwargs = _load_kwargs(checkpoint_dir)
        module_kwargs = _translate_kwargs(raw_kwargs.get("module_kwargs", {}))
        model = _build_model(module_kwargs)
        state_file = checkpoint_dir / "model-params.pkl"
        payload = torch.load(str(state_file), map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"Unsupported EEGNetv4 checkpoint payload type {type(payload)!r} at {state_file}."
            )
        translated = _translate_state_dict(payload)
        missing, unexpected = model.load_state_dict(translated, strict=False)
        model.eval()
        model.to(self.device)
        report = {
            "weight_source": "local_huggingface_snapshot",
            "checkpoint_dir": str(checkpoint_dir),
            "resolved_weight_file": str(state_file),
            "missing_keys": list(missing[:10]),
            "unexpected_keys": list(unexpected[:10]),
            "module_kwargs": dict(module_kwargs),
        }
        return model, module_kwargs, report

    @staticmethod
    def _build_feature_model(model: nn.Module) -> nn.Sequential:
        """Classifier-free feature extractor: everything before `final_layer`."""
        children = [module for name, module in model.named_children() if name != "final_layer"]
        return nn.Sequential(*children).eval()

    def _prepare_input(self, batch: BenchmarkBatch) -> torch.Tensor:
        self.validate_batch(batch)
        try:
            x = self.require_signal(batch, "full_signal")
        except KeyError:
            x = self.require_signal(batch, "eeg")
        x = x.to(self.device, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"EEGNetv4 expects (B, C, T), got {tuple(x.shape)}.")

        meta = batch.get("meta", [{}])
        input_sfreq = meta[0].get("sfreq") if meta else None
        if input_sfreq is not None:
            x = _bandpass_filter(x, float(input_sfreq))
            x = _resample_by_sfreq(x, float(input_sfreq), self.target_sfreq)
        else:
            x = _bandpass_filter(x, self.target_sfreq)

        channels = _infer_batch_channels(batch, x.shape[1])
        x, report = _map_to_motor_channels(x, channels)
        self._last_mapping = report

        x = _center_crop_or_pad(x, self.target_len)
        return x

    def native_head_logits(self, batch: BenchmarkBatch) -> np.ndarray:
        x = self._prepare_input(batch)
        with torch.no_grad():
            logits = self.model(x)
        return logits.detach().cpu().numpy()

    def extract_embeddings(self, batch: BenchmarkBatch) -> np.ndarray:
        x = self._prepare_input(batch)
        with torch.no_grad():
            feats = self.feature_model(x)
        if feats.ndim > 2:
            feats = feats.reshape(feats.shape[0], -1)
        return feats.detach().cpu().numpy()

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            **self._load_report,
            **self._last_mapping,
            "device": self.device,
            "target_sampling_rate": self.target_sfreq,
            "target_length": self.target_len,
            "bandpass_hz": list(_BANDPASS),
        }
