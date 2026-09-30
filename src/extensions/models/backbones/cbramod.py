"""CBraMod backbone wrapper.

Window contract:
    CBraMod is patch-based (1 s patches at 200 Hz = 200 samples each). The
    wrapper accepts any input whose resampled length at 200 Hz is a positive
    multiple of 200 samples. In practice this is ``spec.expected_epoch_seconds
    × 200`` samples. Mismatched lengths raise ``ValueError`` — the wrapper
    never silently stretches, pads, or tiles.

Model-specific transforms applied in _prepare_input
(per MODEL_CONTRACTS.md §0/§2):
    1. Unit conversion to µV via ``unit_to_uv`` (handles V/mV/µV via
       ``meta[i]["unit"]``; missing key → "uV" with a one-shot
       DeprecationWarning).
    2. Strip all-zero channels (per §1; dataio zero-pads missing channels
       for batch-uniform shape).
    3. Drop EOG/EMG/ECG/RESP/EVENT/TEMP auxiliaries by label prefix.
    4. Resample to 200 Hz via polyphase (shared helper).
    5. Amplitude clip ``|x| > 100 µV`` + scale ``x / 100 → ~[-1, 1]``
       (matches CBraMod pretraining bad-sample filter and 100-µV norm).
       Gated by ``apply_amplitude_scale`` runtime override (default True).
    6. Finite check on output.

Weight loading (§4): manual safetensors download from HuggingFace Hub and
explicit allowlisted ``load_state_dict``. ``_ALLOWED_MISSING`` and
``_ALLOWED_UNEXPECTED`` start empty; any state-dict delta raises
``RuntimeError`` until verified and added to the allowlist.

Channel handling: CBraMod's Asymmetric Conditional Positional Encoding
(``Conv2d(kernel=(19,7), groups=d_model)``) is permutation-invariant
across channels and supports variable channel counts. Labels pass through
unchanged (paper §3.1 / MODEL_CONTRACTS §1: index-based, label
pass-through).

Signal cleanliness (bandpass / notch) is the dataset preprocessor's
responsibility, not this wrapper's. See AGENT_GUIDE.md §7.1.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import torch

from ._preproc import (
    _ESAT_DATASETS,
    assert_batch_homogeneity,
    assert_finite,
    read_sampling_rate,
    resample_poly_with_fallback,
    snap_to_epoch_length,
    strip_zero_channels,
    unit_to_uv,
)
from .base import BenchmarkModelWrapper
from benchmarking_helpers import BenchmarkBatch, CheckpointSpec


logger = logging.getLogger(__name__)

_TARGET_SFREQ = 200
_PRETRAIN_EPOCH_SECONDS = 30
_PATCH_SIZE = 200  # fixed by the pretrained encoder (1 s patches at 200 Hz)
_CLIP_UV = 100.0
_SCALE_DIVISOR = 100.0
_NON_EEG_PREFIXES: Tuple[str, ...] = (
    "EOG", "EMG", "ECG", "EKG", "RESP", "TEMP", "EVENT",
)

# Weight-load allowlist: starts empty per MODEL_CONTRACTS §4. The first run
# against a new checkpoint will raise with the observed deltas, which you
# copy in here after verifying they are benign (pretraining-only heads,
# renamed keys, etc.).
_ALLOWED_MISSING: Set[str] = set()
_ALLOWED_UNEXPECTED: Set[str] = set()


def _is_non_eeg(label: str) -> bool:
    if not label:
        return False
    normalized = label.strip().upper()
    for prefix in _NON_EEG_PREFIXES:
        if prefix in normalized:
            return True
    return False


def _select_eeg_channels(
    labels: Sequence[str], n_channels: int
) -> Tuple[List[int], List[str], List[Dict[str, str]], str]:
    """Decide which channel indices to keep.

    Strict contract — raises on missing or mismatched metadata. Callers must
    pass channel labels with exactly ``n_channels`` entries.

    Returns ``(kept_indices, kept_labels, dropped_entries, channel_mode)``
    where ``dropped_entries`` is a list of ``{"label", "reason"}`` dicts
    per MODEL_CONTRACTS.md §1.
    """
    labels = list(labels) if labels is not None else []
    if not labels:
        raise ValueError(
            "CBraMod: batch meta has no 'channels' list; cannot identify EEG "
            "channels. Populate batch['meta'][0]['channels'] with per-channel "
            "labels (or pass a fully-EEG tensor and set labels accordingly)."
        )
    if len(labels) != n_channels:
        raise ValueError(
            f"CBraMod: channel metadata length mismatch — got {len(labels)} "
            f"labels for {n_channels} tensor channels. Labels: {labels!r}."
        )

    kept_idx: List[int] = []
    kept_labels: List[str] = []
    dropped_entries: List[Dict[str, str]] = []
    for i, label in enumerate(labels):
        if _is_non_eeg(label):
            dropped_entries.append({"label": str(label), "reason": "non_eeg_prefix"})
        else:
            kept_idx.append(i)
            kept_labels.append(label)

    if not kept_idx:
        raise ValueError(
            f"CBraMod: all {n_channels} channels classified as non-EEG "
            f"(labels={labels!r}); nothing to embed."
        )

    channel_mode = "filtered_eog_emg" if dropped_entries else "all_eeg"
    return kept_idx, kept_labels, dropped_entries, channel_mode


def describe_channel_mapping(channels):
    """Name-only channel provenance (see channel_provenance.ChannelMapping).

    Reuses ``_select_eeg_channels`` (the exact inference path). CBraMod is
    permutation-invariant (channel-agnostic) — it keeps every EEG channel
    as-is and only drops non-EEG modalities by prefix; there is no fixed
    layout and therefore no zero-insertion. An all-non-EEG batch raises
    (captured upstream).
    """
    from benchmarking_helpers.channels.channel_provenance import (
        ChannelMapping,
        ChannelSlot,
    )

    _kept_idx, kept_labels, dropped_entries, _mode = _select_eeg_channels(
        list(channels), len(channels)
    )
    slots = [ChannelSlot(target=str(n), source=str(n), status="filled") for n in kept_labels]
    dropped = [
        ChannelSlot(target=str(d["label"]), source=str(d["label"]), status="dropped", note=d["reason"])
        for d in dropped_entries
    ]
    return ChannelMapping(
        model_family="cbramod", layout_kind="agnostic", slots=slots, dropped=dropped
    )


def _strict_load_with_allowlist(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
    """Load weights; raise unless the missing/unexpected deltas are allowlisted."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if k not in _ALLOWED_MISSING]
    real_unexpected = [k for k in unexpected if k not in _ALLOWED_UNEXPECTED]
    if real_missing or real_unexpected:
        raise RuntimeError(
            "CBraMod weight load: unaccounted-for state_dict deltas.\n"
            f"  missing (not in allowlist): {real_missing[:20]}\n"
            f"  unexpected (not in allowlist): {real_unexpected[:20]}\n"
            "After verifying benign, add to _ALLOWED_MISSING / _ALLOWED_UNEXPECTED."
        )
    return {
        "missing_keys_allowed": list(missing),
        "unexpected_keys_allowed": list(unexpected),
    }


class CBraModBackbone(BenchmarkModelWrapper):
    """Wrapper around braindecode's CBraMod for benchmark embedding extraction.

    Input contract (from braindecode 1.4.0 ``CBraMod.forward``):
        tensor shape ``(B, C, T)`` with T a positive multiple of 200 samples
        (1 s patches at 200 Hz). The model does the patch split
        ``(n_patch, patch_size=200)`` internally via einops.

    Output contract (with ``return_encoder_output=True``):
        tensor shape ``(B, C, n_patch, 200)``. We mean-pool over ``(C, n_patch)``
        to yield the ``(B, 200)`` embedding advertised by the checkpoint spec.
    """

    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        from ._braindecode_compat import ensure_torchaudio_stub
        ensure_torchaudio_stub()
        from braindecode.models import CBraMod

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.epoch_seconds = float(spec.expected_epoch_seconds)
        raw_target_len = int(round(_TARGET_SFREQ * self.epoch_seconds))
        if raw_target_len <= 0:
            raise ValueError(
                f"CBraMod: expected_epoch_seconds={self.epoch_seconds} yields "
                f"non-positive target length."
            )
        if raw_target_len % _PATCH_SIZE != 0:
            raise ValueError(
                f"CBraMod: expected_epoch_seconds={self.epoch_seconds} does "
                f"not produce a patch-aligned length at 200 Hz (got "
                f"{raw_target_len} samples, patch={_PATCH_SIZE}). Use a "
                f"multiple of 1 s."
            )
        self.target_len = raw_target_len

        if getattr(spec, "source_type", "") == "random_init":
            self.model = CBraMod(return_encoder_output=True)
            self._weight_source = "random_init"
            self._load_report: Dict[str, List[str]] = {
                "missing_keys_allowed": [],
                "unexpected_keys_allowed": [],
            }
        else:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            model_id = spec.checkpoint_path or "braindecode/cbramod-pretrained"
            weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
            state = load_file(weights_path)
            self.model = CBraMod(return_encoder_output=True)
            self._load_report = _strict_load_with_allowlist(self.model, state)
            self._weight_source = "pretrained"

        self.model.eval()
        self.model.to(self.device)

        self._unit_warned = False
        self._scale_off_warned = False
        self._banner_logged = False
        self._last_channel_report: Dict[str, Any] = {
            "input_channels_used": [],
            "dropped_channels": [],
            "channel_mode": "unknown",
            "input_sfreq_observed": None,
            "input_resample_method": "unknown",
            "clip_fraction": 0.0,
            "scale_divisor": _SCALE_DIVISOR,
            "input_unit": "unknown",
            "n_zero_channels_stripped": 0,
        }

    def _resolve_overrides(self) -> Dict[str, bool]:
        ov = getattr(self.spec, "runtime_overrides", {}) or {}
        return {
            "apply_amplitude_scale": bool(ov.get("apply_amplitude_scale", True)),
        }

    def _read_unit(self, batch: BenchmarkBatch) -> str:
        meta = batch.get("meta") or []
        first = meta[0] if meta else {}
        unit = first.get("unit") if isinstance(first, dict) else None
        if unit is None or unit == "":
            if not self._unit_warned:
                warnings.warn(
                    "CBraMod: batch meta[0]['unit'] is missing — defaulting to "
                    "'uV' for backward compatibility (MODEL_CONTRACTS.md §0). "
                    "Adapters should publish the canonical 'unit' key.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                self._unit_warned = True
            return "uV"
        return str(unit)

    def _log_banner_once(
        self, n_target_ch: int, unit: str, apply_amplitude_scale: bool
    ) -> None:
        if self._banner_logged:
            return
        logger.info(
            "[backbone=cbramod] fs=%d Hz window=%g s (patch-multiple of 1 s) "
            "unit=%s scale=/%g clip=|x|<%g µV apply_amplitude_scale=%s "
            "ref=keep n_target_ch=%d weight_source=%s",
            _TARGET_SFREQ, self.epoch_seconds, unit, _SCALE_DIVISOR, _CLIP_UV,
            apply_amplitude_scale, n_target_ch, self._weight_source,
        )
        import json as _json
        logger.info(
            "[backbone=cbramod][report] %s",
            _json.dumps(self._last_channel_report, default=str),
        )
        self._banner_logged = True

    def _prepare_input(self, batch: BenchmarkBatch) -> torch.Tensor:
        self.validate_batch(batch)
        try:
            x = self.require_signal(batch, "eeg")
        except KeyError:
            x = self.require_signal(batch, "full_signal")

        x = x.to(dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(
                f"CBraMod expected signal of shape (B, C, T) or (B, T); "
                f"got shape {tuple(x.shape)}."
            )

        meta = batch.get("meta") or []
        assert_batch_homogeneity(meta, where="cbramod:input")

        # ESAT-8 adaptive epoch_seconds: use 30 s from batch meta instead
        # of spec.expected_epoch_seconds (which may be 10 s on NeuroAtlas).
        dataset = meta[0].get("dataset", "") if meta else ""
        if dataset in _ESAT_DATASETS:
            epoch_sec = float(meta[0].get("epoch_seconds", 30.0))
        else:
            epoch_sec = self.epoch_seconds
        target_len = int(round(_TARGET_SFREQ * epoch_sec))

        # 1. Unit conversion to µV (per §0).
        unit = self._read_unit(batch)
        x = unit_to_uv(x, unit)

        # 2. Strip all-zero (zero-padded missing) channels (per §1).
        first_meta = meta[0] if meta else {}
        raw_labels = list(first_meta.get("channels") or [])
        if raw_labels and len(raw_labels) == x.shape[1]:
            x, raw_labels, _kept_idx_zero = strip_zero_channels(x, raw_labels)
        n_zero_stripped = (
            (len(first_meta.get("channels") or []) - x.shape[1])
            if first_meta.get("channels")
            else 0
        )

        # 3. EEG/non-EEG selection (per §1: drop auxiliary by label prefix).
        keep_idx, kept_labels, dropped_entries, channel_mode = _select_eeg_channels(
            raw_labels, n_channels=x.shape[1]
        )
        if keep_idx != list(range(x.shape[1])):
            x = x[:, keep_idx, :]

        # 4. Resample to 200 Hz. Read fs from meta (canonical 'sampling_rate'),
        # fall back to length-based inference if missing.
        current_len = x.shape[-1]
        meta_fs = read_sampling_rate(first_meta) if first_meta else None
        if meta_fs is not None and meta_fs > 0:
            src_sfreq_f = float(meta_fs)
            expected_T = int(round(src_sfreq_f * epoch_sec))
            if abs(current_len - expected_T) > 1:
                raise ValueError(
                    f"CBraMod: duration mismatch — meta['sampling_rate']="
                    f"{src_sfreq_f} Hz and epoch_seconds={epoch_sec:g} s "
                    f"imply {expected_T} samples, but got T={current_len}. "
                    "Upstream preprocessor contract is broken."
                )
        else:
            src_sfreq_f = current_len / epoch_sec

        _backend = "scipy" if dataset in _ESAT_DATASETS else "auto"
        x, resample_method = resample_poly_with_fallback(
            x, src_sfreq_f, float(_TARGET_SFREQ), backend=_backend
        )
        if dataset in _ESAT_DATASETS:
            x = snap_to_epoch_length(x, float(_TARGET_SFREQ), meta)
        if x.shape[-1] != target_len:
            raise ValueError(
                f"CBraMod: after resampling to 200 Hz, got length "
                f"{x.shape[-1]} but expected {target_len} "
                f"(= {_TARGET_SFREQ} × {epoch_sec} s). "
                f"Input length was {current_len} at fs "
                f"{src_sfreq_f:.3f} Hz."
            )

        # 5. Amplitude clip + scale (gated by apply_amplitude_scale override).
        overrides = self._resolve_overrides()
        if overrides["apply_amplitude_scale"]:
            clipped = (x.abs() > _CLIP_UV).float().mean().item()
            x = x.clamp(-_CLIP_UV, _CLIP_UV)
            x = x / _SCALE_DIVISOR
        else:
            clipped = 0.0
            if not self._scale_off_warned:
                logger.warning(
                    "CBraMod: apply_amplitude_scale=False — skipping ±100 µV "
                    "clip and ÷100 scale. Embedding is OFF-DISTRIBUTION; "
                    "diagnostic-only. Will not re-warn this run."
                )
                self._scale_off_warned = True

        x = x.to(self.device)
        assert_finite(x, "cbramod:output")

        self._last_channel_report = {
            "input_channels_used": list(kept_labels),
            "dropped_channels": list(dropped_entries),
            "channel_mode": channel_mode,
            "input_sfreq_observed": float(src_sfreq_f),
            "input_resample_method": resample_method,
            "clip_fraction": clipped,
            "scale_divisor": _SCALE_DIVISOR,
            "input_unit": unit,
            "n_zero_channels_stripped": int(n_zero_stripped),
            "apply_amplitude_scale": overrides["apply_amplitude_scale"],
        }
        self._log_banner_once(
            len(kept_labels), unit, overrides["apply_amplitude_scale"]
        )
        return x

    def extract_embeddings(self, batch: BenchmarkBatch) -> np.ndarray:
        x = self._prepare_input(batch)
        with torch.inference_mode():
            out = self.model(x)
        if out.ndim != 4:
            raise RuntimeError(
                f"CBraMod encoder output expected 4-D (B, C, n_patch, emb_dim); "
                f"got shape {tuple(out.shape)}."
            )
        features = out.mean(dim=(1, 2))
        return features.detach().float().cpu().numpy()

    def extract_embeddings_perpatch(self, batch: BenchmarkBatch) -> np.ndarray:
        x = self._prepare_input(batch)
        with torch.inference_mode():
            out = self.model(x)
        if out.ndim != 4:
            raise RuntimeError(
                f"CBraMod encoder output expected 4-D (B, C, n_patch, emb_dim); "
                f"got shape {tuple(out.shape)}."
            )
        B, C, n_patch, emb_dim = out.shape
        tokens = out.reshape(B, C * n_patch, emb_dim)
        return tokens.detach().float().cpu().numpy()

    def metadata(self) -> Dict[str, Any]:
        return {
            **super().metadata(),
            "device": self.device,
            "weight_source": self._weight_source,
            "weight_load_report": dict(self._load_report),
            "pretrain_sfreq": _TARGET_SFREQ,
            "pretrain_epoch_seconds": _PRETRAIN_EPOCH_SECONDS,
            "input_epoch_seconds": self.epoch_seconds,
            "input_target_length": self.target_len,
            "input_mode": (
                "native_window_s"
                if abs(self.epoch_seconds - _PRETRAIN_EPOCH_SECONDS) > 1e-6
                else "pretrain_default"
            ),
            "n_patches": self.target_len // _PATCH_SIZE,
            "embedding_reduction": "mean_over_channels_and_patches",
            "wrapper_contract": "model_specific_transforms_only",
            **self._last_channel_report,
        }
