"""LaBraM backbone wrapper.

Window contract:
    LaBraM is patch-based at **200 Hz** with **200-sample patches
    (= 1 s per patch)**, per the paper (§3.2: *"For each EEG sample
    X ∈ R^{C×T} with C channels and T samples, we segment it into
    patches with patch size w"*; the paper uses 200-sample patches at
    200 Hz, i.e. 1 s / patch). The wrapper accepts any
    ``expected_epoch_seconds`` that is a positive multiple of 1 s —
    including non-30-s inputs (e.g. 4 s, 10 s). Mismatched lengths
    raise ``ValueError``; we never silently stretch, pad, or tile.

Model-specific transforms applied in extract_embeddings:
    1. Batch-meta homogeneity check (fs / unit / channels).
    2. ``unit_to_uv`` — convert declared unit to µV.
    3. fs resample to 200 Hz (shared helper).
    4. Amplitude scale x / 100 → ~[-1, 1] (0.1 mV pretraining range).
       Gated by ``runtime_overrides["apply_amplitude_scale"]``.
    5. Finite check on input tensor.

Channel handling: LaBraM's position embedding has 129 rows (CLS + 128
electrodes from ``LABRAM_CHANNEL_ORDER``). The wrapper maps batch channel
names to their canonical indices via ``_resolve_labram_channels`` so each
electrode gets its correct pretrained position embedding.

Pretraining constants: ``_PRETRAIN_N_TIME_PATCHES`` and
``_PRETRAIN_EPOCH_SECONDS`` are **derived from the checkpoint's
``student.time_embed`` shape** at load time (rather than hard-coded)
because the LaBraM paper uses different pretraining windows depending
on channel count (§3.2: *"4–16 s depending on channel count"*). The
checkpoint's time-embedding row count is the ground truth.

Signal cleanliness (bandpass 0.1–75 Hz, 50 Hz notch) is the dataset
preprocessor's responsibility, not this wrapper's. See AGENT_GUIDE.md §7.1.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import torch

from neuroatlas import quiet

from ._preproc import (
    _PAPER_PREPROC_DATASETS,
    StageTimer,
    assert_batch_homogeneity,
    assert_finite,
    is_bci_batch,
    read_sampling_rate,
    resample_poly_with_fallback,
    snap_to_epoch_length,
    strip_zero_channels,
    unit_to_uv,
)
from .base import BenchmarkBackbone
from neuroatlas.benchmarking_helpers import CheckpointSpec


logger = logging.getLogger(__name__)

# Pretraining constants:
#   student.pos_embed:   (1, 1 + N_chans, 200) → N_chans channel positions + CLS
#   student.time_embed:  (1, 1 + N_time_patches, 200) → time patches + CLS
# Paper §3.2: 200 Hz, 200-sample patches (= 1 s / patch). The number of
# time patches varies with pretraining window (4–16 s per channel count);
# the checkpoint's time_embed row count is authoritative.
_PRETRAIN_SFREQ = 200.0
_PRETRAIN_PATCH = 200  # samples per patch → 1 s at 200 Hz
_PRETRAIN_N_CHANS = 128
_SCALE_DIVISOR = 100.0  # 0.1 mV (= 100 µV) → [-1, 1]

# Weight-load allowlist: deltas we know about when mapping the original LaBRAM
# checkpoint onto braindecode's Labram module. Anything outside these sets is
# a real mismatch and raises.
_ALLOWED_MISSING: Set[str] = {
    # fc_norm is a braindecode-specific module not in the checkpoint; stays
    # at its init (weight=1, bias=0 → near-identity).
    "fc_norm.weight",
    "fc_norm.bias",
    # position_embedding and temporal_embedding are popped from the state
    # before load and re-assigned manually (sliced from 128-row / 16-row
    # pretrained tensors) because load_state_dict raises on shape mismatch
    # even under strict=False.
    "position_embedding",
    "temporal_embedding",
}
_ALLOWED_UNEXPECTED_PREFIXES: tuple = (
    "mask_token",
    "lm_head",
)
_ALLOWED_UNEXPECTED_SUBSTRINGS: tuple = (
    ".gamma_1",
    ".gamma_2",
    ".q_norm.",
    ".k_norm.",
    "logit_scale",
)


def _unwrap_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def _remap_labram_keys(state: dict) -> dict:
    """Convert original LaBRAM checkpoint keys (student.*) to braindecode names."""
    remapped: dict = {}
    for k, v in state.items():
        if not k.startswith("student."):
            continue
        k = k[len("student."):]
        k = k.replace("pos_embed", "position_embedding")
        k = k.replace("time_embed", "temporal_embedding")
        if k.startswith("patch_embed."):
            suffix = k[len("patch_embed."):]
            if suffix.startswith(("conv", "norm")):
                k = "patch_embed.temporal_conv." + suffix
        k = k.replace(".mlp.fc1.", ".mlp.0.")
        k = k.replace(".mlp.fc2.", ".mlp.2.")
        remapped[k] = v
    return remapped


def _filter_allowed(keys: List[str], allowed_literals: Set[str],
                    allowed_prefixes: tuple, allowed_substrings: tuple) -> List[str]:
    unexpected = []
    for k in keys:
        if k in allowed_literals:
            continue
        if any(k.startswith(p) for p in allowed_prefixes):
            continue
        if any(s in k for s in allowed_substrings):
            continue
        unexpected.append(k)
    return unexpected


def _strict_load_with_allowlist(model, state: dict) -> Dict[str, List[str]]:
    """Load weights; raise unless the missing/unexpected deltas are all allowlisted."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = _filter_allowed(list(missing), _ALLOWED_MISSING, (), ())
    real_unexpected = _filter_allowed(
        list(unexpected), set(), _ALLOWED_UNEXPECTED_PREFIXES, _ALLOWED_UNEXPECTED_SUBSTRINGS,
    )
    if real_missing or real_unexpected:
        raise RuntimeError(
            "LaBraM weight load: unaccounted-for state_dict deltas.\n"
            f"  missing (not in allowlist): {real_missing}\n"
            f"  unexpected (not in allowlist): {real_unexpected}\n"
            "Update _ALLOWED_MISSING / _ALLOWED_UNEXPECTED_* after verifying "
            "the new checkpoint is compatible."
        )
    return {
        "missing_keys_allowed": list(missing),
        "unexpected_keys_allowed": list(unexpected),
    }


def _labram_match(ch_names: List[str]) -> tuple:
    """Pure name→index match against ``LABRAM_CHANNEL_ORDER``.

    Returns ``(keep_src, labram_indices, matched_names, dropped_names)`` —
    ``labram_indices`` are +1 shifted (CLS at position 0). Raises
    ``ValueError`` if nothing matched. No tensors — shared by
    ``_resolve_labram_channels`` and ``describe_channel_mapping`` so they
    cannot diverge.
    """
    from braindecode.models.labram import LABRAM_CHANNEL_ORDER
    labram_to_idx = {ch.upper(): i for i, ch in enumerate(LABRAM_CHANNEL_ORDER)}

    keep_src: List[int] = []
    labram_indices: List[int] = []
    matched_names: List[str] = []
    dropped_names: List[str] = []
    for i, name in enumerate(ch_names):
        idx = labram_to_idx.get(name.upper())
        if idx is not None:
            keep_src.append(i)
            labram_indices.append(idx + 1)  # +1 for CLS at position 0
            matched_names.append(name)
        else:
            dropped_names.append(name)

    if not keep_src:
        raise ValueError(
            f"LaBraM: no input channels matched LABRAM_CHANNEL_ORDER. "
            f"ch_names={ch_names!r}"
        )
    return keep_src, labram_indices, matched_names, dropped_names


def describe_channel_mapping(channels):
    """Name-only channel provenance (see channel_provenance.ChannelMapping).

    Reuses ``_labram_match`` (the exact inference path). LaBraM uses a fixed
    128-electrode vocabulary but addresses channels via per-channel position
    embeddings (variable layout) — matched channels are kept, unmatched are
    silently dropped, and there is no zero-insertion. An all-unmatched batch
    raises (captured upstream).
    """
    from neuroatlas.benchmarking_helpers.channels.channel_provenance import (
        ChannelMapping,
        ChannelSlot,
    )

    _keep, _idx, matched_names, dropped_names = _labram_match(list(channels))
    slots = [ChannelSlot(target=str(n), source=str(n), status="filled") for n in matched_names]
    dropped = [
        ChannelSlot(target=str(n), source=str(n), status="dropped", note="not_in_vocabulary")
        for n in dropped_names
    ]
    return ChannelMapping(
        model_family="labram", layout_kind="variable", slots=slots, dropped=dropped
    )


def _resolve_labram_channels(
    x: torch.Tensor,
    ch_names: List[str],
) -> tuple:
    """Map channel names to LABRAM_CHANNEL_ORDER indices and filter tensor.

    Returns ``(x_matched, input_chans, matched_names, dropped_names)``
    where ``input_chans`` includes CLS at position 0.
    """
    keep_src, labram_indices, matched_names, dropped_names = _labram_match(ch_names)

    keep_t = torch.tensor(keep_src, device=x.device, dtype=torch.long)
    x_matched = x[:, keep_t, :]
    input_chans = torch.tensor([0] + labram_indices, dtype=torch.long, device=x.device)
    return x_matched, input_chans, matched_names, dropped_names


class LabramBackbone(BenchmarkBackbone):
    """LaBraM backbone with proper weight loading and channel-aware position embeddings.

    Design:
    - Model is built lazily on the first call when the actual channel count C is known.
    - Labram is initialized with n_chans=C (not 128) so there is no zero-padding and
      no mean-pool dilution; forward_features averages only C×n_time_patches tokens.
    - The pretrained 128-channel position embedding is sliced to the first C+1 rows
      (CLS + C channel positions) and copied into the model after weight loading.
    - learned_patcher is set to False (fixed reshape segmentation) because the original
      checkpoint has no learnable Conv1d patcher — only the three temporal-conv layers.
    """

    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        from ._checkpoint_download import ensure_checkpoint
        checkpoint_path = ensure_checkpoint(
            spec.checkpoint_path or "",
            source_type=spec.source_type,
            source_reference=spec.source_reference,
        )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.epoch_seconds = float(spec.expected_epoch_seconds)
        raw_n_times = int(round(_PRETRAIN_SFREQ * self.epoch_seconds))
        if raw_n_times <= 0 or raw_n_times % _PRETRAIN_PATCH != 0:
            raise ValueError(
                f"LaBraM: expected_epoch_seconds={self.epoch_seconds} is not a "
                f"positive multiple of {_PRETRAIN_PATCH / _PRETRAIN_SFREQ:g} s "
                f"(one patch). At {_PRETRAIN_SFREQ:g} Hz this yields "
                f"{raw_n_times} samples; need a positive multiple of "
                f"{_PRETRAIN_PATCH}."
            )
        self.n_times = raw_n_times
        self.n_time_patches = self.n_times // _PRETRAIN_PATCH

        raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state = _unwrap_state_dict(raw)
        state = _remap_labram_keys(state)

        if "position_embedding" not in state:
            raise RuntimeError(
                "LaBraM checkpoint missing remapped 'position_embedding' key."
            )
        self._pretrained_pos_embed: torch.Tensor = state.pop("position_embedding").cpu()
        self._pretrained_temp_embed: torch.Tensor | None = (
            state.pop("temporal_embedding").cpu() if "temporal_embedding" in state else None
        )
        self._state_for_loading: dict = state

        # Pretraining window is authoritative per the checkpoint's
        # temporal_embedding row count. At 200 Hz / 1 s-per-patch, a
        # 16-row temporal_embedding means 15 time patches + CLS → 15 s.
        if self._pretrained_temp_embed is not None:
            self._pretrain_n_time_patches = int(
                self._pretrained_temp_embed.shape[1] - 1  # strip CLS
            )
        else:
            # Without a temporal_embedding in the checkpoint we cannot
            # know; fall back to self.n_time_patches for the strategy
            # report below.
            self._pretrain_n_time_patches = self.n_time_patches
        self._pretrain_epoch_seconds = (
            self._pretrain_n_time_patches * _PRETRAIN_PATCH / _PRETRAIN_SFREQ
        )

        self._model: object = None
        self._model_n_chans: int | None = None
        self._load_report: dict = {}
        self._banner_logged = False
        self._last_report: Dict[str, Any] = {
            "input_sfreq_observed": None,
            "input_resample_method": "unknown",
            "scale_divisor": _SCALE_DIVISOR,
            "n_chans_used": None,
        }

    def _build_model(self, n_chans: int):
        from ._braindecode_compat import ensure_torchaudio_stub
        ensure_torchaudio_stub()
        from braindecode.models import Labram

        model = Labram(
            n_times=self.n_times,
            n_chans=n_chans,
            sfreq=_PRETRAIN_SFREQ,
            n_outputs=0,
        )
        model.patch_embed.segment_patch.learned_patcher = False

        load_info = _strict_load_with_allowlist(model, self._state_for_loading)
        if self.n_time_patches < self._pretrain_n_time_patches:
            strategy = "slice_first_N_pretrained"
        elif self.n_time_patches == self._pretrain_n_time_patches:
            strategy = "full_pretrained"
        else:
            strategy = "interpolated_from_pretrained"
        self._load_report = {
            **load_info,
            "temporal_embed_strategy": strategy,
        }

        model.position_embedding.data = (
            self._pretrained_pos_embed.to(model.position_embedding.device)
        )

        if self._pretrained_temp_embed is not None and hasattr(model, "temporal_embedding"):
            needed = 1 + self.n_time_patches
            pretrain_rows = self._pretrained_temp_embed.shape[1]
            if pretrain_rows >= needed:
                temp_embed = self._pretrained_temp_embed[:, :needed, :]
            else:
                temp_embed = torch.nn.functional.interpolate(
                    self._pretrained_temp_embed.permute(0, 2, 1),
                    size=needed,
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)
                logger.debug(
                    "[backbone=labram] temporal_embedding interpolated: "
                    "%d → %d rows (%d → %d time patches). "
                    "Pretrain window was %g s.",
                    pretrain_rows, needed,
                    self._pretrain_n_time_patches, self.n_time_patches,
                    self._pretrain_epoch_seconds,
                )
            model.temporal_embedding.data = temp_embed.to(model.temporal_embedding.device)

        model.eval()
        model.to(self.device)
        import os as _os
        compile_mode = _os.environ.get("COMPILE_BACKBONE")
        if compile_mode:
            model = torch.compile(
                model,
                mode=compile_mode if compile_mode in {"default", "reduce-overhead", "max-autotune"} else "reduce-overhead",
            )
        return model

    def _log_banner_once(self, n_chans: int, apply_scale: bool) -> None:
        if self._banner_logged:
            return
        scale_repr = f"/{_SCALE_DIVISOR:g}" if apply_scale else "keep"
        logger.info(
            "[backbone=labram] fs=%g Hz window=%d s (patch-multiple of %g s) "
            "scale=%s ref=keep n_chans=%d",
            _PRETRAIN_SFREQ, int(self.epoch_seconds),
            _PRETRAIN_PATCH / _PRETRAIN_SFREQ, scale_repr, n_chans,
        )
        self._banner_logged = True

    def _resolve_overrides(self) -> Dict[str, bool]:
        ov = getattr(self.spec, "runtime_overrides", {}) or {}
        return {
            "apply_amplitude_scale": bool(ov.get("apply_amplitude_scale", True)),
            "apply_recording_normalization": bool(
                ov.get("apply_recording_normalization", True)
            ),
        }

    def extract_embeddings(self, batch) -> np.ndarray:
        with StageTimer("labram", "cast_cpu"):
            x = batch["signals"]["eeg"].to(dtype=torch.float32)
        if x.ndim != 3:
            raise ValueError(
                f"LaBraM expected signal of shape (B, C, T); got {tuple(x.shape)}."
            )
        B, C, T = x.shape

        meta = batch.get("meta") or [{}]
        assert_batch_homogeneity(meta, where="labram:input")

        # Adaptive epoch_seconds: _PAPER_PREPROC_DATASETS take 30 s from the batch meta
        dataset = meta[0].get("dataset", "") if meta else ""
        if dataset in _PAPER_PREPROC_DATASETS:
            epoch_sec = float(meta[0].get("epoch_seconds", 30.0))
        else:
            epoch_sec = self.epoch_seconds
        n_times = int(round(_PRETRAIN_SFREQ * epoch_sec))
        n_time_patches = n_times // _PRETRAIN_PATCH

        # Unit conversion — LaBraM's /100 scaling assumes µV.
        unit = meta[0].get("unit") if meta else None
        x = unit_to_uv(x, unit)

        # Strip all-zero (missing) channels before channel resolution.
        ch_names = (meta[0].get("channels") or [])[:C]
        if ch_names:
            x, ch_names, _ = strip_zero_channels(x, ch_names)
            B, C, T = x.shape

        # Source fs from meta if declared; else inferred.
        meta_sfreq = read_sampling_rate(meta[0]) if meta else None
        src_sfreq_f = (
            float(meta_sfreq) if meta_sfreq is not None else T / epoch_sec
        )
        with StageTimer("labram", "resample_cpu"):
            _backend = "scipy" if dataset in _PAPER_PREPROC_DATASETS else "auto"
            x, resample_method = resample_poly_with_fallback(
                x, src_sfreq_f, _PRETRAIN_SFREQ, backend=_backend
            )
        x = snap_to_epoch_length(x, _PRETRAIN_SFREQ, meta)

        if x.shape[-1] != n_times:
            raise ValueError(
                f"LaBraM: after resampling to {_PRETRAIN_SFREQ:g} Hz, got length "
                f"{x.shape[-1]} but expected {n_times} "
                f"(= {_PRETRAIN_SFREQ:g} × {epoch_sec} s). "
                f"Input was {T} samples at inferred fs {src_sfreq_f:.3f} Hz."
            )

        overrides = self._resolve_overrides()
        with StageTimer("labram", "scale_h2d"):
            if overrides["apply_amplitude_scale"]:
                x = x / _SCALE_DIVISOR
            else:
                quiet.warn_once(
                    logger, "labram scale off",
                    "LaBraM: apply_amplitude_scale=False, so its input is not divided by %g "
                    "as it expects; the embeddings are off its training distribution "
                    "(an ablation)", _SCALE_DIVISOR)
            x = x.to(self.device)
        with StageTimer("labram", "finite_assert"):
            assert_finite(x, "labram:output")

        # Resolve channel names to LABRAM_CHANNEL_ORDER indices.
        with StageTimer("labram", "input_chans"):
            if ch_names:
                x, input_chans, matched, dropped = _resolve_labram_channels(x, ch_names)
                C = x.shape[1]
            else:
                input_chans = torch.arange(C + 1, device=self.device, dtype=torch.long)
                matched, dropped = [], []

        # Update model dimensions for this batch's epoch length
        if self.n_times != n_times or self.n_time_patches != n_time_patches:
            self.n_times = n_times
            self.n_time_patches = n_time_patches
            self._model = None  # force rebuild

        if self._model is None or self._model_n_chans != C:
            self._model = self._build_model(C)
            self._model_n_chans = C

        bci = is_bci_batch(meta)
        with StageTimer("labram", "forward"):
            with torch.inference_mode():
                # BCI parity: the BCI reference embeddings are 400-d, the CLS
                # token next to the mean of the patch tokens.
                if bci:
                    tokens = self._model.forward_features(
                        x, input_chans=input_chans, return_all_tokens=True,
                    )
                    if tokens.ndim == 2:
                        features = tokens
                    else:
                        cls = tokens[:, 0, :]
                        mean_patches = tokens[:, 1:, :].mean(dim=1)
                        features = torch.cat([cls, mean_patches], dim=1)
                else:
                    features = self._model.forward_features(x, input_chans=input_chans)
            if features.ndim > 2:
                features = features.reshape(features.shape[0], -1)

        self._last_report = {
            "input_sfreq_observed": float(src_sfreq_f),
            "input_resample_method": resample_method,
            "scale_divisor": _SCALE_DIVISOR,
            "n_chans_used": C,
            "matched_channels": matched,
            "dropped_channels": dropped,
            "apply_amplitude_scale": overrides["apply_amplitude_scale"],
            "apply_recording_normalization": overrides["apply_recording_normalization"],
            "embedding_extractor": "cls_concat_patch_token_mean" if bci else "fc_norm_of_patch_token_mean",
        }
        self._log_banner_once(C, overrides["apply_amplitude_scale"])
        with StageTimer("labram", "d2h"):
            out = features.detach().cpu().numpy()
        return out

    def extract_embeddings_perpatch(self, batch) -> np.ndarray:
        with StageTimer("labram", "cast_cpu"):
            x = batch["signals"]["eeg"].to(dtype=torch.float32)
        if x.ndim != 3:
            raise ValueError(
                f"LaBraM expected signal of shape (B, C, T); got {tuple(x.shape)}."
            )
        B, C, T = x.shape

        meta = batch.get("meta") or [{}]
        assert_batch_homogeneity(meta, where="labram:input")

        dataset = meta[0].get("dataset", "") if meta else ""
        if dataset in _PAPER_PREPROC_DATASETS:
            epoch_sec = float(meta[0].get("epoch_seconds", 30.0))
        else:
            epoch_sec = self.epoch_seconds
        n_times = int(round(_PRETRAIN_SFREQ * epoch_sec))
        n_time_patches = n_times // _PRETRAIN_PATCH

        unit = meta[0].get("unit") if meta else None
        x = unit_to_uv(x, unit)

        ch_names = (meta[0].get("channels") or [])[:C]
        if ch_names:
            x, ch_names, _ = strip_zero_channels(x, ch_names)
            B, C, T = x.shape

        meta_sfreq = read_sampling_rate(meta[0]) if meta else None
        src_sfreq_f = (
            float(meta_sfreq) if meta_sfreq is not None else T / epoch_sec
        )
        _backend = "scipy" if dataset in _PAPER_PREPROC_DATASETS else "auto"
        with StageTimer("labram", "resample_cpu"):
            x, _ = resample_poly_with_fallback(x, src_sfreq_f, _PRETRAIN_SFREQ, backend=_backend)
        x = snap_to_epoch_length(x, _PRETRAIN_SFREQ, meta)

        if x.shape[-1] != n_times:
            raise ValueError(
                f"LaBraM: after resampling got length {x.shape[-1]} "
                f"but expected {n_times}."
            )

        overrides = self._resolve_overrides()
        if overrides["apply_amplitude_scale"]:
            x = x / _SCALE_DIVISOR
        x = x.to(self.device)
        assert_finite(x, "labram:output")

        if ch_names:
            x, input_chans, _, _ = _resolve_labram_channels(x, ch_names)
            C = x.shape[1]
        else:
            input_chans = torch.arange(C + 1, device=self.device, dtype=torch.long)

        if self._model is None or self._model_n_chans != C:
            self._model = self._build_model(C)
            self._model_n_chans = C

        with torch.inference_mode():
            tokens = self._model.forward_features(
                x, input_chans=input_chans, return_all_tokens=True,
            )
            if tokens.ndim == 2:
                return tokens.unsqueeze(1).detach().cpu().numpy()
            if is_bci_batch(batch.get("meta")):
                return tokens.reshape(tokens.shape[0], -1).detach().cpu().numpy()
            patch_tokens = tokens[:, 1:, :]  # exclude CLS
        return patch_tokens.detach().cpu().numpy()

    def metadata(self):
        return {
            **super().metadata(),
            "device": self.device,
            "pretrain_sfreq": _PRETRAIN_SFREQ,
            "pretrain_n_time_patches": self._pretrain_n_time_patches,
            "pretrain_epoch_seconds": self._pretrain_epoch_seconds,
            "pretrain_n_times": self._pretrain_n_time_patches * _PRETRAIN_PATCH,
            "input_epoch_seconds": self.epoch_seconds,
            "input_n_times": self.n_times,
            "n_time_patches": self.n_time_patches,
            "input_mode": (
                "native_window_s"
                if abs(self.epoch_seconds - self._pretrain_epoch_seconds) > 1e-6
                else "pretrain_default"
            ),
            "wrapper_contract": "model_specific_transforms_only",
            "weight_load_report": {
                **self._load_report,
                "load_mode": "strict_load_with_allowlist",
                "allowed_missing_constants": sorted(_ALLOWED_MISSING),
                "allowed_unexpected_prefixes": list(_ALLOWED_UNEXPECTED_PREFIXES),
                "allowed_unexpected_substrings": list(_ALLOWED_UNEXPECTED_SUBSTRINGS),
            },
            "embedding_provenance": {
                "extractor": "fc_norm_of_patch_token_mean",
                "category": "A",
                "embedding_dim": 200,
                "notes": (
                    "Paper-prescribed: fc_norm(patch_tokens.mean(1)) over "
                    "C × n_time_patches tokens (CLS excluded). braindecode's "
                    "forward_features default return path."
                ),
            },
            **self._last_report,
        }
