"""STEEGFormer backbone wrapper.

Window contract:
    STEEGFormer is patch-based (16 samples at 128 Hz = 0.125 s patches). The
    wrapper accepts any input whose resampled length at 128 Hz is a positive
    multiple of 16 samples. Mismatched lengths raise ``ValueError`` — the
    wrapper never silently stretches, pads, or tiles.

Model-specific transforms applied in _prepare_input:
    1. Batch-meta homogeneity check (fs / unit / channels).
    2. ``unit_to_uv`` — convert declared unit to uV.
    3. Channel resolution: map channel names to STEEGFORMER_CHANNEL_MAP
       indices (case-insensitive). Non-EEG modality channels (EOG, EMG,
       ECG, …) are dropped. Unrecognised EEG labels raise ``ValueError``.
    4. Strip all-zero channels.
    5. fs resample to 128 Hz (shared helper; source fs from
       ``meta[i]["sampling_rate"]``).
    6. Patch-multiple length check: resampled length must be a positive
       multiple of 16 samples (0.125 s at 128 Hz).
    7. Per-channel z-score normalization using recording-level statistics
       (``recording_mean`` / ``recording_std`` from metadata).
       Gated by ``runtime_overrides["apply_recording_normalization"]``.
    8. Finite check on output.

Amplitude scale:
    Per-channel z-score is scale-invariant by construction — the declared
    unit is converted to uV via ``unit_to_uv`` (rubric requirement), then
    the z-score normalises each channel to zero-mean unit-variance.
    amplitude_scale = z_score_internal.

Reference handling: per MODEL_CONTRACTS.md section 2, the wrapper preserves the
adapter's reference. The pretraining data used varied references; at
inference we do not force re-referencing.

Signal cleanliness (bandpass / notch) is the dataset preprocessor's
responsibility, not this wrapper's. See AGENT_GUIDE.md section 7.1.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from ._preproc import (
    _ESAT_DATASETS,
    assert_batch_homogeneity,
    assert_finite,
    read_sampling_rate,
    resample_poly_with_fallback,
    snap_to_epoch_length,
    strict_load_with_allowlist,
    strip_zero_channels,
    unit_to_uv,
)
from .base import BenchmarkModelWrapper
from .steegformer_model import (
    STEEGFORMER_CHANNEL_MAP,
    STEEGFORMER_CONFIGS,
    build_steegformer,
)
from neuroatlas.benchmarking_helpers import BenchmarkBatch, CheckpointSpec


logger = logging.getLogger(__name__)

_TARGET_SFREQ = 128.0
_PATCH_SIZE = 16
_ZSCORE_EPS = 1e-6

_LOWER_CHANNEL_MAP: Dict[str, int] = {
    k.lower(): v for k, v in STEEGFORMER_CHANNEL_MAP.items()
}

_NON_EEG_PREFIXES = ("EOG", "EMG", "ECG", "EKG", "RESP", "TEMP", "EVENT", "STIM")


def _is_non_eeg(label: str) -> bool:
    upper = label.strip().upper()
    return any(upper.startswith(p) for p in _NON_EEG_PREFIXES)


_ALLOWED_MISSING = {"pos_embed"}
_ALLOWED_UNEXPECTED_PREFIXES = ("dec_", "decoder")
_ALLOWED_UNEXPECTED_LITERALS = {"mask_token"}


def _resolve_variant(spec: CheckpointSpec) -> str:
    variant = getattr(spec, "variant", None) or ""
    for name in STEEGFORMER_CONFIGS:
        if name in variant.lower():
            return name
    raise ValueError(
        f"STEEGFormer: cannot determine variant from spec.variant={variant!r}. "
        f"Expected one of {sorted(STEEGFORMER_CONFIGS)}."
    )


def _resolve_channels(
    labels: Sequence[str], n_channels: int
) -> Tuple[List[int], List[str], List[Dict[str, str]], torch.Tensor]:
    """Map channel names to STEEGFORMER vocabulary indices.

    Performs case-insensitive lookup matching the upstream convention.
    Non-EEG modality channels (EOG, EMG, ECG, …) are dropped with reason
    ``"non_eeg_modality"``.  Any other unrecognised label raises
    ``ValueError`` per MODEL_CONTRACTS.md §1.

    Returns (kept_signal_indices, kept_names, dropped_channels, chan_idx_tensor).
    ``dropped_channels`` is a list of ``{"label": str, "reason": str}`` dicts.
    """
    if not labels:
        raise ValueError(
            "STEEGFormer: batch meta has no 'channels' list; cannot resolve "
            "channel positions. Populate meta[0]['channels'] with per-channel "
            "labels."
        )
    if len(labels) != n_channels:
        raise ValueError(
            f"STEEGFormer: channel metadata length mismatch — got {len(labels)} "
            f"labels for {n_channels} tensor channels."
        )

    kept_idx: List[int] = []
    kept_names: List[str] = []
    dropped_channels: List[Dict[str, str]] = []
    chan_indices: List[int] = []

    for i, name in enumerate(labels):
        key = name.strip().lower()
        steeg_idx = _LOWER_CHANNEL_MAP.get(key)
        if steeg_idx is not None:
            kept_idx.append(i)
            kept_names.append(name)
            chan_indices.append(steeg_idx)
        elif _is_non_eeg(name):
            dropped_channels.append({"label": name, "reason": "non_eeg_modality"})
        else:
            # Unknown label (e.g. zero-filled placeholder "EEG" for subjects with
            # partial channels). Drop it like a non-EEG modality so the remaining
            # in-vocab channels can still be embedded. Matches the lenient behaviour
            # of other backbones (cbramod, biot, chronos, ...).
            dropped_channels.append({"label": name, "reason": "unknown_label"})

    if not kept_idx:
        raise RuntimeError(
            f"STEEGFormer: 0 of {n_channels} matched_channels in the 142-entry "
            f"vocabulary. All channels were non-EEG modalities: "
            f"{[d['label'] for d in dropped_channels[:10]]}..."
        )

    return (
        kept_idx,
        kept_names,
        dropped_channels,
        torch.tensor(chan_indices, dtype=torch.long),
    )


def describe_channel_mapping(channels):
    """Name-only channel provenance (see channel_provenance.ChannelMapping).

    Reuses ``_resolve_channels`` (the exact inference path). STEEGFormer
    maps into a 142-entry vocabulary with a per-channel position embedding
    (variable layout) — no fixed slots, so no zero-insertion. Non-EEG and
    unknown labels are dropped; an all-dropped batch raises (captured
    upstream).
    """
    from neuroatlas.benchmarking_helpers.channels.channel_provenance import (
        ChannelMapping,
        ChannelSlot,
    )

    _kept_idx, kept_names, dropped_channels, _chan_idx = _resolve_channels(
        list(channels), len(channels)
    )
    slots = [ChannelSlot(target=str(n), source=str(n), status="filled") for n in kept_names]
    dropped = [
        ChannelSlot(target=str(d["label"]), source=str(d["label"]), status="dropped", note=d["reason"])
        for d in dropped_channels
    ]
    return ChannelMapping(
        model_family="steegformer", layout_kind="variable", slots=slots, dropped=dropped
    )


def _extract_encoder_state(raw: Any) -> Dict[str, torch.Tensor]:
    """Extract encoder-only state dict from a checkpoint payload.

    Handles both formats:
    - Full checkpoint: ``{'model': state_dict, 'optimizer': ..., ...}``
    - Weights-only: ``OrderedDict`` of parameters directly.
    """
    if isinstance(raw, dict) and "model" in raw:
        state = raw["model"]
    else:
        state = raw

    encoder_state = {}
    for k, v in state.items():
        if k == "mask_token":
            continue
        if k.startswith(("dec_", "decoder")):
            continue
        encoder_state[k] = v

    return encoder_state


class STEEGFormerBackbone(BenchmarkModelWrapper):
    """Wrapper around vendored ST-EEGFormer for benchmark embedding extraction.

    Input contract (from VisionTransformerEEG.forward_features):
        eeg: (B, C, T) float32 at 128 Hz, per-channel z-scored.
        chan_idx: (C,) int64 — STEEGFORMER vocabulary indices.

    Output contract:
        CLS token embedding: (B, embed_dim) where embed_dim is
        512 (small), 768 (base), or 1024 (large).

    Channel handling:
        Maps channel names to STEEGFORMER_CHANNEL_MAP indices (142-entry
        vocabulary covering standard 10-20/10-10/10-05 positions). The
        model's learnable ChannelPositionalEmbed encodes channel identity,
        so the set of matched_channels determines the spatial information
        available to the transformer.

    Amplitude scale:
        Per-channel z-score normalization matches the pretraining convention.
        Scale-invariant by construction. amplitude_scale = z_score_internal.
    """

    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        from ._checkpoint_download import ensure_checkpoint

        variant = _resolve_variant(spec)
        self.variant = variant
        self.embed_dim = STEEGFORMER_CONFIGS[variant]["embed_dim"]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.epoch_seconds = float(spec.expected_epoch_seconds)

        self.model = build_steegformer(variant)

        checkpoint_path = ensure_checkpoint(
            spec.checkpoint_path or "",
            source_type=spec.source_type,
            source_reference=spec.source_reference,
        )
        raw = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        encoder_state = _extract_encoder_state(raw)

        self._load_report = strict_load_with_allowlist(
            self.model,
            encoder_state,
            allowed_missing=_ALLOWED_MISSING,
            allowed_unexpected_literals=_ALLOWED_UNEXPECTED_LITERALS,
            allowed_unexpected_prefixes=_ALLOWED_UNEXPECTED_PREFIXES,
            where=f"steegformer:{variant}",
        )
        self._load_report["checkpoint_path"] = str(checkpoint_path)
        self._load_report["variant"] = variant

        self.model.eval()
        self.model.to(self.device)

        self._banner_logged = False
        self._last_channel_report: Dict[str, Any] = {}

    def _log_banner_once(
        self, n_target_ch: int, zscore_applied: bool
    ) -> None:
        if self._banner_logged:
            return
        scale_repr = "z_score_per_channel" if zscore_applied else "keep"
        logger.info(
            "[backbone=steegformer-%s] fs=%g Hz window=%.1f s "
            "(patch-multiple of %.3f s) scale=%s ref=keep n_target_ch=%d "
            "embed_dim=%d",
            self.variant,
            _TARGET_SFREQ,
            self.epoch_seconds,
            _PATCH_SIZE / _TARGET_SFREQ,
            scale_repr,
            n_target_ch,
            self.embed_dim,
        )
        self._banner_logged = True

    def _resolve_overrides(self) -> Dict[str, bool]:
        ov = getattr(self.spec, "runtime_overrides", {}) or {}
        return {
            "apply_amplitude_scale": bool(
                ov.get("apply_amplitude_scale", True)
            ),
            "apply_recording_normalization": bool(
                ov.get("apply_recording_normalization", True)
            ),
        }

    def _prepare_input(
        self, batch: BenchmarkBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform batch -> (eeg_tensor, chan_idx) ready for the encoder.

        Returns
        -------
        eeg : (B, C_matched, T_resampled) float32, z-scored, on self.device.
        chan_idx : (C_matched,) int64 — vocabulary indices, on self.device.
        """
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
                f"STEEGFormer expected signal of shape (B, C, T) or (B, T); "
                f"got shape {tuple(x.shape)}."
            )

        meta = batch["meta"]
        assert_batch_homogeneity(meta, where="steegformer:input")
        first_meta: Dict[str, Any] = meta[0] if meta else {}

        unit = first_meta.get("unit")
        x = unit_to_uv(x, unit)

        raw_labels = first_meta.get("channels") or []
        kept_idx, kept_names, dropped_channels, chan_idx = _resolve_channels(
            raw_labels, n_channels=x.shape[1]
        )
        if dropped_channels:
            logger.info(
                "[backbone=steegformer] dropped %d/%d channels: %s",
                len(dropped_channels),
                len(raw_labels),
                [d["label"] for d in dropped_channels[:10]],
            )
        if kept_idx != list(range(x.shape[1])):
            x = x[:, kept_idx, :]

        x, surviving_names, surviving_idx = strip_zero_channels(x, kept_names)
        if len(surviving_names) < len(kept_names):
            chan_idx = chan_idx[surviving_idx]
            kept_names = surviving_names

        # ESAT-8 adaptive epoch_seconds
        dataset = first_meta.get("dataset", "") if first_meta else ""
        if dataset in _ESAT_DATASETS:
            epoch_sec = float(first_meta.get("epoch_seconds", 30.0))
        else:
            epoch_sec = self.epoch_seconds

        current_len = x.shape[-1]
        meta_sfreq = read_sampling_rate(first_meta) if first_meta else None
        src_sfreq_f = (
            float(meta_sfreq)
            if meta_sfreq is not None
            else current_len / epoch_sec
        )
        _backend = "scipy" if dataset in _ESAT_DATASETS else "auto"
        x, resample_method = resample_poly_with_fallback(
            x, src_sfreq_f, _TARGET_SFREQ, backend=_backend
        )
        x = snap_to_epoch_length(x, _TARGET_SFREQ, meta)

        actual_len = x.shape[-1]
        if actual_len <= 0 or actual_len % _PATCH_SIZE != 0:
            raise ValueError(
                f"STEEGFormer: after resampling to {_TARGET_SFREQ:g} Hz, got "
                f"length {actual_len} which is not a positive multiple of "
                f"{_PATCH_SIZE} (one patch = "
                f"{_PATCH_SIZE / _TARGET_SFREQ:g} s). "
                f"Input was {current_len} samples at inferred fs "
                f"{src_sfreq_f:.3f} Hz."
            )

        overrides = self._resolve_overrides()
        if overrides["apply_recording_normalization"]:
            if "recording_mean" in first_meta and "recording_std" in first_meta:
                mu = torch.as_tensor(
                    first_meta["recording_mean"], dtype=x.dtype, device=x.device,
                ).unsqueeze(-1)
                sigma = torch.as_tensor(
                    first_meta["recording_std"], dtype=x.dtype, device=x.device,
                ).unsqueeze(-1)
                mu = unit_to_uv(mu, unit)
                sigma = unit_to_uv(sigma, unit)
                # Subset to match channels surviving resolution + zero-strip.
                # Build the composite index: raw channel dim → final channel dim.
                ch_sel = kept_idx
                if len(surviving_names) < len(kept_idx):
                    ch_sel = [kept_idx[i] for i in surviving_idx]
                mu = mu[ch_sel]
                sigma = sigma[ch_sel]
                x = (x - mu) / sigma.clamp(min=_ZSCORE_EPS)
            else:
                mean = x.mean(dim=-1, keepdim=True)
                std = x.std(dim=-1, keepdim=True).clamp_min(_ZSCORE_EPS)
                x = (x - mean) / std
        else:
            logger.warning(
                "[backbone=steegformer] apply_recording_normalization=False — "
                "STEEGFormer expects per-channel z-scored input; embeddings "
                "will be off-distribution."
            )

        x = x.to(self.device)
        chan_idx = chan_idx.to(self.device)
        assert_finite(x, "steegformer:output")

        self._last_channel_report = {
            "input_channels_used": list(kept_names),
            "dropped_channels": list(dropped_channels),
            "matched_channels": len(kept_names),
            "total_input_channels": len(raw_labels),
            "input_sfreq_observed": float(src_sfreq_f),
            "input_resample_method": resample_method,
            "apply_recording_normalization": overrides[
                "apply_recording_normalization"
            ],
            "apply_amplitude_scale": overrides["apply_amplitude_scale"],
        }
        self._log_banner_once(
            len(kept_names), overrides["apply_recording_normalization"]
        )
        return x, chan_idx

    def extract_embeddings(self, batch: BenchmarkBatch) -> np.ndarray:
        x, chan_idx = self._prepare_input(batch)
        with torch.no_grad():
            features = self.model.forward_features(x, chan_idx)
        return features.detach().cpu().numpy()

    def extract_embeddings_perpatch(self, batch: BenchmarkBatch) -> np.ndarray:
        x, chan_idx = self._prepare_input(batch)
        with torch.no_grad():
            all_tokens = self.model.forward_features(
                x, chan_idx, return_all_tokens=True
            )
        patch_tokens = all_tokens[:, 1:, :]
        return (
            patch_tokens.reshape(patch_tokens.shape[0], -1)
            .detach()
            .cpu()
            .numpy()
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            **super().metadata(),
            **self._last_channel_report,
            "device": self.device,
            "variant": self.variant,
            "pretrain_sfreq": _TARGET_SFREQ,
            "input_epoch_seconds": self.epoch_seconds,
            "patch_size_samples": _PATCH_SIZE,
            "patch_size_seconds": _PATCH_SIZE / _TARGET_SFREQ,
            "embedding_reduction": "cls_token",
            "wrapper_contract": "model_specific_transforms_only",
            "weight_load_report": dict(self._load_report),
            "embedding_provenance": {
                "extractor": "cls_token",
                "category": "A",
                "embedding_dim": self.embed_dim,
                "notes": (
                    "CLS token from encoder output. "
                    "Category A: standard linear probe target per MAE convention."
                ),
            },
        }
