"""NeuroGPT backbone wrapper.

Window contract:
    Chunk-multiple: input must resample to a positive multiple of 500 samples
    (2 s at 250 Hz). Longer windows are split into non-overlapping 2 s chunks;
    per-chunk encoder embeddings are combined by mean pooling.

Model-specific transforms applied in _prepare_input:
    1. Unit conversion to µV (unit_to_uv) + batch homogeneity check.
    2. Per-channel DC removal (pre-normalization hygiene).
    3. fs resample to 250 Hz (shared helper).
    4. Chunk-multiple length check: resampled length must be a positive
       multiple of 500 samples (2 s). No fixed epoch-length enforced.
    5. Per-channel z-score normalization using recording-level statistics
       (paper: z-transform along time within each recording). Falls back
       to per-epoch stats when recording_mean/recording_std absent. Gated
       by ``runtime_overrides["apply_recording_normalization"]``.
    6. Channel adapter: map caller labels into the fixed 22-channel layout;
       zero-pad unmatched slots. Follows BENDR's channel-mapping pattern.
    7. Finite check on output.

NeuroGPT is scale-invariant (per-channel z-score absorbs any unit),
so amplitude scaling is a no-op.

Reference handling: the wrapper preserves the adapter's reference.
The paper's pretraining re-referenced to the average of 22 channels;
at inference we do not force re-referencing.

Channel handling: NeuroGPT uses legacy 10-20 naming (T1/T2/T3/T4/T5/T6
instead of FT9/FT10/T7/T8/P7/P8). The wrapper normalizes modern 10-10
names back to legacy names. Unknown channels are dropped with logging.

Signal cleanliness (bandpass / notch) is the dataset preprocessor's
responsibility, not this wrapper's. See AGENT_GUIDE.md §7.1.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ._preproc import (
    _ESAT_DATASETS,
    assert_batch_homogeneity,
    assert_finite,
    resample_poly_with_fallback,
    snap_to_epoch_length,
    unit_to_uv,
)
from .base import BenchmarkBackbone
from .neurogpt_encoder import (
    CHUNK_SIZE,
    EMBEDDING_DIM,
    EEGConformerEncoder,
    NEUROGPT_CHANNELS,
)
from benchmarking_helpers import CheckpointSpec

logger = logging.getLogger(__name__)

_TARGET_SFREQ = 250.0
_CHUNK_SIZE = CHUNK_SIZE  # 500 samples = 2 s at 250 Hz
_EMBEDDING_DIM = EMBEDDING_DIM  # 1080 = 27 tokens × 40 features
_ZSCORE_EPS = 1e-25  # paper uses 1e-25

_NEUROGPT_TARGET_CHANNELS: Tuple[str, ...] = NEUROGPT_CHANNELS

# Modern 10-10 → legacy NeuroGPT names.
# NeuroGPT natively uses T1/T2/T3/T4/T5/T6 (legacy 10-20).
_CHANNEL_ALIASES: Dict[str, str] = {
    # Exact equivalences (modern 10-10 → legacy 10-20)
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
    # Approximate (FT9/FT10 are the closest standard positions to T1/T2)
    "FT9": "T1", "FT10": "T2",
    # Case variants
    "FPZ": "FPZ",
    # Sleep-montage bipolar pairs → monopolar anchor
    "C3-A2": "C3", "C4-A1": "C4",
    "O1-A2": "O1", "O2-A1": "O2",
    "F3-A2": "F3", "F4-A1": "F4",
    "EEG FPZ-CZ": "FZ", "FPZ-CZ": "FZ",
    "EEG PZ-OZ": "PZ", "PZ-OZ": "PZ",
    # CHB-MIT double-banana — collision-free mapping
    "FP1-F7": "FP1", "F7-T7": "F7",
    "FP2-F8": "FP2", "F8-T8": "F8",
    "FP1-F3": "FP1", "F3-C3": "F3", "C3-P3": "C3", "P3-O1": "P3",
    "FP2-F4": "FP2", "F4-C4": "F4", "C4-P4": "C4", "P4-O2": "P4",
    "FZ-CZ": "FZ", "CZ-PZ": "CZ",
    "T7-P7": "T5", "P7-O1": "O1",
    "T8-P8": "T6", "P8-O2": "O2",
    # EPILEPSIAE double-banana (legacy T3/T4 naming)
    "F7-T3": "T3", "T3-T5": "T5", "T5-O1": "O1",
    "F8-T4": "T4", "T4-T6": "T6", "T6-O2": "O2",
}

_IGNORED_CHANNEL_PREFIXES: Tuple[str, ...] = (
    "EOG", "EMG", "RESP", "EVENT", "TEMP", "EKG", "ECG", "STIM",
)

_TARGET_INDEX: Dict[str, int] = {
    ch: i for i, ch in enumerate(_NEUROGPT_TARGET_CHANNELS)
}


def _normalize_channel_name(name: str) -> Optional[str]:
    """Normalize a channel label to a NeuroGPT target name.

    Returns the target channel name, or None if the channel should be
    ignored (non-EEG modality prefix).

    Raises ValueError for ambiguous bipolar labels not in the alias table.
    """
    key = name.strip().upper()
    # Strip common dataset prefixes
    if key.startswith("EEG "):
        key = key[4:]

    # Check ignored prefixes
    for prefix in _IGNORED_CHANNEL_PREFIXES:
        if key.startswith(prefix):
            return None

    # Check alias table (handles bipolar overrides)
    if key in _CHANNEL_ALIASES:
        return _CHANNEL_ALIASES[key]

    # Handle generic bipolar labels not in alias table
    if "-" in key:
        raise ValueError(
            f"NeuroGPT: ambiguous bipolar channel label {name!r}. "
            f"Add an explicit entry to _CHANNEL_ALIASES to resolve it."
        )

    return key


def _neurogpt_channel_plan(
    channels: List[str],
) -> Tuple[List[Tuple[int, int, str]], List[Dict[str, str]], List[str]]:
    """Pure name→slot decision for the fixed 22-channel NeuroGPT layout.

    Returns ``(assignments, dropped, padded)`` where ``assignments`` is a
    list of ``(target_idx, src_idx, target_name)``, ``dropped`` is a list of
    ``{"label", "reason"}`` dicts, and ``padded`` is the target names left
    unfilled (zero-padded at runtime). Raises ``RuntimeError`` if nothing
    matched. No tensors — shared by ``_map_signals_to_neurogpt_layout`` and
    ``describe_channel_mapping`` so they cannot diverge.
    """
    used_targets: Dict[str, str] = {}  # target_name → source_name
    assignments: List[Tuple[int, int, str]] = []
    dropped: List[Dict[str, str]] = []

    for src_idx, src_name in enumerate(channels):
        try:
            target_name = _normalize_channel_name(src_name)
        except ValueError:
            dropped.append({"label": src_name, "reason": "ambiguous_bipolar"})
            continue

        if target_name is None:
            dropped.append({"label": src_name, "reason": "non_eeg_modality"})
            continue

        if target_name not in _TARGET_INDEX:
            dropped.append({"label": src_name, "reason": "not_in_vocabulary"})
            continue

        if target_name in used_targets:
            dropped.append({"label": src_name, "reason": "duplicate_target_slot"})
            continue

        used_targets[target_name] = src_name
        assignments.append((_TARGET_INDEX[target_name], src_idx, target_name))

    if not assignments:
        raise RuntimeError(
            f"NeuroGPT: 0 of {len(channels)} channels matched the 22-ch layout. "
            f"Source channels: {channels[:10]}..."
        )

    padded = [ch for ch in _NEUROGPT_TARGET_CHANNELS if ch not in used_targets]
    return assignments, dropped, padded


def describe_channel_mapping(channels):
    """Name-only channel provenance (see channel_provenance.ChannelMapping).

    Reuses ``_neurogpt_channel_plan`` — the exact name→slot decision the
    wrapper runs. The 22 target slots not covered by any source channel are
    reported as ``zero_inserted``. Raises (e.g. 0 matched) are captured
    upstream.
    """
    from benchmarking_helpers.channels.channel_provenance import (
        ChannelMapping,
        ChannelSlot,
    )

    assignments, dropped_d, padded = _neurogpt_channel_plan(list(channels))
    target_to_src = {tname: channels[src_idx] for _ti, src_idx, tname in assignments}
    slots = []
    for tname in _NEUROGPT_TARGET_CHANNELS:
        if tname in target_to_src:
            slots.append(ChannelSlot(target=tname, source=str(target_to_src[tname]), status="filled"))
        else:
            slots.append(ChannelSlot(target=tname, source=None, status="zero_inserted"))
    dropped = [
        ChannelSlot(target=str(d["label"]), source=str(d["label"]), status="dropped", note=d["reason"])
        for d in dropped_d
    ]
    return ChannelMapping(
        model_family="neurogpt", layout_kind="fixed", slots=slots, dropped=dropped
    )


def _map_signals_to_neurogpt_layout(
    x: torch.Tensor,
    channels: List[str],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Map source signals into the fixed 22-channel NeuroGPT layout.

    Parameters
    ----------
    x : Tensor of shape (B, C_src, T)
    channels : list of source channel names (length C_src)

    Returns
    -------
    mapped : Tensor of shape (B, 22, T) — zero-padded for unmatched slots
    report : dict with matched/padded/dropped channel info
    """
    B, C_src, T = x.shape
    mapped = torch.zeros(B, len(_NEUROGPT_TARGET_CHANNELS), T,
                         dtype=x.dtype, device=x.device)

    assignments, dropped, padded = _neurogpt_channel_plan(channels)
    matched: List[str] = []
    for target_idx, src_idx, target_name in assignments:
        mapped[:, target_idx, :] = x[:, src_idx, :]
        matched.append(target_name)

    report = {
        "channels_matched": len(matched),
        "channels_padded": len(padded),
        "channels_dropped": len(dropped),
        "matched_names": matched,
        "padded_names": padded,
        "dropped_channels": dropped,
        "match_ratio": len(matched) / len(_NEUROGPT_TARGET_CHANNELS),
    }
    return mapped, report


def _zscore_per_channel(x: torch.Tensor, eps: float = _ZSCORE_EPS) -> torch.Tensor:
    """Per-channel z-score along the time axis (per-epoch fallback).

    For each channel independently: (x[c] - mean[c]) / (std[c] + eps).
    Fallback when recording-level statistics are unavailable; the preferred
    path uses precomputed recording_mean / recording_std from metadata.
    """
    mu = x.mean(dim=-1, keepdim=True)   # (B, C, 1)
    sigma = x.std(dim=-1, keepdim=True)  # (B, C, 1)
    return (x - mu) / (sigma + eps)


class NeuroGPTBackbone(BenchmarkBackbone):
    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        from ._checkpoint_download import ensure_checkpoint

        checkpoint_path = ensure_checkpoint(
            spec.checkpoint_path or "",
            source_type=spec.source_type,
            source_reference=spec.source_reference,
        )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.window_seconds = float(spec.expected_epoch_seconds)

        self.model = EEGConformerEncoder()

        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

        # Extract encoder.* prefix keys
        encoder_state = {}
        for k, v in state.items():
            if k.startswith("encoder."):
                encoder_state[k[len("encoder."):]] = v

        if not encoder_state:
            raise KeyError(
                "Could not find encoder weights in the NeuroGPT checkpoint. "
                f"Available prefixes: {set(k.split('.')[0] for k in state.keys())}"
            )

        self.model.load_state_dict(encoder_state, strict=True)

        self.model.eval()
        self.model.to(self.device)

        self._load_report = {
            "checkpoint_path": str(checkpoint_path),
            "weight_source": spec.source_type,
            "extracted_prefix": "encoder.",
            "n_state_keys_loaded": len(encoder_state),
            "ignored_non_encoder_prefixes": sorted(
                set(k.split(".")[0] for k in state if not k.startswith("encoder."))
            ),
        }

        self._banner_logged = False
        self._last_channels_used: List[str] = list(_NEUROGPT_TARGET_CHANNELS)
        self._last_mapping_report: Dict[str, Any] = {}
        self._last_resample_report: Dict[str, Any] = {}

    def _resolve_overrides(self) -> Dict[str, bool]:
        ov = getattr(self.spec, "runtime_overrides", {}) or {}
        return {
            "apply_amplitude_scale": bool(ov.get("apply_amplitude_scale", True)),
            "apply_recording_normalization": bool(
                ov.get("apply_recording_normalization", True)
            ),
        }

    def _log_banner_once(
        self, n_chans_matched: int, n_samples: int, n_chunks: int,
        norm_applied: bool,
    ) -> None:
        if self._banner_logged:
            return
        norm_repr = "z_score_per_channel" if norm_applied else "keep"
        logger.info(
            "[backbone=neurogpt] fs=%g Hz input=%d samples (%.2f s, %d chunks) "
            "chunk=%d s scale=invariant norm=%s ref=keep n_chans_matched=%d/22",
            _TARGET_SFREQ, n_samples, n_samples / _TARGET_SFREQ, n_chunks,
            int(_CHUNK_SIZE / _TARGET_SFREQ), norm_repr, n_chans_matched,
        )
        self._banner_logged = True

    def _prepare_input(self, batch):
        signals = batch["signals"]
        x = signals.get("eeg")
        if x is None:
            full = signals.get("full_signal")
            if full is None:
                raise KeyError(
                    "NeuroGPT backbone expects batch['signals']['eeg'] or ['full_signal']."
                )
            x = full
        x = x.to(dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(
                f"NeuroGPT expected signal of shape (B, C, T) or (B, T); got {tuple(x.shape)}."
            )

        meta = batch.get("meta") or [{}]
        assert_batch_homogeneity(meta, where="neurogpt:input")

        unit = meta[0].get("unit", "uV") if meta else "uV"
        x = unit_to_uv(x, unit)

        # Per-channel DC removal
        x = x - x.mean(dim=-1, keepdim=True)

        # ESAT-8 adaptive epoch_seconds
        dataset = meta[0].get("dataset", "") if meta else ""
        if dataset in _ESAT_DATASETS:
            epoch_sec = float(meta[0].get("epoch_seconds", 30.0))
        else:
            epoch_sec = self.window_seconds

        # Resample to 250 Hz
        T = x.shape[-1]
        meta_sfreq = meta[0].get("sfreq") if meta else None
        if meta_sfreq is None:
            meta_sfreq = meta[0].get("sampling_rate") if meta else None
        if meta_sfreq is not None:
            src_sfreq_f = float(meta_sfreq)
        else:
            src_sfreq_f = T / epoch_sec
        _backend = "scipy" if dataset in _ESAT_DATASETS else "auto"
        x, resample_method = resample_poly_with_fallback(x, src_sfreq_f, _TARGET_SFREQ, backend=_backend)
        x = snap_to_epoch_length(x, _TARGET_SFREQ, meta)

        # Chunk-multiple validation
        resampled_len = x.shape[-1]
        if resampled_len <= 0 or resampled_len % _CHUNK_SIZE != 0:
            raise ValueError(
                f"NeuroGPT: after resampling to {_TARGET_SFREQ:g} Hz, got "
                f"{resampled_len} samples which is not a positive multiple "
                f"of the chunk size ({_CHUNK_SIZE} samples = "
                f"{_CHUNK_SIZE / _TARGET_SFREQ:g} s). Input was {T} samples "
                f"at {src_sfreq_f:.3f} Hz."
            )

        overrides = self._resolve_overrides()
        dataset = meta[0].get("dataset") if meta else None

        if dataset == "bci":
            # BCI parity: channel remap first, then z-score (Angeliki order).
            raw_channels = meta[0].get("channels", None) if meta else None
            if raw_channels is None:
                raise ValueError(
                    "NeuroGPT requires channel metadata in meta[i]['channels']. "
                    "Ensure the adapter or channel-map config provides channel labels."
                )
            x, mapping_report = _map_signals_to_neurogpt_layout(x, raw_channels)
            self._last_mapping_report = mapping_report
            self._last_channels_used = mapping_report.get(
                "matched_names", list(_NEUROGPT_TARGET_CHANNELS)
            )
            if overrides["apply_recording_normalization"]:
                x = _zscore_per_channel(x)
        else:
            # NeuroAtlas order: z-score first, then remap.
            if overrides["apply_recording_normalization"]:
                if "recording_mean" in meta[0] and "recording_std" in meta[0]:
                    mu = torch.as_tensor(
                        meta[0]["recording_mean"], dtype=x.dtype, device=x.device,
                    ).unsqueeze(-1)
                    sigma = torch.as_tensor(
                        meta[0]["recording_std"], dtype=x.dtype, device=x.device,
                    ).unsqueeze(-1)
                    mu = unit_to_uv(mu, unit)
                    sigma = unit_to_uv(sigma, unit)
                    x = (x - mu) / (sigma + _ZSCORE_EPS)
                else:
                    x = _zscore_per_channel(x)
            else:
                logger.warning(
                    "[backbone=neurogpt] apply_recording_normalization=False — "
                    "NeuroGPT is scale-invariant via per-channel z-score; "
                    "disabling changes the embedding domain."
                )

            # Channel mapping into fixed 22-slot layout
            raw_channels = meta[0].get("channels", None) if meta else None
            if raw_channels is None:
                raise ValueError(
                    "NeuroGPT requires channel metadata in meta[i]['channels']. "
                    "Ensure the adapter or channel-map config provides channel labels."
                )
            x, mapping_report = _map_signals_to_neurogpt_layout(x, raw_channels)

            self._last_mapping_report = mapping_report
            self._last_channels_used = mapping_report.get(
                "matched_names", list(_NEUROGPT_TARGET_CHANNELS)
            )

        x = x.to(self.device)
        assert_finite(x, "neurogpt:output")

        self._last_resample_report = {
            "input_sfreq_observed": float(src_sfreq_f),
            "input_resample_method": resample_method,
            "apply_amplitude_scale": overrides["apply_amplitude_scale"],
            "apply_recording_normalization": overrides["apply_recording_normalization"],
        }
        return x

    def extract_embeddings(self, batch) -> np.ndarray:
        x = self._prepare_input(batch)  # (B, 22, T)
        B = x.shape[0]
        T = x.shape[-1]
        n_chunks = T // _CHUNK_SIZE

        # Reshape into chunks: (B, 22, n_chunks*500) -> (B*n_chunks, 22, 500)
        x = x[:, :, :n_chunks * _CHUNK_SIZE]
        x = x.reshape(B, len(_NEUROGPT_TARGET_CHANNELS), n_chunks, _CHUNK_SIZE)
        x = x.permute(0, 2, 1, 3).reshape(B * n_chunks, len(_NEUROGPT_TARGET_CHANNELS), _CHUNK_SIZE)

        with torch.inference_mode():
            features = self.model(x)  # (B*n_chunks, 27, 40)
            features = features.reshape(features.shape[0], -1)  # (B*n_chunks, 1080)

        # Mean pool across chunks
        features = features.reshape(B, n_chunks, _EMBEDDING_DIM)
        aggregated = features.mean(dim=1)  # (B, 1080)

        self._log_banner_once(
            n_chans_matched=self._last_mapping_report.get("channels_matched", 22),
            n_samples=T,
            n_chunks=n_chunks,
            norm_applied=self._last_resample_report.get(
                "apply_recording_normalization", True
            ),
        )
        return aggregated.detach().cpu().numpy()

    def extract_embeddings_perpatch(self, batch) -> np.ndarray:
        x = self._prepare_input(batch)
        B = x.shape[0]
        T = x.shape[-1]
        n_chunks = T // _CHUNK_SIZE

        x = x[:, :, :n_chunks * _CHUNK_SIZE]
        x = x.reshape(B, len(_NEUROGPT_TARGET_CHANNELS), n_chunks, _CHUNK_SIZE)
        x = x.permute(0, 2, 1, 3).reshape(B * n_chunks, len(_NEUROGPT_TARGET_CHANNELS), _CHUNK_SIZE)

        with torch.inference_mode():
            features = self.model(x)  # (B*n_chunks, 27, 40)
            features = features.reshape(features.shape[0], -1)  # (B*n_chunks, 1080)

        # Return all chunk embeddings concatenated
        features = features.reshape(B, n_chunks * _EMBEDDING_DIM)
        return features.detach().cpu().numpy()

    def metadata(self):
        return {
            **super().metadata(),
            **self._last_resample_report,
            **self._last_mapping_report,
            "device": self.device,
            "input_channels_used": list(self._last_channels_used),
            "window_seconds_advisory": self.window_seconds,
            "chunk_size": _CHUNK_SIZE,
            "pretrain_sfreq": _TARGET_SFREQ,
            "normalization": "z_score_per_channel",
            "reference": "keep",
            "embedding_reduction": "encoder_mean_chunks",
            "wrapper_contract": "model_specific_transforms_only",
            "weight_load_report": dict(self._load_report),
            "embedding_provenance": {
                "extractor": "encoder_flatten_mean_chunks",
                "category": "C",
                "embedding_dim": _EMBEDDING_DIM,
                "notes": (
                    "Category C: paper's best downstream uses encoder-only with "
                    "a 3-layer MLP classification head (256→32→n_classes), "
                    "not a simple pool. Wrapper's frozen mean-over-chunks is "
                    "a lower bound on paper results."
                ),
            },
        }
