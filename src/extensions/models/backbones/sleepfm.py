"""SleepFM backbone wrapper.

SleepFM (Thapa et al., ICML 2024 / Nat Med 2025) is a multimodal sleep
foundation model trained via leave-one-out contrastive learning on 14k PSGs.
Architecturally it is a single SetTransformer shared across modalities. We
evaluate only the EEG (BAS = brain-activity-signals) branch here.

Input contract (consumed from the benchmark batch):
    signals["eeg"]: (B, C, T) or (B, T) raw EEG at the dataset's native rate.
    meta[0]["channels"]: optional list of channel names.
    meta[0]["unit"]: required (MODEL_CONTRACTS §0).
    meta[0]["recording_mean"], meta[0]["recording_std"]: required —
        per-channel stats over the whole recording. The runner auto-injects
        ``compute_recording_stats=True`` for sleepfm; missing stats raises.

What the wrapper does:
    1. Resample to 128 Hz × ``epoch_seconds`` samples via
       ``resample_poly_with_fallback`` (polyphase FIR with anti-alias).
    2. Apply paper-faithful recording-level z-score per channel using
       ``recording_mean`` / ``recording_std`` (matches ``safe_standardize`` in
       the upstream ``preprocessing.py``). Per-window fallback removed:
       paper has no such path, so missing stats fail loud.
    3. Pad/zero-slot the signal into BAS_CHANNELS=10 channel slots (single-
       channel scalp-EEG datasets land in slot 0; others are masked out).
    4. Call SetTransformer.forward(x, mask) and return the temporal-pooled
       embedding of shape (B, embed_dim=128).
"""
from __future__ import annotations

import json
import logging
import warnings

import numpy as np
import torch

from benchmarking_helpers import BenchmarkBatch, CheckpointSpec

from .base import BenchmarkModelWrapper
from ._checkpoint_download import ensure_checkpoint
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
from .sleepfm_encoder import SetTransformer

logger = logging.getLogger(__name__)

_PRETRAIN_EPOCH_SECONDS = 30


_DEFAULT_CONFIG = {
    "in_channels": 1,            # Tokenizer conv1d input channels (always 1)
    "patch_size": 640,           # 5 s * 128 Hz
    "embed_dim": 128,
    "num_heads": 8,
    "num_layers": 6,
    "pooling_head": 8,
    "dropout": 0.0,              # eval: no dropout regardless of train config
    "sampling_freq": 128,
    "sampling_duration": 5,      # seconds per patch
    "BAS_CHANNELS": 10,
    "max_seq_length": 128,
}


def describe_channel_mapping(channels):
    """Name-only channel provenance (see channel_provenance.ChannelMapping).

    SleepFM has no channel-name vocabulary: ``_prepare_input`` is purely
    positional — it keeps the first ``BAS_CHANNELS`` (=10) channels in
    order, zero-pads the channel axis up to 10 when fewer are present, and
    drops any beyond capacity (``beyond_model_capacity``). This mirrors that
    logic on the given label list. (At runtime an upstream
    ``strip_zero_channels`` may first remove data-side all-zero channels;
    that is data-dependent and not visible from names alone.)
    """
    from benchmarking_helpers.channels.channel_provenance import (
        ChannelMapping,
        ChannelSlot,
    )

    cap = int(_DEFAULT_CONFIG["BAS_CHANNELS"])
    chans = list(channels)
    valid = min(len(chans), cap)
    slots = [ChannelSlot(target=f"slot{i}", source=str(chans[i]), status="filled") for i in range(valid)]
    slots += [ChannelSlot(target=f"slot{i}", source=None, status="zero_inserted") for i in range(valid, cap)]
    dropped = [
        ChannelSlot(target=str(chans[i]), source=str(chans[i]), status="dropped", note="beyond_model_capacity")
        for i in range(cap, len(chans))
    ]
    return ChannelMapping(
        model_family="sleepfm", layout_kind="fixed", slots=slots, dropped=dropped
    )


def _strip_module_prefix(state_dict: dict) -> dict:
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _stats_shape_matches(mean, std, expected_c: int) -> bool:
    """Reject ``recording_mean`` / ``recording_std`` whose length is not C.

    Lifted from ``backbones/reve.py``: meta values can land as tensors,
    numpy arrays, or plain lists; flatten-length must equal ``expected_c``
    so the (1, C, 1) reshape below cannot silently broadcast.
    """
    def _len(v) -> int | None:
        if hasattr(v, "numel"):
            return int(v.numel())
        try:
            return int(np.asarray(v).size)
        except Exception:
            return None
    return _len(mean) == expected_c and _len(std) == expected_c


class SleepFMBackbone(BenchmarkModelWrapper):
    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        checkpoint_path = ensure_checkpoint(
            spec.checkpoint_path or "",
            source_type=spec.source_type,
            source_reference=spec.source_reference,
        )

        # Load companion config if present (same dir as checkpoint).
        cfg_path = checkpoint_path.parent / "config.json"
        cfg = dict(_DEFAULT_CONFIG)
        if cfg_path.exists():
            on_disk = json.loads(cfg_path.read_text())
            for key in ("patch_size", "embed_dim", "num_heads", "num_layers",
                        "pooling_head", "sampling_freq", "BAS_CHANNELS"):
                if key in on_disk:
                    cfg[key] = on_disk[key]

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.bas_channels = int(cfg["BAS_CHANNELS"])
        self.patch_size = int(cfg["patch_size"])
        self.sfreq = int(cfg["sampling_freq"])
        self.epoch_seconds = float(spec.expected_epoch_seconds)
        self.target_len = int(round(self.sfreq * self.epoch_seconds))
        # Strict contract: SetTransformer's patch tokenizer reshapes T into
        # (T / patch_size) patches — T must be ≥ 1 patch AND an exact multiple.
        patch_seconds = self.patch_size / self.sfreq
        if self.target_len < self.patch_size:
            logger.warning(
                "SleepFM: epoch_seconds=%g s yields %d samples, below the "
                "minimum patch size of %d samples (%g s). Embeddings may be "
                "unreliable for short epochs.",
                self.epoch_seconds, self.target_len,
                self.patch_size, patch_seconds,
            )
        if self.target_len % self.patch_size != 0:
            logger.warning(
                "SleepFM: epoch_seconds=%g s yields %d samples, which is "
                "not a multiple of patch_size=%d (%g s). Signal will be "
                "truncated to the nearest patch boundary.",
                self.epoch_seconds, self.target_len,
                self.patch_size, patch_seconds,
            )
        if self.epoch_seconds < _PRETRAIN_EPOCH_SECONDS:
            logger.warning(
                "SleepFM epoch_seconds=%s is below the pretraining regime (%s s); "
                "embeddings are out-of-distribution.",
                self.epoch_seconds, _PRETRAIN_EPOCH_SECONDS,
            )
        self.embed_dim = int(cfg["embed_dim"])

        self.model = SetTransformer(
            in_channels=int(cfg["in_channels"]),
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=int(cfg["num_heads"]),
            num_layers=int(cfg["num_layers"]),
            pooling_head=int(cfg["pooling_head"]),
            dropout=0.0,
            max_seq_length=int(cfg["max_seq_length"]),
        )

        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        state = _strip_module_prefix(state)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)

        self._load_report = {
            "checkpoint_path": str(checkpoint_path),
            "target_len_samples": self.target_len,
            "target_sfreq": self.sfreq,
            "bas_channels": self.bas_channels,
        }
        self._banner_logged = False
        self._warned_missing_unit = False
        self._last_unit_declared: str | None = None
        self._last_resample_method: str = "uninitialized"
        self._last_dropped_channels: list[dict] = []

    def _log_banner(self) -> None:
        if self._banner_logged:
            return
        self._banner_logged = True
        logger.info(
            "SleepFM backbone: fs=%d Hz, epoch=%.3g s (%d samples = %d×%d-sample patches), "
            "BAS slots=%d, norm=recording-z-score (paper-faithful; raises if stats missing), "
            "reference_applied=False, scale=unit-invariant.",
            self.sfreq, self.epoch_seconds, self.target_len,
            self.target_len // self.patch_size, self.patch_size, self.bas_channels,
        )

    def _prepare_input(self, batch: BenchmarkBatch) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate_batch(batch)
        x = self.require_signal(batch, "eeg")
        x = x.to(self.device, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        meta = batch.get("meta") or [{}]
        assert_batch_homogeneity(
            meta,
            keys=("sampling_rate", "unit", "channels"),
            where="sleepfm:input",
        )

        # §0 — declared-unit → µV. Defensive only for SleepFM (z-score below
        # is unit-invariant), but the contract requires reading the field.
        unit = meta[0].get("unit") if meta else None
        if unit is None:
            if not self._warned_missing_unit:
                self._warned_missing_unit = True
                warnings.warn(
                    "SleepFM: batch meta missing 'unit' — defaulting to 'uV'. "
                    "Adapters must publish meta[i]['unit'] per "
                    "MODEL_CONTRACTS.md §0.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            unit = "uV"
        self._last_unit_declared = unit
        x = unit_to_uv(x, unit)

        # ESAT-8 adaptive epoch_seconds
        dataset = meta[0].get("dataset", "") if meta else ""
        if dataset in _ESAT_DATASETS:
            epoch_sec = float(meta[0].get("epoch_seconds", 30.0))
        else:
            epoch_sec = self.epoch_seconds
        _target_len = int(round(self.sfreq * epoch_sec))

        T = x.shape[-1]
        src_sfreq_f = read_sampling_rate(meta[0]) if meta else None
        if src_sfreq_f is not None:
            expected_T = int(round(float(src_sfreq_f) * epoch_sec))
            if abs(T - expected_T) > 1:
                raise ValueError(
                    f"SleepFM: duration mismatch — meta['sampling_rate']={src_sfreq_f} Hz and "
                    f"epoch_seconds={epoch_sec:g} s imply {expected_T} samples, "
                    f"but got T={T}."
                )
            src_sfreq_f = float(src_sfreq_f)
        else:
            src_sfreq_f = T / epoch_sec

        _backend = "scipy" if dataset in _ESAT_DATASETS else "auto"
        x, resample_method = resample_poly_with_fallback(
            x, src_sfreq_f, float(self.sfreq), backend=_backend
        )
        self._last_resample_method = resample_method
        if dataset in _ESAT_DATASETS:
            x = snap_to_epoch_length(x, float(self.sfreq), meta)
        if x.shape[-1] != _target_len:
            if dataset == "bci":
                actual_len = x.shape[-1]
                usable = (actual_len // self.patch_size) * self.patch_size
                if usable > 0:
                    x = x[..., :usable]
                else:
                    pad_needed = self.patch_size - actual_len
                    x = torch.nn.functional.pad(x, (0, pad_needed))
            else:
                raise ValueError(
                    f"SleepFM: window length mismatch — expected {_target_len} samples "
                    f"({epoch_sec:g} s × {self.sfreq} Hz), got {x.shape[-1]}. "
                    f"Input was {T} samples at fs {src_sfreq_f:.3f} Hz."
                )

        # §2 — paper-faithful recording-level z-score per channel. SleepFM's
        # upstream `preprocessing.py::safe_standardize` computes `(x - μ) / σ`
        # over the whole recording per channel; that's the ONLY normalization
        # path. Missing stats raises — there is no paper-faithful fallback.
        # NOTE: stats are computed at the dataset's native fs and applied
        # to the resampled tensor; for properly anti-aliased downsampling
        # the drift is sub-percent.
        # PATCH (parity-2026-05-14): per-window recording stats. Each row
        # uses its own meta[i] instead of broadcasting meta[0] over the batch
        # (cross-recording batches were getting recording-0's stats applied).
        B_in = x.shape[0]
        per_row_means = []
        per_row_stds = []
        dataset = meta[0].get("dataset") if meta else None
        for i in range(B_in):
            m_i = meta[i] if (meta and i < len(meta)) else {}
            rm = m_i.get("recording_mean")
            rs = m_i.get("recording_std")
            if rm is None or rs is None:
                if dataset == "bci":
                    row = x[i]
                    row_mean = row.mean(dim=-1, keepdim=True)
                    row_std = row.std(dim=-1, keepdim=True).clamp(min=1e-6)
                    x[i] = (row - row_mean) / row_std
                    continue
                raise ValueError(
                    "SleepFM requires recording-level z-score stats per window "
                    "(meta[i]['recording_mean'] / meta[i]['recording_std']); "
                    f"row {i} missing."
                )
            if not _stats_shape_matches(rm, rs, x.shape[1]):
                raise ValueError(
                    f"SleepFM row {i}: recording_mean/recording_std shape mismatch "
                    f"— expected {x.shape[1]} per-channel values."
                )
            per_row_means.append(rm)
            per_row_stds.append(rs)
        if per_row_means:
            mu = torch.stack([
                torch.as_tensor(rm, dtype=x.dtype, device=x.device).reshape(x.shape[1])
                for rm in per_row_means
            ]).reshape(B_in, x.shape[1], 1)
            sigma = torch.stack([
                torch.as_tensor(rs, dtype=x.dtype, device=x.device).reshape(x.shape[1])
                for rs in per_row_stds
            ]).reshape(B_in, x.shape[1], 1)
            # Convert stats to µV to match the post-`unit_to_uv` tensor. With the
            # paper's z-score the unit factor cancels in (x - μ) / σ; converting
            # both keeps the numerator and denominator in a single unit system
            # for any downstream code that inspects ``_last_unit_declared``.
            mu = unit_to_uv(mu, unit)
            sigma = unit_to_uv(sigma, unit)
            x = (x - mu) / sigma.clamp(min=1e-6)

        B, C_in, T_out = x.shape

        dropped: list[dict] = []
        channels = meta[0].get("channels", [])[:C_in]
        if channels:
            x_stripped, channels_kept, kept_idx = strip_zero_channels(x, list(channels))
            for orig_idx, name in enumerate(channels):
                if orig_idx not in kept_idx:
                    dropped.append({"label": str(name), "reason": "non_eeg_modality"})
            x = x_stripped
            channels = channels_kept
            B, C_in, T_out = x.shape

        # Truncate to BAS_CHANNELS. Track the cut so over-capacity drops
        # don't vanish silently.
        valid = min(C_in, self.bas_channels)
        if C_in > self.bas_channels:
            for over_idx in range(self.bas_channels, C_in):
                label = channels[over_idx] if over_idx < len(channels) else f"ch{over_idx}"
                dropped.append({"label": str(label), "reason": "beyond_model_capacity"})
        self._last_dropped_channels = dropped

        # Pad channel axis up to BAS_CHANNELS; fill unused slots with zeros.
        padded = torch.zeros(B, self.bas_channels, T_out, device=x.device, dtype=x.dtype)
        padded[:, :valid, :] = x[:, :valid, :]
        # Padding mask: True = padded/invalid, False = real signal.
        mask = torch.ones(B, self.bas_channels, dtype=torch.bool, device=x.device)
        mask[:, :valid] = False

        self._log_banner()
        return padded, mask

    def extract_embeddings(self, batch: BenchmarkBatch) -> np.ndarray:
        x, mask = self._prepare_input(batch)
        with torch.inference_mode():
            pooled, _ = self.model(x, mask)
        assert_finite(pooled, "sleepfm:output")
        return pooled.detach().cpu().numpy()

    def extract_embeddings_perpatch(self, batch: BenchmarkBatch) -> np.ndarray:
        x, mask = self._prepare_input(batch)
        with torch.inference_mode():
            _, embedding = self.model(x, mask)
        # embedding: (B, S, embed_dim) where S = T // patch_size
        # Spatial pooling already masks out padded channels, so all S
        # temporal tokens are valid.
        assert_finite(embedding, "sleepfm:perpatch_output")
        return embedding.detach().cpu().numpy()

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            **self._load_report,
            "device": self.device,
            "modality": "BAS (EEG-branch of SleepFM)",
            "embedding_reduction": "attention_temporal_pooling",
            "alignment_category": "A",
            "norm_source": "recording_stats",
            "unit_declared": self._last_unit_declared,
            "reference_applied": False,
            "resample_method": self._last_resample_method,
            "dropped_channels": list(self._last_dropped_channels),
        }
