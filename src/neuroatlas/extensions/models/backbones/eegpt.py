"""EEGPT backbone wrapper.

Window contract:
    The released checkpoint was pretrained on 4 s × 256 Hz = 1024 samples
    (16 patches of 64). Variable windows are supported via non-overlapping
    4 s tiles plus a patch-aligned trailing tile; per-tile summary-token
    embeddings are combined by duration-weighted mean. Any window that is
    not a positive multiple of the patch size (64 samples @ 256 Hz =
    0.25 s) raises ``ValueError``.

Model-specific transforms applied in _prepare_input:
    1. Unit sanity on input (µV band).
    2. fs resample to 256 Hz (shared helper).
    3. Strict length check: final length must equal epoch_seconds × 256.
    4. Amplitude scale x × 1000 (paper convention: µV → scaled unit).
       BCI trials (``is_bci_batch``) take the BCI reference pipeline's path
       instead: ÷1000, no CAR, no amplitude-band rescue, channels outside the
       58-ch vocabulary dropped, any patch-multiple length accepted.
    5. Strip all-zero channels (defensive guard against dataio padding).
    6. Conditional Common Average Reference (CAR): applied only when the
       remaining channel count is ≥ 3 AND no original label is bipolar
       (contains "-"). EEGPT was pretrained with CAR on 58-ch monopolar
       data, but CAR is degenerate at C<3 (collapses C=1 to zero, mirrors
       C=2) and meaningless on bipolar derivations. The skip reason is
       recorded in metadata and surfaced via the one-shot banner.
    7. Finite check on output.

Channel handling: the wrapper normalizes channel names to EEGPT's CHANNEL_DICT
keys. Known sleep-montage bipolar pairs (C3-A2, etc.) have explicit overrides.
Unknown ``"-"``-containing labels raise with the offending name — we never
silently split a generic bipolar pair because that produces duplicate-target
collisions (e.g. CHB-MIT's ``FP1-F7`` and ``FP1-F3`` would both collapse to
``FP1`` and silently corrupt half the channels).

Signal cleanliness (bandpass / notch) is the dataset preprocessor's
responsibility, not this wrapper's. See AGENT_GUIDE.md §7.1.
"""
from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from ._preproc import (
    _ESAT_DATASETS,
    assert_amplitude_band,
    assert_finite,
    car_reference,
    declared_units_from_batch,
    is_bci_batch,
    read_sampling_rate,
    resample_poly_with_fallback,
    snap_to_epoch_length,
    strip_zero_channels,
    unit_to_uv,
)
from .base import BenchmarkBackbone
from .eegpt_encoder import CHANNEL_DICT, EEGTransformer
from neuroatlas.benchmarking_helpers import CheckpointSpec


logger = logging.getLogger(__name__)

_TARGET_SFREQ = 256.0
_NATIVE_TILE_SECONDS = 4
_NATIVE_TILE_LEN = int(_TARGET_SFREQ * _NATIVE_TILE_SECONDS)  # 1024 samples (16 patches)
_PATCH_SIZE = 64
_AMP_SCALE_FACTOR = 1000.0  # per paper: µV * 1000 → input domain
_AMP_SCALE_FACTOR_BCI = 1e-3  # BCI parity: Angeliki uses µV → mV (÷1000)
_INPUT_AMP_LO_UV = 1.0
_INPUT_AMP_HI_UV = 15_000.0

# Explicit bipolar → monopolar overrides. Acts as the model's "pseudo"
# channel adapter for corpora that ship in a bipolar montage (EEGPT itself
# was pretrained unipolar).
#
# Sleep-montage references (linked-ear) — exact unipolar recovery.
# CHB-MIT double-banana — collision-free unique-electrode assignment across
# the 18 pairs: parasagittal chains keep the proximal electrode, temporal
# chains keep the distal, so each pair maps to a distinct 10-20 target
# (CZ is intentionally unused). The mapping is approximate — signal values
# remain bipolar differences, but positional embeddings now have somewhere
# principled to land instead of raising.  Mark CHB-MIT EEGPT results with
# an asterisk in the paper.
_BIPOLAR_TO_STANDARD: Dict[str, str] = {
    "C3-A2": "C3", "C4-A1": "C4",
    "O1-A2": "O1", "O2-A1": "O2",
    "F3-A2": "F3", "F4-A1": "F4",
    # CHB-MIT double-banana (collision-free, see note above)
    "FP1-F3": "FP1", "F3-C3": "F3", "C3-P3": "C3", "P3-O1": "P3",
    "FP1-F7": "F7",  "F7-T7": "T7", "T7-P7": "P7", "P7-O1": "O1",
    "FP2-F4": "FP2", "F4-C4": "F4", "C4-P4": "C4", "P4-O2": "P4",
    "FP2-F8": "F8",  "F8-T8": "T8", "T8-P8": "P8", "P8-O2": "O2",
    "FZ-CZ":  "FZ",  "CZ-PZ": "PZ",
    # EPILEPSIAE double-banana (legacy 10-20 naming: T3/T4/T5/T6 and
    # an extra central temporal chain). Same distal/proximal scheme as
    # CHB-MIT for the temporal chains; central chain uses 10-10 positions
    # (FC3/FCZ/FC4/CP4) to stay collision-free within the 18-pair set.
    "F7-T3": "T7",  "T3-T5": "P7", "T5-O1": "O1",
    "F8-T4": "T8",  "T4-T6": "P8", "T6-O2": "O2",
    "T3-C3": "FC3", "C3-CZ": "FCZ", "CZ-C4": "FC4", "C4-T4": "CP4",
}

# Legacy 10-20 names → EEGPT's extended 10-10 names.
_LEGACY_1020_TO_1010: Dict[str, str] = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
}

def _normalize_eegpt_channel(name: str) -> str:
    """Map a channel label to an EEGPT CHANNEL_DICT key.

    Strict: only known-safe transforms applied:
    - exact bipolar → monopolar overrides (sleep montages)
    - legacy T3/T4/T5/T6 → T7/T8/P7/P8

    Any ``"-"``-containing name NOT in the override table raises — we do not
    silently split generic bipolar pairs, because that produces silent
    duplicate-target collisions on montages like CHB-MIT double-banana.
    """
    key = name.upper().strip()
    if key in _BIPOLAR_TO_STANDARD:
        key = _BIPOLAR_TO_STANDARD[key]
    elif "-" in key:
        raise ValueError(
            f"EEGPT: ambiguous bipolar channel label {name!r}. Add an explicit "
            f"entry to _BIPOLAR_TO_STANDARD, or reconstruct unipolar signals "
            f"upstream. Silent first-electrode splitting would collide duplicate "
            f"target channels (e.g. 'FP1-F7' and 'FP1-F3' would both become 'FP1')."
        )
    return _LEGACY_1020_TO_1010.get(key, key)


def describe_channel_mapping(channels):
    """Name-only channel provenance (see channel_provenance.ChannelMapping).

    Reuses ``_normalize_eegpt_channel`` (the exact inference path). EEGPT is
    channel-agnostic (per-channel position embeddings) — every input channel
    is kept and mapped to its normalized name; there is no fixed layout and
    therefore no zero-insertion. Ambiguous bipolar labels raise (captured
    upstream). A bipolar→monopolar rewrite is flagged ``approx``.
    """
    from neuroatlas.benchmarking_helpers.channels.channel_provenance import (
        ChannelMapping,
        ChannelSlot,
    )

    slots = []
    for name in channels:
        target = _normalize_eegpt_channel(name)
        status = "approx" if str(name).strip().upper() in _BIPOLAR_TO_STANDARD else "filled"
        slots.append(ChannelSlot(target=target, source=str(name), status=status))
    return ChannelMapping(model_family="eegpt", layout_kind="agnostic", slots=slots)


def _extract_target_encoder_state(payload):
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported EEGPT checkpoint payload type {type(payload)!r}.")
    target_state = {}
    for key, value in payload.items():
        if key.startswith("target_encoder."):
            target_state[key[len("target_encoder."):]] = value
    if not target_state and "target_encoder" in payload and isinstance(payload["target_encoder"], dict):
        target_state = payload["target_encoder"]
    if not target_state:
        raise KeyError("Could not find EEGPT target_encoder weights in the checkpoint payload.")
    return target_state


def _tile_boundaries(n_samples: int) -> List[int]:
    """Return tile end-offsets (exclusive) for an n_samples window.

    Tiles are non-overlapping 1024-sample (4 s) chunks; any remainder is a
    trailing tile of patch-aligned length. Raises if the remainder is not
    a positive multiple of the patch size.
    """
    if n_samples <= 0 or n_samples % _PATCH_SIZE != 0:
        raise ValueError(
            f"EEGPT: expected a positive multiple of {_PATCH_SIZE} samples "
            f"(= {_PATCH_SIZE / _TARGET_SFREQ:g} s @ {_TARGET_SFREQ:g} Hz), "
            f"got {n_samples}."
        )
    bounds = []
    pos = 0
    while pos + _NATIVE_TILE_LEN <= n_samples:
        pos += _NATIVE_TILE_LEN
        bounds.append(pos)
    if pos < n_samples:
        bounds.append(n_samples)
    return bounds


class EEGPTBackbone(BenchmarkBackbone):
    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        from ._checkpoint_download import ensure_checkpoint
        checkpoint_path = ensure_checkpoint(
            spec.checkpoint_path, spec.source_type, spec.source_reference
        )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.window_seconds = float(spec.expected_epoch_seconds)
        self.target_len = int(round(self.window_seconds * _TARGET_SFREQ))
        if self.target_len <= 0 or self.target_len % _PATCH_SIZE != 0:
            raise ValueError(
                f"EEGPT: window_seconds={self.window_seconds:g} yields "
                f"{self.target_len} samples at {_TARGET_SFREQ:g} Hz, which is not "
                f"a positive multiple of the patch size ({_PATCH_SIZE} samples = "
                f"{_PATCH_SIZE / _TARGET_SFREQ:g} s). Choose window_seconds such "
                f"that (window × 256) % 64 == 0."
            )

        self.model = EEGTransformer(
            img_size=(1, _NATIVE_TILE_LEN),
            patch_size=_PATCH_SIZE,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            init_std=0.02,
        )
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        target_state = _extract_target_encoder_state(state)
        self.model.load_state_dict(target_state, strict=True)
        self.model.eval()
        self.model.to(self.device)
        self._load_report = {
            "checkpoint_path": str(checkpoint_path),
        }

        self._banner_logged = False
        self._last_channels_used: List[str] = ["FPZ"]
        self._last_resample_report: Dict[str, Any] = {}
        self._last_reference_applied: str = "unknown"
        self._last_car_skip_reason: str | None = None
        self._last_scale: float = _AMP_SCALE_FACTOR

    def _log_banner_once(self, n_chans: int, n_tiles: int, ref_applied: str, skip_reason: str | None) -> None:
        if self._banner_logged:
            return
        ref_str = ref_applied if skip_reason is None else f"{ref_applied}({skip_reason})"
        logger.info(
            "[backbone=eegpt] fs=%g Hz window=%g s (%d samples, %d tiles) "
            "tile=%d s scale=×%g ref=%s n_chans=%d",
            _TARGET_SFREQ, self.window_seconds, self.target_len, n_tiles,
            _NATIVE_TILE_SECONDS, self._last_scale, ref_str, n_chans,
        )
        self._banner_logged = True

    def _prepare_input(self, batch):
        signals = batch["signals"]
        x = signals.get("eeg")
        if x is None:
            full = signals.get("full_signal")
            if full is None:
                raise KeyError("EEGPT backbone expects batch['signals']['eeg'] or ['full_signal'].")
            x = full[:, :1]
        x = x.to(dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(
                f"EEGPT expected signal of shape (B, C, T) or (B, T); got {tuple(x.shape)}."
            )

        # Unit conversion — convert declared unit to µV.
        meta = batch.get("meta") or [{}]
        unit = meta[0].get("unit") if meta else None
        if unit:
            x = unit_to_uv(x, unit)

        # Tolerate signals that fall outside the µV band by auto-rescaling
        # (some STAGES recordings appear ~1000x too small / too large vs declared µV).
        # BCI parity: Angeliki never rescaled — skip for BCI datasets.
        dataset = meta[0].get("dataset") if meta else None
        bci = is_bci_batch(meta)
        if dataset not in _ESAT_DATASETS and not bci:
            import logging as _lg
            try:
                assert_amplitude_band(
                    x, _INPUT_AMP_LO_UV, _INPUT_AMP_HI_UV, "eegpt:input",
                    declared_units=declared_units_from_batch(batch),
                )
            except ValueError as _e:
                p99 = float(x.abs().quantile(0.99).item()) if x.numel() else 0.0
                if p99 > 0 and p99 < _INPUT_AMP_LO_UV / 10:
                    scale = 1e3 if p99 * 1e3 < _INPUT_AMP_HI_UV else 1.0
                    _lg.getLogger(__name__).warning(
                        "eegpt:input amplitude very low (p99=%.4g uV); rescaling by %g",
                        p99, scale)
                    x = x * scale
                elif p99 > _INPUT_AMP_HI_UV * 10:
                    scale = 1e-3
                    _lg.getLogger(__name__).warning(
                        "eegpt:input amplitude very high (p99=%.4g uV); rescaling by %g",
                        p99, scale)
                    x = x * scale
                else:
                    raise

        # ESAT-8 adaptive epoch_seconds
        if dataset in _ESAT_DATASETS:
            epoch_sec = float(meta[0].get("epoch_seconds", 30.0))
        else:
            epoch_sec = self.window_seconds
        _target_len = int(round(epoch_sec * _TARGET_SFREQ))

        # fs resample to 256 Hz (shared helper). The rate is the one the
        # adapter declares (canonical ``sampling_rate``, MODEL_CONTRACTS §3);
        # it is inferred from the length only when none is declared. This
        # read ``meta['sfreq']`` alone, which no adapter publishes, so the rate
        # was always T / window: a 30 s window handed to a 10 s backbone was
        # silently squashed to 10 s (fs "768 Hz"), and a 10-sample input
        # became a 10 s window at "1 Hz", where every other wrapper refuses
        # the mismatch.
        T = x.shape[-1]
        meta_sfreq = read_sampling_rate(meta[0]) if meta else None
        if meta_sfreq is not None:
            expected_T = int(round(float(meta_sfreq) * epoch_sec))
            # BCI parity: a trial of any patch-multiple length is accepted.
            if not bci and abs(T - expected_T) > 1:
                raise ValueError(
                    f"EEGPT: duration mismatch — meta['sampling_rate']={meta_sfreq} Hz and "
                    f"window_seconds={epoch_sec:g} s imply {expected_T} samples, "
                    f"but got T={T}."
                )
            src_sfreq_f = float(meta_sfreq)
        else:
            src_sfreq_f = T / epoch_sec
        _backend = "scipy" if dataset in _ESAT_DATASETS else "auto"
        x, resample_method = resample_poly_with_fallback(x, src_sfreq_f, _TARGET_SFREQ, backend=_backend)
        if dataset in _ESAT_DATASETS:
            x = snap_to_epoch_length(x, _TARGET_SFREQ, meta)

        resampled_len = x.shape[-1]
        if bci:
            if resampled_len <= 0 or resampled_len % _PATCH_SIZE != 0:
                raise ValueError(
                    f"EEGPT: after resampling to {_TARGET_SFREQ:g} Hz, got "
                    f"{resampled_len} samples which is not a positive multiple "
                    f"of the patch size ({_PATCH_SIZE})."
                )
        elif resampled_len != _target_len:
            raise ValueError(
                f"EEGPT: window length mismatch — expected {_target_len} samples "
                f"({epoch_sec:g} s × {_TARGET_SFREQ:g} Hz) after resampling, "
                f"got {resampled_len}. Input was {T} samples at fs {src_sfreq_f:.3f} Hz."
            )

        # Amplitude scale.
        scale = _AMP_SCALE_FACTOR_BCI if bci else _AMP_SCALE_FACTOR
        x = x * scale
        self._last_scale = scale

        self._last_resample_report = {
            "input_sfreq_observed": float(src_sfreq_f),
            "input_resample_method": resample_method,
        }
        return x

    def forward_features(self, batch) -> torch.Tensor:
        """Grad-enabled pooled features, ``(B, embed_num * embed_dim)``.

        See :meth:`BenchmarkBackbone.forward_features`. ``extract_embeddings``
        is a thin ``inference_mode`` wrapper over this, so the frozen-probe and
        fine-tuning paths share one trunk.
        """
        # Resample + amplitude scale (channel-agnostic transforms).
        x = self._prepare_input(batch)

        # Resolve channel labels. The original (pre-normalization) labels are
        # kept for the bipolar test below — _normalize_eegpt_channel rewrites
        # known bipolar pairs to monopolar anchors, which would mask them.
        meta = batch.get("meta", [{}])
        raw_channels = meta[0].get("channels", None) if meta else None
        if raw_channels is None:
            # TODO(audit): raise per MODEL_CONTRACTS §1 unknown-name policy in MEDIUM cleanup.
            raw_channels = ["FPZ"]
        else:
            raw_channels = list(raw_channels)

        # Strip dataio-padded all-zero channels (defensive; no-op when none present).
        x, raw_kept, _ = strip_zero_channels(x, raw_channels)

        # Channel IDs from surviving (raw → normalized) labels.
        dataset = meta[0].get("dataset") if meta else None
        bci = is_bci_batch(meta)
        if bci:
            keep_indices: List[int] = []
            kept_names: List[str] = []
            dropped: List[str] = []
            for i, ch in enumerate(raw_kept):
                try:
                    std = _normalize_eegpt_channel(ch)
                except ValueError:
                    dropped.append(ch)
                    continue
                if std.upper() in CHANNEL_DICT:
                    keep_indices.append(i)
                    kept_names.append(std)
                else:
                    dropped.append(ch)
            if not kept_names:
                raise RuntimeError(
                    f"EEGPT: 0 of {len(raw_kept)} channels matched the 58-ch vocab. "
                    f"Dropped: {dropped[:10]}..."
                )
            if dropped:
                logger.info(
                    "[backbone=eegpt] BCI: dropped %d/%d channels not in vocab: %s",
                    len(dropped), len(raw_kept), dropped[:10],
                )
            idx = torch.tensor(keep_indices, device=x.device, dtype=torch.long)
            x = x[:, idx, :]
            raw_kept = [raw_kept[i] for i in keep_indices]
            std_names_kept = kept_names
        else:
            std_names_kept = [_normalize_eegpt_channel(c) for c in raw_kept]
        self._last_channels_used = list(std_names_kept)
        channel_ids = self.model.prepare_chan_ids(std_names_kept).to(self.device)

        # Conditional CAR — paper-faithful where safe. EEGPT pretraining applied
        # CAR (via `x - x.mean(-2)` inside temporal_interpolation) to 58-ch
        # monopolar referential data. CAR is degenerate at C<3 (collapses C=1
        # to zero, mirrors C=2) and meaningless on bipolar derivations.
        c_eff = x.shape[-2]
        has_bipolar_label = any("-" in c for c in raw_kept)
        if dataset in _ESAT_DATASETS:
            apply_car, skip_reason = False, "esat_parity"
        elif bci:
            apply_car, skip_reason = False, "bci_parity"
        elif c_eff < 3:
            apply_car, skip_reason = False, "c_lt_3"
        elif has_bipolar_label:
            apply_car, skip_reason = False, "bipolar_labels"
        else:
            apply_car, skip_reason = True, None
        if apply_car:
            x = car_reference(x)
        self._last_reference_applied = "car" if apply_car else "passthrough"
        self._last_car_skip_reason = skip_reason

        x = x.to(self.device)
        assert_finite(x, "eegpt:preprocessed")

        bounds = _tile_boundaries(x.shape[-1])
        prev = 0
        tile_feats: List[torch.Tensor] = []
        tile_weights: List[int] = []
        for end in bounds:
            tile = x[..., prev:end]
            features = self.model(tile, channel_ids)
            # features: (B, num_patches, embed_num, embed_dim) → mean over
            # patches, flatten summary tokens → (B, embed_num * embed_dim)
            tile_feats.append(features.flatten(2).mean(dim=1))
            tile_weights.append(end - prev)
            prev = end
        total = float(sum(tile_weights))
        weights = torch.tensor(
            [w / total for w in tile_weights], device=tile_feats[0].device, dtype=tile_feats[0].dtype,
        )
        stacked = torch.stack(tile_feats, dim=0)  # (n_tiles, B, D)
        aggregated = (stacked * weights.view(-1, 1, 1)).sum(dim=0)
        if not torch.isfinite(aggregated).all():
            n_nan = int(torch.isnan(aggregated).sum())
            n_inf = int(torch.isinf(aggregated).sum())
            n_const = int((x.std(dim=-1) < 1e-6).sum())
            raise ValueError(
                f"EEGPT model produced non-finite output "
                f"(n_nan={n_nan}, n_inf={n_inf}, shape={tuple(aggregated.shape)}). "
                f"Preprocessed input (post-scale, ref={self._last_reference_applied}"
                f"{'(' + skip_reason + ')' if skip_reason else ''}): "
                f"shape={tuple(x.shape)}, min={float(x.min()):.4g}, "
                f"max={float(x.max()):.4g}, std={float(x.std()):.4g}, "
                f"n_constant_channels={n_const}, channels={self._last_channels_used}"
            )
        self._log_banner_once(len(self._last_channels_used), len(bounds), self._last_reference_applied, self._last_car_skip_reason)
        return aggregated

    def extract_embeddings(self, batch) -> np.ndarray:
        with torch.inference_mode():
            return self.forward_features(batch).detach().float().cpu().numpy()

    def extract_embeddings_perpatch(self, batch) -> np.ndarray:
        x = self._prepare_input(batch)
        meta = batch.get("meta", [{}])
        raw_channels = meta[0].get("channels", None) if meta else None
        if raw_channels is None:
            raw_channels = ["FPZ"]
        else:
            raw_channels = list(raw_channels)

        x, raw_kept, _ = strip_zero_channels(x, raw_channels)

        bci = is_bci_batch(meta)
        if bci:
            keep_indices: List[int] = []
            kept_names: List[str] = []
            for i, ch in enumerate(raw_kept):
                try:
                    std = _normalize_eegpt_channel(ch)
                except ValueError:
                    continue
                if std.upper() in CHANNEL_DICT:
                    keep_indices.append(i)
                    kept_names.append(std)
            if not kept_names:
                raise RuntimeError(
                    f"EEGPT: 0 of {len(raw_kept)} channels matched the 58-ch vocab."
                )
            idx = torch.tensor(keep_indices, device=x.device, dtype=torch.long)
            x = x[:, idx, :]
            std_names_kept = kept_names
        else:
            std_names_kept = [_normalize_eegpt_channel(c) for c in raw_kept]
        channel_ids = self.model.prepare_chan_ids(std_names_kept).to(self.device)

        c_eff = x.shape[-2]
        has_bipolar_label = any("-" in c for c in raw_kept)
        if bci:
            apply_car = False
        elif c_eff < 3:
            apply_car = False
        elif has_bipolar_label:
            apply_car = False
        else:
            apply_car = True
        if apply_car:
            x = car_reference(x)

        x = x.to(self.device)
        assert_finite(x, "eegpt:preprocessed")

        bounds = _tile_boundaries(x.shape[-1])
        prev = 0
        all_patch_tokens: List[torch.Tensor] = []
        with torch.inference_mode():
            for end in bounds:
                tile = x[..., prev:end]
                features = self.model(tile, channel_ids)
                # features: (B, num_patches, embed_num, embed_dim)
                B = features.shape[0]
                patch_tokens = features.flatten(2)  # (B, num_patches, embed_num*embed_dim)
                all_patch_tokens.append(patch_tokens)
                prev = end
        concatenated = torch.cat(all_patch_tokens, dim=1)  # (B, total_patches, D)
        return concatenated.detach().float().cpu().numpy()

    def metadata(self):
        return {
            **super().metadata(),
            **self._load_report,
            **self._last_resample_report,
            "device": self.device,
            "input_channels_used": list(self._last_channels_used),
            "window_seconds": self.window_seconds,
            "target_len": self.target_len,
            "native_tile_seconds": _NATIVE_TILE_SECONDS,
            "native_tile_len": _NATIVE_TILE_LEN,
            "patch_size": _PATCH_SIZE,
            "pretrain_sfreq": _TARGET_SFREQ,
            "amplitude_scale_factor": self._last_scale,
            "reference_applied": self._last_reference_applied,
            "car_skip_reason": self._last_car_skip_reason,
            "embedding_reduction": "duration_weighted_mean_of_per_tile_summary_token_means",
            "wrapper_contract": "model_specific_transforms_only",
        }
