"""NeuroRVQ EEG foundation model backbone for frozen linear probing.

Channel contract — CHB-MIT bipolar exception: the 18 double-banana pairs
emitted by CHB-MIT-BIDS are mapped to their anatomically-frontal/midline
anchor (``_BIPOLAR_TO_NEURORVQ``) before vocabulary lookup. Same
collision-free scheme as EEGPT ``_BIPOLAR_TO_STANDARD``. Anchors
use lowercase 10-10 names (legacy T3/T4/T5/T6 → t7/t8/p7/p8) to match
``CH_NAMES_GLOBAL``. Bipolar values are reused as-is at the anchor's
pretrained spatial embedding — pseudo-unipolar evaluation per
``project_chbmit_pseudo.md``.
"""
from __future__ import annotations

import logging
from functools import partial
import math
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import BenchmarkBackbone
from ._preproc import (
    _ESAT_DATASETS,
    assert_batch_homogeneity,
    assert_finite,
    clip_uv,
    read_sampling_rate,
    resample_poly_with_fallback,
    snap_to_epoch_length,
    strip_zero_channels,
    unit_to_uv,
)
from neuroatlas.benchmarking_helpers import CheckpointSpec

_log = logging.getLogger(__name__)

_PRETRAIN_SFREQ = 200.0
_PATCH_SIZE = 200
_MAX_N_PATCHES = 256
_EMBED_DIM = 200
_N_BRANCHES = 4
_CLIP_UV = 500.0  # paper §3.1

# ESAT token-budget channel cap: when C × n_time > 256, keep channels
# in sleep-staging priority order.
_CHANNEL_PRIORITY = [
    b'c3', b'c4', b'f3', b'f4', b'o1', b'o2',
    b'fp1', b'fp2', b'fz', b'cz', b'pz',
]

# CHB-MIT-BIDS double-banana 18-pair → anchor (collision-free; same scheme
# as EEGPT ``_BIPOLAR_TO_STANDARD``). Anchor is the more frontal /
# midline electrode of each pair, written as the lowercase 10-10 vocabulary
# key in ``CH_NAMES_GLOBAL``: legacy temporal anchors are translated
# T3→t7, T4→t8, T5→p7, T6→p8 (same convention as
# ``eegpt._LEGACY_1020_TO_1010``; pattern copied to avoid cross-module
# imports per ``feedback_no_cross_repo_deps``). All 18 pairs reuse 18 of
# 99 ``CH_NAMES_GLOBAL`` spatial-embedding rows. Pseudo-unipolar evaluation
# per ``project_chbmit_pseudo.md`` — bipolar reference is preserved (no LS
# reconstruction); results inherit the asterisk.
_BIPOLAR_TO_NEURORVQ: Dict[str, str] = {
    "FP1-F7": "f7",  "F7-T3": "t7",  "T3-T5": "p7",  "T5-O1": "o1",
    "FP2-F8": "f8",  "F8-T4": "t8",  "T4-T6": "p8",  "T6-O2": "o2",
    "FP1-F3": "fp1", "F3-C3": "f3",  "C3-P3": "c3",  "P3-O1": "p3",
    "FP2-F4": "fp2", "F4-C4": "f4",  "C4-P4": "c4",  "P4-O2": "p4",
    "FZ-CZ":  "fz",  "CZ-PZ": "pz",
}
# Module-load-time guard: collision means two pairs route to the same
# vocabulary slot, which silently shares one spatial embedding row.
assert len(set(_BIPOLAR_TO_NEURORVQ.values())) == len(
    _BIPOLAR_TO_NEURORVQ
), "_BIPOLAR_TO_NEURORVQ has duplicate target slots — fix the table."


def _build_model(device: str = "cpu"):
    from .neurorvq_arch import NeuroRVQFM

    model = NeuroRVQFM(
        n_patches=_MAX_N_PATCHES,
        patch_size=_PATCH_SIZE,
        in_chans=1,
        out_chans=8,
        num_classes=0,
        embed_dim=_EMBED_DIM,
        depth=12,
        num_heads=10,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=1e-5,
        init_scale=0.001,
        vocab_size=8192,
        use_as_encoder=True,
        use_for_pretraining=False,
    )
    model.to(device)
    return model


def _map_channels(
    ch_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Map dataset channel names to NeuroRVQ's global channel list.

    Returns (ch_mask, spatial_indices, matched_names) where:
    - ch_mask: bool array over input channels (True = channel is in global list)
    - spatial_indices: int indices into CH_NAMES_GLOBAL for matched channels
    - matched_names: the matched channel names (bytes)
    """
    from .neurorvq_arch import CH_NAMES_GLOBAL

    # CHB-MIT-BIDS bipolar pairs → anchor (10-10 lowercase). Non-mapped names
    # (e.g. "FP1", "Fp1") fall through to the existing case-insensitive
    # `np.isin` lookup against `CH_NAMES_GLOBAL`.
    translated = [
        _BIPOLAR_TO_NEURORVQ.get(c.upper().strip(), c) for c in ch_names
    ]
    encoded = np.array([c.lower().encode() for c in translated])
    ch_mask = np.isin(encoded, CH_NAMES_GLOBAL)
    matched_b = encoded[ch_mask]
    dropped_b = encoded[~ch_mask]

    spatial_indices = np.array([
        int(np.where(CH_NAMES_GLOBAL == c)[0][0]) for c in matched_b
    ])
    matched = [b.decode() for b in matched_b.tolist()]
    dropped = [b.decode() for b in dropped_b.tolist()]
    return ch_mask, spatial_indices, matched, dropped


def _create_embedding_ix(
    n_time: int,
    n_channels: int,
    spatial_indices: np.ndarray,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build temporal and spatial embedding index tensors.

    Temporal indices are right-aligned within the 256-patch window.
    """
    temp = torch.arange(_MAX_N_PATCHES - n_time, _MAX_N_PATCHES)
    temp = temp.repeat(n_channels).reshape(1, -1).to(device)

    spat = torch.from_numpy(spatial_indices).long()
    spat = spat.repeat_interleave(n_time).reshape(1, -1).to(device)
    return temp, spat


class NeuroRVQBackbone(BenchmarkBackbone):
    """NeuroRVQ EEG foundation model backbone for frozen linear probing.

    Embedding extraction pipeline:
    1. Filter input channels to those present in NeuroRVQ's 99-channel list.
    2. Resample to 200 Hz, pad to multiple of patch_size (200 samples).
    3. Reshape to (B, n_ch, n_time_patches, 200).
    4. Forward through the 4-branch multi-scale encoder + transformer.
    5. Concatenate 4 branch outputs and mean-pool → 800-dim embedding.
    """

    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        checkpoint_path = Path(spec.checkpoint_path or "")
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"NeuroRVQ checkpoint not found at {checkpoint_path}. "
                "Download from https://huggingface.co/ntinosbarmpas/NeuroRVQ"
            )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = _build_model(self.device)

        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        self._load_report = {
            "missing_keys": list(missing[:20]),
            "unexpected_keys": list(unexpected[:20]),
        }
        if missing:
            _log.info("NeuroRVQ missing keys (first 10): %s", missing[:10])
        if unexpected:
            _log.info("NeuroRVQ unexpected keys (first 10): %s", unexpected[:10])

        self.model.eval()

        self.pooling_strategy = getattr(spec, 'pooling_strategy', None) or "mean"

        self._ch_mask: Optional[np.ndarray] = None
        self._spatial_indices: Optional[np.ndarray] = None
        self._matched_names: Optional[List[str]] = None

    def _resolve_channels(self, batch) -> None:
        """One-time channel resolution from batch metadata."""
        if self._ch_mask is not None:
            return
        meta = batch.get("meta", [{}])
        ch_names = meta[0].get("channels") if meta else None
        if ch_names is not None:
            self._ch_mask, self._spatial_indices, self._matched_names, _dropped = (
                _map_channels(ch_names)
            )
            n_matched = int(self._ch_mask.sum())
            _log.info(
                "NeuroRVQ: matched %d / %d input channels to global list",
                n_matched, len(ch_names),
            )
        else:
            self._ch_mask = None
            self._spatial_indices = None
            _log.warning(
                "NeuroRVQ: no channel names in batch metadata; "
                "using all channels with sequential spatial indices"
            )

    def extract_embeddings(self, batch) -> np.ndarray:
        signals = batch["signals"]
        x = signals.get("eeg")
        if x is None:
            x = signals.get("full_signal")
            if x is None:
                raise KeyError(
                    "NeuroRVQ expects batch['signals']['eeg'] or ['full_signal']."
                )
        x = x.to(self.device, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, C, T = x.shape
        meta = batch.get("meta") or [{}]
        dataset = meta[0].get("dataset", "") if meta else ""

        if dataset in _ESAT_DATASETS:
            # ── ESAT preprocessing pipeline ──────────────────────────
            assert_batch_homogeneity(meta, where="neurorvq:input")

            # Unit → µV (NeuroRVQ's ±500 µV clip operates in µV space)
            unit = meta[0].get("unit") if meta else None
            x = unit_to_uv(x, unit)

            # Strip all-zero channels
            ch_names_raw = (meta[0].get("channels") or [])[:C]
            if ch_names_raw:
                x, ch_names_raw, _ = strip_zero_channels(x, ch_names_raw)
                B, C, T = x.shape
            if C == 0:
                return np.full((B, 800), np.nan, dtype=np.float32)

            # Channel vocabulary filtering
            if ch_names_raw:
                ch_mask, spatial_indices, matched, dropped = _map_channels(ch_names_raw)
                if ch_mask.sum() < C:
                    keep = torch.tensor(
                        np.where(ch_mask)[0], device=self.device, dtype=torch.long,
                    )
                    x = x[:, keep, :]
                    C = x.shape[1]
                if C == 0:
                    return np.full((B, 800), np.nan, dtype=np.float32)

                # Deduplicate channels sharing the same spatial index
                seen_ix = {}
                unique_mask = []
                for i, sid in enumerate(spatial_indices):
                    sid_int = int(sid)
                    if sid_int not in seen_ix:
                        seen_ix[sid_int] = matched[i]
                        unique_mask.append(True)
                    else:
                        unique_mask.append(False)
                unique_arr = np.array(unique_mask)
                if not unique_arr.all():
                    keep_u = torch.tensor(
                        np.where(unique_arr)[0], device=self.device, dtype=torch.long,
                    )
                    x = x[:, keep_u, :]
                    spatial_indices = spatial_indices[unique_arr]
                    matched = [matched[i] for i in range(len(matched)) if unique_arr[i]]
                    C = x.shape[1]

                # Token-budget channel cap (C × n_time ≤ 256)
                epoch_sec = meta[0].get("epoch_seconds", 30.0)
                n_time_est = max(1, int(round(float(epoch_sec) * _PRETRAIN_SFREQ / _PATCH_SIZE)))
                max_ch = _MAX_N_PATCHES // n_time_est
                if C > max_ch:
                    encoded_matched = [m.lower().encode() for m in matched]
                    prio = {ch: i for i, ch in enumerate(_CHANNEL_PRIORITY)}
                    order = sorted(
                        range(C),
                        key=lambda i: (prio.get(encoded_matched[i], len(prio)), i),
                    )
                    keep_idx = sorted(order[:max_ch])
                    keep_t = torch.tensor(keep_idx, device=self.device, dtype=torch.long)
                    x = x[:, keep_t, :]
                    spatial_indices = spatial_indices[np.array(keep_idx, dtype=np.intp)]
                    matched = [matched[i] for i in keep_idx]
                    C = x.shape[1]

                self._matched_names = matched
                self._spatial_indices = spatial_indices
                self._ch_mask = ch_mask
            else:
                self._resolve_channels(batch)

            # Resample to 200 Hz
            meta_sfreq = read_sampling_rate(meta[0]) if meta else None
            src_sfreq_f = float(meta_sfreq) if meta_sfreq else float(_PRETRAIN_SFREQ)
            _backend = "scipy" if dataset in _ESAT_DATASETS else "auto"
            x, resample_method = resample_poly_with_fallback(x, src_sfreq_f, _PRETRAIN_SFREQ, backend=_backend)
            x = snap_to_epoch_length(x, float(_PRETRAIN_SFREQ), meta)

            # ±500 µV clip (paper §3.1)
            x, _clip_frac = clip_uv(x, _CLIP_UV)

            assert_finite(x, "neurorvq:output")

            T_rs = x.shape[-1]
            n_time = T_rs // _PATCH_SIZE
            if n_time == 0:
                return np.full((B, 800), np.nan, dtype=np.float32)
            x = x[..., :n_time * _PATCH_SIZE]
            if C * n_time > _MAX_N_PATCHES:
                raise ValueError(
                    f"NeuroRVQ: token count C×n_time={C * n_time} > {_MAX_N_PATCHES}"
                )

            x = x.reshape(B, C, n_time, _PATCH_SIZE)
            if self._spatial_indices is not None:
                spatial_idx = self._spatial_indices
            else:
                spatial_idx = np.arange(C)
            temp_ix, spat_ix = _create_embedding_ix(n_time, C, spatial_idx, self.device)

            with torch.no_grad():
                x1, x2, x3, x4 = self.model.forward_encoder(x, temp_ix, spat_ix)
            features = torch.cat([x1, x2, x3, x4], dim=-1)
            features = features.mean(dim=1)
            return features.detach().cpu().numpy()

        # ── Non-ESAT path (unchanged) ────────────────────────────────
        # 1. Resolve channel mapping once
        self._resolve_channels(batch)

        if self._ch_mask is not None and self._ch_mask.sum() < C:
            keep = torch.tensor(
                np.where(self._ch_mask)[0], device=self.device, dtype=torch.long,
            )
            x = x[:, keep, :]
            C = x.shape[1]

        # 2. Resample to 200 Hz
        # Adapter contract publishes the canonical "sampling_rate" key; some
        # legacy datasets also carry "sfreq" / "fs". Use the shared
        # ``read_sampling_rate`` helper (covers sampling_rate + sfreq); fall
        # back to "fs" for adapters that only carry the legacy key.
        first_meta = meta[0] if meta else {}
        try:
            input_sfreq = read_sampling_rate(first_meta) if first_meta else None
        except ValueError:
            input_sfreq = None
        if input_sfreq is None and first_meta:
            input_sfreq = first_meta.get("fs")

        if input_sfreq is not None and abs(input_sfreq - _PRETRAIN_SFREQ) > 0.5:
            from scipy.signal import resample_poly

            up = int(_PRETRAIN_SFREQ)
            down = int(input_sfreq)
            g = gcd(up, down)
            x_np = x.cpu().numpy()
            x_np = resample_poly(x_np, up // g, down // g, axis=-1)
            x = torch.from_numpy(x_np.astype(np.float32)).to(self.device)

        _, _, T_rs = x.shape

        # 3. Pad temporal dimension to multiple of patch_size
        remainder = T_rs % _PATCH_SIZE
        if remainder != 0:
            pad = _PATCH_SIZE - remainder
            x = torch.nn.functional.pad(x, (0, pad))
            T_rs = x.shape[-1]

        n_time = T_rs // _PATCH_SIZE

        # 4. Reshape to (B, n_ch, n_time_patches, patch_size)
        x = x.reshape(B, C, n_time, _PATCH_SIZE)

        # 5. Build embedding indices
        if self._spatial_indices is not None:
            spatial_idx = self._spatial_indices
        else:
            spatial_idx = np.arange(C)

        temp_ix, spat_ix = _create_embedding_ix(
            n_time, C, spatial_idx, self.device,
        )

        # 6. Forward through encoder
        with torch.no_grad():
            x1, x2, x3, x4 = self.model.forward_encoder(
                x, temp_ix, spat_ix,
            )

        # 7. Concatenate 4 branches and pool
        features = torch.cat([x1, x2, x3, x4], dim=-1)  # (B, seq, 800)
        if self.pooling_strategy == "flatten":
            features = features.reshape(B, -1)  # (B, seq*800)
        else:
            features = features.mean(dim=1)  # (B, 800)

        return features.detach().cpu().numpy()

    def extract_embeddings_perpatch(self, batch) -> np.ndarray:
        signals = batch["signals"]
        x = signals.get("eeg")
        if x is None:
            x = signals.get("full_signal")
            if x is None:
                raise KeyError(
                    "NeuroRVQ expects batch['signals']['eeg'] or ['full_signal']."
                )
        x = x.to(self.device, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, C, T = x.shape
        meta = batch.get("meta") or [{}]
        dataset = meta[0].get("dataset", "") if meta else ""

        if dataset in _ESAT_DATASETS:
            # ── ESAT preprocessing (mirrors extract_embeddings) ──────
            assert_batch_homogeneity(meta, where="neurorvq_perpatch:input")

            unit = meta[0].get("unit") if meta else None
            x = unit_to_uv(x, unit)

            ch_names_raw = (meta[0].get("channels") or [])[:C]
            if ch_names_raw:
                x, ch_names_raw, _ = strip_zero_channels(x, ch_names_raw)
                B, C, T = x.shape
            if C == 0:
                return np.full((B, 1, 800), np.nan, dtype=np.float32)

            if ch_names_raw:
                ch_mask, spatial_indices, matched, dropped = _map_channels(ch_names_raw)
                if ch_mask.sum() < C:
                    keep = torch.tensor(
                        np.where(ch_mask)[0], device=self.device, dtype=torch.long,
                    )
                    x = x[:, keep, :]
                    C = x.shape[1]
                if C == 0:
                    return np.full((B, 1, 800), np.nan, dtype=np.float32)

                # Deduplicate channels sharing the same spatial index
                seen_ix = {}
                unique_mask = []
                for i, sid in enumerate(spatial_indices):
                    sid_int = int(sid)
                    if sid_int not in seen_ix:
                        seen_ix[sid_int] = matched[i]
                        unique_mask.append(True)
                    else:
                        unique_mask.append(False)
                unique_arr = np.array(unique_mask)
                if not unique_arr.all():
                    keep_u = torch.tensor(
                        np.where(unique_arr)[0], device=self.device, dtype=torch.long,
                    )
                    x = x[:, keep_u, :]
                    spatial_indices = spatial_indices[unique_arr]
                    matched = [matched[i] for i in range(len(matched)) if unique_arr[i]]
                    C = x.shape[1]

                # Token-budget channel cap
                epoch_sec = meta[0].get("epoch_seconds", 30.0)
                n_time_est = max(1, int(round(float(epoch_sec) * _PRETRAIN_SFREQ / _PATCH_SIZE)))
                max_ch = _MAX_N_PATCHES // n_time_est
                if C > max_ch:
                    encoded_matched = [m.lower().encode() for m in matched]
                    prio = {ch: i for i, ch in enumerate(_CHANNEL_PRIORITY)}
                    order = sorted(
                        range(C),
                        key=lambda i: (prio.get(encoded_matched[i], len(prio)), i),
                    )
                    keep_idx = sorted(order[:max_ch])
                    keep_t = torch.tensor(keep_idx, device=self.device, dtype=torch.long)
                    x = x[:, keep_t, :]
                    spatial_indices = spatial_indices[np.array(keep_idx, dtype=np.intp)]
                    matched = [matched[i] for i in keep_idx]
                    C = x.shape[1]

                self._matched_names = matched
                self._spatial_indices = spatial_indices
                self._ch_mask = ch_mask
            else:
                self._resolve_channels(batch)

            meta_sfreq = read_sampling_rate(meta[0]) if meta else None
            src_sfreq_f = float(meta_sfreq) if meta_sfreq else float(_PRETRAIN_SFREQ)
            _backend = "scipy"
            x, _ = resample_poly_with_fallback(x, src_sfreq_f, _PRETRAIN_SFREQ, backend=_backend)
            x = snap_to_epoch_length(x, float(_PRETRAIN_SFREQ), meta)

            x, _clip_frac = clip_uv(x, _CLIP_UV)
            assert_finite(x, "neurorvq_perpatch:output")

            T_rs = x.shape[-1]
            n_time = T_rs // _PATCH_SIZE
            if n_time == 0:
                return np.full((B, 1, 800), np.nan, dtype=np.float32)
            x = x[..., :n_time * _PATCH_SIZE]
            if C * n_time > _MAX_N_PATCHES:
                raise ValueError(
                    f"NeuroRVQ perpatch: token count C*n_time={C * n_time} > {_MAX_N_PATCHES}"
                )

            x = x.reshape(B, C, n_time, _PATCH_SIZE)
            if self._spatial_indices is not None:
                spatial_idx = self._spatial_indices
            else:
                spatial_idx = np.arange(C)
            temp_ix, spat_ix = _create_embedding_ix(n_time, C, spatial_idx, self.device)

            with torch.no_grad():
                x1, x2, x3, x4 = self.model.forward_encoder(x, temp_ix, spat_ix)
            features = torch.cat([x1, x2, x3, x4], dim=-1)  # (B, seq, 800)
            return features.detach().cpu().numpy()

        # ── Non-ESAT path (unchanged) ────────────────────────────────
        self._resolve_channels(batch)

        if self._ch_mask is not None and self._ch_mask.sum() < C:
            keep = torch.tensor(
                np.where(self._ch_mask)[0], device=self.device, dtype=torch.long,
            )
            x = x[:, keep, :]
            C = x.shape[1]

        first_meta = meta[0] if meta else {}
        try:
            input_sfreq = read_sampling_rate(first_meta) if first_meta else None
        except ValueError:
            input_sfreq = None
        if input_sfreq is None and first_meta:
            input_sfreq = first_meta.get("fs")

        T_original = T
        if input_sfreq is not None and abs(input_sfreq - _PRETRAIN_SFREQ) > 0.5:
            from scipy.signal import resample_poly
            up = int(_PRETRAIN_SFREQ)
            down = int(input_sfreq)
            g = gcd(up, down)
            x_np = x.cpu().numpy()
            x_np = resample_poly(x_np, up // g, down // g, axis=-1)
            x = torch.from_numpy(x_np.astype(np.float32)).to(self.device)

        _, _, T_rs = x.shape
        n_valid = math.ceil(T_rs / _PATCH_SIZE)

        remainder = T_rs % _PATCH_SIZE
        if remainder != 0:
            pad = _PATCH_SIZE - remainder
            x = torch.nn.functional.pad(x, (0, pad))
            T_rs = x.shape[-1]

        n_time = T_rs // _PATCH_SIZE
        x = x.reshape(B, C, n_time, _PATCH_SIZE)

        if self._spatial_indices is not None:
            spatial_idx = self._spatial_indices
        else:
            spatial_idx = np.arange(C)

        temp_ix, spat_ix = _create_embedding_ix(n_time, C, spatial_idx, self.device)

        with torch.no_grad():
            x1, x2, x3, x4 = self.model.forward_encoder(x, temp_ix, spat_ix)

        features = torch.cat([x1, x2, x3, x4], dim=-1)  # (B, seq, 800)
        features = features[:, :n_valid, :]  # drop zero-padded patches
        return features.detach().cpu().numpy()

    def metadata(self) -> Dict[str, object]:
        n_matched = (
            int(self._ch_mask.sum()) if self._ch_mask is not None else "unknown"
        )
        return {
            **super().metadata(),
            **self._load_report,
            "device": self.device,
            "pretrain_sfreq": _PRETRAIN_SFREQ,
            "patch_size": _PATCH_SIZE,
            "n_branches": _N_BRANCHES,
            "embedding_reduction": "concat_4_branches_mean_pool",
            "n_matched_channels": n_matched,
        }
