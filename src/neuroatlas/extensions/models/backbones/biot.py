"""BIOT backbone wrapper.

Window contract:
    Strict: input must resample to ``spec.expected_epoch_seconds × 200``
    samples. Mismatched lengths raise ``ValueError`` — we never silently
    stretch or pad. BIOT's pretraining window is 10 s; longer / shorter
    inputs are accepted as long as the token count fits the encoder's
    ``max_seq_len`` (handled by the pretrained model itself).

Model-specific transforms applied in ``_prepare_input``:
    1. Batch-meta homogeneity check (fs / unit / channels).
    2. ``unit_to_uv`` — convert declared unit to µV.
    3. Per-channel DC removal (pre-normalization hygiene).
    4. Channel-count validation: BIOT is pretrained at 16 channels;
       non-16-ch datasets are supposed to be handled via the
       ``src/neuroatlas/configs/channel_maps/<dataset>.yaml`` opt-in layer (or
       marked ``skip`` there). Raise loudly rather than zero-padding.
    5. fs resample to 200 Hz (shared helper; source fs from
       ``meta[i]["sampling_rate"]`` with legacy ``sfreq`` fallback).
    6. Per-channel 95-percentile normalization (paper §2.1 formula:
       ``S[i] / (quantile(|S[i]|, 0.95) + 1e-8)``). Gated by
       ``runtime_overrides["apply_recording_normalization"]``.
    7. Finite check on output.

BIOT is **scale-invariant** by architecture — the percentile
normalization is the model's unit-neutralizing first step, so
``apply_amplitude_scale`` is effectively a no-op for this wrapper
(BIOT has no fixed amplitude formula).

Signal cleanliness (bandpass, notch) is the dataset preprocessor's
responsibility. See AGENT_GUIDE.md §7.1.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from neuroatlas import quiet

from .base import BenchmarkBackbone
from ._preproc import (
    _PAPER_PREPROC_DATASETS,
    StageTimer,
    assert_batch_homogeneity,
    assert_finite,
    is_bci_batch,
    percentile_normalize,
    read_sampling_rate,
    resample_poly_with_fallback,
    run_names,
    snap_to_epoch_length,
    unit_to_uv,
)
from neuroatlas.benchmarking_helpers import CheckpointSpec


logger = logging.getLogger(__name__)

_TARGET_SFREQ = 200.0
_N_CHANNELS = 16  # PREST pretraining count; see MODEL_CONTRACTS.md §1.

_BIOT_SLOT_ORDER = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]
_BIOT_NAME_TO_SLOT = {name: i for i, name in enumerate(_BIOT_SLOT_ORDER)}

_BIOT_DERIVATION = [
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]

_10_20_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def _biot_slot_plan(ch_names):
    """Pure name→slot decision for the 16-slot BIOT double-banana layout.

    Returns ``(slot_provenance, name_to_idx, derivations, approximations,
    padded)`` where ``slot_provenance[i]`` is one of:
    ``("direct", src_idx)`` (input already carries the bipolar pair),
    ``("bipolar", idx_a, idx_b)`` (derive ``A-B`` from two electrodes),
    ``("approx", idx)`` (single-electrode approximation), or ``("pad",)``
    (no source → zero slot). No tensors — shared by ``_prepare_input`` and
    ``describe_channel_mapping`` so they cannot diverge.
    """
    name_to_idx = {}
    for i, name in enumerate(ch_names):
        upper = name.upper()
        name_to_idx[upper] = i
        canonical = _10_20_ALIASES.get(upper)
        if canonical and canonical not in name_to_idx:
            name_to_idx[canonical] = i

    slot_provenance: List[tuple] = [("pad",)] * _N_CHANNELS
    derivations, approximations, padded = [], [], []
    for slot_idx, slot_name in enumerate(_BIOT_SLOT_ORDER):
        elec_a, elec_b = _BIOT_DERIVATION[slot_idx]
        if slot_name in name_to_idx:
            slot_provenance[slot_idx] = ("direct", name_to_idx[slot_name])
            derivations.append(f"{slot_name}(direct)")
            continue
        idx_a = name_to_idx.get(elec_a)
        idx_b = name_to_idx.get(elec_b)
        if idx_a is not None and idx_b is not None:
            slot_provenance[slot_idx] = ("bipolar", idx_a, idx_b)
            derivations.append(f"{slot_name}={elec_a}-{elec_b}")
        elif idx_a is not None:
            slot_provenance[slot_idx] = ("approx", idx_a)
            approximations.append(f"{slot_name}~{elec_a}")
        elif idx_b is not None:
            slot_provenance[slot_idx] = ("approx", idx_b)
            approximations.append(f"{slot_name}~{elec_b}")
        else:
            padded.append(slot_name)
    return slot_provenance, name_to_idx, derivations, approximations, padded


def describe_channel_mapping(channels):
    """Name-only channel provenance (see channel_provenance.ChannelMapping).

    Reuses ``_biot_slot_plan`` — the exact name→slot decision the wrapper
    runs. Each of the 16 fixed double-banana slots is reported as
    ``filled`` (input already carries the pair), ``derived`` (A−B from two
    electrodes), ``approx`` (single-electrode stand-in), or ``zero_inserted``
    (no source). Raises (zero matched) are captured upstream.
    """
    from neuroatlas.benchmarking_helpers.channels.channel_provenance import (
        ChannelMapping,
        ChannelSlot,
    )

    slot_provenance, _n2i, _d, _a, _p = _biot_slot_plan(list(channels))
    status_map = {"direct": "filled", "bipolar": "derived", "approx": "approx"}
    slots = []
    matched = False
    for slot_idx, prov in enumerate(slot_provenance):
        target = _BIOT_SLOT_ORDER[slot_idx]
        kind = prov[0]
        if kind == "pad":
            slots.append(ChannelSlot(target=target, source=None, status="zero_inserted"))
            continue
        matched = True
        if kind == "bipolar":
            src = f"{channels[prov[1]]}-{channels[prov[2]]}"
        else:
            src = str(channels[prov[1]])
        slots.append(ChannelSlot(target=target, source=src, status=status_map[kind]))
    if not matched:
        raise ValueError(
            f"BIOT: zero matched_channels for any BIOT double-banana slot. Got: {list(channels)}"
        )
    return ChannelMapping(model_family="biot", layout_kind="fixed", slots=slots)


def _strip_encoder_prefix(state):
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported BIOT checkpoint payload type {type(state)!r}.")
    if any(key.startswith("encoder.") for key in state.keys()):
        return {key[len("encoder."):]: value for key, value in state.items() if key.startswith("encoder.")}
    return state


def _drop_constant_index(state, encoder):
    """Reconcile the PREST checkpoint's ``index`` buffer with the installed braindecode.

    The checkpoint stores ``index`` = arange(n_chans), the channel-token
    lookup table. braindecode < 1.5 registers it as a persistent buffer, so
    it must be loaded; later releases rebuild it in ``__init__`` as a
    non-persistent buffer, so strict loading rejects it as unexpected
    (``Error(s) in loading state_dict for _BIOTEncoder``, F-079). It carries
    no learned value: drop it only when the model has no such key and the
    stored value is exactly the arange the model builds itself.
    """
    if "index" not in state or "index" in encoder.state_dict():
        return state
    stored = state["index"]
    built = getattr(encoder, "index", None)
    expected = built if built is not None else torch.arange(len(stored))
    if not torch.equal(stored.to(torch.long).cpu(), expected.to(torch.long).cpu()):
        return state                       # not the constant: let load_strict report it
    return {k: v for k, v in state.items() if k != "index"}


#: A DC offset above this fraction of the 95th-percentile amplitude is large
#: enough to distort BIOT's q95 normaliser, so the wrapper warns about it.
_DC_OFFSET_WARN_RATIO = 0.05


class BIOTBackbone(BenchmarkBackbone):
    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        from ._checkpoint_download import ensure_checkpoint
        checkpoint_path = ensure_checkpoint(
            spec.checkpoint_path or "",
            source_type=spec.source_type,
            source_reference=spec.source_reference,
        )
        from braindecode.models import BIOT

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.epoch_seconds = float(spec.expected_epoch_seconds)
        self.target_sfreq = _TARGET_SFREQ
        self.target_len = int(round(self.epoch_seconds * self.target_sfreq))
        self.model = BIOT(
            n_chans=_N_CHANNELS,
            n_times=self.target_len,
            sfreq=int(self.target_sfreq),
            n_outputs=5,
            return_feature=True,
        )
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        raw_keys = list(state.keys()) if isinstance(state, dict) else []
        state = _drop_constant_index(_strip_encoder_prefix(state), self.model.encoder)
        from ._state_dict import load_strict

        load_strict(self.model.encoder, state, f"BIOT checkpoint {checkpoint_path}",
                    packages=("braindecode", "torch"))
        self.model.eval()
        self.model.to(self.device)

        self._load_report = {
            "checkpoint_path": str(checkpoint_path),
            "load_mode": "strict=True (encoder submodule)",
            "weight_source": "local_checkpoint",
            "n_state_keys_loaded": len(state),
            "extracted_prefix": "encoder.",
            "ignored_non_encoder_keys": [
                k for k in raw_keys if not k.startswith("encoder.")
            ][:20],
        }

        self._banner_logged = False
        self._dc_offset_warned = False
        self._last_report: Dict[str, Any] = {
            "input_sfreq_observed": None,
            "input_resample_method": "unknown",
        }

        import os as _os
        compile_mode = _os.environ.get("COMPILE_BACKBONE")
        if compile_mode:
            self.model = torch.compile(
                self.model,
                mode=compile_mode if compile_mode in {"default", "reduce-overhead", "max-autotune"} else "reduce-overhead",
            )

    def _resolve_overrides(self) -> Dict[str, bool]:
        ov = getattr(self.spec, "runtime_overrides", {}) or {}
        return {
            "apply_amplitude_scale": bool(ov.get("apply_amplitude_scale", True)),
            "apply_recording_normalization": bool(
                ov.get("apply_recording_normalization", True)
            ),
        }

    def _log_banner_once(self, apply_norm: bool) -> None:
        if self._banner_logged:
            return
        norm_repr = "per_channel_q95" if apply_norm else "keep"
        logger.info(
            "[backbone=biot] fs=%g Hz window=%g s (strict, %d samples) "
            "scale=%s ref=keep n_chans=%d",
            self.target_sfreq, self.epoch_seconds, self.target_len,
            norm_repr, _N_CHANNELS,
        )
        self._banner_logged = True

    @staticmethod
    def _build_q95_vector(
        meta: Dict[str, Any],
        real_mask: torch.Tensor,
        slot_provenance,
    ):
        q95_raw = meta.get("recording_q95")
        q95_bipolar = meta.get("recording_q95_bipolar")
        if q95_raw is None:
            return None
        if slot_provenance is None:
            return list(q95_raw) if len(q95_raw) == _N_CHANNELS else None
        q95_vec = [0.0] * _N_CHANNELS
        for slot_idx in range(_N_CHANNELS):
            prov = slot_provenance[slot_idx]
            if prov[0] == "direct":
                q95_vec[slot_idx] = float(q95_raw[prov[1]])
            elif prov[0] == "bipolar":
                idx_a, idx_b = prov[1], prov[2]
                key = f"{min(idx_a, idx_b)},{max(idx_a, idx_b)}"
                if q95_bipolar and key in q95_bipolar:
                    q95_vec[slot_idx] = float(q95_bipolar[key])
                else:
                    return None
            elif prov[0] == "approx":
                q95_vec[slot_idx] = float(q95_raw[prov[1]])
        return q95_vec

    def _prepare_input(self, batch):
        signals = batch["signals"]
        x = signals.get("eeg")
        if x is None:
            full = signals.get("full_signal")
            if full is None:
                raise KeyError("BIOT backbone expects batch['signals']['eeg'] or ['full_signal'].")
            x = full[:, :1]

        with StageTimer("biot", "h2d"):
            x = x.to(self.device, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(
                f"BIOT expected signal of shape (B, C, T) or (B, T); got {tuple(x.shape)}."
            )

        meta = batch.get("meta") or [{}]
        assert_batch_homogeneity(meta, where="biot:input")

        # Unit conversion — BIOT is scale-invariant after the q95 norm,
        # but the homogeneity check + per-channel DC removal operates on
        # µV-space for consistency with other wrappers.
        unit = meta[0].get("unit") if meta else None
        x = unit_to_uv(x, unit)

        # Each window's per-channel mean is removed before the q95 scaling
        # below. BIOT's own loaders only divide by quantile(|X|, 0.95,
        # axis=-1); on a DC-biased input that quantile is dominated by the
        # offset and x/scale collapses to ~1.0 everywhere. When the offset is
        # large, -v and --log say so once (INFO: it asks nothing of the user).
        with StageTimer("biot", "center_chans"):
            dc = x.mean(dim=-1, keepdim=True)
            if not self._dc_offset_warned:
                # Measured the way the normaliser will see it: per channel,
                # along time, which is what percentile_normalize divides by.
                flat = x.abs().reshape(-1, x.shape[-1])
                q95 = torch.quantile(flat, 0.95, dim=-1).reshape(dc.shape)
                usable = torch.isfinite(q95) & (q95 > 0)
                if bool(usable.any()):
                    ratio = float(
                        (dc.abs()[usable] / q95[usable]).max()
                    )
                    if ratio > _DC_OFFSET_WARN_RATIO:
                        dataset, checkpoint = run_names(meta, self.spec.identifier)
                        logger.info(
                            "%s: %s windows carry a DC offset of up to %.0f%% of their "
                            "95th-percentile amplitude; each window's per-channel mean is "
                            "removed before its 95th-percentile scaling",
                            checkpoint or "BIOT", dataset or "the", 100 * ratio,
                        )
                        self._dc_offset_warned = True
            x = x - dc

        ch_names = meta[0].get("channels", []) if meta else []
        B, C_in, T_raw = x.shape

        if x.shape[1] == _N_CHANNELS and not ch_names:
            real_mask = torch.ones(_N_CHANNELS, dtype=torch.bool)
            slot_provenance = None
        else:
            if not ch_names:
                raise ValueError(
                    "BIOT: meta['channels'] required for channel mapping "
                    "when input is not already 16-ch."
                )

            slot_provenance, _n2i, derivations, approximations, padded = (
                _biot_slot_plan(ch_names)
            )

            x_mapped = torch.zeros(B, _N_CHANNELS, T_raw, dtype=x.dtype, device=x.device)
            real_mask = torch.zeros(_N_CHANNELS, dtype=torch.bool)
            for slot_idx, prov in enumerate(slot_provenance):
                kind = prov[0]
                if kind == "direct" or kind == "approx":
                    x_mapped[:, slot_idx, :] = x[:, prov[1], :]
                    real_mask[slot_idx] = True
                elif kind == "bipolar":
                    x_mapped[:, slot_idx, :] = x[:, prov[1], :] - x[:, prov[2], :]
                    real_mask[slot_idx] = True
                # ("pad",) → leave the slot zero-filled

            if not real_mask.any():
                raise ValueError(
                    f"BIOT: zero matched_channels for any BIOT double-banana slot. "
                    f"Got: {ch_names}"
                )

            logger.debug(
                "[biot] derivations=%s approx=%s padded=%s",
                derivations, approximations, padded,
            )
            x = x_mapped

        T = x.shape[-1]
        meta_sfreq = read_sampling_rate(meta[0]) if meta else None
        src_sfreq_f = (
            float(meta_sfreq) if meta_sfreq is not None else T / self.epoch_seconds
        )
        if meta_sfreq is not None:
            expected_T = int(round(src_sfreq_f * self.epoch_seconds))
            if abs(T - expected_T) > 1:
                raise ValueError(
                    f"BIOT: duration mismatch — meta['sampling_rate']="
                    f"{meta_sfreq} Hz and epoch_seconds={self.epoch_seconds:g} s "
                    f"imply {expected_T} samples, but got T={T}."
                )

        dataset = meta[0].get("dataset", "") if meta else ""
        _backend = "scipy" if dataset in _PAPER_PREPROC_DATASETS else "auto"
        with StageTimer("biot", "resample"):
            x, resample_method = resample_poly_with_fallback(
                x.contiguous(), src_sfreq_f, self.target_sfreq, backend=_backend
            )
        x = snap_to_epoch_length(x, self.target_sfreq, meta)
        if x.shape[-1] != self.target_len:
            raise ValueError(
                f"BIOT: window length mismatch — expected {self.target_len} samples "
                f"({self.epoch_seconds:g} s × {self.target_sfreq:g} Hz), got {x.shape[-1]}. "
                f"Input was {T} samples at fs {src_sfreq_f:.3f} Hz."
            )

        overrides = self._resolve_overrides()
        with StageTimer("biot", "q95_norm"):
            if overrides["apply_recording_normalization"]:
                q95_raw = meta[0].get("recording_q95")
                if q95_raw is None:
                    # Each window divided by its own per-channel 95th
                    # percentile of |x| (percentile_normalize: BIOT's own
                    # formula) instead of the recording's. Expected on BCI:
                    # the MOABB trials carry no recording statistics.
                    if is_bci_batch(meta):
                        logger.debug("BIOT: BCI trials carry no recording 95th percentile; "
                                     "each trial is scaled by its own per-channel 95th "
                                     "percentile")
                    else:
                        dataset, checkpoint = run_names(meta, self.spec.identifier)
                        quiet.warn_once(
                            logger, f"biot recording_q95 missing:{dataset}:{checkpoint}",
                            "%s gives %s no per-recording 95th percentile, so each window "
                            "is scaled by its own per-channel 95th percentile",
                            dataset or "this dataset", checkpoint or "BIOT")
                    if real_mask.any() and not real_mask.all():
                        real_idx = real_mask.nonzero(as_tuple=True)[0]
                        x_real = percentile_normalize(x[:, real_idx, :], q=0.95)
                        x = x.clone()
                        x[:, real_idx, :] = x_real
                    else:
                        x = percentile_normalize(x, q=0.95)
                else:
                    q95_vec = self._build_q95_vector(
                        meta[0], real_mask, slot_provenance,
                    )
                    if q95_vec is not None:
                        q95 = torch.as_tensor(
                            q95_vec, dtype=x.dtype, device=x.device,
                        ).unsqueeze(0).unsqueeze(-1)
                        q95 = unit_to_uv(q95, unit)
                        real_idx = real_mask.nonzero(as_tuple=True)[0]
                        x = x.clone()
                        x[:, real_idx, :] = x[:, real_idx, :] / (
                            q95[:, real_idx, :] + 1e-8
                        )
                    elif real_mask.any() and not real_mask.all():
                        real_idx = real_mask.nonzero(as_tuple=True)[0]
                        x_real = percentile_normalize(x[:, real_idx, :], q=0.95)
                        x = x.clone()
                        x[:, real_idx, :] = x_real
                    else:
                        x = percentile_normalize(x, q=0.95)
            else:
                quiet.warn_once(
                    logger, "biot normalization off",
                    "BIOT: apply_recording_normalization=False, so its per-channel "
                    "95th-percentile normalization is skipped; the embeddings differ from "
                    "the scale it was pretrained on (an ablation)")

        assert_finite(x, "biot:output")

        self._last_report = {
            "input_sfreq_observed": float(src_sfreq_f),
            "input_resample_method": resample_method,
            "apply_amplitude_scale": overrides["apply_amplitude_scale"],
            "apply_recording_normalization": overrides["apply_recording_normalization"],
        }
        self._log_banner_once(overrides["apply_recording_normalization"])
        return x

    def extract_embeddings(self, batch) -> np.ndarray:
        with StageTimer("biot", "prep_total"):
            x = self._prepare_input(batch)
        with StageTimer("biot", "forward"):
            with torch.inference_mode():
                _, emb = self.model(x)
        with StageTimer("biot", "d2h"):
            out = emb.detach().float().cpu().numpy()
        return out


    def extract_embeddings_perpatch(self, batch) -> np.ndarray:
        with StageTimer("biot", "prep_total"):
            x = self._prepare_input(batch)

        pre_pool = {}

        def _hook(_module, _input, output):
            pre_pool["tokens"] = output

        handle = self.model.encoder.transformer.register_forward_hook(_hook)
        try:
            with StageTimer("biot", "forward"):
                with torch.inference_mode():
                    self.model(x)
        finally:
            handle.remove()

        tokens = pre_pool["tokens"]  # (B, n_channels * n_patches, emb_size)
        return tokens.detach().float().cpu().numpy()

    def metadata(self):
        return {
            **super().metadata(),
            "device": self.device,
            "input_channels_used": ["Fpz-Cz"],
            "target_sfreq": self.target_sfreq,
            "epoch_seconds": self.epoch_seconds,
            "target_len": self.target_len,
            "embedding_reduction": "encoder_internal_mean",
            "wrapper_contract": "model_specific_transforms_only",
            "reference": "keep",
            "weight_load_report": dict(self._load_report),
            "embedding_provenance": {
                "extractor": "encoder_internal_mean",
                "category": "A",
                "embedding_dim": 256,
                "notes": (
                    "Paper-prescribed: BIOT encoder's built-in .mean(dim=1) "
                    "over the concatenated channel-token sequence."
                ),
            },
            **self._last_report,
        }
