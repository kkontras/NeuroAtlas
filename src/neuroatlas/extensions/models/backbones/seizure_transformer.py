"""SeizureTransformer wrapper (Wu et al. 2025, EpilepsyBench 2025 winner).

Supervised baseline: U-Net + self-attention trained on Dianalund + Siena +
TUSZ v2.0.3. Native output is per-sample seizure probability at 256 Hz
(sequence-to-sequence).

Window contract: any duration whose resampled length at 256 Hz is a positive
multiple of 32 (the encoder has 5 pooling stages of /2). The paper's training
window is 60 s = 15 360 samples; the wrapper accepts shorter windows (e.g.
10 s = 2 560 samples) so the model can be benchmarked apples-to-apples with
the 10-s FM probe pipeline. The pretrained Conv1d / BatchNorm / transformer
weights are length-agnostic and load `strict=True` at any in_samples; the
resulting embeddings at non-60-s windows are still Category C and inherit an
extra "shorter context" caveat documented in metadata. Raises on any other
length mismatch — no silent stretching/padding.

Channel contract: the pretrained Conv1d learned slot-positional weights for a
specific 19-channel order taken from the upstream wu_2025 package
(``time_step_level/service/handle_data.py: channel_seq``). The wrapper reorders
the adapter's channels to that order before feeding the encoder. Adapter-side
``meta[i]["channels"]`` must be present.

Channel contract — CHB-MIT bipolar exception: the 18 double-banana pairs
emitted by CHB-MIT-BIDS are mapped to their anatomically-frontal/midline
anchor (``_BIPOLAR_TO_PRETRAIN_SLOT``) before slot resolution. Same
collision-free scheme as EEGPT. Bipolar values are reused as-is at
the anchor's pretrained slot — pseudo-unipolar evaluation per
``project_chbmit_pseudo.md``. CZ ends up unfilled (1/19 zero-padded slot).

Amplitude normalization (per-channel z-score) and resampling to 256 Hz are
model-specific and applied here. Bandpass / notch filtering stays with the
dataset preprocessor per project convention (``feedback_filter_scope``).

Embedding tap = mean over time of (transformer_encoder_output + res_cnn_stack
residual), pre-U-Net-decoder. Category C: no paper-supported frozen pooling
exists; the paper's downstream is the supervised head.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .base import BenchmarkBackbone
from ._preproc import (
    assert_batch_homogeneity,
    assert_finite,
    declared_units_from_batch,
    read_sampling_rate,
    resample_poly_with_fallback,
    unit_to_uv,
)
from .third_party.seizure_transformer import SeizureTransformer as _SeizureTransformer
from neuroatlas.benchmarking_helpers import CheckpointSpec

logger = logging.getLogger(__name__)

_TARGET_SFREQ = 256.0
_PAPER_NATIVE_EPOCH_SECONDS = 60.0  # window the released weights were trained on
_PAPER_NATIVE_LEN = int(_TARGET_SFREQ * _PAPER_NATIVE_EPOCH_SECONDS)  # 15360
_POOLING_STRIDE = 32  # encoder has 5 stages of MaxPool1d(2); in_samples must
                     # be a positive multiple of 2**5 for the U-Net halving to
                     # land on an integer length at every stage.
_MIN_EPOCH_SECONDS = _POOLING_STRIDE / _TARGET_SFREQ  # ~0.125 s — but we hard
                                                       # floor at 1 s below to
                                                       # avoid degenerate windows
_HARD_MIN_EPOCH_SECONDS = 1.0
_EXPECTED_CHANNELS = 19
_EMBEDDING_DIM = 512  # transformer hidden size

# Pretrained slot order — taken verbatim from upstream
# `wu_2025/time_step_level/service/handle_data.py: channel_seq`. The first
# Conv1d's 19×32 weights are slot-positional, so any permutation feeds the
# wrong signal into each filter row → silent feature corruption.
_PRETRAIN_CHANNEL_ORDER: Tuple[str, ...] = (
    "FP1", "F3", "C3", "P3", "O1", "F7", "T3", "T5",
    "FZ", "CZ", "PZ",
    "FP2", "F4", "C4", "P4", "O2", "F8", "T4", "T6",
)
_PRETRAIN_CHANNEL_SET = frozenset(_PRETRAIN_CHANNEL_ORDER)

# CHB-MIT-BIDS double-banana 18-pair → unipolar anchor slot. Anchor is the
# more frontal/midline electrode of each pair (collision-free; same scheme
# as EEGPT `_BIPOLAR_TO_STANDARD`). 18 of 19 pretrain slots are
# filled; CZ is intentionally unfilled and remains zero-padded via the
# existing `zero_padded_in_batch` path. Pseudo-unipolar evaluation per
# `project_chbmit_pseudo.md` — bipolar reference is preserved (no LS
# reconstruction); results inherit the asterisk.
_BIPOLAR_TO_PRETRAIN_SLOT: Dict[str, str] = {
    "FP1-F7": "F7",  "F7-T3": "T3",  "T3-T5": "T5",  "T5-O1": "O1",
    "FP2-F8": "F8",  "F8-T4": "T4",  "T4-T6": "T6",  "T6-O2": "O2",
    "FP1-F3": "FP1", "F3-C3": "F3",  "C3-P3": "C3",  "P3-O1": "P3",
    "FP2-F4": "FP2", "F4-C4": "F4",  "C4-P4": "C4",  "P4-O2": "P4",
    "FZ-CZ": "FZ",   "CZ-PZ": "PZ",
}
# Module-load-time guard: collision means two pairs route to the same slot,
# which silently corrupts the slot-positional Conv1d filters.
assert len(set(_BIPOLAR_TO_PRETRAIN_SLOT.values())) == len(
    _BIPOLAR_TO_PRETRAIN_SLOT
), "_BIPOLAR_TO_PRETRAIN_SLOT has duplicate target slots — fix the table."


def _normalize_channel_label(label: str) -> str:
    """Uppercase, strip whitespace, drop common '-REF' / '-LE' suffixes.

    Adapters publish names like 'EEG FP1-REF', 'Fp1', 'FP1-LE'. Map them all
    to the canonical bare slot name (e.g. 'FP1') used by ``_PRETRAIN_CHANNEL_ORDER``.

    CHB-MIT-BIDS bipolar pairs (e.g. 'FP1-F7') are routed via
    ``_BIPOLAR_TO_PRETRAIN_SLOT`` to their anchor slot before the existing
    pretrain-set check. Asterisked per ``project_chbmit_pseudo.md``.
    """
    s = str(label).strip().upper()
    if s.startswith("EEG "):
        s = s[4:].strip()
    # Bipolar lookup: CHB-MIT publishes pairs verbatim ("FP1-F7" etc.); these
    # legitimately contain "-" and must be matched before suffix stripping.
    if s in _BIPOLAR_TO_PRETRAIN_SLOT:
        return _BIPOLAR_TO_PRETRAIN_SLOT[s]
    for suffix in ("-REF", "-LE", "-AVG"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s.strip()


def _build_channel_perm(
    adapter_channels: Sequence[str],
) -> Tuple[List[Optional[int]], List[str]]:
    """Build per-slot map from pretrain order → adapter channel index.

    Returns ``(slot_map, normalized)`` where ``slot_map[i]`` is the input
    column to copy into pretrain slot ``i``, or ``None`` to zero-pad that
    slot. ``normalized`` is the post-normalization label list (after
    `_normalize_channel_label` has applied any bipolar→anchor rewrite).

    Behaviour:
    - If every pretrain slot is present in the adapter (typical unipolar
      19-ch case), the result is a strict permutation (no None).
    - If some slots are missing AND the adapter delivered fewer than
      ``_EXPECTED_CHANNELS`` channels (CHB-MIT bipolar 18-pair → 18
      anchors, CZ unfilled), missing slots map to ``None`` and the
      caller materialises a zero-filled channel for them. This is the
      pseudo-unipolar bipolar route documented in
      ``project_chbmit_pseudo.md`` — bipolar values land at their anchor
      slot's pretrained Conv1d row; the unfilled slot is reported via
      ``dropped_channels`` as ``zero_padded_in_batch``.
    - Any unrecognised adapter label raises ``ValueError`` (no silent
      drop) — slot identity is critical, so foreign channels can't be
      tolerated.
    """
    norm = [_normalize_channel_label(c) for c in adapter_channels]
    name_to_idx = {n: i for i, n in enumerate(norm)}

    extra = [n for n in norm if n not in _PRETRAIN_CHANNEL_SET]
    if extra:
        raise ValueError(
            f"SeizureTransformer: unexpected channel labels {extra} from "
            f"adapter (normalized {norm}). The pretrained Conv1d uses "
            f"slot-positional weights — only labels in "
            f"{list(_PRETRAIN_CHANNEL_ORDER)} (or bipolar pairs in "
            f"_BIPOLAR_TO_PRETRAIN_SLOT) are accepted."
        )
    if len(norm) > _EXPECTED_CHANNELS:
        raise ValueError(
            f"SeizureTransformer: got {len(norm)} channels from adapter, "
            f"more than the {_EXPECTED_CHANNELS} pretrained slots."
        )
    return [name_to_idx.get(n) for n in _PRETRAIN_CHANNEL_ORDER], norm


def describe_channel_mapping(channels):
    """Name-only channel provenance (see channel_provenance.ChannelMapping).

    Reuses ``_build_channel_perm`` — the exact same name→slot logic the
    wrapper runs — so the report matches inference. Missing slots map to
    ``None`` and are reported as ``zero_inserted`` (e.g. CHB-MIT leaves CZ
    unfilled by design). Raises propagate to the caller (captured upstream).
    """
    from neuroatlas.benchmarking_helpers.channels.channel_provenance import (
        ChannelMapping,
        ChannelSlot,
    )

    slot_map, _norm = _build_channel_perm(channels)
    slots = []
    for slot_idx, src in enumerate(slot_map):
        target = _PRETRAIN_CHANNEL_ORDER[slot_idx]
        if src is None:
            slots.append(ChannelSlot(target=target, source=None, status="zero_inserted"))
        else:
            slots.append(ChannelSlot(target=target, source=str(channels[src]), status="filled"))
    return ChannelMapping(model_family="seizure_transformer", layout_kind="fixed", slots=slots)


def _load_checkpoint(path: Path, device: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"no SeizureTransformer weights at {path}\n"
            "fix: neuroatlas models download seizure_transformer_pretrained"
        )
    return torch.load(str(path), map_location=device, weights_only=False)


class SeizureTransformerBackbone(BenchmarkBackbone):
    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.epoch_seconds = float(spec.expected_epoch_seconds)
        if self.epoch_seconds < _HARD_MIN_EPOCH_SECONDS:
            raise ValueError(
                f"SeizureTransformer: epoch_seconds={self.epoch_seconds:g} too short; "
                f"hard minimum is {_HARD_MIN_EPOCH_SECONDS:g} s."
            )
        target_len = int(round(_TARGET_SFREQ * self.epoch_seconds))
        if target_len % _POOLING_STRIDE != 0 or target_len <= 0:
            raise ValueError(
                f"SeizureTransformer: epoch_seconds={self.epoch_seconds:g} → "
                f"{target_len} samples @ 256 Hz, which is not a positive multiple of "
                f"{_POOLING_STRIDE}. Choose seconds such that round(seconds×256) % "
                f"{_POOLING_STRIDE} == 0 (e.g. 10, 15, 30, 60)."
            )
        self.target_sfreq = _TARGET_SFREQ
        self.target_len = target_len
        self.window_matches_paper = (target_len == _PAPER_NATIVE_LEN)

        ckpt_path = Path(spec.checkpoint_path or "")
        state_dict = _load_checkpoint(ckpt_path, device="cpu")
        # Upstream saves a bare state_dict; some forks wrap it under "model".
        if isinstance(state_dict, dict) and "model" in state_dict and all(
            isinstance(v, dict) for v in state_dict.values()
        ):
            state_dict = state_dict["model"]

        # Pretrained weights are length-agnostic for Conv1d / BatchNorm /
        # transformer (verified strict=True at in_samples ∈ {2560, 15360}).
        # Encoder.paddings + Decoder.crops are recomputed at instantiation.
        self.model = _SeizureTransformer(
            in_channels=_EXPECTED_CHANNELS,
            in_samples=target_len,
            drop_rate=0.1,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)

        self._banner_logged = False
        self._last_unit_declared: str | None = None
        self._last_dropped_channels: list[dict] = []
        self._last_input_channels: list[str] = list(_PRETRAIN_CHANNEL_ORDER)
        self._last_structurally_missing: list[str] = []

        logger.info(
            "SeizureTransformer loaded (ckpt=%s, in=(%d,%d), epoch_s=%g, "
            "paper_native=%s, device=%s)",
            ckpt_path.name, _EXPECTED_CHANNELS, target_len, self.epoch_seconds,
            self.window_matches_paper, self.device,
        )

    # --- input pipeline --------------------------------------------------

    def _log_banner_once(self, fs_in: float, unit: str) -> None:
        if self._banner_logged:
            return
        logger.info(
            "[backbone=seizure_transformer] fs_in=%.3f→%g Hz window=%g s "
            "(paper_native=%g s, match=%s) channels=19 unipolar(pretrain_order) "
            "scale=per_channel_zscore ref=keep weight_source=docker_pretrained "
            "alignment_category=C unit=%s",
            fs_in, _TARGET_SFREQ, self.epoch_seconds,
            _PAPER_NATIVE_EPOCH_SECONDS, self.window_matches_paper, unit,
        )
        self._banner_logged = True

    def _prepare_input(self, batch) -> torch.Tensor:
        eeg = self.require_signal(batch, "eeg").to(dtype=torch.float32)
        if eeg.ndim != 3:
            raise ValueError(
                f"SeizureTransformer: expected (B, C, T); got shape {tuple(eeg.shape)}"
            )

        meta = batch.get("meta") or [{}]
        if not meta:
            raise ValueError("SeizureTransformer: batch meta missing.")
        assert_batch_homogeneity(
            meta,
            keys=("sampling_rate", "unit", "channels"),
            where="seizure_transformer:input",
        )

        # Sampling rate (canonical key).
        meta_sfreq = read_sampling_rate(meta[0])
        if meta_sfreq is None:
            raise ValueError(
                "SeizureTransformer: meta[0] missing 'sampling_rate' "
                "(canonical key per MODEL_CONTRACTS §3)."
            )

        # Unit handling — declared by adapter, fail-loud if missing.
        declared = declared_units_from_batch(batch) or []
        unit = meta[0].get("unit")
        if unit is None and declared:
            unit = declared[0]
        if unit is None:
            raise ValueError(
                "SeizureTransformer: meta[0]['unit'] missing. Adapters must "
                "publish the input tensor unit per MODEL_CONTRACTS §0."
            )
        eeg = unit_to_uv(eeg, unit)
        self._last_unit_declared = str(unit)

        # Channel handling — validate label set + reorder to the pretrained
        # slot order. Slot identity is critical (no channel embedding; first
        # Conv1d weights are slot-positional). Unlike FM wrappers, we
        # deliberately do NOT call strip_zero_channels: zero-padded missing
        # channels (e.g. TUSZ FZ/PZ in some sessions) must stay in their
        # pretrained slots — dropping them would re-permute the surviving 17
        # into the wrong filter rows. Zero in a slot just contributes nothing
        # to that filter row, which is the safest fallback for missing
        # electrodes; it's reflected in `dropped_channels` metadata as
        # `zero_padded_in_batch`.
        adapter_channels = meta[0].get("channels")
        if not adapter_channels:
            raise ValueError(
                "SeizureTransformer: meta[0]['channels'] missing. The "
                "pretrained Conv1d has slot-positional weights — adapter "
                "must publish the channel-name list."
            )
        slot_map, _ = _build_channel_perm(list(adapter_channels))
        # Materialise (B, 19, T): copy adapter columns into their assigned
        # pretrained slots; zero-fill missing slots (e.g. CHB-MIT bipolar
        # has no CZ anchor — slot index 9 stays zero). This preserves
        # slot-positional Conv1d semantics; a missing slot just contributes
        # nothing to its filter row.
        B_, _, T_ = eeg.shape
        out = eeg.new_zeros((B_, _EXPECTED_CHANNELS, T_))
        for slot_i, src_i in enumerate(slot_map):
            if src_i is not None:
                out[:, slot_i, :] = eeg[:, src_i, :]
        eeg = out
        missing_slots = [i for i, src in enumerate(slot_map) if src is None]
        # Detect which (now-pretrained-slot-indexed) channels are all-zero
        # across the whole batch — surface them as zero-padded so reports
        # can flag recordings where part of the montage was absent. This
        # captures both structurally-missing slots (slot_map=None, e.g.
        # CHB-MIT CZ) and adapter-supplied zero rows.
        with torch.no_grad():
            zero_mask = (eeg.abs().sum(dim=(0, 2)) == 0)
        zero_padded = [
            {"label": _PRETRAIN_CHANNEL_ORDER[i], "reason": "zero_padded_in_batch"}
            for i, z in enumerate(zero_mask.tolist()) if z
        ]
        self._last_input_channels = list(_PRETRAIN_CHANNEL_ORDER)
        self._last_dropped_channels = zero_padded
        self._last_structurally_missing = [
            _PRETRAIN_CHANNEL_ORDER[i] for i in missing_slots
        ]

        B, C, T = eeg.shape
        if C != _EXPECTED_CHANNELS:
            raise ValueError(
                f"SeizureTransformer: post-reorder C={C} ≠ {_EXPECTED_CHANNELS}; "
                "adapter must provide unipolar 10-20 19-channel montage."
            )
        expected_T = int(round(float(meta_sfreq) * self.epoch_seconds))
        if abs(T - expected_T) > 1:
            raise ValueError(
                f"SeizureTransformer: duration mismatch — "
                f"meta['sampling_rate']={meta_sfreq} Hz and "
                f"epoch_seconds={self.epoch_seconds:g} s imply {expected_T} samples, "
                f"but got T={T}."
            )

        x, _ = resample_poly_with_fallback(eeg, float(meta_sfreq), _TARGET_SFREQ)
        if x.shape[-1] != self.target_len:
            raise ValueError(
                f"SeizureTransformer: resampled length {x.shape[-1]} != target "
                f"{self.target_len} ({self.epoch_seconds:g} s × 256 Hz). "
                f"Input was T={T} at fs={meta_sfreq:.3f} Hz."
            )

        # Per-channel z-score across time (matches authors' inference code).
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (x - mu) / sd

        assert_finite(x, where="seizure_transformer:after_norm")
        self._log_banner_once(float(meta_sfreq), str(unit))
        return x.to(self.device, non_blocking=True)

    # --- forward passes --------------------------------------------------

    def _encoder_trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Run through encoder + res-CNN + transformer; return (B, 512, T_enc)."""
        enc_out, _skips = self.model.encoder(x)
        res_x = self.model.res_cnn_stack(enc_out)
        y = res_x.permute(2, 0, 1)
        y = self.model.position_encoding(y)
        y = self.model.transformer_encoder(y)
        y = y.permute(1, 2, 0) + res_x
        return y  # (B, 512, T_enc)

    def native_head_logits(self, batch) -> np.ndarray:
        """Return per-window seizure probability summary from the native head.

        Output shape: ``(B, 2)`` — ``[prob_non_seizure, prob_seizure]`` where the
        seizure score is the mean of the model's per-sample sigmoid output over
        the 60 s window. Two-column format keeps the linear-probe evaluator
        happy downstream.
        """
        x = self._prepare_input(batch)
        with torch.inference_mode():
            # Model returns sigmoid probabilities, shape (B, 15360).
            per_sample = self.model(x)
        sz = per_sample.mean(dim=-1).clamp(1e-6, 1 - 1e-6)
        probs = torch.stack([1.0 - sz, sz], dim=-1)
        return probs.detach().cpu().numpy()

    def extract_embeddings(self, batch) -> np.ndarray:
        """Global-average pool the transformer-encoder output to (B, 512).

        Category C tap: no paper-supported frozen pooling exists; the paper's
        downstream is the supervised head. Reported numbers are therefore not
        directly comparable to the paper's metrics or to other wrappers'
        Category-A pooling.
        """
        x = self._prepare_input(batch)
        with torch.inference_mode():
            trunk = self._encoder_trunk(x)  # (B, 512, T_enc)
            feats = trunk.mean(dim=-1)
        assert_finite(feats, where="seizure_transformer:embed_out")
        return feats.detach().cpu().numpy()

    def metadata(self) -> dict:
        notes: list[str] = []
        if not self.window_matches_paper:
            notes.append(
                f"epoch_seconds={self.epoch_seconds:g}s deviates from paper_native="
                f"{_PAPER_NATIVE_EPOCH_SECONDS:g}s; trunk has shorter transformer "
                "context — Category-C results are not directly comparable to the "
                "paper's reported numbers and inherit an additional 'shorter "
                "context' caveat over the standard Category-C asterisk."
            )
        return {
            **super().metadata(),
            "device": self.device,
            "target_sfreq": self.target_sfreq,
            "epoch_seconds": self.epoch_seconds,
            "target_len": self.target_len,
            "paper_native_epoch_seconds": _PAPER_NATIVE_EPOCH_SECONDS,
            "window_matches_paper": self.window_matches_paper,
            "alignment_category": "C",
            "alignment_category_reason": (
                "supervised end-to-end model; no paper-supported frozen pooling — "
                "embedding is wrapper-defined GAP of transformer trunk"
            ),
            "extraction_recipe": (
                "encoder→res_cnn_stack→pos_enc→transformer_encoder(+residual)→GAP_over_time"
            ),
            "feature_tap": "transformer_encoder_residual_post",
            "channel_mode": "strict_19_unipolar_pretrain_order",
            "input_channels_used": list(self._last_input_channels),
            "dropped_channels": list(self._last_dropped_channels),
            "unit_declared": self._last_unit_declared,
            "reference_applied": False,
            "wrapper_contract": "model_specific_transforms_only",
            "notes": notes,
        }
