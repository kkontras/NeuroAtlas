"""DeepSOZ-HEM backbone wrapper (Shama et al. MICCAI 2023; SzCORE 2025 #4).

Architecture: per-second per-channel transformer encoder + global-token LSTM
+ MIL-pooled SOZ head. Input contract is `(B, T, C=19, L=256)` where T is
the window length in seconds at the model's strict 256 Hz.

Sampling rate is architecturally hard at 256 Hz: the per-second token IS the
raw 256 samples directly (the transformer's `d_model=256` equals
samples-per-second), with no projection layer. A different fs would not
match the transformer input dimension.

Window length T is pretrained on 600 s but is NOT architecturally fixed:
both the channel transformer (over `(B*T, C+1, 256)`) and the LSTM (rolls
over T tokens) accept any T. We allow short windows (e.g. 10 s) so deepsoz
can join the project's 10 s sweep alongside the other FMs. Probe scores at
T<<600 s are not paper-comparable but ARE the right comparison against
every other 10 s FM in the sweep.

Window contract:
    Any T ≥ 1 s at fs=256 Hz. Wrapper logs a banner whenever T ≠ 600 s so
    logs surface the divergence from pretrain context.

Channel contract:
    Pretrained slot order (upstream main.py:10):
        FP1 FP2 F7 F3 FZ F4 F8 T3 C3 CZ C4 T4 T5 P3 PZ P4 T6 O1 O2.
    Slot identity matters — the channel transformer learns per-slot positional
    embeddings (`pos_encoder = nn.Embedding(20, 256)`). Adapter must publish
    19 unipolar 10-20 channels under any naming; we normalise + reorder.

Reference:
    The pretrained checkpoint expects average-referenced data.

Channel contract — CHB-MIT bipolar exception: the 18 double-banana pairs
emitted by CHB-MIT-BIDS are mapped to their anatomically-frontal/midline
anchor (``_BIPOLAR_TO_PRETRAIN_SLOT``) before slot resolution. Same
collision-free scheme as SeizureTransformer/EEGPT. Bipolar
values are reused as-is at the anchor's pretrained slot — pseudo-unipolar
evaluation per ``project_chbmit_pseudo.md``. CZ ends up unfilled
(1/19 zero-padded slot). Single-channel datasets (Bonn, FZ only) fill
1 of 19 slots and zero-pad the other 18.

Filter scope:
    Upstream applies a 1.6 Hz Butterworth highpass + 30 Hz lowpass + ±2σ
    clip inside `Preprocess.preprocess`. Per project convention
    (`feedback_filter_scope`), bandpass / notch belong to the dataset
    preprocessor. The wrapper does NOT re-filter; the dataset config must
    supply signals already band-limited to ~1-30 Hz. The amplitude clip is
    treated as a model-specific transform and is applied here.
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
from .third_party.deepsoz_hem.baselines import txlstm_szpool
from benchmarking_helpers import CheckpointSpec

logger = logging.getLogger(__name__)


_TARGET_SFREQ = 256.0
_PAPER_EPOCH_SECONDS = 600.0  # documentation only; window length is runtime-driven
_SAMPLES_PER_SEC = int(_TARGET_SFREQ)                    # 256
_CLIP_STD = 2.0
_EXPECTED_CHANNELS = 19
_EMBED_DIM = 256  # transformer hidden size in upstream baselines.py:11

# Upstream channel order (main.py:10 `req_chns`). Slot-positional pos_encoder
# in baselines.py:12 — order matters.
_PRETRAIN_CHANNEL_ORDER: Tuple[str, ...] = (
    "FP1", "FP2", "F7",  "F3",
    "FZ",  "F4",  "F8",  "T3",
    "C3",  "CZ",  "C4",  "T4",
    "T5",  "P3",  "PZ",  "P4",
    "T6",  "O1",  "O2",
)
_PRETRAIN_CHANNEL_SET = frozenset(_PRETRAIN_CHANNEL_ORDER)

# CHB-MIT-BIDS double-banana 18-pair → unipolar anchor slot. Anchor is the
# more frontal/midline electrode of each pair (collision-free; same scheme
# as SeizureTransformer/EEGPT). 18 of 19 pretrain slots are filled;
# CZ is intentionally unfilled and remains zero-padded. Pseudo-unipolar
# evaluation per `project_chbmit_pseudo.md` — bipolar reference is preserved
# (no LS reconstruction); results inherit the asterisk.
_BIPOLAR_TO_PRETRAIN_SLOT: Dict[str, str] = {
    "FP1-F7": "F7",  "F7-T3": "T3",  "T3-T5": "T5",  "T5-O1": "O1",
    "FP2-F8": "F8",  "F8-T4": "T4",  "T4-T6": "T6",  "T6-O2": "O2",
    "FP1-F3": "FP1", "F3-C3": "F3",  "C3-P3": "C3",  "P3-O1": "P3",
    "FP2-F4": "FP2", "F4-C4": "F4",  "C4-P4": "C4",  "P4-O2": "P4",
    "FZ-CZ": "FZ",   "CZ-PZ": "PZ",
}
# Module-load-time guard: collision means two pairs route to the same slot,
# which silently corrupts the slot-positional channel-transformer tokens.
assert len(set(_BIPOLAR_TO_PRETRAIN_SLOT.values())) == len(
    _BIPOLAR_TO_PRETRAIN_SLOT
), "_BIPOLAR_TO_PRETRAIN_SLOT has duplicate target slots — fix the table."


def _normalize_channel_label(label: str) -> str:
    """Canonical form: uppercase, drop 'EEG ' prefix and '-REF'/'-LE'/'-AVG' suffix.

    CHB-MIT-BIDS bipolar pairs (e.g. 'FP1-F7') are routed via
    ``_BIPOLAR_TO_PRETRAIN_SLOT`` to their anchor slot before suffix
    stripping. Asterisked per ``project_chbmit_pseudo.md``.
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
    slot.

    Behaviour:
    - If every pretrain slot is present in the adapter (typical unipolar
      19-ch case), the result is a strict permutation (no None).
    - If some slots are missing AND the adapter delivered fewer than
      ``_EXPECTED_CHANNELS`` channels (CHB-MIT bipolar 18-pair → 18
      anchors, CZ unfilled; Bonn 1-pair → FZ filled, 18 unfilled),
      missing slots map to ``None`` and the caller materialises a
      zero-filled channel for them. Pseudo-unipolar evaluation per
      ``project_chbmit_pseudo.md``; the unfilled slots are reported via
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
            f"DeepSOZ-HEM: unexpected channel labels {extra} from adapter "
            f"(normalized {norm}). The pretrained channel-transformer uses "
            f"slot-positional pos_encoder embeddings — only labels in "
            f"{list(_PRETRAIN_CHANNEL_ORDER)} (or bipolar pairs in "
            f"_BIPOLAR_TO_PRETRAIN_SLOT) are accepted."
        )
    if len(norm) > _EXPECTED_CHANNELS:
        raise ValueError(
            f"DeepSOZ-HEM: got {len(norm)} channels from adapter, more than "
            f"the {_EXPECTED_CHANNELS} pretrained slots."
        )
    return [name_to_idx.get(n) for n in _PRETRAIN_CHANNEL_ORDER], norm


def _load_checkpoint(path: Path, device: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"DeepSOZ-HEM checkpoint not found at {path}. The vendored "
            "third_party/deepsoz_hem/deepsoz_fold4.pth_4.tar should ship with "
            "the repo; restore it from upstream amruth-sn/deepsoz-hem if missing."
        )
    return torch.load(str(path), map_location=device, weights_only=True)


class DeepSOZHEMBackbone(BenchmarkBackbone):
    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.epoch_seconds = float(spec.expected_epoch_seconds)
        # Window length T is pretrained at 600 s but architecturally flexible
        # (the channel transformer is per-second, and extract_embeddings
        # bypasses the LSTM). We allow shorter T so deepsoz can join the
        # project's 10 s sweep; surface the divergence in logs.
        self.t_seconds = int(round(self.epoch_seconds))
        if self.t_seconds < 1 or abs(self.t_seconds - self.epoch_seconds) > 0.01:
            raise ValueError(
                f"DeepSOZ-HEM: epoch_seconds={self.epoch_seconds:g} must be a "
                f"positive integer (per-second tokens); got rounded T={self.t_seconds}."
            )
        if self.t_seconds != int(_PAPER_EPOCH_SECONDS):
            logger.info(
                "DeepSOZ-HEM: running at T=%d s (pretrain context was %d s) — "
                "channel-transformer + mean-over-time still produce a 256-d "
                "feature; native LSTM head's per-second logits are untested at "
                "this length.",
                self.t_seconds, int(_PAPER_EPOCH_SECONDS),
            )
        self.target_sfreq = _TARGET_SFREQ
        self.target_len = int(round(_TARGET_SFREQ * self.epoch_seconds))

        # Detector inside `txlstm_szpool` is built with `pretrained=None` —
        # we'll load the full state dict (detector+head) below.
        self.model = txlstm_szpool(
            transformer_dropout=0.25, device=self.device,
            return_attn=False, pretrained=None,
            modelname="txlstm", pooltype="szpool",
        )
        ckpt_path = Path(spec.checkpoint_path or "")
        state_dict = _load_checkpoint(ckpt_path, device="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)

        self._banner_logged = False
        self._last_unit_declared: str | None = None
        self._last_dropped_channels: list[dict] = []
        self._last_input_channels: list[str] = list(_PRETRAIN_CHANNEL_ORDER)
        self._last_structurally_missing: list[str] = []

        try:
            import setproctitle
            setproctitle.setproctitle(f"deepsoz-{spec.identifier[-3:]}")
        except Exception:
            pass

        logger.info(
            "DeepSOZ-HEM loaded (ckpt=%s, in=(%d,%d,%d), epoch_s=%g, device=%s)",
            ckpt_path.name, self.t_seconds, _EXPECTED_CHANNELS, _SAMPLES_PER_SEC,
            self.epoch_seconds, self.device,
        )

    # --- input pipeline --------------------------------------------------

    def _log_banner_once(self, fs_in: float, unit: str) -> None:
        if self._banner_logged:
            return
        logger.info(
            "[backbone=deepsoz_hem] fs_in=%.3f→%g Hz window=%g s "
            "channels=19 unipolar(pretrain_order) ref=average "
            "scale=clip2sigma weight_source=upstream_pretrained_fold4 unit=%s",
            fs_in, _TARGET_SFREQ, self.epoch_seconds, unit,
        )
        self._banner_logged = True

    def _prepare_input(self, batch) -> torch.Tensor:
        eeg = self.require_signal(batch, "eeg").to(dtype=torch.float32)
        if eeg.ndim != 3:
            raise ValueError(
                f"DeepSOZ-HEM: expected (B, C, T); got shape {tuple(eeg.shape)}"
            )

        meta = batch.get("meta") or [{}]
        if not meta:
            raise ValueError("DeepSOZ-HEM: batch meta missing.")
        assert_batch_homogeneity(
            meta,
            keys=("sampling_rate", "unit", "channels"),
            where="deepsoz_hem:input",
        )

        meta_sfreq = read_sampling_rate(meta[0])
        if meta_sfreq is None:
            raise ValueError(
                "DeepSOZ-HEM: meta[0] missing 'sampling_rate' (canonical key)."
            )

        declared = declared_units_from_batch(batch) or []
        unit = meta[0].get("unit") or (declared[0] if declared else None)
        if unit is None:
            raise ValueError(
                "DeepSOZ-HEM: meta[0]['unit'] missing; adapters must publish "
                "the input unit per MODEL_CONTRACTS §0."
            )
        eeg = unit_to_uv(eeg, unit)
        self._last_unit_declared = str(unit)

        adapter_channels = meta[0].get("channels")
        if not adapter_channels:
            raise ValueError(
                "DeepSOZ-HEM: meta[0]['channels'] missing; pretrained "
                "channel-transformer requires the adapter channel-name list."
            )
        slot_map, _ = _build_channel_perm(list(adapter_channels))
        # Materialise (B, 19, T): copy adapter columns into their assigned
        # pretrained slots; zero-fill missing slots (CHB-MIT bipolar has no
        # CZ anchor — slot index 9 stays zero; Bonn fills only FZ at slot
        # index 4 and zero-pads 18). Slot identity matters because the
        # channel transformer's pos_encoder is per-slot positional.
        B_, _, T_ = eeg.shape
        out = eeg.new_zeros((B_, _EXPECTED_CHANNELS, T_))
        for slot_i, src_i in enumerate(slot_map):
            if src_i is not None:
                out[:, slot_i, :] = eeg[:, src_i, :]
        eeg = out
        missing_slots = [i for i, src in enumerate(slot_map) if src is None]
        # Detect which (now-pretrained-slot-indexed) channels are all-zero
        # across the whole batch; surface as zero-padded so reports flag
        # recordings where part of the montage was absent. Captures both
        # structurally-missing slots (slot_map=None) and adapter-supplied
        # zero rows.
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
                f"DeepSOZ-HEM: post-reorder C={C} ≠ {_EXPECTED_CHANNELS}."
            )
        expected_T = int(round(float(meta_sfreq) * self.epoch_seconds))
        if abs(T - expected_T) > 1:
            raise ValueError(
                f"DeepSOZ-HEM: duration mismatch — "
                f"meta sampling_rate={meta_sfreq} Hz × epoch_seconds="
                f"{self.epoch_seconds:g} s = {expected_T} samples; got T={T}."
            )

        x, _ = resample_poly_with_fallback(eeg, float(meta_sfreq), _TARGET_SFREQ)
        if x.shape[-1] != self.target_len:
            raise ValueError(
                f"DeepSOZ-HEM: resampled length {x.shape[-1]} != target "
                f"{self.target_len} (600 s × 256 Hz). Input was T={T} at "
                f"fs={meta_sfreq:.3f} Hz."
            )

        # Per-channel ±2σ clip (upstream Preprocess.clip).
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = torch.clamp(x, min=mu - _CLIP_STD * sd, max=mu + _CLIP_STD * sd)

        # Reshape (B, 19, T*256) → (B, T, 19, 256). Upstream main.py:177
        # does: data.reshape(C, T_sec, fs).transpose(1, 0, 2) which yields
        # (T_sec, C, fs); here we keep the batch dim and produce the model
        # contract (B, T_sec, C, fs). T is runtime-driven from epoch_seconds.
        x = x.reshape(B, _EXPECTED_CHANNELS, self.t_seconds, _SAMPLES_PER_SEC)
        x = x.permute(0, 2, 1, 3).contiguous()

        assert_finite(x, where="deepsoz_hem:after_clip")
        self._log_banner_once(float(meta_sfreq), str(unit))
        return x.to(self.device, non_blocking=True)

    # --- forward passes --------------------------------------------------

    def native_head_logits(self, batch) -> np.ndarray:
        """Per-window seizure probability summary, shape (B, 2).

        The native head emits per-second softmax logits (B, T, 2). For the
        benchmarking contract we collapse to per-window (B, 2) by taking the
        max over time of the positive-class softmax, mirroring upstream
        `vis()` post-smoothing thresholding (which also operates on per-second
        positive probabilities).

        Note: at T ≠ 600 s the LSTM and pool head are run on a sequence
        length they were not pretrained for. The per-second logits are still
        well-defined (LSTM accepts variable T) but their calibration vs. the
        upstream paper is unverified at short windows.
        """
        x = self._prepare_input(batch)
        with torch.inference_mode():
            proba, _p_soz, _z, _sat = self.model(x)
            # proba: (B, T, 2) raw logits; softmax → per-second probs.
            per_sec = torch.softmax(proba, dim=-1)
            sz = per_sec[..., 1].amax(dim=-1).clamp(1e-6, 1 - 1e-6)
        probs = torch.stack([1.0 - sz, sz], dim=-1)
        return probs.detach().cpu().numpy()

    def extract_embeddings(self, batch) -> np.ndarray:
        """Mean-over-time of the LSTM hidden state, (B, 256).

        The native txlstm_szpool model exposes the per-second 2-class logits
        as `proba` and per-channel SOZ probs as `p_soz`. There is no
        author-supported frozen-feature pooling tap; we collapse the
        transformer trunk output by mean over the time axis to surface a
        Category-C embedding for the linear-probe path.
        """
        x = self._prepare_input(batch)
        with torch.inference_mode():
            # Reproduce the trunk by hand to avoid recomputing inside model:
            B, T, C, L = x.size()
            chn_pos = torch.arange(_EXPECTED_CHANNELS, device=x.device)
            pos_emb = self.model.detector.pos_encoder(chn_pos)[None, None, :, :]
            h_c = x + pos_emb
            h_m = self.model.detector.pos_encoder(
                torch.tensor([_EXPECTED_CHANNELS] * B * T, device=x.device).view(B, T, -1)
            )
            tx_in = torch.cat(
                (h_c.reshape(B * T, C, _EMBED_DIM), h_m.reshape(B * T, 1, _EMBED_DIM)),
                dim=1,
            )
            tx_out = self.model.detector.tx_encoder(tx_in)
            global_tok = tx_out[:, -1, :].view(B, T, _EMBED_DIM)
            feats = global_tok.mean(dim=1)  # (B, 256)
        assert_finite(feats, where="deepsoz_hem:embed_out")
        return feats.detach().cpu().numpy()

    def metadata(self) -> Dict[str, object]:
        return {
            **super().metadata(),
            "device": self.device,
            "target_sfreq": self.target_sfreq,
            "epoch_seconds": self.epoch_seconds,
            "target_len": self.target_len,
            "alignment_category": "C",
            "alignment_category_reason": (
                "supervised end-to-end model; embedding tap is wrapper-defined "
                "mean-over-time of the transformer global-token output"
            ),
            "extraction_recipe": (
                "(B,19,153600)→(B,600,19,256)→tx_encoder[global_token]→mean_over_time"
            ),
            "feature_tap": "tx_encoder_global_token_mean",
            "channel_mode": "19_unipolar_pretrain_order_with_zero_pad",
            "input_channels_used": list(self._last_input_channels),
            "dropped_channels": list(self._last_dropped_channels),
            "structurally_missing": list(self._last_structurally_missing),
            "unit_declared": self._last_unit_declared,
            "reference_required": "average",
            "wrapper_contract": "model_specific_transforms_only",
            "notes": [
                "DeepSOZ-HEM (Shama et al. MICCAI 2023; SzCORE 2025 #4 hard-example-mining variant). "
                "Vendored from amruth-sn/deepsoz-hem (GPL-3.0). Pretrained on TUSZ "
                "(171 patients, 10-fold) — fold 4 weights. "
                "CHB-MIT 18-pair bipolar routes through _BIPOLAR_TO_PRETRAIN_SLOT (CZ "
                "zero-padded); Bonn 1-channel fills FZ slot, 18 zero-padded. "
                "Both routes asterisked per project_chbmit_pseudo.md.",
            ],
        }
