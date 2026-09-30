"""NeuroLM backbone wrapper.

Window contract:
    NeuroLM is dynamic (paper §3 / MODEL_CONTRACTS §3 L555–L573): each
    token is one 200-sample patch at 200 Hz (= 1 s); the encoder accepts
    up to ``block_size = 1024`` total tokens across channels × time
    patches. The wrapper accepts any input whose post-resample length
    at 200 Hz is a positive multiple of 200 samples and raises if the
    token count exceeds ``block_size``. ``expected_epoch_seconds`` from
    the spec is informational only — the wrapper supports variable
    window lengths per batch.

Model-specific transforms applied in ``_prepare_tokens``
(per MODEL_CONTRACTS.md §0/§2):
    1. Unit conversion to µV via ``unit_to_uv`` (handles V/mV/µV via
       ``meta[i]["unit"]``; missing key → "uV" with a one-shot
       ``DeprecationWarning`` per §0 L18-L19).
    2. Strip all-zero channels (per §1; dataio zero-pads missing
       channels for batch-uniform shape).
    3. Resample to 200 Hz via polyphase (shared helper).
    4. Amplitude scale ``x / 100`` (paper §3: *"divided by 100 for
       normalization"*; verified in upstream ``dataset.py``). Gated by
       ``apply_amplitude_scale`` runtime override (default ``True``).
    5. Finite check on output.

Weight loading (§4): VQ encoder weights extracted from ``VQ.pt`` by
unwrapping the ``_orig_mod.VQ.encoder.*`` or ``VQ.encoder.*`` prefix.
``_strict_load_with_allowlist`` raises on any un-allowlisted state-dict
delta. ``_ALLOWED_MISSING`` and ``_ALLOWED_UNEXPECTED`` start empty;
benign deltas are added explicitly after verification.

Channel handling: NeuroLM uses a name-based ``standard_1020`` vocabulary
of ~130 entries that natively contains the 16 modern CHB-MIT bipolar
pairs (``FP1-F7``, ``F7-T7``, ...). ``_BIPOLAR_TO_STANDARD`` covers (i)
sleep-montage bipolars, (ii) the 2 midline CHB-MIT pairs not in the
native vocab, and (iii) Epilepsiae / Helsinki legacy 10-20 double-banana
(``F7-T3``, ``T3-T5``, ...) → modern 10-10 vocab pairs. Unknown labels
raise ``ValueError`` per the uniform §1 contract.

Feature extraction (§5): masked mean over valid tokens of
``forward_features(return_all_tokens=True)``. NeuroLM is **Category C**
(paper has no frozen linear-probe path; downstream is VQ encoder →
learned ``encode_transform_layer`` + GPT2 instruction-tuning) — surfaced
in ``metadata()['alignment_category']``.

Signal cleanliness (bandpass / notch) is the dataset preprocessor's
responsibility, not this wrapper's. See AGENT_GUIDE.md §"Preprocessing
responsibilities".
"""
from __future__ import annotations

import json
import logging
import warnings
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch

from ._preproc import (
    _ESAT_DATASETS,
    assert_batch_homogeneity,
    assert_finite,
    read_sampling_rate,
    replace_nonfinite_with_zero,
    resample_poly_with_fallback,
    snap_to_epoch_length,
    strip_zero_channels,
    unit_to_uv,
)
from .base import BenchmarkBackbone
from .neurolm_encoder import NTConfig, NeuralTransformer
from benchmarking_helpers import CheckpointSpec


logger = logging.getLogger(__name__)


_TARGET_SFREQ = 200
_PATCH_SIZE = 200
_SCALE_DIVISOR = 100.0


_STANDARD_1020 = [
    "FP1", "FPZ", "FP2", "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    "O1", "OZ", "O2", "O9", "CB1", "CB2", "IZ", "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2",
    "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8",
    "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8",
    "T1", "T2", "FTT9h", "TTP7h", "TPP9h", "FTT10h", "TPP8h", "TPP10h",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3",
    "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "pad", "I1", "I2",
]

_PAD_IDX = _STANDARD_1020.index("pad")

# Bipolar / legacy aliases. Maps source labels to *vocab* targets that exist
# in _STANDARD_1020. Any 'A-B' label not handled here raises in
# _channel_idx_or_raise — silent splitting produces collision artifacts on
# 10-20 montages (CHB-MIT modern pairs are tokenised natively, see comment
# below).
_BIPOLAR_TO_STANDARD: Dict[str, str] = {
    # Sleep-montage bipolars → monopolar anchor (matches EEGPT/REVE alias map).
    "C3-A2": "C3", "C4-A1": "C4",
    "O1-A2": "O1", "O2-A1": "O2",
    "F3-A2": "F3", "F4-A1": "F4",

    # NeuroLM's _STANDARD_1020 vocabulary already contains the 16 modern
    # CHB-MIT temporal/parasagittal double-banana pairs as native tokens,
    # so they tokenise directly without aliasing. Only the 2 midline pairs
    # (FZ-CZ, CZ-PZ) need aliasing; they land on FZ / PZ unipolar tokens
    # (collision-free across the 18-pair montage). Signal values remain
    # bipolar differences for these 2 aliased pairs; CHB-MIT NeuroLM
    # results carry a documented asterisk.
    "FZ-CZ": "FZ", "CZ-PZ": "PZ",

    # Epilepsiae / Helsinki Neonatal legacy 10-20 double-banana → modern
    # 10-10 vocab pairs (T3=T7, T4=T8, T5=P7, T6=P8). Same parasagittal
    # left/right structure as CHB-MIT, so the targets are native bipolar
    # vocab tokens (no monopolar anchoring needed).
    "F7-T3": "F7-T7", "T3-T5": "T7-P7", "T5-O1": "P7-O1",
    "F8-T4": "F8-T8", "T4-T6": "T8-P8", "T6-O2": "P8-O2",
    # Central temporal chain (no native pair tokens in NeuroLM vocab).
    # Anchor to 10-10 between-position monopolars (matches EEGPT scheme),
    # collision-free across the 18-pair Epilepsiae montage.
    "T3-C3": "FC3", "C3-CZ": "FCZ", "CZ-C4": "FC4", "C4-T4": "CP4",
}


_NON_EEG_PREFIXES: Tuple[str, ...] = (
    "EOG", "EMG", "ECG", "EKG", "RESP", "TEMP", "EVENT",
)


# Weight-load allowlist: starts empty per MODEL_CONTRACTS §4. The first run
# against a new VQ.pt will raise with the observed deltas; copy the verified-
# benign keys here after manual review.
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


def _channel_idx_or_raise(name: str) -> int:
    """Resolve a label to a NeuroLM ``_STANDARD_1020`` index, or raise.

    Order:
        1. Uppercase + strip.
        2. Exact match in ``_STANDARD_1020`` (covers monopolar tokens and
           the 16 modern CHB-MIT bipolar pairs that are native vocab
           entries).
        3. Exact match in ``_BIPOLAR_TO_STANDARD`` (sleep / midline /
           legacy bipolar aliases).
        4. Otherwise raise — generic ``A-B`` splitting is forbidden because
           it produces silent duplicate-target collisions on 10-20 montages.
    """
    key = name.upper().strip()
    if key in _STANDARD_1020:
        return _STANDARD_1020.index(key), "native"
    if key in _BIPOLAR_TO_STANDARD:
        target = _BIPOLAR_TO_STANDARD[key]
        # Classify the alias for traceability:
        #   * "renamed_pair"      — both source and target are bipolar pairs
        #     (e.g. legacy F7-T3 → modern F7-T7). Position embedding is
        #     still pair-shaped; only the electrode names changed.
        #   * "monopolar_anchor"  — bipolar source aliased to a single
        #     electrode (sleep C3-A2 → C3, central T3-C3 → FC3). Position
        #     embedding was trained on a referential signal at that anchor;
        #     the actual signal is a bipolar difference. Anatomical
        #     approximation per MODEL_CONTRACTS.md §1 L353–L364.
        kind = "renamed_pair" if "-" in target else "monopolar_anchor"
        return _STANDARD_1020.index(target), kind
    raise ValueError(
        f"NeuroLM: channel label {name!r} not in vocabulary. Add an "
        f"explicit entry to _BIPOLAR_TO_STANDARD or rename the adapter's "
        f"channels to a vocab member."
    )


def _select_eeg_channels(
    labels: List[str], n_channels: int
) -> Tuple[List[int], List[str], List[Dict[str, str]]]:
    """Decide which channel indices to keep (drop EOG/EMG/ECG/...).

    Returns ``(kept_indices, kept_labels, dropped_entries)`` where
    ``dropped_entries`` is a list of ``{"label", "reason"}`` dicts per
    MODEL_CONTRACTS.md §1 L195–L212.
    """
    if not labels:
        raise ValueError(
            "NeuroLM: meta[0]['channels'] missing — adapter must publish "
            "per-channel labels."
        )
    if len(labels) != n_channels:
        raise ValueError(
            f"NeuroLM: channel metadata length mismatch — got {len(labels)} "
            f"labels for {n_channels} tensor channels. Labels: {labels!r}."
        )
    kept_idx: List[int] = []
    kept_labels: List[str] = []
    dropped_entries: List[Dict[str, str]] = []
    for i, label in enumerate(labels):
        if _is_non_eeg(label):
            dropped_entries.append({"label": str(label), "reason": "non_eeg_modality"})
        else:
            kept_idx.append(i)
            kept_labels.append(label)
    if not kept_idx:
        raise ValueError(
            f"NeuroLM: all {n_channels} channels classified as non-EEG "
            f"(labels={labels!r}); nothing to embed."
        )
    return kept_idx, kept_labels, dropped_entries


def describe_channel_mapping(channels):
    """Name-only channel provenance (see channel_provenance.ChannelMapping).

    Reuses ``_select_eeg_channels`` + ``_channel_idx_or_raise`` (the exact
    inference path). NeuroLM is token-based (variable layout) — no fixed
    slots, hence no zero-insertion: each kept EEG channel becomes a vocab
    token; non-EEG channels are dropped; unknown labels raise (captured
    upstream).
    """
    from benchmarking_helpers.channels.channel_provenance import (
        ChannelMapping,
        ChannelSlot,
    )

    kept_idx, kept_labels, dropped_entries = _select_eeg_channels(
        list(channels), len(channels)
    )
    slots = []
    for label in kept_labels:
        idx, kind = _channel_idx_or_raise(label)
        target = _STANDARD_1020[idx]
        status = "approx" if kind == "monopolar_anchor" else "filled"
        slots.append(ChannelSlot(target=target, source=str(label), status=status, note=kind))
    dropped = [
        ChannelSlot(target=str(d["label"]), source=str(d["label"]), status="dropped", note=d["reason"])
        for d in dropped_entries
    ]
    return ChannelMapping(
        model_family="neurolm", layout_kind="variable", slots=slots, dropped=dropped
    )


def _extract_encoder_state(state):
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported NeuroLM checkpoint payload type {type(state)!r}.")
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    prefixes = ("_orig_mod.VQ.encoder.", "VQ.encoder.")
    extracted: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix):]] = value
                break
    if not extracted:
        raise KeyError("Could not find NeuroLM VQ encoder weights in the checkpoint payload.")
    return extracted


def _strict_load_with_allowlist(
    model: torch.nn.Module, state: Dict[str, torch.Tensor]
) -> Dict[str, List[str]]:
    """Load weights; raise unless the missing/unexpected deltas are allowlisted."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if k not in _ALLOWED_MISSING]
    real_unexpected = [k for k in unexpected if k not in _ALLOWED_UNEXPECTED]
    if real_missing or real_unexpected:
        raise RuntimeError(
            "NeuroLM weight load: unaccounted-for state_dict deltas.\n"
            f"  missing (not in allowlist): {real_missing[:20]}\n"
            f"  unexpected (not in allowlist): {real_unexpected[:20]}\n"
            "After verifying benign, add to _ALLOWED_MISSING / _ALLOWED_UNEXPECTED."
        )
    return {
        "missing_keys_allowed": list(missing),
        "unexpected_keys_allowed": list(unexpected),
    }


class NeuroLMBackbone(BenchmarkBackbone):
    """Wrapper around the NeuroLM VQ encoder for benchmark embedding extraction.

    Input contract:
        Tensor shape ``(B, C, T)`` with ``T`` a positive multiple of 200
        samples after resampling to 200 Hz, and ``C × (T // 200) ≤
        block_size = 1024`` total tokens.

    Output contract:
        ``[B, embed_dim]`` (typically 768) — masked mean over valid VQ
        encoder tokens via ``forward_features(return_all_tokens=True)``.
        Category C: paper has no frozen linear-probe path; this is a
        benchmark-side adaptation.
    """

    def __init__(self, spec: CheckpointSpec):
        super().__init__(spec)
        from ._checkpoint_download import ensure_checkpoint
        checkpoint_path = ensure_checkpoint(
            spec.checkpoint_path, spec.source_type, spec.source_reference
        )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        encoder_conf = NTConfig(**checkpoint["encoder_args"])
        self.block_size = int(encoder_conf.block_size)
        self.patch_size = int(encoder_conf.patch_size)
        if self.patch_size != _PATCH_SIZE:
            raise ValueError(
                f"NeuroLM: checkpoint patch_size={self.patch_size} disagrees "
                f"with module constant _PATCH_SIZE={_PATCH_SIZE}. Update the "
                f"wrapper if upstream changed the patch convention."
            )
        self.model = NeuralTransformer(encoder_conf)
        state = _extract_encoder_state(checkpoint)
        self._load_report = _strict_load_with_allowlist(self.model, state)
        self._load_report["checkpoint_path"] = str(checkpoint_path)
        self._load_report["encoder_args"] = checkpoint["encoder_args"]
        # Bounds on the channel + time embedding tables. pos_embed indices
        # come from _STANDARD_1020 lookups (capped at len(_STANDARD_1020));
        # time_embed indices come from arange(n_time_patches). Either OOB
        # is silent garbage at inference, so cache the table sizes and
        # validate every input.
        self.pos_vocab_size = int(self.model.pos_embed.num_embeddings)
        self.time_vocab_size = int(self.model.time_embed.num_embeddings)
        if self.pos_vocab_size < len(_STANDARD_1020):
            raise ValueError(
                f"NeuroLM: pos_embed has {self.pos_vocab_size} rows but "
                f"_STANDARD_1020 declares {len(_STANDARD_1020)} channel "
                f"tokens. Wrapper's vocab list cannot index past the "
                f"pretrained table."
            )
        self.model.eval()
        self.model.to(self.device)
        self._weight_source = (
            "random_init" if getattr(spec, "source_type", "") == "random_init"
            else "pretrained"
        )

        self._unit_warned = False
        self._scale_off_warned = False
        self._banner_logged = False
        self._last_channel_report: Dict[str, Any] = {
            "input_channels_used": [],
            "dropped_channels": [],
            "input_sfreq_observed": None,
            "input_resample_method": "unknown",
            "scale_divisor": _SCALE_DIVISOR,
            "input_unit": "unknown",
            "n_zero_channels_stripped": 0,
            "n_time_patches_observed": None,
            "n_tokens_used": None,
            "block_size": self.block_size,
        }

    def _resolve_overrides(self) -> Dict[str, bool]:
        ov = getattr(self.spec, "runtime_overrides", {}) or {}
        return {
            "apply_amplitude_scale": bool(ov.get("apply_amplitude_scale", True)),
        }

    def _read_unit(self, batch) -> str:
        meta = batch.get("meta") or []
        first = meta[0] if meta else {}
        unit = first.get("unit") if isinstance(first, dict) else None
        if unit is None or unit == "":
            if not self._unit_warned:
                warnings.warn(
                    "NeuroLM: batch meta[0]['unit'] is missing — defaulting "
                    "to 'uV' for backward compatibility (MODEL_CONTRACTS.md "
                    "§0 L18-L19). Adapters should publish the canonical "
                    "'unit' key.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                self._unit_warned = True
            return "uV"
        return str(unit)

    def _log_banner_once(
        self,
        n_target_ch: int,
        unit: str,
        n_time_patches: int,
        apply_amplitude_scale: bool,
    ) -> None:
        if self._banner_logged:
            return
        logger.info(
            "[backbone=neurolm] fs=%d Hz patch=%d (1 s) block_size=%d "
            "n_time_patches~%d unit=%s scale=/%g apply_amplitude_scale=%s "
            "ref=keep alignment_category=C n_target_ch=%d weight_source=%s",
            _TARGET_SFREQ, _PATCH_SIZE, self.block_size, n_time_patches,
            unit, _SCALE_DIVISOR, apply_amplitude_scale, n_target_ch,
            self._weight_source,
        )
        logger.info(
            "[neurolm-report] %s",
            json.dumps(self._last_channel_report, default=str),
        )
        self._banner_logged = True

    def _prepare_tokens(self, batch):
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
                f"NeuroLM expected signal of shape (B, C, T) or (B, T); "
                f"got shape {tuple(x.shape)}."
            )

        meta = batch.get("meta") or []
        assert_batch_homogeneity(meta, where="neurolm:input")
        first_meta = meta[0] if meta else {}

        # 1. Unit conversion to µV (per §0).
        unit = self._read_unit(batch)
        x = unit_to_uv(x, unit)

        # 2. Strip all-zero (zero-padded missing) channels (per §1).
        raw_labels = list(first_meta.get("channels") or [])
        n_zero_stripped = 0
        if raw_labels and len(raw_labels) == x.shape[1]:
            x_stripped, raw_labels, _kept_idx_zero = strip_zero_channels(x, raw_labels)
            n_zero_stripped = x.shape[1] - x_stripped.shape[1]
            x = x_stripped

        # 3. Drop non-EEG by label prefix (per §1).
        dataset = first_meta.get("dataset") if first_meta else None
        use_neuroatlas_channel_handling = first_meta.get(
            "neurolm_neuroatlas_channels", False,
        ) if first_meta else False
        if dataset == "bci" and not use_neuroatlas_channel_handling:
            keep_idx = list(range(x.shape[1]))
            kept_labels = list(raw_labels)
            dropped_entries = []
        else:
            keep_idx, kept_labels, dropped_entries = _select_eeg_channels(
                raw_labels, n_channels=x.shape[1]
            )
        if keep_idx != list(range(x.shape[1])):
            x = x[:, keep_idx, :]

        # 4. Resample to 200 Hz. Variable seconds — no fixed target_len:
        # the post-resample length only needs to be a positive multiple of
        # patch_size and to fit within block_size tokens once paired with
        # n_chans.
        current_len = x.shape[-1]
        meta_fs = read_sampling_rate(first_meta) if first_meta else None
        if meta_fs is not None and meta_fs > 0:
            src_sfreq_f = float(meta_fs)
        else:
            # Fallback: assume input is already at 200 Hz (resample helper
            # is identity for src==dst). Adapters should publish
            # 'sampling_rate' to disambiguate this.
            src_sfreq_f = float(_TARGET_SFREQ)

        _backend = "scipy" if dataset in _ESAT_DATASETS else "auto"
        x, resample_method = resample_poly_with_fallback(
            x, src_sfreq_f, float(_TARGET_SFREQ), backend=_backend
        )
        # ESAT-8: snap resampled length to epoch_seconds * target_sfreq
        _dataset = first_meta.get("dataset", "") if first_meta else ""
        if _dataset in _ESAT_DATASETS:
            x = snap_to_epoch_length(x, float(_TARGET_SFREQ), meta)
        post_T = x.shape[-1]
        if post_T < _PATCH_SIZE:
            raise ValueError(
                f"NeuroLM: post-resample length {post_T} samples is shorter "
                f"than one patch ({_PATCH_SIZE}). Need at least 1 s of "
                f"signal at {_TARGET_SFREQ} Hz."
            )
        if post_T % _PATCH_SIZE != 0:
            raise ValueError(
                f"NeuroLM: post-resample length {post_T} is not a positive "
                f"multiple of patch={_PATCH_SIZE} samples (= 1 s at "
                f"{_TARGET_SFREQ} Hz). Input length was {current_len} at "
                f"fs {src_sfreq_f:.3f} Hz; window must be a whole number "
                f"of seconds."
            )
        n_time_patches = post_T // _PATCH_SIZE

        # time_embed table is fixed-size (typically 64 rows). arange(N)
        # would silently index past the trained rows for windows > N
        # seconds. Fail loud.
        if n_time_patches > self.time_vocab_size:
            raise ValueError(
                f"NeuroLM: n_time_patches={n_time_patches} exceeds "
                f"time_embed table size={self.time_vocab_size}. Maximum "
                f"window length at {_TARGET_SFREQ} Hz is "
                f"{self.time_vocab_size} s. Shorten the window upstream."
            )

        # Token-cap check (block_size across channels × time patches).
        n_chans = x.shape[1]
        n_tokens = n_chans * n_time_patches
        if n_tokens > self.block_size:
            if dataset == "bci":
                logger.warning(
                    "NeuroLM: BCI token budget exceeded (%d > %d); truncating channels.",
                    n_tokens, self.block_size,
                )
            else:
                raise ValueError(
                    f"NeuroLM: token budget exceeded — n_chans={n_chans} × "
                    f"n_time_patches={n_time_patches} = {n_tokens} > "
                    f"block_size={self.block_size}. Shorten the window or "
                    f"reduce the channel count upstream."
                )

        # 5. Amplitude scale (gated by apply_amplitude_scale override).
        overrides = self._resolve_overrides()
        if overrides["apply_amplitude_scale"]:
            x = x / _SCALE_DIVISOR
        else:
            if not self._scale_off_warned:
                logger.warning(
                    "NeuroLM: apply_amplitude_scale=False — skipping ÷100 "
                    "scale. Embedding is OFF-DISTRIBUTION; diagnostic-only. "
                    "Will not re-warn this run."
                )
                self._scale_off_warned = True

        x = x.to(self.device)
        assert_finite(x, where="neurolm:after_norm")

        # Resolve channel labels to vocab indices.
        if dataset == "bci":
            ch_indices_with_pos: List[Tuple[int, int, str]] = []
            dropped_ch: List[str] = []
            for i, c in enumerate(kept_labels):
                key = c.upper().strip()
                if key in _STANDARD_1020:
                    ch_indices_with_pos.append((i, _STANDARD_1020.index(key), "native"))
                elif key in _BIPOLAR_TO_STANDARD:
                    target = _BIPOLAR_TO_STANDARD[key]
                    ch_indices_with_pos.append((i, _STANDARD_1020.index(target),
                                                "renamed_pair" if "-" in target else "monopolar_anchor"))
                else:
                    dropped_ch.append(c)
            if not ch_indices_with_pos:
                raise RuntimeError(
                    f"NeuroLM: 0 of {len(kept_labels)} channels matched the vocabulary. "
                    f"Dropped: {dropped_ch[:10]}..."
                )
            if dropped_ch:
                logger.info(
                    "[backbone=neurolm] BCI: dropped %d/%d channels not in vocab: %s",
                    len(dropped_ch), len(kept_labels), dropped_ch[:10],
                )
            keep_signal_idx = [pos for pos, _, _ in ch_indices_with_pos]
            ch_indices = [idx for _, idx, _ in ch_indices_with_pos]
            x = x[:, keep_signal_idx, :]
            kept_labels = [kept_labels[pos] for pos, _, _ in ch_indices_with_pos]
            n_chans = len(ch_indices)
            n_tokens = n_chans * n_time_patches
            channel_alias_approximations = [
                {"label": kept_labels[j], "vocab_idx": idx, "vocab_target": _STANDARD_1020[idx], "alias_kind": kind}
                for j, (_, idx, kind) in enumerate(ch_indices_with_pos)
                if kind != "native"
            ]
        else:
            resolved = [_channel_idx_or_raise(c) for c in kept_labels]
            ch_indices = [idx for idx, _ in resolved]
            channel_alias_approximations = [
                {
                    "label": label,
                    "vocab_idx": idx,
                    "vocab_target": _STANDARD_1020[idx],
                    "alias_kind": kind,
                }
                for label, (idx, kind) in zip(kept_labels, resolved)
                if kind != "native"
            ]

        # Pack tokens (channel-major: all time patches of ch0, then ch1, ...).
        batch_size = x.shape[0]
        tokens = torch.zeros(
            (batch_size, self.block_size, _PATCH_SIZE),
            device=self.device, dtype=torch.float32,
        )
        input_chans = torch.full(
            (batch_size, self.block_size), _PAD_IDX,
            device=self.device, dtype=torch.long,
        )
        input_times = torch.zeros(
            (batch_size, self.block_size),
            device=self.device, dtype=torch.long,
        )
        input_mask = torch.zeros(
            (batch_size, self.block_size),
            device=self.device, dtype=torch.bool,
        )
        time_idx = torch.arange(n_time_patches, device=self.device, dtype=torch.long)
        pos = 0
        for ch_i, ch_idx in enumerate(ch_indices):
            if pos + n_time_patches > self.block_size:
                break
            ch_signal = x[:, ch_i, :]
            patches = ch_signal.reshape(batch_size, n_time_patches, _PATCH_SIZE)
            tokens[:, pos:pos + n_time_patches] = patches
            input_chans[:, pos:pos + n_time_patches] = ch_idx
            input_times[:, pos:pos + n_time_patches] = time_idx
            input_mask[:, pos:pos + n_time_patches] = True
            pos += n_time_patches
        attn_mask = input_mask.unsqueeze(1).repeat(1, self.block_size, 1).unsqueeze(1)

        self._last_channel_report = {
            "input_channels_used": list(kept_labels),
            "dropped_channels": list(dropped_entries),
            "channel_alias_approximations": list(channel_alias_approximations),
            "input_sfreq_observed": float(src_sfreq_f),
            "input_resample_method": resample_method,
            "scale_divisor": _SCALE_DIVISOR,
            "input_unit": unit,
            "n_zero_channels_stripped": int(n_zero_stripped),
            "n_time_patches_observed": int(n_time_patches),
            "n_tokens_used": int(n_tokens),
            "block_size": int(self.block_size),
            "pos_vocab_size": int(self.pos_vocab_size),
            "time_vocab_size": int(self.time_vocab_size),
            "apply_amplitude_scale": overrides["apply_amplitude_scale"],
        }
        self._log_banner_once(
            len(kept_labels), unit, n_time_patches,
            overrides["apply_amplitude_scale"],
        )
        return tokens, input_chans, input_times, input_mask, attn_mask

    def forward_features(self, batch) -> torch.Tensor:
        """Grad-enabled pooled features, ``(B, embed_dim)``. See base class."""
        tokens, input_chans, input_times, input_mask, attn_mask = self._prepare_tokens(batch)
        meta = batch.get("meta") or [{}]
        dataset = meta[0].get("dataset") if meta else None
        if dataset == "bci":
            return self.model.forward_features(
                tokens,
                input_chans=input_chans,
                input_times=input_times,
                mask=attn_mask,
                return_all_tokens=False,
            )
        features = self.model.forward_features(
            tokens,
            input_chans=input_chans,
            input_times=input_times,
            mask=attn_mask,
            return_all_tokens=True,
        )
        valid = input_mask.unsqueeze(-1).to(features.dtype)
        pooled = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return replace_nonfinite_with_zero(pooled, where="neurolm:output")

    def extract_embeddings(self, batch) -> np.ndarray:
        with torch.inference_mode():
            return self.forward_features(batch).detach().float().cpu().numpy()

    def extract_embeddings_perpatch(self, batch) -> np.ndarray:
        tokens, input_chans, input_times, input_mask, attn_mask = self._prepare_tokens(batch)
        with torch.inference_mode():
            features = self.model.forward_features(
                tokens,
                input_chans=input_chans,
                input_times=input_times,
                mask=attn_mask,
                return_all_tokens=True,
            )
        # features: (B, block_size, 768). Select only valid tokens.
        B = features.shape[0]
        n_valid = int(input_mask[0].sum().item())
        valid_features = []
        for i in range(B):
            mask_i = input_mask[i]  # (block_size,)
            valid_features.append(features[i, mask_i, :])  # (n_valid, 768)
        patch_tokens = torch.stack(valid_features, dim=0)  # (B, n_valid, 768)
        return patch_tokens.detach().float().cpu().numpy()

    def metadata(self) -> Dict[str, Any]:
        return {
            **super().metadata(),
            "device": self.device,
            "weight_source": self._weight_source,
            "weight_load_report": dict(self._load_report),
            "pretrain_sfreq": _TARGET_SFREQ,
            "patch_size_samples": _PATCH_SIZE,
            "block_size": self.block_size,
            "tokenization": "channel-major: N_ch × N_time_patches × 200 samples at 200 Hz",
            "embedding_reduction": "mean_over_valid_vq_encoder_tokens",
            "alignment_category": "C",
            "alignment_note": (
                "paper has no frozen linear-probe path; downstream uses "
                "learned encode_transform_layer + GPT2 instruction-tuning. "
                "Masked-mean over valid tokens is a benchmark-side adaptation."
            ),
            "wrapper_contract": "model_specific_transforms_only",
            **self._last_channel_report,
        }
