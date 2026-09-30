"""Per-model channel-name vocabularies used by the channel-map loader.

The loader validates that every rename target declared in
``configs/channel_maps/<dataset>.yaml`` exists in the destination
model's vocabulary. A typo in the YAML (e.g. ``"Fz"`` instead of
``"FZ"``) raises at config-load time rather than producing silent
zero-padded inputs at inference.

These vocabularies are **minimal, conservative supersets** of what the
upstream model code actually accepts. They are intentionally not
exhaustive — the full upstream vocabularies (NeuroLM's ~130-entry
standard_1020, REVE's 543-entry position bank, EEGPT's 61-entry
CHANNEL_DICT) are loaded inside each wrapper. The loader's job is
only to catch obviously-wrong targets at config time; the wrapper's
alias map provides the final fallback.
"""
from __future__ import annotations

from typing import Dict, FrozenSet

# Standard 10-20 / 10-10 labels (uppercase). Superset used by EEGPT,
# LaBraM, NeuroLM. Includes legacy T3/T4/T5/T6 that some datasets still
# publish.
_STANDARD_1020_UPPER: FrozenSet[str] = frozenset({
    "FP1", "FP2", "FPZ",
    # The AF row and FC1/2/5/6 were missing, so a montage naming them was
    # rejected at config time even though the wrappers accept them: EEGPT lists
    # AF7/AF3/AF4/AF8 and FC5..FC6 (eegpt_encoder.py), and NeuroLM
    # (neurolm.py) and S-TEEGformer carry them too. DREAMER's 14-channel
    # Emotiv cap needs AF3/AF4/FC5/FC6, which is how the gap surfaced.
    "AF7", "AF3", "AF4", "AF8",
    "F7", "F3", "FZ", "F4", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T3", "T5", "T7", "T4", "T6", "T8",
    "C3", "CZ", "C4",
    "TP7", "CP3", "CPZ", "CP4", "TP8",
    "P7", "P3", "PZ", "P4", "P8",
    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1", "OZ", "O2",
    "A1", "A2",
    "M1", "M2",
})

# REVE uses lowercase / mixed-case per its position bank.
_REVE_POSITIONS: FrozenSet[str] = frozenset({
    "Fp1", "Fp2", "Fpz",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    # Verified present in the published 543-entry bank
    # (brain-bzh/reve-positions, positions.json): AF3, AF4, FC5, FC6 all
    # resolve. This list is the conservative config-time check, not the bank.
    "AF7", "AF3", "AF4", "AF8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "A1", "A2", "M1", "M2",
})

# CBraMod is index-based + permutation-invariant. Any label is accepted;
# this vocabulary is a placeholder so the loader has a "no-restriction"
# marker.
_ANY_LABEL: FrozenSet[str] = frozenset()  # empty set → no restriction

# BIOT accepts bipolar slot names (direct matches) and monopolar 10-20
# names (the wrapper computes bipolar derivations by subtraction).
_BIOT_VOCABULARY: FrozenSet[str] = frozenset({
    # 16 BIOT double-banana slot names
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "C3-A2", "C4-A1",
    # Monopolar 10-20 names (derivation inputs)
    "FP1", "FP2", "FPZ",
    "F7", "F3", "FZ", "F4", "F8",
    "T3", "T7", "C3", "CZ", "C4", "T4", "T8",
    "T5", "P7", "P3", "PZ", "P4", "T6", "P8",
    "O1", "OZ", "O2",
    # Electrodes BIOT tolerates but derives no slot from. A montage may legally
    # carry them -- DREAMER's 14-channel Emotiv cap has AF3/AF4/FC5/FC6 -- and
    # _biot_slot_plan simply does not use them, filling 14 of 16 slots from the
    # rest. Listing them keeps the typo check meaningful ("FZ" vs "Fz") without
    # rejecting a real electrode the wrapper handles.
    "AF7", "AF3", "AF4", "AF8",
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    # DOD-specific bipolar labels (passed through, ignored by wrapper)
    "F3-F4", "F3-O1", "F4-O2", "FP1-O1", "FP2-O2",
})


# Bipolar label pass-through labels (CBraMod and other index-based
# wrappers accept arbitrary labels); we accept any string that contains
# a hyphen as a valid bipolar pair.
def _accepts_any(_target: str) -> bool:
    return True

# Model-family → vocabulary. Value can be either:
# - a frozenset of permitted names (exact membership check)
# - the literal string "any" meaning no restriction (index-based /
#   permutation-invariant models)
MODEL_VOCABULARIES: Dict[str, object] = {
    "eegpt": _STANDARD_1020_UPPER,
    "labram": _STANDARD_1020_UPPER,
    "neurolm": _STANDARD_1020_UPPER,
    "reve": _REVE_POSITIONS,
    "cbramod": "any",
    "biot": _BIOT_VOCABULARY,
    # Unmigrated / non-EEG models: accept anything (loader is permissive).
    "neurorvq": "any",
    "core_sleep": "any",
    "sleep_transformer": "any",
    "sleepyco": "any",
    "sleepfm": "any",
    "physioex": "any",
    # Univariate TS foundation models — channel-name-agnostic.
    "chronos": "any",
    "moment": "any",
    "timemoe": "any",
    "moirai": "any",
    "lagllama": "any",
    "timesfm": "any",
}

def accepts_label(model_family: str, target: str) -> bool:
    """Return True iff ``target`` is a valid name in ``model_family``'s vocabulary."""
    vocab = MODEL_VOCABULARIES.get(model_family)
    if vocab is None:
        # Model family unknown to this lookup — be permissive; the
        # wrapper's own alias map is the final authority.
        return True
    if vocab == "any":
        return True
    assert isinstance(vocab, frozenset)
    return target in vocab

def suggest_close_matches(model_family: str, target: str, n: int = 3):
    """Return up to ``n`` likely-intended names from the vocabulary for a typo."""
    from difflib import get_close_matches
    vocab = MODEL_VOCABULARIES.get(model_family)
    if not isinstance(vocab, frozenset) or not vocab:
        return []
    return get_close_matches(target, list(vocab), n=n, cutoff=0.6)
