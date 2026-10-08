"""Bonn EEG (Andrzejak et al. 2001) -> flat pre-segmented HDF5 preprocessor.

The Bonn dataset is five sets of 100 single-channel clips (4097 samples
at 173.61 Hz, ~23.6 s each):

    Z = surface EEG, 5 healthy volunteers, eyes open
    O = surface EEG, 5 healthy volunteers, eyes closed
    N = intracranial, 5 patients, interictal, hippocampal formation
        contralateral to the seizure focus
    F = intracranial, 5 patients, interictal, within the epileptogenic zone
    S = intracranial, 5 patients, ictal

Clip->patient mapping is not distributed with the raw files, so
``subject_idx`` is stored as -1 everywhere.  Auxiliary fields
(recording_type / subject_type / state / electrode_location) are
deterministic from the set letter and are stored for convenience.

Reference:
    Andrzejak, R. G. et al. (2001).  Indications of nonlinear
    deterministic and finite-dimensional structures in time series of
    brain electrical activity: Dependence on recording region and brain
    state.  Phys. Rev. E, 64(6), 061907.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Bonn is distributed at this (slightly irrational) sampling rate.
NATIVE_FS: float = 173.61
#: Every clip is exactly this many samples.
SAMPLES_PER_CLIP: int = 4097
#: Clips per set in the canonical release.
CLIPS_PER_SET: int = 100
#: Five set letters, in a stable order — also defines the integer
#: class code: Z=0, O=1, N=2, F=3, S=4.
SET_ORDER: Tuple[str, ...] = ("Z", "O", "N", "F", "S")
#: Mapping from set letter -> integer class code.
SET_TO_CODE: Dict[str, int] = {s: i for i, s in enumerate(SET_ORDER)}

CLASS_DESCRIPTIONS: Tuple[str, ...] = (
    "healthy surface awake eyes open",
    "healthy surface awake eyes closed",
    "interictal intracranial hippocampal contralateral",
    "interictal intracranial epileptogenic zone",
    "ictal intracranial",
)

#: Auxiliary (deterministic from set letter) fields.
RECORDING_TYPE: Dict[str, str] = {
    "Z": "surface", "O": "surface",
    "N": "intracranial", "F": "intracranial", "S": "intracranial",
}
SUBJECT_TYPE: Dict[str, str] = {
    "Z": "healthy", "O": "healthy",
    "N": "patient", "F": "patient", "S": "patient",
}
STATE: Dict[str, str] = {
    "Z": "awake_eyes_open",
    "O": "awake_eyes_closed",
    "N": "interictal",
    "F": "interictal",
    "S": "ictal",
}
ELECTRODE_LOCATION: Dict[str, str] = {
    "Z": "scalp_10_20",
    "O": "scalp_10_20",
    "N": "hippocampal_contralateral",
    "F": "epileptogenic_zone",
    "S": "epileptogenic_zone",
}

PATIENT_INFO: str = (
    "5 patients, all with temporal lobe epilepsy + hippocampal sclerosis "
    "(Andrzejak et al. 2001). Per-patient age and clip->subject mapping "
    "are not distributed with the raw files."
)

CACHE_SCHEMA_TAG: str = "173hz_segments"
DEFAULT_CACHE_FILENAME: str = f"bonn_{CACHE_SCHEMA_TAG}.h5"


# ---------------------------------------------------------------------------
# Raw file discovery
# ---------------------------------------------------------------------------


def data_folder(raw_root: str | Path) -> Path:
    """The folder holding the five set folders: ``raw_root`` itself, or a
    folder under it (``raw/``, or an archive's top folder) that holds them."""
    from neuroatlas.extensions.datasets._layout import descend

    for marker in ("Z", "setA", "A"):
        found = descend(raw_root, ["raw", "*"], marker)
        if (found / marker).is_dir():
            return found
    return Path(raw_root)


def _find_set_directory(raw_root: Path, set_letter: str) -> Path:
    """Locate the directory containing the .TXT files for one set.

    Kaggle mirrors disagree on naming: some use 'Z', 'O', ... directly,
    others use 'setA', 'A', etc.  We accept both.
    """
    aliases_by_letter = {
        "Z": ("Z", "setA", "A"),
        "O": ("O", "setB", "B"),
        "N": ("N", "setC", "C"),
        "F": ("F", "setD", "D"),
        "S": ("S", "setE", "E"),
    }
    for alias in aliases_by_letter[set_letter]:
        cand = raw_root / alias
        if cand.is_dir() and any(
            p.is_file() and p.suffix.lower() == ".txt"
            for p in cand.iterdir()
        ):
            return cand
    raise FileNotFoundError(
        f"Could not find set {set_letter!r} directory under {raw_root} "
        f"(looked for {aliases_by_letter[set_letter]})."
    )


def _list_clip_files(set_dir: Path) -> List[Path]:
    """Sort .TXT files alphabetically — the canonical Bonn ordering."""
    files = sorted(
        p for p in set_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    )
    return files


def _load_clip(path: Path) -> np.ndarray:
    """Load a single Bonn .TXT clip as a ``(SAMPLES_PER_CLIP,)`` float32 array.

    Bonn files are plain ASCII, one integer (microvolt amplitude) per line.
    We use ``np.loadtxt`` with float32 dtype and assert the expected length.
    """
    sig = np.loadtxt(path, dtype=np.float32)
    if sig.ndim != 1:
        raise ValueError(f"{path}: expected 1-D integer stream, got shape {sig.shape}")
    if sig.size != SAMPLES_PER_CLIP:
        # Some mirrors ship clips that are 1 sample shorter/longer — tolerate
        # that by truncating/padding to the canonical length with a warning.
        logger.warning(
            "%s: expected %d samples, got %d — %s",
            path, SAMPLES_PER_CLIP, sig.size,
            "truncating" if sig.size > SAMPLES_PER_CLIP else "zero-padding",
        )
        if sig.size > SAMPLES_PER_CLIP:
            sig = sig[:SAMPLES_PER_CLIP]
        else:
            sig = np.pad(sig, (0, SAMPLES_PER_CLIP - sig.size))
    return sig.astype(np.float32)


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------


def load_raw_corpus(
    raw_root: str | Path,
    sets: Iterable[str] = SET_ORDER,
) -> dict:
    """Read Bonn's plain-text clips into the arrays the cache would hold.

    Bonn ships 500 single-channel clips of 4097 samples -- roughly 8 MB once
    decoded -- so there is no reason to require a prebuilt HDF5 just to read
    it. This is the half of :func:`build_h5_cache` that runs before h5py is
    opened, so the raw path and the cached path cannot drift: the cache is
    literally this dictionary, written out.

    Returns a mapping with the same keys as the HDF5 datasets, plus the
    attributes under ``"attrs"``.
    """
    raw_root = data_folder(raw_root)
    sets = tuple(sets)
    unknown = [s for s in sets if s not in SET_TO_CODE]
    if unknown:
        raise ValueError(f"Unknown set letter(s) {unknown}; expected subset of {SET_ORDER}")

    logger.info("Bonn: reading raw_root=%s sets=%s", raw_root, sets)

    signals: List[np.ndarray] = []
    class_labels: List[int] = []
    subset_letters: List[str] = []

    for letter in sets:
        set_dir = _find_set_directory(raw_root, letter)
        files = _list_clip_files(set_dir)
        logger.info("  set %s: %d files in %s", letter, len(files), set_dir)
        if len(files) != CLIPS_PER_SET:
            logger.warning(
                "  set %s: expected %d files, got %d (continuing anyway)",
                letter, CLIPS_PER_SET, len(files),
            )

        for fp in files:
            signals.append(_load_clip(fp))
            class_labels.append(SET_TO_CODE[letter])
            subset_letters.append(letter)

    n_clips = len(signals)
    if n_clips == 0:
        raise RuntimeError(f"No Bonn clips found under {raw_root}")

    # (N, 1, T) — single-channel
    signals_arr = np.stack(signals, axis=0).astype(np.float32)[:, None, :]
    class_label_arr = np.array(class_labels, dtype=np.uint8)
    subset_arr = np.array(subset_letters, dtype=object)

    # Derivable auxiliary fields
    recording_type_arr = np.array(
        [RECORDING_TYPE[s] for s in subset_letters], dtype=object,
    )
    subject_type_arr = np.array(
        [SUBJECT_TYPE[s] for s in subset_letters], dtype=object,
    )
    state_arr = np.array(
        [STATE[s] for s in subset_letters], dtype=object,
    )
    electrode_location_arr = np.array(
        [ELECTRODE_LOCATION[s] for s in subset_letters], dtype=object,
    )
    subject_idx_arr = np.full(n_clips, -1, dtype=np.int16)

    return {
        "signals": signals_arr,
        "class_label": class_label_arr,
        "subset": subset_arr,
        "subject_idx": subject_idx_arr,
        "recording_type": recording_type_arr,
        "subject_type": subject_type_arr,
        "state": state_arr,
        "electrode_location": electrode_location_arr,
        "attrs": {
            "fs": NATIVE_FS,
            "n_samples_per_segment": SAMPLES_PER_CLIP,
            "corpus": "bonn",
            "schema": "pre_segmented",
            "schema_tag": CACHE_SCHEMA_TAG,
            "class_names": list(SET_ORDER),
            "class_descriptions": list(CLASS_DESCRIPTIONS),
            "patient_info": PATIENT_INFO,
        },
    }


def build_h5_cache(
    raw_root: str | Path,
    output_path: str | Path,
    sets: Iterable[str] = SET_ORDER,
) -> Path:
    """Write the flat pre-segmented HDF5 cache. A fast path, not a requirement.

    The adapter reads the raw clips directly by default; this exists so a
    large sweep does not re-parse 500 text files per job.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus = load_raw_corpus(raw_root, sets)
    signals_arr = corpus["signals"]
    n_clips = len(signals_arr)

    vlen_str = h5py.special_dtype(vlen=str)
    logger.info("Writing HDF5: %d clips total, signals shape %s",
                n_clips, signals_arr.shape)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with h5py.File(str(tmp_path), "w") as h5:
        for _k, _v in corpus["attrs"].items():
            h5.attrs[_k] = _v

        h5.create_dataset("signals", data=signals_arr, chunks=(1, 1, SAMPLES_PER_CLIP))
        h5.create_dataset("class_label", data=corpus["class_label"])
        h5.create_dataset("subset", data=corpus["subset"], dtype=vlen_str)
        h5.create_dataset("subject_idx", data=corpus["subject_idx"])
        h5.create_dataset("recording_type", data=corpus["recording_type"], dtype=vlen_str)
        h5.create_dataset("subject_type", data=corpus["subject_type"], dtype=vlen_str)
        h5.create_dataset("state", data=corpus["state"], dtype=vlen_str)
        h5.create_dataset(
            "electrode_location", data=corpus["electrode_location"], dtype=vlen_str,
        )

    tmp_path.replace(output_path)
    logger.info("Bonn HDF5 cache written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "NATIVE_FS",
    "SAMPLES_PER_CLIP",
    "CLIPS_PER_SET",
    "SET_ORDER",
    "SET_TO_CODE",
    "CLASS_DESCRIPTIONS",
    "RECORDING_TYPE",
    "SUBJECT_TYPE",
    "STATE",
    "ELECTRODE_LOCATION",
    "PATIENT_INFO",
    "CACHE_SCHEMA_TAG",
    "DEFAULT_CACHE_FILENAME",
    "build_h5_cache",
    "load_raw_corpus",
]
