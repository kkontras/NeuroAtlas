"""TUSZ EDF reading, channel normalisation, and subject metadata.

TUSZ recordings come as EDF files with variable per-channel sampling rates
and inconsistent channel label formats (e.g. ``EEG FP1-REF``, ``T7-LE``,
``Fp1``).  This module normalises channel names, resamples each channel to
the target rate, and returns a ``(n_samples, 19)`` unipolar array ordered
according to :data:`UNIPOLAR_ELECTRODES`.

It also provides:

- The TUSZ-specific montage constants (``UNIPOLAR_ELECTRODES``,
  ``BIPOLAR_MONTAGE``, ``TCP_MONTAGE_CHANNELS``).
- A time-first ``bipolar_from_unipolar`` (distinct from the channel-first
  version in ``_common.py`` — see that module for the reason).
- A parser for the sibling ``tuh_eeg_epilepsy`` corpus which supplies
  age/sex/diagnosis labels.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Montage / schema constants
# ---------------------------------------------------------------------------

UNIPOLAR_ELECTRODES: Tuple[str, ...] = (
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "CZ", "FZ",
    "P3", "P4", "PZ", "T3", "T4", "T5", "T6", "O1", "O2",
)

BIPOLAR_MONTAGE: Tuple[str, ...] = (
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "T3-C3", "C3-CZ", "CZ-C4", "C4-T4",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
)

# TCP montage (Temple Central Parietal) used by the original TUSZ annotations.
# Includes ear electrodes A1/A2 and one more bipolar row than BIPOLAR_MONTAGE.
TCP_MONTAGE_CHANNELS: Tuple[str, ...] = (
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "A1-T3", "T3-C3", "C3-CZ", "CZ-C4", "C4-T4", "T4-A2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
)

TARGET_FS: int = 256
CACHE_SCHEMA_TAG: str = "256hz_continuous_unipolar19"

# ``T7/T8/P7/P8`` are the 10-10 names; TUSZ mixes them with classic 10-20
# names.  Collapse to the 10-20 form so our canonical 19 layout is stable.
_ELECTRODE_ALIASES: Dict[str, str] = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
    "FP1": "FP1",
    "FP2": "FP2",
}


# ---------------------------------------------------------------------------
# TUH EDF+ patient-field parsing
# ---------------------------------------------------------------------------


_TUH_AGE_RE = re.compile(r"Age:\s*(\d+)", re.IGNORECASE)
_TUH_SEX_RE = re.compile(r"\b([MF])\b")


def parse_tuh_patient_field(edf_path) -> Dict[str, Any]:
    """Parse the EDF+ Patient field from a TUH corpus EDF.

    TUH encodes demographics in the 80-byte Patient field (bytes 8:88 of the
    EDF header) as:  ``"<patient_id> <M|F> <birthdate> <last_name> Age:<N>"``.
    Birthdate is usually the placeholder ``01-JAN-0000`` (anonymised), so age
    comes from the trailing ``Age:N`` token.

    Returns ``{"age": int, "gender": str}`` with ``age=-1`` and ``gender=""``
    when the respective token is absent or unparseable.
    """
    try:
        with open(edf_path, "rb") as f:
            raw = f.read(88)
        field = raw[8:88].decode("latin-1", errors="replace")
    except Exception:
        return {"age": -1, "gender": ""}
    age = -1
    m = _TUH_AGE_RE.search(field)
    if m:
        try:
            age = int(m.group(1))
        except ValueError:
            age = -1
        # TUH sentinel for anonymised/unknown age is 999 (paired with sex=X).
        if age == 999:
            age = -1
    sex = ""
    # Skip the patient_id token (first whitespace-delimited field) when
    # matching M/F so we don't pick up an id that happens to start with M/F.
    tokens = field.strip().split(None, 1)
    tail = tokens[1] if len(tokens) == 2 else field
    sm = _TUH_SEX_RE.search(tail)
    if sm:
        sex = sm.group(1).lower()  # "m" / "f"
    return {"age": age, "gender": sex}


# ---------------------------------------------------------------------------
# Channel name normalisation
# ---------------------------------------------------------------------------


def normalize_channel_name(name: str) -> str:
    """Normalize a raw EDF channel label to a standard 10-20 name.

    Strips the common ``EEG `` prefix, the reference suffix (``-REF``,
    ``-LE``, ``-AR``, ``-AVG``, ``-A1``, ``-A2``), then applies the 10-10 →
    10-20 alias map.
    """
    name = name.strip().upper()
    name = re.sub(r"^EEG\s+", "", name)
    name = re.sub(r"-(REF|LE|AR|AVG|A[12])$", "", name)
    name = name.strip()
    return _ELECTRODE_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Bipolar helper (time-first convention, distinct from _common.py)
# ---------------------------------------------------------------------------


def bipolar_from_unipolar(signals: np.ndarray) -> np.ndarray:
    """Derive 20-channel bipolar TCP view from 19 unipolar channels.

    Uses the **time-first** axis convention required by the TUSZ dataio
    layer (``(T, 19) → (T, 20)``).  This is deliberately different from the
    channel-first version in ``_common.py``.

    Args:
        signals: ``(n_samples, 19)`` array with columns ordered as
            :data:`UNIPOLAR_ELECTRODES`.

    Returns:
        ``(n_samples, 20)`` array with columns ordered as
        :data:`BIPOLAR_MONTAGE`.
    """
    ch_idx = {name: i for i, name in enumerate(UNIPOLAR_ELECTRODES)}
    out = np.empty((signals.shape[0], len(BIPOLAR_MONTAGE)), dtype=signals.dtype)
    for col, pair in enumerate(BIPOLAR_MONTAGE):
        a, b = pair.split("-")
        out[:, col] = signals[:, ch_idx[a]] - signals[:, ch_idx[b]]
    return out


# ---------------------------------------------------------------------------
# EDF reading
# ---------------------------------------------------------------------------


def read_edf_unipolar(
    edf_path: str,
) -> Tuple[np.ndarray, float, int, List[str]]:
    """Read an EDF and return the 19 unipolar channels resampled to ``TARGET_FS``.

    Missing channels are zero-filled.  Channels with a different native
    sampling rate are resampled with ``scipy.signal.resample``. Per-channel
    physical dimension is read from the EDF header and eagerly scaled to µV
    (uV×1, mV×1e3, V×1e6) so the returned signal is always in µV regardless
    of header declaration; the raw declarations are returned alongside for
    auditability / downstream logging.

    Args:
        edf_path: Path to the EDF file.

    Returns:
        ``(signals, native_fs, n_missing, physical_units)`` where
          - ``signals`` is ``(n_samples_at_TARGET_FS, 19)`` ``float32``, in µV,
          - ``native_fs`` is the source sampling rate (from the first matched channel),
          - ``n_missing`` is the count of the 19 target electrodes not found,
          - ``physical_units`` is a 19-element list of the raw unit strings
            declared in the EDF header (empty string for missing electrodes),
            ordered to match :data:`UNIPOLAR_ELECTRODES`.
    """
    import pyedflib

    from neuroatlas.extensions.datasets.dataio._edf_units import (
        edf_unit_to_uv_scale,
    )

    reader = pyedflib.EdfReader(edf_path)
    try:
        n_signals = reader.signals_in_file
        channel_labels = [reader.getLabel(i) for i in range(n_signals)]
        normalized = [normalize_channel_name(label) for label in channel_labels]

        # Map normalized names to signal indices
        name_to_idx: Dict[str, int] = {}
        for idx, norm_name in enumerate(normalized):
            if norm_name not in name_to_idx:
                name_to_idx[norm_name] = idx

        # Determine native sampling rate from first matched channel
        native_fs = None
        for electrode in UNIPOLAR_ELECTRODES:
            if electrode in name_to_idx:
                native_fs = reader.getSampleFrequency(name_to_idx[electrode])
                break
        if native_fs is None:
            native_fs = reader.getSampleFrequency(0) if n_signals > 0 else TARGET_FS

        duration_s = reader.getFileDuration()
        n_target = int(round(duration_s * TARGET_FS))
        if n_target == 0:
            n_target = 1

        signals = np.zeros((n_target, len(UNIPOLAR_ELECTRODES)), dtype=np.float32)
        n_missing = 0
        physical_units: List[str] = [""] * len(UNIPOLAR_ELECTRODES)

        for col, electrode in enumerate(UNIPOLAR_ELECTRODES):
            if electrode not in name_to_idx:
                n_missing += 1
                continue
            sig_idx = name_to_idx[electrode]
            raw_signal = reader.readSignal(sig_idx)

            unit = reader.getPhysicalDimension(sig_idx)
            if isinstance(unit, bytes):
                unit = unit.decode(errors="ignore")
            unit = unit.strip()
            physical_units[col] = unit
            scale = edf_unit_to_uv_scale(unit)
            if scale != 1.0:
                raw_signal = raw_signal * scale

            ch_fs = reader.getSampleFrequency(sig_idx)
            if abs(ch_fs - TARGET_FS) > 0.5:
                from scipy.signal import resample
                target_len = int(round(len(raw_signal) * TARGET_FS / ch_fs))
                raw_signal = resample(raw_signal, target_len).astype(np.float32)
            else:
                raw_signal = raw_signal.astype(np.float32)
            # Truncate or pad to n_target
            length = min(len(raw_signal), n_target)
            signals[:length, col] = raw_signal[:length]

        return signals, float(native_fs), n_missing, physical_units
    finally:
        reader._close()


def discover_recordings(raw_root: str, split: str) -> List[str]:
    """Find all EDF paths for a given TUSZ split (``train``/``dev``/``eval``).

    ``raw_root`` is the dataset's folder, or the ``edf/`` (or
    ``<version>/edf/``) folder TUH serves under it.
    """
    from neuroatlas.extensions.datasets._layout import descend

    split_dir = descend(raw_root, ["edf", "*/edf"], split) / split
    edfs = sorted(str(p) for p in split_dir.rglob("*.edf"))
    logger.info("Discovered %d EDFs in %s/%s", len(edfs), raw_root, split)
    return edfs


# ---------------------------------------------------------------------------
# Sibling corpus metadata
# ---------------------------------------------------------------------------


def load_tuh_eeg_epilepsy_metadata(
    epilepsy_root: str,
) -> Dict[str, Dict[str, Any]]:
    """Parse age / sex / diagnosis from the sibling ``tuh_eeg_epilepsy`` corpus.

    TUSZ itself doesn't ship demographic data, but the companion
    ``tuh_eeg_epilepsy`` corpus publishes two subject lists with the fields
    we need.  File format example::

        aaaaaanr [Age:35] [M] unknown

    Args:
        epilepsy_root: Directory containing the two ``*_subject_ids_*.list`` files.

    Returns:
        ``{subject_id: {"age": int, "sex": "M"|"F"|"U", "diagnosis": "epilepsy"|"no_epilepsy"}}``.
        Subjects not listed are simply absent from the returned dict.
    """
    root = Path(epilepsy_root)
    meta: Dict[str, Dict[str, Any]] = {}

    for label_file, diagnosis in [
        ("00_subject_ids_epilepsy.list", "epilepsy"),
        ("01_subject_ids_no_epilepsy.list", "no_epilepsy"),
    ]:
        path = root / label_file
        if not path.exists():
            logger.warning("Epilepsy list not found: %s", path)
            continue
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                subject_id = parts[0]
                age = -1
                sex = "U"
                age_match = re.search(r"\[Age:(\d+)\]", line)
                if age_match:
                    age = int(age_match.group(1))
                sex_match = re.search(r"\[([MF])\]", line)
                if sex_match:
                    sex = sex_match.group(1)
                meta[subject_id] = {
                    "age": age,
                    "sex": sex,
                    "diagnosis": diagnosis,
                }
    return meta
