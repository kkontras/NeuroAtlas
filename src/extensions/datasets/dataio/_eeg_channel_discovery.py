"""Shared utility for discovering EEG channels from EDF headers.

Used by per-dataset ``discover_eeg_channels()`` functions when the
``use_all_eeg_channels`` flag is active (timeseries foundation models).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_NON_EEG_PREFIXES: Tuple[str, ...] = (
    "EOG", "EMG", "ECG", "EKG",
    "SPO2", "SAO2",
    "RESP", "FLOW", "AIRFLOW", "SNORE", "CHEST", "ABDOMEN", "THORAX",
    "BODY", "POSITION", "POS",
    "TEMP", "LIGHT", "GRAVITY",
    "EVENT", "STATUS", "MARKER",
    "EDF ANNOTATIONS", "EDF+",
    "CHIN", "LEG", "TIBIALIS",
    "PULSE", "PLETH", "HR", "HEART",
    "CPAP", "PRESS", "LEAK",
    "PH", "CO2", "ETCO2", "PTAF", "THERM",
    "DC", "REF", "A1", "A2", "M1", "M2",
    "LOC", "ROC",
    "LEFTEYE", "RIGHTEYE",
    "NASAL", "ORAL", "NASALORAL", "NASALFLOW", "ORALFLOW", "NAS_PRES",
    "LLEG", "RLEG", "LAT-", "RAT-",
    "SOUND", "RIBCAGE", "ABD", "SUM",
    "E1", "E2",
)

_NON_EEG_EXACT: frozenset = frozenset({
    "LAT", "RAT",
})


def is_eeg_label(label: str) -> bool:
    """Heuristic: return True if an EDF signal label looks like EEG."""
    upper = label.strip().upper()
    if not upper:
        return False
    for prefix in _NON_EEG_PREFIXES:
        if upper.startswith(prefix):
            return False
    if upper in _NON_EEG_EXACT:
        return False
    return True


def read_all_eeg_channels(
    edf_path: str,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    label_filter=None,
) -> Tuple[np.ndarray, List[str], float]:
    """Read ALL EEG channels from an EDF, filtering out non-EEG signals.

    Parameters
    ----------
    edf_path : str
        Path to the EDF file.
    start_sec, duration_sec : float
        Time window to read (full recording if duration_sec is None).
    label_filter : callable or None
        Optional per-dataset filter ``(label: str) -> bool``.
        Falls back to ``is_eeg_label`` when None.

    Returns
    -------
    signals : ndarray, shape (C, T)
    channel_names : list of str
    sfreq : float
    """
    import edfio

    filt = label_filter or is_eeg_label
    edf = edfio.read_edf(edf_path)

    channel_data: List[np.ndarray] = []
    names: List[str] = []
    ref_sfreq: Optional[float] = None

    for sig in edf.signals:
        if not filt(sig.label):
            continue
        sfreq = float(sig.sampling_frequency)
        if ref_sfreq is None:
            ref_sfreq = sfreq
        elif sfreq != ref_sfreq:
            continue

        s_start = int(start_sec * sfreq)
        if duration_sec is not None:
            data = sig.data[s_start : s_start + int(duration_sec * sfreq)]
        elif s_start > 0:
            data = sig.data[s_start:]
        else:
            data = sig.data
        channel_data.append(data.astype(np.float32))
        names.append(sig.label.strip())

    if not channel_data:
        return np.array([]), [], 0.0

    min_samples = min(len(c) for c in channel_data)
    stacked = np.stack([c[:min_samples] for c in channel_data], axis=0)
    return stacked, names, float(ref_sfreq)
