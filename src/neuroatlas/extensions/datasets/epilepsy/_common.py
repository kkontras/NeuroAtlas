"""Shared constants and signal helpers for all epilepsy preprocessors.

Symbols here are identical across the CHB-MIT, Siena, and EPILEPSIAE
preprocessors.  Import from this module instead of copy-pasting.

TUSZ uses a different axis convention for ``bipolar_from_unipolar``
(time-first: ``(T, 19) → (T, 18)``) and keeps its own copy in
``tusz_preprocessor.py``.  Everything else is shared.
"""
from __future__ import annotations

from functools import lru_cache
from math import gcd
from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

# ---------------------------------------------------------------------------
# Canonical 10-20 layout
# ---------------------------------------------------------------------------

CANONICAL_19: Tuple[str, ...] = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T3", "C3", "CZ", "C4", "T4",
    "T5", "P3", "PZ", "P4", "T6",
    "O1", "O2",
)
CANONICAL_SET = frozenset(CANONICAL_19)
CANONICAL_IDX = {ch: i for i, ch in enumerate(CANONICAL_19)}

# ---------------------------------------------------------------------------
# 20-pair bipolar TCP montage (channel-first convention)
#
# Symmetric LL / RL / Central / LP / RP chains. The right-parasagittal chain
# (FP2-F4, F4-C4, C4-P4, P4-O2) matches the left-parasagittal (FP1-F3, F3-C3,
# C3-P3, P3-O1); P4 is otherwise absent from any pair. Without C4-P4 and P4-O2
# the P4 electrode is collapsed out of the bipolar output and cannot be
# algebraically recovered downstream (needed by BIOT's pretrained slots 14-15).
# ---------------------------------------------------------------------------

BIPOLAR_MONTAGE: Tuple[Tuple[str, str], ...] = (
    ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("T3", "C3"), ("C3", "CZ"), ("CZ", "C4"), ("C4", "T4"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
)
BIPOLAR_NAMES: Tuple[str, ...] = tuple(f"{a}-{b}" for a, b in BIPOLAR_MONTAGE)
_BIP_A = np.array([CANONICAL_IDX[a] for a, _ in BIPOLAR_MONTAGE])
_BIP_B = np.array([CANONICAL_IDX[b] for _, b in BIPOLAR_MONTAGE])

# ---------------------------------------------------------------------------
# Cache schema
# ---------------------------------------------------------------------------

TARGET_FS: int = 256
CACHE_SCHEMA_TAG: str = "256hz_continuous_unipolar19"
GAP_LABEL: int = 255

# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def bipolar_from_unipolar(windows: np.ndarray) -> np.ndarray:
    """Derive 20-channel bipolar TCP montage from 19-channel unipolar.

    Uses the channel-first axis convention shared by EPILEPSIAE, Siena,
    and CHB-MIT preprocessors.

    Args:
        windows: ``(..., 19, T)`` unipolar signals.

    Returns:
        ``(..., 20, T)`` bipolar signals via pairwise subtraction.
    """
    return windows[..., _BIP_A, :] - windows[..., _BIP_B, :]


def resample_to(signals: np.ndarray, src_fs: float, tgt_fs: float) -> np.ndarray:
    """Resample ``(C, T)`` signals from ``src_fs`` to ``tgt_fs``.

    Uses polyphase filtering (no spectral leakage).  Returns the input
    unchanged if rates differ by less than 0.5 Hz.
    """
    if abs(src_fs - tgt_fs) < 0.5:
        return signals
    up = int(tgt_fs)
    down = int(src_fs)
    g = gcd(up, down)
    return resample_poly(signals, up // g, down // g, axis=1).astype(np.float32)


def window_binary_from_intervals(
    window_start: int,
    window_end: int,
    intervals_samples,
    threshold: float = 0.0,
) -> Tuple[int, float]:
    """Binary seizure label + fraction for a window from sample-index intervals.

    Args:
        window_start: Window start (samples, inclusive).
        window_end: Window end (samples, exclusive).
        intervals_samples: Iterable of ``(start_sample, end_sample)`` pairs.
        threshold: Label is ``1`` iff ``overlap / window_len > threshold``.

    Returns:
        ``(binary_label, seizure_fraction)``.
    """
    window_len = max(int(window_end) - int(window_start), 1)
    overlap = 0
    for sz_start, sz_end in intervals_samples:
        o = max(0, min(int(window_end), int(sz_end)) - max(int(window_start), int(sz_start)))
        overlap += o
    fraction = overlap / window_len
    label = int(fraction > threshold)
    return label, fraction


def apply_standard_filters(
    signals: np.ndarray,
    fs: float = 256.0,
    notch_hz: float = 50.0,
    axis: int = 1,
) -> np.ndarray:
    """Apply 0.5 Hz highpass + mains notch filters (zero-phase).

    Args:
        signals: float32 array at sampling rate ``fs``.  Time axis is
            ``axis`` (default ``1`` for channel-first ``(C, T)``; pass
            ``axis=0`` for time-first ``(T, C)``).
        fs: sampling rate in Hz.
        notch_hz: mains frequency to notch.  Default ``50.0`` is right
            for European recordings (EPILEPSIAE, Siena unfiltered path,
            NMT).  Pass ``60.0`` for US recordings (TUAB, TUSZ, CHB-MIT).
        axis: time axis of ``signals``.

    Returns:
        Filtered float32 array with the same shape as ``signals``.
    """
    b_hp, a_hp = _hp_coeffs(float(fs))
    signals = filtfilt(b_hp, a_hp, signals, axis=axis).astype(np.float32)

    b_notch, a_notch = _notch_coeffs(float(fs), float(notch_hz))
    signals = filtfilt(b_notch, a_notch, signals, axis=axis).astype(np.float32)

    return signals


@lru_cache(maxsize=16)
def _hp_coeffs(fs: float):
    return butter(4, 0.5, btype="high", fs=fs)


@lru_cache(maxsize=16)
def _notch_coeffs(fs: float, notch_hz: float):
    return iirnotch(notch_hz, Q=30.0, fs=fs)
