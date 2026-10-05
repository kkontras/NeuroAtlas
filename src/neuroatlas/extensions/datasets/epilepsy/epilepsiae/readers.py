"""EPILEPSIAE binary reader — .head/.data parsing, channel normalisation.

Handles the proprietary block-level binary format used by all three
EPILEPSIAE variants (surf30, surfPA, surfCO).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._common import CANONICAL_IDX, CANONICAL_SET

logger = logging.getLogger(__name__)

# Opt-in amplitude rescale for EPILEPSIAE's heterogeneous `conversion_factor`
# values. Some centers mis-export the factor so the output of
# `signals * conversion_factor` lands in V or mV instead of µV. Enabling this
# env at Condor submission time rescales those blocks on-the-fly with a
# single loud warning per block so you can audit what was touched.
_AUTO_RESCALE_ENV = "EPILEPSIAE_AUTO_RESCALE"
_RESCALE_V_TO_UV_THRESHOLD = 1e-4    # abs_max below this → data looked like V, ×1e6
_RESCALE_MV_TO_UV_THRESHOLD = 0.5    # abs_max below this but above V → data in mV, ×1e3

# ---------------------------------------------------------------------------
# EPILEPSIAE-specific channel constants
# ---------------------------------------------------------------------------

# Alias map: variant-specific channel names → canonical uppercase 10-20.
# Handles titlecase (surfPA), new 10-10 names T7/T8/P7/P8 (surfCO), and
# various capitalisation inconsistencies.  Kept local — differs per variant.
CHANNEL_ALIAS: Dict[str, str] = {
    # Titlecase → uppercase (surfPA / surfCO)
    "Fp1": "FP1", "Fp2": "FP2", "Fz": "FZ", "Cz": "CZ", "Pz": "PZ",
    "Fpz": "FPZ", "Oz": "OZ",
    # New 10-10 nomenclature → old 10-20 (surfCO)
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
    # Already canonical (pass-through)
    "FP1": "FP1", "FP2": "FP2", "F3": "F3", "F4": "F4",
    "C3": "C3", "C4": "C4", "P3": "P3", "P4": "P4",
    "O1": "O1", "O2": "O2", "F7": "F7", "F8": "F8",
    "T3": "T3", "T4": "T4", "T5": "T5", "T6": "T6",
    "FZ": "FZ", "CZ": "CZ", "PZ": "PZ",
    # FPz variants
    "FPz": "FPZ", "FPZ": "FPZ",
    "OZ": "OZ",
}

NON_EEG_PREFIXES = ("ECG", "EOG", "EMG", "PHO", "SP1", "SP2", "RS",
                    "MT1", "MT2", "PNG", "PULS", "BEAT", "SpO2",
                    "CB1", "CB2", "elA")

# Seizure type codes (EPILEPSIAE 4-class taxonomy)
SEIZURE_TYPE_CODES: Dict[str, int] = {"UC": 0, "SP": 1, "CP": 2, "SG": 3}
SEIZURE_TYPE_NAMES: "Tuple[str, ...]" = ("UC", "SP", "CP", "SG")

# EEG pattern codes
PATTERN_CODES: Dict[str, int] = {
    "t": 1, "a": 2, "d": 3, "l": 4, "b": 5,
    "s": 6, "r": 7, "m": 8, "c": 9, "p": 10, "e": 11,
}
PATTERN_NAMES: "Tuple[str, ...]" = (
    "", "t", "a", "d", "l", "b", "s", "r", "m", "c", "p", "e",
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BlockMeta:
    """Metadata from one .head file."""

    data_path: Path
    head_path: Path
    start_ts: datetime
    num_samples: int
    sample_freq: int
    conversion_factor: float
    num_channels: int
    elec_names: List[str]
    pat_id: str
    adm_id: str
    rec_id: str
    duration_s: float
    sample_bytes: int
    block_no: int  # extracted from filename

    @property
    def end_ts(self) -> datetime:
        return self.start_ts + timedelta(seconds=self.duration_s)


@dataclass
class ProcessedRecording:
    """Output of preprocessing a single recording (all blocks concatenated)."""

    recording_id: str
    subject_id: str
    variant: str
    signals: np.ndarray  # (T, 19) float32, 256 Hz
    samplewise_label: np.ndarray  # (T,) uint8  0=bckg 1=seiz 255=gap
    samplewise_type: np.ndarray  # (T,) uint8  seizure type code
    events: List[Dict[str, Any]]  # per-seizure event dicts
    duration_s: float
    native_fs: int
    n_missing_channels: int
    channel_mask: np.ndarray  # (19,) bool


@dataclass
class PatientMeta:
    """Discovery info for one patient."""

    pat_id: str
    variant: str
    patient_dir: Path
    recordings: Dict[str, List[BlockMeta]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Timestamp parser
# ---------------------------------------------------------------------------

_TS_FORMATS = [
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
]


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse .head start_ts with flexible format handling.

    Handles both zero-padded (2002-09-30 10:58:46.000) and
    non-padded (2010-2-3 8:40:14.000) timestamps.
    """
    ts_str = ts_str.strip()
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    try:
        parts = ts_str.replace("-", " ").replace(":", " ").replace(".", " ").split()
        if len(parts) >= 6:
            return datetime(
                int(parts[0]), int(parts[1]), int(parts[2]),
                int(parts[3]), int(parts[4]), int(parts[5]),
                int(parts[6]) if len(parts) > 6 else 0,
            )
        elif len(parts) >= 5:
            return datetime(
                int(parts[0]), int(parts[1]), int(parts[2]),
                int(parts[3]), int(parts[4]),
            )
    except (ValueError, IndexError):
        pass
    raise ValueError(f"Cannot parse timestamp: {ts_str!r}")


# ---------------------------------------------------------------------------
# Binary reader
# ---------------------------------------------------------------------------


def read_head(head_path: Path) -> Dict[str, Any]:
    """Parse a .head key=value file into a metadata dict.

    Returns dict with keys: ``start_ts`` (datetime), ``num_samples`` (int),
    ``sample_freq`` (int), ``conversion_factor`` (float), ``num_channels`` (int),
    ``elec_names`` (list[str]), ``pat_id``, ``adm_id``, ``rec_id`` (str),
    ``duration_in_sec`` (float), ``sample_bytes`` (int).
    """
    meta: Dict[str, str] = {}
    with open(head_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            meta[key.strip()] = value.strip()

    elec_str = meta.get("elec_names", "[]")
    elec_names = [n.strip() for n in elec_str.strip("[]").split(",") if n.strip()]

    sample_bytes = int(meta.get("sample_bytes", "2"))
    num_samples = int(meta["num_samples"])
    sample_freq = int(meta["sample_freq"])
    num_channels = int(meta["num_channels"])

    dur_str = meta.get("duration_in_sec", "")
    if dur_str:
        try:
            duration_s = float(dur_str)
        except ValueError:
            duration_s = num_samples / sample_freq
    else:
        duration_s = num_samples / sample_freq

    return {
        "start_ts": _parse_timestamp(meta["start_ts"]),
        "num_samples": num_samples,
        "sample_freq": sample_freq,
        "conversion_factor": float(meta.get("conversion_factor", "1.0")),
        "num_channels": num_channels,
        "elec_names": elec_names,
        "pat_id": meta.get("pat_id", ""),
        "adm_id": meta.get("adm_id", ""),
        "rec_id": meta.get("rec_id", ""),
        "duration_in_sec": duration_s,
        "sample_bytes": sample_bytes,
    }


def load_block_signals(data_path: Path, head: Dict[str, Any]) -> np.ndarray:
    """Read a .data binary file and return float32 signals in microvolts.

    Returns array of shape ``(num_channels, num_samples)``.
    """
    sample_bytes = head["sample_bytes"]
    num_channels = head["num_channels"]
    num_samples = head["num_samples"]
    conversion_factor = head["conversion_factor"]

    dtype = np.dtype(f"<i{sample_bytes}")
    expected_bytes = num_samples * num_channels * sample_bytes
    file_size = data_path.stat().st_size

    if file_size < expected_bytes:
        actual_elements = file_size // sample_bytes
        num_samples = actual_elements // num_channels
        logger.warning(
            "Truncated .data file %s: expected %d bytes, got %d. "
            "Adjusting num_samples to %d.",
            data_path, expected_bytes, file_size, num_samples,
        )

    raw = np.fromfile(str(data_path), dtype=dtype, count=num_samples * num_channels)
    signals = raw.reshape(num_samples, num_channels).T  # (num_channels, num_samples)
    signals = signals.astype(np.float32) * conversion_factor

    if os.environ.get(_AUTO_RESCALE_ENV) == "1":
        signals = _maybe_rescale_units(signals, data_path, head)
    return signals


def _maybe_rescale_units(
    signals: np.ndarray, data_path: Path, head: Dict[str, Any]
) -> np.ndarray:
    """Rescale if abs_max is far below the µV band — opt-in via env.

    Iterative rescale (up to 3 passes) so that multi-order-magnitude unit
    errors (V, mV, extreme miscalibrations) all land in band:

    - ``abs_max < 1e-4`` µV → apply ×1e6 (V → µV)
    - ``1e-4 ≤ abs_max < 0.5`` µV → apply ×1e3 (mV → µV)
    - ``abs_max ≥ 0.5`` µV → in band, stop

    For extreme cases (e.g. 1e-8), pass 1 (×1e6) lands at 1e-2 which is still
    below 0.5, so pass 2 (×1e3) lands at 10 — in band. Zeros and clipping
    (> cap) are left alone; downstream `assert_amplitude_band` handles them.
    """
    abs_max_pre = float(np.abs(signals).max()) if signals.size else 0.0
    if abs_max_pre == 0.0 or abs_max_pre >= _RESCALE_MV_TO_UV_THRESHOLD:
        return signals

    cumulative_factor = 1.0
    for _ in range(3):
        current = float(np.abs(signals).max())
        if current >= _RESCALE_MV_TO_UV_THRESHOLD:
            break
        if current < _RESCALE_V_TO_UV_THRESHOLD:
            step = 1e6
        else:
            step = 1e3
        signals = signals * np.float32(step)
        cumulative_factor *= step

    abs_max_post = float(np.abs(signals).max())
    logger.warning(
        "[epilepsiae_auto_rescale] %s  pat=%s  rec=%s  "
        "orig_conv_factor=%.6g  abs_max_pre=%.3g µV  cumulative_factor=%.3g  "
        "abs_max_post=%.3g µV  data=%s",
        _AUTO_RESCALE_ENV,
        head.get("pat_id", "?"),
        head.get("rec_id", "?"),
        float(head.get("conversion_factor", 1.0)),
        abs_max_pre,
        cumulative_factor,
        abs_max_post,
        data_path,
    )
    return signals


def discover_block(head_path: Path) -> BlockMeta:
    """Parse a .head file and return a :class:`BlockMeta`."""
    head = read_head(head_path)
    data_path = head_path.with_suffix(".data")

    stem = head_path.stem  # e.g. "100102_0075"
    parts = stem.rsplit("_", 1)
    block_no = int(parts[1]) if len(parts) == 2 else 0

    return BlockMeta(
        data_path=data_path,
        head_path=head_path,
        start_ts=head["start_ts"],
        num_samples=head["num_samples"],
        sample_freq=head["sample_freq"],
        conversion_factor=head["conversion_factor"],
        num_channels=head["num_channels"],
        elec_names=head["elec_names"],
        pat_id=str(head["pat_id"]),
        adm_id=str(head["adm_id"]),
        rec_id=str(head["rec_id"]),
        duration_s=head["duration_in_sec"],
        sample_bytes=head["sample_bytes"],
        block_no=block_no,
    )


# ---------------------------------------------------------------------------
# Channel normaliser
# ---------------------------------------------------------------------------


def normalize_channel_name(name: str) -> str:
    """Normalise a raw EPILEPSIAE channel name to canonical uppercase form.

    Strips trailing +/- artefacts (surfCO uses ECG+, EOG+, etc.),
    applies the EPILEPSIAE alias map, then falls back to uppercase.
    """
    name = name.strip().rstrip("+-")
    if name in CHANNEL_ALIAS:
        return CHANNEL_ALIAS[name]
    upper = name.upper()
    if upper in CANONICAL_SET:
        return upper
    return upper


def _is_non_eeg(name: str) -> bool:
    """Return True if the channel name is a known non-EEG signal."""
    upper = name.upper().rstrip("+-").rstrip("0123456789")
    for prefix in NON_EEG_PREFIXES:
        if upper.startswith(prefix.upper()):
            return True
    return False


def select_canonical_channels(
    signals: np.ndarray,
    source_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Select and reorder channels to the canonical 19-channel layout.

    Args:
        signals: ``(num_channels, num_samples)`` raw signals.
        source_names: Channel names from the .head file.

    Returns:
        Tuple of:

        - ``signals_19``: ``(19, num_samples)`` float32, zero-filled for missing.
        - ``channel_mask``: ``(19,)`` bool, True where the channel was present.
    """
    n_samples = signals.shape[1]
    out = np.zeros((19, n_samples), dtype=np.float32)
    mask = np.zeros(19, dtype=bool)

    for ch_idx, raw_name in enumerate(source_names):
        canonical = normalize_channel_name(raw_name)
        if canonical in CANONICAL_IDX:
            target_idx = CANONICAL_IDX[canonical]
            if not mask[target_idx]:  # first match wins
                out[target_idx] = signals[ch_idx]
                mask[target_idx] = True

    return out, mask
