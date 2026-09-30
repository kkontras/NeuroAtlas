"""TUSZ annotation parsers and samplewise label builders.

TUSZ ships two annotation flavours per EDF:

- ``<stem>.csv_bi`` — binary labels on the TERM (whole-recording) channel.
- ``<stem>.csv`` — per-channel multi-class seizure type labels.

Both are simple comma-separated text files. This module parses them and
converts event lists into samplewise arrays aligned to the target sampling
rate, plus flat per-event arrays suitable for HDF5 storage.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Seizure taxonomy (TUSZ-specific: 13-class)
# ---------------------------------------------------------------------------

SEIZURE_TYPE_CODES: Dict[str, int] = {
    "bckg": 0,
    "seiz": 1,   # generic, only used when no specific type overlaps
    "fnsz": 2,   # focal non-specific seizure
    "gnsz": 3,   # generalised non-specific seizure
    "spsz": 4,   # simple partial
    "cpsz": 5,   # complex partial
    "absz": 6,   # absence
    "tnsz": 7,   # tonic
    "cnsz": 8,   # clonic
    "tcsz": 9,   # tonic-clonic
    "atsz": 10,  # atonic
    "mysz": 11,  # myoclonic
    "nesz": 12,  # non-epileptic seizure (psychogenic)
}

SEIZURE_TYPE_NAMES: Tuple[str, ...] = tuple(
    k for k, _ in sorted(SEIZURE_TYPE_CODES.items(), key=lambda kv: kv[1])
)


# ---------------------------------------------------------------------------
# CSV parsers
# ---------------------------------------------------------------------------


def parse_csv_bi(csv_bi_path: str) -> List[Dict[str, Any]]:
    """Parse a ``.csv_bi`` annotation file (binary seizure labels on TERM channel).

    Args:
        csv_bi_path: Path to the ``.csv_bi`` file.

    Returns:
        List of event dicts with keys ``channel``, ``start``, ``stop``,
        ``label`` (``'seiz'`` or ``'bckg'``), ``confidence``.
    """
    events: List[Dict[str, Any]] = []
    with open(csv_bi_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("channel,"):
                continue  # header row
            parts = line.split(",")
            if len(parts) < 4:
                continue
            channel = parts[0].strip()
            start = float(parts[1])
            stop = float(parts[2])
            label = parts[3].strip().lower()
            confidence = float(parts[4]) if len(parts) > 4 else 1.0
            events.append({
                "channel": channel,
                "start": start,
                "stop": stop,
                "label": label,
                "confidence": confidence,
            })
    return events


def parse_csv_multiclass(csv_path: str) -> List[Dict[str, Any]]:
    """Parse a ``.csv`` annotation file (per-channel multi-class seizure types).

    Same format as ``parse_csv_bi`` but the ``label`` field can be any of
    the 13 TUSZ seizure types (see ``SEIZURE_TYPE_CODES``).

    Args:
        csv_path: Path to the ``.csv`` file.

    Returns:
        List of event dicts with keys ``channel``, ``start``, ``stop``,
        ``label``, ``confidence``.
    """
    events: List[Dict[str, Any]] = []
    with open(csv_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("channel,"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            channel = parts[0].strip()
            start = float(parts[1])
            stop = float(parts[2])
            label = parts[3].strip().lower()
            confidence = float(parts[4]) if len(parts) > 4 else 1.0
            events.append({
                "channel": channel,
                "start": start,
                "stop": stop,
                "label": label,
                "confidence": confidence,
            })
    return events


# ---------------------------------------------------------------------------
# Samplewise label builders
# ---------------------------------------------------------------------------


def build_samplewise_labels(
    n_samples: int,
    events: List[Dict[str, Any]],
    fs: int = 256,
) -> np.ndarray:
    """Build sample-wise binary labels from ``parse_csv_bi`` output.

    Args:
        n_samples: Length of the samplewise output array.
        events: Event dicts from :func:`parse_csv_bi`.
        fs: Sampling rate of the target array.

    Returns:
        ``(n_samples,)`` ``uint8`` array, 0 = background, 1 = seizure.
        No forward-fill: only annotated intervals are labelled as seizure.
    """
    labels = np.zeros(n_samples, dtype=np.uint8)
    for ev in events:
        if ev["label"] in ("seiz", "seizure"):
            s0 = max(0, int(round(ev["start"] * fs)))
            s1 = min(n_samples, int(round(ev["stop"] * fs)))
            labels[s0:s1] = 1
    return labels


def build_samplewise_types(
    n_samples: int,
    events: List[Dict[str, Any]],
    fs: int = 256,
) -> np.ndarray:
    """Build sample-wise multi-class seizure type labels.

    Specificity-wins rule: a more specific type (``fnsz``, ``gnsz``, ...)
    overwrites the generic ``seiz`` code at overlapping samples.  Background
    stays 0.

    Args:
        n_samples: Length of the samplewise output array.
        events: Event dicts from :func:`parse_csv_multiclass`.
        fs: Sampling rate of the target array.

    Returns:
        ``(n_samples,)`` ``uint8`` array with codes from ``SEIZURE_TYPE_CODES``.
    """
    labels = np.zeros(n_samples, dtype=np.uint8)
    # Sort events: background first, then generic seiz, then specific types
    # so specific types overwrite generic ones
    priority = {"bckg": 0, "seiz": 1}

    def _sort_key(ev: Dict[str, Any]) -> int:
        return priority.get(ev["label"], 2)

    sorted_events = sorted(events, key=_sort_key)
    for ev in sorted_events:
        code = SEIZURE_TYPE_CODES.get(ev["label"], None)
        if code is None or code == 0:
            continue  # skip background
        s0 = max(0, int(round(ev["start"] * fs)))
        s1 = min(n_samples, int(round(ev["stop"] * fs)))
        if code >= 2:
            # Specific type — always overwrites
            labels[s0:s1] = code
        elif code == 1:
            # Generic "seiz" — only write where still 0
            mask = labels[s0:s1] == 0
            labels[s0:s1][mask] = code
    return labels


# ---------------------------------------------------------------------------
# Per-event flat arrays
# ---------------------------------------------------------------------------


def build_event_arrays(
    events_multiclass: List[Dict[str, Any]],
    recording_idx: int,
    tcp_montage_channels: Tuple[str, ...],
) -> Dict[str, List[Any]]:
    """Build per-event flat arrays from multiclass CSV annotations.

    Args:
        events_multiclass: Event dicts from :func:`parse_csv_multiclass`.
        recording_idx: Index of the source recording in the merged cache.
        tcp_montage_channels: TCP channel names, used to map event channels
            to a small-int code (``-1`` for TERM / unknown).

    Returns:
        Dict of lists, one per event: ``event_start_s``, ``event_stop_s``,
        ``event_type``, ``event_channel_tcp``, ``event_recording_idx``.
    """
    tcp_idx = {ch: i for i, ch in enumerate(tcp_montage_channels)}
    starts: List[float] = []
    stops: List[float] = []
    types: List[int] = []
    channels: List[int] = []
    rec_idxs: List[int] = []
    for ev in events_multiclass:
        label = ev["label"]
        code = SEIZURE_TYPE_CODES.get(label, None)
        if code is None or code == 0:
            continue  # skip background
        starts.append(ev["start"])
        stops.append(ev["stop"])
        types.append(code)
        ch_name = ev["channel"]
        channels.append(tcp_idx.get(ch_name, -1))
        rec_idxs.append(recording_idx)
    return {
        "event_start_s": starts,
        "event_stop_s": stops,
        "event_type": types,
        "event_channel_tcp": channels,
        "event_recording_idx": rec_idxs,
    }
