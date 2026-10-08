"""Direct EDF-backend windowed Dataset for SeizeIt1.

Reads raw ``.edf`` + ``_a1.tsv`` files lazily with a per-recording LRU signal
cache.  Seizure labels are computed on-the-fly from the TSV intervals.  No
HDF5 preprocessing required.

Adds explicit ``meta["unit"] = "uV"`` and ``meta["sampling_rate"] = 256``
per MODEL_CONTRACTS §0.
"""
from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from neuroatlas.extensions.datasets._recording_stats import compute_recording_stats
from neuroatlas.extensions.datasets.epilepsy.sz1_preprocessor import (
    BIPOLAR_NAMES,
    CANONICAL_19,
    TARGET_FS,
    SZ1_SKIP_KEYWORDS,
    _parse_sz1_tsv,
    _read_sz1_edf_to_unipolar19,
    bipolar_from_unipolar,
)

logger = logging.getLogger(__name__)


@dataclass
class _RecordingMeta:
    """Lightweight metadata for one SeizeIt1 recording."""

    rec_index: int
    edf_path: str
    tsv_path: str
    subject_id: str
    n_samples: int
    fs: int
    duration_s: float
    seizure_intervals_samples: List[Tuple[int, int]]


def _build_recording_meta(
    recordings: List[Tuple[str, str, str]],
    recording_indices: Optional[Sequence[int]] = None,
) -> List[_RecordingMeta]:
    """Build metadata for each recording by peeking at the EDF header."""
    rec_set = (
        set(recording_indices)
        if recording_indices is not None
        else set(range(len(recordings)))
    )

    metas: List[_RecordingMeta] = []
    skipped: List[Tuple[str, str]] = []
    # one header read per recording: the item's live line counts them
    from neuroatlas import progress, quiet

    item = progress.current()
    item.phase("indexing windows", total=len(rec_set & set(range(len(recordings)))),
               unit="recordings")
    for global_idx, (edf_path_s, tsv_path_s, subject_id) in enumerate(recordings):
        if global_idx not in rec_set:
            continue
        item.update(advance=1)
        edf_path = Path(edf_path_s)
        tsv_path = Path(tsv_path_s)

        seizures_s = _parse_sz1_tsv(tsv_path) if tsv_path.exists() else []

        try:
            import edfio
            edf = edfio.read_edf(str(edf_path), lazy_load_data=True)
            first_sig = next(
                (s for s in edf.signals if "eeg" in str(s.label).lower()),
                edf.signals[0] if edf.signals else None,
            )
            if first_sig is None:
                raise ValueError("No signals found in EDF")
            fs = int(round(float(first_sig.sampling_frequency)))
            n_samples_raw = edf.num_data_records * first_sig.samples_per_data_record
            if fs != TARGET_FS:
                n_samples = int(round(n_samples_raw * TARGET_FS / fs))
            else:
                n_samples = n_samples_raw
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.info("skipping %s: %s", edf_path, reason)
            skipped.append((str(edf_path), reason))
            continue

        seizure_intervals: List[Tuple[int, int]] = [
            (
                max(0, int(round(s * TARGET_FS))),
                min(n_samples, int(round(e * TARGET_FS))),
            )
            for s, e in seizures_s
        ]

        metas.append(
            _RecordingMeta(
                rec_index=global_idx,
                edf_path=str(edf_path),
                tsv_path=str(tsv_path),
                subject_id=subject_id,
                n_samples=n_samples,
                fs=TARGET_FS,
                duration_s=n_samples / TARGET_FS,
                seizure_intervals_samples=seizure_intervals,
            )
        )

    # one count per dataset at the end of the run (each split and fold reads
    # the headers again); which ones, and why, in the log
    quiet.count("skipped headers:sz1",
                "sz1: {hit} of {of} recordings left out: their EDF header cannot be "
                "read (-v names them, with the reason)",
                hit=[p for p, _ in skipped],
                of=[str(r[0]) for i, r in enumerate(recordings) if i in rec_set])
    if skipped:
        logger.info(
            "Skipped %d / %d recordings due to unreadable EDF headers:\n%s",
            len(skipped),
            len(skipped) + len(metas),
            "\n".join(f"  {p}  [{r}]" for p, r in skipped),
        )
    else:
        logger.info("All %d recordings read successfully.", len(metas))

    return metas


class SeizeIt1EDFDataset(Dataset):
    """Windowed view over SeizeIt1 raw EDF + ``_a1.tsv`` files with lazy reading.

    Args:
        recordings: List of ``(edf_path, tsv_path, subject_id)`` tuples as
            returned by ``discover_recordings``.
        window_s: Window duration in seconds.
        stride_s: Stride in seconds (default: same as window_s).
        recording_indices: Subset of recording indices for k-fold filtering.
        label_mode: ``"binary"`` for seizure detection.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        montage: ``"unipolar"`` (19 ch) or ``"bipolar"`` (20 ch TCP).
        overlap_threshold: Seizure fraction above which binary label is 1.
        signal_cache_size: Max recordings to keep in the LRU signal cache.
    """

    def __init__(
        self,
        recordings: List[Tuple[str, str, str]],
        window_s: float = 10.0,
        stride_s: Optional[float] = None,
        recording_indices: Optional[Sequence[int]] = None,
        label_mode: str = "binary",
        normalize: str = "none",
        montage: str = "unipolar",
        overlap_threshold: float = 0.0,
        signal_cache_size: int = 14,
    ) -> None:
        self._recordings = recordings
        self._window_s = window_s
        self._stride_s = stride_s if stride_s is not None else window_s
        self._label_mode = label_mode
        self._normalize = normalize
        self._montage = montage
        self._overlap_threshold = overlap_threshold
        self._signal_cache_size = signal_cache_size

        self._window_samples = int(round(window_s * TARGET_FS))
        self._stride_samples = int(round(self._stride_s * TARGET_FS))

        self._rec_metas = _build_recording_meta(recordings, recording_indices)
        self._meta_by_index: Dict[int, _RecordingMeta] = {
            m.rec_index: m for m in self._rec_metas
        }

        windows: List[Tuple[int, int]] = []
        for meta in self._rec_metas:
            if meta.n_samples < self._window_samples:
                continue
            pos = 0
            while pos + self._window_samples <= meta.n_samples:
                windows.append((meta.rec_index, pos))
                pos += self._stride_samples
        self._windows = windows

        self._active_rec_indices = [m.rec_index for m in self._rec_metas]

        self._signal_cache: collections.OrderedDict[int, np.ndarray] = (
            collections.OrderedDict()
        )
        # Lazy per-recording amplitude-stats cache (MODEL_CONTRACTS §3).
        # Computed on the unipolar source before any bipolar derivation.
        self._rec_stats_cache: Dict[int, Dict[str, Any]] = {}

        logger.info(
            "SeizeIt1EDFDataset: %d recordings, %d windows "
            "(%.1fs @ %.1fs stride, montage=%s)",
            len(self._rec_metas),
            len(self._windows),
            self._window_s,
            self._stride_s,
            self._montage,
        )

    def _load_signals(self, rec_index: int) -> np.ndarray:
        """Load and preprocess EDF signals for one recording (LRU cached)."""
        if rec_index in self._signal_cache:
            self._signal_cache.move_to_end(rec_index)
            return self._signal_cache[rec_index]

        meta = self._meta_by_index[rec_index]

        signals, _, _ = _read_sz1_edf_to_unipolar19(Path(meta.edf_path))
        n = meta.n_samples
        if signals.shape[1] > n:
            signals = signals[:, :n]
        elif signals.shape[1] < n:
            pad = np.zeros((19, n - signals.shape[1]), dtype=np.float32)
            signals = np.concatenate([signals, pad], axis=1)

        while len(self._signal_cache) >= self._signal_cache_size:
            self._signal_cache.popitem(last=False)

        self._signal_cache[rec_index] = signals
        return signals

    def _compute_seizure_label(
        self, rec_index: int, window_start: int, window_end: int,
    ) -> Tuple[int, float]:
        meta = self._meta_by_index[rec_index]
        window_len = window_end - window_start
        overlap = 0
        for sz_start, sz_end in meta.seizure_intervals_samples:
            o = max(0, min(window_end, sz_end) - max(window_start, sz_start))
            overlap += o
        fraction = overlap / max(window_len, 1)
        label = int(fraction > self._overlap_threshold)
        return label, fraction

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_index, window_start = self._windows[idx]
        window_end = window_start + self._window_samples
        meta = self._meta_by_index[rec_index]

        all_signals = self._load_signals(rec_index)
        signals = all_signals[:, window_start:window_end].copy()  # (19, T)

        # Lazily compute and cache per-recording stats on the full unipolar
        # source (before any bipolar derivation) — MODEL_CONTRACTS §3.
        if rec_index not in self._rec_stats_cache:
            self._rec_stats_cache[rec_index] = compute_recording_stats(all_signals)

        if self._montage == "bipolar":
            signals = bipolar_from_unipolar(signals)  # (20, T)

        if self._normalize == "per_window_zscore":
            mean = signals.mean(axis=1, keepdims=True)
            std = signals.std(axis=1, keepdims=True)
            std[std < 1e-8] = 1.0
            signals = (signals - mean) / std

        label, seizure_fraction = self._compute_seizure_label(
            rec_index, window_start, window_end
        )

        ch_names = list(BIPOLAR_NAMES if self._montage == "bipolar" else CANONICAL_19)
        item_meta: Dict[str, Any] = {
            "dataset": "sz1",
            "split": "",
            "subject_id": meta.subject_id,
            "recording_id": Path(meta.edf_path).name,
            "recording_idx": rec_index,
            "window_start_s": float(window_start / TARGET_FS),
            "window_end_s": float(window_end / TARGET_FS),
            "recording_duration_s": meta.duration_s,
            "seizure_fraction": seizure_fraction,
            "has_seizure": bool(label),
            "binary_label": label,
            "n_channels": signals.shape[0],
            "channels": ch_names,
            "montage": self._montage,
            "normalize": self._normalize,
            # MODEL_CONTRACTS §0: explicit unit declaration
            "unit": "uV",
            "sampling_rate": TARGET_FS,
            "fs": TARGET_FS,
            "window_s": self._window_s,
            "stride_s": self._stride_s,
        }
        item_meta.update(self._rec_stats_cache[rec_index])

        return {
            "eeg": torch.from_numpy(signals),
            "label": label,
            "meta": item_meta,
        }

    @property
    def subject_ids(self) -> List[str]:
        return [m.subject_id for m in self._rec_metas]

    @property
    def n_recordings(self) -> int:
        return len(self._rec_metas)

    def binary_labels(self) -> np.ndarray:
        labels = np.zeros(len(self._windows), dtype=np.int64)
        for i, (rec_index, window_start) in enumerate(self._windows):
            window_end = window_start + self._window_samples
            label, _ = self._compute_seizure_label(rec_index, window_start, window_end)
            labels[i] = label
        return labels


# ---------------------------------------------------------------------------
# Synthetic recordings list (fallback for edf backend when path is unset)
# ---------------------------------------------------------------------------


def _synthetic_recordings_list() -> List[Tuple[str, str, str]]:
    """Return 2 dummy recording entries for testing without real data."""
    return [
        ("SYNTHETIC_EDF_0.edf", "SYNTHETIC_EDF_0_a1.tsv", "SYNTHETIC_SUBJ_0"),
        ("SYNTHETIC_EDF_1.edf", "SYNTHETIC_EDF_1_a1.tsv", "SYNTHETIC_SUBJ_1"),
    ]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def _collate_sz1_edf(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for SeizeIt1EDFDataset batches."""
    eeg = torch.stack([item["eeg"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "eeg": eeg,
        "labels": labels,
        "meta": [item["meta"] for item in batch],
    }
