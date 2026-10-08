"""BIDS-direct windowed Dataset for CHB-MIT.

Reads EDF files lazily from the BIDS directory with a per-recording LRU
signal cache.  Seizure labels are computed on-the-fly from the events TSV
intervals stored in the :class:`BIDSRecordingIndex`.  No HDF5 preprocessing
required.

The CHB-MIT BIDS data is 18-channel bipolar (double-banana) at 256 Hz.
"""
from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from neuroatlas.extensions.datasets.dataio._bids_participants import (
    load_bids_participants,
)
from neuroatlas.extensions.datasets.epilepsy.bids_index import (
    BIDSRecordingIndex,
)

logger = logging.getLogger(__name__)

TARGET_FS = 256

# CHB-MIT BIDS uses an 18-channel double-banana bipolar montage (SzCORE spec).
# Emitted in meta["channels"] so layout-aware backbones (REVE, future models)
# can route channels by name instead of silently zero-padding.
from neuroatlas.extensions.datasets.dataio._edf_units import (
    edf_unit_to_uv_scale as _edf_unit_to_uv_scale,
)


from neuroatlas.extensions.datasets._recording_stats import (
    compute_recording_stats,
    load_cached_recording_stats,
    load_recording_stats,
    save_recording_stats,
)
from pathlib import Path as _Path

# Per-recording stats live in <cache root>/recording_stats/chbmit_bids/
# (_paths.recording_stats_dir); they used to go to
# artifacts/recording_stats/chbmit_bids under the current directory.
_STATS_READER = "chbmit_bids"

# Older 10-20 form (T3/T4/T5/T6) — same physical electrodes as the 1991-
# modified T7/T8/P7/P8 names; aligns with BIOT canonical 16-pair vocabulary
# and TUSZ/Siena/Epilepsiae bipolar montages. See chbmit_preprocessor.BIPOLAR_18.
CHBMIT_BIPOLAR_18 = (
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FZ-CZ", "CZ-PZ",
)


class CHBMITBIDSDataset(Dataset):
    """Windowed view over CHB-MIT BIDS EDF files with lazy reading.

    Args:
        bids_root: Path to BIDS root directory containing ``sub-*`` folders.
        window_s: Window duration in seconds.
        stride_s: Stride in seconds (default: same as window_s).
        recording_indices: Subset of recording indices for k-fold filtering.
        label_mode: ``"binary"`` for seizure detection.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        overlap_threshold: Seizure fraction above which binary label is 1.
        signal_cache_size: Max recordings to keep in the LRU signal cache.
        index_cache_path: Optional path to a pickle cache for the BIDS index.
    """

    def __init__(
        self,
        bids_root: str,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        recording_indices: Optional[Sequence[int]] = None,
        label_mode: str = "binary",
        normalize: str = "none",
        overlap_threshold: float = 0.0,
        signal_cache_size: int = 8,
        index_cache_path: Optional[str] = None,
        # Accepted for interface compat, not used (already bipolar)
        montage: str = "bipolar",
        **kwargs,
    ) -> None:
        self._bids_root = str(bids_root)
        self._window_s = window_s
        self._stride_s = stride_s if stride_s is not None else window_s
        self._label_mode = label_mode
        self._normalize = normalize
        self._overlap_threshold = overlap_threshold
        self._signal_cache_size = signal_cache_size

        self._window_samples = int(round(window_s * TARGET_FS))
        self._stride_samples = int(round(self._stride_s * TARGET_FS))

        # Build BIDS index (fast — reads only JSON + TSV)
        self._index_obj = BIDSRecordingIndex.load_or_build(
            bids_root, cache_path=index_cache_path,
        )
        self._participants = load_bids_participants(bids_root)

        # Filter recordings
        all_recs = self._index_obj.recordings
        if recording_indices is not None:
            rec_set = set(recording_indices)
        else:
            rec_set = set(range(len(all_recs)))

        # Build window index: List[(rec_index, window_start_sample)]
        windows: List[Tuple[int, int]] = []
        for rec in all_recs:
            if rec.rec_index not in rec_set:
                continue
            if rec.n_samples < self._window_samples:
                continue
            pos = 0
            while pos + self._window_samples <= rec.n_samples:
                windows.append((rec.rec_index, pos))
                pos += self._stride_samples

        self._windows = windows
        self._active_rec_indices = sorted(rec_set & set(range(len(all_recs))))

        # LRU signal cache (populated lazily per worker)
        self._signal_cache: collections.OrderedDict[int, np.ndarray] = collections.OrderedDict()
        # Lazy per-recording amplitude-stats cache (MODEL_CONTRACTS §3).
        self._rec_stats_cache: Dict[int, Dict[str, Any]] = {}

        logger.info(
            "CHBMITBIDSDataset: %s — %d recordings, %d windows "
            "(%.1fs @ %.1fs stride)",
            self._bids_root, len(self._active_rec_indices),
            len(self._windows), self._window_s, self._stride_s,
        )

    def _load_signals(self, rec_index: int) -> np.ndarray:
        """Load all channels from an EDF, return (n_ch, n_samples) float32.

        Applies 0.5 Hz HP + 60 Hz notch (CHB-MIT is US recording) per the
        project-wide filter rule. Uses an LRU cache to avoid re-reading and
        re-filtering the same file.
        """
        if rec_index in self._signal_cache:
            self._signal_cache.move_to_end(rec_index)
            return self._signal_cache[rec_index]

        import pyedflib

        from neuroatlas.extensions.datasets.epilepsy._common import (
            apply_standard_filters,
        )

        rec = self._index_obj.recordings[rec_index]
        edf = pyedflib.EdfReader(rec.edf_path)
        try:
            n_ch = edf.signals_in_file
            n_samples = edf.getNSamples()[0]
            ch_fs = edf.getSampleFrequency(0) if n_ch > 0 else 256.0
            signals = np.zeros((n_ch, n_samples), dtype=np.float32)
            for i in range(n_ch):
                sig = edf.readSignal(i)
                # Read declared unit from EDF header. CHB-MIT EDFs all declare "uV".
                unit = edf.getPhysicalDimension(i)
                if isinstance(unit, bytes):
                    unit = unit.decode(errors="ignore")
                unit = unit.strip()
                scale = _edf_unit_to_uv_scale(unit)
                if isinstance(sig, np.ndarray):
                    signals[i, :len(sig)] = (sig * scale).astype(np.float32)
        finally:
            edf.close()

        if signals.shape[1] > 20:
            signals = apply_standard_filters(
                signals, fs=float(ch_fs), notch_hz=60.0, axis=1,
            )

        # Evict oldest if cache is full
        while len(self._signal_cache) >= self._signal_cache_size:
            self._signal_cache.popitem(last=False)

        self._signal_cache[rec_index] = signals
        return signals

    def _compute_seizure_label(
        self, rec_index: int, window_start: int, window_end: int,
    ) -> Tuple[int, float]:
        """Compute binary label and seizure fraction from interval overlap."""
        rec = self._index_obj.recordings[rec_index]
        window_len = window_end - window_start
        overlap = 0
        for sz_start, sz_end in rec.seizure_intervals_samples:
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
        rec = self._index_obj.recordings[rec_index]

        # Load signals (from cache or EDF)
        all_signals = self._load_signals(rec_index)
        signals = all_signals[:, window_start:window_end].copy()  # (n_ch, T)

        # Normalize
        if self._normalize == "per_window_zscore":
            mean = signals.mean(axis=1, keepdims=True)
            std = signals.std(axis=1, keepdims=True)
            std[std < 1e-8] = 1.0
            signals = (signals - mean) / std

        # Label
        label, seizure_fraction = self._compute_seizure_label(
            rec_index, window_start, window_end,
        )

        demo = self._participants.get(rec.subject_id, {})
        meta: Dict[str, Any] = {
            "dataset": "chbmit",
            "split": "",
            "subject_id": rec.subject_id,
            "recording_id": rec.recording_id,
            "recording_idx": rec_index,
            "window_start_s": float(window_start / rec.fs),
            "window_end_s": float(window_end / rec.fs),
            "recording_duration_s": rec.duration_s,
            "seizure_fraction": seizure_fraction,
            "has_seizure": bool(label),
            "binary_label": label,
            "channels": list(CHBMIT_BIPOLAR_18),
            "montage": "bipolar",
            "n_channels": rec.n_channels,
            "fs": rec.fs,
            "sampling_rate": float(rec.fs),
            "units": "uV",
            "unit": "uV",
            "window_s": self._window_s,
            "stride_s": self._stride_s,
            "age": int(demo.get("age", -1)),
            "gender": demo.get("gender", ""),
            "comment": demo.get("comment", ""),
        }

        # Lazy per-recording stats — disk-backed; compute once across all models.
        if rec_index not in self._rec_stats_cache:
            stats_file = f"{rec.recording_id}.npz"
            fingerprint = {
                "target_fs": int(TARGET_FS),
                "n_samples": int(all_signals.shape[1]),
            }
            stats, disk_path = load_cached_recording_stats(_STATS_READER, stats_file, fingerprint)
            if stats is None:
                stats = compute_recording_stats(all_signals)
                try:
                    save_recording_stats(disk_path, stats, fingerprint)
                except Exception:
                    pass
            self._rec_stats_cache[rec_index] = stats
        meta.update(self._rec_stats_cache[rec_index])

        return {
            "eeg": torch.from_numpy(signals),
            "label": label,
            "meta": meta,
        }

    # -- Accessors ----------------------------------------------------------

    @property
    def subject_ids(self) -> List[str]:
        return list({
            self._index_obj.recordings[i].subject_id
            for i in self._active_rec_indices
        })

    @property
    def n_recordings(self) -> int:
        return len(self._active_rec_indices)

    def binary_labels(self) -> np.ndarray:
        """Compute binary labels for all windows without reading signals."""
        labels = np.zeros(len(self._windows), dtype=np.int64)
        for i, (rec_index, window_start) in enumerate(self._windows):
            window_end = window_start + self._window_samples
            label, _ = self._compute_seizure_label(rec_index, window_start, window_end)
            labels[i] = label
        return labels

    def recording_sizes_bytes(self) -> List[Tuple[int, int]]:
        """Return ``(rec_index, estimated_float32_bytes)`` for active recordings.

        Uses header-derived counts; no EDF decode is triggered.
        """
        return [
            (i, self._index_obj.recordings[i].n_channels
                * self._index_obj.recordings[i].n_samples * 4)
            for i in self._active_rec_indices
        ]

    def preload_shard(self, rec_indices: Sequence[int]) -> None:
        """Eagerly decode recordings into the signal cache before model passes.

        Expands ``_signal_cache_size`` to hold the entire shard so no
        recordings are evicted during embedding.  Logs timing and RSS delta.
        """
        import resource
        import time

        n = len(rec_indices)
        if not n:
            return
        self._signal_cache_size = max(self._signal_cache_size, n)

        from neuroatlas import progress

        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()
        cold = 0
        # on the item's live line, not a bar under it
        item = progress.current()
        item.phase("reading the recordings into memory", total=n, unit="recordings")
        for i in rec_indices:
            if i not in self._signal_cache:
                cold += 1
            self._load_signals(i)
            item.update(advance=1)

        elapsed = time.perf_counter() - t0
        rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logger.info(
            "preload_shard: %d recordings (%d cold decodes), %.2fs, RSS +%.1f MB",
            n, cold, elapsed, (rss1 - rss0) / 1024.0,
        )


# ---------------------------------------------------------------------------
# Collate (same as HDF5 version)
# ---------------------------------------------------------------------------


def _collate_chbmit(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    eeg = torch.stack([item["eeg"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "eeg": eeg,
        "labels": labels,
        "meta": [item["meta"] for item in batch],
    }
