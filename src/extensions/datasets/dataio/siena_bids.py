"""BIDS-direct windowed Dataset for Siena Scalp EEG.

Reads EDF files lazily from the BIDS directory with a per-recording LRU
signal cache.  Seizure labels are computed on-the-fly from the events TSV
intervals stored in the :class:`BIDSRecordingIndex`.  No HDF5 preprocessing
required.

The Siena BIDS data is 19-channel unipolar (10-20 common-average) at 256 Hz.
Supports on-the-fly bipolar montage conversion.
"""
from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from extensions.datasets._recording_stats import (
    compute_recording_stats,
    load_recording_stats,
    save_recording_stats,
)
from pathlib import Path as _Path

_SIENA_STATS_CACHE_ROOT = _Path("artifacts/recording_stats/siena_bids")
from extensions.datasets.dataio._bids_participants import (
    load_bids_participants,
)
from extensions.datasets.epilepsy.bids_index import (
    BIDSRecordingIndex,
)
from extensions.datasets.epilepsy.siena_preprocessor import (
    BIPOLAR_NAMES,
    CANONICAL_19,
    CANONICAL_IDX,
    CANONICAL_SET,
    CHANNEL_ALIAS,
    NON_EEG_PREFIXES,
    bipolar_from_unipolar,
)

logger = logging.getLogger(__name__)

TARGET_FS = 256


class SienaBIDSDataset(Dataset):
    """Windowed view over Siena BIDS EDF files with lazy reading.

    Args:
        bids_root: Path to BIDS root directory containing ``sub-*`` folders.
        window_s: Window duration in seconds.
        stride_s: Stride in seconds (default: same as window_s).
        recording_indices: Subset of recording indices for k-fold filtering.
        label_mode: ``"binary"`` for seizure detection.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        montage: ``"unipolar"`` (19 ch) or ``"bipolar"`` (18 ch).
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
        montage: str = "bipolar",
        overlap_threshold: float = 0.0,
        signal_cache_size: int = 8,
        index_cache_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._bids_root = str(bids_root)
        self._window_s = window_s
        self._stride_s = stride_s if stride_s is not None else window_s
        self._label_mode = label_mode
        self._normalize = normalize
        self._montage = montage
        self._overlap_threshold = overlap_threshold
        self._signal_cache_size = signal_cache_size

        self._window_samples = int(round(window_s * TARGET_FS))
        self._stride_samples = int(round(self._stride_s * TARGET_FS))

        # Build BIDS index
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

        # Build window index
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

        # LRU signal cache
        self._signal_cache: collections.OrderedDict[int, np.ndarray] = collections.OrderedDict()
        # Lazy per-recording amplitude-stats cache (MODEL_CONTRACTS §3),
        # computed on the unipolar source before any bipolar derivation.
        self._rec_stats_cache: Dict[int, Dict[str, Any]] = {}

        logger.info(
            "SienaBIDSDataset: %s — %d recordings, %d windows "
            "(%.1fs @ %.1fs stride, montage=%s)",
            self._bids_root, len(self._active_rec_indices),
            len(self._windows), self._window_s, self._stride_s, self._montage,
        )

    def _load_signals(self, rec_index: int) -> np.ndarray:
        """Load EDF and map to canonical 19-ch unipolar layout.

        Returns (19, n_samples) float32. Missing channels are zero-filled.
        Uses an LRU cache.
        """
        if rec_index in self._signal_cache:
            self._signal_cache.move_to_end(rec_index)
            return self._signal_cache[rec_index]

        import pyedflib

        rec = self._index_obj.recordings[rec_index]
        edf = pyedflib.EdfReader(rec.edf_path)
        try:
            n_ch = edf.signals_in_file
            ch_labels = [
                edf.signal_label(i).decode() if isinstance(edf.signal_label(i), bytes)
                else edf.signal_label(i)
                for i in range(n_ch)
            ]
            ch_labels = [s.strip() for s in ch_labels]

            # Map to canonical 19-channel layout
            ch_map: Dict[str, int] = {}
            for i, raw_label in enumerate(ch_labels):
                name = raw_label.replace("EEG ", "").replace("eeg ", "")
                name = name.split("-")[0].strip().upper()
                if any(name.startswith(p) for p in NON_EEG_PREFIXES):
                    continue
                canonical = CHANNEL_ALIAS.get(name)
                if canonical is None:
                    canonical = CHANNEL_ALIAS.get(raw_label.split("-")[0].strip())
                if canonical is not None and canonical in CANONICAL_SET:
                    if canonical not in ch_map:
                        ch_map[canonical] = i

            n_samples = edf.getNSamples()[0]
            signals = np.zeros((19, n_samples), dtype=np.float32)

            for canonical_name, edf_idx in ch_map.items():
                cix = CANONICAL_IDX[canonical_name]
                sig = edf.readSignal(edf_idx)
                # Read declared unit from EDF header. Siena EDFs all declare "uV".
                unit = edf.getPhysicalDimension(edf_idx)
                if isinstance(unit, bytes):
                    unit = unit.decode(errors="ignore")
                from extensions.datasets.dataio._edf_units import edf_unit_to_uv_scale
                scale = edf_unit_to_uv_scale(unit.strip())
                if isinstance(sig, np.ndarray):
                    length = min(len(sig), n_samples)
                    signals[cix, :length] = (sig[:length] * scale).astype(np.float32)

        finally:
            edf.close()

        # Evict oldest
        while len(self._signal_cache) >= self._signal_cache_size:
            self._signal_cache.popitem(last=False)

        self._signal_cache[rec_index] = signals
        return signals

    def _compute_seizure_label(
        self, rec_index: int, window_start: int, window_end: int,
    ) -> Tuple[int, float]:
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

        # Load signals (19-ch unipolar from cache or EDF)
        all_signals = self._load_signals(rec_index)
        signals = all_signals[:, window_start:window_end].copy()  # (19, T)

        # Montage conversion
        if self._montage == "bipolar":
            signals = bipolar_from_unipolar(signals)  # (18, T)

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

        n_ch = signals.shape[0]
        demo = self._participants.get(rec.subject_id, {})
        meta: Dict[str, Any] = {
            "dataset": "siena",
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
            "n_channels": n_ch,
            "channels": list(BIPOLAR_NAMES if self._montage == "bipolar" else CANONICAL_19),
            "montage": self._montage,
            "normalize": self._normalize,
            "fs": rec.fs,
            "sampling_rate": float(rec.fs),
            "units": "uV",
            "unit": "uV",
            "window_s": self._window_s,
            "stride_s": self._stride_s,
            "age": int(demo.get("age", -1)),
            "gender": demo.get("gender", ""),
        }

        # Lazy per-recording stats. When montage='bipolar', recompute mean/
        # std/q95 on the emitted (20, T) bipolar signal so SleepFM / REVE /
        # BIOT all see (C_emit,) stats matching meta["channels"]. Disk cache
        # keyed by montage so unipolar and bipolar variants both persist.
        if rec_index not in self._rec_stats_cache:
            disk_path = _SIENA_STATS_CACHE_ROOT / f"{rec.recording_id}_{self._montage}.npz"
            fingerprint = {
                "target_fs": int(TARGET_FS),
                "n_samples": int(all_signals.shape[1]),
                "montage": self._montage,
            }
            stats = load_recording_stats(disk_path, fingerprint)
            if stats is None:
                if self._montage == "bipolar":
                    bipolar_full = bipolar_from_unipolar(all_signals)  # (20, T)
                    stats = compute_recording_stats(bipolar_full)
                else:
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
        labels = np.zeros(len(self._windows), dtype=np.int64)
        for i, (rec_index, window_start) in enumerate(self._windows):
            window_end = window_start + self._window_samples
            label, _ = self._compute_seizure_label(rec_index, window_start, window_end)
            labels[i] = label
        return labels

    def recording_sizes_bytes(self) -> List[Tuple[int, int]]:
        """Return ``(rec_index, estimated_float32_bytes)`` for active recordings.

        Siena cache always holds 19-channel canonical layout; no EDF decode.
        """
        return [
            (i, 19 * self._index_obj.recordings[i].n_samples * 4)
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

        from tqdm.auto import tqdm

        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()
        cold = 0
        for i in tqdm(rec_indices, desc="preload Siena shard", unit="rec", leave=False):
            if i not in self._signal_cache:
                cold += 1
            self._load_signals(i)

        elapsed = time.perf_counter() - t0
        rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logger.info(
            "preload_shard: %d recordings (%d cold decodes), %.2fs, RSS +%.1f MB",
            n, cold, elapsed, (rss1 - rss0) / 1024.0,
        )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def _collate_siena(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    eeg = torch.stack([item["eeg"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "eeg": eeg,
        "labels": labels,
        "meta": [item["meta"] for item in batch],
    }
