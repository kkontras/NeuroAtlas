"""EPILEPSIAE on-the-fly dataset — reads raw .data/.head files directly.

No HDF5 cache required.  At construction the filesystem is scanned to
build a lightweight block index and seizure annotations are parsed.
On ``__getitem__`` the relevant block is loaded from disk, preprocessed
(channel selection → resample → filter), and the requested window is
extracted.  An LRU cache keeps recently-processed blocks in memory so
consecutive windows from the same block avoid redundant I/O.

The public interface (constructor args, ``__getitem__`` return dict,
convenience properties) is kept compatible with the adapter in
``adapters/epilepsiae.py``.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from neuroatlas.extensions.datasets._recording_stats import (
    compute_recording_stats,
    compute_recording_stats_streaming,
    load_cached_recording_stats,
    load_recording_stats,
    save_recording_stats,
)

# Per-recording stats cache root — one .npz per recording, shared across all
# models. Compute once, read by every subsequent run.
# Per-recording stats live in <cache root>/recording_stats/epilepsiae/
# (_paths.recording_stats_dir); they used to go to
# artifacts/recording_stats/epilepsiae under the current directory.
_STATS_READER = "epilepsiae"
from neuroatlas.extensions.datasets.epilepsy.epilepsiae_preprocessor import (
    BIPOLAR_NAMES,
    CANONICAL_19,
    CANONICAL_IDX,
    GAP_LABEL,
    PATTERN_CODES,
    SEIZURE_TYPE_CODES,
    SEIZURE_TYPE_NAMES,
    BlockMeta,
    EpilepsiAEPreprocessor,
    PatientMeta,
    SeizureEvent,
    _apply_filters,
    _find_patient_sql,
    _resample_to,
    bipolar_from_unipolar,
    build_samplewise_labels,
    load_block_signals,
    load_patient_metadata,
    read_head,
    select_canonical_channels,
    seizures_for_recording,
)

logger = logging.getLogger(__name__)

_LABEL_MODES = ("binary", "multiclass", "pattern")

# Default max blocks kept in the LRU cache.  Each processed block is
# roughly 70 MB (1 h × 256 Hz × 19 ch × 4 B).  40 blocks ≈ 2.8 GB.
# Bumped from 20 → 40 because shard 4 historically OOM'd from LRU thrash:
# the bin-packer leaves the last shard with the most patients (high
# rec_id diversity), so the cache churns on every recording boundary.
# 2.8 GB * NW=2 workers ≈ 5.6 GB cache, still tiny inside the 195 GB cgroup.
_DEFAULT_LRU_SIZE = 40


# ---------------------------------------------------------------------------
# Per-recording metadata (built at init, no signal data)
# ---------------------------------------------------------------------------


@dataclass
class _RecordingInfo:
    """Lightweight descriptor for one recording — no signal data."""
    rec_id: str
    subject_id: str
    variant: str
    blocks: List[BlockMeta]
    block_cumulative_starts: List[int]
    gap_regions: List[Tuple[int, int]]
    total_samples: int  # at target_fs, including gap samples
    channel_mask: np.ndarray  # (19,) bool
    native_fs: int
    samplewise_label: np.ndarray  # (total_samples,) uint8
    samplewise_type: np.ndarray  # (total_samples,) uint8
    events: List[Dict[str, Any]]
    # patient metadata
    gender: str = ""
    age: int = -1
    onset_age: int = -1
    hospital: str = ""
    focus_localisation: str = ""
    n_missing_channels: int = 0


# ---------------------------------------------------------------------------
# Block LRU cache
# ---------------------------------------------------------------------------


class _BlockLRU:
    """Simple LRU cache mapping (data_path_str) -> processed (19, T) array."""

    def __init__(self, maxsize: int = _DEFAULT_LRU_SIZE) -> None:
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: np.ndarray) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = value


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EpilepsiAEContinuousDataset(Dataset):
    """Window-level PyTorch dataset that reads EPILEPSIAE raw files on the fly.

    Args:
        data_root: Root of the raw EPILEPSIAE corpus
            (e.g. ``${EEG_DATA_ROOT}/epilepsiae``).
        patients: Pre-discovered patient list.  If *None* the filesystem
            is scanned (slow on first call but cached by the adapter).
        window_s: Window duration in seconds (default 30).
        stride_s: Window stride in seconds.  ``None`` → non-overlapping.
        overlap_threshold: Seizure fraction threshold for binary label.
        montage: ``"unipolar"`` (19 ch) or ``"bipolar"`` (18 ch TCP).
        label_mode: ``"binary"``, ``"multiclass"``, or ``"pattern"``.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        recording_indices: Subset of recording indices to include.
        target_fs: Target sampling rate (default 256).
        lru_blocks: Max processed blocks kept in memory.
        variants: Which corpus variants to include when discovering.
    """

    def __init__(
        self,
        data_root: str | Path,
        patients: Optional[List[PatientMeta]] = None,
        annotations: Optional[List[Dict[str, Any]]] = None,
        origin_map: Optional[Dict[str, str]] = None,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        overlap_threshold: float = 0.0,
        montage: Literal["unipolar", "bipolar"] = "bipolar",
        label_mode: Literal["binary", "multiclass", "pattern"] = "binary",
        normalize: Literal["none", "per_window_zscore"] = "none",
        recording_indices: Optional[Sequence[int]] = None,
        target_fs: int = 256,
        lru_blocks: int = _DEFAULT_LRU_SIZE,
        variants: Sequence[str] = ("surf30", "surfPA", "surfCO"),
    ) -> None:
        self._data_root = Path(data_root)
        self._window_s = window_s
        self._stride_s = stride_s if stride_s is not None else window_s
        self._overlap_threshold = overlap_threshold
        self._montage = montage
        self._label_mode = label_mode
        self._normalize = normalize
        self._target_fs = target_fs
        self._block_cache = _BlockLRU(maxsize=lru_blocks)
        # Lazy per-recording amplitude-stats cache (MODEL_CONTRACTS §3).
        # Computed via the streaming algorithm in compute_recording_stats_streaming
        # so even 1000+ h recordings (~30 GB unipolar) never get materialised.
        self._rec_stats_cache: Dict[int, Dict[str, Any]] = {}

        # ----- discovery (reuse caller-provided data when available) -----
        preprocessor = EpilepsiAEPreprocessor(
            data_root=self._data_root, target_fs=target_fs,
        )
        if patients is None:
            patients = preprocessor.discover_patients(variants=variants)
        if annotations is None:
            annotations = preprocessor.annotations
        if origin_map is None:
            origin_map = preprocessor.origin_map

        # ----- build per-recording metadata -----
        self._recordings: List[_RecordingInfo] = []
        metadata_dir = preprocessor.annotation_dir / "Metadata"
        for patient in patients:
            sql_path = _find_patient_sql(patient.pat_id, metadata_dir)
            pmeta = {}
            if sql_path is not None:
                try:
                    pmeta = load_patient_metadata(sql_path)
                except Exception:
                    pass

            focus_loc = ""
            foci = pmeta.get("eeg_focus", [])
            if foci:
                focus_loc = foci[0].get("localisation", "")

            for rec_id, blocks in patient.recordings.items():
                seizures = seizures_for_recording(annotations, rec_id, origin_map)
                cum_starts, gap_regions, total_samp, ch_mask = (
                    preprocessor._compute_recording_layout(blocks)
                )
                labels, types, event_dicts = build_samplewise_labels(
                    seizures, blocks, total_samp, cum_starts, target_fs,
                )
                for gs, ge in gap_regions:
                    labels[gs:ge] = GAP_LABEL
                    types[gs:ge] = GAP_LABEL

                n_missing = int((~ch_mask).sum())
                self._recordings.append(_RecordingInfo(
                    rec_id=rec_id,
                    subject_id=patient.pat_id,
                    variant=patient.variant,
                    blocks=blocks,
                    block_cumulative_starts=cum_starts,
                    gap_regions=gap_regions,
                    total_samples=total_samp,
                    channel_mask=ch_mask,
                    native_fs=blocks[0].sample_freq,
                    samplewise_label=labels,
                    samplewise_type=types,
                    events=event_dicts,
                    gender=pmeta.get("gender", ""),
                    age=pmeta.get("age", -1),
                    onset_age=pmeta.get("onset_age", -1),
                    hospital=pmeta.get("hospital", ""),
                    focus_localisation=focus_loc,
                    n_missing_channels=n_missing,
                ))

        # ----- build window index -----
        self._fs = target_fs
        self._window_samples = int(round(window_s * target_fs))
        self._stride_samples = int(round(self._stride_s * target_fs))

        if recording_indices is not None:
            rec_set = set(recording_indices)
        else:
            rec_set = set(range(len(self._recordings)))

        index: List[Tuple[int, int]] = []  # (rec_idx, sample_offset_in_rec)
        for rec_i, rec in enumerate(self._recordings):
            if rec_i not in rec_set:
                continue
            if rec.total_samples < self._window_samples:
                continue
            pos = 0
            while pos + self._window_samples <= rec.total_samples:
                window_labels = rec.samplewise_label[pos:pos + self._window_samples]
                if not np.all(window_labels == GAP_LABEL):
                    index.append((rec_i, pos))
                pos += self._stride_samples

        self._index = np.array(index, dtype=np.int64) if index else np.zeros((0, 2), dtype=np.int64)

    # ----- block loading with LRU cache -----

    def _load_processed_block(self, block: BlockMeta) -> np.ndarray:
        """Load, preprocess, and return a single block as (19, T_resampled).

        Results are LRU-cached by the block's data file path.
        """
        key = str(block.data_path)
        cached = self._block_cache.get(key)
        if cached is not None:
            return cached

        import warnings
        head = read_head(block.head_path)
        signals = load_block_signals(block.data_path, head)
        signals_19, _ = select_canonical_channels(signals, block.elec_names)
        del signals

        signals_19 = _resample_to(signals_19, block.sample_freq, self._target_fs)

        if signals_19.shape[1] > 20:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                signals_19 = _apply_filters(signals_19, self._target_fs)

        self._block_cache.put(key, signals_19)
        return signals_19

    def _read_window_signals(
        self, rec: _RecordingInfo, start: int, end: int,
    ) -> np.ndarray:
        """Read processed signals for sample range [start, end) within a recording.

        Handles windows that span block boundaries and gap regions.
        Returns (19, window_samples) float32.
        """
        window_len = end - start
        out = np.zeros((19, window_len), dtype=np.float32)

        cum_starts = rec.block_cumulative_starts
        blocks = rec.blocks

        for bi, block in enumerate(blocks):
            # Compute the sample range this block occupies in the recording
            block_start = cum_starts[bi]
            block_resampled_len = int(round(
                block.num_samples * self._target_fs / block.sample_freq
            ))
            block_end = block_start + block_resampled_len

            # Check overlap with requested window
            overlap_start = max(start, block_start)
            overlap_end = min(end, block_end)
            if overlap_start >= overlap_end:
                continue

            # Load and process this block
            processed = self._load_processed_block(block)  # (19, T_block)
            actual_len = processed.shape[1]

            # Sample indices within the block
            src_start = overlap_start - block_start
            src_end = overlap_end - block_start
            src_start = min(src_start, actual_len)
            src_end = min(src_end, actual_len)
            if src_start >= src_end:
                continue

            # Sample indices within the output window
            dst_start = overlap_start - start
            dst_end = dst_start + (src_end - src_start)

            out[:, dst_start:dst_end] = processed[:, src_start:src_end]

        # Gap regions are already zero-filled (np.zeros init)
        return out

    # ----- Dataset interface -----

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_i = int(self._index[idx, 0])
        win_start = int(self._index[idx, 1])  # offset within recording
        win_end = win_start + self._window_samples
        rec = self._recordings[rec_i]

        # Read signals
        signals = self._read_window_signals(rec, win_start, win_end)  # (19, T)

        # Montage conversion
        if self._montage == "bipolar":
            signals = bipolar_from_unipolar(signals)  # (18, T)

        # Normalization
        if self._normalize == "per_window_zscore":
            mean = signals.mean(axis=1, keepdims=True)
            std = signals.std(axis=1, keepdims=True)
            std[std < 1e-8] = 1.0
            signals = (signals - mean) / std

        # Labels
        window_labels = rec.samplewise_label[win_start:win_end]
        valid_mask = window_labels != GAP_LABEL
        n_valid = valid_mask.sum()

        seizure_count = ((window_labels == 1) & valid_mask).sum()
        seizure_fraction = seizure_count / max(n_valid, 1)
        binary_label = int(seizure_fraction > self._overlap_threshold)

        if self._label_mode == "binary":
            label = binary_label
        elif self._label_mode == "multiclass":
            window_types = rec.samplewise_type[win_start:win_end]
            valid_types = window_types[(window_types != GAP_LABEL) & (window_types > 0)]
            if len(valid_types) > 0:
                label = int(np.bincount(valid_types).argmax())
            else:
                label = 0
        elif self._label_mode == "pattern":
            label = self._get_pattern_label(rec, win_start, win_end)
        else:
            label = binary_label

        window_start_s = win_start / self._fs
        window_end_s = win_end / self._fs

        meta: Dict[str, Any] = {
            "dataset": "epilepsiae",
            "split": "",
            "subject_id": rec.subject_id,
            "recording_id": rec.rec_id,
            "recording_idx": rec_i,
            "variant": rec.variant,
            "window_start_s": float(window_start_s),
            "window_end_s": float(window_end_s),
            "recording_duration_s": float(rec.total_samples / self._fs),
            "seizure_fraction": float(seizure_fraction),
            "has_seizure": bool(binary_label),
            "binary_label": binary_label,
            "label_mode": self._label_mode,
            "n_missing_channels": rec.n_missing_channels,
            "age": rec.age,
            "onset_age": rec.onset_age,
            "gender": rec.gender,
            "hospital": rec.hospital,
            "focus_localisation": rec.focus_localisation,
            "channels": list(BIPOLAR_NAMES if self._montage == "bipolar" else CANONICAL_19),
            "montage": self._montage,
            "normalize": self._normalize,
            "fs": self._fs,
            "sampling_rate": float(self._fs),
            "units": "uV",
            "unit": "uV",
            "window_s": self._window_s,
            "stride_s": self._stride_s,
        }

        # Lazy per-recording stats — STREAMING (bit-exact to eager
        # compute_recording_stats but never materialises the full recording).
        # A 1000+ h recording would otherwise need ~30 GB for the unipolar
        # array alone (see compute_recording_stats_streaming docstring).
        # Two passes over ``rec.blocks``: Welford for mean/std, histogram +
        # bin refinement for q95 and 171 bipolar-pair q95s. When emitted
        # montage is bipolar, look up the 20 bipolar-aligned pairs from the
        # 171-pair dict and overwrite ``recording_q95`` so BIOT's anontier
        # indexes the emitted channels correctly.
        if rec_i not in self._rec_stats_cache:
            stats_file = f"{rec.rec_id}.npz"
            fingerprint = {
                "target_fs": int(self._target_fs),
                "total_samples": int(rec.total_samples),
                "variant": str(rec.variant),
            }
            stats, disk_path = load_cached_recording_stats(_STATS_READER, stats_file, fingerprint)
            if stats is None:
                def _blocks_iter():
                    for b in rec.blocks:
                        yield self._load_processed_block(b)
                stats = compute_recording_stats_streaming(
                    _blocks_iter, n_channels=19, include_bipolar=True,
                )
                try:
                    save_recording_stats(disk_path, stats, fingerprint)
                except Exception:
                    # Disk-cache write is best-effort; fall through with the
                    # in-memory result if /anonorg is full or read-only.
                    pass
            if self._montage == "bipolar":
                from neuroatlas.extensions.datasets.epilepsy._common import (
                    BIPOLAR_MONTAGE as _PAIRS, CANONICAL_IDX as _IDX,
                )
                bq_dict = stats["recording_q95_bipolar"]
                bq = np.empty(len(_PAIRS), dtype=np.float32)
                for k, (a, b) in enumerate(_PAIRS):
                    ia, ib = _IDX[a], _IDX[b]
                    bq[k] = bq_dict[f"{min(ia, ib)},{max(ia, ib)}"]
                stats["recording_q95"] = bq
            self._rec_stats_cache[rec_i] = stats
        meta.update(self._rec_stats_cache[rec_i])

        return {
            "eeg": torch.from_numpy(signals),
            "label": label,
            "meta": meta,
        }

    def _get_pattern_label(self, rec: _RecordingInfo, start: int, end: int) -> int:
        """Dominant EEG pattern code for the window from event arrays."""
        start_s = start / self._fs
        end_s = end / self._fs
        best = 0
        for ev in rec.events:
            if ev["start_s"] < end_s and ev["stop_s"] > start_s:
                best = max(best, ev.get("pattern", 0))
        return best

    # ----- Convenience accessors (same interface as the HDF5 version) -----

    @property
    def window_s(self) -> float:
        return self._window_s

    @property
    def stride_s(self) -> float:
        return self._stride_s

    @property
    def montage(self) -> str:
        return self._montage

    @property
    def label_mode(self) -> str:
        return self._label_mode

    @property
    def channels(self) -> List[str]:
        return list(BIPOLAR_NAMES if self._montage == "bipolar" else CANONICAL_19)

    @property
    def n_recordings(self) -> int:
        return len(self._recordings)

    @property
    def subject_ids(self) -> List[str]:
        return list(set(r.subject_id for r in self._recordings))

    @property
    def target_fs(self) -> int:
        return self._target_fs

    def samplewise_type(self, rec_i: int) -> np.ndarray:
        return self._recordings[rec_i].samplewise_type

    def binary_labels(self) -> np.ndarray:
        """Vectorised per-window binary labels (no signal reads needed)."""
        labels = np.zeros(len(self._index), dtype=np.int64)
        for i in range(len(self._index)):
            rec_i = int(self._index[i, 0])
            start = int(self._index[i, 1])
            rec = self._recordings[rec_i]
            window = rec.samplewise_label[start:start + self._window_samples]
            valid = window[window != GAP_LABEL]
            if len(valid) > 0:
                frac = (valid == 1).sum() / len(valid)
                labels[i] = int(frac > self._overlap_threshold)
        return labels

    def all_labels(self) -> np.ndarray:
        """Per-window labels for the current label_mode."""
        labels = np.zeros(len(self._index), dtype=np.int64)
        for i in range(len(self._index)):
            labels[i] = self[i]["label"]
        return labels
