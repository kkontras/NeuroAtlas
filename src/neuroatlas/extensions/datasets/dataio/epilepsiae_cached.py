"""EPILEPSIAE runtime Dataset backed by a persistent preprocessed cache.

The raw EPILEPSIAE corpus stores multi-hour ``.data`` blocks that can
exceed 500 MB. Decoding + resampling + scipy ``filtfilt`` each block on
every LRU miss dominates embed wall-clock at 2 s / 1 s overlap.
``preprocess_epilepsiae`` walks the corpus once, applies the same
(channel-select → resample → filter → gap-fill) pipeline, and writes one
compact ``signals.npy`` (float16, 19 ch at ``target_fs``) per recording,
alongside samplewise labels and JSON metadata. This adapter mmap-reads
those files — ``__getitem__`` becomes a slice + optional montage + optional
z-score, with no scipy work on the data path.

Public ``__getitem__`` return signature matches
``EpilepsiAEContinuousDataset`` so the embed entrypoints are
drop-in-swappable via ``--cache-root``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from neuroatlas.extensions.datasets._recording_stats import compute_recording_stats
from neuroatlas.extensions.datasets.epilepsy.epilepsiae_preprocessor import (
    BIPOLAR_NAMES,
    CANONICAL_19,
    GAP_LABEL,
    bipolar_from_unipolar,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache layout (preprocessor and runtime adapter both import from here)
# ---------------------------------------------------------------------------

CACHE_SCHEMA_TAG = "v1"  # bump on any on-disk format change
SIGNALS_DTYPE = np.float16
LABELS_DTYPE = np.uint8
TYPES_DTYPE = np.uint8

_SIGNALS_FILE = "signals.npy"
_LABELS_FILE = "labels.npy"
_TYPES_FILE = "types.npy"
_META_FILE = "meta.json"
_EVENTS_FILE = "events.json"
_PATIENT_META_FILE = "patient_meta.json"
_SCHEMA_MARKER = ".schema_v1"  # written at end of a successful build


def recording_dir(cache_root: str | Path, subject_id: str, rec_id: str) -> Path:
    """Canonical path for one recording's cache payload."""
    return Path(cache_root) / subject_id / rec_id


def is_recording_complete(cache_root: str | Path, subject_id: str, rec_id: str) -> bool:
    """True iff a recording's cache directory has a valid schema marker."""
    return (recording_dir(cache_root, subject_id, rec_id) / _SCHEMA_MARKER).exists()


def mark_recording_complete(rec_dir: Path) -> None:
    (rec_dir / _SCHEMA_MARKER).write_text(CACHE_SCHEMA_TAG)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class _CachedRecordingInfo:
    """Lightweight descriptor — no signal data resident."""
    rec_id: str
    subject_id: str
    variant: str
    total_samples: int
    native_fs: int
    target_fs: int
    gender: str
    age: int
    onset_age: int
    hospital: str
    focus_localisation: str
    n_missing_channels: int
    dir: Path  # where signals.npy / labels.npy live


class EpilepsiAECachedDataset(Dataset):
    """Drop-in replacement for :class:`EpilepsiAEContinuousDataset`.

    Args:
        cache_root: Directory written by ``preprocess_epilepsiae``.
        patients: Optional list of subject_id strings; if given, restrict
            to matching recordings (used by the sharded embed pipeline).
        recording_indices: Optional positional subset of recordings.
        window_s / stride_s / overlap_threshold / montage / label_mode /
        normalize: same semantics as the EDF Dataset.
    """

    def __init__(
        self,
        cache_root: str | Path,
        patients: Optional[Sequence[str]] = None,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        overlap_threshold: float = 0.0,
        montage: Literal["unipolar", "bipolar"] = "bipolar",
        label_mode: Literal["binary", "multiclass", "pattern"] = "binary",
        normalize: Literal["none", "per_window_zscore"] = "none",
        recording_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self._cache_root = Path(cache_root)
        self._window_s = float(window_s)
        self._stride_s = float(stride_s) if stride_s is not None else float(window_s)
        self._overlap_threshold = float(overlap_threshold)
        self._montage = montage
        self._label_mode = label_mode
        self._normalize = normalize

        patient_filter = set(patients) if patients is not None else None

        # ----- discover cached recordings -----
        recs: List[_CachedRecordingInfo] = []
        if not self._cache_root.exists():
            raise FileNotFoundError(f"cache_root does not exist: {self._cache_root}")

        for subj_dir in sorted(self._cache_root.iterdir()):
            if not subj_dir.is_dir():
                continue
            if patient_filter is not None and subj_dir.name not in patient_filter:
                continue
            for rec_dir in sorted(subj_dir.iterdir()):
                if not rec_dir.is_dir():
                    continue
                if not (rec_dir / _SCHEMA_MARKER).exists():
                    continue
                meta = json.loads((rec_dir / _META_FILE).read_text())
                recs.append(_CachedRecordingInfo(
                    rec_id=meta["rec_id"],
                    subject_id=meta["subject_id"],
                    variant=meta["variant"],
                    total_samples=int(meta["total_samples"]),
                    native_fs=int(meta["native_fs"]),
                    target_fs=int(meta["target_fs"]),
                    gender=meta.get("gender", ""),
                    age=int(meta.get("age", -1)),
                    onset_age=int(meta.get("onset_age", -1)),
                    hospital=meta.get("hospital", ""),
                    focus_localisation=meta.get("focus_localisation", ""),
                    n_missing_channels=int(meta.get("n_missing_channels", 0)),
                    dir=rec_dir,
                ))

        self._recordings = recs
        if not recs:
            raise RuntimeError(
                f"No complete cached recordings found under {self._cache_root} "
                f"(patients filter={patients!r})."
            )

        # All recordings must share target_fs
        self._target_fs = recs[0].target_fs
        if any(r.target_fs != self._target_fs for r in recs):
            mismatches = {r.target_fs for r in recs}
            raise RuntimeError(
                f"Inconsistent target_fs in cache: {mismatches}. Rebuild "
                f"with a single target_fs."
            )
        self._fs = self._target_fs
        self._window_samples = int(round(self._window_s * self._target_fs))
        self._stride_samples = int(round(self._stride_s * self._target_fs))

        # ----- filter by positional indices -----
        if recording_indices is not None:
            keep = set(int(i) for i in recording_indices)
            self._recordings = [r for i, r in enumerate(self._recordings) if i in keep]

        # ----- mmap labels/types per recording (cheap — no signals touched) -----
        self._labels_by_rec: List[np.memmap] = []
        self._types_by_rec: List[np.memmap] = []
        for r in self._recordings:
            labels_mm = np.load(r.dir / _LABELS_FILE, mmap_mode="r")
            types_path = r.dir / _TYPES_FILE
            types_mm = np.load(types_path, mmap_mode="r") if types_path.exists() else None
            self._labels_by_rec.append(labels_mm)
            self._types_by_rec.append(types_mm)

        # ----- events & patient_meta (for pattern label + meta dict) -----
        self._events_by_rec: List[List[Dict[str, Any]]] = []
        for r in self._recordings:
            ev_path = r.dir / _EVENTS_FILE
            events = json.loads(ev_path.read_text()) if ev_path.exists() else []
            self._events_by_rec.append(events)

        # ----- build per-window index, excluding all-GAP windows -----
        index: List[Tuple[int, int]] = []
        for rec_i, r in enumerate(self._recordings):
            if r.total_samples < self._window_samples:
                continue
            labels = self._labels_by_rec[rec_i]
            pos = 0
            while pos + self._window_samples <= r.total_samples:
                window = labels[pos:pos + self._window_samples]
                if not np.all(window == GAP_LABEL):
                    index.append((rec_i, pos))
                pos += self._stride_samples
        self._index = (
            np.array(index, dtype=np.int64) if index else np.zeros((0, 2), dtype=np.int64)
        )

        # ----- lazy signal mmaps (opened on first use per worker) -----
        self._signals_by_rec: List[Optional[np.memmap]] = [None] * len(self._recordings)
        # Lazy per-recording amplitude-stats cache (MODEL_CONTRACTS §3),
        # computed on the unipolar mmap source on first access.
        self._rec_stats_cache: Dict[int, Dict[str, Any]] = {}

        logger.info(
            "EpilepsiAECachedDataset: %s — %d recordings, %d windows "
            "(window=%.1fs, stride=%.1fs, fs=%d)",
            self._cache_root, len(self._recordings), len(self._index),
            self._window_s, self._stride_s, self._target_fs,
        )

    # ------------------------------------------------------------------

    def _get_signals_mm(self, rec_i: int) -> np.memmap:
        mm = self._signals_by_rec[rec_i]
        if mm is None:
            path = self._recordings[rec_i].dir / _SIGNALS_FILE
            mm = np.load(path, mmap_mode="r")
            self._signals_by_rec[rec_i] = mm
        return mm

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_i = int(self._index[idx, 0])
        win_start = int(self._index[idx, 1])
        win_end = win_start + self._window_samples
        rec = self._recordings[rec_i]

        mm = self._get_signals_mm(rec_i)
        # Slice + promote to float32. Copy so downstream torch ops don't
        # alias the mmap.
        signals = np.asarray(mm[:, win_start:win_end], dtype=np.float32)

        if self._montage == "bipolar":
            signals = bipolar_from_unipolar(signals)

        if self._normalize == "per_window_zscore":
            mean = signals.mean(axis=1, keepdims=True)
            std = signals.std(axis=1, keepdims=True)
            std[std < 1e-8] = 1.0
            signals = (signals - mean) / std

        labels_mm = self._labels_by_rec[rec_i]
        window_labels = labels_mm[win_start:win_end]
        valid_mask = window_labels != GAP_LABEL
        n_valid = int(valid_mask.sum())
        seizure_count = int(((window_labels == 1) & valid_mask).sum())
        seizure_fraction = seizure_count / max(n_valid, 1)
        binary_label = int(seizure_fraction > self._overlap_threshold)

        if self._label_mode == "binary":
            label = binary_label
        elif self._label_mode == "multiclass":
            types_mm = self._types_by_rec[rec_i]
            if types_mm is None:
                label = 0
            else:
                window_types = types_mm[win_start:win_end]
                valid_types = window_types[(window_types != GAP_LABEL) & (window_types > 0)]
                label = int(np.bincount(valid_types).argmax()) if len(valid_types) else 0
        elif self._label_mode == "pattern":
            label = self._pattern_label(rec_i, win_start, win_end)
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

        # Lazy per-recording stats from the full mmap (unipolar source). When
        # the dataio emits bipolar, overwrite ``recording_q95`` with the
        # 20-d bipolar-aligned q95 so BIOT's anontier indexes the emitted
        # channels correctly. Stream the bipolar q95 per pair so a
        # 1000-hour Epilepsiae recording doesn't peak at ~90 GB RAM
        # (one full copy + np.abs of the bipolar montage would be 60 GB+).
        if rec_i not in self._rec_stats_cache:
            full_unipolar = np.asarray(mm, dtype=np.float32)
            stats = compute_recording_stats(full_unipolar)
            if self._montage == "bipolar":
                from neuroatlas.extensions.datasets.epilepsy._common import (
                    BIPOLAR_MONTAGE as _PAIRS, CANONICAL_IDX as _IDX,
                )
                bq = np.empty(len(_PAIRS), dtype=np.float32)
                for k, (a, b) in enumerate(_PAIRS):
                    diff = np.abs(full_unipolar[_IDX[a]] - full_unipolar[_IDX[b]])
                    bq[k] = np.quantile(diff, 0.95)
                    del diff
                stats["recording_q95"] = bq
            self._rec_stats_cache[rec_i] = stats
            del full_unipolar
        meta.update(self._rec_stats_cache[rec_i])

        return {
            "eeg": torch.from_numpy(signals),
            "label": label,
            "meta": meta,
        }

    def _pattern_label(self, rec_i: int, start: int, end: int) -> int:
        start_s = start / self._fs
        end_s = end / self._fs
        best = 0
        for ev in self._events_by_rec[rec_i]:
            if ev["start_s"] < end_s and ev["stop_s"] > start_s:
                best = max(best, int(ev.get("pattern", 0)))
        return best

    # ----- Convenience accessors (match EDF Dataset interface) -----

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
        return list({r.subject_id for r in self._recordings})

    @property
    def target_fs(self) -> int:
        return self._target_fs

    def samplewise_type(self, rec_i: int) -> np.ndarray:
        mm = self._types_by_rec[rec_i]
        if mm is None:
            return np.zeros(self._recordings[rec_i].total_samples, dtype=LABELS_DTYPE)
        return mm
