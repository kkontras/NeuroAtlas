"""Dataset classes for the TUH EEG Abnormal Corpus (TUAB) v3.0.1.

Two backends are provided so HDF5 caching is **opt-in** rather than
required:

- :class:`TuabEdfDataset` — reads raw EDF files lazily on demand. This
  is the default path: a fresh checkout that has only run
  the fetch step can use the dataset immediately, no preprocess
  step required.

- :class:`TuabH5Dataset` — reads from a pre-built continuous HDF5 cache
  (see :mod:`neuroatlas.extensions.datasets.epilepsy.tuab_preprocessor`).
  Faster random-access, but the cache build itself is heavy (~80 GB of
  EDFs to scan once). Use this when iterating many epochs or when many
  models will reuse the same windows.

Both expose the same interface (``__getitem__``, ``__len__``,
``targets()``, ``n_classes``, ``recording_index_to_subject``) so the
adapter in :mod:`...adapters.tuab` can swap them transparently.
"""
from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from neuroatlas.extensions.datasets._recording_stats import (
    compute_recording_stats,
    load_recording_stats,
    save_recording_stats,
)
from pathlib import Path as _Path

_TUAB_STATS_CACHE_ROOT = _Path("artifacts/recording_stats/tuab")
from neuroatlas.extensions.datasets.epilepsy._common import (
    apply_standard_filters,
)
from neuroatlas.extensions.datasets.epilepsy.tusz.readers import (
    BIPOLAR_MONTAGE,
    TARGET_FS,
    UNIPOLAR_ELECTRODES,
    bipolar_from_unipolar,
    parse_tuh_patient_field,
    read_edf_unipolar,
)

# Filter recipe applied per-recording at EDF read time (parity with
# tuab_preprocessor / tusz_edf). 0.5 Hz HP + 60 Hz notch (US mains —
# Temple University Hospital).
_FILTER_TAG = "hp0.5_notch60"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _decode_vlen(arr: np.ndarray) -> List[str]:
    return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in arr]


def _build_window_index(
    durations_samples: Sequence[int],
    labels: Sequence[int],
    window_samples: int,
    stride_samples: int,
    recording_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Return ``(N, 3) int64`` array of ``(rec_idx, start_sample_in_rec, label)``.

    Drops partial trailing windows.
    """
    keep = (
        set(int(i) for i in recording_indices)
        if recording_indices is not None
        else None
    )
    rows: List[Tuple[int, int, int]] = []
    for rec_idx, n_s in enumerate(durations_samples):
        if keep is not None and rec_idx not in keep:
            continue
        if n_s < window_samples:
            continue
        start = 0
        while start + window_samples <= n_s:
            rows.append((rec_idx, start, int(labels[rec_idx])))
            start += stride_samples
    if not rows:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


# ---------------------------------------------------------------------------
# Discovery (raw-EDF mode)
# ---------------------------------------------------------------------------


def discover_tuab_recordings(
    raw_root: str,
    splits: Sequence[str] = ("train", "eval"),
) -> List[Dict[str, Any]]:
    """Walk ``raw_root/<split>/{normal,abnormal}/<montage>/<subject>/<session>/*.edf``.

    Returns a list of dicts with keys
    ``edf_path, split, is_abnormal, montage_type, subject_id, session_id,
    recording_id``.

    Discovery does **not** read any EDF; per-file durations are obtained
    lazily by :class:`TuabEdfDataset` on first access (and then cached
    in-memory for the lifetime of the dataset).
    """
    out: List[Dict[str, Any]] = []
    raw_root_p = Path(raw_root)
    for split in splits:
        split_dir = raw_root_p / split
        if not split_dir.exists():
            logger.warning("TUAB split dir not found, skipping: %s", split_dir)
            continue
        for label_dir, is_abnormal in (("abnormal", 1), ("normal", 0)):
            root = split_dir / label_dir
            if not root.exists():
                logger.warning("TUAB %s/%s missing, skipping", split, label_dir)
                continue
            for edf_path in sorted(root.rglob("*.edf")):
                rel = edf_path.relative_to(root)
                parts = rel.parts
                montage = parts[0] if len(parts) >= 1 else "unknown"
                subject = parts[1] if len(parts) >= 2 else "unknown"
                session = parts[2] if len(parts) >= 3 else "unknown"
                out.append({
                    "edf_path": str(edf_path),
                    "split": split,
                    "is_abnormal": int(is_abnormal),
                    "montage_type": montage,
                    "subject_id": subject,
                    "session_id": session,
                    "recording_id": edf_path.stem,
                })
    logger.info(
        "Discovered %d TUAB recordings under %s (splits=%s)",
        len(out), raw_root, list(splits),
    )
    return out


# ---------------------------------------------------------------------------
# EDF-backed dataset (default — no preprocess required)
# ---------------------------------------------------------------------------


class TuabEdfDataset(Dataset):
    """Lazy windowed view over raw TUAB EDFs.

    Each ``__getitem__`` call reads the parent recording's full unipolar
    waveform (cached in-memory after the first hit, bounded by
    ``cache_max_recordings`` to avoid blowing memory on large epochs)
    and slices out one window.

    Args:
        recordings: Output of :func:`discover_tuab_recordings`, optionally
            pre-filtered by the caller (e.g. one split, or a fold subset).
        window_s: Window length in seconds (default 30 s).
        stride_s: Stride between consecutive windows in the same recording
            (default = window_s, non-overlapping).
        normalize: ``"none"`` (default) or ``"per_window_zscore"``.
        montage_filter: Optional iterable of allowed montage_type names
            (e.g. ``("01_tcp_ar", "02_tcp_le")``). Recordings outside the
            filter are dropped entirely.
        recording_indices: Optional positional subset of ``recordings``
            (used for fold-level slicing without re-scanning the
            filesystem).
        cache_max_recordings: Maximum number of fully-decoded recordings
            kept in memory at once (LRU). Default 16.
    """

    def __init__(
        self,
        recordings: Sequence[Dict[str, Any]],
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        normalize: str = "none",
        montage: str = "unipolar",
        montage_filter: Optional[Sequence[str]] = None,
        recording_indices: Optional[Sequence[int]] = None,
        cache_max_recordings: int = 16,
    ) -> None:
        if window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        if montage not in ("unipolar", "bipolar"):
            raise ValueError(
                f"montage must be 'unipolar' or 'bipolar', got {montage!r}"
            )
        self._window_s = float(window_s)
        self._stride_s = float(stride_s) if stride_s is not None else float(window_s)
        self._window_samples = int(round(self._window_s * TARGET_FS))
        self._stride_samples = max(1, int(round(self._stride_s * TARGET_FS)))
        self._normalize = normalize
        self._montage = montage
        self._cache_max = int(cache_max_recordings)
        self._lru_cache: "collections.OrderedDict[int, np.ndarray]" = (
            collections.OrderedDict()
        )
        # Lazy per-recording amplitude-stats cache (MODEL_CONTRACTS §3).
        self._rec_stats_cache: Dict[int, Dict[str, Any]] = {}

        recs = list(recordings)
        if montage_filter is not None:
            allowed = set(montage_filter)
            recs = [r for r in recs if r.get("montage_type") in allowed]
        if recording_indices is not None:
            keep = set(int(i) for i in recording_indices)
            recs = [r for i, r in enumerate(recs) if i in keep]

        if not recs:
            raise RuntimeError(
                "TuabEdfDataset has zero recordings after filtering — "
                "check raw_root, montage_filter, and recording_indices."
            )
        self._recordings: List[Dict[str, Any]] = recs

        # Lazy duration discovery: we don't know lengths until we open
        # each EDF.  We do that on first access in __getitem__ via the
        # cache; until then, the index is empty.  For deterministic
        # length / iteration order we eagerly probe durations using the
        # EDF header (cheap — does not read sample data).
        self._durations_samples: List[int] = self._probe_durations(recs)
        # Parse TUH EDF+ patient field for age + sex (cheap header-only read)
        # and attach to each rec dict so __getitem__ can emit demographics.
        for r in recs:
            demo = parse_tuh_patient_field(r["edf_path"])
            r["age"] = demo["age"]
            r["gender"] = demo["gender"]
        self._labels: np.ndarray = np.array(
            [r["is_abnormal"] for r in recs], dtype=np.int64,
        )

        self._index = _build_window_index(
            durations_samples=self._durations_samples,
            labels=self._labels.tolist(),
            window_samples=self._window_samples,
            stride_samples=self._stride_samples,
        )
        if len(self._index) == 0:
            raise RuntimeError(
                f"TuabEdfDataset produced 0 windows for window_s={window_s}, "
                f"stride_s={self._stride_s}.  All {len(recs)} recordings are "
                f"shorter than {self._window_samples} samples?"
            )

        logger.info(
            "TuabEdfDataset: %d recordings, %d windows "
            "(window=%.1fs, stride=%.1fs)",
            len(recs), len(self._index), self._window_s, self._stride_s,
        )

    # ------------------------------------------------------------------
    # Duration probing (header-only, no sample read)
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_durations(recs: Sequence[Dict[str, Any]]) -> List[int]:
        import pyedflib
        durations: List[int] = []
        for r in recs:
            try:
                reader = pyedflib.EdfReader(r["edf_path"])
                try:
                    duration_s = reader.getFileDuration()
                finally:
                    reader._close()
                durations.append(int(round(duration_s * TARGET_FS)))
            except Exception as exc:
                logger.warning(
                    "Failed to read EDF header %s: %s — marking 0 samples",
                    r["edf_path"], exc,
                )
                durations.append(0)
        return durations

    # ------------------------------------------------------------------
    # In-memory LRU cache of decoded recordings
    # ------------------------------------------------------------------

    def _get_decoded(self, rec_idx: int) -> np.ndarray:
        if rec_idx in self._lru_cache:
            self._lru_cache.move_to_end(rec_idx)
            return self._lru_cache[rec_idx]
        signals, _native_fs, _n_missing, _units = read_edf_unipolar(
            self._recordings[rec_idx]["edf_path"],
        )
        # Apply 0.5 Hz HP + 60 Hz notch (parity with tuab_preprocessor and
        # tusz_edf). signals is time-first (T, 19); axis=0 is time.
        if signals.shape[0] > 20:
            signals = apply_standard_filters(
                signals, fs=TARGET_FS, notch_hz=60.0, axis=0,
            )
        self._lru_cache[rec_idx] = signals
        while len(self._lru_cache) > self._cache_max:
            self._lru_cache.popitem(last=False)
        if rec_idx not in self._rec_stats_cache:
            # Disk cache is keyed by montage so unipolar and bipolar variants
            # both persist across runs. Legacy unipolar caches at <rec>.npz
            # (no suffix) are still honoured for backward compat.
            rec_id = self._recordings[rec_idx]["recording_id"]
            if self._montage == "bipolar":
                disk_path = _TUAB_STATS_CACHE_ROOT / f"{rec_id}_bipolar.npz"
            else:
                disk_path = _TUAB_STATS_CACHE_ROOT / f"{rec_id}.npz"
            fingerprint = {
                "target_fs": int(TARGET_FS),
                "n_samples": int(signals.shape[0]),
                "filters": _FILTER_TAG,
                "montage": self._montage,
            }
            stats = load_recording_stats(disk_path, fingerprint)
            if stats is None:
                if self._montage == "bipolar":
                    # signals is (T, 19); compute stats on the emitted (20, T)
                    # bipolar so mean/std/q95 align with meta["channels"].
                    bipolar_full = bipolar_from_unipolar(signals)  # (T, 20)
                    stats = compute_recording_stats(bipolar_full.T)
                else:
                    stats = compute_recording_stats(
                        signals.T.astype(np.float32, copy=False)
                    )
                try:
                    save_recording_stats(disk_path, stats, fingerprint)
                except Exception:
                    pass
            self._rec_stats_cache[rec_idx] = stats
        return signals

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_idx, start, target = (int(x) for x in self._index[idx])
        end = start + self._window_samples
        signals = self._get_decoded(rec_idx)
        # signals shape: (n_samples, 19) in UNIPOLAR_ELECTRODES order.
        win = signals[start:end]  # (T, 19)
        if self._montage == "bipolar":
            win = bipolar_from_unipolar(win)  # (T, 20)
            channels = list(BIPOLAR_MONTAGE)
        else:
            channels = list(UNIPOLAR_ELECTRODES)
        sig = win.T.astype(np.float32, copy=True)

        if self._normalize == "per_window_zscore":
            mean = sig.mean(axis=1, keepdims=True)
            std = sig.std(axis=1, keepdims=True)
            std[std < 1e-8] = 1.0
            sig = (sig - mean) / std

        rec = self._recordings[rec_idx]
        meta: Dict[str, Any] = {
            "dataset": "tuab",
            "recording_id": rec["recording_id"],
            "subject_id": rec["subject_id"],
            "session_id": rec["session_id"],
            "split": rec["split"],
            "montage_type": rec["montage_type"],
            "montage": self._montage,
            "is_abnormal": int(rec["is_abnormal"]),
            "window_start_s": float(start / TARGET_FS),
            "window_end_s": float(end / TARGET_FS),
            "window_s": self._window_s,
            "stride_s": self._stride_s,
            "fs": TARGET_FS,
            "sampling_rate": float(TARGET_FS),
            "units": "uV",
            "unit": "uV",
            "channels": channels,
            "source": "edf",
            "age": int(rec.get("age", -1)),
            "gender": rec.get("gender", ""),
        }
        meta.update(self._rec_stats_cache[rec_idx])
        return {
            "eeg": torch.from_numpy(sig),
            "label": target,
            "meta": meta,
        }

    def targets(self) -> np.ndarray:
        return self._index[:, 2].astype(np.int64)

    @property
    def n_classes(self) -> int:
        return 2

    def recording_index_to_subject(self) -> List[str]:
        """Return the subject id for each recording position."""
        return [r["subject_id"] for r in self._recordings]


# ---------------------------------------------------------------------------
# H5-backed dataset (opt-in fast path)
# ---------------------------------------------------------------------------


class TuabH5Dataset(Dataset):
    """Windowed view over a pre-built TUAB continuous HDF5 cache.

    Use this when the cache exists (build it with
    ``python -m neuroatlas.extensions.datasets.preprocessors.preprocess_tuab``) and you
    want random-access speed, e.g. for many-epoch training or
    hyperparameter sweeps.

    Args:
        h5_path: Path to a per-split TUAB cache file.
        window_s, stride_s, normalize, montage_filter, recording_indices:
            See :class:`TuabEdfDataset`.
    """

    def __init__(
        self,
        h5_path: str,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        normalize: str = "none",
        montage_filter: Optional[Sequence[str]] = None,
        recording_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self._h5: Optional[h5py.File] = None
        self._h5_path = str(h5_path)
        self._window_s = float(window_s)
        self._stride_s = float(stride_s) if stride_s is not None else float(window_s)
        self._window_samples = int(round(self._window_s * TARGET_FS))
        self._stride_samples = max(1, int(round(self._stride_s * TARGET_FS)))
        self._normalize = normalize

        with h5py.File(self._h5_path, "r") as f:
            offsets = f["recording_offsets"][:].astype(np.int64)
            self._is_abnormal = f["is_abnormal"][:].astype(np.int64)
            self._recording_ids = _decode_vlen(f["recording_ids"][:])
            self._subject_ids = _decode_vlen(f["subject_ids"][:])
            self._session_ids = (
                _decode_vlen(f["session_ids"][:]) if "session_ids" in f
                else ["unknown"] * len(self._recording_ids)
            )
            self._montage_types = _decode_vlen(f["montage_types"][:])
            self._splits = [str(f.attrs.get("split", "unknown"))] * len(self._recording_ids)

        # Lengths per recording from the offset array.
        self._durations_samples = [
            int(offsets[i + 1] - offsets[i]) for i in range(len(offsets) - 1)
        ]
        self._global_offsets = offsets

        # Apply filters → derive a final recording-index subset
        keep_idx = list(range(len(self._recording_ids)))
        if montage_filter is not None:
            allowed = set(montage_filter)
            keep_idx = [i for i in keep_idx if self._montage_types[i] in allowed]
        if recording_indices is not None:
            wanted = set(int(i) for i in recording_indices)
            keep_idx = [i for i in keep_idx if i in wanted]
        if not keep_idx:
            raise RuntimeError(
                "TuabH5Dataset has zero recordings after filtering — "
                "check montage_filter and recording_indices."
            )

        # Lazy per-recording amplitude-stats cache (MODEL_CONTRACTS §3),
        # populated on first window access for each rec.
        self._rec_stats_cache: Dict[int, Dict[str, Any]] = {}

        self._index = _build_window_index(
            durations_samples=self._durations_samples,
            labels=self._is_abnormal.tolist(),
            window_samples=self._window_samples,
            stride_samples=self._stride_samples,
            recording_indices=keep_idx,
        )
        if len(self._index) == 0:
            raise RuntimeError(
                f"TuabH5Dataset produced 0 windows for window_s={window_s}."
            )

        logger.info(
            "TuabH5Dataset: %s — %d recordings, %d windows "
            "(window=%.1fs, stride=%.1fs)",
            self._h5_path, len(keep_idx), len(self._index),
            self._window_s, self._stride_s,
        )

    # ------------------------------------------------------------------
    # HDF5 lazy handle
    # ------------------------------------------------------------------

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None or not self._h5.id.valid:
            self._h5 = h5py.File(self._h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_idx, start_in_rec, target = (int(x) for x in self._index[idx])
        end_in_rec = start_in_rec + self._window_samples
        global_start = int(self._global_offsets[rec_idx]) + start_in_rec
        global_end = global_start + self._window_samples

        sig = self.h5["signals"][global_start:global_end, :].astype(np.float32)
        sig = sig.T  # (19, T)

        if self._normalize == "per_window_zscore":
            mean = sig.mean(axis=1, keepdims=True)
            std = sig.std(axis=1, keepdims=True)
            std[std < 1e-8] = 1.0
            sig = (sig - mean) / std

        # Lazy per-recording amplitude stats: load full recording once via
        # the global offset slice, compute, and cache.
        if rec_idx not in self._rec_stats_cache:
            rec_global_start = int(self._global_offsets[rec_idx])
            rec_global_end = int(self._global_offsets[rec_idx + 1])
            rec_full = self.h5["signals"][rec_global_start:rec_global_end, :].T.astype(np.float32)
            self._rec_stats_cache[rec_idx] = compute_recording_stats(rec_full)

        meta: Dict[str, Any] = {
            "dataset": "tuab",
            "recording_id": self._recording_ids[rec_idx],
            "subject_id": self._subject_ids[rec_idx],
            "session_id": self._session_ids[rec_idx],
            "split": self._splits[rec_idx],
            "montage_type": self._montage_types[rec_idx],
            "is_abnormal": int(self._is_abnormal[rec_idx]),
            "window_start_s": float(start_in_rec / TARGET_FS),
            "window_end_s": float(end_in_rec / TARGET_FS),
            "window_s": self._window_s,
            "stride_s": self._stride_s,
            "fs": TARGET_FS,
            "sampling_rate": float(TARGET_FS),
            "units": "uV",
            "unit": "uV",
            "channels": list(UNIPOLAR_ELECTRODES),
            "source": "h5",
        }
        meta.update(self._rec_stats_cache[rec_idx])
        return {
            "eeg": torch.from_numpy(sig),
            "label": target,
            "meta": meta,
        }

    def targets(self) -> np.ndarray:
        return self._index[:, 2].astype(np.int64)

    @property
    def n_classes(self) -> int:
        return 2

    def recording_index_to_subject(self) -> List[str]:
        return list(self._subject_ids)

    def __del__(self) -> None:
        h5 = getattr(self, "_h5", None)
        if h5 is not None and h5.id.valid:
            try:
                h5.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate_tuab(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    eeg = torch.stack([item["eeg"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "eeg": eeg,
        "labels": labels,
        "meta": [item["meta"] for item in batch],
    }
