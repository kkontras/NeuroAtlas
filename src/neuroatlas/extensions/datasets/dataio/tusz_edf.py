"""On-the-fly windowed Dataset that streams TUSZ EDFs directly — no HDF5 cache.

This mirrors :class:`TUSZContinuousDataset` but reads each recording from its
source EDF + ``.csv_bi`` / ``.csv`` annotations on ``__getitem__`` time, so
no ``tusz_v*.h5`` file is written or required.  Reuses the exact same
preprocessing primitives the HDF5 builder uses, so embeddings are
bit-identical to the cached path modulo float32 filter rounding:

- ``read_edf_unipolar`` → (T, 19) µV @ 256 Hz with missing channels zero-filled.
- ``apply_standard_filters`` → 0.5 Hz HP + 60 Hz notch (US mains).
- ``bipolar_from_unipolar`` → (T, 20) TCP bipolar pairs (time-first).
- ``parse_csv_bi`` + ``build_samplewise_labels`` → per-sample binary labels.

A small in-worker LRU amortises EDF open/filter cost across consecutive
windows from the same recording; with ``shuffle=False`` (the embed path) the
LRU hits on every window after the first of each recording.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from neuroatlas.extensions.datasets._recording_stats import (
    compute_recording_stats,
    load_recording_stats,
    save_recording_stats,
)
from pathlib import Path as _Path

# Per-recording stats disk cache — shared across all models, populated by
# entrypoints/cache_recording_stats_tusz.py or first-use here.
_TUSZ_STATS_CACHE_ROOT = _Path("artifacts/recording_stats/tusz_edf_direct")
from neuroatlas.extensions.datasets.epilepsy._common import (
    apply_standard_filters,
)
from neuroatlas.extensions.datasets.epilepsy.tusz import (
    BIPOLAR_MONTAGE,
    TARGET_FS,
    UNIPOLAR_ELECTRODES,
    bipolar_from_unipolar,
    build_samplewise_labels,
    parse_csv_bi,
    parse_tuh_patient_field,
    read_edf_unipolar,
)

logger = logging.getLogger(__name__)


_VALID_SPLITS: Tuple[str, ...] = ("train", "dev", "eval")


def _discover_edf_manifest(raw_root: str, split: str) -> List[Dict[str, Any]]:
    """Enumerate EDFs under ``{raw_root}/{split}`` and derive per-recording meta.

    Uses header-only probing (pyedflib ``getFileDuration``) so the scan is
    fast enough to run at worker start without reading any signals.
    Returns one dict per recording with path, subject/session ids, duration,
    native_fs, and n_samples at TARGET_FS.
    """
    import pyedflib

    split_dir = Path(raw_root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"TUSZ split dir not found: {split_dir} — expected under raw_root={raw_root}"
        )
    edfs = sorted(split_dir.rglob("*.edf"))
    manifest: List[Dict[str, Any]] = []
    for edf_path in edfs:
        stem = edf_path.stem
        parent = edf_path.parent  # .../<montage_dir>/
        montage_type = parent.name

        # Path layout: .../edf/<split>/<subject_id>/<session_dir>/<montage_dir>/<stem>.edf
        parts = edf_path.parts
        subject_id = stem.split("_")[0]
        session_id = "unknown"
        try:
            for i, p in enumerate(parts):
                if p == split and i + 1 < len(parts):
                    subject_id = parts[i + 1]
                    session_id = parts[i + 2] if i + 2 < len(parts) else "unknown"
                    break
        except Exception:  # keep defaults on any oddity
            pass

        csv_bi = parent / f"{stem}.csv_bi"

        try:
            reader = pyedflib.EdfReader(str(edf_path))
            try:
                duration_s = float(reader.getFileDuration())
                native_fs = (
                    float(reader.getSampleFrequency(0))
                    if reader.signals_in_file > 0 else float(TARGET_FS)
                )
            finally:
                reader._close()
        except Exception as exc:
            logger.warning("Skipping EDF with unreadable header %s: %s", edf_path, exc)
            continue

        n_samples = int(round(duration_s * TARGET_FS))
        if n_samples <= 0:
            continue

        demo = parse_tuh_patient_field(str(edf_path))

        manifest.append({
            "edf_path": str(edf_path),
            "csv_bi_path": str(csv_bi) if csv_bi.exists() else "",
            "recording_id": stem,
            "subject_id": subject_id,
            "session_id": session_id,
            "montage_type": montage_type,
            "duration_s": duration_s,
            "native_fs": native_fs,
            "n_samples": n_samples,
            "split": split,
            "age": demo["age"],
            "gender": demo["gender"],
        })
    logger.info(
        "TUSZ EDF-direct: discovered %d recordings under %s/%s",
        len(manifest), raw_root, split,
    )
    return manifest


class TUSZEDFDirectDataset(Dataset):
    """Windowed view that streams TUSZ EDFs directly (no HDF5 cache).

    Args:
        raw_root: Root directory containing ``train/``, ``dev/``, ``eval/``
            sub-trees of EDF files (e.g. the TUSZ v2.0.3 ``.../edf`` dir).
        split: One of ``"train"``, ``"dev"``, ``"eval"``.
        window_s: Window duration in seconds.
        stride_s: Stride in seconds (defaults to ``window_s`` = non-overlapping).
        montage: ``"unipolar"`` or ``"bipolar"``.
        label_mode: ``"binary"`` — the only mode supported in v1. Multiclass/
            demographics labels live in the HDF5 cache's side-metadata and
            are not yet parsed in the EDF-direct path.
        lru_recordings: Number of most-recent filtered recordings to hold in
            memory per worker.  Keeps EDF open + filter cost amortised when
            consecutive windows hit the same recording (embed path with
            ``shuffle=False``).
    """

    def __init__(
        self,
        raw_root: str,
        split: str,
        window_s: float = 10.0,
        stride_s: Optional[float] = None,
        montage: str = "bipolar",
        label_mode: str = "binary",
        lru_recordings: int = 2,
        manifest: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"Unsupported split {split!r}; expected one of {_VALID_SPLITS}"
            )
        if label_mode != "binary":
            raise ValueError(
                f"TUSZEDFDirectDataset v1 supports only label_mode='binary' "
                f"(got {label_mode!r}); use TUSZContinuousDataset for other modes."
            )
        if montage not in ("unipolar", "bipolar"):
            raise ValueError(f"Unsupported montage {montage!r}")

        self._raw_root = str(raw_root)
        self._split = split
        self._window_s = float(window_s)
        self._stride_s = float(stride_s) if stride_s is not None else self._window_s
        self._montage = montage
        self._label_mode = label_mode
        self._window_samples = int(round(self._window_s * TARGET_FS))
        self._stride_samples = int(round(self._stride_s * TARGET_FS))
        self._lru_cap = max(1, int(lru_recordings))

        self._manifest: List[Dict[str, Any]] = list(
            manifest if manifest is not None else _discover_edf_manifest(raw_root, split)
        )

        # Build global (rec_idx, local_sample_start) index.
        self._windows: List[Tuple[int, int]] = []
        n_dropped = 0
        for rec_idx, rec in enumerate(self._manifest):
            n = int(rec["n_samples"])
            if n < self._window_samples:
                n_dropped += 1
                continue
            n_windows = (n - self._window_samples) // self._stride_samples + 1
            for w in range(n_windows):
                start = w * self._stride_samples
                if start + self._window_samples <= n:
                    self._windows.append((rec_idx, start))
        if n_dropped:
            logger.info(
                "TUSZ EDF-direct: dropped %d recordings shorter than %.1fs window",
                n_dropped, self._window_s,
            )
        logger.info(
            "TUSZ EDF-direct(%s): %d recordings, %d windows (%.1fs @ %.1fs)",
            split, len(self._manifest) - n_dropped, len(self._windows),
            self._window_s, self._stride_s,
        )

        # Per-worker LRU of filtered recordings (populated lazily in __getitem__).
        # Each entry: rec_idx -> (signals_uv_filtered (T, 19), samplewise_label (T,)).
        self._lru: "OrderedDict[int, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        # Lazy per-recording amplitude-stats cache (MODEL_CONTRACTS §3),
        # computed on the unipolar source before bipolar derivation.
        self._rec_stats_cache: Dict[int, Dict[str, Any]] = {}

    # -- Dataset API ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec_idx, local_offset = self._windows[index]
        rec = self._manifest[rec_idx]

        signals, sw_label = self._load_recording(rec_idx)
        s0 = local_offset
        s1 = s0 + self._window_samples
        win_uni = signals[s0:s1]  # (T, 19) float32 µV

        if self._montage == "bipolar":
            win = bipolar_from_unipolar(win_uni)  # (T, 20)
            channel_names = list(BIPOLAR_MONTAGE)
        else:
            win = win_uni
            channel_names = list(UNIPOLAR_ELECTRODES)

        eeg = torch.as_tensor(win.T, dtype=torch.float32)  # (C, T)

        window_sw = sw_label[s0:s1]
        seizure_fraction = float(np.mean(window_sw > 0))
        has_seizure = seizure_fraction > 0
        binary_label = int(has_seizure)

        meta: Dict[str, Any] = {
            "dataset": "tusz",
            "recording_id": rec["recording_id"],
            "subject_id": rec["subject_id"],
            "session_id": rec["session_id"],
            "recording_index": rec_idx,
            "window_index": index,
            "channels": channel_names,
            "sfreq": float(TARGET_FS),
            "sampling_rate": float(TARGET_FS),
            "unit": "uV",
            "native_fs": float(rec["native_fs"]),
            "montage_type": rec["montage_type"],
            "seizure_fraction": seizure_fraction,
            "has_seizure": has_seizure,
            "binary_label": binary_label,
            "recording_duration_s": float(rec["duration_s"]),
            "window_end_s": (local_offset + self._window_samples) / TARGET_FS,
            "split": rec["split"],
            "age": int(rec.get("age", -1)),
            "gender": rec.get("gender", ""),
        }
        meta.update(self._rec_stats_cache[rec_idx])

        return {
            "signals": {"eeg": eeg},
            "label": torch.tensor(binary_label, dtype=torch.long),
            "meta": meta,
        }

    # -- Internals -----------------------------------------------------------

    def _load_recording(self, rec_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        cached = self._lru.get(rec_idx)
        if cached is not None:
            self._lru.move_to_end(rec_idx)
            return cached

        rec = self._manifest[rec_idx]
        signals_uni, _native_fs, _n_missing, _units = read_edf_unipolar(rec["edf_path"])
        if signals_uni.shape[0] > 20:  # parity with preprocessor._process_one_recording
            signals_uni = apply_standard_filters(
                signals_uni, fs=TARGET_FS, notch_hz=60.0, axis=0,
            )

        if rec["csv_bi_path"]:
            events = parse_csv_bi(rec["csv_bi_path"])
            sw_label = build_samplewise_labels(
                signals_uni.shape[0], events, fs=TARGET_FS,
            )
        else:
            sw_label = np.zeros(signals_uni.shape[0], dtype=np.uint8)

        self._lru[rec_idx] = (signals_uni, sw_label)
        while len(self._lru) > self._lru_cap:
            self._lru.popitem(last=False)
        # Populate the recording-stats cache. When the dataio emits bipolar,
        # overwrite the 19-d unipolar ``recording_q95`` with a 20-d bipolar-
        # aligned q95 so BIOT's anontier q95 path indexes the emitted channels
        # (not the unipolar source). See siena_bids for the same fix.
        if rec_idx not in self._rec_stats_cache:
            disk_path = _TUSZ_STATS_CACHE_ROOT / f"{rec['recording_id']}.npz"
            fingerprint = {
                "target_fs": int(TARGET_FS),
                "n_samples": int(signals_uni.shape[0]),
            }
            stats = load_recording_stats(disk_path, fingerprint)
            if stats is None:
                stats = compute_recording_stats(
                    signals_uni.T.astype(np.float32, copy=False)
                )
                try:
                    save_recording_stats(disk_path, stats, fingerprint)
                except Exception:
                    pass
            if self._montage == "bipolar":
                bipolar_full = bipolar_from_unipolar(signals_uni)  # (T, 20)
                stats = dict(stats)
                stats["recording_q95"] = np.quantile(
                    np.abs(bipolar_full), 0.95, axis=0,
                ).astype(np.float32)
            self._rec_stats_cache[rec_idx] = stats
        return signals_uni, sw_label

    # -- Public accessors (parity with TUSZContinuousDataset) ---------------

    @property
    def subject_ids(self) -> List[str]:
        return sorted({rec["subject_id"] for rec in self._manifest})

    @property
    def n_recordings(self) -> int:
        return len(self._manifest)

    def binary_labels(self) -> np.ndarray:
        """Return a binary label per window.

        Requires materialising each recording once to read its samplewise
        label stream; expensive but mirrors the contract of
        ``TUSZContinuousDataset.binary_labels`` so callers that pre-compute
        class weights (e.g. ``WeightedRandomSampler`` setup) keep working.
        """
        out = np.zeros(len(self._windows), dtype=np.int64)
        for i, (rec_idx, local_offset) in enumerate(self._windows):
            _sig, sw = self._load_recording(rec_idx)
            s0 = local_offset
            s1 = s0 + self._window_samples
            out[i] = 1 if np.any(sw[s0:s1] > 0) else 0
        return out

    def all_labels(self) -> np.ndarray:
        return self.binary_labels()
