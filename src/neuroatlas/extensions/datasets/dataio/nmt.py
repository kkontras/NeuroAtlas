"""Dataset class for the NMT Scalp EEG Dataset (NUST-SEECS).

Reads raw EDFs lazily on demand.  No preprocessing step and no HDF5
cache: a fresh checkout that has only fetched the corpus can use
the dataset immediately.

Layout expected at ``raw_root``::

    raw_root/
      Labels.csv                (optional — demographic enrichment)
      abnormal/train/*.edf
      abnormal/eval/*.edf
      normal/train/*.edf
      normal/eval/*.edf

Native sampling rate is 200 Hz; :func:`read_edf_unipolar` resamples each
recording to ``TARGET_FS=256 Hz`` on first access and the result is held
in an LRU cache, so the 200→256 cost is amortised across all windows of
a given recording.  Channel aliasing (``T7→T3`` etc.) and missing-channel
zero-fill are also handled by the shared reader.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from pathlib import Path as _Path

from neuroatlas.extensions.datasets._recording_stats import (
    compute_recording_stats,
    load_cached_recording_stats,
    load_recording_stats,
    save_recording_stats,
)

# Per-recording stats live in <cache root>/recording_stats/nmt/
# (_paths.recording_stats_dir); they used to go to
# artifacts/recording_stats/nmt under the current directory.
_STATS_READER = "nmt"
from neuroatlas.extensions.datasets.epilepsy.tusz.readers import (
    BIPOLAR_MONTAGE,
    TARGET_FS,
    UNIPOLAR_ELECTRODES,
    bipolar_from_unipolar,
    read_edf_unipolar,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Window index helper (shared shape with TUAB)
# ---------------------------------------------------------------------------


def _build_window_index(
    durations_samples: Sequence[int],
    labels: Sequence[int],
    window_samples: int,
    stride_samples: int,
    recording_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Return ``(N, 3) int64`` array of ``(rec_idx, start_sample, label)``.

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
# Labels.csv parsing
# ---------------------------------------------------------------------------


def _load_labels_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    """Parse upstream ``Labels.csv``.

    Columns: ``recordname, label, age, gender, loc``.  Returns a mapping
    ``{edf_basename: {"label": "normal"|"abnormal",
                      "age": int | None,
                      "gender": str,
                      "loc": "train"|"eval"}}``.

    Missing or unparseable rows are skipped with a warning.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with open(path, "r") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row.get("recordname") or "").strip()
            if not name:
                continue
            if not name.lower().endswith(".edf"):
                name = f"{name}.edf"
            age_raw = (row.get("age") or "").strip()
            try:
                age: Optional[int] = int(age_raw) if age_raw else None
            except ValueError:
                age = None
            raw_gender = (row.get("gender") or "").strip().lower()
            if raw_gender.startswith("m"):
                gender = "m"
            elif raw_gender.startswith("f"):
                gender = "f"
            else:
                gender = ""
            out[name] = {
                "label": (row.get("label") or "").strip().lower(),
                "age": age,
                "gender": gender,
                "loc": (row.get("loc") or "").strip().lower(),
            }
    return out


# ---------------------------------------------------------------------------
# Recording discovery
# ---------------------------------------------------------------------------


def discover_nmt_recordings(
    raw_root: str,
    splits: Sequence[str] = ("train", "eval"),
) -> List[Dict[str, Any]]:
    """Walk ``raw_root/{abnormal,normal}/{train,eval}/*.edf``.

    Returns a list of dicts with keys
    ``edf_path, split, is_abnormal, subject_id, recording_id, age, gender``.

    ``subject_id`` == ``recording_id`` (== EDF stem) because each NMT EDF
    is one recording per unique participant.  The file stem (e.g.
    ``0000001``) is used as the subject identifier so that subject-
    disjoint k-fold splitting works out of the box.

    ``age`` and ``gender`` come from ``Labels.csv`` when available;
    otherwise they are ``None`` / ``""``.  The filesystem location is the
    authoritative source of ``split`` and ``is_abnormal``.
    """
    raw_root_p = Path(raw_root)
    labels = _load_labels_csv(raw_root_p / "Labels.csv")

    out: List[Dict[str, Any]] = []
    for label_name, is_abnormal in (("abnormal", 1), ("normal", 0)):
        for split in splits:
            split_dir = raw_root_p / label_name / split
            if not split_dir.exists():
                logger.warning(
                    "NMT %s/%s directory missing, skipping: %s",
                    label_name, split, split_dir,
                )
                continue
            for edf_path in sorted(split_dir.glob("*.edf")):
                # Skip macOS AppleDouble metadata stubs (._<name>.edf) that
                # ship inside zip imports made on a Mac. They are 4 KB
                # binary blobs (magic 0x00051607), not EDFs; pyedflib would
                # spend ~50 ms each rejecting them, ~1090 of them on first
                # import = ~1 min wasted on every run.
                if edf_path.name.startswith("._"):
                    continue
                meta = labels.get(edf_path.name, {})
                # Cross-check label if Labels.csv disagrees with the layout
                if meta and meta.get("label") and meta["label"] != label_name:
                    logger.warning(
                        "NMT Labels.csv disagrees with layout for %s "
                        "(csv=%s vs layout=%s) — trusting layout",
                        edf_path.name, meta["label"], label_name,
                    )
                out.append({
                    "edf_path": str(edf_path),
                    "split": split,
                    "is_abnormal": int(is_abnormal),
                    "subject_id": edf_path.stem,
                    "recording_id": edf_path.stem,
                    "age": meta.get("age"),
                    "gender": meta.get("gender", ""),
                })
    logger.info(
        "Discovered %d NMT recordings under %s (splits=%s)",
        len(out), raw_root, list(splits),
    )
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Empirical rescale to match reference scalp EEG amplitudes. NMT EDFs
# declare unit "uV" but the encoded physical_min/max is ~100× too small,
# so pyedflib correctly delivers values ~100× below a typical EEG µV
# scale (NMT std ≈ 0.3 µV vs Siena std ≈ 37 µV). Applied once in
# ``_get_decoded`` so every backbone/model sees consistent amplitudes.
_NMT_AMPLITUDE_RESCALE = 100.0


class NMTEdfDataset(Dataset):
    """Lazy windowed view over raw NMT EDFs.

    Each ``__getitem__`` call loads the parent recording into an LRU
    cache (decoded to ``(n_samples_at_256Hz, 19)`` float32) and slices
    out one window.

    Args:
        recordings: Output of :func:`discover_nmt_recordings`, optionally
            pre-filtered.
        window_s: Window length in seconds (default 30 s).
        stride_s: Stride between consecutive windows in the same
            recording (default = ``window_s``, non-overlapping).
        normalize: ``"none"`` (default) or ``"per_window_zscore"``.
        recording_indices: Optional positional subset of ``recordings``
            (used for fold-level slicing without re-scanning).
        cache_max_recordings: LRU size — maximum fully-decoded
            recordings held in memory (default 16).
    """

    # Class-level flag so the ×100 rescale banner is logged once per
    # process even with many concurrent dataloader workers / instances.
    _rescale_banner_logged: bool = False

    def __init__(
        self,
        recordings: Sequence[Dict[str, Any]],
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        normalize: str = "none",
        montage: str = "unipolar",
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
        self._lru_cache: "Dict[int, np.ndarray]" = {}
        self._lru_order: List[int] = []
        # Per-recording amplitude stats aligned to the emitted channels —
        # populated lazily in _get_decoded. Each entry is the dict returned
        # by compute_recording_stats: recording_mean, recording_std,
        # recording_q95, recording_q95_bipolar (when C>1). Consumed by BIOT
        # (q95), REVE (mean+std), SleepFM (mean+std). Vector length matches
        # UNIPOLAR_ELECTRODES (19) when montage='unipolar', BIPOLAR_MONTAGE
        # (20) when montage='bipolar'.
        self._rec_stats_cache: Dict[int, Dict[str, Any]] = {}

        recs = list(recordings)
        positions = list(range(len(recs)))
        if recording_indices is not None:
            keep = set(int(i) for i in recording_indices)
            positions = [i for i in positions if i in keep]
            recs = [recs[i] for i in positions]
        # Each kept recording's position in ``recordings`` -- the index the
        # adapter's folds are written in -- published as meta["recording_idx"]
        # so a global embedding cache can be split into folds.
        self._source_positions: List[int] = positions

        if not recs:
            raise RuntimeError(
                "NMTEdfDataset has zero recordings after filtering — "
                "check raw_root and recording_indices."
            )
        self._recordings: List[Dict[str, Any]] = recs

        self._durations_samples: List[int] = self._probe_durations(recs)
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
                f"NMTEdfDataset produced 0 windows for "
                f"window_s={window_s}, stride_s={self._stride_s}. "
                f"All {len(recs)} recordings are shorter than "
                f"{self._window_samples} samples (~{window_s:.1f}s)?"
            )

        logger.info(
            "NMTEdfDataset: %d recordings, %d windows "
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
                    "Failed to read NMT EDF header %s: %s — 0 samples",
                    r["edf_path"], exc,
                )
                durations.append(0)
        return durations

    # ------------------------------------------------------------------
    # In-memory LRU cache of decoded recordings
    # ------------------------------------------------------------------

    def _get_decoded(self, rec_idx: int) -> np.ndarray:
        if rec_idx in self._lru_cache:
            self._lru_order.remove(rec_idx)
            self._lru_order.append(rec_idx)
            return self._lru_cache[rec_idx]
        signals, _native_fs, _n_missing, _physical_units = read_edf_unipolar(
            self._recordings[rec_idx]["edf_path"],
        )
        # NMT-specific physical_min/max is off by ~100×: the EDF declares
        # unit "uV" but the scaled digital→physical conversion delivers
        # values with std ≈ 0.3 µV (vs Siena's ≈40 µV). Rescaling by 100×
        # brings NMT in line with reference scalp EEG (std ≈ 26 µV, p99 ≈
        # 86 µV per sanity check on 5 recordings). Applied here so every
        # model sees consistent amplitudes regardless of wrapper.
        signals = signals * _NMT_AMPLITUDE_RESCALE
        if not type(self)._rescale_banner_logged:
            logger.info(
                "NMT: rescaling signals by ×%g to correct EDF metadata "
                "(physical_min/max off by ~100× in the upstream distribution).",
                _NMT_AMPLITUDE_RESCALE,
            )
            type(self)._rescale_banner_logged = True
        self._lru_cache[rec_idx] = signals
        self._lru_order.append(rec_idx)
        while len(self._lru_order) > self._cache_max:
            evict = self._lru_order.pop(0)
            self._lru_cache.pop(evict, None)
        # Populate the emitted-channel-aligned recording stats so BIOT
        # (q95), REVE / SleepFM (mean+std) all have values keyed the same
        # way meta["channels"] is keyed. compute_recording_stats expects
        # (C, T); NMT signals carry (T, C), so transpose at compute time.
        # Disk cache by montage so unipolar and bipolar variants both persist.
        if rec_idx not in self._rec_stats_cache:
            rec_id = self._recordings[rec_idx]["recording_id"]
            stats_file = f"{rec_id}_{self._montage}.npz"
            fingerprint = {
                "target_fs": int(TARGET_FS),
                "n_samples": int(signals.shape[0]),
                "montage": self._montage,
                "amplitude_rescale": float(_NMT_AMPLITUDE_RESCALE),
            }
            stats, disk_path = load_cached_recording_stats(_STATS_READER, stats_file, fingerprint)
            if stats is None:
                src = signals  # (T, 19)
                if self._montage == "bipolar":
                    src = bipolar_from_unipolar(src)  # (T, 20)
                stats = compute_recording_stats(src.T)
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
            "dataset": "nmt",
            "recording_id": rec["recording_id"],
            "recording_idx": self._source_positions[rec_idx],
            "subject_id": rec["subject_id"],
            "split": rec["split"],
            "is_abnormal": int(rec["is_abnormal"]),
            "age": rec.get("age"),
            "gender": rec.get("gender", ""),
            "window_start_s": float(start / TARGET_FS),
            "window_end_s": float(end / TARGET_FS),
            "window_s": self._window_s,
            "stride_s": self._stride_s,
            "fs": TARGET_FS,
            "sampling_rate": float(TARGET_FS),
            "units": "uV",
            "unit": "uV",
            "channels": channels,
            "montage": self._montage,
            "source": "edf",
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
# Collate
# ---------------------------------------------------------------------------


def collate_nmt(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    eeg = torch.stack([item["eeg"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "eeg": eeg,
        "labels": labels,
        "meta": [item["meta"] for item in batch],
    }
