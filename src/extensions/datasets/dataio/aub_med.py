"""Lazy EDF-backed Dataset for the AUB-MED / Nasreddine Epileptic EEG corpus.

Reads EDFs lazily from the flat ``raw/`` directory with a per-recording
LRU signal cache.  Seizure labels are computed on-the-fly from the
clock-time intervals declared in ``annotations/Seizure_times.py``.  No
HDF5 preprocessing required.

AUB-MED specifics:

- 6 patients (p10, p11, p12, p13, p14, p15), ``pN_Record{K}.edf`` per file.
- Native 500 Hz, 21 scalp electrodes (10-20).  Resampled to 256 Hz on load.
- Seizure times stored as ``(hour, minute, second, duration_s)`` where
  ``(h, m, s)`` is *clock time* — the offset from recording start is
  computed against the EDF ``startdatetime``.
- Already bandpass-filtered upstream: HP at 1.6 s time constant
  (≈ 1/(2π·1.6) ≈ 0.1 Hz, Mendeley notation "1/1.6 Hz"), LP at 70 Hz,
  plus a 50 Hz notch.  No extra in-repo filtering is applied.

This is the direct-EDF counterpart to
:mod:`extensions.datasets.dataio.siena_bids`.
"""
from __future__ import annotations

import ast
import collections
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from extensions.datasets._recording_stats import compute_recording_stats
from extensions.datasets.epilepsy._common import (
    BIPOLAR_NAMES,
    CANONICAL_19,
    CANONICAL_IDX,
    CANONICAL_SET,
    TARGET_FS,
    bipolar_from_unipolar,
    resample_to,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel aliasing (inline so AUB-MED doesn't depend on siena_preprocessor)
# ---------------------------------------------------------------------------

_CHANNEL_ALIAS: Dict[str, str] = {
    # Already-canonical names pass through untouched
    **{ch: ch for ch in CANONICAL_19},
    # Titlecase variants seen in the Mendeley recordings
    "Fp1": "FP1", "Fp2": "FP2",
    "Fz": "FZ", "Cz": "CZ", "Pz": "PZ",
    "F3": "F3", "F4": "F4", "F7": "F7", "F8": "F8",
    "C3": "C3", "C4": "C4",
    "P3": "P3", "P4": "P4",
    "O1": "O1", "O2": "O2",
    # 10-10 → 10-20 nomenclature collapse
    "T7": "T3", "T8": "T4",
    "P7": "T5", "P8": "T6",
    "t7": "T3", "t8": "T4", "p7": "T5", "p8": "T6",
    # Lowercase fallbacks
    "fp1": "FP1", "fp2": "FP2", "fz": "FZ", "cz": "CZ", "pz": "PZ",
}

_NON_EEG_PREFIXES = (
    "ECG", "EOG", "EMG", "EKG", "BN", "MK", "DC", "PHOTO",
    "SPO2", "PULS", "BEAT", "STI", "A1", "A2",
)

_NATIVE_FS: float = 500.0

# MNE always returns SI units (Volts) from read_raw_edf, regardless of the
# physical dimension declared in the source file's header. Multiply by this
# constant to get µV. Documented in mne.io.Raw.get_data.
_MNE_V_TO_UV: float = 1e6

# Per-patient demographics transcribed from annotations/Seizures_Information.xlsx
# (6 pediatric patients p10..p15). age_years preserves the half-year resolution
# for p11; integer ``age`` is the floor for the shared cross-dataset field.
_AUB_MED_DEMOGRAPHICS: Dict[str, Dict[str, Any]] = {
    "p10": {"age": 5, "age_years": 5.0, "gender": "f"},
    "p11": {"age": 4, "age_years": 4.5, "gender": "f"},
    "p12": {"age": 8, "age_years": 8.0, "gender": "m"},
    "p13": {"age": 9, "age_years": 9.0, "gender": "m"},
    "p14": {"age": 7, "age_years": 7.0, "gender": "m"},
    "p15": {"age": 8, "age_years": 8.0, "gender": "m"},
}


# ---------------------------------------------------------------------------
# Annotation parsing (Seizure_times.py → intervals in seconds)
# ---------------------------------------------------------------------------


@dataclass
class AUBMedRecording:
    """Metadata for one AUB-MED EDF.

    Attributes:
        rec_index: Positional index inside the index.
        subject_id: ``"p10"`` / ``"p11"`` / ...
        recording_id: EDF stem (e.g. ``"p10_Record1"``).
        record_no: Integer suffix (e.g. ``1``) matching Seizure_times keys.
        edf_path: Absolute path to the EDF.
        n_samples: Post-resample (256 Hz) sample count.
        duration_s: Recording duration in seconds (from EDF header).
        native_fs: Sampling rate in the EDF (expected 500 Hz).
        seizure_intervals_s: Seizure onset/offset in seconds relative to
            recording start (post-resample coordinates).
        seizure_intervals_samples: Same intervals in samples at 256 Hz.
    """

    rec_index: int
    subject_id: str
    recording_id: str
    record_no: int
    edf_path: str
    n_samples: int
    duration_s: float
    native_fs: float
    seizure_intervals_s: List[Tuple[float, float]] = field(default_factory=list)
    seizure_intervals_samples: List[Tuple[int, int]] = field(default_factory=list)


def parse_seizure_times(path: str | Path) -> Dict[str, Dict[int, List[Tuple[int, int, int, int]]]]:
    """Parse ``Seizure_times.py`` into ``{subject_id: {record_no: [(h, m, s, dur_s), ...]}}``.

    The upstream file is pure Python consisting of module-level dict
    assignments named ``seizures_<pid>``.  We parse its AST and evaluate
    each dict literal in isolation with :func:`ast.literal_eval`, so no
    code is executed.
    """
    path = Path(path)
    out: Dict[str, Dict[int, List[Tuple[int, int, int, int]]]] = {}
    if not path.exists():
        logger.warning("AUB-MED Seizure_times.py not found: %s", path)
        return out

    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        m = re.fullmatch(r"seizures_(\d+)", name)
        if m is None:
            continue
        subject_id = f"p{m.group(1)}"
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError) as exc:
            logger.warning("Skipping %s in %s: %s", name, path, exc)
            continue
        if not isinstance(value, dict):
            continue
        cleaned: Dict[int, List[Tuple[int, int, int, int]]] = {}
        for rec_no, intervals in value.items():
            try:
                rec_int = int(rec_no)
            except (TypeError, ValueError):
                continue
            rec_list: List[Tuple[int, int, int, int]] = []
            for entry in intervals:
                if (
                    isinstance(entry, (list, tuple))
                    and len(entry) == 4
                    and all(isinstance(x, (int, float)) for x in entry)
                ):
                    rec_list.append(tuple(int(x) for x in entry))  # type: ignore[arg-type]
            cleaned[rec_int] = rec_list
        out[subject_id] = cleaned
    return out


# ---------------------------------------------------------------------------
# Recording discovery
# ---------------------------------------------------------------------------


_RECORD_NAME_RE = re.compile(r"^(p\d+)_Record(\d+)$", re.IGNORECASE)


def _clock_seconds_since(
    seizure_hms: Tuple[int, int, int],
    start_hms: Tuple[int, int, int],
) -> float:
    """Return seconds from recording start to seizure onset, handling midnight wrap."""
    sz = seizure_hms[0] * 3600 + seizure_hms[1] * 60 + seizure_hms[2]
    st = start_hms[0] * 3600 + start_hms[1] * 60 + start_hms[2]
    delta = sz - st
    if delta < 0:
        delta += 24 * 3600
    return float(delta)


def _probe_edf_header(path: str) -> Tuple[float, float, Tuple[int, int, int]]:
    """Return ``(native_fs, duration_s, (h, m, s))`` using :mod:`mne`.

    Upstream EDFs in this corpus are a mixture of strictly-compliant
    files (readable by pyedflib) and slightly-non-compliant ones that
    pyedflib rejects with "the file is not EDF(+) compliant".  MNE's
    reader is lenient enough to open all of them, so we use it as the
    single source of truth here — the API call is header-only, no
    sample data is decoded.
    """
    import mne

    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    native_fs = float(raw.info["sfreq"])
    duration_s = float(raw.n_times) / native_fs
    meas_date = raw.info.get("meas_date")
    if meas_date is None:
        start_hms = (0, 0, 0)
    else:
        start_hms = (int(meas_date.hour), int(meas_date.minute), int(meas_date.second))
    return native_fs, duration_s, start_hms


def discover_aub_med_recordings(
    raw_dir: str | Path,
    annotations_dir: Optional[str | Path] = None,
) -> List[AUBMedRecording]:
    """Scan ``raw_dir`` for ``pN_Record{K}.edf`` files and attach seizure intervals.

    The EDF header is read *once per file* here to obtain the
    ``startdatetime`` and duration; no sample data is decoded.

    Args:
        raw_dir: Directory containing the raw EDFs (flat layout).
        annotations_dir: Directory containing ``Seizure_times.py``.
            Defaults to ``<raw_dir>/../annotations``.

    Returns:
        List of :class:`AUBMedRecording` sorted by ``(subject_id, record_no)``.
    """
    raw_dir = Path(raw_dir)
    if annotations_dir is None:
        annotations_dir = raw_dir.parent / "annotations"
    annotations_dir = Path(annotations_dir)

    seizure_map = parse_seizure_times(annotations_dir / "Seizure_times.py")

    raw_paths: List[Path] = sorted(raw_dir.glob("*.edf"))
    if not raw_paths:
        logger.warning("AUB-MED raw_dir has no EDFs: %s", raw_dir)
        return []

    recordings: List[AUBMedRecording] = []
    for p in raw_paths:
        m = _RECORD_NAME_RE.match(p.stem)
        if m is None:
            logger.warning("Skipping EDF with unexpected name: %s", p.name)
            continue
        subject_id = m.group(1).lower()
        record_no = int(m.group(2))

        try:
            native_fs, duration_s, start_hms = _probe_edf_header(str(p))
        except Exception as exc:
            logger.warning("Failed to probe %s (%s) — skipping", p.name, exc)
            continue

        n_samples_at_target = int(round(duration_s * TARGET_FS))

        intervals_s: List[Tuple[float, float]] = []
        intervals_samples: List[Tuple[int, int]] = []
        for (h, mi, s, dur) in seizure_map.get(subject_id, {}).get(record_no, []):
            onset_s = _clock_seconds_since((h, mi, s), start_hms)
            offset_s = onset_s + float(dur)
            intervals_s.append((onset_s, offset_s))
            intervals_samples.append((
                max(0, int(round(onset_s * TARGET_FS))),
                min(n_samples_at_target, int(round(offset_s * TARGET_FS))),
            ))

        recordings.append(AUBMedRecording(
            rec_index=0,  # filled after sort
            subject_id=subject_id,
            recording_id=p.stem,
            record_no=record_no,
            edf_path=str(p),
            n_samples=n_samples_at_target,
            duration_s=duration_s,
            native_fs=native_fs,
            seizure_intervals_s=intervals_s,
            seizure_intervals_samples=intervals_samples,
        ))

    recordings.sort(key=lambda r: (r.subject_id, r.record_no))
    for i, r in enumerate(recordings):
        r.rec_index = i

    logger.info(
        "AUB-MED index: %d recordings, %d subjects, %d with seizures",
        len(recordings),
        len({r.subject_id for r in recordings}),
        sum(1 for r in recordings if r.seizure_intervals_s),
    )
    return recordings


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AUBMedDataset(Dataset):
    """Lazy windowed view over AUB-MED EDFs.

    Args:
        raw_dir: Directory containing ``pN_Record{K}.edf`` files.
        annotations_dir: Directory containing ``Seizure_times.py``.
            Defaults to ``<raw_dir>/../annotations``.
        window_s: Window duration in seconds.
        stride_s: Stride in seconds (default = window_s).
        recording_indices: Optional positional subset (used for folds).
        label_mode: Only ``"binary"`` is supported (seizure / background).
        normalize: ``"none"`` or ``"per_window_zscore"``.
        montage: ``"unipolar"`` (19 ch) or ``"bipolar"`` (18 ch).
        overlap_threshold: Seizure fraction above which label = 1.
        signal_cache_size: Max recordings in the LRU cache.
        recordings: Optional precomputed index (skips discovery + header read).
    """

    def __init__(
        self,
        raw_dir: str | Path,
        annotations_dir: Optional[str | Path] = None,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        recording_indices: Optional[Sequence[int]] = None,
        label_mode: str = "binary",
        normalize: str = "none",
        montage: str = "bipolar",
        overlap_threshold: float = 0.0,
        signal_cache_size: int = 8,
        recordings: Optional[Sequence[AUBMedRecording]] = None,
        **kwargs,
    ) -> None:
        if label_mode != "binary":
            raise ValueError(
                f"AUBMedDataset only supports label_mode='binary', got {label_mode!r}"
            )
        if montage not in ("unipolar", "bipolar"):
            raise ValueError(
                f"montage must be 'unipolar' or 'bipolar', got {montage!r}"
            )

        self._raw_dir = str(raw_dir)
        self._annotations_dir = (
            str(annotations_dir) if annotations_dir is not None
            else str(Path(raw_dir).parent / "annotations")
        )
        self._window_s = float(window_s)
        self._stride_s = float(stride_s) if stride_s is not None else float(window_s)
        self._label_mode = label_mode
        self._normalize = normalize
        self._montage = montage
        self._overlap_threshold = float(overlap_threshold)
        self._signal_cache_size = int(signal_cache_size)

        self._window_samples = int(round(self._window_s * TARGET_FS))
        self._stride_samples = max(1, int(round(self._stride_s * TARGET_FS)))
        # Per-recording q95 aligned to the emitted channels — populated
        # lazily by _load_signals, consumed by BIOT anontier. Length matches
        # the emitted channel count (19 for unipolar, 20 for bipolar).
        self._rec_q95_cache: Dict[int, np.ndarray] = {}
        # Per-recording mean/std/q95-bipolar stats aligned to the emitted
        # channels. Populated lazily alongside _rec_q95_cache; consumed by
        # SleepFM/REVE/BIOT (recording-level z-score / scaling). Same dict
        # shape as _common._recording_stats.compute_recording_stats.
        self._rec_stats_cache: Dict[int, Dict[str, Any]] = {}

        if recordings is None:
            recordings = discover_aub_med_recordings(
                self._raw_dir, self._annotations_dir,
            )
        self._recordings: List[AUBMedRecording] = list(recordings)
        if not self._recordings:
            raise RuntimeError(
                f"AUBMedDataset: no recordings discovered under {self._raw_dir}. "
                f"Run `python -m entrypoints.fetch --dataset aub_med --download` to fetch the data."
            )

        if recording_indices is None:
            active = set(range(len(self._recordings)))
        else:
            active = {int(i) for i in recording_indices}

        # Window index: (rec_index, window_start_sample_at_256Hz)
        windows: List[Tuple[int, int]] = []
        for rec in self._recordings:
            if rec.rec_index not in active:
                continue
            if rec.n_samples < self._window_samples:
                continue
            pos = 0
            while pos + self._window_samples <= rec.n_samples:
                windows.append((rec.rec_index, pos))
                pos += self._stride_samples

        self._windows = windows
        self._active_rec_indices = sorted(active & {r.rec_index for r in self._recordings})

        self._signal_cache: "collections.OrderedDict[int, np.ndarray]" = (
            collections.OrderedDict()
        )

        if not self._windows:
            raise RuntimeError(
                f"AUBMedDataset produced 0 windows for window_s={self._window_s}. "
                f"Check that active recordings are long enough "
                f"({self._window_samples} samples @ 256 Hz)."
            )

        logger.info(
            "AUBMedDataset: %s — %d recordings, %d windows "
            "(%.1fs @ %.1fs stride, montage=%s)",
            self._raw_dir,
            len(self._active_rec_indices),
            len(self._windows),
            self._window_s, self._stride_s, self._montage,
        )

    # ------------------------------------------------------------------
    # Signal loading with channel aliasing + on-the-fly resample
    # ------------------------------------------------------------------

    def _load_signals(self, rec_index: int) -> np.ndarray:
        """Return ``(19, n_samples_at_256Hz)`` float32.

        Missing canonicals zero-filled.  Uses the LRU cache.  Reads via
        :mod:`mne` since a subset of the upstream AUB-MED files are
        non-compliant EDFs that pyedflib refuses to open.
        """
        if rec_index in self._signal_cache:
            self._signal_cache.move_to_end(rec_index)
            return self._signal_cache[rec_index]

        import mne

        rec = self._recordings[rec_index]
        raw = mne.io.read_raw_edf(rec.edf_path, preload=True, verbose="ERROR")
        try:
            ch_names = list(raw.ch_names)
            native_fs = float(raw.info["sfreq"])

            ch_map: Dict[str, int] = {}
            for i, raw_label in enumerate(ch_names):
                name = (
                    raw_label
                    .replace("EEG ", "")
                    .replace("eeg ", "")
                    .replace("ECG ", "")
                )
                name = name.split("-")[0].strip()
                name_upper = name.upper()
                if any(name_upper.startswith(p) for p in _NON_EEG_PREFIXES):
                    continue
                canonical = _CHANNEL_ALIAS.get(name)
                if canonical is None:
                    canonical = _CHANNEL_ALIAS.get(name_upper)
                if canonical is None:
                    continue
                if canonical in CANONICAL_SET and canonical not in ch_map:
                    ch_map[canonical] = i

            # MNE always returns SI units (Volts) regardless of what the raw
            # EDF-ish file declared. AUB-MED files are non-strict EDFs that
            # pyedflib refuses, so MNE is our only reader. The V → µV scale
            # is therefore fixed at 1e6 by MNE's documented convention.
            data_v = raw.get_data()  # (n_ch, n_samples) float64
            n_native = data_v.shape[1]
            data_uv = data_v.astype(np.float32, copy=False) * _MNE_V_TO_UV

            native = np.zeros((len(CANONICAL_19), n_native), dtype=np.float32)
            for canonical_name, edf_idx in ch_map.items():
                native[CANONICAL_IDX[canonical_name]] = data_uv[edf_idx]
        finally:
            # mne.Raw holds a file handle open until GC; close explicitly
            # where the public API allows.
            try:
                raw.close()
            except AttributeError:
                pass

        signals_256 = resample_to(native, native_fs, float(TARGET_FS))

        # Align length with header-derived rec.n_samples (guards against
        # polyphase length jitter when duration_s isn't an exact multiple).
        target_n = rec.n_samples
        if signals_256.shape[1] != target_n:
            padded = np.zeros((len(CANONICAL_19), target_n), dtype=np.float32)
            copy_len = min(signals_256.shape[1], target_n)
            padded[:, :copy_len] = signals_256[:, :copy_len]
            signals_256 = padded

        while len(self._signal_cache) >= self._signal_cache_size:
            self._signal_cache.popitem(last=False)
        self._signal_cache[rec_index] = signals_256
        # Populate recording stats (q95, mean, std, q95-bipolar) aligned
        # to the emitted channels. Mean/std are required by SleepFM/REVE
        # auto-inject; q95 by BIOT anontier. Both share the same source
        # array, so compute once.
        if rec_index not in self._rec_q95_cache:
            src = signals_256  # (19, T)
            if self._montage == "bipolar":
                src = bipolar_from_unipolar(src)  # (20, T)
            self._rec_q95_cache[rec_index] = np.quantile(
                np.abs(src), 0.95, axis=1,
            ).astype(np.float32)
            self._rec_stats_cache[rec_index] = compute_recording_stats(src)
        return signals_256

    # ------------------------------------------------------------------
    # Label computation
    # ------------------------------------------------------------------

    def _compute_seizure_label(
        self, rec_index: int, window_start: int, window_end: int,
    ) -> Tuple[int, float]:
        rec = self._recordings[rec_index]
        window_len = window_end - window_start
        overlap = 0
        for sz_start, sz_end in rec.seizure_intervals_samples:
            o = max(0, min(window_end, sz_end) - max(window_start, sz_start))
            overlap += o
        fraction = overlap / max(window_len, 1)
        label = int(fraction > self._overlap_threshold)
        return label, float(fraction)

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_index, window_start = self._windows[idx]
        window_end = window_start + self._window_samples
        rec = self._recordings[rec_index]

        all_signals = self._load_signals(rec_index)
        signals = all_signals[:, window_start:window_end].copy()  # (19, T)

        if self._montage == "bipolar":
            signals = bipolar_from_unipolar(signals)  # (18, T)

        if self._normalize == "per_window_zscore":
            mean = signals.mean(axis=1, keepdims=True)
            std = signals.std(axis=1, keepdims=True)
            std[std < 1e-8] = 1.0
            signals = (signals - mean) / std

        label, seizure_fraction = self._compute_seizure_label(
            rec_index, window_start, window_end,
        )

        demo = _AUB_MED_DEMOGRAPHICS.get(rec.subject_id, {})
        meta: Dict[str, Any] = {
            "dataset": "aub_med",
            "split": "",
            "subject_id": rec.subject_id,
            "recording_id": rec.recording_id,
            "recording_idx": rec_index,
            "record_no": rec.record_no,
            "window_start_s": float(window_start / TARGET_FS),
            "window_end_s": float(window_end / TARGET_FS),
            "recording_duration_s": float(rec.duration_s),
            "seizure_fraction": seizure_fraction,
            "has_seizure": bool(label),
            "binary_label": int(label),
            "n_channels": int(signals.shape[0]),
            "channels": list(
                BIPOLAR_NAMES if self._montage == "bipolar" else CANONICAL_19
            ),
            "montage": self._montage,
            "normalize": self._normalize,
            "fs": TARGET_FS,
            "sampling_rate": float(TARGET_FS),
            "units": "uV",
            "unit": "uV",
            "native_fs": float(rec.native_fs),
            "window_s": self._window_s,
            "stride_s": self._stride_s,
            "source": "edf",
            "recording_q95": self._rec_q95_cache[rec_index],
            "age": demo.get("age", -1),
            "age_years": demo.get("age_years", -1.0),
            "gender": demo.get("gender", ""),
        }
        # Per-recording z-score stats (recording_mean / recording_std / etc.)
        # are required by SleepFM/REVE wrappers (auto-inject path in
        # benchmarking_helpers/runner.py — paper's recipe has no per-window
        # fallback). Mirrors the chbmit / helsinki dataio pattern.
        meta.update(self._rec_stats_cache[rec_index])

        return {
            "eeg": torch.from_numpy(signals.astype(np.float32, copy=False)),
            "label": int(label),
            "meta": meta,
        }

    # -- Accessors ----------------------------------------------------------

    @property
    def subject_ids(self) -> List[str]:
        return sorted({
            self._recordings[i].subject_id for i in self._active_rec_indices
        })

    @property
    def n_recordings(self) -> int:
        return len(self._active_rec_indices)

    @property
    def recordings(self) -> List[AUBMedRecording]:
        return list(self._recordings)

    def binary_labels(self) -> np.ndarray:
        labels = np.zeros(len(self._windows), dtype=np.int64)
        for i, (rec_index, window_start) in enumerate(self._windows):
            window_end = window_start + self._window_samples
            label, _ = self._compute_seizure_label(rec_index, window_start, window_end)
            labels[i] = label
        return labels

    def recording_sizes_bytes(self) -> List[Tuple[int, int]]:
        """Return ``(rec_index, estimated_float32_bytes)`` for active recordings.

        Uses post-resample (256 Hz) sample counts from the recording index.
        """
        return [
            (i, 19 * self._recordings[i].n_samples * 4)
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
        for i in tqdm(rec_indices, desc="preload AUB-MED shard", unit="rec", leave=False):
            if i not in self._signal_cache:
                cold += 1
            self._load_signals(i)

        elapsed = time.perf_counter() - t0
        rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logger.info(
            "preload_shard: %d recordings (%d cold decodes), %.2fs, RSS +%.1f MB",
            n, cold, elapsed, (rss1 - rss0) / 1024.0,
        )

    def targets(self) -> np.ndarray:
        return self.binary_labels()

    @property
    def n_classes(self) -> int:
        return 2


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate_aub_med(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    eeg = torch.stack([item["eeg"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "eeg": eeg,
        "labels": labels,
        "meta": [item["meta"] for item in batch],
    }
