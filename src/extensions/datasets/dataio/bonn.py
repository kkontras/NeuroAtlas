"""Dataset class for the Bonn EEG epilepsy HDF5 cache.

The cache is flat and pre-segmented (see ``bonn_preprocessor.build_h5_cache``).
This Dataset reads it lazily and supports:

- configurable label mapping (``label_mode``: 5-class / 3-class / binary_s_vs_rest
  / binary_ae / recording_type / state)
- variable-length sub-windows within each 23.6-s clip
  (``window_s``, ``stride_s``; default = full clip)
- per-window metadata including the original clip's auxiliary fields
  (subset / recording_type / subject_type / state / electrode_location)

Subject-independence caveat
---------------------------
The public Bonn distribution (Andrzejak et al. 2001) does not ship per-trial
subject identifiers, so ``meta["subject_idx"] == -1`` for every clip in the
cache. As a result, subject-grouped cross-validation
(``StratifiedGroupKFold`` etc.) collapses — all 500 clips land in a single
group. Every published Bonn benchmark uses plain ``StratifiedKFold`` on
windows (within-subject leakage accepted as unavoidable). Downstream probe
code must key on ``dataset == "bonn"`` and skip groups, or exclude Bonn
from subject-independent tables. Not a bug in the cache — an artifact of
the source corpus that cannot be fixed here.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from extensions.datasets._recording_stats import compute_recording_stats
from extensions.datasets.epilepsy.bonn_preprocessor import (
    NATIVE_FS,
    SAMPLES_PER_CLIP,
    SET_ORDER,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label-mode handling
# ---------------------------------------------------------------------------


#: Every label mode maps the 5 class codes (Z=0, O=1, N=2, F=3, S=4)
#: to a new target and optionally filters which class codes are allowed.
_LabelMode = Tuple[Dict[int, int], Optional[frozenset], int]


def _label_mode_spec(label_mode: str) -> _LabelMode:
    """Return (code_map, keep_codes_or_None, n_classes) for a named mode.

    ``keep_codes_or_None`` restricts which class codes are kept in the index;
    ``None`` means "keep all".
    """
    lm = label_mode.lower()
    if lm in ("binary_s_vs_rest", "binary_abcd_vs_e"):
        return {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}, None, 2
    if lm == "binary_ae":
        return {0: 0, 4: 1}, frozenset({0, 4}), 2
    if lm == "three_class":
        return {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}, None, 3
    if lm == "five_class":
        return {i: i for i in range(5)}, None, 5
    if lm == "recording_type":
        # surface (Z, O) -> 0, intracranial (N, F, S) -> 1
        return {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}, None, 2
    if lm == "state":
        # awake_eyes_open (Z) -> 0, awake_eyes_closed (O) -> 1,
        # interictal (N, F) -> 2, ictal (S) -> 3
        return {0: 0, 1: 1, 2: 2, 3: 2, 4: 3}, None, 4
    # -- Controlled comparisons (remove modality confound) ----------------
    if lm == "lateralization":
        # N (contralateral hippocampus) -> 0, F (epileptogenic zone) -> 1
        # Same patients, same time, different electrode location.
        return {2: 0, 3: 1}, frozenset({2, 3}), 2
    if lm == "seizure_intracranial":
        # N+F (interictal) -> 0, S (ictal) -> 1
        # Same patients, same electrode type, different brain state.
        return {2: 0, 3: 0, 4: 1}, frozenset({2, 3, 4}), 2
    if lm == "eyes_open_closed":
        # Z (eyes open) -> 0, O (eyes closed) -> 1
        # Same healthy subjects, surface EEG.
        return {0: 0, 1: 1}, frozenset({0, 1}), 2
    raise ValueError(
        f"Unknown label_mode {label_mode!r}. Expected one of: "
        "binary_s_vs_rest, binary_ae, three_class, five_class, "
        "recording_type, state, lateralization, seizure_intracranial, "
        "eyes_open_closed."
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BonnSegmentDataset(Dataset):
    """Windowed view of the Bonn pre-segmented HDF5 cache.

    Each item is a sub-window of one clip, in ``(C=1, T)`` layout.

    Args:
        h5_path: Path to the HDF5 cache written by
            ``bonn_preprocessor.build_h5_cache``.
        window_s: Window length in seconds.  ``None`` (default) means
            "use the full 23.6-s clip — one window per clip".
        stride_s: Stride in seconds between consecutive windows in the
            same clip.  ``None`` (default) means non-overlapping
            (= ``window_s``).  Ignored when ``window_s is None``.
        label_mode: One of ``binary_s_vs_rest`` (default), ``binary_ae``,
            ``three_class``, ``five_class``, ``recording_type``, ``state``.
        clip_indices: Optional restriction to a subset of clip indices
            (before label-mode filtering).  Used for k-fold splitting.
        normalize: ``"none"`` (default) or ``"per_window_zscore"``.
    """

    def __init__(
        self,
        h5_path: Optional[str] = None,
        raw_dir: Optional[str] = None,
        window_s: Optional[float] = None,
        stride_s: Optional[float] = None,
        label_mode: str = "binary_s_vs_rest",
        clip_indices: Optional[Sequence[int]] = None,
        normalize: str = "none",
    ) -> None:
        # Assigned early so ``__del__`` never AttributeErrors if the
        # constructor bails out before the HDF5 handle is opened.
        self._h5: Optional[h5py.File] = None
        # Bonn is 500 text clips of 4097 samples -- about 8 MB decoded -- so
        # the raw corpus is read straight into memory and the HDF5 cache is a
        # fast path, not a prerequisite. `build_h5_cache` writes exactly what
        # `load_raw_corpus` returns, so the two cannot disagree.
        self._raw_dir = str(raw_dir) if raw_dir else None
        self._corpus = None
        if self._raw_dir is not None:
            from extensions.datasets.epilepsy import (
                bonn_preprocessor,
            )

            self._corpus = bonn_preprocessor.load_raw_corpus(self._raw_dir)
        elif not h5_path:
            raise ValueError(
                "Bonn needs either raw_dir (the directory holding the Z/O/N/F/S "
                "subdirectories of .TXT clips) or h5_path (a prebuilt cache). "
                "Run `fetch --dataset bonn` to get the corpus."
            )
        self._h5_path = str(h5_path) if h5_path else None
        self._label_mode_name = label_mode
        self._label_mode = _label_mode_spec(label_mode)
        self._normalize = normalize
        self._window_s_config = window_s  # None => full clip
        self._stride_s_config = stride_s

        # Derived window sizes (in samples)
        if window_s is None:
            self._window_samples = SAMPLES_PER_CLIP
            self._stride_samples = SAMPLES_PER_CLIP
            self._window_s_actual = SAMPLES_PER_CLIP / NATIVE_FS
            self._stride_s_actual = self._window_s_actual
        else:
            self._window_samples = int(round(window_s * NATIVE_FS))
            stride = stride_s if stride_s is not None else window_s
            self._stride_samples = max(1, int(round(stride * NATIVE_FS)))
            self._window_s_actual = float(window_s)
            self._stride_s_actual = float(stride)
            if self._window_samples < 1:
                raise ValueError(
                    f"window_s={window_s} is too small: rounds to "
                    f"{self._window_samples} samples at {NATIVE_FS} Hz."
                )
            if self._window_samples > SAMPLES_PER_CLIP:
                raise ValueError(
                    f"window_s={window_s} requires {self._window_samples} samples "
                    f"but each Bonn clip is only {SAMPLES_PER_CLIP} samples "
                    f"(~{SAMPLES_PER_CLIP / NATIVE_FS:.2f} s)."
                )

        # --- Load metadata (not signals) eagerly ---
        with self._open_metadata() as f:
            self._class_label = f["class_label"][:].astype(np.int64)
            self._subset = _decode_vlen(f["subset"][:])
            self._subject_idx = f["subject_idx"][:].astype(np.int16)
            self._recording_type = _decode_vlen(f["recording_type"][:])
            self._subject_type = _decode_vlen(f["subject_type"][:])
            self._state = _decode_vlen(f["state"][:])
            self._electrode_location = _decode_vlen(f["electrode_location"][:])

        code_map, keep_codes, n_classes = self._label_mode
        self._n_classes = n_classes

        n_clips = len(self._class_label)
        candidate_clips = range(n_clips) if clip_indices is None else clip_indices

        # Build (clip_idx, start_sample, target_label) index
        index: List[Tuple[int, int, int]] = []
        for ci in candidate_clips:
            code = int(self._class_label[ci])
            if keep_codes is not None and code not in keep_codes:
                continue
            target = code_map.get(code)
            if target is None:
                # This mode doesn't define a target for this class — skip
                continue
            start = 0
            # Non-overlapping default sub-windows; drop partial tails.
            while start + self._window_samples <= SAMPLES_PER_CLIP:
                index.append((ci, start, target))
                start += self._stride_samples
                if self._window_samples == SAMPLES_PER_CLIP:
                    # Full-clip mode — exactly one window per clip
                    break

        if not index:
            raise RuntimeError(
                f"BonnSegmentDataset produced 0 windows for label_mode={label_mode!r} "
                f"and {len(list(candidate_clips))} candidate clips."
            )

        self._index = np.array(index, dtype=np.int64)
        # Lazy per-recording amplitude-stats cache — see __getitem__ comment.
        self._rec_stats_cache: Dict[int, Dict[str, Any]] = {}

        logger.info(
            "BonnSegmentDataset: %s — label_mode=%s n_classes=%d "
            "window=%.3fs stride=%.3fs clips=%d windows=%d",
            self._h5_path, label_mode, n_classes,
            self._window_s_actual, self._stride_s_actual,
            len(set(int(r[0]) for r in index)), len(self._index),
        )

    # ------------------------------------------------------------------
    # HDF5 lazy handle
    # ------------------------------------------------------------------

    def _open_metadata(self):
        """Context manager over whichever source is in use."""
        import contextlib

        if self._corpus is not None:
            return contextlib.nullcontext(_InMemoryCorpus(self._corpus))
        return h5py.File(self._h5_path, "r")

    @property
    def h5(self):
        if self._corpus is not None:
            return _InMemoryCorpus(self._corpus)
        if self._h5 is None or not self._h5.id.valid:
            self._h5 = h5py.File(self._h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # Sample access
    # ------------------------------------------------------------------

    def _load_filtered_clip(self, clip_i: int) -> np.ndarray:
        """Load a full Bonn clip (1, SAMPLES_PER_CLIP) with 0.5 Hz HP +
        50 Hz notch (Bonn was recorded in Germany — 50 Hz mains). Cached
        per-clip so per-window slicing doesn't refilter.
        """
        cached = self._clip_cache.get(clip_i) if hasattr(self, "_clip_cache") else None
        if cached is not None:
            return cached
        from extensions.datasets.epilepsy._common import (
            apply_standard_filters,
        )
        clip = self.h5["signals"][clip_i, :, :].astype(np.float32)  # (1, T)
        if clip.shape[1] > 20:
            clip = apply_standard_filters(
                clip, fs=float(NATIVE_FS), notch_hz=50.0, axis=1,
            )
        if not hasattr(self, "_clip_cache"):
            import collections
            self._clip_cache = collections.OrderedDict()
        self._clip_cache[clip_i] = clip
        while len(self._clip_cache) > 64:
            self._clip_cache.popitem(last=False)
        return clip

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        clip_i = int(self._index[idx, 0])
        start = int(self._index[idx, 1])
        target = int(self._index[idx, 2])
        end = start + self._window_samples

        clip = self._load_filtered_clip(clip_i)
        sig = clip[:, start:end].astype(np.float32, copy=True)  # (1, T)

        if self._normalize == "per_window_zscore":
            mean = sig.mean(axis=1, keepdims=True)
            std = sig.std(axis=1, keepdims=True)
            std[std < 1e-8] = 1.0
            sig = (sig - mean) / std

        subset = self._subset[clip_i]
        meta: Dict[str, Any] = {
            "dataset": "bonn",
            "split": "",
            "clip_idx": clip_i,
            "subset": subset,
            "class_code": int(self._class_label[clip_i]),
            "class_name": SET_ORDER[int(self._class_label[clip_i])],
            "subject_idx": int(self._subject_idx[clip_i]),
            "recording_type": self._recording_type[clip_i],
            "subject_type": self._subject_type[clip_i],
            "state": self._state[clip_i],
            "electrode_location": self._electrode_location[clip_i],
            "window_start_s": float(start / NATIVE_FS),
            "window_end_s": float(end / NATIVE_FS),
            "window_s": self._window_s_actual,
            "stride_s": self._stride_s_actual,
            "fs": NATIVE_FS,
            "sampling_rate": float(NATIVE_FS),
            "units": "uV",
            "unit": "uV",
            "label_mode": self._label_mode_name,
            # Bonn is single-channel; no canonical 10-20 position is
            # distributed, so we claim a neutral midline electrode (FZ) so
            # that foundation-model wrappers that key position embeddings /
            # channel filters on 10-20 names accept the signal instead of
            # dropping it as unknown.
            "channels": ["FZ"],
        }

        # Per-recording amplitude stats so wrappers that need recording-level
        # normalisation (REVE z-score, BIOT q95, SleepFM recording-z) can pull
        # them from meta. Bonn "recordings" are 23.6 s single-channel clips —
        # not paper-faithful to SleepFM's whole-night statistic, but the only
        # statistic available; the wrapper warns about <30 s windows separately.
        if clip_i not in self._rec_stats_cache:
            self._rec_stats_cache[clip_i] = compute_recording_stats(clip)
        meta.update(self._rec_stats_cache[clip_i])

        return {
            "eeg": torch.from_numpy(sig),
            "label": target,
            "meta": meta,
        }

    # ------------------------------------------------------------------
    # Helpers for split / sampler construction
    # ------------------------------------------------------------------

    def targets(self) -> np.ndarray:
        """Return the target label of every window as a ``(N,)`` int64 array."""
        return self._index[:, 2].astype(np.int64)

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def active_clip_indices(self) -> List[int]:
        return sorted(set(int(r[0]) for r in self._index))

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


def _collate_bonn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    eeg = torch.stack([item["eeg"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "eeg": eeg,
        "labels": labels,
        "meta": [item["meta"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


class _InMemoryCorpus:
    """The subset of the h5py.File interface this dataio uses.

    Only ``corpus["name"][...]`` and ``.attrs`` are ever touched, so the raw
    path needs no parallel code in the dataset itself.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, key: str):
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    @property
    def attrs(self) -> dict:
        return self._data.get("attrs", {})


def _decode_vlen(arr: np.ndarray) -> List[str]:
    return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in arr]
