"""EDF-direct windowed Dataset for the Helsinki Neonatal EEG corpus.

Reads ``eeg{1..79}.edf`` lazily from ``raw_root`` with a per-recording LRU
signal cache.  Seizure labels are derived from the three expert MAT
annotations (``annotations_2017_{A,B,C}.mat``) via configurable consensus
(majority / unanimous / any).  Per-expert per-window binary labels and
seizure fractions are always stored in ``meta`` so inter-rater-agreement
studies can proceed without re-opening MAT files.
"""
from __future__ import annotations

import collections
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from pathlib import Path as _Path

from extensions.datasets._recording_stats import (
    compute_recording_stats,
    load_recording_stats,
    save_recording_stats,
)

_HELSINKI_STATS_CACHE_ROOT = _Path("artifacts/recording_stats/helsinki_neonatal")
from extensions.datasets.epilepsy._common import (
    BIPOLAR_NAMES,
    CANONICAL_19,
    TARGET_FS,
    bipolar_from_unipolar,
    window_binary_from_intervals,
)
from extensions.datasets.epilepsy.helsinki_neonatal_preprocessor import (
    EXPERT_IDS,
    consensus_per_second,
    discover_helsinki_recordings,
    load_helsinki_annotations,
    per_second_to_intervals_samples,
    read_edf_canonical19,
    subject_consensus_type,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recording metadata bundle
# ---------------------------------------------------------------------------


class _HelsinkiRecording:
    """Per-recording metadata + lazily-computed annotation intervals."""

    __slots__ = (
        "rec_index", "subject_id", "recording_id", "edf_path", "n_samples",
        "duration_s", "consensus_type",
        "intervals_collapsed", "intervals_per_expert",
        "gender", "bw_grams", "gestational_age_weeks",
        "eeg_to_pma_weeks", "diagnosis",
        "neuroimaging", "primary_localisation",
    )

    def __init__(
        self,
        rec_index: int,
        subject_id: int,
        edf_path: str,
        n_samples: int,
        duration_s: float,
        consensus_type: str,
        intervals_collapsed: List[Tuple[int, int]],
        intervals_per_expert: Dict[str, List[Tuple[int, int]]],
        gender: str = "",
        bw_grams: str = "",
        gestational_age_weeks: str = "",
        eeg_to_pma_weeks: str = "",
        diagnosis: str = "",
        neuroimaging: str = "",
        primary_localisation: str = "",
    ) -> None:
        self.rec_index = rec_index
        self.subject_id = subject_id
        self.recording_id = f"eeg{subject_id}"
        self.edf_path = edf_path
        self.n_samples = n_samples
        self.duration_s = duration_s
        self.consensus_type = consensus_type
        self.intervals_collapsed = intervals_collapsed
        self.intervals_per_expert = intervals_per_expert
        self.gender = gender
        self.bw_grams = bw_grams
        self.gestational_age_weeks = gestational_age_weeks
        self.eeg_to_pma_weeks = eeg_to_pma_weeks
        self.diagnosis = diagnosis
        self.neuroimaging = neuroimaging
        self.primary_localisation = primary_localisation


def _load_helsinki_clinical(raw_root: str) -> Dict[int, Dict[str, str]]:
    """Parse ``clinical_information.csv`` → ``{subject_id: {field: str}}``.

    The CSV is non-quantitative (values like "less than 2500g", "37 to 38")
    so we keep every field as a stripped string; ``"N/A"`` and blanks are
    normalised to ``""`` so downstream consumers can treat them uniformly
    as missing.
    """
    path = Path(raw_root) / "clinical_information.csv"
    if not path.exists():
        return {}

    def _clean(s: str) -> str:
        s = (s or "").strip()
        return "" if s.upper() in ("N/A", "NA", "NAN") else s

    out: Dict[int, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                sid = int(row["ID"])
            except (KeyError, ValueError):
                continue
            gender = _clean(row.get("Gender", "")).lower()
            out[sid] = {
                "gender": gender if gender in ("m", "f") else "",
                "bw_grams": _clean(row.get("BW (g)", "")),
                "gestational_age_weeks": _clean(row.get("GA (weeks)", "")),
                "eeg_to_pma_weeks": _clean(row.get("EEG to PMA (weeks)", "")),
                "diagnosis": _clean(row.get("Diagnosis", "")),
                "neuroimaging": _clean(row.get("Neuroimaging Findings", "")),
                "primary_localisation": _clean(row.get("Primary Localisation", "")),
            }
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class HelsinkiNeonatalEdfDataset(Dataset):
    """Windowed view over Helsinki neonatal EDFs with lazy reading.

    Args:
        raw_root: Directory containing ``eegN.edf`` + ``annotations_2017_{A,B,C}.mat``.
        window_s: Window duration in seconds.
        stride_s: Window stride in seconds.  ``None`` -> non-overlapping.
        recording_indices: Optional subset of recording indices.
        montage: ``"unipolar"`` (19 ch, default) or ``"bipolar"`` (18 ch TCP).
        normalize: ``"none"`` or ``"per_window_zscore"``.
        consensus: ``"majority"`` (≥2/3, default) / ``"unanimous"`` (3/3) / ``"any"``.
        overlap_threshold: Seizure fraction above which binary label is 1.
        signal_cache_size: LRU cache size (decoded recordings).
        include_partial_subjects: If ``False``, drop the Stevenson 2019 18
            mixed-agreement subjects at recording-index build time.  Default
            ``True`` (keep all 79 subjects).
    """

    def __init__(
        self,
        raw_root: str,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        recording_indices: Optional[Sequence[int]] = None,
        montage: str = "unipolar",
        normalize: str = "none",
        consensus: str = "majority",
        overlap_threshold: float = 0.0,
        signal_cache_size: int = 8,
        include_partial_subjects: bool = True,
        **kwargs: Any,
    ) -> None:
        if montage not in ("unipolar", "bipolar"):
            raise ValueError(f"montage must be 'unipolar' or 'bipolar', got {montage!r}")

        self._raw_root = str(raw_root)
        self._window_s = float(window_s)
        self._stride_s = float(stride_s) if stride_s is not None else float(window_s)
        self._montage = montage
        self._normalize = normalize
        self._consensus = consensus
        self._overlap_threshold = float(overlap_threshold)
        self._signal_cache_size = int(signal_cache_size)
        self._include_partial = bool(include_partial_subjects)

        self._window_samples = int(round(self._window_s * TARGET_FS))
        self._stride_samples = max(1, int(round(self._stride_s * TARGET_FS)))
        # Per-recording amplitude stats aligned to the emitted channels —
        # populated lazily by _load_signals. Each entry is the dict returned
        # by compute_recording_stats: recording_mean, recording_std,
        # recording_q95, recording_q95_bipolar (when C>1). Consumed by BIOT
        # (q95), REVE (mean+std), SleepFM (mean+std).
        self._rec_stats_cache: Dict[int, Dict[str, Any]] = {}

        # Discover recordings and load expert annotations once.
        discovered = discover_helsinki_recordings(raw_root)
        n_subjects = len(discovered)
        annotations = load_helsinki_annotations(raw_root, n_subjects=n_subjects)
        clinical = _load_helsinki_clinical(raw_root)

        self._recordings: List[_HelsinkiRecording] = []
        for rec_i, rec in enumerate(discovered):
            subject_id = rec["subject_id"]
            a = annotations["A"][subject_id - 1]
            b = annotations["B"][subject_id - 1]
            c = annotations["C"][subject_id - 1]
            ctype = subject_consensus_type(a, b, c)
            if not self._include_partial and ctype == "mixed":
                continue

            # n_samples from EDF header (cheap); resample accounted for.
            # 3 of 79 Helsinki files (eeg4/eeg29/eeg50) trip pyedflib's
            # strict header validator ("the label is incorrect", caused by
            # a numeric truncation in the per-channel phys_min field).
            # MNE reads them but the dataio is pyedflib-only, so skip them
            # with a warning rather than failing the whole dataset build.
            try:
                n_samples, duration_s = self._probe_edf(rec["edf_path"])
            except OSError as exc:
                logger.warning(
                    "Skipping Helsinki recording %s — pyedflib rejected header: %s",
                    rec["edf_path"], exc,
                )
                continue

            per_sec_collapsed = consensus_per_second(a, b, c, mode=self._consensus)
            intervals_collapsed = per_second_to_intervals_samples(per_sec_collapsed, fs=TARGET_FS)
            intervals_per_expert = {
                "A": per_second_to_intervals_samples(np.asarray(a) != 0, fs=TARGET_FS),
                "B": per_second_to_intervals_samples(np.asarray(b) != 0, fs=TARGET_FS),
                "C": per_second_to_intervals_samples(np.asarray(c) != 0, fs=TARGET_FS),
            }

            demo = clinical.get(subject_id, {})
            self._recordings.append(_HelsinkiRecording(
                rec_index=len(self._recordings),
                subject_id=subject_id,
                edf_path=rec["edf_path"],
                n_samples=n_samples,
                duration_s=duration_s,
                consensus_type=ctype,
                intervals_collapsed=intervals_collapsed,
                intervals_per_expert=intervals_per_expert,
                gender=demo.get("gender", ""),
                bw_grams=demo.get("bw_grams", ""),
                gestational_age_weeks=demo.get("gestational_age_weeks", ""),
                eeg_to_pma_weeks=demo.get("eeg_to_pma_weeks", ""),
                diagnosis=demo.get("diagnosis", ""),
                neuroimaging=demo.get("neuroimaging", ""),
                primary_localisation=demo.get("primary_localisation", ""),
            ))

        if recording_indices is not None:
            keep = set(int(i) for i in recording_indices)
        else:
            keep = set(range(len(self._recordings)))

        # Build window index: List[(rec_index, window_start_sample)]
        self._windows: List[Tuple[int, int]] = []
        for rec in self._recordings:
            if rec.rec_index not in keep:
                continue
            if rec.n_samples < self._window_samples:
                continue
            pos = 0
            while pos + self._window_samples <= rec.n_samples:
                self._windows.append((rec.rec_index, pos))
                pos += self._stride_samples

        self._active_rec_indices = sorted(keep & set(r.rec_index for r in self._recordings))
        self._signal_cache: "collections.OrderedDict[int, np.ndarray]" = collections.OrderedDict()

        logger.info(
            "HelsinkiNeonatalEdfDataset: %s — %d/%d recordings "
            "(consensus=%s, include_partial=%s), %d windows "
            "(%.1fs @ %.1fs stride, montage=%s)",
            self._raw_root, len(self._active_rec_indices), len(self._recordings),
            self._consensus, self._include_partial, len(self._windows),
            self._window_s, self._stride_s, self._montage,
        )

    # ------------------------------------------------------------------
    # EDF header probe + signal loader (LRU cached)
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_edf(edf_path: str) -> Tuple[int, float]:
        """Return ``(n_samples_at_TARGET_FS, duration_s)`` without reading signals."""
        import pyedflib
        reader = pyedflib.EdfReader(edf_path)
        try:
            duration_s = float(reader.getFileDuration())
        finally:
            reader._close()
        return int(round(duration_s * TARGET_FS)), duration_s

    def _load_signals(self, rec_index: int) -> np.ndarray:
        if rec_index in self._signal_cache:
            self._signal_cache.move_to_end(rec_index)
            return self._signal_cache[rec_index]

        rec = self._recordings[rec_index]
        signals, _native_fs, _found = read_edf_canonical19(rec.edf_path)
        # Align length to stored n_samples.
        n = signals.shape[1]
        if n < rec.n_samples:
            pad = np.zeros((signals.shape[0], rec.n_samples - n), dtype=signals.dtype)
            signals = np.concatenate([signals, pad], axis=1)
        elif n > rec.n_samples:
            signals = signals[:, :rec.n_samples]

        while len(self._signal_cache) >= self._signal_cache_size:
            self._signal_cache.popitem(last=False)
        self._signal_cache[rec_index] = signals
        # Populate recording amplitude stats aligned to the emitted channels.
        # Disk cache by montage so unipolar and bipolar variants both persist.
        if rec_index not in self._rec_stats_cache:
            disk_path = _HELSINKI_STATS_CACHE_ROOT / f"{rec.recording_id}_{self._montage}.npz"
            fingerprint = {
                "target_fs": int(TARGET_FS),
                "n_samples": int(signals.shape[1]),
                "montage": self._montage,
            }
            stats = load_recording_stats(disk_path, fingerprint)
            if stats is None:
                src = signals  # (19, T)
                if self._montage == "bipolar":
                    src = bipolar_from_unipolar(src)  # (20, T)
                stats = compute_recording_stats(src)
                try:
                    save_recording_stats(disk_path, stats, fingerprint)
                except Exception:
                    pass
            self._rec_stats_cache[rec_index] = stats
        return signals

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_index, window_start = self._windows[idx]
        window_end = window_start + self._window_samples
        rec = self._recordings[rec_index]

        all_sig = self._load_signals(rec_index)
        signals = all_sig[:, window_start:window_end].copy()  # (19, T)

        if self._montage == "bipolar":
            signals = bipolar_from_unipolar(signals)

        if self._normalize == "per_window_zscore":
            mean = signals.mean(axis=1, keepdims=True)
            std = signals.std(axis=1, keepdims=True)
            std[std < 1e-8] = 1.0
            signals = (signals - mean) / std

        label, seizure_fraction = window_binary_from_intervals(
            window_start, window_end, rec.intervals_collapsed,
            threshold=self._overlap_threshold,
        )

        # Per-expert window labels + fractions so downstream inter-rater
        # studies never need to re-parse MAT files.
        expert_labels: Dict[str, int] = {}
        expert_fractions: Dict[str, float] = {}
        for expert in EXPERT_IDS:
            e_label, e_frac = window_binary_from_intervals(
                window_start, window_end, rec.intervals_per_expert[expert],
                threshold=self._overlap_threshold,
            )
            expert_labels[expert] = e_label
            expert_fractions[expert] = e_frac

        channel_names = list(BIPOLAR_NAMES if self._montage == "bipolar" else CANONICAL_19)

        meta: Dict[str, Any] = {
            "dataset": "helsinki_neonatal",
            "split": "",
            "subject_id": f"{rec.subject_id:03d}",
            "recording_id": rec.recording_id,
            "recording_idx": rec_index,
            "window_start_s": float(window_start / TARGET_FS),
            "window_end_s": float(window_end / TARGET_FS),
            "recording_duration_s": rec.duration_s,
            "consensus": self._consensus,
            "consensus_type": rec.consensus_type,
            "seizure_fraction": seizure_fraction,
            "has_seizure": bool(label),
            "binary_label": label,
            "expert_A_label": expert_labels["A"],
            "expert_B_label": expert_labels["B"],
            "expert_C_label": expert_labels["C"],
            "expert_A_fraction": expert_fractions["A"],
            "expert_B_fraction": expert_fractions["B"],
            "expert_C_fraction": expert_fractions["C"],
            "channels": channel_names,
            "montage": self._montage,
            "n_channels": len(channel_names),
            "fs": TARGET_FS,
            "sampling_rate": float(TARGET_FS),
            "units": "uV",
            "unit": "uV",
            "window_s": self._window_s,
            "stride_s": self._stride_s,
            "age": -1,
            "gender": rec.gender,
            "bw_grams": rec.bw_grams,
            "gestational_age_weeks": rec.gestational_age_weeks,
            "eeg_to_pma_weeks": rec.eeg_to_pma_weeks,
            "diagnosis": rec.diagnosis,
            "neuroimaging": rec.neuroimaging,
            "primary_localisation": rec.primary_localisation,
        }
        meta.update(self._rec_stats_cache[rec_index])

        return {
            "eeg": torch.from_numpy(signals),
            "label": label,
            "meta": meta,
        }

    # ------------------------------------------------------------------
    # Accessors used by the adapter (k-fold + weighted sampler)
    # ------------------------------------------------------------------

    def binary_labels(self) -> np.ndarray:
        """Binary labels for all windows without reading signals."""
        labels = np.zeros(len(self._windows), dtype=np.int64)
        for i, (rec_index, window_start) in enumerate(self._windows):
            rec = self._recordings[rec_index]
            lbl, _ = window_binary_from_intervals(
                window_start, window_start + self._window_samples,
                rec.intervals_collapsed, threshold=self._overlap_threshold,
            )
            labels[i] = lbl
        return labels

    def subject_ids_per_recording(self) -> List[str]:
        return [f"{r.subject_id:03d}" for r in self._recordings]

    def seizure_presence_per_recording(self) -> List[bool]:
        return [len(r.intervals_collapsed) > 0 for r in self._recordings]

    @property
    def n_recordings(self) -> int:
        return len(self._recordings)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def _collate_helsinki(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    eeg = torch.stack([item["eeg"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "eeg": eeg,
        "labels": labels,
        "meta": [item["meta"] for item in batch],
    }
