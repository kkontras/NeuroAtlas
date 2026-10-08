"""ISRUC-SLEEP dataset loader for EEGBenchmarks.

ISRUC-SLEEP is published in three subgroups
(`https://sleeptight.isr.uc.pt/`):

    subgroupI   — 100 sleep-disordered subjects, 1 recording each
    subgroupII  —   8 sleep-disordered subjects, 2 recordings each
    subgroupIII —  10 healthy controls,          1 recording each

Each recording ships as an EDF file named ``<id>.rec`` plus two
plain-text hypnograms ``<id>_{1,2}.txt`` (one integer stage per 30 s
epoch from scorer 1 and scorer 2).

This module:

* walks the three subgroup directories and builds :class:`SubjectRecord`
  entries (with a subgroup prefix so subject ids never collide);
* loads per-subject demographics (age, sex, diagnosis, comorbidities,
  medication, EEG alterations) from the authoritative detail tables
  ``Details_subgroup_{I,II,III}_Submission.xlsx``;
* reads both scorer hypnograms, resamples to the requested
  ``epoch_seconds`` grid, and returns both the **consensus label array**
  (agree → label, disagree or either -1 → -1) and the per-expert
  arrays. This mirrors the DOD pattern so downstream ``label_mode``
  logic is reusable;
* resolves the ``-A1/-A2`` vs ``-M1/-M2`` reference-suffix
  heterogeneity across subgroups (analogous to MASS's
  ``-CLE/-LER``);
* exposes :class:`ISRUCDataset` that lazily opens EDFs per
  ``recording_id`` and caches the processed signals in memory.

Stage codes in the hypnograms: ``0=W, 1=N1, 2=N2, 3=N3, 5=REM``.
Everything else becomes ``-1``. The canonical label space used
elsewhere in the repo is ``["W","N1","N2","N3","REM"]`` indexed
``0..4``.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import edfio
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, firwin, iirnotch, sosfiltfilt
from torch.utils.data import Dataset

from neuroatlas.extensions.datasets.dataio._stage_resample import (
    resample_native_epoch_labels,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Raw hypnogram codes → canonical indices. Anything outside this map → -1.
ISRUC_STAGES_MAP: Dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}

LABEL_NAMES = ["W", "N1", "N2", "N3", "REM"]

DEFAULT_RAW_ROOT = "${EEG_DATA_ROOT}/isruc"

ISRUC_NATIVE_EPOCH = 30
ISRUC_SUBGROUPS: Tuple[str, str, str] = ("I", "II", "III")
ISRUC_SCORER_COLUMNS: Tuple[str, str] = ("scorer_1", "scorer_2")

logger = logging.getLogger(__name__)

DIAGNOSIS_NORMALIZATION: Dict[str, str] = {
    "SAOS": "osa",
    "RONCOPATIA": "snoring",
    "D. Afectiva": "affective_disorder",
    "PRIVAÇÃO DE SONO": "sleep_deprivation",
    "REM Sleep Behaviour Disorder": "rbd",
    "PLMS": "plms",
    "EPILEPSIA": "epilepsy",
    "S. PERNAS INQUIETAS": "rls",
    "Parasomnia": "parasomnia",
    "Parasomnia ": "parasomnia",
    "SRVAS": "uars",
}

DIAGNOSIS_TO_LABEL: Dict[str, int] = {
    "healthy": 0,
    "osa": 1,
    "snoring": 2,
    "affective_disorder": 3,
    "sleep_deprivation": 4,
    "rbd": 5,
    "plms": 6,
    "epilepsy": 7,
    "rls": 8,
    "parasomnia": 9,
    "uars": 10,
}

# Electrode spec: str (monopolar) or tuple (bipolar derivation).
ElectrodeSpec = Union[str, Tuple[str, str]]

# ISRUC's 6 core EEG channels — C3/C4/F3/F4/O1/O2 referenced to the
# contralateral mastoid. Subgroups I/III use "-A1/-A2" labels; subgroup II
# (and some overlapping recordings) use "-M1/-M2" labels. The reference
# resolver below tries all four suffixes.
DEFAULT_CHANNELS: List[str] = ["C3", "C4", "F3", "F4", "O1", "O2"]

# Reference suffixes tried per electrode, in priority order.
# ``-A*`` covers subgroups I and III (and the first recording of most
# subgroup II subjects); ``-M*`` covers the second recording of some
# subgroup II subjects (new 10-20 naming).
_REFERENCE_SUFFIXES_PRIORITY: Tuple[str, ...] = ("-A1", "-A2", "-M1", "-M2")

# For a monopolar C3 electrode, both ``C3-A2`` and ``C3-A1`` may appear in
# different files; we take the first one that matches in priority order.
# For bipolar derivations (should any future montage need them), both
# electrodes must resolve under a consistent reference family; the helper
# below enforces that by trying each family as a group.
_REFERENCE_FAMILIES: Tuple[Tuple[str, ...], ...] = (
    ("-A1", "-A2"),
    ("-M1", "-M2"),
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectRecord:
    """One EDF recording with both scorer hypnograms and subject metadata.

    ``subject_id`` is prefixed by subgroup so there is no collision between
    e.g. subgroupI/1 and subgroupII/1. ``recording_id`` is unique per
    recording (subgroup II contributes two per subject).

    Demographics (age, sex, diagnosis, comorbidities, medication,
    eeg_alterations) are loaded from the authoritative detail tables
    ``Details_subgroup_{I,II,III}_Submission.xlsx``."""

    subject_id: str
    subgroup: str
    recording: int
    recording_id: str
    edf_path: str
    hypnogram_path_scorer1: str
    hypnogram_path_scorer2: str
    age: Optional[float] = None
    sex: Optional[str] = None
    diagnosis: Optional[str] = None
    comorbidities: Optional[str] = None
    medication: Optional[str] = None
    eeg_alterations: Optional[str] = None
    scored_epochs: Optional[int] = None


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------


def _subgroup_root(data_root: str, subgroup: str) -> Path:
    return Path(data_root) / f"subgroup{subgroup}"


def _numeric_dir_entries(path: Path) -> List[str]:
    """Return entries in ``path`` whose names are decimal integers, sorted numerically."""
    if not path.is_dir():
        return []
    entries = [e for e in os.listdir(path) if e.isdigit()]
    entries.sort(key=lambda s: int(s))
    return entries


def _make_subject_id(subgroup: str, numeric_id: str) -> str:
    return f"{subgroup}_{int(numeric_id):03d}"


def _iter_subgroupI_III_records(
    data_root: str,
    subgroup: str,
) -> List[SubjectRecord]:
    """Layout: ``subgroup{I,III}/<id>/<id>.rec`` + ``<id>_{1,2}.txt``."""
    root = _subgroup_root(data_root, subgroup)
    records: List[SubjectRecord] = []
    for numeric_id in _numeric_dir_entries(root):
        subject_dir = root / numeric_id
        edf_path = subject_dir / f"{numeric_id}.rec"
        hyp1 = subject_dir / f"{numeric_id}_1.txt"
        hyp2 = subject_dir / f"{numeric_id}_2.txt"
        if not (edf_path.exists() and hyp1.exists() and hyp2.exists()):
            continue
        subject_id = _make_subject_id(subgroup, numeric_id)
        records.append(SubjectRecord(
            subject_id=subject_id,
            subgroup=subgroup,
            recording=1,
            recording_id=f"{subject_id}_rec1",
            edf_path=str(edf_path),
            hypnogram_path_scorer1=str(hyp1),
            hypnogram_path_scorer2=str(hyp2),
        ))
    return records


def _iter_subgroupII_records(
    data_root: str,
) -> List[SubjectRecord]:
    """Layout: ``subgroupII/<id>/<rec>/<rec>.rec`` + ``<rec>_{1,2}.txt``."""
    root = _subgroup_root(data_root, "II")
    records: List[SubjectRecord] = []
    for numeric_id in _numeric_dir_entries(root):
        subject_dir = root / numeric_id
        subject_id = _make_subject_id("II", numeric_id)
        for rec_name in _numeric_dir_entries(subject_dir):
            rec_dir = subject_dir / rec_name
            edf_path = rec_dir / f"{rec_name}.rec"
            hyp1 = rec_dir / f"{rec_name}_1.txt"
            hyp2 = rec_dir / f"{rec_name}_2.txt"
            if not (edf_path.exists() and hyp1.exists() and hyp2.exists()):
                continue
            recording = int(rec_name)
            records.append(SubjectRecord(
                subject_id=subject_id,
                subgroup="II",
                recording=recording,
                recording_id=f"{subject_id}_rec{recording}",
                edf_path=str(edf_path),
                hypnogram_path_scorer1=str(hyp1),
                hypnogram_path_scorer2=str(hyp2),
            ))
    return records


def scan_isruc_subjects(
    data_root: str = DEFAULT_RAW_ROOT,
    subgroups: Sequence[str] = ISRUC_SUBGROUPS,
) -> List[SubjectRecord]:
    """Discover all ISRUC recordings under ``data_root`` across the listed subgroups.

    Returns a list of :class:`SubjectRecord`. Subgroup II subjects contribute
    two records each; subgroups I and III contribute one.  Demographics are
    loaded from the authoritative detail tables when available.
    """
    details = _load_detail_tables(data_root)

    records: List[SubjectRecord] = []
    for sg in subgroups:
        if sg == "II":
            records.extend(_iter_subgroupII_records(data_root))
        elif sg in ("I", "III"):
            records.extend(_iter_subgroupI_III_records(data_root, sg))
        else:
            raise ValueError(f"Unknown ISRUC subgroup {sg!r}; expected one of {ISRUC_SUBGROUPS}.")

    # Every subgroup that has recordings needs its detail table: without it the
    # recordings would be cached with age/sex/diagnosis = null.
    present = sorted({r.subgroup for r in records})
    absent = [sg for sg in present if not (Path(data_root) / detail_table_name(sg)).exists()]
    if absent:
        raise FileNotFoundError(
            "ISRUC detail table(s) not found: "
            + ", ".join(str(Path(data_root) / detail_table_name(sg)) for sg in absent)
            + ". They ship with the ISRUC-Sleep download and are the only source "
            "of age, sex and diagnosis; place them in data_root."
        )

    # Enrich from detail tables.
    if details:
        records = [
            replace(
                r,
                age=d.age if d.age is not None else r.age,
                sex=d.sex,
                diagnosis=d.diagnosis,
                comorbidities=d.comorbidities,
                medication=d.medication,
                eeg_alterations=d.eeg_alterations,
                scored_epochs=d.scored_epochs,
            )
            if (d := details.get(r.recording_id)) is not None
            else r
            for r in records
        ]

    # Cross-recording propagation for subgroup II: if one recording
    # has demographics and the other doesn't, copy them across.
    demo_by_subject: Dict[str, SubjectRecord] = {}
    for r in records:
        if r.subgroup == "II" and r.age is not None:
            demo_by_subject.setdefault(r.subject_id, r)
    records = [
        replace(
            r,
            age=src.age,
            sex=r.sex or src.sex,
            diagnosis=r.diagnosis or src.diagnosis,
            comorbidities=r.comorbidities or src.comorbidities,
        )
        if (
            r.subgroup == "II"
            and r.age is None
            and (src := demo_by_subject.get(r.subject_id)) is not None
        )
        else r
        for r in records
    ]
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIN_PLAUSIBLE_AGE = 10
_MAX_PLAUSIBLE_AGE = 119


def _is_na(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float):
        return val != val
    return False


# ---------------------------------------------------------------------------
# Detail-table loading (authoritative demographics)
# ---------------------------------------------------------------------------

_SUBJ_REC_RE = re.compile(r"(\d+)_Rec\.(\d+)")


@dataclass
class _DetailRow:
    age: Optional[float]
    sex: Optional[str]
    diagnosis: Optional[str]
    comorbidities: Optional[str]
    medication: Optional[str]
    eeg_alterations: Optional[str]
    scored_epochs: Optional[int]


def _safe_str(val) -> Optional[str]:
    if _is_na(val):
        return None
    s = str(val).strip()
    return s if s else None


def _safe_int(val) -> Optional[int]:
    if _is_na(val):
        return None
    try:
        n = float(val)
        if n == int(n):
            return int(n)
    except (ValueError, TypeError):
        pass
    return None


def _normalize_diagnosis(raw, subgroup: str) -> Optional[str]:
    if _is_na(raw):
        return "healthy" if subgroup == "III" else None
    s = str(raw).strip()
    if not s:
        return "healthy" if subgroup == "III" else None
    return DIAGNOSIS_NORMALIZATION.get(s, s.lower())


def _parse_detail_age(val) -> Optional[float]:
    if _is_na(val):
        return None
    s = str(val).strip()
    if s == "?":
        return None
    try:
        n = float(s)
        if n == int(n) and _MIN_PLAUSIBLE_AGE <= n <= _MAX_PLAUSIBLE_AGE:
            return float(int(n))
    except (ValueError, TypeError):
        pass
    return None


def detail_table_name(subgroup: str) -> str:
    return f"Details_subgroup_{subgroup}_Submission.xlsx"


def _load_detail_tables(data_root: str) -> Dict[str, _DetailRow]:
    """Load demographics from the authoritative detail-table xlsx files.

    Returns a dict keyed by ``recording_id`` (e.g. ``"I_001_rec1"``). A
    subgroup whose table is absent contributes nothing (``scan_isruc_subjects``
    refuses that for a subgroup it has recordings of); a table that is present
    but unreadable raises. It used to be logged and skipped, which left age,
    sex and diagnosis null on every recording of that subgroup -- in the
    embedding cache too, where brain age and the pathology probe then read
    them (F-069).
    """
    result: Dict[str, _DetailRow] = {}
    root = Path(data_root)

    for sg in ISRUC_SUBGROUPS:
        path = root / detail_table_name(sg)
        if not path.exists():
            continue
        try:
            df = pd.read_excel(path, header=None, skiprows=2, engine="openpyxl")
        except ImportError as exc:
            raise RuntimeError(
                f"cannot read the ISRUC detail table {path}: .xlsx needs the "
                f"'openpyxl' package (pip install openpyxl). It is the only source "
                f"of age, sex and diagnosis. ({exc})"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"cannot read the ISRUC detail table {path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        has_medication = df.shape[1] >= 18
        has_eeg_alt = df.shape[1] >= 19

        for _, row in df.iterrows():
            subj_raw = str(row.iloc[0]).strip()

            if sg == "II":
                m = _SUBJ_REC_RE.match(subj_raw)
                if not m:
                    continue
                numeric_id, rec_num = m.group(1), m.group(2)
                rec_id = f"II_{int(numeric_id):03d}_rec{rec_num}"
            elif subj_raw.isdigit():
                rec_id = f"{sg}_{int(subj_raw):03d}_rec1"
            else:
                continue

            comorbidities = _safe_str(row.iloc[4])
            if comorbidities and comorbidities.lower() == "no problem":
                comorbidities = None

            result[rec_id] = _DetailRow(
                age=_parse_detail_age(row.iloc[1]),
                sex=_safe_str(row.iloc[2]),
                diagnosis=_normalize_diagnosis(row.iloc[3], sg),
                comorbidities=comorbidities,
                medication=_safe_str(row.iloc[17]) if has_medication else None,
                eeg_alterations=_safe_str(row.iloc[18]) if has_eeg_alt else None,
                scored_epochs=_safe_int(row.iloc[5]),
            )

    # Subgroup II: propagate demographics from Rec.1 to Rec.2
    sg2_subjects: Dict[str, List[str]] = {}
    for rec_id in result:
        if rec_id.startswith("II_"):
            subj_prefix = rec_id.rsplit("_rec", 1)[0]
            sg2_subjects.setdefault(subj_prefix, []).append(rec_id)
    for subj_prefix, rec_ids in sg2_subjects.items():
        rec1_id = f"{subj_prefix}_rec1"
        if rec1_id not in result:
            continue
        rec1 = result[rec1_id]
        for rec_id in rec_ids:
            if rec_id == rec1_id:
                continue
            rec = result[rec_id]
            result[rec_id] = _DetailRow(
                age=rec.age if rec.age is not None else rec1.age,
                sex=rec.sex if rec.sex is not None else rec1.sex,
                diagnosis=rec.diagnosis if rec.diagnosis is not None else rec1.diagnosis,
                comorbidities=rec.comorbidities if rec.comorbidities is not None else rec1.comorbidities,
                medication=rec.medication if rec.medication is not None else rec1.medication,
                eeg_alterations=rec.eeg_alterations if rec.eeg_alterations is not None else rec1.eeg_alterations,
                scored_epochs=rec.scored_epochs,
            )

    return result


# ---------------------------------------------------------------------------
# Hypnogram reading and consensus
# ---------------------------------------------------------------------------


def _clamp_stage(x: int) -> int:
    return ISRUC_STAGES_MAP.get(int(x), -1)


def read_isruc_hypnogram(hypnogram_path: str) -> np.ndarray:
    """Read one int per line from a ``_1.txt`` or ``_2.txt`` hypnogram.

    Returns a 1-D ``int16`` array at the native 30 s epoch grid, with any
    unknown code mapped to ``-1``. Blank/non-integer lines are skipped.
    """
    stages: List[int] = []
    with open(hypnogram_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                stages.append(_clamp_stage(int(line)))
            except ValueError:
                continue
    return np.asarray(stages, dtype=np.int16)


def _consensus_stages(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Element-wise 2-rater consensus: agree → label, else -1.

    Treats any ``-1`` as disagreement (since the sample is unscored by at
    least one rater). Arrays are truncated to the shorter length when they
    differ.
    """
    n = min(len(s1), len(s2))
    s1 = s1[:n]
    s2 = s2[:n]
    agree = (s1 == s2) & (s1 >= 0)
    out = np.full(n, -1, dtype=np.int16)
    out[agree] = s1[agree]
    return out


def read_isruc_stages(
    scorer1_path: str,
    scorer2_path: str,
    epoch_seconds: float = ISRUC_NATIVE_EPOCH,
) -> Tuple[np.ndarray, float, float]:
    """Read both scorers, build the consensus, and resample to ``epoch_seconds``.

    Returns ``(stages, scores_start, scores_end)`` where ``scores_start`` is
    always ``0.0`` (ISRUC hypnograms start at the recording origin) and
    ``scores_end = len(stages) * epoch_seconds``.
    """
    s1 = read_isruc_hypnogram(scorer1_path)
    s2 = read_isruc_hypnogram(scorer2_path)
    consensus = _consensus_stages(s1, s2)
    resampled = resample_native_epoch_labels(
        consensus.tolist(), int(epoch_seconds), native_epoch=ISRUC_NATIVE_EPOCH,
    )
    arr = np.asarray(resampled, dtype=np.int16)
    return arr, 0.0, float(len(arr) * int(epoch_seconds))


def read_isruc_expert_stages(
    scorer1_path: str,
    scorer2_path: str,
    epoch_seconds: float = ISRUC_NATIVE_EPOCH,
) -> Dict[str, np.ndarray]:
    """Per-rater label arrays resampled to ``epoch_seconds``.

    Returns ``{"scorer_1": ndarray[int16], "scorer_2": ndarray[int16]}``.
    Arrays are truncated to the shorter native length before resampling so
    both rater arrays are the same length at the target grid.
    """
    s1 = read_isruc_hypnogram(scorer1_path)
    s2 = read_isruc_hypnogram(scorer2_path)
    n = min(len(s1), len(s2))
    s1 = s1[:n]
    s2 = s2[:n]
    out: Dict[str, np.ndarray] = {}
    for name, arr in (("scorer_1", s1), ("scorer_2", s2)):
        resampled = resample_native_epoch_labels(
            arr.tolist(), int(epoch_seconds), native_epoch=ISRUC_NATIVE_EPOCH,
        )
        out[name] = np.asarray(resampled, dtype=np.int16)
    return out


# ---------------------------------------------------------------------------
# Channel reading (edfio)
# ---------------------------------------------------------------------------


def _find_monopolar(
    label_to_signal: Dict[str, edfio.EdfSignal],
    electrode: str,
) -> Optional[edfio.EdfSignal]:
    """Return the first signal matching ``<electrode><suffix>`` in priority order.

    Falls back to the bare electrode label (e.g., ``C3``) when no suffixed
    variant is present. Some subgroup-I recordings (e.g., subject 40) ship
    raw electrode labels without a reference suffix.
    """
    for suffix in _REFERENCE_SUFFIXES_PRIORITY:
        key = f"{electrode}{suffix}".upper()
        sig = label_to_signal.get(key)
        if sig is not None:
            return sig
    return label_to_signal.get(electrode.upper())


def _find_bipolar_pair(
    label_to_signal: Dict[str, edfio.EdfSignal],
    electrode_a: str,
    electrode_b: str,
) -> Optional[Tuple[edfio.EdfSignal, edfio.EdfSignal, str]]:
    """Return a pair of signals that share a reference family, else None."""
    for family in _REFERENCE_FAMILIES:
        for suffix_a in family:
            for suffix_b in family:
                ka = f"{electrode_a}{suffix_a}".upper()
                kb = f"{electrode_b}{suffix_b}".upper()
                if ka in label_to_signal and kb in label_to_signal:
                    return label_to_signal[ka], label_to_signal[kb], "|".join(family)
    return None


def resolve_and_read_channels(
    edf_path: str,
    montage_electrodes: Sequence[ElectrodeSpec],
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
) -> Tuple[np.ndarray, List[str], float, bool]:
    """Resolve montage and read channels from an ISRUC EDF in one pass.

    Returns ``(signals, channel_names, sfreq, any_resolved)``. Missing
    channels are zero-filled so model input dimensionality stays constant.
    """
    edf = edfio.read_edf(edf_path)
    label_to_signal: Dict[str, edfio.EdfSignal] = {
        sig.label.upper(): sig for sig in edf.signals
    }

    channel_data: List[Optional[np.ndarray]] = []
    names: List[str] = []
    ref_sfreq: Optional[float] = None

    for spec in montage_electrodes:
        if isinstance(spec, tuple):
            pair = _find_bipolar_pair(label_to_signal, spec[0], spec[1])
            if pair is None:
                channel_data.append(None)
                names.append("MISSING")
                continue
            sig_a, sig_b, _ = pair
            sfreq = float(sig_a.sampling_frequency)
            s_start = int(start_sec * sfreq)
            s_end = s_start + int(duration_sec * sfreq) if duration_sec else None
            data_a = sig_a.data[s_start:s_end] if s_end is not None else sig_a.data[s_start:]
            data_b = sig_b.data[s_start:s_end] if s_end is not None else sig_b.data[s_start:]
            min_len = min(len(data_a), len(data_b))
            channel_data.append(
                (data_a[:min_len] - data_b[:min_len]).astype(np.float32)
            )
            names.append(f"{sig_a.label}-{sig_b.label}")
            if ref_sfreq is None:
                ref_sfreq = sfreq
        else:
            sig = _find_monopolar(label_to_signal, spec)
            if sig is None:
                channel_data.append(None)
                names.append("MISSING")
                continue
            sfreq = float(sig.sampling_frequency)
            s_start = int(start_sec * sfreq)
            if duration_sec is not None:
                data = sig.data[s_start:s_start + int(duration_sec * sfreq)]
            elif s_start > 0:
                data = sig.data[s_start:]
            else:
                data = sig.data
            channel_data.append(data.astype(np.float32))
            names.append(sig.label)
            if ref_sfreq is None:
                ref_sfreq = sfreq

    resolved = [c for c in channel_data if c is not None]
    if not resolved:
        raise ValueError(
            f"No montage channels resolved from {edf_path}; tried electrodes "
            f"{[s if isinstance(s, str) else '-'.join(s) for s in montage_electrodes]}. "
            f"EDF labels present: {sorted(label_to_signal)}"
        )

    min_samples = min(len(c) for c in resolved)
    for i, c in enumerate(channel_data):
        if c is None:
            channel_data[i] = np.zeros(min_samples, dtype=np.float32)

    stacked = np.stack([c[:min_samples] for c in channel_data], axis=0)
    return stacked, names, float(ref_sfreq or 0.0), True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ISRUCDataset(Dataset):
    """Epoch-level PyTorch dataset for the ISRUC-SLEEP collection.

    Each item is a dict with the consensus ``sleep_stage`` plus per-rater
    ``expert_stages``. ``label_mode`` selection happens in the adapter's
    collate (see ``adapters/isruc.py``), identical to the DOD pattern.
    """

    def __init__(
        self,
        subject_records: Sequence[SubjectRecord],
        channel_specs: List[str] = DEFAULT_CHANNELS,
        epoch_seconds: float = ISRUC_NATIVE_EPOCH,
        bandpass: Optional[Tuple[float, float]] = None,
        notch: Optional[float] = None,
        highpass: Optional[float] = None,
        fold_assignments: Optional[Dict[str, Dict[str, str]]] = None,
        compute_recording_stats: bool = False,
        use_all_eeg_channels: bool = False,
    ) -> None:
        if int(epoch_seconds) != epoch_seconds or epoch_seconds <= 0:
            raise ValueError(
                f"epoch_seconds must be a positive integer, got {epoch_seconds!r}"
            )
        self._montage_electrodes: List[ElectrodeSpec] = list(channel_specs)
        self._montage_names = list(channel_specs)
        self.channel_specs = channel_specs
        self.epoch_seconds = float(int(epoch_seconds))
        self.bandpass = bandpass
        self.notch = notch
        self.highpass = highpass
        self._fold_assignments = fold_assignments or {}
        self._compute_recording_stats = compute_recording_stats
        self._use_all_eeg_channels = use_all_eeg_channels

        self._subject_records = list(subject_records)
        self._records_by_rec_id: Dict[str, SubjectRecord] = {}
        self._scores_end: Dict[str, float] = {}
        self._expert_stages: Dict[str, Dict[str, np.ndarray]] = {}
        self._index: List[Tuple[str, int, int]] = []
        self._index_built = False
        self._edf_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._stats_cache: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
        self._discovered_channels: Dict[str, List[str]] = {}

    def _ensure_index_built(self) -> None:
        if self._index_built:
            return
        eps = int(self.epoch_seconds)
        for record in self._subject_records:
            consensus, _, scores_end = read_isruc_stages(
                record.hypnogram_path_scorer1,
                record.hypnogram_path_scorer2,
                epoch_seconds=eps,
            )
            experts = read_isruc_expert_stages(
                record.hypnogram_path_scorer1,
                record.hypnogram_path_scorer2,
                epoch_seconds=eps,
            )

            self._records_by_rec_id[record.recording_id] = record
            self._scores_end[record.recording_id] = scores_end
            self._expert_stages[record.recording_id] = experts

            for ep_idx, stage in enumerate(consensus):
                self._index.append((record.recording_id, ep_idx, int(stage)))
        self._index_built = True

    def _load_recording_edf(self, recording_id: str) -> Tuple[np.ndarray, float]:
        if recording_id in self._edf_cache:
            return self._edf_cache[recording_id]

        record = self._records_by_rec_id[recording_id]
        scores_end = self._scores_end[recording_id]

        if self._use_all_eeg_channels:
            from ._eeg_channel_discovery import read_all_eeg_channels
            signals, ch_names, sfreq = read_all_eeg_channels(
                record.edf_path, start_sec=0.0, duration_sec=scores_end,
            )
            self._discovered_channels[recording_id] = ch_names
        else:
            signals, _, sfreq, _ = resolve_and_read_channels(
                record.edf_path, self._montage_electrodes,
                start_sec=0.0, duration_sec=scores_end,
            )

        if self.highpass is not None:
            nyq = sfreq / 2.0
            sos = butter(5, self.highpass / nyq, btype="high", output="sos")
            signals = sosfiltfilt(sos, signals, axis=-1).astype(np.float32)

        if self.bandpass is not None:
            lo, hi = self.bandpass
            n_taps = 101
            b = firwin(n_taps, [lo, hi], pass_zero=False, fs=sfreq)
            signals = filtfilt(b, 1, signals, axis=-1).astype(np.float32)

        if self.notch is not None and self.notch < sfreq / 2.0:
            b, a = iirnotch(self.notch, Q=30.0, fs=sfreq)
            sos = np.array([[b[0], b[1], b[2], a[0], a[1], a[2]]])
            signals = sosfiltfilt(sos, signals, axis=-1).astype(np.float32)

        rec_stats: Optional[Dict[str, np.ndarray]] = None
        if self._compute_recording_stats and signals.size > 0:
            rec_stats = {
                "recording_mean": signals.mean(axis=1),
                "recording_std": signals.std(axis=1),
                "recording_q95": np.quantile(np.abs(signals), 0.95, axis=1),
            }
            n_ch = signals.shape[0]
            if n_ch > 1:
                bq = {}
                for i in range(n_ch):
                    for j in range(i + 1, n_ch):
                        bq[f"{i},{j}"] = float(
                            np.quantile(np.abs(signals[i] - signals[j]), 0.95)
                        )
                rec_stats["recording_q95_bipolar"] = bq

        self._edf_cache.clear()
        self._stats_cache.clear()
        self._edf_cache[recording_id] = (signals, sfreq)
        self._stats_cache[recording_id] = rec_stats
        return signals, sfreq

    def evict_recording(self, recording_id: str) -> None:
        self._edf_cache.pop(recording_id, None)
        self._stats_cache.pop(recording_id, None)

    def evict_subject(self, recording_id: str) -> None:
        self.evict_recording(recording_id)

    @staticmethod
    def eviction_key(meta: Dict[str, object]) -> Optional[str]:
        rid = meta.get("recording_id")
        return str(rid) if rid is not None else None

    def __len__(self) -> int:
        self._ensure_index_built()
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        self._ensure_index_built()
        recording_id, ep_idx, sleep_stage = self._index[idx]
        signals, sfreq = self._load_recording_edf(recording_id)

        n_samples = int(round(self.epoch_seconds * sfreq))
        start = ep_idx * n_samples
        epoch = signals[:, start : start + n_samples]
        if epoch.shape[1] < n_samples:
            epoch = np.pad(epoch, ((0, 0), (0, n_samples - epoch.shape[1])))

        record = self._records_by_rec_id[recording_id]
        experts = self._expert_stages.get(recording_id, {})

        sample: Dict[str, object] = {
            "eeg": torch.tensor(epoch, dtype=torch.float32),
            "sleep_stage": int(sleep_stage),
            "subject_id": record.subject_id,
            "subgroup": record.subgroup,
            "recording": record.recording,
            "recording_id": recording_id,
            "epoch_idx": ep_idx,
            "channels": (self._discovered_channels.get(recording_id, self._montage_names)),
            "epoch_seconds": float(self.epoch_seconds),
            "unit": "uV",
            "age": record.age,
            "sex": record.sex,
            "location": "Coimbra, Portugal",
            "diagnosis": record.diagnosis,
            "comorbidities": record.comorbidities,
            "medication": record.medication,
            "eeg_alterations": record.eeg_alterations,
            "expert_stages": {
                scorer: int(arr[ep_idx]) if ep_idx < len(arr) else -1
                for scorer, arr in experts.items()
            },
        }
        rec_stats = self._stats_cache.get(recording_id)
        if rec_stats is not None:
            for k, v in rec_stats.items():
                sample[k] = v
        if record.subject_id in self._fold_assignments:
            sample["fold_assignments"] = self._fold_assignments[record.subject_id]
        return sample

    @property
    def recordings(self) -> List[str]:
        return [r.recording_id for r in self._subject_records]
