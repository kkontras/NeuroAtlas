"""SHHS (Sleep Heart Health Study) reader: NSRR's EDFs and NSRR's XML annotations.

Each night is prepared the way the SHHS preprocessing behind the SleepTransformer
and CoRe-Sleep checkpoints prepared SHHS-1 (Phan et al.'s MATLAB scripts,
``process_and_save_1file.m``), so the epochs and their labels are the ones those
models were trained on:

* EEG: the EDF's ``EEG`` signal (C4-A1), resampled to 100 Hz with MATLAB's
  ``resample`` (reproduced by :func:`matlab_resample`) and band-passed
  0.3-40 Hz with a zero-phase 101-tap FIR (``fir1`` + ``filtfilt``).
* 30 s epochs, each labelled from the ``Stages|Stages`` events of the NSRR XML.
* Epochs scored as movement or unscored (codes other than 0-5) are left out.
* When wake outnumbers every sleep stage, wake is trimmed from the start and
  end of the night until it equals the largest sleep stage (the original's
  counting, kept as it is).
* R&K to AASM: stage 4 joins stage 3 as N3. Labels are 0-4 = W, N1, N2, N3, REM.
* Epochs in which a 2 s window of the EEG, the EOG (R - L) or the chin EMG,
  each prepared the same way, is exactly flat are left out of scoring (label
  -1): their log spectrum is infinite, and the original removed them. The
  EOG and EMG are read for this test only.

Every recording is read, including nights that lack one of the five stages:
the benchmark's folds cover all 8444 recordings.

Layout under the dataset's folder (what ``neuroatlas data download shhs``
fetches)::

    polysomnography/edfs/shhs1/shhs1-200001.edf
    polysomnography/edfs/shhs2/shhs2-200077.edf
    polysomnography/annotations-events-nsrr/shhs{1,2}/<recording>-nsrr.xml
    datasets/shhs1-dataset-<version>.csv     nsrrid, age_s1 (age at visit 1)
    datasets/shhs2-dataset-<version>.csv     nsrrid, age_s2 (age at visit 2)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

__all__ = [
    "CHANNEL",
    "EPOCH_SECONDS",
    "LABEL_NAMES",
    "SAMPLING_RATE",
    "Recording",
    "SHHSDataset",
    "age_tables",
    "load_ages",
    "matlab_resample",
    "night_plan",
    "prepare_night",
    "read_stages",
    "scan_recordings",
]

#: The EEG the reader serves: the EDF's ``EEG`` signal, C4 referenced to A1.
CHANNEL = "C4-A1"
SAMPLING_RATE = 100
EPOCH_SECONDS = 30
_EPOCH_SAMPLES = SAMPLING_RATE * EPOCH_SECONDS
LABEL_NAMES = ["W", "N1", "N2", "N3", "REM"]

#: R&K codes of the NSRR XML (0 wake, 1-4 NREM, 5 REM) to the labels served.
_RK_TO_AASM = np.array([0, 1, 2, 3, 3, 4])

#: EDF signal labels: the EEG served, and the EOG pair and chin EMG read
#: only to find the flat epochs the original removed.
_EEG, _EOG_L, _EOG_R, _EMG = "EEG", "EOG(L)", "EOG(R)", "EMG"

_VISITS = ("shhs1", "shhs2")


# ---------------------------------------------------------------------------
# Signal preparation (the MATLAB steps, reproduced)
# ---------------------------------------------------------------------------

def matlab_resample(x: np.ndarray, p: int, q: int, n: int = 10, beta: float = 5.0) -> np.ndarray:
    """MATLAB's ``resample(x, p, q)`` for a vector: the same firls/Kaiser
    anti-aliasing filter, delay compensation and output length, so a night
    resampled here equals the original to float rounding."""
    from scipy import signal as sps

    g = gcd(int(p), int(q))
    p, q = int(p) // g, int(q) // g
    x = np.asarray(x, dtype=np.float64)
    if p == q:
        return x.copy()
    pqmax = max(p, q)
    fc = 1.0 / 2.0 / pqmax
    length = 2 * n * pqmax + 1
    h = sps.firls(length, [0, 2 * fc, 2 * fc, 1], [1, 1, 0, 0]) * np.kaiser(length, beta)
    h = p * h / h.sum()
    half = (length - 1) / 2
    lx = len(x)
    pad = int(np.floor(q - np.mod(half, q)))
    h = np.concatenate([np.zeros(pad), h])
    half += pad
    delay = int(np.floor(np.ceil(half) / q))
    tail = 0
    while np.ceil(((lx - 1) * p + len(h) + tail) / q) - delay < np.ceil(lx * p / q):
        tail += 1
    h = np.concatenate([h, np.zeros(tail)])
    y = sps.upfirdn(h, x, p, q)
    out_len = int(np.ceil(lx * p / q))
    return y[delay:delay + out_len]


def _fir(kind: str) -> np.ndarray:
    """MATLAB ``fir1(100, ...)`` at 100 Hz: 0.3-40 Hz band-pass (EEG, EOG) or
    10 Hz high-pass (EMG), Hamming window, unit gain in the pass band."""
    from scipy import signal as sps

    if kind == "emg":
        return sps.firwin(101, 10.0, pass_zero=False, fs=SAMPLING_RATE)
    return sps.firwin(101, [0.3, 40.0], pass_zero=False, fs=SAMPLING_RATE)


def _prepare(x: np.ndarray, fs: float, kind: str) -> np.ndarray:
    """Resample one signal to 100 Hz and filter it as the original did
    (``filtfilt`` pads with 3 * (taps - 1) samples)."""
    from scipy import signal as sps

    fs_int = int(round(fs))
    if abs(fs - fs_int) > 1e-6:
        raise ValueError(f"a sampling rate of {fs:g} Hz is not a whole number")
    if fs_int != SAMPLING_RATE:
        x = matlab_resample(x, SAMPLING_RATE, fs_int)
    return sps.filtfilt(_fir(kind), [1.0], np.asarray(x, dtype=np.float64),
                        padtype="odd", padlen=300)


def _flat_epochs(x: np.ndarray, n_epochs: int) -> np.ndarray:
    """Epochs with a 2 s window (1 s hop, the 29 windows of the original's
    spectrogram) whose samples are all exactly zero."""
    epochs = np.asarray(x[: n_epochs * _EPOCH_SAMPLES]).reshape(n_epochs, _EPOCH_SAMPLES)
    nonzero = epochs != 0
    flat = np.zeros(n_epochs, dtype=bool)
    win, hop = 2 * SAMPLING_RATE, SAMPLING_RATE
    for start in range(0, _EPOCH_SAMPLES - win + 1, hop):
        flat |= ~nonzero[:, start:start + win].any(axis=1)
    return flat


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

_STAGE_TYPE = b"<EventType>Stages|Stages</EventType>"
_STAGE_EVENT = re.compile(
    rb"<EventType>Stages\|Stages</EventType>\s*"
    rb"<EventConcept>[^<|]*\|(-?\d+)</EventConcept>\s*"
    rb"<Start>([0-9.eE+-]+)</Start>\s*"
    rb"<Duration>([0-9.eE+-]+)</Duration>"
)


def _stage_events(path: Path) -> List[Tuple[int, float, float]]:
    """(code, start s, duration s) of every ``Stages|Stages`` event."""
    data = Path(path).read_bytes()
    events = [(int(c), float(s), float(d)) for c, s, d in _STAGE_EVENT.findall(data)]
    if len(events) == data.count(_STAGE_TYPE):
        return events
    # laid out differently from NSRR's usual file: parse it as XML
    import xml.etree.ElementTree as ET

    events = []
    for event in ET.fromstring(data).iter("ScoredEvent"):
        if (event.findtext("EventType") or "").strip() != "Stages|Stages":
            continue
        concept = (event.findtext("EventConcept") or "").strip()
        events.append((int(concept.rsplit("|", 1)[-1]), float(event.findtext("Start")),
                       float(event.findtext("Duration"))))
    return events


def read_stages(path: Path) -> np.ndarray:
    """The R&K code of every 30 s epoch from the start of the recording (-1
    where no stage event covers it)."""
    events = _stage_events(path)
    if not events:
        return np.zeros(0, dtype=np.int64)
    n = max(int(round((start + duration) / EPOCH_SECONDS)) for _, start, duration in events)
    stages = np.full(n, -1, dtype=np.int64)
    for code, start, duration in events:
        first = int(round(start / EPOCH_SECONDS))
        stages[first:first + int(round(duration / EPOCH_SECONDS))] = code
    return stages


def _trim_wake(stages: np.ndarray) -> np.ndarray:
    """Positions the original's wake trimming keeps (``stages``: R&K codes
    0-5, unscored already removed).

    When wake is the largest class, the wake at the start and end of the
    night is cut until wake equals the largest sleep stage: from the start
    first, the rest from the end. The original counts the evening wake as
    the index of the first change of state, one more than the wake epochs
    before it, and the morning wake as everything after the last change of
    state; both are kept as they are, so the epochs match.
    """
    n = len(stages)
    keep = np.arange(n)
    if n == 0:
        return keep
    codes, counts = np.unique(stages, return_counts=True)
    if codes[0] != 0 or len(codes) < 2 or counts[0] <= counts[1:].max():
        return keep
    second = int(counts[1:].max())
    changes = np.flatnonzero(np.diff((stages == 0).astype(np.int8)) != 0)
    if len(changes) == 0:
        return keep
    evening = int(changes[0]) + 2 if stages[0] == 0 else 0
    morning = n - (int(changes[-1]) + 2) + 1
    if evening + morning <= second:
        return keep
    remove = evening + morning - second
    if evening > remove:
        return keep[remove:]
    return keep[evening:n - (remove - evening)]


def night_plan(stages: np.ndarray, n_signal_epochs: Optional[int] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
    """(epoch numbers kept, their labels 0-4) for one night.

    ``stages``: :func:`read_stages`. ``n_signal_epochs``: whole 30 s epochs
    the EDF holds; scored epochs past its end are dropped.
    """
    stages = np.asarray(stages, dtype=np.int64)
    if n_signal_epochs is not None:
        stages = stages[:max(0, int(n_signal_epochs))]
    numbers = np.arange(len(stages))
    scored = (stages >= 0) & (stages <= 5)
    numbers, stages = numbers[scored], stages[scored]
    keep = _trim_wake(stages)
    return numbers[keep], _RK_TO_AASM[stages[keep]]


# ---------------------------------------------------------------------------
# EDFs
# ---------------------------------------------------------------------------

def _edf_epochs(path: Path) -> int:
    """Whole 30 s epochs in the EDF, from its header."""
    with open(path, "rb") as fh:
        head = fh.read(256)
    records = int(head[236:244].decode("ascii").strip())
    duration = float(head[244:252].decode("ascii").strip())
    return int(records * duration // EPOCH_SECONDS)


def _signals(path: Path, labels: Sequence[str]) -> Dict[str, Tuple[np.ndarray, float]]:
    """Those of the named EDF signals the file has, in µV, with their rates."""
    import edfio

    from neuroatlas.extensions.datasets.dataio._edf_units import edf_unit_to_uv_scale

    edf = edfio.read_edf(str(path))
    by_label = {s.label.strip(): s for s in edf.signals}
    out: Dict[str, Tuple[np.ndarray, float]] = {}
    for label in labels:
        sig = by_label.get(label)
        if sig is None:
            continue
        unit = (sig.physical_dimension or "").strip()
        scale = edf_unit_to_uv_scale(unit) if unit else 1.0
        out[label] = (np.asarray(sig.data, dtype=np.float64) * scale,
                      float(sig.sampling_frequency))
    return out


def prepare_night(edf_path: Path, numbers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(the EEG of epochs ``numbers``, float32 ``(n, 3000)`` at 100 Hz; and
    which of them are flat in the EEG, EOG or EMG)."""
    name = Path(edf_path).name
    sig = _signals(edf_path, (_EEG, _EOG_L, _EOG_R, _EMG))
    if _EEG not in sig:
        raise KeyError(f"shhs: {name} has no EEG signal (C4-A1)")
    prepared = [_prepare(*sig[_EEG], "eeg")]
    if _EOG_L in sig and _EOG_R in sig and sig[_EOG_L][1] == sig[_EOG_R][1]:
        (left, fs), (right, _) = sig[_EOG_L], sig[_EOG_R]
        n_eog = min(len(left), len(right))
        prepared.append(_prepare(right[:n_eog] - left[:n_eog], fs, "eog"))
    if _EMG in sig:
        prepared.append(_prepare(*sig[_EMG], "emg"))
    n = min(len(x) for x in prepared) // _EPOCH_SAMPLES
    numbers = np.asarray(numbers, dtype=np.int64)
    if len(numbers) and int(numbers.max()) >= n:
        raise ValueError(f"shhs: {name} ends at epoch {n}, before scored epoch "
                         f"{int(numbers.max())}")
    flat = np.zeros(n, dtype=bool)
    for x in prepared:
        flat |= _flat_epochs(x, n)
    epochs = prepared[0][: n * _EPOCH_SAMPLES].reshape(n, _EPOCH_SAMPLES)[numbers]
    return epochs.astype(np.float32), flat[numbers]


# ---------------------------------------------------------------------------
# Recordings and ages
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Recording:
    recording_id: str        # "shhs1-200001": the fold manifest's unit
    patient_id: str          # "200001": NSRR's nsrrid, the same person at both visits
    visit: int               # 1 or 2
    edf_path: Path
    xml_path: Path
    age: Optional[float]     # age at this visit, None when the table has none

    @property
    def subject_id(self) -> str:
        """The unit the folds and the embedding chunks are cut by: the recording."""
        return self.recording_id


def polysomnography(root: Path) -> Path:
    """The folder holding ``edfs/`` and ``annotations-events-nsrr/``."""
    from neuroatlas.extensions.datasets._layout import descend

    return descend(root, ["polysomnography"], "edfs")


def _datasets_folder(root: Path) -> Path:
    poly = polysomnography(root)
    for candidate in (Path(root) / "datasets", poly.parent / "datasets"):
        if candidate.is_dir():
            return candidate
    return Path(root) / "datasets"


def _natural(text: str) -> Tuple:
    return tuple((0, int(part), "") if part.isdigit() else (1, 0, part)
                 for part in re.split(r"(\d+)", text) if part)


def age_tables(root: Path) -> Dict[int, Optional[Path]]:
    """``{visit: datasets/shhs<visit>-dataset-<version>.csv}``, the newest
    version when there are several; None where there is none."""
    folder = _datasets_folder(root)
    out: Dict[int, Optional[Path]] = {}
    for visit in (1, 2):
        found = sorted(folder.glob(f"shhs{visit}-dataset-*.csv"), key=lambda p: _natural(p.name))
        out[visit] = found[-1] if found else None
    return out


def load_ages(root: Path) -> Dict[Tuple[int, str], float]:
    """``{(visit, nsrrid): age}`` from the visit tables (``age_s1``, ``age_s2``:
    age in years at that visit; NSRR codes 90 and older as 90)."""
    import pandas as pd

    ages: Dict[Tuple[int, str], float] = {}
    for visit, path in age_tables(root).items():
        if path is None:
            continue
        column = f"age_s{visit}"
        # NSRR's tables are not all UTF-8 (shhs2-dataset has cp1252 quotes)
        table = pd.read_csv(path, usecols=["nsrrid", column], encoding="latin-1",
                            low_memory=False)
        for nsrrid, age in zip(table["nsrrid"], table[column]):
            if pd.notna(nsrrid) and pd.notna(age):
                ages[(visit, str(int(nsrrid)))] = float(age)
    return ages


def scan_recordings(root: Path) -> List[Recording]:
    """Every SHHS recording under ``root`` that has its NSRR annotation file,
    in natural order (SHHS-1 first)."""
    poly = polysomnography(Path(root))
    ages = load_ages(Path(root))
    pattern = re.compile(r"^shhs([12])-(\d+)$")
    out: List[Recording] = []
    for folder in _VISITS:
        edf_dir = poly / "edfs" / folder
        xml_dir = poly / "annotations-events-nsrr" / folder
        if not edf_dir.is_dir():
            continue
        for edf in sorted(edf_dir.glob("*.edf"), key=lambda p: _natural(p.name)):
            match = pattern.match(edf.stem)
            xml = xml_dir / f"{edf.stem}-nsrr.xml"
            if match is None or not xml.is_file():
                continue
            visit, nsrrid = int(match.group(1)), match.group(2)
            out.append(Recording(edf.stem, nsrrid, visit, edf, xml, ages.get((visit, nsrrid))))
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

#: Epoch plans by annotation file: building an index again in the same
#: process (another fold, another split) does not re-read every XML file.
_PLANS: Dict[Tuple[str, float, int], Tuple[np.ndarray, np.ndarray]] = {}


def _plan_for(recording: Recording) -> Tuple[np.ndarray, np.ndarray]:
    stat = recording.xml_path.stat()
    key = (str(recording.xml_path), stat.st_mtime, stat.st_size)
    plan = _PLANS.get(key)
    if plan is None:
        plan = night_plan(read_stages(recording.xml_path), _edf_epochs(recording.edf_path))
        _PLANS[key] = plan
    return plan


class SHHSDataset(Dataset):
    """One item per kept 30 s epoch: C4-A1 at 100 Hz, ``(1, 3000)`` µV.

    The index (which epochs, their labels) comes from the annotations alone;
    a night's signal is read and prepared when one of its epochs is first
    asked for, and kept until another night is read (the loaders batch one
    night at a time).
    """

    def __init__(
        self,
        recordings: Sequence[Recording],
        fold_assignments: Optional[Dict[str, Dict[str, str]]] = None,
        compute_recording_stats: bool = False,
    ) -> None:
        self._recordings = {r.recording_id: r for r in recordings}
        self._order = [r.recording_id for r in recordings]
        self._fold_assignments = fold_assignments or {}
        self._compute_recording_stats = compute_recording_stats
        # (recording_id, epoch number in the night, label 0-4, row of the
        # epoch in its night's plan); a slice of it is a valid index too
        self._index: List[Tuple[str, int, int, int]] = []
        self._index_built = False
        self._night: Optional[Tuple[str, np.ndarray, np.ndarray, Optional[Dict]]] = None

    def _ensure_index_built(self) -> None:
        if self._index_built:
            return
        for rid in self._order:
            numbers, labels = _plan_for(self._recordings[rid])
            for row, (number, label) in enumerate(zip(numbers.tolist(), labels.tolist())):
                self._index.append((rid, int(number), int(label), row))
        self._index_built = True

    def _load_night(self, rid: str):
        if self._night is not None and self._night[0] == rid:
            return self._night
        numbers, _ = _plan_for(self._recordings[rid])
        epochs, flat = prepare_night(self._recordings[rid].edf_path, numbers)
        stats = None
        if self._compute_recording_stats and epochs.size:
            night = epochs.astype(np.float64).reshape(1, -1)
            stats = {
                "recording_mean": night.mean(axis=1),
                "recording_std": night.std(axis=1),
                "recording_q95": np.quantile(np.abs(night), 0.95, axis=1),
            }
        self._night = (rid, epochs, flat, stats)
        return self._night

    def evict_subject(self, recording_id: str) -> None:
        if self._night is not None and self._night[0] == recording_id:
            self._night = None

    @staticmethod
    def eviction_key(meta: Dict[str, object]) -> Optional[str]:
        rid = meta.get("recording_id")
        return str(rid) if rid is not None else None

    def __len__(self) -> int:
        self._ensure_index_built()
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        self._ensure_index_built()
        rid, number, label, row = self._index[idx]
        _, epochs, flat, stats = self._load_night(rid)
        recording = self._recordings[rid]
        sample: Dict[str, object] = {
            "eeg": torch.from_numpy(epochs[row:row + 1].copy()),
            # a flat epoch is left out of scoring, as the original removed it
            "sleep_stage": -1 if bool(flat[row]) else label,
            "subject_id": rid,
            "recording_id": rid,
            "patient_id": recording.patient_id,
            "visit": recording.visit,
            "epoch_idx": number,
            "age": recording.age,
        }
        if stats is not None:
            sample.update(stats)
        if rid in self._fold_assignments:
            sample["fold_assignments"] = self._fold_assignments[rid]
        return sample
