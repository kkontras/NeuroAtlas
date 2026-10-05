"""Shared BIDS scanning and indexing for seizure-monitoring EEG datasets.

Builds a lightweight recording index by reading only JSON sidecars and
events TSV files — no EDF files are opened.  The index is used by the
BIDS-direct dataset backends (``chbmit_bids.py``, ``siena_bids.py``) to
construct a window index and compute seizure labels on the fly.
"""
from __future__ import annotations

import csv
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# (resolved root, root mtime) -> BIDSRecordingIndex; see from_bids_root.
_INDEX_MEMO: Dict[tuple, "BIDSRecordingIndex"] = {}


# ---------------------------------------------------------------------------
# BIDS discovery helpers (deduplicated from preprocessors)
# ---------------------------------------------------------------------------


def find_bids_root(base: Path | str) -> Optional[Path]:
    """Find the directory containing ``sub-*`` folders."""
    base = Path(base)
    if list(base.glob("sub-*")):
        return base
    for d in base.iterdir():
        if d.is_dir() and list(d.glob("sub-*")):
            return d
    for d in base.iterdir():
        if d.is_dir():
            for dd in d.iterdir():
                if dd.is_dir() and list(dd.glob("sub-*")):
                    return dd
    return None


def parse_bids_events(tsv_path: Path | str) -> List[Tuple[float, float, str]]:
    """Parse a BIDS ``_events.tsv`` and extract seizure intervals.

    Returns list of ``(onset_s, offset_s, event_type)`` tuples.
    """
    tsv_path = Path(tsv_path)
    seizures: List[Tuple[float, float, str]] = []
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            event_type = (
                row.get("trial_type", "")
                or row.get("eventType", "")
                or row.get("event_type", "")
                or row.get("value", "")
            ).strip().lower()

            is_seizure = (
                event_type.startswith("sz")
                or "seiz" in event_type
                or "seizure" in event_type
            )
            if is_seizure and event_type != "bckg":
                onset = float(row.get("onset", 0))
                dur_str = row.get("duration", "0")
                duration = float(dur_str) if dur_str not in ("n/a", "") else 0.0
                if duration > 0:
                    seizures.append((onset, onset + duration, event_type))

    return seizures


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BIDSRecording:
    """Metadata for one EDF recording, extracted from BIDS sidecars."""

    rec_index: int
    subject_id: str
    recording_id: str  # relative path from BIDS root (e.g. "sub-01/ses-01/eeg/..._eeg.edf")
    edf_path: str
    n_samples: int
    duration_s: float
    fs: int
    n_channels: int
    seizure_intervals_s: List[Tuple[float, float]] = field(default_factory=list)
    seizure_intervals_samples: List[Tuple[int, int]] = field(default_factory=list)
    seizure_types: List[str] = field(default_factory=list)  # event type per interval


@dataclass
class BIDSRecordingIndex:
    """Full index of a BIDS seizure-monitoring dataset.

    Built by scanning JSON sidecars and events TSV files — no EDF reads.
    """

    bids_root: str
    recordings: List[BIDSRecording]
    _subject_to_recs: Dict[str, List[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._subject_to_recs:
            self._subject_to_recs = {}
            for rec in self.recordings:
                self._subject_to_recs.setdefault(rec.subject_id, []).append(rec.rec_index)

    # -- Construction -------------------------------------------------------

    @classmethod
    def from_bids_root(cls, bids_root: str | Path) -> "BIDSRecordingIndex":
        """Scan a BIDS directory and build the recording index.

        Memoised per process: one datamodule used to scan the tree four times
        (once itself, once per split dataset), and ``check`` once more per
        model. The scan is a recursive glob plus one JSON and one TSV read per
        recording, which on NFS is seconds per call. Keyed on the resolved
        root and the mtimes of it and its ``sub-*`` folders, so a tree that
        gains subjects or sessions is rescanned.
        """
        bids_root = Path(bids_root)
        if not bids_root.exists():
            raise FileNotFoundError(f"BIDS root not found: {bids_root}")
        try:
            stamps = tuple(sorted((p.name, p.stat().st_mtime_ns)
                                  for p in bids_root.glob("sub-*")))
            key = (str(bids_root.resolve()), bids_root.stat().st_mtime_ns, stamps)
        except OSError:
            key = None
        if key is not None and key in _INDEX_MEMO:
            return _INDEX_MEMO[key]
        index = cls._scan(bids_root)
        if key is not None:
            _INDEX_MEMO[key] = index
        return index

    @classmethod
    def _scan(cls, bids_root: Path) -> "BIDSRecordingIndex":

        # Find all JSON sidecars (one per EDF)
        json_files = sorted(bids_root.rglob("*_eeg.json"))
        if not json_files:
            raise RuntimeError(f"No *_eeg.json files found in {bids_root}")

        recordings: List[BIDSRecording] = []

        for idx, json_path in enumerate(json_files):
            # Derive paths
            edf_path = json_path.with_name(json_path.name.replace("_eeg.json", "_eeg.edf"))
            events_path = json_path.with_name(json_path.name.replace("_eeg.json", "_events.tsv"))

            if not edf_path.exists():
                logger.warning("EDF not found for %s, skipping", json_path)
                continue

            # Parse JSON sidecar
            with open(json_path, "r") as f:
                meta = json.load(f)

            fs = int(meta["SamplingFrequency"])
            duration_s = float(meta["RecordingDuration"])
            n_channels = int(meta["EEGChannelCount"])
            n_samples = int(round(duration_s * fs))

            # Parse seizure events (returns list of (onset, offset, type))
            seizures_full: List[Tuple[float, float, str]] = []
            if events_path.exists():
                seizures_full = parse_bids_events(events_path)

            seizures_s = [(s, e) for s, e, _ in seizures_full]
            seizure_types = [t for _, _, t in seizures_full]
            seizures_samples = [
                (int(round(s * fs)), int(round(e * fs)))
                for s, e in seizures_s
            ]

            # Extract subject ID from path
            # Path structure: sub-XX/ses-YY/eeg/*_eeg.edf
            rel = edf_path.relative_to(bids_root)
            subject_id = rel.parts[0]  # "sub-XX"
            recording_id = str(rel)

            recordings.append(BIDSRecording(
                rec_index=len(recordings),
                subject_id=subject_id,
                recording_id=recording_id,
                edf_path=str(edf_path),
                n_samples=n_samples,
                duration_s=duration_s,
                fs=fs,
                n_channels=n_channels,
                seizure_intervals_s=seizures_s,
                seizure_intervals_samples=seizures_samples,
                seizure_types=seizure_types,
            ))

        logger.info(
            "BIDSRecordingIndex: %s — %d recordings, %d subjects, %d with seizures",
            bids_root,
            len(recordings),
            len({r.subject_id for r in recordings}),
            sum(1 for r in recordings if r.seizure_intervals_s),
        )

        return cls(bids_root=str(bids_root), recordings=recordings)

    @classmethod
    def load_or_build(
        cls,
        bids_root: str | Path,
        cache_path: Optional[str | Path] = None,
    ) -> "BIDSRecordingIndex":
        """Load from pickle cache if available, otherwise build and save."""
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                logger.info("Loading BIDS index from cache: %s", cache_path)
                with open(cache_path, "rb") as f:
                    index = pickle.load(f)
                if isinstance(index, cls):
                    return index
                logger.warning("Cache is not a BIDSRecordingIndex, rebuilding")

        index = cls.from_bids_root(bids_root)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saved BIDS index cache to %s", cache_path)

        return index

    # -- Accessors ----------------------------------------------------------

    @property
    def subject_ids(self) -> List[str]:
        """Unique sorted subject IDs."""
        return sorted(self._subject_to_recs.keys())

    def recordings_for_subject(self, subject_id: str) -> List[int]:
        return self._subject_to_recs.get(subject_id, [])

    def subject_ids_per_recording(self) -> List[str]:
        """Subject ID for each recording, ordered by rec_index."""
        return [r.subject_id for r in self.recordings]

    def seizure_presence_per_recording(self) -> List[bool]:
        """Whether each recording has at least one seizure event."""
        return [len(r.seizure_intervals_s) > 0 for r in self.recordings]

    def has_seizures(self, subject_id: str) -> bool:
        for ri in self.recordings_for_subject(subject_id):
            if self.recordings[ri].seizure_intervals_s:
                return True
        return False
