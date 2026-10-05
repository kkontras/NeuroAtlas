"""EPILEPSIAE annotation parsers — seizure CSV, origin CSV, and SQL metadata."""
from __future__ import annotations

import csv
import logging
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .readers import (
    CHANNEL_ALIAS,
    BlockMeta,
    _parse_timestamp,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seizure event dataclass
# ---------------------------------------------------------------------------

# Imported by callers — kept as a plain dataclass here to avoid circular deps.
from dataclasses import dataclass


@dataclass
class SeizureEvent:
    """One seizure annotation with resolved onset/offset."""

    seizure_id: str
    recording_id: str
    block_id: str
    onset: object  # datetime
    offset: object  # datetime
    classification: str  # CP, SP, SG, UC
    pattern: str  # t, a, d, ... or ""
    vigilance: str  # A, 1, 2, 3, 4, R, ?
    onset_electrode: str  # from Annotations_origin.csv, or ""
    onset_is_clinical: bool  # True if eeg_onset was NULL
    offset_is_clinical: bool


# ---------------------------------------------------------------------------
# Seizure annotation CSV
# ---------------------------------------------------------------------------


def load_seizure_annotations(
    csv_path: Optional[Path] = None,
    annotation_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Parse Seizure_annotations.csv into a list of dicts.

    Each dict has keys: ``id``, ``recording``, ``block``, ``eeg_onset``,
    ``clin_onset``, ``first_eeg_change``, ``first_clin_sign``,
    ``eeg_offset``, ``clin_offset``, ``pattern``, ``classification``,
    ``vigilance``, ``focus``, ``commentary``.

    Datetime fields are parsed; NULL values become None.
    """
    if csv_path is None:
        if annotation_dir is None:
            raise ValueError("Either csv_path or annotation_dir must be provided")
        csv_path = annotation_dir / "Seizure_annotations.csv"

    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, quotechar='"')
        for row in reader:
            parsed: Dict[str, Any] = {}
            for key, val in row.items():
                val = val.strip().strip('"')
                if val == "NULL" or val == "":
                    parsed[key] = None
                else:
                    parsed[key] = val
            for dt_key in ("eeg_onset", "clin_onset", "first_eeg_change",
                           "first_clin_sign", "eeg_offset", "clin_offset"):
                if parsed.get(dt_key) is not None:
                    try:
                        parsed[dt_key] = _parse_timestamp(parsed[dt_key])
                    except ValueError:
                        logger.warning("Cannot parse %s=%r in seizure %s",
                                       dt_key, parsed[dt_key], parsed.get("id"))
                        parsed[dt_key] = None
            rows.append(parsed)
    return rows


def load_origin_annotations(
    csv_path: Optional[Path] = None,
    annotation_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Parse Annotations_origin.csv into ``{seizure_sub_id: electrode_name}``.

    Extracts the electrode name from commentary strings like
    ``"Electrode: P4"`` or ``"Electrode:  C4"``.
    """
    if csv_path is None:
        if annotation_dir is None:
            raise ValueError("Either csv_path or annotation_dir must be provided")
        csv_path = annotation_dir / "Annotations_origin.csv"

    mapping: Dict[str, str] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, quotechar='"')
        for row in reader:
            sid = row["id"].strip().strip('"')
            commentary = row.get("commentary", "").strip().strip('"')
            m = re.match(r"Electrode:\s*(\S+)", commentary)
            if m:
                mapping[sid] = m.group(1).strip()
    return mapping


def _match_origin_to_seizure(
    seizure_id: str,
    origin_map: Dict[str, str],
) -> str:
    """Try to find the onset electrode for a seizure.

    The origin CSV uses sub-IDs like ``"100000000102"`` while seizure IDs
    are like ``"100000102"``.  Matches by shared numeric prefix heuristic.
    """
    for origin_id, electrode in origin_map.items():
        if origin_id.startswith(seizure_id[:6]) and origin_id.endswith(seizure_id[-3:]):
            return electrode
    return ""


def seizures_for_recording(
    all_annotations: List[Dict[str, Any]],
    recording_id: str,
    origin_map: Optional[Dict[str, str]] = None,
) -> List[SeizureEvent]:
    """Filter and resolve seizure annotations for one recording.

    Onset/offset fallback logic:
    - onset = eeg_onset if available, else clin_onset
    - offset = eeg_offset if available, else clin_offset (else onset + 60 s)

    Returns list of :class:`SeizureEvent` sorted by onset time.
    """
    if origin_map is None:
        origin_map = {}

    events: List[SeizureEvent] = []
    for ann in all_annotations:
        if str(ann.get("recording")) != str(recording_id):
            continue

        onset_is_clin = False
        offset_is_clin = False

        onset = ann.get("eeg_onset")
        if onset is None:
            onset = ann.get("clin_onset")
            onset_is_clin = True
        if onset is None:
            logger.warning("Seizure %s: both eeg_onset and clin_onset are NULL, skipping.",
                           ann.get("id"))
            continue

        offset = ann.get("eeg_offset")
        if offset is None:
            offset = ann.get("clin_offset")
            offset_is_clin = True
        if offset is None:
            logger.warning("Seizure %s: both offsets NULL, using onset + 60s.", ann.get("id"))
            offset = onset + timedelta(seconds=60)

        events.append(SeizureEvent(
            seizure_id=str(ann.get("id", "")),
            recording_id=str(recording_id),
            block_id=str(ann.get("block", "")),
            onset=onset,
            offset=offset,
            classification=ann.get("classification") or "UC",
            pattern=ann.get("pattern") or "",
            vigilance=ann.get("vigilance") or "?",
            onset_electrode=_match_origin_to_seizure(str(ann.get("id", "")), origin_map),
            onset_is_clinical=onset_is_clin,
            offset_is_clinical=offset_is_clin,
        ))

    events.sort(key=lambda e: e.onset)
    return events


# ---------------------------------------------------------------------------
# SQL metadata parser
# ---------------------------------------------------------------------------

_SQL_INSERT_RE = re.compile(
    r"INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\((.+?)\)\s*;",
    re.IGNORECASE,
)


def _parse_sql_value(val: str) -> Any:
    """Parse a single SQL value literal."""
    val = val.strip()
    if val.upper() == "NULL":
        return None
    if val.upper() == "TRUE":
        return True
    if val.upper() == "FALSE":
        return False
    if (val.startswith("'") and val.endswith("'")) or \
       (val.startswith('"') and val.endswith('"')):
        return val[1:-1]
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


def _parse_sql_values(values_str: str) -> List[Any]:
    """Split a SQL VALUES(...) string respecting quotes."""
    values: List[Any] = []
    current = ""
    in_quote = False
    quote_char = ""
    for ch in values_str:
        if in_quote:
            current += ch
            if ch == quote_char:
                in_quote = False
        elif ch in ("'", '"'):
            in_quote = True
            quote_char = ch
            current += ch
        elif ch == ",":
            values.append(_parse_sql_value(current))
            current = ""
        else:
            current += ch
    if current.strip():
        values.append(_parse_sql_value(current))
    return values


def load_patient_metadata(sql_path: Path) -> Dict[str, Any]:
    """Parse a per-patient SQL dump file into a metadata dict.

    Returns dict with keys: ``gender``, ``onset_age``, ``age``, ``hospital``,
    ``etiology`` (dict), ``eeg_focus`` (list), ``medications`` (list),
    ``electrode_focus_rels`` (dict).
    """
    result: Dict[str, Any] = {
        "gender": "",
        "onset_age": -1,
        "age": -1,
        "hospital": "",
        "etiology": {},
        "eeg_focus": [],
        "medications": [],
        "electrode_focus_rels": {},
        "presurgical": False,
        "surgical_decision": "",
    }

    with open(sql_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    for match in _SQL_INSERT_RE.finditer(content):
        table = match.group(1).lower()
        columns = [c.strip().strip('"').lower() for c in match.group(2).split(",")]
        values = _parse_sql_values(match.group(3))

        if len(values) != len(columns):
            continue
        row = dict(zip(columns, values))

        if table == "patient":
            result["gender"] = str(row.get("gender", "")).strip()
            oa = row.get("onsetage")
            result["onset_age"] = int(oa) if oa is not None else -1

        elif table == "admission":
            age = row.get("age")
            result["age"] = int(age) if age is not None else -1
            result["hospital"] = str(row.get("hospital", "")).strip()
            result["presurgical"] = bool(row.get("presurgical", False))
            result["surgical_decision"] = str(row.get("surgicaldecision", "")).strip()

        elif table == "etiology":
            for col in columns:
                if col not in ("id",) and isinstance(row.get(col), bool):
                    result["etiology"][col] = row[col]

        elif table == "eeg_focus":
            result["eeg_focus"].append({
                "localisation": str(row.get("localisation", "")),
                "focus_number": row.get("focus_number"),
                "ieeg_based": row.get("ieeg_based", False),
            })

        elif table == "electrode":
            name = str(row.get("name", "")).strip()
            focus_rel = str(row.get("focus_rel", "")).strip()
            if name and focus_rel:
                result["electrode_focus_rels"][name] = focus_rel

        elif table == "medication":
            result["medications"].append({
                "medicament": row.get("medicament"),
                "dosage": row.get("dosage"),
                "startdate": row.get("startdate"),
                "enddate": row.get("enddate"),
            })

    return result


def _find_patient_sql(pat_id: str, metadata_dir: Optional[Path] = None,
                      annotation_dir: Optional[Path] = None) -> Optional[Path]:
    """Find the SQL metadata file for a patient.

    Returns ``None`` if the metadata directory is missing — upstream
    callers already treat a ``None`` return as "no metadata available"
    and recover gracefully (see ``dataio/epilepsiae.py``), so a missing
    ``Metadata/`` folder must not raise.
    """
    if metadata_dir is None:
        if annotation_dir is None:
            raise ValueError("Either metadata_dir or annotation_dir must be provided")
        metadata_dir = annotation_dir / "Metadata"
    if not metadata_dir.is_dir():
        return None
    for p in metadata_dir.iterdir():
        if p.name.startswith(f"pat_{pat_id}_") and p.suffix in (".sql", ".txt"):
            return p
        if p.name.startswith(f"pat_{pat_id}_") and p.name.endswith(".sql.txt"):
            return p
    return None
