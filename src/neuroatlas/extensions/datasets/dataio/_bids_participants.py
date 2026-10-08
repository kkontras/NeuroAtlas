"""Shared loader for BIDS ``participants.tsv`` demographics.

BIDS spec columns of interest: ``participant_id``, ``age``, ``sex``. Some
datasets carry additional free-text columns (e.g. CHB-MIT stores a
``comment`` field noting that one session was recorded two years later).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict


def load_bids_participants(bids_root: str) -> Dict[str, Dict[str, Any]]:
    """Parse ``<bids_root>/participants.tsv`` → ``{participant_id: {age, gender, comment}}``.

    - ``age``: int, ``-1`` if missing/unparseable.
    - ``gender``: lowercase ``"m"`` / ``"f"`` / ``""`` (unknown).
    - ``comment``: free-text column if present, else ``""``.
    """
    path = Path(bids_root) / "participants.tsv"
    if not path.exists():
        return {}

    def _age(s: str) -> int:
        s = (s or "").strip()
        if not s or s.lower() in ("n/a", "na"):
            return -1
        try:
            return int(float(s))  # tolerate "11.0"
        except ValueError:
            return -1

    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pid = (row.get("participant_id") or "").strip()
            if not pid:
                continue
            sex = (row.get("sex") or "").strip().lower()
            out[pid] = {
                "age": _age(row.get("age", "")),
                "gender": sex if sex in ("m", "f") else "",
                "comment": (row.get("comment") or "").strip(),
            }
    return out
