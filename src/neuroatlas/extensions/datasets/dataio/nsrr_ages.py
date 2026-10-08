"""Participant ages from an NSRR dataset table, for brain age.

Each NSRR study ships its participant variables as CSV tables in the
release's ``datasets/`` folder, one row per participant and a column that
identifies them: ``mesa-sleep-dataset-0.8.0.csv`` (``mesaid``),
``stages-harmonized-dataset-0.3.0.csv`` (``subject_code``),
``homepap-baseline-dataset-0.2.0.csv`` (``nsrrid``). `data download` fetches
the folder with the recordings. Brain age joins a recording to its row on that
identifier.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = ["find_table", "read_ages", "subject_key"]


def _version(path: Path) -> Tuple[int, ...]:
    """``mesa-sleep-dataset-0.8.0.csv`` -> (0, 8, 0)."""
    found = re.search(r"(\d+(?:\.\d+)*)\.csv$", path.name)
    return tuple(int(p) for p in found.group(1).split(".")) if found else ()


def find_table(datasets_dir: Path, pattern: str) -> Optional[Path]:
    """The table *pattern* names in *datasets_dir* (``*`` stands for the
    release version), the newest release when there are several; None when
    there is none."""
    found = sorted(Path(datasets_dir).glob(pattern), key=_version)
    return found[-1] if found else None


def subject_key(value) -> Optional[str]:
    """An identifier as both sides of the join spell it: a number without
    leading zeros or a decimal point (``0001``, ``1.0`` and ``1`` are all
    ``1``), anything else as written, without surrounding spaces."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def read_ages(path: Path, id_column: str, age_column: str) -> Tuple[Dict[str, float], List[str]]:
    """``({subject id: age in years}, the ids whose age cell is empty)`` from
    the table at *path*.

    Raises ``ValueError`` when the table lacks either column, or gives one
    subject two different ages (the join would then be ambiguous).
    """
    import pandas as pd

    table = pd.read_csv(path, usecols=lambda c: c in (id_column, age_column),
                        dtype={id_column: str}, low_memory=False)
    missing = [c for c in (id_column, age_column) if c not in table.columns]
    if missing:
        raise ValueError(f"{Path(path).name} has no {' or '.join(missing)} column")
    ages: Dict[str, float] = {}
    empty: List[str] = []
    for raw_id, raw_age in zip(table[id_column], table[age_column]):
        key = subject_key(raw_id)
        if key is None:
            continue
        age = pd.to_numeric(raw_age, errors="coerce")
        if pd.isna(age):
            empty.append(key)
            continue
        if key in ages and ages[key] != float(age):
            raise ValueError(f"{Path(path).name} gives {id_column} {key} two ages: "
                             f"{ages[key]:g} and {float(age):g}")
        ages[key] = float(age)
    return ages, [k for k in empty if k not in ages]
