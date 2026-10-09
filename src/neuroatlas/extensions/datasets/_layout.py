"""Where, under a dataset's folder, a reader finds what it reads.

Every dataset's folder defaults to ``<data root>/<dataset>``. What a host
serves often keeps a layout of its own inside it (a version folder, an
``edf/`` folder, the top folder of an archive), and a reader that reads one
of those sub-folders looks for it here instead of asking the user to point
the setting at it. A setting that already points at the sub-folder itself
keeps working: the folder is used as given when it already holds what the
reader looks for.

Standard library only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, Path]


def descend(root: PathLike, candidates: Iterable[str], marker: str) -> Path:
    """The folder under *root* that holds *marker* (a file or folder name,
    or a glob): *root* itself when it does, else the first of *candidates*
    (paths relative to *root*, globs allowed, tried in order) that does.
    *root* unchanged when none does, so the reader's own error names the
    folder the user set.
    """
    if not str(root or ""):
        return Path(root or "")
    base = Path(root)
    if _holds(base, marker):
        return base
    for pattern in candidates:
        hits = sorted(base.glob(pattern)) if any(c in pattern for c in "*?[") \
            else [base / pattern]
        for hit in hits:
            if hit.is_dir() and _holds(hit, marker):
                return hit
    return base


def _holds(folder: Path, marker: str) -> bool:
    if any(c in marker for c in "*?["):
        return next(iter(folder.glob(marker)), None) is not None
    return (folder / marker).exists()
