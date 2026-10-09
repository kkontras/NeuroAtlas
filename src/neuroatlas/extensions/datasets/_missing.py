"""The one shape of a "the data is not here" error, for every reader.

    tuab: no data in /data/TUH/tuh_eeg/tuh_eeg_abnormal/v3.0.1/edf
    fix: neuroatlas data download tuab
    fix: neuroatlas config set tuab.raw_root DIR

The first line names the dataset and the folder it looked in (what the runner
shows of a failure); the ``fix:`` lines are the two remedies, one per line:
``data download`` fetches the dataset, or says where to get it when it has no
download; ``config set`` points the tool at a copy elsewhere.
"""
from __future__ import annotations

from typing import Optional


def no_data(slug: str, where, key: str, *, what: str = "no data in",
            value: str = "DIR", download: bool = True,
            detail: Optional[str] = None) -> str:
    """The text of the error: ``<slug>: <what> <where>`` and its fix lines.

    *key* is the setting ``neuroatlas config set <slug>.<key>`` takes (a
    folder: ``DIR``; a file: ``value="FILE"``). *detail*, when given, is said
    after the folder, in the same sentence."""
    from neuroatlas.cli import _msg

    where = str(where) if where else "(no folder set)"
    head = f"{slug}: {what} {where}" + (f" ({detail})" if detail else "")
    fixes = ([f"neuroatlas data download {slug}"] if download else []) + [
        f"neuroatlas config set {slug}.{key} {value}"]
    return "\n".join([head, *(f"{_msg.FIX}: {fix}" for fix in fixes)])


def warn_no_demographics(log, slug: str, path, *, download: bool = True) -> None:
    """Once per command: *slug*'s demographics table is not at *path*, so its
    recordings carry no age or sex (sleep staging does not need them; brain
    age does)."""
    from neuroatlas import quiet
    from neuroatlas.cli import _msg

    text = (f"{slug}: no demographics table at {path}, so age and sex are unknown "
            f"(brain age needs them)")
    if download:
        text += f"\n{_msg.FIX}: neuroatlas data download {slug}"
    quiet.warn_once(log, f"no demographics:{path}", "%s", text)
