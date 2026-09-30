"""Shared CLI arguments and dataset-path resolution for `embed` and `probe`."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def add_parallel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract embeddings to the cache and exit (no probe fitting).",
    )
    parser.add_argument(
        "--embed-chunk",
        default=None,
        help="Subject chunk for parallel extraction, e.g. '0/4' (chunk 0 of 4 total).",
    )
    parser.add_argument(
        "--task",
        default=None,
        choices=["linear_probe", "native_head_eval", "attention_probe", "lstm_probe"],
        help="Override the evaluation task (default: auto-detect from checkpoint).",
    )
    parser.add_argument(
        "--tune-c",
        default=None,
        help="Comma-separated C values for LR regularization tuning (e.g. 0.001,0.01,0.1,1,10,100).",
    )


def apply_task_override(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if getattr(args, "task", None):
        config["task"] = {"name": args.task}


def parse_embed_chunk(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    parts = value.split("/")
    if len(parts) != 2:
        raise ValueError(f"--embed-chunk must be K/N (e.g. '0/4'), got {value!r}")
    chunk_idx, n_chunks = int(parts[0]), int(parts[1])
    if not (0 <= chunk_idx < n_chunks):
        raise ValueError(f"chunk index {chunk_idx} out of range [0, {n_chunks})")
    return (chunk_idx, n_chunks)


# --------------------------------------------------------------------------
# path resolution, shared by `embed` and `probe`
# --------------------------------------------------------------------------
# 29 of the 40 hand-written dataset specs ship a default like
# ``${EEG_DATA_ROOT}/raw-sleep/cfs``.  Nothing in Python expanded it: the
# cluster wrappers passed a concrete ``--data-root`` instead, so the templates
# worked there and only there.  Run an entrypoint directly, with EEG_DATA_ROOT
# exported exactly as the README says, and the loader looked for a directory
# literally named ``${EEG_DATA_ROOT}`` -- then reported it as "0 recordings",
# or, further downstream, as "cross-validation needs at least 2 subjects".
#
# Expanding here rather than in the manifest keeps the manifest portable: the
# stored value stays a template, and the environment decides at run time.

_PATHISH = ("root", "path", "dir", "file")


class MissingDatasetPath(SystemExit):
    """A required path is neither in the manifest nor on the command line."""


def repo_root() -> str:
    """Where this checkout lives.

    ``${REPO_ROOT}`` appears in nine cache-path defaults. Unlike
    ``${EEG_DATA_ROOT}`` it is not the user's to supply -- we are standing in
    it -- so it is filled in rather than demanded.
    """
    return str(Path(__file__).resolve().parents[2])


def expand_dataset_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand ``${VAR}`` and ``~`` in every string value of a dataset config.

    ``REPO_ROOT`` is supplied from this checkout when the environment does not
    set it.  Variables that remain undefined are left as written, so the error
    downstream still shows the template the user has to fix rather than a path
    with a hole in it.
    """
    environ = dict(os.environ)
    environ.setdefault("REPO_ROOT", repo_root())

    out: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, str) and ("$" in value or value.startswith("~")):
            def _sub(match: "re.Match") -> str:
                name = match.group(1) or match.group(2)
                return environ.get(name, match.group(0))

            expanded = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
                              _sub, value)
            out[key] = os.path.expanduser(expanded)
        else:
            out[key] = value
    return out


def unresolved_paths(config: Dict[str, Any]) -> List[str]:
    """Path-ish keys still holding an unexpanded ``${VAR}`` after expansion."""
    return sorted(
        key for key, value in config.items()
        if isinstance(value, str) and "${" in value
        and any(token in key for token in _PATHISH)
    )


def check_dataset_paths(slug: str, config: Dict[str, Any],
                        required: Optional[List[str]] = None) -> None:
    """Fail early, and say which variable or flag is missing.

    Without this the same mistake surfaces much further downstream -- as a
    ``TypeError: expected str ... not NoneType`` from ``Path(None)``, or as an
    empty recording list that only becomes visible when the splitter is handed
    zero subjects.
    """
    # Keys the manifest marks `runtime_required`: the datamodule takes them
    # positionally, so a null default is not "use the default", it is
    # `Path(None)` several frames later.
    absent = sorted(
        key for key in (required or [])
        if config.get(key) in (None, "")
    )
    if absent:
        raise MissingDatasetPath(
            f"error: {slug} requires " + ", ".join(absent)
            + ", and the manifest declares no default for "
            + ("it" if len(absent) == 1 else "them") + ".\n"
            + "\n".join(f"    --set {key}=/path/to/{key.replace('_', '-')}"
                         for key in absent)
        )

    unresolved = unresolved_paths(config)
    if unresolved:
        wanted = sorted({
            var for key in unresolved
            for var in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", str(config[key]))
        })
        detail = "\n".join(f"    {key} = {config[key]}" for key in unresolved)
        raise MissingDatasetPath(
            f"error: {slug} still has unresolved paths:\n{detail}\n"
            f"Export {' and '.join(wanted)}, or pass the value directly, e.g.\n"
            f"    --set {unresolved[0]}=/path/to/data"
        )
