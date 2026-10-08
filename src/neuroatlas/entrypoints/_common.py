"""Shared CLI arguments and dataset-path resolution for `embed` and `probe`."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from neuroatlas._paths import workspace_root


def add_parallel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract the embeddings to the cache, without fitting probes.",
    )
    parser.add_argument(
        "--embed-chunk",
        default=None,
        help="Extract only chunk K of N of the subjects, as in 0/4.",
    )
    parser.add_argument(
        "--task",
        default=None,
        choices=["linear_probe", "native_head_eval", "attention_probe", "lstm_probe"],
        help="Evaluation task (default: chosen from the checkpoint).",
    )
    parser.add_argument(
        "--tune-c",
        default=None,
        help="C values to try for the logistic regression, separated by commas, as in "
             "0.001,0.01,0.1,1,10,100.",
    )


def resolve_models_arg(value: Optional[str]) -> List[str]:
    """``--models`` -> checkpoint ids; ``[]`` means every registered checkpoint.

    ``all`` and an absent flag keep their old meaning -- the runner's own
    "everything" -- so the paper's command lines run exactly as they did.
    Anything else goes through the selector: an alias (all_fm), a group
    (eeg_fm), a family (reve) or checkpoint ids, comma-separated.
    """
    if value is None or value.strip().lower() == "all":
        return []
    from neuroatlas.selectors import SelectionError, expand_models

    try:
        return expand_models(value)
    except SelectionError as exc:
        # a usage error (exit 2), as `neuroatlas run` reports it
        from neuroatlas.cli import _msg

        _msg.error(str(exc))
        raise SystemExit(2) from None


def report_results(verb: str, results, where: str, *, extraction: bool = False) -> bool:
    """The closing lines of `embed` and `probe`: one ``error:`` line per failed
    (dataset, model, fold) -- its message's first sentence, whole with -v --
    then a line of counts. A failure the runner already reported as it
    happened (it failed, or its data or weights are missing) is counted, not
    repeated; a pair the channel map rules out is neither said nor a
    failure, and one the map has no entry for (invalid) is said as a
    warning. *extraction*: count (dataset, checkpoint) pairs, whatever folds
    their extraction went through, not runs. Returns whether any run failed
    or was skipped (the command's exit status)."""
    from neuroatlas.benchmarking_helpers.runtime.runner import (
        INVALID_CODE,
        RULED_OUT_CODE,
        SKIPPED_CODES,
        outcome_word,
    )
    from neuroatlas.cli import _msg

    said_already = ("runtime_failure", RULED_OUT_CODE, *SKIPPED_CODES)
    invalid_said = set()
    for r in results:
        if r.failure is None or r.failure.code in said_already:
            continue
        fold = (r.metadata or {}).get("fold")
        pair = f"{r.dataset_name}/{r.checkpoint_id}"
        where_pair = pair + (f" fold {fold}" if fold not in (None, "all") else "")
        if r.failure.code == INVALID_CODE:
            # one line per (dataset, checkpoint): no fold of it runs
            if pair not in invalid_said:
                invalid_said.add(pair)
                _msg.warning(f"{pair}: {_msg.brief(r.failure.message)}")
            continue
        _msg.error(f"{where_pair}: {_msg.brief(r.failure.message)}")

    if extraction:
        # a pair's word: the worst of its folds' (one failed fold fails it)
        pairs: Dict[Tuple[str, str], str] = {}
        rank = ("ok", "ruled out", "invalid", "skipped", "failed")
        for r in results:
            key = (r.dataset_name, r.checkpoint_id)
            word = outcome_word(r)
            if rank.index(word) >= rank.index(pairs.get(key, "ok")):
                pairs[key] = word
        tally = {w: sum(1 for v in pairs.values() if v == w) for w in rank}
        datasets = ", ".join(dict.fromkeys(d for d, _ in pairs))
        head = (f"{verb}: {tally['ok']} of {_msg.plural(len(pairs), 'checkpoint')} ok"
                + (f" on {datasets}" if datasets else ""))
    else:
        tally = {w: 0 for w in ("ok", "ruled out", "invalid", "skipped", "failed")}
        for r in results:
            tally[outcome_word(r)] += 1
        head = f"{verb}: {_msg.plural(tally['ok'], 'run')} ok"
    rest = [f"{tally['failed']} failed"] + [
        f"{tally[w]} {w}" for w in ("skipped", "ruled out", "invalid") if tally[w]]
    print(f"{head}, {', '.join(rest)} ({where})")
    return bool(tally["failed"] or tally["skipped"])


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
    """What ``${REPO_ROOT}`` expands to: ``$NEUROATLAS_HOME``.

    No shipped manifest uses it any more (their defaults are written against
    ``${EEG_DATA_ROOT}`` and ``${EEG_CACHE_ROOT}``); a few adapter fallbacks
    still do. It used to be the source checkout, which put downloads inside
    the source tree.
    """
    return str(workspace_root())


def expand_dataset_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand ``${VAR}`` and ``~`` in every string value of a dataset config.

    ``REPO_ROOT`` is supplied from this checkout when the environment does not
    set it.  Variables that remain undefined are left as written, so the error
    downstream still shows the template the user has to fix rather than a path
    with a hole in it.
    """
    environ = dict(os.environ)
    environ.setdefault("REPO_ROOT", repo_root())
    # Prepared-cache defaults are written against the cache root
    # (${EEG_CACHE_ROOT}/prepared/<slug>), which has a default even when no
    # variable is exported.
    if "EEG_CACHE_ROOT" not in environ:
        from neuroatlas import config as _config

        environ["EEG_CACHE_ROOT"] = str(_config.get("cache_root"))

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
            f"error: {slug} needs " + ", ".join(absent)
            + ", and no default folder is set for "
            + ("it" if len(absent) == 1 else "them") + "\n"
            + "\n".join(f"fix: neuroatlas config set {slug}.{key} DIR" for key in absent)
        )

    unresolved = unresolved_paths(config)
    if unresolved:
        wanted = sorted({
            var for key in unresolved
            for var in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", str(config[key]))
        })
        detail = "\n".join(f"  {key} = {config[key]}" for key in unresolved)
        from neuroatlas import config as user_config

        settings = {s.env: s.key for s in user_config.SETTINGS.values()}
        fixes = [f"neuroatlas config set {settings[v]} DIR" if v in settings
                 else f"export {v}=DIR" for v in wanted]
        names = " and no ".join(settings.get(v, v).replace("_", " ") for v in wanted)
        raise MissingDatasetPath(
            f"error: {slug}: no {names} is set, so these folders are unknown:\n{detail}\n"
            + "\n".join(f"fix: {f}" for f in fixes)
            + f"\nfix: neuroatlas config set {slug}.{unresolved[0]} DIR (this dataset alone)"
        )
