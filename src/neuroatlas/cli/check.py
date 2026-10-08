"""``neuroatlas check <benchmark> -m MODELS [--dataset ...]`` -- one real batch
through every (dataset, checkpoint) pair, before you spend GPU hours.

Exit status: 1 if a pair failed, or if not a single pair could be pushed
through (every one skipped for missing data or weights: nothing was
checked); with ``--strict``, also if any pair was skipped or invalid. Library
log lines and warnings go to ``--log`` (or the screen, with ``-v``).
"""
from __future__ import annotations

import contextlib
import re
from typing import Any, Dict, List, Optional

from neuroatlas.cli import MODELS_HELP, Parser
from neuroatlas.cli._table import add_format_arg, render

_FORWARD = re.compile(r"^\(([\d, ]*)\) (finite|constant|(\d+) non-finite)$")
#: a row's fix that is only a weights download (gathered under the table)
_DOWNLOAD_FIX = re.compile(r"^\s*fix: neuroatlas models download (\S+)$")

#: The `--dataset` help of check, run and submit: the flag's values in words.
DATASET_HELP = ("single: the benchmark's quick dataset (`neuroatlas list benchmarks` names "
                "it); full: all its datasets; or dataset names, comma-separated.")

#: The values of the table's columns that need saying (`check` prints the
#: ones its table shows).
LEGEND = {
    "skipped": "not checked: its data or weights are not here (the fix line says how to "
               "get them)",
    "ruled out": "the dataset's channel map excludes this checkpoint: it is never run there",
    "invalid": "the dataset's channel map has no entry for this checkpoint's family: not "
               "checked, and never run there",
    "failed": "the batch did not go through; the line under the row says why",
    "forward pass": "(windows, embedding size) of one batch; finite: no NaN or infinite value",
    "none": "this dataset has no channel map: the model gets its channels as recorded",
}


def build_parser() -> Parser:
    p = Parser(prog="neuroatlas check", description=__doc__)
    p.add_argument("benchmark")
    p.add_argument("-m", "--models", required=True, help=MODELS_HELP)
    p.add_argument("--dataset", default="single", metavar="single|full|NAMES",
                   help=f"Which datasets (default: single). {DATASET_HELP}")
    p.add_argument("--variant", default="default",
                   help="A benchmark variant (default: default; `neuroatlas show <benchmark>` "
                        "lists them).")
    p.add_argument("--num-workers", type=int, default=None, metavar="N",
                   help="Data-loader workers for the one batch (default 0: read in-process, "
                        "the fastest way to get a single batch).")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if any pair was skipped (data or weights missing) or invalid "
                        "(no channel map entry), not only if one failed or none could be "
                        "checked.")
    add_format_arg(p)
    return p


def outcome(pair) -> str:
    """ok (a finite forward pass) | failed | ruled out (the channel map
    excludes it) | invalid (the channel map has no entry for its family) |
    skipped (data or weights missing, nothing pushed through)."""
    from neuroatlas.cli import _msg

    if str(pair.channel_map) == _msg.INVALID:
        return _msg.INVALID
    if pair.error:
        return "failed"
    if str(pair.channel_map).startswith((_msg.RULED_OUT, "n/a")):
        return _msg.RULED_OUT
    match = _FORWARD.match(str(pair.forward))
    return "ok" if match and match.group(2) == "finite" else _msg.SKIPPED


def _row(pair) -> Dict[str, Any]:
    from neuroatlas import progress

    match = _FORWARD.match(str(pair.forward))
    shape = [int(x) for x in match.group(1).split(",") if x.strip()] if match else None
    return {"dataset": pair.dataset, "model": pair.model, "result": outcome(pair),
            "data": pair.data, "weights": pair.weights, "channel_map": pair.channel_map,
            "forward": pair.forward, "shape": shape,
            "finite": (match.group(2) == "finite") if match else None,
            "seconds": round(pair.seconds, 2) if pair.seconds is not None else None,
            "time": progress.duration(pair.seconds) if pair.seconds is not None else None,
            # a failed forward pass's whole message (the table shows its first line)
            "error": getattr(pair, "message", None)}


def _family(ident: str) -> str:
    """A checkpoint id's model family."""
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    return next((s.model_family for s in checkpoint_registry() if s.identifier == ident), ident)


class _LogLines:
    """A stdout for the models while they load: what they print (REVE's hub
    code: "flash_attn not found ...") becomes INFO log lines -- shown with
    -v, kept by --log -- instead of landing in the table or the JSON."""

    def __init__(self, logger):
        self._logger, self._buffer = logger, ""

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._logger.info("%s", line)
        return len(text)

    def flush(self):
        if self._buffer.strip():
            self._logger.info("%s", self._buffer)
        self._buffer = ""


def main(argv: Optional[List[str]] = None) -> None:
    import logging

    from neuroatlas.check import check
    from neuroatlas.cli import _msg

    args = build_parser().parse_args(argv)
    printed = _LogLines(logging.getLogger("neuroatlas.check"))
    with contextlib.redirect_stdout(printed):
        pairs = check(args.benchmark, args.models, args.dataset, args.variant,
                      num_workers=args.num_workers)
    printed.flush()
    rows = [_row(p) for p in pairs]
    notes = {i: list(p.notes) for i, p in enumerate(pairs) if p.notes}
    for i, (pair, row) in enumerate(zip(pairs, rows)):
        # a failed pair whose message names no remedy: the command that shows
        # the whole error and the model's own warnings
        if row["result"] == "failed" and not any(n.strip().startswith("fix:")
                                                 for n in notes.get(i, [])):
            notes.setdefault(i, []).append(
                f"  fix: neuroatlas -v check {args.benchmark} -m {pair.model} --dataset "
                f"{pair.dataset} (the whole error, and the model's warnings)")
    downloads: List[str] = []
    invalid_cmd = None
    if args.format == "table":
        # one `models download` for every pair skipped for its weights, under
        # the table, rather than one fix line per row (JSON keeps each row's)
        for lines in notes.values():
            for line in list(lines):
                match = _DOWNLOAD_FIX.match(line)
                if match:
                    downloads += [i for i in match.group(1).split(",") if i not in downloads]
        if len(downloads) > 1:
            for i, lines in notes.items():
                notes[i] = [line for line in lines if not _DOWNLOAD_FIX.match(line)]
        else:
            downloads = []
        # one command without every family the channel maps have no entry
        # for, under the table, rather than one per family under its rows
        invalid_rows = [i for i, r in enumerate(rows) if r["result"] == _msg.INVALID]
        families = sorted({_family(rows[i]["model"]) for i in invalid_rows})
        if len(families) > 1:
            from neuroatlas.run import invalid_fix

            for i in invalid_rows:
                notes[i] = [line for line in notes.get(i, [])
                            if not line.strip().startswith("fix:")]
            invalid_cmd = invalid_fix(args.models, families,
                                      f"neuroatlas check {args.benchmark} -m "
                                      + ",".join([args.models, *(f"-{f}" for f in families)]))
    shown_rows = rows
    if args.format == "table":
        # a pair that pushed nothing through: say so in its forward-pass cell
        shown_rows = [{**r, "forward": "not run"} if r["forward"] == "-" else r for r in rows]
    render(shown_rows, ["dataset", "model", "data", "weights", "channel_map", "forward", "time"],
           args.format, notes, extra=["result", "shape", "finite", "seconds", "error"],
           labels={"model": "checkpoint", "channel_map": "channel map",
                   "forward": "forward pass"})
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["result"]] = counts.get(r["result"], 0) + 1
    ok, failed, skipped = (counts.get("ok", 0), counts.get("failed", 0),
                           counts.get(_msg.SKIPPED, 0))
    invalid = counts.get(_msg.INVALID, 0)
    nothing = ok == 0 and failed == 0 and skipped > 0
    if args.format == "table":
        from neuroatlas import models, progress

        total = sum(p.seconds or 0 for p in pairs)
        print("\n" + _msg.counts(len(pairs), "pair" if len(pairs) == 1 else "pairs",
                                 [("ok", ok), ("failed", failed), (_msg.SKIPPED, skipped),
                                  (_msg.RULED_OUT, counts.get(_msg.RULED_OUT, 0)),
                                  (_msg.INVALID, invalid)],
                                 keep_zero=("ok", "failed", _msg.SKIPPED))
              + f" ({progress.duration(total)})")
        shown = set(counts) | {r["channel_map"] for r in rows} | {r["weights"] for r in rows}
        explained = [(w, LEGEND[w]) for w in ("skipped", "ruled out", "invalid", "failed", "none")
                     if w in shown]
        explained += [(w, models.STATE_MEANINGS[w]) for w in models.STATE_MEANINGS
                      if w in shown and w != "found"]
        if any(r["shape"] for r in rows):
            explained.append(("forward pass", LEGEND["forward pass"]))
        print("\n".join(_msg.legend(explained)))
        if downloads:
            _msg.note(f"{_msg.plural(len(downloads), 'checkpoint')} skipped for missing "
                      f"weights", f"neuroatlas models download {','.join(downloads)}")
        if invalid_cmd:
            _msg.note(f"{_msg.plural(invalid, 'pair')} invalid: the channel map has no entry "
                      f"for their family", invalid_cmd)
        from neuroatlas import catalog

        left_out = catalog.load(args.benchmark).left_out_note(args.models)
        if left_out:
            _msg.note(left_out)
    if nothing:
        ruled_out = counts.get(_msg.RULED_OUT, 0)
        never = [f"{ruled_out} ruled out" if ruled_out else "",
                 f"{invalid} invalid" if invalid else ""]
        never = [n for n in never if n]
        if skipped == 1:
            only = next(r for r in rows if r["result"] == _msg.SKIPPED)
            data_here = str(only["data"]).startswith(("found", "prepared", "fetched"))
            what = (f"the only pair to check ({only['dataset']}, {only['model']}) was skipped: "
                    f"its {'weights are' if data_here else 'data is'} not here")
        else:
            what = f"{_msg.plural(skipped, 'pair')} skipped for missing data or weights"
        _msg.error(f"nothing was checked: {what}"
                   + (f" ({', '.join(never)})" if never else "")
                   + (" (the fix line above gets it)" if skipped == 1 else
                      " (the fix lines above say how to get them)"))
    elif args.strict and (skipped or invalid) and not failed:
        parts = [f"{_msg.plural(skipped, 'pair')} skipped for missing data or weights"
                 if skipped else "", f"{_msg.plural(invalid, 'pair')} invalid" if invalid else ""]
        _msg.error(f"{' and '.join(p for p in parts if p)}, and --strict fails on them (the "
                   f"lines above say why)")
    if failed or nothing or (args.strict and (skipped or invalid)):
        raise SystemExit(1)
