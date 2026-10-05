"""``neuroatlas check <benchmark> -m MODELS [--dataset ...]`` -- one real batch
through every (dataset, model) pair, before you spend GPU hours.

Exit status: 1 if a pair failed, or if not a single pair could be pushed
through (every one skipped for missing data or weights: nothing was
checked); with ``--strict``, also if any pair was skipped. Library log
lines and warnings go to ``--log`` (or the screen, with ``-v``).
"""
from __future__ import annotations

import contextlib
import re
import sys
from typing import Any, Dict, List, Optional

from neuroatlas.cli import MODELS_HELP, Parser
from neuroatlas.cli._table import add_format_arg, render

_FORWARD = re.compile(r"^\(([\d, ]*)\) (finite|constant|(\d+) non-finite)$")


def build_parser() -> Parser:
    p = Parser(prog="neuroatlas check", description=__doc__)
    p.add_argument("benchmark")
    p.add_argument("-m", "--models", required=True, help=MODELS_HELP)
    p.add_argument("--dataset", default="single", metavar="single|full|SLUGS",
                   help="Which datasets (default: single).")
    p.add_argument("--variant", default="default")
    p.add_argument("--num-workers", type=int, default=None, metavar="N",
                   help="Data-loader workers for the one batch (default 0: read in-process, "
                        "the fastest way to get a single batch).")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if any pair was skipped (data or weights missing), not only "
                        "if one failed or none could be checked.")
    add_format_arg(p)
    return p


def outcome(pair) -> str:
    """ok (a finite forward pass) | error | n/a (the channel map skips it) |
    skipped (data or weights missing, nothing pushed through)."""
    if pair.error:
        return "error"
    if str(pair.channel_map).startswith("n/a"):
        return "n/a"
    match = _FORWARD.match(str(pair.forward))
    return "ok" if match and match.group(2) == "finite" else "skipped"


def _row(pair) -> Dict[str, Any]:
    match = _FORWARD.match(str(pair.forward))
    shape = [int(x) for x in match.group(1).split(",") if x.strip()] if match else None
    return {"dataset": pair.dataset, "model": pair.model, "result": outcome(pair),
            "data": pair.data, "weights": pair.weights, "channel_map": pair.channel_map,
            "forward": pair.forward, "shape": shape,
            "finite": (match.group(2) == "finite") if match else None,
            "seconds": round(pair.seconds, 2) if pair.seconds is not None else None,
            "time": f"{pair.seconds:.1f}s" if pair.seconds is not None else None,
            # a failed forward pass's whole message (the table shows its first line)
            "error": getattr(pair, "message", None)}


def main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas.check import check

    args = build_parser().parse_args(argv)
    # JSON and CSV must parse: whatever the models print while they load
    # (REVE's hub code prints "flash_attn not found ...") goes to stderr.
    quiet = contextlib.redirect_stdout(sys.stderr) if args.format != "table" \
        else contextlib.nullcontext()
    with quiet:
        pairs = check(args.benchmark, args.models, args.dataset, args.variant,
                      num_workers=args.num_workers)
    rows = [_row(p) for p in pairs]
    notes = {i: list(p.notes) for i, p in enumerate(pairs) if p.notes}
    render(rows, ["dataset", "model", "data", "weights", "channel_map", "forward", "time"],
           args.format, notes, extra=["result", "shape", "finite", "seconds", "error"])
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["result"]] = counts.get(r["result"], 0) + 1
    ok, errors, skipped = counts.get("ok", 0), counts.get("error", 0), counts.get("skipped", 0)
    nothing = ok == 0 and errors == 0 and skipped > 0
    if args.format == "table":
        total = sum(p.seconds or 0 for p in pairs)
        print(f"pairs: {len(pairs)}   forward passes: {ok}   errors: {errors}   "
              f"skipped: {skipped}   n/a: {counts.get('n/a', 0)}   time: {total:.1f}s")
        from neuroatlas import catalog

        left_out = catalog.load(args.benchmark).left_out_note(args.models)
        if left_out:
            print(left_out)
    if nothing:
        print(f"nothing was checked: no pair could be pushed through ({skipped} skipped for "
              f"missing data or weights; the ↳ lines say what to fetch)", file=sys.stderr)
    elif args.strict and skipped and not errors:
        print(f"--strict: {skipped} pair(s) skipped for missing data or weights",
              file=sys.stderr)
    if errors or nothing or (args.strict and skipped):
        raise SystemExit(1)
