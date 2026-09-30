"""``neuroatlas check <benchmark> -m MODELS [--dataset ...]`` -- one real batch
through every (dataset, model) pair, before you spend GPU hours."""
from __future__ import annotations

import argparse
from typing import List, Optional

from neuroatlas.cli._table import add_format_arg, render


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neuroatlas check", description=__doc__.split("--", 1)[1].strip())
    p.add_argument("benchmark")
    p.add_argument("-m", "--models", required=True,
                   help="An alias (all_fm ...), group, family or checkpoint ids.")
    p.add_argument("--dataset", default="single", metavar="single|full|SLUGS",
                   help="Which datasets (default: single).")
    p.add_argument("--variant", default="default")
    add_format_arg(p)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas.check import check

    args = build_parser().parse_args(argv)
    pairs = check(args.benchmark, args.models, args.dataset, args.variant)
    rows = [{"dataset": p.dataset, "model": p.model, "data": p.data, "weights": p.weights,
             "channel_map": p.channel_map, "forward": p.forward} for p in pairs]
    notes = {i: p.notes for i, p in enumerate(pairs) if p.notes}
    render(rows, ["dataset", "model", "data", "weights", "channel_map", "forward"],
           args.format, notes if args.format == "table" else None)
    errors = sum(p.error for p in pairs)
    if args.format == "table":
        passes = sum(p.forward.endswith("finite") for p in pairs)
        print(f"pairs: {len(pairs)}   forward passes: {passes}   errors: {errors}")
    if errors:
        raise SystemExit(1)
