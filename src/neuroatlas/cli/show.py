"""``neuroatlas show <benchmark>`` -- one benchmark, explained, with the exact
commands it runs."""
from __future__ import annotations

import argparse
import textwrap
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neuroatlas show", description=__doc__.splitlines()[0])
    parser.add_argument("benchmark")
    parser.add_argument("--dataset", default="full", metavar="single|full|SLUGS",
                        help="Which suite's commands to print (default: full).")
    parser.add_argument("--variant", default="default", help="Which variant (default: default).")
    parser.add_argument("-m", "--models", default=None,
                        help="Put this model selection into the printed commands.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas import catalog

    args = build_parser().parse_args(argv)
    bench = catalog.load(args.benchmark)
    steps = bench.steps(args.dataset, args.variant)

    wrap = lambda text, indent="  ": textwrap.fill(text, 78, initial_indent=indent,
                                                   subsequent_indent=indent)
    m = bench.metrics
    print(f"{bench.name} -- {bench.title}  ({bench.domain}{', ' + bench.paper if bench.paper else ''})")
    print(wrap(bench.question))
    print()
    direction = "higher is better" if m.higher_is_better else "lower is better"
    print(f"scored by   {m.headline} ({direction})"
          + (f"; dummy {m.dummy}" if m.dummy is not None and not isinstance(m.dummy, dict) else ""))
    if m.secondary:
        print(f"also        {', '.join(m.secondary)}")
    if bench.derived_from:
        print(f"computed from the results of `{bench.derived_from}`; no embedding or probe of its own")
    print(f"datasets    {len(bench.datasets)} in the full suite; single = {bench.single or '-'}")
    for e in bench.datasets:
        extra = [f"task {e.task}"] if e.task else []
        extra += [e.note] if e.note else []
        print(f"  {e.slug}" + (f"  ({'; '.join(extra)})" if extra else ""))
    for p in bench.planned:
        print(f"  {p['name']}  planned: {p['reason']}")
    if bench.variants:
        print("variants    default (the headline protocol)")
        for v in bench.variants.values():
            print(f"  {v.name}: {v.description}")
    print()
    print(f"commands ({args.dataset} suite, {args.variant} variant):")
    for step in steps:
        print(f"  {step.command(models=args.models)}")
