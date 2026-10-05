"""``neuroatlas show <benchmark>`` -- one benchmark, explained, with the exact
commands it runs."""
from __future__ import annotations

import argparse
import textwrap
from typing import List, Optional

from neuroatlas.cli import Parser


def build_parser() -> argparse.ArgumentParser:
    parser = Parser(prog="neuroatlas show", description=__doc__)
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
    steps = bench.steps(args.dataset, args.variant, prepare=True)

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
    # The same arguments as run/default_runs.sh, which writes each verb as
    # `python -m neuroatlas.entrypoints.<verb> --models all`; without -m the
    # selection is spelled out here the same way rather than left implicit.
    models = args.models or "all"
    print(f"commands ({args.dataset} suite, {args.variant} variant; the arguments of "
          f"run/default_runs.sh):")
    for step in steps:
        if step.verb == "hypnogram":
            print(f"  {_hypnogram_command(step, bench)}")
        else:
            print(f"  {step.command(models=models)}")
    if bench.derived_from:
        print(f"\n  Each hypnogram line reads the results of `neuroatlas run {bench.derived_from} "
              f"--dataset <slug>`; without --results-dir it also finds those of "
              f"`neuroatlas probe --task sleep_staging`.")
    if any(s.verb == "prepare" for s in steps):
        print("\n  `prepare` builds the file `embed` reads for these cohorts, so it comes "
              "first; `neuroatlas run` does not build it.")
    if not args.models and not bench.derived_from:
        from neuroatlas import selectors

        n = len(selectors.resolve_models("all"))
        print(f"\n  No -m given: `--models all` is every ready checkpoint ({n}). Pass "
              f"-m to print a selection instead, e.g. -m all_fm (`neuroatlas list aliases`).")


def _hypnogram_command(step, bench) -> str:
    """The hypnogram line with the directory `neuroatlas run` writes staging
    results to, so it can be pasted as is."""
    import shlex

    from neuroatlas import _paths

    results = _paths.output_dir(bench.derived_from, step.dataset)
    return f"{step.command()} --results-dir {shlex.quote(str(results))}"
