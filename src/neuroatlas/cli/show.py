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
    parser.add_argument("--dataset", default="full", metavar="single|full|NAMES",
                        help="Whose commands to print: single (the quick dataset), full (all datasets, "
                             "the default) or dataset names.")
    parser.add_argument("--variant", default="default", help="Which variant (default: default).")
    parser.add_argument("-m", "--models", default=None,
                        help="Put this checkpoint selection into the printed commands.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas import catalog

    args = build_parser().parse_args(argv)
    bench = catalog.load(args.benchmark)
    args.variant = bench.variant(args.variant).name     # an earlier name: the variant's
    steps = bench.steps(args.dataset, args.variant, prepare=True)
    # The same arguments as run/default_runs.sh, which writes each verb as
    # `python -m neuroatlas.entrypoints.<verb> --models all`; without -m the
    # selection is spelled out here the same way rather than left implicit.
    # The verbs know no benchmark: a family the benchmark leaves out is
    # removed by name (`all,-sleepyco`), and naming one is refused before
    # anything is printed.
    models = bench.models_arg(args.models or "all") if not bench.derived_from else None

    wrap =lambda text, indent="  ": textwrap.fill(text, 78, initial_indent=indent,
                                                   subsequent_indent=indent)
    m = bench.metrics
    print(f"{bench.name}: {bench.title}  ({bench.domain}{', ' + bench.paper if bench.paper else ''})")
    print(wrap(bench.question))
    print()
    for line in scoring_lines(bench):
        print(line)
    if bench.derived_from:
        print(f"computed from the results of `{bench.derived_from}`; no embedding or probe of its own")
    print(f"datasets    {len(bench.datasets)}; the quick dataset (--dataset single): "
          f"{bench.single or '-'}")
    for e in bench.datasets:
        extra = [f"task {e.task}"] if e.task else []
        extra += [e.note] if e.note else []
        print(f"  {e.slug}" + (f"  ({'; '.join(extra)})" if extra else ""))
    for p in bench.planned:
        print(f"  {p['name']}  planned: {p['reason']}")
    if bench.variants:
        print("variants    (--variant NAME; the default is the headline protocol)")
        for v in [bench.variant(), *bench.variants.values()]:
            print(f"  {v.name}" + (f" (also accepted: {', '.join(v.aliases)})" if v.aliases else ""))
            print(wrap(v.description, "    "))
            if v.paper:
                print(wrap(f"paper: {v.paper}", "    "))
    for exclusion in bench.excluded_models:
        print(f"left out    {', '.join(exclusion.families)} (every checkpoint): "
              f"{exclusion.reason}")
    print()
    suite = {"full": "all datasets", "single": "the quick dataset"}.get(args.dataset, args.dataset)
    print(f"the paper's commands ({suite}, {args.variant} variant; `neuroatlas run` runs "
          f"them, adding --folds and --output-root):")
    for step in steps:
        if step.verb == "hypnogram":
            print(f"  {_hypnogram_command(step, bench)}")
        else:
            print(f"  {step.command(models=models)}")
    if bench.derived_from:
        print(f"\n  Each hypnogram line reads the results of `neuroatlas run {bench.derived_from} "
              f"--dataset <dataset>`; without --results-dir it also finds those of "
              f"`neuroatlas probe --task sleep_staging`.")
    if any(s.verb == "prepare" for s in steps):
        print("\n  `prepare` builds the file `embed` reads for these cohorts, so it comes "
              "first; `neuroatlas run` does not build it.")
    if not args.models and not bench.derived_from:
        from neuroatlas import selectors

        n = len(bench.select_models("all"))
        if bench.excluded_families():
            print(f"\n  No -m given: `--models {models}` is every ready checkpoint the "
                  f"benchmark evaluates ({n} of {len(selectors.resolve_models('all'))}). "
                  f"Pass -m to print a selection instead, e.g. -m all_fm "
                  f"(`neuroatlas list aliases`).")
        else:
            print(f"\n  No -m given: `--models all` is every ready checkpoint ({n}). Pass "
                  f"-m to print a selection instead, e.g. -m all_fm (`neuroatlas list aliases`).")


def _field(label: str, text: str, indent: int = 12) -> str:
    """``label`` then *text* wrapped at 78 columns, continued at *indent*."""
    head = label.ljust(indent - 1) + " " if len(label) < indent else label + ": "
    return textwrap.fill(text, 78, initial_indent=head, subsequent_indent=" " * indent)


def _named(label: str, text: str) -> str:
    """``label: text``, the column label first so a reader finds the column
    in `neuroatlas results`; not twice when *text* starts with it."""
    return text if text.startswith(label + " ") else f"{label}: {text}"


def scoring_lines(bench) -> List[str]:
    """How the benchmark is scored, from the metric registry and its YAML:
    the headline (label, what it is computed over, direction, chance), what a
    fold is and what ± is over, the chance column, then each secondary
    metric's label and meaning."""
    from neuroatlas import metrics_info

    m = bench.metrics
    head = bench.metric_info(m.headline)
    chance = bench.chance()
    direction = "higher is better" if m.higher_is_better else "lower is better"
    fixed = "" if chance in (None, metrics_info.PREVALENCE, metrics_info.ONE_OVER_C) \
        else f"; {metrics_info.chance_text(chance)}"
    lines = [_field("headline", _named(head.label, m.describe or head.describe())
                    + f"; {direction}{fixed}")]
    if m.fold or m.spread:
        lines.append(_field("folds", "; ".join(
            [*([m.fold] if m.fold else []), *([f"± = {m.spread}"] if m.spread else [])])))
    if chance == metrics_info.PREVALENCE:
        units = metrics_info._plural(head.unit or "item")
        lines.append(_field("chance", f"the test prevalence, the share of positive test {units} "
                                      f"(the chance column of `neuroatlas results`)"))
    elif chance == metrics_info.ONE_OVER_C:
        lines.append(_field("chance", "1/C, C = the dataset's number of classes (the chance "
                                      "column of `neuroatlas results`)"))
    if m.note:
        lines.append(_field("", m.note))
    for i, key in enumerate(m.secondary):
        info = bench.metric_info(key)
        lines.append(_field("also" if i == 0 else "",
                            _named(info.label, info.describe() if info.name == info.label
                                   else f"{info.name}, {info.describe()}")))
    return lines


def _hypnogram_command(step, bench) -> str:
    """The hypnogram line with the directory `neuroatlas run` writes staging
    results to, so it can be pasted as is."""
    import shlex

    from neuroatlas import _paths

    results = _paths.output_dir(bench.derived_from, step.dataset)
    return f"{step.command()} --results-dir {shlex.quote(str(results))}"
