"""``neuroatlas rescore <benchmark>`` -- recompute every result's metrics from
the test predictions its probe saved, and rewrite results.json; nothing is
fitted again.

Each fold of a probe saves its test predictions (predictions.npz in its probe
folder; the format is in docs/guide/results.md and ``neuroatlas.predictions``),
so a metric that was fixed or added after a run applies to it without probing
again. A result probed before predictions were saved is listed with the
command that probes it again.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

from neuroatlas.cli import MODELS_HELP, Parser
from neuroatlas.cli._table import add_format_arg, render

#: what each outcome reads as in the table
_STATUS = {"same": "rescored, same", "changed": "rescored, changed",
           "skipped": "skipped", "failed": "failed fold, not rescored"}


def build_parser() -> Parser:
    from neuroatlas.cli.check import BENCHMARK_HELP

    p = Parser(prog="neuroatlas rescore",
               description="Recompute the metrics of every result from the test predictions "
                           "its probes saved, and rewrite results.json. Nothing is fitted "
                           "again. Use it when a metric was fixed or added after a run.")
    p.add_argument("benchmark", help=BENCHMARK_HELP)
    p.add_argument("--dataset", default=None, metavar="NAMES",
                   help="Only these datasets, separated by commas (default: every dataset "
                        "with results).")
    p.add_argument("-m", "--models", default=None,
                   help=MODELS_HELP + " Default: every checkpoint with results.")
    p.add_argument("--variant", default=None,
                   help="Only this variant (default: every variant).")
    p.add_argument("--output-root", type=Path, default=None, metavar="DIR",
                   help="Results folder (default: the output_root setting).")
    add_format_arg(p)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas import _paths, rescore as rs
    from neuroatlas.cli.results import metric_label

    args = build_parser().parse_args(argv)
    output_root = args.output_root.expanduser().resolve() if args.output_root else None
    report = rs.rescore(args.benchmark, datasets=args.dataset, models=args.models,
                        output_root=output_root, variant=args.variant)
    if not report.outcomes:
        where = (output_root or _paths.output_dir()) / report.benchmark
        raise SystemExit(f"error: no results to rescore for {report.benchmark} in {where}\n"
                         f"fix: neuroatlas run {report.benchmark} -m MODELS")

    show_variant = any(o.variant != "default" for o in report.outcomes)
    table = args.format == "table"
    rows, notes = [], {}
    for o in report.outcomes:
        rows.append({"benchmark": report.benchmark, "dataset": o.dataset, "variant": o.variant,
                     "model": o.model, "fold": o.fold, "task": o.task,
                     "result": _STATUS.get(o.status, o.status) if table else o.status,
                     "metric": report.headline, "before": o.before, "after": o.after,
                     "predictions": o.predictions, "results_file": str(o.file),
                     "fix": (rs.fix_command(report.benchmark, o, output_root)
                             if o.status == "skipped" else None)})
        # a table lists the skipped ones under it, with their fix
        if o.reason and not (table and o.status == "skipped"):
            notes[len(rows) - 1] = [o.reason]
    from neuroatlas import catalog

    head = metric_label(report.headline, catalog.load(report.benchmark))
    render(rows, ["dataset", *(["variant"] if show_variant else []), "model", "fold", "result",
                  "before", "after"],
           args.format, notes,
           extra=["benchmark", "variant", "task", "metric", "predictions", "results_file", "fix"],
           labels={"model": "checkpoint", "before": f"{head} before", "after": "after"},
           formats={"before": "{:.6g}", "after": "{:.6g}"})
    if args.format != "table":
        return

    from neuroatlas.cli import _msg

    n_rescored = report.count("same") + report.count("changed")
    # the files whose metrics changed (one rescored without a change keeps its
    # numbers; only the time it was rescored is added)
    updated = len({str(o.file) for o in report.outcomes if o.status == "changed"})
    print("\n" + _msg.counts(n_rescored, "rescored from saved predictions",
                             [("changed", report.count("changed")),
                              ("the same", report.count("same"))],
                             keep_zero=("changed", "the same"))
          + (f"; {_msg.plural(updated, 'results file')} updated" if updated else ""),
          flush=True)
    failed = report.count("failed")
    if failed:
        _msg.note(f"{_msg.plural(failed, 'failed fold')} left as they are: no metrics to rescore")
    # one block per (dataset, variant, model, reason, folder): the folds it
    # covers, why, and the command that probes them again
    skipped = OrderedDict()
    for o in report.outcomes:
        if o.status == "skipped":
            key = (o.dataset, o.variant, o.model, o.reason, str(Path(o.file).parent))
            skipped.setdefault(key, []).append(o)
    for (dataset, variant, model, reason, _), group in skipped.items():
        folds = ", ".join(o.fold for o in group)
        label = "fold" if len(group) == 1 else "folds"
        what = f"{dataset}/{model}" + (f" ({variant})" if variant != "default" else "")
        _msg.say("skipped", f"{what} {label} {folds}: {reason}",
                 rs.fix_command(report.benchmark, group[0], output_root))
