"""``neuroatlas results <benchmark>``.

One schema in every format: a missing number is n/a in a table and null in
JSON, every JSON row names its benchmark and metric, and the lines a table
prints under a row are its ``note``. What each metric is -- its column label,
meaning, unit and chance level -- is the metric registry's
(:mod:`neuroatlas.metrics_info`); the title says, from the benchmark's YAML,
what the headline is computed over, what a fold is and what ± is over.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from neuroatlas.cli import Parser, _msg
from neuroatlas.cli._table import add_format_arg, render


def metric_label(name: str, bench=None) -> str:
    """What a metric's column is called in a table (JSON and CSV keep the key)."""
    from neuroatlas import metrics_info

    return metrics_info.label(name, bench)


def title_lines(bench, variant: Optional[str] = None, *, chance_column: bool = False) -> List[str]:
    """The lines above a `results` table: what the headline is computed over,
    its direction and fixed chance level; what a fold is and what ± is over;
    what the chance column holds; then the secondary columns' names."""
    from neuroatlas import metrics_info

    m = bench.metrics
    head = bench.metric_info(m.headline)
    chance = bench.chance()
    varying = chance in (metrics_info.PREVALENCE, metrics_info.ONE_OVER_C)
    first = (f"{bench.name}" + (f" ({variant} variant)" if variant else "")
             + f": {m.describe or head.name}; "
             + ("higher is better" if m.higher_is_better else "lower is better")
             + ("" if varying or chance is None else f"; {metrics_info.chance_text(chance)}"))
    lines = [first]
    if m.spread:
        lines.append(f"folds: {m.fold or 'the dataset protocol'}; ± = {m.spread}")
    elif m.fold:
        lines.append(m.fold)
    if chance_column and chance == metrics_info.PREVALENCE:
        units = metrics_info._plural(head.unit or "item")
        lines.append(f"chance: the test prevalence, the share of positive test {units} (from "
                     f"each fold's test confusion matrix), mean over the folds; {head.label} "
                     f"at chance equals it")
    elif chance_column and chance == metrics_info.ONE_OVER_C:
        lines.append("chance: 1/C, C = the dataset's number of classes")
    if m.note:
        lines.append(m.note)
    if m.secondary:
        lines.append("also: " + ", ".join(
            f"{metric_label(k, bench)} = {bench.metric_info(k).name}" for k in m.secondary))
    return lines


def build_results_parser() -> Parser:
    from neuroatlas.cli.check import BENCHMARK_HELP

    p = Parser(
        prog="neuroatlas results",
        description="Summarise the results of a benchmark, with one row per dataset, variant "
                    "and checkpoint. Each row shows the mean and spread of the headline metric "
                    "over the folds, the number of folds, the chance level and the other "
                    "metrics.")
    p.add_argument("benchmark", help=BENCHMARK_HELP)
    p.add_argument("paths", nargs="*",
                   help="results.json files, globs or folders to read (default: every "
                        "results.json under `<output root>/<benchmark>/`).")
    p.add_argument("--variant", default=None,
                   help="Only show this variant (default: every variant).")
    p.add_argument("--output-root", type=Path, default=None, metavar="DIR",
                   help="Results folder to read (default: the output_root setting).")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show the error message of each failed fold.")
    add_format_arg(p)
    return p


def _folds_text(s, loso: bool = False) -> str:
    """``3/5``; ``LOSO 1/9`` for a leave-one-subject-out dataset (a fold
    per subject)."""
    text = f"{s.n_folds}/{s.n_expected}" if s.n_expected else str(s.n_folds)
    return f"LOSO {text}" if loso else text


def _na_lines(s) -> List[str]:
    """Why a row has no headline value: the channel map rules the pair out,
    the metric does not apply to the dataset, or no fold recorded what it is
    computed from (shown without -v: they are short)."""
    if s.status == "n/a":
        return _msg.lines(_msg.RULED_OUT, "the dataset's channel map excludes this "
                                          "checkpoint: it is never run there")
    if s.na_reason:
        return _msg.lines("n/a", s.na_reason)
    return []


def _failure_lines(s, machine: bool = False) -> List[str]:
    """One ``error:`` line per distinct failure: the folds it hit, the first
    sentence of the message the run stored in results.json (whole with -v),
    its code; then the folds not run yet."""
    groups: Dict[Tuple[str, str], List[str]] = {}
    for e in s.errors:
        groups.setdefault((e["code"], (e.get("message") or "").strip()), []).append(str(e["fold"]))
    lines = []
    for (code, message), folds in groups.items():
        label = "fold" if len(folds) == 1 else "folds"
        text = _msg.brief(message, keep_fix=False, hint=not machine) if message else "failed"
        first, *rest = text.splitlines() or [""]        # -v: the whole message
        # the failure code is in the JSON's `errors`, not on the line
        lines += _msg.lines("error", "\n".join([f"{label} {', '.join(folds)}: {first}",
                                                *rest]))
    if s.n_expected and s.n_folds + s.n_failed < s.n_expected:
        lines += _msg.lines("warning", f"{s.n_expected - s.n_folds - s.n_failed} of "
                                       f"{s.n_expected} folds not run yet")
    return lines


def _fold_lines(s, show_c: bool) -> List[str]:
    """`results -v`: each fold's headline value, and the C its probe chose
    from the grid where the protocol has one, under the row."""
    if not s.per_fold:
        return []
    parts = []
    for f in s.per_fold:
        value = "n/a" if f["value"] is None else f"{f['value']:.3f}"
        c = f" (C {f['C']:g})" if show_c and f.get("C") is not None else ""
        parts.append(f"{f['fold']}: {value}{c}")
    return [f"by fold: {', '.join(parts)}"]


def _has_c_grid(bench) -> bool:
    """Whether the probe chooses C from a grid: the task preset's (sleep
    staging) or the command's ``--tune-c a,b,...`` (seizure detection)."""
    if _c_grid(bench):
        return True
    args = [a for e in bench.datasets for a in bench.probe_args(e)]
    return any(a == "--tune-c" and i + 1 < len(args) and "," in args[i + 1]
               for i, a in enumerate(args))


def _c_grid(bench) -> Optional[List[float]]:
    """The C grid the benchmark's task preset fixes (sleep staging's), if any."""
    import json

    from neuroatlas import _paths

    path = _paths.configs_dir("tasks", f"{bench.task}.json") if bench.task else None
    if path is None or not path.is_file():
        return None
    grid = (json.loads(path.read_text()).get("probe") or {}).get("c_values")
    return grid if isinstance(grid, list) and len(grid) > 1 else None


def summary_rows(bench, summaries, *, machine: bool = True, verbose: bool = False):
    """`results`' rows, one per summary, in the JSON schema (keys, not
    labels), and ``{row index: the lines under it}``: why a row is n/a,
    always; why folds failed, with -v or in a machine format."""
    m = bench.metrics
    loso = {(slug, v) for v in bench.variant_names() for slug in bench.loso_datasets(v)}
    features = any(s.features for s in summaries)
    show_c = verbose and not machine and _has_c_grid(bench)
    rows, notes = [], {}
    for s in summaries:
        row = {"benchmark": bench.name, "metric": m.headline,
               **({"at": m.at_text()} if m.at else {}),
               "dataset": s.dataset, "variant": s.variant, "model": s.model, "status": s.status,
               "mean": s.mean, "std": s.std,
               "folds": _folds_text(s, (s.dataset, s.variant) in loso),
               "chance": s.chance,
               **({"features": f"{s.features[0]}/{s.features[1]}" if s.features else None}
                  if features else {}),
               "n_folds": s.n_folds, "n_expected": s.n_expected, "n_failed": s.n_failed,
               "failures": s.failures, "errors": s.errors,
               # each fold's C, chosen from the task's grid on validation kappa
               **({"C": s.c_chosen} if s.c_chosen else {}),
               "per_fold": s.per_fold}
        for name in m.secondary:
            row[name] = s.secondary.get(name)
        rows.append(row)
        lines = (_na_lines(s) + (_failure_lines(s, machine) if verbose or machine else [])
                 + (_fold_lines(s, show_c) if verbose and not machine else []))
        if lines:
            notes[len(rows) - 1] = lines
    return rows, notes


def results_main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas import catalog, results as res

    args = build_results_parser().parse_args(argv)
    bench = catalog.load(args.benchmark)
    left_out: List[str] = []
    summaries, dropped = res.benchmark_summary(bench.name, args.paths or None, args.output_root,
                                               variant=args.variant, left_out=left_out)
    m = bench.metrics
    left_out_line = (f"not shown: results of {', '.join(left_out)}, which the {bench.name} "
                     f"benchmark leaves out") if left_out else None
    if not summaries:
        where = ", ".join(args.paths) if args.paths else str(
            (args.output_root or res._paths.output_dir()) / bench.name)
        which = f" (variant {args.variant})" if args.variant else ""
        again = f"neuroatlas run {bench.name} -m MODELS" + (
            f" --variant {args.variant}" if args.variant and args.variant != "default" else "")
        if left_out_line:
            _msg.note(left_out_line)
        raise SystemExit(f"error: no results for {bench.name}{which} in {where}\nfix: {again}")
    # the variant column: always in machine formats, in the table once a
    # variant other than the default has results
    show_variant = any(s.variant != res.DEFAULT_VARIANT for s in summaries)
    # leave-one-subject-out datasets: their folds read "LOSO k/N"
    loso = {(slug, v) for v in bench.variant_names() for slug in bench.loso_datasets(v)}

    from neuroatlas import metrics_info

    # A chance column where chance differs per row (AUPRC: the test
    # prevalence; balanced accuracy: 1/C); a fixed one is in the title.
    chance_column = bench.chance() in (metrics_info.PREVALENCE, metrics_info.ONE_OVER_C)
    # one value per row, no spread (the hypnogram's pooled mean r): no ± and
    # no folds column, but how many features went into it
    spread = bool(m.spread)
    features = any(s.features for s in summaries)
    run = f"neuroatlas run {bench.name}" + (
        f" --output-root {args.output_root}" if args.output_root else "")
    rows, notes = summary_rows(bench, summaries, machine=args.format != "table",
                               verbose=args.verbose)
    columns = ["dataset", *(["variant"] if show_variant else []), "model",
               *(["metric"] if args.format == "csv" else []), "mean",
               *(["std", "folds"] if spread else []),
               *(["chance"] if chance_column else []),
               *(["features"] if features else []), *m.secondary]
    extra = ["benchmark", "metric", *(["at"] if m.at else []), "variant", "status",
             *([] if spread else ["std", "folds"]), *([] if chance_column else ["chance"]),
             "n_folds", "n_expected", "n_failed", "failures", "errors", "per_fold",
             *(["C"] if any(s.c_chosen for s in summaries) else [])]
    if args.format in ("table", "md"):
        # The table names each column by its metric (machine formats keep the
        # keys): the headline's mean and spread, then the secondary metrics.
        if args.format == "table":
            for line in title_lines(bench, args.variant, chance_column=chance_column):
                print(line)
        names = {"model": "checkpoint", "mean": metric_label(m.headline, bench), "std": "±",
                 **{k: metric_label(k, bench) for k in m.secondary}}
        from neuroatlas.cli._table import NOT_APPLICABLE

        # one fold has no spread: `-` (explained under the table), not n/a
        rows = [{names.get(k, k): (NOT_APPLICABLE if k == "std" and v is None
                                   and r.get("n_folds") == 1 else v)
                 for k, v in r.items()} for r in rows]
        columns = [names.get(c, c) for c in columns]
    render(rows, columns, args.format, notes, extra=extra)
    if args.format == "table" and spread and any(s.n_folds == 1 for s in summaries):
        print("\n" + "\n".join(_msg.legend([("± -", "one fold: no spread over folds")])))
    if args.format == "table":
        failed = sum(1 for s in summaries if s.n_failed)
        short = sum(1 for s in summaries if s.status != "n/a" and not s.n_failed and not s.complete)
        if dropped:
            _msg.warning(f"{_msg.plural(dropped, 'result')} recorded more than once (same "
                         f"dataset, variant, checkpoint, fold and task): the newest copy of "
                         f"each is counted")
        if left_out_line:
            _msg.note(left_out_line)
        if failed:
            _msg.warning(f"{_msg.plural(failed, 'row')} with failed folds"
                         + ("" if args.verbose else " (-v: why)"))
        if short:
            rows_short = [s for s in summaries
                          if s.status != "n/a" and not s.n_failed and not s.complete]
            datasets = ",".join(dict.fromkeys(s.dataset for s in rows_short))
            models = ",".join(dict.fromkeys(s.model for s in rows_short))
            _msg.warning(f"{_msg.plural(short, 'row')} with fewer folds than the protocol (the "
                         f"folds column, k/N): not comparable with a full run",
                         f"{run} --dataset {datasets} -m {models}")
        untuned = [s for s in summaries if s.n_folds and not s.c_chosen] if _c_grid(bench) else []
        if untuned:
            # probed without the task's C grid (sleep staging at C = 1):
            # C = 1, not the protocol the title states; probing again tunes it
            datasets = ",".join(dict.fromkeys(s.dataset for s in untuned))
            models = ",".join(dict.fromkeys(s.model for s in untuned))
            _msg.warning(f"{_msg.plural(len(untuned), 'row')} probed with C fixed at 1 (no "
                         f"chosen C recorded); {bench.name} chooses C on validation Cohen's "
                         f"kappa", f"{run} --dataset {datasets} -m {models} --skip-embed")
        unprobed = [s for s in summaries if (s.na_reason or "").startswith(res.NO_PROBE)]
        if unprobed:
            # results made before the benchmark's threshold was in the task's
            # grid (limb 0.5 s): probing again from the saved embeddings adds it
            datasets = ",".join(dict.fromkeys(s.dataset for s in unprobed))
            models = ",".join(dict.fromkeys(s.model for s in unprobed))
            _msg.warning(f"{_msg.plural(len(unprobed), 'row')} without "
                         f"{metric_label(m.headline, bench)} at {m.at_words()}: these results "
                         f"were probed without that threshold",
                         f"{run} --dataset {datasets} -m {models} --skip-embed")
        curveless = [s for s in summaries if s.na_reason == res.NO_CURVE]
        if curveless:
            # probing again from the saved embeddings records it
            datasets = ",".join(dict.fromkeys(s.dataset for s in curveless))
            models = ",".join(dict.fromkeys(s.model for s in curveless))
            _msg.warning(f"{_msg.plural(len(curveless), 'row')} without "
                         f"{metric_label(m.headline, bench)}: {res.NO_CURVE}",
                         f"{run} --dataset {datasets} -m {models} --skip-embed")

