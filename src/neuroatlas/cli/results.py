"""``neuroatlas results <benchmark>``.

One schema in every format: a missing number is n/a in a table and null in
JSON, every JSON row names its benchmark and metric, and the lines a table
prints under a row are its ``note``.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from neuroatlas.cli import Parser, UsageError
from neuroatlas.cli._table import add_format_arg, render


# What a metric is called in a table; JSON and CSV keep the key.
METRIC_LABELS = {
    "auroc": "AUROC(window)",
    "auprc": "AUPRC(window)",
    "balanced_accuracy": "bal_acc",
    "mcc": "MCC",
    "ovlp_f1": "event_F1",
    "sensitivity_at_fpr_h_1_0": "sens@1FA/h",
    "sensitivity_at_fpr_h_0_1": "sens@0.1FA/h",
    "event_sens_fa_auc": "Sens@FA_AUC(event)",
    "cohen_kappa": "kappa",
    "macro_f1": "macro_F1",
    "accuracy": "acc",
    "mae": "MAE(years)",
    "pearson_r": "pearson_r",
    "r2": "R2",
}


# The headline's name in a table's title line.
METRIC_NAMES = {
    "auroc": "window-level AUROC",
    "balanced_accuracy": "balanced accuracy",
    "event_sens_fa_auc": "event-level Sens@FA AUC",
    "mae": "MAE in years",
    "pearson_r": "Pearson r",
    "macro_f1": "macro-F1",
}


def metric_label(name: str) -> str:
    return METRIC_LABELS.get(name, name)


def _reference(benchmark: str) -> Optional[Dict[Tuple[str, str, str, str], float]]:
    """The paper's numbers, configs/reference/<benchmark>.csv:
    dataset,model,metric,value (and variant; a row without one is the
    default variant's), keyed (dataset, variant, model, metric)."""
    from neuroatlas import _paths

    path = _paths.configs_dir("reference", f"{benchmark}.csv")
    if not path.is_file():
        return None
    with open(path, newline="") as f:
        return {(r["dataset"], r.get("variant") or "default", r["model"], r["metric"]):
                float(r["value"]) for r in csv.DictReader(f)}


def build_results_parser() -> Parser:
    p = Parser(
        prog="neuroatlas results",
        description="One row per (dataset, variant, model): the headline metric's mean and "
                    "spread over the folds that succeeded, how many of the protocol's folds "
                    "that is, the normalised score (0 = dummy, 1 = perfect; for benchmarks "
                    "with a dummy) and the secondary metrics. A variant's results (run "
                    "--variant) are their own rows, never merged with the default's.")
    p.add_argument("benchmark")
    p.add_argument("paths", nargs="*",
                   help="results.json files, globs or folders. Default: every results.json "
                        "under <output root>/<benchmark>/.")
    p.add_argument("--variant", default=None,
                   help="Only this variant's results (default: every variant; `neuroatlas "
                        "show <benchmark>` lists them).")
    p.add_argument("--reference", action="store_true",
                   help="Put the paper's numbers beside yours, with the difference.")
    p.add_argument("--all-metrics", action="store_true",
                   help="With --reference: compare every metric, not just the headline.")
    p.add_argument("--output-root", type=Path, default=None,
                   help="The results root to read (default: the configured output root).")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show why failed folds failed: the message each one recorded.")
    add_format_arg(p)
    return p


def _folds_text(s) -> str:
    return f"{s.n_folds}/{s.n_expected}" if s.n_expected else str(s.n_folds)


def _failure_lines(s) -> List[str]:
    """One line per distinct failure: the folds it hit, its code, the first
    line of the message the run stored in results.json."""
    if s.status == "n/a":
        return ["not applicable: the channel map skips this model on this dataset"]
    groups: Dict[Tuple[str, str], List[str]] = {}
    for e in s.errors:
        first = (e.get("message") or "").strip().splitlines()
        groups.setdefault((e["code"], first[0] if first else ""), []).append(str(e["fold"]))
    lines = []
    for (code, message), folds in groups.items():
        label = "fold" if len(folds) == 1 else "folds"
        lines.append(f"{label} {', '.join(folds)} failed: {code}" + (f": {message}" if message else ""))
    if s.n_expected and s.n_folds + s.n_failed < s.n_expected:
        lines.append(f"{s.n_expected - s.n_folds - s.n_failed} of {s.n_expected} folds not run yet")
    return lines


def results_main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas import catalog, results as res

    args = build_results_parser().parse_args(argv)
    if args.all_metrics and not args.reference:
        raise UsageError("--all-metrics compares every metric with the paper's, so it needs "
                         "--reference; without it the table already shows every metric")
    bench = catalog.load(args.benchmark)
    left_out: List[str] = []
    summaries, dropped = res.benchmark_summary(bench.name, args.paths or None, args.output_root,
                                               variant=args.variant, left_out=left_out)
    m = bench.metrics
    left_out_line = (f"not shown: results of {', '.join(left_out)}, which the {bench.name} "
                     f"benchmark leaves out (`neuroatlas show {bench.name}`)") if left_out else None
    if not summaries:
        where = ", ".join(args.paths) if args.paths else str(
            (args.output_root or res._paths.output_dir()) / bench.name)
        which = f" (variant {args.variant})" if args.variant else ""
        again = f"neuroatlas run {bench.name} ..." + (
            f" --variant {args.variant}" if args.variant and args.variant != "default" else "")
        raise SystemExit(f"error: no results for {bench.name}{which} in {where}. "
                         f"Run `{again}` first." + (f"\n{left_out_line}" if left_out_line else ""))
    # the variant column: always in machine formats, in the table once a
    # variant other than the default has results
    show_variant = any(s.variant != res.DEFAULT_VARIANT for s in summaries)

    if args.reference:
        ref = _reference(bench.name)
        if ref is None:
            raise SystemExit(f"error: no reference table ships for {bench.name} "
                             f"(configs/reference/{bench.name}.csv)")
        tol = m.tolerance
        metrics = [m.headline, *m.secondary] if args.all_metrics else [m.headline]
        rows = []
        for s in summaries:
            for name in metrics:
                ours = s.mean if name == m.headline else s.secondary.get(name)
                theirs = ref.get((s.dataset, s.variant, s.model, name))
                if theirs is None:
                    continue
                delta = None if ours is None else ours - theirs
                rows.append({"benchmark": bench.name, "dataset": s.dataset,
                             "variant": s.variant, "model": s.model,
                             "metric": name, "reference": theirs, "ours": ours, "delta": delta,
                             "folds": _folds_text(s), "n_folds": s.n_folds,
                             "n_expected": s.n_expected, "tolerance": tol,
                             "within": None if delta is None or tol is None
                             else ("yes" if abs(delta) <= tol else "no")})
        render(rows, ["dataset", *(["variant"] if show_variant else []), "model", "metric",
                      "reference", "ours", "delta", "folds", "within"],
               args.format, extra=["benchmark", "variant", "n_folds", "n_expected", "tolerance"])
        if args.format == "table":
            print(f"\ntolerance: {tol if tol is not None else 'none recorded'} ({m.headline})")
        return

    direction = "higher is better" if m.higher_is_better else "lower is better"
    has_dummy = m.higher_is_better and any(m.dummy_for(e.slug) is not None for e in bench.datasets)
    rows, notes = [], {}
    for s in summaries:
        row = {"benchmark": bench.name, "metric": m.headline,
               "dataset": s.dataset, "variant": s.variant, "model": s.model, "status": s.status,
               "mean": s.mean, "std": s.std, "folds": _folds_text(s),
               "n_folds": s.n_folds, "n_expected": s.n_expected, "n_failed": s.n_failed,
               "normalized": s.normalized, "failures": s.failures, "errors": s.errors}
        for name in m.secondary:
            row[name] = s.secondary.get(name)
        rows.append(row)
        lines = _failure_lines(s)
        if lines:
            notes[len(rows) - 1] = lines
    columns = ["dataset", *(["variant"] if show_variant else []), "model", "mean", "std", "folds",
               *(["normalized"] if has_dummy else []), *m.secondary]
    extra = ["benchmark", "metric", "variant", "status", "n_folds", "n_expected", "n_failed",
             *([] if has_dummy else ["normalized"]), "failures", "errors"]
    if args.format == "table":
        # The table names each column by its metric (machine formats keep the
        # keys): the headline's mean and spread, then the secondary metrics.
        head = metric_label(m.headline)
        print(f"{bench.name}" + (f" ({args.variant} variant)" if args.variant else "")
              + f": {METRIC_NAMES.get(m.headline, head)}, {direction}; mean ± std over folds"
              + ("; normalized: 0 = chance, 1 = perfect" if has_dummy else ""))
        names = {"mean": head, "std": "±", **{k: metric_label(k) for k in m.secondary}}
        rows = [{names.get(k, k): v for k, v in r.items()} for r in rows]
        columns = [names.get(c, c) for c in columns]
    render(rows, columns, args.format,
           notes if args.verbose or args.format != "table" else None, extra=extra)
    if args.format == "table":
        failed = sum(1 for s in summaries if s.n_failed)
        short = sum(1 for s in summaries if s.status != "n/a" and not s.n_failed and not s.complete)
        if dropped:
            print(f"{dropped} result(s) recorded more than once (same dataset, variant, model, "
                  f"fold and task); counted the newest copy of each")
        if left_out_line:
            print(left_out_line)
        if failed and not args.verbose:
            print(f"{failed} row(s) have failed folds; -v says why")
        if short:
            print(f"{short} row(s) cover fewer folds than the protocol (k/N): "
                  f"not comparable with a full run")

