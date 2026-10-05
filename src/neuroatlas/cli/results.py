"""``neuroatlas results <benchmark>`` and ``neuroatlas leaderboard``.

Both print one schema in every format: a missing number is n/a in a table
and null in JSON, every JSON row names its benchmark (and metric), and the
lines a table prints under a row are its ``note``.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from neuroatlas.cli import Parser, UsageError
from neuroatlas.cli._table import add_format_arg, render


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
    summaries, dropped = res.benchmark_summary(bench.name, args.paths or None, args.output_root,
                                               variant=args.variant)
    m = bench.metrics
    if not summaries:
        where = ", ".join(args.paths) if args.paths else str(
            (args.output_root or res._paths.output_dir()) / bench.name)
        which = f" (variant {args.variant})" if args.variant else ""
        again = f"neuroatlas run {bench.name} ..." + (
            f" --variant {args.variant}" if args.variant and args.variant != "default" else "")
        raise SystemExit(f"error: no results for {bench.name}{which} in {where}. "
                         f"Run `{again}` first.")
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

    direction = "" if m.higher_is_better else ", lower is better"
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
        print(f"{bench.name}" + (f" ({args.variant} variant)" if args.variant else "")
              + f": {m.headline}{direction}, mean ± std over the folds that "
              f"succeeded; folds = succeeded/protocol"
              + ("; normalized: 0 = dummy, 1 = perfect" if has_dummy else ""))
    render(rows, columns, args.format,
           notes if args.verbose or args.format != "table" else None, extra=extra)
    if args.format == "table":
        failed = sum(1 for s in summaries if s.n_failed)
        short = sum(1 for s in summaries if s.status != "n/a" and not s.n_failed and not s.complete)
        if dropped:
            print(f"{dropped} result(s) recorded more than once (same dataset, variant, model, "
                  f"fold and task); counted the newest copy of each")
        if failed and not args.verbose:
            print(f"{failed} row(s) have failed folds; -v says why")
        if short:
            print(f"{short} row(s) ran on fewer folds than the protocol has (e.g. --debug): "
                  f"their mean is not comparable with a full run")


def build_board_parser() -> Parser:
    p = Parser(
        prog="neuroatlas leaderboard",
        description="Rank models within each dataset, average the ranks within a benchmark, "
                    "then across benchmarks. A model missing from some dataset of a suite is "
                    "listed, not ranked; a result on fewer folds than the protocol has is "
                    "ranked and marked. Only the default variant's results are ranked, "
                    "unless --variant names another.")
    p.add_argument("paths", nargs="*", help="results.json files, globs or folders. Default: "
                                            "everything under the output root.")
    p.add_argument("--suite", choices=["single", "full"], default="full",
                   help="Rank on every dataset of each benchmark (full, the paper's "
                        "leaderboard; the default) or on its one quick dataset (single).")
    p.add_argument("--benchmarks", default=None, help="Comma list; default every benchmark.")
    p.add_argument("--variant", default="default",
                   help="Rank this variant's results (default: the default variant, the "
                        "paper's protocol). Another variant ranks the benchmarks that have it.")
    p.add_argument("--output-root", type=Path, default=None,
                   help="The results root to read (default: the configured output root).")
    p.add_argument("-v", "--verbose", action="store_true", help="Add the per-benchmark ranks.")
    add_format_arg(p)
    return p


def leaderboard_main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas import catalog, results as res

    args = build_board_parser().parse_args(argv)
    names = [b.strip() for b in args.benchmarks.split(",") if b.strip()] if args.benchmarks else None
    for name in names or []:
        catalog.load(name)                         # an unknown name: exit 2 with a suggestion
    details: Dict = {}
    rows, per_bench = res.leaderboard(names, args.suite, args.paths or None, args.output_root,
                                      details=details, variant=args.variant)
    if not rows:
        if args.variant != "default":
            raise SystemExit(f"error: no {args.variant} results to rank. Run a benchmark with "
                             f"--variant {args.variant} first.")
        raise SystemExit("error: no results to rank. Run a benchmark first.")
    if not per_bench:
        lines = [f"error: nothing to rank: no benchmark has a result on every dataset of its "
                 f"{args.suite} suite."]
        for bench, gaps in details.get("incomplete", {}).items():
            total = len(catalog.load(bench).suite(args.suite))
            lines.append(f"  {bench}: no result yet on {len(gaps)} of {total} datasets "
                         f"({', '.join(gaps)})")
        if args.suite == "full":
            lines.append("`neuroatlas leaderboard --suite single` ranks on each benchmark's "
                         "quick dataset; `neuroatlas results <benchmark>` shows what there is.")
        raise SystemExit("\n".join(lines))

    table, notes = [], {}
    for i, r in enumerate(rows):
        row = {"model": r.model, "mean_rank": r.mean_rank, "normalized": r.normalized,
               "n_benchmarks": r.n_benchmarks, "suite": args.suite, "variant": args.variant,
               "benchmarks": [b for b in per_bench if r.model in per_bench[b]],
               "missing": r.missing, "partial_folds": r.partial,
               "ranks": {b: per_bench[b].get(r.model) for b in per_bench}}
        for b in per_bench:
            row[b] = per_bench[b].get(r.model)
        table.append(row)
        lines = []
        if r.missing:
            lines.append(f"not ranked: missing from {', '.join(r.missing)}")
        if r.partial:
            lines.append(f"ranked on fewer folds than the protocol: {', '.join(r.partial)}")
        if lines:
            notes[i] = lines
    if args.format == "table":
        print(f"global ranking ({args.suite} suite over {', '.join(per_bench)}"
              + (f"; {args.variant} variant" if args.variant != "default" else "")
              + "); mean_rank 1 = best everywhere")
    render(table, ["model", "mean_rank", "normalized", "n_benchmarks",
                   *(per_bench if args.verbose else [])],
           args.format, notes, extra=["suite", "variant", "benchmarks", "missing",
                                      "partial_folds", "ranks"])
    skipped = {b: g for b, g in details.get("incomplete", {}).items()}
    if args.format == "table" and skipped:
        print(f"not ranked (a {args.suite}-suite dataset has no result yet): "
              + "; ".join(f"{b} ({', '.join(g)})" for b, g in skipped.items()))
