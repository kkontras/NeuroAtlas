"""``neuroatlas results <benchmark>`` and ``neuroatlas leaderboard``."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from neuroatlas.cli._table import add_format_arg, render


def _reference(benchmark: str) -> Optional[Dict[Tuple[str, str, str], float]]:
    """The paper's numbers, configs/reference/<benchmark>.csv: dataset,model,metric,value."""
    from neuroatlas import _paths

    path = _paths.configs_dir("reference", f"{benchmark}.csv")
    if not path.is_file():
        return None
    with open(path, newline="") as f:
        return {(r["dataset"], r["model"], r["metric"]): float(r["value"]) for r in csv.DictReader(f)}


def build_results_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neuroatlas results",
        description="One row per (dataset, model): the headline metric's mean and spread over "
                    "the folds that succeeded, the normalised score (0 = dummy, 1 = perfect) "
                    "and the secondary metrics.")
    p.add_argument("benchmark")
    p.add_argument("paths", nargs="*",
                   help="results.json files, globs or folders. Default: every results.json "
                        "under <output root>/<benchmark>/.")
    p.add_argument("--reference", action="store_true",
                   help="Put the paper's numbers beside yours, with the difference.")
    p.add_argument("--all-metrics", action="store_true",
                   help="With --reference: compare every metric, not just the headline.")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true", help="Show why failed runs failed.")
    add_format_arg(p)
    return p


def results_main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas import catalog, results as res

    args = build_results_parser().parse_args(argv)
    bench = catalog.load(args.benchmark)
    summaries, dropped = res.benchmark_summary(bench.name, args.paths or None, args.output_root)
    m = bench.metrics
    if not summaries:
        where = ", ".join(args.paths) if args.paths else str(
            (args.output_root or res._paths.output_dir()) / bench.name)
        raise SystemExit(f"error: no results for {bench.name} in {where}. "
                         f"Run `neuroatlas run {bench.name} ...` first.")

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
                theirs = ref.get((s.dataset, s.model, name))
                if theirs is None:
                    continue
                delta = None if ours is None else ours - theirs
                rows.append({"dataset": s.dataset, "model": s.model, "metric": name,
                             "reference": theirs, "ours": ours, "delta": delta,
                             "n_folds": s.n_folds,
                             "within": "n/a" if delta is None or tol is None
                             else ("yes" if abs(delta) <= tol else "no")})
        render(rows, ["dataset", "model", "metric", "reference", "ours", "delta", "n_folds", "within"],
               args.format)
        if args.format == "table":
            print(f"\ntolerance: {tol if tol is not None else 'none recorded'} ({m.headline})")
        return

    direction = "" if m.higher_is_better else ", lower is better"
    rows, notes = [], {}
    for s in summaries:
        row = {"dataset": s.dataset, "model": s.model,
               "mean": "n/a" if s.status == "n/a" else s.mean,
               "std": s.std if s.std is not None else "n/a",
               "n_folds": s.n_folds, "normalized": s.normalized if s.normalized is not None else "n/a"}
        for name in m.secondary:
            value = s.secondary.get(name)
            row[name] = value if value is not None else "n/a"
        rows.append(row)
        if s.n_failed:
            notes[len(rows) - 1] = [f"{s.n_failed} fold(s) failed: {', '.join(s.failures)}"]
    columns = ["dataset", "model", "mean", "std", "n_folds", "normalized", *m.secondary]
    if args.format == "table":
        print(f"{bench.name}: {m.headline}{direction}, mean ± std over ok folds; "
              f"normalized: 0 = dummy, 1 = perfect")
    render(rows, columns, args.format, notes if args.format == "table" and args.verbose else None)
    if args.format == "table":
        failed = sum(1 for s in summaries if s.n_failed)
        if dropped:
            print(f"{dropped} result(s) recorded more than once; counted the newest copy of each")
        if failed and not args.verbose:
            print(f"{failed} row(s) have failed folds; -v says why")


def build_board_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neuroatlas leaderboard",
        description="Rank models within each dataset, average the ranks within a benchmark, "
                    "then across benchmarks. A model missing from some dataset of a suite is "
                    "listed, not ranked.")
    p.add_argument("paths", nargs="*", help="results.json files, globs or folders. Default: "
                                            "everything under the output root.")
    p.add_argument("--suite", choices=["single", "full"], default="full")
    p.add_argument("--benchmarks", default=None, help="Comma list; default every benchmark.")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="store_true", help="Add the per-benchmark ranks.")
    add_format_arg(p)
    return p


def leaderboard_main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas import results as res

    args = build_board_parser().parse_args(argv)
    names = [b.strip() for b in args.benchmarks.split(",")] if args.benchmarks else None
    rows, per_bench = res.leaderboard(names, args.suite, args.paths or None, args.output_root)
    if not rows:
        raise SystemExit("error: no results to rank. Run a benchmark first.")
    table = []
    for r in rows:
        row = {"model": r.model, "mean_rank": r.mean_rank if r.mean_rank is not None else "n/a",
               "normalized": r.normalized if r.normalized is not None else "n/a",
               "n_benchmarks": r.n_tasks}
        if args.verbose:
            for b in per_bench:
                row[b] = per_bench[b].get(r.model, "n/a")
        table.append(row)
    notes = {i: [f"not ranked: missing from {', '.join(r.missing)}"]
             for i, r in enumerate(rows) if r.missing}
    if args.format == "table":
        print(f"global ranking ({args.suite} suite over {', '.join(per_bench) or 'no complete benchmark'}); "
              f"mean_rank 1 = best everywhere")
    render(table, ["model", "mean_rank", "normalized", "n_benchmarks", *(per_bench if args.verbose else [])],
           args.format, notes if args.format == "table" else None)
