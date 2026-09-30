"""``neuroatlas run <benchmark> -m MODELS [--dataset single|full|SLUGS]`` --
extract embeddings where missing, fit the probes, write the results."""
from __future__ import annotations

import argparse
from typing import List, Optional

from neuroatlas.cli._table import add_format_arg, render


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neuroatlas run",
        description="Run a benchmark on this machine: for each dataset, `embed` (a no-op "
                    "where the cache already has it), then `probe` over its folds. "
                    "Results go to <output root>/<benchmark>/<dataset>/.")
    p.add_argument("benchmark")
    p.add_argument("-m", "--models", required=True,
                   help="An alias (all_fm, all_ts, ...), group, family or checkpoint ids.")
    p.add_argument("--dataset", default="single", metavar="single|full|SLUGS",
                   help="The one-dataset quick suite (default), every dataset, or a comma list.")
    p.add_argument("--variant", default="default", help="A benchmark variant (see `neuroatlas show`).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan -- datasets, models, folds, runs, data state -- and stop.")
    p.add_argument("--debug", action="store_true",
                   help="Fold 0 only. Embeddings are extracted in full and cached as usual, "
                        "so the real run reuses them.")
    p.add_argument("--folds", default=None, help="Comma list of folds, instead of all.")
    p.add_argument("--limit-batches", type=int, default=None, metavar="N",
                   help="Stop each extraction after N batches: a smoke test. Uses its own "
                        "cache, <cache root>/_limited, and too few subjects may leave a "
                        "fold with nothing to test on.")
    p.add_argument("--skip-embed", action="store_true",
                   help="Probe only; fail where embeddings are missing.")
    p.add_argument("--cache-root", default=None, help="Embedding cache for this run.")
    p.add_argument("--output-root", default=None, help="Results root for this run.")
    p.add_argument("--checkpoint-override", action="append", default=[], metavar="ID.KEY=VALUE",
                   help="Change a checkpoint field for this run, e.g. "
                        "biot_pretrained.checkpoint_path=/my/weights.ckpt. Needs --cache-root: "
                        "the cache is keyed by checkpoint id, not by weights.")
    p.add_argument("--per-model-output", action="store_true",
                   help="Results in <dataset>/<model>/ (what `submit`'s jobs use).")
    add_format_arg(p)
    return p


def _overrides(pairs: List[str]) -> List[str]:
    """ID.KEY=VALUE -> the verbs' --checkpoint ID=... form is per-path only, so
    use the runner's checkpoint_overrides through --checkpoint for paths and
    reject the rest clearly."""
    out = []
    for text in pairs:
        target, sep, value = text.partition("=")
        ident, dot, key = target.partition(".")
        if not (sep and dot):
            raise SystemExit(f"error: --checkpoint-override wants ID.KEY=VALUE, got {text!r}")
        if key != "checkpoint_path":
            raise SystemExit(f"error: only checkpoint_path can be overridden per run for now "
                             f"(got {key}); edit the registry for anything else")
        out += ["--checkpoint", f"{ident}={value}"]
    return out


def main(argv: Optional[List[str]] = None) -> None:
    from pathlib import Path

    from neuroatlas import run as runmod

    args = build_parser().parse_args(argv)
    if args.checkpoint_override and not args.cache_root:
        raise SystemExit("error: --checkpoint-override needs --cache-root: embeddings are cached "
                         "by checkpoint id, so the old cache would be read as if it came from "
                         "the new weights")
    overrides = _overrides(args.checkpoint_override)
    plans = runmod.plan(args.benchmark, args.models, args.dataset, args.variant,
                        debug=args.debug, folds=args.folds, per_model=args.per_model_output,
                        output_root=Path(args.output_root) if args.output_root else None)

    rows = [{
        "benchmark": p.benchmark, "dataset": p.dataset, "task": p.task,
        "n_models": len(p.models), "folds": ", ".join(map(str, p.folds)),
        "seeds": p.seeds, "n_runs": p.n_runs, "data": p.data,
        "n/a": len(p.skipped) or "-",
    } for p in plans]
    notes = {i: p.notes for i, p in enumerate(plans) if p.notes}
    if args.dry_run or args.format != "table":
        render(rows, ["benchmark", "dataset", "task", "n_models", "folds", "seeds", "n_runs",
                      "data", "n/a"], args.format, notes if args.format == "table" else None)
    if args.dry_run:
        if args.format == "table":
            print()
            for p in plans:
                if p.embed_argv:
                    print(f"  neuroatlas embed {' '.join(p.embed_argv)} --models {','.join(p.models)}")
                    print(f"  neuroatlas probe {' '.join(p.probe_argv)} --models {','.join(p.models)} "
                          f"--output-root {p.output}")
                else:
                    print(f"  hypnograms from {p.output.parents[1] / 'sleep_stage' / p.dataset} -> {p.output}")
        return

    summary = runmod.execute(
        plans, cache_root=Path(args.cache_root) if args.cache_root else None,
        limit_batches=args.limit_batches, skip_embed=args.skip_embed,
        extra_embed=overrides, extra_probe=overrides)
    print(f"\nruns: ok {summary['ok']}   failed {summary['failed']}   n/a {summary['n/a']}")
    if summary["datasets_failed"]:
        print(f"datasets that failed before any result: {', '.join(summary['datasets_failed'])}")
    print(f"results: {plans[0].output.parent.parent if plans else '-'}  "
          f"(`neuroatlas results {args.benchmark}`)")
    if summary["failed"]:
        raise SystemExit(1)
