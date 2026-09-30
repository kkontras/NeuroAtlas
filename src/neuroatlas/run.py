"""Run a benchmark: plan it, then execute its steps -- ``embed`` then
``probe`` per dataset -- in this process.

Every step is the catalog's verb invocation (the paper's command line) plus
the few arguments a run adds: the model selection, where results go, and,
for a quick try, fold 0 only. The verbs do the work exactly as they do when
called by hand.

Results land in ``<output_root>/<benchmark>/<dataset>/`` (a variant other
than the default adds ``/<variant>``): the runner's results.json,
results.csv and summary.md, which ``neuroatlas results`` reads.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuroatlas import _paths


@dataclass
class DatasetPlan:
    benchmark: str
    dataset: str
    task: str
    models: List[str]
    skipped: List[str]            # not applicable (channel map)
    folds: List[Any]
    seeds: str
    data: str
    output: Path
    embed_argv: List[str]
    probe_argv: List[str]
    notes: List[str] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.models) * max(1, len(self.folds))


def result_dir(benchmark: str, dataset: str, variant: str = "default",
               output_root: Optional[Path] = None) -> Path:
    base = Path(output_root) if output_root else _paths.output_dir()
    out = base / benchmark / dataset
    return out if variant == "default" else out / variant


def _skipped(dataset: str, specs) -> List[str]:
    from neuroatlas.benchmarking_helpers.channels.channel_map import load_channel_map

    cmap = load_channel_map(dataset)
    if cmap is None:
        return []
    return [s.identifier for s in specs if cmap.is_skip(s.model_family)]


def _folds(dataset: str, probe_argv: List[str]) -> List[Any]:
    from neuroatlas.entrypoints import probe

    try:
        cfg = probe.build_config(probe.build_parser(probe_argv).parse_args(probe_argv))
        block = cfg["datasets"][dataset]
        return list(block.get("folds") or [block.get("fold", "-")])
    except SystemExit:
        return ["?"]
    except Exception as exc:                     # e.g. LOSO needs moabb to count subjects
        return [f"? ({type(exc).__name__})"]


def plan(benchmark: str, models: str, suite: str = "single", variant: str = "default", *,
         debug: bool = False, output_root: Optional[Path] = None,
         folds: Optional[str] = None, per_model: bool = False) -> List[DatasetPlan]:
    """``per_model`` puts each model's results in <dataset>/<model>/: cluster
    jobs, one per (dataset, model), never write the same results.json."""
    from neuroatlas import catalog, data, selectors
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    bench = catalog.load(benchmark)
    ids = selectors.resolve_models(models)
    by_id = {s.identifier: s for s in checkpoint_registry()}
    specs = [by_id[i] for i in ids]
    steps = bench.steps(suite, variant)
    plans: List[DatasetPlan] = []

    if bench.derived_from:
        source = catalog.load(bench.derived_from)
        for step in steps:
            src = result_dir(source.name, step.dataset, "default", output_root)
            state = "staging results found" if (src / "results.json").is_file() else \
                f"needs `neuroatlas run {source.name} --dataset {step.dataset}` first"
            plans.append(DatasetPlan(bench.name, step.dataset, "hypnogram", ids, [], ["all"],
                                     "-", state, result_dir(bench.name, step.dataset, variant, output_root),
                                     [], []))
        return plans

    embeds = {s.dataset: list(s.argv) for s in steps if s.verb == "embed"}
    for step in (s for s in steps if s.verb == "probe"):
        slug = step.dataset
        skipped = _skipped(slug, specs)
        runnable = [i for i in ids if i not in skipped]
        out = result_dir(bench.name, slug, variant, output_root)
        if per_model and len(runnable) == 1:
            out = out / runnable[0]
        probe_argv = list(step.argv)
        if folds:
            probe_argv += ["--folds", folds]
        elif debug:
            probe_argv += ["--folds", "0"]
        fold_list = _folds(slug, probe_argv)
        st = data.status(slug)
        seeds = "per fold"
        plans.append(DatasetPlan(bench.name, slug, bench.task_for(slug) or "default", runnable,
                                 skipped, fold_list, seeds, st.state, out,
                                 embeds[slug], probe_argv))
        if not st.found and st.state != "fetched on first use":
            plans[-1].notes.append(f"data {st.state}: `neuroatlas data status {slug}`")
    return plans


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def _call(verb: str, argv: List[str]) -> int:
    import importlib

    module = importlib.import_module(f"neuroatlas.entrypoints.{verb}")
    print(f"\n$ neuroatlas {verb} {' '.join(argv)}", flush=True)
    try:
        module.main(argv)
    except SystemExit as exc:
        if exc.code in (None, 0):
            return 0
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 1
        return int(exc.code)
    return 0


def read_results(path: Path) -> List[Dict[str, Any]]:
    try:
        return json.loads((path / "results.json").read_text())
    except (OSError, ValueError):
        return []


def execute(plans: List[DatasetPlan], *, cache_root: Optional[Path] = None,
            limit_batches: Optional[int] = None, skip_embed: bool = False,
            extra_embed: Optional[List[str]] = None,
            extra_probe: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the plans. Returns counts of ok / failed / n/a results."""
    import contextlib

    from neuroatlas import config

    summary = {"ok": 0, "failed": 0, "n/a": 0, "datasets_failed": []}
    if limit_batches is not None:
        # A truncated cache must never be read as a complete one.
        base = Path(cache_root) if cache_root else (config.get("cache_root") or _paths.artifacts_dir("embedding_cache"))
        cache_root = Path(base) / "_limited"
    cache_args = ["--cache-root", str(cache_root)] if cache_root else []

    for p in plans:
        if p.task == "hypnogram":
            status = _run_hypnogram(p)
            summary["ok" if status == 0 else "failed"] += 1
            continue
        if not p.models:
            print(f"\n{p.dataset}: no applicable model (channel map skips {', '.join(p.skipped)})")
            summary["n/a"] += len(p.skipped)
            continue
        models = ["--models", ",".join(p.models)]
        ctx = contextlib.nullcontext()
        if limit_batches is not None:
            from neuroatlas.extensions.tasks.linear_probe import limit_batches as _limit

            ctx = _limit(limit_batches)
        with ctx:
            if not skip_embed:
                rc = _call("embed", [*p.embed_argv, *models, *cache_args, *(extra_embed or [])])
                if rc:
                    summary["datasets_failed"].append(p.dataset)
                    summary["failed"] += p.n_runs
                    continue
            p.output.mkdir(parents=True, exist_ok=True)
            rc = _call("probe", [*p.probe_argv, *models, *cache_args,
                                 "--output-root", str(p.output), *(extra_probe or [])])
        rows = read_results(p.output)
        wanted = set(p.models)
        mine = [r for r in rows if r.get("checkpoint_id") in wanted]
        for r in mine:
            code = (r.get("failure") or {}).get("code")
            if r.get("ok"):
                summary["ok"] += 1
            elif code == "channel_map_skip":
                summary["n/a"] += 1
            else:
                summary["failed"] += 1
        if rc and not mine:
            summary["datasets_failed"].append(p.dataset)
            summary["failed"] += p.n_runs
        summary["n/a"] += len(p.skipped)
    return summary


def _run_hypnogram(p: DatasetPlan) -> int:
    """Hypnograms from the staging probes, then their features, into the
    benchmark's own folder, with a results.json the results command reads."""
    import numpy as np

    from neuroatlas import catalog
    from neuroatlas.entrypoints import hypnogram as hyp

    source = catalog.load(catalog.load(p.benchmark).derived_from)
    src = result_dir(source.name, p.dataset, "default", p.output.parents[1])
    if not (src / "results.json").is_file():
        print(f"\n{p.dataset}: no staging results in {src}; run `neuroatlas run {source.name} "
              f"--dataset {p.dataset}` first", file=sys.stderr)
        return 1
    p.output.mkdir(parents=True, exist_ok=True)
    hypno = p.output / "hypnograms.json"
    print(f"\n$ hypnogram reconstruct --results-dir {src} --output {hypno}", flush=True)
    hyp.reconstruct_results_dir(results_dir=src, output=hypno, models=p.models or None,
                                compute_metrics=True, merge=False)
    rows = hyp.rows_from_hypnograms(hypno, p.dataset)
    if not rows:
        print(f"{p.dataset}: no hypnograms reconstructed", file=sys.stderr)
        return 1
    hyp.write_csv(rows, p.output / "hypnogram_features.csv")
    summary = hyp.compute_summary(rows)
    hyp.write_summary_csv(summary, p.output / "hypnogram_features_summary.csv")
    results = []
    for model in sorted({r["model"] for r in summary}):
        feats = [r for r in summary if r["model"] == model]
        rs = [r["pearson_r"] for r in feats if r["pearson_r"] == r["pearson_r"]]
        results.append({
            "checkpoint_id": model, "dataset_name": p.dataset, "evaluation_mode": "hypnogram",
            "ok": bool(rs),
            "metrics": {"pearson_r": float(np.mean(rs)) if rs else float("nan"),
                        "per_feature": {r["feature"]: {k: r[k] for k in ("mae", "bias", "rmse", "pearson_r", "n_valid")}
                                        for r in feats}},
            "metadata": {"fold": "all", "task_name": "hypnogram"},
        })
    (p.output / "results.json").write_text(json.dumps(results, indent=2, default=float))
    return 0
