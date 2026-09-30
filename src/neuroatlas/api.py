"""NeuroAtlas from Python: the command's verbs as functions returning pandas
tables. Importing this module applies ``~/.neuroatlas/config.yaml``, exactly
as the command does.

    from neuroatlas import api

    api.plan("sleep_stage", "all_fm", datasets="full")          # = run --dry-run
    api.check("sleep_stage", "biot_pretrained")                  # = check
    api.run_benchmark("sleep_stage", "biot_pretrained", debug=True)
    api.results("sleep_stage")                                   # = results
    api.leaderboard(suite="single")["global"]                    # = leaderboard
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from neuroatlas import config as _config

_config.apply_to_environ()


def _frame(rows: List[Dict[str, Any]]):
    import pandas as pd

    return pd.DataFrame(rows)


def benchmarks():
    """Every benchmark in the catalog."""
    from neuroatlas import catalog

    return _frame([{"benchmark": b.name, "title": b.title, "domain": b.domain,
                    "task": b.task, "headline": b.metrics.headline,
                    "higher_is_better": b.metrics.higher_is_better,
                    "single": b.single, "datasets": [e.slug for e in b.datasets],
                    "planned": [p["name"] for p in b.planned]}
                   for b in catalog.catalog().values()])


def models(selector: str = "all"):
    """Checkpoints a selector names, with where their weights stand."""
    from neuroatlas import models as ms, selectors
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    by_id = {s.identifier: s for s in checkpoint_registry()}
    rows = []
    for i in selectors.resolve_models(selector):
        st = ms.status(by_id[i])
        rows.append({"checkpoint": i, "family": st.family, "source": st.source,
                     "state": st.state, "path": st.path})
    return _frame(rows)


def data_status(*targets: str):
    """`data status`: is each dataset of these benchmarks / datasets here."""
    from neuroatlas import data
    from neuroatlas.cli.data import expand_targets

    return _frame([{**asdict(data.status(s)), "found": data.status(s).found}
                   for s in expand_targets(list(targets))])


def plan(benchmark: str, models: str, datasets: str = "single", variant: str = "default",
         debug: bool = False):
    from neuroatlas import run

    return _frame([{"benchmark": p.benchmark, "dataset": p.dataset, "task": p.task,
                    "models": p.models, "n_models": len(p.models), "folds": p.folds,
                    "n_runs": p.n_runs, "data": p.data, "not_applicable": p.skipped,
                    "output": str(p.output)}
                   for p in run.plan(benchmark, models, datasets, variant, debug=debug)])


def check(benchmark: str, models: str, datasets: str = "single", variant: str = "default"):
    from neuroatlas.check import check as _check

    return _frame([asdict(p) for p in _check(benchmark, models, datasets, variant)])


def run_benchmark(benchmark: str, models: str, datasets: str = "single",
                  variant: str = "default", *, debug: bool = False,
                  limit_batches: Optional[int] = None, cache_root: Optional[str] = None):
    """Run it here and return its results table (`results`)."""
    from neuroatlas import run

    plans = run.plan(benchmark, models, datasets, variant, debug=debug)
    run.execute(plans, cache_root=Path(cache_root) if cache_root else None,
                limit_batches=limit_batches)
    return results(benchmark)


def results(benchmark: str, paths: Optional[Sequence[str]] = None):
    """One row per (dataset, model): headline mean/std over folds, normalised, secondary."""
    from neuroatlas import results as res

    summaries, _ = res.benchmark_summary(benchmark, paths)
    rows = []
    for s in summaries:
        row = {k: v for k, v in asdict(s).items() if k != "secondary"}
        row.update(s.secondary)
        rows.append(row)
    return _frame(rows)


def leaderboard(benchmarks: Optional[Sequence[str]] = None, suite: str = "full",
                paths: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """{"global": ranking table, "per_benchmark": model x benchmark mean ranks}."""
    import pandas as pd

    from neuroatlas import results as res

    rows, per_bench = res.leaderboard(benchmarks, suite, paths)
    return {"global": _frame([asdict(r) for r in rows]),
            "per_benchmark": pd.DataFrame(per_bench)}
