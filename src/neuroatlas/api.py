"""NeuroAtlas from Python: the command's verbs as functions returning pandas
tables. Importing this module applies ``~/.neuroatlas/config.yaml``, exactly
as the command does.

    from neuroatlas import api

    api.plan("sleep_stage", "all_fm", datasets="full")          # = run --dry-run
    api.check("sleep_stage", "biot_pretrained")                  # = check
    api.run_benchmark("sleep_stage", "biot_pretrained", debug=True)
    api.results("sleep_stage")                                   # = results
    api.rescore("sleep_stage")                                   # = rescore

Every table has exactly the columns of the command's ``--format json``
(``dataset``, ``model``, ``not_applicable``, ``note``, ...),
and a missing number is ``None`` (NaN once pandas makes a float column of it),
never "n/a".

The offline rule holds here as on the command line: no function downloads.
``check`` and ``run_benchmark`` run with downloads switched off
(``NEUROATLAS_OFFLINE=1`` for the duration of the call, unless you set it
yourself), so a model whose weights are not here is refused at once, by name,
instead of fetched. ``run_benchmark(..., online=True)`` is
``neuroatlas --online run``: it fetches a missing checkpoint. Fetch data and
weights with ``neuroatlas data download`` / ``neuroatlas models download``.
"""
from __future__ import annotations

import contextlib as _contextlib
import os as _os
from pathlib import Path as _Path
from typing import Any as _Any, Dict as _Dict, List as _List, Optional as _Optional, \
    Sequence as _Sequence

from neuroatlas import config as _config

__all__ = ["benchmarks", "models", "data_status", "plan", "check", "run_benchmark",
           "results", "rescore"]

_config.apply_to_environ()

#: The variables that switch neuroatlas's own downloads off: the weights
#: pre-flight of `run` (config.is_offline) and the checkpoint downloader
#: (backbones/_checkpoint_download.downloads_off) read them on every call.
#: HF_HUB_OFFLINE is not toggled here: huggingface_hub reads it once, at its
#: import, so a per-call value would stick to the whole Python session; the
#: pre-flight refuses a model whose hub weights are not cached before any
#: hub call is made.
_OFFLINE_VARS = ("NEUROATLAS_OFFLINE", "EEGBENCH_OFFLINE")


@_contextlib.contextmanager
def _downloads(online: bool = False):
    """Downloads off for one call (the offline rule), unless *online*; the
    environment is left as it was found. A variable you set yourself wins
    either way, as on the command line."""
    if online:
        yield
        return
    added = [var for var in _OFFLINE_VARS if var not in _os.environ]
    for var in added:
        _os.environ[var] = "1"
    try:
        yield
    finally:
        for var in added:
            _os.environ.pop(var, None)


def _frame(rows: _List[_Dict[str, _Any]], columns: _Optional[_Sequence[str]] = None):
    import pandas as pd

    return pd.DataFrame(rows, columns=list(columns) if columns and not rows else None)


def benchmarks():
    """Every benchmark in the catalog."""
    from neuroatlas import catalog

    return _frame([{"benchmark": b.name, "title": b.title, "domain": b.domain,
                    "task": b.task, "headline": b.metrics.headline,
                    "higher_is_better": b.metrics.higher_is_better, "chance": b.chance(),
                    "single": b.single, "datasets": [e.slug for e in b.datasets],
                    "planned_datasets": [p["name"] for p in b.planned],
                    "excluded_models": [{"families": list(x.families), "reason": x.reason}
                                        for x in b.excluded_models]}
                   for b in catalog.catalog().values()])


#: `models status -v --format json`'s fields, in its order.
MODEL_COLUMNS = ("checkpoint", "source", "state", "path", "note")


def models(selector: str = "all"):
    """`models status -v`: where each checkpoint's weights stand, with the
    notes the command prints under a row (how to get them, what is missing)."""
    from neuroatlas import models as ms
    from neuroatlas.cli.models import _specs

    rows = []
    for spec in _specs(selector):
        st = ms.status(spec)
        rows.append({"checkpoint": st.identifier, "source": st.source,
                     "state": ms.state_word(st.state),
                     "path": st.path or None, "note": "; ".join(st.notes) or None})
    return _frame(rows, MODEL_COLUMNS)


def data_status(*targets: str):
    """`data status --format json`: is each dataset of these benchmarks /
    datasets / domains here (default: every dataset)."""
    from neuroatlas import data
    from neuroatlas.cli.data import _row, expand_targets

    return _frame([_row(data.status(slug), verbose=False, machine=True)
                   for slug in expand_targets(list(targets))])


def _plan_rows(plans, restricted: bool = False) -> _List[_Dict[str, _Any]]:
    from neuroatlas.cli.run import _incomplete, plan_notes, plan_row

    refused = _incomplete(plans)
    return [{**plan_row(p, restricted), "note": "; ".join(plan_notes(p, p in refused)) or None}
            for p in plans]


def plan(benchmark: str, models: str, datasets: str = "single", variant: str = "default",
         debug: bool = False, output_root: _Optional[str] = None):
    """`run --dry-run --format json`: one row per dataset."""
    from neuroatlas import run

    return _frame(_plan_rows(run.plan(benchmark, models, datasets, variant, debug=debug,
                                      output_root=_Path(output_root) if output_root else None),
                             restricted=debug))


def check(benchmark: str, models: str, datasets: str = "single", variant: str = "default"):
    """`check`: one row per (dataset, model), as `check --format json`.
    Downloads nothing (nor does the command)."""
    from neuroatlas.check import check as _check
    from neuroatlas.cli.check import _row

    with _downloads(online=False):
        pairs = _check(benchmark, models, datasets, variant)
    return _frame([{**_row(p), "note": "; ".join(p.notes) or None} for p in pairs])


def run_benchmark(benchmark: str, models: str, datasets: str = "single",
                  variant: str = "default", *, debug: bool = False,
                  limit_batches: _Optional[int] = None, cache_root: _Optional[str] = None,
                  output_root: _Optional[str] = None, online: bool = False,
                  reprobe: bool = False):
    """Run it here and return the results of *this* run's datasets, models and
    variant (`results`, filtered), read from ``output_root`` (default: the
    configured one).

    ``online=False`` (default): downloads off, as `neuroatlas run`; a model
    whose weights are not here is reported skipped, by name, and the others
    run (a dataset whose data is not here is skipped too). ``online=True``: as `neuroatlas --online run`, a missing checkpoint
    is fetched. ``reprobe=True``: as `run --reprobe`, every fold is fitted
    again instead of rescored from its saved predictions."""
    from neuroatlas import run

    root = _Path(output_root) if output_root else None
    plans = run.plan(benchmark, models, datasets, variant, debug=debug, output_root=root)
    with _downloads(online):
        run.execute(plans, cache_root=_Path(cache_root) if cache_root else None,
                    limit_batches=limit_batches, reprobe=reprobe)
    if limit_batches is not None:
        # execute put this run's results under <root>/_limited (a truncated
        # extraction never lands beside real results)
        root_text = str(plans[0].output_base) if plans and plans[0].output_base else output_root
    else:
        root_text = output_root
    table = results(benchmark, output_root=root_text, variant=variant)
    pairs = {(p.dataset, m) for p in plans for m in [*p.models, *p.skipped]}
    if table.empty:
        return table
    keep = [(d, m) in pairs for d, m in zip(table["dataset"], table["model"])]
    return table[keep].reset_index(drop=True)


def results(benchmark: str, paths: _Optional[_Sequence[str]] = None,
            output_root: _Optional[str] = None, variant: _Optional[str] = None):
    """One row per (dataset, variant, model), as `results --format json`:
    headline mean/std over the folds that succeeded, folds out of the
    protocol's (``LOSO k/N`` for leave-one-subject-out), the headline's
    chance level, failures, and the secondary metrics; ``note`` is the lines
    the table prints under the row (why it is n/a, why folds failed).
    ``variant``: that variant's rows only (default: every variant; an
    earlier name of a variant is that variant)."""
    from neuroatlas import catalog, results as res
    from neuroatlas.cli.results import summary_rows

    bench = catalog.load(benchmark)
    summaries, _ = res.benchmark_summary(benchmark, paths,
                                         _Path(output_root) if output_root else None,
                                         variant=variant)
    rows, notes = summary_rows(bench, summaries, machine=True)
    for i, row in enumerate(rows):
        row["note"] = "; ".join(notes.get(i, [])) or None
    return _frame(rows, ["benchmark", "metric", "dataset", "variant", "model", "status",
                         "mean", "std"])


#: `rescore --format json`'s fields, in its order.
RESCORE_COLUMNS = ("benchmark", "dataset", "variant", "model", "fold", "task", "result",
                   "metric", "before", "after", "predictions", "results_file", "fix", "note")


def rescore(benchmark: str, datasets: _Optional[str] = None, models: _Optional[str] = None,
            output_root: _Optional[str] = None, variant: _Optional[str] = None):
    """`rescore --format json`: recompute every result row's metrics from the
    test predictions its probe saved, rewrite results.json (metadata kept,
    ``metadata.rescored_at`` added), and return one row per result: ``result``
    is same / changed / skipped / failed, ``before`` and ``after`` the
    headline metric, ``fix`` for a skipped row the command that probes it
    again. Nothing is fitted."""
    from neuroatlas import rescore as rs

    root = _Path(output_root).expanduser().resolve() if output_root else None
    report = rs.rescore(benchmark, datasets=datasets, models=models, output_root=root,
                        variant=variant)
    rows = [{"benchmark": report.benchmark, "dataset": o.dataset, "variant": o.variant,
             "model": o.model, "fold": o.fold, "task": o.task, "result": o.status,
             "metric": report.headline, "before": o.before, "after": o.after,
             "predictions": o.predictions, "results_file": str(o.file),
             "fix": rs.fix_command(report.benchmark, o, root) if o.status == "skipped" else None,
             "note": o.reason}
            for o in report.outcomes]
    return _frame(rows, RESCORE_COLUMNS)

