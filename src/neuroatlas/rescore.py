"""Recompute a benchmark's results from the test predictions its probes saved
(``neuroatlas rescore``, ``api.rescore``) -- no probe is fitted again.

Every results.json under ``<output root>/<benchmark>/`` is read; each result
row whose fold saved its predictions (``cache_paths.predictions``, see
:mod:`neuroatlas.predictions`) gets its metrics recomputed by its task's
``score`` function, merged into the row as a run merges a new result (a
metric the task computes is replaced, one it does not is kept), and the file
is rewritten with the row's metadata kept and ``metadata.rescored_at`` added
(``neuroatlas results`` reads it). A row without saved predictions
-- probed before they were kept -- is left as it is and reported with the
command that probes it again.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from neuroatlas.catalog import CatalogError

#: why a row was not rescored, when its fold saved nothing
NOT_SAVED = "probed before predictions were saved"


@dataclass
class Outcome:
    """What happened to one result row."""

    file: Path
    dataset: str
    variant: str
    model: str
    fold: str
    task: str
    status: str                  # same | changed | skipped | failed
    before: Optional[float] = None
    after: Optional[float] = None
    reason: Optional[str] = None
    predictions: Optional[str] = None


@dataclass
class Report:
    benchmark: str
    headline: str
    outcomes: List[Outcome] = field(default_factory=list)
    files_written: List[Path] = field(default_factory=list)
    files_read: List[Path] = field(default_factory=list)
    # where the headline is in a fold's metrics (the benchmark's metrics.at)
    at: Tuple[str, ...] = ()

    def count(self, status: str) -> int:
        return sum(1 for o in self.outcomes if o.status == status)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    from neuroatlas.benchmarking_helpers.runtime.runner import BenchmarkRunner

    return BenchmarkRunner._deep_merge(base, override)


def _wanted_datasets(bench, datasets) -> Optional[set]:
    if not datasets:
        return None
    names = [d.strip() for d in (datasets.split(",") if isinstance(datasets, str) else datasets)
             if d and d.strip()]
    known = {e.slug for e in bench.datasets}
    unknown = [d for d in names if d not in known]
    if unknown:
        raise CatalogError(f"{bench.name} has no dataset {', '.join(unknown)}; "
                           f"its datasets: {', '.join(sorted(known))}\n"
                           f"fix: neuroatlas show {bench.name} (lists them)")
    return set(names)


def _write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def rescore(benchmark: str, *, datasets=None, models: Optional[str] = None,
            output_root: Optional[Path] = None, variant: Optional[str] = None,
            paths: Optional[Sequence[str]] = None) -> Report:
    """Recompute every result row of *benchmark* (or of these *datasets*,
    *models* -- a selector as `run -m` takes -- and *variant*) from its saved
    predictions, and rewrite the results.json files that changed."""
    from neuroatlas import catalog, predictions as preds, results as res
    from neuroatlas.benchmarking_helpers.registry.contracts import BenchmarkResult
    from neuroatlas.benchmarking_helpers.runtime.runner import BenchmarkRunner, results_lock

    bench = catalog.load(benchmark)
    if bench.derived_from:
        raise CatalogError(
            f"{bench.name} is computed from {bench.derived_from}'s results, not probed: "
            f"rescore {bench.derived_from}, then run {bench.name} again\n"
            f"fix: neuroatlas rescore {bench.derived_from}\n"
            f"fix: neuroatlas run {bench.name} -m {models or 'all'}")
    if variant is not None:
        variant = bench.variant(variant).name       # an earlier name: the variant's
    wanted_ds = _wanted_datasets(bench, datasets)
    wanted_models = set(bench.select_models(models)) if models else None
    variants = bench.variant_folders()
    report = Report(bench.name, bench.metrics.headline, at=bench.metrics.at)
    now = datetime.now(timezone.utc).isoformat()

    files = res.result_files(bench.name, paths, output_root)
    # the command's live line (stderr, a terminal only): file by file
    from neuroatlas import progress

    item = progress.current().phase("rescoring", total=len(files), unit="result files")
    for path in files:
        item.update(advance=1)
        report.files_read.append(path)
        with results_lock(path.parent):
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(rows, list):
                continue
            changed_file = False
            for row in rows:
                if not isinstance(row, dict) or "checkpoint_id" not in row:
                    continue
                meta = row.setdefault("metadata", {})
                dataset = str(row.get("dataset_name"))
                model = str(row["checkpoint_id"])
                var = bench.record_variant(res.variant_of(path, dataset, list(variants)), meta)
                if ((wanted_ds is not None and dataset not in wanted_ds)
                        or (wanted_models is not None and model not in wanted_models)
                        or (variant is not None and var != variant)):
                    continue
                out = Outcome(path, dataset, var, model, str(meta.get("fold", "all")),
                              str(meta.get("task_name", row.get("evaluation_mode", ""))), "")
                report.outcomes.append(out)
                if not row.get("ok", True):
                    out.status = "failed"
                    out.reason = (f"its probe did not finish (`neuroatlas results {bench.name} "
                                  f"-v` says why)")
                    continue
                old = row.get("metrics") or {}
                out.before = res.metric(old, report.headline, report.at)
                saved = (row.get("cache_paths") or {}).get("predictions")
                if not saved:
                    out.status, out.reason = "skipped", NOT_SAVED
                    continue
                out.predictions = str(saved)
                if not Path(saved).is_file():
                    out.status, out.reason = "skipped", f"its predictions file is gone ({saved})"
                    continue
                try:
                    pred = preds.load(saved)
                    new = preds.scorer(pred.task)(pred)
                except (preds.PredictionsError, KeyError, ValueError) as exc:
                    out.status, out.reason = "skipped", f"cannot score {saved}: {exc}"
                    continue
                merged = _deep_merge(old, new)
                same = json.dumps(merged, sort_keys=True, default=str) == \
                    json.dumps(old, sort_keys=True, default=str)
                row["metrics"] = merged
                preds.refresh_metadata(meta, pred)
                meta["rescored_at"] = now
                out.after = res.metric(merged, report.headline, report.at)
                out.status = "same" if same else "changed"
                if pred.legacy:
                    out.reason = ("these predictions hold the test columns only: the C and "
                                  "threshold chosen on validation are read from results.json")
                changed_file = True
            if changed_file:
                _write_json(path, rows)
                report.files_written.append(path)
                # a table an older runner wrote beside results.json: the
                # runner brings it in line with the rewritten file
                if (path.parent / "results.csv").exists() or (path.parent / "summary.md").exists():
                    BenchmarkRunner({"benchmark": {"output_root": str(path.parent)}}).write_tables(
                        [BenchmarkResult.from_dict(r) for r in rows
                         if isinstance(r, dict) and "checkpoint_id" in r])
    return report


def fix_command(benchmark: str, outcome: Outcome, output_root: Optional[Path] = None) -> str:
    """The command that probes a skipped row's fold again (and saves its
    predictions)."""
    cmd = f"neuroatlas run {benchmark} --dataset {outcome.dataset} -m {outcome.model} --reprobe"
    if outcome.variant != "default":
        cmd += f" --variant {outcome.variant}"
    if Path(outcome.file).parent.name == outcome.model:
        # a cluster job's folder (<dataset>/<model>/): rewrite that one
        cmd += " --per-model-output"
    if output_root is not None:
        cmd += f" --output-root {output_root}"
    return cmd
