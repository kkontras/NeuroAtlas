"""Read what runs wrote and turn it into tables: one row per (dataset, model)
with the headline metric's mean and spread over folds, and a leaderboard.

Every results.json under ``<output_root>/<benchmark>/`` counts -- a local
``run`` writes ``<dataset>/results.json``, a cluster job
``<dataset>/<model>/results.json`` -- and a result recorded twice (same
dataset, model, fold, task) is counted once, the newest copy.

"n/a" is never a number: a model the channel map rules out, a model missing
from some dataset of a suite, a spread over one fold. It is reported, and
never ranked as zero.
"""
from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from neuroatlas import _paths

# Where a task keeps its held-out numbers, most specific first.
_CONTAINERS = ("best_test", "test")
# Aliases: the name the catalog uses -> names a task may have written.
_ALIASES = {"auroc": ("auroc", "roc_auc"), "r2": ("r2", "r_squared")}


def _unwrap(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        value = value.get("mean")
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric(metrics: Dict[str, Any], name: str) -> Optional[float]:
    """A test metric from a result's metrics, wherever the task put it."""
    names = _ALIASES.get(name, (name,))
    for container in _CONTAINERS:
        block = metrics.get(container)
        if isinstance(block, dict):
            for n in names:
                if n in block:
                    return _unwrap(block[n])
    for n in names:
        if n in metrics:
            return _unwrap(metrics[n])
    # Tasks that report several estimators (brain age) nest one block per
    # estimator; take the first block that has the metric at its top or test level.
    for value in metrics.values():
        if isinstance(value, dict) and value is not metrics:
            for container in (*_CONTAINERS, None):
                block = value.get(container) if container else value
                if isinstance(block, dict):
                    for n in names:
                        if n in block and not isinstance(block[n], dict | list):
                            return _unwrap(block[n])
    return None


@dataclass
class Record:
    dataset: str
    model: str
    fold: str
    task: str
    ok: bool
    failure: Optional[str]
    metrics: Dict[str, Any]
    source: Path
    mtime: float


def _records(path: Path) -> List[Record]:
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    if isinstance(rows, dict):
        rows = [rows]
    mtime = path.stat().st_mtime
    out = []
    for r in rows:
        if not isinstance(r, dict) or "checkpoint_id" not in r:
            continue
        meta = r.get("metadata") or {}
        failure = (r.get("failure") or {}).get("code")
        out.append(Record(
            dataset=str(r.get("dataset_name")), model=str(r["checkpoint_id"]),
            fold=str(meta.get("fold", "all")), task=str(meta.get("task_name", r.get("evaluation_mode", ""))),
            ok=bool(r.get("ok", failure is None)), failure=failure,
            metrics=r.get("metrics") or {}, source=path, mtime=mtime))
    return out


def result_files(benchmark: Optional[str] = None, paths: Optional[Sequence[str]] = None,
                 output_root: Optional[Path] = None) -> List[Path]:
    if paths:
        files: List[Path] = []
        for pattern in paths:
            p = Path(pattern)
            if p.is_dir():
                files += sorted(p.rglob("results.json"))
            else:
                files += [Path(f) for f in sorted(glob.glob(pattern, recursive=True))]
        return files
    base = Path(output_root) if output_root else _paths.output_dir()
    base = base / benchmark if benchmark else base
    return sorted(base.rglob("results.json")) if base.is_dir() else []


def collect(files: Iterable[Path]) -> Tuple[List[Record], int]:
    """Every record, deduplicated by (dataset, model, fold, task); newest wins.
    Returns (records, number of duplicates dropped)."""
    best: Dict[Tuple[str, str, str, str], Record] = {}
    total = 0
    for f in files:
        for rec in _records(f):
            total += 1
            key = (rec.dataset, rec.model, rec.fold, rec.task)
            if key not in best or rec.mtime >= best[key].mtime:
                best[key] = rec
    return list(best.values()), total - len(best)


@dataclass
class Summary:
    dataset: str
    model: str
    mean: Optional[float]
    std: Optional[float]
    n_folds: int
    n_failed: int
    status: str                   # ok | partial | failed | n/a
    normalized: Optional[float] = None
    secondary: Dict[str, Optional[float]] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, None
    return mean, math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def summarize(records: List[Record], headline: str, secondary: Sequence[str] = (),
              dummy_for=None, higher_is_better: bool = True) -> List[Summary]:
    groups: Dict[Tuple[str, str], List[Record]] = {}
    for r in records:
        groups.setdefault((r.dataset, r.model), []).append(r)
    out = []
    for (dataset, model), recs in sorted(groups.items()):
        oks = [r for r in recs if r.ok]
        skips = [r for r in recs if r.failure == "channel_map_skip"]
        failed = [r for r in recs if not r.ok and r.failure != "channel_map_skip"]
        if skips and not oks:
            out.append(Summary(dataset, model, None, None, 0, 0, "n/a"))
            continue
        mean, std = _mean_std([metric(r.metrics, headline) for r in oks])
        s = Summary(dataset, model, mean, std, len(oks), len(failed),
                    "ok" if oks and not failed else "partial" if oks else "failed",
                    secondary={m: _mean_std([metric(r.metrics, m) for r in oks])[0] for m in secondary},
                    failures=sorted({r.failure or "?" for r in failed}))
        dummy = dummy_for(dataset) if dummy_for else None
        if mean is not None and dummy is not None and higher_is_better and dummy < 1:
            s.normalized = (mean - dummy) / (1 - dummy)
        out.append(s)
    return out


def benchmark_summary(benchmark: str, paths: Optional[Sequence[str]] = None,
                      output_root: Optional[Path] = None) -> Tuple[List[Summary], int]:
    from neuroatlas import catalog

    bench = catalog.load(benchmark)
    files = result_files(benchmark, paths, output_root)
    records, dropped = collect(files)
    if paths:
        known = {e.slug for e in bench.datasets}
        records = [r for r in records if r.dataset in known]
    m = bench.metrics
    return summarize(records, m.headline, m.secondary, m.dummy_for, m.higher_is_better), dropped


# --------------------------------------------------------------------------
# leaderboard
# --------------------------------------------------------------------------

def _ranks(values: Dict[str, float], higher_is_better: bool) -> Dict[str, float]:
    """Average ranks (1 = best), ties sharing the mean of their positions."""
    order = sorted(values.items(), key=lambda kv: -kv[1] if higher_is_better else kv[1])
    ranks: Dict[str, float] = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k][0]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


@dataclass
class BoardRow:
    model: str
    mean_rank: Optional[float]
    normalized: Optional[float]
    n_tasks: int
    missing: List[str] = field(default_factory=list)


def leaderboard(benchmarks: Optional[Sequence[str]] = None, suite: str = "full",
                paths: Optional[Sequence[str]] = None,
                output_root: Optional[Path] = None) -> Tuple[List[BoardRow], Dict[str, Dict[str, float]]]:
    """Rank within each dataset, average within a benchmark, then across.

    A model is ranked only on benchmarks where it has a result for every
    dataset of the suite; elsewhere it is listed as missing, not ranked low.
    Returns (global rows, {benchmark: {model: mean rank}}).
    """
    from neuroatlas import catalog

    per_bench: Dict[str, Dict[str, float]] = {}
    per_bench_norm: Dict[str, Dict[str, float]] = {}
    seen_models: set = set()
    names = benchmarks or [b.name for b in catalog.catalog().values() if b.datasets]
    for name in names:
        bench = catalog.load(name)
        wanted = [e.slug for e in bench.suite(suite)]
        summaries, _ = benchmark_summary(name, paths, output_root)
        by_ds: Dict[str, Dict[str, Summary]] = {}
        for s in summaries:
            if s.dataset in wanted and s.mean is not None:
                by_ds.setdefault(s.dataset, {})[s.model] = s
                seen_models.add(s.model)
        if not by_ds:
            continue
        complete = set.intersection(*(set(v) for v in by_ds.values())) if len(by_ds) == len(wanted) else set()
        if not complete:
            continue
        rank_sum: Dict[str, List[float]] = {m: [] for m in complete}
        norm: Dict[str, List[float]] = {m: [] for m in complete}
        for ds, models in by_ds.items():
            ranks = _ranks({m: models[m].mean for m in complete}, bench.metrics.higher_is_better)
            for m in complete:
                rank_sum[m].append(ranks[m])
                if models[m].normalized is not None:
                    norm[m].append(models[m].normalized)
        per_bench[name] = {m: sum(v) / len(v) for m, v in rank_sum.items()}
        per_bench_norm[name] = {m: sum(v) / len(v) for m, v in norm.items() if v}

    rows = []
    for model in sorted(seen_models):
        ranks = [r[model] for r in per_bench.values() if model in r]
        norms = [n[model] for n in per_bench_norm.values() if model in n]
        missing = [b for b in per_bench if model not in per_bench[b]]
        rows.append(BoardRow(model, sum(ranks) / len(ranks) if ranks and not missing else None,
                             sum(norms) / len(norms) if norms and not missing else None,
                             len(ranks), missing))
    rows.sort(key=lambda r: (r.mean_rank is None, r.mean_rank if r.mean_rank is not None else 0))
    return rows, per_bench
