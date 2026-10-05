"""Read what runs wrote and turn it into tables: one row per (dataset,
variant, model) with the headline metric's mean and spread over folds, and a
leaderboard.

Every results.json under ``<output_root>/<benchmark>/`` counts -- a local
``run`` writes ``<dataset>/results.json``, a cluster job
``<dataset>/<model>/results.json``, and a variant other than the default one
folder deeper, ``<dataset>/<variant>/...`` (``run.result_dir``) -- and a result
recorded twice (same dataset, variant, model, fold, task) is counted once, the
newest copy. Variants are never merged: a per_patch result and a default one
of the same model and fold are two results, not a duplicate.

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
    message: Optional[str] = None       # the failure's own message, as the run printed it
    n_folds: Optional[int] = None       # how many folds the dataset's protocol has
    variant: str = "default"            # the benchmark variant, from where the file lies


DEFAULT_VARIANT = "default"


def variant_of(path: Path, dataset: str, variants: Sequence[str] = ()) -> str:
    """The variant a results.json belongs to, from its folder: ``run`` writes a
    variant's results to ``<benchmark>/<dataset>/<variant>/`` (a cluster job to
    ``<dataset>/<variant>/<model>/``), the default variant's straight under
    ``<dataset>/``. *variants* are the names the benchmark defines; a folder
    that is not one of them (a model's) is not a variant."""
    parts = Path(path).parent.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == dataset:
            if i + 1 < len(parts) and parts[i + 1] in variants:
                return parts[i + 1]
            break
    return DEFAULT_VARIANT


def _protocol_folds(meta: Dict[str, Any]) -> Optional[int]:
    """The number of folds a record's dataset protocol has, from the dataset
    config the runner copies into its metadata: ``num_folds`` / ``n_folds``
    (k-fold, LOSO), or 1 for a cohort with one fixed split (``fold`` only)."""
    for key in ("num_folds", "n_folds"):
        value = meta.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    if "fold" in meta and "folds" not in meta:
        return 1
    return None


def _records(path: Path, variants: Sequence[str] = ()) -> List[Record]:
    """The records of one results.json. *variants*: the benchmark's variant
    names, so that a record under ``<dataset>/<variant>/`` is that variant's."""
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
        message = (r.get("failure") or {}).get("message")
        dataset = str(r.get("dataset_name"))
        out.append(Record(
            dataset=dataset, model=str(r["checkpoint_id"]),
            fold=str(meta.get("fold", "all")), task=str(meta.get("task_name", r.get("evaluation_mode", ""))),
            ok=bool(r.get("ok", failure is None)), failure=failure,
            metrics=r.get("metrics") or {}, source=path, mtime=mtime,
            message=str(message).strip() if message else None,
            n_folds=_protocol_folds(meta),
            variant=variant_of(path, dataset, variants)))
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


def collect(files: Iterable[Path], variants: Sequence[str] = ()) -> Tuple[List[Record], int]:
    """Every record, deduplicated by (dataset, variant, model, fold, task);
    newest wins. *variants*: the benchmark's variant names (see
    :func:`variant_of`). Returns (records, number of duplicates dropped)."""
    best: Dict[Tuple[str, str, str, str, str], Record] = {}
    total = 0
    for f in files:
        for rec in _records(f, variants):
            total += 1
            key = (rec.dataset, rec.variant, rec.model, rec.fold, rec.task)
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
    failures: List[str] = field(default_factory=list)       # failure codes
    # how many folds the dataset's protocol has (None: unknown), so a 1-fold
    # --debug result reads "1/5", not like a finished one
    n_expected: Optional[int] = None
    # one {"fold", "code", "message"} per failed fold, as results.json has it
    errors: List[Dict[str, Any]] = field(default_factory=list)
    variant: str = DEFAULT_VARIANT

    @property
    def complete(self) -> bool:
        """Every fold of the protocol succeeded (or the protocol is unknown)."""
        return self.n_failed == 0 and (self.n_expected is None or self.n_folds >= self.n_expected)


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, None
    return mean, math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def _fold_key(fold: str):
    return (0, int(fold), "") if fold.isdigit() else (1, 0, fold)


def _group_order(key: Tuple[str, str, str]):
    """(dataset, variant, model), the default variant first within a dataset."""
    dataset, variant, model = key
    return dataset, variant != DEFAULT_VARIANT, variant, model


def summarize(records: List[Record], headline: str, secondary: Sequence[str] = (),
              dummy_for=None, higher_is_better: bool = True) -> List[Summary]:
    """One summary per (dataset, variant, model)."""
    groups: Dict[Tuple[str, str, str], List[Record]] = {}
    protocol: Dict[Tuple[str, str], int] = {}
    for r in records:
        groups.setdefault((r.dataset, r.variant, r.model), []).append(r)
        if r.n_folds:
            where = (r.dataset, r.variant)
            protocol[where] = max(protocol.get(where, 0), r.n_folds)
    out = []
    for (dataset, variant, model), recs in sorted(groups.items(), key=lambda kv: _group_order(kv[0])):
        oks = [r for r in recs if r.ok]
        skips = [r for r in recs if r.failure == "channel_map_skip"]
        failed = [r for r in recs if not r.ok and r.failure != "channel_map_skip"]
        if skips and not oks:
            out.append(Summary(dataset, model, None, None, 0, 0, "n/a", variant=variant))
            continue
        mean, std = _mean_std([metric(r.metrics, headline) for r in oks])
        s = Summary(dataset, model, mean, std, len(oks), len(failed),
                    "ok" if oks and not failed else "partial" if oks else "failed",
                    secondary={m: _mean_std([metric(r.metrics, m) for r in oks])[0] for m in secondary},
                    failures=sorted({r.failure or "?" for r in failed}),
                    n_expected=protocol.get((dataset, variant)),
                    errors=[{"fold": r.fold, "code": r.failure or "?", "message": r.message}
                            for r in sorted(failed, key=lambda r: _fold_key(r.fold))],
                    variant=variant)
        dummy = dummy_for(dataset) if dummy_for else None
        if mean is not None and dummy is not None and higher_is_better and dummy < 1:
            s.normalized = (mean - dummy) / (1 - dummy)
        out.append(s)
    return out


def benchmark_summary(benchmark: str, paths: Optional[Sequence[str]] = None,
                      output_root: Optional[Path] = None,
                      variant: Optional[str] = None) -> Tuple[List[Summary], int]:
    """Summaries of every variant's results, or of *variant*'s only (a name
    the benchmark does not define is a CatalogError, a usage error)."""
    from neuroatlas import catalog

    bench = catalog.load(benchmark)
    if variant is not None:
        bench.variant(variant)
    files = result_files(benchmark, paths, output_root)
    records, dropped = collect(files, bench.variant_names())
    if paths:
        known = {e.slug for e in bench.datasets}
        records = [r for r in records if r.dataset in known]
    if variant is not None:
        records = [r for r in records if r.variant == variant]
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
    n_benchmarks: int
    missing: List[str] = field(default_factory=list)
    # "benchmark/dataset (k/n folds)" wherever the model was ranked on fewer
    # folds than the dataset's protocol has
    partial: List[str] = field(default_factory=list)


def leaderboard(benchmarks: Optional[Sequence[str]] = None, suite: str = "full",
                paths: Optional[Sequence[str]] = None,
                output_root: Optional[Path] = None,
                details: Optional[Dict[str, Any]] = None,
                variant: str = DEFAULT_VARIANT,
                ) -> Tuple[List[BoardRow], Dict[str, Dict[str, float]]]:
    """Rank within each dataset, average within a benchmark, then across.

    Only *variant*'s results are ranked: the default variant (the paper's
    leaderboard) unless told otherwise. Another variant ranks the benchmarks
    that define it; naming a benchmark that does not is a CatalogError.

    A model is ranked only on benchmarks where it has a result for every
    dataset of the suite; elsewhere it is listed as missing, not ranked low.
    A result on fewer folds than the protocol has is ranked, and marked
    (``BoardRow.partial``).
    Returns (global rows, {benchmark: {model: mean rank}}). ``details``, if
    given, receives ``incomplete``: {benchmark: [suite datasets without any
    result]} for every benchmark that has results but could not be ranked.
    """
    from neuroatlas import catalog

    per_bench: Dict[str, Dict[str, float]] = {}
    per_bench_norm: Dict[str, Dict[str, float]] = {}
    partial: Dict[str, List[str]] = {}
    incomplete: Dict[str, List[str]] = {}
    seen_models: set = set()
    if benchmarks:
        names = list(benchmarks)
        for name in names:
            catalog.load(name).variant(variant)        # a variant it lacks: CatalogError
    else:
        names = [b.name for b in catalog.catalog().values()
                 if b.datasets and variant in b.variant_names()]
        if not names:
            raise catalog.CatalogError(f"no benchmark has a variant {variant!r}")
    for name in names:
        bench = catalog.load(name)
        wanted = [e.slug for e in bench.suite(suite)]
        summaries, _ = benchmark_summary(name, paths, output_root, variant=variant)
        by_ds: Dict[str, Dict[str, Summary]] = {}
        for s in summaries:
            if s.dataset in wanted and s.mean is not None:
                by_ds.setdefault(s.dataset, {})[s.model] = s
                seen_models.add(s.model)
        if not by_ds:
            continue
        complete = set.intersection(*(set(v) for v in by_ds.values())) if len(by_ds) == len(wanted) else set()
        if not complete:
            incomplete[name] = [d for d in wanted if d not in by_ds]
            continue
        rank_sum: Dict[str, List[float]] = {m: [] for m in complete}
        norm: Dict[str, List[float]] = {m: [] for m in complete}
        for ds, models in by_ds.items():
            ranks = _ranks({m: models[m].mean for m in complete}, bench.metrics.higher_is_better)
            for m in complete:
                rank_sum[m].append(ranks[m])
                if models[m].normalized is not None:
                    norm[m].append(models[m].normalized)
                s = models[m]
                if s.n_expected and s.n_folds < s.n_expected:
                    partial.setdefault(m, []).append(
                        f"{name}/{ds} ({s.n_folds}/{s.n_expected} folds)")
        per_bench[name] = {m: sum(v) / len(v) for m, v in rank_sum.items()}
        per_bench_norm[name] = {m: sum(v) / len(v) for m, v in norm.items() if v}

    rows = []
    for model in sorted(seen_models):
        ranks = [r[model] for r in per_bench.values() if model in r]
        norms = [n[model] for n in per_bench_norm.values() if model in n]
        missing = [b for b in per_bench if model not in per_bench[b]]
        rows.append(BoardRow(model, sum(ranks) / len(ranks) if ranks and not missing else None,
                             sum(norms) / len(norms) if norms and not missing else None,
                             len(ranks), missing, partial.get(model, [])))
    rows.sort(key=lambda r: (r.mean_rank is None, r.mean_rank if r.mean_rank is not None else 0))
    if details is not None:
        details["incomplete"] = incomplete
    return rows, per_bench
