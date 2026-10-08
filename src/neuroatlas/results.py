"""Read what runs wrote and turn it into a table: one row per (dataset,
variant, model) with the headline metric's mean and spread over folds.

Every results.json under ``<output_root>/<benchmark>/`` counts -- a local
``run`` writes ``<dataset>/results.json``, a cluster job
``<dataset>/<model>/results.json``, and a variant other than the default one
folder deeper, ``<dataset>/<variant>/...`` (``run.result_dir``) -- and a result
recorded twice (same dataset, variant, model, fold, task) is counted once, the
newest copy. Variants are never merged: a per_patch result and a default one
of the same model and fold are two results, not a duplicate.

"n/a" is never a number: a model the channel map rules out, a spread over
one fold, a metric the task does not define for the dataset (AUROC of a
multi-class diagnosis), a probe that was not fitted (one class only at an
event threshold). It is reported as n/a with the reason, never as zero.

A metric over folds is the mean of the per-fold values ± their population SD,
except the epilepsy headline ``event_sens_fa_auc``: the AUC of the folds'
median sensitivity-vs-FA/h curve ± the sample SD of the per-fold AUCs, as the
paper computed it (:func:`_event_sens_fa`). ``Summary.mean``/``std`` hold
that value and spread.

A task that fits several probes per fold (an event field and a threshold)
files each one's numbers under its own keys; the benchmark's ``metrics.at``
names the one a row reads (``("ahi_fraction", "threshold_10.0s")``), and a
row never falls back to another.
"""
from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from neuroatlas import _paths, progress

# Where a task keeps its held-out numbers, most specific first.
_CONTAINERS = ("best_test", "test")
# Aliases: the name the catalog uses -> names a task may have written.
_ALIASES = {"auroc": ("auroc", "roc_auc"), "r2": ("r2", "r_squared")}
# Validation blocks: never read as a held-out number.
_VALIDATION = ("best_val", "val")


def _unwrap(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        value = value.get("mean")
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _threshold_key(block: Dict[str, Any], key: str) -> Optional[str]:
    """The key of *block* that files the same threshold as *key*
    (``threshold_10s`` and ``threshold_10.0s`` are one threshold)."""
    if key in block:
        return key
    if not (key.startswith("threshold_") and key.endswith("s")):
        return None
    try:
        wanted = float(key[len("threshold_"):-1])
    except ValueError:
        return None
    for k in block:
        if isinstance(k, str) and k.startswith("threshold_") and k.endswith("s"):
            try:
                if float(k[len("threshold_"):-1]) == wanted:
                    return k
            except ValueError:
                continue
    return None


def block_at(metrics: Dict[str, Any], at: Sequence[str] = ()) -> Optional[Dict[str, Any]]:
    """The part of a fold's metrics that *at* names (all of it for ``()``);
    None when the result has no such probe."""
    block: Any = metrics
    for key in at:
        if not isinstance(block, dict):
            return None
        found = _threshold_key(block, key)
        if found is None:
            return None
        block = block[found]
    return block if isinstance(block, dict) else None


def metric(metrics: Dict[str, Any], name: str, at: Sequence[str] = ()) -> Optional[float]:
    """A test metric from a result's metrics, wherever the task put it; with
    *at*, from that probe of the fold only (None if it was not fitted)."""
    if at:
        block = block_at(metrics, at)
        if block is None or block.get("skipped"):
            return None
        return metric(block, name)
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
    for key, value in metrics.items():
        if isinstance(value, dict) and value is not metrics and key not in _VALIDATION:
            for container in (*_CONTAINERS, None):
                block = value.get(container) if container else value
                if isinstance(block, dict):
                    for n in names:
                        if n in block and not isinstance(block[n], dict | list):
                            return _unwrap(block[n])
    return None


#: The start of the n/a reason of a result that has no probe at the
#: benchmark's ``metrics.at`` (made before that threshold was probed).
NO_PROBE = "this result has no probe at"


def _seconds(key: str) -> str:
    return key[len("threshold_"):-1] if key.startswith("threshold_") else key


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _and(words: List[str]) -> str:
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def _at_words(at: Sequence[str], i: int) -> str:
    """The probe ``at[:i + 1]`` names, in words: ``10 s of scored apnea or
    hypopnea``; an event field without a threshold, by its words."""
    from neuroatlas.catalog import EVENT_WORDS

    field_ = next((k for k in at if not k.startswith("threshold_")), None)
    event = EVENT_WORDS.get(field_ or "", field_)
    key = at[i]
    if not key.startswith("threshold_"):
        return event or key
    seconds = _seconds(key)
    seconds = f"{float(seconds):g}" if _is_number(seconds) else seconds
    return f"{seconds} s" + (f" of {event}" if event else "")


def not_applicable(metrics: Dict[str, Any], name: str, at: Sequence[str] = ()) -> Optional[str]:
    """Why a fold's metrics have no *name*, when they say: the probe *at*
    names is not in the result or was not fitted, the task recorded a reason
    (``<metric>_not_applicable``), or the metric is binary and the task is not."""
    block: Any = metrics
    for i, key in enumerate(at):
        found = _threshold_key(block, key) if isinstance(block, dict) else None
        if found is None:
            have = sorted((float(_seconds(k)) for k in (block if isinstance(block, dict) else {})
                           if isinstance(k, str) and k.startswith("threshold_")
                           and _is_number(_seconds(k))))
            return (f"{NO_PROBE} {_at_words(at, i)}"
                    + (f" (it has {_and([f'{h:g} s' for h in have])})" if have else ""))
        block = block[found]
        if isinstance(block, dict) and block.get("skipped"):
            why = str(block.get("reason") or "not fitted").rstrip(".")
            return f"no probe at {_at_words(at, i)} in this fold: {why}"
    if not isinstance(block, dict):
        return None
    for key in (f"{name}_not_applicable", name.rsplit("_", 1)[0] + "_not_applicable"):
        if block.get(key):
            return str(block[key])
    if name in ("auroc", "auprc"):
        n = _n_classes_scored(block)
        if n is not None and n > 2:
            return (f"{name.upper()} is defined here for two classes; this task has {n} "
                    f"(no one-vs-rest protocol)")
    return None


def _n_classes_scored(block: Dict[str, Any]) -> Optional[int]:
    """How many classes a fold's test block scored (the rows of its confusion
    matrix, else its ``labels``: the classes seen in the true or predicted)."""
    for container in _CONTAINERS:
        test = block.get(container)
        if not isinstance(test, dict):
            continue
        for key in ("confusion_matrix", "labels"):
            if isinstance(test.get(key), list) and test[key]:
                return len(test[key])
    return None


def prevalence(metrics: Dict[str, Any], at: Sequence[str] = ()) -> Optional[float]:
    """The share of positive (class 1) test items of a fold's binary probe,
    from the test confusion matrix it recorded: what AUPRC scores at chance."""
    block = block_at(metrics, at)
    if block is None or block.get("skipped"):
        return None
    for container in _CONTAINERS:
        test = block.get(container)
        if not isinstance(test, dict):
            continue
        cm, labels = test.get("confusion_matrix"), test.get("labels")
        if not isinstance(cm, list) or not cm:
            continue
        labels = labels if isinstance(labels, list) and len(labels) == len(cm) else list(range(len(cm)))
        rows = {int(lab): sum(row) for lab, row in zip(labels, cm)}
        total = sum(rows.values())
        return rows.get(1, 0) / total if total else None
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
    n_classes: Optional[int] = None     # the dataset's class count, when the run recorded it


DEFAULT_VARIANT = "default"

Variants = Union[Sequence[str], Mapping[str, str]]


def variant_of(path: Path, dataset: str, variants: Variants = ()) -> str:
    """The variant a results.json belongs to, from its folder: ``run`` writes a
    variant's results to ``<benchmark>/<dataset>/<variant>/`` (a cluster job to
    ``<dataset>/<variant>/<model>/``), the default variant's straight under
    ``<dataset>/``. *variants* are the names the benchmark defines, or
    ``{folder: variant}`` (``Benchmark.variant_folders``: an earlier name of a
    variant reads as the variant); a folder that is not one of them (a
    model's) is not a variant."""
    folders = variants if isinstance(variants, Mapping) else {v: v for v in variants}
    parts = Path(path).parent.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == dataset:
            if i + 1 < len(parts) and parts[i + 1] in folders:
                return folders[parts[i + 1]]
            break
    return DEFAULT_VARIANT


def _protocol_folds(meta: Dict[str, Any], dataset: Optional[str] = None) -> Optional[int]:
    """The number of folds a record's dataset protocol has, from the dataset
    config the runner copies into its metadata: ``num_folds`` / ``n_folds``
    (k-fold, LOSO), or 1 for a cohort with one fixed split (``fold`` only).

    A BCI result written before the count was recorded says ``"loso"``
    (one fold per subject); a MOABB cohort's subject count is known without
    reading anything, so it still reads ``1/9``, not ``1/1``."""
    for key in ("num_folds", "n_folds"):
        value = meta.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        if isinstance(value, str) and value.strip().lower() == "loso" and dataset:
            return _loso_count(dataset)
    if "fold" in meta and "folds" not in meta:
        return 1
    return None


def _loso_count(dataset: str) -> Optional[int]:
    """The subject count of a MOABB cohort (its LOSO fold count), from the
    registered cohort config; None for any other dataset."""
    try:
        from neuroatlas.extensions.datasets.dataio.moabb_loader import MOABB_DATASETS
    except Exception:           # the BCI extra is not installed: count unknown
        return None
    cfg = MOABB_DATASETS.get(dataset)
    return int(cfg.n_subjects) if cfg is not None else None


def _n_classes(meta: Dict[str, Any]) -> Optional[int]:
    """The dataset's class count from the metadata a run recorded: a MOABB
    cohort's ``n_classes``, a pickled cohort's ``canonical_label_space``."""
    value = meta.get("n_classes")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    space = meta.get("canonical_label_space")
    return len(space) if isinstance(space, list) and space else None


def _records(path: Path, variants: Variants = ()) -> List[Record]:
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
            n_folds=_protocol_folds(meta, dataset),
            variant=variant_of(path, dataset, variants),
            n_classes=_n_classes(meta)))
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
    progress.current().phase("looking for result files")
    return sorted(base.rglob("results.json")) if base.is_dir() else []


def collect(files: Iterable[Path], variants: Variants = ()) -> Tuple[List[Record], int]:
    """Every record, deduplicated by (dataset, variant, model, fold, task);
    newest wins. *variants*: the benchmark's variant names (see
    :func:`variant_of`). Returns (records, number of duplicates dropped)."""
    best: Dict[Tuple[str, str, str, str, str], Record] = {}
    total = 0
    files = list(files)
    # the command's live line (stderr, a terminal only)
    item = progress.current().phase("reading", total=len(files), unit="result files")
    for f in files:
        item.update(advance=1)
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
    # The headline's chance level for this row: the benchmark's fixed one
    # (kappa 0, AUROC 0.5), the mean over the folds of the test prevalence
    # (AUPRC), 1/C (balanced accuracy); None where no chance level is stated.
    chance: Optional[float] = None
    secondary: Dict[str, Optional[float]] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)       # failure codes
    # how many folds the dataset's protocol has (None: unknown), so a 1-fold
    # --debug result reads "1/5", not like a finished one
    n_expected: Optional[int] = None
    # one {"fold", "code", "message"} per failed fold, as results.json has it
    errors: List[Dict[str, Any]] = field(default_factory=list)
    variant: str = DEFAULT_VARIANT
    # why the headline has no value although folds succeeded (not_applicable):
    # the task recorded it (``<metric stem>_not_applicable``, e.g. a
    # recording-level cohort has no seizure events for event_sens_fa_auc),
    # the probe at the benchmark's threshold was not fitted, ...
    na_reason: Optional[str] = None
    # a value pooled over features (the hypnogram's mean r): (how many went
    # into it, how many there are); a feature whose r is undefined is left out
    features: Optional[Tuple[int, int]] = None
    # the C each succeeded fold's probe chose from its grid (on validation
    # Cohen's kappa), where it recorded one; [] for a probe without a grid
    c_chosen: List[float] = field(default_factory=list)
    # each succeeded fold: {"fold", "value" (the headline), "C" (the C its
    # probe chose from a grid, or None)}, for `results -v`
    per_fold: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Every fold of the protocol succeeded (or the protocol is unknown)."""
        return self.n_failed == 0 and (self.n_expected is None or self.n_folds >= self.n_expected)


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Mean and population standard deviation (numpy's default, ddof=0) over
    the folds: the convention of every ± the paper prints (its probes and
    table scripts call np.std), so a reproduced row matches the paper's."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, None
    return mean, math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))


def _event_sens_fa(records: List[Record]) -> Tuple[Optional[float], Optional[float]]:
    """The epilepsy headline over folds, as the paper computed it (Fig. 2,
    Tables 3-5): not the mean of the per-fold AUCs but the AUC of the
    point-wise median (np.nanmedian) of the folds' sensitivity-vs-FA/h curves,
    with ± the sample SD (ddof=1) of the per-fold AUCs. The folds are the
    succeeded ones that recorded a curve (``event_sens_fa_curve``); a result
    written before the task recorded curves has none and is not counted."""
    from neuroatlas.extensions.tasks import _event_sens_fa as esf

    curves, grids = [], []
    for r in records:
        curve, grid = r.metrics.get("event_sens_fa_curve"), r.metrics.get("event_sens_fa_grid")
        if isinstance(curve, list) and isinstance(grid, list) and len(curve) == len(grid):
            curves.append(esf.from_json(curve))
            grids.append(esf.from_json(grid))
    if not curves or any(g.shape != grids[0].shape or not (g == grids[0]).all() for g in grids):
        return None, None       # nothing to aggregate, or curves on different grids
    auc, sd = esf.aggregate(curves, grids[0])
    return (None if math.isnan(auc) else auc), (None if math.isnan(sd) else sd)


#: Metrics whose value over folds is not the mean ± population SD of the
#: per-fold values (_mean_std) but the paper's own aggregate of them.
_FOLD_AGGREGATES = {"event_sens_fa_auc": _event_sens_fa}

#: Why such a metric is n/a when no fold says it does not apply (the n/a
#: line under a `results` row).
NO_CURVE = "no fold recorded the curve this metric is computed from"


def _over_folds(records: List[Record], name: str,
                at: Sequence[str] = ()) -> Tuple[Optional[float], Optional[float]]:
    """A metric's value and spread over the folds that succeeded."""
    if name in _FOLD_AGGREGATES:
        return _FOLD_AGGREGATES[name](records)
    return _mean_std([metric(r.metrics, name, at) for r in records])


def _chance(records: List[Record], chance, at: Sequence[str] = ()) -> Optional[float]:
    """The headline's chance level over these folds (Summary.chance)."""
    from neuroatlas import metrics_info

    if chance is None:
        return None
    if chance == metrics_info.PREVALENCE:
        return _mean_std([prevalence(r.metrics, at) for r in records])[0]
    if chance == metrics_info.ONE_OVER_C:
        # the class count the run recorded; else the classes the test fold scored
        counts = [r.n_classes or _n_classes_scored(block_at(r.metrics, at) or {})
                  for r in records]
        return _mean_std([1.0 / n if n else None for n in counts])[0]
    return float(chance)


def _features(records: List[Record]) -> Optional[Tuple[int, int]]:
    """(features with a defined value, features) of a value pooled over
    features: the hypnogram's ``per_feature`` block."""
    for r in records:
        per = r.metrics.get("per_feature")
        if isinstance(per, dict) and per:
            used = sum(1 for v in per.values() if isinstance(v, dict)
                       and _unwrap(v.get("pearson_r")) is not None
                       and not math.isnan(_unwrap(v.get("pearson_r"))))
            return used, len(per)
    return None


def chosen_c(metrics: Dict[str, Any], at: Sequence[str] = ()) -> Optional[float]:
    """The C a fold's probe kept: ``C`` (train_probe's grid) or ``best_c``
    (seizure detection's grid); None if neither."""
    for name in ("C", "best_c"):
        value = metric(metrics, name, at)
        if value is not None:
            return float(value)
    return None


def _fold_key(fold: str):
    return (0, int(fold), "") if fold.isdigit() else (1, 0, fold)


def _group_order(key: Tuple[str, str, str]):
    """(dataset, variant, model), the default variant first within a dataset."""
    dataset, variant, model = key
    return dataset, variant != DEFAULT_VARIANT, variant, model


def summarize(records: List[Record], headline: str, secondary: Sequence[str] = (), *,
              at: Sequence[str] = (), chance=None) -> List[Summary]:
    """One summary per (dataset, variant, model). *at*: the probe of each
    fold the metrics are read from (``Metrics.at``); *chance*: the headline's
    chance level, a number, ``"prevalence"`` or ``"1/C"`` (``Benchmark.chance``)."""
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
        mean, std = _over_folds(oks, headline, at)
        s = Summary(dataset, model, mean, std, len(oks), len(failed),
                    "ok" if oks and not failed else "partial" if oks else "failed",
                    chance=_chance(oks, chance, at),
                    secondary={m: _over_folds(oks, m, at)[0] for m in secondary},
                    failures=sorted({r.failure or "failed" for r in failed}),
                    n_expected=protocol.get((dataset, variant)),
                    errors=[{"fold": r.fold, "code": r.failure or "failed", "message": r.message}
                            for r in sorted(failed, key=lambda r: _fold_key(r.fold))],
                    variant=variant, features=_features(oks),
                    c_chosen=[c for c in (metric(r.metrics, "C", at) for r in
                                          sorted(oks, key=lambda r: _fold_key(r.fold)))
                              if c is not None],
                    per_fold=[{"fold": r.fold, "value": metric(r.metrics, headline, at),
                               "C": chosen_c(r.metrics, at)}
                              for r in sorted(oks, key=lambda r: _fold_key(r.fold))])
        if mean is None and oks:
            reasons = sorted({why for why in (not_applicable(r.metrics, headline, at) for r in oks)
                              if why})
            s.na_reason = "; ".join(reasons) or (NO_CURVE if headline in _FOLD_AGGREGATES
                                                  else None)
        out.append(s)
    return out


def benchmark_summary(benchmark: str, paths: Optional[Sequence[str]] = None,
                      output_root: Optional[Path] = None,
                      variant: Optional[str] = None,
                      left_out: Optional[List[str]] = None) -> Tuple[List[Summary], int]:
    """Summaries of every variant's results, or of *variant*'s only (a name
    the benchmark does not define is a CatalogError, a usage error).

    Results of a model whose family the benchmark leaves out
    (``excluded_models``: written by the verbs, or before the benchmark left
    it out) are not the benchmark's and are not summarised; *left_out*, if
    given, receives their checkpoint ids."""
    from neuroatlas import catalog

    bench = catalog.load(benchmark)
    if variant is not None:
        variant = bench.variant(variant).name       # an earlier name: the variant's
    files = result_files(benchmark, paths, output_root)
    records, dropped = collect(files, bench.variant_folders())
    if paths:
        known = {e.slug for e in bench.datasets}
        records = [r for r in records if r.dataset in known]
    if variant is not None:
        records = [r for r in records if r.variant == variant]
    excluded = bench.excluded_families()
    if excluded and records:
        from neuroatlas import selectors

        family = {s.identifier: s.model_family for s in selectors._registry()}
        out = sorted({r.model for r in records if family.get(r.model) in excluded})
        records = [r for r in records if r.model not in out]
        if left_out is not None:
            left_out.extend(out)
    m = bench.metrics
    return summarize(records, m.headline, m.secondary, at=m.at, chance=bench.chance()), dropped

