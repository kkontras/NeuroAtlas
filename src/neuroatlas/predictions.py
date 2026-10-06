"""Saved test predictions: what every probe writes for each fold, and how its
metrics are recomputed from them.

Each fold of every probe task writes its TEST predictions to::

    <output root>/probes/<dataset>/<checkpoint>/<key>/predictions.npz

(``<key>`` hashes the probe settings, the task settings and the dataset
context, fold included; ``results.json`` records the file under
``cache_paths.predictions``). Every task has one function that computes its
metrics from this file (``TaskSpec.score``), and the probe computes the
metrics it records with that same function, on the file it has just written.
So a fixed or a new metric needs no new probe: ``neuroatlas rescore`` (or
:func:`score`) recomputes every metric from the file, and on a fresh file it
gives exactly the metrics the probe recorded. ``neuroatlas run`` reuses a
fold's file instead of probing again when it was made with the same settings
from the same embeddings (``--reprobe`` probes again).

The format (``format`` = ``neuroatlas.predictions/1``)
------------------------------------------------------
A numpy ``.npz`` file; ``np.load(path)`` reads it (no pickles).

0-d text entries -- what produced the file:
    ``format``, ``dataset``, ``checkpoint_id``, ``task`` (the task slug:
    linear_probe, brain_age, seizure_detection, ...), ``fold`` (text; empty
    for a cohort with one fixed split), and ``info`` (JSON, below).
0-d number entries:
    settings the task's metrics read, e.g. the seizure probe's ``window_s``
    (seconds per window) and ``threshold`` (the decision threshold tuned on
    validation).
Per-row columns, one row per test item the metrics count (a 30 s epoch, a
10 s window, a BCI trial, a subject):
    ``y_true``     the true class (an integer) or value (an age in years)
    ``y_pred``     the predicted class, or the predicted value
    ``y_proba``    (n, k) class probabilities, float32, columns in the order of
    ``classes``    (k,) -- the class label of each ``y_proba`` column
    ``y_score``    (n,) float64 score of the positive class, binary tasks only:
                   what AUROC, AUPRC and the event-level Sens@FA read
    ids            ``subject_id``, ``recording_id``, ``session_id`` (text) and
                   ``epoch_index`` (sleep), ``trial_idx`` (BCI) or
                   ``window_start_s`` (epilepsy), whichever the rows carry
Several probes per fold (groups):
    A task that fits more than one probe on a fold stores each probe's
    columns as ``<group>/<column>`` and lists the groups in
    ``info["groups"]``: arousal one per threshold (``threshold_1.0s/y_true``),
    respiratory and limb events one per event field and threshold
    (``apnea_any_fraction/threshold_1.0s/y_pred``), brain age one per
    estimator besides the headline at the top level
    (``ridge_subject/alpha_y_pred``: one row of test predictions per ridge
    alpha; ``epoch_regression/...``: the epoch-level rows; ``holdout/...``).
    A group without id columns of its own has the top level's rows; when it
    covers only some of them, ``<group>/row`` holds their indices.
Several seeds (``probe --seed-mode shared``):
    ``seed_y_pred`` (s, n) and ``seed_y_score`` (s, n) hold every seed's
    predictions, in the order of ``info["seeds"]``; ``y_pred``, ``y_proba``
    and ``y_score`` are those of the seed selected on validation.
``info`` (JSON):
    ``fit``             what the probe chose on the training and validation
                        splits -- the seed, C, ridge alpha, decision threshold,
                        and the validation scores it chose them by. A rescore
                        keeps these (they are part of fitting) and recomputes
                        everything measured on the test rows.
    ``score``           other settings the metrics need (seconds per window, ...)
    ``groups``          the group names, in the order the metrics list them
    ``seeds``, ``pooling``, ``probe``, ``task``, ``dataset_context``,
    ``checkpoint_path``, ``embeddings`` (each embedding cache read, with its
    files' size and modification time): what the probe was given -- `run`
    reuses the file only when all of it still holds
    ``created_at``, ``neuroatlas_version``

Example -- the macro-F1 of one sleep-staging fold, by hand::

    import numpy as np
    from sklearn.metrics import f1_score
    z = np.load("predictions.npz")
    f1_score(z["y_true"], z["y_pred"], average="macro")

or every metric the task records: ``neuroatlas.predictions.score(path)``.
"""
from __future__ import annotations

import io
import json
import math
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

FORMAT = "neuroatlas.predictions/1"
#: The predictions of one fold, in its probe folder.
FILENAME = "predictions.npz"
#: The result row the probe recorded beside them: what `run` restores, with
#: the metrics recomputed, when it reuses the predictions.
RESULT_FILENAME = "result.json"

_IDENTITY = ("format", "dataset", "checkpoint_id", "task", "fold", "info")
#: Id columns copied from the rows' metadata, when the rows carry them.
TEXT_IDS = ("subject_id", "recording_id", "session_id")
INDEX_IDS = ("epoch_index", "trial_idx")


class PredictionsError(ValueError):
    """A file that is not a predictions file this version can read."""


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def dumps(value: Any) -> str:
    return json.dumps(value, default=_jsonable)


def fold_text(fold: Any) -> str:
    return "" if fold is None else str(fold)


@dataclass
class Predictions:
    """One fold's test predictions (see the module docstring for the format)."""

    dataset: str
    checkpoint_id: str
    task: str
    fold: str
    columns: Dict[str, np.ndarray] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None
    #: saved by the seizure probe before the common format (2026-10-06):
    #: test-set columns only, no info
    legacy: bool = False

    # -- reading ------------------------------------------------------------
    def group(self, name: str = "") -> Dict[str, np.ndarray]:
        """The columns of one group (``""``: the top level), without prefix."""
        if not name:
            return {k: v for k, v in self.columns.items() if "/" not in k}
        prefix = name + "/"
        return {k[len(prefix):]: v for k, v in self.columns.items()
                if k.startswith(prefix) and "/" not in k[len(prefix):]}

    @property
    def fit(self) -> Dict[str, Any]:
        return self.info.get("fit") or {}

    @property
    def groups(self) -> List[str]:
        return list(self.info.get("groups") or [])

    # -- writing ------------------------------------------------------------
    def _arrays(self) -> Dict[str, np.ndarray]:
        out = {
            "format": np.asarray(FORMAT),
            "dataset": np.asarray(str(self.dataset)),
            "checkpoint_id": np.asarray(str(self.checkpoint_id)),
            "task": np.asarray(str(self.task)),
            "fold": np.asarray(str(self.fold)),
            "info": np.asarray(dumps(self.info)),
        }
        for key, value in self.columns.items():
            if key in _IDENTITY or key.split("/")[-1] in ("", "file"):
                raise PredictionsError(f"column name {key!r} is reserved")
            arr = np.asarray(value)
            if arr.dtype == object:          # no pickles in the file
                arr = arr.astype(str)
            out[key] = arr
        return out

    def save(self, path: Path) -> Path:
        """Write the file (atomically: a crash leaves no half file)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "wb") as handle:
                np.savez_compressed(handle, **self._arrays())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        self.path = path
        return path

    def roundtrip(self) -> "Predictions":
        """What a reader of the saved file gets: the same arrays and the JSON
        info as read back (``score`` of this == ``score`` of the file)."""
        buf = io.BytesIO()
        np.savez(buf, **self._arrays())
        buf.seek(0)
        with np.load(buf, allow_pickle=False) as z:
            return _from_npz(z, None)


def _from_npz(z, path: Optional[Path]) -> Predictions:
    keys = list(z.files)
    if "format" not in keys:
        return _legacy(z, path)
    fmt = str(z["format"])
    if fmt != FORMAT:
        raise PredictionsError(f"{path}: format {fmt!r}; this version of neuroatlas reads {FORMAT!r}")
    info = json.loads(str(z["info"]))
    columns = {k: z[k] for k in keys if k not in _IDENTITY}
    return Predictions(str(z["dataset"]), str(z["checkpoint_id"]), str(z["task"]),
                       str(z["fold"]), columns, info, path)


def _legacy(z, path: Optional[Path]) -> Predictions:
    """The seizure probe's first predictions files (commit 1803e9a): window
    labels, scores, ids, window length and threshold -- no format, no info."""
    keys = set(z.files)
    need = {"y_true", "y_score", "recording_id", "window_s", "dataset", "checkpoint_id", "fold"}
    if not need <= keys:
        raise PredictionsError(f"{path}: not a neuroatlas predictions file "
                               f"(no 'format' entry, and not the seizure probe's first layout)")
    columns = {k: z[k] for k in ("y_true", "y_score", "recording_id", "subject_id",
                                 "window_start_s", "window_s", "threshold") if k in keys}
    threshold = float(columns.get("threshold", np.nan))
    if np.isfinite(threshold):
        columns["y_pred"] = (np.asarray(columns["y_score"]) >= threshold).astype(np.int64)
    info = {"legacy": "saved by the seizure probe before the common format: test columns only",
            "groups": [], "fit": {}, "score": {}}
    return Predictions(str(z["dataset"]), str(z["checkpoint_id"]), "seizure_detection",
                       str(z["fold"]), columns, info, path, legacy=True)


def load(path) -> Predictions:
    """Read a predictions file."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as z:
            return _from_npz(z, path)
    except PredictionsError:
        raise
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise PredictionsError(f"{path}: cannot read it as a predictions file ({exc})") from None


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def scorer(task: str) -> Callable[[Predictions], Dict[str, Any]]:
    """The function that computes *task*'s metrics from its predictions."""
    from neuroatlas.benchmarking_helpers.registry.discovery import load_task_spec

    spec = load_task_spec(task)
    if getattr(spec, "score", None) is None:
        raise PredictionsError(f"task {task!r} has no function that scores saved predictions")
    return spec.score


def score(source) -> Dict[str, Any]:
    """Every metric the task records, recomputed from a predictions file (a
    path or a :class:`Predictions`)."""
    pred = source if isinstance(source, Predictions) else load(source)
    return scorer(pred.task)(pred)


def finalize(pred: Predictions, probe_dir: Optional[Path],
             score_fn: Callable[[Predictions], Dict[str, Any]]
             ) -> Tuple[Dict[str, Any], Predictions, Optional[Path]]:
    """Save *pred* in *probe_dir* and compute the metrics from the file as
    saved, with the function a rescore uses -- so a rescore gives exactly
    these metrics. Without a probe folder (a task called directly), from an
    in-memory copy identical to that file. Returns (metrics, the saved
    predictions as read back, the path or None)."""
    if probe_dir is not None:
        path = pred.save(Path(probe_dir) / FILENAME)
        saved = load(path)
    else:
        path, saved = None, pred.roundtrip()
    return score_fn(saved), saved, path


# --------------------------------------------------------------------------
# building a file in a task
# --------------------------------------------------------------------------

def id_columns(meta: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """The id columns of these rows: text ids, epoch/trial indices (-1 where a
    row has none) and window start times (NaN where none), for the keys any
    row carries."""
    meta = list(meta)
    out: Dict[str, np.ndarray] = {}
    for key in TEXT_IDS:
        if any(key in m for m in meta):
            out[key] = np.asarray(["" if m.get(key) is None else str(m.get(key)) for m in meta],
                                  dtype=str)
    for key in INDEX_IDS:
        if any(key in m for m in meta):
            out[key] = np.asarray([-1 if m.get(key) is None else int(m.get(key)) for m in meta],
                                  dtype=np.int64)
    if any("window_start_s" in m for m in meta):
        out["window_start_s"] = np.asarray(
            [np.nan if m.get("window_start_s") is None else float(m["window_start_s"]) for m in meta],
            dtype=np.float64)
    return out


def embedding_stamp(cache_dir) -> Optional[Dict[str, List[int]]]:
    """Size and modification time (ns) of an embedding cache's files: changes
    when the cache is extracted again."""
    d = Path(cache_dir)
    if not d.is_dir():
        return None
    out: Dict[str, List[int]] = {}
    for name in ("features.npy", "labels.npy", "items.json", "metadata.json"):
        try:
            st = (d / name).stat()
        except OSError:
            continue
        out[name] = [int(st.st_size), int(st.st_mtime_ns)]
    return out or None


def embeddings_used(cache_paths: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The embedding caches a probe read (``*cache_dir`` entries of its
    cache_paths), each with its stamp."""
    out: List[Dict[str, Any]] = []
    seen = set()
    for key, value in (cache_paths or {}).items():
        if not str(key).endswith("cache_dir") or not value:
            continue
        path = str(value)
        if path in seen:
            continue
        seen.add(path)
        prefix = str(key)[: -len("cache_dir")]
        out.append({"key": str(key), "path": path,
                    "external": bool((cache_paths or {}).get(prefix + "external_cache")),
                    "stamp": embedding_stamp(path)})
    return out


def make_info(*, datamodule, checkpoint_spec, probe_config=None, task_config=None, seeds=(),
              cache_paths=None, fit=None, score=None, groups=None) -> Dict[str, Any]:
    """The ``info`` of a fold's file: what the probe was given (for `run`'s
    reuse check), what it chose on validation (``fit``), what scoring needs."""
    from neuroatlas import __version__

    try:
        from neuroatlas.extensions.tasks.linear_probe import current_pooling

        pooling = current_pooling()
    except Exception:  # pragma: no cover -- linear_probe always imports here
        pooling = "mean"
    return {
        "fit": fit or {},
        "score": score or {},
        "groups": list(groups or []),
        "seeds": [int(s) for s in (seeds or [])],
        "pooling": pooling,
        "probe": dict(probe_config or {}),
        "task": dict(task_config or {}),
        "dataset_context": dict(getattr(datamodule, "metadata", None) or {}),
        "checkpoint_path": getattr(checkpoint_spec, "checkpoint_path", None),
        "embeddings": embeddings_used(cache_paths or {}),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "neuroatlas_version": __version__,
    }


def new(task: str, *, dataset_name: str, checkpoint_spec, datamodule,
        columns: Dict[str, Any], info: Dict[str, Any]) -> Predictions:
    fold = (getattr(datamodule, "metadata", None) or {}).get("fold")
    return Predictions(str(dataset_name), str(checkpoint_spec.identifier), task,
                       fold_text(fold), dict(columns), info)


def prefixed(group: str, columns: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{group}/{k}": v for k, v in columns.items()}


# --------------------------------------------------------------------------
# seed-selected probes (train_probe, the LSTM and attention probes)
# --------------------------------------------------------------------------

def probe_columns(result, *, with_row: bool = False) -> Dict[str, np.ndarray]:
    """The test columns of a :class:`ProbeResult`: labels, the selected
    seed's predictions (and every seed's, when there are several), and --
    with *with_row* -- the indices of the rows it scored when it dropped some
    (non-finite embeddings)."""
    outputs = result.test_outputs
    if outputs is None or result.test_y is None:
        raise PredictionsError("this probe result kept no test predictions")
    seeds = [int(r["seed"]) for r in result.per_seed]
    best = outputs[seeds.index(int(result.best_seed))]
    cols: Dict[str, np.ndarray] = {"y_true": np.asarray(result.test_y),
                                   "y_pred": np.asarray(best["y_pred"])}
    if best.get("y_score") is not None:
        cols["y_score"] = np.asarray(best["y_score"], dtype=np.float64)
    if best.get("y_proba") is not None:
        cols["y_proba"] = np.asarray(best["y_proba"], dtype=np.float32)
        if result.classes is not None:
            cols["classes"] = np.asarray(result.classes)
    if len(seeds) > 1:
        cols["seed_y_pred"] = np.stack([np.asarray(o["y_pred"]) for o in outputs])
        if any(o.get("y_score") is not None for o in outputs):
            n = len(cols["y_true"])
            cols["seed_y_score"] = np.stack([
                np.asarray(o["y_score"], dtype=np.float64) if o.get("y_score") is not None
                else np.full(n, np.nan) for o in outputs])
    keep = result.test_keep
    if with_row and keep is not None and not bool(np.all(keep)):
        cols["row"] = np.flatnonzero(np.asarray(keep)).astype(np.int64)
    return cols


def probe_fit(result, *, selection_metric: str, probe_type: str, mode: str = "classification",
              higher_is_better: bool = True, hidden_dims=None) -> Dict[str, Any]:
    """What a seed-selected probe chose on validation: every seed's
    validation scores and the metric the seed was selected by."""
    fit = {"seeds": [int(r["seed"]) for r in result.per_seed],
           "val": [r["val"] for r in result.per_seed],
           "selection_metric": str(selection_metric),
           "higher_is_better": bool(higher_is_better),
           "probe_type": str(probe_type), "mode": mode}
    if hidden_dims:
        fit["hidden_dims"] = list(hidden_dims)
    return fit


def _seed_scores(cols: Dict[str, np.ndarray], n_seeds: int) -> List[Optional[np.ndarray]]:
    if n_seeds == 1:
        return [cols.get("y_score")]
    stacked = cols.get("seed_y_score")
    if stacked is None:
        return [None] * n_seeds
    return [None if np.all(np.isnan(row)) else row for row in stacked]


def score_probe(cols: Dict[str, np.ndarray], fit: Dict[str, Any]
                ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """A seed-selected probe's metrics from its test columns and its ``fit``:
    exactly the summary ``train_probe`` reports (per-metric mean and std over
    seeds, the selected seed's validation and test blocks), and the per-seed
    rows."""
    from neuroatlas.benchmarking_helpers.probes.metrics import (
        compute_classification_metrics,
        compute_regression_metrics,
    )

    seeds = [int(s) for s in fit["seeds"]]
    regression = fit.get("mode") == "regression"
    y_true = cols["y_true"]
    preds = [cols["y_pred"]] if len(seeds) == 1 else list(cols["seed_y_pred"])
    scores = _seed_scores(cols, len(seeds))
    per_seed: List[Dict[str, Any]] = []
    for seed, val, y_pred, y_score in zip(seeds, fit["val"], preds, scores):
        test = (compute_regression_metrics(y_true, y_pred) if regression
                else compute_classification_metrics(y_true, y_pred, y_score=y_score))
        per_seed.append({"seed": seed, "val": val, "test": test})
    selection = fit["selection_metric"]
    pick = max if fit.get("higher_is_better", True) else min
    best_row = pick(per_seed, key=lambda row: float(row["val"][selection]))
    names = (["mae", "rmse", "r2", "pearson_r"] if regression
             else ["accuracy", "macro_f1", "weighted_f1", "cohen_kappa"])
    summary: Dict[str, Any] = {}
    for name in names:
        values = np.asarray([row["test"][name] for row in per_seed], dtype=float)
        summary[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    if not regression:
        summary["confusion_matrix"] = per_seed[0]["test"]["confusion_matrix"]
    summary["best_seed"] = int(best_row["seed"])
    summary["best_val_metric"] = float(best_row["val"][selection])
    summary["best_val"] = best_row["val"]
    summary["best_test"] = best_row["test"]
    summary["probe_type"] = fit.get("probe_type", "linear")
    if fit.get("hidden_dims"):
        summary["hidden_dims"] = list(fit["hidden_dims"])
    return summary, per_seed


def per_seed(pred: Predictions) -> Optional[List[Dict[str, Any]]]:
    """The per-seed rows of a seed-selected probe's file (results.json keeps
    them in metadata.per_seed), or None for other tasks."""
    fit = pred.fit
    if "seeds" not in fit or "val" not in fit:
        return None
    return score_probe(pred.group(""), fit)[1]


# --------------------------------------------------------------------------
# reuse
# --------------------------------------------------------------------------

def _inside(path: Path, root: Path) -> bool:
    try:
        Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root)))
        return True
    except ValueError:
        return False


def reuse_problem(pred: Predictions, *, task: str, seeds: Sequence[int], pooling: str,
                  checkpoint_path: Optional[str], cache_root: Optional[Path]) -> Optional[str]:
    """Why these saved predictions are not what a probe with these inputs
    would make, or None if they are. (The probe folder already fixes the
    probe settings, the task settings and the dataset context, fold
    included; this checks what the folder's key does not.)"""
    if pred.legacy:
        return "they were saved in an older format"
    info = pred.info
    if pred.task != task:
        return f"they were made by task {pred.task}, this probe is {task}"
    if [int(s) for s in info.get("seeds") or []] != [int(s) for s in seeds]:
        return f"they were probed with seeds {info.get('seeds')}, this probe uses {list(seeds)}"
    if (info.get("pooling") or "mean") != pooling:
        return f"they were probed on --pooling {info.get('pooling')} embeddings, not {pooling}"
    if (info.get("checkpoint_path") or None) != (checkpoint_path or None):
        return (f"the checkpoint's weights are another file now ({checkpoint_path}, "
                f"they were made with {info.get('checkpoint_path')})")
    for emb in info.get("embeddings") or []:
        path = Path(emb.get("path", ""))
        if cache_root is not None and not emb.get("external") and not _inside(path, cache_root):
            return f"they were fitted on embeddings under another cache root ({path})"
        if embedding_stamp(path) != emb.get("stamp"):
            return f"the embeddings they were fitted on were extracted again or are gone ({path})"
    return None


def refresh_metadata(metadata: Dict[str, Any], pred: Predictions) -> None:
    """Metadata a result row derives from its predictions (the per-seed
    blocks of a seed-selected probe), recomputed with the metrics."""
    if "per_seed" in metadata:
        rows = per_seed(pred)
        if rows is not None:
            metadata["per_seed"] = rows


def write_result(probe_dir: Path, row: Dict[str, Any]) -> Path:
    """Keep the result row a probe recorded next to its predictions."""
    path = Path(probe_dir) / RESULT_FILENAME
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(row, indent=2, default=_jsonable), encoding="utf-8")
    os.replace(tmp, path)
    return path


def finite_or_none(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None
