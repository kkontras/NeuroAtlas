from __future__ import annotations

import contextlib
import fcntl
import functools
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from neuroatlas import progress

from .cache import build_cache_key, cache_exists
from ..channels.channel_map import (
    ChannelMap,
    ChannelMapSkip,
    load_channel_map,
    rewrite_meta_channels,
)
from ..registry.contracts import BenchmarkFailure, BenchmarkResult
from ..registry.discovery import load_backbone, load_checkpoint_registry, load_dataset_spec
from ..registry.discovery import evaluate_task


class _PrecomputedStubBackbone:
    """Stand-in backbone for datamodules that serve precomputed embeddings.

    Task evaluators only call ``backbone.metadata()`` on the surface path
    (it's merged into ``BenchmarkResult.metadata``). They never call
    ``extract_embeddings`` when the precomputed-cache hook fires.
    """

    def __init__(self, checkpoint_spec) -> None:
        self._spec = checkpoint_spec

    def metadata(self) -> Dict[str, Any]:
        return {
            "wrapper_name": self._spec.wrapper_name or "precomputed_stub",
            "backbone_source": "precomputed_stub",
            "checkpoint_id": self._spec.identifier,
        }

    def extract_embeddings(self, batch):
        raise RuntimeError(
            "PrecomputedStubBackbone.extract_embeddings was called — the "
            "precomputed-embedding hook should have short-circuited the task."
        )


class _LazyBackbone:
    """The checkpoint's backbone, its weights loaded the first time something
    uses it for more than its metadata: a probe that reads embeddings already
    extracted never loads them, so it runs where the weights are not (another
    machine's cache, weights removed after extraction). A probe that has to
    extract loads them then, and a missing checkpoint fails as it always did."""

    def __init__(self, checkpoint_spec, load) -> None:
        self._spec = checkpoint_spec
        self._load = load
        self._backbone = None

    def loaded(self):
        if self._backbone is None:
            self._backbone = self._load(self._spec)
        return self._backbone

    def metadata(self) -> Dict[str, Any]:
        if self._backbone is not None:
            return self._backbone.metadata()
        # what every backbone reports from its registry entry; the extraction
        # details (device, rates) belong to the run that made the embeddings
        spec = self._spec
        return {"checkpoint_id": spec.identifier, "model_family": spec.model_family,
                "embedding_key": spec.embedding_key, "embedding_dim": spec.embedding_dim,
                "wrapper_name": spec.wrapper_name,
                "backbone_source": "embeddings read from the cache"}

    def __getattr__(self, name):
        return getattr(self.loaded(), name)


def _precomputed_cache_available(datamodule, checkpoint_spec) -> bool:
    fn = getattr(datamodule, "precomputed_embedding_cache_dir", None)
    if not callable(fn):
        return False
    try:
        path = fn(
            checkpoint_id=checkpoint_spec.identifier,
            split="all",
            purpose="global_embeddings",
        )
    except Exception:
        return False
    return path is not None and cache_exists(Path(path))


def _serves_every_fold(result: BenchmarkResult) -> bool:
    """Whether an extraction wrote (or found) embeddings no fold changes: the
    global ``all/<key>`` cache, where every fold's split is cut at probe time.
    Per-split caches hold one fold's train/val/test and serve only that fold."""
    if result.failure is not None:
        return False
    layout = (result.metadata or {}).get("embedding_cache_layout")
    if layout is not None:
        return layout == "all"
    return any(str(k).startswith("global_") or k == "chunk_dir"
               for k in (result.cache_paths or {}))


def _is_regression_result(result: BenchmarkResult) -> bool:
    if getattr(result, "evaluation_mode", None) == "brain_age_eval":
        return True
    best_test = result.metrics.get("best_test", {}) if isinstance(result.metrics, dict) else {}
    if "mae" in best_test or "mae" in (result.metrics or {}):
        return True
    return False


class _ChannelMapDataloaderWrapper:
    """Iterable adapter that rewrites ``meta[i]['channels']`` per batch.

    Only the ``meta`` list is altered; ``signals``, ``label``, and
    ``raw_batch`` pass through untouched. ``len()`` is proxied when the
    underlying loader supports it so probe/task code can size progress
    bars.
    """

    def __init__(self, loader, cmap: ChannelMap, model_family: str):
        self._loader = loader
        self._cmap = cmap
        self._model_family = model_family
        self.dataset = getattr(loader, "dataset", None)
        self.batch_size = getattr(loader, "batch_size", None)
        _inner_collate = getattr(loader, "collate_fn", None)
        if _inner_collate is not None:
            def _collate_with_remap(batch, _c=_inner_collate, _cm=cmap, _mf=model_family):
                result = _c(batch)
                meta = result.get("meta")
                if meta:
                    result = dict(result)
                    result["meta"] = rewrite_meta_channels(meta, _cm, _mf)
                return result
            self.collate_fn = _collate_with_remap
        else:
            self.collate_fn = None
        self._num_workers = getattr(loader, "_num_workers", 0)
        self.batch_sampler = getattr(loader, "batch_sampler", None)

    def __iter__(self):
        for batch in self._loader:
            meta = batch.get("meta")
            if meta:
                batch = dict(batch)
                batch["meta"] = rewrite_meta_channels(
                    meta, self._cmap, self._model_family
                )
            yield batch

    def __len__(self):
        return len(self._loader)


def _Ticker(label: str, every: float = 1.0, *, verb: str = "probing", stream=None,
            start_line_off_tty: bool = False) -> progress.Progress:
    """Shows that a run is alive while it works. On a terminal: one line,
    "<label>: fitting 33% (1/3 seeds, 0m 05s)", naming the phase the code
    inside reports (loading the data, reading the embeddings, loading
    weights, fitting seed by seed, scoring), rewritten every second and
    cleared when the run ends (its result line follows). Off a terminal
    nothing, unless *start_line_off_tty*: the result line alone (a
    leave-one-subject-out probe over many models is thousands of folds)."""
    return progress.Progress(label, verb=verb, every=every, stream=stream,
                             start_line_off_tty=start_line_off_tty,
                             screen_only=not start_line_off_tty)


# The metric a progress line shows for a fold that belongs to no benchmark
# (a probe run by hand on another task), first one present.
_PROGRESS_METRICS = ("event_sens_fa_auc", "auroc", "balanced_accuracy", "mae", "pearson_r")


@functools.lru_cache(maxsize=None)
def _benchmark_headline(dataset: str, task_name: str) -> Optional[Tuple[str, Tuple[str, ...], str]]:
    """(metric key, where in the fold's metrics, column label) of the
    benchmark a probe of *dataset* with the registered task *task_name*
    belongs to: sleep staging's kappa, BCI's balanced accuracy, AUPRC at an
    event benchmark's threshold. None when no benchmark runs that pair."""
    try:
        from neuroatlas import catalog, metrics_info
        from neuroatlas.benchmarking_helpers.registry.discovery import load_dataset_spec
        from neuroatlas.entrypoints.probe import load_task_preset

        for bench in catalog.catalog().values():
            if bench.derived_from:
                continue
            for entry in bench.datasets:
                if entry.slug != dataset:
                    continue
                # a benchmark that names no task (BCI) runs the dataset's own
                preset = entry.task or bench.task or load_dataset_spec(dataset).default_task
                if not preset:
                    continue
                if (load_task_preset(preset)["task"].get("name") or preset) == task_name:
                    m = bench.metrics
                    return m.headline, tuple(m.at), metrics_info.label(m.headline, bench)
    except (Exception, SystemExit):     # a catalog or preset that does not load: no headline
        return None
    return None


def _headline(result) -> Optional[Tuple[str, float]]:
    """(label, value) a fold's result line shows: the headline of the
    benchmark the probe belongs to, read as `results` reads it and named by
    its column there (``kappa``, ``bal_acc``, ``MAE(years)``); else the first
    of _PROGRESS_METRICS among the result's metrics; for a regression probe
    (brain age), its headline estimator's test block (``best_test``)."""
    metrics = result.metrics or {}
    task_name = (result.metadata or {}).get("task_name")
    spec = _benchmark_headline(result.dataset_name, task_name) if task_name else None
    if spec:
        from neuroatlas import results as res

        key, at, label = spec
        value = res.metric(metrics, key, at)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:
            return label, float(value)
    blocks = [metrics]
    if _is_regression_result(result) and isinstance(metrics.get("best_test"), dict):
        blocks.append(metrics["best_test"])
    from neuroatlas import metrics_info

    for block in blocks:
        for name in _PROGRESS_METRICS:
            value = block.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return metrics_info.label(name), float(value)
    return None


#: A run's failure codes that are not failures: the channel map rules the
#: pair out (``skip``), or has no entry for its family (``invalid``).
RULED_OUT_CODE = "channel_map_skip"
INVALID_CODE = "channel_map_invalid"
#: Failure codes of a run that could not start here: the dataset's data, or
#: the checkpoint's weights, are not on this machine (``skipped``).
DATA_MISSING_CODE = "data_missing"
WEIGHTS_MISSING_CODE = "weights_missing"
SKIPPED_CODES = (DATA_MISSING_CODE, WEIGHTS_MISSING_CODE)


def outcome_word(result) -> str:
    """``ok``, ``ruled out``, ``invalid`` or ``failed``: what a run's result
    line and the closing count call it."""
    if result.ok:
        return "ok"
    code = getattr(getattr(result, "failure", None), "code", None)
    if code == RULED_OUT_CODE:
        return "ruled out"
    if code == INVALID_CODE:
        return "invalid"
    if code in SKIPPED_CODES:
        return "skipped"
    return "failed"


def _progress_line(done: int, total: int, result, seconds: float) -> str:
    fold = (result.metadata or {}).get("fold")
    where = f"{result.dataset_name} {result.checkpoint_id}" + ("" if fold is None else f" fold {fold}")
    if not result.ok:
        what = outcome_word(result)
        if what != "failed":
            # decided before any data is read: no time to show
            return progress.result_line(done, total, where, what, None)
    else:
        value = _headline(result)
        # reused: recomputed from the fold's saved predictions, not fitted
        what = "ok" + (", reused" if (result.metadata or {}).get("reused_predictions") else "") \
            + (f", {value[0]} {value[1]:.3f}" if value else "")
    return progress.result_line(done, total, where, what, seconds)


def _load_weights(checkpoint_spec):
    """The backbone with its weights, the item's live line saying so. What a
    model's code prints while it loads (REVE's remote code: "flash_attn not
    found") becomes log lines (-v, --log), and a hub download inside it
    shows on the line, not as huggingface_hub's bars."""
    from neuroatlas import quiet
    from neuroatlas.extensions.models.backbones._checkpoint_download import hub_progress

    progress.current().phase("loading weights")
    try:
        with quiet.printed_to_log(logging.getLogger(__name__)), hub_progress(at_first_byte=True):
            return load_backbone(checkpoint_spec)
    except FileNotFoundError as exc:
        # what `run` and `check` say before they start, naming this checkpoint
        # and the command that gets its weights
        from neuroatlas.run import weights_problem_for

        try:
            problem = weights_problem_for(checkpoint_spec)
        except Exception:
            problem = None
        if problem:
            raise WeightsMissing(problem) from exc
        raise


def _exception_text(exc: BaseException) -> str:
    """What a run's failure says: our own errors, written for a user, as
    they are; anything else with its type where that says something
    (:func:`neuroatlas.cli._msg.exception_text`)."""
    from neuroatlas.cli import _msg

    from ..registry.contracts import DatamoduleConfigError
    from ..registry.discovery import UnknownName
    from .cache import CacheCorruptError, CacheLockError

    if isinstance(exc, (DatamoduleConfigError, UnknownName, CacheCorruptError, CacheLockError)):
        return str(exc)
    return _msg.exception_text(exc)


class WeightsMissing(FileNotFoundError):
    """The checkpoint's weights are not on this machine: its runs are
    skipped (the message names the command that gets them)."""


#: What `data status` says of a dataset that is not here, in a sentence
#: about its folder (the run that needed it is skipped).
_DATA_STATES = {
    "missing": "no {ds} data here: {path} does not exist",
    "empty": "no {ds} data here: {path} holds no recordings",
    "partial": "the {ds} data is incomplete: {path} holds {count}",
    "not prepared": "the file {ds} is read from is not here: it is not in {path}",
    "not downloaded": "no {ds} data here: it is not in the MOABB data folder {path}",
    "not configured": "no {ds} data here: no data root is set, so its folder is unknown",
}


from neuroatlas import quiet as _quiet

#: dataset -> data_missing_text, for one command (a status walks the folder)
_DATA_MISSING: Dict[str, Optional[str]] = _quiet.per_command({})


def data_missing_text(dataset_name: str) -> Optional[str]:
    """When `data status` finds *dataset_name* is not here: what to say and
    the commands that bring it (``what\nfix: ...``); else None. A run that
    fails on such a dataset fails for that reason, whatever it raised."""
    if not _quiet.in_command():
        return _data_missing_text(dataset_name)
    if dataset_name not in _DATA_MISSING:
        _DATA_MISSING[dataset_name] = _data_missing_text(dataset_name)
    return _DATA_MISSING[dataset_name]


def _data_missing_text(dataset_name: str) -> Optional[str]:
    try:
        from neuroatlas import data as data_mod
        from neuroatlas.check import data_fix

        st = data_mod.status(dataset_name)
    except Exception:       # a status that cannot be taken: say what was raised
        return None
    state, _, rest = str(st.state).partition(" (")
    if st.found or state not in _DATA_STATES:
        return None
    what = _DATA_STATES[state].format(ds=dataset_name, path=st.path or "its folder",
                                      count=rest.rstrip(")") or "part of it")
    return f"{what}\nfix: {data_fix(dataset_name, st)}"


#: An extraction pass that found its embeddings: its result line (with its
#: time when that was long: the data was read to match them).
ALREADY_CACHED = "embeddings already in the cache"
#: A cached extraction's result line shows its time from this many seconds on.
_CACHED_SHOWN_AFTER = 2.0


def _count(n: int, noun: str) -> str:
    """``1 checkpoint``, ``3 checkpoints``."""
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _embed_outcome(results: List[BenchmarkResult], counts: Dict[str, int]) -> str:
    """What an extraction pass over one (dataset, model) did, for its result
    line: ``ok, 50,749 windows``, ``already in the cache``, ``ruled out``,
    ``invalid``, ``failed``."""
    words = {outcome_word(r) for r in results}
    for word in ("failed", "skipped", "invalid", "ruled out"):
        if word in words:
            return word
    if not results:
        return "failed"
    if counts.get("extracted"):
        windows = counts.get("windows")
        unit = _window_unit(results[0])
        return f"ok, {windows:,} {unit}" if windows else "ok"
    return ALREADY_CACHED


def _window_unit(result: BenchmarkResult) -> str:
    """What one embedded window is on this dataset, for the extraction's
    result line: ``trials`` (a BCI trial), ``epochs (30 s)`` (a sleep
    epoch), else ``windows (10 s)``."""
    meta = result.metadata or {}
    seconds = meta.get("epoch_seconds") or meta.get("window_s")
    try:
        domain = (load_dataset_spec(result.dataset_name).manifest or {}).get("domain") or ""
    except Exception:
        domain = ""
    if domain == "bci":
        return "trials"
    length = ""
    try:
        if seconds:
            length = f" ({float(seconds):g} s)"
    except (TypeError, ValueError):
        length = ""
    return ("epochs" if domain == "sleep" else "windows") + length


def _wrap_datamodule_with_channel_map(datamodule, cmap: ChannelMap, model_family: str):
    """Wrap each of train/val/test dataloaders with channel rewriting.

    Returns the same datamodule object — mutation happens by replacing
    the three ``*_dataloader`` methods with bound wrappers so downstream
    code that calls e.g. ``datamodule.train_dataloader()`` sees the
    rewritten batches.
    """
    for name in ("train_dataloader", "val_dataloader", "test_dataloader", "full_embedding_dataloader"):
        original = getattr(datamodule, name, None)
        if original is None:
            continue

        def _wrap(original_fn=original):
            def _wrapped(*args, **kwargs):
                return _ChannelMapDataloaderWrapper(
                    original_fn(*args, **kwargs), cmap, model_family
                )
            return _wrapped

        setattr(datamodule, name, _wrap())
    return datamodule


@contextlib.contextmanager
def results_lock(output_root: Path):
    """Exclusive use of ``<output_root>/results.json`` (and the tables made
    from it) while it is read and rewritten: two jobs writing one folder, or
    a rescore beside a run, must not lose each other's rows."""
    lock_path = Path(output_root) / ".results.lock"
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.lockf(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.lockf(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
        lock_path.unlink(missing_ok=True)


class BenchmarkRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        benchmark_cfg = config.get("benchmark", {})
        self.output_root = Path(benchmark_cfg.get("output_root", "artifacts/benchmarks"))
        self.cache_root = Path(benchmark_cfg.get("cache_root", self.output_root / "cache"))
        self.seeds = list(benchmark_cfg.get("seeds", [0, 1, 2]))
        self.extract_only: bool = bool(benchmark_cfg.get("extract_only", False))
        # Set by `probe`: a cache miss is then an error naming `embed`, rather
        # than a silent backbone extraction inside a probing job.
        self.require_cached_embeddings: bool = bool(
            benchmark_cfg.get("require_cached_embeddings", False)
        )
        self.embed_chunk: Optional[Tuple[int, int]] = benchmark_cfg.get("embed_chunk")
        # A fold whose predictions.npz is already in its probe folder, made
        # from the same inputs, is rescored rather than probed again, unless
        # `probe --reprobe` (neuroatlas.predictions).
        self.reprobe: bool = bool(benchmark_cfg.get("reprobe", False))

    def _selected_datasets(self) -> List[str]:
        return list(self.config.get("datasets", {}).keys())

    def _selected_models(self) -> List[str]:
        return list(self.config.get("benchmark", {}).get("models", []))

    def _checkpoint_overrides(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.config.get("benchmark", {}).get("checkpoint_overrides", {}))

    def _seed_per_fold(self) -> bool:
        return bool(self.config.get("benchmark", {}).get("seed_per_fold", False))

    def _task_config(self) -> Dict[str, Any]:
        return dict(self.config.get("task", {}))

    def _probe_config(self) -> Dict[str, Any]:
        return dict(self.config.get("probe", {}))

    def _apply_spec_overrides(self, spec):
        override = self._checkpoint_overrides().get(spec.identifier) or self._checkpoint_overrides().get(spec.model_family)
        if not override:
            return spec
        payload = spec.to_dict()
        payload.update(override)
        return type(spec)(**payload)

    def _effective_seeds(self, datamodule) -> List[int]:
        if self._seed_per_fold():
            fold = getattr(datamodule, "metadata", {}).get("fold")
            if fold is not None:
                return [int(fold)]
        return list(self.seeds)

    def _dataset_runs(self) -> List[tuple[str, Dict[str, Any]]]:
        runs: List[tuple[str, Dict[str, Any]]] = []
        for dataset_name in self._selected_datasets():
            dataset_spec = load_dataset_spec(dataset_name)
            dataset_config = dict(self.config["datasets"][dataset_name])
            for run_config in dataset_spec.expand_runs(dataset_config):
                runs.append((dataset_name, run_config))
        return runs

    def _task_name_for(self, dataset_spec, checkpoint_spec) -> str:
        explicit = self._task_config().get("name")
        if explicit:
            return str(explicit)
        if dataset_spec.default_task not in {"linear_probe", "native_head_eval"}:
            return str(dataset_spec.default_task)
        if checkpoint_spec.evaluation_mode(dataset_spec.slug) == "native_head_eval":
            return "native_head_eval"
        return "linear_probe"

    def _probe_dir(self, dataset_name: str, checkpoint_id: str, datamodule) -> Path:
        probe_parts = {
            "probe": self._probe_config(),
            "task": self._task_config(),
            "dataset_context": dict(getattr(datamodule, "metadata", {})),
        }
        from neuroatlas.extensions.tasks.linear_probe import current_pooling

        # Only when it is not the default, as in the embedding cache key: a
        # per_patch probe must not land in (and be reused as) the mean one's
        # folder, and the default keeps every existing folder's key.
        if current_pooling() != "mean":
            probe_parts["pooling"] = current_pooling()
        key = build_cache_key(probe_parts)
        return self.output_root / "probes" / dataset_name / checkpoint_id / key

    def _saved_result(self, dataset_name: str, checkpoint_spec, datamodule, task_name: str,
                      probe_dir: Path) -> Optional[BenchmarkResult]:
        """This fold's result recomputed from the predictions an earlier probe
        saved in *probe_dir*, when they were made from the same inputs;
        otherwise None (the fold is probed). The folder's key already fixes
        the probe and task settings and the dataset context, fold included;
        predictions.reuse_problem checks the rest (seeds, pooling, weights,
        and that the embeddings are the ones they were fitted on)."""
        from neuroatlas import predictions as preds
        from neuroatlas.extensions.tasks.linear_probe import current_pooling

        path = probe_dir / preds.FILENAME
        if not path.is_file():
            return None
        fold = (getattr(datamodule, "metadata", None) or {}).get("fold")
        where = f"{dataset_name}/{checkpoint_spec.identifier}" + (
            f" fold {fold}" if fold is not None else "")
        sidecar = probe_dir / preds.RESULT_FILENAME
        try:
            pred = preds.load(path)
            why = None if sidecar.is_file() else "they were saved without their result record"
            why = why or preds.reuse_problem(
                pred, task=task_name, seeds=self._effective_seeds(datamodule),
                pooling=current_pooling(), checkpoint_path=checkpoint_spec.checkpoint_path,
                cache_root=self.cache_root)
            if why is None:
                row = BenchmarkResult.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))
                if (row.dataset_name, row.checkpoint_id) != (dataset_name, checkpoint_spec.identifier):
                    why = "their result.json is another dataset's or model's"
        except Exception as exc:  # unreadable: probe again, and say so
            from neuroatlas.cli import _msg

            why = f"they cannot be read ({_msg.first_sentence(_msg.exception_text(exc))})"
        if why is not None:
            from neuroatlas import quiet

            pair = f"{dataset_name}/{checkpoint_spec.identifier}"
            if quiet.once(f"not reused:{pair}"):
                # once per (dataset, model): its other folds say so in the log
                from neuroatlas.cli import _msg

                _msg.note(f"{pair}: fitting again, not scoring the saved predictions: {why}")
            else:
                logging.getLogger(__name__).info(
                    "%s: fitting again, not scoring the saved predictions: %s", where, why)
            return None
        progress.current().phase("scoring the saved predictions")
        row.metrics = preds.scorer(task_name)(pred)
        preds.refresh_metadata(row.metadata, pred)
        row.cache_paths["predictions"] = str(path)
        row.metadata.pop("written_at", None)
        row.metadata["reused_predictions"] = True
        # the fold's progress line says "ok, reused"; the file, for -v and --log
        logging.getLogger(__name__).info("%s: reused the saved predictions %s", where, path)
        return row

    @staticmethod
    def _remember(result: BenchmarkResult, probe_dir: Path) -> None:
        """Keep the result row beside the predictions it was computed from:
        what a later run restores when it reuses them."""
        if result.ok and (result.cache_paths or {}).get("predictions"):
            from neuroatlas import predictions as preds

            try:
                preds.write_result(probe_dir, result.to_dict())
            except OSError as exc:
                from neuroatlas.cli import _msg

                fold = (result.metadata or {}).get("fold")
                _msg.warning(f"{result.dataset_name}/{result.checkpoint_id}"
                             + (f" fold {fold}" if fold is not None else "")
                             + f": could not save this fold's result next to its predictions "
                             f"({_msg.first_sentence(_msg.exception_text(exc))}); a later run "
                             f"fits it again")

    @staticmethod
    def _pair_config(dataset_config: Dict[str, Any], cmap: Optional[ChannelMap],
                     checkpoint_spec) -> Dict[str, Any]:
        """The dataset config one checkpoint sees: its channels, and recording
        statistics for the models that normalise by them."""
        if cmap is not None:
            family = checkpoint_spec.model_family.lower()
            dataset_config = {**dataset_config, "channel_specs": cmap.channels_for(family)}
            montage = cmap.montage_for(family)
            if montage is not None:
                # The montage this model was pretrained on (the map's
                # model_montage), not one montage for every model.
                dataset_config["montage"] = montage
        if checkpoint_spec.model_family.lower() in {"reve", "biot", "sleepfm", "steegformer", "neurogpt"}:
            dataset_config = {**dataset_config, "compute_recording_stats": True}
        return dataset_config

    @staticmethod
    def _fit_checkpoint_to_window(datamodule, checkpoint_spec, *, verbose: bool = False):
        """The checkpoint as it must see this datamodule's windows.

        Where the dataset fixes the window (``FIXED_WINDOW``: a BCI trial of
        3-5 s), the backbone is told that length instead of its pretraining
        window, or every wrapper's duration check refuses the batch ("epoch_seconds=30 s
        imply 3840 samples, but got T=512"). Elsewhere the spec is unchanged:
        sleep and epilepsy windows already follow the model, or the benchmark
        passes ``--expected-epoch-seconds``.
        """
        if not getattr(datamodule, "FIXED_WINDOW", False):
            return checkpoint_spec
        window = (getattr(datamodule, "metadata", None) or {}).get("epoch_seconds")
        if window is None:
            return checkpoint_spec
        window = float(window)
        if abs(window - float(checkpoint_spec.expected_epoch_seconds)) < 1e-9:
            return checkpoint_spec
        if verbose:
            # detail for -v and the --log file, not for every run's screen
            logging.getLogger(__name__).info(
                "%s/%s: windows are the dataset's %g s trials; the backbone is told %g s "
                "instead of its %g s", getattr(datamodule, "name", None) or "the dataset",
                checkpoint_spec.identifier, window, window,
                float(checkpoint_spec.expected_epoch_seconds))
        import dataclasses

        return dataclasses.replace(checkpoint_spec, expected_epoch_seconds=window)

    def _prepare_pair(self, dataset_name: str, dataset_config: Dict[str, Any],
                      checkpoint_spec, cmap: Optional[ChannelMap], *, load_weights: bool = True):
        """The datamodule and backbone for one (dataset, checkpoint), exactly as
        a run builds them: channel map, recording stats, sequence windowing,
        the precomputed-cache stub. `neuroatlas check` uses this too, so what
        it verifies is what a run executes.

        ``load_weights=False`` returns ``backbone=None`` -- for checking the
        data side when the weights are not on this machine.
        """
        dataset_spec = load_dataset_spec(dataset_name)
        (self.cache_root / dataset_name).mkdir(parents=True, exist_ok=True)
        # what the item's live line says meanwhile; a reader that can count
        # names its own phase (BCI: "loading the data 3/9 subjects")
        progress.current().phase("loading the data")
        datamodule = dataset_spec.create_datamodule(dataset_config, checkpoint=checkpoint_spec)
        checkpoint_spec = self._fit_checkpoint_to_window(datamodule, checkpoint_spec, verbose=True)
        if self.embed_chunk is not None:
            datamodule.embed_chunk = self.embed_chunk
        if cmap is not None:
            datamodule = _wrap_datamodule_with_channel_map(
                datamodule, cmap, checkpoint_spec.model_family
            )

        # Sequence windowing is an *extraction-time* concern: it groups raw
        # epochs into the multi-epoch context a model was pretrained on.
        # When the datamodule serves precomputed embeddings there is no
        # signal left to window, and wrapping it fails on the no-op loader
        # ("'_EmptyLoader' object has no attribute 'dataset'"). EEGBenchmarks
        # has no wrapper here at all, which is why CoRe-Sleep ran fine on the
        # precomputed brain-age datasets there.
        precomputed = _precomputed_cache_available(datamodule, checkpoint_spec)
        seq_len = getattr(checkpoint_spec, "expected_sequence_length", 1)
        if seq_len > 1 and precomputed:
            logging.getLogger(__name__).info(
                "%s/%s: no sequential wrapper (expected_sequence_length=%d): this datamodule "
                "serves precomputed embeddings", dataset_name, checkpoint_spec.identifier, seq_len)
        if seq_len > 1 and not precomputed:
            from .sequential_epochs import SequentialDataModuleWrapper
            overrides = getattr(checkpoint_spec, "runtime_overrides", {}) or {}
            stride = int(overrides.get("stride", 1))
            target_idx = overrides.get("target_idx", "all")
            emb_stride_raw = overrides.get("embedding_stride")
            emb_stride = int(emb_stride_raw) if emb_stride_raw is not None else None
            emb_target = overrides.get("embedding_target_idx")
            datamodule = SequentialDataModuleWrapper(
                datamodule,
                window_size=seq_len,
                stride=stride,
                target_idx=target_idx,
                batch_size=int(overrides.get("sequential_batch_size", 1)),
                embedding_stride=emb_stride,
                embedding_target_idx=emb_target,
            )

        # Re-check: the wrappers above can change what the datamodule
        # exposes, so this is not simply the `precomputed` value from before.
        if _precomputed_cache_available(datamodule, checkpoint_spec):
            backbone = _PrecomputedStubBackbone(checkpoint_spec)
        elif load_weights:
            backbone = _load_weights(checkpoint_spec)
        else:
            backbone = None
        if backbone is not None:
            # until the task names its own phase (reading the embeddings,
            # the extraction loop's batches): building its loader
            progress.current().phase("preparing the windows")
        return datamodule, backbone

    def _extracted_already(self, task_name: str, dataset_name: str, checkpoint_spec,
                           datamodule) -> bool:
        """Whether an extraction pass over this pair would only find its
        global cache (every window, every fold): then it needs no weights.
        The same test the task makes before it would extract (linear probe,
        seizure detection); any other task loads them as before."""
        if self.embed_chunk is not None:
            return False
        if task_name == "seizure_detection":
            from neuroatlas.extensions.tasks.seizure_detection import _uses_global_cache

            if not _uses_global_cache(datamodule):
                return False
        elif task_name == "linear_probe":
            try:
                if not datamodule.supports_global_embedding_cache():
                    return False
            except Exception:
                return False
            if getattr(datamodule, "limit_windows_per_split", None) is not None:
                return False
        else:
            return False
        from neuroatlas.extensions.tasks.linear_probe import _embedding_cache_dir

        try:
            path = _embedding_cache_dir(self.cache_root, dataset_name, checkpoint_spec, "all",
                                        datamodule, purpose="global_embeddings")
        except Exception:
            return False
        return cache_exists(Path(path))

    def _run_one(self, dataset_name: str, dataset_config: Dict[str, Any], checkpoint_spec) -> BenchmarkResult:
        # what a message said deep inside the run names it by (pair.where)
        from .pair import working_on

        with working_on(dataset_name, checkpoint_spec.identifier, dataset_config.get("fold")):
            return self._run_one_pair(dataset_name, dataset_config, checkpoint_spec)

    def _run_one_pair(self, dataset_name: str, dataset_config: Dict[str, Any],
                      checkpoint_spec) -> BenchmarkResult:
        dataset_spec = load_dataset_spec(dataset_name)

        # Opt-in channel-map layer: if src/neuroatlas/configs/channel_maps/<slug>.yaml
        # exists, apply resolved labels before the wrapper sees them.
        channel_map_name = dataset_config.pop("channel_map_name", None) or dataset_name
        cmap = load_channel_map(channel_map_name)
        task_name = self._task_name_for(dataset_spec, checkpoint_spec)
        family = checkpoint_spec.model_family
        if cmap is not None and (cmap.is_skip(family) or not cmap.has_entry(family)):
            # decided before any data is read: the map rules the pair out
            # (`skip`, with its note as the reason), or has no entry for the
            # family (`invalid`)
            state, detail = cmap.state_for(family)
            if state == "skip":
                code = RULED_OUT_CODE
                reason = " ".join(str(detail or "").split()).rstrip(".")
                message = (f"ruled out: the {dataset_name} channel map excludes the {family} "
                           f"family" + (f" ({reason[:1].lower() + reason[1:]})" if reason else ""))
            else:
                code, message = INVALID_CODE, detail
            return BenchmarkResult(
                checkpoint_id=checkpoint_spec.identifier,
                dataset_name=dataset_name,
                evaluation_mode=checkpoint_spec.evaluation_mode(dataset_name),
                failure=BenchmarkFailure(
                    code=code,
                    message=message,
                    details={
                        "channel_map_path": str(
                            Path("neuroatlas") / "configs" / "channel_maps" / f"{dataset_name}.yaml"
                        ),
                        "notes": cmap.notes.get(family, ""),
                    },
                ),
                metadata={
                    "task_name": task_name,
                    "spec": checkpoint_spec.to_dict(),
                    "channel_map_applied": False,
                    **dataset_config,
                },
            )

        dataset_config = self._pair_config(dataset_config, cmap, checkpoint_spec)

        try:
            effective_extract_only = self.extract_only or self.embed_chunk is not None
            reuse = not (self.reprobe or effective_extract_only)
            # Weights only once the pair is known to need them: a fold whose
            # saved predictions are reused, an extraction that finds its
            # cache, and a probe (or --reprobe) of embeddings already
            # extracted never load the backbone.
            datamodule, backbone = self._prepare_pair(
                dataset_name, dataset_config, checkpoint_spec, cmap, load_weights=False)
            # The spec the backbone was built from (results record it).
            checkpoint_spec = self._fit_checkpoint_to_window(datamodule, checkpoint_spec)
            probe_dir = self._probe_dir(dataset_name, checkpoint_spec.identifier, datamodule)
            if reuse:
                saved = self._saved_result(dataset_name, checkpoint_spec, datamodule,
                                           task_name, probe_dir)
                if saved is not None:
                    if cmap is not None:
                        saved.metadata.setdefault("channel_map_applied", True)
                    return saved
            if backbone is None:
                if effective_extract_only and self._extracted_already(
                        task_name, dataset_name, checkpoint_spec, datamodule):
                    backbone = _PrecomputedStubBackbone(checkpoint_spec)
                elif effective_extract_only:
                    backbone = _load_weights(checkpoint_spec)
                else:
                    # probing: the weights only if the task has to extract
                    backbone = _LazyBackbone(checkpoint_spec, _load_weights)
                progress.current().phase("preparing the windows")
            from neuroatlas.extensions.tasks.linear_probe import (
                require_cached_embeddings,
            )

            with require_cached_embeddings(
                self.require_cached_embeddings and not effective_extract_only
            ):
                result = evaluate_task(
                    task_name,
                    dataset_name=dataset_name,
                    dataset_spec=dataset_spec,
                    dataset_config=dataset_config,
                    checkpoint_spec=checkpoint_spec,
                    datamodule=datamodule,
                    backbone=backbone,
                    runner=self,
                    probe_config=self._probe_config(),
                    task_config=self._task_config(),
                    seeds=self._effective_seeds(datamodule),
                    probe_dir=probe_dir,
                    cache_root=self.cache_root,
                    extract_only=effective_extract_only,
                    embed_chunk=self.embed_chunk,
                )
            if cmap is not None:
                result.metadata.setdefault("channel_map_applied", True)
            if not effective_extract_only:
                self._remember(result, probe_dir)
            return result
        except Exception as exc:
            from neuroatlas import quiet
            from neuroatlas.cli import _msg

            # The pair failed and the run goes on: one line now, its first
            # sentence (-v and --log: the whole message; results.json keeps it).
            fold = dataset_config.get("fold")
            pair = f"{dataset_name}/{checkpoint_spec.identifier}" + (
                f" fold {fold}" if fold is not None else "")
            code, message = "runtime_failure", _exception_text(exc)
            if isinstance(exc, WeightsMissing):
                code, message = WEIGHTS_MISSING_CODE, str(exc)
            else:
                # a dataset that is not here fails every run on it, whatever
                # the reader raised: say that, once, with what brings it
                missing = data_missing_text(dataset_name)
                if missing:
                    code, message = DATA_MISSING_CODE, missing
            if code == "runtime_failure" or quiet.once(f"{code}:{dataset_name}"
                                                       if code == DATA_MISSING_CODE
                                                       else f"{code}:{pair}"):
                _msg.error(f"{pair}: " + _msg.brief(message))
            else:
                logging.getLogger(__name__).info("%s: %s", pair, message)
            # The traceback is for a bug report: the --log file, or -v on screen.
            logging.getLogger("neuroatlas.traceback").debug(
                "%s/%s failed", dataset_name, checkpoint_spec.identifier, exc_info=True)
            return BenchmarkResult(
                checkpoint_id=checkpoint_spec.identifier,
                dataset_name=dataset_name,
                evaluation_mode=checkpoint_spec.evaluation_mode(dataset_name),
                failure=BenchmarkFailure(
                    code=code,
                    message=message if code != "runtime_failure" else str(exc),
                    details={
                        "checkpoint_status": checkpoint_spec.status,
                        "wrapper_name": checkpoint_spec.wrapper_name,
                        **({"exception": _exception_text(exc)}
                           if code != "runtime_failure" else {}),
                    },
                ),
                metadata={"task_name": task_name, "spec": checkpoint_spec.to_dict(), **dataset_config},
            )

    def run(self) -> List[BenchmarkResult]:
        from .. import ensure_dataloader_sharing_strategy

        ensure_dataloader_sharing_strategy()

        self.output_root.mkdir(parents=True, exist_ok=True)
        results: List[BenchmarkResult] = []
        selected_models = self._selected_models()
        specs = [self._apply_spec_overrides(spec) for spec in load_checkpoint_registry(selected_models)]
        runs = list(self._dataset_runs())
        if self.extract_only or self.embed_chunk is not None:
            return self._extract(runs, specs)
        total, done = len(runs) * len(specs), 0
        if total:
            datasets = sorted({name for name, _ in runs})
            progress.say(f"probing {_count(len(specs), 'checkpoint')} on {', '.join(datasets)}: "
                         f"{_count(total, 'run')} (checkpoint x fold), one line each as it "
                         f"finishes")
        for dataset_name, dataset_config in runs:
            for spec in specs:
                started = time.monotonic()
                fold = dataset_config.get("fold")
                label = (f"[{done + 1}/{total}] {dataset_name} {spec.identifier}"
                         + ("" if fold is None else f" fold {fold}"))
                # a live line through the fold's phases on a terminal; off
                # one, its result line alone
                with _Ticker(label):
                    result = self._run_one(dataset_name, dataset_config, spec)
                results.append(result)
                # One line per probed fold, whatever the log level: a probe
                # over many models runs for hours with nothing else to show.
                done += 1
                progress.say(_progress_line(done, total, result, time.monotonic() - started))
        self._write_outputs(results)
        return results

    def _extract(self, runs: List[tuple], specs) -> List[BenchmarkResult]:
        """The extraction pass (`embed`): one item per (dataset, model), its
        folds inside it, a live line through its phases and one result line.

        An extraction over several folds (`embed --folds`, which `run` passes
        so per-split cohorts get every fold the probe reads) is done for a
        (dataset, model) once one fold reports a fold-independent cache: the
        other folds would only rebuild the datamodule to find it.
        """
        results: List[BenchmarkResult] = []
        by_dataset: Dict[str, List[Dict[str, Any]]] = {}
        for dataset_name, dataset_config in runs:
            by_dataset.setdefault(dataset_name, []).append(dataset_config)
        total, done = len(by_dataset) * len(specs), 0
        if total:
            chunk = (f", subject chunk {self.embed_chunk[0]}/{self.embed_chunk[1]}"
                     if self.embed_chunk is not None else "")
            progress.say(f"embedding {_count(len(specs), 'checkpoint')} on {', '.join(by_dataset)}"
                         f"{chunk}, one line each as it finishes")
        for dataset_name, configs in by_dataset.items():
            for spec in specs:
                done += 1
                where = f"{dataset_name} {spec.identifier}"
                pair_results: List[BenchmarkResult] = []
                with progress.Progress(f"[{done}/{total}] {where}", verb="embedding") as item:
                    for i, dataset_config in enumerate(configs):
                        fold = dataset_config.get("fold")
                        if i and fold is not None:
                            # the first fold's cache serves only that fold
                            # (a per-split cohort): name the fold from here on
                            item.relabel(f"[{done}/{total}] {where} fold {fold}")
                        result = self._run_one(dataset_name, dataset_config, spec)
                        pair_results.append(result)
                        if _serves_every_fold(result):
                            break
                results.extend(pair_results)
                outcome = _embed_outcome(pair_results, item.counts)
                # a cached pair's time only when it was not a moment: loading
                # a BCI cohort to find its cache takes minutes
                seconds = item.seconds
                if outcome == ALREADY_CACHED and seconds <= _CACHED_SHOWN_AFTER:
                    seconds = None
                progress.say(progress.result_line(done, total, where, outcome, seconds))
        return results

    def _result_key(self, result: BenchmarkResult) -> tuple:
        task_name = result.metadata.get("task_name", result.evaluation_mode)
        aggregation = result.metadata.get("aggregation", "")
        mode = result.metadata.get("mode", "")
        return (
            result.dataset_name,
            result.checkpoint_id,
            task_name,
            result.evaluation_mode,
            result.metadata.get("fold", ""),
            aggregation,
            mode,
        )

    @staticmethod
    def _attempt_key(result: BenchmarkResult) -> tuple:
        """(dataset, checkpoint, task, fold): what one attempt at a fold is,
        whatever evaluation mode or aggregation its record ended up with."""
        fold = result.metadata.get("fold")
        return (
            result.dataset_name,
            result.checkpoint_id,
            str(result.metadata.get("task_name", result.evaluation_mode)),
            "" if fold is None else str(fold),
        )

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = BenchmarkRunner._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    def _merge_with_existing(self, new_results: List[BenchmarkResult]) -> List[BenchmarkResult]:
        existing_path = self.output_root / "results.json"
        if not existing_path.exists():
            return new_results
        try:
            existing_dicts = json.loads(existing_path.read_text(encoding="utf-8"))
            existing = [BenchmarkResult.from_dict(d) for d in existing_dicts]
        except Exception:
            return new_results
        existing_by_key = {self._result_key(r): r for r in existing}
        for r in new_results:
            key = self._result_key(r)
            old = existing_by_key.get(key)
            if old is not None and old.metrics and r.metrics:
                r.metrics = self._deep_merge(old.metrics, r.metrics)
            if old is not None:
                for field in ("thresholds_seconds", "event_fields"):
                    old_vals = old.metadata.get(field, [])
                    new_vals = r.metadata.get(field, [])
                    if old_vals or new_vals:
                        r.metadata[field] = sorted(set(old_vals) | set(new_vals))
        new_keys = {self._result_key(r) for r in new_results}
        # An earlier attempt's failure is superseded by any new result for the
        # same checkpoint, task and fold, even when it was recorded under a
        # different evaluation mode or aggregation (a failure is written before
        # the task knows them), and a failure recorded before any fold by any
        # new result for that checkpoint and task. Otherwise the failed rows
        # stay next to the new ones and `results` counts both.
        retried = {self._attempt_key(r) for r in new_results}
        retried_any_fold = {k[:3] for k in retried}

        def _superseded(r: BenchmarkResult) -> bool:
            if r.ok:
                return False
            key = self._attempt_key(r)
            return key in retried or (key[3] == "" and key[:3] in retried_any_fold)

        merged = [r for r in existing
                  if self._result_key(r) not in new_keys and not _superseded(r)] + new_results
        merged.sort(
            key=lambda r: (
                r.dataset_name,
                r.checkpoint_id,
                str(r.metadata.get("task_name", r.evaluation_mode)),
                str(r.metadata.get("aggregation", "")),
                str(r.metadata.get("mode", "")),
                str(r.metadata.get("fold", "")),
            )
        )
        return merged

    def _write_outputs(self, results: Iterable[BenchmarkResult]) -> None:
        with results_lock(self.output_root):
            self._write_outputs_unlocked(results)

    def _write_outputs_unlocked(self, results: Iterable[BenchmarkResult]) -> None:
        results = list(results)
        # When each row was written: tells a re-run's row from an identical
        # one an earlier run left (`neuroatlas run` counts only its own).
        written_at = datetime.now(timezone.utc).isoformat()
        for r in results:
            r.metadata["written_at"] = written_at
        results = self._merge_with_existing(results)
        with open(self.output_root / "results.json", "w", encoding="utf-8") as handle:
            json.dump([result.to_dict() for result in results], handle, indent=2)
        self.write_tables(results)

    #: Tables an earlier version of this runner wrote beside results.json, by
    #: their first line. `neuroatlas results` summarises results.json; these
    #: had no column for most benchmarks' headline and their ± was another SD.
    _OLD_TABLES = {
        "results.csv": "dataset,task,mode,aggregation,fold,checkpoint_id,evaluation_mode,status,",
        "results.md": "| Dataset | Task | Mode | Aggregation | Fold | Checkpoint | Eval Mode |",
        "summary.md": "| Dataset | Task | Mode | Aggregation | Fold | Checkpoint | Eval Mode |",
    }

    def write_tables(self, results: List[BenchmarkResult]) -> None:
        """results.json is the folder's one record; `neuroatlas results`
        summarises it. No table is written beside it, and one this runner
        once wrote there is removed when results.json is rewritten (it would
        no longer match it). *results* is not read."""
        for name, head in self._OLD_TABLES.items():
            path = self.output_root / name
            try:
                with open(path, encoding="utf-8") as handle:
                    first = handle.readline()
            except OSError:
                continue
            if first.startswith(head):
                try:
                    path.unlink()
                except OSError:
                    pass
