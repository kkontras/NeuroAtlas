from __future__ import annotations

import contextlib
import fcntl
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


def _Ticker(label: str, every: float = 2.0, *, verb: str = "probing", stream=None,
            start_line_off_tty: bool = True) -> progress.Progress:
    """Shows that a run is alive while it works. On a terminal: one line,
    "<label>: probing (1m 05s)", rewritten every few seconds and cleared when
    the run ends (its result line follows). Elsewhere (a log, a cluster job):
    one "<label>: probing" line at the start. (neuroatlas.progress, with the
    line saying only the verb.)"""
    return progress.Progress(label, verb=verb, every=every, stream=stream,
                             start_line_off_tty=start_line_off_tty, phases=False)


# The metric a progress line shows for a fold, first one present.
_PROGRESS_METRICS = ("event_sens_fa_auc", "auroc", "balanced_accuracy", "mae", "pearson_r")


def _progress_line(done: int, total: int, result, seconds: float) -> str:
    fold = (result.metadata or {}).get("fold")
    where = f"{result.dataset_name} {result.checkpoint_id}" + ("" if fold is None else f" fold {fold}")
    if not result.ok:
        what = "failed"
    else:
        value = next(((m, (result.metrics or {}).get(m)) for m in _PROGRESS_METRICS
                      if isinstance((result.metrics or {}).get(m), (int, float))), None)
        # (reused): recomputed from the fold's saved predictions, not fitted
        what = "ok" + (" (reused)" if (result.metadata or {}).get("reused_predictions") else "") \
            + (f", {value[0]} {value[1]:.3f}" if value else "")
    return progress.result_line(done, total, where, what, seconds)


def _embed_outcome(results: List[BenchmarkResult], counts: Dict[str, int]) -> str:
    """What an extraction pass over one (dataset, model) did, for its result
    line: ``ok, 50,749 windows``, ``cached (already extracted)``, ``failed``."""
    if not results or any(not r.ok for r in results):
        return "failed"
    if counts.get("extracted"):
        windows = counts.get("windows")
        return f"ok, {windows:,} windows" if windows else "ok"
    return "cached (already extracted)"


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


def _metric_scalar(metrics: Dict[str, Any], name: str):
    value = metrics.get(name)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def _format_metric(value, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _format_f1_per_class(metrics: Dict[str, Any], digits: int = 4) -> str:
    values = metrics.get("f1_per_class")
    if not values:
        return ""
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in values) + "]"


def _select_val_metrics(result: BenchmarkResult) -> Dict[str, Any]:
    if "best_val" in result.metrics:
        return result.metrics["best_val"]
    return result.metrics.get("val", {})


def _select_test_metrics(result: BenchmarkResult) -> Dict[str, Any]:
    if "best_test" in result.metrics:
        return result.metrics["best_test"]
    return result.metrics.get("test", result.metrics)


def _mean_metric(rows: List[Dict[str, Any]], name: str):
    values = [_metric_scalar(row, name) for row in rows]
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def _mean_f1_per_class(rows: List[Dict[str, Any]]):
    series = [row.get("f1_per_class") for row in rows if row.get("f1_per_class")]
    if not series:
        return []
    min_len = min(len(values) for values in series)
    arr = np.asarray([values[:min_len] for values in series], dtype=float)
    return arr.mean(axis=0).tolist()


def _std_metric(rows: List[Dict[str, Any]], name: str):
    values = [_metric_scalar(row, name) for row in rows]
    values = [float(value) for value in values if value is not None]
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1))


def _std_f1_per_class(rows: List[Dict[str, Any]]):
    series = [row.get("f1_per_class") for row in rows if row.get("f1_per_class")]
    if len(series) < 2:
        return []
    min_len = min(len(values) for values in series)
    arr = np.asarray([values[:min_len] for values in series], dtype=float)
    return arr.std(axis=0, ddof=1).tolist()


def _format_mean_std(mean, std, digits: int = 4) -> str:
    if mean is None:
        return ""
    if std is None:
        return _format_metric(mean, digits)
    return f"{float(mean):.{digits}f}±{float(std):.{digits}f}"


def _format_mean_std_f1_per_class(mean_values, std_values, digits: int = 4) -> str:
    if not mean_values:
        return ""
    if not std_values or len(std_values) != len(mean_values):
        return "[" + ", ".join(f"{float(v):.{digits}f}" for v in mean_values) + "]"
    pairs = ", ".join(f"{float(m):.{digits}f}±{float(s):.{digits}f}" for m, s in zip(mean_values, std_values))
    return "[" + pairs + "]"


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
            why = None if sidecar.is_file() else "no result.json beside them (an older version saved them)"
            why = why or preds.reuse_problem(
                pred, task=task_name, seeds=self._effective_seeds(datamodule),
                pooling=current_pooling(), checkpoint_path=checkpoint_spec.checkpoint_path,
                cache_root=self.cache_root)
            if why is None:
                row = BenchmarkResult.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))
                if (row.dataset_name, row.checkpoint_id) != (dataset_name, checkpoint_spec.identifier):
                    why = "their result.json is another dataset's or model's"
        except Exception as exc:  # unreadable: probe again, and say so
            why = f"they cannot be read ({type(exc).__name__}: {exc})"
        if why is not None:
            from neuroatlas.cli import _msg

            _msg.note(f"{where}: saved predictions not reused, probing again: {why}")
            return None
        row.metrics = preds.scorer(task_name)(pred)
        preds.refresh_metadata(row.metadata, pred)
        row.cache_paths["predictions"] = str(path)
        row.metadata.pop("written_at", None)
        row.metadata["reused_predictions"] = True
        # the fold's progress line says "ok (reused)"; the file, for -v and --log
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

                _msg.warning(f"{result.dataset_name}/{result.checkpoint_id}: could not keep "
                             f"result.json beside the predictions ({exc}); a later run fits "
                             f"this fold again")

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
                "instead of its %g s", getattr(datamodule, "name", "?"),
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
        # what an `embed` item's live line says meanwhile (a probe's keeps "probing")
        progress.current().phase("indexing recordings")
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
            progress.current().phase("loading weights")
            backbone = load_backbone(checkpoint_spec)
        else:
            backbone = None
        # until the extraction loop counts batches: building its loader, or
        # finding the cache already there
        progress.current().phase("preparing the windows")
        return datamodule, backbone

    def _run_one(self, dataset_name: str, dataset_config: Dict[str, Any], checkpoint_spec) -> BenchmarkResult:
        dataset_spec = load_dataset_spec(dataset_name)

        # Opt-in channel-map layer: if src/neuroatlas/configs/channel_maps/<slug>.yaml
        # exists, apply resolved labels before the wrapper sees them.
        channel_map_name = dataset_config.pop("channel_map_name", None) or dataset_name
        cmap = load_channel_map(channel_map_name)
        task_name = self._task_name_for(dataset_spec, checkpoint_spec)
        if cmap is not None and cmap.is_skip(checkpoint_spec.model_family):
            return BenchmarkResult(
                checkpoint_id=checkpoint_spec.identifier,
                dataset_name=dataset_name,
                evaluation_mode=checkpoint_spec.evaluation_mode(dataset_name),
                failure=BenchmarkFailure(
                    code="channel_map_skip",
                    message=(
                        f"({dataset_name!r}, {checkpoint_spec.model_family!r}) "
                        f"explicitly skipped by src/neuroatlas/configs/channel_maps/"
                        f"{dataset_name}.yaml; remove from benchmark matrix "
                        f"or remove the skip marker."
                    ),
                    details={
                        "channel_map_path": str(
                            Path("neuroatlas") / "configs" / "channel_maps" / f"{dataset_name}.yaml"
                        ),
                        "notes": cmap.notes.get(checkpoint_spec.model_family, ""),
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
            # Weights only once the fold is known to need probing: a fold
            # whose saved predictions are reused never loads its backbone.
            datamodule, backbone = self._prepare_pair(
                dataset_name, dataset_config, checkpoint_spec, cmap, load_weights=not reuse)
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
                    backbone = load_backbone(checkpoint_spec)
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
            from neuroatlas.cli import _msg

            # The pair failed and the run goes on: one line now, its first
            # sentence (-v and --log: the whole message; results.json keeps it).
            _msg.error(f"{dataset_name}/{checkpoint_spec.identifier} "
                       f"(fold {dataset_config.get('fold', '?')}): "
                       + _msg.brief(_msg.exception_text(exc)))
            # The traceback is for a bug report: the --log file, or -v on screen.
            logging.getLogger("neuroatlas.traceback").debug(
                "%s/%s failed", dataset_name, checkpoint_spec.identifier, exc_info=True)
            return BenchmarkResult(
                checkpoint_id=checkpoint_spec.identifier,
                dataset_name=dataset_name,
                evaluation_mode=checkpoint_spec.evaluation_mode(dataset_name),
                failure=BenchmarkFailure(
                    code="runtime_failure",
                    message=str(exc),
                    details={
                        "checkpoint_status": checkpoint_spec.status,
                        "wrapper_name": checkpoint_spec.wrapper_name,
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
            print(f"probing {len(specs)} model(s) on {', '.join(datasets)}: {total} run(s), "
                  f"one line each as it finishes", flush=True)
        for dataset_name, dataset_config in runs:
            for spec in specs:
                started = time.monotonic()
                fold = dataset_config.get("fold")
                label = (f"[{done + 1}/{total}] {dataset_name} {spec.identifier}"
                         + ("" if fold is None else f" fold {fold}"))
                with _Ticker(label):
                    result = self._run_one(dataset_name, dataset_config, spec)
                results.append(result)
                # One line per probed fold, whatever the log level: a probe
                # over many models runs for hours with nothing else to show.
                done += 1
                print(_progress_line(done, total, result, time.monotonic() - started),
                      flush=True)
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
            print(f"embedding {len(specs)} model(s) on {', '.join(by_dataset)}: "
                  f"{total} run(s){chunk}", flush=True)
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
                seconds = None if outcome.startswith("cached") else item.seconds
                print(progress.result_line(done, total, where, outcome, seconds), flush=True)
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

    def write_tables(self, results: List[BenchmarkResult]) -> None:
        """results.csv, summary.md and results.md from the rows of results.json
        (`neuroatlas rescore` rewrites them after changing the metrics)."""
        csv_rows = [
            "dataset,task,mode,aggregation,fold,checkpoint_id,evaluation_mode,status,"
            "val_accuracy,val_macro_f1,val_weighted_f1,val_cohen_kappa,val_f1_per_class,"
            "test_accuracy,test_macro_f1,test_weighted_f1,test_cohen_kappa,test_f1_per_class"
        ]
        for result in results:
            val_metrics = _select_val_metrics(result)
            test_metrics = _select_test_metrics(result)
            csv_rows.append(
                ",".join([
                    result.dataset_name,
                    str(result.metadata.get("task_name", result.evaluation_mode)),
                    str(result.metadata.get("mode", "")),
                    str(result.metadata.get("aggregation", "")),
                    str(result.metadata.get("fold", "")),
                    result.checkpoint_id,
                    result.evaluation_mode,
                    "ok" if result.ok else "failed",
                    _format_metric(_metric_scalar(val_metrics, "accuracy"), digits=6),
                    _format_metric(_metric_scalar(val_metrics, "macro_f1"), digits=6),
                    _format_metric(_metric_scalar(val_metrics, "weighted_f1"), digits=6),
                    _format_metric(_metric_scalar(val_metrics, "cohen_kappa"), digits=6),
                    '"' + _format_f1_per_class(val_metrics, digits=6) + '"',
                    _format_metric(_metric_scalar(test_metrics, "accuracy"), digits=6),
                    _format_metric(_metric_scalar(test_metrics, "macro_f1"), digits=6),
                    _format_metric(_metric_scalar(test_metrics, "weighted_f1"), digits=6),
                    _format_metric(_metric_scalar(test_metrics, "cohen_kappa"), digits=6),
                    '"' + _format_f1_per_class(test_metrics, digits=6) + '"',
                ])
            )
        (self.output_root / "results.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")

        lines = [
            "| Dataset | Task | Mode | Aggregation | Fold | Checkpoint | Eval Mode | Status | Val Acc | Val Macro-F1 | Val Weighted-F1 | Val Kappa | Val F1/Class | Test Acc | Test Macro-F1 | Test Weighted-F1 | Test Kappa | Test F1/Class |",
            "|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
        ]
        for result in results:
            val_metrics = _select_val_metrics(result)
            test_metrics = _select_test_metrics(result)
            lines.append(
                f"| {result.dataset_name} | {result.metadata.get('task_name', result.evaluation_mode)} | "
                f"{result.metadata.get('mode', '')} | {result.metadata.get('aggregation', '')} | "
                f"{result.metadata.get('fold', '')} | {result.checkpoint_id} | {result.evaluation_mode} | "
                f"{'ok' if result.ok else 'failed'} | "
                f"{_format_metric(_metric_scalar(val_metrics, 'accuracy'))} | "
                f"{_format_metric(_metric_scalar(val_metrics, 'macro_f1'))} | "
                f"{_format_metric(_metric_scalar(val_metrics, 'weighted_f1'))} | "
                f"{_format_metric(_metric_scalar(val_metrics, 'cohen_kappa'))} | "
                f"{_format_f1_per_class(val_metrics)} | "
                f"{_format_metric(_metric_scalar(test_metrics, 'accuracy'))} | "
                f"{_format_metric(_metric_scalar(test_metrics, 'macro_f1'))} | "
                f"{_format_metric(_metric_scalar(test_metrics, 'weighted_f1'))} | "
                f"{_format_metric(_metric_scalar(test_metrics, 'cohen_kappa'))} | "
                f"{_format_f1_per_class(test_metrics)} |"
            )

        grouped: Dict[tuple[str, str, str, str, str], List[BenchmarkResult]] = {}
        for result in results:
            key = (
                result.dataset_name,
                str(result.metadata.get("task_name", result.evaluation_mode)),
                str(result.metadata.get("mode", "")),
                str(result.metadata.get("aggregation", "")),
                result.checkpoint_id,
            )
            grouped.setdefault(key, []).append(result)

        if grouped:
            lines.extend(["", "## Fold Averages (mean±std)", ""])
            lines.extend([
                "| Dataset | Task | Mode | Aggregation | Folds | Checkpoint | Eval Mode | Status | Val Acc | Val Macro-F1 | Val Weighted-F1 | Val Kappa | Val F1/Class | Test Acc | Test Macro-F1 | Test Weighted-F1 | Test Kappa | Test F1/Class |",
                "|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
            ])
            for (dataset_name, task_name, mode, aggregation, checkpoint_id), group in sorted(grouped.items()):
                ok_group = [result for result in group if result.ok]
                evaluation_mode = group[0].evaluation_mode
                if not ok_group:
                    lines.append(
                        f"| {dataset_name} | {task_name} | {mode} | {aggregation} | — | {checkpoint_id} | {evaluation_mode} | failed |  |  |  |  |  |  |  |  |  |  |"
                    )
                    continue
                val_rows = [_select_val_metrics(result) for result in ok_group]
                test_rows = [_select_test_metrics(result) for result in ok_group]
                fold_ids = [str(result.metadata.get("fold", "?")) for result in ok_group]
                avg_val = {m: _mean_metric(val_rows, m) for m in ("accuracy", "macro_f1", "weighted_f1", "cohen_kappa")}
                std_val = {m: _std_metric(val_rows, m) for m in ("accuracy", "macro_f1", "weighted_f1", "cohen_kappa")}
                avg_val["f1_per_class"] = _mean_f1_per_class(val_rows)
                std_val["f1_per_class"] = _std_f1_per_class(val_rows)
                avg_test = {m: _mean_metric(test_rows, m) for m in ("accuracy", "macro_f1", "weighted_f1", "cohen_kappa")}
                std_test = {m: _std_metric(test_rows, m) for m in ("accuracy", "macro_f1", "weighted_f1", "cohen_kappa")}
                avg_test["f1_per_class"] = _mean_f1_per_class(test_rows)
                std_test["f1_per_class"] = _std_f1_per_class(test_rows)
                lines.append(
                    f"| {dataset_name} | {task_name} | {mode} | {aggregation} | {','.join(fold_ids)} | {checkpoint_id} | {evaluation_mode} | ok ({len(ok_group)} folds) | "
                    f"{_format_mean_std(avg_val['accuracy'], std_val['accuracy'])} | "
                    f"{_format_mean_std(avg_val['macro_f1'], std_val['macro_f1'])} | "
                    f"{_format_mean_std(avg_val['weighted_f1'], std_val['weighted_f1'])} | "
                    f"{_format_mean_std(avg_val['cohen_kappa'], std_val['cohen_kappa'])} | "
                    f"{_format_mean_std_f1_per_class(avg_val['f1_per_class'], std_val['f1_per_class'])} | "
                    f"{_format_mean_std(avg_test['accuracy'], std_test['accuracy'])} | "
                    f"{_format_mean_std(avg_test['macro_f1'], std_test['macro_f1'])} | "
                    f"{_format_mean_std(avg_test['weighted_f1'], std_test['weighted_f1'])} | "
                    f"{_format_mean_std(avg_test['cohen_kappa'], std_test['cohen_kappa'])} | "
                    f"{_format_mean_std_f1_per_class(avg_test['f1_per_class'], std_test['f1_per_class'])} |"
                )
        (self.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (self.output_root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
