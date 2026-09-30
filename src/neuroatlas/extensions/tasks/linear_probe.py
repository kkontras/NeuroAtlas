from __future__ import annotations

import contextlib
import copy
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from neuroatlas.benchmarking_helpers import BenchmarkResult, EmbeddingPayload, TaskSpec
from neuroatlas.benchmarking_helpers.runtime.cache import (
    IncrementalEmbeddingWriter,
    build_cache_key,
    cache_exists,
    load_embedding_payload,
    merge_embedding_chunks,
    save_embedding_payload,
    save_probe_payload,
)
from neuroatlas.benchmarking_helpers.probes.probe import train_probe

# Provisional fix: route precomputed-embedding datamodules to their external
# cache before falling back to the internal cache / backbone extraction.
# Without this, runner.py substitutes a stub backbone (no GPU needed) but
# _extract_or_load_embeddings never finds the external cache, writes an
# empty features.npy, and crashes.  The permanent fix is to always check
# datamodule.precomputed_embedding_cache_dir (as EEGBenchmarks does);
# set this flag to False to revert to the old behaviour for debugging.
_PRECOMPUTED_CACHE_FIX = True


# --------------------------------------------------------------------------
# embed / probe separation
# --------------------------------------------------------------------------
# The pipeline is two commands: `embed` turns raw recordings into embeddings
# for a (dataset, model), `probe` fits a task on those embeddings.  They share
# this function, which is why the split needs enforcing rather than assuming:
# without it a probe job that misses the cache quietly loads a backbone and
# extracts, so a CPU-only allocation dies on a GPU-shaped workload, or worse,
# succeeds slowly and hides the fact that nothing had been embedded.
#
# A module-level switch rather than a keyword argument: there are 36 call
# sites across 8 task modules, and this is a property of the *run*, not of any
# one call.  The runner owns it, and sets it from the benchmark config.
_REQUIRE_CACHED_EMBEDDINGS = False


@contextlib.contextmanager
def require_cached_embeddings(active: bool = True):
    """Within this block, a cache miss is an error instead of an extraction."""
    global _REQUIRE_CACHED_EMBEDDINGS
    previous = _REQUIRE_CACHED_EMBEDDINGS
    _REQUIRE_CACHED_EMBEDDINGS = bool(active)
    try:
        yield
    finally:
        _REQUIRE_CACHED_EMBEDDINGS = previous


class EmbeddingsNotFound(FileNotFoundError):
    """No embeddings for this (dataset, model, split), and probing may not make them."""


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------
# Both settings describe what happens *inside one window* -- the window itself
# is set by --set window_s/stride_s and does not change here. A backbone cuts
# a window into patch tokens; the question is what it hands back for that
# window:
#
#   mean       one vector per window, averaged over that window's tokens
#   per_patch  that window's tokens, kept separate: (n_tokens, token_dim)
#              instead of (token_dim,)
#
# Which one a run wants is a property of the run, so it rides the same
# module-level switch as the embed/probe split above.
#
# The two land in different cache cells (see ``_cache_spec``), so `embed
# --pooling per_patch` and `embed --pooling mean` do not overwrite each other
# and `probe --pooling ...` picks the one it wants.
POOLINGS = ("mean", "per_patch")

_POOLING = "mean"


@contextlib.contextmanager
def embedding_pooling(mode: str = "mean"):
    """Within this block, extract and look up embeddings pooled *mode*-wise."""
    if mode not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {mode!r}")
    global _POOLING
    previous = _POOLING
    _POOLING = mode
    try:
        yield
    finally:
        _POOLING = previous


_MAX_BATCHES: Optional[int] = None


@contextlib.contextmanager
def limit_batches(n: Optional[int]):
    """Within this block, stop each extraction after *n* batches.

    For smoke tests only. The cache written is truncated, so the caller must
    point the run at a cache root of its own -- `neuroatlas run
    --limit-batches` uses <cache_root>/_limited -- or a later full run would
    read the truncated cache as complete.
    """
    global _MAX_BATCHES
    previous = _MAX_BATCHES
    _MAX_BATCHES = None if n is None else int(n)
    try:
        yield
    finally:
        _MAX_BATCHES = previous


def current_pooling() -> str:
    """The pooling this run is extracting and reading."""
    return _POOLING


class PoolingNotSupported(NotImplementedError):
    """This backbone cannot hand back per-patch tokens."""


def _sanitize_meta(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        elif isinstance(v, dict):
            out[k] = {
                dk: (dv.item() if isinstance(dv, (np.integer, np.floating)) else dv)
                for dk, dv in v.items()
            }
        else:
            out[k] = v
    return out


class _OffsetSequentialSampler(Sampler):
    """Yield indices ``offset..len(dataset)-1`` in order.

    Used during resume (R4) to short-circuit already-extracted samples so
    the DataLoader never calls ``__getitem__`` on them — avoiding the
    expensive EDF/H5 re-reads the legacy fast path used to incur.
    """

    def __init__(self, data_source, offset: int) -> None:
        self.data_source = data_source
        self.offset = int(offset)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.offset, len(self.data_source)))

    def __len__(self) -> int:
        return max(0, len(self.data_source) - self.offset)


def _eviction_key(dataset, meta: Dict[str, object]) -> Optional[str]:
    """Dataset-correct identity for eviction.

    WSC keys its signal cache by ``recording_id``; MASS/DCSM/DOD/UCDDB key
    by ``subject_id``. Falls back to a best-effort lookup for datasets that
    don't yet implement ``eviction_key``.
    """
    if hasattr(dataset, "eviction_key"):
        return dataset.eviction_key(meta)
    sid = meta.get("subject_id")
    if sid is not None:
        return str(sid)
    rid = meta.get("recording_id")
    return str(rid) if rid is not None else None


def _flatten_labels(batch) -> np.ndarray:
    labels = batch["label"]
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels).reshape(-1)


def _dataset_context(datamodule, purpose: str = "default") -> Dict[str, object]:
    if hasattr(datamodule, "cache_context"):
        return datamodule.cache_context(purpose=purpose)
    return dict(getattr(datamodule, "metadata", {}))


def _cache_spec(dataset_name: str, checkpoint_spec, split_name: str, datamodule, purpose: str = "default") -> Dict[str, object]:
    parts = {
        "dataset": dataset_name,
        "split": split_name,
        "checkpoint_id": checkpoint_spec.identifier,
        "embedding_key": checkpoint_spec.embedding_key,
        "input_kind": checkpoint_spec.input_kind,
        "channels": list(checkpoint_spec.expected_channels),
        "sampling_rate": checkpoint_spec.expected_sampling_rate,
    }
    parts.update(_dataset_context(datamodule, purpose=purpose))
    # Only when it is not the default: adding a key unconditionally would
    # change every hash and orphan every embedding cached before pooling
    # was a choice.
    if _POOLING != "mean":
        parts["pooling"] = _POOLING
    return parts


def _embedding_cache_dir(cache_root: Path, dataset_name: str, checkpoint_spec, split_name: str, datamodule, purpose: str = "default") -> Path:
    key = build_cache_key(_cache_spec(dataset_name, checkpoint_spec, split_name, datamodule, purpose=purpose))
    return cache_root / dataset_name / checkpoint_spec.identifier / split_name / key


def _split_dataset_by_subject(dataset, n_splits: int) -> List:
    """Return *n_splits* shallow copies of *dataset*, each with a disjoint subject slice."""
    if hasattr(dataset, "_ensure_index_built"):
        dataset._ensure_index_built()
    index = dataset._index
    # Find subject boundaries.
    boundaries = [0]
    prev = None
    for i, entry in enumerate(index):
        if entry[0] != prev:
            if prev is not None:
                boundaries.append(i)
            prev = entry[0]
    boundaries.append(len(index))
    n_subjects = len(boundaries) - 1
    splits = []
    for w in range(n_splits):
        start_s = w * n_subjects // n_splits
        end_s = (w + 1) * n_subjects // n_splits
        start_i = boundaries[start_s]
        end_i = boundaries[end_s]
        ds = copy.copy(dataset)
        ds._index = index[start_i:end_i]
        for attr in ("_edf_cache", "_h5_cache"):
            if hasattr(ds, attr):
                setattr(ds, attr, {})
        splits.append(ds)
    return splits


def _sequential_round_robin_iterator(loaders: Sequence[DataLoader]) -> Iterator:
    """Yield ``(loader_idx, batch)`` in strict round-robin order, single-threaded."""
    iters = [iter(loader) for loader in loaders]
    sentinel = object()
    active = list(range(len(iters)))
    while active:
        next_active = []
        for i in active:
            batch = next(iters[i], sentinel)
            if batch is sentinel:
                continue
            yield i, batch
            next_active.append(i)
        active = next_active


def _extract(backbone, batch):
    """One batch of embeddings, pooled the way this run asked for."""
    if _POOLING == "per_patch":
        fn = getattr(backbone, "extract_embeddings_perpatch", None)
        if not callable(fn):
            raise PoolingNotSupported(
                f"{type(backbone).__name__} does not implement "
                "extract_embeddings_perpatch, so it has no per-patch tokens to "
                "hand back. Re-run with --pooling mean."
            )
        return fn(batch)
    return backbone.extract_embeddings(batch)


def _extract_or_load_embeddings(
    cache_root: Path,
    dataset_name: str,
    split_name: str,
    checkpoint_spec,
    backbone,
    dataloader,
    datamodule,
    cache_purpose: str = "default",
    cache_dir_override: Optional[Path] = None,
):
    if _PRECOMPUTED_CACHE_FIX:
        precomputed_fn = getattr(datamodule, "precomputed_embedding_cache_dir", None)
        if callable(precomputed_fn):
            ext_dir = precomputed_fn(
                checkpoint_id=checkpoint_spec.identifier,
                split=split_name,
                purpose=cache_purpose,
            )
            if ext_dir is not None:
                ext_dir = Path(ext_dir)
                if cache_exists(ext_dir):
                    import time
                    t0 = time.time()
                    payload = load_embedding_payload(ext_dir, mmap_mode="r")
                    dt = time.time() - t0
                    print(
                        f"[embedding] External HIT {split_name:<6s}  "
                        f"{len(payload.labels):>7,d} samples  from {ext_dir}  "
                        f"loaded in {dt:.1f}s",
                        flush=True,
                    )
                    return payload, {
                        "cache_dir": str(ext_dir),
                        "cache_hit": True,
                        "external_cache": True,
                    }
                raise FileNotFoundError(
                    f"precomputed_embedding_cache_dir returned {ext_dir} but it does not "
                    f"contain features.npy + labels.npy. Refusing to fall back to backbone "
                    f"extraction for dataset={dataset_name!r} "
                    f"checkpoint={checkpoint_spec.identifier!r}."
                )

    cache_dir = cache_dir_override if cache_dir_override is not None else _embedding_cache_dir(cache_root, dataset_name, checkpoint_spec, split_name, datamodule, purpose=cache_purpose)
    if cache_exists(cache_dir):
        payload = load_embedding_payload(cache_dir, mmap_mode="r")
        return payload, {"cache_dir": str(cache_dir), "cache_hit": True}

    if _REQUIRE_CACHED_EMBEDDINGS:
        raise EmbeddingsNotFound(
            f"No embeddings for dataset={dataset_name!r} "
            f"checkpoint={checkpoint_spec.identifier!r} split={split_name!r}.\n"
            f"  looked in: {cache_dir}\n"
            f"Probing reads embeddings; it does not create them. Run:\n"
            f"  python -m neuroatlas.entrypoints.embed "
            f"--dataset {dataset_name} --models {checkpoint_spec.identifier}\n"
            f"(then re-run this probe with the same --cache-root)."
        )

    _MAX_SPLITS = 8
    num_workers = min(getattr(dataloader, "_num_workers", 0) or 0, _MAX_SPLITS)
    loader_dataset = getattr(dataloader, "dataset", None)
    collate_fn = getattr(dataloader, "collate_fn", None)
    batch_size_val = getattr(dataloader, "batch_size", None)
    use_subject_split = (
        num_workers > 0
        and loader_dataset is not None
        and collate_fn is not None
        and batch_size_val is not None
        and hasattr(loader_dataset, "_index")
    )

    with IncrementalEmbeddingWriter(cache_dir, num_workers=0) as writer:
        skip_rows = writer.n_rows_done
        efficient_resume = (
            skip_rows > 0
            and writer.has_restored_items
            and batch_size_val is not None
            and skip_rows % batch_size_val == 0
        )

        try:
            total_batches = len(dataloader)
        except TypeError:
            total_batches = None

        def _has_evict(ds) -> bool:
            return hasattr(ds, "evict_subject") or hasattr(ds, "evict_recording")

        def _do_evict(ds, key: str) -> None:
            if hasattr(ds, "evict_subject"):
                ds.evict_subject(key)
            elif hasattr(ds, "evict_recording"):
                ds.evict_recording(key)

        if use_subject_split:
            splits = _split_dataset_by_subject(loader_dataset, num_workers)
            sub_offsets = [0] * num_workers
            if efficient_resume:
                k_batches = skip_rows // batch_size_val
                for i in range(num_workers):
                    if i < k_batches:
                        skipped = -(-(k_batches - i) // num_workers)  # ceil div
                    else:
                        skipped = 0
                    sub_offsets[i] = skipped * batch_size_val
            sub_loaders = []
            for ds, off in zip(splits, sub_offsets):
                sampler = _OffsetSequentialSampler(ds, off) if off > 0 else None
                sub_loaders.append(
                    DataLoader(
                        ds,
                        batch_size=batch_size_val,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=collate_fn,
                        sampler=sampler,
                        pin_memory=False,
                    )
                )
            raw_iter = _sequential_round_robin_iterator(sub_loaders)
            can_evict_per_loader = all(_has_evict(ds) for ds in splits)
        else:
            splits = None
            can_evict_per_loader = False
            if efficient_resume:
                resume_loader = DataLoader(
                    loader_dataset,
                    batch_size=batch_size_val,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_fn,
                    sampler=_OffsetSequentialSampler(loader_dataset, skip_rows),
                    pin_memory=False,
                )
                raw_iter = ((0, batch) for batch in resume_loader)
            elif num_workers > 0 and loader_dataset is not None and collate_fn is not None and batch_size_val is not None:
                safe_loader = DataLoader(
                    loader_dataset,
                    batch_size=batch_size_val,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_fn,
                    pin_memory=False,
                )
                raw_iter = ((0, batch) for batch in safe_loader)
            elif num_workers > 0 and loader_dataset is not None and collate_fn is not None:
                batch_sampler = getattr(dataloader, "batch_sampler", None)
                if batch_sampler is not None:
                    safe_loader = DataLoader(
                        loader_dataset,
                        batch_sampler=batch_sampler,
                        num_workers=0,
                        collate_fn=collate_fn,
                        pin_memory=False,
                    )
                else:
                    safe_loader = DataLoader(
                        loader_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=collate_fn,
                        pin_memory=False,
                    )
                raw_iter = ((0, batch) for batch in safe_loader)
            else:
                raw_iter = ((0, batch) for batch in dataloader)

        can_evict_single = (
            not use_subject_split
            and loader_dataset is not None
            and _has_evict(loader_dataset)
        )
        prev_subjects: set = set()
        prev_subjects_per_loader: Dict[int, set] = {}
        rows_seen = skip_rows if efficient_resume else 0
        desc = f"Embedding {dataset_name}/{checkpoint_spec.model_family}/{split_name}"
        if skip_rows:
            mode = "efficient" if efficient_resume else "re-walking"
            desc += f" (resuming from {skip_rows} rows, {mode})"
        n_extracted = 0
        for loader_idx, batch in tqdm(raw_iter, total=total_batches, desc=desc, leave=True):
            if _MAX_BATCHES is not None and n_extracted >= _MAX_BATCHES:
                break
            batch_len = len(batch["meta"])
            if not efficient_resume and rows_seen + batch_len <= skip_rows:
                if not writer.has_restored_items:
                    meta = [_sanitize_meta(m) for m in batch["meta"]]
                    writer.extend_items_without_append(meta)
                rows_seen += batch_len
                del batch
                continue
            rows_seen += batch_len
            n_extracted += 1
            features = np.asarray(_extract(backbone, batch))
            labels = _flatten_labels(batch)
            meta = [_sanitize_meta(m) for m in batch["meta"]]
            writer.append(features, labels, meta)
            del features, labels, batch
            if can_evict_per_loader:
                batch_subjects = {
                    k for k in (_eviction_key(splits[loader_idx], m) for m in meta) if k
                }
                prev = prev_subjects_per_loader.get(loader_idx, set())
                for sid in prev - batch_subjects:
                    _do_evict(splits[loader_idx], sid)
                prev_subjects_per_loader[loader_idx] = batch_subjects
            elif can_evict_single:
                batch_subjects = {
                    k for k in (_eviction_key(loader_dataset, m) for m in meta) if k
                }
                for sid in prev_subjects - batch_subjects:
                    _do_evict(loader_dataset, sid)
                prev_subjects = batch_subjects
        if can_evict_per_loader:
            for li, subjs in prev_subjects_per_loader.items():
                for sid in subjs:
                    _do_evict(splits[li], sid)
        elif can_evict_single:
            for sid in prev_subjects:
                _do_evict(loader_dataset, sid)

        cache_metadata = _cache_spec(dataset_name, checkpoint_spec, split_name, datamodule, purpose=cache_purpose)
        paths = writer.finalize(metadata=cache_metadata)
    paths["cache_hit"] = False
    payload = load_embedding_payload(cache_dir, mmap_mode="r")
    return payload, paths


def _extract_only_result(
    dataset_name: str, checkpoint_spec, datamodule, backbone, status: str, cache_paths: dict,
) -> BenchmarkResult:
    return BenchmarkResult(
        checkpoint_id=checkpoint_spec.identifier,
        dataset_name=dataset_name,
        evaluation_mode="linear_probe_eval",
        metrics=None,
        cache_paths=cache_paths,
        metadata={
            **backbone.metadata(),
            **_dataset_context(datamodule),
            "task_name": "linear_probe",
            "extract_only_status": status,
        },
    )


def evaluate_linear_probe(
    *,
    dataset_name: str,
    checkpoint_spec,
    datamodule,
    backbone,
    probe_config: Dict[str, Any],
    seeds,
    probe_dir: Path,
    cache_root: Path,
    extract_only: bool = False,
    embed_chunk: Optional[tuple] = None,
    **_,
) -> BenchmarkResult:
    cache_paths: dict = {}
    use_global_cache = datamodule.supports_global_embedding_cache() and getattr(datamodule, "limit_windows_per_split", None) is None

    if use_global_cache:
        global_cache_dir = _embedding_cache_dir(
            cache_root, dataset_name, checkpoint_spec, "all", datamodule, purpose="global_embeddings",
        )

        if cache_exists(global_cache_dir):
            # Full cache already ready.
            if extract_only:
                print(f"[extract-only] global cache already exists at {global_cache_dir}")
                return _extract_only_result(dataset_name, checkpoint_spec, datamodule, backbone, "already_cached", {"global_cache_dir": str(global_cache_dir)})
            full_payload = load_embedding_payload(global_cache_dir, mmap_mode="r")
            full_paths: dict = {"cache_dir": str(global_cache_dir), "cache_hit": True}

        elif embed_chunk is not None:
            # Chunk extraction mode: extract this chunk and return.
            chunk_idx, n_chunks = embed_chunk
            chunk_dir = global_cache_dir / "_chunks" / f"{chunk_idx}_of_{n_chunks}"
            if cache_exists(chunk_dir):
                print(f"[embed-chunk] chunk {chunk_idx}/{n_chunks} already cached at {chunk_dir}")
            else:
                _extract_or_load_embeddings(
                    cache_root, dataset_name, "all", checkpoint_spec, backbone,
                    datamodule.full_embedding_dataloader(), datamodule,
                    cache_purpose="global_embeddings",
                    cache_dir_override=chunk_dir,
                )
            merge_embedding_chunks(global_cache_dir)
            return _extract_only_result(dataset_name, checkpoint_spec, datamodule, backbone, "chunk_extracted", {"chunk_dir": str(chunk_dir)})

        else:
            # Normal path: try merging chunks first, then full extraction.
            if merge_embedding_chunks(global_cache_dir):
                full_payload = load_embedding_payload(global_cache_dir, mmap_mode="r")
                full_paths = {"cache_dir": str(global_cache_dir), "cache_hit": True, "merged_from_chunks": True}
            else:
                full_payload, full_paths = _extract_or_load_embeddings(
                    cache_root, dataset_name, "all", checkpoint_spec, backbone,
                    datamodule.full_embedding_dataloader(), datamodule,
                    cache_purpose="global_embeddings",
                )

        if extract_only:
            return _extract_only_result(dataset_name, checkpoint_spec, datamodule, backbone, "extracted", {"global_cache_dir": str(global_cache_dir)})

        split_payloads = datamodule.split_global_embedding_payload(full_payload)
        train_payload = split_payloads["train"]
        val_payload = split_payloads["val"]
        test_payload = split_payloads["test"]
        cache_paths.update({f"global_{key}": value for key, value in full_paths.items()})
    else:
        if embed_chunk is not None:
            print(f"[embed-chunk] dataset {dataset_name!r} does not support global embedding cache; ignoring --embed-chunk")
        train_payload, train_paths = _extract_or_load_embeddings(
            cache_root, dataset_name, "train", checkpoint_spec, backbone, datamodule.train_dataloader(), datamodule
        )
        val_payload, val_paths = _extract_or_load_embeddings(
            cache_root, dataset_name, "val", checkpoint_spec, backbone, datamodule.val_dataloader(), datamodule
        )
        test_payload, test_paths = _extract_or_load_embeddings(
            cache_root, dataset_name, "test", checkpoint_spec, backbone, datamodule.test_dataloader(), datamodule
        )
        cache_paths.update({f"train_{key}": value for key, value in train_paths.items()})
        cache_paths.update({f"val_{key}": value for key, value in val_paths.items()})
        cache_paths.update({f"test_{key}": value for key, value in test_paths.items()})

        if extract_only:
            return _extract_only_result(dataset_name, checkpoint_spec, datamodule, backbone, "extracted", cache_paths)

    # Filter out -1 (unscored) epochs for sleep-staging label modes only.
    label_mode = datamodule.metadata.get("label_mode", "sleep_stage")
    if label_mode == "sleep_stage" or label_mode.startswith("scorer_"):
        for payload in (train_payload, val_payload, test_payload):
            mask = np.asarray(payload.labels) >= 0
            payload.features = np.asarray(payload.features)[mask]
            payload.labels = np.asarray(payload.labels)[mask]
            payload.metadata = [m for m, keep in zip(payload.metadata, mask) if keep]

    probe_result = train_probe(
        train_payload.features,
        train_payload.labels,
        val_payload.features,
        val_payload.labels,
        test_payload.features,
        test_payload.labels,
        seeds=list(seeds),
        probe_type=str(probe_config.get("type", "linear")),
        max_iter=int(probe_config.get("max_iter", 10_000)),
        hidden_dims=probe_config.get("hidden_dims"),
        selection_metric=str(probe_config.get("selection_metric", "macro_f1")),
        class_weight=probe_config.get("class_weight"),
        c_values=probe_config.get("c_values"),
    )
    probe_artifacts = save_probe_payload(
        probe_dir,
        probe_result.estimator,
        metadata={
            "checkpoint_id": checkpoint_spec.identifier,
            "dataset_name": dataset_name,
            "dataset_context": _dataset_context(datamodule),
            "probe_config": probe_config,
            "best_seed": probe_result.best_seed,
            "best_val_metric": probe_result.best_val_metric,
        },
    )
    cache_paths.update(probe_artifacts)
    return BenchmarkResult(
        checkpoint_id=checkpoint_spec.identifier,
        dataset_name=dataset_name,
        evaluation_mode="linear_probe_eval",
        metrics=probe_result.metrics,
        cache_paths=cache_paths,
        metadata={
            **backbone.metadata(),
            **_dataset_context(datamodule),
            "task_name": "linear_probe",
            "embedding_split_sizes": {
                "train": int(len(train_payload.labels)),
                "val": int(len(val_payload.labels)),
                "test": int(len(test_payload.labels)),
            },
            "probe_seeds": list(seeds),
            "per_seed": probe_result.per_seed,
        },
    )


TASK_SPECS = [
    TaskSpec(
        slug="linear_probe",
        description="Extract embeddings and fit a train/val/test linear or nonlinear probe.",
        evaluator=evaluate_linear_probe,
    )
]
