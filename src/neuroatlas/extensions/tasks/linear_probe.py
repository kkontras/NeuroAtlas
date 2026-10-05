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
    discard_partial,
    load_embedding_payload,
    merge_embedding_chunks,
    read_progress,
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


def _json(value) -> str:
    import json

    return json.dumps(value, sort_keys=True, default=str)


def _benchmarks_with(dataset_name: str) -> List[str]:
    try:
        from neuroatlas import catalog

        return [name for name in catalog.benchmarks_using(dataset_name)
                if not catalog.load(name).derived_from]
    except Exception:
        return []


def _not_found_message(cache_dir: Path, dataset_name: str, checkpoint_spec, split_name: str,
                       wanted: Dict[str, object]) -> str:
    """Say which cache was wanted, which ones exist, and how they differ.

    The usual cause is not a missing extraction but a different key: the
    embed step and the probe were given different dataset settings (a
    ``--set window_s=10`` on one and the manifest's 10.0 on the other is
    already a different key). So this prints the inputs that differ between
    the wanted key and the nearest cache that exists, and suggests only a
    command that gives both steps the same settings.
    """
    import json

    model = checkpoint_spec.identifier
    pair_dir = cache_dir.parent.parent            # <cache_root>/<dataset>/<model>
    found = []
    if pair_dir.is_dir():
        for meta in sorted(pair_dir.glob("*/*/metadata.json")):
            try:
                have = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            keys = set(have) | set(wanted)
            diff = sorted(k for k in keys if _json(have.get(k)) != _json(wanted.get(k)))
            found.append((len(diff), meta.parent, have, diff))
    lines = [
        f"No embeddings for dataset={dataset_name!r} checkpoint={model!r} split={split_name!r}.",
        f"  looked in: {cache_dir}",
    ]
    if found:
        found.sort(key=lambda t: (t[0], str(t[1])))
        _, where, have, diff = found[0]
        lines.append(f"  nearest existing cache for this dataset and model: {where}")
        lines.append("  it was made with different inputs:")
        for k in diff:
            lines.append(f"    {k}: that cache {_json(have.get(k)) if k in have else '(absent)'}, "
                         f"this probe {_json(wanted.get(k)) if k in wanted else '(absent)'}")
        if len(found) > 1:
            lines.append(f"  ({len(found) - 1} other cache(s) for this pair under {pair_dir})")
        lines.append("A probe reads only embeddings made with the same dataset settings "
                     "(--set ..., --expected-epoch-seconds) it is given itself.")
    else:
        lines.append(f"  no embeddings exist yet for this dataset and model under {pair_dir.parent.parent}")
    if split_name in ("train", "val", "test") and "fold" in wanted:
        # A per-split cache holds one fold's split: `embed` must be given
        # that fold (`run` passes the probe's folds to its embed step).
        lines.append(f"  this dataset caches its embeddings per fold and split: embed fold "
                     f"{wanted.get('fold')} with `neuroatlas embed ... --folds {wanted.get('fold')}`.")
    lines.append("Probing reads embeddings; it does not create them.")
    benches = _benchmarks_with(dataset_name)
    if len(benches) == 1:
        lines.append(f"To extract and probe with one set of settings: "
                     f"neuroatlas run {benches[0]} --dataset {dataset_name} -m {model}")
    else:
        lines.append(f"To extract: neuroatlas embed --dataset {dataset_name} --models {model}, "
                     f"with the same --set/--expected-epoch-seconds flags as this probe, "
                     f"and the same --cache-root.")
    return "\n".join(lines)


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


def probe_features(features) -> np.ndarray:
    """The 2-D ``(n_windows, dim)`` matrix a probe fits on.

    ``--pooling mean`` caches one vector per window, already 2-D.
    ``--pooling per_patch`` caches each window's tokens, ``(n_windows,
    n_tokens, token_dim)``; they are concatenated in token order into one
    ``n_tokens * token_dim`` vector per window. That is the paper's per-patch
    protocol: the per-patch BCI numbers (Fig. 20 / 21, the per_patch columns
    of the BCI tables) were probed by ``probe_bci_from_embeddings.py``, whose
    ``_flatten_features`` reshapes ``(n_trials, n_patches, dim)`` to
    ``(n_trials, -1)`` -- "flatten the patch axis into the feature dim ...
    (concatenate patches)" -- before the same StandardScaler + logistic
    regression. Averaging the tokens instead would be ``--pooling mean``.
    """
    arr = features if isinstance(features, np.ndarray) else np.asarray(features)
    if arr.ndim <= 2:
        return arr
    return np.asarray(arr).reshape(arr.shape[0], -1)


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


def _sequential_round_robin_iterator(loaders: Sequence[DataLoader], skip_first_round=()) -> Iterator:
    """Yield ``(loader_idx, batch)`` in strict round-robin order, single-threaded.

    ``skip_first_round``: loaders that already served their batch of the
    round an interrupted run stopped in; resuming serves the rest of that
    round first, so the rows arrive in exactly the uninterrupted order.
    """
    iters = [iter(loader) for loader in loaders]
    sentinel = object()
    active = list(range(len(iters)))
    skip = set(skip_first_round)
    while active:
        next_active = []
        for i in active:
            if i in skip:
                next_active.append(i)
                continue
            batch = next(iters[i], sentinel)
            if batch is sentinel:
                continue
            yield i, batch
            next_active.append(i)
        skip = set()
        active = next_active


def _round_robin_resume_point(lengths: Sequence[int], batch_size: int, skip_rows: int):
    """Where an interrupted round-robin over *lengths*-row loaders stopped.

    Returns ``(batches consumed per loader, loaders done in the open round)``
    when *skip_rows* falls exactly on a batch boundary, else None. Replays
    :func:`_sequential_round_robin_iterator` on the row counts alone, so it
    is exact even after some loaders have run out (their last batch is short,
    which is why ``skip_rows % batch_size`` cannot locate the point).
    """
    n_batches = [-(-int(n) // batch_size) for n in lengths]
    consumed = [0] * len(lengths)
    rows = 0
    active = [i for i, nb in enumerate(n_batches) if nb > 0]
    while active:
        for p, i in enumerate(active):
            if rows == skip_rows:
                return consumed, set(active[:p])
            rows += min(batch_size, int(lengths[i]) - consumed[i] * batch_size)
            consumed[i] += 1
            if rows > skip_rows:
                return None
        active = [i for i in active if consumed[i] < n_batches[i]]
    return (consumed, set()) if rows == skip_rows else None


class _ListBatchSampler(Sampler):
    def __init__(self, batches: List[List[int]]) -> None:
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def _inner_dataloader(loader):
    """``(wrappers outermost first, torch DataLoader or None)``."""
    chain = []
    inner = loader
    while not isinstance(inner, DataLoader) and hasattr(inner, "_loader"):
        chain.append(inner)
        inner = inner._loader
    return chain, (inner if isinstance(inner, DataLoader) else None)


def _loader_order_tag(loader) -> Optional[str]:
    """The batch sampler's order tag, when its order depends on its config."""
    _, inner = _inner_dataloader(loader)
    return getattr(getattr(inner, "batch_sampler", None), "order_tag", None)


def _resume_loader(loader, skip_rows: int, order_tag: Optional[str] = None):
    """A copy of *loader* that starts after its first *skip_rows* rows.

    Walks the wrapper chain (each wrapper keeps its inner loader in
    ``_loader``) down to the torch DataLoader, replays its batch sampler --
    index lists only, no data is read -- to find the batch where *skip_rows*
    ends, and rebuilds the chain over a DataLoader that starts there with the
    same workers and collate. Returns ``(loader, n_batches_skipped)``, or
    None when the order is not reproducible (a random sampler) or the row
    count does not fall on a batch boundary.
    """
    from torch.utils.data import BatchSampler, SequentialSampler

    chain, inner = _inner_dataloader(loader)
    if inner is None:
        return None
    batch_sampler = inner.batch_sampler
    current_tag = getattr(batch_sampler, "order_tag", None)
    if current_tag is not None and order_tag != current_tag:
        # Written in another order (another worker count): replay that one.
        if order_tag is None or not hasattr(batch_sampler, "with_order"):
            return None
        batch_sampler = batch_sampler.with_order(order_tag)
    deterministic = getattr(batch_sampler, "deterministic", False) or (
        isinstance(batch_sampler, BatchSampler)
        and isinstance(batch_sampler.sampler, SequentialSampler)
    )
    if not deterministic:
        return None
    batches = [list(b) for b in batch_sampler]
    rows = 0
    k = 0
    while k < len(batches) and rows < skip_rows:
        rows += len(batches[k])
        k += 1
    if rows != skip_rows:
        return None
    kwargs = dict(
        batch_sampler=_ListBatchSampler(batches[k:]),
        num_workers=inner.num_workers,
        collate_fn=inner.collate_fn,
        pin_memory=inner.pin_memory,
        timeout=inner.timeout,
        worker_init_fn=inner.worker_init_fn,
        multiprocessing_context=inner.multiprocessing_context,
        persistent_workers=inner.persistent_workers,
    )
    if inner.num_workers > 0:
        kwargs["prefetch_factor"] = inner.prefetch_factor
    rebuilt = DataLoader(inner.dataset, **kwargs)
    for wrapper in reversed(chain):
        outer = copy.copy(wrapper)
        outer._loader = rebuilt
        rebuilt = outer
    return rebuilt, k


def _require_row_per_item(features: np.ndarray, labels: np.ndarray, meta: List[dict],
                          checkpoint_spec) -> None:
    """The cache stores features, labels and items side by side, one row each
    per item of the batch; refuse a batch whose counts differ rather than
    write a cache whose rows no longer line up.

    A sequence checkpoint's batch is ``B`` windows of ``W`` epochs, and how
    many items it carries depends on its target epoch: every epoch (``B * W``)
    for ``target_idx="all"``, one per window (``B``) otherwise. The backbone
    must return that many rows. CoRe-Sleep once returned one ``W * dim`` row
    per window, which wrote 21,807 feature rows next to 457,947 items on
    Sleep-EDF and failed every probe on "cache row-count mismatch".
    """
    n_items = len(meta)
    n_feat = int(features.shape[0]) if features.ndim else 0
    n_label = int(labels.shape[0])
    if n_feat == n_items and n_label == n_items:
        return
    seq = int(getattr(checkpoint_spec, "expected_sequence_length", 1) or 1)
    overrides = getattr(checkpoint_spec, "runtime_overrides", None) or {}
    hint = ""
    if seq > 1:
        hint = (f" It reads windows of {seq} epochs with embedding_target_idx="
                f"{overrides.get('embedding_target_idx', overrides.get('target_idx', 'all'))!r}: "
                f"its extract_embeddings must return one row per labelled epoch "
                f"(every epoch for 'all', the target epoch otherwise).")
    raise RuntimeError(
        f"{checkpoint_spec.identifier}: the backbone returned {n_feat} embedding rows "
        f"(shape {tuple(features.shape)}) for a batch of {n_items} items with "
        f"{n_label} labels; the cache needs one row per item.{hint}"
    )


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
        raise EmbeddingsNotFound(_not_found_message(
            cache_dir, dataset_name, checkpoint_spec, split_name,
            _cache_spec(dataset_name, checkpoint_spec, split_name, datamodule, purpose=cache_purpose),
        ))

    order_tag = _loader_order_tag(dataloader)
    if order_tag is not None:
        progress = read_progress(cache_dir)
        if progress and progress.get("n_rows") and not progress.get("order"):
            # Rows written in an unrecorded order cannot be skipped by index.
            print(f"[embedding] discarding an interrupted extraction at {cache_dir}: "
                  f"its row order was not recorded", flush=True)
            discard_partial(cache_dir)

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
        if skip_rows == 0:
            writer.order_tag = order_tag
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

        initial_batches = 0
        if use_subject_split:
            splits = _split_dataset_by_subject(loader_dataset, num_workers)
            sub_offsets = [0] * num_workers
            done_in_open_round: set = set()
            point = None
            if skip_rows > 0 and writer.has_restored_items:
                point = _round_robin_resume_point([len(ds) for ds in splits], batch_size_val, skip_rows)
            if point is not None:
                consumed, done_in_open_round = point
                sub_offsets = [c * batch_size_val for c in consumed]
                initial_batches = sum(consumed)
                efficient_resume = True
            else:
                efficient_resume = False
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
            raw_iter = _sequential_round_robin_iterator(sub_loaders, done_in_open_round)
            can_evict_per_loader = all(_has_evict(ds) for ds in splits)
        else:
            splits = None
            can_evict_per_loader = False
            resumed = None
            if (skip_rows > 0 and writer.has_restored_items and not efficient_resume
                    and not (num_workers > 0 and loader_dataset is not None and collate_fn is not None)):
                # The loader is iterated as given (the epilepsy readers, any
                # wrapper that hides its dataset): skip by batch index rather
                # than re-reading every finished window.
                resumed = _resume_loader(dataloader, skip_rows, writer.restored_order_tag)
            if resumed is not None:
                resume_dl, initial_batches = resumed
                efficient_resume = True
                raw_iter = ((0, batch) for batch in resume_dl)
            elif efficient_resume:
                resume_loader = DataLoader(
                    loader_dataset,
                    batch_size=batch_size_val,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate_fn,
                    sampler=_OffsetSequentialSampler(loader_dataset, skip_rows),
                    pin_memory=False,
                )
                initial_batches = skip_rows // batch_size_val
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
        # A resumed bar starts at the resume point, not at 0.
        for loader_idx, batch in tqdm(raw_iter, total=total_batches, desc=desc, leave=True,
                                      initial=initial_batches if efficient_resume else 0):
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
            _require_row_per_item(features, labels, meta, checkpoint_spec)
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
            # "all": one cache serves every fold, so an extraction pass over
            # several folds (embed --folds) is done after this one.
            "embedding_cache_layout": (
                "all" if any(k.startswith("global_") or k == "chunk_dir" for k in cache_paths)
                else "per_split"),
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

    # Per-patch tokens (--pooling per_patch) become one concatenated vector
    # per window -- the paper's per-patch probe (see probe_features).
    for payload in (train_payload, val_payload, test_payload):
        payload.features = probe_features(payload.features)

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
            "pooling": _POOLING,
            "probe_feature_dim": int(np.asarray(train_payload.features).shape[1])
            if np.asarray(train_payload.features).ndim == 2 else None,
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
