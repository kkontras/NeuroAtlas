"""Embed an epilepsy cohort once, then assign folds when probing.

Why this exists
---------------
The epilepsy adapters were written for training: ``train_dataloader`` draws
its windows through a ``WeightedRandomSampler`` (with replacement, as many
draws as there are windows) and ``seizure_detection`` extracted each fold's
train, val and test split through those loaders. Two consequences:

* **Speed.** A random draw visits a different recording for nearly every
  window, and the readers decode a whole EDF (one hour of CHB-MIT, 18
  channels: ~1.6 s, plus filtering) for each recording they touch, keeping
  only eight in memory. A 64-window batch therefore decoded ~50 recordings:
  30-45 s per batch on CHB-MIT, against ~0.3 s for the same loader read in
  order.
* **Per-fold caches.** The cached "train split" was a random resample (on
  the tester's Siena fold-0 cache: 27,951 rows, 11,274 distinct windows, one
  seizure window drawn 98 times), so it could only belong to one fold, and
  every fold extracted again.

The global cache instead holds every window of every recording exactly once,
in a fixed order, and :meth:`split_global_embedding_payload` hands each fold
its train/val/test windows from it -- the protocol of the paper's original
CHB-MIT/Siena pipeline (``embed_chbmit_siena.py``: "embeds the entire dataset
once (no fold filtering) ... then runs k-fold cross-validated linear probes").
The probe balances classes with ``class_weight="balanced"``, as it already did.

Only valid when val/test and train windows coincide, i.e. ``stride_s`` is
unset or equal to ``window_s`` (the benchmark's 10 s / 10 s); otherwise the
adapter falls back to per-split extraction, and ``neuroatlas run`` hands
``embed`` every fold the probe will read (see ``run.plan``).

Who uses it: CHB-MIT (bids backend) and Siena first; then Helsinki, NMT,
Bonn, EPILEPSIAE, SeizeIT1/2 and TUAB (EDF backend), which all cut their
train split per fold and so could not serve fold 1 from the fold-0
extraction. A "recording" is whatever a cohort's folds are made of -- a
clip for Bonn (``_recording_key``).

Reading order
-------------
With W loader workers, DataLoader hands batch ``j`` to worker ``j % W``. Read
in natural order that means every worker touches every recording, so each
recording is decoded W times. :class:`ShardedBatchSampler` cuts the batch
list into W contiguous shards and interleaves them, so worker ``w`` reads only
shard ``w`` and each recording is decoded once. Batch *composition* is the
natural one (batch k = windows k*B .. k*B+B-1), only the order in which
batches arrive changes; the split puts every fold's windows back in their
natural (recording, start) order.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


class ShardedBatchSampler(Sampler[List[int]]):
    """Natural-order batches, interleaved so worker w reads shard w only.

    Deterministic: the order is a function of ``(n_items, batch_size,
    n_shards)``. ``deterministic = True`` tells the resume logic it may skip
    already-written batches by index.
    """

    deterministic = True

    def __init__(self, n_items: int, batch_size: int, n_shards: int = 1) -> None:
        self.n_items = int(n_items)
        self.batch_size = int(batch_size)
        self.n_shards = max(1, int(n_shards))
        n_batches = -(-self.n_items // self.batch_size) if self.n_items else 0
        shards = np.array_split(np.arange(n_batches), min(self.n_shards, max(1, n_batches)))
        order: List[int] = []
        longest = max((len(s) for s in shards), default=0)
        for r in range(longest):
            for s in shards:
                if r < len(s):
                    order.append(int(s[r]))
        self._order = order

    @property
    def order_tag(self) -> str:
        return f"sharded:{self.n_shards}"

    def with_order(self, tag: str) -> "ShardedBatchSampler":
        """The same batches in the order *tag* names (a resume after the
        worker count changed must replay the order the rows were written in)."""
        kind, _, n = str(tag).partition(":")
        if kind != "sharded" or not n.isdigit():
            raise ValueError(f"not a ShardedBatchSampler order: {tag!r}")
        return ShardedBatchSampler(self.n_items, self.batch_size, int(n))

    def batch_indices(self, k: int) -> List[int]:
        start = k * self.batch_size
        return list(range(start, min(start + self.batch_size, self.n_items)))

    def __iter__(self) -> Iterator[List[int]]:
        for k in self._order:
            yield self.batch_indices(k)

    def __len__(self) -> int:
        return len(self._order)


_FOLD_KEYS = ("fold", "fold_source", "fold_stats", "n_folds", "folds_manifest")


def default_montage(datamodule) -> Optional[str]:
    """The montage a cohort reads when nobody asks for another: its manifest's
    ``runtime_defaults.montage`` (what ``build_config`` passes), else the
    datamodule's own constructor default. None if it has no montage option."""
    import inspect

    from neuroatlas.benchmarking_helpers.registry.discovery import load_dataset_spec

    try:
        value = load_dataset_spec(datamodule.name).config_defaults.get("montage")
    except Exception:                       # not a registered cohort (a test double)
        value = None
    if value is not None:
        return str(value)
    try:
        param = inspect.signature(type(datamodule).__init__).parameters.get("montage")
    except (TypeError, ValueError):
        return None
    if param is None or param.default is inspect.Parameter.empty:
        return None
    return param.default


def montage_context(datamodule, context: Dict[str, Any]) -> Dict[str, Any]:
    """Add the montage to a cache context.

    The readers re-reference the signal by montage (unipolar electrodes or
    bipolar pairs: other channels, other values), so the montage is part of
    what an embedding is. It used to enter the key only when it differed from
    the cohort's default, which kept old keys stable but meant that changing a
    default (Siena: bipolar -> unipolar, the montage the paper used, with
    BIOT on its bipolar pairs) would read the old embeddings as the new
    montage's. It now always enters the key, so a cache names its montage;
    caches made before this are not reused. A context that already carries
    the montage is left alone: TUSZ's metadata names it, and EPILEPSIAE's
    ``channel_policy`` is its montage.

    TUAB's ``montage_filter`` (keep only recordings of these montage types)
    changes which recordings exist, so a filter, never the default, enters
    the key too.
    """
    montage = getattr(datamodule, "_montage", None)
    if (montage is not None and "montage" not in context
            and context.get("channel_policy") != montage):
        context["montage"] = montage
    kept = getattr(datamodule, "_montage_filter", None)
    if kept and "montage_filter" not in context:
        context["montage_filter"] = sorted(str(m) for m in kept)
    return context


class RecordingWindowGlobalCache:
    """Mixin for adapters whose folds are subsets of one recording index.

    The adapter provides ``_train_recs``/``_val_recs``/``_test_recs``,
    ``_window_s``, ``_stride_s``, ``_batch_size``, ``_num_workers``,
    ``_collate_fn``, ``_n_recordings()``, ``_subject_of(rec_index)`` and
    ``_window_dataset(rec_indices, stride_s, signal_cache_size)``, whose
    windows carry ``meta[_recording_key]`` and ``meta["window_start_s"]``.
    """

    #: ``seizure_detection`` embeds through the global cache only for adapters
    #: that set this; the others keep their per-split caches until someone
    #: verifies them on their data.
    global_cache_for_seizure_detection = True

    #: The per-window meta key naming the unit the folds are made of. The
    #: split gives each window the role of its unit and orders a role's
    #: windows by (unit, window_start_s), the order the per-split readers
    #: emit.
    _recording_key = "recording_idx"

    def supports_global_embedding_cache(self) -> bool:
        stride, window = self._stride_s, self._window_s
        if window is None:                  # one window per unit (Bonn's whole clip)
            return stride is None
        return stride is None or float(stride) == float(window)

    def cache_context(self, purpose: str = "default") -> Dict[str, Any]:
        # per-split and global caches alike: a non-default montage is its own cache
        context = montage_context(self, dict(self.metadata))
        if purpose == "global_embeddings":
            # Nothing about the fold changes which windows exist.
            for key in _FOLD_KEYS:
                context.pop(key, None)
            # `--set window_s=10` arrives as int, the manifest's 10.0 as a
            # float; they cut identical windows and must find one cache.
            if self._window_s is not None:
                context["epoch_seconds"] = float(self._window_s)
                context["stride_s"] = float(self._window_s)
            else:
                context["stride_s"] = None
            context["global_layout"] = "all_recordings_v1"
        return context

    # -- the loader over every recording --------------------------------

    def _all_recording_indices(self) -> List[int]:
        """Every unit a fold can draw from (default: every recording)."""
        return list(range(self._n_recordings()))

    def _global_recordings(self) -> List[int]:
        recs = self._all_recording_indices()
        chunk = getattr(self, "embed_chunk", None)
        if chunk is None:
            return recs
        k, n = chunk
        subjects = sorted({self._subject_of(r) for r in recs})
        size, extra = divmod(len(subjects), n)
        start = k * size + min(k, extra)
        mine = set(subjects[start:start + size + (1 if k < extra else 0)])
        return [r for r in recs if self._subject_of(r) in mine]

    def full_embedding_dataloader(self):
        from torch.utils.data import DataLoader

        from neuroatlas import progress
        from neuroatlas.extensions.datasets.adapters._loader_adapters import SplitTaggingLoader

        # the window index of every recording (a header read each): an
        # embed item's live line says so meanwhile
        progress.current().phase("indexing windows")
        ds = self._window_dataset(self._global_recordings(), self._window_s, signal_cache_size=2)
        workers = int(self._num_workers or 0)
        loader = DataLoader(
            ds,
            batch_sampler=ShardedBatchSampler(len(ds), self._batch_size, max(1, workers)),
            num_workers=workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
        )
        return SplitTaggingLoader(loader, split_name="all")

    # -- folds, at probe time ---------------------------------------------

    def split_global_embedding_payload(self, payload) -> Dict[str, Any]:
        from neuroatlas.benchmarking_helpers import EmbeddingPayload

        role: Dict[int, str] = {}
        for name, recs in (("train", self._train_recs), ("val", self._val_recs),
                           ("test", self._test_recs)):
            for r in recs:
                role[int(r)] = name

        metadata = payload.metadata
        unit = self._recording_key
        groups: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        for i, m in enumerate(metadata):
            name = role.get(int(m[unit]))
            if name is not None:
                groups[name].append(i)

        features = payload.features
        labels = np.asarray(payload.labels)
        out: Dict[str, Any] = {}
        for name, idx in groups.items():
            # The per-split readers emit (recording, start) order; so does this.
            idx.sort(key=lambda i: (int(metadata[i][unit]),
                                    float(metadata[i]["window_start_s"])))
            arr = np.asarray(idx, dtype=np.int64)
            items = []
            for i in idx:
                m = dict(metadata[i])
                m["split"] = name
                items.append(m)
            out[name] = EmbeddingPayload(
                features=np.asarray(features[arr]) if arr.size else
                np.zeros((0,) + tuple(features.shape[1:]), dtype=features.dtype),
                labels=labels[arr] if arr.size else np.zeros((0,), dtype=labels.dtype),
                metadata=items,
            )
        return out


def describe_split(payloads: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={len(v.labels)}" for k, v in payloads.items())
