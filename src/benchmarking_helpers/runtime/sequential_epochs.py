"""Sequential epoch infrastructure for multi-epoch sleep staging models.

Provides a sampler, collate wrapper, and datamodule wrapper that convert
single-epoch dataloaders into multi-epoch sliding-window dataloaders.
Also includes SeqSleepNet-style multiplicative aggregation.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Sampler

from ..registry.contracts import BenchmarkDataModule

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class SequentialEpochSampler(Sampler[List[int]]):
    """Yields sliding windows of *window_size* consecutive epoch indices.

    Each yielded batch is a list of exactly ``window_size`` dataset indices
    forming one contiguous window within a single recording.

    Parameters
    ----------
    recording_ids : per-sample recording identifier.
    epoch_indices : per-sample sequential epoch index within its recording.
    window_size : epochs per window.
    stride : step between window starts (1 = fully overlapping).
    shuffle : randomise window order (epoch order within a window is preserved).
    pad_mode : ``"replicate"`` pads boundary / short recordings by repeating
        edge epochs; ``"drop"`` skips them.
    """

    def __init__(
        self,
        recording_ids: Sequence[str],
        epoch_indices: Sequence[int],
        window_size: int,
        stride: int = 1,
        shuffle: bool = False,
        pad_mode: str = "replicate",
        batch_size: int = 1,
    ) -> None:
        assert len(recording_ids) == len(epoch_indices)
        self._shuffle = shuffle
        self._batch_size = batch_size

        by_rec: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for ds_idx, (rec_id, ep_idx) in enumerate(
            zip(recording_ids, epoch_indices)
        ):
            by_rec[rec_id].append((ep_idx, ds_idx))

        self._windows: List[List[int]] = []
        self._window_rec_ids: List[str] = []
        for rec_id in sorted(by_rec):
            items = sorted(by_rec[rec_id], key=lambda x: x[0])
            ds_indices = [ds_idx for _, ds_idx in items]
            n = len(ds_indices)

            if n < window_size:
                if pad_mode == "drop":
                    continue
                padded = list(ds_indices)
                while len(padded) < window_size:
                    padded.append(ds_indices[-1])
                self._windows.append(padded)
                self._window_rec_ids.append(rec_id)
                continue

            for start in range(0, n - window_size + 1, stride):
                self._windows.append(ds_indices[start : start + window_size])
                self._window_rec_ids.append(rec_id)

            if stride > 1:
                last_complete = ((n - window_size) // stride) * stride
                next_start = last_complete + stride
                if next_start < n and next_start + window_size > n:
                    if pad_mode == "replicate":
                        window = list(ds_indices[next_start:])
                        while len(window) < window_size:
                            window.append(ds_indices[-1])
                        self._windows.append(window)
                        self._window_rec_ids.append(rec_id)

    def __iter__(self) -> Iterator[List[int]]:
        by_rec: Dict[str, List[int]] = defaultdict(list)
        for i, rec_id in enumerate(self._window_rec_ids):
            by_rec[rec_id].append(i)

        rec_batches: List[List[int]] = []
        for rec_id in sorted(by_rec):
            win_indices = by_rec[rec_id]
            for start in range(0, len(win_indices), self._batch_size):
                rec_batches.append(win_indices[start : start + self._batch_size])

        if self._shuffle:
            order = torch.randperm(len(rec_batches)).tolist()
        else:
            order = list(range(len(rec_batches)))

        for i in order:
            batch = []
            for win_idx in rec_batches[i]:
                batch.extend(self._windows[win_idx])
            yield batch

    def __len__(self) -> int:
        import math
        from collections import Counter
        counts = Counter(self._window_rec_ids)
        return sum(math.ceil(c / self._batch_size) for c in counts.values())


# ---------------------------------------------------------------------------
# Sequential collate wrapper
# ---------------------------------------------------------------------------


def make_sequential_collate(
    base_collate,
    window_size: int,
    target_idx: Union[str, int] = "all",
):
    """Wrap a per-epoch collate to produce ``(B, W, C, T)`` sequence batches.

    The base collate receives ``B * window_size`` individual dataset items
    and produces ``signals["eeg"]: (B*W, C, T)``, ``label: (B*W,)``, and
    ``meta: [dict]*(B*W)``.  This wrapper reshapes every signal tensor to
    ``(B, W, C, T)`` and optionally slices labels / meta according to
    *target_idx*.

    The batch dict also receives a ``_seq_target_idx`` key so that
    downstream backbone ``extract_embeddings`` can slice features to the
    correct epoch position.

    Parameters
    ----------
    base_collate : callable
        The dataset adapter's collate (with metadata injection from
        ``_LoaderAdapter``).
    window_size : int
        Expected items per window.
    target_idx : ``"all"`` | int
        ``"all"`` keeps all *B*W* labels (models predicting every epoch).
        An int selects that index (``-1`` = last epoch, for SleePyCo;
        ``10`` = center epoch, for SleepTransformer).
    """

    def _collate(batch):
        result = base_collate(batch)
        result = dict(result)
        signals = dict(result["signals"])

        total = len(batch)
        B = total // window_size

        for key in signals:
            if torch.is_tensor(signals[key]):
                signals[key] = signals[key].view(B, window_size, *signals[key].shape[1:])
        result["signals"] = signals
        result["_seq_target_idx"] = target_idx

        if target_idx != "all":
            idx = int(target_idx) % window_size
            labels = result["label"]
            labels_2d = labels.view(B, window_size) if hasattr(labels, "view") else labels.reshape(B, window_size)
            result["label"] = labels_2d[:, idx]
            result["meta"] = [result["meta"][w * window_size + idx] for w in range(B)]

        return result

    return _collate


# ---------------------------------------------------------------------------
# Loader proxy
# ---------------------------------------------------------------------------


class _SequentialLoaderProxy:
    """Minimal proxy returned by :class:`SequentialDataModuleWrapper`."""

    def __init__(self, loader: DataLoader, dataset, batch_sampler):
        self._loader = loader
        self.dataset = dataset
        self.batch_size = None
        self.batch_sampler = batch_sampler
        self.collate_fn = loader.collate_fn
        self._num_workers = loader.num_workers

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)


# ---------------------------------------------------------------------------
# DataModule wrapper
# ---------------------------------------------------------------------------


class SequentialDataModuleWrapper(BenchmarkDataModule):
    """Wraps any :class:`BenchmarkDataModule` to yield sequential windows.

    Intercepts ``train_dataloader`` / ``val_dataloader`` /
    ``test_dataloader`` / ``full_embedding_dataloader``, replaces the
    sampler with :class:`SequentialEpochSampler`, and wraps the collate
    so that each batch is ``(B, window_size, C, T)``.

    For ``full_embedding_dataloader`` (used by linear-probe embedding
    extraction), separate ``embedding_stride`` and ``embedding_target_idx``
    are used to control the sliding window and epoch selection independently
    from the native-head evaluation config.
    """

    def __init__(
        self,
        base_dm: BenchmarkDataModule,
        window_size: int,
        stride: int = 1,
        target_idx: Union[str, int] = "all",
        pad_mode: str = "replicate",
        batch_size: int = 1,
        embedding_stride: Optional[int] = None,
        embedding_target_idx: Optional[Union[str, int]] = None,
    ) -> None:
        meta = dict(getattr(base_dm, "metadata", {}))
        meta["sequence_length"] = window_size
        meta["sequence_stride"] = stride
        meta["sequence_target_idx"] = str(target_idx)
        super().__init__(name=base_dm.name, metadata=meta)
        self._base = base_dm
        self._window_size = window_size
        self._stride = stride
        self._target_idx = target_idx
        self._pad_mode = pad_mode
        self._batch_size = batch_size
        self._emb_stride = embedding_stride if embedding_stride is not None else window_size
        self._emb_target_idx = embedding_target_idx if embedding_target_idx is not None else target_idx
        if hasattr(base_dm, "embed_chunk"):
            self.embed_chunk = base_dm.embed_chunk

    def _wrap_loader(
        self,
        adapter,
        shuffle: bool = False,
        *,
        override_stride: Optional[int] = None,
        override_target_idx: Optional[Union[str, int]] = None,
        override_batch_size: Optional[int] = None,
        override_pad_mode: Optional[str] = None,
    ):
        dataset = adapter.dataset
        if hasattr(dataset, "_ensure_index_built"):
            dataset._ensure_index_built()

        recording_ids = [dataset._index[i][0] for i in range(len(dataset))]
        epoch_indices = [dataset._index[i][1] for i in range(len(dataset))]

        stride = override_stride if override_stride is not None else self._stride
        target_idx = override_target_idx if override_target_idx is not None else self._target_idx
        batch_size = override_batch_size if override_batch_size is not None else self._batch_size
        pad_mode = override_pad_mode if override_pad_mode is not None else self._pad_mode

        sampler = SequentialEpochSampler(
            recording_ids,
            epoch_indices,
            window_size=self._window_size,
            stride=stride,
            shuffle=shuffle,
            pad_mode=pad_mode,
            batch_size=batch_size,
        )

        base_collate = getattr(adapter, "collate_fn", None)
        if base_collate is None:
            raise ValueError(
                "SequentialDataModuleWrapper requires a collate_fn on "
                "the base loader/adapter."
            )

        seq_collate = make_sequential_collate(
            base_collate, self._window_size, target_idx=target_idx,
        )

        num_workers = getattr(adapter, "_num_workers", 0)

        new_loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=seq_collate,
            pin_memory=False,
        )
        return _SequentialLoaderProxy(new_loader, dataset=dataset, batch_sampler=sampler)

    # -- public dataloader methods ------------------------------------------

    def train_dataloader(self):
        return self._wrap_loader(self._base.train_dataloader(), shuffle=True)

    def val_dataloader(self):
        return self._wrap_loader(self._base.val_dataloader(), shuffle=False)

    def test_dataloader(self):
        return self._wrap_loader(self._base.test_dataloader(), shuffle=False)

    # -- global embedding cache (non-overlapping for linear probe) ----------

    def cache_context(self, purpose: str = "default") -> Dict[str, Any]:
        return self._base.cache_context(purpose)

    def supports_global_embedding_cache(self) -> bool:
        return self._base.supports_global_embedding_cache()

    def full_embedding_dataloader(self):
        return self._wrap_loader(
            self._base.full_embedding_dataloader(),
            shuffle=False,
            override_stride=self._emb_stride,
            override_target_idx=self._emb_target_idx,
            override_pad_mode="drop",
        )

    def split_global_embedding_payload(self, payload):
        return self._base.split_global_embedding_payload(payload)
