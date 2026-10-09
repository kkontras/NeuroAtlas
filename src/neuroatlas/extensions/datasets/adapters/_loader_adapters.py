"""Loader wrappers shared by the dataset adapters.

Every adapter wraps its DataLoader in a small class that makes the batch
look the way the wrappers expect. Those classes were written per adapter
and there were 29 of them, most byte-identical apart from a docstring --
so a change to the batch contract meant editing seventeen files, and the
missing ``unit`` key that broke eleven wrappers had to be chased one
adapter at a time.

Two shapes cover them. Adapters keep calling their local name, so nothing
at the call sites changes.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional

from torch.utils.data import DataLoader


class SplitTaggingLoader:
    """Reshape a raw batch into the benchmark format and tag its split.

    Standard format::

        {
            "signals": {"eeg": Tensor(B, C, T)},
            "label":   Tensor(B,),
            "meta":    List[dict],
            "raw_batch": original raw batch dict,
        }

    Used by the epilepsy adapters, whose dataio yields ``eeg``/``labels``
    directly.
    """

    def __init__(self, loader: DataLoader, split_name: str = "") -> None:
        self._loader = loader
        self._split_name = split_name

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        for batch in self._loader:
            for m in batch["meta"]:
                m["split"] = self._split_name
            yield {
                "signals": {"eeg": batch["eeg"]},
                "label": batch["labels"],
                "meta": batch["meta"],
                "raw_batch": batch,
            }


class ContractMetaLoader:
    """Publish the MODEL_CONTRACTS §0 keys on every sample's meta.

    ``unit`` and ``channels`` are constants of the cohort. *sampling_rate*
    is optional because the cohorts differ: most declare one rate for the
    whole corpus, while WSC, MASS and PhysioNet 2026 carry it per sample and
    must not have it overwritten. Passing None leaves whatever the dataio
    already published.

    The keys are injected twice on purpose -- once through a wrapped
    ``collate_fn`` for workers that collate in a subprocess, and once on
    iteration for callers that bypass the collate. Both are idempotent.
    """

    def __init__(
        self,
        loader: DataLoader,
        *,
        unit: str,
        channels: List[str],
        sampling_rate: Optional[float] = None,
    ) -> None:
        self._loader = loader
        self.dataset = loader.dataset
        self.batch_size = getattr(loader, "batch_size", None)

        per: Dict[str, object] = {"unit": unit, "channels": list(channels)}
        if sampling_rate is not None:
            per["sampling_rate"] = float(sampling_rate)

        _raw_collate = loader.collate_fn

        def _collate_with_meta(batch, _c=_raw_collate, _p=per):
            result = _c(batch)
            for m in result["meta"]:
                m.update(_p)
            return result

        self.collate_fn = _collate_with_meta
        self._num_workers = loader.num_workers
        self.batch_sampler = getattr(loader, "batch_sampler", None)
        self._per_sample = per
        self._unit = unit
        self._channels = list(channels)
        self._sampling_rate = None if sampling_rate is None else float(sampling_rate)

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        for batch in self._loader:
            for m in batch["meta"]:
                m.update(self._per_sample)
            yield batch
