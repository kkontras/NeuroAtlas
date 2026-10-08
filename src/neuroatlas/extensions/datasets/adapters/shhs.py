"""SHHS benchmark datamodule: NSRR's EDFs read by :mod:`dataio.shhs`.

Folds are the five of ``configs/folds/shhs_combined.json``: 8444 recordings
(SHHS-1 and SHHS-2), partitioned so that a participant's two visits are always
in the same fold. The unit is the recording (``subject_id`` is
``shhs1-200001``); ``patient_id`` names the participant.

Label modes: ``sleep_stage`` (W, N1, N2, N3, REM per 30 s epoch) and ``age``
(the participant's age at that visit, from NSRR's ``shhs1-dataset`` /
``shhs2-dataset`` tables, joined on ``nsrrid``). Both read the same embedding
cache: brain age reuses the sleep-staging embeddings.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from neuroatlas.benchmarking_helpers.runtime.subject_sampler import SubjectBatchSampler
from neuroatlas.extensions.datasets.dataio.shhs import (
    CHANNEL,
    EPOCH_SECONDS,
    LABEL_NAMES,
    SAMPLING_RATE,
    Recording,
    SHHSDataset,
    age_tables,
    scan_recordings,
)

from ._loader_adapters import ContractMetaLoader
from ._runtime_keys import RAW_ONLY
from .base import BenchmarkDataModule

logger = logging.getLogger(__name__)

LABEL_MODES: Tuple[str, ...] = ("sleep_stage", "age")

#: The fold manifest (configs/folds/shhs_combined.json): its five folds.
FOLD_MANIFEST = "shhs_combined"
N_FOLDS = 5


def _age_labels(items) -> List[float]:
    """Age at the visit per row, NaN where the tables have none."""
    out: List[float] = []
    for item in items:
        age = item.get("age")
        out.append(float("nan") if age is None else float(age))
    return out


def _collate(batch: List[Dict[str, object]], *, label_mode: str) -> Dict[str, object]:
    eeg = torch.stack([item["eeg"] for item in batch], dim=0)
    if label_mode == "age":
        labels = torch.tensor(_age_labels(batch), dtype=torch.float32)
    else:
        labels = torch.tensor([item["sleep_stage"] for item in batch], dtype=torch.long)
    meta: List[Dict[str, object]] = []
    for item in batch:
        m: Dict[str, object] = {
            "dataset": "shhs",
            "subject_id": item["subject_id"],
            "recording_id": item["recording_id"],
            "patient_id": item["patient_id"],
            "visit": item["visit"],
            "epoch_index": item["epoch_idx"],
            "sleep_stage": item["sleep_stage"],
            "epoch_seconds": float(EPOCH_SECONDS),
            "age": item.get("age"),
        }
        for key in ("recording_mean", "recording_std", "recording_q95"):
            if key in item:
                m[key] = item[key]
        if "fold_assignments" in item:
            m["fold_assignments"] = item["fold_assignments"]
        meta.append(m)
    return {"signals": {"eeg": eeg}, "label": labels, "meta": meta, "raw_batch": batch}


def _fold_assignments() -> Dict[str, Dict[str, str]]:
    """``{recording: {"fold_k": "train" | "valid" | "test"}}`` for the five
    folds of the manifest (test = fold k, validation = fold k+1)."""
    from neuroatlas.benchmarking_helpers.registry.fold_manifest import load_fold_split

    out: Dict[str, Dict[str, str]] = {}
    for fold in range(N_FOLDS):
        split = load_fold_split(FOLD_MANIFEST, fold)
        for name, ids in (("train", split.train), ("valid", split.val), ("test", split.test)):
            for rid in ids:
                out.setdefault(rid, {})[f"fold_{fold}"] = name
    return out


class _NightBatches(Sampler):
    """Batches of one night's epochs (:class:`SubjectBatchSampler`), made on
    first use: a loader built for an embedding cache that already exists
    never reads the annotations."""

    def __init__(self, dataset: SHHSDataset, batch_size: int, shuffle: bool) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._batches: Optional[SubjectBatchSampler] = None

    def _sampler(self) -> SubjectBatchSampler:
        if self._batches is None:
            self._dataset._ensure_index_built()
            self._batches = SubjectBatchSampler(
                [entry[0] for entry in self._dataset._index],
                batch_size=self._batch_size, shuffle=self._shuffle)
        return self._batches

    @property
    def deterministic(self) -> bool:
        """Unshuffled, the same batches every time: an interrupted extraction
        resumes where it stopped."""
        return not self._shuffle

    def __iter__(self):
        return iter(self._sampler())

    def __len__(self) -> int:
        return len(self._sampler())


class SHHSBenchmarkDataModule(BenchmarkDataModule):
    """SHHS sleep staging and brain age from NSRR's EDFs (C4-A1, 100 Hz)."""

    RUNTIME_KEYS_FIXED = RAW_ONLY
    RUNTIME_KEYS_IGNORED = {
        "epoch_seconds": (
            "the reader serves the scored 30 s epochs at 100 Hz (3000 samples); a "
            "model's own window is met by its wrapper"
        ),
    }

    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        fold: int = 0,
        n_folds: int = N_FOLDS,
        num_workers: int = 0,
        label_mode: str = "sleep_stage",
        channel_specs: Optional[Sequence[str]] = None,
        compute_recording_stats: bool = False,
    ) -> None:
        if label_mode not in LABEL_MODES:
            raise ValueError(f"shhs: label_mode must be one of {', '.join(LABEL_MODES)}, "
                             f"not {label_mode!r}")
        if int(n_folds) != N_FOLDS:
            raise ValueError(f"shhs: n_folds={n_folds}, but its folds are the {N_FOLDS} of "
                             f"the benchmark; ask for {N_FOLDS}")
        if not 0 <= int(fold) < N_FOLDS:
            raise ValueError(f"shhs has {N_FOLDS} folds (0 to {N_FOLDS - 1}); there is no "
                             f"fold {fold}")
        other = sorted({str(c) for c in (channel_specs or [])} - {CHANNEL})
        if other:
            raise ValueError(f"shhs: the reader serves {CHANNEL} only, not "
                             f"{', '.join(other)}")
        meta: Dict[str, object] = {
            "canonical_label_space": list(LABEL_NAMES),
            "epoch_seconds": EPOCH_SECONDS,
            "channel_policy": ["eeg"],
            "signal_kind": "raw",
            "source": "nsrr_edf",
            "fold": int(fold),
            "n_folds": N_FOLDS,
            "label_mode": label_mode,
        }
        super().__init__(name="shhs", metadata=meta)
        self._data_root = Path(str(data_root))
        self._batch_size = int(batch_size)
        self._num_workers = int(num_workers)
        self._fold = int(fold)
        self._label_mode = label_mode
        self._compute_recording_stats = bool(compute_recording_stats)
        self._collate_fn = functools.partial(_collate, label_mode=label_mode)

        recordings = scan_recordings(self._data_root)
        if not recordings:
            from neuroatlas.extensions.datasets._missing import no_data

            raise FileNotFoundError(no_data(
                "shhs", self._data_root, "data_root",
                detail="no SHHS EDF with its NSRR annotation file"))
        if label_mode == "age" and not any(age_tables(self._data_root).values()):
            from neuroatlas.extensions.datasets._missing import no_data

            raise FileNotFoundError(no_data(
                "shhs", self._data_root / "datasets", "data_root",
                what="no age table (shhs1-dataset, shhs2-dataset) in"))
        self._fold_assignments = _fold_assignments()
        self._records: List[Recording] = [
            r for r in recordings if r.recording_id in self._fold_assignments]
        self._age_by_recording = {r.recording_id: r.age for r in self._records}
        self._datasets: Dict[str, SHHSDataset] = {}

    # -- splits ---------------------------------------------------------------

    def _split_records(self, split: str) -> List[Recording]:
        key = f"fold_{self._fold}"
        return [r for r in self._records
                if self._fold_assignments[r.recording_id].get(key) == split]

    def _dataset(self, split: str) -> SHHSDataset:
        if split not in self._datasets:
            records = self._records if split == "all" else self._split_records(split)
            self._datasets[split] = SHHSDataset(
                records, fold_assignments=self._fold_assignments,
                compute_recording_stats=self._compute_recording_stats)
        return self._datasets[split]

    def _make_loader(self, dataset: SHHSDataset, shuffle: bool) -> ContractMetaLoader:
        sampler = _NightBatches(dataset, self._batch_size, shuffle)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=self._num_workers,
                            collate_fn=self._collate_fn, pin_memory=False)
        return ContractMetaLoader(loader, unit="uV", channels=[CHANNEL],
                                  sampling_rate=float(SAMPLING_RATE))

    def train_dataloader(self) -> ContractMetaLoader:
        return self._make_loader(self._dataset("train"), shuffle=True)

    def val_dataloader(self) -> ContractMetaLoader:
        return self._make_loader(self._dataset("valid"), shuffle=False)

    def test_dataloader(self) -> ContractMetaLoader:
        return self._make_loader(self._dataset("test"), shuffle=False)

    # -- global embedding cache: every recording once, split per fold --------

    def cache_context(self, purpose: str = "default") -> Dict[str, object]:
        context = dict(self.metadata)
        if purpose == "global_embeddings":
            context.pop("fold", None)
            context.pop("label_mode", None)
        return context

    def supports_global_embedding_cache(self) -> bool:
        return True

    def full_embedding_dataloader(self) -> ContractMetaLoader:
        if self.embed_chunk is not None:
            from neuroatlas.benchmarking_helpers.registry.contracts import chunk_records

            records = chunk_records(self._records, *self.embed_chunk)
            dataset = SHHSDataset(records, fold_assignments=self._fold_assignments,
                                  compute_recording_stats=self._compute_recording_stats)
            return self._make_loader(dataset, shuffle=False)
        return self._make_loader(self._dataset("all"), shuffle=False)

    def split_global_embedding_payload(self, payload) -> Dict[str, object]:
        """Fold ``k``'s train/val/test rows of the cache. Epochs left out of
        scoring (flat signal) are dropped for both label modes; ``age`` comes
        from the visit tables read now, not from the cache."""
        from neuroatlas.benchmarking_helpers import EmbeddingPayload

        key = f"fold_{self._fold}"

        def subset(split: str) -> EmbeddingPayload:
            keep = [i for i, m in enumerate(payload.metadata)
                    if (m.get("fold_assignments") or {}).get(key) == split
                    and 0 <= int(m.get("sleep_stage", -1)) < len(LABEL_NAMES)]
            if not keep:
                raise ValueError(f"shhs: the embedding cache holds no {split} recording of "
                                 f"fold {self._fold}")
            idx = np.asarray(keep, dtype=int)
            rows = [payload.metadata[i] for i in keep]
            if self._label_mode == "age":
                labels = np.asarray(
                    [np.nan if self._age_by_recording.get(str(m.get("recording_id"))) is None
                     else self._age_by_recording[str(m.get("recording_id"))] for m in rows],
                    dtype=np.float32)
            else:
                labels = np.asarray([int(m["sleep_stage"]) for m in rows], dtype=np.int64)
            return EmbeddingPayload(features=payload.features[idx], labels=labels,
                                    metadata=rows)

        return {"train": subset("train"), "val": subset("valid"), "test": subset("test")}
