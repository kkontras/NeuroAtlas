"""UCDDB benchmark datamodule adapter."""

from __future__ import annotations

import functools

from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuroatlas.benchmarking_helpers.runtime.subject_sampler import SubjectBatchSampler
from neuroatlas.extensions.datasets.dataio.ucddb import (
    DEFAULT_CHANNELS,
    RESP_EVENT_FRACTION_FIELDS,
    SubjectRecord,
    UCDDBDataset,
    scan_ucddb_subjects,
)

from .base import BenchmarkDataModule


# ---------------------------------------------------------------------------
# Label modes
# ---------------------------------------------------------------------------

LABEL_MODES: Tuple[str, ...] = ("sleep_stage", "sex")

SEX_TO_LABEL = {"M": 0, "F": 1}


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate_ucddb(batch: List[Dict[str, object]], *, label_mode: str) -> Dict[str, object]:
    eeg = torch.stack([item["eeg"] for item in batch], dim=0)
    if label_mode == "sex":
        labels = torch.tensor(
            [SEX_TO_LABEL.get(item.get("sex"), -1) for item in batch],
            dtype=torch.long,
        )
    else:
        labels = torch.tensor(
            [item["sleep_stage"] for item in batch], dtype=torch.long,
        )
    meta: List[Dict[str, object]] = []
    for item in batch:
        m: Dict[str, object] = {
            "dataset": "ucddb",
            "subject_id": item["subject_id"],
            "epoch_index": item["epoch_idx"],
            "sleep_stage": item["sleep_stage"],
            "channels": item["channels"],
            "epoch_seconds": item["epoch_seconds"],
            "age": item.get("age"),
            "sex": item.get("sex"),
        }
        for key in ("recording_mean", "recording_std", "recording_q95", "recording_q95_bipolar"):
            if key in item:
                m[key] = item[key]
        for field in RESP_EVENT_FRACTION_FIELDS:
            if field in item:
                m[field] = item[field]
        if "fold_assignments" in item:
            m["fold_assignments"] = item["fold_assignments"]
        meta.append(m)
    return {
        "signals": {"eeg": eeg},
        "label": labels,
        "meta": meta,
        "raw_batch": batch,
    }

def _make_collate(label_mode: str):
    return functools.partial(_collate_ucddb, label_mode=label_mode)


# ---------------------------------------------------------------------------
# Fold splitting
# ---------------------------------------------------------------------------

def _patient_splits(
    all_records: Sequence[SubjectRecord],
    num_folds: int,
    fold: int,
    seed: int = 42,
) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
    """Patient-level k-fold split with rotating test/val/train assignment."""
    seen: Dict[str, SubjectRecord] = {}
    for r in all_records:
        if r.subject_id not in seen:
            seen[r.subject_id] = r
    unique_records = list(seen.values())

    rng = np.random.RandomState(seed)
    indices = np.arange(len(unique_records))
    rng.shuffle(indices)
    chunks = np.array_split(indices, num_folds)

    test_idx = set(chunks[fold % num_folds].tolist())
    val_idx = set(chunks[(fold + 1) % num_folds].tolist())
    train_idx = set(range(len(unique_records))) - test_idx - val_idx

    return (
        [unique_records[i] for i in sorted(train_idx)],
        [unique_records[i] for i in sorted(val_idx)],
        [unique_records[i] for i in sorted(test_idx)],
    )


# ---------------------------------------------------------------------------
# Loader adapter
# ---------------------------------------------------------------------------

# The shared wrapper; the local name is kept so call sites are unchanged.
from ._loader_adapters import ContractMetaLoader as _LoaderAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# BenchmarkDataModule
# ---------------------------------------------------------------------------

class UCDDBBenchmarkDataModule(BenchmarkDataModule):
    def __init__(
        self,
        data_root: str,
        age_csv: Optional[str] = None,
        age_xls: Optional[str] = None,
        batch_size: int = 8,
        fold: int = 0,
        num_folds: int = 5,
        num_workers: int = 0,
        channel_specs: List[str] = DEFAULT_CHANNELS,
        epoch_seconds: float = 30,
        bandpass: Optional[Tuple[float, float]] = None,
        notch: Optional[float] = None,
        highpass: Optional[float] = None,
        signal_kind: str = "raw",
        label_mode: str = "sleep_stage",
        compute_recording_stats: bool = False,
        use_all_eeg_channels: bool = False,
    ) -> None:
        self._channel_names = list(channel_specs)

        if signal_kind != "raw":
            raise ValueError(
                f"ucddb reads only the raw signal (signal_kind=raw), not "
                f"signal_kind={signal_kind!r}"
            )
        if label_mode not in LABEL_MODES:
            raise ValueError(
                f"label_mode must be one of {LABEL_MODES}, got {label_mode!r}."
            )
        meta: Dict[str, object] = {
            "canonical_label_space": ["W", "N1", "N2", "N3", "REM"],
            "epoch_seconds": epoch_seconds,
            "channel_policy": ["eeg"],
            "signal_kind": signal_kind,
            "fold": fold,
            "num_folds": num_folds,
            "label_mode": label_mode,
        }
        super().__init__(name="ucddb", metadata=meta)

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._fold = fold
        self._num_folds = num_folds
        self._label_mode = label_mode
        self._collate = _make_collate(label_mode)
        self._compute_recording_stats = compute_recording_stats

        self._all_records = scan_ucddb_subjects(
            data_root, age_csv=age_csv, age_xls=age_xls,
        )

        self._fold_assignments: Dict[str, Dict[str, str]] = {}
        for f in range(num_folds):
            train_f, val_f, test_f = _patient_splits(
                self._all_records, num_folds=num_folds, fold=f,
            )
            fold_key = f"fold_{f}"
            for r in train_f:
                self._fold_assignments.setdefault(r.subject_id, {})[fold_key] = "train"
            for r in val_f:
                self._fold_assignments.setdefault(r.subject_id, {})[fold_key] = "valid"
            for r in test_f:
                self._fold_assignments.setdefault(r.subject_id, {})[fold_key] = "test"

        self._ds_kwargs = dict(
            channel_specs=channel_specs,
            epoch_seconds=epoch_seconds,
            bandpass=bandpass,
            notch=notch,
            highpass=highpass,
            compute_recording_stats=compute_recording_stats,
            use_all_eeg_channels=use_all_eeg_channels,
        )

        self._train_ds: Optional[UCDDBDataset] = None
        self._val_ds: Optional[UCDDBDataset] = None
        self._test_ds: Optional[UCDDBDataset] = None
        self._all_ds: Optional[UCDDBDataset] = None

    def _get_split_records(self) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
        return _patient_splits(
            self._all_records, num_folds=self._num_folds, fold=self._fold,
        )

    def _get_train_ds(self) -> UCDDBDataset:
        if self._train_ds is None:
            train_records, _, _ = self._get_split_records()
            self._train_ds = UCDDBDataset(subject_records=train_records, **self._ds_kwargs)
        return self._train_ds

    def _get_val_ds(self) -> UCDDBDataset:
        if self._val_ds is None:
            _, val_records, _ = self._get_split_records()
            self._val_ds = UCDDBDataset(subject_records=val_records, **self._ds_kwargs)
        return self._val_ds

    def _get_test_ds(self) -> UCDDBDataset:
        if self._test_ds is None:
            _, _, test_records = self._get_split_records()
            self._test_ds = UCDDBDataset(subject_records=test_records, **self._ds_kwargs)
        return self._test_ds

    def _get_all_ds(self) -> UCDDBDataset:
        if self._all_ds is None:
            self._all_ds = UCDDBDataset(
                subject_records=self._all_records,
                fold_assignments=self._fold_assignments,
                **self._ds_kwargs,
            )
        return self._all_ds

    def _make_loader(self, dataset: UCDDBDataset, shuffle: bool) -> _LoaderAdapter:
        dataset._ensure_index_built()
        subject_ids = [dataset._index[i][0] for i in range(len(dataset))]
        sampler = SubjectBatchSampler(
            subject_ids, batch_size=self._batch_size, shuffle=shuffle,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate,
            pin_memory=False,
        )
        channels = self._channel_names
        native_sfreq = 128.0
        return _LoaderAdapter(
            loader,
            sampling_rate=native_sfreq,
            unit="mV",
            channels=channels,
        )

    def train_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._get_train_ds(), shuffle=True)

    def val_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._get_val_ds(), shuffle=False)

    def test_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._get_test_ds(), shuffle=False)

    # -- global embedding cache ----------------------------------------------

    def cache_context(self, purpose: str = "default") -> Dict[str, object]:
        context = dict(self.metadata)
        if purpose == "global_embeddings":
            context.pop("fold", None)
            context.pop("label_mode", None)
        return context

    def supports_global_embedding_cache(self) -> bool:
        return True

    def full_embedding_dataloader(self) -> _LoaderAdapter:
        if self.embed_chunk is not None:
            from neuroatlas.benchmarking_helpers.registry.contracts import chunk_records
            records = chunk_records(self._all_records, *self.embed_chunk)
            ds = UCDDBDataset(subject_records=records, fold_assignments=self._fold_assignments, **self._ds_kwargs)
            return self._make_loader(ds, shuffle=False)
        return self._make_loader(self._get_all_ds(), shuffle=False)

    def split_global_embedding_payload(
        self, payload: "EmbeddingPayload",
    ) -> Dict[str, "EmbeddingPayload"]:
        from neuroatlas.benchmarking_helpers import EmbeddingPayload

        fold_key = f"fold_{self._fold}"

        def _subset(split_name: str) -> EmbeddingPayload:
            keep = [
                i for i, m in enumerate(payload.metadata)
                if m.get("fold_assignments", {}).get(fold_key) == split_name
            ]
            if not keep:
                from collections import Counter
                seen: Counter = Counter()
                for m in payload.metadata:
                    fa = m.get("fold_assignments") or {}
                    seen[fa.get(fold_key, "<missing>")] += 1
                raise ValueError(
                    f"No embedding rows for {split_name!r} in {fold_key}. "
                    f"Values seen under {fold_key}: {dict(seen)}"
                )
            idx = np.asarray(keep, dtype=int)
            labels = payload.labels[idx]
            if self._label_mode == "sex":
                labels = np.asarray(
                    [SEX_TO_LABEL.get(payload.metadata[i].get("sex"), -1) for i in keep],
                    dtype=labels.dtype,
                )
            else:
                labels = np.asarray(
                    [int(payload.metadata[i]["sleep_stage"]) for i in keep],
                    dtype=labels.dtype,
                )
            return EmbeddingPayload(
                features=payload.features[idx],
                labels=labels,
                metadata=[payload.metadata[i] for i in keep],
            )

        return {
            "train": _subset("train"),
            "val": _subset("valid"),
            "test": _subset("test"),
        }
