"""PhysioNet Challenge 2026 benchmark datamodule adapter."""

from __future__ import annotations

import functools

from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarking_helpers.runtime.subject_sampler import SubjectBatchSampler
from extensions.datasets.dataio.physionet2026 import (
    ALL_EVENT_FRACTION_FIELDS,
    CAISR_FIELDS,
    DEFAULT_CHANNELS,
    PhysioNet2026Dataset,
    SubjectRecord,
    scan_physionet2026_subjects,
)

from .base import BenchmarkDataModule


LABEL_MODES: Tuple[str, ...] = ("sleep_stage", "cognitive_impairment", "sex", "age")

SEX_TO_LABEL = {"M": 0, "F": 1}


def _age_labels(items) -> List[float]:
    """Chronological age per row, NaN where the record has none.

    Brain age is a task, not a dataset: this cohort already carries `age`
    in every metadata row, so exposing it as a label mode is all that
    `probe --task brain_age` needs to run on the cohort's own embeddings.
    Missing ages stay NaN rather than becoming a sentinel age.
    """
    out: List[float] = []
    for item in items:
        age = item.get("age")
        out.append(float("nan") if age is None else float(age))
    return out


CI_TO_LABEL = {True: 1, False: 0}


def _collate_physionet2026(batch: List[Dict[str, object]], *, label_mode: str) -> Dict[str, object]:
    eeg = torch.stack([item["eeg"] for item in batch], dim=0)

    if label_mode == "age":
        labels = torch.tensor(_age_labels(batch), dtype=torch.float32)
    elif label_mode == "cognitive_impairment":
        labels = torch.tensor(
            [CI_TO_LABEL.get(item.get("cognitive_impairment"), -1) for item in batch],
            dtype=torch.long,
        )
    elif label_mode == "sex":
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
            "dataset": "physionet2026",
            "subject_id": item["subject_id"],
            "site_id": item["site_id"],
            "epoch_index": item["epoch_idx"],
            "sleep_stage": item["sleep_stage"],
            "channels": item["channels"],
            "sampling_rate": item.get("sampling_rate", 250.0),
            "epoch_seconds": item["epoch_seconds"],
            "location": item.get("location", ""),
            "has_human_annotations": item.get("has_human_annotations", False),
            "age": item.get("age"),
            "sex": item.get("sex"),
            "cognitive_impairment": item.get("cognitive_impairment"),
            "ci_label": CI_TO_LABEL.get(item.get("cognitive_impairment"), -1),
            "time_to_event": item.get("time_to_event"),
            "time_to_last_visit": item.get("time_to_last_visit"),
            "bmi": item.get("bmi"),
        }
        # Recording stats
        for key in ("recording_mean", "recording_std", "recording_q95", "recording_q95_bipolar"):
            if key in item:
                m[key] = item[key]
        # Human event fractions
        if item.get("arousal_fraction") is not None:
            m["arousal_fraction"] = item["arousal_fraction"]
        for field in ALL_EVENT_FRACTION_FIELDS:
            if item.get(field) is not None:
                m[field] = item[field]
        # CAISR metadata
        for field in CAISR_FIELDS:
            if item.get(field) is not None:
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
    return functools.partial(_collate_physionet2026, label_mode=label_mode)


def _patient_splits(
    all_records: Sequence[SubjectRecord],
    num_folds: int,
    fold: int,
    seed: int = 42,
) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
    """Patient-level k-fold split, stratified by (cognitive_impairment, site_id).

    Each (CI, site) group is shuffled independently with a deterministic
    seed and split into num_folds chunks. This ensures balanced CI/non-CI
    and site representation in every fold.
    """
    seen: Dict[str, SubjectRecord] = {}
    for r in all_records:
        if r.subject_id not in seen:
            seen[r.subject_id] = r

    by_group: Dict[Tuple, List[SubjectRecord]] = {}
    for r in seen.values():
        key = (r.cognitive_impairment, r.site_id)
        by_group.setdefault(key, []).append(r)

    train_records: List[SubjectRecord] = []
    val_records: List[SubjectRecord] = []
    test_records: List[SubjectRecord] = []

    rng = np.random.RandomState(seed)
    for group_key in sorted(by_group, key=str):
        records = by_group[group_key]
        indices = np.arange(len(records))
        rng.shuffle(indices)
        chunks = np.array_split(indices, num_folds)
        test_idx = set(chunks[fold % num_folds].tolist())
        val_idx = set(chunks[(fold + 1) % num_folds].tolist())
        for i, rec in enumerate(records):
            if i in test_idx:
                test_records.append(rec)
            elif i in val_idx:
                val_records.append(rec)
            else:
                train_records.append(rec)

    return train_records, val_records, test_records


# The shared wrapper; the local name is kept so call sites are unchanged.
from ._loader_adapters import ContractMetaLoader as _LoaderAdapter  # noqa: E402


class PhysioNet2026BenchmarkDataModule(BenchmarkDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
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
        require_human_annotations: bool = False,
        compute_recording_stats: bool = False,
        use_all_eeg_channels: bool = False,
        subset_filter: Optional[List[str]] = None,
    ) -> None:
        self._channel_names = list(channel_specs)

        if signal_kind != "raw":
            raise ValueError(
                f"PhysioNet2026 currently only supports signal_kind='raw', got {signal_kind!r}."
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
        if subset_filter:
            meta["subset_filter"] = sorted(subset_filter)
        super().__init__(name="physionet2026", metadata=meta)

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._fold = fold
        self._num_folds = num_folds
        self._label_mode = label_mode
        self._require_human_annotations = require_human_annotations
        self._collate = _make_collate(label_mode)
        self._compute_recording_stats = compute_recording_stats
        self._subset_filter = set(subset_filter) if subset_filter else None

        self._all_records = scan_physionet2026_subjects(data_root)

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

        self._train_ds: Optional[PhysioNet2026Dataset] = None
        self._val_ds: Optional[PhysioNet2026Dataset] = None
        self._test_ds: Optional[PhysioNet2026Dataset] = None
        self._all_ds: Optional[PhysioNet2026Dataset] = None

    def _get_split_records(self) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
        return _patient_splits(
            self._all_records, num_folds=self._num_folds, fold=self._fold,
        )

    def _get_train_ds(self) -> PhysioNet2026Dataset:
        if self._train_ds is None:
            train_records, _, _ = self._get_split_records()
            self._train_ds = PhysioNet2026Dataset(
                subject_records=train_records, **self._ds_kwargs,
            )
        return self._train_ds

    def _get_val_ds(self) -> PhysioNet2026Dataset:
        if self._val_ds is None:
            _, val_records, _ = self._get_split_records()
            self._val_ds = PhysioNet2026Dataset(
                subject_records=val_records, **self._ds_kwargs,
            )
        return self._val_ds

    def _get_test_ds(self) -> PhysioNet2026Dataset:
        if self._test_ds is None:
            _, _, test_records = self._get_split_records()
            self._test_ds = PhysioNet2026Dataset(
                subject_records=test_records, **self._ds_kwargs,
            )
        return self._test_ds

    def _get_all_ds(self) -> PhysioNet2026Dataset:
        if self._all_ds is None:
            self._all_ds = PhysioNet2026Dataset(
                subject_records=self._all_records,
                fold_assignments=self._fold_assignments,
                **self._ds_kwargs,
            )
        return self._all_ds

    def _make_loader(self, dataset: PhysioNet2026Dataset, shuffle: bool) -> _LoaderAdapter:
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
        return _LoaderAdapter(
            loader,
            unit="uV",
            channels=channels,
        )

    def train_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._get_train_ds(), shuffle=True)

    def val_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._get_val_ds(), shuffle=False)

    def test_dataloader(self) -> _LoaderAdapter:
        return self._make_loader(self._get_test_ds(), shuffle=False)

    # -- global embedding cache ------------------------------------------------

    def cache_context(self, purpose: str = "default") -> Dict[str, object]:
        context = dict(self.metadata)
        if purpose == "global_embeddings":
            context.pop("fold", None)
            context.pop("label_mode", None)
            context.pop("subset_filter", None)
        return context

    def supports_global_embedding_cache(self) -> bool:
        return True

    def full_embedding_dataloader(self) -> _LoaderAdapter:
        if self.embed_chunk is not None:
            from benchmarking_helpers.registry.contracts import chunk_records
            records = chunk_records(self._all_records, *self.embed_chunk)
            ds = PhysioNet2026Dataset(subject_records=records, fold_assignments=self._fold_assignments, **self._ds_kwargs)
            return self._make_loader(ds, shuffle=False)
        return self._make_loader(self._get_all_ds(), shuffle=False)

    def split_global_embedding_payload(
        self, payload: "EmbeddingPayload",
    ) -> Dict[str, "EmbeddingPayload"]:
        from benchmarking_helpers import EmbeddingPayload

        fold_key = f"fold_{self._fold}"

        def _subset(split_name: str) -> EmbeddingPayload:
            keep = [
                i for i, m in enumerate(payload.metadata)
                if m.get("fold_assignments", {}).get(fold_key) == split_name
            ]
            if self._subset_filter:
                keep = [
                    i for i in keep
                    if payload.metadata[i].get("site_id") in self._subset_filter
                ]
            if self._require_human_annotations:
                keep = [
                    i for i in keep
                    if payload.metadata[i].get("has_human_annotations", False)
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
            features = payload.features[idx]

            if self._label_mode == "age":
                labels = np.asarray(
                    _age_labels(payload.metadata[i] for i in keep),
                    dtype=np.float32,
                )
            elif self._label_mode == "cognitive_impairment":
                labels = np.asarray(
                    [CI_TO_LABEL.get(payload.metadata[i].get("cognitive_impairment"), -1) for i in keep],
                    dtype=np.int64,
                )
            elif self._label_mode == "sex":
                labels = np.asarray(
                    [SEX_TO_LABEL.get(payload.metadata[i].get("sex"), -1) for i in keep],
                    dtype=np.int64,
                )
            else:
                labels = np.asarray(
                    [int(payload.metadata[i]["sleep_stage"]) for i in keep],
                    dtype=np.int64,
                )

            return EmbeddingPayload(
                features=features,
                labels=labels,
                metadata=[payload.metadata[i] for i in keep],
            )

        return {
            "train": _subset("train"),
            "val": _subset("valid"),
            "test": _subset("test"),
        }
