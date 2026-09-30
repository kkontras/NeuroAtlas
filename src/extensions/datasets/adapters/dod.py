"""DOD benchmark datamodule adapter."""

from __future__ import annotations

import functools

from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarking_helpers.runtime.subject_sampler import SubjectBatchSampler
from extensions.datasets.dataio.dod import (
    DEFAULT_CHANNELS,
    DOD_GROUP_TO_OSA_LABEL,
    DODDataset,
    SubjectRecord,
    scan_dod_subjects,
)

from .base import BenchmarkDataModule


LABEL_MODES = (
    "sleep_stage",
    "osa_group",
    "scorer_1",
    "scorer_2",
    "scorer_3",
    "scorer_4",
    "scorer_5",
)


def _collate_dod(batch: List[Dict[str, object]], *, label_mode: str) -> Dict[str, object]:
    eeg = torch.stack([item["eeg"] for item in batch], dim=0)
    if label_mode == "osa_group":
        labels = torch.tensor(
            [DOD_GROUP_TO_OSA_LABEL[item["group"]] for item in batch],
            dtype=torch.long,
        )
    elif label_mode.startswith("scorer_"):
        labels = torch.tensor(
            [
                int((item.get("expert_stages") or {}).get(label_mode, -1))
                for item in batch
            ],
            dtype=torch.long,
        )
    else:
        labels = torch.tensor(
            [item["sleep_stage"] for item in batch], dtype=torch.long,
        )
    meta: List[Dict[str, object]] = []
    for item in batch:
        m: Dict[str, object] = {
            "dataset": "dod",
            "subject_id": item["subject_id"],
            "group": item["group"],
            "osa_label": DOD_GROUP_TO_OSA_LABEL[item["group"]],
            "epoch_index": item["epoch_idx"],
            "sleep_stage": item["sleep_stage"],
            "channels": item["channels"],
            "epoch_seconds": item["epoch_seconds"],
        }
        for key in ("recording_mean", "recording_std", "recording_q95", "recording_q95_bipolar"):
            if key in item:
                m[key] = item[key]
        if "expert_stages" in item:
            m["expert_stages"] = item["expert_stages"]
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
    return functools.partial(_collate_dod, label_mode=label_mode)


def _patient_splits(
    all_records: Sequence[SubjectRecord],
    num_folds: int,
    fold: int,
    seed: int = 42,
) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
    """Patient-level k-fold split, stratified by DOD group (dodh/dodo).

    Each group is shuffled independently with a deterministic seed and
    split into ``num_folds`` chunks. Fold *k* takes chunk *k* from each
    group as the test set and chunk *(k + 1) mod num_folds* as the
    validation set, with the remaining chunks forming the training set.
    Every fold therefore holds ~1/num_folds of each group's subjects,
    preserving the overall ~31 % / 69 % dodh/dodo balance.
    """
    seen: Dict[str, SubjectRecord] = {}
    for r in all_records:
        if r.subject_id not in seen:
            seen[r.subject_id] = r

    by_group: Dict[str, List[SubjectRecord]] = {}
    for r in seen.values():
        by_group.setdefault(r.group, []).append(r)

    train_records: List[SubjectRecord] = []
    val_records: List[SubjectRecord] = []
    test_records: List[SubjectRecord] = []

    rng = np.random.RandomState(seed)
    for group_name in sorted(by_group):
        records = by_group[group_name]
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


class DODBenchmarkDataModule(BenchmarkDataModule):
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
        notch_map: Optional[Dict[str, float]] = None,
        signal_kind: str = "raw",
        label_mode: str = "sleep_stage",
        expert_scorers_root: Optional[str] = None,
        compute_recording_stats: bool = False,
        use_all_eeg_channels: bool = False,
        subset_filter: Optional[List[str]] = None,
    ) -> None:
        if signal_kind != "raw":
            raise ValueError(
                f"DOD currently only supports signal_kind='raw', got {signal_kind!r}."
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
        super().__init__(name="dod", metadata=meta)

        self._channel_names = list(channel_specs)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._fold = fold
        self._num_folds = num_folds
        self._label_mode = label_mode
        self._collate = _make_collate(label_mode)
        self._compute_recording_stats = compute_recording_stats
        self._subset_filter = set(subset_filter) if subset_filter else None

        self._all_records = scan_dod_subjects(data_root)

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
            notch_map=notch_map,
            expert_scorers_root=expert_scorers_root,
            compute_recording_stats=compute_recording_stats,
            use_all_eeg_channels=use_all_eeg_channels,
        )

        self._train_ds: Optional[DODDataset] = None
        self._val_ds: Optional[DODDataset] = None
        self._test_ds: Optional[DODDataset] = None
        self._all_ds: Optional[DODDataset] = None

    def _get_split_records(self) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
        return _patient_splits(
            self._all_records, num_folds=self._num_folds, fold=self._fold,
        )

    def _get_train_ds(self) -> DODDataset:
        if self._train_ds is None:
            train_records, _, _ = self._get_split_records()
            self._train_ds = DODDataset(subject_records=train_records, **self._ds_kwargs)
        return self._train_ds

    def _get_val_ds(self) -> DODDataset:
        if self._val_ds is None:
            _, val_records, _ = self._get_split_records()
            self._val_ds = DODDataset(subject_records=val_records, **self._ds_kwargs)
        return self._val_ds

    def _get_test_ds(self) -> DODDataset:
        if self._test_ds is None:
            _, _, test_records = self._get_split_records()
            self._test_ds = DODDataset(subject_records=test_records, **self._ds_kwargs)
        return self._test_ds

    def _get_all_ds(self) -> DODDataset:
        if self._all_ds is None:
            self._all_ds = DODDataset(
                subject_records=self._all_records,
                fold_assignments=self._fold_assignments,
                **self._ds_kwargs,
            )
        return self._all_ds

    def _make_loader(self, dataset: DODDataset, shuffle: bool) -> _LoaderAdapter:
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
        return _LoaderAdapter(
            loader,
            sampling_rate=250.0,
            unit="uV",
            channels=self._channel_names,
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
            context.pop("subset_filter", None)
        return context

    def supports_global_embedding_cache(self) -> bool:
        return True

    def full_embedding_dataloader(self) -> _LoaderAdapter:
        if self.embed_chunk is not None:
            from benchmarking_helpers.registry.contracts import chunk_records
            records = chunk_records(self._all_records, *self.embed_chunk)
            ds = DODDataset(subject_records=records, fold_assignments=self._fold_assignments, **self._ds_kwargs)
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
                    if payload.metadata[i].get("group") in self._subset_filter
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
            labels = payload.labels[idx]
            # Re-apply label_mode so one cached embedding set serves every
            # probe configuration (consensus, per-expert, OSA).
            if self._label_mode == "osa_group":
                labels = np.asarray(
                    [DOD_GROUP_TO_OSA_LABEL[payload.metadata[i]["group"]] for i in keep],
                    dtype=labels.dtype,
                )
            elif self._label_mode.startswith("scorer_"):
                first_meta = payload.metadata[keep[0]] if keep else {}
                if "expert_stages" in first_meta:
                    labels = np.asarray(
                        [
                            int(
                                (payload.metadata[i].get("expert_stages") or {}).get(
                                    self._label_mode, -1
                                )
                            )
                            for i in keep
                        ],
                        dtype=labels.dtype,
                    )
                else:
                    all_ds = self._get_all_ds()
                    all_ds._ensure_index_built()
                    scorer = self._label_mode
                    raw_labels = []
                    for i in keep:
                        m = payload.metadata[i]
                        subj_experts = all_ds._expert_stages.get(m["subject_id"])
                        if subj_experts and scorer in subj_experts:
                            arr = subj_experts[scorer]
                            ep = m["epoch_index"]
                            raw_labels.append(int(arr[ep]) if ep < len(arr) else -1)
                        else:
                            raw_labels.append(-1)
                    labels = np.asarray(raw_labels, dtype=labels.dtype)
            else:
                labels = np.asarray(
                    [int(payload.metadata[i]["sleep_stage"]) for i in keep],
                    dtype=labels.dtype,
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
