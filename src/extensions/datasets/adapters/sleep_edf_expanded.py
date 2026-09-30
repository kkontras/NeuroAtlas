"""Sleep-EDF Expanded benchmark datamodule adapter."""

from __future__ import annotations

import functools

from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarking_helpers.runtime.subject_sampler import SubjectBatchSampler
from extensions.datasets.dataio.sleep_edf_expanded import (
    DEFAULT_CHANNELS,
    SleepEDFExpandedDataset,
    SubjectRecord,
    scan_sleep_edf_expanded_subjects,
)

from .base import BenchmarkDataModule


# ---------------------------------------------------------------------------
# Label modes
# ---------------------------------------------------------------------------

LABEL_MODES: Tuple[str, ...] = ("sleep_stage", "sex", "condition", "age")

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


CONDITION_TO_LABEL = {"placebo": 0, "temazepam": 1}


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate_sleep_edf_expanded(batch: List[Dict[str, object]], *, label_mode: str) -> Dict[str, object]:
    eeg = torch.stack([item["eeg"] for item in batch], dim=0)
    if label_mode == "age":
        labels = torch.tensor(_age_labels(batch), dtype=torch.float32)
    elif label_mode == "sex":
        labels = torch.tensor(
            [SEX_TO_LABEL.get(item.get("sex"), -1) for item in batch],
            dtype=torch.long,
        )
    elif label_mode == "condition":
        labels = torch.tensor(
            [CONDITION_TO_LABEL.get(item.get("condition"), -1) for item in batch],
            dtype=torch.long,
        )
    else:
        labels = torch.tensor(
            [item["sleep_stage"] for item in batch], dtype=torch.long,
        )
    meta: List[Dict[str, object]] = []
    for item in batch:
        m: Dict[str, object] = {
            "dataset": "sleep_edf_expanded",
            "subject_id": item["subject_id"],
            "recording_id": item["recording_id"],
            "subset": item["subset"],
            "night": item["night"],
            "condition": item["condition"],
            "location": item["location"],
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
    return functools.partial(_collate_sleep_edf_expanded, label_mode=label_mode)


# ---------------------------------------------------------------------------
# Fold splitting — subset-stratified, subject-grouped
# ---------------------------------------------------------------------------

def _patient_splits(
    all_records: Sequence[SubjectRecord],
    num_folds: int,
    fold: int,
    seed: int = 42,
) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
    """Patient-level k-fold split, stratified by subset (cassette/telemetry).

    All recordings from the same subject are always assigned to the same split.
    Each subset is shuffled and split independently, then merged, so every fold
    holds proportional cassette and telemetry representation.
    """
    seen: Dict[str, SubjectRecord] = {}
    for r in all_records:
        if r.subject_id not in seen:
            seen[r.subject_id] = r

    by_subset: Dict[str, List[SubjectRecord]] = {}
    for r in seen.values():
        by_subset.setdefault(r.subset, []).append(r)

    train_subjects: set[str] = set()
    val_subjects: set[str] = set()
    test_subjects: set[str] = set()

    rng = np.random.RandomState(seed)
    for subset_name in sorted(by_subset):
        records = by_subset[subset_name]
        indices = np.arange(len(records))
        rng.shuffle(indices)
        chunks = np.array_split(indices, num_folds)
        test_idx = set(chunks[fold % num_folds].tolist())
        val_idx = set(chunks[(fold + 1) % num_folds].tolist())
        for i, rec in enumerate(records):
            if i in test_idx:
                test_subjects.add(rec.subject_id)
            elif i in val_idx:
                val_subjects.add(rec.subject_id)
            else:
                train_subjects.add(rec.subject_id)

    train_records = [r for r in all_records if r.subject_id in train_subjects]
    val_records = [r for r in all_records if r.subject_id in val_subjects]
    test_records = [r for r in all_records if r.subject_id in test_subjects]

    return train_records, val_records, test_records


# ---------------------------------------------------------------------------
# Loader adapter
# ---------------------------------------------------------------------------

class _LoaderAdapter:
    """Wrap a DataLoader to inject MODEL_CONTRACTS.md metadata into each batch.

    ``meta[i]`` publishes ``sampling_rate``, ``unit`` and ``channels``
    per MODEL_CONTRACTS.md.  Sleep-EDF Expanded native
    sfreq is 100 Hz, unit is ``"uV"``.
    """

    def __init__(
        self,
        loader: DataLoader,
        *,
        sampling_rate: float,
        unit: str,
        channels: List[str],
    ) -> None:
        self._loader = loader
        self.dataset = loader.dataset
        self.batch_size = getattr(loader, "batch_size", None)
        self._num_workers = loader.num_workers
        self.batch_sampler = getattr(loader, "batch_sampler", None)
        self._sampling_rate = float(sampling_rate)
        self._unit = unit
        self._channels = list(channels)

        _raw_collate = loader.collate_fn
        _per: Dict[str, object] = {"sampling_rate": float(sampling_rate), "unit": unit, "channels": list(channels)}

        def _collate_with_meta(batch, _c=_raw_collate, _p=_per):
            result = _c(batch)
            for m in result["meta"]:
                m.update(_p)
            return result

        self.collate_fn = _collate_with_meta

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        for batch in self._loader:
            per_sample: Dict[str, object] = {
                "sampling_rate": self._sampling_rate,
                "unit": self._unit,
                "channels": list(self._channels),
            }
            for m in batch["meta"]:
                m.update(per_sample)
            yield batch


# ---------------------------------------------------------------------------
# BenchmarkDataModule
# ---------------------------------------------------------------------------

class SleepEDFExpandedBenchmarkDataModule(BenchmarkDataModule):
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
        compute_recording_stats: bool = False,
        use_all_eeg_channels: bool = False,
        subset_filter: Optional[List[str]] = None,
        folds_manifest: Optional[str] = None,
        strict_folds: bool = True,
    ) -> None:
        self._channel_names = list(channel_specs)

        if signal_kind != "raw":
            raise ValueError(
                f"SleepEDF Expanded currently only supports signal_kind='raw', got {signal_kind!r}."
            )
        if label_mode not in LABEL_MODES:
            raise ValueError(
                f"label_mode must be one of {LABEL_MODES}, got {label_mode!r}."
            )
        canonical = (
            ["placebo", "temazepam"] if label_mode == "condition"
            else ["W", "N1", "N2", "N3", "REM"]
        )
        meta: Dict[str, object] = {
            "canonical_label_space": canonical,
            "epoch_seconds": epoch_seconds,
            "channel_policy": ["eeg"],
            "signal_kind": signal_kind,
            "fold": fold,
            "num_folds": num_folds,
            "subsets": ["cassette", "telemetry"],
            "label_mode": label_mode,
        }
        if subset_filter:
            meta["subset_filter"] = sorted(subset_filter)
        super().__init__(name="sleep_edf_expanded", metadata=meta)

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._fold = fold
        self._num_folds = num_folds
        self._label_mode = label_mode
        self._collate = _make_collate(label_mode)
        self._compute_recording_stats = compute_recording_stats
        self._subset_filter = set(subset_filter) if subset_filter else None
        self._folds_manifest = folds_manifest
        self._strict_folds = strict_folds
        if folds_manifest:
            meta["folds_manifest"] = folds_manifest

        self._all_records = scan_sleep_edf_expanded_subjects(data_root)

        self._fold_assignments: Dict[str, Dict[str, str]] = {}
        for f in range(num_folds):
            train_f, val_f, test_f = self._splits_for_fold(f)
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

        self._train_ds: Optional[SleepEDFExpandedDataset] = None
        self._val_ds: Optional[SleepEDFExpandedDataset] = None
        self._test_ds: Optional[SleepEDFExpandedDataset] = None
        self._all_ds: Optional[SleepEDFExpandedDataset] = None

    def _splits_for_fold(
        self, fold: int,
    ) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
        """Split records for ``fold``, from the frozen manifest when one is named.

        Falls back to the built-in subject-grouped ``_patient_splits`` when
        ``folds_manifest`` is None, preserving the historical behaviour.
        """
        if not self._folds_manifest:
            return _patient_splits(
                self._all_records, num_folds=self._num_folds, fold=fold,
            )

        from benchmarking_helpers.registry.fold_manifest import (
            load_fold_split,
            recording_indices_for_split,
        )

        split = load_fold_split(self._folds_manifest, fold)
        recs = recording_indices_for_split(
            [r.recording_id for r in self._all_records], split,
            strict=self._strict_folds,
        )
        return ([self._all_records[i] for i in recs["train"]],
                [self._all_records[i] for i in recs["val"]],
                [self._all_records[i] for i in recs["test"]])

    def _get_split_records(self) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
        return self._splits_for_fold(self._fold)

    def _get_train_ds(self) -> SleepEDFExpandedDataset:
        if self._train_ds is None:
            train_records, _, _ = self._get_split_records()
            self._train_ds = SleepEDFExpandedDataset(subject_records=train_records, **self._ds_kwargs)
        return self._train_ds

    def _get_val_ds(self) -> SleepEDFExpandedDataset:
        if self._val_ds is None:
            _, val_records, _ = self._get_split_records()
            self._val_ds = SleepEDFExpandedDataset(subject_records=val_records, **self._ds_kwargs)
        return self._val_ds

    def _get_test_ds(self) -> SleepEDFExpandedDataset:
        if self._test_ds is None:
            _, _, test_records = self._get_split_records()
            self._test_ds = SleepEDFExpandedDataset(subject_records=test_records, **self._ds_kwargs)
        return self._test_ds

    def _get_all_ds(self) -> SleepEDFExpandedDataset:
        if self._all_ds is None:
            self._all_ds = SleepEDFExpandedDataset(
                subject_records=self._all_records,
                fold_assignments=self._fold_assignments,
                **self._ds_kwargs,
            )
        return self._all_ds

    def _make_loader(self, dataset: SleepEDFExpandedDataset, shuffle: bool) -> _LoaderAdapter:
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
        native_sfreq = 100.0
        return _LoaderAdapter(
            loader,
            sampling_rate=native_sfreq,
            unit="uV",
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
            context.pop("subset_filter", None)
            context["canonical_label_space"] = ["W", "N1", "N2", "N3", "REM"]
        return context

    def supports_global_embedding_cache(self) -> bool:
        return True

    def full_embedding_dataloader(self) -> _LoaderAdapter:
        if self.embed_chunk is not None:
            from benchmarking_helpers.registry.contracts import chunk_records
            records = chunk_records(self._all_records, *self.embed_chunk)
            ds = SleepEDFExpandedDataset(subject_records=records, fold_assignments=self._fold_assignments, **self._ds_kwargs)
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
                    if payload.metadata[i].get("subset") in self._subset_filter
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
            if self._label_mode == "age":
                labels = np.asarray(
                    _age_labels(payload.metadata[i] for i in keep),
                    dtype=np.float32,
                )
            elif self._label_mode == "condition":
                subj_condition: Dict[str, str] = {
                    r.subject_id: r.condition for r in self._all_records
                }
                def _get_condition(i: int) -> str:
                    m = payload.metadata[i]
                    cond = m.get("condition")
                    if cond is not None and cond in CONDITION_TO_LABEL:
                        return cond
                    return subj_condition.get(m.get("subject_id", ""), "")
                labels = np.asarray(
                    [CONDITION_TO_LABEL.get(_get_condition(i), -1) for i in keep],
                    dtype=labels.dtype,
                )
                valid = labels >= 0
                if not valid.all():
                    idx = idx[valid]
                    labels = labels[valid]
                    keep = [keep[j] for j in range(len(keep)) if valid[j]]
            elif self._label_mode == "sex":
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
