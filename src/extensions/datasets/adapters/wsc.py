"""
WSC benchmark datamodule adapter.

Wraps :class:`WSCDataset` into the standard :class:`BenchmarkDataModule`
interface. Fold splitting is by subject_id (not recording) to prevent
data leakage across visits.
"""

from __future__ import annotations

import functools

from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarking_helpers.runtime.subject_sampler import SubjectBatchSampler
from extensions.datasets.dataio.wsc import (
    DEFAULT_CHANNELS,
    WSCDataset,
    SubjectRecord,
    scan_wsc_subjects,
)

from .base import BenchmarkDataModule


# ---------------------------------------------------------------------------
# Label modes
# ---------------------------------------------------------------------------

LABEL_MODES: Tuple[str, ...] = ("sleep_stage", "sex", "age")

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


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate_wsc(batch: List[Dict[str, object]], *, label_mode: str) -> Dict[str, object]:
    eeg = torch.stack([item["eeg"] for item in batch], dim=0)
    if label_mode == "age":
        labels = torch.tensor(_age_labels(batch), dtype=torch.float32)
    elif label_mode == "sex":
        labels = torch.tensor(
            [SEX_TO_LABEL.get(item.get("sex"), -1) for item in batch],
            dtype=torch.long,
        )
    else:
        labels = torch.tensor(
            [item["sleep_stage"] for item in batch], dtype=torch.long,
        )
    meta = []
    for item in batch:
        m = {
            "dataset": "wsc",
            "subject_id": item["subject_id"],
            "recording_id": item["recording_id"],
            "visit": item["visit"],
            "epoch_index": item["epoch_idx"],
            "sleep_stage": item["sleep_stage"],
            "channels": item["channels"],
            "sampling_rate": item.get("sampling_rate"),
            "epoch_seconds": item.get("epoch_seconds", 30.0),
            "age": item["age"],
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
    return functools.partial(_collate_wsc, label_mode=label_mode)


# ---------------------------------------------------------------------------
# Fold splitting (by subject_id, not recording)
# ---------------------------------------------------------------------------

def _balanced_subject_chunks(
    all_records: Sequence[SubjectRecord],
    num_folds: int,
    seed: int = 42,
) -> List[List[str]]:
    """Partition subject_ids into ``num_folds`` chunks balanced by recording count.

    Uses Longest-Processing-Time (LPT) greedy bin-packing: subjects are sorted
    by visit count descending, then each is assigned to the chunk with the
    currently smallest total recording count. This keeps recording counts
    approximately equal across folds even though subjects have 1–5 visits.
    """
    visits_by_subject: Dict[str, int] = {}
    for r in all_records:
        visits_by_subject[r.subject_id] = visits_by_subject.get(r.subject_id, 0) + 1

    subjects = sorted(visits_by_subject.keys())
    if num_folds > len(subjects):
        raise ValueError(
            f"num_folds={num_folds} exceeds unique subjects={len(subjects)}"
        )

    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    subjects.sort(key=lambda s: -visits_by_subject[s])  # stable, preserves shuffle within ties

    chunks: List[List[str]] = [[] for _ in range(num_folds)]
    loads = [0] * num_folds
    for sid in subjects:
        i = min(range(num_folds), key=lambda j: (loads[j], j))
        chunks[i].append(sid)
        loads[i] += visits_by_subject[sid]
    return chunks


def _patient_splits(
    all_records: Sequence[SubjectRecord],
    num_folds: int,
    fold: int,
    seed: int = 42,
) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
    """Patient-level k-fold split with balanced recording counts per fold.

    Splits by unique ``subject_id`` so all visits from the same subject go to
    the same fold (no data leakage). Chunks are sized by total recording count
    (LPT bin-packing) so train/val/test sizes stay consistent across folds.
    """
    chunks = _balanced_subject_chunks(all_records, num_folds, seed)
    test_subj = set(chunks[fold % num_folds])
    val_subj = set(chunks[(fold + 1) % num_folds])
    train_subj = set().union(*chunks) - test_subj - val_subj

    return (
        [r for r in all_records if r.subject_id in train_subj],
        [r for r in all_records if r.subject_id in val_subj],
        [r for r in all_records if r.subject_id in test_subj],
    )


# ---------------------------------------------------------------------------
# Loader adapter
# ---------------------------------------------------------------------------

# The shared wrapper; the local name is kept so call sites are unchanged.
from ._loader_adapters import ContractMetaLoader as _LoaderAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# BenchmarkDataModule
# ---------------------------------------------------------------------------

class WSCBenchmarkDataModule(BenchmarkDataModule):
    """BenchmarkDataModule for the WSC dataset.

    Parameters
    ----------
    data_root : str
        Path to the WSC polysomnography directory.
    csv_path : str
        Path to the WSC CSV with age/demographics.
    visits : list of int or None
        Which visits to include. None = all.
    batch_size : int
    fold : int
    num_folds : int
    num_workers : int
    channel_specs : str
    epoch_seconds : float
    bandpass : tuple or None
    notch : float or None
    signal_kind : str
        Optional amplitude range in microvolts for MODEL_CONTRACTS.md.
    compute_recording_stats : bool
        When *True*, compute per-recording mean / std / q95 statistics.
    """

    def __init__(
        self,
        data_root: str,
        csv_path: str = "${EEG_DATA_ROOT}/data/raw/wsc/datasets/wsc-dataset-0.8.0.csv",
        visits: Optional[Sequence[int]] = None,
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
    ) -> None:
        if signal_kind != "raw":
            raise ValueError(
                f"WSC currently only supports signal_kind='raw', got {signal_kind!r}."
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
        super().__init__(name="wsc", metadata=meta)

        self._channel_names = list(channel_specs)
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._fold = fold
        self._num_folds = num_folds
        self._label_mode = label_mode
        self._collate = _make_collate(label_mode)
        self._compute_recording_stats = compute_recording_stats

        self._all_records = scan_wsc_subjects(data_root, csv_path=csv_path, visits=visits)

        # Compute fold assignments by subject_id (not recording)
        self._fold_assignments: Dict[str, Dict[str, str]] = {}
        for f in range(num_folds):
            train_f, val_f, test_f = _patient_splits(self._all_records, num_folds=num_folds, fold=f)
            fold_key = f"fold_{f}"
            for r in train_f:
                self._fold_assignments.setdefault(r.subject_id, {})[fold_key] = "train"
            for r in val_f:
                self._fold_assignments.setdefault(r.subject_id, {})[fold_key] = "valid"
            for r in test_f:
                self._fold_assignments.setdefault(r.subject_id, {})[fold_key] = "test"
            print(
                f"[WSC folds] fold {f}: train={len(train_f)} val={len(val_f)} test={len(test_f)}"
            )

        self._ds_kwargs = dict(
            channel_specs=channel_specs,
            epoch_seconds=epoch_seconds,
            bandpass=bandpass,
            notch=notch,
            highpass=highpass,
            compute_recording_stats=compute_recording_stats,
            use_all_eeg_channels=use_all_eeg_channels,
        )

        # Datasets created lazily
        self._train_ds: Optional[WSCDataset] = None
        self._val_ds: Optional[WSCDataset] = None
        self._test_ds: Optional[WSCDataset] = None
        self._all_ds: Optional[WSCDataset] = None

    def _get_split_records(self) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
        return _patient_splits(self._all_records, num_folds=self._num_folds, fold=self._fold)

    def _get_train_ds(self) -> WSCDataset:
        if self._train_ds is None:
            train_records, _, _ = self._get_split_records()
            self._train_ds = WSCDataset(subject_records=train_records, **self._ds_kwargs)
        return self._train_ds

    def _get_val_ds(self) -> WSCDataset:
        if self._val_ds is None:
            _, val_records, _ = self._get_split_records()
            self._val_ds = WSCDataset(subject_records=val_records, **self._ds_kwargs)
        return self._val_ds

    def _get_test_ds(self) -> WSCDataset:
        if self._test_ds is None:
            _, _, test_records = self._get_split_records()
            self._test_ds = WSCDataset(subject_records=test_records, **self._ds_kwargs)
        return self._test_ds

    def _get_all_ds(self) -> WSCDataset:
        if self._all_ds is None:
            self._all_ds = WSCDataset(
                subject_records=self._all_records,
                fold_assignments=self._fold_assignments,
                **self._ds_kwargs,
            )
        return self._all_ds

    def _make_loader(self, dataset: WSCDataset, shuffle: bool) -> _LoaderAdapter:
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
            from benchmarking_helpers.registry.contracts import chunk_records
            records = chunk_records(self._all_records, *self.embed_chunk)
            ds = WSCDataset(subject_records=records, fold_assignments=self._fold_assignments, **self._ds_kwargs)
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
