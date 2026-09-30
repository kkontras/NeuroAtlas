"""ISRUC benchmark datamodule adapter.

Thin wrapper around :class:`ISRUCDataset` that exposes the canonical
train/val/test loaders plus a global-embedding cache path. Subgroup-
stratified patient-level fold splitting mirrors MASS's subset
stratification; the ``label_mode`` knob and cache-drop pattern mirror
DOD so that one embedding cache serves consensus-probing and per-expert
probing without rerunning the backbone.
"""

from __future__ import annotations

import functools

from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarking_helpers.runtime.subject_sampler import SubjectBatchSampler
from extensions.datasets.dataio.isruc import (
    DEFAULT_CHANNELS,
    DIAGNOSIS_TO_LABEL,
    ISRUC_SUBGROUPS,
    ISRUCDataset,
    SubjectRecord,
    scan_isruc_subjects,
)

from .base import BenchmarkDataModule


# ---------------------------------------------------------------------------
# Label modes
# ---------------------------------------------------------------------------

LABEL_MODES: Tuple[str, ...] = ("sleep_stage", "diagnosis", "has_diagnosis", "scorer_1", "scorer_2", "sex", "age")

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

def _collate_isruc(batch: List[Dict[str, object]], *, label_mode: str) -> Dict[str, object]:
    eeg = torch.stack([item["eeg"] for item in batch], dim=0)
    if label_mode == "age":
        labels = torch.tensor(_age_labels(batch), dtype=torch.float32)
    elif label_mode == "diagnosis":
        labels = torch.tensor(
            [DIAGNOSIS_TO_LABEL.get(item.get("diagnosis", ""), -1) for item in batch],
            dtype=torch.long,
        )
    elif label_mode == "has_diagnosis":
        labels = torch.tensor(
            [0 if item.get("diagnosis") == "healthy" else 1 for item in batch],
            dtype=torch.long,
        )
    elif label_mode == "sex":
        labels = torch.tensor(
            [SEX_TO_LABEL.get(item.get("sex"), -1) for item in batch],
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
            "dataset": "isruc",
            "subject_id": item["subject_id"],
            "subgroup": item["subgroup"],
            "diagnosis": item.get("diagnosis"),
            "diagnosis_label": DIAGNOSIS_TO_LABEL.get(item.get("diagnosis", ""), -1),
            "recording": item["recording"],
            "recording_id": item["recording_id"],
            "epoch_index": item["epoch_idx"],
            "sleep_stage": item["sleep_stage"],
            "channels": item["channels"],
            "epoch_seconds": item["epoch_seconds"],
            "age": item.get("age"),
            "sex": item.get("sex"),
            "comorbidities": item.get("comorbidities"),
            "medication": item.get("medication"),
            "eeg_alterations": item.get("eeg_alterations"),
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
    return functools.partial(_collate_isruc, label_mode=label_mode)


# ---------------------------------------------------------------------------
# Fold splitting — stratified per subgroup, patient-level
# ---------------------------------------------------------------------------

def _patient_splits(
    all_records: Sequence[SubjectRecord],
    num_folds: int,
    fold: int,
    seed: int = 42,
) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
    """Subgroup-stratified, diagnosis-stratified patient-level k-fold split.

    Within each subgroup, subjects are split using
    ``sklearn.model_selection.StratifiedKFold`` on their diagnosis label
    so every fold sees all diagnosis classes present in that subgroup.
    If a subgroup has only one diagnosis class (e.g. SG-III: all healthy),
    falls back to plain shuffled splitting.

    Subgroup II's two recordings per subject follow the subject into
    whichever split they land in.
    """
    from sklearn.model_selection import StratifiedKFold

    seen: Dict[str, SubjectRecord] = {}
    for r in all_records:
        if r.subject_id not in seen:
            seen[r.subject_id] = r

    by_subgroup: Dict[str, List[SubjectRecord]] = {}
    for r in seen.values():
        by_subgroup.setdefault(r.subgroup, []).append(r)

    train_subj: set[str] = set()
    val_subj: set[str] = set()
    test_subj: set[str] = set()

    for sg in sorted(by_subgroup):
        records = sorted(by_subgroup[sg], key=lambda r: r.subject_id)
        diagnoses = [r.diagnosis or "unknown" for r in records]
        class_counts = Counter(diagnoses)
        min_class_count = min(class_counts.values())

        if len(class_counts) > 1 and min_class_count >= num_folds:
            skf = StratifiedKFold(
                n_splits=num_folds, shuffle=True, random_state=seed,
            )
            all_folds = list(skf.split(records, diagnoses))
            test_idx = set(all_folds[fold % num_folds][1].tolist())
            val_idx = set(all_folds[(fold + 1) % num_folds][1].tolist())
        else:
            rng = np.random.RandomState(seed)
            indices = np.arange(len(records))
            rng.shuffle(indices)
            chunks = np.array_split(indices, num_folds)
            test_idx = set(chunks[fold % num_folds].tolist())
            val_idx = set(chunks[(fold + 1) % num_folds].tolist())

        for i, rec in enumerate(records):
            if i in test_idx:
                test_subj.add(rec.subject_id)
            elif i in val_idx:
                val_subj.add(rec.subject_id)
            else:
                train_subj.add(rec.subject_id)

    train_records: List[SubjectRecord] = []
    val_records: List[SubjectRecord] = []
    test_records: List[SubjectRecord] = []
    for r in all_records:
        if r.subject_id in train_subj:
            train_records.append(r)
        elif r.subject_id in val_subj:
            val_records.append(r)
        elif r.subject_id in test_subj:
            test_records.append(r)

    return train_records, val_records, test_records


# ---------------------------------------------------------------------------
# Loader adapter
# ---------------------------------------------------------------------------

# The shared wrapper; the local name is kept so call sites are unchanged.
from ._loader_adapters import ContractMetaLoader as _LoaderAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# BenchmarkDataModule
# ---------------------------------------------------------------------------

class ISRUCBenchmarkDataModule(BenchmarkDataModule):
    """BenchmarkDataModule for the ISRUC-SLEEP dataset."""

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
        subgroups: Sequence[str] = ISRUC_SUBGROUPS,
        label_mode: str = "sleep_stage",
        min_diagnosis_prevalence: Optional[float] = None,
        diagnosis_filter: Optional[List[str]] = None,
        compute_recording_stats: bool = False,
        use_all_eeg_channels: bool = False,
        subset_filter: Optional[List[str]] = None,
    ) -> None:
        self._channel_names = list(channel_specs)

        if signal_kind != "raw":
            raise ValueError(
                f"ISRUC currently only supports signal_kind='raw', got {signal_kind!r}."
            )
        if label_mode not in LABEL_MODES:
            raise ValueError(
                f"label_mode must be one of {LABEL_MODES}, got {label_mode!r}."
            )
        subgroups = tuple(subgroups)
        meta: Dict[str, object] = {
            "canonical_label_space": ["W", "N1", "N2", "N3", "REM"],
            "epoch_seconds": epoch_seconds,
            "channel_policy": ["eeg"],
            "signal_kind": signal_kind,
            "fold": fold,
            "num_folds": num_folds,
            "subgroups": list(subgroups),
            "label_mode": label_mode,
        }
        if subset_filter:
            meta["subset_filter"] = sorted(subset_filter)
        # diagnosis_filter intentionally NOT added to meta — it only affects
        # label assignment at probe time, not extraction, so the full cache is reused.
        super().__init__(name="isruc", metadata=meta)

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._fold = fold
        self._num_folds = num_folds
        self._subgroups = subgroups
        self._label_mode = label_mode
        self._collate = _make_collate(label_mode)
        self._compute_recording_stats = compute_recording_stats
        self._subset_filter = set(subset_filter) if subset_filter else None

        self._all_records = scan_isruc_subjects(data_root, subgroups=subgroups)

        # Prevalence filtering for multi-class diagnosis.
        self._filtered_diagnosis_map: Optional[Dict[str, int]] = None
        self._allowed_diagnoses: Optional[Set[str]] = None
        if min_diagnosis_prevalence is not None and label_mode == "diagnosis":
            subj_diag: Dict[str, Optional[str]] = {}
            for r in self._all_records:
                if r.subject_id not in subj_diag:
                    subj_diag[r.subject_id] = r.diagnosis
            counts = Counter(subj_diag.values())
            total = len(subj_diag)
            self._allowed_diagnoses = {
                d for d, c in counts.items()
                if d is not None and c / total >= min_diagnosis_prevalence
            }
            self._all_records = [
                r for r in self._all_records
                if r.diagnosis in self._allowed_diagnoses
            ]
            self._filtered_diagnosis_map = {
                d: i for i, d in enumerate(
                    sorted(self._allowed_diagnoses, key=lambda d: DIAGNOSIS_TO_LABEL.get(d, 999))
                )
            }

        if diagnosis_filter is not None and label_mode == "diagnosis":
            valid_names = set(DIAGNOSIS_TO_LABEL.keys())
            for d in diagnosis_filter:
                if d not in valid_names:
                    raise ValueError(
                        f"Unknown diagnosis {d!r} in diagnosis_filter. "
                        f"Valid: {sorted(valid_names)}"
                    )
            self._allowed_diagnoses = set(diagnosis_filter)
            self._filtered_diagnosis_map = {
                d: i for i, d in enumerate(
                    sorted(self._allowed_diagnoses, key=lambda d: DIAGNOSIS_TO_LABEL.get(d, 999))
                )
            }

        # Compute fold assignments once for every fold.
        self._fold_assignments: Dict[str, Dict[str, str]] = {}
        for f in range(num_folds):
            train_f, val_f, test_f = _patient_splits(
                self._all_records, num_folds=num_folds, fold=f,
            )
            fold_key = f"fold_{f}"
            for rec in train_f:
                self._fold_assignments.setdefault(rec.subject_id, {})[fold_key] = "train"
            for rec in val_f:
                self._fold_assignments.setdefault(rec.subject_id, {})[fold_key] = "valid"
            for rec in test_f:
                self._fold_assignments.setdefault(rec.subject_id, {})[fold_key] = "test"

        # Startup sanity print — recording counts, not subject counts, so the
        # log reflects what each split will actually load.
        for f in range(num_folds):
            fold_key = f"fold_{f}"
            tr = sum(1 for r in self._all_records
                     if self._fold_assignments.get(r.subject_id, {}).get(fold_key) == "train")
            va = sum(1 for r in self._all_records
                     if self._fold_assignments.get(r.subject_id, {}).get(fold_key) == "valid")
            te = sum(1 for r in self._all_records
                     if self._fold_assignments.get(r.subject_id, {}).get(fold_key) == "test")
            print(f"[ISRUC folds] fold {f}: train={tr} val={va} test={te}")

        self._ds_kwargs = dict(
            channel_specs=channel_specs,
            epoch_seconds=epoch_seconds,
            bandpass=bandpass,
            notch=notch,
            highpass=highpass,
            compute_recording_stats=compute_recording_stats,
            use_all_eeg_channels=use_all_eeg_channels,
        )

        self._train_ds: Optional[ISRUCDataset] = None
        self._val_ds: Optional[ISRUCDataset] = None
        self._test_ds: Optional[ISRUCDataset] = None
        self._all_ds: Optional[ISRUCDataset] = None

    def _get_split_records(self) -> Tuple[List[SubjectRecord], List[SubjectRecord], List[SubjectRecord]]:
        return _patient_splits(
            self._all_records, num_folds=self._num_folds, fold=self._fold,
        )

    def _get_train_ds(self) -> ISRUCDataset:
        if self._train_ds is None:
            train_records, _, _ = self._get_split_records()
            self._train_ds = ISRUCDataset(subject_records=train_records, **self._ds_kwargs)
        return self._train_ds

    def _get_val_ds(self) -> ISRUCDataset:
        if self._val_ds is None:
            _, val_records, _ = self._get_split_records()
            self._val_ds = ISRUCDataset(subject_records=val_records, **self._ds_kwargs)
        return self._val_ds

    def _get_test_ds(self) -> ISRUCDataset:
        if self._test_ds is None:
            _, _, test_records = self._get_split_records()
            self._test_ds = ISRUCDataset(subject_records=test_records, **self._ds_kwargs)
        return self._test_ds

    def _get_all_ds(self) -> ISRUCDataset:
        if self._all_ds is None:
            self._all_ds = ISRUCDataset(
                subject_records=self._all_records,
                fold_assignments=self._fold_assignments,
                **self._ds_kwargs,
            )
        return self._all_ds

    def _make_loader(self, dataset: ISRUCDataset, shuffle: bool) -> _LoaderAdapter:
        dataset._ensure_index_built()
        # ISRUC indexes by recording_id (not subject_id) because subgroup II
        # has two recordings per subject with potentially different stats.
        recording_ids = [dataset._index[i][0] for i in range(len(dataset))]
        sampler = SubjectBatchSampler(
            recording_ids, batch_size=self._batch_size, shuffle=shuffle,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate,
            pin_memory=False,
        )
        channels = self._channel_names
        native_sfreq = 200.0
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
        return context

    def supports_global_embedding_cache(self) -> bool:
        return True

    def full_embedding_dataloader(self) -> _LoaderAdapter:
        if self.embed_chunk is not None:
            from benchmarking_helpers.registry.contracts import chunk_records
            records = chunk_records(self._all_records, *self.embed_chunk)
            ds = ISRUCDataset(subject_records=records, fold_assignments=self._fold_assignments, **self._ds_kwargs)
            return self._make_loader(ds, shuffle=False)
        return self._make_loader(self._get_all_ds(), shuffle=False)

    def split_global_embedding_payload(
        self, payload: "EmbeddingPayload",
    ) -> Dict[str, "EmbeddingPayload"]:
        from benchmarking_helpers import EmbeddingPayload

        fold_key = f"fold_{self._fold}"
        diag_map = self._filtered_diagnosis_map
        allowed = self._allowed_diagnoses

        rec_diagnosis: Dict[str, Optional[str]] = {
            r.recording_id: r.diagnosis for r in self._all_records
        }

        def _get_diagnosis(m: Dict[str, Any]) -> Optional[str]:
            d = m.get("diagnosis")
            if d is not None:
                return d
            return rec_diagnosis.get(m.get("recording_id", ""))

        def _subset(split_name: str) -> EmbeddingPayload:
            keep = [
                i for i, m in enumerate(payload.metadata)
                if m.get("fold_assignments", {}).get(fold_key) == split_name
            ]
            if self._subset_filter:
                keep = [
                    i for i in keep
                    if payload.metadata[i].get("subgroup") in self._subset_filter
                ]
            if allowed is not None and self._label_mode == "diagnosis":
                keep = [
                    i for i in keep
                    if _get_diagnosis(payload.metadata[i]) in allowed
                ]
            if not keep:
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
            if self._label_mode == "age":
                labels = np.asarray(
                    _age_labels(payload.metadata[i] for i in keep),
                    dtype=np.float32,
                )
            elif self._label_mode == "diagnosis":
                label_map = diag_map if diag_map is not None else DIAGNOSIS_TO_LABEL
                labels = np.asarray(
                    [
                        label_map.get(
                            _get_diagnosis(payload.metadata[i]) or "", -1
                        )
                        for i in keep
                    ],
                    dtype=labels.dtype,
                )
            elif self._label_mode == "has_diagnosis":
                labels = np.asarray(
                    [
                        0 if _get_diagnosis(payload.metadata[i]) == "healthy" else 1
                        for i in keep
                    ],
                    dtype=labels.dtype,
                )
            elif self._label_mode == "sex":
                labels = np.asarray(
                    [
                        SEX_TO_LABEL.get(payload.metadata[i].get("sex"), -1)
                        for i in keep
                    ],
                    dtype=labels.dtype,
                )
            elif self._label_mode.startswith("scorer_"):
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
