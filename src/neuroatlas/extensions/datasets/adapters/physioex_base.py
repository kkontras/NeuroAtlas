"""Generic benchmark adapter for any physioex BasePhysioDataset.

Wraps a physioex dataset (imported read-only via ``sys.path``) and converts
its output to the EEGBenchmarks batch contract. Subclasses only need to
specify the dataset class, EEG channel requests, and channel-to-standard
name mapping.
"""
from __future__ import annotations

import logging
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from neuroatlas.benchmarking_helpers.registry.contracts import BenchmarkDataModule
from neuroatlas.extensions.datasets.pipelines import PIPELINE_REGISTRY

logger = logging.getLogger(__name__)


_CANONICAL_LABELS = ["W", "N1", "N2", "N3", "REM"]


def _ensure_physioex():
    pass  # No-op: physioex_data is now bundled in the repo.


def _strip_to_standard(name: str) -> str:
    """Best-effort normalisation of EEG channel names to standard 10-20 labels.

    Examples:
        'EEG C3-M2' → 'C3'     (prefix + dash ref)
        'EEG_C3-A2' → 'C3'     (underscore prefix + dash ref)
        'C3-M2'     → 'C3'     (dash ref)
        'C3_M2'     → 'C3'     (underscore ref)
        'C3M2'      → 'C3'     (STAGES BOGN: no separator)
        'F4M1'      → 'F4'     (STAGES BOGN: no separator)
        'C3'        → 'C3'     (already standard)
    """
    name = name.strip()
    # Remove common prefixes (space or underscore separated)
    for prefix in ("EEG ", "EEG_", "EOG ", "EOG_", "EMG ", "EMG_", "ECG ", "ECG_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Take part before reference (dash or underscore)
    parts = re.split(r"[-_]", name)
    name = parts[0] if parts else name
    # Handle bare concatenated refs like C4M1, F3M2, O2M1 (no separator)
    # Pattern: electrode (letter + digit(s)) + reference (letter + digit(s))
    m = re.match(r"^([A-Za-z]+\d+)[A-Za-z]+\d+$", name)
    if m:
        name = m.group(1)
    return name


# ---------------------------------------------------------------------------
# Per-split dataset with label filtering
# ---------------------------------------------------------------------------

class _PhysioExSplitDataset(Dataset):
    """View over a physioex dataset restricted to specific subjects,
    excluding epochs with invalid labels (label == -1)."""

    def __init__(self, physio_ds, subject_ids: Sequence[str]):
        self._ds = physio_ds
        self._indices = self._build_valid_indices(subject_ids)

    def _build_valid_indices(self, subject_ids: Sequence[str]) -> List[int]:
        id_set = set(subject_ids)
        valid: List[int] = []
        for sid, start, end in self._ds._subject_ranges:
            if sid not in id_set:
                continue
            spec = next(s for s in self._ds._subjects if s.subject_id == sid)
            labels = self._ds._get_labels(spec)
            n_items = end - start
            for local_idx in range(n_items):
                if local_idx < len(labels) and 0 <= int(labels[local_idx]) <= 4:
                    valid.append(start + local_idx)
        return valid

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._ds[self._indices[idx]]


# ---------------------------------------------------------------------------
# Collation: physioex → benchmark batch contract
# ---------------------------------------------------------------------------

def _collate_physioex(batch: List[Dict[str, Any]],
                      to_standard: Optional[Dict[str, str]] = None,
                      dataset_name: str = "unknown") -> Dict[str, Any]:
    """Convert physioex per-item dicts to EEGBenchmarks batch contract."""
    channel_order = batch[0]["channel_order"]

    signals_list, labels_list, meta_list = [], [], []

    for item in batch:
        channels = []
        for ch_key in channel_order:
            sig = item["signals"][ch_key]
            if sig.ndim == 2:
                sig = sig[0]
            channels.append(sig)
        stacked = torch.stack(channels, dim=0)
        signals_list.append(stacked)

        label = item["labels"]
        if label.ndim >= 1:
            label = label[0]
        labels_list.append(label)

        subject_info = item.get("subject", {})
        if to_standard:
            std_names = [to_standard.get(ch, _strip_to_standard(ch)) for ch in channel_order]
        else:
            std_names = [_strip_to_standard(ch) for ch in channel_order]

        # Read sampling rate and unit from physioex channel_info
        ch_info = item.get("channel_info", {})
        first_ch_info = ch_info.get(channel_order[0], {}) if ch_info else {}
        fs_out = first_ch_info.get("fs_out", None)
        # physioex converts all signals to µV during EDF reading (_unit_scale_to_uv)
        unit = "uV"

        meta_entry = {
            "dataset": dataset_name,
            "subject_id": subject_info.get("id", ""),
            "window_idx": int(item.get("epoch_indices", torch.tensor([0]))[0].item()),
            "channels": std_names,
            "channels_raw": list(channel_order),
            "full_signal_channels": list(channel_order),
            "unit": unit,
        }
        if fs_out is not None:
            meta_entry["sampling_rate"] = float(fs_out)

        meta_list.append(meta_entry)

    eeg_tensor = torch.stack(signals_list, dim=0)
    label_tensor = torch.stack(labels_list, dim=0)

    return {
        "signals": {"eeg": eeg_tensor, "full_signal": eeg_tensor},
        "label": label_tensor,
        "meta": meta_list,
    }


# ---------------------------------------------------------------------------
# Generic Benchmark DataModule
# ---------------------------------------------------------------------------

class PhysioExBenchmarkDataModule(BenchmarkDataModule):
    """Generic physioex-backed benchmark adapter.

    Subclasses should set class attributes:
        DATASET_NAME: str
        PHYSIOEX_CLASS: str  — e.g. "neuroatlas.extensions.datasets.physioex.datasets.dcsm.DCSMDataset"
        EEG_CHANNELS: list   — channel requests for physioex
        CHANNEL_MAP: dict    — physical name → standard 10-20 name
        DATASET_KWARGS: dict — extra kwargs for the physioex constructor
    """

    DATASET_NAME: str = "generic"
    PHYSIOEX_CLASS: str = ""
    EEG_CHANNELS: List[str] = ["EEG", "EEG"]
    CHANNEL_MAP: Dict[str, str] = {}
    DATASET_KWARGS: Dict[str, Any] = {}

    def _check_fold_count(self, n_folds: Optional[int]) -> None:
        """Refuse a fold count the cohort's manifest does not have.

        ``n_folds`` is what makes ``neuroatlas run`` probe every fold (the
        manifest cohorts declare 5, the paper's count); a different count has
        no published folds behind it, and fold ``k`` would otherwise still be
        read from the 5-fold file.
        """
        if n_folds is None:
            return
        from neuroatlas.benchmarking_helpers.registry.fold_manifest import load_fold_split

        try:
            have = load_fold_split(self.DATASET_NAME, 0).n_folds
        except FileNotFoundError:
            return          # no manifest: the loader's own split, any count
        if int(n_folds) != have:
            raise ValueError(
                f"{self.DATASET_NAME}: n_folds={n_folds}, but its fold manifest "
                f"has {have} folds (the paper's); ask for {have}."
            )

    def _resolve_splits(self, fold: int, discovered: List[str]):
        """Folds from the cohort's manifest when it has one, else the loader's.

        `neuroatlas.extensions.datasets.physioex`'s `get_splits` is a random 70/15/15 draw seeded
        42+fold. That is not k-fold: the test sets of different folds overlap
        arbitrarily instead of partitioning the cohort, so five "folds" are
        five overlapping samples of the same people. A manifest under
        src/neuroatlas/configs/folds is a 5-fold patient-level partition, which is what the
        paper reports, so it wins wherever one exists.

        An overriding `get_splits` does not change that. STAGES has one, and
        it is still the same random draw -- the override exists to keep the
        split coherent across a subject's repeat recordings, not to supply
        benchmark folds. No physioex loader reads src/neuroatlas/configs/folds, so there is
        no override this would be wrong to overrule.
        """
        from neuroatlas.benchmarking_helpers.registry.fold_manifest import load_fold_split

        try:
            split = load_fold_split(self.DATASET_NAME, fold)
        except FileNotFoundError:
            logger.info(
                "%s: no fold manifest, falling back to the loader's own split "
                "(random 70/15/15, seed 42+fold -- not k-fold)",
                self.DATASET_NAME,
            )
            return self._physio_ds.get_splits(fold=fold)

        present = set(discovered)
        named = set(split.train) | set(split.val) | set(split.test)
        overlap = named & present
        if not overlap:
            raise ValueError(
                f"{self.DATASET_NAME}: fold manifest {split.source_path} names "
                f"{len(named)} ids, none of which match the {len(present)} "
                f"subjects discovered under data_root (manifest "
                f"{sorted(named)[:2]} vs discovered {sorted(present)[:2]}). "
                "Refusing to fall back to a random split, which would silently "
                "not be the paper's folds."
            )
        logger.info(
            "%s: folds from %s (fold %d of %d), %d/%d manifest ids present",
            self.DATASET_NAME, split.source_path, fold, split.n_folds,
            len(overlap), len(named),
        )
        return (
            [s for s in split.train if s in present],
            [s for s in split.val if s in present],
            [s for s in split.test if s in present],
        )


    def __init__(
        self,
        data_root: str,
        batch_size: int = 8,
        signal_kind: str = "raw",
        pipeline_name: str = "legacy",
        fold: Optional[int] = None,
        num_workers: int = 0,
        max_records: Optional[int] = None,
        n_folds: Optional[int] = None,
        **kwargs,
    ):
        if pipeline_name not in PIPELINE_REGISTRY:
            raise ValueError(f"Unknown pipeline_name {pipeline_name!r}. Available: {sorted(PIPELINE_REGISTRY)}")

        pipeline = PIPELINE_REGISTRY[pipeline_name]()
        fold = int(fold) if fold is not None else 0
        self._check_fold_count(n_folds)

        _ensure_physioex()
        # Dynamic import of the physioex dataset class
        parts = self.PHYSIOEX_CLASS.rsplit(".", 1)
        mod = __import__(parts[0], fromlist=[parts[1]])
        DatasetCls = getattr(mod, parts[1])

        self._physio_ds = DatasetCls(
            root=str(data_root),
            channels=list(self.EEG_CHANNELS),
            pipelines=pipeline,
            sequence_length=1,
            trim_excess_wake=False,
            cache_enabled=True,
            **self.DATASET_KWARGS,
        )

        all_subject_ids = [s.subject_id for s in self._physio_ds._subjects]
        if max_records is not None:
            all_subject_ids = all_subject_ids[:int(max_records)]

        train_ids, val_ids, test_ids = self._resolve_splits(fold, all_subject_ids)

        if max_records is not None:
            present = set(all_subject_ids)
            train_ids = [s for s in train_ids if s in present]
            val_ids = [s for s in val_ids if s in present]
            test_ids = [s for s in test_ids if s in present]

        if not (train_ids and val_ids and test_ids):
            raise ValueError(
                f"Empty split (fold={fold}, max_records={max_records}): "
                f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
            )

        self.train_ds: Dataset = _PhysioExSplitDataset(self._physio_ds, train_ids)
        self.val_ds: Dataset = _PhysioExSplitDataset(self._physio_ds, val_ids)
        self.test_ds: Dataset = _PhysioExSplitDataset(self._physio_ds, test_ids)

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self._channel_map = self.CHANNEL_MAP
        self._dataset_name = self.DATASET_NAME

        metadata: Dict[str, Any] = {
            "canonical_label_space": list(_CANONICAL_LABELS),
            "epoch_seconds": 30,
            "signal_kind": signal_kind,
            "pipeline_name": pipeline_name,
            "fold": fold,
            "num_records": len(all_subject_ids),
            "split_subjects": {
                "train": list(train_ids),
                "valid": list(val_ids),
                "test": list(test_ids),
            },
        }
        super().__init__(name=self.DATASET_NAME, metadata=metadata)

        logger.info(
            "%s adapter: pipeline=%s, fold=%d, train=%d/%d, val=%d/%d, test=%d/%d",
            self.DATASET_NAME, pipeline_name, fold,
            len(train_ids), len(self.train_ds),
            len(val_ids), len(self.val_ds),
            len(test_ids), len(self.test_ds),
        )

    def _collate(self, batch):
        return _collate_physioex(batch, self._channel_map, self._dataset_name)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=self._collate)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=self._collate)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=self._collate)
