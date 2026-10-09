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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from neuroatlas.benchmarking_helpers.registry.contracts import BenchmarkDataModule
from neuroatlas.extensions.datasets.pipelines import PIPELINE_REGISTRY

logger = logging.getLogger(__name__)


_CANONICAL_LABELS = ["W", "N1", "N2", "N3", "REM"]

#: What a recording is labelled with: the sleep stage of each 30 s epoch, or
#: (a cohort with an ``AGE_TABLE``) its participant's age, for brain age.
LABEL_MODES = ("sleep_stage", "age")


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

    and, for brain age (``label_mode=age``):
        AGE_TABLE: str       — the cohort's NSRR dataset table under
                               <data_root>/datasets/ (``*`` = its version)
        AGE_ID_COLUMN: str   — the table's subject-id column
        AGE_COLUMN: str      — its age column
        age_key()            — the table id of a recording
    """

    DATASET_NAME: str = "generic"
    PHYSIOEX_CLASS: str = ""
    EEG_CHANNELS: List[str] = ["EEG", "EEG"]
    CHANNEL_MAP: Dict[str, str] = {}
    DATASET_KWARGS: Dict[str, Any] = {}
    AGE_TABLE: Optional[str] = None
    AGE_ID_COLUMN: str = "nsrrid"
    AGE_COLUMN: str = "age"

    @staticmethod
    def age_key(recording_id: str) -> Optional[str]:
        """The table id of a recording (``mesa-sleep-0001`` -> ``1``): by
        default the number its file name ends in."""
        from neuroatlas.extensions.datasets.dataio.nsrr_ages import subject_key

        return subject_key(re.split(r"[-_]", recording_id)[-1])

    def _read_ages(self, recordings: Sequence[str]) -> Dict[str, float]:
        """{recording: its participant's age} from the cohort's NSRR table.

        Brain age joins these to the cached embeddings at probe time
        (:meth:`with_ages`), so the sleep-staging cache serves it as it is. A
        recording the table gives no age is left out of brain age, and said
        so at INFO.
        """
        from neuroatlas.extensions.datasets._missing import no_data
        from neuroatlas.extensions.datasets.dataio.nsrr_ages import find_table, read_ages

        folder = Path(self._data_root) / "datasets"
        path = find_table(folder, str(self.AGE_TABLE))
        if path is None:
            raise FileNotFoundError(no_data(
                self.DATASET_NAME, folder, "data_root", what=f"no {self.AGE_TABLE} "
                f"(the participants' ages brain age predicts) in"))
        by_subject, _empty = read_ages(path, self.AGE_ID_COLUMN, self.AGE_COLUMN)
        ages = {r: by_subject[k] for r in recordings
                if (k := self.age_key(r)) is not None and k in by_subject}
        if not ages and recordings:
            raise ValueError(
                f"{self.DATASET_NAME}: {path.name} gives an age for none of the "
                f"{len(recordings)} recordings in {self._data_root} (joined on its "
                f"{self.AGE_ID_COLUMN} column, e.g. {recordings[0]} -> "
                f"{self.age_key(recordings[0])})")
        without = [r for r in recordings if r not in ages]
        if without:
            logger.info("%s: %s gives no age for %d of %d recordings, which brain age "
                        "leaves out: %s%s", self.DATASET_NAME, path.name, len(without),
                        len(recordings), ", ".join(without[:10]),
                        " ..." if len(without) > 10 else "")
        logger.info("%s: ages of %d recordings from %s (%s, joined on %s)",
                    self.DATASET_NAME, len(ages), path, self.AGE_COLUMN, self.AGE_ID_COLUMN)
        return ages

    def with_ages(self, payload):
        """*payload* with each row's participant age, in ``age`` and as its
        label (NaN where the table has none).

        The cached embeddings are the sleep-staging ones, labelled with
        stages; brain age calls this on what it reads.
        """
        from neuroatlas.benchmarking_helpers import EmbeddingPayload

        if self._ages is None:
            from neuroatlas.cli import _msg, command_with

            raise ValueError(_msg.compose(
                f"{self.DATASET_NAME}: brain age reads the participants' ages with "
                f"label_mode=age", command_with("--set", "label_mode=age")))
        ages = self._ages
        rows = [{**row, "age": ages.get(str(row.get("subject_id")))}
                for row in payload.metadata]
        labels = np.asarray([np.nan if r["age"] is None else r["age"] for r in rows],
                            dtype=np.float64)
        return EmbeddingPayload(features=payload.features, labels=labels, metadata=rows)

    def cache_context(self, purpose: str = "default") -> Dict[str, Any]:
        """The cache key's dataset part: the label mode is not in it, since
        brain age reads the sleep-staging embeddings (and joins the ages)."""
        return {k: v for k, v in self.metadata.items() if k != "label_mode"}

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
            # A random split instead would not be the paper's folds: refuse.
            from neuroatlas.cli import _msg

            where = getattr(self, "_data_root", None) or "the data folder"
            found = (f"it holds {len(present)} other subjects (e.g. "
                     f"{', '.join(sorted(present)[:2])}; the folds name e.g. "
                     f"{', '.join(sorted(named)[:2])})" if present
                     else "it holds no subjects")
            raise ValueError(_msg.compose(
                f"{self.DATASET_NAME}: none of the {len(named)} subjects of the benchmark's "
                f"folds is in {where}: {found}",
                f"neuroatlas config set {self.DATASET_NAME}.data_root DIR"))
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
        label_mode: str = "sleep_stage",
        **kwargs,
    ):
        if pipeline_name not in PIPELINE_REGISTRY:
            raise ValueError(f"Unknown pipeline_name {pipeline_name!r}. Available: {sorted(PIPELINE_REGISTRY)}")
        modes = LABEL_MODES if self.AGE_TABLE else LABEL_MODES[:1]
        if label_mode not in modes:
            raise ValueError(f"{self.DATASET_NAME}: label_mode must be "
                             f"{' or '.join(modes)}, not {label_mode!r}")

        pipeline = PIPELINE_REGISTRY[pipeline_name]()
        fold = int(fold) if fold is not None else 0
        self._check_fold_count(n_folds)
        self._data_root = str(data_root)

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
        self._ages: Optional[Dict[str, float]] = (
            self._read_ages(all_subject_ids) if label_mode == "age" else None)

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
        if label_mode != "sleep_stage":
            metadata["label_mode"] = label_mode
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
