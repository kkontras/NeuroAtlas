"""BenchmarkDataModule adapter for the TUH EEG Abnormal Corpus (TUAB).

Two backends are supported, chosen automatically by the adapter:

- **EDF backend** (default) — reads raw EDFs directly from
  ``raw_root``.  No preprocessing required: a fresh checkout that
  only fetched the corpus works out of the box.

- **H5 backend** (opt-in fast path) — used when ``cache_root`` is
  provided AND the expected per-split cache files exist there.  Build
  them with::

      python -m neuroatlas.extensions.datasets.preprocessors.preprocess_tuab \\
          --raw-root <raw_root> --cache-root <cache_root>

Two split modes:

- ``"official"`` — TUAB ships its own train/eval split.  We split the
  train recordings into train + val (subject-disjoint, seeded), and use
  eval as the test set.
- ``"kfold"`` — pool both splits and run subject-stratified k-fold on
  the per-subject ``is_abnormal`` label.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from neuroatlas.benchmarking_helpers.registry.splits import make_subject_kfold
from torch.utils.data import DataLoader, WeightedRandomSampler

from neuroatlas.extensions.datasets.dataio.tuab import (
    TuabEdfDataset,
    TuabH5Dataset,
    collate_tuab,
    discover_tuab_recordings,
)
from neuroatlas.extensions.datasets.epilepsy.tuab_preprocessor import (
    CACHE_SCHEMA_TAG,
)

from .base import BenchmarkDataModule

logger = logging.getLogger(__name__)


_DEFAULT_RAW_ROOT = "${EEG_DATA_ROOT}/TUH/tuh_eeg/tuh_eeg_abnormal/v3.0.1/edf"
_DEFAULT_CACHE_ROOT = (
    "${REPO_ROOT}/tuab_cache/hdf5"
)


# ---------------------------------------------------------------------------
# Loader adapter (benchmark batch format)
# ---------------------------------------------------------------------------


# The shared wrapper; the local name is kept so call sites are unchanged.
from ._loader_adapters import SplitTaggingLoader as _LoaderAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Subject-disjoint split helpers
# ---------------------------------------------------------------------------


def _subject_label_table(
    subject_ids: Sequence[str], is_abnormal: Sequence[int],
) -> Tuple[List[str], np.ndarray]:
    """Aggregate to one label per subject (any abnormal recording → abnormal)."""
    by_subject: Dict[str, int] = {}
    for s, a in zip(subject_ids, is_abnormal):
        by_subject[s] = max(by_subject.get(s, 0), int(a))
    subjects = sorted(by_subject.keys())
    y = np.array([by_subject[s] for s in subjects], dtype=np.int64)
    return subjects, y


def _stratified_subject_split(
    subjects: List[str],
    y_subject: np.ndarray,
    fold: int,
    n_folds: int,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Return ``(train_subjects, val_subjects, test_subjects)`` for one fold.

    Two-level: outer split selects the test fold, inner split carves
    train/val out of the remainder.  Used by the ``kfold`` split mode.
    """
    # Shared rule: benchmarking_helpers/splits — keep the requested folds,
    # drop stratification if the classes cannot carry them.
    skf, _stratified, n_folds = make_subject_kfold(n_folds, y_subject, seed=seed)
    splits = list(skf.split(subjects, y_subject))
    tv_idx, test_idx = splits[fold % n_folds]

    test_subjects = [subjects[i] for i in test_idx]
    tv_subjects = [subjects[i] for i in tv_idx]
    tv_y = y_subject[tv_idx]

    inner_skf, _inner_strat, inner_folds = make_subject_kfold(n_folds, tv_y, seed=seed)
    inner = list(inner_skf.split(tv_subjects, tv_y))
    val_fold = (fold + 1) % inner_folds
    inner_train_idx, inner_val_idx = inner[val_fold % len(inner)]
    train_subjects = [tv_subjects[i] for i in inner_train_idx]
    val_subjects = [tv_subjects[i] for i in inner_val_idx]
    return train_subjects, val_subjects, test_subjects


def _train_val_subject_split(
    subjects: List[str],
    y_subject: np.ndarray,
    fold: int,
    n_folds: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Single StratifiedKFold pass — used by the ``official`` split mode
    where test is supplied externally (the eval cache) and we only need
    a train/val carve-out."""
    # This used to raise when a class had a single subject. It now falls back
    # to an unstratified split, because refusing to carve a validation set is
    # worse than carving one that is not class-balanced.
    skf, _stratified, effective_folds = make_subject_kfold(
        n_folds, y_subject, seed=seed)
    splits = list(skf.split(subjects, y_subject))
    train_idx, val_idx = splits[fold % effective_folds]
    return (
        [subjects[i] for i in train_idx],
        [subjects[i] for i in val_idx],
    )


def _recordings_for_subjects(
    subject_ids: Sequence[str], wanted: Sequence[str],
) -> List[int]:
    wset = set(wanted)
    return [i for i, s in enumerate(subject_ids) if s in wset]


# ---------------------------------------------------------------------------
# Cache discovery
# ---------------------------------------------------------------------------


def _h5_cache_paths(
    cache_root: Optional[str], corpus_version: str,
) -> Dict[str, Optional[Path]]:
    """Return ``{split: Path or None}`` for both TUAB splits."""
    if cache_root is None:
        return {"train": None, "eval": None}
    root = Path(cache_root)
    paths: Dict[str, Optional[Path]] = {}
    for split in ("train", "eval"):
        p = root / f"tuab_{corpus_version}_{split}_{CACHE_SCHEMA_TAG}.h5"
        paths[split] = p if p.exists() else None
    return paths


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class TuabBenchmarkDataModule(BenchmarkDataModule):
    """BenchmarkDataModule for the TUH EEG Abnormal Corpus (TUAB).

    Args:
        raw_root: TUAB ``edf/`` directory.  Required for the EDF backend
            (the default).  Even when the H5 backend is used, this is
            consulted as a fallback if a per-split cache is missing.
        cache_root: Directory containing pre-built per-split HDF5 caches.
            If ``None`` or the expected files are absent, falls back to
            the EDF backend.  Defaults to a workspace-local path; pass
            ``None`` to disable the fast path entirely.
        corpus_version: TUAB version string used in cache filenames
            (default ``"v3.0.1"``).
        split_mode: ``"official"`` (default) or ``"kfold"``.
        fold: Fold index for ``kfold`` mode and for the inner train/val
            split in ``official`` mode (seeds the val carve-out).
        n_folds: Number of folds (default 5).
        batch_size, num_workers: DataLoader knobs.
        window_s: Window length in seconds (default 30).
        stride_s: Stride; ``None`` = non-overlapping (= window_s).
        normalize: ``"none"`` or ``"per_window_zscore"``.
        balance: ``"weighted_sampler"`` for class-balanced training, or
            ``"none"``.
        montage_filter: Optional list of montage_type names to keep
            (e.g. ``["01_tcp_ar", "02_tcp_le"]``).
        cache_max_recordings: LRU size for the EDF backend (in fully
            decoded recordings).  Larger = more RAM, fewer re-reads.
        seed: RNG seed for the subject-stratified splits (default 42).
    """

    def __init__(
        self,
        raw_root: str = _DEFAULT_RAW_ROOT,
        cache_root: Optional[str] = _DEFAULT_CACHE_ROOT,
        corpus_version: str = "v3.0.1",
        split_mode: str = "official",
        fold: int = 0,
        n_folds: int = 5,
        batch_size: int = 64,
        num_workers: int = 4,
        window_s: float = 30.0,
        stride_s: Optional[float] = None,
        label_mode: str = "binary",
        normalize: str = "none",
        balance: str = "weighted_sampler",
        montage_filter: Optional[Sequence[str]] = None,
        cache_max_recordings: int = 16,
        seed: int = 42,
        signal_kind: str = "raw",
        **_unused: Any,
    ) -> None:
        if split_mode not in ("official", "kfold"):
            raise ValueError(f"split_mode must be 'official' or 'kfold', got {split_mode!r}")
        if label_mode != "binary":
            raise ValueError(f"TUAB only supports label_mode='binary', got {label_mode!r}")

        metadata = {
            "canonical_label_space": "binary",
            "epoch_seconds": float(window_s),
            "channel_policy": ["eeg"],
            "signal_kind": signal_kind,
            "fold": fold,
            "n_folds": n_folds,
        }
        super().__init__(name="tuab", metadata=metadata)

        self._raw_root = raw_root
        self._cache_root = cache_root
        self._corpus_version = corpus_version
        self._split_mode = split_mode
        self._fold = fold
        self._n_folds = n_folds
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._window_s = float(window_s)
        self._stride_s = float(stride_s) if stride_s is not None else float(window_s)
        self._normalize = normalize
        self._balance = balance
        self._montage_filter = list(montage_filter) if montage_filter else None
        self._cache_max = int(cache_max_recordings)
        self._seed = seed

        # ----- Decide backend -----
        self._cache_paths = _h5_cache_paths(cache_root, corpus_version)
        self._use_h5 = all(p is not None for p in self._cache_paths.values())

        if self._use_h5:
            logger.info(
                "TUAB: using H5 backend (caches: %s, %s)",
                self._cache_paths["train"], self._cache_paths["eval"],
            )
        else:
            logger.info(
                "TUAB: using EDF backend (raw_root=%s; "
                "no complete H5 cache at %s — set or build one to use the fast path)",
                raw_root, cache_root,
            )
            if not Path(raw_root).exists():
                raise FileNotFoundError(
                    f"TUAB raw_root not found: {raw_root}\n"
                    f"Either:\n"
                    f"  - run `python -m neuroatlas.entrypoints.fetch --dataset tuab --download` to fetch it, or\n"
                    f"  - point `raw_root` at an existing mirror, or\n"
                    f"  - build the H5 cache and set `cache_root`."
                )

        # ----- Build per-split recording_indices -----
        # We need subject ids + is_abnormal labels per recording for the
        # subject-disjoint split, regardless of backend.  Both backends
        # expose the same metadata via a transient probe object.
        (self._all_recordings,
         self._subject_ids,
         self._is_abnormal,
         self._splits_per_rec) = self._scan_metadata()

        self._train_idx, self._val_idx, self._test_idx = self._compute_splits()
        logger.info(
            "TUAB splits (mode=%s, fold=%d/%d): train=%d val=%d test=%d recordings",
            split_mode, fold, n_folds,
            len(self._train_idx), len(self._val_idx), len(self._test_idx),
        )

        self._datasets: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Metadata scan
    # ------------------------------------------------------------------

    def _scan_metadata(self) -> Tuple[
        List[Dict[str, Any]], List[str], List[int], List[str],
    ]:
        """Build a unified recording table indexed positionally.

        Returns:
            (all_recordings, subject_ids, is_abnormal, split_per_rec).
            Indices into these lists are the canonical "recording index"
            used by the split logic.
        """
        if self._use_h5:
            import h5py
            recs: List[Dict[str, Any]] = []
            subj: List[str] = []
            isab: List[int] = []
            splt: List[str] = []
            for split in ("train", "eval"):
                p = self._cache_paths[split]
                with h5py.File(str(p), "r") as f:
                    rec_ids = [s.decode() if isinstance(s, bytes) else str(s)
                               for s in f["recording_ids"][:]]
                    s_ids = [s.decode() if isinstance(s, bytes) else str(s)
                             for s in f["subject_ids"][:]]
                    montages = [s.decode() if isinstance(s, bytes) else str(s)
                                for s in f["montage_types"][:]]
                    abn = f["is_abnormal"][:].astype(int).tolist()
                for rid, sid, mt, a in zip(rec_ids, s_ids, montages, abn):
                    if self._montage_filter and mt not in self._montage_filter:
                        continue
                    recs.append({
                        "recording_id": rid, "subject_id": sid,
                        "montage_type": mt, "is_abnormal": int(a),
                        "split": split,
                        "_h5_local_idx": len(recs),  # not used; reserved
                    })
                    subj.append(sid)
                    isab.append(int(a))
                    splt.append(split)
            return recs, subj, isab, splt

        # EDF backend
        all_recs = discover_tuab_recordings(self._raw_root, splits=("train", "eval"))
        if self._montage_filter:
            allowed = set(self._montage_filter)
            all_recs = [r for r in all_recs if r["montage_type"] in allowed]
        subj = [r["subject_id"] for r in all_recs]
        isab = [int(r["is_abnormal"]) for r in all_recs]
        splt = [r["split"] for r in all_recs]
        return all_recs, subj, isab, splt

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def _compute_splits(self) -> Tuple[List[int], List[int], List[int]]:
        if self._split_mode == "official":
            train_recs = [i for i, s in enumerate(self._splits_per_rec) if s == "train"]
            eval_recs = [i for i, s in enumerate(self._splits_per_rec) if s == "eval"]

            # Carve a subject-disjoint val out of the train recordings.
            train_subj_ids = [self._subject_ids[i] for i in train_recs]
            train_abn = [self._is_abnormal[i] for i in train_recs]
            subjects, y = _subject_label_table(train_subj_ids, train_abn)
            tr_subj, va_subj = _train_val_subject_split(
                subjects, y,
                fold=self._fold, n_folds=self._n_folds, seed=self._seed,
            )
            tr_set, va_set = set(tr_subj), set(va_subj)
            train_idx = [i for i in train_recs if self._subject_ids[i] in tr_set]
            val_idx = [i for i in train_recs if self._subject_ids[i] in va_set]
            test_idx = eval_recs
            return train_idx, val_idx, test_idx

        # kfold
        subjects, y = _subject_label_table(self._subject_ids, self._is_abnormal)
        tr_subj, va_subj, te_subj = _stratified_subject_split(
            subjects, y,
            fold=self._fold, n_folds=self._n_folds, seed=self._seed,
        )
        tr_set, va_set, te_set = set(tr_subj), set(va_subj), set(te_subj)
        train_idx = [i for i, s in enumerate(self._subject_ids) if s in tr_set]
        val_idx = [i for i, s in enumerate(self._subject_ids) if s in va_set]
        test_idx = [i for i, s in enumerate(self._subject_ids) if s in te_set]
        return train_idx, val_idx, test_idx

    # ------------------------------------------------------------------
    # Dataset construction (per split)
    # ------------------------------------------------------------------

    def _get_dataset(self, split: str) -> Any:
        if split in self._datasets:
            return self._datasets[split]

        rec_indices = {
            "train": self._train_idx,
            "val": self._val_idx,
            "test": self._test_idx,
        }[split]
        # Non-training splits use non-overlapping windows for stable eval.
        stride_s = self._stride_s if split == "train" else self._window_s

        if self._use_h5:
            ds = self._build_h5_dataset(split, rec_indices, stride_s)
        else:
            ds = TuabEdfDataset(
                recordings=self._all_recordings,
                window_s=self._window_s,
                stride_s=stride_s,
                normalize=self._normalize,
                montage_filter=None,  # already applied in _scan_metadata
                recording_indices=rec_indices,
                cache_max_recordings=self._cache_max,
            )
        self._datasets[split] = ds
        return ds

    def _build_h5_dataset(
        self, split: str, rec_indices: Sequence[int], stride_s: float,
    ) -> Any:
        """Construct a TuabH5Dataset by remapping global recording indices to
        per-cache-file indices."""
        # Recording table is concatenated train-then-eval; figure out how
        # many train recordings there are so we can rebase indices.
        train_count = sum(1 for s in self._splits_per_rec if s == "train")
        if split == "test" and self._split_mode == "official":
            target_split = "eval"
        elif split in ("train", "val") and self._split_mode == "official":
            target_split = "train"
        else:
            target_split = None  # kfold — may need both files, see below

        if target_split is not None:
            # Single file path
            local = []
            for gi in rec_indices:
                if self._splits_per_rec[gi] != target_split:
                    continue
                # Local index inside that split's cache
                base = 0 if target_split == "train" else train_count
                local.append(gi - base)
            return TuabH5Dataset(
                h5_path=str(self._cache_paths[target_split]),
                window_s=self._window_s,
                stride_s=stride_s,
                normalize=self._normalize,
                recording_indices=local,
            )

        # kfold mode may need recordings from both files — wrap in a
        # ConcatDataset-style helper.
        from torch.utils.data import ConcatDataset

        train_local = [
            gi for gi in rec_indices if self._splits_per_rec[gi] == "train"
        ]
        eval_local = [
            gi - train_count for gi in rec_indices if self._splits_per_rec[gi] == "eval"
        ]
        parts: List[Any] = []
        if train_local:
            parts.append(TuabH5Dataset(
                h5_path=str(self._cache_paths["train"]),
                window_s=self._window_s, stride_s=stride_s,
                normalize=self._normalize,
                recording_indices=train_local,
            ))
        if eval_local:
            parts.append(TuabH5Dataset(
                h5_path=str(self._cache_paths["eval"]),
                window_s=self._window_s, stride_s=stride_s,
                normalize=self._normalize,
                recording_indices=eval_local,
            ))
        if not parts:
            raise RuntimeError(
                f"TUAB H5 backend produced 0 datasets for split={split!r} "
                "in kfold mode — splits are empty?"
            )
        if len(parts) == 1:
            return parts[0]

        # ConcatDataset doesn't expose targets() out of the box; provide
        # a tiny wrapper.
        class _ConcatWithTargets(ConcatDataset):
            def __init__(self, datasets):
                super().__init__(datasets)
                ts = [d.targets() for d in datasets]
                self._targets = np.concatenate(ts) if ts else np.array([], dtype=np.int64)
                self.n_classes = 2

            def targets(self) -> np.ndarray:
                return self._targets

        return _ConcatWithTargets(parts)

    # ------------------------------------------------------------------
    # Loader construction
    # ------------------------------------------------------------------

    def _make_loader(self, split: str) -> _LoaderAdapter:
        ds = self._get_dataset(split)
        # Embed-only path doesn't benefit from shuffle and it kills LRU
        # locality on lazy EDF dataios. Probes run on cached embeddings,
        # not on this loader.
        shuffle = False
        sampler = None

        if (split == "train"
                and self._balance == "weighted_sampler"
                and len(ds) > 0):
            targets = ds.targets()
            unique, counts = np.unique(targets, return_counts=True)
            class_weight = {int(c): 1.0 / float(n) for c, n in zip(unique, counts)}
            weights = np.array(
                [class_weight[int(t)] for t in targets], dtype=np.float64,
            )
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(weights).double(),
                num_samples=len(ds),
                replacement=True,
            )
            shuffle = False

        loader = DataLoader(
            ds,
            batch_size=self._batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self._num_workers,
            collate_fn=collate_tuab,
            pin_memory=True,
            drop_last=False,
        )
        return _LoaderAdapter(loader, split_name=split)

    def train_dataloader(self):
        return self._make_loader("train")

    def val_dataloader(self):
        return self._make_loader("val")

    def test_dataloader(self):
        return self._make_loader("test")

    def datasets(self) -> Dict[str, Any]:
        for split in ("train", "val", "test"):
            self._get_dataset(split)
        return dict(self._datasets)
