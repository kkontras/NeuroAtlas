"""BenchmarkDataModule adapter for the Bonn EEG epilepsy dataset.

Reads a pre-built flat HDF5 cache (see ``bonn_preprocessor.build_h5_cache``)
and exposes train/val/test dataloaders.

Splits are computed at the **clip level** (not window level) via stratified
k-fold on the class label, so sub-windows from the same clip always stay in
the same split.  We stratify at the class-code level: Bonn only has 5
clip-to-patient groups (unknown mapping) per intracranial set, so
patient-disjoint splitting isn't feasible — class-stratified splits are
the standard choice in the literature.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from benchmarking_helpers.registry.splits import make_subject_kfold
from torch.utils.data import DataLoader, WeightedRandomSampler

from extensions.datasets.dataio.bonn import (
    BonnSegmentDataset,
    _collate_bonn,
)

from .base import BenchmarkDataModule

logger = logging.getLogger(__name__)


_CACHE_ROOT_DEFAULT = (
    "${REPO_ROOT}/bonn_cache/hdf5"
)
_CACHE_FILENAME = "bonn_173hz_segments.h5"


# ---------------------------------------------------------------------------
# Loader adapter (benchmark batch format)
# ---------------------------------------------------------------------------


# The shared wrapper; the local name is kept so call sites are unchanged.
from ._loader_adapters import SplitTaggingLoader as _LoaderAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Clip-level stratified split
# ---------------------------------------------------------------------------


def _stratified_clip_splits(
    class_codes_per_clip: np.ndarray,
    allowed_clip_indices: List[int],
    fold: int,
    n_folds: int,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Patient info is unknown, so we stratify on class code at the clip level.

    Returns ``(train, val, test)`` clip-index lists (indices into the full
    HDF5 clip array, not positions within ``allowed_clip_indices``).
    """
    if len(allowed_clip_indices) == 0:
        return [], [], []

    allowed = np.asarray(allowed_clip_indices, dtype=np.int64)
    strat = class_codes_per_clip[allowed]

    # Shared rule: benchmarking_helpers/splits. This used to reduce the fold
    # count to the smallest class (100 clips per class normally, but only Z+S
    # for binary_ae), which silently ran fewer folds than asked for.
    skf_outer, _stratified, effective_folds = make_subject_kfold(
        n_folds, strat, seed=seed)
    outer_splits = list(skf_outer.split(allowed, strat))
    train_val_local, test_local = outer_splits[fold % effective_folds]

    # Inner split for val
    tv_allowed = allowed[train_val_local]
    tv_strat = strat[train_val_local]
    skf_inner, _inner_strat, inner_folds = make_subject_kfold(
        effective_folds, tv_strat, seed=seed)
    inner_splits = list(skf_inner.split(tv_allowed, tv_strat))
    val_fold = (fold + 1) % inner_folds
    inner_train_local, inner_val_local = inner_splits[val_fold % len(inner_splits)]

    train_clips = sorted(int(x) for x in tv_allowed[inner_train_local])
    val_clips = sorted(int(x) for x in tv_allowed[inner_val_local])
    test_clips = sorted(int(x) for x in allowed[test_local])

    return train_clips, val_clips, test_clips


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class BonnBenchmarkDataModule(BenchmarkDataModule):
    """BenchmarkDataModule for the Bonn EEG epilepsy dataset.

    Args:
        cache_root: Directory containing the Bonn HDF5 cache.
        cache_filename: File name inside ``cache_root`` (default
            ``bonn_173hz_segments.h5``).
        fold: Outer fold index (0-based).
        n_folds: Total k-fold count (default 5).
        batch_size: Batch size for all loaders.
        num_workers: DataLoader worker count.
        window_s: Sub-window length in seconds.  ``None`` = full 23.6-s clip.
        stride_s: Stride in seconds between sub-windows.  ``None`` =
            non-overlapping.
        label_mode: Passed through to :class:`BonnSegmentDataset`.  One of
            ``binary_s_vs_rest`` (default), ``binary_ae``, ``three_class``,
            ``five_class``, ``recording_type``, ``state``.
        normalize: ``"none"`` or ``"per_window_zscore"``.
        balance: ``"weighted_sampler"`` (rebalances class frequency in
            training) or ``"none"``.
        seed: RNG seed for the StratifiedKFold split (default 42).
    """

    def __init__(
        self,
        raw_dir: str | None = None,
        cache_root: str = _CACHE_ROOT_DEFAULT,
        cache_filename: str = _CACHE_FILENAME,
        fold: int = 0,
        n_folds: int = 5,
        batch_size: int = 64,
        num_workers: int = 0,
        window_s: Optional[float] = None,
        stride_s: Optional[float] = None,
        label_mode: str = "binary_s_vs_rest",
        normalize: str = "none",
        balance: str = "weighted_sampler",
        seed: int = 42,
        signal_kind: str = "raw",
        **kwargs: Any,
    ) -> None:
        metadata = {
            "canonical_label_space": label_mode,
            "epoch_seconds": float(window_s) if window_s is not None else 23.593,
            "channel_policy": ["eeg"],
            "signal_kind": signal_kind,
            "fold": fold,
            "n_folds": n_folds,
        }
        super().__init__(name="bonn", metadata=metadata)

        # Bonn is 500 text clips (~8 MB); reading them directly means a fresh
        # checkout works after `fetch` alone, with no build step. The HDF5
        # cache stays as the fast path for large sweeps.
        # Prefer the raw clips, but only if they are actually there. The
        # manifest ships a raw_dir default, so a user who has only ever built
        # the cache must not be broken by it -- and a user who has only the
        # raw corpus must not need to build anything.
        raw = Path(str(raw_dir)) if raw_dir else None
        self._raw_dir = str(raw) if raw is not None and raw.is_dir() else None
        self._h5_path = str(Path(cache_root) / cache_filename) if cache_root else None
        if self._raw_dir is None and (not self._h5_path
                                      or not Path(self._h5_path).exists()):
            raise FileNotFoundError(
                f"No Bonn data found. Either source works:\n"
                f"  raw clips : {raw_dir or '(no raw_dir given)'}\n"
                f"  HDF5 cache: {self._h5_path or '(no cache_root given)'}\n"
                "Get the corpus with:\n"
                "  python -m entrypoints.fetch --dataset bonn --download\n"
                "The raw clips are enough; building the cache is optional:\n"
                "  python -m extensions.datasets.preprocessors.preprocess_bonn "
                f"--raw-dir {raw_dir or '<raw>'} --output {self._h5_path or '<cache.h5>'}"
            )

        self._fold = fold
        self._n_folds = n_folds
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._window_s = window_s
        self._stride_s = stride_s
        self._label_mode = label_mode
        self._normalize = normalize
        self._balance = balance
        self._seed = seed

        # --- Read clip-level class codes + determine the allowed clip set ---
        if self._raw_dir is not None:
            from extensions.datasets.epilepsy import (
                bonn_preprocessor,
            )

            self._class_codes = bonn_preprocessor.load_raw_corpus(
                self._raw_dir)["class_label"].astype(np.int64)
        else:
            import h5py

            with h5py.File(self._h5_path, "r") as f:
                self._class_codes = f["class_label"][:].astype(np.int64)

        # Honour label_mode's class-code filter (e.g. binary_ae keeps only Z+S)
        from extensions.datasets.dataio.bonn import _label_mode_spec

        _, keep_codes, _ = _label_mode_spec(label_mode)
        n_clips = len(self._class_codes)
        if keep_codes is None:
            allowed = list(range(n_clips))
        else:
            allowed = [i for i in range(n_clips) if int(self._class_codes[i]) in keep_codes]

        if len(allowed) == 0:
            raise RuntimeError(
                f"No clips remain after applying label_mode={label_mode!r}"
            )

        self._train_clips, self._val_clips, self._test_clips = _stratified_clip_splits(
            self._class_codes, allowed, fold, n_folds, seed=seed,
        )
        logger.info(
            "Bonn splits (fold %d/%d, label_mode=%s): train=%d val=%d test=%d clips",
            fold, n_folds, label_mode,
            len(self._train_clips), len(self._val_clips), len(self._test_clips),
        )

        self._datasets: Dict[str, BonnSegmentDataset] = {}

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _get_dataset(self, split: str) -> BonnSegmentDataset:
        if split not in self._datasets:
            clip_indices = {
                "train": self._train_clips,
                "val": self._val_clips,
                "test": self._test_clips,
            }[split]
            # Non-training splits use non-overlapping windows only
            stride_s = self._stride_s if split == "train" else self._window_s

            self._datasets[split] = BonnSegmentDataset(
                h5_path=None if self._raw_dir else self._h5_path,
                raw_dir=self._raw_dir,
                window_s=self._window_s,
                stride_s=stride_s,
                label_mode=self._label_mode,
                clip_indices=clip_indices,
                normalize=self._normalize,
            )
        return self._datasets[split]

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

        if split == "train" and self._balance == "weighted_sampler" and len(ds) > 0:
            targets = ds.targets()
            # 1 / class frequency, normalised per window
            _, counts = np.unique(targets, return_counts=True)
            class_weight = {int(c): 1.0 / float(n) for c, n in zip(
                np.unique(targets), counts,
            )}
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
            collate_fn=_collate_bonn,
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

    def datasets(self) -> Dict[str, BonnSegmentDataset]:
        for split in ("train", "val", "test"):
            self._get_dataset(split)
        return dict(self._datasets)
