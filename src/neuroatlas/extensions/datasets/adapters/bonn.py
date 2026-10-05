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
from neuroatlas.benchmarking_helpers.registry.splits import make_subject_kfold
from torch.utils.data import DataLoader, WeightedRandomSampler

from neuroatlas.extensions.datasets.dataio.bonn import (
    BonnSegmentDataset,
    _collate_bonn,
)

from .base import BenchmarkDataModule
from neuroatlas.extensions.datasets.epilepsy._global_cache import RecordingWindowGlobalCache

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
# The paper's actual folds (split_mode="paper_segments")
# ---------------------------------------------------------------------------

#: ``clip`` -- the default: folds over the 500 clips, a clip's windows always together.
#: ``paper_segments`` -- the folds the published Bonn numbers were drawn on (App. D.1.3:
#: "stratified 5-fold cross-validation at the segment level"), frozen in
#: ``configs/cohorts/bonn/folds_paper_segments.json``: the 1,000 10-s segments were assigned
#: independently, so the two halves of one 23.6-s clip can sit in train and test. Offered only
#: to reproduce the published numbers; defined for 10 s / 10 s windows and S vs rest only.
SPLIT_MODES = ("clip", "paper_segments")
_SEGMENT_SETS = ("Z", "O", "N", "F", "S")    # bonn_preprocessor.SET_ORDER: class code = position


def paper_segments_manifest_path() -> Path:
    from neuroatlas._paths import configs_dir

    return configs_dir("cohorts", "bonn", "folds_paper_segments.json")


def _segment_key(segment_id: str, class_codes: np.ndarray) -> Tuple[int, int]:
    """'Z037:1' -> (clip index, segment): clip Z037 is the 37th clip of set Z in the
    canonical order (sets Z,O,N,F,S; files sorted), segment 1 its [10, 20) s window."""
    stem, seg = segment_id.split(":")
    code, ordinal = _SEGMENT_SETS.index(stem[0]), int(stem[1:]) - 1
    clips_of_set = np.flatnonzero(np.asarray(class_codes) == code)
    if not 0 <= ordinal < len(clips_of_set):
        raise ValueError(f"segment {segment_id!r}: set {stem[0]} has {len(clips_of_set)} clips")
    return int(clips_of_set[ordinal]), int(seg)


def _paper_segment_roles(
    class_codes: np.ndarray,
    fold: int,
    n_folds: int,
    manifest_path: Optional[Path] = None,
) -> Tuple[Dict[Tuple[int, int], str], str]:
    """{(clip index, segment): "train"|"val"|"test"} of the paper's fold, and the manifest path."""
    import json

    path = Path(manifest_path) if manifest_path is not None else paper_segments_manifest_path()
    manifest = json.loads(path.read_text())
    if int(manifest["n_folds"]) != int(n_folds):
        raise ValueError(
            f"split_mode=paper_segments is the paper's {manifest['n_folds']}-fold split; "
            f"n_folds={n_folds} has no paper counterpart")
    folds = manifest["folds"][str(int(fold) % int(n_folds))]
    roles: Dict[Tuple[int, int], str] = {}
    for role in ("train", "val", "test"):
        for sid in folds[role]:
            roles[_segment_key(sid, class_codes)] = role
    if len(roles) != sum(len(folds[r]) for r in ("train", "val", "test")):
        raise ValueError(f"{path}: a segment is listed twice in fold {fold}")
    return roles, str(path)


def _check_canonical_clip_names(raw_dir: str) -> None:
    """The manifest names clips <set><NNN>; make sure this copy sorts them as NNN = 1..100."""
    from neuroatlas.extensions.datasets.epilepsy import bonn_preprocessor as bp

    for code, letter in enumerate(_SEGMENT_SETS):
        files = bp._list_clip_files(bp._find_set_directory(Path(raw_dir), letter))
        stems = [f.stem.upper() for f in files]
        want = [f"{letter}{i + 1:03d}" for i in range(len(files))]
        if stems != want:
            raise ValueError(
                f"split_mode=paper_segments: set {letter} files are not named {want[0]}..{want[-1]} "
                f"in sorted order (found {stems[:3]}...), so the paper's segment ids cannot be mapped")


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class BonnBenchmarkDataModule(RecordingWindowGlobalCache, BenchmarkDataModule):
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
        split_mode: ``"clip"`` (default) or ``"paper_segments"`` (the paper's
            segment-level folds; see ``SPLIT_MODES``).
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
        split_mode: str = "clip",
        **kwargs: Any,
    ) -> None:
        if split_mode not in SPLIT_MODES:
            raise ValueError(f"split_mode must be one of {SPLIT_MODES}, got {split_mode!r}")
        if split_mode == "paper_segments":
            if label_mode != "binary_s_vs_rest":
                raise ValueError("split_mode=paper_segments is the paper's S-vs-rest split; "
                                 f"label_mode={label_mode!r} has no paper folds")
            if window_s is None or float(window_s) != 10.0 or (
                    stride_s is not None and float(stride_s) != float(window_s)):
                raise ValueError("split_mode=paper_segments is defined on the paper's 10 s / 10 s "
                                 f"segments; got window_s={window_s!r} stride_s={stride_s!r}")
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
                "  python -m neuroatlas.entrypoints.fetch --dataset bonn --download\n"
                "The raw clips are enough; building the cache is optional:\n"
                "  python -m neuroatlas.extensions.datasets.preprocessors.preprocess_bonn "
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
            from neuroatlas.extensions.datasets.epilepsy import (
                bonn_preprocessor,
            )

            self._class_codes = bonn_preprocessor.load_raw_corpus(
                self._raw_dir)["class_label"].astype(np.int64)
        else:
            import h5py

            with h5py.File(self._h5_path, "r") as f:
                self._class_codes = f["class_label"][:].astype(np.int64)

        # Honour label_mode's class-code filter (e.g. binary_ae keeps only Z+S)
        from neuroatlas.extensions.datasets.dataio.bonn import _label_mode_spec

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

        self._allowed_clips = list(allowed)
        self._split_mode = split_mode
        self._segment_roles: Optional[Dict[Tuple[int, int], str]] = None
        if split_mode == "paper_segments":
            if self._raw_dir is not None:
                _check_canonical_clip_names(self._raw_dir)
            self._segment_roles, source = _paper_segment_roles(self._class_codes, fold, n_folds)
            # A clip is listed under every role one of its segments has; the
            # segment-level assignment itself is applied in split_global_embedding_payload.
            by_role: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
            for (clip, _seg), role in self._segment_roles.items():
                by_role[role].add(clip)
            self._train_clips, self._val_clips, self._test_clips = (
                sorted(by_role["train"]), sorted(by_role["val"]), sorted(by_role["test"]))
            # Under keys the embedding caches treat as fold bookkeeping (never in the
            # global cache key): which folds produced a result is recorded with it.
            self.metadata["fold_source"] = source
            self.metadata["fold_stats"] = {"split_mode": split_mode}
        else:
            self._train_clips, self._val_clips, self._test_clips = _stratified_clip_splits(
                self._class_codes, allowed, fold, n_folds, seed=seed,
            )
        # The names the global-cache mixin reads: a Bonn "recording" is a clip.
        self._train_recs, self._val_recs, self._test_recs = (
            self._train_clips, self._val_clips, self._test_clips)
        self._collate_fn = _collate_bonn
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
        if self._segment_roles is not None:
            # A per-split reader takes whole clips; the paper's folds split clips.
            raise RuntimeError(
                "split_mode=paper_segments assigns the two segments of a clip separately, which "
                "only the global embedding cache can express (seizure_detection with window_s=10, "
                "stride_s=10 uses it); per-split loaders are not available in this mode")
        if split not in self._datasets:
            clip_indices = {
                "train": self._train_clips,
                "val": self._val_clips,
                "test": self._test_clips,
            }[split]
            # Non-training splits use non-overlapping windows only
            stride_s = self._stride_s if split == "train" else self._window_s
            self._datasets[split] = self._window_dataset(clip_indices, stride_s)
        return self._datasets[split]

    def _window_dataset(self, clip_indices, stride_s, signal_cache_size: int = 0):
        # signal_cache_size: unused, Bonn holds the whole 8 MB corpus in memory.
        return BonnSegmentDataset(
            h5_path=None if self._raw_dir else self._h5_path,
            raw_dir=self._raw_dir,
            window_s=self._window_s,
            stride_s=stride_s,
            label_mode=self._label_mode,
            clip_indices=clip_indices,
            normalize=self._normalize,
        )

    # ------------------------------------------------------------------
    # Global embedding cache (epilepsy/_global_cache.py): every window of
    # every clip once, folds assigned when probing. The per-split caches it
    # replaces held one fold's (resampled) train split, so fold 1 found
    # nothing to read.
    # ------------------------------------------------------------------

    _recording_key = "clip_idx"

    def _n_recordings(self) -> int:
        return len(self._class_codes)

    def _all_recording_indices(self) -> List[int]:
        # Only the clips the label mode keeps (binary_ae drops O/N/F).
        return list(self._allowed_clips)

    def _subject_of(self, clip_index: int) -> str:
        # The public corpus names no subjects; the clip is the unit.
        return str(int(clip_index))

    def split_global_embedding_payload(self, payload) -> Dict[str, Any]:
        if self._segment_roles is None:
            return super().split_global_embedding_payload(payload)
        # split_mode=paper_segments: the role belongs to the (clip, segment), not the clip.
        from neuroatlas.benchmarking_helpers import EmbeddingPayload

        metadata = payload.metadata
        features = payload.features
        labels = np.asarray(payload.labels)
        window = float(self._window_s)
        groups: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        for i, m in enumerate(metadata):
            role = self._segment_roles.get(
                (int(m["clip_idx"]), int(round(float(m["window_start_s"]) / window))))
            if role is not None:
                groups[role].append(i)
        n_assigned = sum(len(v) for v in groups.values())
        if n_assigned != len(self._segment_roles):
            raise RuntimeError(
                f"split_mode=paper_segments: {len(self._segment_roles)} segments in the paper's fold, "
                f"{n_assigned} found in the embedding cache")
        out: Dict[str, Any] = {}
        for name, idx in groups.items():
            idx.sort(key=lambda i: (int(metadata[i]["clip_idx"]), float(metadata[i]["window_start_s"])))
            arr = np.asarray(idx, dtype=np.int64)
            items = []
            for i in idx:
                m = dict(metadata[i])
                m["split"] = name
                items.append(m)
            out[name] = EmbeddingPayload(
                features=np.asarray(features[arr]) if arr.size else
                np.zeros((0,) + tuple(features.shape[1:]), dtype=features.dtype),
                labels=labels[arr] if arr.size else np.zeros((0,), dtype=labels.dtype),
                metadata=items,
            )
        return out

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
