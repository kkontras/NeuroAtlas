"""Adapters for the cognitive and affective cohorts.

EEGMat, ArithmeticTask and DREAMER are not MOABB datasets. Each is read from
the preprocessed pickles its own ``preprocess_*.py`` wrote -- the same files
the published embeddings and probes were built from -- so there is no download
path and no MOABB fallback: if the pickle is absent the run stops and says so.

One class per cohort because ``spec.datamodule`` names a class, but they differ
only in which ``DATASET_CONFIGS`` key they read. DREAMER is two cohorts, valence
and arousal, split that way in the pickles, the embeddings and the probes.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

from neuroatlas.extensions.datasets.dataio.bci import (
    DATASET_CONFIGS,
    get_subject_split,
    load_preprocessed_dataset,
)

from neuroatlas.extensions.models.backbones._preproc import BCI_DOMAIN

from ._runtime_keys import BCI_TRIALS, PREPARED_FILTERS, RAW_ONLY
from .base import BenchmarkDataModule
from .cho2017 import _PreprocessedDataset, _collate


class _LoaderAdapter:
    """Emit the standard batch for one cohort, with *its* rate and channels.

    These cohorts used to borrow Cho2017's adapter, which stamped every batch
    with ``dataset="cho2017"``, Cho2017's 100 Hz and its 64 channel names --
    so a 128 Hz, 19-channel EEGMat trial would have reached the wrappers
    mislabelled -- and read ``subject_ids`` from a batch that the default
    collate had keyed ``subject_id``.
    """

    def __init__(self, loader, *, slug: str, sampling_rate: float,
                 channels: List[str], channel_index: Optional[List[int]]) -> None:
        self._loader = loader
        self._slug = slug
        self._sampling_rate = float(sampling_rate)
        self._channels = list(channels)
        self._channel_index = channel_index

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        for batch in self._loader:
            eeg = batch["eeg"]
            if self._channel_index is not None:
                eeg = eeg[:, self._channel_index, :]
            meta = [
                {
                    "dataset": self._slug,
                    "domain": BCI_DOMAIN,
                    "subject_id": batch["subject_ids"][i],
                    "trial_idx": batch["trial_idxs"][i],
                    "sampling_rate": self._sampling_rate,
                    "unit": "uV",
                    "channels": list(self._channels),
                }
                for i in range(len(batch["subject_ids"]))
            ]
            yield {"signals": {"eeg": eeg}, "label": batch["label"],
                   "meta": meta, "raw_batch": batch}


def _channel_selection(slug: str, available: Sequence[str],
                       channel_specs: Optional[Sequence[str]]) -> Optional[List[int]]:
    """Indices of *channel_specs* in the file's channel order; None keeps them as stored."""
    if channel_specs is None:
        return None
    wanted = list(channel_specs)
    missing = [c for c in wanted if c not in available]
    if missing:
        raise ValueError(
            f"{slug}: channel_specs names channels the preprocessed file does not "
            f"have: {missing}. It has: {list(available)}"
        )
    index = [list(available).index(c) for c in wanted]
    return None if index == list(range(len(available))) else index


class _PickleCohortDataModule(BenchmarkDataModule):
    """Shared implementation. Subclasses set :attr:`SLUG`."""

    SLUG: str = ""
    RUNTIME_KEYS_FIXED = RAW_ONLY
    RUNTIME_KEYS_IGNORED = {**BCI_TRIALS, **PREPARED_FILTERS}
    FIXED_WINDOW = True

    def __init__(
        self,
        fold: int = 0,
        n_folds: int = 5,
        batch_size: int = 64,
        num_workers: int = 0,
        subject_ids: Optional[Sequence[int]] = None,
        preprocessed_path: Optional[str] = None,
        channel_specs: Optional[Sequence[str]] = None,
    ) -> None:
        cfg = DATASET_CONFIGS[self.SLUG]
        available = list(cfg.channels)
        # The channel map's channels_used, in its order; the file's own when absent.
        self._channel_index = _channel_selection(self.SLUG, available, channel_specs)
        channels = list(channel_specs) if channel_specs is not None else available
        subjects = list(subject_ids) if subject_ids is not None else list(cfg.subjects)
        train_subj, val_subj, test_subj = get_subject_split(
            subjects, fold=fold, n_folds=n_folds
        )
        meta: Dict[str, object] = {
            "domain": BCI_DOMAIN,
            "canonical_label_space": list(cfg.targets),
            "epoch_seconds": cfg.tmax - cfg.tmin,
            "channel_policy": ["eeg"],
            "signal_kind": "raw",
            "fold": fold,
            "n_folds": n_folds,
            "n_channels": len(channels),
            "sfreq": cfg.resample_sfreq,
            "sampling_rate": float(cfg.resample_sfreq),
            "unit": "uV",
            "channels": channels,
            "source": "preprocessed",
        }
        super().__init__(name=self.SLUG, metadata=meta)

        mat = load_preprocessed_dataset(self.SLUG, preprocessed_path=preprocessed_path)
        if mat is None:
            raise FileNotFoundError(
                f"{self.SLUG}: no preprocessed pickle found. This cohort is read "
                f"only from the files preprocess_{self.SLUG}.py produced; there is "
                f"no download path. Point --set preprocessed_path=<file> at one, or "
                f"place it where dataio/bci.py's PREPROCESSED_SEARCH_PATHS expects."
            )

        names = np.asarray(mat["subject_name"]).squeeze()
        index = {int(np.asarray(names[i]).flat[0]): i for i in range(names.shape[0])}
        pick = lambda group: [index[s] for s in group if s in index]
        self._train_ds = _PreprocessedDataset(mat, pick(train_subj))
        self._val_ds = _PreprocessedDataset(mat, pick(val_subj))
        self._test_ds = _PreprocessedDataset(mat, pick(test_subj))
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._channels = channels
        self._sampling_rate = float(cfg.resample_sfreq)

    def _make_loader(self, dataset, shuffle: bool):
        from torch.utils.data import DataLoader

        return _LoaderAdapter(
            DataLoader(
                dataset,
                batch_size=self._batch_size,
                shuffle=shuffle,
                num_workers=self._num_workers,
                collate_fn=_collate,
            ),
            slug=self.SLUG,
            sampling_rate=self._sampling_rate,
            channels=self._channels,
            channel_index=self._channel_index,
        )

    def train_dataloader(self):
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self._val_ds, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self._test_ds, shuffle=False)


class EEGMatBenchmarkDataModule(_PickleCohortDataModule):
    """EEGMat: rest (0) against mental arithmetic (1), 36 subjects."""

    SLUG = "eegmat"


class ArithmeticTaskBenchmarkDataModule(_PickleCohortDataModule):
    """ArithmeticTask: rest (0), arithmetic (1), meditation/breathing (2)."""

    SLUG = "arithmetic_task"


class DreamerValenceBenchmarkDataModule(_PickleCohortDataModule):
    """DREAMER valence, binarised at 3 on the 1-5 self-report scale."""

    SLUG = "dreamer_valence"


class DreamerArousalBenchmarkDataModule(_PickleCohortDataModule):
    """DREAMER arousal, binarised at 3 on the 1-5 self-report scale."""

    SLUG = "dreamer_arousal"
