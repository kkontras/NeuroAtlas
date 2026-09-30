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

from typing import Dict, Optional, Sequence

import numpy as np

from neuroatlas.extensions.datasets.dataio.bci import (
    DATASET_CONFIGS,
    get_subject_split,
    load_preprocessed_dataset,
)

from .base import BenchmarkDataModule
from .cho2017 import _PreprocessedDataset, _LoaderAdapter


class _PickleCohortDataModule(BenchmarkDataModule):
    """Shared implementation. Subclasses set :attr:`SLUG`."""

    SLUG: str = ""

    def __init__(
        self,
        fold: int = 0,
        n_folds: int = 5,
        batch_size: int = 64,
        num_workers: int = 0,
        subject_ids: Optional[Sequence[int]] = None,
        preprocessed_path: Optional[str] = None,
    ) -> None:
        cfg = DATASET_CONFIGS[self.SLUG]
        subjects = list(subject_ids) if subject_ids is not None else list(cfg.subjects)
        train_subj, val_subj, test_subj = get_subject_split(
            subjects, fold=fold, n_folds=n_folds
        )
        meta: Dict[str, object] = {
            "canonical_label_space": list(cfg.targets),
            "epoch_seconds": cfg.tmax - cfg.tmin,
            "channel_policy": ["eeg"],
            "signal_kind": "raw",
            "fold": fold,
            "n_folds": n_folds,
            "n_channels": len(cfg.channels),
            "sfreq": cfg.resample_sfreq,
            "sampling_rate": float(cfg.resample_sfreq),
            "unit": "uV",
            "channels": list(cfg.channels),
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

    def _make_loader(self, dataset, shuffle: bool):
        from torch.utils.data import DataLoader

        return _LoaderAdapter(
            DataLoader(
                dataset,
                batch_size=self._batch_size,
                shuffle=shuffle,
                num_workers=self._num_workers,
            )
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
