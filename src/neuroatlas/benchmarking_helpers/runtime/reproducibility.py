"""Reproducibility utilities for deterministic benchmark runs."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set all random-number-generator seeds for reproducible runs.

    Covers Python stdlib, NumPy, PyTorch CPU/CUDA, and cuDNN.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataloader_worker_init_fn(worker_id: int) -> None:
    """Worker init function that re-seeds NumPy and stdlib random per worker.

    Derives a per-worker seed from ``torch.initial_seed()`` so that each
    worker is deterministic yet different from its siblings.  Pass this as
    ``worker_init_fn`` when constructing a :class:`torch.utils.data.DataLoader`.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
