"""Batch sampler that groups epochs by subject.

Every batch contains epochs from exactly one subject, guaranteeing
homogeneous ``sampling_rate``, ``unit``, ``channels``, and per-recording
statistics within each batch.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, List

import torch
from torch.utils.data import Sampler


class SubjectBatchSampler(Sampler[List[int]]):

    def __init__(
        self,
        subject_ids: List[str],
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        by_subject: dict[str, list[int]] = defaultdict(list)
        for i, sid in enumerate(subject_ids):
            by_subject[sid].append(i)

        self._batches: List[List[int]] = []
        for indices in by_subject.values():
            for start in range(0, len(indices), batch_size):
                batch = indices[start : start + batch_size]
                if drop_last and len(batch) < batch_size:
                    continue
                self._batches.append(batch)
        self._shuffle = shuffle

    def __iter__(self) -> Iterator[List[int]]:
        if self._shuffle:
            order = torch.randperm(len(self._batches)).tolist()
        else:
            order = list(range(len(self._batches)))
        return (self._batches[i] for i in order)

    def __len__(self) -> int:
        return len(self._batches)
