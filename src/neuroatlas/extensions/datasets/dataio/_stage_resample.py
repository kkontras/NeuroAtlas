"""Shared stage-label resampler.

Datasets in this project store sleep-stage annotations on a native 30 s grid
(WSC ``.stg.txt``, DCSM hypnograms, UCDDB ``_stage.txt``, …). Downstream
datamodules expose samples at a configurable ``epoch_seconds``. This helper
converts a sequence of native-epoch labels to any **integer** target epoch
length using a single, uniform rule:

    Expand each native label into ``native_epoch`` per-second copies.
    Group the per-second array into chunks of ``epoch_seconds``.
      - if every per-second label in the chunk agrees -> that label
      - otherwise                                      -> ``unscored`` (-1)

Applied at divisor targets this is exact duplication; at integer multiples of
the native epoch it's the familiar "mixed = unscored" merge. At non-divisor
targets (20, 25, 45, …) it's the principled continuation of the same rule:
windows that happen to lie entirely inside a run of identical native labels
get that label; only windows that genuinely straddle a change of native label
become unscored.
"""

from __future__ import annotations

from typing import List, Sequence


def resample_native_epoch_labels(
    stages_native: Sequence[int],
    epoch_seconds: int,
    native_epoch: int = 30,
    unscored: int = -1,
) -> List[int]:
    """Resample per-native-epoch labels to per-``epoch_seconds`` labels."""
    if int(epoch_seconds) != epoch_seconds or epoch_seconds <= 0:
        raise ValueError(
            f"epoch_seconds must be a positive integer, got {epoch_seconds!r}"
        )
    eps = int(epoch_seconds)
    per_second: List[int] = []
    for s in stages_native:
        per_second.extend([int(s)] * native_epoch)

    out: List[int] = []
    for i in range(0, len(per_second), eps):
        group = per_second[i : i + eps]
        if len(set(group)) == 1:
            out.append(group[0])
        else:
            out.append(unscored)
    return out
