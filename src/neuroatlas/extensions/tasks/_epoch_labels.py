"""Epoch-length-aware binary labeling for fraction-valued meta fields.

Tasks in this package store per-epoch event overlap as *fractions*
(``0.0-1.0`` = overlap_seconds / epoch_seconds). Thresholds are expressed
in **absolute seconds** so the comparison is invariant to epoch length:

    positive  <=>  fraction * epoch_seconds > threshold_seconds

``epoch_seconds`` is read per-sample from ``meta["epoch_seconds"]``.
``default_epoch_seconds`` is a fallback for caches that predate the
per-sample field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def binary_labels_from_fraction_seconds(
    metadata: List[Dict[str, Any]],
    field: str,
    threshold_seconds: float,
    default_epoch_seconds: Optional[float] = None,
) -> np.ndarray:
    """Binary labels from a fraction-valued meta field, thresholded in seconds."""
    labels = np.zeros(len(metadata), dtype=np.int64)
    for i, m in enumerate(metadata):
        frac = m.get(field)
        if frac is None:
            continue
        eps = m.get("epoch_seconds", default_epoch_seconds)
        if eps is None:
            continue
        if float(frac) * float(eps) > threshold_seconds:
            labels[i] = 1
    return labels
