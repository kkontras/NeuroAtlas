"""EPILEPSIAE samplewise label builder.

Converts seizure event annotations into per-sample binary and type arrays,
accounting for the block-level concatenation layout of continuous HDF5 caches.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._common import TARGET_FS
from .readers import (
    PATTERN_CODES,
    PATTERN_NAMES,
    SEIZURE_TYPE_CODES,
    SEIZURE_TYPE_NAMES,
    BlockMeta,
)

logger = logging.getLogger(__name__)

# Re-export seizure taxonomy so callers can import from one place
__all__ = [
    "PATTERN_CODES", "PATTERN_NAMES",
    "SEIZURE_TYPE_CODES", "SEIZURE_TYPE_NAMES",
    "build_samplewise_labels",
]


def _timestamp_to_sample(
    dt: object,  # datetime
    blocks: List[BlockMeta],
    block_cumulative_starts: List[int],
    target_fs: int,
) -> Optional[int]:
    """Convert an absolute datetime to a sample index in the concatenated recording.

    ``blocks`` must be sorted by ``start_ts``.
    ``block_cumulative_starts[i]`` is the first sample index of block ``i``
    in the concatenated output (accounting for gap insertions).

    Returns None if the timestamp falls outside all blocks and gaps.
    """
    for i, block in enumerate(blocks):
        if block.start_ts <= dt <= block.end_ts:
            offset_s = (dt - block.start_ts).total_seconds()
            sample_in_block = int(round(offset_s * target_fs))
            block_len = int(round(block.num_samples * target_fs / block.sample_freq))
            sample_in_block = min(sample_in_block, block_len - 1)
            return block_cumulative_starts[i] + sample_in_block

    # Timestamp falls in a gap — snap to end of the preceding block
    for i in range(len(blocks) - 1):
        if blocks[i].end_ts < dt < blocks[i + 1].start_ts:
            block_len = int(round(blocks[i].num_samples * target_fs / blocks[i].sample_freq))
            return block_cumulative_starts[i] + block_len - 1

    return None


def build_samplewise_labels(
    seizures: List[object],  # List[SeizureEvent]
    blocks: List[BlockMeta],
    total_samples: int,
    block_cumulative_starts: List[int],
    target_fs: int = TARGET_FS,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Build samplewise binary and type label arrays for a concatenated recording.

    Args:
        seizures: List of :class:`~epilepsiae.annotations.SeizureEvent` objects.
        blocks: Block metadata sorted by ``start_ts``.
        total_samples: Total sample count in the concatenated output.
        block_cumulative_starts: Cumulative sample start for each block.
        target_fs: Target sampling rate (default 256 Hz).

    Returns:
        ``(samplewise_label, samplewise_type, event_dicts)``

        - ``samplewise_label``: ``(total_samples,)`` uint8, 0=bckg, 1=seizure.
          Gap regions are marked separately by the caller.
        - ``samplewise_type``: ``(total_samples,)`` uint8, seizure type code.
        - ``event_dicts``: Per-seizure metadata dicts for HDF5 event arrays.
    """
    labels = np.zeros(total_samples, dtype=np.uint8)
    types = np.zeros(total_samples, dtype=np.uint8)
    event_dicts: List[Dict[str, Any]] = []

    for sz in seizures:
        onset_sample = _timestamp_to_sample(sz.onset, blocks, block_cumulative_starts, target_fs)
        offset_sample = _timestamp_to_sample(sz.offset, blocks, block_cumulative_starts, target_fs)

        if onset_sample is None:
            logger.warning("Seizure %s: onset %s outside all blocks, skipping.",
                           sz.seizure_id, sz.onset)
            continue
        if offset_sample is None:
            logger.warning("Seizure %s: offset %s outside all blocks, snapping to last sample.",
                           sz.seizure_id, sz.offset)
            offset_sample = total_samples - 1

        if offset_sample <= onset_sample:
            logger.warning(
                "Seizure %s: non-positive duration (onset_sample=%d, offset_sample=%d), skipping.",
                sz.seizure_id, onset_sample, offset_sample,
            )
            continue

        onset_sample = max(0, min(onset_sample, total_samples - 1))
        offset_sample = max(0, min(offset_sample, total_samples))

        labels[onset_sample:offset_sample] = 1
        type_code = SEIZURE_TYPE_CODES.get(sz.classification, 0)
        types[onset_sample:offset_sample] = np.maximum(
            types[onset_sample:offset_sample], type_code
        )

        rec_start = blocks[0].start_ts
        event_dicts.append({
            "start_s": (sz.onset - rec_start).total_seconds(),
            "stop_s": (sz.offset - rec_start).total_seconds(),
            "type": type_code,
            "pattern": PATTERN_CODES.get(sz.pattern, 0),
            "classification": sz.classification,
            "vigilance": sz.vigilance,
            "onset_electrode": sz.onset_electrode,
        })

    return labels, types, event_dicts
