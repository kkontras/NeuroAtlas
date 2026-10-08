"""Build a persistent preprocessed cache for EPILEPSIAE.

Walks the raw corpus once, applies the exact same pipeline the on-the-fly
Dataset uses (channel-select → resample → filter), concatenates per
recording into a single (19, T) float16 array, and writes it to a
per-recording directory under ``--cache-root`` alongside samplewise
labels and JSON metadata. The runtime
:class:`~neuroatlas.extensions.datasets.dataio.epilepsiae_cached.EpilepsiAECachedDataset`
mmap-reads those files, eliminating the 500 MB ``np.fromfile`` + scipy
``filtfilt`` per-block cost from the embed path.

Submit via Condor (one shard per patient bin). See
a scheduler job file that is not shipped.

Layout::

    <cache-root>/
      <subject_id>/
        <rec_id>/
          signals.npy      # (19, T) float16, resampled+filtered unipolar
          labels.npy       # (T,) uint8 — samplewise binary label
          types.npy        # (T,) uint8 — samplewise seizure type
          meta.json        # rec_id, subject_id, variant, fs, …
          events.json      # seizure event list with pattern codes
          .schema_v1       # written last; marks completion
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-root",
                   default="${EEG_DATA_ROOT}/epilepsiae",
                   help="Root of the raw EPILEPSIAE corpus.")
    p.add_argument("--cache-root", required=True,
                   help="Output directory. One subdir per (subject_id, rec_id).")
    p.add_argument("--variants", nargs="+",
                   default=["surf30", "surfPA", "surfCO"])
    p.add_argument("--target-fs", type=int, default=256)
    p.add_argument("--shard-index", type=int, default=None,
                   help="0-based shard index; pair with --num-shards.")
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--force", action="store_true",
                   help="Rebuild recordings even if the schema marker is present.")
    return p


def _shard_patients(patients, num_shards: int, shard_index: int):
    """Volume-aware bin-pack patients into shards; split heavy ones."""
    total_hours = sum(
        b.duration_s / 3600
        for p in patients for blocks in p.recordings.values() for b in blocks
    )
    mean_budget = total_hours / num_shards
    units = []
    for p in patients:
        pat_total = sum(
            b.duration_s / 3600
            for bs in p.recordings.values() for b in bs
        )
        if pat_total <= mean_budget * 1.5:
            units.append((pat_total, p))
        else:
            n_chunks = max(2, math.ceil(pat_total / mean_budget))
            all_blocks = []
            for rec_id, blocks in p.recordings.items():
                for b in blocks:
                    all_blocks.append((b.duration_s / 3600, rec_id, b))
            all_blocks.sort(key=lambda x: x[0], reverse=True)
            chunk_recs = [{} for _ in range(n_chunks)]
            chunk_h = [0.0] * n_chunks
            for bh, rec_id, block in all_blocks:
                lightest = min(range(n_chunks), key=lambda i: chunk_h[i])
                chunk_recs[lightest].setdefault(rec_id, []).append(block)
                chunk_h[lightest] += bh
            for i in range(n_chunks):
                if chunk_recs[i]:
                    units.append((chunk_h[i], replace(p, recordings=chunk_recs[i])))
    units.sort(key=lambda x: x[0], reverse=True)
    bins = [[] for _ in range(num_shards)]
    bin_load = [0.0] * num_shards
    for hours, u in units:
        lightest = min(range(num_shards), key=lambda i: bin_load[i])
        bins[lightest].append(u)
        bin_load[lightest] += hours
    logger.info(
        "Shard %d/%d: %d patient chunks (%.0f h, max shard %.0f h)",
        shard_index, num_shards, len(bins[shard_index]),
        bin_load[shard_index], max(bin_load),
    )
    return bins[shard_index]


def _preprocess_recording(rec, dataset, cache_root: Path, force: bool) -> bool:
    """Write one recording's cache payload. Returns True if built."""
    from neuroatlas.extensions.datasets.dataio.epilepsiae_cached import (
        recording_dir,
        is_recording_complete,
        mark_recording_complete,
        _SIGNALS_FILE,
        _LABELS_FILE,
        _TYPES_FILE,
        _META_FILE,
        _EVENTS_FILE,
        SIGNALS_DTYPE,
    )

    if not force and is_recording_complete(cache_root, rec.subject_id, rec.rec_id):
        return False

    rec_dir = recording_dir(cache_root, rec.subject_id, rec.rec_id)
    rec_dir.mkdir(parents=True, exist_ok=True)

    total = rec.total_samples
    signals = np.zeros((19, total), dtype=np.float32)
    cum_starts = rec.block_cumulative_starts
    for bi, block in enumerate(rec.blocks):
        block_start = cum_starts[bi]
        processed = dataset._load_processed_block(block)  # (19, T_block), fp32
        actual_len = processed.shape[1]
        end = min(block_start + actual_len, total)
        n = end - block_start
        if n > 0:
            signals[:, block_start:end] = processed[:, :n]
    # Gap regions are already zero (np.zeros init).

    # Float16 storage: EEG amplitudes are sub-millivolt; fp16 mantissa
    # gives ≈0.1 µV quantisation at 100 µV — far below sensor noise.
    signals_f16 = signals.astype(SIGNALS_DTYPE)

    np.save(rec_dir / _SIGNALS_FILE, signals_f16)
    np.save(rec_dir / _LABELS_FILE, rec.samplewise_label)
    np.save(rec_dir / _TYPES_FILE, rec.samplewise_type)

    meta = {
        "rec_id": rec.rec_id,
        "subject_id": rec.subject_id,
        "variant": rec.variant,
        "native_fs": int(rec.native_fs),
        "target_fs": int(dataset._target_fs),
        "total_samples": int(total),
        "n_missing_channels": int(rec.n_missing_channels),
        "gender": rec.gender,
        "age": int(rec.age),
        "onset_age": int(rec.onset_age),
        "hospital": rec.hospital,
        "focus_localisation": rec.focus_localisation,
        "signals_dtype": str(np.dtype(SIGNALS_DTYPE)),
        "n_blocks": len(rec.blocks),
        "gap_regions": [list(gr) for gr in rec.gap_regions],
    }
    (rec_dir / _META_FILE).write_text(json.dumps(meta))
    (rec_dir / _EVENTS_FILE).write_text(json.dumps(rec.events))

    mark_recording_complete(rec_dir)
    return True


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from neuroatlas.extensions.datasets.dataio.epilepsiae import (
        EpilepsiAEContinuousDataset,
    )
    from neuroatlas.extensions.datasets.epilepsy.epilepsiae_preprocessor import (
        EpilepsiAEPreprocessor,
    )

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    pp = EpilepsiAEPreprocessor(
        data_root=Path(args.data_root), target_fs=args.target_fs,
    )
    patients = pp.discover_patients(variants=args.variants)

    if args.shard_index is not None and args.num_shards is not None:
        patients = _shard_patients(patients, args.num_shards, args.shard_index)

    if not patients:
        logger.warning("epilepsiae: no patient to prepare in this shard")
        return 0

    # Instantiate the on-the-fly Dataset purely to reuse its block loader
    # (_load_processed_block) and metadata layout. No windowing is done here;
    # we iterate ds._recordings directly.
    ds = EpilepsiAEContinuousDataset(
        data_root=args.data_root,
        patients=patients,
        annotations=pp.annotations,
        origin_map=pp.origin_map,
        window_s=1.0,
        stride_s=1.0,
        montage="unipolar",
        label_mode="binary",
        normalize="none",
        variants=args.variants,
        target_fs=args.target_fs,
    )

    from neuroatlas import progress

    built = skipped = 0
    # on the live line of `neuroatlas data prepare` (nothing when run on its own)
    item = progress.current().phase("preparing recordings", total=len(ds._recordings),
                                    unit="recordings")
    for rec in ds._recordings:
        if _preprocess_recording(rec, ds, cache_root, args.force):
            built += 1
        else:
            skipped += 1
        item.update(advance=1)
    logger.info("epilepsiae: %d recordings prepared, %d already there (%d in all)",
                built, skipped, built + skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
