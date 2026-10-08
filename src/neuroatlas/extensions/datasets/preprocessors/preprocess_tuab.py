#!/usr/bin/env python
"""CLI entrypoint for building TUAB v3.0.1 continuous HDF5 caches from raw EDFs.

Usage::

    python -m neuroatlas.extensions.datasets.preprocessors.preprocess_tuab \\
        --raw-root ${EEG_DATA_ROOT}/tuab \\
        --cache-root ${EEG_CACHE_ROOT}/prepared/tuab \\
        --splits train eval \\
        --workers 4

Supports sharding for Condor parallelism (``--shard-index`` / ``--num-shards``).
"""

from __future__ import annotations

import argparse
import logging
import os

from neuroatlas.benchmarking_helpers import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build TUAB continuous HDF5 caches.")
    parser.add_argument(
        "--raw-root", required=True,
        help="Root of TUAB EDF data (the directory that contains train/ and eval/).",
    )
    parser.add_argument(
        "--cache-root", required=True,
        help="Output directory for HDF5 files.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "eval"],
        help="Splits to process (default: train eval).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel EDF-reading workers (per-recording level).",
    )
    parser.add_argument(
        "--max-recordings", type=int, default=None,
        help="Cap for debugging.",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print cache summary instead of building.",
    )
    parser.add_argument(
        "--corpus-version", default="v3.0.1",
        help="TUAB corpus version (default: v3.0.1).",
    )
    parser.add_argument(
        "--shard-index", type=int, default=None,
        help="Shard index (0-based) for parallel builds.",
    )
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Total number of shards.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed for reproducibility.",
    )
    args = parser.parse_args(argv)

    seed_everything(args.seed)

    from neuroatlas.extensions.datasets.epilepsy.tuab_preprocessor import (
        CACHE_SCHEMA_TAG,
        build_cache,
        summarize_cache,
    )

    if args.summary:
        for split in args.splits:
            fname = f"tuab_{args.corpus_version}_{split}_{CACHE_SCHEMA_TAG}.h5"
            path = os.path.join(args.cache_root, fname)
            if os.path.exists(path):
                summarize_cache(path)
            else:
                logger.warning("Cache not found: %s", path)
        return

    for split in args.splits:
        logger.info("Processing TUAB split: %s", split)
        out = build_cache(
            raw_root=args.raw_root,
            cache_root=args.cache_root,
            split=split,
            corpus_version=args.corpus_version,
            workers=args.workers,
            max_recordings=args.max_recordings,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        logger.info("Done: %s", out)


if __name__ == "__main__":
    main()
