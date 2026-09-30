"""Preprocess the Bonn EEG epilepsy dataset (Andrzejak et al. 2001).

Input:
    --raw-dir    Directory containing the five per-set subdirectories
                 {Z,O,N,F,S}/, each with 100 plain-text .TXT clips.
                 (Run ``python -m neuroatlas.entrypoints.fetch --dataset bonn --download`` first.)

Output:
    --output     Destination HDF5 file (default:
                 ${EEG_DATA_ROOT}/.../bonn_cache/hdf5/bonn_173hz_segments.h5)

The output is a flat pre-segmented HDF5 with:
    /signals            (N, 1, 4097)  float32   native 173.61 Hz
    /class_label        (N,)          uint8     0=Z, 1=O, 2=N, 3=F, 4=S
    /subset             (N,)          str       "Z"|"O"|"N"|"F"|"S"
    /subject_idx        (N,)          int16     -1 (mapping not public)
    /recording_type     (N,)          str       "surface" | "intracranial"
    /subject_type       (N,)          str       "healthy" | "patient"
    /state              (N,)          str       awake_* | interictal | ictal
    /electrode_location (N,)          str       scalp_10_20 | hippocampal_* | ...
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


_DEFAULT_RAW_ROOT = (
    "${REPO_ROOT}/bonn_cache/raw"
)
_DEFAULT_OUTPUT = (
    "${REPO_ROOT}/bonn_cache/"
    "hdf5/bonn_173hz_segments.h5"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--raw-dir",
        default=_DEFAULT_RAW_ROOT,
        help="Directory containing per-set subdirectories (default: %(default)s).",
    )
    p.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help="Output HDF5 file path (default: %(default)s).",
    )
    p.add_argument(
        "--sets",
        nargs="+",
        default=None,
        help=(
            "Subset of set letters to include (default: all five). "
            "Example: --sets Z S"
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the output HDF5 already exists.",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Print a short summary of the cache after building.",
    )
    return p


def _print_summary(h5_path: Path) -> None:
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        n = f["signals"].shape[0]
        per_set = {}
        for s in f["subset"][:]:
            s = s.decode() if isinstance(s, bytes) else str(s)
            per_set[s] = per_set.get(s, 0) + 1

        print("\n" + "=" * 60)
        print("  Bonn HDF5 Cache Summary")
        print(f"  {h5_path}")
        print("=" * 60)
        print(f"  Clips:             {n}")
        print(f"  Sampling rate:     {f.attrs.get('fs', '?')} Hz")
        print(f"  Samples per clip:  {f.attrs.get('n_samples_per_segment', '?')}")
        print(f"  Per-set counts:    {per_set}")
        print(f"  Schema tag:        {f.attrs.get('schema_tag', '?')}")
        print("=" * 60 + "\n")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    raw_dir = Path(args.raw_dir)
    output = Path(args.output)

    if output.exists() and not args.force:
        logger.info("Output %s already exists — skipping (pass --force to rebuild).",
                    output)
        if args.summary:
            _print_summary(output)
        return

    from neuroatlas.extensions.datasets.epilepsy.bonn_preprocessor import (
        SET_ORDER,
        build_h5_cache,
    )

    sets = tuple(args.sets) if args.sets else SET_ORDER
    build_h5_cache(raw_root=raw_dir, output_path=output, sets=sets)

    if args.summary:
        _print_summary(output)


if __name__ == "__main__":
    main()
