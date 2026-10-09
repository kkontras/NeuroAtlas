"""Build a dataset's prepared file: a step between `data download` and `run`.

Four datasets need it: the bci_cognitive ones (dreamer_valence,
dreamer_arousal, eegmat, arithmetic_task) are read from the files their
builder writes from the raw data. Every sleep and brain-age dataset reads its
raw data directly, and so do the epilepsy ones: `chbmit` and `siena` read
BIDS, `sz1`, `sz2` and `tusz` read EDF, `tuab` reads EDF, `bonn` reads its
plain-text clips. For five of them (bonn, epilepsiae, sz1, tuab, tusz) a
prepared file is a speed-up for large sweeps and nothing else -- `embed`
works without it. The MOABB datasets load through MOABB's own reader; the
pickle the five motor-imagery builders write (bnci2014_001, bnci2014_004,
bnci2015_001, shin2017a, weibo2014) is not read by `run` (``required: false``
in their manifests; :func:`neuroatlas.catalog.prepare_required`).

It dispatches on ``pipeline.preprocessor`` in the dataset manifest, which
records the builder module and which of its flags take the raw corpus and the
destination -- those flag names differ per builder (``--raw-dir`` vs
``--raw-root`` vs ``--data-root``), so they are declared rather than guessed.

Prepared files go under ``<cache root>/prepared`` (the manifests write
``${EEG_CACHE_ROOT}/prepared/<slug>``; the BCI pickles go to
``<cache root>/prepared/<Name>/``), never into the source tree.

``neuroatlas data prepare <dataset>`` is the front door: it refuses until
the raw data is there. This module is the old ``prepare`` verb it calls.

Usage
-----
    neuroatlas data prepare --list
    neuroatlas data prepare bonn --dry-run
    neuroatlas data prepare tusz --shard 0/8
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuroatlas import config as user_config
from neuroatlas.cli import ErrorParser, LinesFormatter
from neuroatlas.entrypoints._common import (
    expand_dataset_paths,
    parse_embed_chunk,
)

logger = logging.getLogger(__name__)

# Keys a manifest may use for the raw corpus, most specific first.
_RAW_KEYS = ("raw_dir", "raw_root", "bids_root", "data_root")
_OUT_KEYS = ("cache_root", "output")


def _spec(slug: str):
    from neuroatlas.benchmarking_helpers.registry.discovery import load_dataset_spec

    return load_dataset_spec(slug)


def builder_for(slug: str) -> Optional[Dict[str, Any]]:
    """The declared preprocessor block, or None when the cohort needs none."""
    manifest = _spec(slug).manifest or {}
    block = (manifest.get("pipeline") or {}).get("preprocessor")
    return block if isinstance(block, dict) else None


def build_argv(slug: str, overrides: Dict[str, Any],
               shard: Optional[tuple], dest: Optional[str]) -> List[str]:
    """Turn the manifest plus the CLI into the builder's own arguments."""
    block = builder_for(slug)
    if block is None:
        raise SystemExit(
            f"error: {slug} has no build step: `neuroatlas run {_benchmark_of(slug)} "
            f"--dataset {slug}` reads its data as it is\n"
            f"fix: neuroatlas data status {slug} (whether that is here)"
        )
    args = block.get("args") or {}
    spec = _spec(slug)
    manifest = spec.manifest or {}

    # A raw root may live under the selected backend rather than at the top
    # level -- tusz keeps raw_root under backends.edf, because the hdf5 backend
    # has no use for it.
    merged = dict(spec.config_defaults)
    backends = manifest.get("backends") or {}
    chosen = overrides.get("backend", merged.get("backend"))
    for name, cfg in backends.items():
        if name == chosen or chosen is None:
            merged.update((cfg or {}).get("defaults") or {})
    merged.update(user_config.dataset_paths(slug))
    defaults = expand_dataset_paths({**merged, **overrides})

    argv: List[str] = []
    if "slug" in args:                       # MOABB builders take the slug
        argv += [args["slug"], slug]
    if "raw" in args:
        raw = next((defaults[k] for k in _RAW_KEYS if defaults.get(k)), None)
        if not raw and not any(k in defaults for k in _RAW_KEYS):
            # a reader that reads only the built file (bci_cognitive): the
            # raw data is where `data download` puts it
            from neuroatlas import data

            _, download_dir, unresolved = data.raw_location(slug)
            raw = str(download_dir) if download_dir is not None and not unresolved else None
        if not raw:
            # the key its manifest names for the raw data (set, but empty)
            key = next((k for k in _RAW_KEYS if k in defaults), _RAW_KEYS[0])
            raise SystemExit(
                f"error: {slug}: no raw data folder is set for it\n"
                f"fix: neuroatlas config set {slug}.{key} DIR"
            )
        argv += [args["raw"], str(raw)]
    out = dest or next((defaults[k] for k in _OUT_KEYS if defaults.get(k)), None)
    if out is None and "out" in args and "raw" in args:
        # A raw-corpus builder whose manifest names no cache folder (sz1):
        # under the cache root, never wherever the builder's own default is.
        # (MOABB builders take no raw path and write their PREPROCESSED_SEARCH_PATHS
        # entry, which is already under the cache root.)
        from neuroatlas import _paths

        out = str(_paths.prepared_dir(slug))
    if "out" in args and out:
        # Some builders write a single file (bonn's HDF5), others a directory.
        # `cache_filename` in the manifest is what distinguishes them.
        filename = defaults.get("cache_filename")
        if filename and not str(out).endswith(str(filename)):
            out = str(Path(out) / str(filename))
        argv += [args["out"], str(out)]

    if shard is not None:
        shard_args = block.get("shard")
        if not shard_args:
            raise SystemExit(
                f"error: {slug} cannot be built in shards\n"
                f"fix: neuroatlas data prepare {slug} (without --shard)"
            )
        index, count = shard
        argv += [shard_args["index"], str(index), shard_args["count"], str(count)]

    # anything else the user set that the builder names as a flag
    known = {v for k, v in args.items() if k not in ("slug",)}
    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        if flag not in argv and flag not in known:
            argv += [flag, str(value)]
    return argv


def _benchmark_of(slug: str) -> str:
    """A benchmark that runs *slug* (the first, when several do), for a hint."""
    try:
        from neuroatlas import catalog

        # a benchmark of its own (not one derived from another's results)
        names = [n for n in catalog.benchmarks_using(slug) if not catalog.load(n).derived_from]
    except Exception:
        names = []
    return names[0] if names else "BENCHMARK"


def _flag_value(argv: List[str], flag: Optional[str]) -> Optional[str]:
    """The value after *flag* in a builder's arguments."""
    if flag and flag in argv:
        i = argv.index(flag)
        return argv[i + 1] if i + 1 < len(argv) else None
    return None


def describe(slug: str, builder_argv: List[str]) -> str:
    """What a build does, in words: ``bonn's prepared file at /x.h5 from /raw``."""
    args = (builder_for(slug) or {}).get("args") or {}
    raw = _flag_value(builder_argv, args.get("raw"))
    out = _flag_value(builder_argv, args.get("out"))
    source = raw or ("its MOABB data" if "slug" in args else None)
    text = f"{slug}'s prepared file"
    if out:
        text += f" at {out}"
    if source:
        text += f" from {source}"
    return text


def build_parser(argv: Optional[List[str]] = None) -> argparse.ArgumentParser:
    from neuroatlas.entrypoints import _help

    parser = ErrorParser(
        prog="neuroatlas prepare",
        description=(
            "Build the files dreamer_valence, dreamer_arousal, eegmat and arithmetic_task are "
            "read from, or a faster-to-read copy of bonn, epilepsiae, sz1, tuab or tusz "
            "(optional). `neuroatlas data prepare` does the same, after checking that the raw "
            "data is there. The files go to `<cache root>/prepared`."),
        formatter_class=LinesFormatter,
        epilog=_help.build_epilog(argv, show_models=False),
    )
    parser.add_argument("--dataset", help="Dataset name, as listed by `neuroatlas list "
                                           "datasets`.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Change a dataset setting, as in --set raw_root=/data.")
    parser.add_argument("--dest", default=None,
                        help="Where to write the copy (default: `<cache root>/prepared`).")
    parser.add_argument("--shard", default=None, metavar="K/N",
                        help="Build only shard K of N, to split the work across cluster jobs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be built and from where, then exit.")
    parser.add_argument("--list", action="store_true",
                        help="List the datasets that have a build step.")
    return parser


def _parse_set(pairs: List[str]) -> Dict[str, Any]:
    from neuroatlas.entrypoints.probe import _parse_set as shared

    return shared(pairs)


def main(argv: Optional[List[str]] = None) -> None:
    user_config.apply_to_environ()
    argv = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser(argv).parse_args(argv)

    if args.list:
        from neuroatlas.benchmarking_helpers.registry.discovery import dataset_specs
        from neuroatlas.catalog import prepare_required

        rows = []
        for spec in sorted(dataset_specs(), key=lambda s: s.slug):
            manifest = spec.manifest or {}
            if not manifest.get("paper_dataset"):
                continue
            block = builder_for(spec.slug)
            rows.append((spec.slug, manifest.get("domain"), block, prepare_required(spec.slug)))
        needed = [r for r in rows if r[2] and r[3]]
        # the motor-imagery builders write a file no benchmark reads: not
        # offered (`data prepare <it>` still builds it)
        built = [r for r in rows if r[2] and (r[3] or r[1] != "bci")]
        print(f"{'dataset':20s} {'domain':10s} build step")
        for slug, domain, block, required in built:
            what = ("required: `run` reads the prepared file" if required else
                    "optional: a faster-to-read copy")
            print(f"{slug:20s} {str(domain):10s} {what}")
        print(f"\n{len(built)} of the paper's {len(rows)} datasets have a build step"
              + (f", {len(needed)} of them required" if needed else
                 "; none is required: every dataset runs without it")
              + ". `neuroatlas data prepare DATASET` builds one.")
        return

    if not args.dataset:
        build_parser(argv).error("--dataset is required (or use --list)")

    shard = parse_embed_chunk(args.shard) if args.shard else None
    builder_argv = build_argv(args.dataset, _parse_set(args.overrides),
                              shard, args.dest)
    block = builder_for(args.dataset)
    module = block["module"]

    what = describe(args.dataset, builder_argv)
    # the builder's own command line, for -v
    logger.debug("builder: python -m %s %s", module, " ".join(builder_argv))
    if args.dry_run:
        print(f"would build {what}", flush=True)
        return
    from neuroatlas import progress

    progress.say(f"building {what}")
    # A builder that reports only to the log (per recording, at INFO) would
    # leave the screen still for hours: a live line with its time meanwhile
    # (what a builder prints clears it first and stays on screen).
    with progress.Progress(f"{args.dataset}", verb="building its prepared file") as item:
        importlib.import_module(module).main(builder_argv)
    progress.say(f"{args.dataset}: ok ({progress.duration(item.seconds)})")


if __name__ == "__main__":
    main()
