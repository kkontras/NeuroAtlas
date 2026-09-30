"""Build a dataset's cache — the optional step between `fetch` and `embed`.

Most cohorts do not need this.  Every sleep and brain-age dataset reads its
raw corpus directly, and so do the epilepsy ones: `chbmit` and `siena` read
BIDS, `sz1`, `sz2` and `tusz` read EDF, `tuab` reads EDF, `bonn` reads its
plain-text clips.  For those, a prebuilt cache is a speed optimisation for
large sweeps and nothing else -- `embed` works without it.

The exception is BCI, where MOABB has to download and epoch the corpus before
anything can read it.  There, `prepare` is required.

It dispatches on ``pipeline.preprocessor`` in the dataset manifest, which
records the builder module and which of its flags take the raw corpus and the
destination -- those flag names differ per builder (``--raw-dir`` vs
``--raw-root`` vs ``--data-root``), so they are declared rather than guessed.

Usage
-----
    python -m neuroatlas.entrypoints.prepare --list
    python -m neuroatlas.entrypoints.prepare --dataset bonn --dry-run
    python -m neuroatlas.entrypoints.prepare --dataset tusz --shard 0/8
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuroatlas import config as user_config
from neuroatlas.entrypoints._common import (
    expand_dataset_paths,
    parse_embed_chunk,
)

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
            f"error: {slug} declares no pipeline.preprocessor — it reads its raw "
            f"corpus directly, so there is nothing to prepare.\n"
            f"Run `embed --dataset {slug}` once `fetch` has the data."
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
        if not raw:
            raise SystemExit(
                f"error: {slug} has no raw corpus path in its manifest "
                f"(looked for {', '.join(_RAW_KEYS)}). Pass --set raw_root=..."
            )
        argv += [args["raw"], str(raw)]
    out = dest or next((defaults[k] for k in _OUT_KEYS if defaults.get(k)), None)
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
                f"error: {slug}'s builder does not support sharding "
                f"(no pipeline.preprocessor.shard in its manifest)."
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


def build_parser(argv: Optional[List[str]] = None) -> argparse.ArgumentParser:
    from neuroatlas.entrypoints import _help

    parser = argparse.ArgumentParser(
        prog="neuroatlas prepare",
        description="Build a dataset's cache. Optional for every cohort that "
                    "reads raw; required for BCI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_help.build_epilog(argv, show_models=False),
    )
    parser.add_argument("--dataset", help="DatasetSpec slug.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Override a dataset config key, e.g. --set raw_root=/data.")
    parser.add_argument("--dest", default=None,
                        help="Where the cache is written. Default: the manifest's "
                             "cache_root.")
    parser.add_argument("--shard", default=None, metavar="K/N",
                        help="Build shard K of N, for scheduler fan-out.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the builder command and exit.")
    parser.add_argument("--list", action="store_true",
                        help="Show which datasets have a build step and whether "
                             "it is required.")
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

        rows = []
        for spec in sorted(dataset_specs(), key=lambda s: s.slug):
            manifest = spec.manifest or {}
            if not manifest.get("paper_dataset"):
                continue
            block = builder_for(spec.slug)
            rows.append((spec.slug, manifest.get("domain"), block))
        needed = [r for r in rows if r[1] == "bci" and r[2]]
        optional = [r for r in rows if r[1] != "bci" and r[2]]
        print(f"{len(needed)} datasets REQUIRE a build step (BCI: MOABB epochs "
              f"the corpus first).")
        print(f"{len(optional)} have an OPTIONAL fast-path cache; they read raw "
              f"without it.")
        print(f"{len(rows) - len(needed) - len(optional)} need nothing.\n")
        print(f"{'dataset':28s} {'domain':10s} {'build step':11s} builder")
        print("-" * 84)
        for slug, domain, block in rows:
            if block is None:
                kind, module = "none", "-"
            else:
                kind = "required" if domain == "bci" else "optional"
                module = str(block.get("module", "?")).rsplit(".", 1)[-1]
            print(f"{slug:28s} {str(domain):10s} {kind:11s} {module}")
        return

    if not args.dataset:
        build_parser(argv).error("--dataset is required (or use --list)")

    shard = parse_embed_chunk(args.shard) if args.shard else None
    builder_argv = build_argv(args.dataset, _parse_set(args.overrides),
                              shard, args.dest)
    block = builder_for(args.dataset)
    module = block["module"]

    print(f"$ python -m {module} " + " ".join(builder_argv), flush=True)
    if args.dry_run:
        return
    importlib.import_module(module).main(builder_argv)


if __name__ == "__main__":
    main()
