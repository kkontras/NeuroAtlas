"""Extract frozen-backbone embeddings for any registered dataset.

This is the single extraction entrypoint. It is the only module that should
load a backbone and touch raw signal; probing reads what this writes.

It is deliberately a thin wrapper over :class:`BenchmarkRunner` with
``extract_only=True`` -- the same code path ``probe_<dataset>_fm.py
--extract-only`` uses today. Every per-model detail (amplitude scaling,
recording normalization, resampling, channel mapping, sequence windowing)
stays where it already lives: in ``CheckpointSpec`` and the backbone
wrappers. Consolidating the CLI does not move any of it.

Per-dataset defaults come from ``DatasetSpec.config_defaults`` rather than
being hardcoded here, so adding a dataset never means adding an entrypoint.

Usage
-----
    # What can I extract, and what are this dataset's defaults?
    python -m neuroatlas.entrypoints.embed --list-datasets
    python -m neuroatlas.entrypoints.embed --dataset dod --dry-run

    # Extract two models over DOD
    python -m neuroatlas.entrypoints.embed --dataset dod \\
        --models biot,labram

    # Dataset-specific options pass through with --set
    python -m neuroatlas.entrypoints.embed --dataset isruc \\
        --models reve --set montage=isruc_eeg_6 --set subgroups=I,II

    # Shard a large cohort across 4 parallel jobs
    python -m neuroatlas.entrypoints.embed --dataset shhs \\
        --models labram --embed-chunk 0/4

Notes
-----
The embedding cache is keyed globally (``purpose="global_embeddings"``), so
one extraction pass serves every fold, label mode and probe type for that
``(dataset, model)`` pair. ``--folds`` therefore defaults to a single pass;
pass it explicitly only if a dataset's windowing genuinely differs per fold.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuroatlas.benchmarking_helpers import BenchmarkRunner, seed_everything
from neuroatlas.benchmarking_helpers.runtime.cache import SHARED_EMBEDDING_CACHE_ROOT
from neuroatlas.benchmarking_helpers.registry.discovery import (
    checkpoint_registry,
    dataset_specs,
    load_dataset_spec,
)
from neuroatlas.entrypoints._common import (
    check_dataset_paths,
    expand_dataset_paths,
    parse_embed_chunk,
)
from neuroatlas.entrypoints import _help


def default_cache_root() -> Path:
    """Where embeddings are cached, in precedence order.

    ``--cache-root``  >  ``$EEG_CACHE_ROOT``  >  ``SHARED_EMBEDDING_CACHE_ROOT``.

    The README has always told users to export ``EEG_CACHE_ROOT``; until now
    nothing read it, so every run silently defaulted to an absolute path on the
    authors' filesystem (audit finding P3/Q2).
    """
    env = os.environ.get("EEG_CACHE_ROOT")
    return Path(env) if env else SHARED_EMBEDDING_CACHE_ROOT


def _parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _coerce(raw: str) -> Any:
    """Coerce a ``--set`` value to bool / int / float / list / str.

    Mirrors how the per-dataset entrypoints typed these options by hand:
    ``--montage dod_eeg_5`` stays a string, ``--n-folds 5`` becomes an int,
    ``--subgroups I,II`` becomes a list.
    """
    low = raw.strip().lower()
    if low in {"true", "yes"}:
        return True
    if low in {"false", "no"}:
        return False
    if low in {"none", "null"}:
        return None
    if "," in raw:
        return [_coerce(part) for part in raw.split(",") if part.strip()]
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def _parse_set(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key.strip()] = _coerce(value)
    return out


def _parse_checkpoints(pairs: List[str]) -> Dict[str, Dict[str, Any]]:
    """``--checkpoint labram=/path/to.pt`` -> checkpoint_overrides entry."""
    out: Dict[str, Dict[str, Any]] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--checkpoint expects model=path, got {pair!r}")
        model, _, path = pair.partition("=")
        out[model.strip()] = {"checkpoint_path": path.strip(), "status": "ready"}
    return out


def build_parser(argv: Optional[List[str]] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neuroatlas.entrypoints.embed",
        description="Extract frozen-backbone embeddings for any registered dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_help.build_epilog(argv, show_models=True),
    )
    parser.add_argument("--dataset", help="DatasetSpec slug (see --list-datasets).")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model families or checkpoint ids, "
                             "or `all` for every checkpoint in the registry "
                             "(also the default when the flag is omitted).")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                        help="Dataset config override, repeatable. Merged on top of "
                             "DatasetSpec.config_defaults (e.g. --set montage=dod_eeg_5).")
    parser.add_argument("--checkpoint", dest="checkpoints", action="append", metavar="MODEL=PATH",
                        help="Override a model's checkpoint path, repeatable.")
    parser.add_argument("--folds", default=None,
                        help="Comma-separated fold indices. Default: one pass (the "
                             "embedding cache is shared across folds).")
    parser.add_argument("--data-root", default=None, help="Override the dataset's data root.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cache-root", default=None,
                        help="Embedding cache root. Default: $EEG_CACHE_ROOT, else "
                             f"{SHARED_EMBEDDING_CACHE_ROOT}.")
    parser.add_argument("--output-root", default=None,
                        help="Run directory. Extraction writes no results here, but the "
                             "runner creates it (default: artifacts/embeddings/<dataset>).")
    parser.add_argument("--embed-chunk", default=None, metavar="K/N",
                        help="Extract subject chunk K of N for parallel jobs, e.g. '0/4'. "
                             "Chunks are merged automatically on the first read, so no "
                             "separate merge step is needed.")
    parser.add_argument("--expected-epoch-seconds", type=float, default=None,
                        help="Override each selected checkpoint's expected epoch length. "
                             "This is the model-to-data contract: a backbone trained on "
                             "30 s epochs fed 10 s windows will extract, and be wrong. "
                             "Every epilepsy launcher sets it, which is why it is here "
                             "and not in --set (it belongs to the checkpoint, not the "
                             "dataset).")
    # No --device: the per-dataset extractors had one, but nothing in the
    # unified runner reads such a key, so the flag would be accepted and
    # silently ignored. Device selection is `cuda` when available else `cpu`;
    # pin a GPU with CUDA_VISIBLE_DEVICES, which the schedulers already set.
    parser.add_argument("--no-recording-norm", action="store_true",
                        help="Disable per-recording normalization (BIOT q95, REVE z-score).")
    parser.add_argument("--no-amplitude-scale", action="store_true",
                        help="Disable fixed amplitude scaling (EEGPT x1000, LaBraM /100, etc.).")
    parser.add_argument("--pooling", choices=["mean", "per_patch"], default="mean",
                        help="What a backbone hands back for each window (the "
                             "window itself is set by --set window_s/stride_s). "
                             "mean: one vector per window, averaged over that "
                             "window's patch tokens. per_patch: that window's "
                             "tokens kept separate. Cached separately, so both "
                             "can coexist.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved config as JSON and exit without extracting. "
                             "Used by the config-parity tests.")
    parser.add_argument("--list-datasets", action="store_true",
                        help="List registered dataset slugs and exit.")
    parser.add_argument("--paper-only", action="store_true",
                        help="With --list-datasets, show only the cohorts the paper "
                             "reports (excludes the ~150 additionally-loadable MOABB "
                             "datasets).")
    return parser


def _list_datasets(*, paper_only: bool = False) -> None:
    """One line per dataset: slug, domain, and whether the paper reports it.

    Domain and paper membership come from the dataset manifest when there is
    one; MOABB datasets have no manifest and are never paper cohorts here.
    """
    rows = []
    for spec in dataset_specs():
        manifest = spec.manifest or {}
        in_paper = bool(manifest.get("paper_dataset"))
        if paper_only and not in_paper:
            continue
        rows.append((spec.slug, manifest.get("domain", "-"),
                     "paper" if in_paper else "", spec.description))
    width = max((len(r[0]) for r in rows), default=20)
    for slug, domain, flag, description in rows:
        print(f"{slug:<{width}}  {domain:<10} {flag:<6} {description}")
    print(f"\n{len(rows)} datasets"
          f"{' in the paper' if paper_only else ' registered'}")


def _resolve_dataset(slug: str):
    """Look up a dataset, turning a miss into advice instead of a traceback.

    A retired slug (tusz_edf_direct) and a typo both arrive here as KeyError;
    the registry's message already says what to type instead, so the only job
    left is to print it rather than a stack trace.
    """
    from neuroatlas.benchmarking_helpers.registry.discovery import (
        dataset_specs,
        load_dataset_spec,
    )

    try:
        return load_dataset_spec(slug)
    except KeyError as exc:
        import difflib

        message = exc.args[0] if exc.args else str(exc)
        slugs = [s.slug for s in dataset_specs()]
        # Substring first (so "sleep" finds all the sleep cohorts), then close
        # spellings, which is what catches an ordinary typo.
        near = [s for s in slugs if slug.lower() in s.lower()][:8]
        near = near or difflib.get_close_matches(slug, slugs, n=5, cutoff=0.6)
        if near and "--dataset" not in message:
            message += "\nDid you mean: " + ", ".join(near)
        raise SystemExit(f"error: {message}\nRun with --list-datasets to see them all.")


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve CLI + DatasetSpec defaults into a BenchmarkRunner config."""
    spec = _resolve_dataset(args.dataset)

    # Per-dataset defaults first, CLI on top -- never the other way round.
    dataset_config: Dict[str, Any] = dict(spec.config_defaults)
    dataset_config.update(_parse_set(args.overrides))
    if args.data_root is not None:
        dataset_config["data_root"] = args.data_root
    if args.batch_size is not None:
        dataset_config["batch_size"] = args.batch_size
    dataset_config["num_workers"] = (
        args.num_workers if args.num_workers is not None
        else dataset_config.get("num_workers", max(1, (os.cpu_count() or 2) - 1))
    )
    if args.folds is not None:
        dataset_config["folds"] = _parse_int_list(args.folds)

    # ${EEG_DATA_ROOT} and friends are expanded here, not in the manifest, so
    # the stored default stays portable and the environment decides at run time.
    dataset_config = expand_dataset_paths(dataset_config)

    # An empty list means every registered model. `--models all` says that
    # out loud, so a command reads the same as what it does instead of
    # relying on the reader knowing that the absent flag means everything.
    if args.models is None or args.models.strip().lower() == "all":
        models = []
    else:
        models = _parse_csv(args.models)
    overrides = _parse_checkpoints(args.checkpoints)

    if args.expected_epoch_seconds is not None:
        # Applies to whichever checkpoints this run selects; with no --models
        # that is every checkpoint in the registry, which is what the launchers
        # that pass it are doing.
        targets = models or sorted({c.model_family for c in checkpoint_registry()})
        for model in targets:
            overrides.setdefault(model, {})["expected_epoch_seconds"] = (
                args.expected_epoch_seconds)

    if args.no_recording_norm or args.no_amplitude_scale:
        runtime: Dict[str, Any] = {}
        if args.no_recording_norm:
            runtime["apply_recording_normalization"] = False
        if args.no_amplitude_scale:
            runtime["apply_amplitude_scale"] = False
        targets = models or sorted({c.model_family for c in checkpoint_registry()})
        for model in targets:
            overrides.setdefault(model, {})["runtime_overrides"] = runtime

    cache_root = Path(args.cache_root) if args.cache_root else default_cache_root()
    output_root = (
        Path(args.output_root) if args.output_root
        else Path("artifacts/embeddings") / spec.slug
    )

    return {
        "benchmark": {
            "output_root": str(output_root),
            "cache_root": str(cache_root),
            "models": models,
            "checkpoint_overrides": overrides,
            "extract_only": True,
            "embed_chunk": parse_embed_chunk(args.embed_chunk),
        },
        "datasets": {spec.slug: dataset_config},
    }



def _required_keys(slug: str):
    """Keys the dataset's manifest marks as having no default."""
    try:
        from neuroatlas.benchmarking_helpers.registry.discovery import load_dataset_spec

        return (load_dataset_spec(slug).manifest or {}).get("runtime_required")
    except Exception:
        return None


def main(argv: List[str] | None = None) -> None:
    argv = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser(argv).parse_args(argv)

    if args.list_datasets:
        _list_datasets(paper_only=args.paper_only)
        return

    if not args.dataset:
        build_parser().error("--dataset is required (or use --list-datasets)")

    config = build_config(args)

    if args.dry_run:
        json.dump(config, sys.stdout, indent=2, sort_keys=True, default=str)
        sys.stdout.write("\n")
        return

    # Last moment before anything is read: every path must be real by now.
    # Deliberately after --dry-run, so `--dry-run` still shows what *would*
    # run on a machine that has the data.
    for slug, block in config.get("datasets", {}).items():
        check_dataset_paths(slug, block, _required_keys(slug))

    seed_everything(args.seed)
    benchmark = config["benchmark"]
    Path(benchmark["output_root"]).mkdir(parents=True, exist_ok=True)
    Path(benchmark["cache_root"]).mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(config)
    # Imported here, not at module scope: linear_probe pulls in torch, and
    # probing is CPU-only by design -- `--help` and config assembly must not
    # drag in the GPU stack.
    from neuroatlas.extensions.tasks.linear_probe import embedding_pooling

    with embedding_pooling(args.pooling):
        results = runner.run()

    failures = [r for r in results if r.failure is not None]
    for r in failures:
        print(f"FAIL {r.dataset_name}/{r.checkpoint_id}: "
              f"{r.failure.code}: {r.failure.message}", file=sys.stderr)
    print(f"[embed] dataset={args.dataset} models={benchmark['models'] or 'ALL'} "
          f"cells={len(results)} failed={len(failures)} cache={benchmark['cache_root']}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
