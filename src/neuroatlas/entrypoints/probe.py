"""Fit a probe on extracted embeddings — any dataset, any model, any task.

This is the single probing entrypoint.  It replaces the 23 near-identical
``probe_<dataset>_fm.py`` modules, whose entire per-dataset content was a
dataset slug, a data root, a montage string and a small task preset — all of
which now live in the dataset manifest and in ``src/neuroatlas/configs/tasks/``.

Probing is CPU-only and runs once per ``(dataset, model, fold, C)``; extraction
is GPU-bound and runs once per ``(dataset, model)``.  Keeping them apart is why
``embed`` and ``probe`` are separate commands: a probe job should never need a
GPU allocation.

Usage
-----
    # sleep staging, the default task for the dataset
    python -m neuroatlas.entrypoints.probe --dataset isruc --models biot,labram

    # a different task, with its preset
    python -m neuroatlas.entrypoints.probe --dataset isruc --task sex

    # override anything the manifest declares
    python -m neuroatlas.entrypoints.probe --dataset dod \\
        --set label_mode=osa_group --task patient_classification

    # see exactly what would run, without touching data
    python -m neuroatlas.entrypoints.probe --dataset mass --task arousal --dry-run

    # run a saved JSON config verbatim (the old `benchmark` entrypoint)
    python -m neuroatlas.entrypoints.probe --config my_sweep.json

Task presets
------------
``--task`` accepts either a registered task slug (``linear_probe``,
``patient_classification``, ``seizure_detection``, ...) or the name of a preset
in ``src/neuroatlas/configs/tasks/``.  A preset is the ``task`` block plus any dataset keys the
task needs — for instance ``mass_arousal`` carries ``subset_filter: SS01``,
because only MASS subset SS01 has arousal annotations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuroatlas import config as user_config
from neuroatlas.cli import ErrorParser
from neuroatlas.entrypoints._common import (
    check_dataset_paths,
    expand_dataset_paths,
    parse_embed_chunk,
    resolve_models_arg,
)
from neuroatlas.entrypoints import _help
from neuroatlas._paths import configs_dir, output_dir

TASKS_DIR = configs_dir("tasks")


# --------------------------------------------------------------------------
# argument coercion — shared with embed.py's --set
# --------------------------------------------------------------------------

def _parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_list(value: str) -> List[int]:
    out: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:                      # "0-4" -> 0,1,2,3,4
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _coerce(raw: str) -> Any:
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


def _parse_set(pairs: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key.strip()] = _coerce(value)
    return out


def _parse_checkpoints(pairs: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--checkpoint expects model=path, got {pair!r}")
        model, _, path = pair.partition("=")
        out[model.strip()] = {"checkpoint_path": path.strip(), "status": "ready"}
    return out


# --------------------------------------------------------------------------
# task presets
# --------------------------------------------------------------------------

def available_tasks() -> List[str]:
    if not TASKS_DIR.is_dir():
        return []
    return sorted(p.stem for p in TASKS_DIR.glob("*.json"))


def load_task_preset(name: str) -> Dict[str, Any]:
    """Return ``{"task", "probe", "dataset_defaults", "datasets"}`` for a preset.

    A bare task slug (one registered in ``extensions/tasks/``) is accepted and
    becomes ``{"task": {"name": name}}``, so ``--task linear_probe`` works
    without a preset file.
    """
    path = TASKS_DIR / f"{name}.json"
    if path.is_file():
        preset = json.loads(path.read_text(encoding="utf-8"))
        if "task" not in preset:
            raise SystemExit(f"error: task preset {path} has no 'task' block")
        return preset

    from neuroatlas.benchmarking_helpers.registry.discovery import task_specs

    registered = {spec.slug for spec in task_specs()}
    if name in registered:
        return {"task": {"name": name}}

    raise SystemExit(
        f"error: unknown task {name!r}\n"
        f"fix: neuroatlas list tasks names the registered tasks and presets"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser(argv: Optional[List[str]] = None) -> argparse.ArgumentParser:
    parser = ErrorParser(
        prog="neuroatlas probe",
        description="Fit a probe on extracted embeddings for any dataset/model/task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_help.build_epilog(argv, show_models=True),
    )
    parser.add_argument("--config", default=None,
                        help="Run a JSON benchmark config verbatim (the former "
                             "`benchmark` entrypoint). Mutually exclusive with --dataset.")
    parser.add_argument("--dataset", default=None, help="DatasetSpec slug.")
    parser.add_argument("-m", "--models", default=None,
                        help="An alias (all_fm, all_ts, all_supervised, all_random), a "
                             "group, a family or checkpoint ids, comma-separated; `all` "
                             "(the default) is every checkpoint in the registry. See "
                             "`neuroatlas list aliases`.")
    parser.add_argument("--task", default=None,
                        help="Task slug or preset name. Default: the dataset's "
                             "DatasetSpec.default_task.")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                        help="Dataset config override, repeatable.")
    parser.add_argument("--checkpoint", dest="checkpoints", action="append",
                        metavar="MODEL=PATH", help="Override a checkpoint path, repeatable.")
    parser.add_argument("--folds", default=None,
                        help="Fold indices: '0,1,2' or a range '0-4'. "
                             "Default: the dataset's fold count.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--probe-type", default="linear",
                        choices=["linear", "sklearn_linear", "nonlinear"])
    parser.add_argument("--hidden-dims", default="256,128",
                        help="Hidden sizes for --probe-type nonlinear.")
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--class-weight", choices=["balanced"], default=None)
    parser.add_argument("--selection-metric", default="macro_f1",
                        help="Validation metric that picks the probe's C (and seed). Seizure "
                             "detection: auprc (the paper's), auroc, or event_sens_fa_auc "
                             "(the event-level Sens@FA AUC on the validation fold).")
    parser.add_argument("--tune-c", default=None,
                        help="Comma-separated C values for the regularisation sweep.")
    parser.add_argument("--aggregation", default=None,
                        help="Subject aggregation for subject-level tasks, e.g. "
                             "'mean' or 'mean,mean_std' (runs once per value).")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--seed-mode", choices=["fold", "shared"], default="fold")
    parser.add_argument("--pooling", choices=["mean", "per_patch"], default="mean",
                        help="Which cached embeddings to probe, for each "
                             "window. mean: the one vector averaged over that "
                             "window's patch tokens. per_patch: that window's "
                             "tokens kept separate. Must match what `embed "
                             "--pooling` produced.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--cache-root", default=None,
                        help="Embedding cache root. Default: $EEG_CACHE_ROOT, else the "
                             "shared cache.")
    parser.add_argument("--extract-only", action="store_true",
                        help="Extract embeddings and exit (prefer `embed`).")
    parser.add_argument("--embed-chunk", default=None, metavar="K/N")
    parser.add_argument("--no-recording-norm", action="store_true")
    parser.add_argument("--no-amplitude-scale", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved config as JSON and exit.")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List registered tasks and presets, then exit.")
    return parser


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
            message += " (did you mean " + ", ".join(near) + "?)"
        raise SystemExit(f"error: {message}\nfix: neuroatlas probe --list-datasets names them all")


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve manifest defaults + task preset + CLI into a runner config."""
    from neuroatlas.benchmarking_helpers.runtime.cache import SHARED_EMBEDDING_CACHE_ROOT
    from neuroatlas.benchmarking_helpers.registry.discovery import (
        checkpoint_registry,
        load_dataset_spec,
    )

    spec = _resolve_dataset(args.dataset)

    task_name = args.task or spec.default_task
    preset = load_task_preset(task_name)
    task_block: Dict[str, Any] = dict(preset["task"])

    # precedence: manifest defaults < config.yaml dataset_paths < task preset
    #             < --set < explicit flags
    dataset_config: Dict[str, Any] = dict(spec.config_defaults)
    dataset_config.update(user_config.dataset_paths(spec.slug))
    # `dataset_defaults` applies to whichever dataset the preset is run on;
    # `datasets.<slug>` then refines it for one cohort.
    dataset_config.update(preset.get("dataset_defaults") or {})
    dataset_config.update((preset.get("datasets") or {}).get(spec.slug, {}))
    dataset_config.update(_parse_set(args.overrides))
    if args.data_root is not None:
        dataset_config["data_root"] = args.data_root
    if args.batch_size is not None:
        dataset_config["batch_size"] = args.batch_size
    from neuroatlas.entrypoints.embed import resolve_num_workers

    dataset_config["num_workers"] = resolve_num_workers(args.num_workers, dataset_config)

    if args.folds is not None:
        dataset_config["folds"] = _parse_int_list(args.folds)
    elif "folds" not in dataset_config:
        n_folds = dataset_config.get("n_folds") or dataset_config.get("num_folds")
        if spec.supports_folds and n_folds:
            # "loso" is one fold per subject, so the count comes from the
            # cohort rather than from the flag.
            if isinstance(n_folds, str) and n_folds.strip().lower() == "loso":
                from neuroatlas.extensions.datasets.dataio.moabb_loader import loso_fold_count

                n_folds = loso_fold_count(spec.slug)
                dataset_config["n_folds"] = n_folds
            dataset_config["folds"] = list(range(int(n_folds)))

    if args.aggregation is not None:
        task_block["aggregation"] = args.aggregation

    overrides = _parse_checkpoints(args.checkpoints)
    # An empty list means every registered model. `--models all` says that
    # out loud, so a command reads the same as what it does instead of
    # relying on the reader knowing that the absent flag means everything.
    models = resolve_models_arg(args.models)
    if args.no_recording_norm or args.no_amplitude_scale:
        runtime: Dict[str, Any] = {}
        if args.no_recording_norm:
            runtime["apply_recording_normalization"] = False
        if args.no_amplitude_scale:
            runtime["apply_amplitude_scale"] = False
        for model in models or sorted({c.model_family for c in checkpoint_registry()}):
            overrides.setdefault(model, {})["runtime_overrides"] = runtime

    probe_block: Dict[str, Any] = {
        "type": args.probe_type,
        "max_iter": args.max_iter,
        "class_weight": args.class_weight,
        "selection_metric": args.selection_metric,
    }
    if args.probe_type == "nonlinear":
        probe_block["hidden_dims"] = _parse_int_list(args.hidden_dims)
    if args.tune_c:
        probe_block["c_values"] = [float(c) for c in _parse_csv(args.tune_c)]
    probe_block.update(preset.get("probe") or {})

    dataset_config = expand_dataset_paths(dataset_config)

    cache_env = os.environ.get("EEG_CACHE_ROOT")
    cache_root = (Path(args.cache_root) if args.cache_root
                  else Path(cache_env) if cache_env else SHARED_EMBEDDING_CACHE_ROOT)
    output_root = (Path(args.output_root) if args.output_root
                   else output_dir(spec.slug, task_name))

    return {
        "benchmark": {
            "output_root": str(output_root),
            "cache_root": str(cache_root),
            "seeds": _parse_int_list(args.seeds),
            "seed_per_fold": args.seed_mode == "fold",
            "models": models,
            "checkpoint_overrides": overrides,
            "extract_only": args.extract_only,
            "embed_chunk": parse_embed_chunk(args.embed_chunk),
            # Probing reads embeddings; it does not make them. A cache miss
            # here is an error naming `embed`, not a backbone load — otherwise
            # a CPU-only probe job silently turns into a GPU extraction.
            # --extract-only and --embed-chunk are the deliberate exceptions.
            "require_cached_embeddings": not args.extract_only,
        },
        "probe": probe_block,
        "task": task_block,
        "datasets": {spec.slug: dataset_config},
    }



def _required_keys(slug: str):
    """Keys the dataset's manifest marks as having no default."""
    try:
        from neuroatlas.benchmarking_helpers.registry.discovery import load_dataset_spec

        return (load_dataset_spec(slug).manifest or {}).get("runtime_required")
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> None:
    user_config.apply_to_environ()
    argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser(argv)
    args = parser.parse_args(argv)

    if args.list_tasks:
        from neuroatlas.benchmarking_helpers.registry.discovery import task_specs

        print("registered tasks:")
        for spec in task_specs():
            print(f"  {spec.slug:<28} {spec.description}")
        presets = available_tasks()
        print("\npresets in src/neuroatlas/configs/tasks/:" if presets else "\nno presets in src/neuroatlas/configs/tasks/")
        for name in presets:
            preset = json.loads((TASKS_DIR / f"{name}.json").read_text())
            print(f"  {name:<28} task={preset['task'].get('name')}")
        return

    if bool(args.config) == bool(args.dataset):
        parser.error("give exactly one of --config or --dataset")

    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    else:
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

    from neuroatlas.benchmarking_helpers import BenchmarkRunner, seed_everything

    _refuse_unsupported_probe_settings(config)
    seed_everything(config.get("benchmark", {}).get("seed", args.seed))
    runner = BenchmarkRunner(config)
    # Imported here, not at module scope: linear_probe pulls in torch, and
    # probing is CPU-only by design -- `--help` and config assembly must not
    # drag in the GPU stack.
    import contextlib

    from neuroatlas.benchmarking_helpers.runtime.resources import limit_probe_threads
    from neuroatlas.extensions.tasks.linear_probe import embedding_pooling

    # BLAS/OpenMP threads: at most 8 (see resources.MAX_PROBE_THREADS). Left
    # alone, OpenBLAS starts one thread per core of the node -- 28 on
    # deanston, where the probe ran 2.5x slower than on an 8-core desktop.
    threads = (contextlib.nullcontext() if config["benchmark"].get("extract_only")
               else limit_probe_threads())
    with embedding_pooling(args.pooling), threads:
        results = runner.run()

    from neuroatlas.entrypoints._common import report_results

    if report_results("probe", results, f"results: {runner.output_root}"):
        sys.exit(1)


def _refuse_unsupported_probe_settings(config) -> None:
    """Refuse, once and before anything is read, probe flags the task cannot
    honour; otherwise every (model, fold) fails with the same message."""
    from neuroatlas.benchmarking_helpers.registry.discovery import load_dataset_spec
    from neuroatlas.extensions.tasks.seizure_detection import ProbeSettingsError, probe_settings

    task = (config.get("task") or {}).get("name")
    tasks = {task} if task else {load_dataset_spec(slug).default_task
                                 for slug in config.get("datasets", {})}
    if "seizure_detection" in tasks:
        try:
            probe_settings(config.get("probe"))
        except ProbeSettingsError as exc:
            from neuroatlas.cli import _msg

            _msg.error(str(exc), "neuroatlas probe --help (the options seizure detection takes)")
            raise SystemExit(2) from None


if __name__ == "__main__":
    main()
