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
from neuroatlas.cli import ErrorParser, LinesFormatter
from neuroatlas.entrypoints._common import (
    check_dataset_paths,
    expand_dataset_paths,
    parse_embed_chunk,
    resolve_models_arg,
)
from neuroatlas.entrypoints import _help
from neuroatlas._paths import configs_dir, output_dir

TASKS_DIR = configs_dir("tasks")

#: The probe command's iteration cap when none is given.
DEFAULT_MAX_ITER = 10_000


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


def usage_error(text: str, fix: str) -> "SystemExit":
    """A usage error from a verb: ``error:`` and ``fix:`` on stderr, exit 2."""
    from neuroatlas.cli import _msg

    _msg.error(text, fix)
    return SystemExit(2)


def _parse_set(pairs: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise usage_error(f"--set expects key=value, got {pair!r}",
                              f"neuroatlas probe --set KEY=VALUE (e.g. --set window_s=10)")
        key, _, value = pair.partition("=")
        out[key.strip()] = _coerce(value)
    return out


def _parse_checkpoints(pairs: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise usage_error(f"--checkpoint expects ID=PATH, got {pair!r}",
                              f"neuroatlas probe --checkpoint ID=PATH (e.g. "
                              f"--checkpoint biot_pretrained=/my/weights.ckpt)")
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
        description=(
            "Fit probes on extracted embeddings, for any dataset, checkpoint and task. "
            "`neuroatlas run` does this for you. Use it directly to try another task or other "
            "probe settings."),
        formatter_class=LinesFormatter,
        epilog=_help.build_epilog(argv, show_models=True),
    )
    parser.add_argument("--config", default=None, metavar="FILE",
                        help="Run a saved JSON config file as written, instead of --dataset.")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name, as listed by `neuroatlas list datasets --all`.")
    parser.add_argument("-m", "--models", default=None, help=_help.MODELS_HELP)
    parser.add_argument("--task", default=None,
                        help="Task to fit, as listed by `neuroatlas list tasks` (default: the "
                             "dataset's own task).")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                        help="Change a dataset setting. Use the same --set values as the "
                             "`embed` that made the embeddings. Can be repeated.")
    parser.add_argument("--checkpoint", dest="checkpoints", action="append",
                        metavar="ID=PATH",
                        help="Load the weights of checkpoint ID from PATH. Can be repeated.")
    parser.add_argument("--folds", default=None,
                        help="Folds to fit, as in 0,1,2 or 0-4 (default: all).")
    parser.add_argument("--data-root", default=None, metavar="DIR",
                        help="Read the dataset from DIR instead of its configured folder.")
    parser.add_argument("--batch-size", type=int, default=None, metavar="N",
                        help="Windows per batch (default: the dataset's own).")
    parser.add_argument("--num-workers", type=int, default=None, metavar="N",
                        help="Data loader workers (default: the dataset's own, else the CPUs "
                             "this job may use minus one, at most 16).")
    parser.add_argument("--probe-type", default="linear",
                        choices=["linear", "sklearn_linear", "nonlinear"],
                        help="Probe model (default: linear). linear and sklearn_linear are both "
                             "the logistic regression of the benchmarks. nonlinear is an MLP.")
    parser.add_argument("--hidden-dims", default="256,128", metavar="N,N",
                        help="Hidden layer sizes for --probe-type nonlinear (default: 256,128).")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, metavar="N",
                        help="Iteration limit of the solver (default: 10000). The BCI "
                             "benchmarks use 1000, and seizure detection keeps its 500 unless "
                             "this is given.")
    parser.add_argument("--class-weight", choices=["balanced"], default=None,
                        help="Weight each class by its inverse frequency (default: unweighted, "
                             "unless the task sets it).")
    parser.add_argument("--selection-metric", default="macro_f1", metavar="METRIC",
                        help="Validation metric that picks the seed (default: macro_f1). "
                             "Seizure detection also ranks C by it, from auprc (its default), "
                             "auroc or event_sens_fa_auc. Other logistic regression probes "
                             "choose C on validation Cohen's kappa.")
    parser.add_argument("--tune-c", default=None, metavar="C,C,...",
                        help="C values to try for the logistic regression, separated by "
                             "commas. A task without a logistic regression refuses it.")
    parser.add_argument("--aggregation", default=None, metavar="NAME[,NAME]",
                        help="How subject-level tasks combine the windows of a subject, as in "
                             "mean or mean,mean_std. Each value is a separate run.")
    parser.add_argument("--seeds", default="0,1,2", metavar="N,N,...",
                        help="Probe seeds for --seed-mode shared, separated by commas "
                             "(default: 0,1,2).")
    parser.add_argument("--seed-mode", choices=["fold", "shared"], default="fold",
                        help="How probe seeds are chosen (default: fold). fold uses the fold "
                             "number as the seed. shared fits each of --seeds on every fold.")
    parser.add_argument("--pooling", choices=["mean", "per_patch"], default="mean",
                        help="Which embeddings to probe (default: mean). It must match the "
                             "--pooling of `embed`.")
    parser.add_argument("--seed", type=int, default=42, metavar="N",
                        help="Random seed (default: 42).")
    parser.add_argument("--output-root", default=None, metavar="DIR",
                        help="Where to write results.json and the fold probes (default: the "
                             "output_root setting). Each dataset and task gets its own "
                             "folder.")
    parser.add_argument("--cache-root", default=None, metavar="DIR",
                        help="Where to read the embeddings (default: the cache_root setting).")
    parser.add_argument("--reprobe", action="store_true",
                        help="Fit every fold again. Without it, a fold with saved predictions "
                             "from the same settings, weights and embeddings is only "
                             "rescored.")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract the embeddings, as `neuroatlas embed` does.")
    parser.add_argument("--embed-chunk", default=None, metavar="K/N",
                        help="With --extract-only, extract only chunk K of N of the subjects, "
                             "as in 0/4.")
    parser.add_argument("--no-recording-norm", action="store_true",
                        help="Probe the embeddings made with `embed --no-recording-norm`.")
    parser.add_argument("--no-amplitude-scale", action="store_true",
                        help="Probe the embeddings made with `embed --no-amplitude-scale`.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved settings as JSON and exit.")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List the tasks and exit.")
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
        raise usage_error(message, "neuroatlas list datasets --all (names every one)")


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
        # n_folds stays as given ("loso" too): the dataset config then equals
        # the one `run` hands `embed` for the same folds. The BCI readers
        # record what "loso" counts to (one fold per subject) in the result's
        # metadata, so a --debug run of one fold reads "LOSO 1/9", not "1/1".
        dataset_config["folds"] = _parse_int_list(args.folds)
    elif "folds" not in dataset_config:
        n_folds = dataset_config.get("n_folds") or dataset_config.get("num_folds")
        if spec.supports_folds and n_folds:
            # "loso" is one fold per subject, so the count comes from the
            # cohort rather than from the flag; n_folds stays "loso", which
            # the BCI readers split by (dataio/bci.loso_split)
            if isinstance(n_folds, str) and n_folds.strip().lower() == "loso":
                from neuroatlas.extensions.datasets.dataio.moabb_loader import loso_fold_count

                n_folds = loso_fold_count(spec.slug)
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
            # only when asked: the default leaves the resolved config as it was
            **({"reprobe": True} if getattr(args, "reprobe", False) else {}),
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

        print("tasks:")
        for spec in task_specs():
            print(f"  {spec.slug:<28} {spec.description}")
        presets = available_tasks()
        print("\npresets (a registered task with its settings):" if presets
              else "\nno presets")
        for name in presets:
            preset = json.loads((TASKS_DIR / f"{name}.json").read_text())
            print(f"  {name:<28} task={preset['task'].get('name')}")
        return

    if bool(args.config) == bool(args.dataset):
        parser.error("give exactly one of --config or --dataset")

    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if args.reprobe:
            config.setdefault("benchmark", {})["reprobe"] = True
    else:
        config = build_config(args)

    # A probe flag the task would not apply is refused before anything runs
    # (a dry run included): the run would otherwise look as asked and not be.
    _refuse_ignored_probe_flags(config, args, argv)

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
    from neuroatlas.benchmarking_helpers.probes import probe as probe_fit

    with embedding_pooling(args.pooling), threads, \
            probe_fit.max_iter_changed(_changed_max_iter(args, argv)):
        results = runner.run()
        note = probe_fit.user_cap_note()
    if note:
        from neuroatlas.cli import _msg

        _msg.note(note)

    from neuroatlas.entrypoints._common import report_results

    extract = bool(config["benchmark"].get("extract_only"))
    if report_results("probe", results,
                      f"embeddings in {runner.cache_root}" if extract
                      else f"results in {runner.output_root}", extraction=extract):
        sys.exit(1)



def _changed_max_iter(args, argv: List[str]) -> Optional[int]:
    """The benchmark protocol's iteration cap when the command line sets
    another (``--max-iter``), else None: the cap in use is the protocol's.
    The protocol's cap is the ``--max-iter`` the benchmark's probe line
    passes for this dataset (BCI: 1000), or the probe command's default
    where it passes none (seizure detection keeps its own, whatever this
    says)."""
    if args.config or not args.dataset or not _given(argv, "--max-iter"):
        return None
    caps = set()
    try:
        from neuroatlas import catalog

        for name in catalog.benchmarks_using(args.dataset):
            bench = catalog.load(name)
            if bench.derived_from:
                continue
            entry = next(e for e in bench.datasets if e.slug == args.dataset)
            for variant in bench.variant_names():
                try:
                    line = bench.probe_args(entry, variant)
                except catalog.CatalogError:
                    continue
                caps.add(_flag_int(line, "--max-iter") or DEFAULT_MAX_ITER)
    except Exception:       # a catalog that does not load: no protocol to compare with
        return None
    caps = caps or {DEFAULT_MAX_ITER}
    return None if int(args.max_iter) in caps else min(caps)


def _flag_int(argv, flag: str) -> Optional[int]:
    """The integer after *flag* in *argv* (``--flag N`` or ``--flag=N``)."""
    argv = list(argv)
    for i, a in enumerate(argv):
        try:
            if a == flag and i + 1 < len(argv):
                return int(argv[i + 1])
            if a.startswith(flag + "="):
                return int(a.split("=", 1)[1])
        except ValueError:
            return None
    return None


#: The probe flags each task applies, of --class-weight (``class_weight``)
#: and --tune-c (``c_values``); a task not listed applies both, as far as
#: this check knows (seizure_detection checks its own: probe_settings).
#: Each line says why the others would be ignored.
_TASK_APPLIES: Dict[str, tuple] = {
    "patient_classification": ((), "it fits an unweighted logistic regression at C = 1 on each "
                                   "subject's mean embedding (the published diagnosis probe)"),
    "brain_age": ((), "it fits a ridge regression of age and picks its alpha by nested "
                      "cross-validation"),
    "native_head_eval": ((), "it scores the checkpoint's own classifier head; no probe is fitted"),
    "lstm_probe": (("class_weight",), "it trains a neural probe, which has no C"),
    "attention_probe": (("class_weight",), "it trains a neural probe, which has no C"),
    "attention_probe_patient": (("class_weight",), "it trains a neural probe, which has no C"),
}
#: Tasks whose probe is train_probe: --probe-type nonlinear there is an MLP,
#: which takes neither a class weight nor a C.
_TRAIN_PROBE_TASKS = ("linear_probe", "arousal_detection", "respiratory_event_detection",
                      "patient_classification")
_FLAGS = {"class_weight": "--class-weight", "c_values": "--tune-c"}


def _given(argv: List[str], flag: str) -> bool:
    """Whether *flag* is on the command line (``--flag v`` or ``--flag=v``)."""
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _without(argv: List[str], flags) -> List[str]:
    """*argv* less *flags* and their values: the command that runs."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in flags:
            skip = True
            continue
        if any(a.startswith(f + "=") for f in flags):
            continue
        out.append(a)
    return out


def _refuse_ignored_probe_flags(config, args, argv: List[str]) -> None:
    """Refuse (exit 2, before anything runs) a probe flag the run would not
    apply: one the task preset fixes to another value (sleep_staging fixes
    the C grid, the selection metric and an unweighted loss), or one the task
    does not use at all (--class-weight / --tune-c on diagnosis or brain age,
    --tune-c on a neural probe, either on an MLP probe)."""
    from neuroatlas.benchmarking_helpers.registry.discovery import load_dataset_spec

    task = (config.get("task") or {}).get("name")
    if not task:
        tasks = {load_dataset_spec(slug).default_task for slug in config.get("datasets", {})}
        task = next(iter(tasks)) if len(tasks) == 1 else None
    problems, drop = [], []
    probe = config.get("probe") or {}
    if args.config:
        # a JSON config: its probe block is what the user set
        given = {k for k in _FLAGS if probe.get(k) not in (None, [], "")}
        preset_name, fixed = None, {}
    else:
        given = {k for k, flag in _FLAGS.items() if _given(argv, flag)}
        preset_name = args.task or load_dataset_spec(args.dataset).default_task
        fixed = load_task_preset(preset_name).get("probe") or {}
        cli = {"type": ("--probe-type", args.probe_type),
               "class_weight": ("--class-weight", args.class_weight),
               "c_values": ("--tune-c", [float(c) for c in _parse_csv(args.tune_c)]
                            if args.tune_c else None),
               "selection_metric": ("--selection-metric", args.selection_metric),
               "max_iter": ("--max-iter", args.max_iter)}
        clash = [(flag, key) for key, (flag, value) in cli.items()
                 if key in fixed and _given(argv, flag) and fixed[key] != value]
        if clash:
            what = "; ".join(f"{_SETTING[key]} {_setting_text(key, fixed[key])}"
                             for _, key in clash)
            problems.append(f"the {preset_name} task fixes {what}, so "
                            f"{' and '.join(flag for flag, _ in clash)} would be ignored")
            drop += [flag for flag, _ in clash]
            given -= {key for _, key in clash}
    applies, why = _TASK_APPLIES.get(task, (tuple(_FLAGS), ""))
    if task in _TRAIN_PROBE_TASKS and str(probe.get("type") or "linear") == "nonlinear":
        applies, why = (), "--probe-type nonlinear fits an MLP, which takes no class weight or C"
    ignored = [k for k in _FLAGS if k in given and k not in applies]
    if ignored:
        flags = [_FLAGS[k] for k in ignored]
        problems.append(f"the {preset_name or task} task does not apply "
                        f"{' or '.join(flags)}: {why}")
        drop += flags
    if problems:
        from neuroatlas.cli import _msg

        import shlex

        _msg.error("; ".join(problems),
                   "neuroatlas probe " + shlex.join(_without(argv, set(drop))))
        raise SystemExit(2)


#: A task preset's probe setting, in words (_refuse_ignored_probe_flags).
_SETTING = {"type": "the probe type", "class_weight": "the class weight",
            "c_values": "the C grid", "selection_metric": "the selection metric",
            "max_iter": "the iteration cap"}


def _setting_text(key: str, value) -> str:
    if key == "class_weight" and value is None:
        return "(none: an unweighted loss)"
    if isinstance(value, list):
        return "(" + ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in value) + ")"
    return f"({value})"


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
