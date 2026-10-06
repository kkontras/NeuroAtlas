"""``neuroatlas run <benchmark> -m MODELS [--dataset single|full|SLUGS]`` --
extract embeddings where missing, fit the probes, write the results."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional

from neuroatlas.cli import MODELS_HELP, Parser, UsageError
from neuroatlas.cli._table import add_format_arg, render

FIXED_SPLIT = "fixed split"


def build_parser() -> Parser:
    p = Parser(
        prog="neuroatlas run",
        description="Run a benchmark on this machine: for each dataset, `embed` (a no-op "
                    "where the cache already has it), then `probe` over its folds. "
                    "Results go to <output root>/<benchmark>/<dataset>/.")
    p.add_argument("benchmark")
    p.add_argument("-m", "--models", required=True, help=MODELS_HELP)
    p.add_argument("--dataset", default="single", metavar="single|full|SLUGS",
                   help="The one-dataset quick suite (default), every dataset, or a comma list.")
    p.add_argument("--variant", default="default", help="A benchmark variant (see `neuroatlas show`).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan -- datasets, models, folds, runs, data state -- and stop.")
    p.add_argument("--debug", action="store_true",
                   help="Fold 0 only. Most datasets keep one embedding cache for every fold: "
                        "it is extracted in full, and a later full run reuses it. Those that "
                        "cache per fold and split -- HMC, MESA, STAGES and HomePAP (the "
                        "PhysioEx-format sleep cohorts), SHHS, the four bci_cognitive cohorts, "
                        "TUAB's H5 fast path, CHB-MIT's HDF5 backend, any run with a stride "
                        "other than the window -- extract fold 0 only, and the full run "
                        "extracts the other folds.")
    p.add_argument("--folds", default=None, help="Comma list of folds, instead of all.")
    p.add_argument("--limit-batches", type=int, default=None, metavar="N",
                   help="Stop each extraction after N batches: a smoke test. Uses its own "
                        "cache, <cache root>/_limited, and writes its results under "
                        "<output root>/_limited; too few subjects may leave a fold with "
                        "nothing to test on.")
    p.add_argument("--skip-embed", action="store_true",
                   help="Probe only; fail where embeddings are missing.")
    p.add_argument("--reprobe", action="store_true",
                   help="Fit every fold again. Without it, a fold whose saved predictions "
                        "(predictions.npz in its probe folder) were made with the same "
                        "settings, weights and embeddings is not fitted again: its metrics "
                        "are recomputed from them.")
    p.add_argument("--num-workers", type=int, default=None, metavar="N",
                   help="Data-loader workers for extraction. Default: the CPUs this job may "
                        "use (CPU affinity and cgroup quota) minus one, at most 16, unless "
                        "the dataset pins its own.")
    p.add_argument("--cache-root", default=None, help="Embedding cache for this run.")
    p.add_argument("--output-root", default=None, help="Results root for this run.")
    p.add_argument("--checkpoint-override", action="append", default=[], metavar="ID.KEY=VALUE",
                   help="Change a checkpoint field for this run, e.g. "
                        "biot_pretrained.checkpoint_path=/my/weights.ckpt (only checkpoint_path "
                        "for now; the file must exist). ID is a checkpoint id or family among "
                        "the -m selection. Needs --cache-root: the cache is keyed by checkpoint "
                        "id, not by weights.")
    p.add_argument("--per-model-output", action="store_true",
                   help="Results in <dataset>/<model>/ (what `submit`'s jobs use).")
    p.add_argument("--allow-partial", action="store_true",
                   help="Run on a dataset that `data status` calls partial or empty (a "
                        "half-finished download). Without it, run refuses: the results would "
                        "silently cover only the subjects that arrived.")
    add_format_arg(p)
    return p


INCOMPLETE = ("partial", "empty")


def _incomplete(plans) -> List[Any]:
    """Plans whose data is a half-finished download (`data status`: partial, empty)."""
    return [p for p in plans if p.models and str(p.data).startswith(INCOMPLETE)]


_HUB_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


def _overrides(pairs: List[str], selected: List[str]) -> List[str]:
    """ID.KEY=VALUE -> the verbs' ``--checkpoint ID=PATH``, every value
    checked before any work starts: a bad one is a usage error (exit 2)."""
    import difflib

    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry
    from neuroatlas.cli import _msg

    specs = {s.identifier: s for s in checkpoint_registry()}
    families = {s.model_family: s for s in specs.values()}
    out = []
    for text in pairs:
        target, sep, value = text.partition("=")
        ident, dot, key = target.partition(".")
        if not (sep and dot and ident and key and value):
            raise UsageError(f"--checkpoint-override wants ID.KEY=VALUE, got {text!r}\n"
                             f"fix: --checkpoint-override ID.checkpoint_path=/my/weights")
        if key != "checkpoint_path":
            raise UsageError(f"--checkpoint-override {text}: only checkpoint_path can be "
                             f"overridden, not {key!r}")
        spec = specs.get(ident) or families.get(ident)
        if spec is None:
            close = difflib.get_close_matches(ident, [*specs, *families], n=1)
            hint = f" (did you mean {close[0]}?)" if close else ""
            raise UsageError(f"--checkpoint-override {text}: no checkpoint or family "
                             f"{ident!r}{hint}")
        hits = [m for m in selected if m == ident or specs[m].model_family == ident]
        if not hits:
            _msg.warning(f"--checkpoint-override {text} changes nothing: {ident} is not "
                         f"among the selected models ({', '.join(selected)})")
        path = Path(value).expanduser()
        if not path.exists():
            if not (spec.source_type == "huggingface" and _HUB_ID.match(value)):
                raise UsageError(f"--checkpoint-override {text}: {path} does not exist")
        else:
            value = str(path.absolute())
        out += ["--checkpoint", f"{ident}={value}"]
    return out


def fold_label(p, restricted: bool) -> str:
    """How a plan's folds read in the table: a short list, ``LOSO (87)`` for
    leave-one-subject-out, ``N folds`` for a long list, ``fixed split`` for a
    cohort that ships one train/val/test split."""
    folds = list(p.folds)
    if any(str(f).startswith("?") for f in folds):
        return ", ".join(map(str, folds))
    if folds in ([], ["-"]) or (folds == [0] and not restricted):
        return FIXED_SPLIT
    if not restricted and any(a.replace(" ", "").lower() == "n_folds=loso" for a in p.probe_argv):
        return f"LOSO ({len(folds)})"
    if len(folds) > 8:
        return f"{len(folds)} folds ({folds[0]}-{folds[-1]})"
    return ", ".join(map(str, folds))


def _task(p) -> str:
    """The task a plan runs: the benchmark's, or the dataset's own default."""
    if p.task != "default":
        return p.task
    from neuroatlas.benchmarking_helpers.registry.discovery import dataset_specs

    spec = next((s for s in dataset_specs() if s.slug == p.dataset), None)
    return getattr(spec, "default_task", None) or "default"


def _command(verb: str, argv: List[str], models: List[str], extra: List[str]) -> str:
    return " ".join(["neuroatlas", verb, *argv, "--models", ",".join(models), *extra])


def plan_row(p, restricted: bool, overrides: List[str] = (), reprobe: bool = False) -> dict:
    """One plan as `run --dry-run --format json` prints it (without its note);
    ``api.plan`` builds its table from the same rows."""
    runs = bool(p.embed_argv and p.models)
    probe_extra = ["--output-root", str(p.output), *overrides, *(["--reprobe"] if reprobe else [])]
    return {
        "benchmark": p.benchmark, "dataset": p.dataset, "task": _task(p),
        "models": list(p.models), "n_models": len(p.models),
        "folds": fold_label(p, restricted), "fold_ids": [f for f in p.folds],
        "seeds": p.seeds, "n_runs": p.n_runs, "data": p.data,
        "not_applicable": list(p.skipped), "n_not_applicable": len(p.skipped),
        "invalid": list(p.invalid), "n_invalid": len(p.invalid),
        "output": str(p.output),
        "embed_command": _command("embed", p.embed_argv, p.models, list(overrides)) if runs else None,
        "probe_command": _command("probe", p.probe_argv, p.models, probe_extra) if runs else None,
    }


def plan_notes(p, refused_partial: bool) -> List[str]:
    """The lines a plan's row carries (its `note` in machine formats). The
    plan's own notes come first: a map that fails to load, the pairs the map
    has no entry for (invalid, not run), data that is not here."""
    from neuroatlas.cli import _msg

    lines = list(p.notes)
    if refused_partial:
        lines += _msg.lines("warning", "a half-finished download: run refuses it",
                            f"neuroatlas data download {p.dataset}, or add --allow-partial")
    if p.skipped:
        lines += _msg.lines("n/a", f"{name_ids(p.skipped)} (the channel map skips them)")
    return lines


def name_ids(ids: List[str]) -> str:
    """Checkpoint ids for a message: themselves when few, else by family
    (``eegnetv4 (12 checkpoints)``); JSON keeps every id."""
    if len(ids) <= 3:
        return ", ".join(ids)
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    family = {s.identifier: s.model_family for s in checkpoint_registry()}
    groups: dict = {}
    for i in ids:
        groups.setdefault(family.get(i, i), []).append(i)
    return ", ".join(f"{f} ({len(members)} checkpoints)" if len(members) > 1 else members[0]
                     for f, members in groups.items())


def main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas import catalog, run as runmod
    from neuroatlas.cli import _msg

    args = build_parser().parse_args(argv)
    if args.checkpoint_override and not args.cache_root:
        raise UsageError("--checkpoint-override needs --cache-root (the embedding cache is "
                         "keyed by checkpoint id, not by its weights)\n"
                         "fix: add --cache-root DIR, a folder for this run's embeddings")
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else None
    bench = catalog.load(args.benchmark)
    selected = bench.select_models(args.models)
    left_out = bench.left_out_note(args.models)
    overrides = _overrides(args.checkpoint_override, selected)
    plans = runmod.plan(args.benchmark, args.models, args.dataset, args.variant,
                        debug=args.debug, folds=args.folds, per_model=args.per_model_output,
                        output_root=output_root)

    incomplete = _incomplete(plans)
    if incomplete and not args.allow_partial and not args.dry_run:
        names = " ".join(p.dataset for p in incomplete)
        raise UsageError(
            "half-finished download: "
            + "; ".join(f"{p.dataset} is {p.data}" for p in incomplete)
            + f"\nfix: neuroatlas data download {names}, or add --allow-partial to run on "
              f"what is there")
    restricted = bool(args.debug or args.folds)
    rows: List[dict] = []
    notes = {}
    for i, p in enumerate(plans):
        rows.append(plan_row(p, restricted, overrides, reprobe=args.reprobe))
        lines = plan_notes(p, p in incomplete and not args.allow_partial)
        if lines:
            notes[i] = lines
    if args.dry_run or args.format != "table":
        # `invalid` (pairs the channel map has no entry for) is a column only
        # when there are some; it is always in the machine formats
        any_invalid = any(r["n_invalid"] for r in rows)
        render(rows, ["benchmark", "dataset", "task", "n_models", "folds", "seeds", "n_runs",
                      "data", "n_not_applicable", *(["n_invalid"] if any_invalid else [])],
               args.format, notes,
               extra=["models", "fold_ids", "not_applicable", "n_invalid", "invalid", "output",
                      "embed_command", "probe_command"],
               labels={"n_not_applicable": "n/a", "n_invalid": "invalid"})
    if left_out and args.format == "table":
        _msg.note(left_out)
    if args.dry_run:
        if args.format == "table":
            if any(r["folds"] == FIXED_SPLIT for r in rows):
                print(f"\n{FIXED_SPLIT}: one train/val/test split, so one run per model "
                      f"and no spread")
            if any(r["folds"].startswith("LOSO") for r in rows):
                print("LOSO (N): leave-one-subject-out, one fold per subject")
            print()
            for p, r in zip(plans, rows):
                if r["embed_command"]:
                    print(f"  {r['embed_command']}")
                    print(f"  {r['probe_command']}")
                elif not p.embed_argv:
                    print(f"  hypnograms from {p.output.parents[1] / 'sleep_stage' / p.dataset} "
                          f"-> {p.output}")
        return

    summary = runmod.execute(
        plans, cache_root=Path(args.cache_root) if args.cache_root else None,
        limit_batches=args.limit_batches, skip_embed=args.skip_embed,
        extra_embed=overrides, extra_probe=overrides, num_workers=args.num_workers,
        reprobe=args.reprobe)
    kept = summary.get("kept", 0)
    invalid = summary.get("invalid", 0)
    total = summary["ok"] + summary["failed"] + summary["n/a"] + invalid
    print("\n" + _msg.counts(total, "run" if total == 1 else "runs",
                             [("ok", summary["ok"]), ("failed", summary["failed"]),
                              ("n/a", summary["n/a"]), ("invalid", invalid)],
                             keep_zero=("ok", "failed", "n/a"))
          + (f"; results.json also keeps {_msg.plural(kept, 'earlier result')}" if kept else ""),
          flush=True)
    if summary.get("reused"):
        # folds whose saved predictions matched: rescored, not fitted
        _msg.note(f"reused saved predictions for {_msg.plural(summary['reused'], 'fold')}; "
                  f"--reprobe probes again")
    if invalid:
        pairs = [f"{p.dataset}/{m}" for p in plans for m in p.invalid]
        from neuroatlas.benchmarking_helpers.channels.channel_map import _default_yaml_path

        maps = sorted({_default_yaml_path(p.dataset).name for p in plans if p.invalid})
        _msg.warning(f"invalid, not run: {', '.join(pairs)} (the channel map has no entry "
                     f"for the model's family)",
                     f"add the family to configs/channel_maps/{', '.join(maps)}: a mapping, "
                     f"`mode: label_pass_through` or `skip`")
    if summary["datasets_failed"]:
        _msg.error(f"failed before any result: {', '.join(summary['datasets_failed'])}")
    if plans:
        from neuroatlas import _paths

        # where execute() put the results: --limit-batches moves them under
        # <output root>/_limited (output_base follows), so the line points there
        base = plans[0].output_base or output_root or _paths.output_dir()
        root = Path(base) / plans[0].benchmark
        moved = args.limit_batches is not None and plans[0].task != "hypnogram"
        again = f"neuroatlas results {args.benchmark}" + (
            f" --output-root {base}" if output_root or moved else "") + (
            f" --variant {args.variant}" if args.variant != "default" else "")
        print(f"results: {root}  (`{again}`)")
    if summary["failed"]:
        raise SystemExit(1)
