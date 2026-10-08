"""``neuroatlas run <benchmark> -m MODELS [--dataset single|full|SLUGS]`` --
extract embeddings where missing, fit the probes, write the results."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional

from neuroatlas.cli import MODELS_HELP, Parser, UsageError
from neuroatlas.cli._table import add_format_arg, render

FIXED_SPLIT = "fixed split"

#: The dry-run table's headers (the JSON keys stay as they are).
PLAN_LABELS = {"n_models": "checkpoints", "n_runs": "runs", "n_not_applicable": "ruled out",
               "n_invalid": "invalid"}


def build_parser() -> Parser:
    p = Parser(
        prog="neuroatlas run",
        description="Run a benchmark on this machine. For each dataset it extracts the "
                    "embeddings that are not cached yet, fits the probes on every fold and "
                    "writes the results to `<output root>/<benchmark>/<dataset>/`.")
    from neuroatlas.cli.check import BENCHMARK_HELP, DATASET_HELP, VARIANT_HELP

    p.add_argument("benchmark", help=BENCHMARK_HELP)
    p.add_argument("-m", "--models", required=True, help=MODELS_HELP)
    p.add_argument("--dataset", default="single", metavar="single|full|NAMES",
                   help=f"{DATASET_HELP} (default: single).")
    p.add_argument("--variant", default="default", help=VARIANT_HELP)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the datasets, checkpoints, folds and runs, and whether the "
                        "data is here, then stop.")
    p.add_argument("--debug", action="store_true",
                   help="Run fold 0 only, as a quick test. Most datasets still extract the "
                        "embeddings of every fold, which a later full run reuses. HMC, MESA, "
                        "STAGES, HomePAP, the four bci_cognitive datasets, TUAB and "
                        "CHB-MIT when read from an HDF5 file, and runs with a stride other "
                        "than the window extract fold 0 only.")
    p.add_argument("--folds", default=None, metavar="LIST",
                   help="Only run these folds, as in 0,1 or 0-4 (default: all).")
    p.add_argument("--limit-batches", type=int, default=None, metavar="N",
                   help="Stop each extraction after N batches, for a smoke test. Its "
                        "embeddings and results go to `_limited` folders under the cache "
                        "and output roots. A fold may end up with no test subjects.")
    p.add_argument("--skip-embed", action="store_true",
                   help="Only fit the probes. Fail where embeddings are missing.")
    p.add_argument("--reprobe", action="store_true",
                   help="Fit every fold again. Without it, a fold with saved predictions "
                        "from the same settings, weights and embeddings is only rescored.")
    p.add_argument("--num-workers", type=int, default=None, metavar="N",
                   help="Data loader workers for extraction (default: the CPUs this job may "
                        "use minus one, at most 16). Some datasets set their own.")
    p.add_argument("--cache-root", default=None, metavar="DIR",
                   help="Where to save and read embeddings (default: the cache_root "
                        "setting).")
    p.add_argument("--output-root", default=None, metavar="DIR",
                   help="Where to write results (default: the output_root setting).")
    p.add_argument("--checkpoint-override", action="append", default=[],
                   metavar="ID.checkpoint_path=PATH",
                   help="Load the weights of checkpoint ID from PATH, as in "
                        "biot_pretrained.checkpoint_path=/my/weights.ckpt. ID must be in the "
                        "-m selection. Needs its own --cache-root, because embeddings are "
                        "cached by checkpoint id.")
    p.add_argument("--per-model-output", action="store_true",
                   help="Write results to `<dataset>/<checkpoint>/`, as the jobs of `submit` "
                        "do.")
    p.add_argument("--allow-partial", action="store_true",
                   help="Run on a dataset that `data status` reports as partial or empty. "
                        "Without it, run refuses, because the results would cover only some "
                        "subjects.")
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
            example = f"{ident or (selected[0] if selected else 'ID')}.checkpoint_path=PATH"
            raise UsageError(f"--checkpoint-override wants ID.checkpoint_path=PATH, got {text!r}"
                             f"\nfix: {_override_fix(text, example)}")
        if key != "checkpoint_path":
            raise UsageError(f"--checkpoint-override {text}: only checkpoint_path can be "
                             f"changed, not {key!r}\n"
                             f"fix: {_override_fix(text, f'{ident}.checkpoint_path=PATH')}")
        spec = specs.get(ident) or families.get(ident)
        if spec is None:
            close = difflib.get_close_matches(ident, [*specs, *families], n=1)
            hint = f" (did you mean {close[0]}?)" if close else ""
            raise UsageError(f"--checkpoint-override {text}: no checkpoint or family "
                             f"{ident!r}{hint}\nfix: neuroatlas list models")
        hits = [m for m in selected if m == ident or specs[m].model_family == ident]
        if not hits:
            from neuroatlas.cli import command_with_value, command_without

            models = _selection()
            add = command_with_value(("-m", "--models"), f"{models},{ident}") if models else None
            _msg.warning(f"--checkpoint-override {text} changes nothing: {ident} is not "
                         f"among the selected checkpoints ({', '.join(selected)})",
                         "\n".join(f for f in (add, command_without("--checkpoint-override"))
                                   if f) or None)
        path = Path(value).expanduser()
        if not path.exists():
            if not (spec.source_type == "huggingface" and _HUB_ID.match(value)):
                raise UsageError(f"--checkpoint-override {text}: {path} does not exist")
        else:
            value = str(path.absolute())
        out += ["--checkpoint", f"{ident}={value}"]
    return out


def _selection() -> Optional[str]:
    """The -m value of the command being run, as typed."""
    from neuroatlas.cli import _COMMAND_LINE

    words = list(_COMMAND_LINE)
    for i, word in enumerate(words):
        if word in ("-m", "--models") and i + 1 < len(words):
            return words[i + 1]
        if word.startswith(("-m=", "--models=")):
            return word.split("=", 1)[1]
    return None


def _override_fix(given: str, example: str) -> str:
    """The command being run with this --checkpoint-override value in place of
    *given*; the flag alone when the command is not known."""
    from neuroatlas.cli import corrected_command

    return (corrected_command({given: example})
            or corrected_command({f"--checkpoint-override={given}":
                                  f"--checkpoint-override={example}"})
            or f"--checkpoint-override {example}")


def fold_label(p, restricted: bool) -> str:
    """How a plan's folds read in the table: a short list, ``LOSO (87)`` for
    leave-one-subject-out, ``N folds`` for a long list, ``fixed split`` for a
    cohort that ships one train/val/test split."""
    folds = list(p.folds)
    if any(str(f).startswith("?") for f in folds):
        # not counted here (LOSO needs moabb to count the subjects)
        why = str(folds[0])[1:].strip().strip("()") or "not counted"
        loso = any(a.replace(" ", "").lower() == "n_folds=loso" for a in p.probe_argv)
        return f"LOSO ({why})" if loso else why
    if folds in ([], ["-"]) or (folds == [0] and not restricted):
        return FIXED_SPLIT
    if not restricted and any(a.replace(" ", "").lower() == "n_folds=loso" for a in p.probe_argv):
        return f"LOSO ({len(folds)})"
    if len(folds) > 8:
        return f"{len(folds)} folds ({folds[0]}-{folds[-1]})"
    return ", ".join(map(str, folds))


def _task(p) -> str:
    """The task a plan runs, as `list benchmarks` names it: the benchmark's
    (or the dataset's own) preset, ``hypnogram``, or on BCI ``trial
    classification`` (a linear probe on each dataset's own classes)."""
    if p.task != "default":
        return p.task
    from neuroatlas import catalog
    from neuroatlas.cli.listing import task_label

    try:
        return task_label(catalog.load(p.benchmark), p.dataset)
    except catalog.CatalogError:
        return "default"


def _command(verb: str, argv: List[str], models: List[str], extra: List[str]) -> str:
    return " ".join(["neuroatlas", verb, *argv, "--models", ",".join(models), *extra])


def plan_row(p, restricted: bool, overrides: List[str] = (), reprobe: bool = False, *,
             cache_root=None, limit_batches: Optional[int] = None,
             skip_embed: bool = False) -> dict:
    """One plan as `run --dry-run --format json` prints it (without its note);
    ``api.plan`` builds its table from the same rows. Its commands are the
    ones `run` executes, with the same folders: under --limit-batches the
    embed step stops after N batches into <cache root>/_limited, which the
    probe step reads (``runmod.limit`` has already moved ``p.output``)."""
    runs = bool(p.embed_argv and p.models)
    given = ["--cache-root", str(cache_root)] if cache_root else []
    embed_extra = [*given, *(["--limit-batches", str(limit_batches)]
                             if limit_batches is not None else []), *overrides]
    if limit_batches is not None:
        from neuroatlas import run as runmod

        given = ["--cache-root", str(runmod.limit([], cache_root))]
    probe_extra = [*given, "--output-root", str(p.output), *overrides,
                   *(["--reprobe"] if reprobe else [])]
    return {
        "benchmark": p.benchmark, "dataset": p.dataset, "task": _task(p),
        "models": list(p.models), "n_models": len(p.models),
        "folds": fold_label(p, restricted), "fold_ids": [f for f in p.folds],
        "seeds": p.seeds, "n_runs": p.n_runs, "data": p.data,
        "not_applicable": list(p.skipped), "n_not_applicable": len(p.skipped),
        "invalid": list(p.invalid), "n_invalid": len(p.invalid),
        "output": str(p.output),
        "embed_command": (_command("embed", p.embed_argv, p.models, embed_extra)
                          if runs and not skip_embed else None),
        "probe_command": _command("probe", p.probe_argv, p.models, probe_extra) if runs else None,
    }


def plan_notes(p, refused_partial: bool) -> List[str]:
    """The lines a plan's row carries (its `note` in machine formats). The
    plan's own notes come first: a map that fails to load, the pairs the map
    has no entry for (invalid, not run), data that is not here."""
    from neuroatlas.cli import _msg

    lines = list(p.notes)
    if refused_partial:
        lines += _msg.lines("warning", "a half-finished download: run refuses it (add "
                                       "--allow-partial to run on what is there)",
                            f"neuroatlas data download {p.dataset}")
    if p.skipped:
        why = ruled_out_reason(p.dataset, p.skipped)
        lines += _msg.lines(_msg.RULED_OUT, f"{name_ids(p.skipped)}, by the {p.dataset} "
                                            f"channel map" + (f" ({why})" if why else ""))
    return lines


def ruled_out_reason(dataset: str, ids: List[str]) -> Optional[str]:
    """Why the channel map rules these checkpoints out (its note's first
    sentence), when it gives one reason for all of them."""
    from neuroatlas.benchmarking_helpers.channels.channel_map import load_channel_map
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry
    from neuroatlas.cli import _msg

    try:
        cmap = load_channel_map(dataset)
    except (ValueError, KeyError):
        return None
    if cmap is None:
        return None
    family = {s.identifier: s.model_family for s in checkpoint_registry()}
    reasons = {_msg.first_sentence(cmap.state_for(family.get(i, i))[1]) for i in ids}
    reasons.discard("")
    if len(reasons) != 1:
        return None
    reason = reasons.pop().rstrip(".")
    return reason[:1].lower() + reason[1:]


def _family_of(ids: List[str]) -> List[str]:
    """Each checkpoint id's model family, in order (one per id)."""
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    family = {s.identifier: s.model_family for s in checkpoint_registry()}
    return [family.get(i, i) for i in ids]


def _hypnogram_lines(p) -> List[str]:
    """What `run` executes for a hypnogram plan: one `neuroatlas hypnogram
    reconstruct` per folder of sleep_stage results it reads (the folder it
    expects when there are none yet)."""
    from neuroatlas import run as runmod

    base = p.output_base if p.output_base is not None else p.output.parents[1]
    sources = runmod.staging_results(p.dataset, base) or [base / "sleep_stage" / p.dataset]
    hypno = p.output / "hypnograms.json"
    return [f"neuroatlas hypnogram reconstruct --results-dir {src} --output {hypno}"
            for src in sources]


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

    from neuroatlas.cli import command_with

    args = build_parser().parse_args(argv)
    if args.checkpoint_override and not args.cache_root:
        raise UsageError("--checkpoint-override needs --cache-root: saved embeddings are filed "
                         "under the checkpoint id, not its weights, so other weights need a "
                         "cache folder of their own\n"
                         f"fix: {command_with('--cache-root', 'DIR') or '--cache-root DIR'} "
                         f"(a folder for this run's embeddings)")
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
        partial = command_with("--allow-partial")
        raise UsageError(
            "half-finished download: "
            + "; ".join(f"{p.dataset} is {p.data}" for p in incomplete)
            + f"\nfix: neuroatlas data download {names} (finishes it)"
            + (f"\nfix: {partial} (runs on what is there)" if partial else ""))
    restricted = bool(args.debug or args.folds)
    rows: List[dict] = []
    notes = {}
    if args.limit_batches is not None:
        # the folders a --limit-batches run uses, shown before it runs
        runmod.limit(plans, Path(args.cache_root) if args.cache_root else None)
    for i, p in enumerate(plans):
        rows.append(plan_row(p, restricted, overrides, reprobe=args.reprobe,
                             cache_root=args.cache_root, limit_batches=args.limit_batches,
                             skip_embed=args.skip_embed))
        lines = plan_notes(p, p in incomplete and not args.allow_partial)
        if lines:
            notes[i] = lines
    if args.dry_run or args.format != "table":
        # `invalid` (pairs the channel map has no entry for) is a column only
        # when there are some; it is always in the machine formats. `seeds`
        # (always one per fold) only there.
        any_invalid = any(r["n_invalid"] for r in rows)
        human = args.format in ("table", "md")
        render(rows, ["benchmark", "dataset", "task", "n_models", "folds",
                      *([] if human else ["seeds"]), "n_runs", "data", "n_not_applicable",
                      *(["n_invalid"] if any_invalid else [])],
               args.format, notes,
               extra=["models", "fold_ids", "not_applicable", "n_invalid", "invalid", "output",
                      "seeds", "embed_command", "probe_command"],
               labels=PLAN_LABELS)
    if left_out and args.format == "table":
        _msg.note(left_out)
    if args.dry_run:
        if args.format == "table":
            explained = [("runs", "one probe fit per checkpoint and fold"
                          if any(p.task != "hypnogram" for p in plans) else
                          "one per checkpoint: its hypnograms over every fold")]
            if any(r["folds"] == FIXED_SPLIT for r in rows):
                explained.append((FIXED_SPLIT, "one train/validation/test split: one run per "
                                               "checkpoint, no spread over folds"))
            if any(r["folds"].startswith("LOSO") for r in rows):
                explained.append(("LOSO (N)", "leave-one-subject-out: N folds, each testing "
                                              "one held-out subject"))
            if any(r["n_not_applicable"] for r in rows):
                explained.append((_msg.RULED_OUT, "checkpoints the dataset's channel map "
                                                  "excludes: never run there"))
            if any_invalid:
                explained.append(("invalid", "checkpoints the channel map has no entry for: "
                                             "not run"))
            print("\n" + "\n".join(_msg.legend(explained)))
            print("\nthe commands it runs:")
            if args.skip_embed and any(r["probe_command"] for r in rows):
                print("  (no embedding step: --skip-embed; the probe reads the embeddings "
                      "already in the cache)")
            for p, r in zip(plans, rows):
                if r["probe_command"]:
                    if r["embed_command"]:
                        print(f"  {r['embed_command']}")
                    print(f"  {r['probe_command']}")
                elif not p.embed_argv:
                    for line in _hypnogram_lines(p):
                        print(f"  {line}")
        return

    summary = runmod.execute(
        plans, cache_root=Path(args.cache_root) if args.cache_root else None,
        limit_batches=args.limit_batches, skip_embed=args.skip_embed,
        extra_embed=overrides, extra_probe=overrides, num_workers=args.num_workers,
        reprobe=args.reprobe)
    kept = summary.get("kept", 0)
    invalid = summary.get("invalid", 0)
    skipped = summary.get("skipped", 0)
    total = summary["ok"] + summary["failed"] + skipped
    # runs are (checkpoint, fold) results; what the channel map rules out or
    # has no entry for never ran, and is counted in (dataset, checkpoint) pairs
    never = [f"{summary['n/a']} ruled out by the channel map" if summary["n/a"] else "",
             f"{invalid} invalid (no channel map entry)" if invalid else ""]
    never = [n for n in never if n]
    print("\n" + _msg.counts(total, "run (checkpoint x fold)" if total == 1
                             else "runs (checkpoint x fold)",
                             [("ok", summary["ok"]), ("failed", summary["failed"]),
                              (_msg.SKIPPED, skipped)],
                             keep_zero=("ok", "failed"))
          + (" (data or weights not here)" if skipped else "")
          + (f"; pairs (dataset x checkpoint) never run: {', '.join(never)}" if never else "")
          + (f"; the results folder also holds {_msg.plural(kept, 'result')} of earlier runs"
             if kept else ""),
          flush=True)
    if args.limit_batches is not None and summary["failed"]:
        # a smoke test's truncated embeddings can miss a fold's subjects
        from neuroatlas.cli import command_without

        n = args.limit_batches
        _msg.note(f"--limit-batches {n} embeds only the first {_msg.plural(n, 'batch', 'batches')} "
                  f"of each dataset, so a fold whose subjects come later has no embeddings "
                  f"to fit or test on; the same run without it embeds them all",
                  command_without("--limit-batches"))
    if summary.get("reused"):
        # folds whose saved predictions matched: rescored, not fitted
        n = summary["reused"]
        _msg.note(f"{_msg.plural(n, 'fold')} scored from {'its' if n == 1 else 'their'} saved "
                  f"predictions (same settings, weights and embeddings); add --reprobe to fit "
                  f"{'it' if n == 1 else 'them'} again")
    if invalid:
        # one line per dataset, as the dry run says it; the command without
        # every family concerned after the last
        with_invalid = [p for p in plans if p.invalid]
        families = sorted({f for p in with_invalid for f in _family_of(p.invalid)})
        fix = runmod.invalid_fix(args.models, families, runmod._run_line(
            args.benchmark, args.models, args.dataset, args.variant,
            {f: f for f in families}))
        for k, p in enumerate(with_invalid):
            ids = dict(zip(p.invalid, _family_of(p.invalid)))
            _msg.warning(f"invalid, not run: {runmod.ids_text(ids)} (the {p.dataset} channel "
                         f"map has no entry for {runmod.family_text(sorted(set(ids.values())))})",
                         fix if k == len(with_invalid) - 1 else None)
    if summary["datasets_failed"]:
        failed = summary["datasets_failed"]
        _msg.error(f"failed before any result: {', '.join(failed)} (the messages above say why)",
                   f"neuroatlas check {args.benchmark} -m {args.models} --dataset "
                   f"{','.join(dict.fromkeys(failed))} (one batch per pair, to find the cause)")
    wrote = summary["ok"] + summary["failed"] > 0
    if plans and wrote:
        from neuroatlas import _paths

        # where execute() put the results: --limit-batches moves them under
        # <output root>/_limited (output_base follows), so the line points there
        base = plans[0].output_base or output_root or _paths.output_dir()
        root = Path(base) / plans[0].benchmark
        moved = args.limit_batches is not None and plans[0].task != "hypnogram"
        again = f"neuroatlas results {args.benchmark}" + (
            f" --output-root {base}" if output_root or moved else "") + (
            f" --variant {args.variant}" if args.variant != "default" else "")
        print(f"results: {root}; `{again}` summarises them")
    if summary["failed"] or skipped:
        raise SystemExit(1)
