"""``neuroatlas list`` -- what exists: benchmarks, datasets, models, aliases, tasks.

Listing reads the registry and the shipped tables only. It never looks at
your data or weights; `data status` and `models status` do that.

``-v`` adds the detail a listing has beyond its table -- each benchmark's
datasets, each alias's members, where a dataset or a checkpoint comes from,
a task's full description; ``--format json`` always carries all of it.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from neuroatlas.cli import Parser, _msg
from neuroatlas.cli._table import NOT_APPLICABLE, add_format_arg, render


def _grep(rows: List[Dict[str, Any]], pattern: Optional[str], keys: List[str]) -> List[Dict[str, Any]]:
    if not pattern:
        return rows
    p = pattern.lower()
    return [r for r in rows if any(p in str(r.get(k, "")).lower() for k in keys)]


def first_sentence(text: Any) -> str:
    """The first sentence of a description, whole -- never cut mid-word."""
    flat = " ".join(str(text or "").split())
    match = re.search(r"(?<=[.!?])\s+(?=[A-Z(])", flat)
    return flat[:match.start()] if match else flat


def _done(args, rows: List[Dict[str, Any]], total: int, what: str, scope: str = "") -> bool:
    """The table footer's count; False (and a message instead of a bare
    header) when --grep matched nothing."""
    if args.format != "table":
        return True                     # [] in JSON, a header-only CSV: still the schema
    if not rows:
        print(f"no {what} match {args.grep!r}" + (f" ({total} {scope})" if scope else "")
              + f". `neuroatlas list {args.what}` shows them all.")
        return False
    return True


def _count(args, n: int, total: int, what: str, scope: str) -> str:
    if (n if not args.grep else total) == 1 and what.endswith("s"):
        what = what[:-1]
    if args.grep:
        return f"{n} of {total} {what} {scope} match {args.grep!r}"
    return f"{n} {what} {scope}".rstrip()


# -- benchmarks ------------------------------------------------------------------

#: The task a benchmark without a `task` of its own runs on every BCI dataset:
#: a linear probe on that dataset's own trial labels.
BCI_TASK = "trial classification"


def task_label(bench, slug: Optional[str] = None) -> str:
    """What a benchmark's probe predicts, as `list benchmarks` and `run
    --dry-run` print it: the task preset (``sleep_staging``), ``per dataset``
    when each dataset has its own, ``hypnogram`` for the benchmark built on
    another's predictions, ``trial classification`` on BCI (each dataset's
    own classes)."""
    entry = next((e for e in bench.datasets if e.slug == slug), None) if slug else None
    if entry is not None and entry.task:
        return entry.task
    if bench.task:
        return bench.task
    if bench.derived_from:
        return "hypnogram"
    if slug is None and any(e.task for e in bench.datasets):
        return "per dataset"
    return BCI_TASK if bench.domain == "bci" else "per dataset"


def headline_text(bench) -> str:
    """The headline as `list benchmarks` prints it: the metric's name, the
    event threshold its probe is read at, and the direction when lower is
    better (``AUPRC (> 3 s arousal per epoch)``, ``MAE in years (lower is
    better)``)."""
    m = bench.metrics
    text = bench.metric_info(m.headline).name
    seconds = m.at_seconds()
    if seconds is not None:
        event = (m.event_words() or "").replace("scored ", "")
        text += f" (> {seconds:g} s{' ' + event if event else ''} per epoch)"
    return text + ("" if m.higher_is_better else " (lower is better)")


def chance_text(bench) -> Any:
    """The headline's chance level where one is defined: a number, ``1/C``
    (balanced accuracy, per dataset) or ``prevalence`` (AUPRC), ``-`` when
    there is none (the YAML's ``metrics.chance``, else the metric registry's)."""
    value = bench.chance()
    if value is None:
        return NOT_APPLICABLE
    return f"{value:g}" if isinstance(value, float) else value


def _left_out_line(bench) -> Optional[str]:
    """``left out: core_sleep, sleep_transformer, sleepyco (why); eegnetv4
    (why)``: the checkpoint families a benchmark does not evaluate, each group
    with the benchmark file's reason, or None when it leaves none out."""
    groups = []
    for x in bench.excluded_models:
        reason = " ".join(str(x.reason or "").split()).rstrip(".")
        groups.append(", ".join(sorted(x.families)) + (f" ({reason})" if reason else ""))
    return f"left out: {'; '.join(groups)}" if groups else None


def list_benchmarks(args) -> None:
    from neuroatlas import catalog

    every = list(catalog.catalog().values())
    rows, notes = [], {}
    for bench in _grep_benchmarks(every, args.grep):
        m = bench.metrics
        rows.append({
            "benchmark": bench.name,
            "domain": bench.domain,
            "task": task_label(bench),
            "headline": headline_text(bench),
            "headline_key": m.headline,
            "chance": chance_text(bench),
            "single": bench.single or NOT_APPLICABLE,
            "n_full": len(bench.datasets),
            "planned": len(bench.planned),
            "variants": list(bench.variants),
            "datasets": [e.slug for e in bench.datasets],
            "planned_datasets": [p["name"] for p in bench.planned],
            "excluded_models": [{"families": list(x.families), "reason": x.reason}
                                for x in bench.excluded_models],
        })
        listed = [e.slug for e in bench.datasets] + [f"{p['name']} (planned)" for p in bench.planned]
        if args.verbose and args.format == "table":
            # one line per dataset: its name, how it is obtained, its own task or
            # note; then the families the benchmark leaves out, on one line
            left_out = _left_out_line(bench)
            notes[len(rows) - 1] = [_dataset_line(e) for e in bench.datasets] + [
                f"{p['name']}: planned ({p['reason']})" for p in bench.planned] + (
                [left_out] if left_out else [])
        elif listed:
            notes[len(rows) - 1] = [", ".join(listed)]
    if not _done(args, rows, len(every), "benchmarks"):
        return
    render(rows, ["benchmark", "domain", "task", "headline", "chance", "single",
                  "n_full", "planned", "variants"], args.format, notes,
           extra=["headline_key", "datasets", "planned_datasets", "excluded_models"],
           labels={"single": "quick dataset", "n_full": "datasets"})
    if args.format == "table":
        print("\n" + "\n".join(_msg.legend([
            ("quick dataset", "the one dataset `run` and `check` use by default "
                              "(--dataset single); --dataset full runs all of them"),
            ("datasets", "how many datasets the benchmark has (listed under its row); "
                         "planned: how many more the paper lists that cannot run here yet"),
            ("chance", "the headline's value for a classifier that guesses (-: the "
                       "headline has no fixed chance value)"),
            ("variants", "other protocols beside the default one (--variant NAME; "
                         "`neuroatlas show <benchmark>` explains each)"),
        ])))
        print(f"\n{_count(args, len(rows), len(every), 'benchmarks', '')}. "
              + ("" if args.verbose else "-v adds each dataset's name, task and notes, and "
                                         "the checkpoint families each benchmark leaves out. ")
              + "`neuroatlas show <benchmark>` explains one; "
                "`neuroatlas run <benchmark> --dry-run` plans it.")


def _dataset_line(entry) -> str:
    from neuroatlas import data
    from neuroatlas.benchmarking_helpers.registry.discovery import dataset_specs

    spec = next((s for s in dataset_specs() if s.slug == entry.slug), None)
    manifest = (spec.manifest if spec else None) or {}
    parts = [str(manifest.get("name") or (spec.description if spec else "") or "").strip()]
    acq = manifest.get("acquisition") or {}
    kind = acq.get("kind")
    how = data.download_word(kind, data.HANDLERS.get(kind, "manual")) if kind else ""
    where = "" if how == "from the authors" else f" from {data.host_word(kind, acq)}"
    parts += [f"{how}{where}" if kind else "",
              f"task {entry.task}" if entry.task else "", entry.note or ""]
    return f"{entry.slug}: " + "; ".join(p for p in parts if p)


def _grep_benchmarks(benches, pattern):
    if not pattern:
        return list(benches)
    p = pattern.lower()
    return [b for b in benches if p in b.name or p in b.domain
            or any(p in e.slug for e in b.datasets)]


# -- datasets --------------------------------------------------------------------

def _size(gb: Any) -> Any:
    """``392 GB``, ``8.1 GB``, ``3.0 MB``; None when the size is not recorded."""
    from neuroatlas import progress

    try:
        return progress.size(float(gb) * 1e9)
    except (TypeError, ValueError):
        return None


def list_datasets(args) -> None:
    from neuroatlas import catalog, data
    from neuroatlas.benchmarking_helpers.registry.discovery import dataset_specs

    rows = []
    for spec in sorted(dataset_specs(), key=lambda s: s.slug):
        manifest = spec.manifest or {}
        in_paper = bool(manifest.get("paper_dataset")) or "paper_cohort" in manifest
        if not args.all and not in_paper:
            continue
        acq = data.acquisition(spec.slug) if not manifest.get("acquisition") else \
            dict(manifest["acquisition"])
        kind = acq.get("kind")
        handler = "refused" if acq.get("unusable_download") else \
            data.HANDLERS.get(kind or "manual", "manual")
        rows.append({
            "dataset": spec.slug,
            "name": manifest.get("name") or spec.description,
            "domain": manifest.get("domain") or NOT_APPLICABLE,
            "access": kind or NOT_APPLICABLE,
            "host": data.host_word(kind, acq) if kind else NOT_APPLICABLE,
            "download": data.download_word(kind or "manual", handler),
            "size_gb": acq.get("size_gb"),
            "size": _size(acq.get("size_gb")),
            "benchmarks": catalog.benchmarks_using(spec.slug),
            "source": data.origin_text(kind, acq, spec.slug) or NOT_APPLICABLE,
        })
    total = len(rows)
    rows = _grep(rows, args.grep, ["dataset", "name", "domain", "access", "host", "download",
                                   "benchmarks"])
    scope = "readable here" if args.all else "in the paper"
    if not _done(args, rows, total, "datasets", scope):
        return
    human = args.format in ("table", "md")
    columns = ["dataset", "name", "domain", "host", "download",
               "size" if human else "size_gb", "benchmarks"]
    if not human:
        columns.insert(3, "access")
    render(rows, columns + (["source"] if args.verbose else []), args.format,
           extra=[] if args.verbose else ["source"], labels={"source": "source"})
    if args.format == "table":
        words = [w for w in dict.fromkeys(r["download"] for r in rows)]
        explained = [(w, data.DOWNLOAD_MEANINGS[w]) for w in words
                     if w in data.DOWNLOAD_MEANINGS]
        if any(r["size"] is None for r in rows):
            explained.append(("size n/a", "the host does not publish a size"))
        print("\n" + "\n".join(_msg.legend(explained)))
        count = _count(args, len(rows), total, "datasets", scope)
        more = "" if args.all else " (--all adds the ones readable here that the paper " \
                                   "does not evaluate)"
        print(f"\n{count}{more}. `neuroatlas data status <dataset>` says whether one is here "
              f"and how to get it" + ("" if args.verbose else "; -v adds where each comes from")
              + ".")


# -- models ----------------------------------------------------------------------

def list_models(args) -> None:
    from neuroatlas import selectors
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    # a registry entry that is not ready is not part of the release
    specs = [s for s in checkpoint_registry() if s.status == "ready"]
    if args.benchmark:
        from neuroatlas import catalog

        # what the benchmark evaluates: the selection (default: every
        # checkpoint) without the families the benchmark leaves out
        wanted = set(catalog.load(args.benchmark).select_models(args.selector or "all"))
        specs = [s for s in specs if s.identifier in wanted]
    elif args.selector:
        wanted = set(selectors.resolve_models(args.selector))
        specs = [s for s in specs if s.identifier in wanted]
    from neuroatlas import models

    rows = [{
        "checkpoint": s.identifier,
        "family": s.model_family,
        "group": "baseline" if selectors.is_baseline(s) else selectors.group_of(s.model_family),
        "weights": s.source_type,
        "weights_from": models.source_word(s.source_type, s.model_family),
        "hz": int(s.expected_sampling_rate) if s.expected_sampling_rate else "any",
        "window_s": (float(s.expected_epoch_seconds) if s.expected_epoch_seconds
                     else NOT_APPLICABLE),
        "dim": s.embedding_dim,
        "source": s.source_reference or NOT_APPLICABLE,
    } for s in specs]
    total = len(rows)
    rows = _grep(rows, args.grep, ["checkpoint", "family", "group", "weights", "weights_from"])
    if not _done(args, rows, total, "checkpoints"):
        return
    human = args.format in ("table", "md")
    columns = ["checkpoint", "family", "group",
               "weights_from" if human else "weights", "hz", "window_s", "dim"]
    render(rows, columns + (["source"] if args.verbose else []), args.format,
           extra=[] if args.verbose else ["source"], formats={"window_s": "{:.3g}"},
           labels={"weights_from": "weights from", "hz": "rate (Hz)", "window_s": "window (s)",
                   "dim": "embedding size", "source": "from (URL)"})
    if args.format == "table":
        print("\n" + "\n".join(_msg.legend([
            ("rate, window", "the input the model was built for. A benchmark cuts its own "
                             "windows (30 s epochs in sleep, 10 s windows in epilepsy, the "
                             "trial in BCI), and each model gets them resampled to its rate; "
                             "any: the recording's own rate"),
            ("embedding size", "the length of the vector the model gives per window, which "
                               "the probe is fitted on"),
        ])))
        scope = f"the {args.benchmark} benchmark evaluates" if args.benchmark else ""
        print(f"\n{_count(args, len(rows), total, 'checkpoints', scope)}"
              + ". `neuroatlas models status` says whether their weights are on this machine."
              + ("" if args.verbose else " -v adds where each comes from."))
        print("Every value of the `group` column, every family and every checkpoint id "
              "is a selector for -m / --models.")


def list_aliases(args) -> None:
    from neuroatlas import selectors

    groups = selectors.model_groups()
    rows, notes = [], {}
    for name, rule in groups["aliases"].items():
        members = selectors.alias_members(name)
        rows.append({"alias": name, "kind": "alias", "n": len(members),
                     "selects": rule.get("description", ""), "members": members})
        notes[len(rows) - 1] = [", ".join(members)]
    for name, group in groups["groups"].items():
        members = selectors.resolve_models(name)
        rows.append({"alias": name, "kind": "group", "n": len(members),
                     "selects": first_sentence(group["description"]), "members": members})
        notes[len(rows) - 1] = [", ".join(members)]
    total = len(rows)
    rows = _grep(rows, args.grep, ["alias", "selects", "members"])
    notes = {i: [", ".join(r["members"])] for i, r in enumerate(rows)}
    if not _done(args, rows, total, "aliases"):
        return
    render(rows, ["alias", "kind", "n", "selects"], args.format,
           notes if args.verbose or args.format != "table" else None, extra=["members"],
           labels={"alias": "name", "n": "checkpoints", "selects": "what it selects"})
    if args.format == "table":
        print("\nkind: alias, a name for a set of checkpoints; group, one of the paper's "
              "model groups (the `group` column of `neuroatlas list models`).")
        print("Use any of these, a family (reve: its trained checkpoints) or a checkpoint id "
              "with -m / --models; combine them with commas."
              + ("" if args.verbose else " -v lists each one's checkpoints."))


def _task_about(preset: Dict[str, Any]) -> str:
    """A preset's description for users: its ``about`` (the ``_doc`` beside
    it is a developer note, never printed); a preset without one gets the
    first sentence of its ``_doc``."""
    about = preset.get("about")
    if about:
        return " ".join(str(about).split())
    return first_sentence(preset.get("_doc", ""))


def _tasks_in_use() -> Dict[str, List[str]]:
    """task -> the benchmarks whose probe runs it (a BCI benchmark runs
    linear_probe on each dataset's own labels)."""
    from neuroatlas import catalog

    used: Dict[str, List[str]] = {}
    for bench in catalog.catalog().values():
        if bench.derived_from:
            continue
        for entry in bench.datasets:
            task = entry.task or bench.task or "linear_probe"
            names = used.setdefault(task, [])
            if bench.name not in names:
                names.append(bench.name)
    return used


def list_tasks(args) -> None:
    from neuroatlas.benchmarking_helpers.registry.discovery import task_specs
    from neuroatlas.entrypoints.probe import available_tasks, load_task_preset

    used = _tasks_in_use()
    presets = available_tasks()
    # `--task brain_age` is the preset when there is one of that name
    rows = [{"task": s.slug, "kind": "task",
             "benchmarks": [] if s.slug in presets else used.get(s.slug, []),
             "about": " ".join(str(s.description).split())}
            for s in task_specs()]
    for name in presets:
        preset = load_task_preset(name)
        rows.append({"task": name, "kind": f"preset of {preset['task']['name']}",
                     "benchmarks": used.get(name, []), "about": _task_about(preset)})
    # The tasks a user can run: those a benchmark uses. -v (and every
    # machine format) adds the rest of the registry.
    every = len(rows)
    if not args.verbose and args.format == "table":
        rows = [r for r in rows if r["benchmarks"]]
        rows.sort(key=lambda r: (r["benchmarks"][0], r["task"]))
    total = len(rows)
    rows = _grep(rows, args.grep, ["task", "kind", "about", "benchmarks"])
    if not _done(args, rows, total, "tasks"):
        return
    columns = ["task", *(["kind"] if args.verbose or args.format != "table" else []),
               "benchmarks", "about"]
    render(rows, columns, args.format, labels={"benchmarks": "used by"})
    if args.format == "table":
        print(f"\n{_count(args, len(rows), total, 'tasks', '')}, the ones the benchmarks run"
              if not args.verbose else f"\n{_count(args, len(rows), total, 'tasks', '')}",
              end="")
        print(". `neuroatlas run <benchmark>` picks its task itself; "
              "`neuroatlas probe --task <task>` runs one by hand."
              + (f" -v lists all {every}." if not args.verbose
                 else " kind: a task is the code that fits the probe; a preset is a task with "
                      "the settings a benchmark runs it with (--task takes either)."))


LISTINGS = {
    "benchmarks": (list_benchmarks, "What each benchmark predicts, on which datasets, scored how."),
    "datasets": (list_datasets, "Every paper dataset: domain, access, size, benchmarks."),
    "models": (list_models, "Checkpoints: family, group, input rate and window, embedding size."),
    "aliases": (list_aliases, "Names for groups of checkpoints, for -m / --models."),
    "tasks": (list_tasks, "Probe tasks, and the settings each benchmark runs them with."),
}

_VERBOSE_HELP = {
    "benchmarks": "Also show each dataset's full name, how it is obtained and its task.",
    "datasets": "Also show where each dataset comes from (its web page, or its MOABB name).",
    "models": "Also show where each checkpoint's weights come from.",
    "aliases": "Also list each alias's checkpoints.",
    "tasks": "Also list the tasks no benchmark runs, and each one's kind.",
}


def build_parser() -> Parser:
    parser = Parser(prog="neuroatlas list", description=__doc__)
    sub = parser.add_subparsers(dest="what", metavar="<what>")
    for name, (_, summary) in LISTINGS.items():
        p = sub.add_parser(name, help=summary, description=summary)
        p.add_argument("--grep", metavar="TEXT", help="Only rows mentioning TEXT.")
        p.add_argument("-v", "--verbose", action="store_true", help=_VERBOSE_HELP[name])
        add_format_arg(p)
        if name == "models":
            p.add_argument("selector", nargs="?",
                           help="Only these: an alias, group, family or ids (e.g. all_fm, "
                                "baseline, reve).")
            p.add_argument("--benchmark", metavar="NAME", default=None,
                           help="Only the checkpoints this benchmark evaluates: the selector "
                                "(default: every checkpoint) without the model families the "
                                "benchmark leaves out.")
        if name == "datasets":
            p.add_argument("--all", action="store_true",
                           help="Also list the datasets readable here that the paper does "
                                "not evaluate.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.what is None:
        parser.print_help()
        raise SystemExit(2)
    LISTINGS[args.what][0](args)
