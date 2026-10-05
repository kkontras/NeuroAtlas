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

from neuroatlas.cli import Parser
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

def list_benchmarks(args) -> None:
    from neuroatlas import catalog

    every = list(catalog.catalog().values())
    rows, notes = [], {}
    for bench in _grep_benchmarks(every, args.grep):
        m = bench.metrics
        dummy = ("per dataset" if isinstance(m.dummy, dict) or m.dummy is None else m.dummy)
        rows.append({
            "benchmark": bench.name,
            "domain": bench.domain,
            "task": bench.task or ("derived" if bench.derived_from else "per dataset"),
            "headline": m.headline + ("" if m.higher_is_better else " (lower better)"),
            "dummy": dummy if bench.metrics.higher_is_better else NOT_APPLICABLE,
            "single": bench.single or NOT_APPLICABLE,
            "n_full": len(bench.datasets),
            "planned": len(bench.planned),
            "variants": list(bench.variants),
            "datasets": [e.slug for e in bench.datasets],
            "planned_datasets": [p["name"] for p in bench.planned],
        })
        listed = [e.slug for e in bench.datasets] + [f"{p['name']} (planned)" for p in bench.planned]
        if args.verbose and args.format == "table":
            # one line per dataset: its name, how it is obtained, its own task or note
            notes[len(rows) - 1] = [_dataset_line(e) for e in bench.datasets] + [
                f"{p['name']}: planned -- {p['reason']}" for p in bench.planned]
        elif listed:
            notes[len(rows) - 1] = [", ".join(listed)]
    if not _done(args, rows, len(every), "benchmarks"):
        return
    render(rows, ["benchmark", "domain", "task", "headline", "dummy", "single",
                  "n_full", "planned", "variants"], args.format, notes,
           extra=["datasets", "planned_datasets"])
    if args.format == "table":
        print(f"\n{_count(args, len(rows), len(every), 'benchmarks', '')}. "
              + ("" if args.verbose else "-v adds each dataset's task and notes. ")
              + "`neuroatlas show <benchmark>` explains one; "
                "`neuroatlas run <benchmark> --dry-run` plans it.")


def _dataset_line(entry) -> str:
    from neuroatlas.benchmarking_helpers.registry.discovery import dataset_specs

    spec = next((s for s in dataset_specs() if s.slug == entry.slug), None)
    manifest = (spec.manifest if spec else None) or {}
    parts = [str(manifest.get("name") or (spec.description if spec else "") or "").strip()]
    kind = (manifest.get("acquisition") or {}).get("kind")
    parts += [f"access {kind}" if kind else "", f"task {entry.task}" if entry.task else "",
              entry.note or ""]
    return f"{entry.slug}: " + "; ".join(p for p in parts if p)


def _grep_benchmarks(benches, pattern):
    if not pattern:
        return list(benches)
    p = pattern.lower()
    return [b for b in benches if p in b.name or p in b.domain
            or any(p in e.slug for e in b.datasets)]


# -- datasets --------------------------------------------------------------------

def list_datasets(args) -> None:
    from neuroatlas import catalog
    from neuroatlas.benchmarking_helpers.registry.discovery import dataset_specs

    rows = []
    for spec in sorted(dataset_specs(), key=lambda s: s.slug):
        manifest = spec.manifest or {}
        in_paper = bool(manifest.get("paper_dataset")) or "paper_cohort" in manifest
        if not args.all and not in_paper:
            continue
        acq = manifest.get("acquisition") or {}
        rows.append({
            "dataset": spec.slug,
            "name": manifest.get("name") or spec.description,
            "domain": manifest.get("domain") or NOT_APPLICABLE,
            "access": acq.get("kind") or NOT_APPLICABLE,
            "size_gb": acq.get("size_gb"),
            "benchmarks": catalog.benchmarks_using(spec.slug),
            "source": acq.get("ref") or NOT_APPLICABLE,
        })
    total = len(rows)
    rows = _grep(rows, args.grep, ["dataset", "name", "domain", "access", "benchmarks"])
    scope = "registered" if args.all else "in the paper"
    if not _done(args, rows, total, "datasets", scope):
        return
    columns = ["dataset", "name", "domain", "access", "size_gb", "benchmarks"]
    render(rows, columns + (["source"] if args.verbose else []), args.format,
           extra=[] if args.verbose else ["source"])
    if args.format == "table":
        count = _count(args, len(rows), total, "datasets", scope)
        more = "" if args.all else " (--all for every registered one)"
        print(f"\n{count}{more}. `neuroatlas data status <dataset>` says whether one is here "
              f"and how to get it" + ("" if args.verbose else "; -v adds where each comes from")
              + ".")


# -- models ----------------------------------------------------------------------

def list_models(args) -> None:
    from neuroatlas import selectors
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    specs = checkpoint_registry()
    if args.selector:
        wanted = set(selectors.resolve_models(args.selector))
        specs = [s for s in specs if s.identifier in wanted]
    elif not args.all:
        specs = [s for s in specs if s.status == "ready"]
    rows = [{
        "checkpoint": s.identifier,
        "family": s.model_family,
        "group": "baseline" if selectors.is_baseline(s) else selectors.group_of(s.model_family),
        "status": s.status,
        "weights": s.source_type,
        "hz": int(s.expected_sampling_rate) if s.expected_sampling_rate else "any",
        "window_s": (float(s.expected_epoch_seconds) if s.expected_epoch_seconds
                     else NOT_APPLICABLE),
        "dim": s.embedding_dim,
        "source": s.source_reference or NOT_APPLICABLE,
    } for s in specs]
    total = len(rows)
    rows = _grep(rows, args.grep, ["checkpoint", "family", "group", "weights"])
    if not _done(args, rows, total, "checkpoints"):
        return
    columns = ["checkpoint", "family", "group", "status", "weights", "hz", "window_s", "dim"]
    render(rows, columns + (["source"] if args.verbose else []), args.format,
           extra=[] if args.verbose else ["source"], formats={"window_s": "{:.3g}"})
    if args.format == "table":
        print(f"\n{_count(args, len(rows), total, 'checkpoints', '')}. `ready` means the "
              f"model's code is in place, not that its weights are on this machine "
              f"(`neuroatlas models status`)." + ("" if args.verbose else
                                                   " -v adds where each comes from."))
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
        rows.append({"alias": name, "kind": "paper group", "n": len(members),
                     "selects": first_sentence(group["description"]), "members": members})
        notes[len(rows) - 1] = [", ".join(members)]
    total = len(rows)
    rows = _grep(rows, args.grep, ["alias", "selects", "members"])
    notes = {i: [", ".join(r["members"])] for i, r in enumerate(rows)}
    if not _done(args, rows, total, "aliases"):
        return
    render(rows, ["alias", "kind", "n", "selects"], args.format,
           notes if args.verbose or args.format != "table" else None, extra=["members"])
    if args.format == "table":
        print("\nUse any of these, a family (reve: its trained checkpoints) or a checkpoint id "
              "with -m / --models; combine them with commas."
              + ("" if args.verbose else " -v lists each one's members."))


def _task_about(doc: Any, verbose: bool) -> str:
    """A preset's description for users: its first sentence (the full text
    with -v), without the developer history ("Was probe_x.py.")."""
    flat = " ".join(str(doc or "").split())
    sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", flat)
                 if s and not s.startswith("Was ")]
    if verbose:
        return " ".join(sentences)
    about = sentences[0] if sentences else ""
    if re.search(r"\bnot part of the neuroatlas benchmark\b", flat, re.I):
        about += " Not part of the benchmark."
    return about


def list_tasks(args) -> None:
    from neuroatlas.benchmarking_helpers.registry.discovery import task_specs
    from neuroatlas.entrypoints.probe import available_tasks, load_task_preset

    rows = [{"task": s.slug, "kind": "registered",
             "about": " ".join(str(s.description).split()) if args.verbose
             else first_sentence(s.description)} for s in task_specs()]
    for name in available_tasks():
        preset = load_task_preset(name)
        rows.append({"task": name, "kind": f"preset -> {preset['task']['name']}",
                     "about": _task_about(preset.get("_doc", ""), args.verbose)})
    total = len(rows)
    rows = _grep(rows, args.grep, ["task", "kind", "about"])
    if not _done(args, rows, total, "tasks"):
        return
    render(rows, ["task", "kind", "about"], args.format)
    if args.format == "table":
        print(f"\n{_count(args, len(rows), total, 'tasks', '')}. A preset is a registered task "
              f"with its settings; `--task <preset>` uses it."
              + ("" if args.verbose else " -v prints each description in full."))


LISTINGS = {
    "benchmarks": (list_benchmarks, "What each benchmark predicts, on which datasets, scored how."),
    "datasets": (list_datasets, "Every paper dataset: domain, access, size, benchmarks."),
    "models": (list_models, "Checkpoints: family, group, input rate and window, embedding size."),
    "aliases": (list_aliases, "Names for groups of checkpoints, for -m / --models."),
    "tasks": (list_tasks, "Registered probe tasks and the presets built on them."),
}

_VERBOSE_HELP = {
    "benchmarks": "Also list each benchmark's datasets.",
    "datasets": "Also show where each dataset comes from (URL or DOI).",
    "models": "Also show where each checkpoint's weights come from.",
    "aliases": "Also list each alias's checkpoints.",
    "tasks": "Print each description in full.",
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
            p.add_argument("--all", action="store_true", help="Include planned checkpoints.")
        if name == "datasets":
            p.add_argument("--all", action="store_true",
                           help="Include registered datasets the paper does not evaluate.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.what is None:
        parser.print_help()
        raise SystemExit(2)
    LISTINGS[args.what][0](args)
