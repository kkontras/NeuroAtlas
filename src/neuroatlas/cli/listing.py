"""``neuroatlas list`` -- what exists: benchmarks, datasets, models, aliases, tasks.

Listing reads the registry and the shipped tables only. It never looks at
your data or weights; `data status` and `models status` do that.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from neuroatlas.cli._table import add_format_arg, render


def _grep(rows: List[Dict[str, Any]], pattern: Optional[str], keys: List[str]) -> List[Dict[str, Any]]:
    if not pattern:
        return rows
    p = pattern.lower()
    return [r for r in rows if any(p in str(r.get(k, "")).lower() for k in keys)]


# -- benchmarks ------------------------------------------------------------------

def list_benchmarks(args) -> None:
    from neuroatlas import catalog

    rows, notes = [], {}
    for bench in _grep_benchmarks(catalog.catalog().values(), args.grep):
        m = bench.metrics
        dummy = ("per dataset" if isinstance(m.dummy, dict) or m.dummy is None else m.dummy)
        rows.append({
            "benchmark": bench.name,
            "domain": bench.domain,
            "task": bench.task or ("derived" if bench.derived_from else "per dataset"),
            "headline": m.headline + ("" if m.higher_is_better else " (lower better)"),
            "dummy": dummy if bench.metrics.higher_is_better else "-",
            "single": bench.single,
            "n_full": len(bench.datasets),
            "planned": len(bench.planned),
            "variants": ", ".join(bench.variants) or "-",
        })
        listed = [e.slug for e in bench.datasets] + [f"{p['name']} (planned)" for p in bench.planned]
        notes[len(rows) - 1] = [", ".join(listed)] if listed else []
    render(rows, ["benchmark", "domain", "task", "headline", "dummy", "single",
                  "n_full", "planned", "variants"], args.format, notes if args.verbose or args.format == "table" else None)
    if args.format == "table":
        print("\n`neuroatlas show <benchmark>` explains one; "
              "`neuroatlas run <benchmark> --dry-run` plans it.")


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
            "domain": manifest.get("domain", "-"),
            "access": acq.get("kind", "-"),
            "size_gb": acq.get("size_gb"),
            "benchmarks": ", ".join(catalog.benchmarks_using(spec.slug)) or "-",
        })
    rows = _grep(rows, args.grep, ["dataset", "name", "domain", "access", "benchmarks"])
    render(rows, ["dataset", "name", "domain", "access", "size_gb", "benchmarks"], args.format)
    if args.format == "table":
        scope = "registered" if args.all else "in the paper (--all for every registered one)"
        print(f"\n{len(rows)} datasets {scope}. `neuroatlas fetch --dataset X` says how to get one.")


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
        "window_s": (f"{s.expected_epoch_seconds:g}" if s.expected_epoch_seconds else "-"),
        "dim": s.embedding_dim,
    } for s in specs]
    rows = _grep(rows, args.grep, ["checkpoint", "family", "group", "weights"])
    render(rows, ["checkpoint", "family", "group", "status", "weights", "hz", "window_s", "dim"],
           args.format)
    if args.format == "table":
        print(f"\n{len(rows)} checkpoints. `ready` means the model's code is in place, not that "
              f"its weights are on this machine.")


def list_aliases(args) -> None:
    from neuroatlas import selectors

    groups = selectors.model_groups()
    rows, notes = [], {}
    for name, rule in groups["aliases"].items():
        members = selectors.alias_members(name)
        rows.append({"alias": name, "n": len(members), "selects": rule.get("description", "")})
        if args.verbose or args.format == "table":
            notes[len(rows) - 1] = [", ".join(members)]
    for name, group in groups["groups"].items():
        members = selectors.resolve_models(name)
        rows.append({"alias": name, "n": len(members),
                     "selects": f"paper group: {' '.join(str(group['description']).split())[:70]}..."})
    render(rows, ["alias", "n", "selects"], args.format, notes)
    if args.format == "table":
        print("\nUse any of these, a family (reve) or a checkpoint id with -m / --models; "
              "combine them with commas.")


def list_tasks(args) -> None:
    from neuroatlas.benchmarking_helpers.registry.discovery import task_specs
    from neuroatlas.entrypoints.probe import available_tasks, load_task_preset

    rows = [{"task": s.slug, "kind": "registered", "about": s.description} for s in task_specs()]
    for name in available_tasks():
        preset = load_task_preset(name)
        rows.append({"task": name, "kind": f"preset -> {preset['task']['name']}",
                     "about": " ".join(str(preset.get("_doc", "")).split())[:90]})
    rows = _grep(rows, args.grep, ["task", "kind", "about"])
    render(rows, ["task", "kind", "about"], args.format)


LISTINGS = {
    "benchmarks": (list_benchmarks, "What each benchmark predicts, on which datasets, scored how."),
    "datasets": (list_datasets, "Every paper dataset: domain, access, size, benchmarks."),
    "models": (list_models, "Checkpoints: family, group, input rate and window, embedding size."),
    "aliases": (list_aliases, "Names for groups of checkpoints, for -m / --models."),
    "tasks": (list_tasks, "Registered probe tasks and the presets built on them."),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neuroatlas list", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="what", metavar="<what>")
    for name, (_, summary) in LISTINGS.items():
        p = sub.add_parser(name, help=summary, description=summary)
        p.add_argument("--grep", metavar="TEXT", help="Only rows mentioning TEXT.")
        p.add_argument("-v", "--verbose", action="store_true", help="Show members / datasets.")
        add_format_arg(p)
        if name == "models":
            p.add_argument("selector", nargs="?",
                           help="Only these: an alias, group, family or ids (e.g. all_fm).")
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
