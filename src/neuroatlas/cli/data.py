"""``neuroatlas data`` -- is each dataset here, how to get it, and build its cache.

    neuroatlas data status  <benchmark|dataset|domain>...  [-v]
    neuroatlas data download <dataset>... [--dry-run]
    neuroatlas data prepare <dataset> [prepare options]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

from neuroatlas.cli._table import add_format_arg, render

DOMAINS = ("epilepsy", "sleep", "brain_age", "bci", "all")


def expand_targets(targets: List[str]) -> List[str]:
    """Benchmarks, domains and slugs -> dataset slugs, in order, deduplicated."""
    from neuroatlas import catalog
    from neuroatlas.benchmarking_helpers.registry.discovery import dataset_specs

    specs = {s.slug: s for s in dataset_specs()}
    out: List[str] = []
    for target in targets or ["all"]:
        if target in catalog.catalog():
            slugs = [e.slug for e in catalog.load(target).datasets]
        elif target in DOMAINS:
            slugs = sorted({e.slug for b in catalog.catalog().values() for e in b.datasets
                            if target == "all" or b.domain == target})
        elif target in specs:
            slugs = [target]
        else:
            import difflib

            close = difflib.get_close_matches(target, [*catalog.catalog(), *DOMAINS, *specs], n=3)
            hint = f" Did you mean {', '.join(close)}?" if close else ""
            raise catalog.CatalogError(f"{target!r} is not a benchmark, domain or dataset.{hint}")
        out.extend(s for s in slugs if s not in out)
    return out


def _data_rel(path: Optional[Path]) -> str:
    """Show paths under the data root as $DATA/..., which is what users type."""
    from neuroatlas import config

    if path is None:
        return "-"
    root = config.get("data_root")
    if root:
        try:
            return "$DATA/" + str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def cmd_status(args) -> int:
    from neuroatlas import data

    slugs = expand_targets(args.targets)
    rows, notes, missing = [], {}, []
    for slug in slugs:
        st = data.status(slug)
        row = {"dataset": slug, "access": st.access, "state": st.state, "handler": st.handler,
               "map": "present" if st.channel_map else "absent"}
        if args.verbose:
            row["path"] = _data_rel(st.path)
        rows.append(row)
        if st.notes:
            notes[len(rows) - 1] = st.notes
        if not st.found and st.path is not None:
            missing.append((slug, st))
    columns = ["dataset", "access", "state", "handler", "map"] + (["path"] if args.verbose else [])
    render(rows, columns, args.format, notes if args.format == "table" else None)
    if args.format != "table":
        return 0
    counts = Counter(r["state"].split(" (")[0] for r in rows)
    print("\nstates: " + "   ".join(f"{k} {v}" for k, v in counts.most_common()))
    if missing and not args.verbose:
        print("\nnot found (looked in):")
        width = max(len(s) for s, _ in missing)
        for slug, st in missing:
            print(f"  {slug:<{width}}  {_data_rel(st.path)}")
        slug, st = missing[0]
        key = st.path_key or "data_root"
        print("If you have one under a different name or folder:\n"
              f"  neuroatlas config set {slug}.{key} /your/path")
    return 0


def cmd_download(args) -> int:
    from neuroatlas import data

    status = 0
    for slug in expand_targets(args.datasets):
        plan = data.plan_download(slug)
        if args.dry_run:
            if plan.commands:
                where = f"  (in {plan.cwd})" if plan.cwd else ""
                for argv in plan.commands:
                    print(f"command: {' '.join(argv)}{where}")
                for line in plan.after:
                    print(f"note: {line}")
                print(f"{slug}: dry-run")
            else:
                print(f"{slug}: {plan.handler} — {plan.message}")
            continue
        status = max(status, data.run_download(plan))
    return status


def cmd_prepare(argv: List[str]) -> int:
    from neuroatlas import data
    from neuroatlas.entrypoints import prepare

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--dataset")
    probe.add_argument("--list", action="store_true")
    probe.add_argument("--dry-run", action="store_true")
    known, _ = probe.parse_known_args(argv)
    if known.dataset and not known.list and not known.dry_run:
        st = data.status(known.dataset)
        # MOABB builders fetch as they go; everything else needs the raw corpus
        if st.kind != "moabb" and not st.found:
            print(f"error: {known.dataset}'s raw corpus is not there ({st.state}: "
                  f"{_data_rel(st.path)}). Run `neuroatlas data download {known.dataset}` "
                  f"or point at your copy with `neuroatlas config set`.", file=sys.stderr)
            return 2
    prepare.main(argv)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neuroatlas data", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", metavar="<action>")
    st = sub.add_parser("status", help="Is each dataset of a benchmark on this machine?",
                        description="Is each dataset on this machine, where was it looked "
                                    "for, and how would `data download` get it. Takes "
                                    "benchmarks, domains (epilepsy, sleep, brain_age, bci, "
                                    "all) or dataset slugs; default all.")
    st.add_argument("targets", nargs="*")
    st.add_argument("-v", "--verbose", action="store_true", help="Show every path as a column.")
    add_format_arg(st)
    dl = sub.add_parser("download", help="Fetch a dataset, or print how to.",
                        description="Fetch datasets into the folder their reader expects. "
                                    "Credentialed and manual ones print instructions.")
    dl.add_argument("datasets", nargs="+")
    dl.add_argument("--dry-run", action="store_true", help="Print the commands; run nothing.")
    sub.add_parser("prepare", add_help=False,
                   help="Build a dataset's cache (required for the MOABB MI cohorts).")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["prepare"]:
        rest = argv[1:]
        # `data prepare bonn` reads more naturally than `--dataset bonn`
        if rest and not rest[0].startswith("-"):
            rest = ["--dataset", rest[0], *rest[1:]]
        raise SystemExit(cmd_prepare(rest))
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action is None:
        parser.print_help()
        raise SystemExit(2)
    status = {"status": cmd_status, "download": cmd_download}[args.action](args)
    if status:
        raise SystemExit(status)
