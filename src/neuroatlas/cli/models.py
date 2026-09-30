"""``neuroatlas models`` -- are a selection's weights here, and fetch them.

    neuroatlas models status [SELECTOR] [-v]
    neuroatlas models download SELECTOR
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import List, Optional

from neuroatlas.cli._table import add_format_arg, render


def _specs(selector: Optional[str]):
    from neuroatlas import selectors
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    specs = checkpoint_registry()
    if not selector:
        return [s for s in specs if s.status == "ready"]
    wanted = selectors.resolve_models(selector)
    by_id = {s.identifier: s for s in specs}
    return [by_id[i] for i in wanted]


def cmd_status(args) -> int:
    from neuroatlas import models

    rows, notes = [], {}
    for spec in _specs(args.selector):
        st = models.status(spec)
        row = {"checkpoint": st.identifier, "source": st.source, "state": st.state}
        if args.verbose:
            row["path"] = st.path or "-"
        rows.append(row)
        if st.notes:
            notes[len(rows) - 1] = st.notes
    render(rows, ["checkpoint", "source", "state"] + (["path"] if args.verbose else []),
           args.format, notes if args.format == "table" else None)
    if args.format == "table":
        counts = Counter(r["state"] for r in rows)
        print("\nstates: " + "   ".join(f"{k} {v}" for k, v in counts.most_common()))
        if counts.get("auto") or counts.get("hub"):
            print(f"fetch them with `neuroatlas models download {args.selector or 'all'}`")
    return 0


def cmd_download(args) -> int:
    from neuroatlas import models

    failed = 0
    for spec in _specs(args.selector):
        before = models.status(spec)
        if before.ready:
            continue
        if before.state in ("manual", "planned"):
            print(f"{spec.identifier}: {before.state} -- {'; '.join(before.notes) or 'fetch it by hand'}"
                  + (f" and place it at {before.path}" if before.path else ""))
            continue
        print(f"{spec.identifier}: downloading ({before.source})", flush=True)
        try:
            after = models.download(spec)
        except Exception as exc:                       # report and carry on
            failed += 1
            print(f"  failed: {exc}", file=sys.stderr)
            continue
        print(f"  {after.state}")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neuroatlas models", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", metavar="<action>")
    st = sub.add_parser("status", help="Whether each checkpoint's weights are here. No network.")
    st.add_argument("selector", nargs="?", help="An alias, group, family or ids (default: every ready one).")
    st.add_argument("-v", "--verbose", action="store_true", help="Show where each is expected.")
    add_format_arg(st)
    dl = sub.add_parser("download", help="Fetch the weights a selection is missing.")
    dl.add_argument("selector", help="An alias, group, family or ids, e.g. all_fm.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action is None:
        parser.print_help()
        raise SystemExit(2)
    status = {"status": cmd_status, "download": cmd_download}[args.action](args)
    if status:
        raise SystemExit(status)
