"""``neuroatlas models`` -- are a selection's weights here, and fetch them.

    neuroatlas models status [SELECTOR] [-v]
    neuroatlas models download SELECTOR
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import List, Optional

from neuroatlas.cli import Parser
from neuroatlas.cli._table import add_format_arg, render


def _specs(selector: Optional[str], include_planned: bool = True):
    """The checkpoints a selector names; planned ones are kept as rows here.

    Elsewhere naming a planned checkpoint is an error (it cannot run), but
    asking whether its weights are here deserves the answer `planned`.
    """
    from neuroatlas import selectors
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    specs = checkpoint_registry()
    if not selector:
        return [s for s in specs if include_planned or s.status == "ready"]
    by_id = {s.identifier: s for s in specs}
    out = []
    for name in (n.strip() for n in selector.split(",")):
        if not name:
            continue
        spec = by_id.get(name.lower())
        picked = [spec] if spec is not None and spec.status != "ready" else \
            [by_id[i] for i in selectors.resolve_models([name])]
        out.extend(s for s in picked if s not in out)
    if not out:
        raise selectors.SelectionError("no models selected")
    return out


def cmd_status(args) -> int:
    from neuroatlas import models

    rows, notes, missing = [], {}, []
    for spec in _specs(args.selector):
        st = models.status(spec)
        row = {"checkpoint": st.identifier, "source": st.source, "state": st.state,
               "note": "; ".join(st.notes) or None}
        if args.verbose:
            row["path"] = st.path or "-"
        rows.append(row)
        if st.notes:
            notes[len(rows) - 1] = st.notes
        if st.fetchable:
            missing.append(st.identifier)
    columns = ["checkpoint", "source", "state"] + (["path"] if args.verbose else [])
    # the renderer prints the notes under the table, or as every row's `note` field
    render(rows, columns, args.format, notes)
    if args.format == "table":
        counts = Counter(r["state"] for r in rows)
        print("\nstates: " + "   ".join(f"{k} {v}" for k, v in counts.most_common()))
        if missing:
            target = ",".join(missing) if len(missing) <= 6 else (args.selector or "all")
            print(f"fetch the {len(missing)} missing with `neuroatlas models download {target}`")
    return 0


def cmd_download(args) -> int:
    """Fetch what the selection lacks; exit 1 unless every one ends up usable."""
    from neuroatlas import models

    failed = unobtained = 0
    for spec in _specs(args.selector):
        before = models.status(spec)
        note = "; ".join(before.notes)
        if before.ready:
            print(f"{spec.identifier}: {before.state}")
            continue
        if not before.fetchable:
            unobtained += 1
            print(f"{spec.identifier}: {before.state} -- {note or 'nothing to download'}")
            continue
        print(f"{spec.identifier}: downloading ({before.source})"
              + (f" -- {note}" if note and before.state != "package missing" else ""), flush=True)
        try:
            after = models.download(spec)
        except Exception as exc:                       # report and carry on
            failed += 1
            print(f"  failed: {exc}", file=sys.stderr)
            continue
        print(f"  {after.state}" + (f" -- {after.notes[0]}" if after.state == "package missing" else ""))
        if not after.ready:
            unobtained += 1
    if failed or unobtained:
        print(f"\n{failed} failed, {unobtained} not usable yet (see the lines above)", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = Parser(prog="neuroatlas models", description=__doc__)
    sub = parser.add_subparsers(dest="action", metavar="<action>")
    st = sub.add_parser("status", help="Whether each checkpoint's weights are here. No network.")
    st.add_argument("selector", nargs="?", help="An alias, group, family or ids (default: every checkpoint).")
    st.add_argument("-v", "--verbose", action="store_true", help="Show where each is expected.")
    add_format_arg(st)
    dl = sub.add_parser("download", help="Fetch the weights a selection is missing; "
                                         "exit 1 unless every one ends up usable.")
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
