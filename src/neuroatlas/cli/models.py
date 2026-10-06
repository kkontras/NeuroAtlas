"""``neuroatlas models`` -- are a selection's weights here, and fetch them.

    neuroatlas models status [SELECTOR] [-v]
    neuroatlas models download SELECTOR
"""
from __future__ import annotations

import argparse
from collections import Counter
from typing import List, Optional

from neuroatlas.cli import Parser, _msg
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
        raise selectors.SelectionError("no models selected\nfix: neuroatlas list aliases")
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
        print("\n" + _msg.counts(len(rows), "checkpoint" if len(rows) == 1 else "checkpoints",
                                 counts.most_common()))
        if missing:
            target = ",".join(missing) if len(missing) <= 6 else (args.selector or "all")
            print(f"download the {len(missing)} missing: neuroatlas models download {target}")
    return 0


def cmd_download(args) -> int:
    """Fetch what the selection lacks; exit 1 unless every one ends up usable."""
    from neuroatlas import models

    failed = unobtained = done = 0
    specs = _specs(args.selector)
    for spec in specs:
        before = models.status(spec)
        if before.ready:
            print(f"{spec.identifier}: {before.state}")
            done += 1
            continue
        if not before.fetchable:
            # nothing `models download` can fetch: say why, and what to do
            unobtained += 1
            what, fix = models.weights_problem(before) or (before.state, None)
            _msg.error(f"{spec.identifier}: {what}", fix)
            continue
        print(f"{spec.identifier}: downloading ({before.source})", flush=True)
        for note in before.notes:
            # what the download means: a licence it accepts (REVE, DeepSOZ's
            # GPL-3.0), a token it needs, a big image it streams
            if before.state != "package missing":
                print(f"  {note}", flush=True)
        try:
            after = models.download(spec)
        except Exception as exc:                       # report and carry on
            failed += 1
            _msg.error(f"{spec.identifier}: " + _msg.brief(_msg.exception_text(exc)))
            continue
        if after.ready:
            done += 1
            print(f"{spec.identifier}: {after.state}")
        else:
            unobtained += 1
            what, fix = models.weights_problem(after) or (after.state, None)
            _msg.error(f"{spec.identifier}: downloaded, but {what}", fix)
    if len(specs) > 1 or failed or unobtained:
        print(_msg.counts(len(specs), "checkpoint" if len(specs) == 1 else "checkpoints",
                          [("ready", done), ("failed", failed), ("not usable", unobtained)]))
    return 1 if failed or unobtained else 0


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
