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


def _size_on_disk(path: Optional[str]) -> int:
    """Bytes of a checkpoint file or folder (0 for a hub id or nothing)."""
    from pathlib import Path

    p = Path(path) if path else None
    if p is None or not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def cmd_download(args) -> int:
    """Fetch what the selection lacks; exit 1 unless every one ends up usable.

    As every long command shows its work (neuroatlas.progress): a header, a
    live line while a checkpoint downloads (bytes, percent, rate), and one
    result line each: ``[2/3] neurogpt_pretrained: downloaded, 318 MB (6s)``,
    ``found``, or ``failed`` after its error and fix lines."""
    from neuroatlas import models, progress

    failed = unobtained = done = 0
    specs = _specs(args.selector)
    n = len(specs)
    progress.say(f"downloading {n} checkpoint(s)")
    for k, spec in enumerate(specs, 1):
        label = f"[{k}/{n}] {spec.identifier}"
        before = models.status(spec)
        if before.ready:
            progress.say(progress.result_line(k, n, spec.identifier, before.state))
            done += 1
            continue
        if not before.fetchable:
            # nothing `models download` can fetch: say why, and what to do
            unobtained += 1
            what, fix = models.weights_problem(before) or (before.state, None)
            _msg.error(f"{spec.identifier}: {what}", fix)
            progress.say(progress.result_line(k, n, spec.identifier,
                                              f"not downloaded ({before.state})"))
            continue
        # what the download means: a licence it accepts (REVE, DeepSOZ's
        # GPL-3.0), a token it needs, a big image it streams
        notes = [note for note in before.notes if before.state != "package missing"]
        if notes:
            progress.say(f"{label}: downloading ({before.source})")
            for note in notes:
                progress.say(f"  {note}")
        with progress.Progress(label, verb=f"downloading ({before.source})",
                               start_line_off_tty=not notes) as item:
            try:
                after = models.download(spec)
            except Exception as exc:                   # report and carry on
                after, error = None, exc
        if after is None:
            failed += 1
            _msg.error(f"{spec.identifier}: " + _msg.brief(_msg.exception_text(error)))
            progress.say(progress.result_line(k, n, spec.identifier, "failed", item.seconds))
            continue
        fetched = item.counts.get("bytes") or _size_on_disk(after.path)
        what = "downloaded" + (f", {progress.size(fetched)}" if fetched else "")
        if after.ready:
            done += 1
        else:
            unobtained += 1
            problem, fix = models.weights_problem(after) or (after.state, None)
            _msg.error(f"{spec.identifier}: downloaded, but {problem}", fix)
            what += ", not usable"
        progress.say(progress.result_line(k, n, spec.identifier, what, item.seconds))
    if len(specs) > 1 or failed or unobtained:
        print(_msg.counts(len(specs), "checkpoint" if len(specs) == 1 else "checkpoints",
                          [("ready", done), ("failed", failed), ("not usable", unobtained)]),
              flush=True)
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
