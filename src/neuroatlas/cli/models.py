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


def _specs(selector: Optional[str]):
    """The checkpoints a selector names (every checkpoint of this release
    when none is given); naming one that is not part of it is an error."""
    from neuroatlas import selectors
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    specs = [s for s in checkpoint_registry() if s.status == "ready"]
    if not selector:
        return specs
    by_id = {s.identifier: s for s in specs}
    out = []
    for name in (n.strip() for n in selector.split(",")):
        if not name:
            continue
        picked = [by_id[i] for i in selectors.resolve_models([name])]
        out.extend(s for s in picked if s not in out)
    if not out:
        raise selectors.SelectionError("no checkpoint selected\nfix: neuroatlas list aliases")
    return out


def status_row(st, verbose: bool = False) -> dict:
    """One checkpoint as `models status` prints it, in every format: the
    state in words (``downloadable``, not the code ``auto``); ``source`` keeps
    the source type (``github_release``), which the table shows in words."""
    from neuroatlas import models

    row = {"checkpoint": st.identifier, "source": st.source,
           "weights_from": models.source_word(st.source, st.family),
           "state": models.state_word(st.state),
           "note": "; ".join(st.notes) or None}
    if verbose:
        row["path"] = st.path or "-"
    return row


def cmd_status(args) -> int:
    from neuroatlas import models

    rows, notes, missing = [], {}, []
    for spec in _specs(args.selector):
        st = models.status(spec)
        rows.append(status_row(st, args.verbose))
        if st.notes:
            notes[len(rows) - 1] = st.notes
        if st.fetchable:
            missing.append(st.identifier)
    human = args.format in ("table", "md")
    columns = ["checkpoint", "weights_from" if human else "source", "state"] + (
        ["path"] if args.verbose else [])
    # the renderer prints the notes under the table, or as every row's `note` field
    render(rows, columns, args.format, notes, labels={"weights_from": "weights from"})
    if args.format == "table":
        counts = Counter(r["state"] for r in rows)
        print("\n" + _msg.counts(len(rows), "checkpoint" if len(rows) == 1 else "checkpoints",
                                 counts.most_common()))
        print("\n".join(_msg.legend([(s, models.STATE_MEANINGS[s]) for s in counts
                                     if s in models.STATE_MEANINGS
                                     and (args.verbose or s != "found")])))
        if missing:
            target = ",".join(missing) if len(missing) <= 6 else (args.selector or "all")
            # the ones whose package is missing too: their weights download,
            # and they run once the package is installed (the line under the row)
            need = sum(1 for r in rows if r["checkpoint"] in missing
                       and r["state"] == "package missing")
            _msg.note(f"{_msg.plural(len(missing), 'checkpoint')} not here can be downloaded"
                      + (f" ({need} of them also need a Python package to run: the line "
                         f"under each row gives the install)" if need else ""),
                      f"neuroatlas models download {target}")
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
    """Fetch what the selection lacks; exit 1 if a checkpoint's weights could
    not be had (a checkpoint whose weights are here but whose package is not
    installed is not a failure: it says what to install).

    As every long command shows its work (neuroatlas.progress): a header, a
    live line while a checkpoint downloads (bytes, percent, rate), and one
    result line each: ``[2/3] neurogpt_pretrained: downloaded, 318 MB (6s)``,
    ``found``, or ``failed`` after its error and fix lines."""
    from neuroatlas import models, progress

    failed = done = need_package = 0
    specs = _specs(args.selector)
    n = len(specs)
    progress.say(f"downloading {_msg.plural(n, 'checkpoint')}")
    for k, spec in enumerate(specs, 1):
        label = f"[{k}/{n}] {spec.identifier}"
        before = models.status(spec)
        source = models.source_word(before.source)
        if before.ready:
            progress.say(progress.result_line(k, n, spec.identifier,
                                              models.state_word(before.state)))
            done += 1
            continue
        if not before.fetchable:
            what, fix = models.weights_problem(before) or (models.state_word(before.state), None)
            if before.state == "package missing" and (before.weights or "") in (
                    "found", "hub (cached)", "nothing needed"):
                # the weights are here; only the package is missing
                need_package += 1
                _msg.warning(f"{spec.identifier}: its weights are here; {what}", fix)
                progress.say(progress.result_line(k, n, spec.identifier,
                                                  f"found; needs {_package_name(spec)} to run"))
                continue
            # nothing `models download` can fetch: say why, and what to do
            failed += 1
            _msg.error(f"{spec.identifier}: {what}", fix)
            progress.say(progress.result_line(k, n, spec.identifier,
                                              f"failed ({models.state_word(before.state)})"))
            continue
        # what the download means: a licence it accepts (REVE, DeepSOZ's
        # GPL-3.0), a token it needs, a big image it streams
        notes = [note for note in before.notes if before.state != "package missing"]
        if notes:
            progress.say(f"{label}: downloading from {source}")
            for note in notes:
                progress.say(f"  {note}")
        with progress.Progress(label, verb=f"downloading from {source}",
                               start_line_off_tty=not notes) as item:
            try:
                after = models.download(spec)
            except Exception as exc:                   # report and carry on
                after, error = None, exc
        if after is None:
            failed += 1
            text = _msg.brief(_msg.exception_text(error))
            _msg.error(f"{spec.identifier}: {text}",
                       None if _msg.split(text)[2] else
                       f"neuroatlas -v models download {spec.identifier} (shows the whole error)")
            progress.say(progress.result_line(k, n, spec.identifier, "failed", item.seconds))
            continue
        fetched = item.counts.get("bytes") or _size_on_disk(after.path)
        what = "downloaded" + (f", {progress.size(fetched)}" if fetched else "")
        if after.ready:
            done += 1
        elif after.state == "package missing":
            # the weights arrived; the model's Python package is the next step
            need_package += 1
            problem, fix = models.weights_problem(after) or (models.state_word(after.state), None)
            _msg.warning(f"{spec.identifier}: {problem}; it runs once that is installed", fix)
            what += f"; needs {_package_name(spec)} to run"
        else:
            failed += 1
            problem, fix = models.weights_problem(after) or (models.state_word(after.state), None)
            _msg.error(f"{spec.identifier}: downloaded, but its weights are not complete: "
                       f"{problem}", fix)
            what += ", incomplete"
        progress.say(progress.result_line(k, n, spec.identifier, what, item.seconds))
    if len(specs) > 1 or failed or need_package:
        print(_msg.counts(len(specs), "checkpoint" if len(specs) == 1 else "checkpoints",
                          [("ready", done), ("need a package", need_package),
                           ("failed", failed)]),
              flush=True)
    return 1 if failed else 0


def _package_name(spec) -> str:
    """The Python package a checkpoint's family needs (``uni2ts``)."""
    from neuroatlas import models

    entry = models.PACKAGES.get(spec.model_family)
    return entry[1] if entry else "its package"


def build_parser() -> argparse.ArgumentParser:
    parser = Parser(prog="neuroatlas models", description=__doc__)
    sub = parser.add_subparsers(dest="action", metavar="<action>")
    st = sub.add_parser("status", help="Whether each checkpoint's weights are here. No network.")
    st.add_argument("selector", nargs="?", help="An alias, group, family or ids (default: every checkpoint).")
    st.add_argument("-m", "--models", dest="models", default=None, metavar="SELECTOR",
                    help="The same selection, as run and check take it.")
    st.add_argument("-v", "--verbose", action="store_true",
                    help="Show where each is expected, and explain every state.")
    add_format_arg(st)
    dl = sub.add_parser("download", help="Fetch the weights a selection is missing; exit 1 "
                                         "if one could not be fetched.")
    dl.add_argument("selector", nargs="?", help="An alias, group, family or ids, e.g. all_fm.")
    dl.add_argument("-m", "--models", dest="models", default=None, metavar="SELECTOR",
                    help="The same selection, as run and check take it.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action is None:
        parser.print_help()
        raise SystemExit(2)
    if args.models:
        if args.selector and args.selector != args.models:
            from neuroatlas.cli import UsageError

            raise UsageError(f"two selections: {args.selector} and -m {args.models}\n"
                             f"fix: neuroatlas models {args.action} {args.models}")
        args.selector = args.models
    if args.action == "download" and not args.selector:
        from neuroatlas.cli import UsageError

        raise UsageError("models download needs a selection: which checkpoints to fetch\n"
                         "fix: neuroatlas models download all_fm\n"
                         "fix: neuroatlas list aliases")
    status = {"status": cmd_status, "download": cmd_download}[args.action](args)
    if status:
        raise SystemExit(status)
