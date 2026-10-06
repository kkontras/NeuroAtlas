"""``neuroatlas data`` -- is each dataset here, how to get it, and build its cache.

    neuroatlas data status   [benchmark|dataset|domain ...]  [-v] [--format F]
    neuroatlas data download <dataset ...> [--dry-run] [--mirror aws] [--keep-archive]
    neuroatlas data prepare  <dataset> [--dry-run] [--set KEY=VALUE] [--shard K/N]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from neuroatlas.cli import Parser, _msg
from neuroatlas.cli._table import add_format_arg, render

DOMAINS = ("epilepsy", "sleep", "brain_age", "bci", "all")


def _split(targets: List[str]) -> List[str]:
    """``a,b c`` -> [a, b, c]: commas work here as in every other selection."""
    return [t.strip() for arg in targets for t in arg.split(",") if t.strip()]


def expand_targets(targets: List[str]) -> List[str]:
    """Benchmarks, domains and slugs -> dataset slugs, in order, deduplicated."""
    from neuroatlas import catalog
    from neuroatlas.benchmarking_helpers.registry.discovery import dataset_specs

    specs = {s.slug: s for s in dataset_specs()}
    out: List[str] = []
    for target in _split(targets) or ["all"]:
        if target in catalog.catalog():
            slugs = [e.slug for e in catalog.load(target).datasets]
        elif target in DOMAINS:
            slugs = sorted({e.slug for b in catalog.catalog().values() for e in b.datasets
                            if target == "all" or b.domain == target})
        elif target in specs:
            slugs = [target]
        else:
            import difflib

            close = difflib.get_close_matches(target, [*catalog.catalog(), *DOMAINS, *specs], n=1)
            hint = f" (did you mean {close[0]}?)" if close else ""
            raise catalog.CatalogError(
                f"{target!r} is not a benchmark, domain or dataset{hint}\n"
                f"fix: neuroatlas list benchmarks, or neuroatlas list datasets",
                suggest={target: close[0]} if close else None)
        out.extend(s for s in slugs if s not in out)
    return out


def _data_rel(path: Optional[Path]) -> str:
    """Show paths under the data root as $DATA/..., defined under the table."""
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


def _row(st, verbose: bool, machine: bool) -> Dict[str, object]:
    row: Dict[str, object] = {"dataset": st.slug, "access": st.access, "state": st.state,
                              "download": st.download,
                              "map": "yes" if st.channel_map else "no"}
    if machine:
        # Machine-readable output: absolute paths, the key that moves the
        # folder, and every note -- nothing a script would have to re-derive.
        row.update({"path": str(st.path) if st.path else None, "path_key": st.path_key,
                    "setting": st.setting, "n_files": st.n_files, "expected": st.expected,
                    "found": st.found, "note": "; ".join(st.notes) or None})
    elif verbose:
        row.update({"path": _data_rel(st.path), "key": st.setting or "-"})
    return row


def cmd_status(args) -> int:
    from neuroatlas import config, data

    slugs = expand_targets(args.targets)
    machine = args.format in ("json", "csv")
    rows, notes, missing, tokenless = [], {}, [], []
    for slug in slugs:
        st = data.status(slug)
        rows.append(_row(st, args.verbose, machine))
        lines = list(st.notes)
        if args.format == "table" and data.NO_NSRR_TOKEN in lines:
            # one line under the table for all of them, not one per row
            lines.remove(data.NO_NSRR_TOKEN)
            tokenless.append(slug)
        if lines:
            notes[len(rows) - 1] = lines
        if not st.found and st.path is not None:
            missing.append(st)
    columns = ["dataset", "access", "state", "download", "map"]
    if machine:
        columns += ["path", "path_key", "setting", "n_files", "expected", "found", "note"]
    elif args.verbose:
        columns += ["path", "key"]
    # The renderer turns the notes into the table's indented lines, or into the
    # `note` field of every row in machine formats.
    render(rows, [c for c in columns if c != "note"], args.format, notes)
    if args.format != "table":
        return 0
    counts = Counter(str(r["state"]).split(" (")[0] for r in rows)
    if len(rows) > 1:
        print("\n" + _msg.counts(len(rows), "datasets", counts.most_common()))
    # Explain only what is not obvious: "found" needs no legend (-v shows all).
    explain = [s for s in counts if s in data.STATES and (args.verbose or s != "found")]
    for state in explain:
        print(f"  {state:<14} {data.STATES[state]}")
    root = config.get("data_root")
    if args.verbose:
        print("  map: yes      a channel map ships for it (configs/channel_maps/<dataset>.yaml)")
    if missing and not args.verbose:
        # where each was looked for, and the setting that points it at a copy
        print("\nnot found here (to use your own copy: neuroatlas config set <setting> DIR):")
        width = max(len(st.slug) for st in missing)
        places = [_data_rel(st.path) for st in missing]
        wide = max(len(w) for w in places)
        for st, where in zip(missing, places):
            key = (st.setting or ("$MNE_DATA (export MNE_DATA=DIR moves it)"
                                  if st.kind == "moabb" else ""))
            print(f"  {st.slug:<{width}}  {where:<{wide}}  {key}".rstrip())
    shown = [_data_rel(st.path) for st in missing] if not args.verbose else []
    if args.verbose or any(w.startswith("$DATA") for w in shown):
        print(f"$DATA = {root if root else '(no data root: neuroatlas config init --data-root DIR)'}")
    if tokenless:
        _msg.note(f"no NSRR token: {', '.join(tokenless)} cannot be downloaded yet",
                  "neuroatlas config token nsrr")
    return 0


def cmd_download(args) -> int:
    from neuroatlas import data

    status = 0
    for slug in expand_targets(args.datasets):
        plan = data.plan_download(slug, mirror=args.mirror, keep_archive=args.keep_archive)
        if args.dry_run:
            if plan.handler in ("manual", "internal"):
                for line in data.describe_manual(plan):
                    print(line)
                continue
            why = data.refusal(plan)
            for line in data.describe(plan):
                print(line)
            if why:
                _msg.error(f"{slug}: {why}")
                status = max(status, 2)
            else:
                size = f", about {plan.size_gb:g} GB" if plan.size_gb else ""
                print(f"{slug}: dry run{size}; nothing was transferred")
            continue
        status = max(status, data.run_download(plan))
    return status


def cmd_prepare(args) -> int:
    from neuroatlas import data
    from neuroatlas.entrypoints import prepare

    if args.list:
        prepare.main(["--list"])
        return 0
    if not args.dataset:
        _msg.error("name a dataset", "neuroatlas data prepare --list names those with a "
                                     "build step")
        return 2
    slug = expand_targets([args.dataset])
    if len(slug) != 1:
        _msg.error(f"data prepare takes one dataset; {args.dataset!r} is {len(slug)} of them")
        return 2
    slug = slug[0]
    acq = data.acquisition(slug)
    if prepare.builder_for(slug) is None:
        if acq.get("prepared_only"):
            _msg.error(f"{slug} cannot be prepared here: " + data.manifest_text(acq["prepared_only"]))
            return 2
        print(f"nothing to prepare: {slug} reads its raw data directly (no build step)")
        return 0

    raw = data.status(slug)
    if raw.kind == "moabb":
        # The prepared state hides the raw one; ask about the raw download.
        probe = data.DatasetStatus(slug, raw.kind, raw.access, raw.handler, "missing")
        data._nemar_state(slug, acq, probe)
        raw = probe
    if not raw.found and not args.dry_run:
        # Refuse before the builder runs: a MOABB builder would otherwise
        # download the corpus itself, as a side effect of `prepare`.
        _msg.error(f"{slug}: no raw data to prepare from ({raw.state}: {_data_rel(raw.path)})",
                   f"neuroatlas data download {slug}"
                   + (f", or neuroatlas config set {raw.setting} DIR" if raw.setting else ""))
        return 2
    # Quiet MNE's per-file INFO lines (O-8); the builder prints its own
    # progress. MNE reads the variable when imported, which status() may
    # already have done, so set the live level too.
    os.environ.setdefault("MNE_LOGGING_LEVEL", "WARNING")
    if "mne" in sys.modules:
        sys.modules["mne"].set_log_level(os.environ["MNE_LOGGING_LEVEL"])
    argv = ["--dataset", slug]
    for pair in args.overrides:
        argv += ["--set", pair]
    if args.dest:
        argv += ["--dest", args.dest]
    if args.shard:
        argv += ["--shard", args.shard]
    if args.dry_run:
        argv.append("--dry-run")
        if not raw.found:
            _msg.warning(f"{slug}: no raw data yet ({raw.state}); the real run refuses",
                         f"neuroatlas data download {slug}")
    prepare.main(argv)
    if not args.dry_run:
        after = data.status(slug)
        print(f"{slug}: {after.state}" + (f" ({after.path})" if after.path else ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = Parser(prog="neuroatlas data", description=__doc__)
    sub = parser.add_subparsers(dest="action", metavar="<action>")
    st = sub.add_parser("status", help="Is each dataset of a benchmark on this machine?",
                        description="Is each dataset on this machine, where was it looked "
                                    "for, which setting moves it, and how `data download` "
                                    "would get it. Takes benchmarks, domains (epilepsy, "
                                    "sleep, brain_age, bci, all) or dataset names, "
                                    "space- or comma-separated; default all.")
    st.add_argument("targets", nargs="*")
    st.add_argument("-v", "--verbose", action="store_true",
                    help="Show every path, and the setting that moves it, as columns.")
    add_format_arg(st)
    dl = sub.add_parser("download", help="Fetch a dataset, or print how to.",
                        description="Fetch datasets into the folder their reader expects "
                                    "(under the data root; MOABB ones under $MNE_DATA). "
                                    "Credentialed and manual ones print where to get them. "
                                    "Exit 0 done, 1 a transfer failed, 2 refused.")
    dl.add_argument("datasets", nargs="+")
    dl.add_argument("--dry-run", action="store_true",
                    help="Print the plan and run every check (token, disk, tools); "
                         "transfer nothing.")
    dl.add_argument("--mirror", choices=("physionet", "aws"), default="physionet",
                    help="PhysioNet datasets: physionet.org (default) or PhysioNet's "
                         "open-data copy on AWS (anonymous, usually much faster).")
    dl.add_argument("--keep-archive", action="store_true",
                    help="Keep a downloaded .zip after unpacking it (default: delete it).")
    pr = sub.add_parser("prepare", help="Build a dataset's optional cache (no benchmark needs one).",
                        description="Build a dataset's optional prepared file: a speed-up "
                                    "for a few epilepsy cohorts, or dataio/bci.py's pickle "
                                    "for five MOABB motor-imagery ones, which `embed` and "
                                    "`run` do not read. Refuses until the raw data is "
                                    "there (`data download` first), and writes under "
                                    "<cache root>/prepared.")
    pr.add_argument("dataset", nargs="?")
    pr.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="Override a dataset setting, e.g. raw_root=/data.")
    pr.add_argument("--dest", default=None,
                    help="Where the prepared file goes. Default: under <cache root>/prepared.")
    pr.add_argument("--shard", default=None, metavar="K/N",
                    help="Build shard K of N, for scheduler fan-out (epilepsy builders).")
    pr.add_argument("--dry-run", action="store_true", help="Print the builder command; run nothing.")
    pr.add_argument("--list", action="store_true",
                    help="Which datasets have a build step, and whether it is required.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action is None:
        parser.print_help()
        raise SystemExit(2)
    status = {"status": cmd_status, "download": cmd_download,
              "prepare": cmd_prepare}[args.action](args)
    if status:
        raise SystemExit(status)
