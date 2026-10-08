"""``neuroatlas data`` -- is each dataset here; download it; build its optional prepared file.

    neuroatlas data status   [benchmark|dataset|domain ...]  [-v] [--format F]
    neuroatlas data download <dataset ...> [--dry-run] [--mirror aws] [--keep-archive] [--first N]
    neuroatlas data prepare  <dataset> [--dry-run] [--set KEY=VALUE] [--shard K/N]
"""
from __future__ import annotations

import argparse
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
                f"fix: neuroatlas list benchmarks\n"
                f"fix: neuroatlas list datasets",
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
    row: Dict[str, object] = {"dataset": st.slug, "access": st.access, "host": st.host,
                              "state": st.state, "download": st.download,
                              "map": "yes" if st.channel_map else "no"}
    if machine:
        # Machine-readable output: absolute paths, the key that moves the
        # folder, and every note -- nothing a script would have to re-derive.
        row.update({"path": str(st.path) if st.path else None, "path_key": st.path_key,
                    "setting": st.setting, "n_files": st.n_files, "expected": st.expected,
                    "found": st.found, "note": "; ".join(st.notes) or None})
    elif verbose:
        row.update({"path": _data_rel(st.path), "setting": st.setting or "-"})
    return row


#: The `map` column (-v, and every machine format) explained.
MAP_LEGEND = ("channel map", "yes: a channel map ships for this dataset (which of its channels "
                             "each model family gets, or which families it rules out)")


def cmd_status(args) -> int:
    from neuroatlas import config, data, progress

    slugs = expand_targets(args.targets)
    machine = args.format in ("json", "csv")
    rows, notes, missing, tokenless = [], {}, [], []
    # the command's live line (stderr, a terminal only): "checking 40% (2/5 datasets)"
    item = progress.current().phase("checking", total=len(slugs), unit="datasets")
    for slug in slugs:
        item.update(note=slug)
        st = data.status(slug)
        item.update(advance=1)
        rows.append(_row(st, args.verbose, machine))
        lines = list(st.notes)
        if args.format == "table" and data.NO_NSRR_TOKEN in lines:
            # one line under the table for all of them, not one per row
            lines.remove(data.NO_NSRR_TOKEN)
            tokenless.append(slug)
        if not st.found and st.path is not None:
            missing.append(st)
            if not any(line.strip().startswith("fix:") for line in lines):
                # how to get it, under its row (the first remedy; with one
                # dataset, every one, the setting for a copy elsewhere too)
                from neuroatlas.check import data_fix

                fixes = _msg.split(_msg.compose("-", data_fix(slug, st)))[2]
                if slug in tokenless:
                    # the NSRR token: one note under the table for all of them
                    fixes = [f for f in fixes if not f.startswith("neuroatlas config token")]
                lines += [f"  fix: {f}" for f in (fixes if len(slugs) == 1 else fixes[:1])]
        if lines:
            notes[len(rows) - 1] = lines
    # the channel map column only with -v and in the machine formats: it
    # says nothing a first look needs
    columns = ["dataset", "host", "state", "download"]
    if machine:
        columns += ["map", "access", "path", "path_key", "setting", "n_files", "expected",
                    "found", "note"]
    elif args.verbose:
        columns += ["map", "path", "setting"]
    # The renderer turns the notes into the table's indented lines, or into the
    # `note` field of every row in machine formats.
    render(rows, [c for c in columns if c != "note"], args.format, notes,
           labels={"map": "channel map"})
    if args.format != "table":
        return 0
    counts = Counter(str(r["state"]).split(" (")[0] for r in rows)
    if len(rows) > 1:
        print("\n" + _msg.counts(len(rows), "datasets", counts.most_common()))
    # Explain only what is not obvious: "found" and "downloadable" need no
    # legend (-v explains every value shown).
    states = [(s, data.STATES[s]) for s in counts
              if s in data.STATES and (args.verbose or s != "found")]
    words = [(w, data.DOWNLOAD_MEANINGS[w]) for w in dict.fromkeys(r["download"] for r in rows)
             if w in data.DOWNLOAD_MEANINGS and (args.verbose or w != "downloadable")]
    explained = states + words + ([MAP_LEGEND] if args.verbose else [])
    print("\n".join(_msg.legend(explained)))
    root = config.get("data_root")
    if missing and not args.verbose:
        # where each was looked for, and the setting that points it at a copy
        print("\nnot here; where each was looked for, and the setting that points it at a "
              "copy elsewhere:")
        width = max(len(st.slug) for st in missing)
        places = [_data_rel(st.path) for st in missing]
        wide = max(len(w) for w in places)
        for st, where in zip(missing, places):
            key = st.setting or ("MNE_DATA (an environment variable)"
                                 if st.kind == "moabb" else "")
            print(f"  {st.slug:<{width}}  {where:<{wide}}  {key}".rstrip())
    shown = [_data_rel(st.path) for st in missing] if not args.verbose else []
    if args.verbose or any(w.startswith("$DATA") for w in shown):
        print("$DATA = " + (str(root) if root else
                            "(no data root is set: neuroatlas config set data_root DIR)"))
    if missing:
        settable = [st for st in missing if st.setting]
        moabb = [st for st in missing if st.kind == "moabb" and not st.setting]
        if len(settable) > 1:
            # the setting of the first one a folder holds (a preprocessed_path
            # names a FILE); the table above names each dataset's
            example = next((st for st in settable if not st.setting.endswith("_path")),
                           settable[0])
            what = "FILE" if example.setting.endswith("_path") else "DIR"
            _msg.note("a dataset you already have elsewhere: point its setting (listed "
                      "above) at it", f"neuroatlas config set {example.setting} {what}")
        mne = config.mne_data()
        if moabb and mne.origin != "env":
            _msg.note(f"MOABB datasets are read from one folder, {mne.value}",
                      "export MNE_DATA=DIR (to keep them elsewhere)")
    if tokenless:
        _msg.note(f"no NSRR token: `data download` needs one to fetch "
                  f"{', '.join(tokenless)}", "neuroatlas config token nsrr")
    return 0


def cmd_download(args) -> int:
    from neuroatlas import data

    plans = [data.plan_download(slug, mirror=args.mirror, keep_archive=args.keep_archive,
                                first=getattr(args, "first", None))
             for slug in expand_targets(args.datasets)]
    if not args.dry_run:
        return _download(plans)
    status = 0
    for plan in plans:
        slug = plan.slug
        if plan.handler in ("manual", "internal"):
            for line in data.describe_manual(plan):
                print(line)
            continue
        why = data.refusal(plan)
        for line in data.describe(plan):
            data.say(line)             # a `note:` line to stderr, the plan to stdout
        if why:
            _msg.error(f"{slug}: {why}")
            status = max(status, 2)
        else:
            from neuroatlas import progress

            size = f", about {progress.size(plan.size_gb * 1e9)}" if plan.size_gb else ""
            print(f"{slug}: dry run{size}; nothing was transferred")
    return status


def _download(plans) -> int:
    """The downloads, as every long command shows its work (neuroatlas.progress):
    a header; per dataset a start line, a live line while it transfers (one
    line per file under it) and a result line. Datasets fetched by hand print
    how, as they come."""
    from neuroatlas import data, progress

    status = 0
    auto = [p for p in plans if p.handler not in ("manual", "internal")]
    if auto:
        progress.say(f"downloading {_msg.plural(len(auto), 'dataset')}: "
                     f"{', '.join(p.slug for p in auto)}")
    done = 0
    for plan in plans:
        if plan.handler in ("manual", "internal"):
            for line in data.describe_manual(plan):
                print(line)
            continue
        done += 1
        label = f"[{done}/{len(auto)}] {plan.slug}"
        why = data.refusal(plan)
        if why:
            _msg.error(f"{plan.slug}: {why}")
            progress.say(progress.result_line(done, len(auto), plan.slug, "refused"))
            status = max(status, 2)
            continue
        progress.say(f"{label}: downloading {data.source_text(plan)}".rstrip())
        with progress.Progress(label, verb="downloading", start_line_off_tty=False) as item:
            rc = data.run_download(plan)
        progress.say(progress.result_line(done, len(auto), plan.slug,
                                          data.result_text(rc, item.counts), item.seconds))
        status = max(status, rc)
    return status


def cmd_prepare(args) -> int:
    from neuroatlas import data
    from neuroatlas.entrypoints import prepare

    if args.list:
        prepare.main(["--list"])
        return 0
    if not args.dataset:
        _msg.error("name a dataset", "neuroatlas data prepare --list (the datasets with a "
                                     "build step)")
        return 2
    slug = expand_targets([args.dataset])
    if len(slug) != 1:
        _msg.error(f"data prepare takes one dataset; {args.dataset!r} is {len(slug)} of them",
                   f"neuroatlas data prepare {slug[0]}" if slug else None)
        return 2
    slug = slug[0]
    acq = data.acquisition(slug)
    pipeline = (data._manifest(slug).get("pipeline") or {})
    if prepare.builder_for(slug) is None or pipeline.get("required", True) is False:
        if acq.get("prepared_only"):
            _, what, fixes = _msg.split(data.manifest_text(acq["prepared_only"]))
            _msg.error(f"{slug} has no build step: " + " ".join(what),
                       "\n".join(fixes) or None)
            return 2
        print(f"nothing to prepare: {slug} is read from its "
              + ("MOABB download" if acq.get("kind") == "moabb" else "raw data")
              + " (no build step)")
        return 0

    raw = data.status(slug, raw_only=True)
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
                   + (f"\nfix: neuroatlas config set {raw.setting} DIR (a copy elsewhere)"
                      if raw.setting else ""))
        return 2
    # MNE's per-file INFO lines (O-8) are hidden and counted unless -v, for
    # every command (neuroatlas.quiet.route_libraries); the builder prints its
    # own progress.
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
            _msg.warning(f"{slug}: its raw data is not here ({raw.state}); `data prepare` "
                         f"without --dry-run refuses until it is",
                         f"neuroatlas data download {slug}")
    prepare.main(argv)
    if not args.dry_run:
        after = data.status(slug)
        print(f"{slug}: {after.state}" + (f" ({after.path})" if after.path else ""))
    return 0


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of recordings, at least 1; "
                                         f"got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = Parser(prog="neuroatlas data",
                    description="Check which datasets are on this machine, download them, and "
                                "build optional faster-to-read copies.")
    sub = parser.add_subparsers(dest="action", metavar="<action>")
    st = sub.add_parser("status", help="Show which datasets are on this machine.",
                        description="Show which datasets are on this machine, where they were "
                                    "looked for and how to get the missing ones.")
    st.add_argument("targets", nargs="*",
                    help="Benchmarks, domains (epilepsy, sleep, brain_age, bci or all) or "
                         "dataset names, separated by spaces or commas (default: all).")
    st.add_argument("-v", "--verbose", action="store_true",
                    help="Add columns for each path, the setting that moves it and the "
                         "channel map, and explain every value.")
    add_format_arg(st)
    dl = sub.add_parser("download", help="Download datasets, or print how to get them.",
                        description="Download datasets into the folder each one is read from. "
                                    "MOABB datasets go to $MNE_DATA. For datasets that need an "
                                    "account or a request, it prints where to get them.")
    dl.add_argument("datasets", nargs="+",
                    help="Dataset names, as listed by `neuroatlas list datasets`.")
    dl.add_argument("--dry-run", action="store_true",
                    help="Print the plan and run every check (token, disk space, tools), but "
                         "download nothing.")
    dl.add_argument("--mirror", choices=("physionet", "aws"), default="physionet",
                    help="Where to get PhysioNet datasets (default: physionet). aws is "
                         "PhysioNet's open-data copy on AWS, which is usually much faster.")
    dl.add_argument("--keep-archive", action="store_true",
                    help="Keep downloaded .zip files after unpacking them.")
    dl.add_argument("--first", type=_positive, default=None, metavar="N",
                    help="Download only the first N recordings, for a quick test. `check` "
                         "works on them, but `run` needs the whole dataset. Works for "
                         "PhysioNet, NSRR and file-by-file Zenodo downloads.")
    pr = sub.add_parser("prepare", help="Build the files a bci_cognitive dataset is read from, "
                                        "or a faster copy of an epilepsy dataset.",
                        description="Build the two files that dreamer_valence, dreamer_arousal, "
                                    "eegmat or arithmetic_task are read from. Run it once before "
                                    "`run`. For bonn, epilepsiae, sz1, tuab and tusz it builds an "
                                    "optional faster-to-read copy. Download the raw data first. "
                                    "The files go to `<cache root>/prepared`.")
    pr.add_argument("dataset", nargs="?",
                    help="Dataset name. --list shows the datasets with a build step.")
    pr.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="Change a dataset setting, as in raw_root=/data.")
    pr.add_argument("--dest", default=None, metavar="PATH",
                    help="Where to write the copy (default: `<cache root>/prepared`).")
    pr.add_argument("--shard", default=None, metavar="K/N",
                    help="Build only shard K of N, to split the work across cluster jobs.")
    pr.add_argument("--dry-run", action="store_true",
                    help="Print what would be built, without building it.")
    pr.add_argument("--list", action="store_true",
                    help="List the datasets that have a build step.")
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
