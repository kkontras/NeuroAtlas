"""Obtain a dataset's raw data -- the same code as ``neuroatlas data download``.

Kept so the commands in ``run/default_runs.sh`` and older notes still work,
but it no longer has a download path of its own: the plan, the destination
and the transfer all come from :mod:`neuroatlas.data`, so ``fetch`` and
``data download`` cannot disagree about where a corpus goes or whether it
can be fetched. (Its own path used to save a PhysioNet directory listing
into one file, in a folder no reader looks at.)

Nothing is transferred unless ``--download`` is given. ``--dest DIR`` is the
data root to plan against (default: the configured one); the corpus lands
in the sub-folder its reader expects, as with ``data download``.

Usage
-----
    neuroatlas fetch --dataset siena                    # = data download siena --dry-run
    neuroatlas fetch --dataset siena --download         # = data download siena
    neuroatlas fetch --list
"""
from __future__ import annotations

import argparse
import contextlib
import os
import subprocess  # noqa: F401 -- kept: tests assert planning never calls it
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from neuroatlas import config as user_config
from neuroatlas.cli import ErrorParser, LinesFormatter


# How each acquisition.kind is obtained. `auto` means `data download` (and so
# this verb) can do it; the rest need a human, and get instructions instead.
KIND_HELP = {
    "zenodo": ("auto", "public Zenodo record"),
    "physionet": ("auto", "public PhysioNet project"),
    "url": ("auto", "public files at fixed addresses"),
    "moabb": ("auto", "MOABB downloads it into $MNE_DATA"),
    "mendeley": ("manual", "public Mendeley Data record; no download API here"),
    "figshare": ("manual", "public figshare record; no download API here"),
    "nsrr": ("manual", "NSRR: needs an approved data-access request; "
                       "`neuroatlas data download` runs the `nsrr` gem with your token"),
    "tuh": ("manual", "TUH EEG Corpus: needs a signed data use agreement; "
                      "credentials arrive by email"),
    "manual": ("manual", "a person has to fetch it; see the landing page"),
    "internal": ("internal", "not publicly licensed; obtain from the authors"),
}

_AUTO_HANDLERS = ("physionet", "zenodo", "url", "moabb")


@contextlib.contextmanager
def _data_root(dest: Optional[Path]) -> Iterator[None]:
    """Plan against ``dest`` as the data root, for this call only."""
    if dest is None:
        yield
        return
    old = os.environ.get("EEG_DATA_ROOT")
    os.environ["EEG_DATA_ROOT"] = str(Path(dest).expanduser().absolute())
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("EEG_DATA_ROOT", None)
        else:
            os.environ["EEG_DATA_ROOT"] = old


def _manifest(slug: str) -> Dict[str, Any]:
    from neuroatlas.benchmarking_helpers.registry.manifest import load_manifest

    return load_manifest(slug)


def plan(slug: str, dest: Optional[Path] = None) -> Dict[str, Any]:
    """What `data download` would do for ``slug``, as a dict for `render`."""
    from neuroatlas import data

    manifest = _manifest(slug)
    acq = manifest.get("acquisition") or {}
    kind = acq.get("kind") or "manual"
    with _data_root(dest):
        dl = data.plan_download(slug)
        why_not = data.refusal(dl) if dl.handler not in ("manual", "internal") else None
    mode, why = KIND_HELP.get(kind, ("manual", kind))
    if dl.handler not in _AUTO_HANDLERS:
        mode = "internal" if dl.handler == "internal" else "manual"
        why = dl.message or why
    url = None
    if dl.fetches:
        url = dl.fetches[0].url + (f" (+{len(dl.fetches) - 1} more)" if len(dl.fetches) > 1 else "")
    elif dl.listing:
        url = dl.listing
    elif dl.commands:
        url = dl.commands[0][-1] if dl.commands[0][0] == "wget" else " ".join(dl.commands[0])
    return {
        "slug": slug,
        "name": manifest.get("name"),
        "kind": kind,
        "mode": mode,
        "why": why,
        "url": url,
        "upstream": acq.get("upstream"),
        "size_gb": acq.get("size_gb"),
        "checksum": acq.get("checksum"),
        "dest": str(dl.dest) if dl.dest else None,
        "data_root": str(dest) if dest else None,
        "note": acq.get("note"),
        "refusal": why_not,
    }


def render(p: Dict[str, Any]) -> str:
    size = f"{p['size_gb']} GB" if p["size_gb"] else "size not recorded"
    lines = [
        f"{p['slug']}: {p['name']}",
        f"  source       {p['kind']} ({p['why']})",
        f"  size         {size}",
        f"  destination  {p['dest'] or '-'}",
    ]
    if p["url"]:
        lines.append(f"  url          {p['url']}")
    if p["upstream"]:
        lines.append(f"  landing      {p['upstream']}")
    if p["note"]:
        lines.append(f"  about        {' '.join(str(p['note']).split())}")
    if p["checksum"] in (None, "TBD") and p["kind"] not in ("zenodo", "moabb"):
        lines.append("  checksum     not recorded")
    if p["mode"] == "auto" and p["url"]:
        if p.get("refusal"):
            from neuroatlas.cli import _msg

            _, what, fixes = _msg.split(str(p["refusal"]))
            lines.append(f"  needs first: {' '.join(' '.join(what).split())}")
            lines += [f"    fix: {f}" for f in fixes]
        lines.append(f"  to fetch it: add --download (the same as "
                     f"`neuroatlas data download {p['slug']}`)")
    else:
        lines.append(f"  not downloadable here: {p['why']}")
    return "\n".join(lines)


def download(p: Dict[str, Any]) -> int:
    """Run `data download`'s own code for the plan's dataset."""
    from neuroatlas import data

    if p["mode"] != "auto" or not p["url"]:
        from neuroatlas.cli import _msg

        print(render(p))
        _msg.error(f"{p['slug']} cannot be downloaded here (the lines above say how to get it)",
                   f"neuroatlas data download {p['slug']} --dry-run (says how to get it)")
        return 2
    from neuroatlas.cli.data import _download

    root = Path(p["data_root"]) if p.get("data_root") else None
    with _data_root(root):
        # data download's own lines: header, live line, result line
        return _download([data.plan_download(p["slug"])])


def build_parser(argv: Optional[List[str]] = None) -> argparse.ArgumentParser:
    from neuroatlas.entrypoints import _help

    parser = ErrorParser(
        prog="neuroatlas fetch",
        description=(
            "Download a dataset, or print how to get it. It is another name for `neuroatlas "
            "data download`, and it only downloads when given --download."),
        formatter_class=LinesFormatter,
        epilog=_help.build_epilog(argv, show_models=False),
    )
    parser.add_argument("--dataset", help="Dataset name, as listed by `neuroatlas list "
                                           "datasets`.")
    parser.add_argument("--dest", default=None, metavar="DIR",
                        help="Data folder to download into (default: the data_root setting). "
                             "The dataset goes to its own sub-folder.")
    parser.add_argument("--download", action="store_true",
                        help="Download the data. Without it, only the plan is printed.")
    parser.add_argument("--list", action="store_true",
                        help="List the paper's datasets and how to get each one.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    user_config.apply_to_environ()
    argv = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser(argv).parse_args(argv)
    dest = Path(args.dest) if args.dest else None

    if args.list:
        from neuroatlas.benchmarking_helpers.registry.discovery import dataset_specs

        rows = [plan(s.slug, dest) for s in sorted(dataset_specs(), key=lambda x: x.slug)
                if (s.manifest or {}).get("paper_dataset")]
        auto = [r for r in rows if r["mode"] == "auto" and r["url"]]
        from neuroatlas import data

        print(f"{len(auto)} of {len(rows)} paper datasets are downloadable "
              f"(`neuroatlas data download DATASET`).")
        print()
        table = []
        for r in rows:
            acq = (_manifest(r["slug"]).get("acquisition") or {})
            handler = "refused" if acq.get("unusable_download") else \
                data.HANDLERS.get(r["kind"], "manual")
            how = ("downloadable" if (r["mode"] == "auto" and r["url"]) else
                   data.download_word(r["kind"], handler))
            if r["mode"] == "auto" and r.get("refusal"):
                how += (f" (`neuroatlas data download {r['slug']} --dry-run` says what it "
                        f"needs first)")
            table.append((r["slug"], data.host_word(r["kind"], acq), how))
        wide = [max(len(t[i]) for t in table + [("dataset", "host", "")]) for i in (0, 1)]
        print(f"{'dataset':<{wide[0]}}  {'host':<{wide[1]}}  download")
        for slug, host, how in table:
            print(f"{slug:<{wide[0]}}  {host:<{wide[1]}}  {how}")
        return

    if not args.dataset:
        build_parser(argv).error("--dataset is required (or use --list)")

    p = plan(args.dataset, dest)
    if not args.download:
        print(render(p))
        return
    raise SystemExit(download(p))


if __name__ == "__main__":
    main()
