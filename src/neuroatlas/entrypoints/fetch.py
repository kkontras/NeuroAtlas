"""Obtain a dataset's raw corpus — the first thing an outside user runs.

This is the answer to "I cloned the repo, now what?".  Before it existed the
answer was eight bash scripts under ``run/`` covering eight of the 32 paper
cohorts, each hardcoding a Zenodo record and a destination, and nothing at all
for the other 24 — so the honest instruction was "ask us".

It dispatches on ``acquisition.kind`` in the dataset manifest, which is the
single place those facts live.  For an open record it can do the download; for
a credentialed corpus (NSRR, TUH, the internal cohorts) it prints exactly what
to request and where to put it, which is the useful thing it *can* do without
your credentials.

Nothing is transferred unless ``--download`` is given.  A plain
``fetch --dataset X`` prints the plan: where it would come from, how large, and
where it would land.  That default is deliberate — these are multi-gigabyte
pulls from other people's servers.

Usage
-----
    python -m neuroatlas.entrypoints.fetch --dataset siena
    python -m neuroatlas.entrypoints.fetch --dataset siena --download --dest /data/eeg
    python -m neuroatlas.entrypoints.fetch --list
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# How each acquisition.kind is obtained. `auto` means this verb can do it;
# the rest need a human with credentials, and get instructions instead.
KIND_HELP = {
    "zenodo": ("auto", "public Zenodo record"),
    "physionet": ("auto", "public PhysioNet project"),
    "mendeley": ("auto", "public Mendeley Data record"),
    "figshare": ("auto", "public figshare record"),
    "moabb": ("auto", "downloaded on first use by MOABB itself"),
    "nsrr": ("manual", "NSRR — needs an approved data-access request and the "
                       "`nsrr` ruby gem with your token"),
    "tuh": ("manual", "TUH EEG Corpus — needs a signed data use agreement; "
                      "credentials arrive by email"),
    "manual": ("manual", "no download API — see acquisition.upstream"),
    "internal": ("internal", "not publicly licensed; obtain from the authors"),
}


def _manifest(slug: str) -> Dict[str, Any]:
    from neuroatlas.benchmarking_helpers.registry.manifest import load_manifest

    return load_manifest(slug)


def _url_for(kind: str, acq: Dict[str, Any]) -> Optional[str]:
    ref, name = acq.get("ref"), acq.get("file")
    if not ref:
        return None
    if kind == "zenodo":
        if name and not name.startswith("("):
            return f"https://zenodo.org/api/records/{ref}/files/{name}/content"
        return f"https://zenodo.org/api/records/{ref}"
    if kind == "physionet":
        return f"https://physionet.org/files/{ref}/"
    if kind == "mendeley":
        return f"https://data.mendeley.com/datasets/{str(ref).split('/')[-1]}"
    if kind == "figshare":
        return f"https://figshare.com/articles/dataset/{ref}"
    return None


def plan(slug: str, dest: Path) -> Dict[str, Any]:
    manifest = _manifest(slug)
    acq = manifest.get("acquisition") or {}
    kind = acq.get("kind") or "manual"
    mode, why = KIND_HELP.get(kind, ("manual", kind))
    return {
        "slug": slug,
        "name": manifest.get("name"),
        "kind": kind,
        "mode": mode,
        "why": why,
        "url": _url_for(kind, acq),
        "upstream": acq.get("upstream"),
        "size_gb": acq.get("size_gb"),
        "checksum": acq.get("checksum"),
        "dest": str(dest / slug),
        "note": acq.get("note"),
    }


def render(p: Dict[str, Any]) -> str:
    size = f"{p['size_gb']} GB" if p["size_gb"] else "size not recorded"
    lines = [
        f"{p['slug']} — {p['name']}",
        f"  source   : {p['kind']} ({p['why']})",
        f"  size     : {size}",
        f"  destination: {p['dest']}",
    ]
    if p["url"]:
        lines.append(f"  url      : {p['url']}")
    if p["upstream"]:
        lines.append(f"  landing  : {p['upstream']}")
    if p["note"]:
        lines.append(f"  note     : {' '.join(str(p['note']).split())}")
    if p["checksum"] in (None, "TBD"):
        lines.append("  checksum : NOT RECORDED — nothing verifies this download")
    if p["mode"] == "auto" and p["url"]:
        lines.append("  -> add --download to fetch it")
    else:
        lines.append(f"  -> cannot be downloaded here: {p['why']}")
    return "\n".join(lines)


def download(p: Dict[str, Any]) -> int:
    if p["mode"] != "auto" or not p["url"]:
        print(render(p))
        print(f"\nerror: {p['slug']} cannot be fetched automatically.",
              file=sys.stderr)
        return 2
    dest = Path(p["dest"])
    dest.mkdir(parents=True, exist_ok=True)
    tool = shutil.which("wget") or shutil.which("curl")
    if tool is None:
        print("error: neither wget nor curl is on PATH", file=sys.stderr)
        return 3
    target = dest / (p["slug"] + Path(str(p["url"])).suffix or ".download")
    cmd = ([tool, "-c", "-O", str(target), p["url"]] if tool.endswith("wget")
           else [tool, "-L", "-C", "-", "-o", str(target), p["url"]])
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def build_parser(argv: Optional[List[str]] = None) -> argparse.ArgumentParser:
    from neuroatlas.entrypoints import _help

    parser = argparse.ArgumentParser(
        prog="python -m neuroatlas.entrypoints.fetch",
        description="Obtain a dataset's raw corpus, or say exactly how to.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_help.build_epilog(argv, show_models=False),
    )
    parser.add_argument("--dataset", help="DatasetSpec slug.")
    parser.add_argument("--dest", default=None,
                        help="Where the corpus should land. Default: $EEG_DATA_ROOT, "
                             "which is the variable every dataset default is written "
                             "against.")
    parser.add_argument("--download", action="store_true",
                        help="Actually transfer. Without it, the plan is printed and "
                             "nothing leaves or enters this machine.")
    parser.add_argument("--list", action="store_true",
                        help="Show every paper dataset and how it is obtained.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    import os

    argv = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser(argv).parse_args(argv)

    dest = Path(args.dest or os.environ.get("EEG_DATA_ROOT") or "data")

    if args.list:
        from neuroatlas.benchmarking_helpers.registry.discovery import dataset_specs

        rows = [plan(s.slug, dest) for s in sorted(dataset_specs(), key=lambda x: x.slug)
                if (s.manifest or {}).get("paper_dataset")]
        auto = [r for r in rows if r["mode"] == "auto" and r["url"]]
        no_ref = [r for r in rows if r["mode"] == "auto" and not r["url"]]
        print(f"{len(auto)} of {len(rows)} paper datasets can be fetched directly.")
        if no_ref:
            print(f"{len(no_ref)} more are public but have no acquisition.ref "
                  f"recorded: {', '.join(r['slug'] for r in no_ref)}")
        print()
        print(f"{'dataset':36s} {'source':10s} how")
        print("-" * 60)
        for r in rows:
            # "auto" with no url means the manifest has no acquisition.ref --
            # the source is public but we never wrote down which record. Saying
            # "auto" there would promise something this verb cannot do.
            how = ("automatic" if (r["mode"] == "auto" and r["url"])
                   else "NO REF" if r["mode"] == "auto" else r["mode"])
            print(f"{r['slug']:36s} {r['kind']:10s} {how}")
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
