"""``neuroatlas config`` -- write, change and inspect the settings file.

    neuroatlas config init --data-root DIR [--cache-root DIR] ...
    neuroatlas config show
    neuroatlas config set KEY VALUE        # data_root, or ucddb.data_root
    neuroatlas config unset KEY
    neuroatlas config path
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neuroatlas import _paths
from neuroatlas import config as cfg


def _dataset_key(text: str) -> Tuple[str, str, str]:
    """``ucddb.data_root=/mnt/ucddb`` -> ("ucddb", "data_root", "/mnt/ucddb")."""
    target, sep, value = text.partition("=")
    slug, dot, key = target.partition(".")
    if not (sep and dot and slug and key and value):
        raise argparse.ArgumentTypeError(
            f"expected DATASET.KEY=PATH, e.g. ucddb.data_root=/mnt/ucddb; got {text!r}")
    return slug, key, value


def _known_dataset(slug: str) -> bool:
    # Cheap check against the shipped dossiers; MOABB-generated specs have
    # none, so an unknown slug is a warning, not an error.
    return _paths.configs_dir("cohorts", slug).is_dir()


def _abs(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuroatlas config",
        description="Where NeuroAtlas finds your data and puts what it makes. "
                    f"The settings file is {cfg.config_path()}; set "
                    "$NEUROATLAS_HOME to keep it elsewhere.",
    )
    sub = parser.add_subparsers(dest="action", metavar="<action>")

    init = sub.add_parser("init", help="Write the settings file.",
                          description="Write the settings file. Only --data-root is "
                                      "required; every other folder has a default.")
    init.add_argument("--data-root", required=True, metavar="DIR",
                      help="Raw datasets, one sub-folder each.")
    for setting in cfg.SETTINGS.values():
        if setting.key == "data_root":
            continue
        default = setting.default()
        init.add_argument(f"--{setting.key.replace('_', '-')}", metavar="DIR",
                          help=f"{setting.help[0].upper()}{setting.help[1:]}. "
                               f"Default: {default if default else 'none'}.")
    init.add_argument("--dataset-path", action="append", default=[], type=_dataset_key,
                      metavar="DATASET.KEY=PATH",
                      help="A dataset stored outside its default sub-folder, e.g. "
                           "ucddb.data_root=/mnt/ucddb. Repeatable.")
    init.add_argument("--force", action="store_true",
                      help="Replace an existing settings file.")

    sub.add_parser("show", help="Print every setting, whether it exists, and where it came from.")

    set_ = sub.add_parser("set", help="Change one setting.",
                          description="Change one setting: a root such as cache_root, "
                                      "or a dataset path such as ucddb.data_root.")
    set_.add_argument("key")
    set_.add_argument("value")

    unset = sub.add_parser("unset", help="Remove one setting, returning it to its default.")
    unset.add_argument("key")

    sub.add_parser("path", help="Print where the settings file is.")
    return parser


# --------------------------------------------------------------------------

def _init(args: argparse.Namespace) -> int:
    path = cfg.config_path()
    if path.exists() and not args.force:
        print(f"error: {path} already exists. Add --force to replace it, or change "
              f"one setting with `neuroatlas config set KEY VALUE`.", file=sys.stderr)
        return 2
    data: Dict[str, Any] = {"data_root": _abs(args.data_root)}
    for key in cfg.SETTINGS:
        value = getattr(args, key, None)
        if key != "data_root" and value:
            data[key] = _abs(value)
    paths: Dict[str, Dict[str, str]] = {}
    for slug, key, value in args.dataset_path:
        paths.setdefault(slug, {})[key] = _abs(value)
    if paths:
        data["dataset_paths"] = paths
    cfg.save_file(data, path)
    print(f"wrote {path}")
    for warning in _warnings(data):
        print(f"  warning: {warning}")
    print("next: `neuroatlas config show`, then `neuroatlas fetch --list`")
    return 0


def _warnings(data: Dict[str, Any]) -> List[str]:
    out = []
    if not Path(data["data_root"]).is_dir():
        out.append(f"data root {data['data_root']} does not exist yet")
    for slug in (data.get("dataset_paths") or {}):
        if not _known_dataset(slug):
            out.append(f"{slug} is not a dataset NeuroAtlas ships a manifest for")
    return out


def _set(args: argparse.Namespace, unset: bool = False) -> int:
    data = cfg.load_file()
    key = args.key
    if key in cfg.SETTINGS:
        if unset:
            data.pop(key, None)
        else:
            data[key] = _abs(args.value)
    elif "." in key:
        slug, _, sub_key = key.partition(".")
        paths = data.setdefault("dataset_paths", {})
        if unset:
            (paths.get(slug) or {}).pop(sub_key, None)
            if not paths.get(slug):
                paths.pop(slug, None)
        else:
            if not _known_dataset(slug):
                print(f"  warning: {slug} is not a dataset NeuroAtlas ships a manifest for")
            paths.setdefault(slug, {})[sub_key] = _abs(args.value)
    else:
        import difflib

        close = difflib.get_close_matches(key, cfg.SETTINGS, n=1)
        hint = f" Did you mean {close[0]}?" if close else ""
        print(f"error: unknown setting {key!r}.{hint} Settings: {', '.join(cfg.SETTINGS)}, "
              f"or DATASET.KEY for a dataset path.", file=sys.stderr)
        return 2
    path = cfg.save_file(data)
    print(f"{'removed' if unset else 'set'} {key} in {path}")
    return 0


def _state(path: Optional[Path]) -> str:
    if path is None:
        return "not set"
    if not path.exists():
        return "missing"
    if not path.is_dir():
        return "not a directory"
    return "exists, writable" if os.access(path, os.W_OK) else "exists, read-only"


def _home_relative(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def _show() -> int:
    path = cfg.config_path()
    data = cfg.load_file()
    print(f"config file: {_home_relative(path)}  [{'found' if path.is_file() else 'not found'}]")
    if not path.is_file():
        print("  -> create it with `neuroatlas config init --data-root DIR`")
    checkout = _paths.checkout_root()
    print(f"package: neuroatlas from {'checkout ' + str(checkout) if checkout else 'installed wheel'}")

    print("roots  [state] [origin: env, file or default]")
    width = max(len(k) for k in cfg.SETTINGS)
    for key, setting in cfg.SETTINGS.items():
        r = cfg.resolve(key, data)
        if r.value is None:
            print(f"  {key.replace('_', ' '):<{width}}  -  [not set]")
            continue
        origin = f"${setting.env}" if r.origin == "env" else r.origin
        print(f"  {key.replace('_', ' '):<{width}}  {_home_relative(r.value)}  "
              f"[{_state(r.value)}] [{origin}]")

    paths = data.get("dataset_paths") or {}
    if paths:
        print("dataset paths")
        for slug, block in sorted(paths.items()):
            for key, value in sorted(block.items()):
                print(f"  {slug}.{key}  {value}  [{_state(Path(value))}]")

    print("downloads: off for every command except `fetch --download` and `prepare`; "
          "`--online` allows them for one run")
    from_user = cfg.offline_vars_from_user()
    if from_user:
        print("  your environment sets " + ", ".join(f"{v}={os.environ[v]}" for v in from_user)
              + ", which takes precedence")

    print("credentials (where, never what)")
    for name, advice in (("hf", "Hugging Face, for gated model weights"),
                         ("nsrr", "NSRR, for the NSRR sleep cohorts")):
        where = cfg.locate_token(name)
        if where is None:
            print(f"  {name:<5} not found  ({advice}: put it in "
                  f"{_home_relative(cfg.token_file(name))}, chmod 600)")
            continue
        note = ""
        if not where.startswith("$") and not cfg.token_permissions_ok(Path(where)):
            note = "  -- readable by others: chmod 600 it"
        print(f"  {name:<5} found in {_home_relative(Path(where)) if not where.startswith('$') else where}{note}")
    return 0


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action is None:
        parser.print_help()
        raise SystemExit(2)
    if args.action == "path":
        print(cfg.config_path())
        return
    status = {
        "init": lambda: _init(args),
        "show": _show,
        "set": lambda: _set(args),
        "unset": lambda: _set(args, unset=True),
    }[args.action]()
    if status:
        raise SystemExit(status)
