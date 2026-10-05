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
from neuroatlas.cli import Parser


def _dataset_key(text: str) -> Tuple[str, str, str]:
    """``ucddb.data_root=/mnt/ucddb`` -> ("ucddb", "data_root", "/mnt/ucddb")."""
    target, sep, value = text.partition("=")
    slug, dot, key = target.partition(".")
    if not (sep and dot and slug and key and value):
        raise argparse.ArgumentTypeError(
            f"expected DATASET.KEY=PATH, e.g. ucddb.data_root=/mnt/ucddb; got {text!r}")
    return slug, key, value


def _dataset_key_error(slug: str, key: str) -> Optional[str]:
    """Why ``slug.key`` would be written and then never read, or None."""
    return cfg.dataset_key_problem(slug, [key])


def _abs(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = Parser(
        prog="neuroatlas config",
        description="Where NeuroAtlas finds your data and puts what it makes. "
                    "The settings file is $NEUROATLAS_HOME/config.yaml (default "
                    "~/.neuroatlas/config.yaml); `neuroatlas config path` prints it.",
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
        init.add_argument(f"--{setting.key.replace('_', '-')}", metavar="DIR",
                          help=f"{setting.help[0].upper()}{setting.help[1:]}. "
                               f"Default: {setting.default_text}.")
    init.add_argument("--dataset-path", action="append", default=[], type=_dataset_key,
                      metavar="DATASET.KEY=PATH",
                      help="A dataset stored outside its default sub-folder, e.g. "
                           "ucddb.data_root=/mnt/ucddb. Repeatable.")
    init.add_argument("--force", action="store_true",
                      help="Replace an existing settings file.")

    sub.add_parser("show", help="Print every setting, whether it exists, and where it came from.")

    set_ = sub.add_parser("set", help="Change one setting.",
                          description="Change one setting: a root (data_root, cache_root, "
                                      "output_root, models_root), or a dataset path "
                                      "DATASET.KEY such as chbmit.bids_root. A dataset "
                                      "or key that nothing reads is refused, with the "
                                      "keys that dataset does read; `neuroatlas data "
                                      "status <dataset>` names the one to set.")
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
    for slug, key, _ in args.dataset_path:
        problem = _dataset_key_error(slug, key)
        if problem:
            print(f"error: --dataset-path {slug}.{key}: {problem}", file=sys.stderr)
            return 2
    # All four roots are written out, defaults included, so the file says
    # where everything goes and nothing depends on how the package was
    # installed or which directory the command ran from.
    data: Dict[str, Any] = {"data_root": _abs(args.data_root)}
    for key, setting in cfg.SETTINGS.items():
        value = getattr(args, key, None)
        if key == "data_root":
            continue
        data[key] = _abs(value) if value else str(setting.default())
    paths: Dict[str, Dict[str, str]] = {}
    for slug, key, value in args.dataset_path:
        paths.setdefault(slug, {})[key] = _abs(value)
    if paths:
        data["dataset_paths"] = paths
    cfg.save_file(data, path)
    print(f"wrote {path}")
    for key in cfg.SETTINGS:
        print(f"  {key.replace('_', ' '):<11}  {data[key]}")
    for warning in _warnings(data):
        print(f"  warning: {warning}")
    for line in _legacy_notes(data):
        print(line)
    print(NEXT_STEPS)
    return 0


NEXT_STEPS = ("next: `neuroatlas config show`, then `neuroatlas list benchmarks` and "
              "`neuroatlas data status <benchmark>`")

DOWNLOADS_LINE = ("downloads: off, except in `data download`, `data prepare`, `models download` "
                  "and `fetch --download`; `neuroatlas --online <command>` allows them for one run")


def _warnings(data: Dict[str, Any]) -> List[str]:
    out = []
    if not Path(data["data_root"]).is_dir():
        out.append(f"data root {data['data_root']} does not exist yet")
    return out


def _unset_message(key: str, path: Path) -> str:
    return f"{key} was not set in {path}; nothing to remove"


def _set(args: argparse.Namespace, unset: bool = False) -> int:
    # Lenient: this is how a file with a misspelt setting gets fixed.
    data = cfg.load_file(strict=False)
    key = args.key
    path = cfg.config_path()
    if key in cfg.SETTINGS or (unset and key in cfg.unknown_settings(data)):
        if unset:
            if key not in data:
                print(_unset_message(key, path))
                return 0
            data.pop(key)
        else:
            data[key] = _abs(args.value)
    elif "." in key:
        slug, _, sub_key = key.partition(".")
        paths = data.setdefault("dataset_paths", {})
        if unset:
            if sub_key not in (paths.get(slug) or {}):
                print(_unset_message(key, path))
                return 0
            paths[slug].pop(sub_key)
            if not paths[slug]:
                paths.pop(slug)
        else:
            problem = _dataset_key_error(slug, sub_key)
            if problem:
                print(f"error: {problem}", file=sys.stderr)
                return 2
            paths.setdefault(slug, {})[sub_key] = _abs(args.value)
    else:
        import difflib

        close = difflib.get_close_matches(key, cfg.SETTINGS, n=1)
        hint = f" Did you mean {close[0]}?" if close else ""
        print(f"error: unknown setting {key!r}.{hint} Settings: {', '.join(cfg.SETTINGS)}, "
              f"or DATASET.KEY for a dataset path (`neuroatlas data status <dataset> -v` "
              f"shows the key each dataset reads).", file=sys.stderr)
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


def _legacy_notes(data: Dict[str, Any]) -> List[str]:
    """What an earlier version left in the source checkout, where the setting
    that replaces it now points elsewhere: say how to keep using it."""
    out = []
    for what, legacy, setting in _paths.legacy_locations():
        current = cfg.resolve(setting, data).value
        if legacy.name == "preprocessed":
            target = _paths.prepared_dir()
            out.append(f"note: {what} from an earlier version are in {legacy}; they are now "
                       f"read from {target}: mkdir -p {target} && mv {legacy}/* {target}/")
            continue
        if current is not None and Path(current) == legacy:
            continue
        out.append(f"note: {what} from an earlier version are in {legacy}; to keep using "
                   f"them: neuroatlas config set {setting} {legacy}")
    return out


def _show() -> int:
    path = cfg.config_path()
    data = cfg.load_file(strict=False)       # warns about, rather than stops at, a bad key
    problems = cfg.unknown_settings(data)
    print(f"config file: {_home_relative(path)}  [{'found' if path.is_file() else 'not found'}]")
    if not path.is_file():
        print("  -> create it with `neuroatlas config init --data-root DIR`")
    if problems:
        print(f"  -> unknown settings {', '.join(problems)}: every other command refuses to "
              f"run until you `neuroatlas config unset` them")
    checkout = _paths.checkout_root()
    print(f"package: neuroatlas from {'checkout ' + str(checkout) if checkout else 'installed wheel'}"
          f" (defaults hang off $NEUROATLAS_HOME = {_home_relative(_paths.home())} either way)")

    print("roots  [state] [origin: env, file or default]")
    width = max(len(k) for k in cfg.SETTINGS)
    resolved = {}
    for key, setting in cfg.SETTINGS.items():
        r = resolved[key] = cfg.resolve(key, data)
        if r.value is None:
            print(f"  {key.replace('_', ' '):<{width}}  -  [not set]")
            continue
        origin = f"${setting.env}" if r.origin == "env" else r.origin
        print(f"  {key.replace('_', ' '):<{width}}  {_home_relative(r.value)}  "
              f"[{_state(r.value)}] [{origin}]")
    mne = cfg.mne_data(data)
    mne_origin = {"env": "$MNE_DATA", "mne config": "MNE's config file",
                  "data root": "data root/mne_data", "default": "MNE's default"}[mne.origin]
    print(f"  {'MOABB data':<{width}}  {_home_relative(mne.value)}  [{_state(mne.value)}] "
          f"[{mne_origin}]")
    per_dataset = cfg.mne_per_dataset_keys()
    if per_dataset:
        print(f"  {'':<{width}}  (MNE's config file {_home_relative(cfg.mne_config_file())} names "
              f"its own folder for {len(per_dataset)} dataset(s); NeuroAtlas uses the one above "
              f"for all of them and never edits that file)")

    paths = data.get("dataset_paths") or {}
    if paths:
        print("dataset paths")
        for slug, block in sorted(paths.items()):
            for key, value in sorted(block.items()):
                print(f"  {slug}.{key}  {value}  [{_state(Path(str(value)))}]")

    for line in _legacy_notes(data):
        print(line)

    print(DOWNLOADS_LINE)
    from_user = cfg.offline_vars_from_user()
    if from_user:
        print("  your environment sets " + ", ".join(f"{v}={os.environ[v]}" for v in from_user)
              + ", which takes precedence")

    print("credentials (where, never what)")
    for name, advice in (("hf", "Hugging Face, for gated model weights"),
                         ("nsrr", "NSRR, for the NSRR sleep cohorts"),
                         ("github", "GitHub, for private release assets")):
        where = cfg.locate_token(name)
        if where is None:
            print(f"  {name:<5} not found  ({advice}: put it in "
                  f"{_home_relative(cfg.token_file(name))}, chmod 600)")
            continue
        note = ""
        if not where.startswith("$") and not cfg.token_permissions_ok(Path(where)):
            note = "  -- readable by others: chmod 600 it"
        print(f"  {name:<5} found in {_home_relative(Path(where)) if not where.startswith('$') else where}{note}")

    from neuroatlas import catalog

    try:
        benches = catalog.catalog()
        print(f"catalog: {len(benches)} benchmarks ({', '.join(benches)}); validation ok")
    except catalog.CatalogError as exc:
        print(f"catalog: INVALID -- {exc}")
        return 1
    if problems:
        print(f"error: {path} has unknown settings: {', '.join(problems)}", file=sys.stderr)
        return 2
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
