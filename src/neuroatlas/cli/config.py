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
from neuroatlas.cli import Parser, UsageError, _msg


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
        description="Set where NeuroAtlas reads datasets and writes caches, results and "
                    "model weights. The settings are saved in ~/.neuroatlas/config.yaml, "
                    "or in $NEUROATLAS_HOME/config.yaml when that variable is set.",
    )
    sub = parser.add_subparsers(dest="action", metavar="<action>")

    init = sub.add_parser("init", help="Write the settings file.",
                          description="Write the settings file. With a project folder DIR, "
                                      "everything goes to DIR/data, DIR/cache, DIR/results "
                                      "and DIR/models. A `--*-root` option moves one of them.")
    init.add_argument("root", nargs="?", metavar="DIR",
                      help="Project folder. Its sub-folders are created when needed.")
    init.add_argument("--data-root", metavar="DIR",
                      help="Folder with the raw datasets, one sub-folder per dataset "
                           "(default: DIR/data).")
    for setting in cfg.SETTINGS.values():
        if setting.key == "data_root":
            continue
        init.add_argument(f"--{setting.key.replace('_', '-')}", metavar="DIR",
                          help=f"{setting.help} (default: DIR/{PROJECT_FOLDERS[setting.key]}, "
                               f"else {setting.default_text}).")
    init.add_argument("--dataset-path", action="append", default=[], type=_dataset_key,
                      metavar="DATASET.KEY=PATH",
                      help="Folder of a dataset kept elsewhere, as in "
                           "ucddb.data_root=/mnt/ucddb. Can be repeated.")
    init.add_argument("--force", action="store_true",
                      help="Overwrite an existing settings file.")

    sub.add_parser("show", help="Print each setting, whether its folder exists and where "
                                "the value comes from.")

    set_ = sub.add_parser("set", help="Change one setting.",
                          description="Change one setting. KEY is data_root, cache_root, "
                                      "output_root, models_root or a dataset folder such as "
                                      "chbmit.bids_root. `neuroatlas data status DATASET` "
                                      "shows the key a dataset reads.")
    set_.add_argument("key", help="Setting name, such as data_root or ucddb.data_root.")
    set_.add_argument("value", help="New value, usually a folder.")

    unset = sub.add_parser("unset", help="Remove one setting, so it goes back to its default.")
    unset.add_argument("key", help="Setting name.")

    sub.add_parser("path", help="Print the path of the settings file.")

    token = sub.add_parser(
        "token", help="Save a Hugging Face, GitHub or NSRR token.",
        description="Save an access token for Hugging Face (hf), GitHub (github) or NSRR "
                    "(nsrr). Type it when asked, or pipe it in as in `neuroatlas config "
                    "token hf < file`. It is saved to `$NEUROATLAS_HOME/<name>_token`, "
                    "readable only by you, and never printed.")
    token.add_argument("name", choices=sorted(_TOKENS),
                       help="hf for gated Hugging Face repositories, github when GitHub "
                            "refuses a download, or nsrr for the NSRR sleep datasets.")
    token.add_argument("--remove", action="store_true", help="Delete the saved token.")
    return parser


# name -> (service, the variables that take precedence over the file)
_TOKENS = {
    "hf": ("Hugging Face", ("HF_TOKEN",)),
    "github": ("GitHub", ("GITHUB_TOKEN", "GH_TOKEN")),
    "nsrr": ("NSRR", ("NSRR_TOKEN",)),
}


def _token(args: argparse.Namespace) -> int:
    service, variables = _TOKENS[args.name]
    path = cfg.token_file(args.name)
    if args.remove:
        if path.is_file():
            path.unlink()
            print(f"removed the {service} token ({_home_relative(path)})")
        else:
            print(f"no {service} token saved at {_home_relative(path)}; nothing to remove")
        return 0
    if sys.stdin.isatty():
        import getpass

        value = getpass.getpass(f"{service} token (input hidden): ")
    else:
        value = sys.stdin.read()
    value = value.strip()
    if not value or any(c.isspace() for c in value):
        _msg.error(f"that is not a {service} token (empty, or it contains spaces)",
                   f"neuroatlas config token {args.name} (paste the token alone)")
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.is_file()
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(value + "\n")
    os.chmod(path, 0o600)
    print(f"{'replaced' if existed else 'saved'} the {service} token: {_home_relative(path)} "
          f"(chmod 600)")
    shadowing = [v for v in variables if os.environ.get(v)]
    if shadowing:
        _msg.warning(f"${shadowing[0]} is set in this shell and is used instead of the file",
                     f"unset {shadowing[0]}")
    return 0


# --------------------------------------------------------------------------

def _init(args: argparse.Namespace) -> int:
    path = cfg.config_path()
    if path.exists() and not args.force:
        from neuroatlas.cli import command_with

        _msg.error(f"{path} already exists",
                   "neuroatlas config set data_root DIR (changes one setting)\n"
                   f"fix: {command_with('--force') or 'neuroatlas config init --data-root DIR --force'}"
                   f" (replaces the file)")
        return 2
    for slug, key, _ in args.dataset_path:
        problem = _dataset_key_error(slug, key)
        if problem:
            _msg.error(f"--dataset-path {slug}.{key}: {problem}")
            return 2
    if not (args.root or args.data_root):
        raise UsageError("config init needs a project folder (or --data-root)\n"
                         "fix: neuroatlas config init ~/neuroatlas")
    # All four roots are written out, defaults included, so the file says
    # where everything goes and nothing depends on how the package was
    # installed or which directory the command ran from.
    root = Path(_abs(args.root)) if args.root else None
    data: Dict[str, Any] = {}
    for key, setting in cfg.SETTINGS.items():
        value = getattr(args, key, None)
        if value:
            data[key] = _abs(value)
        elif root is not None:
            data[key] = str(root / PROJECT_FOLDERS[key])
        else:
            data[key] = str(setting.default())
    if root is not None:
        for key in cfg.SETTINGS:
            Path(data[key]).mkdir(parents=True, exist_ok=True)
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
        _msg.warning(warning)
    _print_legacy_notes(data)
    print(NEXT_STEPS)
    return 0


#: The sub-folders of a project folder (`config init DIR`), per setting.
PROJECT_FOLDERS = {"data_root": "data", "cache_root": "cache", "output_root": "results",
                   "models_root": "models"}

NEXT_STEPS = ("next: neuroatlas config show; neuroatlas list benchmarks; "
              "neuroatlas data status <benchmark>")

DOWNLOADS_LINE = ("downloads: off, except in data download, data prepare and models download "
                  "(--online allows them for one run)")


def _warnings(data: Dict[str, Any]) -> List[str]:
    out = []
    if not Path(data["data_root"]).is_dir():
        out.append(f"the data root {data['data_root']} does not exist yet; `data download` "
                   f"creates it, or point it at your datasets\n"
                   f"fix: neuroatlas config set data_root DIR")
    return out


def _unset_message(key: str, path: Path) -> str:
    return f"{key} is not set in {path}; nothing to remove"


def _set(args: argparse.Namespace, unset: bool = False) -> int:
    # Lenient: this is how a file with a misspelt setting gets fixed.
    data = cfg.load_file(strict=False)
    key = args.key
    path = cfg.config_path()
    if not unset and not str(args.value).strip():
        _msg.error(f"{key} needs a value", f"neuroatlas config unset {key} (back to the default)")
        return 2
    if not unset and key in cfg.SETTINGS and Path(args.value).expanduser().is_file():
        _msg.error(f"{key} must be a folder; {_abs(args.value)} is a file",
                   f"neuroatlas config set {key} DIR")
        return 2
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
                if cfg.dataset_path_keys(slug) == []:          # a MOABB dataset: no key
                    fix = "export MNE_DATA=DIR (the folder of every MOABB dataset)"
                elif slug in cfg.known_dataset_slugs():
                    # the command again with the key the dataset reads: the
                    # closest one, or its only one
                    import difflib

                    valid = cfg.dataset_path_keys(slug) or []
                    close = difflib.get_close_matches(sub_key, valid, n=1)
                    right = close[0] if close else valid[0] if len(valid) == 1 else None
                    fix = (f"neuroatlas config set {slug}.{right} {args.value}" if right
                           else f"neuroatlas data status {slug} -v (names the key it reads)")
                else:
                    # a mistyped dataset: the command with the closest one
                    import difflib

                    close = difflib.get_close_matches(slug, cfg.known_dataset_slugs(), n=1)
                    fix = (f"neuroatlas config set {close[0]}.{sub_key} {args.value}" if close
                           else "neuroatlas list datasets")
                _msg.error(problem, fix)
                return 2
            paths.setdefault(slug, {})[sub_key] = _abs(args.value)
    else:
        import difflib

        from neuroatlas.cli import corrected_command

        close = difflib.get_close_matches(key, cfg.SETTINGS, n=1)
        fix = corrected_command({key: close[0]}) if close else None
        _msg.error(f"unknown setting {key!r}" + (f" (did you mean {close[0]}?)" if close else
                                                  f"; settings: {', '.join(cfg.SETTINGS)}, or "
                                                  f"DATASET.KEY for a dataset's folder"),
                   fix or "neuroatlas data status <dataset> -v names a dataset's key")
        return 2
    path = cfg.save_file(data)
    if unset:
        print(f"removed {key} from {path}")
    else:
        value = data.get(key) if key in cfg.SETTINGS else \
            (data.get("dataset_paths") or {}).get(key.partition(".")[0], {}).get(
                key.partition(".")[2])
        print(f"{key} = {value}  (saved in {path})")
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


def _legacy_notes(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """``(what, fix)`` for what an earlier version left in the source
    checkout, where the setting that replaces it now points elsewhere."""
    out = []
    for what, legacy, setting in _paths.legacy_locations():
        current = cfg.resolve(setting, data).value
        if legacy.name == "preprocessed":
            target = _paths.prepared_dir()
            out.append((f"{what} in {legacy} are not read: they belong in {target}",
                        f"mkdir -p {target} && mv {legacy}/* {target}/"))
            continue
        if current is not None and Path(current) == legacy:
            continue
        out.append((f"{what} in {legacy} are not read: {setting} points elsewhere",
                    f"neuroatlas config set {setting} {legacy}"))
    return out


def _print_legacy_notes(data: Dict[str, Any]) -> None:
    for what, fix in _legacy_notes(data):
        _msg.note(what, fix)


def _show() -> int:
    path = cfg.config_path()
    data = cfg.load_file(strict=False, warn=False)     # a bad key: the error at the end
    problems = cfg.unknown_settings(data)
    print(f"config file: {_home_relative(path)}  [{'found' if path.is_file() else 'not found'}]")
    checkout = _paths.checkout_root()
    print(f"home:        {_home_relative(_paths.home())}  ($NEUROATLAS_HOME; the defaults "
          f"below are under it)")
    print(f"package:     {'checkout ' + str(checkout) if checkout else 'installed wheel'}")

    print("folders  [state] [set by: an environment variable, the config file, or the "
          "default]")
    width = max(len(k) for k in cfg.SETTINGS)
    resolved = {}
    for key, setting in cfg.SETTINGS.items():
        r = resolved[key] = cfg.resolve(key, data)
        if r.value is None:
            print(f"  {key.replace('_', ' '):<{width}}  -  [not set]")
            continue
        origin = f"${setting.env}" if r.origin == "env" else \
            {"file": "config file"}.get(r.origin, r.origin)
        print(f"  {key.replace('_', ' '):<{width}}  {_home_relative(r.value)}  "
              f"[{_state(r.value)}] [{origin}]")
    mne = cfg.mne_data(data)
    mne_origin = {"env": "$MNE_DATA", "mne config": "MNE's config file",
                  "data root": "under the data root", "default": "MNE's default"}[mne.origin]
    print(f"  {'MOABB data':<{width}}  {_home_relative(mne.value)}  [{_state(mne.value)}] "
          f"[{mne_origin}]")
    per_dataset = cfg.mne_per_dataset_keys()
    if per_dataset:
        print(f"  {'':<{width}}  (used for every MOABB dataset; the "
              f"{len(per_dataset)} per-dataset folders in "
              f"{_home_relative(cfg.mne_config_file())} are not)")

    paths = data.get("dataset_paths") or {}
    if paths:
        print("dataset paths")
        for slug, block in sorted(paths.items()):
            for key, value in sorted(block.items()):
                print(f"  {slug}.{key}  {value}  [{_state(Path(str(value)))}]")

    print(DOWNLOADS_LINE)

    print("credentials (where, never what)")
    loose = []
    for name, use in (("hf", "needed only for a gated Hugging Face repository"),
                      ("nsrr", "needed to download the NSRR sleep datasets"),
                      ("github", "needed only when a GitHub download is refused")):
        where = cfg.locate_token(name)
        if where is None:
            print(f"  {name:<6}  not found  ({use}: neuroatlas config token {name})")
            continue
        shown = where if where.startswith("$") else _home_relative(Path(where))
        print(f"  {name:<6}  found in {shown}")
        if not where.startswith("$") and not cfg.token_permissions_ok(Path(where)):
            loose.append(where)

    from neuroatlas import catalog

    try:
        catalog.catalog()
    except catalog.CatalogError as exc:
        from neuroatlas.run import REINSTALL

        _msg.error(f"the benchmark definitions shipped with the package do not load: {exc}",
                                   REINSTALL)
        return 1
    _print_legacy_notes(data)
    from_user = cfg.offline_vars_from_user()
    if from_user:
        _msg.warning("your environment sets " + ", ".join(f"{v}={os.environ[v]}" for v in from_user)
                     + ", which overrides the downloads setting above",
                     "unset " + " ".join(from_user) + " (to restore it)")
    for where in loose:
        _msg.warning(f"{_home_relative(Path(where))} is readable by others", f"chmod 600 {where}")
    if not path.is_file():
        _msg.note("no config file yet: every root above is its default",
                  "neuroatlas config init --data-root DIR")
    if problems:
        _msg.error(cfg._unknown_text(path, problems))
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
        "token": lambda: _token(args),
    }[args.action]()
    if status:
        raise SystemExit(status)
