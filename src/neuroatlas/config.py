"""The user's NeuroAtlas settings: where data, caches, results and weights live.

One file, ``$NEUROATLAS_HOME/config.yaml`` (default ``~/.neuroatlas``),
written by ``neuroatlas config init``::

    data_root: /data/eeg
    cache_root: /scratch/neuroatlas/cache
    dataset_paths:
      ucddb:
        data_root: /mnt/ucddb

Every setting is resolved in the same order: an environment variable, then
the file, then a default. The environment wins so that a cluster job can
redirect one run without editing anyone's file.

The engine predates this file and reads environment variables
(``${EEG_DATA_ROOT}`` in the cohort manifests, ``$EEG_CACHE_ROOT``,
``$NEUROATLAS_MODELS_ROOT``). :func:`apply_to_environ` exports the file's
values into exactly those variables, when they are unset, so the file reaches
every loader without each one learning about it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from neuroatlas import _paths

CONFIG_NAME = "config.yaml"


@dataclass(frozen=True)
class Setting:
    key: str
    env: str
    default: Callable[[], Optional[Path]]
    help: str
    default_text: str = "none"      # the default as `--help` states it, machine-independent


# The order here is the order `config show` prints. The defaults depend on
# $NEUROATLAS_HOME and nothing else -- not on whether the package is an
# editable install or a wheel, nor on the working directory.
SETTINGS: Dict[str, Setting] = {s.key: s for s in (
    Setting("data_root", "EEG_DATA_ROOT", lambda: None,
            "Folder with the raw datasets, one sub-folder per dataset"),
    Setting("cache_root", "EEG_CACHE_ROOT", lambda: _paths.artifacts_dir("embedding_cache"),
            "Folder for saved embeddings and prepared datasets, which can grow large",
            "$NEUROATLAS_HOME/artifacts/embedding_cache"),
    Setting("output_root", "NEUROATLAS_OUTPUT_ROOT", lambda: _paths.artifacts_dir("benchmarks"),
            "Folder for the probe results", "$NEUROATLAS_HOME/artifacts/benchmarks"),
    Setting("models_root", "NEUROATLAS_MODELS_ROOT", lambda: _paths.artifacts_dir("models"),
            "Folder for model weights", "$NEUROATLAS_HOME/artifacts/models"),
)}

# Keys a config file may hold besides the roots.
_OTHER_KEYS = {"dataset_paths"}

# A per-dataset key names a path when it contains one of these (data_root,
# bids_root, raw_dir, cache_root, preprocessed_path, ...).
_PATHISH = ("root", "dir", "path", "file")


class ConfigError(SystemExit):
    """A config file that cannot be used as written.

    A SystemExit carrying its message, like ``MissingDatasetPath``: raised
    from ``python -m ...`` it prints and exits 1; the ``neuroatlas`` command
    catches it and exits 2, the usage-error status.
    """


def config_path() -> Path:
    return _paths.home() / CONFIG_NAME


_WARNED: set = set()


def _warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        from neuroatlas.cli import _msg

        _msg.warning(message)


def unknown_settings(data: Dict[str, Any]) -> List[str]:
    """Top-level keys of a config file that are not settings."""
    return sorted(set(data) - set(SETTINGS) - _OTHER_KEYS)


def _unknown_text(path: Path, unknown: List[str]) -> str:
    import difflib

    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, [*SETTINGS, *_OTHER_KEYS], n=1)
        hints.append(f"{key} (did you mean {close[0]}?)" if close else key)
    return (f"{path}: unknown setting{'s' if len(unknown) > 1 else ''} {', '.join(hints)}\n"
            f"fix: " + "; ".join(f"neuroatlas config unset {key}" for key in unknown))


def load_file(path: Optional[Path] = None, strict: bool = True,
              warn: bool = True) -> Dict[str, Any]:
    """The config file as written, or ``{}`` when there is none.

    ``strict`` (every command but ``config``): an unknown setting is an error,
    because a run that silently ignored a misspelt ``cahce_root`` would write
    somewhere the user did not ask for. The ``config`` commands load it
    leniently -- warn and carry on -- since they are how it gets fixed.
    ``warn=False``: not even that (``config show`` reports it itself).
    """
    import yaml

    path = path or config_path()
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        where = getattr(exc, "problem_mark", None)
        line = f" (line {where.line + 1})" if where is not None else ""
        raise ConfigError(f"error: {path} is not valid YAML{line}: "
                          f"{getattr(exc, 'problem', None) or exc}\n"
                          f"fix: correct that line, or rewrite the file with neuroatlas "
                          f"config init --force --data-root DIR") from None
    if not isinstance(data, dict):
        raise ConfigError(f"error: {path} must be a mapping of settings, not a "
                          f"{type(data).__name__}\nfix: neuroatlas config init --force "
                          f"--data-root DIR rewrites it")
    unknown = unknown_settings(data)
    if unknown:
        if strict:
            raise ConfigError(f"error: {_unknown_text(path, unknown)}")
        if warn:
            _warn_once(_unknown_text(path, unknown))
    paths = data.get("dataset_paths") or {}
    if not isinstance(paths, dict) or not all(isinstance(v, dict) for v in paths.values()):
        message = (f"{path}: dataset_paths must map each dataset to its keys\n"
                   f"fix: write it as dataset_paths: {{ucddb: {{data_root: /mnt/ucddb}}}}, "
                   f"or neuroatlas config set ucddb.data_root /mnt/ucddb")
        if strict:
            raise ConfigError(f"error: {message}")
        _warn_once(message.replace("\n", " (ignored until fixed)\n", 1))
        data = {k: v for k, v in data.items() if k != "dataset_paths"}
        paths = {}
    for slug, block in paths.items():
        problem = dataset_key_problem(str(slug), [str(k) for k in block])
        if problem:
            _warn_once(f"{path}: {problem}; it has no effect\n"
                       f"fix: neuroatlas config unset {slug}.{next(iter(block), 'KEY')}")
    return data


# --------------------------------------------------------------------------
# per-dataset path keys
# --------------------------------------------------------------------------

def _manifest_defaults(slug: str) -> Optional[Dict[str, Any]]:
    """runtime_defaults plus every backend's defaults, or None for no such dataset.

    Read from the shipped manifest when there is one (fast: no registry
    import); the registry only for the MOABB datasets it generates.
    """
    import yaml

    dossier = _paths.configs_dir("cohorts", slug, "cohort.yaml")
    if dossier.is_file():
        manifest = yaml.safe_load(dossier.read_text()) or {}
        merged = dict(manifest.get("runtime_defaults") or {})
        for block in (manifest.get("backends") or {}).values():
            merged.update((block or {}).get("defaults") or {})
        return merged
    try:
        from neuroatlas.benchmarking_helpers.registry.discovery import load_dataset_spec

        spec = load_dataset_spec(slug)
    except Exception:
        return None
    return dict(spec.config_defaults or {})


def dataset_path_keys(slug: str) -> Optional[List[str]]:
    """The keys `config set <slug>.KEY` accepts, or None when there is no such dataset."""
    defaults = _manifest_defaults(slug)
    if defaults is None:
        return None
    return sorted(k for k in defaults if any(token in k for token in _PATHISH))


def known_dataset_slugs() -> List[str]:
    root = _paths.configs_dir("cohorts")
    return sorted(p.name for p in root.iterdir() if (p / "cohort.yaml").is_file())


def dataset_key_problem(slug: str, keys: List[str]) -> Optional[str]:
    """Why ``<slug>.<key>`` would have no effect, or None when it is read."""
    import difflib

    valid = dataset_path_keys(slug)
    if valid is None:
        close = difflib.get_close_matches(slug, known_dataset_slugs(), n=3)
        hint = f" (did you mean {', '.join(close)}?)" if close else ""
        return f"no dataset {slug!r}{hint}"
    for key in keys:
        if key in valid:
            continue
        if not valid:
            return (f"{slug} reads no path from the settings: it is a MOABB dataset, read "
                    f"from $MNE_DATA with every other MOABB dataset")
        close = difflib.get_close_matches(key, valid, n=1)
        hint = f" (did you mean {close[0]}?)" if close else ""
        return f"{slug} has no path key {key!r}{hint}; its keys: {', '.join(valid)}"
    return None


def save_file(data: Dict[str, Any], path: Optional[Path] = None) -> Path:
    import yaml

    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unknown keys are kept: `config set` must not silently drop a misspelt
    # setting the user has not seen yet. `config unset KEY` removes one.
    keys = [*SETTINGS, *sorted(_OTHER_KEYS), *unknown_settings(data)]
    ordered = {k: data[k] for k in keys if data.get(k) not in (None, {}, "")}
    path.write_text(yaml.safe_dump(ordered, sort_keys=False, default_flow_style=False))
    return path


# Variables this process exported from the file (apply_to_environ) or set to
# go offline (set_offline), so reports can tell them from the user's own.
_EXPORTED: set = set()
_EXPORTED_VALUES: Dict[str, str] = {}
_OFFLINE_SET: set = set()


def _export(var: str, value: str) -> None:
    os.environ[var] = value
    _EXPORTED.add(var)
    _EXPORTED_VALUES[var] = value


def _ours(var: str) -> bool:
    """True when *var* still holds the value this process exported. Once the
    user (or a caller of the Python API) sets it to something else, it is
    theirs, and wins like any variable they exported."""
    return var in _EXPORTED and os.environ.get(var) == _EXPORTED_VALUES.get(var)


@dataclass(frozen=True)
class Resolved:
    key: str
    value: Optional[Path]
    origin: str          # "env", "file", "default" or "unset"


def _from_env(value: str) -> Path:
    """An environment value as a path. A relative one is taken against the
    current directory, here, once -- so every later reader (a datamodule, a
    job script, `config show`) sees the same absolute folder."""
    return _paths._absolute(Path(value))


def _from_file(value: Any) -> Path:
    """A file value as a path. `config set` writes absolute paths; a relative
    one typed by hand is taken against the folder the file lives in, so it
    does not depend on where the command runs."""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else config_path().parent / path


def resolve(key: str, data: Optional[Dict[str, Any]] = None) -> Resolved:
    """One setting, with where its value came from."""
    setting = SETTINGS[key]
    env = os.environ.get(setting.env)
    # A variable apply_to_environ exported came from the file, not the user.
    if env and not _ours(setting.env):
        return Resolved(key, _from_env(env), "env")
    data = load_file(strict=False) if data is None else data
    if data.get(key):
        return Resolved(key, _from_file(data[key]), "file")
    default = setting.default()
    return Resolved(key, default, "default" if default is not None else "unset")


def get(key: str) -> Optional[Path]:
    return resolve(key).value


def dataset_paths(slug: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Per-dataset path keys from the file, e.g. ``{"data_root": "/mnt/ucddb"}``."""
    data = load_file(strict=False) if data is None else data
    block = (data.get("dataset_paths") or {}).get(slug) or {}
    return {k: str(_from_file(v)) for k, v in block.items()}


def apply_to_environ(data: Optional[Dict[str, Any]] = None, strict: bool = True) -> List[str]:
    """Export file values into the variables the engine reads, if unset.

    Returns the root variables it set. Idempotent, and never overrides a
    variable the user or a job script already exported. A relative value
    the user exported is made absolute in place (see :func:`_from_env`).
    Also exports ``MNE_DATA`` (see :func:`mne_data`), so MOABB reads and
    writes where `data status` looks and never edits MNE's own config file.
    """
    data = load_file(strict=strict) if data is None else data
    exported = []
    for setting in SETTINGS.values():
        if setting.env in _EXPORTED and not _ours(setting.env):
            _EXPORTED.discard(setting.env)        # set by the user since: theirs now
        if setting.env in _EXPORTED:
            # Ours from an earlier call: follow the file, which may have changed.
            if data.get(setting.key):
                _export(setting.env, str(_from_file(data[setting.key])))
            else:
                os.environ.pop(setting.env, None)
                _EXPORTED.discard(setting.env)
            continue
        current = os.environ.get(setting.env)
        if current:
            absolute = str(_from_env(current))
            if absolute != current:
                os.environ[setting.env] = absolute
            continue
        if not data.get(setting.key):
            continue
        _export(setting.env, str(_from_file(data[setting.key])))
        exported.append(setting.env)
    _export_mne_data(data)
    return exported


# --------------------------------------------------------------------------
# MOABB's folder
# --------------------------------------------------------------------------
# MOABB downloads through MNE, which keeps its datasets under MNE_DATA. When
# MNE finds no MNE_DATA anywhere it picks ~/mne_data and *writes that into
# ~/.mne/mne-python.json* -- a user-wide file NeuroAtlas has no business
# editing (F-029). MNE also honours per-dataset keys from that file
# (MNE_DATASETS_BNCI_PATH, ...) ahead of MNE_DATA, which can send one
# dataset's download and its later read to different folders.
#
# So NeuroAtlas resolves one folder itself and exports it as MNE_DATA, with
# MNE_DONTWRITE_HOME=true: MNE then neither writes nor reads the user's
# ~/.mne/mne-python.json in a NeuroAtlas process, and `data download`,
# `data prepare`, `data status` and `embed` all use the same folder.

def mne_config_file() -> Path:
    """MNE's own config file (``_MNE_FAKE_HOME_DIR`` is MNE's override of ``~``)."""
    base = os.environ.get("_MNE_FAKE_HOME_DIR") or str(Path.home())
    return Path(base).expanduser() / ".mne" / "mne-python.json"


def _mne_config() -> Dict[str, Any]:
    import json

    try:
        data = json.loads(mne_config_file().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def mne_per_dataset_keys() -> List[str]:
    """MNE_DATASETS_*_PATH entries in MNE's config file, which NeuroAtlas does not follow."""
    return sorted(k for k in _mne_config() if k.startswith("MNE_DATASETS_") and k.endswith("_PATH"))


def mne_data(data: Optional[Dict[str, Any]] = None) -> Resolved:
    """Where MOABB data lives: $MNE_DATA, else MNE_DATA from MNE's config file
    (read, never written), else ``<data root>/mne_data``, else ``~/mne_data``
    (MNE's own default).

    ``<data root>/mne_data`` comes before MNE's default so a MOABB download
    lands with every other raw dataset, under the root the user named.
    """
    env = os.environ.get("MNE_DATA")
    if env and not _ours("MNE_DATA"):
        return Resolved("mne_data", _from_env(env), "env")
    from_mne = _mne_config().get("MNE_DATA")
    if from_mne:
        return Resolved("mne_data", Path(str(from_mne)).expanduser(), "mne config")
    root = resolve("data_root", data).value
    if root is not None:
        return Resolved("mne_data", root / "mne_data", "data root")
    return Resolved("mne_data", Path.home() / "mne_data", "default")


def _export_mne_data(data: Dict[str, Any]) -> None:
    resolved = mne_data(data)
    if resolved.origin != "env":
        _export("MNE_DATA", str(resolved.value))
    if "MNE_DONTWRITE_HOME" not in os.environ:
        _export("MNE_DONTWRITE_HOME", "true")


# --------------------------------------------------------------------------
# offline mode
# --------------------------------------------------------------------------
# Only the download commands touch the network. Everything else -- and every
# cluster job -- runs with downloads switched off, so a job never stalls
# halfway on a missing file; it fails at once and names the download command.

OFFLINE_VARS = ("NEUROATLAS_OFFLINE", "EEGBENCH_OFFLINE", "HF_HUB_OFFLINE")


def set_offline() -> List[str]:
    """Switch downloads off, unless a variable was already set explicitly."""
    changed = []
    for var in OFFLINE_VARS:
        if var not in os.environ:
            os.environ[var] = "1"
            _OFFLINE_SET.add(var)
            changed.append(var)
    return changed


def offline_vars_from_user() -> List[str]:
    """Offline variables set in the user's environment, not by this command."""
    return [v for v in OFFLINE_VARS if v in os.environ and v not in _OFFLINE_SET]


def is_offline() -> bool:
    return any(os.environ.get(var) == "1" for var in ("NEUROATLAS_OFFLINE", "EEGBENCH_OFFLINE"))


# --------------------------------------------------------------------------
# credentials: where, never what
# --------------------------------------------------------------------------

def token_file(name: str) -> Path:
    """``$NEUROATLAS_HOME/<name>_token`` -- e.g. hf_token, nsrr_token."""
    return _paths.home() / f"{name}_token"


def locate_token(name: str) -> Optional[str]:
    """Where a token was found (a variable name or a path), never its value."""
    if name == "hf":
        if os.environ.get("HF_TOKEN"):
            return "$HF_TOKEN"
        candidates = [os.environ.get("NEUROATLAS_HF_TOKEN_FILE"), token_file("hf")]
        checkout = _paths.checkout_root()
        if checkout:
            candidates.append(checkout / ".secrets" / "hf_token")
        candidates.append(Path.home() / ".cache" / "huggingface" / "token")
    elif name == "nsrr":
        if os.environ.get("NSRR_TOKEN"):
            return "$NSRR_TOKEN"
        candidates = [token_file("nsrr")]
    elif name == "github":
        # read by models download for private release assets (_checkpoint_download)
        for var in ("GITHUB_TOKEN", "GH_TOKEN"):
            if os.environ.get(var):
                return f"${var}"
        candidates = [token_file("github")]
    else:
        raise KeyError(name)
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and Path(candidate).read_text().strip():
            return str(candidate)
    return None


def token_permissions_ok(path: Path) -> bool:
    """False when a token file is readable by group or others."""
    try:
        return (path.stat().st_mode & 0o077) == 0
    except OSError:
        return True
