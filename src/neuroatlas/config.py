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


# The order here is the order `config show` prints. <workspace> is the source
# checkout when running from one, else $NEUROATLAS_HOME.
SETTINGS: Dict[str, Setting] = {s.key: s for s in (
    Setting("data_root", "EEG_DATA_ROOT", lambda: None,
            "raw datasets, one sub-folder each"),
    Setting("cache_root", "EEG_CACHE_ROOT", lambda: _paths.artifacts_dir("embedding_cache"),
            "saved embeddings; can grow large", "<workspace>/artifacts/embedding_cache"),
    Setting("output_root", "NEUROATLAS_OUTPUT_ROOT", lambda: _paths.artifacts_dir("benchmarks"),
            "probe results (results.json, results.csv)", "<workspace>/artifacts/benchmarks"),
    Setting("models_root", "NEUROATLAS_MODELS_ROOT", lambda: _paths.artifacts_dir("models"),
            "model weights", "<workspace>/artifacts/models"),
)}

# Keys a config file may hold besides the roots.
_OTHER_KEYS = {"dataset_paths"}


class ConfigError(SystemExit):
    """A config file that cannot be used as written.

    A SystemExit carrying its message, like ``MissingDatasetPath``: raised
    from ``python -m ...`` it prints and exits 1; the ``neuroatlas`` command
    catches it and exits 2, the usage-error status.
    """


def config_path() -> Path:
    return _paths.home() / CONFIG_NAME


def load_file(path: Optional[Path] = None) -> Dict[str, Any]:
    """The config file as written, or ``{}`` when there is none."""
    import yaml

    path = path or config_path()
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"error: {path} is not valid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"error: {path} must be a mapping of settings, got {type(data).__name__}")
    unknown = sorted(set(data) - set(SETTINGS) - _OTHER_KEYS)
    if unknown:
        import difflib

        hints = []
        for key in unknown:
            close = difflib.get_close_matches(key, [*SETTINGS, *_OTHER_KEYS], n=1)
            hints.append(f"{key} (did you mean {close[0]}?)" if close else key)
        raise ConfigError(f"error: {path} has unknown settings: {', '.join(hints)}")
    paths = data.get("dataset_paths") or {}
    if not isinstance(paths, dict) or not all(isinstance(v, dict) for v in paths.values()):
        raise ConfigError(
            f"error: {path}: dataset_paths must map a dataset to its keys, e.g.\n"
            f"  dataset_paths:\n    ucddb:\n      data_root: /mnt/ucddb")
    return data


def save_file(data: Dict[str, Any], path: Optional[Path] = None) -> Path:
    import yaml

    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {k: data[k] for k in [*SETTINGS, *sorted(_OTHER_KEYS)] if data.get(k) not in (None, {}, "")}
    path.write_text(yaml.safe_dump(ordered, sort_keys=False, default_flow_style=False))
    return path


# Variables this process exported from the file (apply_to_environ) or set to
# go offline (set_offline), so reports can tell them from the user's own.
_EXPORTED: set = set()
_OFFLINE_SET: set = set()


@dataclass(frozen=True)
class Resolved:
    key: str
    value: Optional[Path]
    origin: str          # "env", "file", "default" or "unset"


def resolve(key: str, data: Optional[Dict[str, Any]] = None) -> Resolved:
    """One setting, with where its value came from."""
    setting = SETTINGS[key]
    env = os.environ.get(setting.env)
    # A variable apply_to_environ exported came from the file, not the user.
    if env and setting.env not in _EXPORTED:
        return Resolved(key, Path(env).expanduser(), "env")
    data = load_file() if data is None else data
    if data.get(key):
        return Resolved(key, Path(str(data[key])).expanduser(), "file")
    default = setting.default()
    return Resolved(key, default, "default" if default is not None else "unset")


def get(key: str) -> Optional[Path]:
    return resolve(key).value


def dataset_paths(slug: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Per-dataset path keys from the file, e.g. ``{"data_root": "/mnt/ucddb"}``."""
    data = load_file() if data is None else data
    block = (data.get("dataset_paths") or {}).get(slug) or {}
    return {k: str(Path(str(v)).expanduser()) for k, v in block.items()}


def apply_to_environ(data: Optional[Dict[str, Any]] = None) -> List[str]:
    """Export file values into the variables the engine reads, if unset.

    Returns the variables it set. Idempotent, and never overrides a variable
    the user or a job script already exported.
    """
    data = load_file() if data is None else data
    exported = []
    for setting in SETTINGS.values():
        if setting.env in _EXPORTED:
            # Ours from an earlier call: follow the file, which may have changed.
            if data.get(setting.key):
                os.environ[setting.env] = str(Path(str(data[setting.key])).expanduser())
            else:
                os.environ.pop(setting.env, None)
                _EXPORTED.discard(setting.env)
            continue
        if os.environ.get(setting.env) or not data.get(setting.key):
            continue
        os.environ[setting.env] = str(Path(str(data[setting.key])).expanduser())
        _EXPORTED.add(setting.env)
        exported.append(setting.env)
    return exported


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
