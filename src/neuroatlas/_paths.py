"""Where NeuroAtlas finds its own tables and puts the user's files.

Two kinds of location, and they must not be confused:

* **Shipped tables** -- cohort manifests, channel maps, fold manifests, task
  presets, model groups -- live inside the package (``neuroatlas/configs``)
  and are the same for every user. They are found relative to this file, so
  they work from a checkout and from an installed wheel alike.

* **User files** -- model weights, embedding caches, prepared corpora,
  results -- are large, machine-specific and never shipped. Every one of them
  goes under one of the four roots of :mod:`neuroatlas.config` (data, cache,
  output, models), and the defaults of those roots hang off
  ``$NEUROATLAS_HOME`` (``~/.neuroatlas``) and nothing else.

  They used to default to ``<checkout>/artifacts`` whenever the package was
  imported from a source checkout. That made the defaults depend on *how the
  package was installed* (an editable install and a wheel, with the same
  config, wrote to different folders) and let two different
  ``NEUROATLAS_HOME`` share one checkout's caches. :func:`legacy_locations`
  finds what an earlier version left in a checkout, so ``config show`` and
  ``data status`` can say how to keep using it.

Every module that used to count ``Path(__file__).parents[N]`` up to the repo
root asks here instead, so moving a file can no longer silently re-point it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = PACKAGE_ROOT / "configs"


def configs_dir(*parts: str) -> Path:
    """A shipped table, e.g. ``configs_dir("channel_maps")``."""
    return CONFIGS_DIR.joinpath(*parts)


def checkout_root() -> Optional[Path]:
    """The source checkout this package runs from, or None when installed.

    ``src/neuroatlas`` -> ``src`` -> checkout. Recognised by its
    ``pyproject.toml``, so a wheel under site-packages is never mistaken for
    one. Used to report the install and to find what an earlier version
    left there -- never to decide where user files go.
    """
    candidate = PACKAGE_ROOT.parents[1]
    if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
        return candidate
    return None


def _absolute(path: Path) -> Path:
    """A path made absolute against the current directory, ``~`` expanded."""
    path = path.expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def home() -> Path:
    """``$NEUROATLAS_HOME``, default ``~/.neuroatlas``: config, tokens, defaults."""
    return _absolute(Path(os.environ.get("NEUROATLAS_HOME") or Path.home() / ".neuroatlas"))


def workspace_root() -> Path:
    """The folder default user files hang off: always ``$NEUROATLAS_HOME``.

    The same for an editable install, a wheel and any working directory, so
    a setting never moves because of how the package was installed.
    """
    return home()


def artifacts_dir(*parts: str) -> Path:
    """``$NEUROATLAS_HOME/artifacts/...`` -- the default cache, output and models roots."""
    return workspace_root().joinpath("artifacts", *parts)


def models_dir(*parts: str) -> Path:
    """Model weights. ``$NEUROATLAS_MODELS_ROOT`` overrides the default."""
    override = os.environ.get("NEUROATLAS_MODELS_ROOT")
    base = _absolute(Path(override)) if override else artifacts_dir("models")
    return base.joinpath(*parts)


def output_dir(*parts: str) -> Path:
    """Probe results. ``$NEUROATLAS_OUTPUT_ROOT`` overrides the default."""
    override = os.environ.get("NEUROATLAS_OUTPUT_ROOT")
    base = _absolute(Path(override)) if override else artifacts_dir("benchmarks")
    return base.joinpath(*parts)


def prepared_dir(*parts: str) -> Path:
    """``<cache root>/prepared/...`` -- what `data prepare` builds.

    Prepared corpora (the BCI pickles, the epilepsy HDF5 fast paths) are
    derived, rebuildable and machine-specific, like embeddings, so they live
    under the cache root rather than next to the raw data, which is often a
    read-only shared copy.
    """
    from neuroatlas import config

    return Path(config.get("cache_root")).joinpath("prepared", *parts)


def recording_stats_dir(*parts: str) -> Path:
    """``<cache root>/recording_stats/...`` -- per-recording amplitude
    statistics (mean, std, q95) the epilepsy readers compute once per
    recording for the models that normalise by them.

    Derived and rebuildable like the embeddings beside them. They used to go
    to ``artifacts/recording_stats`` under whatever directory the command was
    started in; ``_recording_stats.load_cached_recording_stats`` still reads
    those (and a checkout's) and copies a hit here.
    """
    from neuroatlas import config

    return Path(config.get("cache_root")).joinpath("recording_stats", *parts)


def data_dir(*parts: str) -> Path:
    """Deprecated alias of :func:`prepared_dir` (it used to be ``<checkout>/data``)."""
    return prepared_dir(*parts)


# What an earlier version wrote into a source checkout, by default:
# (what, path relative to the checkout, the setting that now replaces it).
_LEGACY = (
    ("model weights", "artifacts/models", "models_root"),
    ("embedding cache", "artifacts/embedding_cache", "cache_root"),
    ("results", "artifacts/benchmarks", "output_root"),
    ("prepared BCI corpora", "data/preprocessed", "cache_root"),
)


def legacy_locations() -> List[Tuple[str, Path, str]]:
    """``(what, path, setting)`` for every legacy default that exists and is not empty."""
    checkout = checkout_root()
    if checkout is None:
        return []
    found = []
    for what, rel, setting in _LEGACY:
        path = checkout / rel
        try:
            if path.is_dir() and any(path.iterdir()):
                found.append((what, path, setting))
        except OSError:
            continue
    return found


def legacy_raw(slug: str) -> Optional[Path]:
    """``<checkout>/<slug>_cache/raw``, where an earlier version downloaded the
    Zenodo epilepsy cohorts, when it exists."""
    checkout = checkout_root()
    if checkout is None:
        return None
    path = checkout / f"{slug}_cache" / "raw"
    return path if path.is_dir() else None
