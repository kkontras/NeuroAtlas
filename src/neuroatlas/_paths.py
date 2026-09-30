"""Where NeuroAtlas finds its own tables and puts the user's files.

Two kinds of location, and they must not be confused:

* **Shipped tables** -- cohort manifests, channel maps, fold manifests, task
  presets, model groups -- live inside the package (``neuroatlas/configs``)
  and are the same for every user. They are found relative to this file, so
  they work from a checkout and from an installed wheel alike.

* **User files** -- model weights, embedding caches, preprocessed corpora,
  results -- are large, machine-specific and never shipped. In a checkout they
  default to ``<checkout>/artifacts`` and ``<checkout>/data``, which is where
  every existing run already put them. In an installed package there is no
  checkout, so they default to ``$NEUROATLAS_HOME`` (``~/.neuroatlas``).

Every module that used to count ``Path(__file__).parents[N]`` up to the repo
root asks here instead, so moving a file can no longer silently re-point it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = PACKAGE_ROOT / "configs"


def configs_dir(*parts: str) -> Path:
    """A shipped table, e.g. ``configs_dir("channel_maps")``."""
    return CONFIGS_DIR.joinpath(*parts)


def checkout_root() -> Optional[Path]:
    """The source checkout this package runs from, or None when installed.

    ``src/neuroatlas`` -> ``src`` -> checkout. Recognised by its
    ``pyproject.toml``, so a wheel under site-packages is never mistaken for
    one.
    """
    candidate = PACKAGE_ROOT.parents[1]
    if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
        return candidate
    return None


def home() -> Path:
    """``$NEUROATLAS_HOME``, default ``~/.neuroatlas``: config, tokens, caches."""
    return Path(os.environ.get("NEUROATLAS_HOME") or Path.home() / ".neuroatlas").expanduser()


def workspace_root() -> Path:
    """The folder user files hang off: the checkout if there is one, else home."""
    return checkout_root() or home()


def artifacts_dir(*parts: str) -> Path:
    """``<workspace>/artifacts/...`` -- model weights, embedding caches."""
    return workspace_root().joinpath("artifacts", *parts)


def models_dir(*parts: str) -> Path:
    """Model weights. ``$NEUROATLAS_MODELS_ROOT`` overrides the default."""
    override = os.environ.get("NEUROATLAS_MODELS_ROOT")
    base = Path(override).expanduser() if override else artifacts_dir("models")
    return base.joinpath(*parts)


def output_dir(*parts: str) -> Path:
    """Probe results. ``$NEUROATLAS_OUTPUT_ROOT`` overrides the default."""
    override = os.environ.get("NEUROATLAS_OUTPUT_ROOT")
    base = Path(override).expanduser() if override else artifacts_dir("benchmarks")
    return base.joinpath(*parts)


def data_dir(*parts: str) -> Path:
    """``<workspace>/data/...`` -- corpora a builder writes next to the code."""
    return workspace_root().joinpath("data", *parts)
