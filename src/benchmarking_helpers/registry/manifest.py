"""Dataset manifests — the single home for per-dataset facts.

A manifest lives at ``configs/cohorts/<slug>/cohort.yaml`` and is described
by ``configs/cohorts/_schema.yaml``.  It holds everything factual about a cohort: where
it comes from, what the signal is, which label modes exist, how it is split,
and which shared machinery it needs.

This module turns a manifest into the :class:`DatasetSpec` the registry already
uses, so nothing downstream has to change.  The mapping is deliberately dull:

    manifest["runtime_defaults"]      -> DatasetSpec.config_defaults
    manifest["spec"]["default_task"]  -> DatasetSpec.default_task
    manifest["signal"]["notch_freq"]  -> DatasetSpec.notch_freq
    manifest (whole)                  -> DatasetSpec.manifest

Why YAML rather than more Python: a manifest is data, and data should be
reviewable in a diff by someone who does not read Python.  The one thing a
manifest cannot hold is the datamodule callable, which is why
:func:`dataset_spec_from_manifest` takes it as an argument.

``load_manifest`` is cached and does not import torch, so
``docs datasets`` and the completeness tests stay cheap.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .contracts import DatasetSpec


class ManifestError(RuntimeError):
    """Raised when a manifest is missing, malformed, or inconsistent."""


def cohorts_root() -> Path:
    """Where the dataset dossiers live: ``configs/cohorts/``.

    Resolved relative to this file so it works from an installed package and
    from a source checkout alike.
    """
    return Path(__file__).resolve().parents[3] / "configs" / "cohorts"


def available_manifests(root: Optional[Path] = None) -> List[str]:
    """Every slug that has a manifest, sorted."""
    root = root or cohorts_root()
    if not root.is_dir():
        return []
    return sorted(
        p.parent.name
        for p in root.glob("*/cohort.yaml")
        if not p.parent.name.startswith("_")
    )


@lru_cache(maxsize=None)
def load_manifest(slug: str, root: Optional[Path] = None) -> Dict[str, Any]:
    """Read and validate ``configs/cohorts/<slug>/cohort.yaml``."""
    root = root or cohorts_root()
    path = root / slug / "cohort.yaml"
    if not path.is_file():
        raise ManifestError(
            f"No manifest for {slug!r} at {path}. "
            f"Known: {', '.join(available_manifests(root)) or '(none)'}"
        )

    import yaml  # local import: yaml is a soft dependency, as in channel_map.py

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ManifestError(f"{path}: top level must be a mapping")

    declared = data.get("slug")
    if declared != slug:
        raise ManifestError(
            f"{path}: declares slug={declared!r} but lives in directory {slug!r}. "
            "The directory name is authoritative; fix the file."
        )

    for required in ("name", "domain", "description", "cohort", "acquisition",
                     "signal", "labels", "splits"):
        if required not in data:
            raise ManifestError(f"{path}: missing required section {required!r}")

    labels = data["labels"]
    default_mode = labels.get("default")
    modes = labels.get("modes") or {}
    if default_mode not in modes:
        raise ManifestError(
            f"{path}: labels.default={default_mode!r} is not among "
            f"labels.modes ({', '.join(sorted(modes)) or 'none'})"
        )

    grouping = data["splits"].get("grouping")
    if grouping not in ("patient", "recording", "clip", "session"):
        raise ManifestError(
            f"{path}: splits.grouping={grouping!r} must be one of "
            "patient | recording | clip | session"
        )

    data.setdefault("_path", str(path))
    return data


def fold_manifest_path(slug: str, root: Optional[Path] = None) -> Optional[Path]:
    """Absolute path to this dataset's fold manifest, if it declares one."""
    manifest = load_manifest(slug, root)
    rel = (manifest.get("splits") or {}).get("manifest")
    if not rel:
        return None
    return (root or cohorts_root()) / slug / rel


def dataset_spec_from_manifest(
    slug: str,
    datamodule_cls: Callable[..., Any],
    *,
    root: Optional[Path] = None,
    **overrides: Any,
) -> DatasetSpec:
    """Build the registry's :class:`DatasetSpec` from a manifest.

    Args:
        slug: directory name under ``datasets/``.
        datamodule_cls: the callable the spec constructs — the one thing a
            YAML manifest cannot carry.
        overrides: escape hatch for a spec field a manifest cannot yet express.
            Using it is a sign the schema needs a new field; prefer that.
    """
    manifest = load_manifest(slug, root)
    spec_block = manifest.get("spec") or {}

    kwargs: Dict[str, Any] = {
        "slug": manifest["slug"],
        "description": " ".join(manifest["description"].split()),
        "datamodule_cls": datamodule_cls,
        "config_defaults": dict(manifest.get("runtime_defaults") or {}),
        "metadata_keys": tuple(spec_block.get("metadata_keys") or ()),
        "supports_folds": bool(spec_block.get("supports_folds", False)),
        "default_task": spec_block.get("default_task", "linear_probe"),
        "input_kind_signal_map": dict(spec_block.get("input_kind_signal_map") or {}),
        # NOT cohort["powerline_hz"].  powerline_hz is a *fact* about the
        # recording site; notch_freq is *behaviour* — DatasetSpec.build_config
        # injects `notch` and `highpass` into every run when it is set.  Most
        # epilepsy specs deliberately leave it unset today, so deriving it from
        # the cohort would silently change their preprocessing.  Unifying the
        # two is a deliberate follow-up, gated on audit OQ-3.2.
        "notch_freq": (manifest.get("signal") or {}).get("notch_freq"),
        "manifest": manifest,
    }
    kwargs.update(overrides)
    return DatasetSpec(**kwargs)
