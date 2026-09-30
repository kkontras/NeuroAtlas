"""Whether each checkpoint's weights are on this machine, and fetching them.

States, as `models status` prints them:

    found           the weights are on disk
    auto            not here; `models download` can fetch them
    hub             a Hugging Face model loaded through a hub cache, not cached yet
    hub (cached)    ... already in the cache
    manual          must be fetched by hand; the note says from where
    nothing needed  an untrained baseline
    planned         the wrapper is not ready

Only :func:`download` uses the network.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from neuroatlas import _paths

# source types ensure_checkpoint can fetch into a local path
AUTO_SOURCES = {"github_release", "github_release_asset", "huggingface", "figshare_private_share"}

# families that load straight from the hub, with the cache each one uses
_HUB_CACHE_FAMILIES = {"chronos", "moirai", "moment"}


@dataclass
class ModelStatus:
    identifier: str
    family: str
    source: str
    state: str
    path: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.state in ("found", "hub (cached)", "nothing needed")


def _is_hub_id(value: Optional[str]) -> bool:
    return bool(value) and not Path(value).is_absolute() and value.count("/") == 1 \
        and not value.startswith(("artifacts", "src"))


def hub_cache_dirs(family: str) -> List[Path]:
    dirs = []
    if family in _HUB_CACHE_FAMILIES:
        dirs.append(_paths.models_dir("foundation", "huggingface_cache"))
    hub = os.environ.get("HF_HUB_CACHE") or (
        Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")
    dirs.append(Path(hub))
    return dirs


def hub_cached(repo_id: str, family: str) -> bool:
    folder = "models--" + repo_id.replace("/", "--")
    for d in hub_cache_dirs(family):
        snapshots = d / folder / "snapshots"
        if snapshots.is_dir() and any(snapshots.iterdir()):
            return True
    return False


def status(spec) -> ModelStatus:
    st = ModelStatus(spec.identifier, spec.model_family, spec.source_type, "manual",
                     spec.checkpoint_path)
    if spec.status != "ready":
        st.state = "planned"
        st.notes.append(spec.notes or "wrapper not ready")
        return st
    if spec.source_type == "random_init":
        st.state = "nothing needed"
        return st
    path = spec.checkpoint_path
    if _is_hub_id(path):
        st.state = "hub (cached)" if hub_cached(path, spec.model_family) else "hub"
        return st
    if path and Path(path).exists():
        st.state = "found"
        return st
    if spec.source_type in AUTO_SOURCES:
        st.state = "auto"
    else:
        st.state = "manual"
        ref = spec.source_reference
        st.notes.append(f"from {ref}" if ref and ref != path else "no download source recorded")
    return st


def download(spec) -> ModelStatus:
    """Fetch one checkpoint. Raises on failure with the reason."""
    st = status(spec)
    if st.ready or st.state in ("planned", "manual"):
        return st
    if st.state == "hub":
        from huggingface_hub import snapshot_download

        cache = hub_cache_dirs(spec.model_family)[0]
        snapshot_download(spec.checkpoint_path,
                          cache_dir=str(cache) if spec.model_family in _HUB_CACHE_FAMILIES else None)
        return status(spec)
    from neuroatlas.extensions.models.backbones._checkpoint_download import ensure_checkpoint

    ensure_checkpoint(spec.checkpoint_path, spec.source_type, spec.source_reference)
    return status(spec)
