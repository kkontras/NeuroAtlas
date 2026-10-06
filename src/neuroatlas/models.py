"""Whether each checkpoint's weights are on this machine, and fetching them.

States, as `models status` prints them:

    found            the weights are on disk
    auto             not here; `models download` can fetch them
    hub              a Hugging Face model loaded through a hub cache, not cached yet
    hub (cached)     ... already in the cache
    manual           must be fetched by hand; the note says from where and how
    nothing needed   an untrained baseline that needs no files
    package missing  the Python package the wrapper imports is not installed;
                     the note gives the command (weights are reported in the note)
    planned          the wrapper is not ready

Only :func:`download` uses the network.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from neuroatlas import _paths

# source types ensure_checkpoint can fetch into a local path
AUTO_SOURCES = {"github_release", "github_release_asset", "github_commit_files", "huggingface",
                "google_drive_zip", "docker_image"}

# families that load straight from the hub, with the cache each one uses
_HUB_CACHE_FAMILIES = {"chronos", "moirai", "moment"}

# The package each time-series family imports, and the install that works next
# to the rest of the stack (the `ts` extra holds both). momentfm pins
# transformers==4.33.3 and numpy==1.25.2 on every release, so a plain
# `pip install momentfm` downgrades the stack every other model uses; it runs
# with the current one, so it goes in without its dependencies. Chronos 1.5.2+
# accepts transformers 4.48-4.x (1.5.0, the TS-FM freeze, wanted <4.48).
# uni2ts (Moirai) needs torch<2.5 and scipy~=1.11: it cannot share this
# environment at all.
# family -> (module imported, distribution name, install command)
PACKAGES = {
    "moment": ("momentfm", "momentfm", 'pip install --no-deps "momentfm==0.1.4"'),
    "chronos": ("chronos", "chronos-forecasting", 'pip install "chronos-forecasting>=1.5.2,<2"'),
    "moirai": ("uni2ts", "uni2ts", "Moirai runs in its own environment: see requirements-tsfm.txt "
                                   "(Python 3.10, torch 2.4.1, jax)"),
}

# Per-checkpoint instructions for what `models download` cannot do on its own,
# and for licences a download accepts.
_NOTES = {
    "reve_pretrained": "REVE Responsible Use License v1.0 (https://huggingface.co/brain-bzh/reve-base): "
                       "downloading it accepts the licence",
    "core_sleep_shhs_fold0": "a private release asset: needs a GitHub token that can read "
                             "kkontras/NeuroAtlas (neuroatlas config token github)",
    "seizure_transformer_pretrained": "pulled out of the authors' Docker image yujjio/seizure_transformer "
                                      "(streams up to 3.4 GB of image layers; keeps the 168 MB model.pth)",
    "eegpt_pretrained": "fetched from the bit-identical hub copy eeg-telecom-paris/eegpt-large-official "
                        "(upstream's Figshare share is browser-only)",
}
_NOTES["core_sleep_shhs_fold0_seq1"] = _NOTES["core_sleep_shhs_fold0"]
_NOTES["sleep_transformer_shhs_fold0"] = _NOTES["core_sleep_shhs_fold0"]
_NOTES["sleep_transformer_shhs_fold0_seq1"] = _NOTES["core_sleep_shhs_fold0"]


@dataclass
class ModelStatus:
    identifier: str
    family: str
    source: str
    state: str
    path: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    weights: Optional[str] = None     # the weights' own state, when `state` is about the package

    @property
    def ready(self) -> bool:
        return self.state in ("found", "hub (cached)", "nothing needed")

    @property
    def fetchable(self) -> bool:
        """`models download` has something to fetch."""
        return (self.weights or self.state) in ("auto", "hub")


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


_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".msgpack", ".h5")


def hub_cached(repo_id: str, family: str) -> bool:
    """A cached snapshot of *repo_id* that holds weights, not only a config.

    An interrupted download or a config-only fetch (AutoConfig) leaves a
    snapshot folder without them, which used to count as cached.
    """
    folder = "models--" + repo_id.replace("/", "--")
    for d in hub_cache_dirs(family):
        snapshots = d / folder / "snapshots"
        if not snapshots.is_dir():
            continue
        for snap in snapshots.iterdir():
            if any(f.suffix in _WEIGHT_SUFFIXES and f.is_file() for f in snap.rglob("*")):
                return True
    return False


def missing_package(family: str) -> Optional[str]:
    """The distribution a family needs, when its module is not importable."""
    entry = PACKAGES.get(family)
    if entry is None:
        return None
    try:
        found = importlib.util.find_spec(entry[0]) is not None
    except (ImportError, ValueError):
        found = False
    return None if found else entry[1]


def _manual_note(spec) -> str:
    ref, path = spec.source_reference, spec.checkpoint_path
    if ref and ref != path:
        return f"get it from {ref} and place it at {path}"
    return f"no public source; place the file at {path} if you have it"


def _weights_status(spec, st: ModelStatus) -> None:
    from neuroatlas.extensions.models.backbones._checkpoint_download import (
        missing_files, reve_sources,
    )

    if spec.model_family == "reve":
        # REVE loads from the models root, or from a complete snapshot in the
        # Hugging Face cache, which the wrapper reads offline too.
        random_init = spec.source_type == "random_init"
        sources = reve_sources(spec.checkpoint_path, random_init)
        if not sources.lacking:
            st.state = ("nothing needed" if random_init
                        else "hub (cached)" if sources.from_hub_cache else "found")
            if sources.from_hub_cache:
                st.notes.append("read from the Hugging Face cache "
                                + ", ".join(s for s in (sources.model, sources.positions)
                                            if "/" in s and not Path(s).is_absolute()))
            return
        st.state = "auto"
        if random_init:
            st.notes.append("needs REVE's config and code (no weights) and its position bank "
                            "from huggingface.co")
        elif Path(spec.checkpoint_path).is_dir():
            base = Path(spec.checkpoint_path).parent
            st.notes.append("incomplete: missing " + ", ".join(
                str(Path(p).relative_to(base)) for p in sources.lacking))
        return
    if spec.source_type == "random_init":
        st.state = "nothing needed"
        return
    path = spec.checkpoint_path
    if _is_hub_id(path):
        st.state = "hub (cached)" if hub_cached(path, spec.model_family) else "hub"
        return
    lacking = missing_files(path, spec.source_type, spec.source_reference)
    if path and not lacking:
        st.state = "found"
        return
    if spec.source_type in AUTO_SOURCES:
        st.state = "auto"
        if path and Path(path).is_dir() and lacking:
            base = Path(path).parent
            st.notes.append("incomplete: missing " + ", ".join(
                str(Path(p).relative_to(base)) if Path(p).is_relative_to(base) else p for p in lacking))
    else:
        st.state = "manual"
        st.notes.append(_manual_note(spec))


def status(spec) -> ModelStatus:
    st = ModelStatus(spec.identifier, spec.model_family, spec.source_type, "manual",
                     spec.checkpoint_path)
    if spec.status != "ready":
        st.state = "planned"
        st.notes.append(spec.notes or "wrapper not ready")
        return st
    _weights_status(spec, st)
    if spec.identifier in _NOTES and not st.ready:
        st.notes.append(_NOTES[spec.identifier])
    if spec.source_type == "github_commit_files":
        # code and weights under a licence the package does not ship (DeepSOZ:
        # GPL-3.0): named whatever the state, since what runs is that code
        from neuroatlas.extensions.models.backbones._checkpoint_download import commit_files_note

        note = commit_files_note(spec.source_reference)
        if note:
            st.notes.append(note)
    package = missing_package(spec.model_family)
    if package:
        st.weights, st.state = st.state, "package missing"
        st.notes.insert(0, f"{package} is not installed: {PACKAGES[spec.model_family][2]}"
                           f" (weights: {st.weights})")
    return st


#: Checkpoints whose download needs a GitHub token (private release assets).
_GITHUB_TOKEN_NEEDED = ("core_sleep_shhs_fold0", "core_sleep_shhs_fold0_seq1",
                        "sleep_transformer_shhs_fold0", "sleep_transformer_shhs_fold0_seq1")


def weights_problem(st: ModelStatus) -> Optional[tuple]:
    """``(what, fix)`` when this checkpoint cannot be loaded here, else None:
    what `check`, `run` and `submit` say for a pair they do not run."""
    if st.ready:
        return None
    if st.state == "package missing":
        dist, command = PACKAGES[st.family][1], PACKAGES[st.family][2]
        return f"{dist} is not installed", command
    state = getattr(st, "weights", None) or st.state
    incomplete = next((n for n in st.notes if n.startswith("incomplete")), None)
    if state in ("auto", "hub"):
        what = "weights not downloaded" + (f" ({incomplete})" if incomplete else "")
        ident = getattr(st, "identifier", None) or "<checkpoint>"
        fix = f"neuroatlas models download {ident}"
        if ident in _GITHUB_TOKEN_NEEDED:
            fix = f"neuroatlas config token github, then {fix} (a private release asset)"
        return what, fix
    if state == "manual":
        manual = next((n for n in st.notes if n.startswith(("get it from", "no public source"))),
                      None)
        return "weights not here (no automatic download)", manual
    if state == "planned":
        return "the wrapper is not ready", None
    return f"weights {state}", None


def download(spec) -> ModelStatus:
    """Fetch one checkpoint. Raises on failure with the reason.

    A checkpoint whose package is missing still gets its weights; the returned
    status keeps saying `package missing` until the package is installed.
    """
    st = status(spec)
    if not st.fetchable:
        return st
    if spec.source_type == "random_init" and spec.model_family == "reve":
        from neuroatlas.extensions.models.backbones._checkpoint_download import fetch_reve_code

        fetch_reve_code(spec.checkpoint_path)
        return status(spec)
    if (st.weights or st.state) == "hub":
        from huggingface_hub import snapshot_download

        cache = hub_cache_dirs(spec.model_family)[0]
        snapshot_download(spec.checkpoint_path,
                          cache_dir=str(cache) if spec.model_family in _HUB_CACHE_FAMILIES else None)
        return status(spec)
    from neuroatlas.extensions.models.backbones._checkpoint_download import ensure_checkpoint

    ensure_checkpoint(spec.checkpoint_path, spec.source_type, spec.source_reference)
    return status(spec)
