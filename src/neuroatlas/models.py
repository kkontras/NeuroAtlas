"""Whether each checkpoint's weights are on this machine, and fetching them.

States (``ModelStatus.state``, the code) and the words `models status`,
`check` and every message print for them (:data:`STATE_WORDS`):

    found            found                   the weights are on disk
    auto             downloadable            not here; `models download` can fetch them
    hub              downloadable            a Hugging Face model, not in its cache yet
    hub (cached)     in Hugging Face cache   already in Hugging Face's cache
    manual           manual                  must be fetched by hand; the note says from where
    nothing needed   no weights needed       an untrained (random-init) baseline
    package missing  package missing         the Python package the model needs is not
                                             installed; the note gives the command (and
                                             the weights' own state)

A registry entry that is not ``ready`` is not part of the release: no
command lists or selects it (:mod:`neuroatlas.selectors`).

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

#: state code -> the word a user reads (tables, JSON, messages)
STATE_WORDS = {
    "found": "found",
    "auto": "downloadable",
    "hub": "downloadable",
    "hub (cached)": "in Hugging Face cache",
    "manual": "manual",
    "nothing needed": "no weights needed",
    "package missing": "package missing",
}

#: word -> its meaning, for the legend under `models status` and `check`
STATE_MEANINGS = {
    "found": "the weights are on this machine",
    "downloadable": "not here yet; `neuroatlas models download <checkpoint>` fetches them",
    "in Hugging Face cache": "already in Hugging Face's local cache: nothing to download",
    "manual": "must be fetched by hand; the line under the row says from where",
    "no weights needed": "an untrained (random-init) baseline: it has no weights",
    "package missing": "a Python package the model needs is not installed; the line under "
                       "the row gives the command",
}

#: source type -> where the weights come from, as `list models` and
#: `models status` print it (JSON keeps the source type)
SOURCE_WORDS = {
    "github_release": "GitHub release",
    "github_release_asset": "GitHub release",
    "github_commit_files": "GitHub repository",
    "huggingface": "Hugging Face",
    "google_drive_zip": "Google Drive",
    "docker_image": "Docker image",
    "random_init": "none (random init)",
    "local": "local file",
}


def state_word(state: Optional[str]) -> str:
    """The word for a state code (``auto`` -> ``downloadable``)."""
    return STATE_WORDS.get(str(state), str(state))


def source_word(source_type: Optional[str], family: Optional[str] = None) -> str:
    """Where a checkpoint's weights come from, in words. REVE's untrained
    control has no weights, but its config and code come from Hugging Face."""
    if source_type == "random_init" and family == "reve":
        return "Hugging Face (config only)"
    return SOURCE_WORDS.get(str(source_type), str(source_type))

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
    "moirai": ("uni2ts", "uni2ts", "pip install -r requirements-tsfm.txt (in a separate Python "
                                   "3.10 environment, from your NeuroAtlas clone: uni2ts needs "
                                   "torch 2.4.1)"),
}

# Per-checkpoint instructions for what `models download` cannot do on its own,
# and for licences a download accepts.
_NOTES = {
    "reve_pretrained": "REVE Responsible Use License v1.0 (https://huggingface.co/brain-bzh/reve-base): "
                       "downloading it accepts the licence",
    "seizure_transformer_pretrained": "pulled out of the authors' Docker image yujjio/seizure_transformer "
                                      "(streams up to 3.4 GB of image layers; keeps the 168 MB model.pth)",
    "eegpt_pretrained": "from Hugging Face (eeg-telecom-paris/eegpt-large-official), byte-identical "
                        "to the authors' Figshare release",
}


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
        # not part of this release (no command lists or selects it)
        st.state = "not in this release"
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
                           f" (weights: {state_word(st.weights)})")
    return st


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
        return what, f"neuroatlas models download {ident}"
    if state == "manual":
        manual = next((n for n in st.notes if n.startswith(("get it from", "no public source"))),
                      None)
        return "weights not here, and they must be fetched by hand", manual
    if state == "not in this release":
        return ("not a checkpoint of this release", "neuroatlas list models")
    return f"weights {state_word(state)}", None


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

        from neuroatlas.extensions.models.backbones._checkpoint_download import hub_progress

        cache = hub_cache_dirs(spec.model_family)[0]
        # the bytes on the live line of `models download` (its bars off)
        with hub_progress(cache / ("models--" + spec.checkpoint_path.replace("/", "--"))):
            snapshot_download(spec.checkpoint_path,
                              cache_dir=str(cache) if spec.model_family in _HUB_CACHE_FAMILIES
                              else None)
        return status(spec)
    from neuroatlas.extensions.models.backbones._checkpoint_download import ensure_checkpoint

    ensure_checkpoint(spec.checkpoint_path, spec.source_type, spec.source_reference)
    return status(spec)
