"""Where each dataset's corpus is, whether it is there, and how to get it.

The facts come from the cohort manifest (``acquisition``, ``runtime_defaults``)
and the user's settings (``data_root``, ``dataset_paths``) -- the same
resolution ``embed`` and ``probe`` use, so a dataset `data status` reports as
found is the one a run will read.

Everything ``data download`` writes goes under the data root (or, for
MOABB, under ``$MNE_DATA``, see :func:`neuroatlas.config.mne_data`); nothing
lands in the source tree.

Nothing here touches the network except :func:`run_download`, and that only
when asked to.
"""
from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from neuroatlas import _paths, progress
from neuroatlas import config as user_config
from neuroatlas.cli import _msg

# Keys a manifest may use for the raw corpus, most specific first (as prepare).
RAW_KEYS = ("raw_dir", "raw_root", "bids_root", "data_root")

# File types counted as a recording when a manifest does not say otherwise.
EEG_SUFFIXES = (".edf", ".bdf", ".rec", ".fif", ".set", ".vhdr", ".h5", ".hdf5", ".mat",
                ".npz", ".xdf", ".cnt", ".eeg")

# `data status` must answer in seconds even on a network filesystem holding
# a 2 TB cohort, so the count stops early and says it did.
SCAN_SECONDS = 8.0
SCAN_ENTRIES = 200_000

# How each acquisition.kind is fetched by `data download`.
HANDLERS = {
    "physionet": "physionet",
    "zenodo": "zenodo",
    "url": "url",
    "nsrr": "nsrr",
    "moabb": "moabb",
    "tuh": "manual",
    "figshare": "manual",
    "mendeley": "manual",
    "manual": "manual",
    "internal": "internal",
}

# How a dataset is obtained -- the `download` column of `data status` and
# `list datasets`, by handler (a TUH dataset, fetched by hand, is
# credentialed: it needs a signed agreement first).
DOWNLOAD = {
    "physionet": "downloadable", "zenodo": "downloadable", "url": "downloadable",
    "moabb": "downloadable", "nsrr": "credentialed", "manual": "manual",
    "internal": "from the authors", "refused": "refused", "blocked": "blocked",
}

#: What each `download` word means: the legend under `data status` and
#: `list datasets`.
DOWNLOAD_MEANINGS = {
    "downloadable": "`neuroatlas data download <dataset>` fetches it; no account needed",
    "credentialed": "needs an approved account or a signed agreement first (NSRR: then "
                    "`data download` fetches it with your token; TUH: by hand)",
    "manual": "no download API: `neuroatlas data download <dataset>` says where to get it "
              "and where to put it",
    "from the authors": "the benchmark reads the authors' own copy: ask the authors; "
                        "`neuroatlas data download <dataset>` says what to ask for and where "
                        "to put it",
    "refused": "`data download` does not fetch it: the benchmark reads a different copy "
               "(`neuroatlas data status <dataset>` says which)",
    "blocked": "`data download` cannot plan it: no data root is set "
               "(`neuroatlas config set data_root DIR`)",
}

#: Where a dataset is served, by acquisition kind (`host` column); a manual
#: dataset shows the host of its landing page.
HOSTS = {
    "physionet": "PhysioNet", "zenodo": "Zenodo", "url": "web", "moabb": "MOABB",
    "nsrr": "NSRR", "tuh": "TUH", "figshare": "figshare", "mendeley": "Mendeley Data",
    "internal": "the authors",
}

# Kept for callers of the old column; `access` is now the manifest's kind,
# the same word `list datasets` prints.
ACCESS = {kind: kind for kind in HANDLERS}


def download_word(kind: str, handler: str) -> str:
    """The `download` word of a dataset: how `data download` gets it."""
    if kind == "tuh" and handler == "manual":
        return "credentialed"
    return DOWNLOAD.get(handler, handler)


#: A landing page's host name, as the `host` column words it (the HOSTS words).
KNOWN_HOSTS = {"zenodo.org": "Zenodo", "physionet.org": "PhysioNet",
               "figshare.com": "figshare", "data.mendeley.com": "Mendeley Data",
               "sleepdata.org": "NSRR"}


def host_word(kind: str, acq: Optional[Dict[str, Any]] = None) -> str:
    """Where a dataset is served (``PhysioNet``, ``MOABB``, ``sleeptight.isr.uc.pt``)."""
    if kind in HOSTS:
        return HOSTS[kind]
    upstream = str((acq or {}).get("upstream") or "")
    host = urllib.parse.urlparse(upstream).hostname if "://" in upstream else None
    host = host[4:] if host and host.startswith("www.") else host
    return KNOWN_HOSTS.get(host or "", host or "-")


def origin_text(kind: Optional[str], acq: Dict[str, Any], slug: str = "") -> Optional[str]:
    """Where a dataset comes from, for `list datasets -v`: its web page (the
    manifest's landing page, else one built from the record id), or its
    MOABB dataset; None when nothing is recorded."""
    ref, version = acq.get("ref"), acq.get("version")
    if acq.get("upstream"):
        return str(acq["upstream"])
    if kind == "physionet" and ref:
        return f"https://physionet.org/content/{ref}/" + (f"{version}/" if version else "")
    if kind == "zenodo" and ref:
        return f"https://zenodo.org/records/{ref}"
    if kind == "nsrr" and ref:
        return f"https://sleepdata.org/datasets/{ref}"
    if kind == "mendeley" and ref:
        return f"https://doi.org/{ref}"
    if kind == "moabb":
        from neuroatlas.extensions.datasets.dataio.moabb_loader import MOABB_DATASETS

        cfg = MOABB_DATASETS.get(slug)
        return f"MOABB {cfg.moabb_name}" if cfg else (f"MOABB {ref}" if ref else "MOABB")
    return None


# Every state `data status` can report, with its meaning: printed under the
# table and the vocabulary the guide should list.
STATES = {
    "found": "every expected file is there (n files, or n/N when the total is known)",
    "partial": "fewer files than a complete copy (n/N): a download that stopped part-way; "
               "running it again finishes it",
    "empty": "the folder exists but holds no recordings",
    "missing": "the folder does not exist",
    "prepared": "the prepared file this dataset is read from is there",
    "not prepared": "the prepared file this dataset is read from is not there "
                    "(the line under the row says where it comes from)",
    "not downloaded": "not in the MOABB data folder ($MNE_DATA) yet",
    "not configured": "no data root is set: `neuroatlas config set data_root DIR`",
    "no path": "no folder is set for this dataset: point it at your copy "
               "(`neuroatlas config set <dataset>.data_root DIR`)",
    "package missing": "a Python package needed to check it is not installed; the line "
                       "under the row gives the command",
}

#: A folder `data download --first N` filled: enough for `check`, not for `run`.
STATES["sample"] = ("only the first recordings (`data download --first N`): enough for "
                    "`neuroatlas check`, not for `run`")


#: The note a dataset whose download needs an NSRR token carries when there is
#: none (`data status` gathers them into one line under the table).
NO_NSRR_TOKEN = "no NSRR token"
#: The note of a dataset neuroatlas has no folder for.
NO_FOLDER = "no folder is set for this dataset: point it at your copy"

#: The one install line for MOABB (the BCI datasets), as the README gives it.
MOABB_INSTALL = ('pip install -e ".[fm,bci]" -c requirements-fm.txt && '
                 'pip install --no-deps "moabb==1.2.0"')
MOABB_WHERE = "in your NeuroAtlas clone"


def manifest_text(value: Any) -> str:
    """A manifest's message (``unusable_download``, ``prepared_only``) as a
    message: what, then the ``fix:`` it gives on its own line."""
    flat = " ".join(str(value).split())
    what, sep, fix = flat.partition(" fix: ")
    return _msg.compose(what.strip(), fix.strip() if sep else None)


def _note_lines(text: str, fix: Optional[str] = None) -> List[str]:
    """A row note: its lines, then ``  fix: ...`` (no kind word: a dataset's
    state is not an error)."""
    _, lines, fixes = _msg.split(_msg.compose(text, fix))
    return lines + [f"  fix: {f}" for f in fixes]


def _spec(slug: str):
    from neuroatlas.benchmarking_helpers.registry.discovery import load_dataset_spec

    return load_dataset_spec(slug)


def _manifest(slug: str) -> Dict[str, Any]:
    return dict(_spec(slug).manifest or {})


def resolved_config(slug: str) -> Dict[str, Any]:
    """Manifest defaults + backend defaults + config.yaml, with ${VAR} expanded."""
    from neuroatlas.entrypoints._common import expand_dataset_paths

    spec = _spec(slug)
    manifest = spec.manifest or {}
    merged = dict(spec.config_defaults)
    chosen = merged.get("backend")
    for name, cfg in (manifest.get("backends") or {}).items():
        if name == chosen:
            merged.update((cfg or {}).get("defaults") or {})
    merged.update(user_config.dataset_paths(slug))
    return expand_dataset_paths(merged)


def acquisition(slug: str) -> Dict[str, Any]:
    acq = dict(_manifest(slug).get("acquisition") or {})
    if not acq.get("kind"):
        # a MOABB dataset without a manifest of its own: MOABB serves it
        from neuroatlas.extensions.datasets.dataio.moabb_loader import MOABB_DATASETS

        if slug in MOABB_DATASETS:
            acq["kind"] = "moabb"
    return acq


def _unresolved(value: str) -> List[str]:
    return re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", str(value))


def _setting_of(var: str) -> Optional[str]:
    """The setting an environment variable stands for (EEG_DATA_ROOT -> data_root)."""
    return next((s.key for s in user_config.SETTINGS.values() if s.env == var), None)


def _unresolved_text(variables: List[str]) -> str:
    """``no data root is set``: the settings a dataset's folder is written against."""
    names = [(_setting_of(v) or v).replace("_", " ") for v in variables]
    return f"no {' and no '.join(names)} {'is' if len(names) == 1 else 'are'} set"


def _unresolved_fix(variables: List[str]) -> str:
    fixes = [f"neuroatlas config set {_setting_of(v)} DIR" if _setting_of(v)
             else f"export {v}=DIR" for v in variables]
    return ", and ".join(fixes)


def raw_location(slug: str) -> Tuple[Optional[str], Optional[Path], List[str]]:
    """(key, path, unresolved variables) of the raw corpus.

    ``key`` is the per-dataset setting that moves it (``chbmit.bids_root``);
    for a cohort whose reader reads no raw path, ``acquisition.dest`` says
    where a download goes and the key is None.
    """
    cfg = resolved_config(slug)
    for key in RAW_KEYS:
        value = cfg.get(key)
        if value:
            unresolved = _unresolved(value)
            return key, (None if unresolved else Path(str(value))), unresolved
    dest = acquisition(slug).get("dest")
    if dest:
        from neuroatlas.entrypoints._common import expand_dataset_paths

        value = expand_dataset_paths({"dest": dest})["dest"]
        unresolved = _unresolved(value)
        return None, (None if unresolved else Path(str(value))), unresolved
    return None, None, []


def _matcher(pattern: Optional[str]) -> Callable[[str], bool]:
    if pattern:
        name_pattern = pattern.rsplit("/", 1)[-1]
        return lambda name: fnmatch.fnmatchcase(name, name_pattern)
    return lambda name: name.lower().endswith(EEG_SUFFIXES)


def _count(root: Path, pattern: Optional[str]) -> Tuple[int, bool]:
    """(files matching, whether the scan stopped early).

    ``pattern`` is matched against file names (a leading ``**/`` or
    directory part is ignored); macOS ``._`` resource files never count.
    """
    started, seen, hits = time.monotonic(), 0, 0
    match = _matcher(pattern)
    stack = [root]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for entry in it:
                    seen += 1
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name != "__MACOSX":
                            stack.append(Path(entry.path))
                    elif not entry.name.startswith("._") and match(entry.name):
                        hits += 1
                    if seen >= SCAN_ENTRIES or time.monotonic() - started > SCAN_SECONDS:
                        return hits, True
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
    return hits, False


@dataclass
class DatasetStatus:
    slug: str
    kind: str
    access: str                      # the manifest's kind, as `list datasets` says it
    handler: str                     # how `data download` gets it
    state: str                       # a key of STATES, maybe with "(n/N files)"
    path: Optional[Path] = None
    path_key: Optional[str] = None   # `config set <slug>.<path_key> DIR` moves it
    n_files: Optional[int] = None
    expected: Optional[int] = None
    notes: List[str] = field(default_factory=list)
    channel_map: bool = False
    download: str = ""               # how `data download` gets it (DOWNLOAD)
    host: str = ""                   # where it is served (HOSTS)

    @property
    def found(self) -> bool:
        return self.state.startswith("found") or self.state == "prepared"

    @property
    def setting(self) -> Optional[str]:
        """The `config set` key for this dataset's folder, or None."""
        return f"{self.slug}.{self.path_key}" if self.path_key else None


def mne_data_dir() -> Path:
    return user_config.mne_data().value


def _moabb_keeps_cohort_folders() -> bool:
    """True when the installed moabb predates 1.6, which moved every cohort
    under NEMAR/<id>: moabb 1.2.0 (the numpy<2 stack) keeps its own folders."""
    from importlib import metadata

    try:
        major, minor = (int(x) for x in metadata.version("moabb").split(".")[:2])
    except (metadata.PackageNotFoundError, ValueError):
        return False
    return (major, minor) < (1, 6)


def _nemar_state(slug: str, acq: Dict[str, Any], st: DatasetStatus) -> None:
    """found / partial / not downloaded, from MOABB's NEMAR deposit."""
    base = mne_data_dir()
    nemar = acq.get("nemar")
    if not nemar:
        st.path = base
        st.state = "not downloaded" if not base.is_dir() else "found"
        st.notes.append(f"which folder under {base} holds it is not recorded, so its files "
                        f"were not counted")
        return
    store = base / "NEMAR" / str(nemar)
    st.path = store
    legacy = acq.get("moabb_folder")
    if not store.is_dir() and legacy:
        # moabb 1.2.0 (the numpy<2 stack) stores each cohort in its own folder
        # under MNE_DATA; NEMAR/ is where moabb 1.6+ puts it.
        folder = base / str(legacy)
        n = sum(1 for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else 0
        if n:
            st.path, st.n_files = folder, n
            st.state = f"found ({n} files)"
            return
    if not store.is_dir():
        if legacy and _moabb_keeps_cohort_folders():
            st.path = base / str(legacy)      # where this moabb will put it
        st.state = "not downloaded"
        st.notes.append(f"fix: neuroatlas data download {slug}")
        return
    expected = None
    provenance = store / "sourcedata" / "sourcedata_provenance.json"
    try:
        expected = int(json.loads(provenance.read_text()).get("n_files"))
    except (OSError, ValueError, TypeError):
        pass
    sourcedata = store / "sourcedata"
    n = sum(1 for p in sourcedata.rglob("*") if p.is_file()
            and p.name != "sourcedata_provenance.json") if sourcedata.is_dir() else 0
    st.n_files, st.expected = n, expected
    if expected and n < expected:
        st.state = f"partial ({n}/{expected} files)"
        st.notes += _note_lines(f"{n} of {expected} files: the download stopped part-way; "
                                f"running it again finishes it",
                                f"neuroatlas data download {slug}")
    elif n:
        st.state = f"found ({n}/{expected} files)" if expected else f"found ({n} files)"
    else:
        st.state = "empty"
        st.notes.append(f"fix: neuroatlas data download {slug}")


def _prepared_hit(slug: str, explicit: Optional[str]) -> Tuple[Optional[Path], List[Path]]:
    """(the prepared file a reader would open, or None; every candidate)."""
    from neuroatlas.extensions.datasets.dataio.bci_paths import (
        COGNITIVE_PICKLES,
        CONFOUND_CONTROLLED,
        PREPROCESSED_SEARCH_PATHS,
        first_existing,
    )

    if explicit:
        hit = first_existing([explicit])
        return (Path(hit) if hit else None), [Path(explicit)]
    from neuroatlas.entrypoints._common import expand_dataset_paths

    # A candidate may be a glob (the authors' pickle, whichever model's). A
    # cognitive cohort has two pickles, the confound-filtered one (what the
    # benchmark's default reads) first; either one is the data.
    search = [*(PREPROCESSED_SEARCH_PATHS[slug + CONFOUND_CONTROLLED]
                if slug in COGNITIVE_PICKLES else []),
              *PREPROCESSED_SEARCH_PATHS.get(slug, [])]
    candidates = [Path(p) for p in expand_dataset_paths(
        {f"p{i}": p for i, p in enumerate(search)}).values()]
    hit = first_existing(str(p) for p in candidates)
    return (Path(hit) if hit else None), candidates


def _legacy_prepared(candidate: Path) -> Optional[Path]:
    """The same prepared file where an earlier version wrote it (the checkout)."""
    checkout = _paths.checkout_root()
    if checkout is None:
        return None
    try:
        rel = candidate.relative_to(_paths.prepared_dir())
    except ValueError:
        return None
    old = checkout / "data" / "preprocessed" / rel
    return old if old.is_file() else None


def _prepared_status(slug: str, st: DatasetStatus, acq: Dict[str, Any],
                     needs_build: bool) -> DatasetStatus:
    """prepared / not prepared, for readers that read only a built file."""
    explicit = user_config.dataset_paths(slug).get("preprocessed_path")
    try:
        hit, candidates = _prepared_hit(slug, explicit)
    except ImportError as exc:                    # braindecode / moabb not installed
        st.state = "package missing"
        st.notes += _note_lines(f"{exc.name} is not installed", MOABB_INSTALL)
        return st
    if explicit:
        st.path_key = "preprocessed_path"
    if hit:
        st.state, st.path = "prepared", hit
        return st
    st.state = "not prepared"
    st.path = candidates[0] if candidates else None
    old = _legacy_prepared(st.path) if st.path else None
    if old:
        st.notes += _note_lines(f"a copy in the source tree is not read: {old}",
                                f"mkdir -p {st.path.parent} && mv {old} {st.path}")
    if needs_build:
        return st
    st.path_key = "preprocessed_path"
    st.notes += _note_lines(manifest_text(acq["prepared_only"]))
    return st


def status(slug: str, *, raw_only: bool = False) -> DatasetStatus:
    """What `data status` says about *slug*. A cohort read from a file `data
    prepare` builds reports that file (prepared / not prepared, and then its
    raw data in the notes); *raw_only* reports the raw data alone."""
    acq = acquisition(slug)
    kind = acq.get("kind") or "manual"
    handler = HANDLERS.get(kind, "manual")
    if acq.get("unusable_download"):
        handler = "refused"
    st = DatasetStatus(slug, kind, kind, handler, "missing",
                       channel_map=_paths.configs_dir("channel_maps", f"{slug}.yaml").is_file(),
                       download=download_word(kind, handler), host=host_word(kind, acq))
    if kind == "nsrr" and not user_config.locate_token("nsrr"):
        st.notes.append(NO_NSRR_TOKEN)
    manifest = _manifest(slug)
    pipeline = manifest.get("pipeline") or {}
    has_builder = isinstance(pipeline.get("preprocessor"), dict)
    # A MOABB cohort whose pickle nothing reads (pipeline.required: false) is
    # reported by its raw data, like the MOABB cohorts with no builder.
    needs_build = has_builder and pipeline.get("required", True) is not False

    if kind == "moabb":
        if needs_build:
            _prepared_status(slug, st, acq, needs_build=True)
            if st.state == "not prepared":
                raw = DatasetStatus(slug, kind, kind, handler, "missing")
                _nemar_state(slug, acq, raw)
                if raw.found:
                    st.notes += _note_lines(f"raw data {raw.state}",
                                            f"neuroatlas data prepare {slug}")
                else:
                    st.notes += _note_lines("its raw data is not here either",
                                            f"neuroatlas data download {slug}, then "
                                            f"neuroatlas data prepare {slug}")
            return st
        _nemar_state(slug, acq, st)
        return st
    if acq.get("prepared_only"):
        _prepared_status(slug, st, acq, needs_build=False)
        return st
    if needs_build and not pipeline["preprocessor"].get("optional") and not raw_only:
        # read from the file `data prepare` builds from the raw data;
        # `preprocessed_path` points at one built elsewhere
        _prepared_status(slug, st, acq, needs_build=True)
        st.path_key = "preprocessed_path"
        if st.state == "not prepared":
            raw = status(slug, raw_only=True)
            if raw.found:
                st.notes += _note_lines(f"raw data {raw.state}", f"neuroatlas data prepare {slug}")
            else:
                st.notes += _note_lines(
                    "its raw data is not here either",
                    f"neuroatlas data download {slug}, then neuroatlas data prepare {slug}")
        return st

    key, path, unresolved = raw_location(slug)
    st.path_key = key
    if key is None and path is None:
        st.state = "no path"
        st.notes += _note_lines(NO_FOLDER, f"neuroatlas config set {slug}.data_root DIR")
        return st
    if unresolved:
        st.state = "not configured"
        st.notes += _note_lines(_unresolved_text(unresolved), _unresolved_fix(unresolved))
        return st
    st.path = path
    if acq.get("unusable_download"):
        st.notes += _note_lines(manifest_text(acq["unusable_download"]))
    expect = acq.get("expect") or {}
    count = expect.get("count")
    st.expected = int(count) if count else None
    if not path.exists():
        legacy = _paths.legacy_raw(slug)
        if legacy is not None:
            old = legacy / path.name if (legacy / path.name).is_dir() else legacy
            st.notes += _note_lines(f"a copy in the source tree is not read: {old}",
                                    f"neuroatlas config set {st.setting} {old}")
        return st
    n, truncated = _count(path, expect.get("glob"))
    st.n_files = n
    minimum = int(count or expect.get("min", 1))
    total = f"/{count}" if count else ""
    sample = sample_of(_nsrr_folder(path, str(acq["ref"])) if kind == "nsrr" and acq.get("ref")
                       else path)
    if sample is not None and n:
        k = len(sample.get("recordings") or []) or int(sample.get("first") or 0)
        st.state = f"sample (first {_msg.plural(k, 'recording')})"
        st.notes += _note_lines(f"the first {_msg.plural(k, 'recording')} only (`data download "
                                f"--first`): `neuroatlas check` runs on them; `run` needs the "
                                f"whole dataset", f"neuroatlas data download {slug}")
    elif n >= minimum or (truncated and n):
        st.state = f"found ({'≥' if truncated else ''}{n}{total} files)"
    elif n:
        st.state = f"partial ({n}{total} files)"
        if handler in ("physionet", "zenodo", "url"):
            st.notes += _note_lines(f"{n} of {minimum} expected files: the download stopped "
                                    f"part-way; running it again resumes it",
                                    f"neuroatlas data download {slug}")
        else:
            st.notes += _note_lines(f"{n} of {minimum} expected files are there",
                                    f"copy the rest into {path} (`neuroatlas data download "
                                    f"{slug}` says where it comes from)")
    else:
        st.state = "empty"
        st.notes += _note_lines("the folder exists but holds no recordings"
                                + (f" matching {expect['glob']}" if expect.get("glob") else ""),
                                f"neuroatlas data download {slug}")
    return st


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------

@dataclass
class Fetch:
    """One file to transfer: url -> dest, checked against size and/or md5."""
    url: str
    dest: Path
    size: Optional[int] = None
    md5: Optional[str] = None


@dataclass
class DownloadPlan:
    slug: str
    handler: str
    commands: List[List[str]] = field(default_factory=list)   # argv lists (wget, nsrr)
    cwd: Optional[Path] = None
    dest: Optional[Path] = None
    message: Optional[str] = None      # manual / internal / refused: what a person must do
    stdin_token: Optional[str] = None  # name of the token fed on stdin
    size_gb: Optional[float] = None
    after: List[str] = field(default_factory=list)
    fetches: List[Fetch] = field(default_factory=list)       # done in Python, resumable
    listing: Optional[str] = None      # zenodo record / S3 prefix to list when it runs
    unpack: List[Path] = field(default_factory=list)          # zips to unpack into dest
    keep_archive: bool = False
    mirror: str = "physionet"
    first: Optional[int] = None        # --first N: the first N recordings only
    pattern: Optional[str] = None      # which listed files are recordings (expect.glob)
    source: Optional[str] = None       # files come from here when listed elsewhere
    link: Optional[str] = None         # nsrr: the folder the tool writes, when not dest's name
    ref: Optional[str] = None          # nsrr: the NSRR dataset
    folders: List[str] = field(default_factory=list)          # nsrr: what to fetch
    recordings: List[str] = field(default_factory=list)       # --first: the ones taken


#: Written into a dataset's folder by `data download --first N` once the
#: sample is complete (JSON: the N and the recordings); a whole download
#: removes it. `data status` reports such a folder as a sample.
SAMPLE_MARK = ".neuroatlas-first.json"

#: NSRR's listing of one folder of a dataset (the API the nsrr tool reads).
NSRR_FILES = "https://sleepdata.org/api/v1/datasets/{ref}/files.json"


def _nsrr_folder(path: Path, ref: str) -> Path:
    """The dataset's own folder, the one NSRR's tree goes in: ``path``, or,
    for a setting that points inside that tree (``.../wsc/polysomnography``),
    the folder named after the NSRR dataset above it."""
    posix = path.as_posix().rstrip("/")
    marker = f"/{ref}/"
    if marker in posix + "/" and not posix.endswith(f"/{ref}"):
        return Path(posix[: posix.rindex(marker) + len(marker) - 1])
    return path


def _archive_root(path: Path, archive: str) -> Path:
    """Where to unpack an archive: into the dataset's folder ``path``
    (BIDS_CHB-MIT.zip -> $DATA/chbmit/BIDS_CHB-MIT/, which the reader finds
    there; dodh.zip -> $DATA/dod/dodh/). A setting that points at the
    archive's own top folder (.../BIDS_CHB-MIT) unpacks into its parent, so
    the files land where it points."""
    return path.parent if Path(archive).stem == path.name else path


ZENODO_FILE = "https://zenodo.org/api/records/{ref}/files/{name}/content"
ZENODO_RECORD = "https://zenodo.org/api/records/{ref}"
S3_BUCKET = "https://physionet-open.s3.amazonaws.com"


def _no_first(plan: DownloadPlan, how: str) -> DownloadPlan:
    """Refuse --first for a host that cannot serve part of a dataset."""
    plan.handler = "refused"
    plan.message = _msg.compose(
        f"--first takes the first recordings of a PhysioNet, NSRR or file-by-file Zenodo "
        f"download; {plan.slug} comes {how}",
        f"neuroatlas data download {plan.slug} (the whole dataset)")
    return plan


def plan_download(slug: str, *, mirror: str = "physionet",
                  keep_archive: bool = False, first: Optional[int] = None) -> DownloadPlan:
    acq = acquisition(slug)
    kind = acq.get("kind") or "manual"
    handler = HANDLERS.get(kind, "manual")
    plan = DownloadPlan(slug, handler, size_gb=acq.get("size_gb"),
                        keep_archive=keep_archive, mirror=mirror)
    landing = acq.get("upstream")

    if acq.get("unusable_download"):
        plan.handler = "refused"
        plan.message = manifest_text(acq["unusable_download"])
        return plan

    if first is not None and handler not in ("manual", "internal"):
        plan.first = int(first)
        plan.size_gb = None          # the dataset's size, not the sample's
        plan.pattern = (acq.get("expect") or {}).get("glob")
        if handler == "moabb":
            # MOABB fetches a missing subject when a reader asks for it
            return _no_first(plan, "through MOABB, which fetches every other subject "
                                   "on first use")
        if handler == "url":
            files = acq.get("files") or []
            url = files[0]["url"] if files and isinstance(files[0], dict) else \
                str(files[0]) if files else ""
            host = urllib.parse.urlparse(url).hostname or host_word(kind, acq)
            return _no_first(plan, f"as {len(files)} archives from "
                                   f"{host[4:] if host.startswith('www.') else host}")
        if handler == "zenodo" and (acq.get("file") or acq.get("files")):
            n = 1 if acq.get("file") else len(acq["files"])
            return _no_first(plan, "as one archive from Zenodo" if n == 1
                             else f"as {n} archives from Zenodo")

    if handler == "moabb":
        from neuroatlas.extensions.datasets.dataio.moabb_loader import MOABB_DATASETS

        cfg = MOABB_DATASETS.get(slug)
        if cfg is None:
            plan.handler = "manual"
            plan.message = (f"this package has no MOABB download for {slug}\n"
                            f"fix: pip install --force-reinstall neuroatlas")
            return plan
        plan.dest = mne_data_dir()
        plan.commands = [["moabb", f"{cfg.moabb_name}().download()"]]
        pipeline = _manifest(slug).get("pipeline") or {}
        if (isinstance(pipeline.get("preprocessor"), dict)
                and pipeline.get("required", True) is not False):
            plan.after = [f"note: {slug} is read from a prepared file, built from the "
                          f"download\nfix: neuroatlas data prepare {slug}"]
        return plan

    key, path, unresolved = raw_location(slug)
    if handler not in ("manual", "internal") and (unresolved or path is None):
        plan.handler = "blocked"
        plan.message = _msg.compose(*((_unresolved_text(unresolved), _unresolved_fix(unresolved))
                                      if unresolved else
                                      (NO_FOLDER, f"neuroatlas config set {slug}.data_root DIR")))
        return plan
    plan.dest = path
    pipeline = _manifest(slug).get("pipeline") or {}
    block = pipeline.get("preprocessor")
    if (isinstance(block, dict) and not block.get("optional")
            and pipeline.get("required", True) is not False):
        plan.after = [f"note: {slug} is read from a prepared file, built from the "
                      f"download\nfix: neuroatlas data prepare {slug}"]

    if handler == "physionet" and acq.get("ref") and acq.get("version"):
        # The version folder's contents go straight into the dataset's folder
        # (wget: --cut-dirs=3 drops files/<ref>/<version>/).
        prefix = f"{acq['ref']}/{acq['version']}/"
        if mirror == "aws" or plan.first:
            # PhysioNet's open-data mirror on AWS: anonymous HTTPS, no
            # credentials, typically far faster than physionet.org. Its
            # listing also says which files the first N recordings are.
            plan.listing = f"{S3_BUCKET}/?list-type=2&prefix={prefix}"
            if mirror != "aws":
                plan.source = f"https://physionet.org/files/{prefix}"
            return plan
        url = f"https://physionet.org/files/{prefix}"
        # --progress=dot:mega: wget's output is read, not shown (_ToolOutput),
        # and its dots become the bytes on the live line
        plan.commands = [["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3",
                          "--progress=dot:mega", "--reject", "index.html*", "-P", str(path), url]]
        return plan

    if handler == "zenodo" and acq.get("ref"):
        ref = str(acq["ref"])
        archives = [acq["file"]] if acq.get("file") else list(acq.get("files") or [])
        if archives:
            for name in archives:
                where = _archive_root(path, str(name))
                plan.fetches.append(Fetch(ZENODO_FILE.format(ref=ref, name=name), where / str(name)))
                if str(name).endswith(".zip"):
                    plan.unpack.append(where / str(name))
            if acq.get("checksum", "").startswith("md5:") and len(plan.fetches) == 1:
                plan.fetches[0].md5 = acq["checksum"].split(":", 1)[1]
            plan.listing = ZENODO_RECORD.format(ref=ref)     # sizes and md5s, when it runs
        else:
            plan.listing = ZENODO_RECORD.format(ref=ref)     # every file of the record
        return plan

    if handler == "url" and acq.get("files"):
        for item in acq["files"]:
            url = item["url"] if isinstance(item, dict) else str(item)
            # `name`: the file's own name, where the URL does not end in it
            name = item.get("name") if isinstance(item, dict) else None
            target = path / (name or url.rstrip("/").rsplit("/", 1)[-1])
            plan.fetches.append(Fetch(url, target, md5=(item.get("md5") if isinstance(item, dict) else None)))
            if target.suffix == ".zip":
                plan.unpack.append(target)
        return plan

    if handler == "nsrr" and acq.get("ref"):
        # The nsrr tool writes <its folder>/<ref>/<path> under the folder it
        # runs in; NSRR's tree (polysomnography/, datasets/) goes into the
        # dataset's folder, where its reader looks for it.
        ref = str(acq["ref"])
        dest = _nsrr_folder(path, ref)
        plan.dest, plan.ref = dest, ref
        plan.folders = [str(f).strip("/") for f in acq.get("folders") or []]
        if dest.name == ref:
            plan.cwd = dest.parent
        else:
            # run in a scratch folder where <ref> links to the dataset's folder
            plan.link = ref
        plan.commands = [["nsrr", "download", f"{ref}/{f}"] for f in plan.folders] \
            or [["nsrr", "download", ref]]
        plan.stdin_token = "nsrr"
        if plan.first:
            # which files: from NSRR's listing, when it runs (it needs the token)
            plan.listing = NSRR_FILES.format(ref=ref)
            plan.commands = []
        return plan

    plan.handler = "manual" if handler != "internal" else "internal"
    if acq.get("prepared_only"):
        # what the benchmark reads is the authors' file: that, where it is
        # read without a setting, and how to point at it elsewhere
        _, what, fixes = _msg.split(manifest_text(acq["prepared_only"]))
        where = _prepared_home(slug)
        if where is not None:
            what.append(f"put it in {where.parent}/ ({where.name}) and it is read without a "
                        f"setting")
        plan.message = "\n".join(what + [f"{_msg.FIX}: {f}" for f in fixes])
        return plan
    reason = {
        "tuh": "needs a signed TUH data use agreement (credentials arrive by email)",
        "internal": "from the authors: not publicly released",
    }.get(kind, "fetched by hand (no download API, or access on request)")
    # one line each: why by hand, where from, what the reader reads, where it goes
    lines = [reason]
    if landing:
        lines.append(f"get it from {landing}")
    if acq.get("note"):
        lines.append(" ".join(str(acq["note"]).split()))
    if path is not None:
        setting = f"{slug}.{key}" if key else None
        lines.append(f"put it at {path}")
        if setting:
            lines.append(f"a copy elsewhere: neuroatlas config set {setting} DIR")
    plan.message = "\n".join(lines)
    return plan


def _prepared_home(slug: str) -> Optional[Path]:
    """Where a prepared file the authors provide goes by default: the first
    place its reader looks under the data root (a glob: any model's file)."""
    try:
        _, candidates = _prepared_hit(slug, None)
    except ImportError:
        return None
    root = user_config.get("data_root")
    for candidate in candidates:
        if root and Path(candidate).is_relative_to(Path(root)):
            return candidate
    return None


def free_gb(path: Path) -> Optional[float]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / 1e9
    except OSError:
        return None


NSRR_MIN_FREE_GB = 50.0
#: the fix for a download tool that is not on PATH ({slug}: the dataset)
TOOL_HINTS = {"nsrr": "gem install --user-install nsrr irb (the nsrr tool, once)",
              "wget": "neuroatlas data download {slug} --mirror aws (needs no wget), "
                      "or install wget"}


def check_nsrr_allowed(slug: str) -> Optional[str]:
    """An NSRR token opens only the datasets its owner is approved for."""
    allowed = os.environ.get("NEUROATLAS_NSRR_DATASETS")
    if not allowed:
        return None
    names = {n.strip().lower() for n in allowed.split(",") if n.strip()}
    ref = str(acquisition(slug).get("ref") or slug).lower()
    if ref not in names and slug not in names:
        return _msg.compose(
            f"{slug} is not in NEUROATLAS_NSRR_DATASETS ({allowed}): your NSRR account is not "
            f"approved for it",
            f"request access to {ref} at sleepdata.org, then add {ref} to "
            f"NEUROATLAS_NSRR_DATASETS")
    return None


def refusal(plan: DownloadPlan) -> Optional[str]:
    """Why this plan must not run, or None. The same checks for a real run and a
    dry run, so a dry run never prints a command the real one would refuse."""
    if plan.handler in ("refused", "blocked"):
        return plan.message
    if plan.stdin_token:
        why = check_nsrr_allowed(plan.slug)
        if why:
            return why
        if user_config.locate_token(plan.stdin_token) is None:
            service = "NSRR" if plan.stdin_token == "nsrr" else plan.stdin_token
            return (f"no {service} token\n"
                    f"fix: neuroatlas config token {plan.stdin_token}")
    need = plan.size_gb or (NSRR_MIN_FREE_GB if plan.handler == "nsrr" else None)
    if plan.first:
        need = None          # a few recordings: their size is known once listed
    if need and plan.unpack:
        need *= 2            # the archive and what it unpacks to, until the archive goes
    if need and plan.dest is not None:
        free = free_gb(plan.dest)
        if free is not None and free < need * 1.1:
            return _msg.compose(
                f"{plan.slug} needs about {need:g} GB; {free:.0f} GB free under {plan.dest}",
                "export MNE_DATA=DIR (a disk with room), then the same download"
                if plan.handler == "moabb" else
                "neuroatlas config set data_root DIR (a disk with room), then the same download")
    if plan.handler == "moabb":
        import importlib.util

        if importlib.util.find_spec("moabb") is None:
            return _msg.compose(f"MOABB is not installed (install it {MOABB_WHERE})",
                                MOABB_INSTALL)
    tools = [argv[0] for argv in plan.commands if argv and argv[0] != "moabb"]
    if plan.handler == "nsrr":
        tools.append("nsrr")       # --first: its commands are known once listed
    for tool in dict.fromkeys(tools):
        if shutil.which(tool) is None:
            hint = TOOL_HINTS.get(tool)
            return _msg.compose(f"`{tool}` is not on PATH",
                                hint.format(slug=plan.slug) if hint else None)
    return None


def describe(plan: DownloadPlan) -> List[str]:
    """What a plan would do, one line each (the dry run)."""
    lines = []
    where = f"  (into {plan.dest})" if plan.dest is not None and plan.handler == "nsrr" \
        else (f"  (in {plan.cwd})" if plan.cwd else "")
    for argv in plan.commands:
        lines.append(f"command: {' '.join(argv)}{where}")
    if plan.first and plan.listing:
        what = ", ".join(plan.folders) if plan.handler == "nsrr" else ""
        origin = (", with the nsrr tool" if plan.handler == "nsrr"
                  else f", from {urllib.parse.urlparse(plan.source).hostname}"
                  if plan.source else "")
        lines.append(f"fetch: the first {_msg.plural(plan.first, 'recording')} and the files "
                     f"they share{' under ' + what if what else ''}, listed by {plan.listing} "
                     f"when it runs{origin}, into {plan.dest}/")
        return lines + list(plan.after)
    if plan.listing and not plan.fetches:
        source = ("every file of the record" if "zenodo" in plan.listing
                  else "every file under the prefix")
        lines.append(f"fetch: {source} listed by {plan.listing}, into {plan.dest}/")
    for f in plan.fetches:
        lines.append(f"fetch: {f.url}, into {f.dest}" + (f"  (md5 {f.md5})" if f.md5 else ""))
    for archive in plan.unpack:
        lines.append(f"unpack: {archive}, into {archive.parent}/"
                     + ("" if plan.keep_archive else "  (then delete the archive; "
                        "--keep-archive keeps it)"))
    for line in plan.after:
        lines.append(line)
    if plan.commands and plan.commands[0][0] == "wget" and "physionet.org" in plan.commands[0][-1]:
        # before a transfer only: PhysioNet's open-data copy on AWS
        lines.append(f"note: PhysioNet's open-data copy on AWS is usually much faster (no "
                     f"account needed)\nfix: neuroatlas data download {plan.slug} --mirror aws")
    return lines


# -- transfers done in Python: resumable, verified, one line per file (and a
#    live line meanwhile, neuroatlas.progress)

def _http_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _zenodo_files(listing: str) -> List[Dict[str, Any]]:
    return list(_http_json(listing).get("files") or [])


def _s3_keys(listing: str) -> List[Tuple[str, int, Optional[str]]]:
    """(key, size, md5 or None) under an S3 prefix, following continuation tokens."""
    import xml.etree.ElementTree as ET

    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    out, token = [], None
    while True:
        url = listing + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        for item in root.findall("s3:Contents", ns):
            key = item.findtext("s3:Key", namespaces=ns)
            size = int(item.findtext("s3:Size", default="0", namespaces=ns))
            etag = (item.findtext("s3:ETag", default="", namespaces=ns) or "").strip('"')
            out.append((key, size, etag if etag and "-" not in etag else None))
        if root.findtext("s3:IsTruncated", namespaces=ns) != "true":
            return out
        token = root.findtext("s3:NextContinuationToken", namespaces=ns)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(f: Fetch, echo, label: str, overall: bool = False) -> None:
    """Transfer one file, resuming a partial one; raise on any failure.

    The bytes go to the live line of the item running (neuroatlas.progress):
    with *overall*, toward the total of every file the listing sized; else
    as this file's own percent. One line per file goes to *echo* when it is
    done.
    """
    item = progress.current()
    f.dest.parent.mkdir(parents=True, exist_ok=True)
    if f.dest.is_file() and (f.size is not None or f.md5 is not None) \
            and (f.size is None or f.dest.stat().st_size == f.size):
        if f.md5 is not None:
            item.update(note=f"{label} checking {f.dest.name}")
        if f.md5 is None or _md5(f.dest) == f.md5:
            echo(f"  {label} {f.dest.name}: already complete")
            item.count("skipped")
            if overall:
                item.update(advance=f.dest.stat().st_size, carried=f.dest.stat().st_size)
            return
    part = f.dest.with_name(f.dest.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    request = urllib.request.Request(f.url, headers={"Range": f"bytes={have}-"} if have else {})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=120) as response:
        if have and response.status != 206:        # server ignored the range: start over
            have = 0
        total = f.size or (int(response.headers.get("Content-Length", 0)) + have) or None
        if overall:
            item.update(advance=have, carried=have, note=f"{label} {f.dest.name}")
        else:
            item.phase("downloading", total=total, done=have, unit="bytes")
            item.update(note=f"{label} {f.dest.name}")
        with open(part, "ab" if have else "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                item.update(advance=len(chunk))
                item.count("bytes", len(chunk))
    if f.size is not None and part.stat().st_size != f.size:
        raise IOError(f"{f.dest.name}: got {part.stat().st_size} bytes, expected {f.size}; "
                      f"running the same download again resumes it")
    if f.md5:
        item.update(note=f"{label} checking {f.dest.name}")
        if _md5(part) != f.md5:
            part.unlink()
            raise IOError(f"{f.dest.name}: md5 mismatch (expected {f.md5}); the partial file "
                          f"was removed, so running the same download again starts it over")
    part.replace(f.dest)
    item.count("files")
    echo(f"  {label} {f.dest.name}: {progress.size(f.dest.stat().st_size)} in "
         f"{progress.duration(time.monotonic() - started)}" + (", md5 ok" if f.md5 else ""))


def _natural(text: str) -> Tuple:
    """Sort key: digits as numbers (eeg2 before eeg10)."""
    return tuple((0, int(part), "") if part.isdigit() else (1, 0, part)
                 for part in re.split(r"(\d+)", text) if part)


#: Shared files a sample leaves out: archives of the whole dataset.
_ARCHIVES = (".zip", ".tar", ".gz", ".tgz", ".7z", ".bz2")


def first_files(paths: List[str], n: int, pattern: Optional[str]) -> Tuple[List[str], List[str]]:
    """``--first N``: which of a dataset's files to fetch.

    *paths* are relative to the dataset's folder. The recordings are the
    files *pattern* (``expect.glob``; else any EEG file type) matches, in
    natural order; the first *n* are taken with the files that belong to
    them -- a file whose name shares a longer start with one recording than
    with any other, a start that reaches into its number
    (``SN001_sleepscoring.edf``, ``mesa-sleep-0001-nsrr.xml``,
    ``SC4001EC-Hypnogram.edf``) -- and every file that belongs to none
    (metadata sheets, RECORDS), archives of the whole dataset left out.

    Returns (the files to fetch, in listing order; the recordings taken).
    """
    import bisect

    match = _matcher(pattern)
    name = {p: p.rsplit("/", 1)[-1] for p in paths}
    recordings = sorted((p for p in paths if match(name[p])), key=_natural)
    chosen = recordings[:max(0, int(n))]
    owners = sorted({name[r] for r in recordings})
    taken = {name[r] for r in chosen}

    def owner(file_name: str) -> Optional[str]:
        """The one recording this file shares its longest start with, or None."""
        i = bisect.bisect_left(owners, file_name)
        best = max((len(os.path.commonprefix([file_name, owners[j]]))
                    for j in (i - 1, i) if 0 <= j < len(owners)), default=0)
        head = file_name[:best]
        if not any(ch.isdigit() for ch in head):      # a sheet: "SC-subjects.xls"
            return None
        lo = bisect.bisect_left(owners, head)
        hi = bisect.bisect_left(owners, head + "\U0010ffff")
        return owners[lo] if hi - lo == 1 else None

    keep = set(chosen)
    recording_set = set(recordings)
    for p in paths:
        if p in recording_set:
            continue
        who = owner(name[p])
        if who is None:
            if not name[p].lower().endswith(_ARCHIVES):
                keep.add(p)
        elif who in taken:
            keep.add(p)
    return [p for p in paths if p in keep], chosen


def _resolve_listing(plan: DownloadPlan) -> None:
    """Fill sizes and md5s (or the whole file list) from the record's API."""
    if not plan.listing:
        return
    if "zenodo.org" in plan.listing:
        files = {f["key"]: f for f in _zenodo_files(plan.listing)}
        if not plan.fetches:                          # the whole record
            ref = plan.listing.rstrip("/").rsplit("/", 1)[-1]
            names = sorted(files)
            if plan.first:
                names, plan.recordings = first_files(names, plan.first, plan.pattern)
                if not plan.recordings:
                    raise IOError(f"{plan.listing} lists no recording")
            for name in names:
                plan.fetches.append(Fetch(ZENODO_FILE.format(ref=ref, name=name),
                                          plan.dest / name))
        for fetch in plan.fetches:
            meta = files.get(fetch.dest.name)
            if meta is None:
                raise IOError(f"{fetch.dest.name} is not in {plan.listing}")
            fetch.size = int(meta.get("size") or 0) or None
            checksum = str(meta.get("checksum") or "")
            if checksum.startswith("md5:"):
                fetch.md5 = fetch.md5 or checksum.split(":", 1)[1]
        return
    prefix = plan.listing.split("prefix=", 1)[1]
    listed = {key[len(prefix):]: (key, size, md5) for key, size, md5 in _s3_keys(plan.listing)}
    rels = [rel for rel in listed if rel and not rel.endswith("/")]
    if plan.first:
        rels, plan.recordings = first_files(rels, plan.first, plan.pattern)
        if not plan.recordings:
            raise IOError(f"{plan.listing} lists no recording")
    for rel in rels:
        key, size, md5 = listed[rel]
        # the open-data copy's sizes and md5s hold for physionet.org's files too
        url = f"{plan.source}{rel}" if plan.source else f"{S3_BUCKET}/{key}"
        plan.fetches.append(Fetch(url, plan.dest / rel, size=size, md5=md5))


def _unpack(archive: Path, echo, keep: bool) -> None:
    target = archive.parent
    item = progress.current()
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.filename.startswith("__MACOSX/")
                   and not m.filename.rsplit("/", 1)[-1].startswith("._")]
        item.phase("unpacking", total=len(members), unit="entries")
        item.update(note=archive.name)
        for member in members:
            out = target / member.filename
            if member.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            elif not (out.is_file() and out.stat().st_size == member.file_size):
                zf.extract(member, target)
            item.update(advance=1)
    files = sum(1 for m in members if not m.is_dir())
    echo(f"  unpacked {archive.name} into {target}/ ({files} files)")
    if not keep:
        archive.unlink()
        echo(f"  deleted {archive.name} (pass --keep-archive to keep it)")
    # an archive of archives (ArithmeticTask: Experiment 1.zip, Experiment 2.zip):
    # each inner one into a folder of its name
    for member in members:
        inner = target / member.filename
        if not member.is_dir() and inner.suffix == ".zip" and inner.is_file():
            folder = inner.parent / inner.stem
            folder.mkdir(parents=True, exist_ok=True)
            moved = folder / inner.name
            inner.replace(moved)
            _unpack(moved, echo, keep)


@contextlib.contextmanager
def _quiet(log: io.StringIO):
    """Collect a library's prints (MOABB, nemar-py) instead of showing them."""
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        yield


def _moabb_download(plan: DownloadPlan, echo) -> int:
    import logging

    from neuroatlas.extensions.datasets.dataio.moabb_loader import (
        MOABB_DATASETS,
        moabb_dataset_class,                   # refusal() checked moabb is installed
    )
    plan.dest.mkdir(parents=True, exist_ok=True)
    # Exported, so MNE never falls back to writing ~/.mne/mne-python.json, and
    # never follows a per-dataset folder from it (see config.mne_data).
    os.environ["MNE_DATA"] = str(plan.dest)
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    cfg = MOABB_DATASETS[plan.slug]
    try:
        # the installed moabb's class, or the vendored copy it predates
        dataset = moabb_dataset_class(cfg.moabb_name)()
    except Exception as exc:                          # noqa: BLE001
        echo(_msg.format("error", f"{plan.slug}: MOABB cannot build {cfg.moabb_name}: "
                                  + _msg.brief(_msg.exception_text(exc)),
                         f"neuroatlas -v data download {plan.slug} (shows the whole error)"))
        return 1
    subjects = list(getattr(dataset, "subject_list", []) or [None])
    echo(f"  MOABB {cfg.moabb_name}: {_msg.plural(len(subjects), 'subject')}, into {plan.dest}")
    if logging.getLogger().level > logging.DEBUG:    # `neuroatlas -v` keeps them
        # One request line per file per subject otherwise (O-8).
        for name in ("moabb", "mne", "pooch", "httpx", "httpcore", "nemar", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)
    # MOABB says nothing while it fetches: the live line counts subjects, and
    # the bytes arriving under the MOABB folder (when that is cheap to sum)
    item = progress.current()
    item.phase("downloading", total=len(subjects), unit="subjects")
    written = progress.DiskBytes(plan.dest) if isinstance(item, progress.Progress) else None
    item.watch(written)
    started = time.monotonic()
    try:
        for i, subject in enumerate(subjects, 1):
            log = io.StringIO()
            item.update(note=f"subject {subject}" if subject is not None else None)
            began = time.monotonic()
            try:
                with _quiet(log):
                    if subject is None:
                        dataset.download(path=str(plan.dest), update_path=False)
                    else:
                        # accept stays False: a dataset whose licence must be
                        # accepted says so in the one-line error below.
                        dataset.download(subject_list=[subject], path=str(plan.dest),
                                         update_path=False)
            except Exception as exc:                  # noqa: BLE001 -- one line, not a traceback
                tail = " | ".join(line for line in log.getvalue().splitlines()[-3:] if line.strip())
                if tail:
                    logging.getLogger(__name__).info("%s: MOABB's last output: %s", plan.slug, tail)
                echo(_msg.format("error", f"{plan.slug}: subject {subject}: "
                                          + _msg.brief(_msg.exception_text(exc))
                                          + "; running the same download again resumes it",
                                 f"neuroatlas data download {plan.slug}"))
                return 1
            item.update(advance=1)
            item.count("subjects")
            echo(f"  [{i}/{len(subjects)}] subject {subject} "
                 f"({progress.duration(time.monotonic() - began)})")
    finally:
        if written is not None:
            written._next = 0.0
            item.count("bytes", written() or 0)
    logging.getLogger(__name__).info("%s: %s in %s", plan.slug,
                                     _msg.plural(len(subjects), "subject"),
                                     progress.duration(time.monotonic() - started))
    return 0


class _ToolOutput:
    """What a download tool (wget, nsrr) writes, read while it runs.

    The tool's stdout and stderr go to a file; :meth:`poll` reads what is new
    each time the item's line ticks. wget's lines (``--progress=dot:mega``)
    become files done, bytes and the file in transfer on the live line;
    another tool's lines are passed on to *echo*, so it reads as before. The
    last lines are kept for the error message when the tool fails.
    """

    _START = re.compile(r"^--\d{4}-\d\d-\d\d \d\d:\d\d:\d\d--\s+(\S+)")
    _LENGTH = re.compile(r"^Length: (\d+)(?: \([^)]*\))?(?:, (\d+) \([^)]*\) remaining)?")
    _SAVING = re.compile(r"^Saving to: ['\u2018\"](.+?)['\u2019\"]\s*$")
    _SAVED = re.compile(r" saved \[(\d+)(?:/\d+)?\]\s*$")
    _KEPT = re.compile(r"^(?:File ['\u2018\"](.+?)['\u2019\"] (?:not modified on server|already "
                       r"there)|Server file no newer than local file ['\u2018\"](.+?)['\u2019\"])")
    _DOTS = re.compile(r"^\s*(\d+)K ([.,\s]*)")

    #: An nsrr line for a file its --file pattern leaves out (``--first``):
    #: one per other file of the folder, so not shown.
    _NOT_TAKEN = re.compile(r"^(?:\x1b\[[0-9;]*m)?\s*skipped(?:\x1b\[[0-9;]*m)?\s")
    #: An nsrr line for a file sleepdata.org did not serve; its reason is the
    #: next line ("Token Not Authorized to Access Specified File"). The tool
    #: still exits 0.
    _FAILED = re.compile(r"^(?:\x1b\[[0-9;]*m)?\s*failed(?:\x1b\[[0-9;]*m)?\s+(\S+)")

    def __init__(self, path: str, *, wget: bool, item=None, echo=None, written=None,
                 drop_not_taken: bool = False):
        self.path, self.wget, self.echo = path, wget, echo
        self.drop_not_taken = drop_not_taken
        self.item = item if item is not None else progress.current()
        self.written = written            # DiskBytes for a tool that does not say
        self.offset = 0
        self.buffer = ""
        self.files = self.skipped = 0
        self.present = 0                  # bytes of the files finished or kept
        self.current = 0                  # bytes of the file in transfer
        self.had = 0                      # of which there before (a resumed file)
        self.listing = False              # the file in transfer is a directory listing
        self.tail: List[str] = []
        self.failed: List[str] = []       # nsrr: the files it did not get
        self.reasons: List[str] = []      # and why, as it says
        self._reason_next = False
        self._lock = threading.Lock()

    def poll(self, final: bool = False):
        with self._lock:
            try:
                with open(self.path, "rb") as fh:
                    fh.seek(self.offset)
                    new = fh.read()
            except OSError:
                new = b""
            self.offset += len(new)
            self.buffer += new.decode("utf-8", errors="replace")
            *lines, self.buffer = self.buffer.split("\n")
            if final and self.buffer:
                lines, self.buffer = lines + [self.buffer], ""
            for line in lines:
                self._line(line.rstrip("\r"))
            if self.wget:
                dots = self._DOTS.match(self.buffer)
                if dots:
                    self._dots(dots)
                self.item.update(done=self.present + self.current)
        return self.written() if self.written is not None else None

    def _dots(self, match) -> None:
        marks = sum(match.group(2).count(c) for c in ".,")
        self.current = int(match.group(1)) * 1024 + marks * 65536

    def _line(self, line: str) -> None:
        if not line.strip():
            return
        if not self.wget:
            if self.drop_not_taken and self._NOT_TAKEN.match(line):
                return
            failed = self._FAILED.match(line)
            if failed:
                self.failed.append(failed.group(1))
            elif self._reason_next and line.strip() not in self.reasons:
                self.reasons.append(line.strip())
            self._reason_next = bool(failed)
            self.tail = (self.tail + [line])[-20:]
            if self.echo is not None:
                self.echo(f"  {line.rstrip()}")
            return
        dots = self._DOTS.match(line)
        if dots:
            self._dots(dots)
            return
        self.tail = (self.tail + [line])[-20:]
        start = self._START.match(line)
        if start:
            name = start.group(1).rstrip("/").rsplit("/", 1)[-1]
            self.current = self.had = 0
            self.listing = start.group(1).endswith("/") or name.startswith("robots.txt")
            if not self.listing:
                self.item.update(note=name)
            return
        length = self._LENGTH.match(line)
        if length and length.group(2):
            self.had = int(length.group(1)) - int(length.group(2))
            self.item.update(carried=self.had)
            return
        saving = self._SAVING.match(line)
        if saving:
            self.listing = self.listing or Path(saving.group(1)).name.startswith("index.html")
            return
        saved = self._SAVED.search(line)
        if saved:
            if not self.listing:
                got = int(saved.group(1))
                self.present += got
                self.files += 1
                self.item.count("files").count("bytes", max(0, got - self.had))
            self.current = self.had = 0
            return
        kept = self._KEPT.match(line)
        if kept:
            path = Path(kept.group(1) or kept.group(2))
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            self.skipped += 1
            self.present += size
            self.item.count("skipped").update(carried=size)
            self.current = self.had = 0
            return
        if "fully retrieved" in line:
            self.skipped += 1
            self.item.count("skipped")


@contextlib.contextmanager
def _token_on_stdin(token: Optional[str]):
    """The ``subprocess.run`` arguments that answer the tool's token prompt.

    The nsrr tool reads the token with echo off, which needs a terminal: on a
    pipe it stops with ``Errno::ENOTTY``. Where the platform has
    pseudo-terminals the token goes in through one of ours (the terminal's
    echo comes back to us, never into the tool's output); elsewhere through a
    pipe.
    """
    if not token:
        yield {}
        return
    try:
        import pty

        master, slave = pty.openpty()
    except (ImportError, OSError):
        yield {"input": token + "\n", "text": True}
        return
    try:
        os.write(master, (token + "\n").encode())
        yield {"stdin": slave}
    finally:
        os.close(slave)
        os.close(master)


def _run_tool(argv: List[str], plan: DownloadPlan, token: Optional[str], echo,
              cwd: Optional[Path] = None) -> int:
    """Run wget or nsrr with their output read as it comes (:class:`_ToolOutput`):
    a live line instead of silence, the tool's last lines when it fails."""
    import tempfile

    item = progress.current()
    wget = argv[0] == "wget"
    fd, path = tempfile.mkstemp(prefix=f"neuroatlas-{argv[0]}-", suffix=".log")
    if wget:
        # the size is the manifest's estimate: shown as "of ~1.3 GB"
        total = plan.size_gb * 1e9 if plan.size_gb else None
        item.phase("downloading", total=total, unit="bytes", approx=True)
        output = _ToolOutput(path, wget=True, item=item)
        extra = {"env": {**os.environ, "LC_ALL": "C"}}      # wget's words, in English
    else:
        item.phase(" ".join(argv[:3]))
        written = (progress.DiskBytes(plan.dest or cwd or plan.cwd)
                   if isinstance(item, progress.Progress) else None)
        output = _ToolOutput(path, wget=False, item=item, echo=echo, written=written,
                             drop_not_taken=any(a.startswith("--file=") for a in argv))
        extra = {}
    item.watch(output.poll)
    try:
        # The token goes to the tool's own prompt on stdin: never argv (visible
        # in `ps`), never the environment, never printed.
        with _token_on_stdin(token) as stdin:
            done = subprocess.run(argv, cwd=cwd if cwd is not None else plan.cwd,
                                  stdout=fd, stderr=subprocess.STDOUT, **stdin, **extra)
    finally:
        os.close(fd)
        item.watch(None)
        output.poll(final=True)
        if output.written is not None:
            output.written._next = 0.0
            item.count("bytes", output.written() or 0)
        try:
            os.unlink(path)
        except OSError:
            pass
    if done.returncode:
        last = [line for line in output.tail if "robots.txt" not in line][-3:]
        detail = ("\n" + "\n".join(last)) if last else ""
        how = (f"was stopped (signal {-done.returncode})" if done.returncode < 0
               else f"exited {done.returncode}")
        echo(_msg.format("error", f"{plan.slug}: `{argv[0]}` {how}; running the same download "
                                  f"again resumes it{detail}",
                         f"neuroatlas data download {plan.slug}"))
        return 1
    if output.failed:
        # the nsrr tool exits 0 with files it did not get
        n = len(output.failed)
        names = ", ".join(output.failed[:3]) + (", ..." if n > 3 else "")
        why = f" ({'; '.join(output.reasons)})" if output.reasons else ""
        refused = any("not authorized" in r.lower() for r in output.reasons)
        fix = (f"request access to {plan.ref or plan.slug} at sleepdata.org, then the same "
               f"download" if refused else f"neuroatlas data download {plan.slug}")
        echo(_msg.format("error", f"{plan.slug}: sleepdata.org did not serve "
                                  f"{_msg.plural(n, 'file')}{why}: {names}", fix))
        return 1
    return 0


class _NothingListed(IOError):
    """A listing with no recording in it: ``args`` = (what, the fix)."""


def _nsrr_list(ref: str, folder: str, token: Optional[str]) -> List[Dict[str, Any]]:
    """Every file under one folder of an NSRR dataset (its ``archive/``
    folders of superseded sheets left out), from the listing the nsrr tool
    reads."""
    query = urllib.parse.urlencode({"path": folder, "auth_token": token or ""})
    try:
        entries = _http_json(f"{NSRR_FILES.format(ref=ref)}?{query}")
    except (urllib.error.URLError, ValueError) as exc:     # never the URL: it holds the token
        raise IOError(f"sleepdata.org did not list {ref}/{folder}: "
                      f"{getattr(exc, 'reason', None) or type(exc).__name__}") from None
    if not isinstance(entries, list):
        raise IOError(f"sleepdata.org did not list {ref}/{folder}")
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.get("is_file"):
            out.append(entry)
        elif entry.get("file_name") != "archive" and entry.get("full_path"):
            out += _nsrr_list(ref, str(entry["full_path"]), token)
    return out


def _nsrr_first_commands(plan: DownloadPlan, token: Optional[str]) -> List[List[str]]:
    """The nsrr commands that fetch the first N recordings and their files:
    one per folder, naming its files (``--shallow --file=^(?:a|b)$``)."""
    ref = str(plan.ref)
    paths = [str(e["full_path"]).strip("/") for folder in (plan.folders or [""])
             for e in _nsrr_list(ref, folder, token)]
    chosen, plan.recordings = first_files(paths, int(plan.first or 0), plan.pattern)
    if not plan.recordings:
        raise _NothingListed(f"sleepdata.org lists no recording of {ref} for this token: an "
                             f"NSRR account sees a dataset's files once it is approved for it",
                             f"request access to {ref} at sleepdata.org, then the same download")
    by_folder: Dict[str, List[str]] = {}
    for path in chosen:
        folder, _, name = path.rpartition("/")
        by_folder.setdefault(folder, []).append(name)
    return [["nsrr", "download", f"{ref}/{folder}" if folder else ref, "--shallow",
             "--file=^(?:" + "|".join(re.escape(n) for n in names) + ")$"]
            for folder, names in by_folder.items()]


def _mark(plan: DownloadPlan) -> None:
    """After a download: a sample's folder says it is one (:data:`SAMPLE_MARK`);
    a whole download's folder no longer does."""
    if plan.dest is None:
        return
    mark = plan.dest / SAMPLE_MARK
    if plan.first:
        plan.dest.mkdir(parents=True, exist_ok=True)
        mark.write_text(json.dumps({"first": plan.first, "recordings": plan.recordings},
                                   indent=1) + "\n")
    elif mark.is_file():
        mark.unlink()


def sample_of(folder: Optional[Path]) -> Optional[Dict[str, Any]]:
    """What `data download --first N` left in *folder*: ``{"first": N,
    "recordings": [...]}``, or None when it holds a whole download."""
    if folder is None:
        return None
    try:
        mark = json.loads((folder / SAMPLE_MARK).read_text())
    except (OSError, ValueError):
        return None
    return mark if isinstance(mark, dict) else None


def run_download(plan: DownloadPlan, *, echo=None) -> int:
    """Execute a plan. Exit status: 0 done (or instructions printed), 1 a
    transfer or tool failed, 2 refused before anything ran.

    Progress goes to the item running (neuroatlas.progress), when there is
    one; *echo* gets one line per file, unpacked archive or subject."""
    if echo is None:
        echo = say
    if plan.handler in ("manual", "internal"):
        for line in describe_manual(plan):
            echo(line)
        return 0
    why = refusal(plan)
    if why:
        echo(_msg.format("error", f"{plan.slug}: {why}"))
        return 2

    token = None
    if plan.stdin_token:
        where = user_config.locate_token(plan.stdin_token)
        token = os.environ["NSRR_TOKEN"] if where == "$NSRR_TOKEN" else Path(where).read_text().strip()

    if plan.handler == "moabb":
        return _moabb_download(plan, echo)

    item = progress.current()
    if plan.handler == "nsrr" and plan.first:
        try:
            item.phase("listing the files")
            plan.commands = _nsrr_first_commands(plan, token)
        except _NothingListed as exc:
            echo(_msg.format("error", f"{plan.slug}: {exc.args[0]}", exc.args[1]))
            return 1
        except (OSError, ValueError) as exc:
            echo(_msg.format("error", f"{plan.slug}: {_msg.brief(str(exc))}",
                             f"neuroatlas data download {plan.slug} --first {plan.first}"))
            return 1
    elif plan.fetches or plan.listing:
        try:
            if plan.listing:
                item.phase("listing the files")
            _resolve_listing(plan)
            sizes = [f.size for f in plan.fetches]
            overall = bool(sizes) and all(sizes)
            if overall:
                # every size is known: one percent for the whole dataset
                item.phase("downloading", total=sum(sizes), unit="bytes")
            for i, f in enumerate(plan.fetches, 1):
                _fetch(f, echo, f"[{i}/{len(plan.fetches)}]", overall=overall)
            for archive in plan.unpack:
                _unpack(archive, echo, plan.keep_archive)
        except (OSError, urllib.error.URLError, zipfile.BadZipFile, ValueError) as exc:
            echo(_msg.format("error", f"{plan.slug}: {_msg.brief(str(exc))}",
                             f"neuroatlas data download {plan.slug}"))
            return 1
        _mark(plan)
        for line in plan.after:
            echo(line)
        return 0

    if plan.cwd:
        plan.cwd.mkdir(parents=True, exist_ok=True)
    if plan.dest:
        plan.dest.mkdir(parents=True, exist_ok=True)
    with _tool_folder(plan) as cwd:
        for argv in plan.commands:
            echo(f"  $ {' '.join(argv)}")
            if _run_tool(argv, plan, token, echo, cwd=cwd):
                return 1
    _mark(plan)
    for line in plan.after:
        echo(line)
    return 0


@contextlib.contextmanager
def _tool_folder(plan: DownloadPlan):
    """The folder a download tool runs in: ``plan.cwd``; or, for the nsrr
    tool when the dataset's folder is not named after the NSRR dataset
    (hpap_lab_full holds NSRR's homepap), a scratch folder where that name
    links to the dataset's folder, so what the tool writes under it lands
    there (and a rerun finds, and skips, what is already there)."""
    if not plan.link:
        yield plan.cwd
        return
    import tempfile

    scratch = Path(tempfile.mkdtemp(prefix="neuroatlas-nsrr-"))
    link = scratch / plan.link
    try:
        os.symlink(plan.dest, link, target_is_directory=True)
        yield scratch
    finally:
        # the link only, never what it points at; then the (empty) folder
        with contextlib.suppress(OSError):
            link.unlink()
        with contextlib.suppress(OSError):
            scratch.rmdir()


def source_text(plan: DownloadPlan) -> str:
    """Where a download comes from and goes, for its start line."""
    if plan.handler == "moabb":
        what = "with MOABB"
    elif plan.handler == "nsrr":
        what = "with the nsrr tool"
    elif plan.commands and plan.commands[0][0] == "wget":
        what = "from physionet.org with wget (--mirror aws is usually faster)"
    elif plan.source:
        what = f"from {urllib.parse.urlparse(plan.source).hostname}"
    elif plan.listing and "amazonaws" in plan.listing:
        what = "from PhysioNet's open-data copy on AWS"
    elif plan.listing or plan.fetches:
        url = plan.listing or plan.fetches[0].url
        what = f"from {urllib.parse.urlparse(url).hostname}"
    else:
        what = ""
    where = plan.dest if plan.dest is not None else plan.cwd
    return " ".join(part for part in (what, f"into {where}" if where else "") if part)


def result_text(status: int, counts: Dict[str, int]) -> str:
    """What a dataset's download did, for its result line: ``downloaded, 5
    files, 3.2 MB``, ``found, 25 files already complete``, ``failed``."""
    if status == 2:
        return "refused"
    if status:
        return "failed"
    files, skipped = counts.get("files", 0), counts.get("skipped", 0)
    nbytes, subjects = counts.get("bytes", 0), counts.get("subjects", 0)
    if subjects:
        return f"downloaded, {_msg.plural(subjects, 'subject')}" + (
            f", {progress.size(nbytes)}" if nbytes else "")
    if files:
        return (f"downloaded, {_msg.plural(files, 'file')}, {progress.size(nbytes)}"
                + (f"; {skipped} already complete" if skipped else ""))
    if skipped:
        return f"found, {_msg.plural(skipped, 'file')} already complete"
    if nbytes:
        return f"done, {progress.size(nbytes)} written"
    return "done"


def say(line: str) -> None:
    """What `data download` prints: an ``error:`` (or ``warning:``, ``note:``)
    line to stderr, everything else -- progress -- to stdout, flushed so it
    shows up at once in a log or a job's output file."""
    kind = line.split(":", 1)[0]
    if kind in _msg.KINDS:
        _msg.say(kind, line)
    else:
        print(line, flush=True)


def describe_manual(plan: DownloadPlan) -> List[str]:
    """How to get a dataset `data download` cannot fetch: ``<slug>: <why>``,
    then one indented line per step."""
    first, *rest = (plan.message or "").splitlines() or [""]
    return [f"{plan.slug}: {first}", *(f"  {line}" for line in rest)]
