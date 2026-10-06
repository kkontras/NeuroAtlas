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

# What `data download` does for each handler -- the `download` column.
DOWNLOAD = {
    "physionet": "automatic", "zenodo": "automatic", "url": "automatic",
    "moabb": "automatic", "nsrr": "with NSRR token", "manual": "instructions",
    "internal": "from the authors", "refused": "refused", "blocked": "blocked",
}

# Kept for callers of the old column; `access` is now the manifest's kind,
# the same word `list datasets` prints.
ACCESS = {kind: kind for kind in HANDLERS}

# Every state `data status` can report, with its meaning: printed under the
# table and the vocabulary the guide should list.
STATES = {
    "found": "every expected file is there (n files, or n/N when the total is known)",
    "partial": "some files, fewer than expected (n/N): an interrupted download?",
    "empty": "the folder exists but holds no recordings",
    "missing": "the folder does not exist",
    "prepared": "the prepared file a reader needs is there",
    "not prepared": "the raw data may be there, but the prepared file the reader needs is not",
    "not downloaded": "MOABB data not under $MNE_DATA yet",
    "not configured": "no data root set: `neuroatlas config init --data-root DIR`",
    "no path": "the manifest names no folder for it",
}


#: The note a dataset whose download needs an NSRR token carries when there is
#: none (`data status` gathers them into one line under the table).
NO_NSRR_TOKEN = "no nsrr token"


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
    return dict(_manifest(slug).get("acquisition") or {})


def _unresolved(value: str) -> List[str]:
    return re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", str(value))


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
    download: str = ""               # what `data download` does (DOWNLOAD)

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
        st.notes.append(f"which folder under {base} holds it is not recorded (no acquisition.nemar)")
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
        st.notes.append(f"fix: neuroatlas data download {slug} (finishes it)")
    elif n:
        st.state = f"found ({n}/{expected} files)" if expected else f"found ({n} files)"
    else:
        st.state = "empty"
        st.notes.append(f"fix: neuroatlas data download {slug}")


def _prepared_hit(slug: str, explicit: Optional[str]) -> Tuple[Optional[Path], List[Path]]:
    """(the prepared file a reader would open, or None; every candidate)."""
    if explicit:
        return (Path(explicit) if Path(explicit).is_file() else None), [Path(explicit)]
    from neuroatlas.extensions.datasets.dataio.bci import PREPROCESSED_SEARCH_PATHS
    from neuroatlas.entrypoints._common import expand_dataset_paths

    candidates = [Path(p) for p in expand_dataset_paths(
        {f"p{i}": p for i, p in enumerate(PREPROCESSED_SEARCH_PATHS.get(slug, []))}).values()]
    return next((p for p in candidates if p.is_file()), None), candidates


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
        st.state = "unknown"
        st.notes += _note_lines(f"cannot check: {exc.name} is not installed",
                                "pip install -e '.[bci]' && pip install --no-deps 'moabb==1.2.0'")
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


def status(slug: str) -> DatasetStatus:
    acq = acquisition(slug)
    kind = acq.get("kind") or "manual"
    handler = HANDLERS.get(kind, "manual")
    if acq.get("unusable_download"):
        handler = "refused"
    st = DatasetStatus(slug, kind, kind, handler, "missing",
                       channel_map=_paths.configs_dir("channel_maps", f"{slug}.yaml").is_file(),
                       download=DOWNLOAD.get(handler, handler))
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
                    st.notes.append(f"fix: neuroatlas data download {slug}, then "
                                    f"neuroatlas data prepare {slug}")
            return st
        _nemar_state(slug, acq, st)
        return st
    if acq.get("prepared_only"):
        _prepared_status(slug, st, acq, needs_build=False)
        if st.state == "not prepared" and acq.get("dest"):
            _, raw_path, _ = raw_location(slug)
            if raw_path is not None:
                n, _ = _count(raw_path, (acq.get("expect") or {}).get("glob")) \
                    if raw_path.is_dir() else (0, False)
                have = f"{n} there now" if n else "none there yet"
                st.notes.append(f"data download fetches the raw EDFs into {raw_path} ({have}), "
                                f"which the benchmark does not read")
        return st

    key, path, unresolved = raw_location(slug)
    st.path_key = key
    if key is None and path is None:
        st.state = "no path"
        st.notes.append("the manifest names no raw path")
        return st
    if unresolved:
        st.state = "not configured"
        st.notes += _note_lines(f"{' and '.join(unresolved)} not set",
                                "neuroatlas config init --data-root DIR")
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
    if n >= minimum or (truncated and n):
        st.state = f"found ({'≥' if truncated else ''}{n}{total} files)"
    elif n:
        st.state = f"partial ({n}{total} files)"
        st.notes += _note_lines(f"{n} of {minimum} expected files: an interrupted download?",
                                f"neuroatlas data download {slug} (resumes it)"
                                if handler in ("physionet", "zenodo", "url") else
                                "complete the copy")
    else:
        st.state = "empty"
        st.notes.append("the folder exists but holds no recordings"
                        + (f" matching {expect['glob']}" if expect.get("glob") else ""))
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


def _download_root(path: Path, acq: Dict[str, Any]) -> Path:
    """Strip acquisition.subdir: the reader's path may point inside the download."""
    subdir = acq.get("subdir")
    if subdir and path.as_posix().rstrip("/").endswith("/" + str(subdir).strip("/")):
        return Path(path.as_posix().rstrip("/")[: -len(str(subdir).strip("/")) - 1])
    return path


def _archive_root(path: Path, archive: str) -> Path:
    """Where to unpack an archive so its top folder lands at ``path``.

    BIDS_CHB-MIT.zip holds BIDS_CHB-MIT/..., and the reader reads
    .../BIDS_CHB-MIT, so it unpacks into the parent. An archive whose stem
    is not the reader's folder (dodh.zip -> dodh/) unpacks into ``path``.
    """
    return path.parent if Path(archive).stem == path.name else path


ZENODO_FILE = "https://zenodo.org/api/records/{ref}/files/{name}/content"
ZENODO_RECORD = "https://zenodo.org/api/records/{ref}"
S3_BUCKET = "https://physionet-open.s3.amazonaws.com"


def plan_download(slug: str, *, mirror: str = "physionet",
                  keep_archive: bool = False) -> DownloadPlan:
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

    if handler == "moabb":
        from neuroatlas.extensions.datasets.dataio.moabb_loader import MOABB_DATASETS

        cfg = MOABB_DATASETS.get(slug)
        if cfg is None:
            plan.handler, plan.message = "manual", f"{slug} has no MOABB configuration"
            return plan
        plan.dest = mne_data_dir()
        plan.commands = [["moabb", f"{cfg.moabb_name}().download()"]]
        pipeline = _manifest(slug).get("pipeline") or {}
        if (isinstance(pipeline.get("preprocessor"), dict)
                and pipeline.get("required", True) is not False):
            plan.after = [f"then: neuroatlas data prepare {slug}"]
        return plan

    key, path, unresolved = raw_location(slug)
    if handler not in ("manual", "internal") and (unresolved or path is None):
        plan.handler = "blocked"
        plan.message = ("no data root set\nfix: neuroatlas config init --data-root DIR"
                        if unresolved else "the manifest names no raw path")
        return plan
    plan.dest = path

    if handler == "physionet" and acq.get("ref") and acq.get("version"):
        root = _download_root(path, acq)
        plan.dest = root
        prefix = f"{acq['ref']}/{acq['version']}/"
        if mirror == "aws":
            # PhysioNet's open-data mirror on AWS: anonymous HTTPS, no
            # credentials, typically far faster than physionet.org.
            plan.listing = f"{S3_BUCKET}/?list-type=2&prefix={prefix}"
            return plan
        url = f"https://physionet.org/files/{prefix}"
        # --progress=dot:mega: wget's output is read, not shown (_ToolOutput),
        # and its dots become the bytes on the live line
        plan.commands = [["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3",
                          "--progress=dot:mega", "--reject", "index.html*", "-P", str(root), url]]
        plan.after = [f"faster: neuroatlas data download {slug} --mirror aws (PhysioNet's "
                      f"open-data copy on AWS, no account needed)"]
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
            target = path / url.rstrip("/").rsplit("/", 1)[-1]
            plan.fetches.append(Fetch(url, target, md5=(item.get("md5") if isinstance(item, dict) else None)))
            if target.suffix == ".zip":
                plan.unpack.append(target)
        return plan

    if handler == "nsrr" and acq.get("ref"):
        ref = str(acq["ref"])
        posix = path.as_posix()
        marker = f"/{ref}/"
        if marker in posix + "/":
            cwd = Path((posix + "/").split(marker)[0])
            inside = (posix + "/").split(marker, 1)[1].strip("/")
            subpaths = [f"{ref}/{inside}" if inside else ref]
            if inside:
                subpaths.append(f"{ref}/datasets")          # the metadata sheets
        else:
            cwd, subpaths = path.parent, [ref]
            plan.after = [f"the nsrr tool writes {ref}/ under {cwd}; the reader expects {path} "
                          f"(neuroatlas config set {slug}.{key} {cwd / ref} if they differ)"]
        plan.cwd, plan.dest = cwd, cwd / ref
        plan.commands = [["nsrr", "download", sub] for sub in subpaths]
        plan.stdin_token = "nsrr"
        return plan

    plan.handler = "manual" if handler != "internal" else "internal"
    reason = {
        "tuh": "needs a signed TUH data use agreement (credentials arrive by email)",
        "internal": "from the authors only (not publicly licensed)",
    }.get(kind, "fetched by hand (no download API, or access on request)")
    # one line each: why by hand, where from, what the reader reads, where it goes
    lines = [reason]
    if landing:
        lines.append(f"get it from {landing}")
    if acq.get("note"):
        lines.append(" ".join(str(acq["note"]).split()))
    if acq.get("prepared_only"):
        _, what, fixes = _msg.split(manifest_text(acq["prepared_only"]))
        lines += what + fixes
    elif path is not None:
        setting = f"{slug}.{key}" if key else None
        lines.append(f"put it at {path}" + (f", or point at your copy: neuroatlas config set "
                                            f"{setting} DIR" if setting else ""))
    plan.message = "\n".join(lines)
    return plan


def free_gb(path: Path) -> Optional[float]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / 1e9
    except OSError:
        return None


NSRR_MIN_FREE_GB = 50.0
TOOL_HINTS = {"nsrr": "install it once: gem install --user-install nsrr irb",
              "wget": "install wget", "unzip": "install unzip"}


def check_nsrr_allowed(slug: str) -> Optional[str]:
    """An NSRR token opens only the cohorts its owner is approved for."""
    allowed = os.environ.get("NEUROATLAS_NSRR_DATASETS")
    if not allowed:
        return None
    names = {n.strip().lower() for n in allowed.split(",") if n.strip()}
    ref = str(acquisition(slug).get("ref") or slug).lower()
    if ref not in names and slug not in names:
        return (f"{slug} is not in NEUROATLAS_NSRR_DATASETS ({allowed}); your NSRR token "
                f"is not approved for it")
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
            return (f"no {plan.stdin_token} token\n"
                    f"fix: neuroatlas config token {plan.stdin_token}")
    need = plan.size_gb or (NSRR_MIN_FREE_GB if plan.handler == "nsrr" else None)
    if need and plan.unpack:
        need *= 2            # the archive and what it unpacks to, until the archive goes
    if need and plan.dest is not None:
        free = free_gb(plan.dest)
        if free is not None and free < need * 1.1:
            return f"{plan.slug} needs about {need:g} GB; {free:.0f} GB free under {plan.dest}"
    if plan.handler == "moabb":
        import importlib.util

        if importlib.util.find_spec("moabb") is None:
            return ("MOABB is not installed\n"
                    "fix: pip install -e '.[fm,bci]' && pip install --no-deps 'moabb==1.2.0' "
                    "(in your NeuroAtlas clone)")
    tools = [argv[0] for argv in plan.commands if argv and argv[0] != "moabb"]
    for tool in dict.fromkeys(tools):
        if shutil.which(tool) is None:
            return _msg.compose(f"`{tool}` is not on PATH", TOOL_HINTS.get(tool))
    return None


def describe(plan: DownloadPlan) -> List[str]:
    """What a plan would do, one line each (the dry run)."""
    lines = []
    where = f"  (in {plan.cwd})" if plan.cwd else ""
    for argv in plan.commands:
        lines.append(f"command: {' '.join(argv)}{where}")
    if plan.listing and not plan.fetches:
        source = ("every file of the record" if "zenodo" in plan.listing
                  else "every file under the prefix")
        lines.append(f"fetch: {source} listed by {plan.listing} -> {plan.dest}/")
    for f in plan.fetches:
        lines.append(f"fetch: {f.url} -> {f.dest}" + (f"  (md5 {f.md5})" if f.md5 else ""))
    for archive in plan.unpack:
        lines.append(f"unpack: {archive} -> {archive.parent}/"
                     + ("" if plan.keep_archive else "  (then delete the archive; "
                        "--keep-archive keeps it)"))
    for line in plan.after:
        lines.append(line)
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
        raise IOError(f"{f.dest.name}: got {part.stat().st_size} bytes, expected {f.size}\n"
                      f"fix: run the same download again (it resumes)")
    if f.md5:
        item.update(note=f"{label} checking {f.dest.name}")
        if _md5(part) != f.md5:
            part.unlink()
            raise IOError(f"{f.dest.name}: md5 mismatch (expected {f.md5}); the partial file "
                          f"was removed\nfix: run the same download again")
    part.replace(f.dest)
    item.count("files")
    echo(f"  {label} {f.dest.name}: {progress.size(f.dest.stat().st_size)} in "
         f"{progress.duration(time.monotonic() - started)}" + (", md5 ok" if f.md5 else ""))


def _resolve_listing(plan: DownloadPlan) -> None:
    """Fill sizes and md5s (or the whole file list) from the record's API."""
    if not plan.listing:
        return
    if "zenodo.org" in plan.listing:
        files = {f["key"]: f for f in _zenodo_files(plan.listing)}
        if not plan.fetches:                          # the whole record
            ref = plan.listing.rstrip("/").rsplit("/", 1)[-1]
            for name, f in sorted(files.items()):
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
    for key, size, md5 in _s3_keys(plan.listing):
        rel = key[len(prefix):]
        if not rel or rel.endswith("/"):
            continue
        plan.fetches.append(Fetch(f"{S3_BUCKET}/{key}", plan.dest / rel, size=size, md5=md5))


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
    echo(f"  unpacked {archive.name} -> {target}/ ({files} files)")
    if not keep:
        archive.unlink()
        echo(f"  deleted {archive.name} (pass --keep-archive to keep it)")


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
                                  + _msg.brief(_msg.exception_text(exc))))
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
                                          + _msg.brief(_msg.exception_text(exc))))
                return 1
            item.update(advance=1)
            item.count("subjects")
            echo(f"  [{i}/{len(subjects)}] subject {subject} "
                 f"({progress.duration(time.monotonic() - began)})")
    finally:
        if written is not None:
            written._next = 0.0
            item.count("bytes", written() or 0)
    logging.getLogger(__name__).info("%s: %d subject(s) in %.0f s", plan.slug, len(subjects),
                                     time.monotonic() - started)
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

    def __init__(self, path: str, *, wget: bool, item=None, echo=None, written=None):
        self.path, self.wget, self.echo = path, wget, echo
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


def _run_tool(argv: List[str], plan: DownloadPlan, token: Optional[str], echo) -> int:
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
        written = (progress.DiskBytes(plan.dest or plan.cwd)
                   if isinstance(item, progress.Progress) else None)
        output = _ToolOutput(path, wget=False, item=item, echo=echo, written=written)
        extra = {}
    item.watch(output.poll)
    try:
        # The token goes to the tool's own prompt on stdin: never argv (visible
        # in `ps`), never the environment, never printed.
        done = subprocess.run(argv, cwd=plan.cwd,
                              input=(token + "\n") if token else None, text=True,
                              stdout=fd, stderr=subprocess.STDOUT, **extra)
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
        echo(_msg.format("error", f"{plan.slug}: `{argv[0]}` {how}{detail}",
                         f"neuroatlas data download {plan.slug} (it carries on where this "
                         f"stopped)"))
        return 1
    return 0


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
    if plan.fetches or plan.listing:
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
            echo(_msg.format("error", f"{plan.slug}: {_msg.brief(str(exc))}"))
            return 1
        for line in plan.after:
            echo(line)
        return 0

    if plan.cwd:
        plan.cwd.mkdir(parents=True, exist_ok=True)
    elif plan.dest:
        plan.dest.mkdir(parents=True, exist_ok=True)
    for argv in plan.commands:
        echo(f"  $ {' '.join(argv)}")
        if _run_tool(argv, plan, token, echo):
            return 1
    for line in plan.after:
        echo(line)
    return 0


def source_text(plan: DownloadPlan) -> str:
    """Where a download comes from and goes, for its start line."""
    if plan.handler == "moabb":
        what = "with MOABB"
    elif plan.handler == "nsrr":
        what = "with the nsrr tool"
    elif plan.commands and plan.commands[0][0] == "wget":
        what = "from physionet.org with wget"
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
