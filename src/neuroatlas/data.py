"""Where each dataset's corpus is, whether it is there, and how to get it.

The facts come from the cohort manifest (``acquisition``, ``runtime_defaults``)
and the user's settings (``data_root``, ``dataset_paths``) -- the same
resolution ``embed`` and ``probe`` use, so a dataset `data status` reports as
found is the one a run will read.

Nothing here touches the network except :func:`download`, and that only when
asked to.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neuroatlas import _paths
from neuroatlas import config as user_config

# Keys a manifest may use for the raw corpus, most specific first (as prepare).
RAW_KEYS = ("raw_dir", "raw_root", "bids_root", "data_root")

# File types counted as a recording when a manifest does not say otherwise.
EEG_SUFFIXES = (".edf", ".bdf", ".rec", ".fif", ".set", ".vhdr", ".h5", ".hdf5", ".mat",
                ".npz", ".xdf", ".cnt", ".eeg")

# `data status` must answer in seconds even on a network filesystem holding
# a 2 TB cohort, so the count stops early and says it did.
SCAN_SECONDS = 8.0
SCAN_ENTRIES = 200_000

# How each acquisition.kind is fetched, and whether `data download` can do it.
HANDLERS = {
    "physionet": "physionet",
    "zenodo": "zenodo",
    "nsrr": "nsrr",
    "moabb": "moabb",
    "tuh": "manual",
    "figshare": "manual",
    "mendeley": "manual",
    "manual": "manual",
    "internal": "internal",
}

ACCESS = {
    "physionet": "open", "zenodo": "open", "moabb": "open", "figshare": "open",
    "mendeley": "open", "manual": "manual", "nsrr": "nsrr", "tuh": "tuh DUA",
    "internal": "internal",
}


def _spec(slug: str):
    from neuroatlas.benchmarking_helpers.registry.discovery import load_dataset_spec

    return load_dataset_spec(slug)


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
    return dict((_spec(slug).manifest or {}).get("acquisition") or {})


def raw_location(slug: str) -> Tuple[Optional[str], Optional[Path], List[str]]:
    """(key, path, unresolved variables) of the raw corpus."""
    cfg = resolved_config(slug)
    for key in RAW_KEYS:
        value = cfg.get(key)
        if value:
            unresolved = re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", str(value))
            return key, (None if unresolved else Path(str(value))), unresolved
    return None, None, []


def _count(root: Path, pattern: Optional[str]) -> Tuple[int, bool]:
    """(files matching, whether the scan stopped early)."""
    started, seen, hits = time.monotonic(), 0, 0
    suffix_re = re.compile(pattern.replace(".", r"\.").replace("**/", "").replace("*", ".*") + "$") \
        if pattern else None
    stack = [root]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for entry in it:
                    seen += 1
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif suffix_re is not None:
                        hits += bool(suffix_re.match(entry.name))
                    elif entry.name.lower().endswith(EEG_SUFFIXES):
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
    access: str
    handler: str
    state: str                       # found | partial | missing | not configured | ...
    path: Optional[Path] = None
    path_key: Optional[str] = None
    n_files: Optional[int] = None
    notes: List[str] = field(default_factory=list)
    channel_map: bool = False

    @property
    def found(self) -> bool:
        return self.state.startswith("found") or self.state == "prepared"


def _moabb_status(slug: str, st: DatasetStatus) -> DatasetStatus:
    manifest = _spec(slug).manifest or {}
    has_builder = bool((manifest.get("pipeline") or {}).get("preprocessor"))
    mne_data = os.environ.get("MNE_DATA") or str(Path.home() / "mne_data")
    st.path = Path(mne_data)
    if not has_builder:
        st.state = "fetched on first use"
        st.notes.append(f"MOABB downloads it into {mne_data} the first time `embed` reads it")
        return st
    try:
        from neuroatlas.extensions.datasets.dataio.bci import PREPROCESSED_SEARCH_PATHS
    except ImportError as exc:                    # braindecode / moabb not installed
        st.state = "unknown"
        st.notes.append(f"install the [bci] extra to check it ({exc.name} missing)")
        return st
    from neuroatlas.entrypoints._common import expand_dataset_paths

    candidates = [Path(p) for p in expand_dataset_paths(
        {f"p{i}": p for i, p in enumerate(PREPROCESSED_SEARCH_PATHS.get(slug, []))}).values()]
    hit = next((p for p in candidates if p.exists()), None)
    if hit:
        st.state, st.path = "prepared", hit
    else:
        st.state = "not prepared"
        if candidates:
            st.path = candidates[0]
        st.notes.append(f"`neuroatlas data download {slug}`, then `neuroatlas data prepare {slug}`")
    return st


def status(slug: str) -> DatasetStatus:
    acq = acquisition(slug)
    kind = acq.get("kind") or "manual"
    st = DatasetStatus(slug, kind, ACCESS.get(kind, kind), HANDLERS.get(kind, "manual"), "missing",
                       channel_map=_paths.configs_dir("channel_maps", f"{slug}.yaml").is_file())
    if kind == "nsrr" and not user_config.locate_token("nsrr"):
        st.notes.append("nsrr token not found")
    if kind == "moabb":
        return _moabb_status(slug, st)

    key, path, unresolved = raw_location(slug)
    st.path_key = key
    if key is None:
        st.state = "no path"
        st.notes.append("the manifest names no raw path")
        return st
    if unresolved:
        st.state = "not configured"
        st.notes.append(f"set {' and '.join(unresolved)}: `neuroatlas config init --data-root DIR`")
        return st
    st.path = path
    if not path.exists():
        return st
    expect = acq.get("expect") or {}
    n, truncated = _count(path, expect.get("glob"))
    st.n_files = n
    minimum = int(expect.get("min", 1))
    if n >= minimum:
        st.state = f"found ({'≥' if truncated else ''}{n} files)"
    elif n:
        st.state = "partial"
        st.notes.append(f"{n} files, expected at least {minimum}")
    else:
        st.state = "partial"
        st.notes.append("the folder exists but holds no recordings")
    return st


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------

@dataclass
class DownloadPlan:
    slug: str
    handler: str
    commands: List[List[str]] = field(default_factory=list)   # argv lists
    cwd: Optional[Path] = None
    dest: Optional[Path] = None
    message: Optional[str] = None      # for manual/internal: what a person must do
    stdin_token: Optional[str] = None  # name of the token fed on stdin
    size_gb: Optional[float] = None
    after: List[str] = field(default_factory=list)


def _download_root(path: Path, acq: Dict[str, Any]) -> Path:
    """Strip acquisition.subdir: the reader's path may point inside the download."""
    subdir = acq.get("subdir")
    if subdir and path.as_posix().rstrip("/").endswith("/" + str(subdir).strip("/")):
        return Path(path.as_posix().rstrip("/")[: -len(str(subdir).strip("/")) - 1])
    return path


def plan_download(slug: str) -> DownloadPlan:
    acq = acquisition(slug)
    kind = acq.get("kind") or "manual"
    handler = HANDLERS.get(kind, "manual")
    plan = DownloadPlan(slug, handler, size_gb=acq.get("size_gb"))
    landing = acq.get("upstream")

    if handler == "moabb":
        from neuroatlas.extensions.datasets.dataio.moabb_loader import MOABB_DATASETS

        cfg = MOABB_DATASETS.get(slug)
        if cfg is None:
            plan.handler, plan.message = "manual", f"{slug} has no MOABB configuration"
            return plan
        plan.dest = Path(os.environ.get("MNE_DATA") or Path.home() / "mne_data")
        plan.commands = [["moabb", f"{cfg.moabb_name}().download()"]]
        plan.after = [f"then `neuroatlas data prepare {slug}`"] if (
            ((_spec(slug).manifest or {}).get("pipeline") or {}).get("preprocessor")) else []
        return plan

    key, path, unresolved = raw_location(slug)
    if unresolved or path is None:
        plan.handler = "blocked"
        plan.message = ("set the data root first: `neuroatlas config init --data-root DIR`"
                        if unresolved else "the manifest names no raw path")
        return plan
    plan.dest = path

    if handler == "physionet" and acq.get("ref") and acq.get("version"):
        root = _download_root(path, acq)
        url = f"https://physionet.org/files/{acq['ref']}/{acq['version']}/"
        plan.dest = root
        plan.commands = [["wget", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3",
                          "-P", str(root), url]]
        return plan

    if handler == "zenodo" and acq.get("ref") and acq.get("file"):
        archive = str(acq["file"])
        parent = path.parent
        url = f"https://zenodo.org/api/records/{acq['ref']}/files/{archive}/content"
        plan.dest = parent
        plan.commands = [["wget", "-c", "-O", str(parent / archive), url]]
        if archive.endswith(".zip"):
            plan.commands.append(["unzip", "-n", "-q", str(parent / archive), "-d", str(parent)])
            plan.after = [f"expects the archive to unpack to {path}"]
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
            plan.after = [f"the nsrr tool writes {ref}/ under {cwd}; the reader expects {path} -- "
                          f"`neuroatlas config set {slug}.{key} {cwd / ref}` if they differ"]
        plan.cwd, plan.dest = cwd, cwd / ref
        plan.commands = [["nsrr", "download", sub] for sub in subpaths]
        plan.stdin_token = "nsrr"
        if slug == "shhs":
            plan.after.append("the SHHS reader needs the CoRe-Sleep preprocessed version, which "
                              "this does not build")
        return plan

    plan.handler = "manual" if handler != "internal" else "internal"
    reason = {
        "tuh": "needs a signed TUH data use agreement; credentials arrive by email",
        "internal": "not publicly licensed; obtain it from the authors",
    }.get(kind, "has no download API")
    parts = [f"{slug} {reason}."]
    if landing:
        parts.append(f"Get it from {landing}")
    if acq.get("note"):
        parts.append(" ".join(str(acq["note"]).split()))
    parts.append(f"Put it at {path}")
    plan.message = " ".join(parts)
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


def run_download(plan: DownloadPlan, *, echo=print) -> int:
    """Execute a plan. Returns a process exit status."""
    if plan.handler in ("manual", "internal", "blocked"):
        echo(f"{plan.slug}: {plan.handler} — {plan.message}")
        return 2 if plan.handler == "blocked" else 0

    token = None
    if plan.stdin_token:
        refusal = check_nsrr_allowed(plan.slug)
        if refusal:
            echo(f"error: {refusal}")
            return 2
        where = user_config.locate_token(plan.stdin_token)
        if where is None:
            echo(f"error: no {plan.stdin_token} token; put it in "
                 f"{user_config.token_file(plan.stdin_token)} (chmod 600)")
            return 2
        token = os.environ["NSRR_TOKEN"] if where == "$NSRR_TOKEN" else Path(where).read_text().strip()

    need = plan.size_gb or (NSRR_MIN_FREE_GB if plan.handler == "nsrr" else None)
    if need and plan.dest is not None:
        free = free_gb(plan.dest)
        if free is not None and free < need * 1.1:
            echo(f"error: {plan.slug} needs about {need:g} GB; {free:.0f} GB free under {plan.dest}")
            return 1

    if plan.handler == "moabb":
        from neuroatlas.extensions.datasets.dataio.moabb_loader import MOABB_DATASETS
        import moabb.datasets as moabb_ds

        os.environ.setdefault("MNE_DATA", str(plan.dest))
        cfg = MOABB_DATASETS[plan.slug]
        echo(f"moabb: {cfg.moabb_name}().download() into {plan.dest}")
        getattr(moabb_ds, cfg.moabb_name)().download()
        return 0

    tool = plan.commands[0][0]
    if shutil.which(tool) is None:
        hint = {"nsrr": "install it once: gem install --user-install nsrr irb",
                "wget": "install wget", "unzip": "install unzip"}.get(tool, "")
        echo(f"error: `{tool}` is not on PATH. {hint}")
        return 3
    if plan.cwd:
        plan.cwd.mkdir(parents=True, exist_ok=True)
    elif plan.dest:
        plan.dest.mkdir(parents=True, exist_ok=True)
    for argv in plan.commands:
        echo(f"$ {' '.join(argv)}")
        # The token goes to the tool's own prompt on stdin: never argv (visible
        # in `ps`), never the environment, never printed.
        done = subprocess.run(argv, cwd=plan.cwd,
                              input=(token + "\n") if token else None, text=True,
                              stdout=None, stderr=None)
        if done.returncode:
            echo(f"error: `{argv[0]}` exited {done.returncode}")
            return done.returncode
    for line in plan.after:
        echo(f"note: {line}")
    return 0
