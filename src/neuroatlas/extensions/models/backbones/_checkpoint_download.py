"""Fetching model checkpoints into the models root.

Each source type knows how to bring one checkpoint to its ``checkpoint_path``:
a file of a Hugging Face repository, a folder of one, a file committed to a
GitHub repository, a GitHub release asset, one member of a Google Drive zip,
or one file inside a public Docker image. Where the upstream file's SHA-256 is
recorded below it is checked before the file is put in place, so a changed or
truncated upstream file fails loudly instead of loading.

:func:`missing_files` says what a folder checkpoint still lacks, so a folder
that holds only a config is not mistaken for the weights.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from neuroatlas._paths import checkout_root, home

logger = logging.getLogger(__name__)


# --------------- what each source holds --------------- #

@dataclass(frozen=True)
class HubFolder:
    """A checkpoint that is a folder of a Hugging Face repository.

    ``per_checkpoint_subfolder``: the repository holds one subfolder per
    checkpoint, named like the local folder (PierreGtch/EEGNetv4 keeps
    ``EEGNetv4_BNCI2014001/{kwargs.pkl,model-params.pkl,...}``); otherwise the
    whole repository is the checkpoint. ``companions`` are other repositories
    the model also loads, each fetched whole into a sibling folder.
    """
    required: Tuple[str, ...]
    per_checkpoint_subfolder: bool = False
    companions: Tuple[Tuple[str, str], ...] = ()   # (sibling folder name, repo id)


HUB_FOLDERS: Dict[str, HubFolder] = {
    "PierreGtch/EEGNetv4": HubFolder(required=("kwargs.pkl", "model-params.pkl"),
                                     per_checkpoint_subfolder=True),
    "brain-bzh/reve-base": HubFolder(
        required=("config.json", "configuration_reve.py", "modeling_reve.py", "model.safetensors"),
        companions=(("reve-positions", "brain-bzh/reve-positions"),)),
}

# Files each companion repository must provide (it is downloaded whole).
COMPANION_REQUIRED: Dict[str, Tuple[str, ...]] = {
    "brain-bzh/reve-positions": ("config.json", "configuration_bank.py", "position_bank.py",
                                 "model.safetensors"),
}

# What an untrained REVE needs from brain-bzh/reve-base: the config and the
# modelling code it runs with trust_remote_code -- not the weights.
REVE_REPO = "brain-bzh/reve-base"
REVE_CODE_FILES = ("config.json", "configuration_reve.py", "modeling_reve.py")

# Hub files that re-package an upstream checkpoint: fetched, checked by SHA-256,
# and written at checkpoint_path as the ``{"state_dict": ...}`` torch file the
# wrapper loads.
#   eeg-telecom-paris/eegpt-large-official holds the 413 tensors of EEGPT's
#   eegpt_mcae_58chs_4s_large4E.ckpt (sha256 9d63ebc8...), without the
#   optimiser state; all 102 ``target_encoder.*`` tensors the wrapper loads are
#   bit-identical to the upstream file (compared 2026-10-02). The upstream
#   Figshare share is behind a browser challenge that no script passes.
HUB_SAFETENSORS_AS_TORCH: Dict[str, Tuple[str, str]] = {
    "eeg-telecom-paris/eegpt-large-official": (
        "weights.safetensors",
        "c45c31cb42f68f8b9c630a7739a8084ae5dd6a6608a0d732a5165f180c755a96"),
}

# Docker images that are the only official distribution of a checkpoint:
# image -> (path of the file inside the image, its SHA-256).
DOCKER_FILES: Dict[str, Tuple[str, str]] = {
    "yujjio/seizure_transformer": (
        "usr/local/lib/python3.10/dist-packages/wu_2025/model.pth",
        "79b14e4715fef055ba252ae3f7a072325907e3d5d11bf91a8070d4220995f65d"),
}

# Google Drive zips: url -> (the member to keep, its SHA-256).
DRIVE_ZIP_MEMBERS: Dict[str, Tuple[str, str]] = {
    "https://drive.google.com/uc?export=download&id=1FwjtO3JLd1Di0yRmz7g4B0niyY0gzQEd": (
        "ckpt_fold-01.pth",
        "402ce091662f297e0faf53cb903ed84829eaed68c8599a1706dbdd6aaff1f68c"),
}


def hub_repo(reference: str) -> str:
    """``org/repo`` from either that or the browser URL of the repository."""
    for prefix in ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"):
        if reference.startswith(prefix):
            reference = reference[len(prefix):]
            break
    return reference.strip("/")


def missing_files(checkpoint_path, source_type: str, source_reference: str) -> List[str]:
    """What a checkpoint still lacks on disk, as paths; [] when it is complete.

    A single-file checkpoint lacks itself when absent. A folder checkpoint
    (see :data:`HUB_FOLDERS`) lacks each required file, and each required
    file of its companion folders.
    """
    path = Path(checkpoint_path) if checkpoint_path else Path("")
    folder = HUB_FOLDERS.get(hub_repo(source_reference or "")) if source_type == "huggingface" else None
    if folder is None:
        return [] if str(path) not in ("", ".") and path.exists() else [str(path)]
    lacking = [str(path / name) for name in folder.required if not (path / name).is_file()]
    for sibling, repo in folder.companions:
        lacking += [str(path.parent / sibling / name) for name in COMPANION_REQUIRED.get(repo, ())
                    if not (path.parent / sibling / name).is_file()]
    return lacking


def reve_code_missing(checkpoint_dir) -> List[str]:
    """What an untrained REVE lacks: the config and code, and the position bank."""
    path = Path(checkpoint_dir)
    lacking = [str(path / n) for n in REVE_CODE_FILES if not (path / n).is_file()]
    lacking += [str(path.parent / "reve-positions" / n)
                for n in COMPANION_REQUIRED["brain-bzh/reve-positions"]
                if not (path.parent / "reve-positions" / n).is_file()]
    return lacking


def default_hub_cache() -> Path:
    """The Hugging Face cache transformers and huggingface_hub read by default."""
    return Path(os.environ.get("HF_HUB_CACHE") or (
        Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"))


def hub_cache_snapshot(repo_id: str, files, cache_dirs=None) -> Optional[Path]:
    """A cached snapshot of *repo_id* that holds every one of *files*, else None.

    A snapshot folder alone is not enough: an interrupted download or a
    config-only fetch leaves one without the weights.
    """
    folder = "models--" + repo_id.replace("/", "--")
    for cache in (cache_dirs or [default_hub_cache()]):
        snapshots = Path(cache) / folder / "snapshots"
        if not snapshots.is_dir():
            continue
        for snap in sorted(snapshots.iterdir()):
            if all((snap / name).is_file() for name in files):   # follows the blob symlinks
                return snap
    return None


@dataclass(frozen=True)
class ReveSources:
    """Where REVE's model files and its position bank load from.

    Each is the local folder when it is complete, else the repo id when the
    Hugging Face cache holds a complete snapshot (transformers reads it from
    there, offline too), else None; ``lacking`` lists what neither has.
    """
    model: Optional[str]
    positions: Optional[str]
    lacking: Tuple[str, ...]

    @property
    def from_hub_cache(self) -> bool:
        return REVE_REPO in (self.model,) or "brain-bzh/reve-positions" in (self.positions,)


def reve_sources(checkpoint_dir, random_init: bool) -> ReveSources:
    path = Path(checkpoint_dir)

    def pick(local: Path, repo: str, files):
        if all((local / n).is_file() for n in files):
            return str(local), []
        if hub_cache_snapshot(repo, files) is not None:
            return repo, []
        return None, [str(local / n) for n in files if not (local / n).is_file()]

    model, lacking = pick(path, REVE_REPO,
                          REVE_CODE_FILES if random_init else HUB_FOLDERS[REVE_REPO].required)
    positions, lacking_pos = pick(path.parent / "reve-positions", "brain-bzh/reve-positions",
                                  COMPANION_REQUIRED["brain-bzh/reve-positions"])
    return ReveSources(model, positions, tuple(lacking + lacking_pos))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: Optional[str], what: str) -> None:
    if not expected:
        return
    got = _sha256(path)
    if got != expected:
        raise FileNotFoundError(
            f"{what} has SHA-256 {got}, not the recorded {expected}: the upstream file "
            f"changed or the download was truncated. Nothing was put in place.")


def downloads_off() -> bool:
    return "1" in (os.environ.get("NEUROATLAS_OFFLINE"), os.environ.get("EEGBENCH_OFFLINE"))


def ensure_checkpoint(checkpoint_path, source_type: str, source_reference: str) -> Path:
    """Return a valid local path to the checkpoint, downloading if needed.

    Raises FileNotFoundError, saying what to do, when the checkpoint is absent
    and cannot be fetched (downloads off, a manual source, or a failure).
    """
    path = Path(checkpoint_path) if checkpoint_path else Path("")
    lacking = missing_files(path, source_type, source_reference)
    if not lacking:
        return path

    if downloads_off():
        detail = (f" (missing {', '.join(Path(p).name for p in lacking)})"
                  if path.is_dir() else "")
        raise FileNotFoundError(
            f"Checkpoint not found at {path}{detail}, and downloads are off. "
            f"Fetch it with `neuroatlas models download`, or pass --online."
        )

    logger.info("Checkpoint not found at %s — attempting auto-download (source_type=%s)", path, source_type)
    path.parent.mkdir(parents=True, exist_ok=True)

    handlers = {
        "huggingface": _download_huggingface,
        "github_release": _download_github,
        "github_release_asset": _download_release_asset,
        "google_drive_zip": _download_google_drive_zip,
        "docker_image": _download_docker_image_file,
        "figshare_private_share": _download_figshare_private_share,
    }
    handler = handlers.get(source_type)
    if handler is None:
        if source_type in ("github_figshare", "github", "local", "local_artifact"):
            raise FileNotFoundError(
                f"Checkpoint not found at {path}. It cannot be downloaded automatically: "
                f"get it from {source_reference} and place it at {path}."
            )
        raise FileNotFoundError(
            f"Checkpoint not found at {path}, and source_type={source_type!r} has no "
            f"download handler. Get it from {source_reference} and place it at {path}."
        )
    try:
        return handler(path, source_reference)
    except FileNotFoundError:
        raise
    except Exception as exc:
        hint = ""
        if source_type == "huggingface" and (
                type(exc).__name__ in ("GatedRepoError", "RepositoryNotFoundError")
                or any(f" {code} " in f" {exc} " for code in ("401", "403"))):
            hint = (f" The repository needs a Hugging Face token with access: accept its terms on "
                    f"https://huggingface.co/{hub_repo(source_reference)}, then put the token in "
                    f"{home() / 'hf_token'} (chmod 600) or $HF_TOKEN.")
        raise FileNotFoundError(
            f"Download failed for {path} from {source_reference}: "
            f"{type(exc).__name__}: {str(exc).strip()}{hint}"
        ) from exc


# --------------- tokens: where, never what --------------- #

# Where a Hugging Face token may live, most explicit first. The search order
# is the one `config show` reports (neuroatlas.config.locate_token).
_HF_TOKEN_SOURCES = (
    "NEUROATLAS_HF_TOKEN_FILE",           # explicit override, a path
    "$NEUROATLAS_HOME/hf_token",          # written next to config.yaml
    "${REPO}/.secrets/hf_token",           # per-checkout, gitignored
    "~/.cache/huggingface/token",          # what `huggingface-cli login` writes
)


def resolve_hf_token() -> str | None:
    """Return a Hugging Face token, or None with a warning naming where we looked.

    Never logs the token itself, and never writes it anywhere.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()

    # A checkout's .secrets/hf_token (config.locate_token looks there too).
    repo_root = checkout_root() or home()
    candidates = [
        os.environ.get("NEUROATLAS_HF_TOKEN_FILE"),
        home() / "hf_token",
        repo_root / ".secrets" / "hf_token",
        Path.home() / ".cache" / "huggingface" / "token",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            token = path.read_text().strip()
            if token:
                logger.info("Using the Hugging Face token from %s", path)
                return token

    logger.warning(
        "No Hugging Face token found (looked at $HF_TOKEN, then %s). "
        "Gated models will fail to download; the open ones are unaffected.",
        ", ".join(_HF_TOKEN_SOURCES[1:]),
    )
    return None


def resolve_github_token() -> str | None:
    """A GitHub token from $GITHUB_TOKEN, $GH_TOKEN or $NEUROATLAS_HOME/github_token.

    Only private release assets need one. Never logs the token.
    """
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    path = home() / "github_token"
    if path.is_file():
        token = path.read_text().strip()
        if token:
            logger.info("Using the GitHub token from %s", path)
            return token
    return None


# --------------- download backends --------------- #

_HF_FILENAME_MAP = {
    # Maps HF repo_id → path-within-repo when that differs from the local
    # checkpoint_path's basename. Only needed when the HF repo stores the
    # weight under a subfolder (e.g. neurolm's "checkpoints/VQ.pt").
    "Weibang/NeuroLM": "checkpoints/VQ.pt",
}


def download_hub_folder(local_dir: Path, repo_id: str, allow_patterns=None,
                        token: Optional[str] = None) -> Path:
    """Fetch (part of) a hub repository into *local_dir*, as plain files."""
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), allow_patterns=allow_patterns,
                      token=token)
    return local_dir


def fetch_reve_code(checkpoint_dir) -> Path:
    """What an untrained REVE needs: reve-base's config and code, and the position bank."""
    path = Path(checkpoint_dir)
    token = resolve_hf_token()
    download_hub_folder(path, REVE_REPO, allow_patterns=list(REVE_CODE_FILES) + ["LICENSE", "README.md"],
                        token=token)
    download_hub_folder(path.parent / "reve-positions", "brain-bzh/reve-positions", token=token)
    lacking = reve_code_missing(path)
    if lacking:
        raise FileNotFoundError(f"{REVE_REPO} did not provide {', '.join(lacking)}.")
    return path


def _download_huggingface(local_path: Path, repo_id: str) -> Path:
    """Download a checkpoint from HuggingFace Hub.

    *repo_id* may be written either way a spec author would naturally write
    it -- "org/repo" or the URL you get from the browser address bar. The
    Hub API only accepts the former, and a full URL otherwise fails at
    download time rather than at registration, so normalise it here.
    """
    repo_id = hub_repo(repo_id)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise FileNotFoundError(
            f"Checkpoint not found at {local_path}. Install `huggingface_hub` "
            f"to download it, or get it from https://huggingface.co/{repo_id}."
        )

    token = resolve_hf_token()
    folder = HUB_FOLDERS.get(repo_id)
    if folder is not None:
        return _download_hub_checkpoint_folder(local_path, repo_id, folder, token)
    if repo_id in HUB_SAFETENSORS_AS_TORCH:
        return _download_safetensors_as_torch(local_path, repo_id, token)

    repo_filename = _HF_FILENAME_MAP.get(repo_id, local_path.name)
    # local_dir is chosen so that hf_hub_download(local_dir/repo_filename)
    # lands at local_path exactly. local_path may be nested inside a
    # per-model subtree (artifacts/models/foundation/<model>/...).
    if "/" in repo_filename:
        depth = repo_filename.count("/")
        local_dir = local_path
        for _ in range(depth + 1):
            local_dir = local_dir.parent
    else:
        local_dir = local_path.parent
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=repo_filename,
        local_dir=str(local_dir),
        token=token,
    )
    logger.info("Downloaded %s from HuggingFace %s", repo_filename, repo_id)
    return Path(downloaded)


def _download_hub_checkpoint_folder(local_path: Path, repo_id: str, folder: HubFolder,
                                    token: Optional[str]) -> Path:
    if folder.per_checkpoint_subfolder:
        # the repo's <name>/ lands at local_path when local_dir is its parent
        download_hub_folder(local_path.parent, repo_id, allow_patterns=[f"{local_path.name}/*"],
                            token=token)
    else:
        download_hub_folder(local_path, repo_id, token=token)
    for sibling, companion in folder.companions:
        download_hub_folder(local_path.parent / sibling, companion, token=token)
    lacking = missing_files(local_path, "huggingface", repo_id)
    if lacking:
        raise FileNotFoundError(
            f"https://huggingface.co/{repo_id} did not provide "
            f"{', '.join(lacking)}; the repository layout may have changed.")
    logger.info("Downloaded %s from HuggingFace %s", local_path.name, repo_id)
    return local_path


def _download_safetensors_as_torch(local_path: Path, repo_id: str, token: Optional[str]) -> Path:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load as load_bytes

    filename, sha256 = HUB_SAFETENSORS_AS_TORCH[repo_id]
    with tempfile.TemporaryDirectory(prefix="neuroatlas_dl_", dir=str(local_path.parent)) as tmp:
        fetched = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=tmp, token=token))
        _verify(fetched, sha256, f"https://huggingface.co/{repo_id}/{filename}")
        # read into memory rather than memory-map: an open map keeps the file
        # busy and the temporary folder cannot be removed on NFS
        state = load_bytes(fetched.read_bytes())
        partial = Path(tmp) / local_path.name
        torch.save({"state_dict": state,
                    "neuroatlas_source": f"https://huggingface.co/{repo_id}/{filename} (sha256 {sha256})"},
                   str(partial))
        del state
        shutil.move(str(partial), str(local_path))
    logger.info("Wrote %s from HuggingFace %s/%s", local_path, repo_id, filename)
    return local_path


def _download_github(local_path: Path, repo_url: str) -> Path:
    """Clone a GitHub repo (shallow) and copy the checkpoint file.

    Supports repos where weights are committed directly (SleepFM, TF-C).
    Uses a mapping of known repos → internal checkpoint paths.
    """
    _GITHUB_CHECKPOINT_MAP = {
        "zou-group/sleepfm-clinical": "sleepfm/checkpoints/model_base/best.pt",
        "ycq091044/BIOT": "pretrained-models/EEG-PREST-16-channels.ckpt",
        "935963004/LaBraM": "checkpoints/labram-base.pth",
    }

    # Extract org/repo from URL
    repo_slug = repo_url.rstrip("/").split("github.com/")[-1].split(".git")[0]
    internal_path = _GITHUB_CHECKPOINT_MAP.get(repo_slug)

    if internal_path is None:
        raise FileNotFoundError(
            f"Checkpoint not found at {local_path}. No known file path for "
            f"GitHub repo {repo_slug}: get it from {repo_url} and place it at {local_path}."
        )

    with tempfile.TemporaryDirectory(prefix="eegbench_dl_") as tmpdir:
        clone_url = f"https://github.com/{repo_slug}.git"
        logger.info("Cloning %s (shallow) to download checkpoint...", clone_url)
        subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, os.path.join(tmpdir, "repo")],
            check=True, capture_output=True,
        )
        src = Path(tmpdir) / "repo" / internal_path
        if not src.exists():
            raise FileNotFoundError(
                f"Expected checkpoint at {internal_path} in {repo_slug} but not found after cloning."
            )
        shutil.copy2(str(src), str(local_path))

        # Also copy config.json / LICENSE if present alongside
        for extra in ("config.json", "LICENSE"):
            extra_src = src.parent / extra
            extra_dst = local_path.parent / extra
            if extra_src.exists() and not extra_dst.exists():
                shutil.copy2(str(extra_src), str(extra_dst))

    logger.info("Downloaded %s from %s", local_path.name, repo_slug)
    return local_path


def _download_release_asset(local_path: Path, asset_url: str) -> Path:
    """Fetch a file that is a GitHub release asset, by its direct URL.

    Unlike _download_github, which clones a repository and copies a committed
    file, this expects source_reference to be the asset link itself
    (``.../releases/download/<tag>/<file>``). Three S-TEEGformer checkpoints and
    the supervised sleep baselines are published that way.

    A private repository answers 404 to an unauthenticated request, so the error
    says so rather than reporting a missing file.
    """
    import json
    import urllib.error
    import urllib.request

    token = resolve_github_token()

    def fetch(url, accept="application/octet-stream"):
        request = urllib.request.Request(url, headers={"Accept": accept})
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        return urllib.request.urlopen(request)

    def save(response):
        partial = local_path.with_name(local_path.name + ".partial")
        with open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle)
        partial.replace(local_path)

    logger.info("Downloading release asset %s", asset_url)
    try:
        with fetch(asset_url) as response:
            save(response)
        return local_path
    except urllib.error.HTTPError as exc:
        if exc.code not in (401, 403, 404):
            raise
        # A browse-style /releases/download/ link 404s on a private repository
        # even with a token; only the API asset endpoint honours one. Resolve
        # the asset id and retry there before giving up.
        if not token:
            raise FileNotFoundError(
                f"{asset_url} returned HTTP {exc.code}. If the repository is still "
                f"private, put a GitHub token that can read it in "
                f"{home() / 'github_token'} (chmod 600) or $GITHUB_TOKEN, or download "
                f"the asset by hand and place it at {local_path}."
            ) from exc
        try:
            _, _, rest = asset_url.partition("github.com/")
            owner, repo, _, _, tag, filename = rest.split("/")[:6]
            api = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
            with fetch(api, "application/vnd.github+json") as response:
                release = json.load(response)
            asset_id = next(a["id"] for a in release["assets"] if a["name"] == filename)
            with fetch(f"https://api.github.com/repos/{owner}/{repo}/releases/assets/{asset_id}") as response:
                save(response)
            return local_path
        except Exception as inner:
            raise FileNotFoundError(
                f"{asset_url} returned HTTP {exc.code}, and the API fallback with your "
                f"GitHub token failed ({inner}). Download the asset by hand and place it "
                f"at {local_path}."
            ) from exc


def _download_google_drive_zip(local_path: Path, drive_url: str) -> Path:
    """Fetch a zip the upstream authors host on Google Drive and pull one file out.

    SleePyCo publishes its checkpoints this way, linked from the Main Results
    table of github.com/gist-ailab/SleePyCo. We download from them rather than
    redistributing the weights, so the file lands here byte-identical to theirs.

    Only the recorded member (see :data:`DRIVE_ZIP_MEMBERS`) is kept, written
    to *local_path* after its SHA-256 is checked.
    """
    import urllib.request
    import zipfile

    member, sha256 = DRIVE_ZIP_MEMBERS.get(drive_url, (None, None))
    with tempfile.TemporaryDirectory(prefix="eegbench_gd_") as tmpdir:
        archive = Path(tmpdir) / "download.zip"
        logger.info("Downloading %s", drive_url)
        with urllib.request.urlopen(drive_url) as response, open(archive, "wb") as handle:
            shutil.copyfileobj(response, handle)
        if not zipfile.is_zipfile(archive):
            # Drive serves an HTML consent page instead of the file when the
            # object is large enough to trigger its virus-scan interstitial.
            raise FileNotFoundError(
                f"{drive_url} did not return a zip. Open it in a browser, unzip it, "
                f"and place {member or 'the .pth inside'} at {local_path}."
            )
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            if member is None:
                members = [n for n in names if n.endswith(".pth")]
                if not members:
                    raise FileNotFoundError(f"No .pth inside the archive at {drive_url}.")
                member = members[0]
            elif member not in names:
                raise FileNotFoundError(
                    f"The archive at {drive_url} no longer holds {member} (it has {', '.join(names)}).")
            extracted = Path(tmpdir) / "member"
            with zf.open(member) as src, open(extracted, "wb") as dst:
                shutil.copyfileobj(src, dst)
        _verify(extracted, sha256, f"{member} from {drive_url}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(local_path))
        logger.info("Extracted %s from the upstream archive", member)
    return local_path


def _download_docker_image_file(local_path: Path, image_ref: str) -> Path:
    """Pull one file out of a public Docker Hub image, without docker.

    The registry's HTTP API is anonymous for public images: fetch a pull token,
    the image manifest, then stream the layers (newest first) through tarfile
    until the recorded path turns up. Only that file is written; the layers are
    never stored. SeizureTransformer's authors publish their weights only this
    way (``docker pull yujjio/seizure_transformer``).
    """
    import json
    import tarfile
    import urllib.request

    image = image_ref.split("://", 1)[-1]
    repo, _, tag = image.partition(":")
    tag = tag or "latest"
    if repo not in DOCKER_FILES:
        raise FileNotFoundError(f"No file recorded for docker image {repo}.")
    member, sha256 = DOCKER_FILES[repo]
    if "/" not in repo:
        repo = f"library/{repo}"

    with urllib.request.urlopen(
            f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull",
            timeout=60) as response:
        token = json.load(response)["token"]

    def get(url, accept):
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": accept})
        return urllib.request.urlopen(request, timeout=120)

    manifest_types = ", ".join([
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.index.v1+json",
    ])
    registry = f"https://registry-1.docker.io/v2/{repo}"
    with get(f"{registry}/manifests/{tag}", manifest_types) as response:
        manifest = json.load(response)
    if "manifests" in manifest:          # a multi-platform index: take linux/amd64
        chosen = next((m for m in manifest["manifests"]
                       if m.get("platform", {}).get("os") == "linux"
                       and m.get("platform", {}).get("architecture") == "amd64"), manifest["manifests"][0])
        with get(f"{registry}/manifests/{chosen['digest']}", manifest_types) as response:
            manifest = json.load(response)

    wanted = {member, "./" + member, "/" + member}
    local_path.parent.mkdir(parents=True, exist_ok=True)
    partial = local_path.with_name(local_path.name + ".partial")
    for layer in reversed(manifest["layers"]):
        logger.info("Scanning layer %s (%.0f MB) of %s for %s",
                    layer["digest"][:19], layer.get("size", 0) / 1e6, repo, member)
        with get(f"{registry}/blobs/{layer['digest']}", "*/*") as response, \
                tarfile.open(fileobj=response, mode="r|*") as tar:
            for entry in tar:
                if entry.name in wanted and entry.isfile():
                    with tar.extractfile(entry) as src, open(partial, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    try:
                        _verify(partial, sha256, f"{member} from docker://{image}")
                    except FileNotFoundError:
                        partial.unlink(missing_ok=True)
                        raise
                    partial.replace(local_path)
                    logger.info("Extracted %s from docker://%s", member, image)
                    return local_path
    raise FileNotFoundError(f"docker://{image} does not contain {member}.")


def _download_figshare_private_share(local_path: Path, share_url: str) -> Path:
    """Figshare private shares cannot be fetched by a script.

    The share page sits behind a browser challenge, and the v2 API has no
    endpoint that resolves a private-link token to its articles: the listing
    this used to call (``/v2/articles?private_link=...``) ignores the token
    and returns the newest *public* articles, so it downloaded strangers'
    files. Say what to do instead.
    """
    raise FileNotFoundError(
        f"{share_url} is a Figshare private share, which only a browser can open. "
        f"Download {local_path.name} from it and place it at {local_path}."
    )
