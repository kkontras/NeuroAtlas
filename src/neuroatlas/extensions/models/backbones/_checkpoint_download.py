"""Auto-download utility for foundation model checkpoints.

When a checkpoint file is missing locally, attempts to fetch it from the
source specified in the CheckpointSpec (HuggingFace, GitHub, Google Drive).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from neuroatlas._paths import home, workspace_root

logger = logging.getLogger(__name__)


def ensure_checkpoint(checkpoint_path: str | Path, source_type: str, source_reference: str) -> Path:
    """Return a valid local path to the checkpoint, downloading if needed.

    Args:
        checkpoint_path: Expected local path (may not exist yet).
        source_type: One of "huggingface", "github_release", "google_drive".
        source_reference: URL or repo ID for downloading.

    Returns:
        Path to the local checkpoint file.

    Raises:
        FileNotFoundError: If download fails or source_type is unknown.
    """
    path = Path(checkpoint_path) if checkpoint_path else Path("")
    if path.exists():
        return path

    if "1" in (os.environ.get("NEUROATLAS_OFFLINE"), os.environ.get("EEGBENCH_OFFLINE")):
        raise FileNotFoundError(
            f"Checkpoint not found at {path}, and downloads are off. "
            f"Fetch it with `neuroatlas models download`, or pass --online."
        )

    logger.info("Checkpoint not found at %s — attempting auto-download (source_type=%s)", path, source_type)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if source_type == "huggingface":
            return _download_huggingface(path, source_reference)
        elif source_type == "github_release":
            return _download_github(path, source_reference)
        elif source_type == "figshare_private_share":
            return _download_figshare_private_share(path, source_reference)
        elif source_type == "github_figshare":
            raise FileNotFoundError(
                f"Checkpoint not found at {path}. "
                f"Please download manually from {source_reference} and place it at {path}."
            )
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at {path} and source_type={source_type!r} "
                f"has no auto-download handler. Download manually from {source_reference}."
            )
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise FileNotFoundError(
            f"Auto-download failed for {path} from {source_reference}: {exc}. "
            f"Please download manually."
        ) from exc


# --------------- download backends --------------- #

_HF_FILENAME_MAP = {
    # Maps HF repo_id → path-within-repo when that differs from the local
    # checkpoint_path's basename. Only needed when the HF repo stores the
    # weight under a subfolder (e.g. neurolm's "checkpoints/VQ.pt").
    "Weibang/NeuroLM": "checkpoints/VQ.pt",
}



# Where a Hugging Face token may live, most explicit first. This used to be
# a shell helper that is no longer shipped, which only helped if you sourced it
# from a launcher -- running `embed` directly on a gated model just failed with
# a 401. The search order is that script's, unchanged.
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
    import os

    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()

    repo_root = workspace_root()
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


def _download_huggingface(local_path: Path, repo_id: str) -> Path:
    """Download a checkpoint from HuggingFace Hub.

    *repo_id* may be written either way a spec author would naturally write
    it -- "org/repo" or the URL you get from the browser address bar. The
    Hub API only accepts the former, and a full URL otherwise fails at
    download time rather than at registration, so normalise it here.
    """
    for prefix in ("https://huggingface.co/", "http://huggingface.co/",
                   "huggingface.co/"):
        if repo_id.startswith(prefix):
            repo_id = repo_id[len(prefix):]
            break
    repo_id = repo_id.strip("/")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise FileNotFoundError(
            f"Checkpoint not found at {local_path}. Install `huggingface_hub` "
            f"to auto-download, or manually download from https://huggingface.co/{repo_id}."
        )

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
        token=resolve_hf_token(),
    )
    logger.info("Downloaded %s from HuggingFace %s", repo_filename, repo_id)
    return Path(downloaded)


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
            f"Checkpoint not found at {local_path}. No known internal path for "
            f"GitHub repo {repo_slug}. Download manually from {repo_url}."
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


def _download_figshare_private_share(local_path: Path, share_url: str) -> Path:
    """Download a file from a Figshare private-share URL.

    ``share_url`` is of the form ``https://figshare.com/s/<token>``. The share
    may wrap one or more articles; we pull the article listing via the
    Figshare v2 API, download each article as a zip, and extract the file
    matching ``local_path.name``.
    """
    import urllib.request
    import urllib.parse
    import json
    import zipfile

    token = share_url.rstrip("/").rsplit("/", 1)[-1]
    api_url = f"https://api.figshare.com/v2/articles?private_link={token}"
    logger.info("Resolving Figshare private share token=%s...", token)
    with urllib.request.urlopen(api_url, timeout=60) as resp:
        articles = json.load(resp)

    target_name = local_path.name
    with tempfile.TemporaryDirectory(prefix="eegbench_dl_fs_") as tmpdir:
        for article in articles:
            art_id = article.get("id")
            if art_id is None:
                continue
            zip_url = (
                f"https://figshare.com/ndownloader/articles/{art_id}?private_link={token}"
            )
            zip_path = os.path.join(tmpdir, f"{art_id}.zip")
            logger.info("Downloading Figshare article %s (%s)...", art_id, zip_url)
            urllib.request.urlretrieve(zip_url, zip_path)
            if not zipfile.is_zipfile(zip_path):
                continue
            with zipfile.ZipFile(zip_path, "r") as zf:
                matches = [n for n in zf.namelist() if n.endswith(target_name)]
                if not matches:
                    continue
                extracted = zf.extract(matches[0], tmpdir)
                shutil.copy2(extracted, str(local_path))
                logger.info("Downloaded %s from Figshare article %s", target_name, art_id)
                return local_path

    raise FileNotFoundError(
        f"Figshare share {share_url} did not contain a file ending with {target_name!r}. "
        f"Check the share contents and adjust spec.checkpoint_path."
    )
