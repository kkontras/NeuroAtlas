from __future__ import annotations

import fcntl
import hashlib
import io
import json
import logging
import os
import pickle
import socket
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..registry.contracts import EmbeddingPayload
from neuroatlas._paths import artifacts_dir


# Where embeddings land when nothing else says. Every entrypoint resolves
# ``--cache-root`` > ``$EEG_CACHE_ROOT`` > this, so a subdirectory of one
# root is shared across probe types and an embedding extracted once is reused.
#
# Repo-relative, because a checkout has to work on a machine that is not the
# one this was written on. It used to be a hardcoded path on a specific
# scratch volume, which meant a fresh clone defaulted to a directory it had
# no access to -- and carried two people's usernames into the public tree.
#
# To keep using an existing cache, export EEG_CACHE_ROOT rather than editing
# this: the env var already takes precedence, so nothing re-extracts.
SHARED_EMBEDDING_CACHE_ROOT = (
    artifacts_dir("embedding_cache")
)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


class CacheCorruptError(RuntimeError):
    """Raised when a finalized cache fails the load-time integrity check."""


class CacheLockError(RuntimeError):
    """Raised when a cache directory is locked by another writer."""


def _locked_text(cache_dir: Path, info: str) -> str:
    """Another process holds the lock (the kernel frees it when that process
    ends, so only a live writer holds it): who, and what to do."""
    who = ""
    try:
        holder = json.loads(info) if info else {}
        if holder.get("pid"):
            who = f" (pid {holder['pid']}" + (f" on {holder['host']}" if holder.get("host")
                                              else "") + ")"
    except (ValueError, TypeError, AttributeError):
        who = ""
    try:
        from neuroatlas.cli import command_with

        again = command_with()
    except Exception:
        again = None
    return (f"the embeddings in {cache_dir} are being written by another neuroatlas "
            f"process{who}" + (f"\nfix: {again} (once that process has finished)"
                               if again else ""))


def _extract_again(cache_dir: Path) -> str:
    """The fix lines for embeddings that cannot be read: remove them, then
    run the command again (it extracts them anew)."""
    lines = f"\nfix: rm -r {cache_dir}"
    try:
        from neuroatlas.cli import command_with

        again = command_with()
    except Exception:
        again = None
    return lines + (f"\nfix: {again}" if again else "")


def build_cache_key(parts: Dict[str, object]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_exists(cache_dir: Path) -> bool:
    return (cache_dir / "features.npy").exists() and (cache_dir / "labels.npy").exists()


def save_embedding_payload(cache_dir: Path, payload: EmbeddingPayload, metadata: Dict[str, object]) -> Dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "features.npy", payload.features)
    np.save(cache_dir / "labels.npy", payload.labels)
    with open(cache_dir / "items.json", "w", encoding="utf-8") as handle:
        json.dump(payload.metadata, handle, indent=2, cls=_NumpyEncoder)
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, cls=_NumpyEncoder)
    return {
        "cache_dir": str(cache_dir),
        "features": str(cache_dir / "features.npy"),
        "labels": str(cache_dir / "labels.npy"),
        "items": str(cache_dir / "items.json"),
        "metadata": str(cache_dir / "metadata.json"),
    }


def load_embedding_payload(cache_dir: Path, mmap_mode: Optional[str] = None) -> EmbeddingPayload:
    features = np.load(cache_dir / "features.npy", mmap_mode=mmap_mode)
    labels = np.load(cache_dir / "labels.npy", mmap_mode=mmap_mode)
    with open(cache_dir / "items.json", "r", encoding="utf-8") as handle:
        items = json.load(handle)
    # R3 — fail loud on half-finalized or corrupted caches instead of surfacing
    # later as IndexError deep inside the probe.
    n_feat = int(features.shape[0])
    n_label = int(labels.shape[0])
    if n_feat != n_label or n_feat != len(items):
        raise CacheCorruptError(
            f"the embeddings in {cache_dir} do not line up: {n_feat:,} embeddings, "
            f"{n_label:,} labels and {len(items):,} window records"
            + _extract_again(cache_dir))
    if (cache_dir / "progress.json").exists():
        raise CacheCorruptError(
            f"the embeddings in {cache_dir} are incomplete: their writing was interrupted"
            + _extract_again(cache_dir))
    return EmbeddingPayload(features=features, labels=labels, metadata=items)


def merge_embedding_chunks(cache_dir: Path) -> bool:
    """Merge chunk sub-caches into the main *cache_dir*.

    Returns ``True`` when the merged cache is ready (either freshly merged or
    already present).  Returns ``False`` when not all chunks are available yet.
    """
    if cache_exists(cache_dir):
        return True
    chunks_dir = cache_dir / "_chunks"
    if not chunks_dir.is_dir():
        return False

    # Discover chunks: directory names like "0_of_4", "1_of_4", …
    n_chunks: Optional[int] = None
    found: Dict[int, Path] = {}
    for entry in sorted(chunks_dir.iterdir()):
        if not entry.is_dir():
            continue
        parts = entry.name.split("_of_")
        if len(parts) != 2:
            continue
        try:
            idx, total = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if n_chunks is None:
            n_chunks = total
        elif total != n_chunks:
            return False
        found[idx] = entry

    if n_chunks is None or set(found) != set(range(n_chunks)):
        return False

    for chunk_dir in found.values():
        if not cache_exists(chunk_dir) or (chunk_dir / "progress.json").exists():
            return False

    # All chunks present and finalized — merge under an exclusive lock.
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".merge.lock"
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.lockf(fd, fcntl.LOCK_EX)
        if cache_exists(cache_dir):
            return True

        all_features, all_labels, all_items = [], [], []
        for i in range(n_chunks):
            chunk = found[i]
            all_features.append(np.load(chunk / "features.npy"))
            all_labels.append(np.load(chunk / "labels.npy"))
            with open(chunk / "items.json", "r", encoding="utf-8") as fh:
                all_items.extend(json.load(fh))

        np.save(cache_dir / "features.npy", np.concatenate(all_features, axis=0))
        np.save(cache_dir / "labels.npy", np.concatenate(all_labels, axis=0))
        with open(cache_dir / "items.json", "w", encoding="utf-8") as fh:
            json.dump(all_items, fh, indent=2, cls=_NumpyEncoder)

        # Copy metadata from chunk 0.
        meta_src = found[0] / "metadata.json"
        if meta_src.exists():
            import shutil
            shutil.copy2(meta_src, cache_dir / "metadata.json")

        total_rows = sum(f.shape[0] for f in all_features)
        logging.getLogger(__name__).info("merged %d chunks into %d rows at %s",
                                         n_chunks, total_rows, cache_dir)
        return True
    finally:
        try:
            fcntl.lockf(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
        lock_path.unlink(missing_ok=True)


def read_progress(cache_dir: Path) -> Optional[Dict[str, object]]:
    """The progress record of an interrupted extraction, or None."""
    try:
        return json.loads((Path(cache_dir) / "progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def discard_partial(cache_dir: Path) -> None:
    """Remove an interrupted extraction's partial files (not a finished cache)."""
    cache_dir = Path(cache_dir)
    for name in ("features.npy.tmp", "labels.npy.tmp", "items.jsonl.tmp", "progress.json"):
        (cache_dir / name).unlink(missing_ok=True)


def _npy_header(dtype: np.dtype, shape: Tuple[int, ...]) -> bytes:
    """Build a NumPy v1.0 .npy header for the given dtype and shape."""
    header_dict = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (
        np.lib.format.dtype_to_descr(dtype),
        repr(shape),
    )
    header_bytes = header_dict.encode("latin1")
    preamble_len = 6 + 2 + 2
    padding = 64 - ((preamble_len + len(header_bytes) + 1) % 64)
    if padding == 64:
        padding = 0
    header_bytes = header_bytes + b" " * padding + b"\n"
    header_len = len(header_bytes)
    return b"\x93NUMPY\x01\x00" + header_len.to_bytes(2, "little") + header_bytes


class IncrementalEmbeddingWriter:
    """Streams embedding batches to .npy files on disk with resume support.

    Adds three safety mechanisms on top of the raw tensor persistence:
    - R2: advisory lock on ``cache_dir/.lock`` via ``fcntl.flock`` to prevent
      two processes from silently truncating each other's output. Fail-fast on
      contention; ``EEGBENCHMARKS_FORCE_LOCK=1`` or ``force_lock=True`` to
      override a stale lockfile.
    - R4: per-batch append of metadata to ``items.jsonl.tmp`` so resume no
      longer has to walk the DataLoader to rebuild per-sample metadata. The
      finalized cache format is unchanged (``items.json`` list).
    """

    def __init__(self, cache_dir: Path, num_workers: int = 0, force_lock: bool = False):
        self.cache_dir = cache_dir
        self._num_workers = num_workers
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._feat_tmp = cache_dir / "features.npy.tmp"
        self._label_tmp = cache_dir / "labels.npy.tmp"
        self._items_tmp = cache_dir / "items.jsonl.tmp"
        self._progress_path = cache_dir / "progress.json"
        self._lock_path = cache_dir / ".lock"
        self._lock_info_path = cache_dir / ".lock.info"
        self._lock_fd: Optional[int] = None
        self._items_file: Optional[io.TextIOBase] = None
        self._feat_file: Optional[io.BufferedWriter] = None
        self._label_file: Optional[io.BufferedWriter] = None
        self._items: List[Dict] = []
        self._items_restored = False
        self._n_rows = 0
        self._feat_dtype: Optional[np.dtype] = None
        self._feat_trailing: Optional[Tuple[int, ...]] = None
        self._label_dtype: Optional[np.dtype] = None
        self._header_len = 0
        # The order the loader delivers rows in, when that order depends on
        # how it was configured (e.g. "sharded:16"). Saved with the progress
        # so a resume replays the same order even if the worker count changed.
        self.order_tag: Optional[str] = None
        self.restored_order_tag: Optional[str] = None

        force_lock = force_lock or os.environ.get("EEGBENCHMARKS_FORCE_LOCK") == "1"
        self._acquire_lock(force=force_lock)

        if self._feat_tmp.exists() and self._label_tmp.exists() and self._progress_path.exists():
            self._try_resume()

    # ------------------------------------------------------------------ locking

    def _acquire_lock(self, force: bool) -> None:
        """Acquire an advisory exclusive lock on the cache directory.

        Fail-fast on contention unless ``force`` is set, in which case the
        stale lockfile is unlinked and re-acquired. The kernel automatically
        releases the lock when the holding process dies, so Condor kills
        never leave a lock behind on filesystems that support ``flock``.
        """
        self._lock_fd = os.open(str(self._lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            info = self._read_lock_info()
            if not force:
                os.close(self._lock_fd)
                self._lock_fd = None
                raise CacheLockError(_locked_text(self.cache_dir, info))
            # Force path: close this fd, remove both lock files, re-acquire.
            os.close(self._lock_fd)
            self._lock_info_path.unlink(missing_ok=True)
            self._lock_path.unlink(missing_ok=True)
            self._lock_fd = os.open(str(self._lock_path), os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        try:
            self._lock_info_path.write_text(
                json.dumps({
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "ts": time.time(),
                })
            )
        except OSError:
            # Info file is advisory; don't fail the run if we can't write it.
            pass

    def _read_lock_info(self) -> str:
        try:
            return self._lock_info_path.read_text().strip()
        except OSError:
            return ""

    def _release_lock(self) -> None:
        if self._lock_fd is None:
            return
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self._lock_fd)
        except OSError:
            pass
        self._lock_fd = None
        self._lock_path.unlink(missing_ok=True)
        self._lock_info_path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._close_files()
            self._release_lock()
        return False

    def __del__(self):
        # Best-effort lock release on GC. Kernel cleans up the flock regardless.
        try:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
        except Exception:
            pass

    def _close_files(self):
        for f in (self._feat_file, self._label_file, self._items_file):
            if f is not None:
                try:
                    f.close()
                except OSError:
                    pass
        self._feat_file = None
        self._label_file = None
        self._items_file = None

    # ------------------------------------------------------------------- state

    @property
    def n_rows_done(self) -> int:
        return self._n_rows

    @property
    def has_restored_items(self) -> bool:
        """True iff resume re-populated ``self._items`` from ``items.jsonl.tmp``.

        When True, the extraction loop can safely skip already-extracted rows
        without re-walking the DataLoader to rebuild per-sample metadata.
        """
        return self._items_restored

    def _discard_partial(self):
        """Delete .tmp + progress to force a fresh extraction."""
        self._feat_tmp.unlink(missing_ok=True)
        self._label_tmp.unlink(missing_ok=True)
        self._items_tmp.unlink(missing_ok=True)
        self._progress_path.unlink(missing_ok=True)

    def _try_resume(self):
        try:
            with open(self._progress_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            n_rows = int(info["n_rows"])
            feat_dtype = np.dtype(info["feat_dtype"])
            feat_trailing = tuple(info["feat_trailing"])
            label_dtype = np.dtype(info["label_dtype"])
            header_len = int(info["header_len"])
            saved_workers = int(info.get("num_workers", 0))
            saved_order = info.get("order")
        except (json.JSONDecodeError, KeyError, ValueError):
            return  # corrupt progress file → start fresh
        if n_rows == 0:
            return
        if saved_workers != self._num_workers:
            self._discard_partial()
            return

        # Truncate tensor tmp files to exactly n_rows (guards partial last write).
        feat_row_bytes = int(np.prod(feat_trailing)) * feat_dtype.itemsize
        expected_feat = header_len + n_rows * feat_row_bytes
        expected_label = header_len + n_rows * label_dtype.itemsize
        try:
            with open(self._feat_tmp, "r+b") as f:
                f.truncate(expected_feat)
            with open(self._label_tmp, "r+b") as f:
                f.truncate(expected_label)
        except OSError:
            self._discard_partial()
            return

        # R4 — restore per-sample metadata from items.jsonl.tmp. Keep exactly
        # n_rows lines so downstream consumers see a consistent view.
        items: List[Dict] = []
        items_restored = False
        if self._items_tmp.exists():
            try:
                with open(self._items_tmp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        items.append(json.loads(line))
                        if len(items) >= n_rows:
                            break
                if len(items) < n_rows:
                    # jsonl is short — cannot rebuild reliably, fall back to
                    # legacy "re-walk DataLoader" path.
                    items = []
                else:
                    # Truncate any surplus lines so the tmp file matches n_rows.
                    self._rewrite_items_tmp(items)
                    items_restored = True
            except (OSError, json.JSONDecodeError):
                items = []
                items_restored = False
                self._items_tmp.unlink(missing_ok=True)
        # If the tmp jsonl is missing/corrupt, keep items_restored=False.
        # The extraction loop will fall back to its legacy rebuild path.

        self._n_rows = n_rows
        self.restored_order_tag = saved_order
        self.order_tag = saved_order
        self._feat_dtype = feat_dtype
        self._feat_trailing = feat_trailing
        self._label_dtype = label_dtype
        self._header_len = header_len
        self._items = items
        self._items_restored = items_restored

        self._feat_file = open(self._feat_tmp, "ab")
        self._label_file = open(self._label_tmp, "ab")
        if items_restored:
            self._items_file = open(self._items_tmp, "a", encoding="utf-8")
        else:
            # Start fresh jsonl so subsequent appends land on a known-consistent
            # file; the extraction loop will repopulate _items by walking.
            self._items_file = open(self._items_tmp, "w", encoding="utf-8")

    def _rewrite_items_tmp(self, items: List[Dict]) -> None:
        tmp = self._items_tmp.with_suffix(self._items_tmp.suffix + ".rewrite")
        with open(tmp, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, cls=_NumpyEncoder))
                f.write("\n")
        os.replace(tmp, self._items_tmp)

    # -------------------------------------------------------------- write path

    def _open(self, feat_dtype: np.dtype, feat_trailing: Tuple[int, ...], label_dtype: np.dtype):
        self._feat_dtype = feat_dtype
        self._feat_trailing = feat_trailing
        self._label_dtype = label_dtype
        feat_header = _npy_header(feat_dtype, (0, *feat_trailing))
        label_header = _npy_header(label_dtype, (0,))
        self._header_len = len(feat_header)
        self._feat_file = open(self._feat_tmp, "wb")
        self._feat_file.write(feat_header)
        self._label_file = open(self._label_tmp, "wb")
        self._label_file.write(label_header)
        # Fresh items.jsonl.tmp to match the fresh tensor tmp files.
        self._items_file = open(self._items_tmp, "w", encoding="utf-8")
        self._items_restored = False

    def _save_progress(self):
        tmp = self._progress_path.with_suffix(self._progress_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "n_rows": self._n_rows,
                "feat_dtype": str(self._feat_dtype),
                "feat_trailing": list(self._feat_trailing),
                "label_dtype": str(self._label_dtype),
                "header_len": self._header_len,
                "num_workers": self._num_workers,
                "order": self.order_tag,
            }, f)
        os.replace(tmp, self._progress_path)

    def append(self, features: np.ndarray, labels: np.ndarray, metadata: List[Dict]):
        """Append one batch to the on-disk cache.

        Order matters for crash-safety: tensor writes → metadata write →
        progress update. If we die after tensor writes but before metadata,
        the next run truncates the tensor tmp files back to the last
        progress-recorded n_rows and the batch is re-extracted.
        """
        features = np.ascontiguousarray(features)
        labels = np.ascontiguousarray(labels.reshape(-1))
        if self._feat_file is None:
            self._open(features.dtype, features.shape[1:], labels.dtype)
        self._feat_file.write(features.tobytes())
        self._feat_file.flush()
        self._label_file.write(labels.tobytes())
        self._label_file.flush()
        # Persist per-sample metadata incrementally (R4).
        for item in metadata:
            self._items_file.write(json.dumps(item, cls=_NumpyEncoder))
            self._items_file.write("\n")
        self._items_file.flush()
        self._items.extend(metadata)
        self._n_rows += features.shape[0]
        self._save_progress()

    def extend_items_without_append(self, metadata: List[Dict]) -> None:
        """Append metadata to the in-memory item list without writing any tensor.

        Used by the legacy resume fast-path when ``items.jsonl.tmp`` was missing
        or corrupt and the loop re-walks the DataLoader to rebuild metadata for
        rows that already exist in the tensor tmp files. Writes to
        ``items.jsonl.tmp`` are intentionally not performed here — the on-disk
        jsonl is rebuilt from ``self._items`` at finalize time.
        """
        self._items.extend(metadata)

    def finalize(self, metadata: Dict[str, object]) -> Dict[str, str]:
        try:
            if self._feat_file is None:
                paths = save_embedding_payload(
                    self.cache_dir,
                    EmbeddingPayload(features=np.empty((0,)), labels=np.empty((0,)), metadata=[]),
                    metadata,
                )
                self._progress_path.unlink(missing_ok=True)
                self._items_tmp.unlink(missing_ok=True)
                return paths
            self._feat_file.close()
            self._label_file.close()
            if self._items_file is not None:
                self._items_file.close()
                self._items_file = None
            # Patch headers with the real row count.
            feat_header = _npy_header(self._feat_dtype, (self._n_rows, *self._feat_trailing))
            label_header = _npy_header(self._label_dtype, (self._n_rows,))
            with open(self._feat_tmp, "r+b") as f:
                f.seek(0)
                f.write(feat_header)
            with open(self._label_tmp, "r+b") as f:
                f.seek(0)
                f.write(label_header)
            final_feat = self.cache_dir / "features.npy"
            final_label = self.cache_dir / "labels.npy"
            self._feat_tmp.rename(final_feat)
            self._label_tmp.rename(final_label)
            with open(self.cache_dir / "items.json", "w", encoding="utf-8") as handle:
                json.dump(self._items, handle, indent=2, cls=_NumpyEncoder)
            with open(self.cache_dir / "metadata.json", "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, cls=_NumpyEncoder)
            # Clean up transient artifacts.
            self._progress_path.unlink(missing_ok=True)
            self._items_tmp.unlink(missing_ok=True)
            return {
                "cache_dir": str(self.cache_dir),
                "features": str(final_feat),
                "labels": str(final_label),
                "items": str(self.cache_dir / "items.json"),
                "metadata": str(self.cache_dir / "metadata.json"),
            }
        finally:
            self._release_lock()


def save_probe_payload(probe_dir: Path, estimator, metadata: Dict[str, object]) -> Dict[str, str]:
    probe_dir.mkdir(parents=True, exist_ok=True)
    with open(probe_dir / "probe.pkl", "wb") as handle:
        pickle.dump(estimator, handle)
    with open(probe_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, cls=_NumpyEncoder)
    return {
        "probe_dir": str(probe_dir),
        "probe": str(probe_dir / "probe.pkl"),
        "probe_metadata": str(probe_dir / "metadata.json"),
    }
