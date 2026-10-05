"""TUH EEG Abnormal Corpus (TUAB) v3.0.1 -> continuous HDF5 cache.

TUAB is the binary normal/abnormal EEG sub-corpus of TUH EEG.  Layout
under ``raw_root``::

    edf/
      train/
        abnormal/<montage>/<subject>/<session>/*.edf
        normal/  <montage>/<subject>/<session>/*.edf
      eval/
        abnormal/<montage>/<subject>/<session>/*.edf
        normal/  <montage>/<subject>/<session>/*.edf

Labels are file-level binary (one label per recording — every sample of
that recording shares it).  The continuous HDF5 schema mirrors TUSZ
(``signals (N, 19) float32`` at 256 Hz, ``recording_offsets``,
per-recording metadata) so downstream windowing helpers can be reused.
The TUAB-specific addition is the per-recording ``is_abnormal`` array.

This module reuses ``read_edf_unipolar`` and the canonical 19-channel
10-20 unipolar layout from ``tusz.readers`` — TUAB EDFs use the same
channel naming conventions.
"""
from __future__ import annotations

import gc
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from ._common import apply_standard_filters
from .tusz.readers import (
    BIPOLAR_MONTAGE,
    TARGET_FS,
    UNIPOLAR_ELECTRODES,
    read_edf_unipolar,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: TUAB has only train and eval splits (no dev split, unlike TUSZ).
SPLITS: Tuple[str, ...] = ("train", "eval")

#: Cache schema tag — distinct from the TUSZ tag so split filenames don't
#: collide in a shared cache root.
CACHE_SCHEMA_TAG: str = "256hz_continuous_unipolar19_tuab"

#: I/O chunk size for streaming writes (samples per chunk).  Same value as
#: the TUSZ preprocessor uses.
_CHUNK_SAMPLES: int = 5_000_000


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_recordings(raw_root: str, split: str) -> List[Tuple[str, int, str, str, str]]:
    """Find all EDF files for one split, with their per-recording metadata.

    The TUAB layout encodes the binary label and the montage type as
    directory names.

    Args:
        raw_root: Path to the ``edf/`` directory of a TUAB release.
        split: ``train`` or ``eval``.

    Returns:
        Sorted list of tuples
        ``(edf_path, is_abnormal, montage_type, subject_id, session_id)``.
    """
    split_dir = Path(raw_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"TUAB split directory not found: {split_dir}")

    recordings: List[Tuple[str, int, str, str, str]] = []
    for label_dir, is_abnormal in (("abnormal", 1), ("normal", 0)):
        root = split_dir / label_dir
        if not root.exists():
            logger.warning("TUAB %s/%s directory not present, skipping", split, label_dir)
            continue
        for edf_path in sorted(root.rglob("*.edf")):
            # Layout: <root>/<montage>/<subject>/<session>/<file>.edf
            try:
                rel = edf_path.relative_to(root)
                parts = rel.parts
                montage_type = parts[0] if len(parts) >= 1 else "unknown"
                subject_id = parts[1] if len(parts) >= 2 else "unknown"
                session_id = parts[2] if len(parts) >= 3 else "unknown"
            except Exception:
                montage_type = "unknown"
                subject_id = "unknown"
                session_id = "unknown"
            recordings.append((str(edf_path), is_abnormal, montage_type, subject_id, session_id))

    logger.info(
        "Discovered %d TUAB recordings for split=%s (abnormal=%d, normal=%d)",
        len(recordings), split,
        sum(1 for r in recordings if r[1] == 1),
        sum(1 for r in recordings if r[1] == 0),
    )
    return recordings


# ---------------------------------------------------------------------------
# Per-recording processing
# ---------------------------------------------------------------------------


def _process_one_recording(
    edf_path: str,
    is_abnormal: int,
    montage_type: str,
    subject_id: str,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """Read and normalise a single TUAB EDF.  Returns None on failure."""
    try:
        signals, native_fs, n_missing, _physical_units = read_edf_unipolar(edf_path)
    except Exception as exc:
        logger.warning("Failed to read EDF %s: %s", edf_path, exc)
        return None

    # 0.5 Hz HP + 60 Hz notch (US mains — Temple University Hospital).
    # ``read_edf_unipolar`` returns time-first ``(T, 19)``.
    if signals.shape[0] > 20:
        signals = apply_standard_filters(
            signals, fs=TARGET_FS, notch_hz=60.0, axis=0,
        )

    n_samples = signals.shape[0]
    duration_s = n_samples / TARGET_FS
    recording_id = Path(edf_path).stem

    return {
        "signals": signals,
        "is_abnormal": int(is_abnormal),
        "recording_id": recording_id,
        "subject_id": subject_id,
        "session_id": session_id,
        "montage_type": montage_type,
        "duration_s": duration_s,
        "native_fs": native_fs,
        "n_missing_electrodes": n_missing,
    }


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------


def build_cache(
    raw_root: str,
    cache_root: str,
    split: str,
    corpus_version: str = "v3.0.1",
    workers: int = 1,
    max_recordings: Optional[int] = None,
    shard_index: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> str:
    """Build a continuous HDF5 cache for one TUAB split (or one shard of it).

    Args:
        raw_root: Path to the TUAB ``edf/`` directory.
        cache_root: Where to write the HDF5 file.
        split: ``train`` or ``eval``.
        corpus_version: TUAB corpus version string (default ``v3.0.1``).
        workers: Number of parallel EDF-reading workers.
        max_recordings: Cap for debugging.
        shard_index: 0-based shard index for parallel builds (optional).
        num_shards: Total number of shards (optional).

    Returns:
        Path to the written HDF5 file.
    """
    if split not in SPLITS:
        raise ValueError(f"TUAB split must be one of {SPLITS}, got {split!r}")

    os.makedirs(cache_root, exist_ok=True)

    recordings = discover_recordings(raw_root, split)
    if max_recordings is not None:
        recordings = recordings[:max_recordings]

    if shard_index is not None and num_shards is not None:
        recordings = [
            r for i, r in enumerate(recordings) if i % num_shards == shard_index
        ]
        logger.info("Shard %d/%d: %d EDFs", shard_index, num_shards, len(recordings))

    if shard_index is not None and num_shards is not None:
        fname = f"tuab_{corpus_version}_{split}_{CACHE_SCHEMA_TAG}_shard{shard_index:03d}.h5"
    else:
        fname = f"tuab_{corpus_version}_{split}_{CACHE_SCHEMA_TAG}.h5"
    out_path = os.path.join(cache_root, fname)

    n_failed = 0

    # Process recordings deterministically (sorted by edf path) so that the
    # output ordering is stable across re-runs and across worker counts.
    recordings_sorted = sorted(recordings, key=lambda r: r[0])

    # We stream recordings to disk as soon as each worker returns, so total
    # peak memory is bounded by ~workers × largest_recording rather than by
    # the entire corpus.  HDF5 datasets start empty and grow incrementally
    # via ``resize``.
    n_channels = len(UNIPOLAR_ELECTRODES)

    recording_ids: List[str] = []
    subject_ids: List[str] = []
    session_ids: List[str] = []
    montage_types: List[str] = []
    is_abnormal_list: List[int] = []
    durations_list: List[float] = []
    native_fs_list: List[float] = []
    n_missing_list: List[int] = []
    offsets_list: List[int] = [0]
    cursor = 0

    with h5py.File(out_path, "w") as h5:
        ds_signals = h5.create_dataset(
            "signals",
            shape=(0, n_channels),
            maxshape=(None, n_channels),
            dtype=np.float32,
            chunks=(min(_CHUNK_SAMPLES, 1_000_000), n_channels),
        )
        # samplewise_label broadcast from the file-level is_abnormal so
        # downstream task code that already reads samplewise_label keeps
        # working unchanged.
        ds_label = h5.create_dataset(
            "samplewise_label",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=(min(_CHUNK_SAMPLES, 1_000_000),),
        )

        def _append(rec: Dict[str, Any]) -> None:
            nonlocal cursor
            n_s = int(rec["signals"].shape[0])
            new_total = cursor + n_s
            ds_signals.resize((new_total, n_channels))
            ds_label.resize((new_total,))

            label_value = np.uint8(rec["is_abnormal"])
            written = 0
            while written < n_s:
                chunk_end = min(written + _CHUNK_SAMPLES, n_s)
                ds_signals[cursor + written : cursor + chunk_end] = rec["signals"][written:chunk_end]
                ds_label[cursor + written : cursor + chunk_end] = label_value
                written = chunk_end

            cursor = new_total
            offsets_list.append(cursor)
            recording_ids.append(rec["recording_id"])
            subject_ids.append(rec["subject_id"])
            session_ids.append(rec["session_id"])
            montage_types.append(rec["montage_type"])
            is_abnormal_list.append(int(rec["is_abnormal"]))
            durations_list.append(float(rec["duration_s"]))
            native_fs_list.append(float(rec["native_fs"]))
            n_missing_list.append(int(rec["n_missing_electrodes"]))

            # Free the (potentially large) signal buffer before the next
            # worker result arrives.
            rec["signals"] = None
            del rec

        if workers <= 1:
            for edf_path, is_abnormal, montage_type, subject_id, session_id in recordings_sorted:
                result = _process_one_recording(
                    edf_path, is_abnormal, montage_type, subject_id, session_id,
                )
                if result is None:
                    n_failed += 1
                    continue
                _append(result)
        else:
            # Buffer worker output to enforce sorted-by-recording-id write
            # order (so the on-disk layout is reproducible regardless of
            # worker scheduling).  ``pending`` maps recording_id -> result.
            target_order = [r[0] for r in recordings_sorted]  # edf paths
            id_to_pos = {Path(p).stem: i for i, p in enumerate(target_order)}
            next_pos = 0
            pending: Dict[int, Dict[str, Any]] = {}

            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _process_one_recording, ep, isa, mt, sid, sess,
                    ): ep
                    for ep, isa, mt, sid, sess in recordings_sorted
                }
                for future in as_completed(futures):
                    edf_path = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        logger.warning("Worker failed for %s: %s", edf_path, exc)
                        n_failed += 1
                        # Skip slot in the order so the rest can still flush.
                        skip_pos = id_to_pos[Path(edf_path).stem]
                        pending[skip_pos] = None  # sentinel
                    else:
                        if result is None:
                            n_failed += 1
                            skip_pos = id_to_pos[Path(edf_path).stem]
                            pending[skip_pos] = None
                        else:
                            pos = id_to_pos[Path(edf_path).stem]
                            pending[pos] = result

                    # Drain any contiguous prefix that's now ready.
                    while next_pos in pending:
                        rec = pending.pop(next_pos)
                        next_pos += 1
                        if rec is not None:
                            _append(rec)
                            # Periodic progress + flush to keep the file
                            # readable mid-run if the job dies.
                            if len(recording_ids) % 100 == 0:
                                h5.flush()
                                logger.info(
                                    "Wrote %d/%d recordings (cursor=%d samples, %.1f h)",
                                    len(recording_ids), len(recordings_sorted),
                                    cursor, cursor / TARGET_FS / 3600,
                                )

        n_rec = len(recording_ids)
        if n_rec == 0:
            raise RuntimeError(f"No TUAB recordings for split={split}")

        total_samples = cursor
        logger.info(
            "Wrote %d TUAB recordings, %d total samples (%.1f hours) to %s",
            n_rec, total_samples, total_samples / TARGET_FS / 3600, out_path,
        )

        recording_offsets = np.asarray(offsets_list, dtype=np.int64)
        is_abnormal_arr = np.asarray(is_abnormal_list, dtype=np.uint8)
        durations_s = np.asarray(durations_list, dtype=np.float32)
        native_fs_arr = np.asarray(native_fs_list, dtype=np.float32)
        n_missing_arr = np.asarray(n_missing_list, dtype=np.int16)

        h5.create_dataset("recording_offsets", data=recording_offsets)
        h5.create_dataset("recording_ids",
                          data=np.array(recording_ids, dtype=h5py.string_dtype()))
        h5.create_dataset("subject_ids",
                          data=np.array(subject_ids, dtype=h5py.string_dtype()))
        h5.create_dataset("session_ids",
                          data=np.array(session_ids, dtype=h5py.string_dtype()))
        h5.create_dataset("montage_types",
                          data=np.array(montage_types, dtype=h5py.string_dtype()))
        h5.create_dataset("is_abnormal", data=is_abnormal_arr)
        h5.create_dataset("durations_s", data=durations_s)
        h5.create_dataset("native_fs", data=native_fs_arr)
        h5.create_dataset("n_missing_electrodes", data=n_missing_arr)

        unique_subjects = set(subject_ids)
        n_abnormal = int(is_abnormal_arr.sum())
        pos_fraction = float(n_abnormal) / max(n_rec, 1)

        h5.attrs["corpus"] = "TUAB"
        h5.attrs["version"] = corpus_version
        h5.attrs["split"] = split
        h5.attrs["fs"] = TARGET_FS
        h5.attrs["schema"] = "continuous"
        h5.attrs["schema_tag"] = CACHE_SCHEMA_TAG
        h5.attrs["channels"] = list(UNIPOLAR_ELECTRODES)
        h5.attrs["bipolar_montage"] = list(BIPOLAR_MONTAGE)
        h5.attrs["n_subjects"] = len(unique_subjects)
        h5.attrs["n_recordings_ok"] = n_rec
        h5.attrs["n_recordings_failed"] = n_failed
        h5.attrs["n_abnormal"] = n_abnormal
        h5.attrs["n_normal"] = n_rec - n_abnormal
        h5.attrs["pos_fraction"] = pos_fraction
        h5.attrs["total_samples"] = total_samples
        h5.attrs["total_missing_electrodes"] = int(n_missing_arr.sum())

        if shard_index is not None:
            h5.attrs["shard_index"] = shard_index
            h5.attrs["num_shards"] = num_shards

        h5.flush()

    logger.info("Wrote %s (%d recordings, %d failed)", out_path, n_rec, n_failed)
    gc.collect()
    return out_path


# ---------------------------------------------------------------------------
# Cache summary
# ---------------------------------------------------------------------------


def summarize_cache(path: str) -> None:
    """Pretty-print a one-shot summary of a TUAB cache file."""
    with h5py.File(path, "r") as h5:
        n_rec = int(h5["recording_ids"].shape[0])
        total_samples = int(h5["signals"].shape[0])
        fs = float(h5.attrs.get("fs", TARGET_FS))
        n_abnormal = int(h5["is_abnormal"][:].sum())
        n_normal = n_rec - n_abnormal
        n_subjects = len(set(h5["subject_ids"][:].tolist()))
        n_missing_total = int(h5["n_missing_electrodes"][:].sum())
        montages = sorted(set(
            (s.decode() if isinstance(s, bytes) else str(s))
            for s in h5["montage_types"][:]
        ))

        print("=" * 72)
        print(f"  TUAB Cache Summary — {path}")
        print("=" * 72)
        print(f"  Corpus:            {h5.attrs.get('corpus', '?')} "
              f"{h5.attrs.get('version', '?')}")
        print(f"  Split:             {h5.attrs.get('split', '?')}")
        print(f"  Sampling rate:     {fs} Hz")
        print(f"  Recordings:        {n_rec} ({n_abnormal} abnormal / {n_normal} normal)")
        print(f"  Subjects:          {n_subjects}")
        print(f"  Total samples:     {total_samples} ({total_samples / fs / 3600:.2f} h)")
        print(f"  Missing electrodes (sum across recs): {n_missing_total}")
        print(f"  Montages present:  {montages}")
        print(f"  Schema tag:        {h5.attrs.get('schema_tag', '?')}")
        print("=" * 72)
