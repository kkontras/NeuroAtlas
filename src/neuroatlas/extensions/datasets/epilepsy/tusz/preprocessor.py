"""TUSZ cache builder, shard merger, split merger, and summary tools.

Two-level merge pipeline:

1. **Shard cache** (``build_cache``) — per-split HDF5 file, optionally
   sharded across Condor workers for parallelism.
2. **Shard merge** (``merge_shards``) — combine the ``_shardNNN`` files for
   one split into the canonical ``tusz_<ver>_<split>_<schema>.h5``.
3. **Split merge** (``merge_splits``) — combine ``train``/``dev``/``eval``
   files into a single ``_all_`` cache that also records ``split_labels``
   (needed for subject-disjoint k-fold splitting).

All caches follow the same schema: continuous 19-unipolar float32 signals
at 256 Hz with ``samplewise_label`` (binary) and ``samplewise_type``
(13-class), plus per-recording and per-event metadata datasets.
"""
from __future__ import annotations

import gc
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from .._common import apply_standard_filters
from .annotations import (
    SEIZURE_TYPE_NAMES,
    build_event_arrays,
    build_samplewise_labels,
    build_samplewise_types,
    parse_csv_bi,
    parse_csv_multiclass,
)
from .readers import (
    BIPOLAR_MONTAGE,
    CACHE_SCHEMA_TAG,
    TARGET_FS,
    TCP_MONTAGE_CHANNELS,
    UNIPOLAR_ELECTRODES,
    discover_recordings,
    load_tuh_eeg_epilepsy_metadata,
    read_edf_unipolar,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Split / chunking constants
# ---------------------------------------------------------------------------

SPLITS: Tuple[str, ...] = ("train", "dev", "eval")

# Chunk size for streaming I/O — tuned to keep NFS reads under a few GB
# per chunk and fit comfortably in RAM alongside the current recording.
_CHUNK_SAMPLES: int = 5_000_000


# ---------------------------------------------------------------------------
# Per-recording processing
# ---------------------------------------------------------------------------


def _process_one_recording(
    edf_path: str,
    epilepsy_meta: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Process a single EDF + annotations. Returns None on failure."""
    edf = Path(edf_path)
    stem = edf.stem  # e.g. aaaaabnn_s002_t004
    parent = edf.parent  # .../03_tcp_ar_a/
    montage_type = parent.name  # e.g. "03_tcp_ar_a"

    csv_bi_path = parent / f"{stem}.csv_bi"
    csv_path = parent / f"{stem}.csv"

    # Extract subject/session from path layout:
    #   .../edf/{split}/{subject_id}/{session_dir}/{montage_dir}/{stem}.edf
    parts = edf.parts
    try:
        split_idx = None
        for i, p in enumerate(parts):
            if p in SPLITS:
                split_idx = i
                break
        if split_idx is not None and split_idx + 1 < len(parts):
            subject_id = parts[split_idx + 1]
            session_id = parts[split_idx + 2] if split_idx + 2 < len(parts) else "unknown"
        else:
            subject_id = stem.split("_")[0]
            session_id = "unknown"
    except Exception:
        subject_id = stem.split("_")[0] if "_" in stem else "unknown"
        session_id = "unknown"

    recording_id = stem

    try:
        signals, native_fs, n_missing, physical_units = read_edf_unipolar(edf_path)
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

    events_bi = parse_csv_bi(str(csv_bi_path)) if csv_bi_path.exists() else []
    samplewise_label = build_samplewise_labels(n_samples, events_bi, fs=TARGET_FS)

    events_mc = parse_csv_multiclass(str(csv_path)) if csv_path.exists() else []
    samplewise_type = build_samplewise_types(n_samples, events_mc, fs=TARGET_FS)

    subj_meta = epilepsy_meta.get(subject_id, {})
    age = subj_meta.get("age", -1)
    sex = subj_meta.get("sex", "U")
    diagnosis = subj_meta.get("diagnosis", "unknown")

    duration_s = n_samples / TARGET_FS

    return {
        "signals": signals,
        "samplewise_label": samplewise_label,
        "samplewise_type": samplewise_type,
        "recording_id": recording_id,
        "subject_id": subject_id,
        "session_id": session_id,
        "montage_type": montage_type,
        "duration_s": duration_s,
        "native_fs": native_fs,
        "n_missing_electrodes": n_missing,
        "physical_units": physical_units,
        "age": age,
        "sex": sex,
        "epilepsy_diagnosis": diagnosis,
        "events_multiclass": events_mc,
    }


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------


def build_cache(
    raw_root: str,
    cache_root: str,
    split: str,
    corpus_version: str = "v2.0.3",
    epilepsy_corpus_root: Optional[str] = None,
    workers: int = 1,
    max_recordings: Optional[int] = None,
    shard_index: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> str:
    """Build a continuous HDF5 cache for one split (or one shard of it).

    Args:
        raw_root: Root of TUSZ EDF data, containing ``{split}/`` subdirectories.
        cache_root: Where to write the HDF5 file.
        split: One of ``train``, ``dev``, ``eval``.
        corpus_version: TUSZ corpus version string.
        epilepsy_corpus_root: Path to ``tuh_eeg_epilepsy`` for metadata joins.
        workers: Number of parallel EDF-reading workers (per-recording level).
        max_recordings: Cap for debugging (process at most this many EDFs).
        shard_index: If sharding, 0-based index of this shard.
        num_shards: If sharding, total number of shards.

    Returns:
        Path to the written HDF5 file.
    """
    os.makedirs(cache_root, exist_ok=True)

    edfs = discover_recordings(raw_root, split)
    if max_recordings is not None:
        edfs = edfs[:max_recordings]

    if shard_index is not None and num_shards is not None:
        edfs = [e for i, e in enumerate(edfs) if i % num_shards == shard_index]
        logger.info("Shard %d/%d: %d EDFs", shard_index, num_shards, len(edfs))

    if shard_index is not None and num_shards is not None:
        fname = f"tusz_{corpus_version}_{split}_{CACHE_SCHEMA_TAG}_shard{shard_index:03d}.h5"
    else:
        fname = f"tusz_{corpus_version}_{split}_{CACHE_SCHEMA_TAG}.h5"
    out_path = os.path.join(cache_root, fname)

    epilepsy_meta: Dict[str, Dict[str, Any]] = {}
    if epilepsy_corpus_root:
        epilepsy_meta = load_tuh_eeg_epilepsy_metadata(epilepsy_corpus_root)
        logger.info("Loaded epilepsy metadata for %d subjects", len(epilepsy_meta))

    results: List[Dict[str, Any]] = []
    n_failed = 0

    def _process(edf_path: str) -> Optional[Dict[str, Any]]:
        return _process_one_recording(edf_path, epilepsy_meta)

    if workers <= 1:
        for edf_path in edfs:
            result = _process(edf_path)
            if result is not None:
                results.append(result)
            else:
                n_failed += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, p): p for p in edfs}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    else:
                        n_failed += 1
                except Exception as exc:
                    logger.warning("Worker failed for %s: %s", futures[future], exc)
                    n_failed += 1

    # Deterministic order for downstream fold assignment
    results.sort(key=lambda r: r["recording_id"])

    n_rec = len(results)
    if n_rec == 0:
        logger.error("No recordings processed successfully for split=%s", split)
        raise RuntimeError(f"No recordings for {split}")

    total_samples = sum(r["signals"].shape[0] for r in results)
    logger.info(
        "Writing %d recordings, %d total samples (%.1f hours) to %s",
        n_rec, total_samples, total_samples / TARGET_FS / 3600, out_path,
    )

    with h5py.File(out_path, "w") as h5:
        ds_signals = h5.create_dataset(
            "signals",
            shape=(total_samples, len(UNIPOLAR_ELECTRODES)),
            dtype=np.float32,
            chunks=(min(_CHUNK_SAMPLES, total_samples), len(UNIPOLAR_ELECTRODES)),
        )
        ds_label = h5.create_dataset(
            "samplewise_label",
            shape=(total_samples,),
            dtype=np.uint8,
            chunks=(min(_CHUNK_SAMPLES, total_samples),),
        )
        ds_type = h5.create_dataset(
            "samplewise_type",
            shape=(total_samples,),
            dtype=np.uint8,
            chunks=(min(_CHUNK_SAMPLES, total_samples),),
        )

        recording_offsets = np.zeros(n_rec + 1, dtype=np.int64)
        recording_ids: List[str] = []
        subject_ids: List[str] = []
        session_ids: List[str] = []
        montage_types: List[str] = []
        durations_s = np.zeros(n_rec, dtype=np.float32)
        native_fs_arr = np.zeros(n_rec, dtype=np.float32)
        n_missing_arr = np.zeros(n_rec, dtype=np.int16)
        ages = np.zeros(n_rec, dtype=np.int16)
        sexes: List[str] = []
        epilepsy_diag: List[str] = []
        physical_units_list: List[List[str]] = []

        all_event_start: List[float] = []
        all_event_stop: List[float] = []
        all_event_type: List[int] = []
        all_event_channel: List[int] = []
        all_event_rec_idx: List[int] = []

        cursor = 0
        for rec_idx, rec in enumerate(results):
            n_s = rec["signals"].shape[0]
            recording_offsets[rec_idx] = cursor

            written = 0
            while written < n_s:
                chunk_end = min(written + _CHUNK_SAMPLES, n_s)
                ds_signals[cursor + written : cursor + chunk_end] = rec["signals"][written:chunk_end]
                ds_label[cursor + written : cursor + chunk_end] = rec["samplewise_label"][written:chunk_end]
                ds_type[cursor + written : cursor + chunk_end] = rec["samplewise_type"][written:chunk_end]
                written = chunk_end

            cursor += n_s

            recording_ids.append(rec["recording_id"])
            subject_ids.append(rec["subject_id"])
            session_ids.append(rec["session_id"])
            montage_types.append(rec["montage_type"])
            durations_s[rec_idx] = rec["duration_s"]
            native_fs_arr[rec_idx] = rec["native_fs"]
            n_missing_arr[rec_idx] = rec["n_missing_electrodes"]
            ages[rec_idx] = rec["age"]
            sexes.append(rec["sex"])
            epilepsy_diag.append(rec["epilepsy_diagnosis"])
            physical_units_list.append(rec["physical_units"])

            ev_arrays = build_event_arrays(
                rec["events_multiclass"], rec_idx, TCP_MONTAGE_CHANNELS,
            )
            all_event_start.extend(ev_arrays["event_start_s"])
            all_event_stop.extend(ev_arrays["event_stop_s"])
            all_event_type.extend(ev_arrays["event_type"])
            all_event_channel.extend(ev_arrays["event_channel_tcp"])
            all_event_rec_idx.extend(ev_arrays["event_recording_idx"])

            # Free memory as we stream
            del rec["signals"]
            del rec["samplewise_label"]
            del rec["samplewise_type"]

        recording_offsets[n_rec] = cursor

        h5.create_dataset("recording_offsets", data=recording_offsets)
        h5.create_dataset("recording_ids", data=np.array(recording_ids, dtype=h5py.string_dtype()))
        h5.create_dataset("subject_ids", data=np.array(subject_ids, dtype=h5py.string_dtype()))
        h5.create_dataset("session_ids", data=np.array(session_ids, dtype=h5py.string_dtype()))
        h5.create_dataset("montage_types", data=np.array(montage_types, dtype=h5py.string_dtype()))
        h5.create_dataset("durations_s", data=durations_s)
        h5.create_dataset("native_fs", data=native_fs_arr)
        h5.create_dataset("n_missing_electrodes", data=n_missing_arr)
        h5.create_dataset("ages", data=ages)
        h5.create_dataset("sexes", data=np.array(sexes, dtype=h5py.string_dtype()))
        h5.create_dataset("epilepsy_diagnosis", data=np.array(epilepsy_diag, dtype=h5py.string_dtype()))
        h5.create_dataset(
            "physical_units",
            data=np.asarray(physical_units_list, dtype=h5py.string_dtype()),
        )

        n_events = len(all_event_start)
        h5.create_dataset("event_start_s", data=np.array(all_event_start, dtype=np.float32))
        h5.create_dataset("event_stop_s", data=np.array(all_event_stop, dtype=np.float32))
        h5.create_dataset("event_type", data=np.array(all_event_type, dtype=np.uint8))
        h5.create_dataset("event_channel_tcp", data=np.array(all_event_channel, dtype=np.int8))
        h5.create_dataset("event_recording_idx", data=np.array(all_event_rec_idx, dtype=np.int32))

        unique_subjects = set(subject_ids)
        # Memory-bounded pos_fraction: for big files, sum in chunks.
        if total_samples < 50_000_000:
            pos_samples = int(np.sum(ds_label[:] > 0))
        else:
            pos_samples = 0
            for start in range(0, total_samples, _CHUNK_SAMPLES):
                end = min(start + _CHUNK_SAMPLES, total_samples)
                pos_samples += int(np.sum(ds_label[start:end] > 0))
        pos_fraction = pos_samples / max(total_samples, 1)

        h5.attrs["corpus"] = "TUSZ"
        h5.attrs["version"] = corpus_version
        h5.attrs["split"] = split
        h5.attrs["fs"] = TARGET_FS
        h5.attrs["schema"] = "continuous"
        h5.attrs["schema_tag"] = CACHE_SCHEMA_TAG
        h5.attrs["channels"] = list(UNIPOLAR_ELECTRODES)
        h5.attrs["bipolar_montage"] = list(BIPOLAR_MONTAGE)
        h5.attrs["tcp_montage"] = list(TCP_MONTAGE_CHANNELS)
        h5.attrs["seizure_type_names"] = list(SEIZURE_TYPE_NAMES)
        h5.attrs["n_subjects"] = len(unique_subjects)
        h5.attrs["n_recordings_ok"] = n_rec
        h5.attrs["n_recordings_failed"] = n_failed
        h5.attrs["pos_fraction"] = pos_fraction
        h5.attrs["total_samples"] = total_samples
        h5.attrs["total_missing_electrodes"] = int(n_missing_arr.sum())

        if shard_index is not None:
            h5.attrs["shard_index"] = shard_index
            h5.attrs["num_shards"] = num_shards

        h5.flush()

    logger.info("Wrote %s (%d recordings, %d events)", out_path, n_rec, n_events)
    gc.collect()
    return out_path


# ---------------------------------------------------------------------------
# Shard merging
# ---------------------------------------------------------------------------


def _copy_datasets_chunked(
    src: h5py.File,
    dst: h5py.File,
    sample_offset: int,
    rec_offset: int,
    event_offset: int,
) -> Tuple[int, int, int]:
    """Append one source HDF5 file's data into the destination at given offsets.

    Handles the three offset families in one pass:
      - Sample-level arrays (``signals``, ``samplewise_label``, ``samplewise_type``)
        copied in chunks to avoid OOM.
      - Per-recording arrays copied whole; ``recording_offsets`` is shifted by
        ``sample_offset``.
      - Per-event arrays copied whole; ``event_recording_idx`` is shifted by
        ``rec_offset``.

    Returns:
        ``(n_samples_copied, n_recordings_copied, n_events_copied)``.
    """
    n_samples = src["signals"].shape[0]
    n_rec = src["recording_ids"].shape[0]
    n_events = src["event_start_s"].shape[0]

    for key in ("signals", "samplewise_label", "samplewise_type"):
        written = 0
        while written < n_samples:
            chunk_end = min(written + _CHUNK_SAMPLES, n_samples)
            dst[key][sample_offset + written : sample_offset + chunk_end] = src[key][written:chunk_end]
            written = chunk_end

    # Copy the n_rec+1 recording offsets, shifted; the last slot is left for the
    # caller to finalise with the running sample cursor.
    src_offsets = src["recording_offsets"][:]
    dst["recording_offsets"][rec_offset : rec_offset + n_rec + 1] = src_offsets + sample_offset

    for key in ("recording_ids", "subject_ids", "session_ids", "montage_types",
                "sexes", "epilepsy_diagnosis"):
        dst[key][rec_offset : rec_offset + n_rec] = src[key][:]

    for key in ("durations_s", "native_fs", "n_missing_electrodes", "ages"):
        dst[key][rec_offset : rec_offset + n_rec] = src[key][:]

    if "physical_units" in src and "physical_units" in dst:
        dst["physical_units"][rec_offset : rec_offset + n_rec] = src["physical_units"][:]

    if n_events > 0:
        dst["event_start_s"][event_offset : event_offset + n_events] = src["event_start_s"][:]
        dst["event_stop_s"][event_offset : event_offset + n_events] = src["event_stop_s"][:]
        dst["event_type"][event_offset : event_offset + n_events] = src["event_type"][:]
        dst["event_channel_tcp"][event_offset : event_offset + n_events] = src["event_channel_tcp"][:]
        rec_idx_arr = src["event_recording_idx"][:] + rec_offset
        dst["event_recording_idx"][event_offset : event_offset + n_events] = rec_idx_arr

    return n_samples, n_rec, n_events


def merge_shards(
    cache_root: str,
    split: str,
    corpus_version: str = "v2.0.3",
    cleanup: bool = False,
) -> str:
    """Merge shard HDF5 files for one split into a single per-split file.

    Args:
        cache_root: Directory containing shard files.
        split: One of ``train``, ``dev``, ``eval``.
        corpus_version: TUSZ corpus version string.
        cleanup: If True, delete shard files after successful merge.

    Returns:
        Path to the merged HDF5 file.
    """
    import glob as glob_mod

    pattern = os.path.join(cache_root, f"tusz_{corpus_version}_{split}_{CACHE_SCHEMA_TAG}_shard*.h5")
    shard_paths = sorted(glob_mod.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matching {pattern}")

    logger.info("Merging %d shards for split=%s", len(shard_paths), split)

    # First pass: total sizes (fits in memory — just shape reads).
    total_samples = 0
    total_rec = 0
    total_events = 0
    for sp in shard_paths:
        with h5py.File(sp, "r") as sh:
            total_samples += sh["signals"].shape[0]
            total_rec += sh["recording_ids"].shape[0]
            total_events += sh["event_start_s"].shape[0]

    out_fname = f"tusz_{corpus_version}_{split}_{CACHE_SCHEMA_TAG}.h5"
    out_path = os.path.join(cache_root, out_fname)
    logger.info("Merged file: %s (%d recordings, %d samples)", out_path, total_rec, total_samples)

    with h5py.File(out_path, "w") as dst:
        dst.create_dataset("signals", shape=(total_samples, len(UNIPOLAR_ELECTRODES)),
                           dtype=np.float32, chunks=(min(_CHUNK_SAMPLES, total_samples), len(UNIPOLAR_ELECTRODES)))
        dst.create_dataset("samplewise_label", shape=(total_samples,), dtype=np.uint8,
                           chunks=(min(_CHUNK_SAMPLES, total_samples),))
        dst.create_dataset("samplewise_type", shape=(total_samples,), dtype=np.uint8,
                           chunks=(min(_CHUNK_SAMPLES, total_samples),))

        dst.create_dataset("recording_offsets", shape=(total_rec + 1,), dtype=np.int64)

        for key in ("recording_ids", "subject_ids", "session_ids", "montage_types",
                     "sexes", "epilepsy_diagnosis"):
            dst.create_dataset(key, shape=(total_rec,), dtype=h5py.string_dtype())

        dst.create_dataset("durations_s", shape=(total_rec,), dtype=np.float32)
        dst.create_dataset("native_fs", shape=(total_rec,), dtype=np.float32)
        dst.create_dataset("n_missing_electrodes", shape=(total_rec,), dtype=np.int16)
        dst.create_dataset("ages", shape=(total_rec,), dtype=np.int16)
        dst.create_dataset(
            "physical_units",
            shape=(total_rec, len(UNIPOLAR_ELECTRODES)),
            dtype=h5py.string_dtype(),
        )

        dst.create_dataset("event_start_s", shape=(total_events,), dtype=np.float32)
        dst.create_dataset("event_stop_s", shape=(total_events,), dtype=np.float32)
        dst.create_dataset("event_type", shape=(total_events,), dtype=np.uint8)
        dst.create_dataset("event_channel_tcp", shape=(total_events,), dtype=np.int8)
        dst.create_dataset("event_recording_idx", shape=(total_events,), dtype=np.int32)

        sample_offset = 0
        rec_offset = 0
        event_offset = 0
        all_subject_ids: List[str] = []

        for sp in shard_paths:
            logger.info("Merging shard %s", sp)
            with h5py.File(sp, "r") as src:
                ns, nr, ne = _copy_datasets_chunked(src, dst, sample_offset, rec_offset, event_offset)
                all_subject_ids.extend(str(s) for s in src["subject_ids"][:])
            sample_offset += ns
            rec_offset += nr
            event_offset += ne
            dst.flush()

        dst["recording_offsets"][total_rec] = total_samples

        pos_samples = 0
        for start in range(0, total_samples, _CHUNK_SAMPLES):
            end = min(start + _CHUNK_SAMPLES, total_samples)
            pos_samples += int(np.sum(dst["samplewise_label"][start:end] > 0))
        pos_fraction = pos_samples / max(total_samples, 1)

        unique_subjects = set(all_subject_ids)
        dst.attrs["corpus"] = "TUSZ"
        dst.attrs["version"] = corpus_version
        dst.attrs["split"] = split
        dst.attrs["fs"] = TARGET_FS
        dst.attrs["schema"] = "continuous"
        dst.attrs["schema_tag"] = CACHE_SCHEMA_TAG
        dst.attrs["channels"] = list(UNIPOLAR_ELECTRODES)
        dst.attrs["bipolar_montage"] = list(BIPOLAR_MONTAGE)
        dst.attrs["tcp_montage"] = list(TCP_MONTAGE_CHANNELS)
        dst.attrs["seizure_type_names"] = list(SEIZURE_TYPE_NAMES)
        dst.attrs["n_subjects"] = len(unique_subjects)
        dst.attrs["n_recordings_ok"] = total_rec
        dst.attrs["n_recordings_failed"] = 0
        dst.attrs["pos_fraction"] = pos_fraction
        dst.attrs["total_samples"] = total_samples
        dst.attrs["total_missing_electrodes"] = int(np.sum(dst["n_missing_electrodes"][:]))
        dst.attrs["merged_from_shards"] = len(shard_paths)
        dst.flush()

    logger.info("Merge complete: %s", out_path)

    if cleanup:
        for sp in shard_paths:
            os.remove(sp)
            logger.info("Removed shard %s", sp)

    gc.collect()
    return out_path


# ---------------------------------------------------------------------------
# Split merging (for k-fold)
# ---------------------------------------------------------------------------


def merge_splits(
    cache_root: str,
    corpus_version: str = "v2.0.3",
    splits: Sequence[str] = SPLITS,
) -> str:
    """Merge per-split HDF5 files into a single all-splits file (for k-fold).

    Adds a ``split_labels`` dataset recording which split each recording
    originally came from, so downstream code can still discriminate if needed.

    Returns:
        Path to the merged all-splits HDF5 file.
    """
    split_paths = []
    for sp in splits:
        fname = f"tusz_{corpus_version}_{sp}_{CACHE_SCHEMA_TAG}.h5"
        path = os.path.join(cache_root, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Split file not found: {path}")
        split_paths.append((sp, path))

    total_samples = 0
    total_rec = 0
    total_events = 0
    for sp_name, sp_path in split_paths:
        with h5py.File(sp_path, "r") as sh:
            total_samples += sh["signals"].shape[0]
            total_rec += sh["recording_ids"].shape[0]
            total_events += sh["event_start_s"].shape[0]

    out_fname = f"tusz_{corpus_version}_all_{CACHE_SCHEMA_TAG}.h5"
    out_path = os.path.join(cache_root, out_fname)
    logger.info("Merging splits %s into %s (%d recordings, %d samples)", splits, out_path, total_rec, total_samples)

    with h5py.File(out_path, "w") as dst:
        dst.create_dataset("signals", shape=(total_samples, len(UNIPOLAR_ELECTRODES)),
                           dtype=np.float32, chunks=(min(_CHUNK_SAMPLES, total_samples), len(UNIPOLAR_ELECTRODES)))
        dst.create_dataset("samplewise_label", shape=(total_samples,), dtype=np.uint8,
                           chunks=(min(_CHUNK_SAMPLES, total_samples),))
        dst.create_dataset("samplewise_type", shape=(total_samples,), dtype=np.uint8,
                           chunks=(min(_CHUNK_SAMPLES, total_samples),))
        dst.create_dataset("recording_offsets", shape=(total_rec + 1,), dtype=np.int64)

        for key in ("recording_ids", "subject_ids", "session_ids", "montage_types",
                     "sexes", "epilepsy_diagnosis"):
            dst.create_dataset(key, shape=(total_rec,), dtype=h5py.string_dtype())

        dst.create_dataset("durations_s", shape=(total_rec,), dtype=np.float32)
        dst.create_dataset("native_fs", shape=(total_rec,), dtype=np.float32)
        dst.create_dataset("n_missing_electrodes", shape=(total_rec,), dtype=np.int16)
        dst.create_dataset("ages", shape=(total_rec,), dtype=np.int16)
        dst.create_dataset(
            "physical_units",
            shape=(total_rec, len(UNIPOLAR_ELECTRODES)),
            dtype=h5py.string_dtype(),
        )
        dst.create_dataset("split_labels", shape=(total_rec,), dtype=h5py.string_dtype())

        dst.create_dataset("event_start_s", shape=(total_events,), dtype=np.float32)
        dst.create_dataset("event_stop_s", shape=(total_events,), dtype=np.float32)
        dst.create_dataset("event_type", shape=(total_events,), dtype=np.uint8)
        dst.create_dataset("event_channel_tcp", shape=(total_events,), dtype=np.int8)
        dst.create_dataset("event_recording_idx", shape=(total_events,), dtype=np.int32)

        sample_offset = 0
        rec_offset = 0
        event_offset = 0
        all_subject_ids: List[str] = []

        for sp_name, sp_path in split_paths:
            logger.info("Merging split %s from %s", sp_name, sp_path)
            with h5py.File(sp_path, "r") as src:
                n_rec_src = src["recording_ids"].shape[0]
                ns, nr, ne = _copy_datasets_chunked(src, dst, sample_offset, rec_offset, event_offset)
                dst["split_labels"][rec_offset : rec_offset + n_rec_src] = [sp_name] * n_rec_src
                all_subject_ids.extend(str(s) for s in src["subject_ids"][:])
            sample_offset += ns
            rec_offset += nr
            event_offset += ne
            dst.flush()

        dst["recording_offsets"][total_rec] = total_samples

        pos_samples = 0
        for start in range(0, total_samples, _CHUNK_SAMPLES):
            end = min(start + _CHUNK_SAMPLES, total_samples)
            pos_samples += int(np.sum(dst["samplewise_label"][start:end] > 0))
        pos_fraction = pos_samples / max(total_samples, 1)

        unique_subjects = set(all_subject_ids)
        dst.attrs["corpus"] = "TUSZ"
        dst.attrs["version"] = corpus_version
        dst.attrs["split"] = "all"
        dst.attrs["fs"] = TARGET_FS
        dst.attrs["schema"] = "continuous"
        dst.attrs["schema_tag"] = CACHE_SCHEMA_TAG
        dst.attrs["channels"] = list(UNIPOLAR_ELECTRODES)
        dst.attrs["bipolar_montage"] = list(BIPOLAR_MONTAGE)
        dst.attrs["tcp_montage"] = list(TCP_MONTAGE_CHANNELS)
        dst.attrs["seizure_type_names"] = list(SEIZURE_TYPE_NAMES)
        dst.attrs["n_subjects"] = len(unique_subjects)
        dst.attrs["pos_fraction"] = pos_fraction
        dst.attrs["total_samples"] = total_samples
        dst.attrs["merged_from_splits"] = list(splits)
        dst.flush()

    logger.info("Merge-splits complete: %s", out_path)
    gc.collect()
    return out_path


# ---------------------------------------------------------------------------
# Summary / validation
# ---------------------------------------------------------------------------


def summarize_cache(cache_path: str) -> Dict[str, Any]:
    """Print and return a summary dict for a TUSZ HDF5 cache file."""
    with h5py.File(cache_path, "r") as h5:
        info: Dict[str, Any] = {}
        info["path"] = cache_path
        for key in ("corpus", "version", "split", "fs", "schema", "schema_tag",
                     "n_subjects", "pos_fraction", "total_samples"):
            info[key] = h5.attrs.get(key)
        info["n_recordings"] = h5["recording_ids"].shape[0]
        info["n_events"] = h5["event_start_s"].shape[0]
        info["signal_shape"] = list(h5["signals"].shape)
        info["hours"] = h5["signals"].shape[0] / TARGET_FS / 3600
        if "split_labels" in h5:
            labels_arr = h5["split_labels"][:]
            unique, counts = np.unique(labels_arr, return_counts=True)
            info["split_distribution"] = {str(u): int(c) for u, c in zip(unique, counts)}
    for k, v in sorted(info.items()):
        logger.info("  %s: %s", k, v)
    return info
