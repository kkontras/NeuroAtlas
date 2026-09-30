"""SZ1 preprocessor — write a per-recording memmap cache for fast iteration.

Mirrors the EpilepsiAE pattern (``epilepsiae_cached.py``) so the runtime
adapter is uniform across datasets:

    cache_root/<subject>/<recording>/
        signals.npy        # float16 (C=19, T_total) at TARGET_FS=256 Hz
        labels.npy         # uint8 per-sample seizure label
        meta.json          # subject_id, rec_id, fs, channels, recording stats
        .schema_v1         # atomic completion marker (written last)

A single cohort-level ``windows.npy`` index is produced after all shards
finish. The ``finetune_sz1`` entrypoint slices windows from there.

Condor-shardable per subject; resumable (recordings with the schema marker
are skipped). Patient folds come from ``src/neuroatlas/configs/folds/sz1.json`` and are
not (re-)derived here.

Usage:
    # one shard
    python -m neuroatlas.extensions.datasets.preprocessors.preprocess_sz1 \
        --cache-root /anonorg/.../caches/sz1 --shard-index 0 --num-shards 12

    # finalize the cohort window index after all shards complete
    python -m neuroatlas.extensions.datasets.preprocessors.preprocess_sz1 \
        --cache-root /anonorg/.../caches/sz1 --build-windows-index \
        --window-s 10 --stride-s 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from neuroatlas.benchmarking_helpers.registry.fold_manifest import load_folds
from neuroatlas.extensions.datasets._recording_stats import compute_recording_stats
from neuroatlas.extensions.datasets.epilepsy.sz1_preprocessor import (
    CANONICAL_19,
    SZ1_DATA_ROOT,
    TARGET_FS,
    _parse_sz1_tsv,
    _read_sz1_edf_to_unipolar19,
    discover_recordings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocess_sz1")

CACHE_SCHEMA_TAG = "v1"
SIGNALS_DTYPE = np.float16
LABELS_DTYPE = np.uint8
SCHEMA_MARKER = ".schema_v1"
SIGNALS_FILE = "signals.npy"
LABELS_FILE = "labels.npy"
META_FILE = "meta.json"
WINDOWS_INDEX = "windows.npy"
COHORT_META = "cohort_meta.json"


def _rec_id(edf_path: Path, root: Path) -> str:
    try:
        return str(edf_path.relative_to(root)).replace("/", "__").replace(".edf", "")
    except ValueError:
        return edf_path.stem


def _shard_subjects(subjects: List[str], num_shards: int, shard_index: int) -> List[str]:
    return [s for i, s in enumerate(sorted(subjects)) if i % num_shards == shard_index]


def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp, arr, allow_pickle=False)
    # numpy auto-appends .npy to .save, but only if missing; we passed .npy.tmp
    # so it stays as-is. Verify before replace:
    if not tmp.exists():
        # numpy added a .npy suffix — locate and rename.
        candidate = tmp.with_suffix(tmp.suffix + ".npy")
        if candidate.exists():
            os.replace(candidate, tmp)
        else:
            raise RuntimeError(f"np.save produced no file at {tmp} or {candidate}")
    os.replace(tmp, path)


def _preprocess_recording(
    edf_path: Path,
    tsv_path: Path,
    subject_id: str,
    rec_id: str,
    cache_root: Path,
) -> Dict[str, Any]:
    rec_dir = cache_root / subject_id / rec_id
    if (rec_dir / SCHEMA_MARKER).exists():
        return {"status": "skipped", "rec_id": rec_id}

    signals, fs, found_chs = _read_sz1_edf_to_unipolar19(edf_path)
    n_samples = signals.shape[1]

    seizures = _parse_sz1_tsv(tsv_path) if tsv_path.exists() else []
    labels = np.zeros(n_samples, dtype=LABELS_DTYPE)
    for start_s, end_s in seizures:
        s = max(0, int(round(start_s * TARGET_FS)))
        e = min(n_samples, int(round(end_s * TARGET_FS)))
        labels[s:e] = 1

    stats = compute_recording_stats(signals)

    rec_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_npy(rec_dir / SIGNALS_FILE, signals.astype(SIGNALS_DTYPE, copy=False))
    _atomic_write_npy(rec_dir / LABELS_FILE, labels)

    meta = {
        "schema_version": CACHE_SCHEMA_TAG,
        "dataset": "sz1",
        "subject_id": subject_id,
        "rec_id": rec_id,
        "edf_path": str(edf_path),
        "tsv_path": str(tsv_path),
        "native_fs": int(fs),
        "target_fs": int(TARGET_FS),
        "n_samples": int(n_samples),
        "duration_s": float(n_samples / TARGET_FS),
        "channels": list(CANONICAL_19),
        "n_channels": len(CANONICAL_19),
        "n_found_channels": len(found_chs),
        "n_missing_channels": len(CANONICAL_19) - len(found_chs),
        "unit": "uV",
        "n_seizure_samples": int(labels.sum()),
        "n_seizure_intervals": len(seizures),
        "recording_mean": stats["recording_mean"].astype(np.float32).tolist(),
        "recording_std": stats["recording_std"].astype(np.float32).tolist(),
        "recording_q95": stats["recording_q95"].astype(np.float32).tolist(),
    }
    (rec_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    (rec_dir / SCHEMA_MARKER).write_text(CACHE_SCHEMA_TAG)

    return {
        "status": "wrote",
        "rec_id": rec_id,
        "subject_id": subject_id,
        "n_samples": n_samples,
        "n_seizure_samples": int(labels.sum()),
    }


def _build_windows_index(cache_root: Path, window_s: float, stride_s: float) -> Path:
    """Walk the completed cache and emit one global ``windows.npy`` index.

    Columns: rec_idx (int32), t_start (int64), label (int8), subject_idx (int32).
    Subject index is into ``cohort_meta.json['subjects']`` (sorted).
    """
    window_samples = int(round(window_s * TARGET_FS))
    stride_samples = int(round(stride_s * TARGET_FS))

    recordings: List[Dict[str, Any]] = []
    for subj_dir in sorted(cache_root.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith("."):
            continue
        for rec_dir in sorted(subj_dir.iterdir()):
            if not (rec_dir / SCHEMA_MARKER).exists():
                continue
            meta = json.loads((rec_dir / META_FILE).read_text())
            recordings.append({
                "subject_id": meta["subject_id"],
                "rec_id": meta["rec_id"],
                "n_samples": int(meta["n_samples"]),
                "rel_dir": str(rec_dir.relative_to(cache_root)),
            })

    subjects = sorted({r["subject_id"] for r in recordings})
    subject_to_idx = {s: i for i, s in enumerate(subjects)}

    rec_idx_list, t_start_list, label_list, subj_idx_list = [], [], [], []
    n_windows_per_rec: List[int] = []
    for rec_i, rec in enumerate(recordings):
        labels_mm = np.load(cache_root / rec["rel_dir"] / LABELS_FILE, mmap_mode="r")
        n = rec["n_samples"]
        n_w = 0
        pos = 0
        while pos + window_samples <= n:
            window_lbl = labels_mm[pos:pos + window_samples]
            label = int(window_lbl.any())
            rec_idx_list.append(rec_i)
            t_start_list.append(pos)
            label_list.append(label)
            subj_idx_list.append(subject_to_idx[rec["subject_id"]])
            pos += stride_samples
            n_w += 1
        n_windows_per_rec.append(n_w)

    if not rec_idx_list:
        raise RuntimeError(f"No windows produced from cache at {cache_root}")

    dtype = np.dtype([
        ("rec_idx", np.int32),
        ("t_start", np.int64),
        ("label", np.int8),
        ("subj_idx", np.int32),
    ])
    arr = np.empty(len(rec_idx_list), dtype=dtype)
    arr["rec_idx"] = rec_idx_list
    arr["t_start"] = t_start_list
    arr["label"] = label_list
    arr["subj_idx"] = subj_idx_list

    out_path = cache_root / WINDOWS_INDEX
    np.save(out_path, arr, allow_pickle=False)

    cohort_meta = {
        "schema_version": CACHE_SCHEMA_TAG,
        "dataset": "sz1",
        "target_fs": int(TARGET_FS),
        "channels": list(CANONICAL_19),
        "subjects": subjects,
        "n_subjects": len(subjects),
        "n_recordings": len(recordings),
        "n_windows": int(arr.shape[0]),
        "n_pos_windows": int(arr["label"].sum()),
        "window_s": float(window_s),
        "stride_s": float(stride_s),
        "window_samples": int(window_samples),
        "stride_samples": int(stride_samples),
        "recordings": recordings,
        "n_windows_per_rec": n_windows_per_rec,
    }
    (cache_root / COHORT_META).write_text(json.dumps(cohort_meta, indent=2))

    # Validate against folds manifest (raises on mismatch).
    folds = load_folds("sz1")
    folds.assert_covers(set(subjects))

    logger.info(
        "windows.npy written: %d windows, %d pos (%.4f%%), %d recordings, %d subjects",
        arr.shape[0], int(arr["label"].sum()),
        100 * float(arr["label"].mean()), len(recordings), len(subjects),
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=SZ1_DATA_ROOT,
                   help="SeizeIt1 raw EDF root (defaults to SZ1_DATA_ROOT).")
    p.add_argument("--cache-root", required=True,
                   help="Output cache directory (will be created).")
    p.add_argument("--shard-index", type=int, default=None,
                   help="Subject-shard index (0-based). With --num-shards.")
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--build-windows-index", action="store_true",
                   help="Skip preprocessing; build cohort-level windows.npy.")
    p.add_argument("--window-s", type=float, default=10.0)
    p.add_argument("--stride-s", type=float, default=10.0)
    args = p.parse_args()

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    if args.build_windows_index:
        out = _build_windows_index(cache_root, args.window_s, args.stride_s)
        logger.info("Wrote %s", out)
        return

    # Discover + filter recordings to this shard.
    recordings = discover_recordings(args.data_root)
    if not recordings:
        raise SystemExit(f"No SZ1 recordings under {args.data_root}")

    by_subject: Dict[str, List[Tuple[Path, Path, str]]] = defaultdict(list)
    for edf, tsv, subj in recordings:
        by_subject[subj].append((Path(edf), Path(tsv), subj))

    all_subjects = sorted(by_subject.keys())
    if args.shard_index is not None and args.num_shards is not None:
        if not 0 <= args.shard_index < args.num_shards:
            raise SystemExit(f"shard {args.shard_index} out of range [0,{args.num_shards})")
        my_subjects = _shard_subjects(all_subjects, args.num_shards, args.shard_index)
        logger.info("Shard %d/%d: %d subjects: %s",
                    args.shard_index, args.num_shards, len(my_subjects), my_subjects)
    else:
        my_subjects = all_subjects
        logger.info("No shard args — preprocessing all %d subjects.", len(my_subjects))

    n_total = sum(len(by_subject[s]) for s in my_subjects)
    n_done = n_skip = n_fail = 0
    for subj in my_subjects:
        for edf_path, tsv_path, subject_id in by_subject[subj]:
            rec_id = _rec_id(edf_path, Path(args.data_root))
            try:
                result = _preprocess_recording(
                    edf_path, tsv_path, subject_id, rec_id, cache_root,
                )
            except Exception as exc:
                n_fail += 1
                logger.exception("FAIL %s — %s", rec_id, exc)
                continue
            if result["status"] == "skipped":
                n_skip += 1
            else:
                n_done += 1
                logger.info(
                    "[%d/%d] %s/%s  n_samples=%d  n_seiz=%d",
                    n_done + n_skip, n_total, subject_id, rec_id,
                    result["n_samples"], result["n_seizure_samples"],
                )

    logger.info("Shard done: wrote=%d skipped=%d failed=%d (total=%d)",
                n_done, n_skip, n_fail, n_total)


if __name__ == "__main__":
    main()
