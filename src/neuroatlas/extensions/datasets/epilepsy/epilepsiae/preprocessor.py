"""EPILEPSIAE HDF5 cache builder and shard merger.

Orchestrates discovery → layout → stream-write for the three EPILEPSIAE
surface variants (surf30, surfPA, surfCO).
"""
from __future__ import annotations

import logging
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from .annotations import (
    _find_patient_sql,
    load_origin_annotations,
    load_patient_metadata,
    load_seizure_annotations,
    seizures_for_recording,
)
from .labels import build_samplewise_labels
from .readers import (
    BlockMeta,
    PatientMeta,
    PATTERN_NAMES,
    SEIZURE_TYPE_NAMES,
    discover_block,
    load_block_signals,
    normalize_channel_name,
    read_head,
    select_canonical_channels,
)
from .._common import (
    BIPOLAR_NAMES,
    CACHE_SCHEMA_TAG,
    CANONICAL_19,
    CANONICAL_IDX,
    GAP_LABEL,
    TARGET_FS,
    apply_standard_filters,
    resample_to,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path("${EEG_DATA_ROOT}/epilepsiae")
ANNOTATION_DIR = DATA_ROOT / "Annotation"

VARIANTS = ("surf30", "surfPA", "surfCO")

GAP_THRESHOLD_S: float = 2.0  # gaps >= this are real disconnections

# Patients to exclude (pat_300 uses anonymous EL001-EL025 channel names)
EXCLUDED_PATIENTS = frozenset({"300"})


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class EpilepsiAEPreprocessor:
    """Orchestrates discovery, preprocessing, and HDF5 cache building."""

    def __init__(
        self,
        data_root: Path = DATA_ROOT,
        annotation_dir: Optional[Path] = None,
        target_fs: int = TARGET_FS,
    ):
        self.data_root = Path(data_root)
        self.annotation_dir = Path(annotation_dir) if annotation_dir else self.data_root / "Annotation"
        self.target_fs = target_fs
        self._annotations: Optional[List[Dict[str, Any]]] = None
        self._origin_map: Optional[Dict[str, str]] = None

    @property
    def annotations(self) -> List[Dict[str, Any]]:
        if self._annotations is None:
            self._annotations = load_seizure_annotations(
                self.annotation_dir / "Seizure_annotations.csv"
            )
        return self._annotations

    @property
    def origin_map(self) -> Dict[str, str]:
        if self._origin_map is None:
            origin_path = self.annotation_dir / "Annotations_origin.csv"
            if origin_path.exists():
                self._origin_map = load_origin_annotations(origin_path)
            else:
                self._origin_map = {}
        return self._origin_map

    def discover_patients(
        self,
        variants: Sequence[str] = VARIANTS,
    ) -> List[PatientMeta]:
        """Walk the filesystem and discover all patients across requested variants."""
        patients: List[PatientMeta] = []
        for variant in variants:
            variant_dir = self.data_root / variant
            if not variant_dir.is_dir():
                logger.warning("Variant directory not found: %s", variant_dir)
                continue
            for entry in sorted(variant_dir.iterdir()):
                if not entry.is_dir() or not entry.name.startswith("pat_"):
                    continue
                pat_id = entry.name[4:]  # strip "pat_"
                if pat_id in EXCLUDED_PATIENTS:
                    logger.info("Skipping excluded patient %s", pat_id)
                    continue

                patient = PatientMeta(
                    pat_id=pat_id, variant=variant, patient_dir=entry,
                )

                # Discover all blocks grouped by recording
                for head_path in sorted(entry.rglob("*.head")):
                    try:
                        block = discover_block(head_path)
                    except Exception as exc:
                        logger.warning("Failed to parse %s: %s", head_path, exc)
                        continue
                    patient.recordings.setdefault(block.rec_id, []).append(block)

                # Sort blocks within each recording by start_ts
                for rec_id in patient.recordings:
                    patient.recordings[rec_id].sort(key=lambda b: b.start_ts)

                if patient.recordings:
                    patients.append(patient)
                else:
                    logger.warning("No valid blocks for patient %s", pat_id)

        logger.info("Discovered %d patients across variants %s", len(patients), variants)
        return patients

    def _compute_recording_layout(
        self,
        blocks: List[BlockMeta],
    ) -> Tuple[List[int], List[Tuple[int, int]], int, np.ndarray]:
        """Pre-compute the sample layout for a recording without loading data.

        Returns:
            (block_cumulative_starts, gap_regions, total_samples, channel_mask)
        """
        block_cumulative_starts: List[int] = []
        gap_regions: List[Tuple[int, int]] = []
        total_samples = 0
        channel_mask_union = np.zeros(19, dtype=bool)

        for i, block in enumerate(blocks):
            # Check which canonical channels are present
            for raw_name in block.elec_names:
                canonical = normalize_channel_name(raw_name)
                if canonical in CANONICAL_IDX:
                    channel_mask_union[CANONICAL_IDX[canonical]] = True

            # Gap insertion before this block
            if i > 0:
                gap_s = (block.start_ts - blocks[i - 1].end_ts).total_seconds()
                if gap_s >= GAP_THRESHOLD_S:
                    gap_samples = int(round(gap_s * self.target_fs))
                    gap_samples = min(gap_samples, self.target_fs * 3600)
                    gap_start = total_samples
                    total_samples += gap_samples
                    gap_regions.append((gap_start, total_samples))

            block_cumulative_starts.append(total_samples)
            # Estimate resampled block length
            block_len = int(round(block.num_samples * self.target_fs / block.sample_freq))
            total_samples += block_len

        return block_cumulative_starts, gap_regions, total_samples, channel_mask_union

    def _process_and_write_block(
        self,
        block: BlockMeta,
        ds_signals: h5py.Dataset,
        write_offset: int,
    ) -> int:
        """Load, process, and write one block directly to HDF5. Returns samples written."""
        head = read_head(block.head_path)
        signals = load_block_signals(block.data_path, head)
        signals_19, _ = select_canonical_channels(signals, block.elec_names)
        del signals  # free raw memory immediately

        signals_19 = resample_to(signals_19, block.sample_freq, self.target_fs)

        if signals_19.shape[1] > 20:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                signals_19 = apply_standard_filters(signals_19, self.target_fs)

        n_samp = signals_19.shape[1]
        ds_signals[write_offset:write_offset + n_samp] = signals_19.T  # (T, 19)
        return n_samp

    def build_cache(
        self,
        patients: List[PatientMeta],
        output_path: Path,
        n_workers: int = 1,
        shard_index: Optional[int] = None,
        num_shards: Optional[int] = None,
    ) -> Path:
        """Build a continuous HDF5 cache, streaming blocks to disk.

        Processes one block at a time to avoid accumulating entire recordings
        in memory. Two-pass per recording: (1) compute layout from headers,
        (2) stream-write each processed block to HDF5.
        """
        if shard_index is not None and num_shards is not None:
            patients = patients[shard_index::num_shards]
            logger.info("Shard %d/%d: processing %d patients", shard_index, num_shards, len(patients))

        # --- Pass 1: discover recordings and compute total sizes ---
        rec_plans: List[Dict[str, Any]] = []
        all_patient_meta: Dict[str, Dict[str, Any]] = {}
        grand_total_samples = 0
        grand_total_events = 0

        for patient in patients:
            sql_path = _find_patient_sql(patient.pat_id, metadata_dir=self.annotation_dir / "Metadata")
            pmeta = {}
            if sql_path is not None:
                try:
                    pmeta = load_patient_metadata(sql_path)
                except Exception as exc:
                    logger.warning("Failed to parse metadata for patient %s: %s",
                                   patient.pat_id, exc)
            all_patient_meta[patient.pat_id] = pmeta

            for rec_id, blocks in patient.recordings.items():
                seizures = seizures_for_recording(self.annotations, rec_id, self.origin_map)
                cum_starts, gap_regions, total_samp, ch_mask = self._compute_recording_layout(blocks)

                # Pre-compute labels (small: just uint8 arrays)
                labels, types, event_dicts = build_samplewise_labels(
                    seizures, blocks, total_samp, cum_starts, self.target_fs,
                )
                for gs, ge in gap_regions:
                    labels[gs:ge] = GAP_LABEL
                    types[gs:ge] = GAP_LABEL

                rec_plans.append({
                    "rec_id": rec_id,
                    "pat_id": patient.pat_id,
                    "variant": patient.variant,
                    "blocks": blocks,
                    "cum_starts": cum_starts,
                    "gap_regions": gap_regions,
                    "total_samples": total_samp,
                    "channel_mask": ch_mask,
                    "native_fs": blocks[0].sample_freq,
                    "labels": labels,
                    "types": types,
                    "events": event_dicts,
                })
                grand_total_samples += total_samp
                grand_total_events += len(event_dicts)

        if not rec_plans:
            logger.warning("No recordings found, skipping cache write.")
            return output_path

        # Count total blocks across all recordings for progress tracking
        total_blocks = sum(len(plan["blocks"]) for plan in rec_plans)
        logger.info("Layout computed: %d recordings, %d blocks, %d total samples, %d events",
                     len(rec_plans), total_blocks, grand_total_samples, grand_total_events)

        # --- Pass 2: stream-write to HDF5 ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(".h5.tmp")
        progress_path = output_path.with_suffix(".progress")
        vlen_str = h5py.special_dtype(vlen=str)
        n_rec = len(rec_plans)

        with h5py.File(str(tmp_path), "w") as f:
            ds_signals = f.create_dataset("signals", shape=(grand_total_samples, 19), dtype="float32")
            ds_label = f.create_dataset("samplewise_label", shape=(grand_total_samples,), dtype="uint8")
            ds_type = f.create_dataset("samplewise_type", shape=(grand_total_samples,), dtype="uint8")

            offsets = np.zeros(n_rec + 1, dtype=np.int64)
            rec_ids, subject_ids, variants_list = [], [], []
            durations = np.zeros(n_rec, dtype=np.float32)
            native_fs_arr = np.zeros(n_rec, dtype=np.float32)
            n_missing_arr = np.zeros(n_rec, dtype=np.int16)
            ch_masks = np.zeros((n_rec, 19), dtype=bool)
            genders, hospitals, focus_locs = [], [], []
            ages = np.full(n_rec, -1, dtype=np.int16)
            onset_ages = np.full(n_rec, -1, dtype=np.int16)

            ev_start_l, ev_stop_l, ev_type_l, ev_pattern_l = [], [], [], []
            ev_class_l, ev_vig_l, ev_rec_idx_l, ev_elec_l = [], [], [], []

            sample_cursor = 0
            blocks_done = 0

            for rec_i, plan in enumerate(rec_plans):
                rec_start = sample_cursor
                offsets[rec_i] = rec_start
                blocks = plan["blocks"]
                cum_starts = plan["cum_starts"]
                gap_regions = plan["gap_regions"]

                # Write labels (already computed, small arrays)
                n_samp = plan["total_samples"]
                ds_label[rec_start:rec_start + n_samp] = plan["labels"]
                ds_type[rec_start:rec_start + n_samp] = plan["types"]
                del plan["labels"], plan["types"]  # free label memory

                # Write gap regions as zeros in signals
                for gs, ge in gap_regions:
                    ds_signals[rec_start + gs:rec_start + ge] = 0.0

                # Stream-write each block
                for b_i, block in enumerate(blocks):
                    block_write_offset = rec_start + cum_starts[b_i]
                    try:
                        self._process_and_write_block(
                            block, ds_signals, block_write_offset,
                        )
                    except Exception as exc:
                        logger.error("Failed block %s/%s: %s", plan["rec_id"], block.block_no, exc)
                        # Zero-fill this block region
                        est_len = int(round(block.num_samples * self.target_fs / block.sample_freq))
                        ds_signals[block_write_offset:block_write_offset + est_len] = 0.0

                    blocks_done += 1
                    if blocks_done % 10 == 0 or blocks_done == total_blocks:
                        pct = 100.0 * blocks_done / total_blocks
                        progress_path.write_text(
                            f"{blocks_done}/{total_blocks} blocks ({pct:.1f}%) | "
                            f"rec {rec_i+1}/{n_rec} | "
                            f"pat {plan['pat_id']} ({plan['variant']})\n"
                        )

                sample_cursor += n_samp

                # Recording metadata
                rec_ids.append(plan["rec_id"])
                subject_ids.append(plan["pat_id"])
                variants_list.append(plan["variant"])
                durations[rec_i] = n_samp / self.target_fs
                native_fs_arr[rec_i] = plan["native_fs"]
                n_missing_arr[rec_i] = int((~plan["channel_mask"]).sum())
                ch_masks[rec_i] = plan["channel_mask"]

                pmeta = all_patient_meta.get(plan["pat_id"], {})
                genders.append(pmeta.get("gender", ""))
                age_val = pmeta.get("age", -1)
                ages[rec_i] = age_val if age_val is not None else -1
                oa_val = pmeta.get("onset_age", -1)
                onset_ages[rec_i] = oa_val if oa_val is not None else -1
                hospitals.append(pmeta.get("hospital", ""))
                foci = pmeta.get("eeg_focus", [])
                focus_locs.append(";".join(
                    fo.get("localisation", "") for fo in foci if fo.get("localisation")
                ) if foci else "")

                for ev in plan["events"]:
                    ev_start_l.append(ev["start_s"])
                    ev_stop_l.append(ev["stop_s"])
                    ev_type_l.append(ev["type"])
                    ev_pattern_l.append(ev["pattern"])
                    ev_class_l.append(ev["classification"])
                    ev_vig_l.append(ev["vigilance"])
                    ev_rec_idx_l.append(rec_i)
                    ev_elec_l.append(ev["onset_electrode"])

                if (rec_i + 1) % 5 == 0 or rec_i == n_rec - 1:
                    logger.info("Written %d/%d recordings (%d/%d samples)",
                                rec_i + 1, n_rec, sample_cursor, grand_total_samples)

            offsets[n_rec] = sample_cursor

            # Write metadata datasets
            f.create_dataset("recording_offsets", data=offsets)
            f.create_dataset("recording_ids", data=np.array(rec_ids, dtype=object), dtype=vlen_str)
            f.create_dataset("subject_ids", data=np.array(subject_ids, dtype=object), dtype=vlen_str)
            f.create_dataset("variant", data=np.array(variants_list, dtype=object), dtype=vlen_str)
            f.create_dataset("durations_s", data=durations)
            f.create_dataset("native_fs", data=native_fs_arr)
            f.create_dataset("n_missing_channels", data=n_missing_arr)
            f.create_dataset("channel_mask", data=ch_masks)
            f.create_dataset("genders", data=np.array(genders, dtype=object), dtype=vlen_str)
            f.create_dataset("ages", data=ages)
            f.create_dataset("onset_ages", data=onset_ages)
            f.create_dataset("hospitals", data=np.array(hospitals, dtype=object), dtype=vlen_str)
            f.create_dataset("focus_localisation", data=np.array(focus_locs, dtype=object), dtype=vlen_str)

            n_ev = len(ev_start_l)
            f.create_dataset("event_start_s", data=np.array(ev_start_l, dtype=np.float32))
            f.create_dataset("event_stop_s", data=np.array(ev_stop_l, dtype=np.float32))
            f.create_dataset("event_type", data=np.array(ev_type_l, dtype=np.uint8))
            f.create_dataset("event_pattern", data=np.array(ev_pattern_l, dtype=np.uint8))
            f.create_dataset("event_classification", data=np.array(ev_class_l, dtype=object), dtype=vlen_str)
            f.create_dataset("event_vigilance", data=np.array(ev_vig_l, dtype=object), dtype=vlen_str)
            f.create_dataset("event_recording_idx", data=np.array(ev_rec_idx_l, dtype=np.int32))
            f.create_dataset("event_onset_electrode", data=np.array(ev_elec_l, dtype=object), dtype=vlen_str)

            # Root attributes
            unique_subjects = set(subject_ids)
            pos_samples = int(np.sum(ds_label[:] == 1))
            variant_counts: Dict[str, int] = {}
            for v in variants_list:
                variant_counts[v] = variant_counts.get(v, 0) + 1

            f.attrs["fs"] = self.target_fs
            f.attrs["schema"] = "continuous"
            f.attrs["schema_tag"] = CACHE_SCHEMA_TAG
            f.attrs["channels"] = list(CANONICAL_19)
            f.attrs["bipolar_montage"] = list(BIPOLAR_NAMES)
            f.attrs["seizure_type_names"] = list(SEIZURE_TYPE_NAMES)
            f.attrs["pattern_names"] = list(PATTERN_NAMES)
            f.attrs["corpus"] = "epilepsiae"
            f.attrs["n_subjects"] = len(unique_subjects)
            f.attrs["n_recordings"] = n_rec
            f.attrs["total_samples"] = grand_total_samples
            f.attrs["total_events"] = n_ev
            f.attrs["pos_fraction"] = pos_samples / max(grand_total_samples, 1)
            for vk, vc in variant_counts.items():
                f.attrs[f"variant_{vk}_count"] = vc

        os.replace(str(tmp_path), str(output_path))
        progress_path.write_text(f"DONE | {n_rec} recordings, {grand_total_samples} samples, {n_ev} events\n")
        logger.info("Wrote HDF5 cache: %s (%d recordings, %d samples, %d events)",
                     output_path, n_rec, grand_total_samples, n_ev)
        return output_path


# ---------------------------------------------------------------------------
# Shard merging
# ---------------------------------------------------------------------------


def _read_vlen(f: h5py.File, name: str) -> List[str]:
    """Read a variable-length string dataset as a list of Python strings."""
    ds = f[name][:]
    return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in ds]


def merge_shards(
    shard_dir: Path,
    output_path: Path,
    cleanup: bool = False,
) -> Path:
    """Merge per-shard HDF5 files into a single cache file.

    Shard files are expected at: shard_dir/shard_{K:03d}_of_{N:03d}.h5
    """
    shard_paths = sorted(shard_dir.glob("shard_*.h5"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {shard_dir}")

    logger.info("Merging %d shards from %s", len(shard_paths), shard_dir)

    # First pass: collect sizes
    shard_infos = []
    total_samples = 0
    total_events = 0
    total_recordings = 0

    for sp in shard_paths:
        with h5py.File(str(sp), "r") as f:
            n_samp = f["signals"].shape[0]
            n_rec = f["recording_ids"].shape[0]
            n_ev = f["event_start_s"].shape[0]
            shard_infos.append((sp, n_samp, n_rec, n_ev))
            total_samples += n_samp
            total_recordings += n_rec
            total_events += n_ev

    # Second pass: write merged file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".h5.tmp")
    vlen_str = h5py.special_dtype(vlen=str)

    with h5py.File(str(tmp_path), "w") as out:
        # Pre-allocate
        out.create_dataset("signals", shape=(total_samples, 19), dtype="float32")
        out.create_dataset("samplewise_label", shape=(total_samples,), dtype="uint8")
        out.create_dataset("samplewise_type", shape=(total_samples,), dtype="uint8")

        offsets = np.zeros(total_recordings + 1, dtype=np.int64)
        all_rec_ids: List[str] = []
        all_subject_ids: List[str] = []
        all_variants: List[str] = []
        all_durations: List[float] = []
        all_native_fs: List[float] = []
        all_n_missing: List[int] = []
        all_ch_masks: List[np.ndarray] = []
        all_genders: List[str] = []
        all_ages: List[int] = []
        all_onset_ages: List[int] = []
        all_hospitals: List[str] = []
        all_focus_locs: List[str] = []

        all_ev_start: List[np.ndarray] = []
        all_ev_stop: List[np.ndarray] = []
        all_ev_type: List[np.ndarray] = []
        all_ev_pattern: List[np.ndarray] = []
        all_ev_class: List[str] = []
        all_ev_vig: List[str] = []
        all_ev_rec_idx: List[np.ndarray] = []
        all_ev_electrode: List[str] = []

        sample_cursor = 0
        rec_cursor = 0

        for sp, n_samp, n_rec, n_ev in shard_infos:
            with h5py.File(str(sp), "r") as f:
                out["signals"][sample_cursor:sample_cursor + n_samp] = f["signals"][:]
                out["samplewise_label"][sample_cursor:sample_cursor + n_samp] = f["samplewise_label"][:]
                out["samplewise_type"][sample_cursor:sample_cursor + n_samp] = f["samplewise_type"][:]

                # Fixup recording offsets
                shard_offsets = f["recording_offsets"][:]
                for i in range(n_rec):
                    offsets[rec_cursor + i] = sample_cursor + shard_offsets[i]

                # Recording metadata
                all_rec_ids.extend(_read_vlen(f, "recording_ids"))
                all_subject_ids.extend(_read_vlen(f, "subject_ids"))
                all_variants.extend(_read_vlen(f, "variant"))
                all_durations.extend(f["durations_s"][:].tolist())
                all_native_fs.extend(f["native_fs"][:].tolist())
                all_n_missing.extend(f["n_missing_channels"][:].tolist())
                all_ch_masks.append(f["channel_mask"][:])

                all_genders.extend(_read_vlen(f, "genders"))
                all_ages.extend(f["ages"][:].tolist())
                all_onset_ages.extend(f["onset_ages"][:].tolist())
                all_hospitals.extend(_read_vlen(f, "hospitals"))
                all_focus_locs.extend(_read_vlen(f, "focus_localisation"))

                # Events with recording index fixup
                if n_ev > 0:
                    all_ev_start.append(f["event_start_s"][:])
                    all_ev_stop.append(f["event_stop_s"][:])
                    all_ev_type.append(f["event_type"][:])
                    all_ev_pattern.append(f["event_pattern"][:])
                    all_ev_class.extend(_read_vlen(f, "event_classification"))
                    all_ev_vig.extend(_read_vlen(f, "event_vigilance"))
                    ev_ridx = f["event_recording_idx"][:]
                    all_ev_rec_idx.append(ev_ridx + rec_cursor)
                    all_ev_electrode.extend(_read_vlen(f, "event_onset_electrode"))

            sample_cursor += n_samp
            rec_cursor += n_rec

        offsets[total_recordings] = sample_cursor

        # Write merged metadata
        out.create_dataset("recording_offsets", data=offsets)
        out.create_dataset("recording_ids", data=np.array(all_rec_ids, dtype=object), dtype=vlen_str)
        out.create_dataset("subject_ids", data=np.array(all_subject_ids, dtype=object), dtype=vlen_str)
        out.create_dataset("variant", data=np.array(all_variants, dtype=object), dtype=vlen_str)
        out.create_dataset("durations_s", data=np.array(all_durations, dtype=np.float32))
        out.create_dataset("native_fs", data=np.array(all_native_fs, dtype=np.float32))
        out.create_dataset("n_missing_channels", data=np.array(all_n_missing, dtype=np.int16))
        out.create_dataset("channel_mask", data=np.concatenate(all_ch_masks, axis=0) if all_ch_masks else np.zeros((0, 19), dtype=bool))

        out.create_dataset("genders", data=np.array(all_genders, dtype=object), dtype=vlen_str)
        out.create_dataset("ages", data=np.array(all_ages, dtype=np.int16))
        out.create_dataset("onset_ages", data=np.array(all_onset_ages, dtype=np.int16))
        out.create_dataset("hospitals", data=np.array(all_hospitals, dtype=object), dtype=vlen_str)
        out.create_dataset("focus_localisation", data=np.array(all_focus_locs, dtype=object), dtype=vlen_str)

        # Events
        if all_ev_start:
            out.create_dataset("event_start_s", data=np.concatenate(all_ev_start))
            out.create_dataset("event_stop_s", data=np.concatenate(all_ev_stop))
            out.create_dataset("event_type", data=np.concatenate(all_ev_type))
            out.create_dataset("event_pattern", data=np.concatenate(all_ev_pattern))
            out.create_dataset("event_classification", data=np.array(all_ev_class, dtype=object), dtype=vlen_str)
            out.create_dataset("event_vigilance", data=np.array(all_ev_vig, dtype=object), dtype=vlen_str)
            out.create_dataset("event_recording_idx", data=np.concatenate(all_ev_rec_idx))
            out.create_dataset("event_onset_electrode", data=np.array(all_ev_electrode, dtype=object), dtype=vlen_str)
        else:
            for name, dtype_ in [("event_start_s", "float32"), ("event_stop_s", "float32"),
                                  ("event_type", "uint8"), ("event_pattern", "uint8"),
                                  ("event_recording_idx", "int32")]:
                out.create_dataset(name, shape=(0,), dtype=dtype_)
            for name in ("event_classification", "event_vigilance", "event_onset_electrode"):
                out.create_dataset(name, data=np.array([], dtype=object), dtype=vlen_str)

        # Copy attrs from first shard as base, then update totals
        with h5py.File(str(shard_paths[0]), "r") as first:
            for key in first.attrs:
                out.attrs[key] = first.attrs[key]

        out.attrs["n_subjects"] = len(set(all_subject_ids))
        out.attrs["n_recordings"] = total_recordings
        out.attrs["total_samples"] = total_samples
        out.attrs["total_events"] = total_events
        pos_samples = int(np.sum(out["samplewise_label"][:] == 1))
        out.attrs["pos_fraction"] = pos_samples / max(total_samples, 1)

    os.replace(str(tmp_path), str(output_path))
    logger.info("Merged %d shards → %s (%d recordings, %d samples)",
                len(shard_paths), output_path, total_recordings, total_samples)

    if cleanup:
        shutil.rmtree(str(shard_dir))
        logger.info("Cleaned up shard directory: %s", shard_dir)

    return output_path
