"""SeizeIt2 -> continuous-HDF5 preprocessing pipeline.

Reads raw EDF files and companion ``_a1.tsv`` annotation files of SeizeIT2
and builds a continuous HDF5 cache with samplewise binary seizure labels.
``build_h5_cache`` raises when the folder holds no recordings or a file cannot
be read.

Annotation file format (``*_a1.tsv``):
    - 5 header rows to skip
    - Columns: start, stop, class, type, hemisphere, lobe, extra
    - Rows with ``class == "seizure"`` are labelled 1 in the cache.

EDF channel labels use a ``"EEG <name>"`` prefix (e.g. ``"EEG F4"``, ``"EEG Fz"``).
``SZ2_CHANNEL_ALIAS`` maps those to the canonical CANONICAL_19 names.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from ._common import (
    BIPOLAR_NAMES,
    CANONICAL_19,
    CANONICAL_IDX,
    CANONICAL_SET,
    GAP_LABEL,
    TARGET_FS,
    bipolar_from_unipolar,
    resample_to,
    apply_standard_filters,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

SZ2_DATA_ROOT: str = "${EEG_DATA_ROOT}/sz2"

# ---------------------------------------------------------------------------
# Channel mapping
# ---------------------------------------------------------------------------

# Maps SeizeIt2 EDF channel labels (with "EEG " prefix) to canonical 10-20 names.
SZ2_CHANNEL_ALIAS: Dict[str, str] = {
    "EEG F4":  "F4",  "EEG F8":  "F8",  "EEG F7":  "F7",  "EEG F3":  "F3",
    "EEG Fz":  "FZ",  "EEG T4":  "T4",  "EEG T6":  "T6",  "EEG T3":  "T3",
    "EEG T5":  "T5",  "EEG O1":  "O1",  "EEG O2":  "O2",  "EEG P4":  "P4",
    "EEG P3":  "P3",  "EEG Pz":  "PZ",  "EEG Fp1": "FP1", "EEG Fp2": "FP2",
    "EEG Cz":  "CZ",  "EEG C3":  "C3",  "EEG C4":  "C4",
}

# Channel label substrings that indicate a non-EEG or artefact channel to skip.
SZ2_SKIP_KEYWORDS: Tuple[str, ...] = ("oor", "sd")

# ---------------------------------------------------------------------------
# Annotation parser
# ---------------------------------------------------------------------------


def _parse_sz2_tsv(tsv_path: Path) -> List[Tuple[float, float]]:
    """Parse a SeizeIt2 ``_a1.tsv`` annotation file.

    The file has 5 header rows followed by tab-separated rows with columns:
    ``start``, ``stop``, ``class``, ``type``, ``hemisphere``, ``lobe``, ``extra``.

    Returns:
        List of ``(start_s, stop_s)`` tuples for all rows where
        ``class == "seizure"``, regardless of seizure type.
    """
    seizures: List[Tuple[float, float]] = []
    try:
        import pandas as pd
        df = pd.read_csv(
            str(tsv_path),
            delimiter="\t",
            skiprows=5,
            names=["start", "stop", "class", "type", "hemisphere", "lobe", "extra"],
        )
        for _, row in df.iterrows():
            if str(row["class"]).strip().lower() == "seizure":
                try:
                    seizures.append((float(row["start"]), float(row["stop"])))
                except (ValueError, TypeError):
                    pass
    except Exception as exc:
        logger.warning("Could not parse annotation file %s: %s", tsv_path, exc)
    return seizures


# ---------------------------------------------------------------------------
# EDF reader
# ---------------------------------------------------------------------------


def _read_sz2_edf_to_unipolar19(
    edf_path: Path,
) -> Tuple[np.ndarray, int, List[str]]:
    """Read a SeizeIt2 EDF and map to canonical 19-channel unipolar layout.

    Uses ``edfio`` for fast reading.  Channel labels are normalised through
    ``SZ2_CHANNEL_ALIAS``.  Missing canonical channels are zero-filled.
    Standard filters (0.5 Hz HPF + 50 Hz notch) are applied after resampling
    to ``TARGET_FS`` (256 Hz).

    Returns:
        signals: ``(19, n_samples)`` float32 @ TARGET_FS.
        fs: effective sampling rate (= TARGET_FS after resampling).
        found_channels: list of canonical channel names that were present.
    """
    import edfio  # lazy import — only needed in the preprocessor

    edf = edfio.read_edf(str(edf_path))

    # Build a map: canonical_name -> (raw_data_array, sampling_frequency)
    ch_map: Dict[str, Tuple[np.ndarray, float]] = {}
    for sig in edf.signals:
        raw_label = str(sig.label).strip()

        # Skip channels matching skip keywords
        if any(kw in raw_label.lower() for kw in SZ2_SKIP_KEYWORDS):
            continue

        canonical = SZ2_CHANNEL_ALIAS.get(raw_label)
        if canonical is None:
            # Try stripping "EEG " prefix and upper-casing
            trimmed = raw_label.replace("EEG ", "").replace("eeg ", "").strip().upper()
            canonical = SZ2_CHANNEL_ALIAS.get(f"EEG {trimmed.capitalize()}")
        if canonical is not None and canonical in CANONICAL_SET:
            if canonical not in ch_map:
                ch_map[canonical] = (
                    np.array(sig.data, dtype=np.float32),
                    float(sig.sampling_frequency),
                )

    if not ch_map:
        raise ValueError(f"No canonical EEG channels found in {edf_path}")

    # All channels should share the same sampling rate
    fs_values = {fs for _, fs in ch_map.values()}
    src_fs = float(next(iter(fs_values)))
    if len(fs_values) > 1:
        logger.warning(
            "%s has mixed sampling rates %s; using %.0f Hz",
            edf_path, fs_values, src_fs,
        )

    # Reference length from the first channel
    ref_len = len(next(iter(ch_map.values()))[0])

    signals = np.zeros((19, ref_len), dtype=np.float32)
    found: List[str] = []
    for canonical_name, (data, _) in ch_map.items():
        cix = CANONICAL_IDX[canonical_name]
        length = min(len(data), ref_len)
        signals[cix, :length] = data[:length]
        found.append(canonical_name)

    # Resample to TARGET_FS if needed
    if abs(src_fs - TARGET_FS) >= 0.5:
        signals = resample_to(signals, src_fs, TARGET_FS)

    # Apply standard filters (0.5 Hz HPF + 50 Hz notch)
    signals = apply_standard_filters(signals, fs=float(TARGET_FS))

    return signals, TARGET_FS, found


# ---------------------------------------------------------------------------
# Recording discovery
# ---------------------------------------------------------------------------


def data_folder(root: str | Path) -> Path:
    """The folder holding the SeizeIt2 subject folders: ``root`` itself, or the
    ``Anonymous_Hospital_Adult`` folder the dataset is delivered in, under it."""
    from neuroatlas.extensions.datasets._layout import descend

    return descend(root, ["Anonymous_Hospital_Adult", "*/Anonymous_Hospital_Adult"], "*/*.edf")


def discover_recordings(root: str | Path) -> List[Tuple[Path, Path, str]]:
    """Discover all SeizeIt2 EDF + annotation pairs under ``root``.

    Expects each EDF file to have a companion ``*_a1.tsv`` in the same
    directory (filename without extension + ``_a1.tsv``).  Subject ID is
    taken from the immediate parent directory name of the EDF file.

    Returns:
        List of ``(edf_path, tsv_path, subject_id)`` tuples, sorted by path.
    """
    root = data_folder(root)
    results: List[Tuple[Path, Path, str]] = []

    for edf_path in sorted(root.rglob("*.edf")):
        tsv_path = edf_path.with_suffix("").with_name(
            edf_path.stem + "_a1.tsv"
        )
        if not tsv_path.exists():
            logger.debug("No annotation for %s, skipping", edf_path)
            continue
        subject_id = edf_path.parent.name
        results.append((edf_path, tsv_path, subject_id))

    logger.info("Discovered %d recordings under %s", len(results), root)
    return results


# ---------------------------------------------------------------------------
# Synthetic HDF5 (fallback for testing without real data)
# ---------------------------------------------------------------------------


def _build_synthetic_h5_cache(output_path: str | Path) -> None:
    """Write a minimal synthetic HDF5 cache with the same schema as the real one.

    Creates 2 recordings of 60 s each with random float32 signals and
    alternating 0/1 samplewise labels.  Used as a fallback when the real
    data path is not yet available so adapters and tests can run end-to-end.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    n_recs = 4  # 4 subjects, each one recording — enough for n_folds=2 k-fold
    n_samples_per_rec = TARGET_FS * 60  # 60 s
    total_samples = n_recs * n_samples_per_rec

    signals = rng.standard_normal((total_samples, 19)).astype(np.float32)

    labels = np.zeros(total_samples, dtype=np.uint8)
    # Add a synthetic seizure in the second half of each recording
    for i in range(n_recs):
        s = i * n_samples_per_rec + n_samples_per_rec // 2
        e = s + TARGET_FS * 5  # 5 s seizure
        labels[s:e] = 1

    offsets = np.array(
        [i * n_samples_per_rec for i in range(n_recs + 1)], dtype=np.int64
    )
    rec_ids = [f"synthetic_rec_{i:02d}.edf" for i in range(n_recs)]
    subj_ids = [f"SYNTHETIC_SUBJ_{i}" for i in range(n_recs)]
    durations = np.full(n_recs, float(n_samples_per_rec) / TARGET_FS, dtype=np.float32)
    n_missing = np.zeros(n_recs, dtype=np.int16)
    ch_masks = np.ones((n_recs, 19), dtype=bool)

    vlen_str = h5py.special_dtype(vlen=str)

    with h5py.File(str(output_path), "w") as h5:
        h5.attrs["fs"] = TARGET_FS
        h5.attrs["channels"] = list(CANONICAL_19)
        h5.attrs["montage"] = "unipolar"
        h5.attrs["synthetic"] = True

        h5.create_dataset("signals", data=signals)
        h5.create_dataset("samplewise_label", data=labels)
        h5.create_dataset("recording_offsets", data=offsets)
        h5.create_dataset(
            "recording_ids", data=np.array(rec_ids, dtype=object), dtype=vlen_str
        )
        h5.create_dataset(
            "subject_ids", data=np.array(subj_ids, dtype=object), dtype=vlen_str
        )
        h5.create_dataset("durations_s", data=durations)
        h5.create_dataset("n_missing_channels", data=n_missing)
        h5.create_dataset("channel_mask", data=ch_masks)
        h5.create_dataset(
            "ages", data=np.full(n_recs, -1, dtype=np.int16)
        )
        h5.create_dataset(
            "genders",
            data=np.array(["U"] * n_recs, dtype=object),
            dtype=vlen_str,
        )

    logger.info("Synthetic HDF5 cache written to %s", output_path)


# ---------------------------------------------------------------------------
# HDF5 cache builder
# ---------------------------------------------------------------------------


def build_h5_cache(root: str | Path, output_path: str | Path) -> Path:
    """Build a continuous HDF5 cache from SeizeIt2 EDF + TSV files.

    When ``root`` holds no recordings, or a file cannot be read, it raises.

    Args:
        root: Root directory containing SeizeIt2 `.edf` + `_a1.tsv` files.
        output_path: Destination HDF5 file path.

    Returns:
        Path to the written HDF5 file.
    """
    output_path = Path(output_path)
    root = data_folder(root)      # the recording ids are relative to it

    # --- real data path -------------------------------------------
    recordings = discover_recordings(root)
    if not recordings:
        raise FileNotFoundError(
            f"sz2: no EDF and annotation (_a1.tsv) pairs in {root} (give the folder "
            "with --data-root)"
        )

    all_signals: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    rec_ids: List[str] = []
    subj_ids: List[str] = []
    durations: List[float] = []
    n_missing: List[int] = []
    ch_masks: List[np.ndarray] = []
    offsets: List[int] = [0]

    for edf_path, tsv_path, subject_id in recordings:
        try:
            signals, fs, found = _read_sz2_edf_to_unipolar19(edf_path)
        except Exception as e:
            logger.warning("Skipping %s: %s", edf_path, e)
            continue

        n_samples = signals.shape[1]
        seizures = _parse_sz2_tsv(tsv_path)

        labels = np.zeros(n_samples, dtype=np.uint8)
        for start_s, end_s in seizures:
            s_idx = max(0, int(round(start_s * TARGET_FS)))
            e_idx = min(n_samples, int(round(end_s * TARGET_FS)))
            labels[s_idx:e_idx] = 1

        mask = np.zeros(19, dtype=bool)
        for ch in found:
            mask[CANONICAL_IDX[ch]] = True

        # Use path relative to root as recording ID
        try:
            rec_id = str(edf_path.relative_to(Path(root)))
        except ValueError:
            rec_id = str(edf_path)

        all_signals.append(signals.T)   # (N, 19)
        all_labels.append(labels)
        rec_ids.append(rec_id)
        subj_ids.append(subject_id)
        durations.append(n_samples / TARGET_FS)
        n_missing.append(19 - len(found))
        ch_masks.append(mask)
        offsets.append(offsets[-1] + n_samples)

        n_seiz = int(labels.sum())
        logger.info(
            "  %s: %d/19 channels, %d samples (%.1fs), %d seizure samples",
            rec_id, len(found), n_samples, n_samples / TARGET_FS, n_seiz,
        )

    if not rec_ids:
        raise RuntimeError(
            "All recordings were skipped during preprocessing."
        )

    # --- write HDF5 -----------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_samples = offsets[-1]
    logger.info(
        "Writing HDF5: %d recordings, %d total samples (%.1f hours)",
        len(rec_ids), total_samples, total_samples / TARGET_FS / 3600,
    )

    vlen_str = h5py.special_dtype(vlen=str)
    with h5py.File(str(output_path), "w") as h5:
        h5.attrs["fs"] = TARGET_FS
        h5.attrs["channels"] = list(CANONICAL_19)
        h5.attrs["montage"] = "unipolar"

        chunk = min(total_samples, TARGET_FS * 60)
        h5.create_dataset(
            "signals", shape=(total_samples, 19), dtype=np.float32,
            chunks=(chunk, 19),
        )
        h5.create_dataset(
            "samplewise_label", shape=(total_samples,), dtype=np.uint8,
            chunks=(chunk,),
        )
        for i, (sig, lbl) in enumerate(zip(all_signals, all_labels)):
            s, e = offsets[i], offsets[i + 1]
            h5["signals"][s:e] = sig
            h5["samplewise_label"][s:e] = lbl

        h5.create_dataset(
            "recording_offsets", data=np.array(offsets, dtype=np.int64)
        )
        h5.create_dataset(
            "recording_ids",
            data=np.array(rec_ids, dtype=object),
            dtype=vlen_str,
        )
        h5.create_dataset(
            "subject_ids",
            data=np.array(subj_ids, dtype=object),
            dtype=vlen_str,
        )
        h5.create_dataset(
            "durations_s", data=np.array(durations, dtype=np.float32)
        )
        h5.create_dataset(
            "n_missing_channels", data=np.array(n_missing, dtype=np.int16)
        )
        h5.create_dataset(
            "channel_mask", data=np.array(ch_masks, dtype=bool)
        )
        h5.create_dataset(
            "ages", data=np.full(len(rec_ids), -1, dtype=np.int16)
        )
        h5.create_dataset(
            "genders",
            data=np.array(["U"] * len(rec_ids), dtype=object),
            dtype=vlen_str,
        )

    logger.info("HDF5 cache written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="SeizeIt2 EDF+TSV -> continuous HDF5 preprocessor"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=SZ2_DATA_ROOT,
        help="Root directory containing SeizeIt2 .edf + _a1.tsv files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output HDF5 file path",
    )
    args = parser.parse_args()

    build_h5_cache(args.data_root, args.output)


if __name__ == "__main__":
    main()
