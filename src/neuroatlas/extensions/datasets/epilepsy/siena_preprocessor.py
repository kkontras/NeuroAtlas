"""Siena Scalp EEG -> continuous-HDF5 preprocessing pipeline.

Downloads the BIDS-formatted Siena dataset from Zenodo (preprocessed by
the SzCORE team: 19-channel 10-20 common-average reference, 256 Hz) and
builds a continuous HDF5 cache with samplewise binary seizure labels.

BIDS version reference:
    Dan, J. et al., 2024.  SzCORE: Seizure Community Open-Source Research
    Evaluation framework.  Epilepsia.
    https://zenodo.org/records/10640762

Original dataset reference:
    Detti, P., et al., 2020.  EEG Synchronization Analysis for Seizure
    Prediction: A Study on Data of Noninvasive Recordings.
    https://physionet.org/content/siena-scalp-eeg/1.0.0/
"""
from __future__ import annotations

import csv
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from ._common import (
    BIPOLAR_MONTAGE,
    BIPOLAR_NAMES,
    CACHE_SCHEMA_TAG,
    CANONICAL_19,
    CANONICAL_IDX,
    CANONICAL_SET,
    GAP_LABEL,
    TARGET_FS,
    _BIP_A,
    _BIP_B,
    bipolar_from_unipolar,
    resample_to,
)
from .bids_index import find_bids_root, parse_bids_events

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZENODO_URL = "https://zenodo.org/api/records/10640762/files/BIDS_Siena.zip/content"

# Per-dataset channel alias map (local — differs from chbmit/epilepsiae)
CHANNEL_ALIAS: Dict[str, str] = {
    "FP1": "FP1", "FP2": "FP2", "F3": "F3", "F4": "F4",
    "C3": "C3", "C4": "C4", "P3": "P3", "P4": "P4",
    "O1": "O1", "O2": "O2", "F7": "F7", "F8": "F8",
    "T3": "T3", "T4": "T4", "T5": "T5", "T6": "T6",
    "FZ": "FZ", "CZ": "CZ", "PZ": "PZ",
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
    "Fp1": "FP1", "Fp2": "FP2", "Fz": "FZ", "Cz": "CZ", "Pz": "PZ",
}

NON_EEG_PREFIXES = ("ECG", "EOG", "EMG", "EKG", "BN", "MK", "DC", "PHOTO",
                     "SpO2", "PULS", "BEAT", "STI")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_bids(dest_dir: str | Path) -> Path:
    """Download Siena BIDS from Zenodo and unzip.

    Returns path to the unzipped BIDS root directory.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "BIDS_Siena.zip"

    bids_root = find_bids_root(dest)
    if bids_root is not None:
        logger.info("BIDS data already present at %s, skipping download", bids_root)
        return bids_root

    if not zip_path.exists():
        logger.info("Downloading Siena BIDS from Zenodo (~4.5 GB) ...")
        subprocess.run(
            ["wget", "-O", str(zip_path), "-c", ZENODO_URL],
            check=True,
        )
    else:
        logger.info("ZIP already downloaded at %s", zip_path)

    logger.info("Unzipping %s ...", zip_path)
    subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(dest)], check=True)

    bids_root = find_bids_root(dest)
    if bids_root is None:
        raise RuntimeError(f"Could not find BIDS root after unzipping in {dest}")

    logger.info("BIDS root: %s", bids_root)
    return bids_root


def _read_bids_edf_to_unipolar19(edf_path: Path) -> Tuple[np.ndarray, int, List[str]]:
    """Read a BIDS EDF and map to canonical 19-channel unipolar layout.

    The Siena BIDS data is already 19-channel 10-20 at 256 Hz.

    Returns:
        signals: (19, n_samples) float32.  Missing channels zero-filled.
        fs: sampling rate
        found_channels: list of canonical channel names found
    """
    import pyedflib

    edf = pyedflib.EdfReader(str(edf_path))
    try:
        n_ch = edf.signals_in_file
        ch_labels = [
            edf.signal_label(i).decode() if isinstance(edf.signal_label(i), bytes)
            else edf.signal_label(i)
            for i in range(n_ch)
        ]
        ch_labels = [s.strip() for s in ch_labels]
        ch_fs = [int(edf.getSampleFrequency(i)) for i in range(n_ch)]

        ch_map: Dict[str, int] = {}
        for i, raw_label in enumerate(ch_labels):
            name = raw_label.replace("EEG ", "").replace("eeg ", "")
            name = name.split("-")[0].strip().upper()
            if any(name.startswith(p) for p in NON_EEG_PREFIXES):
                continue
            canonical = CHANNEL_ALIAS.get(name)
            if canonical is None:
                canonical = CHANNEL_ALIAS.get(raw_label.split("-")[0].strip())
            if canonical is not None and canonical in CANONICAL_SET:
                if canonical not in ch_map:
                    ch_map[canonical] = i

        eeg_indices = list(ch_map.values())
        if not eeg_indices:
            raise ValueError(f"No canonical EEG channels found in {edf_path}")
        fs = ch_fs[eeg_indices[0]]

        n_samples = edf.getNSamples()[eeg_indices[0]]
        signals = np.zeros((19, n_samples), dtype=np.float32)
        found: List[str] = []

        for canonical_name, edf_idx in ch_map.items():
            cix = CANONICAL_IDX[canonical_name]
            sig = edf.readSignal(edf_idx).astype(np.float32)
            if len(sig) < n_samples:
                sig = np.pad(sig, (0, n_samples - len(sig)))
            elif len(sig) > n_samples:
                sig = sig[:n_samples]
            signals[cix] = sig
            found.append(canonical_name)

    finally:
        edf.close()

    return signals, fs, found


# ---------------------------------------------------------------------------
# HDF5 cache builder (BIDS)
# ---------------------------------------------------------------------------


def build_h5_cache_bids(
    bids_root: str | Path,
    output_path: str | Path,
) -> Path:
    """Build continuous HDF5 from BIDS-formatted Siena data.

    The Siena BIDS version has 19-channel unipolar 10-20 at 256 Hz — same
    schema as TUSZ and EPILEPSIAE caches.
    """
    bids_root = Path(bids_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(bids_root.glob("sub-*"))
    if not subject_dirs:
        raise RuntimeError(f"No sub-* directories found in {bids_root}")

    all_signals: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    rec_ids: List[str] = []
    subj_ids: List[str] = []
    durations: List[float] = []
    n_missing: List[int] = []
    ch_masks: List[np.ndarray] = []
    offsets: List[int] = [0]

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name

        edf_files = sorted(subj_dir.rglob("*.edf"))
        if not edf_files:
            logger.warning("No EDF files for %s, skipping", subj_id)
            continue

        for edf_path in edf_files:
            # Find matching events.tsv
            events_path = Path(str(edf_path).replace("_eeg.edf", "_events.tsv"))
            if not events_path.exists():
                stem = edf_path.stem.replace("_eeg", "")
                events_path = edf_path.parent / f"{stem}_events.tsv"
            if not events_path.exists():
                for candidate in edf_path.parent.glob("*events.tsv"):
                    if edf_path.stem.split("_")[0] in candidate.stem:
                        events_path = candidate
                        break

            try:
                signals, fs, found = _read_bids_edf_to_unipolar19(edf_path)
            except Exception as e:
                logger.warning("Skipping %s: %s", edf_path, e)
                continue

            n_samples = signals.shape[1]

            # Resample if needed (should already be 256 Hz)
            if fs != TARGET_FS:
                signals = resample_to(signals, fs, TARGET_FS)
                n_samples = signals.shape[1]

            # Parse seizure annotations
            seizures: List[Tuple[float, float, str]] = []
            if events_path.exists():
                seizures = parse_bids_events(events_path)

            labels = np.zeros(n_samples, dtype=np.uint8)
            for start_s, end_s, _ in seizures:
                s_idx = max(0, int(round(start_s * TARGET_FS)))
                e_idx = min(n_samples, int(round(end_s * TARGET_FS)))
                labels[s_idx:e_idx] = 1

            mask = np.zeros(19, dtype=bool)
            for ch in found:
                mask[CANONICAL_IDX[ch]] = True

            rel = edf_path.relative_to(bids_root)
            rec_id = str(rel)

            all_signals.append(signals.T)  # (N, 19)
            all_labels.append(labels)
            rec_ids.append(rec_id)
            subj_ids.append(subj_id)
            durations.append(n_samples / TARGET_FS)
            n_missing.append(19 - len(found))
            ch_masks.append(mask)
            offsets.append(offsets[-1] + n_samples)

            n_seiz_samples = int(labels.sum())
            logger.info(
                "  %s: %d/19 channels, %d samples (%.1fs), %d seizure events, "
                "%d seizure samples",
                rec_id, len(found), n_samples, n_samples / TARGET_FS,
                len(seizures), n_seiz_samples,
            )

    if not rec_ids:
        raise RuntimeError("No recordings found in BIDS root")

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

        ds_sig = h5.create_dataset(
            "signals", shape=(total_samples, 19), dtype=np.float32,
            chunks=(min(total_samples, TARGET_FS * 60), 19),
        )
        ds_lbl = h5.create_dataset(
            "samplewise_label", shape=(total_samples,), dtype=np.uint8,
            chunks=(min(total_samples, TARGET_FS * 60),),
        )

        for i, (sig, lbl) in enumerate(zip(all_signals, all_labels)):
            s = offsets[i]
            e = offsets[i + 1]
            ds_sig[s:e] = sig
            ds_lbl[s:e] = lbl

        h5.create_dataset("recording_offsets", data=np.array(offsets, dtype=np.int64))
        h5.create_dataset("recording_ids", data=np.array(rec_ids, dtype=object), dtype=vlen_str)
        h5.create_dataset("subject_ids", data=np.array(subj_ids, dtype=object), dtype=vlen_str)
        h5.create_dataset("durations_s", data=np.array(durations, dtype=np.float32))
        h5.create_dataset("n_missing_channels", data=np.array(n_missing, dtype=np.int16))
        h5.create_dataset("channel_mask", data=np.array(ch_masks, dtype=bool))
        h5.create_dataset("ages", data=np.full(len(rec_ids), -1, dtype=np.int16))
        h5.create_dataset("genders", data=np.array(["U"] * len(rec_ids), dtype=object), dtype=vlen_str)

    logger.info("HDF5 cache written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Siena BIDS -> HDF5 preprocessor")
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Directory to download/store BIDS data")
    parser.add_argument("--output", type=str, required=True,
                        help="Output HDF5 file path")
    parser.add_argument("--download", action="store_true",
                        help="Download BIDS from Zenodo before preprocessing")
    parser.add_argument("--bids-root", type=str, default=None,
                        help="Path to already-unzipped BIDS root")
    args = parser.parse_args()

    if args.download:
        bids_root = download_bids(args.raw_dir)
    elif args.bids_root:
        bids_root = Path(args.bids_root)
    else:
        bids_root = find_bids_root(Path(args.raw_dir))
        if bids_root is None:
            raise RuntimeError(
                f"No BIDS root found in {args.raw_dir}. "
                "Use --download to fetch from Zenodo or --bids-root to specify."
            )

    build_h5_cache_bids(bids_root, args.output)


if __name__ == "__main__":
    main()
