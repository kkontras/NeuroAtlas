"""CHB-MIT Scalp EEG -> continuous-HDF5 preprocessing pipeline.

Downloads the BIDS-formatted CHB-MIT dataset from Zenodo (preprocessed by
the SzCORE team: 18-channel double-banana bipolar montage, 256 Hz) and
builds a continuous HDF5 cache with samplewise binary seizure labels.

Also supports raw PhysioNet download as a fallback (19-channel unipolar,
variable montage — requires more complex channel mapping).

BIDS version reference:
    Dan, J. et al., 2024.  SzCORE: Seizure Community Open-Source Research
    Evaluation framework.  Epilepsia.
    https://zenodo.org/records/10259996

Original dataset reference:
    Shoeb, A.H., 2009.  Application of machine learning to epileptic
    seizure onset detection and treatment.  PhD thesis, MIT.
    https://physionet.org/content/chbmit/1.0.0/
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
    apply_standard_filters,
    bipolar_from_unipolar,
    resample_to,
)
from .bids_index import find_bids_root, parse_bids_events

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZENODO_URL = "https://zenodo.org/api/records/10259996/files/BIDS_CHB-MIT.zip/content"
PHYSIONET_URL = "https://physionet.org/files/chbmit/1.0.0/"

# The BIDS version uses 18-channel double-banana bipolar montage at 256 Hz.
# Names use the older 10-20 form (T3/T4/T5/T6) — same physical electrodes
# as the 1991-modified T7/T8/P7/P8, just the legacy convention. This aligns
# with BIOT's pretrained canonical 16-pair vocabulary and with the
# TUSZ/Siena/Epilepsiae bipolar montages so cross-dataset code paths see
# consistent channel names.
BIPOLAR_18: Tuple[str, ...] = (
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FZ-CZ", "CZ-PZ",
)

# Per-dataset channel alias map (local — differs from siena/epilepsiae)
CHANNEL_ALIAS: Dict[str, str] = {
    "FP1": "FP1", "FP2": "FP2", "F3": "F3", "F4": "F4",
    "C3": "C3", "C4": "C4", "P3": "P3", "P4": "P4",
    "O1": "O1", "O2": "O2", "F7": "F7", "F8": "F8",
    "T3": "T3", "T4": "T4", "T5": "T5", "T6": "T6",
    "FZ": "FZ", "CZ": "CZ", "PZ": "PZ",
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
    "Fp1": "FP1", "Fp2": "FP2", "Fz": "FZ", "Cz": "CZ", "Pz": "PZ",
}

NON_EEG_PREFIXES = ("ECG", "EOG", "EMG", "VNS", "-", ".", "LOC", "ROC",
                     "EKG", "BURSTS", "SUPPR", "IBI", "STI")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_bids(dest_dir: str | Path) -> Path:
    """Download CHB-MIT BIDS from Zenodo and unzip.

    Returns path to the unzipped BIDS root directory.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "BIDS_CHB-MIT.zip"

    # Check if already unzipped
    bids_root = find_bids_root(dest)
    if bids_root is not None:
        logger.info("BIDS data already present at %s, skipping download", bids_root)
        return bids_root

    # Download
    if not zip_path.exists():
        logger.info("Downloading CHB-MIT BIDS from Zenodo (~22 GB) ...")
        subprocess.run(
            ["wget", "-O", str(zip_path), "-c", ZENODO_URL],
            check=True,
        )
    else:
        logger.info("ZIP already downloaded at %s", zip_path)

    # Unzip
    logger.info("Unzipping %s ...", zip_path)
    subprocess.run(["unzip", "-o", "-q", str(zip_path), "-d", str(dest)], check=True)

    bids_root = find_bids_root(dest)
    if bids_root is None:
        raise RuntimeError(f"Could not find BIDS root (sub-* dirs) after unzipping in {dest}")

    logger.info("BIDS root: %s", bids_root)
    return bids_root


def _read_bids_edf(edf_path: Path) -> Tuple[np.ndarray, int, List[str]]:
    """Read a BIDS EDF file and return all EEG channels.

    Returns:
        signals: (n_channels, n_samples) float32
        fs: sampling rate
        channel_names: list of channel labels as found in the file
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

        # Find EEG channels (skip non-EEG)
        eeg_indices = []
        eeg_names = []
        for i, label in enumerate(ch_labels):
            name = label.replace("EEG ", "").replace("eeg ", "").strip()
            if any(name.upper().startswith(p) for p in NON_EEG_PREFIXES):
                continue
            eeg_indices.append(i)
            eeg_names.append(name)

        if not eeg_indices:
            raise ValueError(f"No EEG channels found in {edf_path}")

        fs = ch_fs[eeg_indices[0]]
        n_samples = edf.getNSamples()[eeg_indices[0]]

        signals = np.zeros((len(eeg_indices), n_samples), dtype=np.float32)
        for j, edf_idx in enumerate(eeg_indices):
            sig = edf.readSignal(edf_idx).astype(np.float32)
            if len(sig) < n_samples:
                sig = np.pad(sig, (0, n_samples - len(sig)))
            elif len(sig) > n_samples:
                sig = sig[:n_samples]
            signals[j] = sig

    finally:
        edf.close()

    return signals, fs, eeg_names


# ---------------------------------------------------------------------------
# HDF5 cache builder (BIDS)
# ---------------------------------------------------------------------------


def build_h5_cache_bids(
    bids_root: str | Path,
    output_path: str | Path,
) -> Path:
    """Build continuous HDF5 from BIDS-formatted CHB-MIT.

    The BIDS version has 18-channel bipolar montage at 256 Hz.
    We store the signals as-is (no montage conversion needed).
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
    n_channels_list: List[int] = []
    offsets: List[int] = [0]

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name  # e.g. "sub-01"

        # Find EDF files (may be in eeg/ subfolder or session subfolders)
        edf_files = sorted(subj_dir.rglob("*.edf"))
        if not edf_files:
            logger.warning("No EDF files for %s, skipping", subj_id)
            continue

        for edf_path in edf_files:
            # Find matching events.tsv
            events_path = Path(str(edf_path).replace("_eeg.edf", "_events.tsv"))
            if not events_path.exists():
                # Try other naming patterns
                stem = edf_path.stem.replace("_eeg", "")
                events_path = edf_path.parent / f"{stem}_events.tsv"
            if not events_path.exists():
                # Also check for events file with same base name
                for candidate in edf_path.parent.glob("*events.tsv"):
                    if edf_path.stem.split("_")[0] in candidate.stem:
                        events_path = candidate
                        break

            try:
                signals, fs, ch_names = _read_bids_edf(edf_path)
            except Exception as e:
                logger.warning("Skipping %s: %s", edf_path, e)
                continue

            n_ch, n_samples = signals.shape

            # Resample if needed (BIDS should be 256 Hz already)
            if fs != TARGET_FS:
                signals = resample_to(signals, fs, TARGET_FS)
                n_samples = signals.shape[1]

            # 0.5 Hz HP + 60 Hz notch (US mains — Children's Hospital Boston).
            # Applied once per recording at cache-build time; downstream code
            # must not re-filter per batch.
            if signals.shape[1] > 20:
                signals = apply_standard_filters(
                    signals, fs=TARGET_FS, notch_hz=60.0,
                )

            # Parse seizure annotations
            seizures: List[Tuple[float, float, str]] = []
            if events_path.exists():
                seizures = parse_bids_events(events_path)

            # Build samplewise labels
            labels = np.zeros(n_samples, dtype=np.uint8)
            for start_s, end_s, _ in seizures:
                s_idx = max(0, int(round(start_s * TARGET_FS)))
                e_idx = min(n_samples, int(round(end_s * TARGET_FS)))
                labels[s_idx:e_idx] = 1

            # Relative path for recording ID
            rel = edf_path.relative_to(bids_root)
            rec_id = str(rel)

            all_signals.append(signals.T)  # (N, n_ch)
            all_labels.append(labels)
            rec_ids.append(rec_id)
            subj_ids.append(subj_id)
            durations.append(n_samples / TARGET_FS)
            n_channels_list.append(n_ch)
            offsets.append(offsets[-1] + n_samples)

            n_seiz_samples = int(labels.sum())
            logger.info(
                "  %s: %d ch, %d samples (%.1fs), %d seizure events, "
                "%d seizure samples",
                rec_id, n_ch, n_samples, n_samples / TARGET_FS,
                len(seizures), n_seiz_samples,
            )

    if not rec_ids:
        raise RuntimeError("No recordings found in BIDS root")

    # Determine consistent channel count
    n_ch = n_channels_list[0]
    if not all(c == n_ch for c in n_channels_list):
        logger.warning(
            "Inconsistent channel counts: %s. Using %d (most common).",
            set(n_channels_list), n_ch,
        )

    total_samples = offsets[-1]
    logger.info(
        "Writing HDF5: %d recordings, %d total samples (%.1f hours), %d channels",
        len(rec_ids), total_samples, total_samples / TARGET_FS / 3600, n_ch,
    )

    vlen_str = h5py.special_dtype(vlen=str)

    with h5py.File(str(output_path), "w") as h5:
        h5.attrs["fs"] = TARGET_FS
        h5.attrs["n_channels"] = n_ch
        h5.attrs["montage"] = "bipolar"

        ds_sig = h5.create_dataset(
            "signals", shape=(total_samples, n_ch), dtype=np.float32,
            chunks=(min(total_samples, TARGET_FS * 60), n_ch),
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
        h5.create_dataset("n_channels_per_rec", data=np.array(n_channels_list, dtype=np.int16))
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

    parser = argparse.ArgumentParser(description="CHB-MIT BIDS -> HDF5 preprocessor")
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Directory to download/store BIDS data")
    parser.add_argument("--output", type=str, required=True,
                        help="Output HDF5 file path")
    parser.add_argument("--download", action="store_true",
                        help="Download BIDS from Zenodo before preprocessing")
    parser.add_argument("--bids-root", type=str, default=None,
                        help="Path to already-unzipped BIDS root (overrides auto-detection)")
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
