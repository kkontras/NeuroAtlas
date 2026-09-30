"""Helsinki Neonatal EEG Dataset — readers, annotations, optional HDF5 cache.

End-to-end support for the Stevenson et al. 2019 neonatal seizure corpus
(Zenodo 4940267 / 2547147).  79 neonates, NicoletOne 10-20, 256 Hz, three
independent expert seizure annotations (A/B/C) stored as MATLAB files.

Dataset reference:
    Stevenson, N.J., Tapani, K., Lauronen, L., Vanhatalo, S., 2019.
    A dataset of neonatal EEG recordings with seizure annotations.
    Scientific Data 6, 190039.  https://doi.org/10.1038/sdata.2019.39

Archive: https://zenodo.org/record/4940267  (CC BY 4.0, direct download)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._common import (
    CACHE_SCHEMA_TAG,
    CANONICAL_19,
    CANONICAL_IDX,
    CANONICAL_SET,
    GAP_LABEL,
    TARGET_FS,
    apply_standard_filters,
    resample_to,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZENODO_RECORD_ID = "4940267"
ZENODO_FILES = (
    "eeg1.edf", "annotations_2017_A.mat", "annotations_2017_B.mat",
    "annotations_2017_C.mat", "clinical_information.csv",
)  # plus eeg2.edf..eeg79.edf — fetched in a loop by the shell helper

# Per-dataset channel alias map (Zenodo EDFs use "EEG Fp1-Ref", "EEG T3-Ref",
# "Fp1", etc.).  Maps normalised names to CANONICAL_19 entries.
CHANNEL_ALIAS: Dict[str, str] = {
    "FP1": "FP1", "FP2": "FP2",
    "F3": "F3", "F4": "F4", "F7": "F7", "F8": "F8", "FZ": "FZ",
    "C3": "C3", "C4": "C4", "CZ": "CZ",
    "P3": "P3", "P4": "P4", "PZ": "PZ",
    "T3": "T3", "T4": "T4", "T5": "T5", "T6": "T6",
    "O1": "O1", "O2": "O2",
    # 10-10 names that map to the 10-20 canonical form
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
}

NON_EEG_PREFIXES = (
    "ECG", "EOG", "EMG", "EKG", "RESP", "SPO2", "PULS", "BEAT",
    "STI", "MK", "DC", "BN", "PHOTO",
)

EXPERT_IDS: Tuple[str, ...] = ("A", "B", "C")
CONSENSUS_MODES = ("majority", "unanimous", "any")

# ---------------------------------------------------------------------------
# Channel-name normalisation
# ---------------------------------------------------------------------------

_REF_SUFFIX_RE = re.compile(r"-(REF|LE|AR|AVG|A[12])$", re.IGNORECASE)


def _normalize_channel_label(raw: str) -> Optional[str]:
    """Normalise a raw EDF channel label to a CANONICAL_19 name, or ``None``.

    Strips a leading ``EEG `` prefix, trailing ``-REF``/``-LE``/``-AR``/
    ``-AVG``/``-A1``/``-A2`` references, upper-cases, and consults
    :data:`CHANNEL_ALIAS`.  Returns ``None`` for non-EEG channels (ECG, EOG,
    EMG, etc.) or channels that do not resolve to a canonical name.
    """
    s = (raw or "").strip()
    if not s:
        return None
    s = re.sub(r"^EEG\s+", "", s, flags=re.IGNORECASE)
    s = _REF_SUFFIX_RE.sub("", s).strip().upper()
    if any(s.startswith(p) for p in NON_EEG_PREFIXES):
        return None
    return CHANNEL_ALIAS.get(s)


# ---------------------------------------------------------------------------
# EDF reader — (19, n_samples) CANONICAL_19-ordered float32
# ---------------------------------------------------------------------------


def read_edf_canonical19(
    edf_path: str | Path,
) -> Tuple[np.ndarray, float, List[str]]:
    """Read a Helsinki EDF and return CANONICAL_19-ordered unipolar signals.

    Channel-first convention (matches Siena / CHB-MIT / EPILEPSIAE) so
    downstream helpers like ``_common.bipolar_from_unipolar`` apply directly.

    Returns:
        signals: ``(19, n_samples_at_TARGET_FS)`` float32.  Missing canonical
            channels are zero-filled.
        native_fs: native sampling rate (Hz) of the first matched channel.
        found: canonical channel names actually present in the source EDF.
    """
    import pyedflib

    reader = pyedflib.EdfReader(str(edf_path))
    try:
        n_signals = reader.signals_in_file
        raw_labels = [reader.getLabel(i) for i in range(n_signals)]
        fs_per_sig = [reader.getSampleFrequency(i) for i in range(n_signals)]

        ch_map: Dict[str, int] = {}
        for i, raw in enumerate(raw_labels):
            canonical = _normalize_channel_label(raw)
            if canonical is None or canonical not in CANONICAL_SET:
                continue
            ch_map.setdefault(canonical, i)

        if not ch_map:
            raise ValueError(f"No canonical 10-20 channels found in {edf_path}")

        native_fs = float(fs_per_sig[next(iter(ch_map.values()))])
        duration_s = float(reader.getFileDuration())
        n_target = max(1, int(round(duration_s * TARGET_FS)))

        signals = np.zeros((19, n_target), dtype=np.float32)
        found: List[str] = []

        for canonical, edf_idx in ch_map.items():
            raw = reader.readSignal(edf_idx).astype(np.float32)
            ch_fs = float(fs_per_sig[edf_idx])
            if abs(ch_fs - TARGET_FS) > 0.5:
                raw = resample_to(raw[None, :], ch_fs, TARGET_FS)[0]
            length = min(len(raw), n_target)
            signals[CANONICAL_IDX[canonical], :length] = raw[:length]
            found.append(canonical)

    finally:
        reader._close()

    return signals, native_fs, found


# ---------------------------------------------------------------------------
# MAT annotation parsing
# ---------------------------------------------------------------------------


def _expert_file(raw_root: Path, expert: str) -> Tuple[Path, str]:
    """Locate the per-expert annotation file and return (path, format_tag).

    Accepts both Zenodo release formats:
      - v1 (2547147): ``annotations_2017_{A,B,C}.mat`` — cell/matrix MAT.
      - v2 (4940267): ``annotations_2017_{A_fixed,B,C}.csv`` — columns are
        subject ids, rows are per-second labels.
    """
    for name in (
        f"annotations_2017_{expert}_fixed.csv",
        f"annotations_2017_{expert}.csv",
    ):
        p = raw_root / name
        if p.exists():
            return p, "csv"
    for name in (
        f"annotations_2017_{expert}.mat",
        f"annotations_2017_{expert.lower()}.mat",
        f"annotations_{expert}.mat",
    ):
        p = raw_root / name
        if p.exists():
            return p, "mat"
    raise FileNotFoundError(
        f"Expert-{expert} annotations not found under {raw_root}. "
        f"Expected annotations_2017_{expert}.csv or annotations_2017_{expert}.mat "
        f"from Zenodo {ZENODO_RECORD_ID}."
    )


def _extract_per_subject_annotations(mat_dict: Dict[str, Any], n_subjects: int) -> List[np.ndarray]:
    """Pull per-subject per-second binary seizure vectors from a loaded MAT.

    The Stevenson release stores annotations in a cell-array variable whose
    name has varied across Zenodo versions.  This helper tries known names
    (``annotat_new``, ``annotations``, ``seizures``, …) and accepts either
    a (n_subjects × n_seconds_max) numeric matrix or a length-n_subjects cell
    of per-subject row vectors.
    """
    candidate_keys = [
        "annotat_new", "annotations", "annotat",
        "seizures", "seizure", "annot",
    ]
    data = None
    for key in candidate_keys:
        if key in mat_dict:
            data = mat_dict[key]
            break
    if data is None:
        data_keys = [k for k in mat_dict if not k.startswith("__")]
        if len(data_keys) == 1:
            data = mat_dict[data_keys[0]]
        else:
            raise KeyError(
                f"Could not find a seizure-annotation variable in MAT file. "
                f"Tried {candidate_keys}; file variables: {data_keys}"
            )

    out: List[np.ndarray] = []

    if isinstance(data, np.ndarray) and data.dtype == object:
        # Cell array — iterate rows.
        cells = data.reshape(-1)
        if len(cells) < n_subjects:
            raise ValueError(
                f"MAT cell array has {len(cells)} entries, expected >= {n_subjects}"
            )
        for i in range(n_subjects):
            vec = np.asarray(cells[i]).reshape(-1)
            out.append((vec != 0).astype(np.uint8))
        return out

    arr = np.asarray(data)
    if arr.ndim == 1:
        if n_subjects == 1:
            return [(arr != 0).astype(np.uint8)]
        raise ValueError(
            f"MAT array is 1-D (length {arr.size}) but n_subjects={n_subjects}."
        )
    if arr.ndim != 2:
        raise ValueError(f"Unexpected MAT annotation shape: {arr.shape}")

    # 2-D: prefer subject-along-rows, else transpose.
    if arr.shape[0] != n_subjects and arr.shape[1] == n_subjects:
        arr = arr.T
    if arr.shape[0] < n_subjects:
        raise ValueError(
            f"MAT matrix has {arr.shape[0]} rows, expected >= {n_subjects}"
        )
    for i in range(n_subjects):
        out.append((arr[i] != 0).astype(np.uint8))
    return out


def _load_csv_annotations(csv_path: Path, n_subjects: int) -> List[np.ndarray]:
    """Read a v2-release expert CSV and return per-subject per-second vectors.

    CSV layout: header = subject ids (``1..N``), each row = one second,
    values are 0/1.  Subjects with shorter recordings have empty cells past
    their recording length — treated as "beyond end of recording" and the
    returned per-subject vector is trimmed to the last non-empty entry.
    """
    import csv as _csv

    with open(csv_path, "r") as fh:
        reader = _csv.reader(fh)
        header = next(reader)
        n_cols = len(header)
        rows: List[List[str]] = [row for row in reader]

    if n_cols < n_subjects:
        raise ValueError(
            f"{csv_path.name} has {n_cols} subject columns, expected >= {n_subjects}"
        )

    out: List[np.ndarray] = []
    for col in range(n_subjects):
        last_valid = -1
        for r, row in enumerate(rows):
            if col < len(row) and row[col].strip():
                last_valid = r
        length = last_valid + 1
        vec = np.zeros(length, dtype=np.uint8)
        for r in range(length):
            v = rows[r][col].strip() if col < len(rows[r]) else ""
            if v and v != "0":
                vec[r] = 1
        out.append(vec)
    return out


def load_helsinki_annotations(
    raw_root: str | Path,
    n_subjects: int = 79,
) -> Dict[str, List[np.ndarray]]:
    """Load per-second binary seizure vectors from all three experts.

    Supports both Zenodo release formats (v1 MAT, v2 CSV).  Returns
    ``{"A": [vec_1, ..., vec_N], "B": [...], "C": [...]}`` with 1-based
    subject indexing (``vec_{i-1}`` is subject ``i``).
    """
    raw_root = Path(raw_root)
    out: Dict[str, List[np.ndarray]] = {}
    for expert in EXPERT_IDS:
        path, fmt = _expert_file(raw_root, expert)
        if fmt == "csv":
            out[expert] = _load_csv_annotations(path, n_subjects)
        else:
            from scipy.io import loadmat
            mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
            out[expert] = _extract_per_subject_annotations(mat, n_subjects)
    return out


# ---------------------------------------------------------------------------
# Per-second consensus collapse
# ---------------------------------------------------------------------------


def consensus_per_second(
    a: np.ndarray, b: np.ndarray, c: np.ndarray,
    mode: str = "majority",
) -> np.ndarray:
    """Collapse three per-second expert vectors to one binary vector.

    The three inputs need not be equal length; the output length equals the
    shortest common prefix (trailing gaps in any expert are treated as
    missing, not 0).

    Args:
        a, b, c: 1-D uint8 per-second labels.
        mode: ``"majority"`` (≥2/3), ``"unanimous"`` (3/3), ``"any"`` (≥1/3).

    Returns:
        1-D uint8 of length ``min(len(a), len(b), len(c))``.
    """
    if mode not in CONSENSUS_MODES:
        raise ValueError(f"mode must be one of {CONSENSUS_MODES}, got {mode!r}")
    L = min(len(a), len(b), len(c))
    aa = np.asarray(a[:L], dtype=np.uint8)
    bb = np.asarray(b[:L], dtype=np.uint8)
    cc = np.asarray(c[:L], dtype=np.uint8)
    if mode == "majority":
        return ((aa.astype(np.int16) + bb + cc) >= 2).astype(np.uint8)
    if mode == "unanimous":
        return (aa & bb & cc).astype(np.uint8)
    return (aa | bb | cc).astype(np.uint8)


def subject_consensus_type(
    a: np.ndarray, b: np.ndarray, c: np.ndarray,
) -> str:
    """Classify a subject as ``"unanimous_seizure"``, ``"unanimous_nonseizure"``, or ``"mixed"``.

    Matches Stevenson 2019's 39/22/18 partition driven by inter-rater agreement
    at the recording level.
    """
    L = min(len(a), len(b), len(c))
    any_sz = [np.any(np.asarray(v[:L]) != 0) for v in (a, b, c)]
    if all(any_sz):
        return "unanimous_seizure"
    if not any(any_sz):
        return "unanimous_nonseizure"
    return "mixed"


# ---------------------------------------------------------------------------
# Per-second vector -> sample-index intervals
# ---------------------------------------------------------------------------


def per_second_to_intervals_samples(
    per_second: np.ndarray,
    fs: int = TARGET_FS,
) -> List[Tuple[int, int]]:
    """Convert a per-second binary vector to a list of sample-index intervals."""
    if len(per_second) == 0:
        return []
    # Pad with zeros on both sides so edges produce transitions.
    padded = np.concatenate(([0], (np.asarray(per_second) != 0).astype(np.int8), [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]  # seconds
    ends = np.where(diffs == -1)[0]   # seconds (exclusive)
    return [(int(s * fs), int(e * fs)) for s, e in zip(starts, ends)]


# ---------------------------------------------------------------------------
# EDF discovery
# ---------------------------------------------------------------------------


_SUBJECT_RE = re.compile(r"eeg(\d+)\.edf$", re.IGNORECASE)


def discover_helsinki_recordings(raw_root: str | Path) -> List[Dict[str, Any]]:
    """Return ``[{subject_id, edf_path}, ...]`` sorted by integer subject id.

    Subject ids are taken from the ``eegN.edf`` filename (1-79).
    """
    raw_root = Path(raw_root)
    recs: List[Dict[str, Any]] = []
    for edf_path in raw_root.glob("eeg*.edf"):
        m = _SUBJECT_RE.search(edf_path.name)
        if not m:
            continue
        recs.append({
            "subject_id": int(m.group(1)),
            "edf_path": str(edf_path),
        })
    recs.sort(key=lambda r: r["subject_id"])
    if not recs:
        raise FileNotFoundError(
            f"No eegN.edf files under {raw_root}. "
            f"Run `python -m entrypoints.fetch --dataset helsinki_neonatal --download` to stage it."
        )
    return recs


# ---------------------------------------------------------------------------
# Optional HDF5 cache builder (Siena-schema-compatible)
# ---------------------------------------------------------------------------


def build_h5_cache(
    raw_root: str | Path,
    output_path: str | Path,
    consensus: str = "majority",
    apply_filters: bool = False,
    notch_hz: float = 50.0,
) -> Path:
    """Build a continuous HDF5 cache (19-ch unipolar @ 256 Hz) for Helsinki.

    Schema identical to ``siena_preprocessor.build_h5_cache_bids``.
    Samplewise labels use the configured ``consensus`` collapse.

    NOTE: the in-tree H5 cache reader (``HelsinkiNeonatalContinuousDataset``)
    has been removed; this builder is currently orphaned and kept only for
    external scripts that may still consume the schema. The live benchmark
    path uses the EDF-direct ``HelsinkiNeonatalEdfDataset`` reader.
    """
    import h5py

    raw_root = Path(raw_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    recordings = discover_helsinki_recordings(raw_root)
    annotations = load_helsinki_annotations(raw_root, n_subjects=len(recordings))

    all_signals: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    rec_ids: List[str] = []
    subj_ids: List[str] = []
    durations: List[float] = []
    n_missing: List[int] = []
    ch_masks: List[np.ndarray] = []
    offsets: List[int] = [0]

    for rec in recordings:
        subject_id = rec["subject_id"]
        try:
            signals, fs, found = read_edf_canonical19(rec["edf_path"])
        except Exception as exc:
            logger.warning("Skipping eeg%d: %s", subject_id, exc)
            continue

        if apply_filters:
            signals = apply_standard_filters(signals, fs=TARGET_FS, notch_hz=notch_hz, axis=1)

        n_samples = signals.shape[1]
        # Align per-second expert vectors and collapse.
        a = annotations["A"][subject_id - 1]
        b = annotations["B"][subject_id - 1]
        c = annotations["C"][subject_id - 1]
        per_sec = consensus_per_second(a, b, c, mode=consensus)
        labels = np.zeros(n_samples, dtype=np.uint8)
        for s_idx, e_idx in per_second_to_intervals_samples(per_sec, fs=TARGET_FS):
            labels[max(0, s_idx):min(n_samples, e_idx)] = 1

        mask = np.zeros(19, dtype=bool)
        for ch in found:
            mask[CANONICAL_IDX[ch]] = True

        rec_id = f"eeg{subject_id}"
        all_signals.append(signals.T)  # (N, 19)
        all_labels.append(labels)
        rec_ids.append(rec_id)
        subj_ids.append(f"{subject_id:03d}")
        durations.append(n_samples / TARGET_FS)
        n_missing.append(19 - len(found))
        ch_masks.append(mask)
        offsets.append(offsets[-1] + n_samples)

        logger.info(
            "  %s: %d/19 channels, %d samples (%.1fs), %d seizure samples (consensus=%s)",
            rec_id, len(found), n_samples, n_samples / TARGET_FS,
            int(labels.sum()), consensus,
        )

    if not rec_ids:
        raise RuntimeError("No recordings produced in Helsinki cache build.")

    total_samples = offsets[-1]
    vlen_str = h5py.special_dtype(vlen=str)

    with h5py.File(str(output_path), "w") as h5:
        h5.attrs["fs"] = TARGET_FS
        h5.attrs["channels"] = list(CANONICAL_19)
        h5.attrs["montage"] = "unipolar"
        h5.attrs["schema_tag"] = CACHE_SCHEMA_TAG
        h5.attrs["consensus"] = consensus

        ds_sig = h5.create_dataset(
            "signals", shape=(total_samples, 19), dtype=np.float32,
            chunks=(min(total_samples, TARGET_FS * 60), 19),
        )
        ds_lbl = h5.create_dataset(
            "samplewise_label", shape=(total_samples,), dtype=np.uint8,
            chunks=(min(total_samples, TARGET_FS * 60),),
        )
        for i, (sig, lbl) in enumerate(zip(all_signals, all_labels)):
            s, e = offsets[i], offsets[i + 1]
            ds_sig[s:e] = sig
            ds_lbl[s:e] = lbl

        h5.create_dataset("recording_offsets", data=np.array(offsets, dtype=np.int64))
        h5.create_dataset("recording_ids", data=np.array(rec_ids, dtype=object), dtype=vlen_str)
        h5.create_dataset("subject_ids", data=np.array(subj_ids, dtype=object), dtype=vlen_str)
        h5.create_dataset("durations_s", data=np.array(durations, dtype=np.float32))
        h5.create_dataset("n_missing_channels", data=np.array(n_missing, dtype=np.int16))
        h5.create_dataset("channel_mask", data=np.array(ch_masks, dtype=bool))

    logger.info("HDF5 cache written to %s", output_path)
    return output_path


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Helsinki neonatal EEG -> HDF5 preprocessor")
    parser.add_argument("--raw-root", type=str, required=True,
                        help="Directory containing eegN.edf + annotations_2017_{A,B,C}.mat")
    parser.add_argument("--output", type=str, required=True, help="Output HDF5 path")
    parser.add_argument("--consensus", type=str, default="majority",
                        choices=list(CONSENSUS_MODES),
                        help="Inter-rater consensus for samplewise labels")
    parser.add_argument("--apply-filters", action="store_true",
                        help="Apply 0.5 Hz HP + 50 Hz notch during caching")
    args = parser.parse_args()
    build_h5_cache(args.raw_root, args.output, consensus=args.consensus,
                   apply_filters=args.apply_filters)


if __name__ == "__main__":
    main()
