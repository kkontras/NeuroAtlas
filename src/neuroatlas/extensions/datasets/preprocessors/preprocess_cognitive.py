"""Build the bci_cognitive datasets' files from their public raw data.

One cohort per call (``--dataset``): EEGMat (PhysioNet ``eegmat`` EDFs),
ArithmeticTask (the OSF release, ``Experiment 1/`` and ``Experiment 2/``),
DREAMER valence or arousal (``DREAMER.mat``). Each call writes the two files
the reader reads (``bci_paths.COGNITIVE_PICKLES``):

    <stem>_preprocessed_steegformer.pkl          no filtering: 0.1-64 Hz
    <stem>_preprocessed_trackD_steegformer.pkl   confound filtering: 4-40 Hz

both at 128 Hz, with a 50 Hz notch (and its harmonics below the native
Nyquist) and the average reference, in microvolts. The preparation is the
one the published files were made with (the cohorts' ``preprocess_*.py``
scripts, their ``steegformer`` and ``trackD_steegformer`` rows):

    EEGMat          19 channels; rest (``SubjectNN_1.edf``) = 0 and mental
                    arithmetic (``SubjectNN_2.edf``) = 1; each recording
                    filtered whole, then cut into 4 s windows
    ArithmeticTask  19 channels; rest = 0, arithmetic = 1, meditation or
                    breath focus = 2; the first 30 s skipped, at most 5 min
                    kept (a recording up to 10 min: its last 5 min; longer:
                    5-10 min); 1 s windows
    DREAMER         14 channels; the last 60 s of each of the 18 film clips
                    in 4 s windows, labelled by the clip's self-rated
                    valence or arousal, low (1-3) = 0, high (4-5) = 1

The subjects are the cohort's own list (``DATASET_CONFIGS[slug].subjects``).

Each file is a pickle of ``data_raw`` (per subject, windows x channels x
samples), ``condition`` (per subject, the labels), ``subject_name`` and
``preprocessing_meta`` (band, notch, rate, window), which the reader checks.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: (fmin, fmax, resample rate, notch) of the two files: the published files'
#: ``steegformer`` and ``trackD_steegformer`` rows
FORMATS: Dict[bool, Tuple[float, float, float, float]] = {
    False: (0.1, 64.0, 128.0, 50.0),
    True: (4.0, 40.0, 128.0, 50.0),
}
FORMAT_NAME = "steegformer"

# --------------------------------------------------------------------------- EEGMat

_EEGMAT_RENAME = {
    "EEG Fp1": "Fp1", "EEG Fp2": "Fp2", "EEG F3": "F3", "EEG F4": "F4",
    "EEG F7": "F7", "EEG F8": "F8", "EEG T3": "T7", "EEG T4": "T8",
    "EEG C3": "C3", "EEG C4": "C4", "EEG T5": "P7", "EEG T6": "P8",
    "EEG P3": "P3", "EEG P4": "P4", "EEG O1": "O1", "EEG O2": "O2",
    "EEG Fz": "Fz", "EEG Cz": "Cz", "EEG Pz": "Pz",
}
_EEGMAT_CHANNELS = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "T7", "T8", "C3", "C4",
                    "P7", "P8", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz"]

# ------------------------------------------------------------------ ArithmeticTask

_ARITH_RENAME = {"FP1": "Fp1", "FP2": "Fp2", "T3": "T7", "T4": "T8",
                 "T5": "P7", "T6": "P8", "PZ": "Pz"}
_ARITH_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
                   "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
_ARITH_LABELS = {"R": 0, "A": 1, "M": 2, "B": 2}
#: experiment -> (subject-id offset, conditions, folder names it may have)
_ARITH_EXPERIMENTS = {
    1: (1000, ("R", "A", "M"), ("Experiment 1", "Experiment1_new", "Experiment1")),
    2: (2000, ("R", "A", "B"), ("Experiment 2", "Experiment2_new", "Experiment2")),
}
_ARITH_SKIP_S, _ARITH_MAX_S = 30.0, 300.0

# ------------------------------------------------------------------------ DREAMER

_DREAMER_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8",
                     "FC6", "F4", "F8", "AF4"]
_DREAMER_SFREQ = 128.0
_DREAMER_SEGMENT_S = 60.0
_DREAMER_THRESHOLD = 3


def _windows(data: np.ndarray, n: int) -> Optional[np.ndarray]:
    """Non-overlapping windows of *n* samples, (windows, channels, n)."""
    k = data.shape[1] // n
    if k == 0:
        return None
    return np.stack([data[:, i * n:(i + 1) * n] for i in range(k)]).astype(np.float32)


def _filter(raw, fmt, *, upsample_first: bool = False):
    """Notch, band-pass, average reference and resample *raw* (MNE Raw), in
    the published files' order."""
    fmin, fmax, sfreq, notch = fmt
    if upsample_first and fmax >= raw.info["sfreq"] / 2.0:
        # DREAMER is recorded at 128 Hz: a 64 Hz low-pass needs a higher rate first
        raw.resample(max(sfreq, 2 * fmax + 2), verbose=False)
    freqs = np.arange(notch, raw.info["sfreq"] / 2, notch)
    if len(freqs):
        raw.notch_filter(freqs, verbose=False)
    raw.filter(l_freq=fmin, h_freq=fmax, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    if abs(raw.info["sfreq"] - sfreq) > 0.1:
        raw.resample(sfreq, verbose=False)
    return raw


def _eegmat_edf(path: Path, fmt) -> Optional[np.ndarray]:
    import mne

    raw = mne.io.read_raw_edf(str(path), preload=True, encoding="latin1", verbose=False)
    raw.rename_channels({c: _EEGMAT_RENAME[c] for c in raw.ch_names if c in _EEGMAT_RENAME})
    raw.pick(_EEGMAT_CHANNELS)
    raw.reorder_channels(_EEGMAT_CHANNELS)
    raw = _filter(raw, fmt)
    return _windows(raw.get_data() * 1e6, int(4.0 * raw.info["sfreq"]))


def eegmat_subject(raw_dir: Path, subject: int, fmt) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    rest, task = (raw_dir / f"Subject{subject:02d}_{k}.edf" for k in (1, 2))
    if not (rest.is_file() and task.is_file()):
        return None
    r, t = _eegmat_edf(rest, fmt), _eegmat_edf(task, fmt)
    if r is None or t is None:
        return None
    return (np.concatenate([r, t]),
            np.concatenate([np.zeros(len(r), int), np.ones(len(t), int)]))


def _arith_segment(duration: float) -> Tuple[float, float]:
    if duration <= _ARITH_MAX_S:
        return _ARITH_SKIP_S, duration
    if duration <= 600.0:
        return duration - _ARITH_MAX_S, duration
    return 300.0, 600.0


def _arith_file(raw_dir: Path, experiment: int, number: int, condition: str) -> Optional[Path]:
    """One recording, under any of the layouts the release has been kept in:
    ``Experiment 2/S2A.edf`` (the OSF release), ``Experiment2_new/S02/S02A.edf``."""
    folders = _ARITH_EXPERIMENTS[experiment][2]
    for folder in folders:
        base = raw_dir / folder
        for name in (f"S{number:02d}{condition}.edf", f"S{number}{condition}.edf"):
            for path in (base / name, base / f"S{number:02d}" / name, base / f"S{number}" / name):
                if path.is_file():
                    return path
    return None


def _arith_edf(path: Path, fmt) -> Optional[np.ndarray]:
    import mne

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    raw.rename_channels({c: _ARITH_RENAME[c] for c in raw.ch_names if c in _ARITH_RENAME})
    raw.pick(_ARITH_CHANNELS)
    raw.reorder_channels(_ARITH_CHANNELS)
    start, end = _arith_segment(raw.times[-1])
    if end - start < 1.0:
        return None
    raw.crop(tmin=start, tmax=end)
    raw = _filter(raw, fmt)
    return _windows(raw.get_data() * 1e6, int(1.0 * raw.info["sfreq"]))


def arithmetic_subject(raw_dir: Path, subject: int, fmt) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    experiment, number = divmod(subject, 1000)
    if experiment not in _ARITH_EXPERIMENTS:
        return None
    xs, ys = [], []
    for condition in _ARITH_EXPERIMENTS[experiment][1]:
        path = _arith_file(raw_dir, experiment, number, condition)
        if path is None:
            return None
        w = _arith_edf(path, fmt)
        if w is None:
            return None
        xs.append(w)
        ys.append(np.full(len(w), _ARITH_LABELS[condition], int))
    return np.concatenate(xs), np.concatenate(ys)


def _dreamer_mat(raw_dir: Path) -> Dict:
    import scipy.io

    path = raw_dir / "DREAMER.mat" if raw_dir.is_dir() else raw_dir
    return scipy.io.loadmat(str(path), simplify_cells=True)["DREAMER"]


def dreamer_subject(mat: Dict, subject: int, fmt, label: str
                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    import mne

    data = mat["Data"][subject - 1]
    scores = np.asarray(data["ScoreValence" if label == "valence" else "ScoreArousal"])
    n_segment = int(_DREAMER_SEGMENT_S * _DREAMER_SFREQ)
    xs, ys = [], []
    for t, trial in enumerate(data["EEG"]["stimuli"]):
        trial = np.asarray(trial)
        if trial.shape[0] < n_segment:
            continue
        segment = trial[-n_segment:, :].T.astype(np.float64)      # already in microvolts
        info = mne.create_info(list(_DREAMER_CHANNELS), _DREAMER_SFREQ, "eeg")
        raw = _filter(mne.io.RawArray(segment, info, verbose=False), fmt, upsample_first=True)
        w = _windows(raw.get_data(), int(4.0 * raw.info["sfreq"]))
        if w is None:
            continue
        xs.append(w)
        ys.append(np.full(len(w), 0 if scores[t] <= _DREAMER_THRESHOLD else 1, int))
    if not xs:
        return None
    return np.concatenate(xs), np.concatenate(ys)


# --------------------------------------------------------------------------- build

def _payload(subjects: List[int], windows: List[np.ndarray], labels: List[np.ndarray],
             fmt, slug: str, channels: Sequence[str], window_s: float) -> Dict:
    data_raw = np.empty(len(subjects), dtype=object)
    condition = np.empty(len(subjects), dtype=object)
    for i in range(len(subjects)):
        data_raw[i], condition[i] = windows[i], labels[i]
    fmin, fmax, sfreq, notch = fmt
    return {
        "data_raw": data_raw, "condition": condition,
        "subject_name": np.asarray(subjects, dtype=int),
        "preprocessing_meta": {
            "slug": slug, "channels": list(channels), "n_channels": len(channels),
            "fmin": fmin, "fmax": fmax, "notch_freq": notch, "resample_sfreq": sfreq,
            "use_car": True, "reference": "average", "window_sec": window_s, "unit": "uV",
            "built_by": "neuroatlas data prepare",
        },
    }


def build(slug: str, raw_dir: Path, out_dir: Path, *, force: bool = False,
          subjects: Optional[Sequence[int]] = None, say: Callable[[str], None] = print
          ) -> List[Path]:
    """Write *slug*'s two files into *out_dir*; returns their paths."""
    from neuroatlas.extensions.datasets.dataio.bci import DATASET_CONFIGS
    from neuroatlas.extensions.datasets.dataio.bci_paths import (
        COGNITIVE_PICKLES,
        CONFOUND_FILTERED_TAG,
    )

    if slug not in COGNITIVE_PICKLES:
        raise SystemExit(f"error: {slug} is not a bci_cognitive dataset; these are: "
                         f"{', '.join(COGNITIVE_PICKLES)}")
    stem = COGNITIVE_PICKLES[slug][1]
    cfg = DATASET_CONFIGS[slug]
    wanted = list(subjects) if subjects is not None else list(cfg.subjects)
    outputs = {confound: out_dir / (f"{stem}_preprocessed"
                                    f"{CONFOUND_FILTERED_TAG if confound else '_'}{FORMAT_NAME}.pkl")
               for confound in (False, True)}
    if not force and all(p.is_file() for p in outputs.values()):
        say(f"{slug}: already built ({', '.join(p.name for p in outputs.values())}); "
            f"--set force=true builds them again")
        return list(outputs.values())

    if slug == "eegmat":
        one, window_s, channels = (lambda s, f: eegmat_subject(raw_dir, s, f)), 4.0, _EEGMAT_CHANNELS
    elif slug == "arithmetic_task":
        one, window_s, channels = (lambda s, f: arithmetic_subject(raw_dir, s, f)), 1.0, _ARITH_CHANNELS
    else:
        mat = _dreamer_mat(raw_dir)
        label = slug.split("_", 1)[1]
        one, window_s, channels = (lambda s, f: dreamer_subject(mat, s, f, label)), 4.0, _DREAMER_CHANNELS

    import mne

    mne.set_log_level("WARNING")
    out_dir.mkdir(parents=True, exist_ok=True)
    for confound, path in outputs.items():
        fmt = FORMATS[confound]
        t0 = time.time()
        kept, xs, ys, missing = [], [], [], []
        for s in wanted:
            got = one(s, fmt)
            if got is None:
                missing.append(s)
                continue
            kept.append(s)
            xs.append(got[0])
            ys.append(got[1])
        if not kept:
            raise SystemExit(f"error: {slug}: no recording of any subject in {raw_dir}\n"
                             f"fix: neuroatlas data download {slug}")
        tmp = path.with_suffix(".pkl.part")
        with open(tmp, "wb") as fh:
            pickle.dump(_payload(kept, xs, ys, fmt, slug, channels, window_s), fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        n = sum(len(y) for y in ys)
        say(f"{slug}: {path.name}, {len(kept)} subjects, {n:,} windows, "
            f"{fmt[0]:g}-{fmt[1]:g} Hz ({time.time() - t0:.0f}s)"
            + (f"; no recordings for subjects {', '.join(map(str, missing))}" if missing else ""))
    return list(outputs.values())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="eegmat, arithmetic_task, dreamer_valence or dreamer_arousal")
    p.add_argument("--raw-dir", required=True, help="The dataset's raw data folder.")
    p.add_argument("--output", required=True, help="The folder the two files go in.")
    p.add_argument("--force", default="false", help="true: build again over existing files.")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    build(args.dataset, Path(args.raw_dir), Path(args.output),
          force=str(args.force).lower() in ("1", "true", "yes"))


if __name__ == "__main__":
    main()
