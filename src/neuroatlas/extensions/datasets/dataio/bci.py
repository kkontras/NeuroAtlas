"""Low-level MOABB-based data loading and preprocessing for BCI datasets.

Supports PhysionetMI, Cho2017, Lee2019_MI (motor imagery) and Hinss2021
(cognitive workload) via braindecode + MOABB.
Each dataset has its own configuration (channels, events, sampling rate, etc.)
extracted from the original BCI_DG_RPA preprocessing pipeline.
"""

from __future__ import annotations

import sys as _sys
import types as _types

# ---------------------------------------------------------------------------
# Numpy 2.x → 1.x pickle compatibility shim.
# Numpy 2.x moved numpy.core → numpy._core.  Files pickled with numpy 2.x
# fail to unpickle under numpy 1.x.  Register the new module paths so
# pickle.load() finds them regardless of numpy version.
# ---------------------------------------------------------------------------
try:
    import numpy._core.numeric  # noqa: F401 — works on numpy 2.x
except (ImportError, ModuleNotFoundError):
    # numpy 1.26 has numpy._core as a real package but without numeric /
    # multiarray submodules.  Point them at numpy.core.* so pickle.load()
    # can resolve references written by numpy 2.x.
    import numpy.core.numeric as _ncn
    import numpy.core.multiarray as _ncm
    _sys.modules["numpy._core.numeric"] = _ncn
    _sys.modules["numpy._core.multiarray"] = _ncm

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mne
import numpy as np
import pandas as pd
from braindecode.datasets import BaseConcatDataset, WindowsDataset
from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess
from neuroatlas._paths import prepared_dir

# ---------------------------------------------------------------------------
# Per-dataset constants
# ---------------------------------------------------------------------------

# --- Events (class name -> integer label) ---
EVENTS_PHYSIONET_MI = {"left_hand": 0, "rest": 1, "right_hand": 2, "feet": 3, "hands": 4}
EVENTS_CHO2017 = {"left_hand": 0, "right_hand": 1}
EVENTS_LEE2019_MI = {"right_hand": 0, "left_hand": 1}
EVENTS_HINSS2021 = {"easy": 2, "medium": 3, "difficult": 4, "rest": 1}

# --- Cognitive / affective cohorts -------------------------------------------
# Not MOABB datasets: each is read from the preprocessed pickles the cohort was
# benchmarked on. Label integers are the ones actually stored, read from that
# upstream preprocessing rather than inferred -- see the note on ArithmeticTask.
EVENTS_EEGMAT = {"rest": 0, "arithmetic": 1}
# Upstream preprocessing writes rest=0, arithmetic=1, meditation/breathing=2.
# Experiment 1 recorded meditation ("M"), experiment 2 breathing ("B"); both
# map to 2, which is why the class is named for the pair.
EVENTS_ARITHMETIC_TASK = {"rest": 0, "arithmetic": 1, "meditation_breathing": 2}
# 1-5 self-report binarised at 3: <=3 low, >3 high. Not presence/absence.
EVENTS_DREAMER_VALENCE = {"low_valence": 0, "high_valence": 1}
EVENTS_DREAMER_AROUSAL = {"low_arousal": 0, "high_arousal": 1}

# --- Targets (class names kept for this benchmark) ---
TARGETS_PHYSIONET_MI = ["left_hand", "rest", "right_hand", "feet", "hands"]
TARGETS_CHO2017 = ["left_hand", "right_hand"]
TARGETS_LEE2019_MI = ["left_hand", "right_hand"]
TARGETS_HINSS2021 = ["easy", "medium", "difficult", "rest"]
TARGETS_EEGMAT = ["rest", "arithmetic"]
TARGETS_ARITHMETIC_TASK = ["rest", "arithmetic", "meditation_breathing"]
TARGETS_DREAMER_VALENCE = ["low_valence", "high_valence"]
TARGETS_DREAMER_AROUSAL = ["low_arousal", "high_arousal"]

# --- Subject lists ---
SUBJECTS_PHYSIONET_MI = [
    x for x in range(1, 110) if x not in {88, 90, 92, 100, 104, 106}
]
SUBJECTS_CHO2017 = [x for x in range(1, 53) if x not in {32, 46, 49}]
SUBJECTS_LEE2019_MI = list(range(1, 55))
SUBJECTS_HINSS2021 = list(range(1, 16))
SUBJECTS_EEGMAT = list(range(36))            # ids 0..35, one per PhysioNet subject
# The upstream preprocessing declares 52 (exp1 S01-S21 -> 1001-1021, exp2
# S02-S33 excluding S07 -> 2002-2033). The pickles hold 45: experiment 2's
# S02-S09 are absent, because the script skips any subject whose directory or
# EDF is missing from the raw tree. 45 is what the embeddings and probes were
# built from, so it is what this cohort declares.
SUBJECTS_ARITHMETIC_TASK = list(range(1001, 1022)) + list(range(2010, 2034))
SUBJECTS_DREAMER = list(range(1, 24))        # 23 subjects x 18 clips

# --- Channel lists ---
CHANNELS_PHYSIONET_MI = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
]
CHANNELS_CHO2017 = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7",
    "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7", "TP7",
    "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9",
    "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz",
    "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4",
    "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz",
    "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]
CHANNELS_LEE2019_MI = [
    "FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4",
    "CP5", "CP1", "CP2", "CP6", "FC3", "FC4",
    "C5", "C1", "C2", "C6", "CP3", "CPz", "CP4",
]
# 19 standard 10-20, as the upstream preprocessing picks them. EEGMAT and
# ArithmeticTask differ only in ordering; the set is identical.
CHANNELS_EEGMAT = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "T7", "T8", "C3", "C4",
    "P7", "P8", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
]
CHANNELS_ARITHMETIC_TASK = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]
# Emotiv EPOC, 14 channels.
CHANNELS_DREAMER = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]
CHANNELS_HINSS2021 = [
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3",
    "T7", "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2",
    "P4", "P8", "CP6", "CP2", "FCz", "C4", "T8", "FT8", "FC6", "FC2",
    "F4", "F8", "Fp2", "AF7", "AF3", "AFz", "F1", "F5", "FT7", "FC3",
    "C1", "C5", "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8",
    "P6", "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT10", "F6", "AF8",
    "AF4", "F2",
]

# --- Native sampling rates ---
SFREQ_PHYSIONET_MI = 160
SFREQ_CHO2017 = 512
SFREQ_LEE2019_MI = 1000
SFREQ_HINSS2021 = 500
SFREQ_EEGMAT = 500
SFREQ_ARITHMETIC_TASK = 256
SFREQ_DREAMER = 128

# ---------------------------------------------------------------------------
# Additional MOABB MI datasets (broad_mi_loso sweep)
# ---------------------------------------------------------------------------
# Metadata gathered via direct MOABB introspection. Each dataset uses its
# native EEG channel layout (all EEG channels, EOG/EMG excluded). 2-class L/R
# targets where both are present; else the dataset's own 2-class (e.g.
# bnci2015_001 = right_hand vs feet).

# --- Events (0-based codes matching PhysionetMI's convention:
#     left_hand=0, rest=1, right_hand=2, feet=3, tongue=4, hands=5, ...).
#     This keeps `probe_bci_loso --targets 0 2` meaning "L/R" across every
#     dataset that has both. `_filter_target_classes` is recording-level and
#     drops a recording only when NONE of its classes are kept, so TARGETS_*
#     must include every class present to avoid dropping all data. Class
#     subsetting for the probe happens at probe-time via --targets.)
EVENTS_SCHIRRMEISTER2017 = {"left_hand": 0, "rest": 1, "right_hand": 2, "feet": 3}
EVENTS_SHIN2017A = {"left_hand": 0, "right_hand": 2}
# experiment: PatientPopulation-EEG — Shin2017B mental arithmetic paradigm
EVENTS_SHIN2017B = {"subtraction": 0, "rest": 1}
EVENTS_WEIBO2014 = {
    "left_hand": 0, "rest": 1, "right_hand": 2, "feet": 3,
    "hands": 5, "left_hand_right_foot": 6, "right_hand_left_foot": 7,
}
EVENTS_BNCI2014_001 = {"left_hand": 0, "right_hand": 2, "feet": 3, "tongue": 4}
EVENTS_DREYER2023A = {"left_hand": 0, "right_hand": 2}
EVENTS_BNCI2014_004 = {"left_hand": 0, "right_hand": 2}
EVENTS_BNCI2015_001 = {"right_hand": 2, "feet": 3}
# experiment: HefmiIch2025-PatientPopulation — HEFMI-ICH MI dataset
EVENTS_HEFMIICH2025 = {"left_hand": 0, "right_hand": 1}

# --- Targets: keep ALL classes present (subset at probe-time via --targets).
TARGETS_SCHIRRMEISTER2017 = ["left_hand", "rest", "right_hand", "feet"]
TARGETS_SHIN2017A = ["left_hand", "right_hand"]
TARGETS_SHIN2017B = ["subtraction", "rest"]
TARGETS_WEIBO2014 = [
    "left_hand", "rest", "right_hand", "feet",
    "hands", "left_hand_right_foot", "right_hand_left_foot",
]
TARGETS_BNCI2014_001 = ["left_hand", "right_hand", "feet", "tongue"]
TARGETS_DREYER2023A = ["left_hand", "right_hand"]
TARGETS_BNCI2014_004 = ["left_hand", "right_hand"]
TARGETS_BNCI2015_001 = ["right_hand", "feet"]
TARGETS_HEFMIICH2025 = ["left_hand", "right_hand"]

# --- Subject lists ---
SUBJECTS_SCHIRRMEISTER2017 = list(range(1, 15))   # 14
SUBJECTS_SHIN2017A = list(range(1, 30))           # 29
SUBJECTS_SHIN2017B = list(range(1, 30))           # 29 (same subjects as Shin2017A)
SUBJECTS_WEIBO2014 = list(range(1, 11))           # 10
SUBJECTS_BNCI2014_001 = list(range(1, 10))        # 9
SUBJECTS_DREYER2023A = list(range(1, 61))         # 60
SUBJECTS_BNCI2014_004 = list(range(1, 10))        # 9
SUBJECTS_BNCI2015_001 = list(range(1, 13))        # 12
SUBJECTS_HEFMIICH2025 = list(range(1, 38))        # 37 (17 healthy + 20 ICH patients)

# --- Channel lists (EEG only, in native MOABB order) ---
CHANNELS_SCHIRRMEISTER2017 = [
    "Fp1", "Fp2", "Fpz", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "M1", "T7", "C3", "Cz", "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7",
    "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2", "AF7", "AF3", "AF4", "AF8",
    "F5", "F1", "F2", "F6", "FC3", "FCz", "FC4", "C5", "C1", "C2", "C6", "CP3",
    "CPz", "CP4", "P5", "P1", "P2", "P6", "PO5", "PO3", "PO4", "PO6", "FT7",
    "FT8", "TP7", "TP8", "PO7", "PO8", "FT9", "FT10", "TPP9h", "TPP10h", "PO9",
    "PO10", "P9", "P10", "AFF1", "AFz", "AFF2", "FFC5h", "FFC3h", "FFC4h",
    "FFC6h", "FCC5h", "FCC3h", "FCC4h", "FCC6h", "CCP5h", "CCP3h", "CCP4h",
    "CCP6h", "CPP5h", "CPP3h", "CPP4h", "CPP6h", "PPO1", "PPO2", "I1", "Iz",
    "I2", "AFp3h", "AFp4h", "AFF5h", "AFF6h", "FFT7h", "FFC1h", "FFC2h", "FFT8h",
    "FTT9h", "FTT7h", "FCC1h", "FCC2h", "FTT8h", "FTT10h", "TTP7h", "CCP1h",
    "CCP2h", "TTP8h", "TPP7h", "CPP1h", "CPP2h", "TPP8h", "PPO9h", "PPO5h",
    "PPO6h", "PPO10h", "POO9h", "POO3h", "POO4h", "POO10h", "OI1h", "OI2h",
]
CHANNELS_SHIN2017A = [
    "F7", "AFF5h", "F3", "AFp1", "AFp2", "AFF6h", "F4", "F8", "AFF1h", "AFF2h",
    "Cz", "Pz", "FCC5h", "FCC3h", "CCP5h", "CCP3h", "T7", "P7", "P3", "PPO1h",
    "POO1", "POO2", "PPO2h", "P4", "FCC4h", "FCC6h", "CCP4h", "CCP6h", "P8", "T8",
]
CHANNELS_SHIN2017B = CHANNELS_SHIN2017A  # identical 30-ch EEG layout
CHANNELS_WEIBO2014 = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4",
    "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3",
    "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2",
    "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "O1",
    "Oz", "O2",
]
CHANNELS_BNCI2014_001 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]
CHANNELS_DREYER2023A = [
    "Fz", "FCz", "Cz", "CPz", "Pz", "C1", "C3", "C5", "C2", "C4", "C6", "F4",
    "FC2", "FC4", "FC6", "CP2", "CP4", "CP6", "P4", "F3", "FC1", "FC3", "FC5",
    "CP1", "CP3", "CP5", "P3",
]
CHANNELS_BNCI2014_004 = ["C3", "Cz", "C4"]
CHANNELS_BNCI2015_001 = [
    "FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CPz",
    "CP4",
]
# experiment: HefmiIch2025-PatientPopulation — 32-ch EEG (biosemi32 layout)
CHANNELS_HEFMIICH2025 = [
    "FC1", "AF3", "AF4", "CP1", "CP2", "CP6", "Cz", "C3", "C4", "T7",
    "T8", "FC2", "FC5", "FC6", "Pz", "CP5", "PO3", "PO4", "Oz", "Fp2",
    "Fp1", "Fz", "F3", "F4", "F7", "F8", "P3", "P4", "P7", "P8", "O1", "O2",
]

# --- Native sampling rates ---
SFREQ_SCHIRRMEISTER2017 = 500.0
SFREQ_SHIN2017A = 200.0
SFREQ_SHIN2017B = 200.0
SFREQ_WEIBO2014 = 200.0
SFREQ_BNCI2014_001 = 250.0
SFREQ_DREYER2023A = 512.0
SFREQ_BNCI2014_004 = 250.0
SFREQ_BNCI2015_001 = 512.0
SFREQ_HEFMIICH2025 = 256.0


# ---------------------------------------------------------------------------
# ERP / P300 datasets
# ---------------------------------------------------------------------------

EVENTS_ERP = {"Target": 0, "NonTarget": 1}
TARGETS_ERP = ["Target", "NonTarget"]

# --- BI2013a (BrainInvaders 2013a, France) ---
SUBJECTS_BI2013A = list(range(1, 25))  # 24
CHANNELS_BI2013A = [
    "Fp1", "Fp2", "F5", "AFz", "F6", "T7", "Cz", "T8",
    "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
]
SFREQ_BI2013A = 512.0

# --- BI2014a (BrainInvaders 2014a, France) ---
SUBJECTS_BI2014A = list(range(1, 65))  # 64
CHANNELS_BI2014A = [
    "Fp1", "Fp2", "F3", "AFz", "F4", "T7", "Cz", "T8",
    "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
]
SFREQ_BI2014A = 512.0

# --- BI2015a (BrainInvaders 2015a, France) ---
SUBJECTS_BI2015A = list(range(1, 44))  # 43
CHANNELS_BI2015A = [
    "Fp1", "Fp2", "AFz", "F7", "F3", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
    "Pz", "P4", "P8", "PO7", "O1", "Oz", "O2", "PO8", "PO9", "PO10",
]
SFREQ_BI2015A = 512.0

# --- BNCI2014_008 (P300 speller, Austria) ---
SUBJECTS_BNCI2014_008 = list(range(1, 9))  # 8
CHANNELS_BNCI2014_008 = ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]
SFREQ_BNCI2014_008 = 256.0

# --- BNCI2014_009 (P300 speller, Austria) ---
SUBJECTS_BNCI2014_009 = list(range(1, 11))  # 10
CHANNELS_BNCI2014_009 = [
    "Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8",
    "F3", "F4", "FCz", "C3", "C4", "CP3", "CPz", "CP4",
]
SFREQ_BNCI2014_009 = 256.0

# --- EPFLP300 (Switzerland) ---
SUBJECTS_EPFLP300 = [1, 2, 3, 4, 6, 7, 8, 9]  # 8 (subject 5 missing)
CHANNELS_EPFLP300 = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5", "P7",
    "P3", "Pz", "PO3", "O1", "Oz", "O2", "PO4", "P4", "P8", "CP6", "CP2",
    "C4", "T8", "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
]
SFREQ_EPFLP300 = 2048.0

# --- Lee2019_ERP (South Korea) ---
SUBJECTS_LEE2019_ERP = list(range(1, 55))  # 54
CHANNELS_LEE2019_ERP = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10", "FC3",
    "FC4", "C5", "C1", "C2", "C6", "CP3", "CPz", "CP4", "P1", "P2", "POz",
    "FT9", "FTT9h", "TTP7h", "TP7", "TPP9h", "FT10", "FTT10h", "TPP8h",
    "TP8", "TPP10h", "F9", "F10", "AF7", "AF3", "AF4", "AF8", "PO3", "PO4",
]
SFREQ_LEE2019_ERP = 1000.0


@dataclass(frozen=True)
class BCIDatasetConfig:
    """All per-dataset knobs needed for loading and preprocessing."""

    moabb_name: str
    subjects: Sequence[int]
    channels: Sequence[str]
    events: Dict[str, int]
    targets: Sequence[str]
    native_sfreq: float
    resample_sfreq: float = 100.0
    fmin: float = 4.0
    fmax: float = 40.0
    notch_freq: float = 60.0
    use_car: bool = True
    tmin: float = 1.0
    tmax: float = 4.0
    trial_duration: float = 4.0  # forced annotation duration (seconds)


DATASET_CONFIGS: Dict[str, BCIDatasetConfig] = {
    # --- Cognitive / affective cohorts. Read from preprocessed pickles; the
    # moabb_name is a label only, since load_preprocessed_dataset resolves the
    # file before any MOABB path is considered.
    "eegmat": BCIDatasetConfig(
        moabb_name="EEGMat", subjects=SUBJECTS_EEGMAT, channels=CHANNELS_EEGMAT,
        events=EVENTS_EEGMAT, targets=TARGETS_EEGMAT, native_sfreq=SFREQ_EEGMAT,
        resample_sfreq=128.0, fmin=0.1, fmax=64.0, notch_freq=50.0, use_car=True,
        tmin=0.0, tmax=4.0, trial_duration=4.0,
    ),
    "arithmetic_task": BCIDatasetConfig(
        moabb_name="ArithmeticTask", subjects=SUBJECTS_ARITHMETIC_TASK,
        channels=CHANNELS_ARITHMETIC_TASK, events=EVENTS_ARITHMETIC_TASK,
        targets=TARGETS_ARITHMETIC_TASK, native_sfreq=SFREQ_ARITHMETIC_TASK,
        resample_sfreq=128.0, fmin=0.1, fmax=64.0, notch_freq=50.0, use_car=True,
        tmin=0.0, tmax=1.0, trial_duration=1.0,
    ),
    "dreamer_valence": BCIDatasetConfig(
        moabb_name="DREAMER", subjects=SUBJECTS_DREAMER, channels=CHANNELS_DREAMER,
        events=EVENTS_DREAMER_VALENCE, targets=TARGETS_DREAMER_VALENCE,
        native_sfreq=SFREQ_DREAMER, resample_sfreq=128.0, fmin=0.1, fmax=64.0,
        notch_freq=50.0, use_car=True, tmin=0.0, tmax=4.0, trial_duration=4.0,
    ),
    "dreamer_arousal": BCIDatasetConfig(
        moabb_name="DREAMER", subjects=SUBJECTS_DREAMER, channels=CHANNELS_DREAMER,
        events=EVENTS_DREAMER_AROUSAL, targets=TARGETS_DREAMER_AROUSAL,
        native_sfreq=SFREQ_DREAMER, resample_sfreq=128.0, fmin=0.1, fmax=64.0,
        notch_freq=50.0, use_car=True, tmin=0.0, tmax=4.0, trial_duration=4.0,
    ),
    "physionet_mi": BCIDatasetConfig(
        moabb_name="PhysionetMI",
        subjects=SUBJECTS_PHYSIONET_MI,
        channels=CHANNELS_PHYSIONET_MI,
        events=EVENTS_PHYSIONET_MI,
        targets=TARGETS_PHYSIONET_MI,
        native_sfreq=SFREQ_PHYSIONET_MI,
        fmin=4.0,
        fmax=40.0,
        use_car=True,
        tmin=0.0,
        tmax=4.0,
    ),
    "cho2017": BCIDatasetConfig(
        moabb_name="Cho2017",
        subjects=SUBJECTS_CHO2017,
        channels=CHANNELS_CHO2017,
        events=EVENTS_CHO2017,
        targets=TARGETS_CHO2017,
        native_sfreq=SFREQ_CHO2017,
        fmin=4.0,
        fmax=40.0,
        use_car=True,
        tmin=0.0,
        tmax=4.0,
    ),
    "cho2017_labram": BCIDatasetConfig(
        moabb_name="Cho2017",
        subjects=SUBJECTS_CHO2017,
        channels=CHANNELS_CHO2017,
        events=EVENTS_CHO2017,
        targets=TARGETS_CHO2017,
        native_sfreq=SFREQ_CHO2017,
        resample_sfreq=200.0,
        fmin=0.1,
        fmax=75.0,
        notch_freq=50.0,
        use_car=True,
        tmin=0.0,
        tmax=4.0,
    ),
    "lee2019_mi": BCIDatasetConfig(
        moabb_name="Lee2019_MI",
        subjects=SUBJECTS_LEE2019_MI,
        channels=CHANNELS_LEE2019_MI,
        events=EVENTS_LEE2019_MI,
        targets=TARGETS_LEE2019_MI,
        native_sfreq=SFREQ_LEE2019_MI,
    ),
    "lee2019_mi_labram": BCIDatasetConfig(
        moabb_name="Lee2019_MI",
        subjects=SUBJECTS_LEE2019_MI,
        channels=CHANNELS_LEE2019_MI,
        events=EVENTS_LEE2019_MI,
        targets=TARGETS_LEE2019_MI,
        native_sfreq=SFREQ_LEE2019_MI,
        resample_sfreq=200.0,
        fmin=0.1,
        fmax=75.0,
        notch_freq=50.0,
        use_car=True,
        tmin=1.0,
        tmax=4.0,
    ),
    "hinss2021": BCIDatasetConfig(
        moabb_name="Hinss2021",
        subjects=SUBJECTS_HINSS2021,
        channels=CHANNELS_HINSS2021,
        events=EVENTS_HINSS2021,
        targets=TARGETS_HINSS2021,
        native_sfreq=SFREQ_HINSS2021,
        resample_sfreq=250.0,
        fmin=1.0,
        fmax=40.0,
        notch_freq=60.0,
        use_car=True,
        tmin=1.0,
        tmax=4.0,
    ),
    # experiment: MOABB-Extension — Track B configs for PhysionetMI, Cho2017, Lee2019_MI.
    # 4-40 Hz bandpass at model-native sfreq, per-dataset notch.
    "physionet_mi_mi_band": BCIDatasetConfig(
        moabb_name="PhysionetMI",
        subjects=SUBJECTS_PHYSIONET_MI,
        channels=CHANNELS_PHYSIONET_MI,
        events=EVENTS_PHYSIONET_MI,
        targets=TARGETS_PHYSIONET_MI,
        native_sfreq=SFREQ_PHYSIONET_MI,
        resample_sfreq=200.0,
        fmin=4.0, fmax=40.0, notch_freq=60.0,  # US → 60 Hz
        use_car=True, tmin=0.0, tmax=4.0,
    ),
    "cho2017_mi_band": BCIDatasetConfig(
        moabb_name="Cho2017",
        subjects=SUBJECTS_CHO2017,
        channels=CHANNELS_CHO2017,
        events=EVENTS_CHO2017,
        targets=TARGETS_CHO2017,
        native_sfreq=SFREQ_CHO2017,
        resample_sfreq=200.0,
        fmin=4.0, fmax=40.0, notch_freq=60.0,  # South Korea → 60 Hz
        use_car=True, tmin=0.0, tmax=4.0,
    ),
    "lee2019_mi_mi_band": BCIDatasetConfig(
        moabb_name="Lee2019_MI",
        subjects=SUBJECTS_LEE2019_MI,
        channels=CHANNELS_LEE2019_MI,
        events=EVENTS_LEE2019_MI,
        targets=TARGETS_LEE2019_MI,
        native_sfreq=SFREQ_LEE2019_MI,
        resample_sfreq=200.0,
        fmin=4.0, fmax=40.0, notch_freq=60.0,  # South Korea → 60 Hz
        use_car=True, tmin=1.0, tmax=4.0,
    ),
    # experiment: MOABB-CorrectPreprocessing — tracks C (= A + tmin=1s) and
    # D (= B + tmin=1s) remove the first second of each trial to eliminate
    # cue-onset eye-gaze artefacts. For Lee2019_MI track A/B already used
    # tmin=1.0, so tracks C/D produce identical output — separate files are
    # still written to keep output paths non-overlapping.
    "physionet_mi_labram_confound_controlled": BCIDatasetConfig(
        moabb_name="PhysionetMI",
        subjects=SUBJECTS_PHYSIONET_MI,
        channels=CHANNELS_PHYSIONET_MI,
        events=EVENTS_PHYSIONET_MI,
        targets=TARGETS_PHYSIONET_MI,
        native_sfreq=SFREQ_PHYSIONET_MI,
        resample_sfreq=200.0,
        fmin=0.1, fmax=75.0, notch_freq=50.0,  # LaBraM fixed 50 Hz notch
        use_car=True, tmin=1.0, tmax=4.0,
    ),
    "physionet_mi_mi_band_confound_controlled": BCIDatasetConfig(
        moabb_name="PhysionetMI",
        subjects=SUBJECTS_PHYSIONET_MI,
        channels=CHANNELS_PHYSIONET_MI,
        events=EVENTS_PHYSIONET_MI,
        targets=TARGETS_PHYSIONET_MI,
        native_sfreq=SFREQ_PHYSIONET_MI,
        resample_sfreq=200.0,
        fmin=4.0, fmax=40.0, notch_freq=60.0,
        use_car=True, tmin=1.0, tmax=4.0,
    ),
    "cho2017_labram_confound_controlled": BCIDatasetConfig(
        moabb_name="Cho2017",
        subjects=SUBJECTS_CHO2017,
        channels=CHANNELS_CHO2017,
        events=EVENTS_CHO2017,
        targets=TARGETS_CHO2017,
        native_sfreq=SFREQ_CHO2017,
        resample_sfreq=200.0,
        fmin=0.1, fmax=75.0, notch_freq=50.0,
        use_car=True, tmin=1.0, tmax=4.0,
    ),
    "cho2017_mi_band_confound_controlled": BCIDatasetConfig(
        moabb_name="Cho2017",
        subjects=SUBJECTS_CHO2017,
        channels=CHANNELS_CHO2017,
        events=EVENTS_CHO2017,
        targets=TARGETS_CHO2017,
        native_sfreq=SFREQ_CHO2017,
        resample_sfreq=200.0,
        fmin=4.0, fmax=40.0, notch_freq=60.0,
        use_car=True, tmin=1.0, tmax=4.0,
    ),
    "lee2019_mi_labram_confound_controlled": BCIDatasetConfig(
        moabb_name="Lee2019_MI",
        subjects=SUBJECTS_LEE2019_MI,
        channels=CHANNELS_LEE2019_MI,
        events=EVENTS_LEE2019_MI,
        targets=TARGETS_LEE2019_MI,
        native_sfreq=SFREQ_LEE2019_MI,
        resample_sfreq=200.0,
        fmin=0.1, fmax=75.0, notch_freq=50.0,
        use_car=True, tmin=1.0, tmax=4.0,
    ),
    "lee2019_mi_mi_band_confound_controlled": BCIDatasetConfig(
        moabb_name="Lee2019_MI",
        subjects=SUBJECTS_LEE2019_MI,
        channels=CHANNELS_LEE2019_MI,
        events=EVENTS_LEE2019_MI,
        targets=TARGETS_LEE2019_MI,
        native_sfreq=SFREQ_LEE2019_MI,
        resample_sfreq=200.0,
        fmin=4.0, fmax=40.0, notch_freq=60.0,
        use_car=True, tmin=1.0, tmax=4.0,
    ),
    # experiment: BENDR-Alignment
    # BENDR was pretrained on wideband clinical EEG (TUEG corpus). Feeding
    # the default 4-40 Hz motor-imagery bandpass strips frequencies BENDR
    # relies on and collapses its downstream accuracy to chance (50.2 %).
    # Wideband 0.5-70 Hz at 256 Hz → 60.4 % linear LOSO on 19 subjects
    # (+10 pp over 4-40 Hz); matches the paper's MMI result envelope.
    # experiment: LaBraM-Alignment
    # PhysionetMI preprocessed to match LaBraM pretraining parameters:
    # 200 Hz resample, 0.1-75 Hz bandpass, 50 Hz notch, CAR, tmin=0, tmax=4.
    "physionet_mi_labram": BCIDatasetConfig(
        moabb_name="PhysionetMI",
        subjects=SUBJECTS_PHYSIONET_MI,
        channels=CHANNELS_PHYSIONET_MI,
        events=EVENTS_PHYSIONET_MI,
        targets=TARGETS_PHYSIONET_MI,
        native_sfreq=SFREQ_PHYSIONET_MI,
        resample_sfreq=200.0,  # experiment: LaBraM-Alignment — match pretrain rate
        fmin=0.1,              # experiment: LaBraM-Alignment — match pretrain filter
        fmax=75.0,             # experiment: LaBraM-Alignment — match pretrain filter
        notch_freq=50.0,       # experiment: LaBraM-Alignment — match pretrain notch
        use_car=True,
        tmin=0.0,
        tmax=4.0,
    ),
    "physionet_mi_eegpt": BCIDatasetConfig(
        moabb_name="PhysionetMI",
        subjects=SUBJECTS_PHYSIONET_MI,
        channels=CHANNELS_PHYSIONET_MI,
        events=EVENTS_PHYSIONET_MI,
        targets=TARGETS_PHYSIONET_MI,
        native_sfreq=SFREQ_PHYSIONET_MI,
        resample_sfreq=256.0,
        fmin=0.5,
        fmax=70.0,
        notch_freq=60.0,
        use_car=True,
        tmin=0.0,
        tmax=4.0,
    ),
    "cho2017_eegpt": BCIDatasetConfig(
        moabb_name="Cho2017",
        subjects=SUBJECTS_CHO2017,
        channels=CHANNELS_CHO2017,
        events=EVENTS_CHO2017,
        targets=TARGETS_CHO2017,
        native_sfreq=SFREQ_CHO2017,
        resample_sfreq=256.0,
        fmin=0.5,
        fmax=70.0,
        notch_freq=60.0,
        use_car=True,
        tmin=0.0,
        tmax=4.0,
    ),
    "lee2019_mi_eegpt": BCIDatasetConfig(
        moabb_name="Lee2019_MI",
        subjects=SUBJECTS_LEE2019_MI,
        channels=CHANNELS_LEE2019_MI,
        events=EVENTS_LEE2019_MI,
        targets=TARGETS_LEE2019_MI,
        native_sfreq=SFREQ_LEE2019_MI,
        resample_sfreq=256.0,
        fmin=0.5,
        fmax=70.0,
        notch_freq=60.0,
        use_car=True,
        tmin=1.0,
        tmax=4.0,
    ),
    # --- CBraMod variants (200 Hz, 0.3–75 Hz, 50 Hz notch) ---
    "physionet_mi_cbramod": BCIDatasetConfig(
        moabb_name="PhysionetMI",
        subjects=SUBJECTS_PHYSIONET_MI,
        channels=CHANNELS_PHYSIONET_MI,
        events=EVENTS_PHYSIONET_MI,
        targets=TARGETS_PHYSIONET_MI,
        native_sfreq=SFREQ_PHYSIONET_MI,
        resample_sfreq=200.0,
        fmin=0.3, fmax=75.0, notch_freq=50.0,
        use_car=True, tmin=0.0, tmax=4.0,
    ),
    "cho2017_cbramod": BCIDatasetConfig(
        moabb_name="Cho2017",
        subjects=SUBJECTS_CHO2017,
        channels=CHANNELS_CHO2017,
        events=EVENTS_CHO2017,
        targets=TARGETS_CHO2017,
        native_sfreq=SFREQ_CHO2017,
        resample_sfreq=200.0,
        fmin=0.3, fmax=75.0, notch_freq=50.0,
        use_car=True, tmin=0.0, tmax=4.0,
    ),
    "lee2019_mi_cbramod": BCIDatasetConfig(
        moabb_name="Lee2019_MI",
        subjects=SUBJECTS_LEE2019_MI,
        channels=CHANNELS_LEE2019_MI,
        events=EVENTS_LEE2019_MI,
        targets=TARGETS_LEE2019_MI,
        native_sfreq=SFREQ_LEE2019_MI,
        resample_sfreq=200.0,
        fmin=0.3, fmax=75.0, notch_freq=50.0,
        use_car=True, tmin=1.0, tmax=4.0,
    ),
    # --- REVE variants (200 Hz, 0.5–75 Hz, per-dataset notch) ---
    "physionet_mi_reve": BCIDatasetConfig(
        moabb_name="PhysionetMI",
        subjects=SUBJECTS_PHYSIONET_MI,
        channels=CHANNELS_PHYSIONET_MI,
        events=EVENTS_PHYSIONET_MI,
        targets=TARGETS_PHYSIONET_MI,
        native_sfreq=SFREQ_PHYSIONET_MI,
        resample_sfreq=200.0,
        fmin=0.5, fmax=75.0, notch_freq=60.0,
        use_car=True, tmin=0.0, tmax=4.0,
    ),
    "cho2017_reve": BCIDatasetConfig(
        moabb_name="Cho2017",
        subjects=SUBJECTS_CHO2017,
        channels=CHANNELS_CHO2017,
        events=EVENTS_CHO2017,
        targets=TARGETS_CHO2017,
        native_sfreq=SFREQ_CHO2017,
        resample_sfreq=200.0,
        fmin=0.5, fmax=75.0, notch_freq=60.0,
        use_car=True, tmin=0.0, tmax=4.0,
    ),
    "lee2019_mi_reve": BCIDatasetConfig(
        moabb_name="Lee2019_MI",
        subjects=SUBJECTS_LEE2019_MI,
        channels=CHANNELS_LEE2019_MI,
        events=EVENTS_LEE2019_MI,
        targets=TARGETS_LEE2019_MI,
        native_sfreq=SFREQ_LEE2019_MI,
        resample_sfreq=200.0,
        fmin=0.5, fmax=75.0, notch_freq=60.0,
        use_car=True, tmin=1.0, tmax=4.0,
    ),
}


# ---------------------------------------------------------------------------
# Broad MI sweep — 7 MOABB datasets over two orthogonal preprocessing axes.
# Every entry is a standard 4-second MI trial with the dataset's native EEG
# channel layout; the axes decide what the backbone actually sees.
#
#   alignment         the passband and sample rate a backbone was pretrained
#                     with, so its input looks like its training data
#   confound_control  whether the first second of each trial is dropped
#
# The second axis is the one that decides how a number can be read. An MI
# trial opens with a visual cue, so its first second carries the evoked
# response and the eye movement towards the cue -- a probe can ride that
# instead of the motor imagery. Holding the band to the 4-40 Hz motor range
# and starting at tmin=1.0 s removes that confound; a backbone's own wideband
# passband over the whole trial leaves it in. Both are built, because the
# pair is the comparison: the gap between them is how much of a score came
# from the cue rather than from the imagery.
# ---------------------------------------------------------------------------

#: alignment -> (fmin, fmax, resample_sfreq, notch); notch None = per-dataset
MI_ALIGNMENTS: Dict[str, Tuple[float, float, float, Optional[float]]] = {
    "default": (4.0, 40.0, 100.0, None),   # shared MI baseline
    "eegpt":   (0.5, 70.0, 256.0, None),
    "labram":  (0.1, 75.0, 200.0, 50.0),   # also NeuroLM
    "cbramod": (0.3, 75.0, 200.0, 50.0),
    "reve":    (0.5, 75.0, 200.0, None),
    "mi_band": (4.0, 40.0, 200.0, None),   # the 4-40 motor band at FM sfreq
}

#: seconds dropped from the head of each trial when confound control is on
CONFOUND_CONTROL_TMIN = 1.0

#: appended to a config key when confound control is on
CONFOUND_CONTROLLED = "_confound_controlled"

def _mi_cfg(
    *,
    moabb_name: str,
    subjects: Sequence[int],
    channels: Sequence[str],
    events: Dict[str, int],
    targets: Sequence[str],
    native_sfreq: float,
    notch_freq: float,
    alignment: str = "default",
    confound_control: bool = False,
) -> BCIDatasetConfig:
    """Build a BCIDatasetConfig for a broad-sweep MOABB MI dataset.

    *alignment* picks a row of :data:`MI_ALIGNMENTS`; *confound_control*
    starts each trial at :data:`CONFOUND_CONTROL_TMIN` instead of 0.0.
    """
    try:
        fmin, fmax, resample, notch = MI_ALIGNMENTS[alignment]
    except KeyError:
        raise ValueError(
            f"unknown alignment {alignment!r}; known: {sorted(MI_ALIGNMENTS)}"
        ) from None
    return BCIDatasetConfig(
        moabb_name=moabb_name,
        subjects=subjects,
        channels=channels,
        events=events,
        targets=targets,
        native_sfreq=native_sfreq,
        resample_sfreq=resample,
        fmin=fmin,
        fmax=fmax,
        notch_freq=notch_freq if notch is None else notch,
        use_car=True,
        tmin=CONFOUND_CONTROL_TMIN if confound_control else 0.0,
        tmax=4.0,
    )


def mi_config_key(slug: str, alignment: str = "default",
                  confound_control: bool = False) -> str:
    """The DATASET_CONFIGS key for one point on the two axes."""
    key = slug if alignment == "default" else f"{slug}_{alignment}"
    return key + CONFOUND_CONTROLLED if confound_control else key


_BROAD_MI_SPECS = [
    # (slug_root, moabb_name, subjects, channels, events, targets, native_sfreq, notch)
    ("schirrmeister2017", "Schirrmeister2017", SUBJECTS_SCHIRRMEISTER2017,
     CHANNELS_SCHIRRMEISTER2017, EVENTS_SCHIRRMEISTER2017,
     TARGETS_SCHIRRMEISTER2017, SFREQ_SCHIRRMEISTER2017, 50.0),  # Germany
    ("shin2017a", "Shin2017A", SUBJECTS_SHIN2017A,
     CHANNELS_SHIN2017A, EVENTS_SHIN2017A, TARGETS_SHIN2017A,
     SFREQ_SHIN2017A, 60.0),  # Korea
    ("weibo2014", "Weibo2014", SUBJECTS_WEIBO2014,
     CHANNELS_WEIBO2014, EVENTS_WEIBO2014, TARGETS_WEIBO2014,
     SFREQ_WEIBO2014, 50.0),  # China
    ("bnci2014_001", "BNCI2014_001", SUBJECTS_BNCI2014_001,
     CHANNELS_BNCI2014_001, EVENTS_BNCI2014_001, TARGETS_BNCI2014_001,
     SFREQ_BNCI2014_001, 50.0),  # Austria
    ("dreyer2023a", "Dreyer2023A", SUBJECTS_DREYER2023A,
     CHANNELS_DREYER2023A, EVENTS_DREYER2023A, TARGETS_DREYER2023A,
     SFREQ_DREYER2023A, 50.0),  # France
    ("bnci2014_004", "BNCI2014_004", SUBJECTS_BNCI2014_004,
     CHANNELS_BNCI2014_004, EVENTS_BNCI2014_004, TARGETS_BNCI2014_004,
     SFREQ_BNCI2014_004, 50.0),  # Austria
    ("bnci2015_001", "BNCI2015_001", SUBJECTS_BNCI2015_001,
     CHANNELS_BNCI2015_001, EVENTS_BNCI2015_001, TARGETS_BNCI2015_001,
     SFREQ_BNCI2015_001, 50.0),  # Austria
]

for _slug, _name, _subj, _chs, _ev, _tgt, _sf, _notch in _BROAD_MI_SPECS:
    for _align in MI_ALIGNMENTS:
        for _cc in (False, True):
            DATASET_CONFIGS[mi_config_key(_slug, _align, _cc)] = _mi_cfg(
                moabb_name=_name,
                subjects=_subj,
                channels=_chs,
                events=_ev,
                targets=_tgt,
                native_sfreq=_sf,
                notch_freq=_notch,
                alignment=_align,
                confound_control=_cc,
            )


# ---------------------------------------------------------------------------
# experiment: PatientPopulation-EEG — Shin2017B (mental arithmetic)
# Shin2017B is NOT in _BROAD_MI_SPECS because:
#   1. Different event mapping (subtraction/rest, not left_hand/right_hand)
#   2. Track D uses user-specified 0.5-50 Hz (not standard 4-40 Hz)
# Track A uses the standard FM-aligned spec (identical to other datasets).
# ---------------------------------------------------------------------------

# FM-aligned: labram 0.1-75 Hz @ 200 Hz
DATASET_CONFIGS["shin2017b_labram"] = BCIDatasetConfig(
    moabb_name="Shin2017B",
    subjects=SUBJECTS_SHIN2017B,
    channels=CHANNELS_SHIN2017B,
    events=EVENTS_SHIN2017B,
    targets=TARGETS_SHIN2017B,
    native_sfreq=SFREQ_SHIN2017B,
    resample_sfreq=200.0,
    fmin=0.1, fmax=75.0, notch_freq=50.0,  # LaBraM fixed 50 Hz
    use_car=True, tmin=0.0, tmax=4.0,
)

# Track D — user-specified 0.5-50 Hz bandpass + tmin=1s for mental arithmetic
DATASET_CONFIGS["shin2017b_eegpt_confound_controlled"] = BCIDatasetConfig(
    moabb_name="Shin2017B",
    subjects=SUBJECTS_SHIN2017B,
    channels=CHANNELS_SHIN2017B,
    events=EVENTS_SHIN2017B,
    targets=TARGETS_SHIN2017B,
    native_sfreq=SFREQ_SHIN2017B,
    resample_sfreq=256.0,
    fmin=0.5, fmax=50.0, notch_freq=60.0,  # 0.5-50 Hz user-specified
    use_car=True, tmin=1.0, tmax=4.0,
)
DATASET_CONFIGS["shin2017b_labram_confound_controlled"] = BCIDatasetConfig(
    moabb_name="Shin2017B",
    subjects=SUBJECTS_SHIN2017B,
    channels=CHANNELS_SHIN2017B,
    events=EVENTS_SHIN2017B,
    targets=TARGETS_SHIN2017B,
    native_sfreq=SFREQ_SHIN2017B,
    resample_sfreq=200.0,
    fmin=0.5, fmax=50.0, notch_freq=60.0,  # 0.5-50 Hz user-specified
    use_car=True, tmin=1.0, tmax=4.0,
)

# experiment: HefmiIch2025-PatientPopulation — Track A (FM-aligned)
DATASET_CONFIGS["hefmiich2025_eegpt"] = BCIDatasetConfig(
    moabb_name="HefmiIch2025",
    subjects=SUBJECTS_HEFMIICH2025,
    channels=CHANNELS_HEFMIICH2025,
    events=EVENTS_HEFMIICH2025,
    targets=TARGETS_HEFMIICH2025,
    native_sfreq=SFREQ_HEFMIICH2025,
    resample_sfreq=256.0,
    fmin=0.5, fmax=70.0, notch_freq=50.0,
    use_car=True, tmin=0.0, tmax=10.0,
)
DATASET_CONFIGS["hefmiich2025_labram"] = BCIDatasetConfig(
    moabb_name="HefmiIch2025",
    subjects=SUBJECTS_HEFMIICH2025,
    channels=CHANNELS_HEFMIICH2025,
    events=EVENTS_HEFMIICH2025,
    targets=TARGETS_HEFMIICH2025,
    native_sfreq=SFREQ_HEFMIICH2025,
    resample_sfreq=200.0,
    fmin=0.1, fmax=75.0, notch_freq=50.0,
    use_car=True, tmin=0.0, tmax=10.0,
)
# experiment: HefmiIch2025-PatientPopulation — Track D (4-40 Hz + notch, tmin=1s)
DATASET_CONFIGS["hefmiich2025_eegpt_confound_controlled"] = BCIDatasetConfig(
    moabb_name="HefmiIch2025",
    subjects=SUBJECTS_HEFMIICH2025,
    channels=CHANNELS_HEFMIICH2025,
    events=EVENTS_HEFMIICH2025,
    targets=TARGETS_HEFMIICH2025,
    native_sfreq=SFREQ_HEFMIICH2025,
    resample_sfreq=256.0,
    fmin=4.0, fmax=40.0, notch_freq=50.0,
    use_car=True, tmin=1.0, tmax=10.0,
)
DATASET_CONFIGS["hefmiich2025_labram_confound_controlled"] = BCIDatasetConfig(
    moabb_name="HefmiIch2025",
    subjects=SUBJECTS_HEFMIICH2025,
    channels=CHANNELS_HEFMIICH2025,
    events=EVENTS_HEFMIICH2025,
    targets=TARGETS_HEFMIICH2025,
    native_sfreq=SFREQ_HEFMIICH2025,
    resample_sfreq=200.0,
    fmin=4.0, fmax=40.0, notch_freq=50.0,
    use_car=True, tmin=1.0, tmax=10.0,
)


# ---------------------------------------------------------------------------
# ERP / P300 dataset configs — Track A (FM-aligned preprocessing)
# ---------------------------------------------------------------------------

_ERP_SPECS = [
    # (slug_root, moabb_name, subjects, channels, events, targets, native_sfreq, notch, trial_dur)
    ("bi2013a", "BI2013a", SUBJECTS_BI2013A,
     CHANNELS_BI2013A, EVENTS_ERP, TARGETS_ERP,
     SFREQ_BI2013A, 50.0, 1.0),  # France
    ("bi2014a", "BI2014a", SUBJECTS_BI2014A,
     CHANNELS_BI2014A, EVENTS_ERP, TARGETS_ERP,
     SFREQ_BI2014A, 50.0, 1.0),  # France
    ("bi2015a", "BI2015a", SUBJECTS_BI2015A,
     CHANNELS_BI2015A, EVENTS_ERP, TARGETS_ERP,
     SFREQ_BI2015A, 50.0, 1.0),  # France
    ("bnci2014_008", "BNCI2014_008", SUBJECTS_BNCI2014_008,
     CHANNELS_BNCI2014_008, EVENTS_ERP, TARGETS_ERP,
     SFREQ_BNCI2014_008, 50.0, 1.0),  # Austria
    ("bnci2014_009", "BNCI2014_009", SUBJECTS_BNCI2014_009,
     CHANNELS_BNCI2014_009, EVENTS_ERP, TARGETS_ERP,
     SFREQ_BNCI2014_009, 50.0, 0.8),  # Austria, 0.8s interval
    ("epflp300", "EPFLP300", SUBJECTS_EPFLP300,
     CHANNELS_EPFLP300, EVENTS_ERP, TARGETS_ERP,
     SFREQ_EPFLP300, 50.0, 1.0),  # Switzerland
    ("lee2019_erp", "Lee2019_ERP", SUBJECTS_LEE2019_ERP,
     CHANNELS_LEE2019_ERP, EVENTS_ERP, TARGETS_ERP,
     SFREQ_LEE2019_ERP, 60.0, 1.0),  # South Korea
]


def _erp_cfg(
    moabb_name, subjects, channels, events, targets, native_sfreq,
    notch_freq, trial_dur, variant,
):
    """Build a BCIDatasetConfig for an ERP dataset + variant."""
    if variant in ("eegpt"):
        fmin, fmax, resample, notch = 0.5, 70.0, 256.0, notch_freq
    elif variant == "labram":
        fmin, fmax, resample, notch = 0.1, 75.0, 200.0, 50.0
    elif variant == "cbramod":
        fmin, fmax, resample, notch = 0.3, 75.0, 200.0, 50.0
    elif variant == "reve":
        fmin, fmax, resample, notch = 0.5, 75.0, 200.0, notch_freq
    else:
        raise ValueError(f"unknown ERP variant {variant!r}")
    return BCIDatasetConfig(
        moabb_name=moabb_name,
        subjects=subjects,
        channels=channels,
        events=events,
        targets=targets,
        native_sfreq=native_sfreq,
        resample_sfreq=resample,
        fmin=fmin,
        fmax=fmax,
        notch_freq=notch,
        use_car=True,
        tmin=0.0,
        tmax=trial_dur,
        trial_duration=trial_dur,
    )


for _slug, _name, _subj, _chs, _ev, _tgt, _sf, _notch, _tdur in _ERP_SPECS:
    for _var in ("labram", "eegpt", "cbramod", "reve"):
        _key = f"{_slug}_{_var}"
        DATASET_CONFIGS[_key] = _erp_cfg(
            moabb_name=_name,
            subjects=_subj,
            channels=_chs,
            events=_ev,
            targets=_tgt,
            native_sfreq=_sf,
            notch_freq=_notch,
            trial_dur=_tdur,
            variant=_var,
        )


# ---------------------------------------------------------------------------
# SSVEP datasets — experiment: MOABB-SSVEP-Extension
# 3 MOABB SSVEP datasets × (labram / eegpt / cbramod / reve / ssvep_band)
# Track A = FM-aligned (same as MI Track A)
# Track B = 1-50 Hz at model-native sfreq + per-dataset notch
# ---------------------------------------------------------------------------

# --- Events (SSVEP frequency label → integer code, 0-based) ---
EVENTS_LIU2020BETA = {
    "8.6": 0, "8.8": 1, "9": 2, "9.2": 3, "9.4": 4, "9.6": 5, "9.8": 6,
    "10": 7, "10.2": 8, "10.4": 9, "10.6": 10, "10.8": 11, "11": 12,
    "11.2": 13, "11.4": 14, "11.6": 15, "11.8": 16, "12": 17, "12.2": 18,
    "12.4": 19, "12.6": 20, "12.8": 21, "13": 22, "13.2": 23, "13.4": 24,
    "13.6": 25, "13.8": 26, "14": 27, "14.2": 28, "14.4": 29, "14.6": 30,
    "14.8": 31, "15": 32, "15.2": 33, "15.4": 34, "15.6": 35, "15.8": 36,
    "8": 37, "8.2": 38, "8.4": 39,
}
EVENTS_WANG2016 = {
    "8": 0, "9": 1, "10": 2, "11": 3, "12": 4, "13": 5, "14": 6, "15": 7,
    "8.2": 8, "9.2": 9, "10.2": 10, "11.2": 11, "12.2": 12, "13.2": 13,
    "14.2": 14, "15.2": 15, "8.4": 16, "9.4": 17, "10.4": 18, "11.4": 19,
    "12.4": 20, "13.4": 21, "14.4": 22, "15.4": 23, "8.6": 24, "9.6": 25,
    "10.6": 26, "11.6": 27, "12.6": 28, "13.6": 29, "14.6": 30, "15.6": 31,
    "8.8": 32, "9.8": 33, "10.8": 34, "11.8": 35, "12.8": 36, "13.8": 37,
    "14.8": 38, "15.8": 39,
}
EVENTS_NAKANISHI2015 = {
    "9.25": 0, "11.25": 1, "13.25": 2, "9.75": 3, "11.75": 4, "13.75": 5,
    "10.25": 6, "12.25": 7, "14.25": 8, "10.75": 9, "12.75": 10, "14.75": 11,
}

# --- Targets: keep ALL classes present (subset at probe-time via --targets).
TARGETS_LIU2020BETA = list(EVENTS_LIU2020BETA.keys())
TARGETS_WANG2016 = list(EVENTS_WANG2016.keys())
TARGETS_NAKANISHI2015 = list(EVENTS_NAKANISHI2015.keys())

# --- Subject lists ---
SUBJECTS_LIU2020BETA = list(range(1, 71))     # 70
SUBJECTS_WANG2016 = list(range(1, 35))         # 34
SUBJECTS_NAKANISHI2015 = list(range(1, 10))    # 9

# --- Channel lists (EEG only, Tsinghua 64-ch 10-10 layout for Liu/Wang) ---
CHANNELS_LIU2020BETA = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
    "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8",
    "CB1", "O1", "Oz", "O2", "CB2",
]
CHANNELS_WANG2016 = CHANNELS_LIU2020BETA  # identical Tsinghua 64-ch layout
CHANNELS_NAKANISHI2015 = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]

# --- Native sampling rates ---
SFREQ_LIU2020BETA = 250.0
SFREQ_WANG2016 = 250.0
SFREQ_NAKANISHI2015 = 256.0


def _ssvep_cfg(
    *,
    moabb_name: str,
    subjects: Sequence[int],
    channels: Sequence[str],
    events: Dict[str, int],
    targets: Sequence[str],
    native_sfreq: float,
    notch_freq: float,
    tmin: float,
    tmax: float,
    trial_duration: float,
    variant: str,
) -> BCIDatasetConfig:
    """Build a BCIDatasetConfig for a broad-sweep MOABB SSVEP dataset.

    variant:
        "labram"        → 0.1–75 Hz @ 200 Hz  (NeuroLM / LaBraM, Track A)
        "cbramod"       → 0.3–75 Hz @ 200 Hz  (CBraMod alignment, 50 Hz notch)
        "reve"          → 0.5–75 Hz @ 200 Hz  (REVE alignment, per-dataset notch)
        → 1–50 Hz @ 256 Hz    (SSVEP-standard, Track B)
        "ssvep_band" → 1–50 Hz @ 200 Hz    (SSVEP-standard, Track B)
    """
    if variant in ("eegpt"):
        fmin, fmax, resample, notch = 0.5, 70.0, 256.0, notch_freq
    elif variant == "labram":
        fmin, fmax, resample, notch = 0.1, 75.0, 200.0, 50.0
    elif variant == "cbramod":
        fmin, fmax, resample, notch = 0.3, 75.0, 200.0, 50.0
    elif variant == "reve":
        fmin, fmax, resample, notch = 0.5, 75.0, 200.0, notch_freq
    elif variant == "ssvep_band":
        fmin, fmax, resample, notch = 1.0, 50.0, 200.0, notch_freq
    else:
        raise ValueError(f"unknown SSVEP variant {variant!r}")
    return BCIDatasetConfig(
        moabb_name=moabb_name,
        subjects=subjects,
        channels=channels,
        events=events,
        targets=targets,
        native_sfreq=native_sfreq,
        resample_sfreq=resample,
        fmin=fmin,
        fmax=fmax,
        notch_freq=notch,
        use_car=True,
        tmin=tmin,
        tmax=tmax,
        trial_duration=trial_duration,
    )


# --- Lee2019_SSVEP (South Korea, 62 ch, 4-class) ---
EVENTS_LEE2019_SSVEP = {"12.0": 0, "8.57": 1, "6.67": 2, "5.45": 3}
TARGETS_LEE2019_SSVEP = list(EVENTS_LEE2019_SSVEP.keys())
SUBJECTS_LEE2019_SSVEP = list(range(1, 55))  # 54
CHANNELS_LEE2019_SSVEP = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", "TP10",
    "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10",
    "FC3", "FC4", "C5", "C1", "C2", "C6", "CP3", "CPz", "CP4", "P1", "P2",
    "POz", "FT9", "FTT9h", "TTP7h", "TP7", "TPP9h", "FT10", "FTT10h",
    "TPP8h", "TP8", "TPP10h", "F9", "F10", "AF7", "AF3", "AF4", "AF8",
    "PO3", "PO4",
]
SFREQ_LEE2019_SSVEP = 1000.0

# --- Liu2022EldBETA (China, 64 ch Tsinghua, 9-class) ---
EVENTS_LIU2022ELDBETA = {
    "8": 0, "9.5": 1, "11": 2, "8.5": 3, "10": 4,
    "11.5": 5, "9": 6, "10.5": 7, "12": 8,
}
TARGETS_LIU2022ELDBETA = list(EVENTS_LIU2022ELDBETA.keys())
SUBJECTS_LIU2022ELDBETA = list(range(1, 101))  # 100
# Tsinghua layout — raw uses uppercase FP1/FPZ/FZ/FCZ; must match exactly
CHANNELS_LIU2022ELDBETA = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
    "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "Pz", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8",
    "CB1", "O1", "Oz", "O2", "CB2",
]
SFREQ_LIU2022ELDBETA = 1000.0

# --- GuttmannFlury2025_SSVEP (Austria, 66 ch, 4-class) ---
EVENTS_GUTTMANNFLURY2025_SSVEP = {"10.0": 0, "11.0": 1, "12.0": 2, "13.0": 3}
TARGETS_GUTTMANNFLURY2025_SSVEP = list(EVENTS_GUTTMANNFLURY2025_SSVEP.keys())
SUBJECTS_GUTTMANNFLURY2025_SSVEP = list(range(1, 32))  # 31
CHANNELS_GUTTMANNFLURY2025_SSVEP = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "O1", "Oz", "O2",
    "CB1", "CB2", "M1", "M2", "HEO", "Trig",
]
SFREQ_GUTTMANNFLURY2025_SSVEP = 1000.0

# --- Han2024Fatigue (China, 64 ch Tsinghua, 32-class) ---
EVENTS_HAN2024FATIGUE = {
    "8": 0, "8.5": 1, "9": 2, "9.5": 3, "10": 4, "10.5": 5,
    "11": 6, "11.5": 7, "12": 8, "12.5": 9, "13": 10, "13.5": 11,
    "14": 12, "14.5": 13, "15": 14, "15.5": 15,
    "25.5": 16, "26": 17, "26.5": 18, "27": 19, "27.5": 20, "28": 21,
    "28.5": 22, "29": 23, "29.5": 24, "30": 25, "30.5": 26, "31": 27,
    "31.5": 28, "32": 29, "32.5": 30, "33": 31,
}
TARGETS_HAN2024FATIGUE = list(EVENTS_HAN2024FATIGUE.keys())
SUBJECTS_HAN2024FATIGUE = list(range(1, 25))  # 24
CHANNELS_HAN2024FATIGUE = CHANNELS_LIU2020BETA  # identical Tsinghua 64-ch
SFREQ_HAN2024FATIGUE = 1000.0

# --- Kim2025BetaRange (South Korea, 31 ch centro-parieto-occipital, 40-class) ---
EVENTS_KIM2025BETARANGE = {
    "14": 0, "15": 1, "16": 2, "17": 3, "18": 4, "19": 5, "20": 6, "21": 7,
    "14.2": 8, "15.2": 9, "16.2": 10, "17.2": 11, "18.2": 12, "19.2": 13,
    "20.2": 14, "21.2": 15, "14.4": 16, "15.4": 17, "16.4": 18, "17.4": 19,
    "18.4": 20, "19.4": 21, "20.4": 22, "21.4": 23, "14.6": 24, "15.6": 25,
    "16.6": 26, "17.6": 27, "18.6": 28, "19.6": 29, "20.6": 30, "21.6": 31,
    "14.8": 32, "15.8": 33, "16.8": 34, "17.8": 35, "18.8": 36, "19.8": 37,
    "20.8": 38, "21.8": 39,
}
TARGETS_KIM2025BETARANGE = list(EVENTS_KIM2025BETARANGE.keys())
SUBJECTS_KIM2025BETARANGE = list(range(1, 41))  # 40
CHANNELS_KIM2025BETARANGE = [
    "C3", "Cz", "TP7", "CP5", "CP3", "CP1", "CPz", "P9", "P7", "P5",
    "P3", "P1", "Pz", "PO7", "PO3", "POz", "O1", "Oz", "O2", "PO8",
    "PO4", "P10", "P8", "P6", "P4", "P2", "TP8", "CP6", "CP4", "CP2", "C4",
]
SFREQ_KIM2025BETARANGE = 1024.0


# (slug_root, moabb_name, subjects, channels, events, targets,
#  native_sfreq, notch, tmin, tmax, trial_duration)
_BROAD_SSVEP_SPECS = [
    ("liu2020beta", "Liu2020BETA", SUBJECTS_LIU2020BETA,
     CHANNELS_LIU2020BETA, EVENTS_LIU2020BETA, TARGETS_LIU2020BETA,
     SFREQ_LIU2020BETA, 50.0, 0.0, 3.0, 3.0),    # China, 40-class
    ("wang2016", "Wang2016", SUBJECTS_WANG2016,
     CHANNELS_WANG2016, EVENTS_WANG2016, TARGETS_WANG2016,
     SFREQ_WANG2016, 50.0, 0.5, 5.5, 5.0),        # China, 40-class (annot dur=5.0s)
    ("nakanishi2015", "Nakanishi2015", SUBJECTS_NAKANISHI2015,
     CHANNELS_NAKANISHI2015, EVENTS_NAKANISHI2015, TARGETS_NAKANISHI2015,
     SFREQ_NAKANISHI2015, 60.0, 0.15, 4.3, 4.15),  # USA, 12-class (annot dur=4.15s)
    ("lee2019_ssvep", "Lee2019_SSVEP", SUBJECTS_LEE2019_SSVEP,
     CHANNELS_LEE2019_SSVEP, EVENTS_LEE2019_SSVEP, TARGETS_LEE2019_SSVEP,
     SFREQ_LEE2019_SSVEP, 60.0, 0.0, 4.0, 4.0),   # Korea, 4-class (annot dur=4.0s)
    ("liu2022eldbeta", "Liu2022EldBETA", SUBJECTS_LIU2022ELDBETA,
     CHANNELS_LIU2022ELDBETA, EVENTS_LIU2022ELDBETA, TARGETS_LIU2022ELDBETA,
     SFREQ_LIU2022ELDBETA, 50.0, 0.0, 6.0, 6.0),  # China, 9-class (annot dur=6.0s)
    ("guttmannflury2025_ssvep", "GuttmannFlury2025_SSVEP", SUBJECTS_GUTTMANNFLURY2025_SSVEP,
     CHANNELS_GUTTMANNFLURY2025_SSVEP, EVENTS_GUTTMANNFLURY2025_SSVEP, TARGETS_GUTTMANNFLURY2025_SSVEP,
     SFREQ_GUTTMANNFLURY2025_SSVEP, 50.0, 0.0, 5.0, 5.0),  # Austria, 4-class (annot dur=5.0s)
    ("han2024fatigue", "Han2024Fatigue", SUBJECTS_HAN2024FATIGUE,
     CHANNELS_HAN2024FATIGUE, EVENTS_HAN2024FATIGUE, TARGETS_HAN2024FATIGUE,
     SFREQ_HAN2024FATIGUE, 50.0, 0.14, 2.14, 2.0),  # China, 32-class (annot dur=2.0s)
    ("kim2025betarange", "Kim2025BetaRange", SUBJECTS_KIM2025BETARANGE,
     CHANNELS_KIM2025BETARANGE, EVENTS_KIM2025BETARANGE, TARGETS_KIM2025BETARANGE,
     SFREQ_KIM2025BETARANGE, 60.0, 0.0, 5.0, 5.0),  # Korea, 40-class beta (annot dur=5.0s)
]

for _slug, _name, _subj, _chs, _ev, _tgt, _sf, _notch, _tmin, _tmax, _tdur in _BROAD_SSVEP_SPECS:
    for _var in ("labram", "eegpt", "cbramod", "reve", "ssvep_band"):
        _key = f"{_slug}_{_var}"
        DATASET_CONFIGS[_key] = _ssvep_cfg(
            moabb_name=_name,
            subjects=_subj,
            channels=_chs,
            events=_ev,
            targets=_tgt,
            native_sfreq=_sf,
            notch_freq=_notch,
            tmin=_tmin,
            tmax=_tmax,
            trial_duration=_tdur,
            variant=_var,
        )


# ---------------------------------------------------------------------------
# Paths to already-preprocessed data on disk.
# Each dataset has a search list: repo-local paths are tried first, then the
# original external paths from the BCI_DG_RPA pipeline.  The first existing
# file wins.  Users can always override via the ``preprocessed_path`` argument.
# ---------------------------------------------------------------------------

from pathlib import Path as _Path

# Built by `neuroatlas data prepare`: <cache root>/prepared/<Name>/... (it was
# <checkout>/data/preprocessed, inside the source tree). The table below is
# written against a placeholder for that folder, and every read resolves it
# against the cache root as it is *then*: it used to be resolved at import,
# so a process that changed the cache root afterwards (a test, an API user,
# `config set` in the same session) still looked in the old one.
_PREPARED_TOKEN = "@prepared@"
_DATA_DIR = _Path(_PREPARED_TOKEN)


def _resolve_prepared(path: str) -> str:
    if path.startswith(_PREPARED_TOKEN):
        return str(prepared_dir()) + path[len(_PREPARED_TOKEN):]
    return path


class _SearchPaths(dict):
    """``{slug: [candidate path, ...]}``; reads resolve the prepared folder
    against the current cache root (see ``_PREPARED_TOKEN``)."""

    def __getitem__(self, key):
        return [_resolve_prepared(p) for p in super().__getitem__(key)]

    def get(self, key, default=None):
        return self[key] if key in self else default

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return [(k, self[k]) for k in self]


class _FirstPaths:
    """``{slug: first candidate}`` over :data:`PREPROCESSED_SEARCH_PATHS`."""

    def __init__(self, search):
        self._search = search

    def __getitem__(self, key):
        return self._search[key][0]

    def get(self, key, default=None):
        return self[key] if key in self._search else default

    def __contains__(self, key):
        return key in self._search

    def __iter__(self):
        return iter(self._search)

    def __len__(self):
        return len(self._search)

    def keys(self):
        return self._search.keys()

    def items(self):
        return [(k, self[k]) for k in self._search]

    def values(self):
        return [self[k] for k in self._search]


PREPROCESSED_SEARCH_PATHS: Dict[str, List[str]] = _SearchPaths({
    # Cognitive/affective cohorts: <Dataset>_preprocessed_<variant>.pkl, the
    # naming their preprocess_*.py writes. steegformer first only because it is
    # the smallest; any variant carries the same signal.
    "eegmat": [
        str(_DATA_DIR / "EEGMat" / "EEGMat_preprocessed_steegformer.pkl"),
        str(_DATA_DIR / "EEGMat" / "EEGMat_preprocessed_labram.pkl"),
        str(_DATA_DIR / "EEGMat" / "EEGMat_preprocessed_bendr.pkl"),
    ],
    "arithmetic_task": [
        str(_DATA_DIR / "ArithmeticTask" / "ArithmeticTask_preprocessed_steegformer.pkl"),
        str(_DATA_DIR / "ArithmeticTask" / "ArithmeticTask_preprocessed_labram.pkl"),
        str(_DATA_DIR / "ArithmeticTask" / "ArithmeticTask_preprocessed_bendr.pkl"),
    ],
    "dreamer_valence": [
        str(_DATA_DIR / "DREAMER" / "DREAMER_valence_preprocessed_steegformer.pkl"),
        str(_DATA_DIR / "DREAMER" / "DREAMER_valence_preprocessed_labram.pkl"),
        str(_DATA_DIR / "DREAMER" / "DREAMER_valence_preprocessed_bendr.pkl"),
    ],
    "dreamer_arousal": [
        str(_DATA_DIR / "DREAMER" / "DREAMER_arousal_preprocessed_steegformer.pkl"),
        str(_DATA_DIR / "DREAMER" / "DREAMER_arousal_preprocessed_labram.pkl"),
        str(_DATA_DIR / "DREAMER" / "DREAMER_arousal_preprocessed_bendr.pkl"),
    ],
    # PhysionetMI: repo-local .pkl exists → try it first
    "physionet_mi": [
        str(_DATA_DIR / "PhysionetMI" / "physionetMI_preprocessed.pkl"),
        str(_DATA_DIR / "PhysionetMI" / "physionetMI_preprocessed.mat"),
        "${EEG_DATA_ROOT}/PhysionetMI_data/physionetMI_preprocessed.mat",
    ],
    # experiment: LaBraM-Alignment — PhysionetMI preprocessed with LaBraM params
    # (200 Hz, 0.1-75 Hz, 50 Hz notch)
    "physionet_mi_labram": [
        str(_DATA_DIR / "PhysionetMI" / "physionetMI_preprocessed_labram.pkl"),
    ],
    "physionet_mi_eegpt": [
        str(_DATA_DIR / "PhysionetMI" / "physionetMI_preprocessed_eegpt.pkl"),
        # Historical filename. That file was written with 0.5-70 Hz @ 256 Hz,
        # which is exactly the eegpt alignment, so it is still the right input
        # -- only the wrapper it was originally made for is gone.
        str(_DATA_DIR / "PhysionetMI" / "PhysionetMI_preprocessed_bendr.pkl"),
    ],
    # Cho2017: repo-local .pkl exists → try it first
    "cho2017": [
        str(_DATA_DIR / "Cho2017" / "cho2017_preprocessed.pkl"),
        "${EEG_DATA_ROOT}/PhysionetMI_data/Cho2017_preprocessed.mat",
        str(_DATA_DIR / "Cho2017" / "Cho2017_preprocessed.mat"),
    ],
    "cho2017_labram": [
        str(_DATA_DIR / "Cho2017" / "Cho2017_preprocessed_labram.pkl"),
    ],
    "cho2017_eegpt": [
        str(_DATA_DIR / "Cho2017" / "Cho2017_preprocessed_eegpt.pkl"),
        str(_DATA_DIR / "Cho2017" / "Cho2017_preprocessed_bendr.pkl"),
    ],
    # Lee2019_MI: only the external .mat exists for now
    "lee2019_mi": [
        "${EEG_DATA_ROOT}/PhysionetMI_data/Lee2019_MI_preprocessed_1s.mat",
        str(_DATA_DIR / "Lee2019_MI" / "Lee2019_MI_preprocessed.pkl"),
        str(_DATA_DIR / "Lee2019_MI" / "Lee2019_MI_preprocessed.mat"),
    ],
    "lee2019_mi_labram": [
        str(_DATA_DIR / "Lee2019_MI" / "Lee2019_MI_preprocessed_labram.pkl"),
    ],
    "lee2019_mi_eegpt": [
        str(_DATA_DIR / "Lee2019_MI" / "Lee2019_MI_preprocessed_eegpt.pkl"),
        str(_DATA_DIR / "Lee2019_MI" / "Lee2019_MI_preprocessed_bendr.pkl"),
    ],
    # CBraMod/REVE core MI variants — fall back to labram preprocessed (same 200 Hz)
    "physionet_mi_cbramod": [
        str(_DATA_DIR / "PhysionetMI" / "physionetMI_preprocessed_cbramod.pkl"),
        str(_DATA_DIR / "PhysionetMI" / "physionetMI_preprocessed_labram.pkl"),
    ],
    "physionet_mi_reve": [
        str(_DATA_DIR / "PhysionetMI" / "physionetMI_preprocessed_reve.pkl"),
        str(_DATA_DIR / "PhysionetMI" / "physionetMI_preprocessed_labram.pkl"),
    ],
    "cho2017_cbramod": [
        str(_DATA_DIR / "Cho2017" / "Cho2017_preprocessed_cbramod.pkl"),
        str(_DATA_DIR / "Cho2017" / "Cho2017_preprocessed_labram.pkl"),
    ],
    "cho2017_reve": [
        str(_DATA_DIR / "Cho2017" / "Cho2017_preprocessed_reve.pkl"),
        str(_DATA_DIR / "Cho2017" / "Cho2017_preprocessed_labram.pkl"),
    ],
    "lee2019_mi_cbramod": [
        str(_DATA_DIR / "Lee2019_MI" / "Lee2019_MI_preprocessed_cbramod.pkl"),
        str(_DATA_DIR / "Lee2019_MI" / "Lee2019_MI_preprocessed_labram.pkl"),
    ],
    "lee2019_mi_reve": [
        str(_DATA_DIR / "Lee2019_MI" / "Lee2019_MI_preprocessed_reve.pkl"),
        str(_DATA_DIR / "Lee2019_MI" / "Lee2019_MI_preprocessed_labram.pkl"),
    ],
    # Hinss2021: only the external .pkl exists for now
    "hinss2021": [
        "${EEG_DATA_ROOT}/PhysionetMI_data/Hinss2021/Hinss2021_preprocessed_2s.pkl",
        str(_DATA_DIR / "Hinss2021" / "Hinss2021_preprocessed.pkl"),
        str(_DATA_DIR / "Hinss2021" / "Hinss2021_preprocessed_2s.pkl"),
    ],
})

# Broad MI sweep: auto-register repo-local search paths for each (slug, variant).
# cbramod/reve/eegpt fall back to labram-preprocessed files (same sfreq).
_FALLBACK_VARIANTS = {"cbramod": "labram", "reve": "labram"}
for _slug, _name, *_ in _BROAD_MI_SPECS:
    for _align in MI_ALIGNMENTS:
        for _cc in (False, True):
            _key = mi_config_key(_slug, _align, _cc)
            # The file is named after the key's variant part, so the search
            # paths and DATASET_CONFIGS cannot drift apart.
            _tail = _key[len(_slug):]
            _paths = [str(_DATA_DIR / _name / f"{_name}_preprocessed{_tail}.pkl")]
            if _align in _FALLBACK_VARIANTS:
                _fb = mi_config_key(_slug, _FALLBACK_VARIANTS[_align], _cc)[len(_slug):]
                _paths.append(str(_DATA_DIR / _name / f"{_name}_preprocessed{_fb}.pkl"))
            PREPROCESSED_SEARCH_PATHS[_key] = _paths

# The three core datasets are not in _BROAD_MI_SPECS, so their alignment x
# confound-control search paths are registered here.
for _slug, _name in [
    ("physionet_mi", "PhysionetMI"),
    ("cho2017", "Cho2017"),
    ("lee2019_mi", "Lee2019_MI"),
]:
    for _align in ("labram", "mi_band"):
        for _cc in (False, True):
            _key = mi_config_key(_slug, _align, _cc)
            PREPROCESSED_SEARCH_PATHS[_key] = [
                str(_DATA_DIR / _name / f"{_name}_preprocessed{_key[len(_slug):]}.pkl"),
            ]

# experiment: PatientPopulation-EEG — Shin2017B search paths
for _var in ("labram", "labram_confound_controlled"):
    PREPROCESSED_SEARCH_PATHS[f"shin2017b_{_var}"] = [
        str(_DATA_DIR / "Shin2017B" / f"Shin2017B_preprocessed_{_var}.pkl"),
    ]

# experiment: HefmiIch2025-PatientPopulation — search paths under anonstorage root
_HEFMI_ROOT = "${EEG_DATA_ROOT}/anonuser/mne_data_patients"
PREPROCESSED_SEARCH_PATHS["hefmiich2025_eegpt"] = [
    _HEFMI_ROOT + "/HefmiIch2025-EEGonly-TrackA-MI/hefmiich2025_preprocessed_tracka_bendr.pkl",
]
PREPROCESSED_SEARCH_PATHS["hefmiich2025_labram"] = [
    _HEFMI_ROOT + "/HefmiIch2025-EEGonly-TrackA-MI/hefmiich2025_preprocessed_tracka_labram.pkl",
]
PREPROCESSED_SEARCH_PATHS["hefmiich2025_eegpt_confound_controlled"] = [
    _HEFMI_ROOT + "/HefmiIch2025-EEGonly-TrackD-MI/hefmiich2025_preprocessed_trackd_bendr.pkl",
]
PREPROCESSED_SEARCH_PATHS["hefmiich2025_labram_confound_controlled"] = [
    _HEFMI_ROOT + "/HefmiIch2025-EEGonly-TrackD-MI/hefmiich2025_preprocessed_trackd_labram.pkl",
]

# ERP dataset search paths — auto-register for each (slug, variant).
_ERP_DIR_NAMES = {
    "bi2013a": "BI2013a", "bi2014a": "BI2014a", "bi2015a": "BI2015a",
    "bnci2014_008": "BNCI2014_008", "bnci2014_009": "BNCI2014_009",
    "epflp300": "EPFLP300", "lee2019_erp": "Lee2019_ERP",
}
for _slug, _dir_name in _ERP_DIR_NAMES.items():
    for _var in ("labram", "eegpt", "cbramod", "reve"):
        _key = f"{_slug}_{_var}"
        _paths = [str(_DATA_DIR / _dir_name / f"{_dir_name}_preprocessed_{_var}.pkl")]
        if _var in _FALLBACK_VARIANTS:
            _fb = _FALLBACK_VARIANTS[_var]
            _paths.append(str(_DATA_DIR / _dir_name / f"{_dir_name}_preprocessed_{_fb}.pkl"))
        PREPROCESSED_SEARCH_PATHS[_key] = _paths

# experiment: MOABB-SSVEP-Extension — search paths for SSVEP datasets.
_SSVEP_DIR_NAMES = {
    "liu2020beta": "Liu2020BETA",
    "wang2016": "Wang2016",
    "nakanishi2015": "Nakanishi2015",
    "lee2019_ssvep": "Lee2019_SSVEP",
    "liu2022eldbeta": "Liu2022EldBETA",
    "guttmannflury2025_ssvep": "GuttmannFlury2025_SSVEP",
    "han2024fatigue": "Han2024Fatigue",
    "kim2025betarange": "Kim2025BetaRange",
}
for _slug, _dir_name in _SSVEP_DIR_NAMES.items():
    for _var in ("labram", "eegpt", "cbramod", "reve", "ssvep_band"):
        _key = f"{_slug}_{_var}"
        _paths = [str(_DATA_DIR / _dir_name / f"{_dir_name}_preprocessed_{_var}.pkl")]
        if _var in _FALLBACK_VARIANTS:
            _fb = _FALLBACK_VARIANTS[_var]
            _paths.append(str(_DATA_DIR / _dir_name / f"{_dir_name}_preprocessed_{_fb}.pkl"))
        PREPROCESSED_SEARCH_PATHS[_key] = _paths

# Legacy alias — first path in the search list (for code that reads this
# directly), resolved on read like the search list itself.
PREPROCESSED_PATHS: Dict[str, str] = _FirstPaths(PREPROCESSED_SEARCH_PATHS)


def _load_preprocessed_mat(path: str) -> Dict[str, Any]:
    """Load a preprocessed .mat file and normalise its keys.

    Returns a dict with at least ``data_raw``, ``subject_name``,
    ``condition``, ``subject_name_vectors``.
    """
    import os, pickle
    from scipy.io import loadmat

    if path.endswith(".pkl"):
        with open(path, "rb") as fh:
            dataset = pickle.load(fh)
    else:
        dataset = loadmat(path)

    # clean scipy metadata keys
    for k in ["__header__", "__version__", "__globals__"]:
        dataset.pop(k, None)

    # normalise aliases used across different .mat exports.
    # Some .mat files have both "condition" (with None) and "labels_condition"
    # (with actual data).  Always prefer labels_condition / labels_group.
    if "labels_condition" in dataset:
        dataset["condition"] = dataset.pop("labels_condition")
    if "labels_group" in dataset:
        dataset["group"] = dataset.pop("labels_group")
    if "subject_id" in dataset and "subject_name" not in dataset:
        dataset["subject_name"] = dataset.pop("subject_id")

    return dataset


def _resolve_preprocessed_path(
    dataset_slug: str,
    preprocessed_path: Optional[str] = None,
) -> Optional[str]:
    """Return the first existing preprocessed file for *dataset_slug*.

    Search order:
      1. Explicit *preprocessed_path* if given.
      2. Each candidate in ``PREPROCESSED_SEARCH_PATHS[dataset_slug]``, first
         match wins (repo-local paths vs external paths, see list above).

    Returns ``None`` when no file is found.
    """
    import os

    if preprocessed_path is not None:
        return preprocessed_path if os.path.isfile(preprocessed_path) else None
    for candidate in PREPROCESSED_SEARCH_PATHS.get(dataset_slug, []):
        if os.path.isfile(candidate):
            return candidate
    return None


def load_preprocessed_dataset(
    dataset_slug: str,
    preprocessed_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Try to load an already-preprocessed .mat / .pkl.

    Returns ``None`` when the file does not exist (caller should fall back to
    MOABB-based loading).
    """
    path = _resolve_preprocessed_path(dataset_slug, preprocessed_path)
    if path is None:
        return None
    return _load_preprocessed_mat(path)


# ---------------------------------------------------------------------------
# Windowing — uses the patched version from windows_mine.py (NOT braindecode's
# built-in) which forces 4-second trial durations for MI datasets.
# ---------------------------------------------------------------------------

def _create_windows_from_events(dataset, trial_start_offset_samples, trial_stop_offset_samples, mapping=None, forced_trial_duration=4.0):
    """Wrapper using the patched windowing from windows_mine."""
    from neuroatlas.extensions.datasets.dataio.bci_windowing import (
        create_windows_from_events,
    )

    return create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        mapping=mapping,
        preload=True,
        forced_trial_duration=forced_trial_duration,
    )


def _volts_to_microvolts(x):
    """Convert raw EEG from volts to microvolts (picklable for joblib)."""
    return x * 1e6


# ---------------------------------------------------------------------------
# Core loading / preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(
    cfg: BCIDatasetConfig,
    subject_ids: Optional[Sequence[int]] = None,
    n_jobs: int = 1,
) -> BaseConcatDataset:
    """Download (if needed), preprocess, and window a MOABB BCI dataset.

    Parameters
    ----------
    cfg : BCIDatasetConfig
        Dataset-specific configuration.
    subject_ids : sequence of int, optional
        Subset of subjects to load.  Defaults to all subjects in *cfg*.
    n_jobs : int
        Number of parallel workers for preprocessing.

    Returns
    -------
    BaseConcatDataset
        A braindecode windowed dataset ready for iteration.
    """
    subjects = list(subject_ids) if subject_ids is not None else list(cfg.subjects)
    from neuroatlas.extensions.datasets.dataio.moabb_loader import (
        moabb_dataset_arg,
        no_download_when_offline,
    )

    with no_download_when_offline(cfg.moabb_name.lower()):
        dataset = MOABBDataset(dataset_name=moabb_dataset_arg(cfg.moabb_name),
                               subject_ids=subjects)

    # --- Preprocessing pipeline ---
    notch_freqs = np.arange(cfg.notch_freq, cfg.native_sfreq / 2, cfg.notch_freq)
    preprocessors = [
        Preprocessor(_volts_to_microvolts),  # V -> uV
        Preprocessor("pick_channels", ch_names=list(cfg.channels), ordered=True),
        Preprocessor("notch_filter", freqs=notch_freqs),
        Preprocessor("filter", l_freq=cfg.fmin, h_freq=cfg.fmax),
    ]
    if cfg.use_car:
        preprocessors.append(
            Preprocessor("set_eeg_reference", ref_channels="average", ch_type="eeg")
        )
    preprocessors.append(Preprocessor("resample", sfreq=cfg.resample_sfreq))

    preprocess(dataset, preprocessors, n_jobs=n_jobs)

    # --- Windowing ---
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    trial_start_offset_samples = int(cfg.tmin * sfreq)
    trial_stop_offset_samples = 0
    windows_dataset = _create_windows_from_events(
        dataset, trial_start_offset_samples, trial_stop_offset_samples,
        mapping=cfg.events,
        forced_trial_duration=cfg.trial_duration,
    )

    # --- Filter to keep only target classes ---
    windows_dataset = _filter_target_classes(windows_dataset, cfg)

    return windows_dataset


def load_subjects_on_the_fly(
    dataset_slug: str,
    subject_ids: Optional[Sequence[int]] = None,
    n_jobs: int = 1,
    skip_missing_raw: bool = True,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    resample_sfreq: Optional[float] = None,
    use_car: Optional[bool] = None,
):
    """Preprocess BCI subjects on the fly from MOABB cache.

    Returns ``(subject_ids, data_per_subject, labels_per_subject)`` in the exact
    same format as ``probe_bci_loso.load_dataset_subjects`` — so diagnostic
    scripts can transparently fall back to on-the-fly preprocessing when the
    preprocessed .pkl does not exist yet.

    Each subject is processed one at a time so RAM stays bounded. Subjects
    whose raw .edf files are not already in the MNE/MOABB cache are skipped
    (when ``skip_missing_raw=True``) rather than triggering an interactive
    download — useful while a separate preprocessing job is still downloading.

    Parameters
    ----------
    dataset_slug : {"physionet_mi", "physionet_mi_labram", "cho2017", ...}
        Must be a key of ``DATASET_CONFIGS``.
    subject_ids : sequence of int, optional
        Subset of subjects to load. Defaults to all subjects in the config.
    n_jobs : int
        Parallel workers for braindecode preprocess (per subject).
    skip_missing_raw : bool
        If True, skip subjects whose raw MOABB data isn't cached yet.
    """
    cfg = DATASET_CONFIGS[dataset_slug]

    # Optional overrides — useful for FM-specific preprocessing (e.g., feeding
    # BENDR wideband EEG instead of MI-specific 4-40 Hz bandpass).
    overrides = {k: v for k, v in {
        "fmin": fmin, "fmax": fmax,
        "resample_sfreq": resample_sfreq, "use_car": use_car,
    }.items() if v is not None}
    if overrides:
        from dataclasses import replace as _dc_replace
        cfg = _dc_replace(cfg, **overrides)

    requested = list(subject_ids) if subject_ids is not None else list(cfg.subjects)

    if skip_missing_raw and cfg.moabb_name == "PhysionetMI":
        requested = _filter_cached_physionet_subjects(requested)

    out_ids: list = []
    out_data: list = []
    out_labels: list = []

    for sid in requested:
        try:
            windows = load_and_preprocess(cfg, subject_ids=[sid], n_jobs=n_jobs)
        except Exception as exc:
            # Any error (missing raw, MOABB quirk) → skip this subject.
            print(f"  skip subject {sid}: {type(exc).__name__}: {exc}")
            continue

        # Stack all trials across runs for this subject.
        trials, labels = [], []
        for ds in windows.datasets:
            for j in range(len(ds)):
                x, y, _ = ds[j]
                trials.append(np.asarray(x, dtype=np.float32))
                labels.append(int(y))
        if not trials:
            continue
        out_ids.append(int(sid))
        out_data.append(np.stack(trials, axis=0))
        out_labels.append(np.asarray(labels, dtype=int))

    return out_ids, out_data, out_labels


def _filter_cached_physionet_subjects(requested):
    """Return the subset of *requested* whose MNE-eegbci raw data is already cached.

    A subject is considered cached if at least one of the imagery runs
    (4, 8, 12) exists as an .edf file under the MNE cache directory.
    """
    from pathlib import Path as _P
    import os

    # Follow mne.datasets.eegbci.load_data()'s cache layout.
    # Consult in priority order: env var → MNE config file → legacy home default.
    # MNE itself uses `mne.get_config("MNE_DATA")` (which also reads the env var)
    # so we match that lookup here. Checking the env var first keeps the function
    # cheap when a process-level override is in place.
    try:
        import mne
        _mne_config = mne.get_config("MNE_DATA")
    except Exception:
        _mne_config = None
    candidates = [
        os.environ.get("MNE_DATA"),
        _mne_config,
        str(_P.home() / "mne_data"),
    ]
    for base in candidates:
        if not base:
            continue
        root = _P(base) / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
        if root.is_dir():
            break
    else:
        return requested  # no cache found → let MOABB handle it

    cached = []
    for sid in requested:
        subj_dir = root / f"S{sid:03d}"
        if not subj_dir.is_dir():
            continue
        # At least one MI-imagery run .edf present (runs 4, 8, 12)?
        if any((subj_dir / f"S{sid:03d}R{run:02d}.edf").is_file() for run in (4, 8, 12)):
            cached.append(sid)
    return cached


def _filter_target_classes(
    windows_dataset: BaseConcatDataset, cfg: BCIDatasetConfig
) -> BaseConcatDataset:
    """Remove recording-level datasets that contain only non-target classes."""
    target_codes = set(cfg.events[t] for t in cfg.targets if t in cfg.events)
    keep_indices = []
    for idx, ds in enumerate(windows_dataset.datasets):
        codes_in_ds = set(ds.windows.metadata.target.values)
        if codes_in_ds.issubset(target_codes):
            keep_indices.append(idx)
    if len(keep_indices) < len(windows_dataset.datasets):
        drop_indices = [
            i for i in range(len(windows_dataset.datasets)) if i not in keep_indices
        ]
        split = windows_dataset.split([keep_indices, drop_indices])
        return split["0"]
    return windows_dataset


# ---------------------------------------------------------------------------
# Hinss2021-specific helpers
# ---------------------------------------------------------------------------

def _make_2s_events_from_1s_stream(raw: mne.io.BaseRaw, unit_sec: float = 1.0, win_sec: float = 2.0) -> np.ndarray:
    """Stitch consecutive 1-second events of the same class into 2-second windows."""
    sfreq = raw.info["sfreq"]
    unit_samp = int(round(unit_sec * sfreq))
    win_samp = int(round(win_sec * sfreq))

    events = mne.find_events(raw)
    if len(events) == 0:
        return np.empty((0, 3), dtype=int)

    events = events[np.argsort(events[:, 0])]

    # split into runs of consecutive, same-label, 1 s apart
    runs = []
    s = 0
    for i in range(1, len(events)):
        same = events[i, 2] == events[i - 1, 2]
        consecutive_1s = (events[i, 0] - events[i - 1, 0]) == unit_samp
        if not (same and consecutive_1s):
            runs.append((s, i - 1))
            s = i
    runs.append((s, len(events) - 1))

    # carve 2 s onsets inside each run
    new_events = []
    n_raw = len(raw)
    for s, e in runs:
        lab = int(events[s, 2])
        first = int(events[s, 0])
        last_inclusive = int(events[e, 0] + unit_samp - 1)
        run_len = last_inclusive - first + 1
        n_full = run_len // win_samp
        for k in range(n_full):
            onset = first + k * win_samp
            if onset + win_samp <= n_raw:
                new_events.append([onset, 0, lab])

    return np.asarray(new_events, dtype=int)


def _set_2s_annotations(raw: mne.io.BaseRaw, new_events: np.ndarray, label_prefix: str = "class_"):
    """Replace annotations on *raw* with 2-second windows from *new_events*."""
    sfreq = raw.info["sfreq"]
    onsets_sec = new_events[:, 0] / sfreq
    labels = new_events[:, 2]
    descs = [f"{label_prefix}{lab}" for lab in labels]
    ann = mne.Annotations(onset=onsets_sec, duration=[2.0] * len(new_events), description=descs)
    raw.set_annotations(ann)

    unique_labs = np.unique(labels)
    mapping = {f"{label_prefix}{lab}": int(lab) for lab in unique_labs}
    return mapping, raw


def _epoch_hinss(dataset, events_id) -> Tuple[List[WindowsDataset], list, list]:
    """Create braindecode WindowsDatasets from Hinss2021 epochs."""
    list_of_windows_ds = []
    epochs_list = []
    row_descs: list = []

    orig_desc = getattr(dataset, "description", None)

    for i, ds in enumerate(dataset.datasets):
        raw = ds.raw
        sf = raw.info["sfreq"]

        events2, event_id = mne.events_from_annotations(raw, event_id=events_id)
        ep = mne.Epochs(
            raw, events2, event_id=event_id, tmin=0.0,
            tmax=2.0 - 1.0 / sf, baseline=None, preload=True,
        )

        inv = {v: k for k, v in event_id.items()}

        def _get_field(field_name, idx):
            if isinstance(orig_desc, pd.DataFrame) and field_name in orig_desc:
                return orig_desc.loc[idx, field_name]
            if hasattr(ds, "description") and ds.description is not None:
                desc = ds.description
                if isinstance(desc, dict):
                    return desc.get(field_name, idx)
                if isinstance(desc, pd.Series):
                    return desc.get(field_name, idx)
            return idx

        ep.metadata = pd.DataFrame({
            "target": ep.events[:, 2].astype(int),
            "label": [inv.get(c, c) for c in ep.events[:, 2]],
            "subject": _get_field("subject", i),
            "i_trial": np.arange(len(ep.events)),
            "session": _get_field("session", i),
        })

        epochs_list.append(ep)
        row_descs.append({
            "target": ep.events[:, 2].astype(int),
            "label": [inv.get(c, c) for c in ep.events[:, 2]],
            "subject": [_get_field("subject", i)] * len(ep.events),
            "i_trial": np.arange(len(ep.events)),
            "session": [_get_field("session", i)] * len(ep.events),
        })
        windows_ds = WindowsDataset(ep, description=ds.description)
        list_of_windows_ds.append(windows_ds)

    return list_of_windows_ds, epochs_list, row_descs


def load_and_preprocess_hinss(
    cfg: BCIDatasetConfig,
    subject_ids: Optional[Sequence[int]] = None,
    n_jobs: int = 1,
) -> BaseConcatDataset:
    """Load, preprocess, and window the Hinss2021 dataset.

    Hinss2021 needs special handling compared to the motor-imagery datasets:
    - Uses a custom MOABB dataset class (not the standard MOABBDataset wrapper)
    - Preprocessing uses montage instead of channel picking + notch
    - 1-second events are stitched into 2-second windows before epoching
    """
    from neuroatlas.extensions.datasets.dataio.moabb_hinss2021 import Hinss2021 as Hinss2021Dataset

    subjects = list(subject_ids) if subject_ids is not None else list(cfg.subjects)
    dataset = MOABBDataset(dataset_name="Hinss2021", subject_ids=subjects)

    # --- Preprocessing (Hinss2021-specific: montage, no channel pick, no notch) ---
    preprocessors = [
        Preprocessor(_volts_to_microvolts),  # V -> uV
        Preprocessor("set_montage", montage="standard_1020"),
        Preprocessor("filter", l_freq=cfg.fmin, h_freq=cfg.fmax),
    ]
    if cfg.use_car:
        preprocessors.append(
            Preprocessor("set_eeg_reference", ref_channels="average", ch_type="eeg")
        )
    preprocessors.append(Preprocessor("resample", sfreq=cfg.resample_sfreq))

    preprocess(dataset, preprocessors, n_jobs=n_jobs)

    # --- Re-annotate with 2-second windows ---
    all_mappings = []
    for i, ds in enumerate(dataset.datasets):
        raw = ds.raw
        ev2 = _make_2s_events_from_1s_stream(raw)
        mapping, raw = _set_2s_annotations(raw, ev2, label_prefix="class_")
        all_mappings.append(mapping)
        raw.add_events(ev2, stim_channel="stim", replace=True)
        dataset.datasets[i].raw = raw

    mapping = all_mappings[0] if all_mappings else {}

    # --- Windowing via patched create_windows_from_events (from windows_mine) ---
    from neuroatlas.extensions.datasets.dataio.bci_windowing import (
        create_windows_from_events,
    )

    sfreq = dataset.datasets[0].raw.info["sfreq"]
    trial_start_offset_samples = int(cfg.tmin * sfreq)
    trial_stop_offset_samples = 0
    window_size_samples = int(2.0 * sfreq)
    window_stride_samples = int(2.0 * sfreq)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=True,
        preload=True,
        drop_bad_windows=False,
        n_jobs=n_jobs,
        mapping=mapping,
    )

    return windows_dataset


#: ``n_folds`` value asking for leave-one-subject-out
LOSO = "loso"


def resolve_n_folds(n_folds: Union[int, str], n_subjects: int) -> int:
    """How many folds to cut *n_subjects* into.

    ``"loso"`` means one fold per subject -- leave-one-subject-out, which is
    the BCI protocol (App. C.4). It used to be spelled as a large integer and
    left to the clamp below, which worked but read as a magic number.

    The clamp stays for any integer: without it, an *n_folds* above the
    subject count makes ``np.array_split`` return empty chunks and the test
    split silently vanishes.
    """
    if isinstance(n_folds, str):
        if n_folds.strip().lower() != LOSO:
            raise ValueError(
                f"n_folds must be an integer or {LOSO!r}, got {n_folds!r}"
            )
        n_folds = n_subjects
    return max(2, min(int(n_folds), n_subjects))


def get_subject_split(
    all_subjects: Sequence[int],
    fold: int,
    n_folds: Union[int, str] = 5,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split subjects into train / val / test by fold index.

    Uses a simple deterministic split: subjects are shuffled once with
    *random_state*, divided into *n_folds* roughly equal chunks.  The chunk
    at index *fold* is the test set; the next chunk is the validation set;
    the rest is training.
    """
    rng = np.random.RandomState(random_state)
    subjects = np.array(all_subjects)
    perm = rng.permutation(len(subjects))
    n_folds = resolve_n_folds(n_folds, len(subjects))
    chunks = np.array_split(perm, n_folds)

    test_idx = chunks[fold % n_folds]
    val_idx = chunks[(fold + 1) % n_folds]
    train_idx = np.concatenate(
        [chunks[i] for i in range(n_folds) if i != fold % n_folds and i != (fold + 1) % n_folds]
    )

    return (
        subjects[train_idx].tolist(),
        subjects[val_idx].tolist(),
        subjects[test_idx].tolist(),
    )
