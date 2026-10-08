"""Generic MOABB dataset loading and preprocessing.

Provides MOABBDatasetConfig, a registry of 140+ MOABB datasets across all BCI
paradigms, and a single ``load_and_preprocess_moabb()`` function that handles
download, preprocessing, and windowing for any registered dataset.

The four *original* BCI datasets (physionet_mi, cho2017, lee2019_mi, hinss2021)
are NOT included here — they have dedicated hand-tuned configs in ``dataio/bci.py``
with hardcoded channel lists and preprocessed-file support.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MOABBDatasetConfig:
    """Configuration for a generic MOABB dataset.

    Metadata fields (n_subjects, n_channels, …) are informational — actual
    subject and channel lists are discovered from MOABB at load time.
    """

    slug: str
    moabb_name: str
    paradigm: str  # "mi", "p300", "ssvep", "cvep", "resting"
    n_subjects: int
    n_sessions: int
    n_channels: int
    n_classes: int
    native_sfreq: float
    trial_duration: float  # seconds
    resample_sfreq: float = 128.0
    fmin: float = 1.0
    fmax: float = 40.0
    use_car: bool = True
    exclude_subjects: tuple = ()


#: passband held to the motor rhythms under confound filtering
MI_MOTOR_BAND = (4.0, 40.0)

#: seconds dropped from the head of a motor-imagery trial under confound filtering
CONFOUND_CONTROL_TMIN = 1.0

#: Confound filtering, per paradigm (paper App. D.6): ``(fmin, fmax)`` and
#: the seconds dropped from the head of each trial. ``fmin`` None keeps the
#: reader's own high-pass; ``fmax`` None is no low-pass.
#:   mi     4-40 Hz, the trial from 1 s after the cue (the cue's evoked
#:          response and the eye movement towards it fall outside)
#:   p300   0.5-40 Hz band-pass, the trial unchanged
#:   ssvep  high-pass only, so the stimulus harmonics stay in
CONFOUND_FILTERING: Dict[str, Tuple[Tuple[Optional[float], Optional[float]], float]] = {
    "mi": (MI_MOTOR_BAND, CONFOUND_CONTROL_TMIN),
    "p300": ((0.5, 40.0), 0.0),
    "ssvep": ((None, None), 0.0),
}

#: The paper's motor-imagery trial: the first 4 s after the cue. Every MI
#: config the published BCI embeddings were preprocessed with
#: (dataio/bci.py DATASET_CONFIGS, tmax=4.0) cuts the trial there --
#: bci_windowing forces each annotation to 4.0 s -- whatever MOABB's own
#: interval is (BNCI2014_004 4.5 s, BNCI2015_001 and Dreyer2023 5 s,
#: Shin2017A 10 s). The published embeddings agree: STEEGFormer's per-patch
#: BCI embeddings hold 32 patches of 0.125 s (4.0 s) for every MI cohort, and
#: 24 (3.0 s) in the confound-controlled track.
MI_TRIAL_SECONDS = 4.0


def trial_window(cfg: "MOABBDatasetConfig", annotation_seconds: float):
    """``(start, stop)`` of the paper's trial, in s after the annotation onset.

    The annotation is the trial as MOABB publishes it (onset at the cue plus
    the dataset's ``interval[0]``, ``annotation_seconds`` long). The window is
    the one the paper's own preprocessing cut for this cohort:

    1. the cohort's config in dataio/bci.py ``DATASET_CONFIGS`` (its own key,
       else its ``_labram`` variant -- every variant has the same window):
       start ``tmin``, stop ``trial_duration``, as bci.load_and_preprocess
       cuts it (Nakanishi2015: 0.15-4.15 s, a 4.0 s trial);
    2. otherwise, for motor imagery, the first :data:`MI_TRIAL_SECONDS` --
       never more than the annotation, so a cohort with shorter trials keeps
       them whole;
    3. otherwise MOABB's trial as published.

    Confound control's 1 s offset is added on top by the caller.
    """
    from neuroatlas.extensions.datasets.dataio.bci import DATASET_CONFIGS

    paper = DATASET_CONFIGS.get(cfg.slug) or DATASET_CONFIGS.get(f"{cfg.slug}_labram")
    if paper is not None:
        return float(paper.tmin), float(paper.trial_duration)
    if cfg.paradigm == "mi":
        return 0.0, min(MI_TRIAL_SECONDS, float(annotation_seconds))
    return 0.0, float(annotation_seconds)


def _annotation_seconds(dataset, trials: Optional[Dict[str, int]] = None) -> Optional[float]:
    """The single trial duration MOABB gave the trial annotations, or None.

    braindecode builds every annotation of a MOABB recording from the
    dataset's one ``interval``, so the trials share a duration. Only the
    annotations named in *trials* count when it is given: MNE cuts an event
    at a recording's edge short (Dreyer2023's run-start marker, 0.0625 s).
    None when they differ (or there are none): the caller then leaves MOABB's
    trials as they are.
    """
    durations = set()
    for rec in dataset.datasets:
        raw = getattr(rec, "raw", None)
        if raw is not None and len(raw.annotations):
            durations.update(round(float(d), 6)
                             for d, name in zip(raw.annotations.duration,
                                                raw.annotations.description)
                             if trials is None or name in trials)
    return durations.pop() if len(durations) == 1 else None


def loso_fold_count(slug: str) -> int:
    """How many folds ``n_folds="loso"`` means for *slug*.

    One per subject. The count comes from the registered cohort metadata, so
    this answers without downloading anything.
    """
    cfg = MOABB_DATASETS.get(slug)
    if cfg is not None:
        return int(cfg.n_subjects)

    # The cognitive/affective cohorts are not MOABB datasets -- they are read
    # from preprocessed pickles -- but their subject list is pinned in
    # DATASET_CONFIGS, which is the same answer without a download.
    from neuroatlas.extensions.datasets.dataio.bci import DATASET_CONFIGS

    bci_cfg = DATASET_CONFIGS.get(slug)
    if bci_cfg is not None and bci_cfg.subjects:
        return len(bci_cfg.subjects)

    from neuroatlas.cli import _msg

    raise ValueError(_msg.compose(
        f"{slug}: n_folds=loso needs the dataset's list of subjects, which it does not "
        "have; give a number of folds",
        f"neuroatlas probe --dataset {slug} --set n_folds=5"))


def resolve_confound_control(cfg, enabled: bool):
    """Return ``(fmin, fmax, trial_start_offset_samples)`` for this trial;
    ``fmax`` None is no low-pass.

    *enabled* is the paper's confound filtering (App. D.6), which differs by
    paradigm (:data:`CONFOUND_FILTERING`). A motor-imagery trial opens with a
    visual cue, so its first second carries the evoked response and the eye
    movement towards it -- a probe can ride that instead of the imagery: the
    passband is held to the motor rhythms and the window starts
    :data:`CONFOUND_CONTROL_TMIN` in. For ERP and SSVEP the cue-locked
    response is the signal, so the trial is kept whole and only the band
    changes: 0.5-40 Hz for ERP, a high-pass alone for SSVEP. Without
    *enabled* ("no filtering"), the cohort's own passband and the whole trial.

    The paper defines confound filtering for these three paradigms only; for
    another (c-VEP, resting state) it is an error, not a guess.
    """
    if not enabled:
        return cfg.fmin, cfg.fmax, 0
    rule = CONFOUND_FILTERING.get(cfg.paradigm)
    if rule is None:
        raise ValueError(
            f"{cfg.slug}: confound filtering is defined for motor-imagery, ERP and "
            f"SSVEP cohorts, and {cfg.slug} is a {cfg.paradigm} cohort\n"
            f"fix: neuroatlas embed --dataset {cfg.slug} --set confound_control=false")
    (fmin, fmax), tmin = rule
    if fmin is None:
        fmin = cfg.fmin
    return fmin, fmax, int(round(tmin * cfg.resample_sfreq))


# ---------------------------------------------------------------------------
# Per-paradigm preprocessing defaults
# ---------------------------------------------------------------------------

_PARADIGM_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "mi":      {"resample_sfreq": 128.0, "fmin": 4.0, "fmax": 40.0},
    "p300":    {"resample_sfreq": 128.0, "fmin": 1.0, "fmax": 30.0},
    "ssvep":   {"resample_sfreq": 256.0, "fmin": 1.0, "fmax": 50.0},
    "cvep":    {"resample_sfreq": 256.0, "fmin": 1.0, "fmax": 50.0},
    "resting": {"resample_sfreq": 128.0, "fmin": 1.0, "fmax": 40.0},
}


def _cfg(slug, moabb_name, paradigm, n_subj, n_sess, n_ch, n_cls, sfreq, trial_dur, **kw):
    """Shorthand builder — fills defaults from paradigm table."""
    defaults = _PARADIGM_DEFAULTS.get(paradigm, {})
    return MOABBDatasetConfig(
        slug=slug,
        moabb_name=moabb_name,
        paradigm=paradigm,
        n_subjects=n_subj,
        n_sessions=n_sess,
        n_channels=n_ch,
        n_classes=n_cls,
        native_sfreq=sfreq,
        trial_duration=trial_dur,
        resample_sfreq=kw.get("resample_sfreq", defaults.get("resample_sfreq", 128.0)),
        fmin=kw.get("fmin", defaults.get("fmin", 1.0)),
        fmax=kw.get("fmax", defaults.get("fmax", 40.0)),
        use_car=kw.get("use_car", True),
        exclude_subjects=tuple(kw.get("exclude_subjects", ())),
    )


# ============================================================================
# Motor Imagery datasets  (physionet_mi, cho2017, lee2019_mi excluded)
# ============================================================================

_MI = [
    _cfg("alex_mi",                "AlexMI",                "mi",  8,  1,  16,  3,   512,  3.0),
    _cfg("bnci2003_004",           "BNCI2003_004",          "mi",  5,  1, 118,  2,   100,  3.5),
    _cfg("bnci2014_001",           "BNCI2014_001",          "mi",  9,  2,  22,  4,   250,  4.0),
    _cfg("bnci2014_002",           "BNCI2014_002",          "mi", 14,  1,  15,  2,   512,  5.0),
    _cfg("bnci2014_004",           "BNCI2014_004",          "mi",  9,  5,   3,  2,   250,  4.5),
    _cfg("bnci2015_001",           "BNCI2015_001",          "mi", 12,  3,  13,  2,   512,  5.0),
    _cfg("bnci2015_004",           "BNCI2015_004",          "mi",  9,  2,  30,  5,   256,  7.0),
    _cfg("bnci2019_001",           "BNCI2019_001",          "mi", 10,  1,  61,  5,   256,  3.0),
    _cfg("bnci2020_001",           "BNCI2020_001",          "mi", 15,  3,  58,  3,   256,  5.0),
    _cfg("bnci2022_001",           "BNCI2022_001",          "mi", 13,  1,  64,  4,   256, 90.0),
    _cfg("bnci2024_001",           "BNCI2024_001",          "mi", 20,  1,  60, 10,   500,  8.5),
    _cfg("bnci2025_001",           "BNCI2025_001",          "mi", 20,  1,  67, 16,   500,  1.0),
    _cfg("bnci2025_002",           "BNCI2025_002",          "mi", 10,  3,  60,  3,   200, 23.0),
    _cfg("beetl2021_a",            "Beetl2021_A",           "mi",  3,  1,  63,  4,   500,  4.0),
    _cfg("beetl2021_b",            "Beetl2021_B",           "mi",  2,  1,  32,  4,   200,  4.0),
    _cfg("brandl2020",             "Brandl2020",            "mi", 16,  1,  63,  2,  1000,  4.5),
    _cfg("chang2025",              "Chang2025",             "mi", 28,  4,  59,  3,  1000,  6.0),
    _cfg("dreyer2023",             "Dreyer2023",            "mi", 87,  1,  27,  2,   512,  8.0),
    _cfg("dreyer2023a",            "Dreyer2023A",           "mi", 60,  1,  27,  2,   512,  8.0),
    _cfg("dreyer2023b",            "Dreyer2023B",           "mi", 21,  1,  27,  2,   512,  8.0),
    _cfg("dreyer2023c",            "Dreyer2023C",           "mi",  6,  1,  27,  2,   512,  8.0),
    _cfg("forenzo2023",            "Forenzo2023",           "mi", 25,  5,  64,  2,  1000,  6.0),
    _cfg("gao2026",                "Gao2026",               "mi", 22,  2,  32, 10,  1000,  4.0),
    _cfg("grossewentrup2009",      "GrosseWentrup2009",     "mi", 10,  1, 128,  2,   500, 10.0),
    _cfg("guttmannflury2025_me",   "GuttmannFlury2025_ME",  "mi", 31,  3,  64,  2,  1000,  7.5),
    _cfg("guttmannflury2025_mi",   "GuttmannFlury2025_MI",  "mi", 31,  3,  64,  2,  1000,  7.5),
    _cfg("hefmiich2025",           "HefmiIch2025",          "mi", 37,  3,  32,  2,   256, 27.0),
    _cfg("jeong2020",              "Jeong2020",             "mi", 25,  3,  60, 11,  1000,  4.0),
    _cfg("kaya2018",               "Kaya2018",              "mi",  7,  3,  19,  3,   200,  1.0),
    _cfg("kumar2024",              "Kumar2024",             "mi", 18,  6,  22,  2,   512,  5.0),
    _cfg("liu2024",                "Liu2024",               "mi", 50,  1,  29,  2,   500,  8.0),
    _cfg("liu2025",                "Liu2025",               "mi", 27,  1,  60,  2,  1000,  5.0),
    _cfg("ma2020",                 "Ma2020",                "mi", 25, 15,  62,  2,  1000,  4.0),
    _cfg("ofner2017",              "Ofner2017",             "mi", 15,  1,  61,  7,   512,  2.0),
    _cfg("pressel2016",            "Pressel2016",           "mi", 15,  1,   6, 11,  1024,  4.0),
    _cfg("rozado2015",             "Rozado2015",            "mi", 30,  1,  32,  2,   512,  6.0),
    _cfg("schirrmeister2017",      "Schirrmeister2017",     "mi", 14,  1, 128,  4,   500,  4.0),
    _cfg("shin2017a",              "Shin2017A",             "mi", 29,  3,  30,  2,   200, 10.0),
    _cfg("shin2017b",              "Shin2017B",             "mi", 29,  3,  30,  2,   200, 10.0),
    _cfg("stieger2021",            "Stieger2021",           "mi", 62, 11,  62,  4,  1000, 11.0),
    _cfg("tavakolan2017",          "Tavakolan2017",         "mi", 12,  4,  32,  3,  1000,  3.0),
    _cfg("trianaguzman2024",       "TrianaGuzman2024",      "mi", 32,  1,  17,  4,   250, 15.0),
    _cfg("wairagkar2018",          "Wairagkar2018",         "mi", 14,  1,  19,  3,  1024,  6.0),
    _cfg("weibo2014",              "Weibo2014",             "mi", 10,  1,  60,  7,   200,  8.0),
    _cfg("yang2025",               "Yang2025",              "mi", 51,  3,  59,  2,  1000,  7.5),
    _cfg("yi2025",                 "Yi2025",                "mi", 18,  1,  62,  8,  1000,  4.0),
    _cfg("zhang2017",              "Zhang2017",             "mi", 12,  1,  17, 10,  1000,  5.0),
    _cfg("zhou2016",               "Zhou2016",              "mi",  4,  3,  14,  3,   250, 10.0),
    _cfg("zhou2020",               "Zhou2020",              "mi", 20,  7,  41,  4,   500,  5.0),
    _cfg("zuo2025",                "Zuo2025",               "mi", 30,  5,  30,  2,   500,  4.0),
    _cfg("aguilerarodriguez2025",  "AguileraRodriguez2025", "mi", 15,  1,  24,  4,   500, 11.8),
    _cfg("bcicomp2020_is",         "BCIComp2020IS",         "mi", 15,  1,  64,  5,   256,  3.1),
    _cfg("bcicomp2020_upperlimb",  "BCIComp2020UpperLimb",  "mi", 15,  3,  60,  3,   250,  4.0),
    _cfg("nguyen2017_l",           "Nguyen2017_L",          "mi",  6,  1,  60,  2,   256,  5.0),
    _cfg("nguyen2017_s",           "Nguyen2017_S",          "mi",  6,  1,  60,  3,   256,  5.0),
    _cfg("nguyen2017_sl",          "Nguyen2017_SL",         "mi",  6,  1,  60,  2,   256,  5.0),
    _cfg("nguyen2017_v",           "Nguyen2017_V",          "mi",  8,  1,  60,  3,   256,  5.0),
]

# ============================================================================
# P300 / ERP datasets
# ============================================================================

_P300 = [
    _cfg("bcicomp2020_walkingerp", "BCIComp2020WalkingERP", "p300", 15,  1, 46,  2,  100,  1.0),
    _cfg("bi2012",                 "BI2012",                "p300", 25,  2, 16,  2,  128,  1.0),
    _cfg("bi2013a",                "BI2013a",               "p300", 24,  2, 16,  2,  512,  8.0),
    _cfg("bi2014a",                "BI2014a",               "p300", 64,  1, 16,  2,  512,  1.0),
    _cfg("bi2014b",                "BI2014b",               "p300", 38,  1, 32,  2,  512,  1.0),
    _cfg("bi2015a",                "BI2015a",               "p300", 43,  1, 32,  2,  512,  3.0),
    _cfg("bi2015b",                "BI2015b",               "p300", 44,  4, 32,  2,  512,  1.0),
    _cfg("bnci2014_008",           "BNCI2014_008",          "p300",  8,  1,  8,  2,  256,  1.0),
    _cfg("bnci2014_009",           "BNCI2014_009",          "p300", 10,  3, 16,  2,  256, 16.0),
    _cfg("bnci2015_003",           "BNCI2015_003",          "p300", 10,  2,  8,  2,  256,  1.0),
    _cfg("bnci2015_006",           "BNCI2015_006",          "p300", 11,  1, 64,  2,  200, 40.0),
    _cfg("bnci2015_007",           "BNCI2015_007",          "p300", 16,  1, 63,  2,  100, 30.0),
    _cfg("bnci2015_008",           "BNCI2015_008",          "p300", 13,  1, 63,  2,  250, 30.0),
    _cfg("bnci2015_009",           "BNCI2015_009",          "p300", 21,  1, 60,  2,  250,  0.8),
    _cfg("bnci2015_010",           "BNCI2015_010",          "p300", 12,  1, 63,  2,  200, 46.5),
    _cfg("bnci2015_012",           "BNCI2015_012",          "p300", 10,  2, 63,  2,  250,  1.0),
    _cfg("bnci2015_013",           "BNCI2015_013",          "p300",  6, 20, 64,  2,  512,  2.0),
    _cfg("bnci2016_002",           "BNCI2016_002",          "p300", 15,  1, 59,  2,  200,  3.0),
    _cfg("bnci2020_002",           "BNCI2020_002",          "p300", 18,  1, 30,  2,  250,  1.0),
    _cfg("cattan2019_vr",          "Cattan2019_VR",         "p300", 21, 60, 16,  2,  512,  1.0),
    _cfg("chailloux2020",          "Chailloux2020",         "p300", 19,  7,  8,  2,  256,  1.0),
    _cfg("epflp300",               "EPFLP300",              "p300",  8,  6, 32,  2, 2048,  1.0),
    _cfg("erpcore2021",            "ErpCore2021",           "p300", 40,  1, 30,  1, 1024,  1.0),
    _cfg("erpcore2021_ern",        "ErpCore2021_ERN",       "p300", 40,  1, 30,  2, 1024,  1.0),
    _cfg("erpcore2021_lrp",        "ErpCore2021_LRP",       "p300", 40,  1, 30,  2, 1024,  1.0),
    _cfg("erpcore2021_mmn",        "ErpCore2021_MMN",       "p300", 40,  1, 30,  2, 1024,  1.0),
    _cfg("erpcore2021_n170",       "ErpCore2021_N170",      "p300", 40,  1, 30,  2, 1024,  1.0),
    _cfg("erpcore2021_n2pc",       "ErpCore2021_N2pc",      "p300", 40,  1, 30,  2, 1024,  1.0),
    _cfg("erpcore2021_n400",       "ErpCore2021_N400",      "p300", 40,  1, 30,  2, 1024,  1.0),
    _cfg("erpcore2021_p3",         "ErpCore2021_P3",        "p300", 40,  1, 30,  2, 1024,  1.0),
    _cfg("guttmannflury2025_p300", "GuttmannFlury2025_P300","p300", 31,  1, 64,  2, 1000,  3.0),
    _cfg("huebner2017",            "Huebner2017",           "p300", 13,  9, 31,  2, 1000, 25.0),
    _cfg("huebner2018",            "Huebner2018",           "p300", 12,  1, 31,  2, 1000, 17.0),
    _cfg("kaneshiro2015",          "Kaneshiro2015",         "p300", 10,  1,124,  6,  62.5, 0.5),
    _cfg("kojima2024a",            "Kojima2024A",           "p300", 11,  6, 64,  2, 1000,  1.0),
    _cfg("kojima2024b",            "Kojima2024B",           "p300", 15, 12, 64,  2, 1000, 90.0),
    _cfg("lee2019_erp",            "Lee2019_ERP",           "p300", 54,  2, 62,  2, 1000,  2.0),
    _cfg("lee2021mobile_erp",      "Lee2021Mobile_ERP",     "p300", 24,  5, 73,  2,  100,  1.0),
    _cfg("lee2024_ac",             "Lee2024_AC",            "p300", 10,  1, 25,  2,  500,  1.0),
    _cfg("lee2024_bs",             "Lee2024_BS",            "p300", 14,  1, 31,  2,  500,  1.0),
    _cfg("lee2024_dl",             "Lee2024_DL",            "p300", 15,  1, 31,  2,  500,  1.0),
    _cfg("lee2024_el",             "Lee2024_EL",            "p300", 15,  1, 31,  2,  500,  1.0),
    _cfg("lee2024_tv",             "Lee2024_TV",            "p300", 30,  1, 31,  2,  500,  1.0),
    _cfg("mainsah2025_a",          "Mainsah2025_A",         "p300", 13,  1, 32,  2,  256,  1.0),
    _cfg("mainsah2025_b",          "Mainsah2025_B",         "p300", 19,  1, 16,  2,  256,  8.0),
    _cfg("mainsah2025_c",          "Mainsah2025_C",         "p300", 19,  1, 32,  2,  256,  1.0),
    _cfg("mainsah2025_d",          "Mainsah2025_D",         "p300", 17,  1, 32,  2,  256,  1.0),
    _cfg("mainsah2025_e",          "Mainsah2025_E",         "p300",  8,  1, 16,  2,  256,  1.0),
    _cfg("mainsah2025_f",          "Mainsah2025_F",         "p300", 10,  1, 16,  2,  256,  3.0),
    _cfg("mainsah2025_g",          "Mainsah2025_G",         "p300", 20,  1, 16,  2,  256,  1.0),
    _cfg("mainsah2025_h",          "Mainsah2025_H",         "p300", 16,  1, 16,  2,  256,  1.0),
    _cfg("mainsah2025_i",          "Mainsah2025_I",         "p300", 13,  1, 16,  2,  256,  1.0),
    _cfg("mainsah2025_j",          "Mainsah2025_J",         "p300", 20,  1, 16,  2,  256,  1.0),
    _cfg("mainsah2025_k",          "Mainsah2025_K",         "p300",  5,  1, 16,  2,  256,  2.0),
    _cfg("mainsah2025_l",          "Mainsah2025_L",         "p300", 11,  1, 16,  2,  256,  1.0),
    _cfg("mainsah2025_m",          "Mainsah2025_M",         "p300", 21,  1, 16,  2,  256,  1.0),
    _cfg("mainsah2025_n",          "Mainsah2025_N",         "p300",  8,  1, 16,  2,  256,  2.0),
    _cfg("mainsah2025_o",          "Mainsah2025_O",         "p300", 18,  1, 32,  2,  256,  2.0),
    _cfg("mainsah2025_p",          "Mainsah2025_P",         "p300", 19,  1, 32,  2,  256,  2.0),
    _cfg("mainsah2025_q",          "Mainsah2025_Q",         "p300", 36,  1, 32,  2,  256,  3.0),
    _cfg("mainsah2025_r",          "Mainsah2025_R",         "p300", 20,  1, 32,  2,  256,  2.0),
    _cfg("mainsah2025_s1",         "Mainsah2025_S1",        "p300", 10,  1, 32,  2,  256,  1.0),
    _cfg("mainsah2025_s2",         "Mainsah2025_S2",        "p300", 24,  1, 32,  2,  256,  1.0),
    _cfg("romanibf2025erp",        "RomaniBF2025ERP",       "p300", 22,  1,  8,  2,  250,  0.9),
    _cfg("simoes2020",             "Simoes2020",            "p300", 15,  7,  8,  2,  250,  1.2),
    _cfg("sosulski2019",           "Sosulski2019",          "p300", 13,  1, 31,  2, 1000, 80.0),
    _cfg("speier2017",             "Speier2017",            "p300", 10,  3, 32,  2,  256,  1.0),
    _cfg("zheng2020",              "Zheng2020",             "p300", 14,  3, 62,  2, 1000,  1.0),
    _cfg("zhang2025",              "Zhang2025",             "p300", 15,  4, 57,  2, 1000,  1.0),
]

# ============================================================================
# SSVEP datasets
# ============================================================================

_SSVEP = [
    _cfg("chen2017singleflicker",    "Chen2017SingleFlicker",   "ssvep", 12,  1,  32,  4,  512, 2.0),
    _cfg("dong2023",                 "Dong2023",                "ssvep", 59,  1,   8, 40,  250, 4.0),
    _cfg("guttmannflury2025_ssvep",  "GuttmannFlury2025_SSVEP", "ssvep", 31, 3,  64,  4, 1000, 7.0),
    _cfg("han2024fatigue",           "Han2024Fatigue",          "ssvep", 24,  2,  64, 32, 1000, 2.0),
    _cfg("kalunga2016",              "Kalunga2016",             "ssvep", 12,  1,   8,  4,  256, 6.0),
    _cfg("kim2025betarange",         "Kim2025BetaRange",        "ssvep", 40,  6,  31, 40, 1024, 5.0),
    _cfg("lee2019_ssvep",            "Lee2019_SSVEP",           "ssvep", 54,  2,  62,  4, 1000, 4.0),
    _cfg("lee2021mobile_ssvep",      "Lee2021Mobile_SSVEP",     "ssvep", 23,  4,  73,  3,  100, 5.0),
    _cfg("liu2020beta",              "Liu2020BETA",             "ssvep", 70,  1,  64, 40,  250, 3.0),
    _cfg("liu2022eldbeta",           "Liu2022EldBETA",          "ssvep",100,  7,  64,  9, 1000, 5.0),
    _cfg("mamem1",                   "MAMEM1",                  "ssvep", 11,  1, 256,  5,  250, 5.0),
    _cfg("mamem2",                   "MAMEM2",                  "ssvep", 11,  5, 256,  5,  250, 5.0),
    _cfg("mamem3",                   "MAMEM3",                  "ssvep", 11, 10,  14,  5,  128, 5.0),
    _cfg("nakanishi2015",            "Nakanishi2015",           "ssvep",  9,  1,   8, 12,  256, 4.0),
    _cfg("wang2016",                 "Wang2016",                "ssvep", 34,  1,  64, 40,  250, 6.0),
    _cfg("wang2021combined",         "Wang2021Combined",        "ssvep",  8,  1,  31,  4, 1000, 5.0),
]

# ============================================================================
# c-VEP datasets
# ============================================================================

_CVEP = [
    _cfg("castillosburstvep100",       "CastillosBurstVEP100",       "cvep", 12, 1, 32,  2,  500, 2.2),
    _cfg("castillosburstvep40",        "CastillosBurstVEP40",        "cvep", 12, 1, 32,  2,  500, 2.2),
    _cfg("castilloscvep100",           "CastillosCVEP100",           "cvep", 12, 1, 32,  2,  500, 2.2),
    _cfg("castilloscvep40",            "CastillosCVEP40",            "cvep", 12, 1, 32,  2,  500, 2.2),
    _cfg("martinezcagigal2023checker", "MartinezCagigal2023Checker", "cvep", 16, 3, 16,  2,  256, 8.0),
    _cfg("martinezcagigal2023pary",    "MartinezCagigal2023Pary",    "cvep", 16, 8, 16, 11,  256, 5.0),
    _cfg("thielen2015",                "Thielen2015",                "cvep", 12, 3, 64,  2, 2048, 4.2),
    _cfg("thielen2021",                "Thielen2021",                "cvep", 30, 5,  8,  2,  512, 31.5),
]

# ============================================================================
# Resting-state datasets  (hinss2021 excluded — has dedicated config)
# ============================================================================

_RESTING = [
    _cfg("cattan2019_phmd", "Cattan2019_PHMD", "resting", 12, 1, 16, 2, 512, 60.0),
    _cfg("rodrigues2017",   "Rodrigues2017",   "resting", 19, 1, 16, 2, 512, 10.0),
]

# ============================================================================
# Unified registry
# ============================================================================

MOABB_DATASETS: Dict[str, MOABBDatasetConfig] = {}
for _lst in (_MI, _P300, _SSVEP, _CVEP, _RESTING):
    for _c in _lst:
        MOABB_DATASETS[_c.slug] = _c


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

class MOABBDatasetUnavailable(LookupError):
    """The installed moabb has no class for this cohort, and none is vendored."""


def moabb_dataset_class(moabb_name: str) -> type:
    """The MOABB dataset class *moabb_name*.

    The installed moabb's own class when it has one, else the copy in
    ``moabb_vendored`` (Dreyer2023 and Kim2025BetaRange, which moabb 1.2.0 --
    the last release on numpy<2 -- predates). Anything else raises
    :class:`MOABBDatasetUnavailable`, naming the moabb version: most of the
    MOABB cohorts registered here beyond the paper's 14 arrived in moabb
    releases that need numpy>=2.
    """
    import moabb

    from neuroatlas.extensions.datasets.dataio.moabb_vendored import dataset_class

    cls = dataset_class(moabb_name)
    if cls is None:
        raise MOABBDatasetUnavailable(
            f"moabb {moabb.__version__} has no dataset {moabb_name!r}, and NeuroAtlas "
            f"vendors only Dreyer2023 and Kim2025BetaRange. It is in a later moabb, "
            f"which needs numpy>=2 (the benchmark's stack is numpy 1.26 with moabb 1.2.0)."
        )
    return cls


def moabb_dataset_arg(moabb_name: str):
    """What braindecode's ``MOABBDataset(dataset_name=...)`` should get.

    The name, when the installed moabb registers the class (braindecode looks
    it up in moabb's ``dataset_list``); an instance of the vendored class
    otherwise -- braindecode accepts a ``BaseDataset`` instance too. Without
    moabb the name is passed through, for braindecode to report.
    """
    try:
        import moabb.datasets as upstream
    except ImportError:
        return moabb_name
    if hasattr(upstream, moabb_name):
        return moabb_name
    from neuroatlas.extensions.datasets.dataio.moabb_vendored import VENDORED

    return moabb_dataset_class(moabb_name)() if moabb_name in VENDORED else moabb_name


class MOABBDataMissing(FileNotFoundError):
    """MOABB would have to download this cohort, and downloads are off."""


@contextlib.contextmanager
def no_download_when_offline(slug: str):
    """With downloads switched off, refuse instead of letting MOABB fetch.

    MOABB downloads whatever a dataset lacks the moment it is read, through
    whichever transport that dataset uses (pooch, urllib, mne's fetcher,
    NEMAR's client). Every one of them opens an internet socket, so under
    offline mode (``neuroatlas.config.is_offline``: every command but the
    download ones, unless ``--online``) the read runs with internet sockets
    refused; the first attempt ends it with one message naming the command
    that fetches the cohort. A run would otherwise fetch it unannounced, as
    `run` did with 132 MB of Nakanishi2015.
    """
    from neuroatlas.config import is_offline

    if not is_offline():
        yield
        return
    attempted: List[Any] = []
    real_connect = socket.socket.connect

    def _refuse(sock, address, *args, **kwargs):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            attempted.append(address)
            raise OSError(f"NeuroAtlas is offline: refused a connection to {address!r}")
        return real_connect(sock, address, *args, **kwargs)

    message = (
        f"{slug}: part of its data is not in $MNE_DATA "
        f"({os.environ.get('MNE_DATA') or '~/mne_data'}), and downloads are off\n"
        f"fix: neuroatlas data download {slug}"
    )
    socket.socket.connect = _refuse
    try:
        yield
    except Exception as exc:
        if attempted:
            raise MOABBDataMissing(message) from exc
        raise
    finally:
        socket.socket.connect = real_connect
    if attempted:                                    # a transport that swallowed the error
        raise MOABBDataMissing(message)


def get_moabb_subjects(cfg: MOABBDatasetConfig) -> List[int]:
    """Discover available subjects from MOABB, minus exclusions."""
    with no_download_when_offline(cfg.slug):
        ds = moabb_dataset_class(cfg.moabb_name)()
    return [s for s in ds.subject_list if s not in cfg.exclude_subjects]


# ---------------------------------------------------------------------------
# Generic loading pipeline
# ---------------------------------------------------------------------------

def model_input(cfg: MOABBDatasetConfig, bci_format: Optional[str]):
    """``(cfg, notch)``: the cohort's config at the model format's rate and
    band (``dataio/bci_formats.py``), and the notch frequency (None: none).
    Without a format, the cohort's own rate and band and no notch."""
    if bci_format is None:
        return cfg, None
    import dataclasses

    from neuroatlas.extensions.datasets.dataio.bci_formats import check_format, line_frequency

    fmt = check_format(bci_format)
    notch = fmt.notch if fmt.notch is not None else line_frequency(cfg.slug)
    return dataclasses.replace(cfg, resample_sfreq=fmt.rate, fmin=fmt.fmin, fmax=fmt.fmax), notch


def load_and_preprocess_moabb(
    cfg: MOABBDatasetConfig,
    subject_ids: Optional[Sequence[int]] = None,
    n_jobs: int = 1,
    confound_control: bool = False,
    bci_format: Optional[str] = None,
):
    """Download (if needed), preprocess, and window any MOABB dataset.

    Pipeline:
      1. Load raw data via braindecode ``MOABBDataset``
      2. Keep the EEG channels; scale V → µV; with a model format
         (*bci_format*), a notch at the line frequency and its harmonics
      3. Band-pass filter (the model format's band, else the paradigm's): the paradigm's own band, or under confound
         filtering the paper's per paradigm (:func:`resolve_confound_control`;
         SSVEP: a high-pass alone)
      4. Optional common average reference
      5. Resample to target sampling rate
      6. Window from event annotations: the paper's trial window
         (:func:`trial_window`), plus confound filtering's 1 s offset for
         motor imagery

    Returns
    -------
    braindecode.datasets.BaseConcatDataset
        A windowed dataset ready for iteration.
    """
    from braindecode.datasets import BaseConcatDataset
    from braindecode.datasets.moabb import MOABBDataset
    from braindecode.preprocessing import Preprocessor, preprocess
    from braindecode.preprocessing.windowers import create_windows_from_events

    from neuroatlas import progress, quiet

    subjects = list(subject_ids) if subject_ids else get_moabb_subjects(cfg)
    # The cohort as the last call in this command made it: `run` embeds and
    # then probes, and every model and fold builds its datamodule, each one
    # loading and filtering the same subjects (minutes on BCI) to the same
    # windows. Only within one `neuroatlas` command (quiet.per_command).
    cfg, notch = model_input(cfg, bci_format)
    key = (repr(cfg), notch, tuple(subjects), bool(confound_control), os.environ.get("MNE_DATA"))
    if quiet.in_command() and _LAST_COHORT.get("key") == key:
        logger.info("%s: the %d subjects loaded earlier in this command are reused",
                    cfg.slug, len(subjects))
        return _LAST_COHORT["windows"]
    _LAST_COHORT.clear()

    fmin, fmax, start_offset = resolve_confound_control(cfg, confound_control)

    preprocessors = [
        # EEG channels only. Without this the windows also carried the
        # recording's EOG and stimulus channels -- BNCI2014_001 came out as
        # 22 EEG + EOG1-3 + STI = 26 channels against the 22 its config and
        # metadata declare -- and, since the scaling below only touches data
        # channels, the EOG stayed in volts (~1e-6) next to EEG in µV.
        Preprocessor("pick", picks="eeg"),
        Preprocessor(lambda x: x * 1e6),  # V -> µV
    ]
    if notch:
        import numpy as np

        preprocessors.append(Preprocessor(
            "notch_filter", freqs=np.arange(notch, cfg.native_sfreq / 2, notch)))
    preprocessors.append(Preprocessor("filter", l_freq=fmin, h_freq=fmax))
    if cfg.use_car:
        preprocessors.append(
            Preprocessor("set_eeg_reference", ref_channels="average", ch_type="eeg")
        )
    preprocessors.append(Preprocessor("resample", sfreq=cfg.resample_sfreq))

    # Subject by subject, so the item's live line counts them ("loading the
    # data 33% (3/9 subjects)"): each recording is read and filtered on its
    # own either way, so the windows are the ones a single call makes.
    item = progress.current()
    item.phase("loading the data", total=len(subjects), unit="subjects")
    parts = []
    with no_download_when_offline(cfg.slug):
        for subject in subjects:
            part = MOABBDataset(dataset_name=moabb_dataset_arg(cfg.moabb_name),
                                subject_ids=[subject])
            preprocess(part, preprocessors, n_jobs=n_jobs)
            parts.append(part)
            item.update(advance=1)
    dataset = BaseConcatDataset([ds for part in parts for ds in part.datasets])
    # numbered across the cohort, as one MOABBDataset call numbers them
    for i, ds in enumerate(dataset.datasets):
        ds.description.name = i

    item.phase("cutting the trials")
    # The paper's trial, not MOABB's: BNCI2014_004's 4.5 s and BNCI2015_001's
    # 5 s trials were cut to the 4 s after the cue. Left at 4.5 s the trial
    # was a whole number of patches for no 1 s-patch model, so LaBraM,
    # CBraMod and NeuroLM refused it.
    trials = _trial_mapping(cfg)
    start, stop = 0, 0
    annotation_seconds = _annotation_seconds(dataset, trials)
    if annotation_seconds is not None:
        t0, t1 = trial_window(cfg, annotation_seconds)
        start = int(round(t0 * cfg.resample_sfreq))
        stop = int(round((t1 - annotation_seconds) * cfg.resample_sfreq))

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=start + start_offset,
        trial_stop_offset_samples=stop,
        mapping=trials,
        preload=True,
    )
    if quiet.in_command():
        _LAST_COHORT.update(key=key, windows=windows_dataset)
    return windows_dataset


def _per_command_store() -> Dict[str, Any]:
    from neuroatlas import quiet

    return quiet.per_command({})


#: The last cohort load_and_preprocess_moabb made in this command: its key
#: (config, subjects, confound control, MNE_DATA) and windows.
_LAST_COHORT: Dict[str, Any] = _per_command_store()


def _trial_mapping(cfg: MOABBDatasetConfig) -> Optional[Dict[str, int]]:
    """The cohort's own trial events -> 0-based labels, or None to infer.

    braindecode turns every annotation into a trial unless told which. For a
    cohort read from BIDS (Dreyer2023, ErpCore2021) the annotations are every
    event in events.tsv -- run starts, fixation crosses, feedback -- and
    Dreyer2023 failed outright ("Overlapping trials detected"). The mapping
    keeps the classes the MOABB dataset declares (its ``event_id``), numbered
    in sorted order, which is the numbering braindecode infers when those are
    the only annotations -- so the other cohorts' labels are unchanged. (The
    paper's pipeline passed its own class mapping the same way.)
    """
    try:
        events = moabb_dataset_class(cfg.moabb_name)().event_id
    except Exception:                   # no moabb, or a cohort it does not know
        return None
    return {name: i for i, name in enumerate(sorted(events))}
