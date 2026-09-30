"""Generic MOABB dataset loading and preprocessing.

Provides MOABBDatasetConfig, a registry of 140+ MOABB datasets across all BCI
paradigms, and a single ``load_and_preprocess_moabb()`` function that handles
download, preprocessing, and windowing for any registered dataset.

The four *original* BCI datasets (physionet_mi, cho2017, lee2019_mi, hinss2021)
are NOT included here — they have dedicated hand-tuned configs in ``dataio/bci.py``
with hardcoded channel lists and preprocessed-file support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


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


#: passband held to the motor rhythms when confound control is on
MI_MOTOR_BAND = (4.0, 40.0)

#: seconds dropped from the head of each trial when confound control is on
CONFOUND_CONTROL_TMIN = 1.0


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

    raise KeyError(
        f"n_folds='loso' needs a subject count for {slug!r}, which is neither a "
        "registered MOABB cohort nor a DATASET_CONFIGS entry with a subject "
        "list. Give an integer instead."
    )


def resolve_confound_control(cfg, enabled: bool):
    """Return ``(fmin, fmax, trial_start_offset_samples)`` for this trial.

    A motor-imagery trial opens with a visual cue, so its first second carries
    the evoked response and the eye movement towards it -- a probe can ride
    that instead of the imagery. With *enabled*, the passband is held to the
    motor rhythms and the window starts :data:`CONFOUND_CONTROL_TMIN` in;
    without it, the cohort's own passband and the whole trial are used. The
    difference between the two is how much of a score came from the cue.

    Motor imagery only. For a P300, ERP or SSVEP cohort the cue-locked
    response *is* the signal, so dropping the first second would remove what
    is being measured -- that is an error, not a quieter result.
    """
    if not enabled:
        return cfg.fmin, cfg.fmax, 0
    if cfg.paradigm != "mi":
        raise ValueError(
            f"confound_control is defined for motor imagery, but {cfg.slug!r} is "
            f"a {cfg.paradigm!r} paradigm. There the cue-locked response is the "
            "signal, not a confound, so dropping the first second would remove "
            "what is being measured."
        )
    fmin, fmax = MI_MOTOR_BAND
    return fmin, fmax, int(round(CONFOUND_CONTROL_TMIN * cfg.resample_sfreq))


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

def get_moabb_subjects(cfg: MOABBDatasetConfig) -> List[int]:
    """Discover available subjects from MOABB, minus exclusions."""
    import moabb.datasets as moabb_ds

    ds_cls = getattr(moabb_ds, cfg.moabb_name)
    ds = ds_cls()
    return [s for s in ds.subject_list if s not in cfg.exclude_subjects]


# ---------------------------------------------------------------------------
# Generic loading pipeline
# ---------------------------------------------------------------------------

def load_and_preprocess_moabb(
    cfg: MOABBDatasetConfig,
    subject_ids: Optional[Sequence[int]] = None,
    n_jobs: int = 1,
    confound_control: bool = False,
):
    """Download (if needed), preprocess, and window any MOABB dataset.

    Pipeline:
      1. Load raw data via braindecode ``MOABBDataset``
      2. Scale V → µV
      3. Bandpass filter  (paradigm-specific defaults)
      4. Optional common average reference
      5. Resample to target sampling rate
      6. Window from event annotations

    Returns
    -------
    braindecode.datasets.BaseConcatDataset
        A windowed dataset ready for iteration.
    """
    from braindecode.datasets.moabb import MOABBDataset
    from braindecode.preprocessing import Preprocessor, preprocess
    from braindecode.preprocessing.windowers import create_windows_from_events

    subjects = list(subject_ids) if subject_ids else get_moabb_subjects(cfg)
    dataset = MOABBDataset(dataset_name=cfg.moabb_name, subject_ids=subjects)

    fmin, fmax, start_offset = resolve_confound_control(cfg, confound_control)

    preprocessors = [
        Preprocessor(lambda x: x * 1e6),  # V -> µV
        Preprocessor("filter", l_freq=fmin, h_freq=fmax),
    ]
    if cfg.use_car:
        preprocessors.append(
            Preprocessor("set_eeg_reference", ref_channels="average", ch_type="eeg")
        )
    preprocessors.append(Preprocessor("resample", sfreq=cfg.resample_sfreq))

    preprocess(dataset, preprocessors, n_jobs=n_jobs)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=start_offset,
        trial_stop_offset_samples=0,
        preload=True,
    )

    return windows_dataset
