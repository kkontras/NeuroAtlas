"""Where the prepared BCI pickles are looked for, and the config keys they
are named after.

Split out of ``dataio/bci.py`` so that what only asks where a prepared file
is -- ``neuroatlas data status`` -- does not import MNE, braindecode and
torch with it. ``dataio/bci.py`` re-exports every name here, so readers that
import them from there are unchanged.

Standard library only.
"""
from __future__ import annotations

import glob as _glob
import os as _os
import re as _re
from pathlib import Path as _Path
from typing import Dict, Iterable, List, Optional, Tuple

from neuroatlas._paths import prepared_dir

# ---------------------------------------------------------------------------
# Config keys: alignment x confound control
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


def mi_config_key(slug: str, alignment: str = "default",
                  confound_control: bool = False) -> str:
    """The DATASET_CONFIGS key for one point on the two axes."""
    key = slug if alignment == "default" else f"{slug}_{alignment}"
    return key + CONFOUND_CONTROLLED if confound_control else key


#: (slug, MOABB name) of the broad-sweep MI cohorts, in ``_BROAD_MI_SPECS``
#: order (``dataio/bci.py`` checks the two agree at import).
BROAD_MI_NAMES: List[Tuple[str, str]] = [
    ("schirrmeister2017", "Schirrmeister2017"),
    ("shin2017a", "Shin2017A"),
    ("weibo2014", "Weibo2014"),
    ("bnci2014_001", "BNCI2014_001"),
    ("dreyer2023a", "Dreyer2023A"),
    ("bnci2014_004", "BNCI2014_004"),
    ("bnci2015_001", "BNCI2015_001"),
]


# ---------------------------------------------------------------------------
# Paths to already-preprocessed data on disk.
# Each dataset has a search list: files under the prepared folder (what
# `data prepare` builds) and files in the dataset's folder under the data root
# (what the authors provide), in the order given; a candidate may be a glob
# (see matching_files). The first existing file wins. The ``preprocessed_path``
# setting replaces the list.
# ---------------------------------------------------------------------------

# Built by `neuroatlas data prepare`: <cache root>/prepared/<Name>/... (it was
# <checkout>/data/preprocessed, inside the source tree). The table below is
# written against a placeholder for that folder, and every read resolves it
# against the cache root as it is *then*: it used to be resolved at import,
# so a process that changed the cache root afterwards (a test, an API user,
# `config set` in the same session) still looked in the old one.
_PREPARED_TOKEN = "@prepared@"
_DATA_DIR = _Path(_PREPARED_TOKEN)


def _resolve_prepared(path: str) -> str:
    """The prepared folder resolved against the cache root, and ``${VAR}``
    (the data root) against the environment, as it is *now*."""
    if path.startswith(_PREPARED_TOKEN):
        return str(prepared_dir()) + path[len(_PREPARED_TOKEN):]
    return _re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
                   lambda m: _os.environ.get(m.group(1), m.group(0)), path)


#: When a folder holds the same pickle prepared for several models, the one
#: read is the first of these (the smallest first; each carries the same
#: signal), then any other in name order.
PREFERRED_VARIANTS = ("steegformer", "labram", "bendr")


def matching_files(candidate: str) -> List[str]:
    """The files a candidate names: itself when it is a file, or, for a glob
    (``EEGMat_preprocessed_*.pkl``), its matches in :data:`PREFERRED_VARIANTS`
    order."""
    if not any(c in candidate for c in "*?["):
        return [candidate] if _os.path.isfile(candidate) else []

    def rank(path: str):
        stem = _Path(path).stem
        for i, variant in enumerate(PREFERRED_VARIANTS):
            if stem.endswith("_" + variant):
                return (i, path)
        return (len(PREFERRED_VARIANTS), path)

    return sorted((p for p in _glob.glob(candidate) if _os.path.isfile(p)), key=rank)


def first_existing(candidates: Iterable[str]) -> Optional[str]:
    """The first file the candidates name, in order (globs as in
    :func:`matching_files`), or None."""
    for candidate in candidates:
        hits = matching_files(candidate)
        if hits:
            return hits[0]
    return None


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
    # Cognitive/affective cohorts: the pickle the authors provide,
    # <Dataset>_preprocessed_<variant>.pkl, in the dataset's folder under the
    # data root (then under the prepared folder). Any variant carries the same
    # signal; PREFERRED_VARIANTS picks one when there are several.
    "eegmat": [
        "${EEG_DATA_ROOT}/eegmat/EEGMat_preprocessed_*.pkl",
        str(_DATA_DIR / "EEGMat" / "EEGMat_preprocessed_*.pkl"),
    ],
    "arithmetic_task": [
        "${EEG_DATA_ROOT}/arithmetic_task/ArithmeticTask_preprocessed_*.pkl",
        str(_DATA_DIR / "ArithmeticTask" / "ArithmeticTask_preprocessed_*.pkl"),
    ],
    "dreamer_valence": [
        "${EEG_DATA_ROOT}/dreamer_valence/DREAMER_valence_preprocessed_*.pkl",
        str(_DATA_DIR / "DREAMER" / "DREAMER_valence_preprocessed_*.pkl"),
    ],
    "dreamer_arousal": [
        "${EEG_DATA_ROOT}/dreamer_arousal/DREAMER_arousal_preprocessed_*.pkl",
        str(_DATA_DIR / "DREAMER" / "DREAMER_arousal_preprocessed_*.pkl"),
    ],
    # PhysionetMI: repo-local .pkl exists → try it first
    "physionet_mi": [
        str(_DATA_DIR / "PhysionetMI" / "physionetMI_preprocessed.pkl"),
        str(_DATA_DIR / "PhysionetMI" / "physionetMI_preprocessed.mat"),
        "${EEG_DATA_ROOT}/physionet_mi/physionetMI_preprocessed.mat",
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
        "${EEG_DATA_ROOT}/cho2017/Cho2017_preprocessed.mat",
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
        "${EEG_DATA_ROOT}/lee2019_mi/Lee2019_MI_preprocessed_1s.mat",
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
        "${EEG_DATA_ROOT}/hinss2021/Hinss2021_preprocessed_2s.pkl",
        str(_DATA_DIR / "Hinss2021" / "Hinss2021_preprocessed.pkl"),
        str(_DATA_DIR / "Hinss2021" / "Hinss2021_preprocessed_2s.pkl"),
    ],
})

# Broad MI sweep: auto-register repo-local search paths for each (slug, variant).
# cbramod/reve/eegpt fall back to labram-preprocessed files (same sfreq).
_FALLBACK_VARIANTS = {"cbramod": "labram", "reve": "labram"}
for _slug, _name in BROAD_MI_NAMES:
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

# experiment: HefmiIch2025-PatientPopulation — search paths in its folder
# under the data root
_HEFMI_ROOT = "${EEG_DATA_ROOT}/hefmiich2025"
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
