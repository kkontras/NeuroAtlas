"""Each model's own BCI input: the sampling rate, band and notch a model's
trials are prepared with.

A model reads its BCI trials in one of three formats, named after the model
it was first made for:

==============  ===========  =======  ===========================
format         band         rate     notch
=============  ===========  =======  ===========================
``labram``     0.1-75 Hz    200 Hz   50 Hz
``bendr``      0.5-70 Hz    256 Hz   the cohort's line frequency
``steegformer`` 0.1-64 Hz    128 Hz   the cohort's line frequency
=============  ===========  =======  ===========================

(the cognitive cohorts' files have a 50 Hz notch in every format). EEGPT
reads ``bendr``; ST-EEGFormer and SleepFM read ``steegformer``; every other
model reads ``labram``. Confound filtering keeps the format's rate and notch
and changes the band (``moabb_loader.CONFOUND_FILTERING``; the cognitive
cohorts: 4-40 Hz). The notch covers the line frequency and its harmonics
below the native Nyquist frequency.

Standard library only.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Optional


class Format(NamedTuple):
    fmin: float
    fmax: float
    rate: float
    notch: Optional[float]          # None: the cohort's line frequency


FORMATS: Dict[str, Format] = {
    "labram": Format(0.1, 75.0, 200.0, 50.0),
    "bendr": Format(0.5, 70.0, 256.0, None),
    "steegformer": Format(0.1, 64.0, 128.0, None),
}

#: model family -> format; any family not listed reads ``labram``
FAMILY_FORMAT: Dict[str, str] = {
    "eegpt": "bendr",
    "mirepnet": "bendr",
    "steegformer": "steegformer",
    "sleepfm": "steegformer",
    "unet1": "steegformer",
}
DEFAULT_FORMAT = "labram"


def format_for(family: Optional[str]) -> str:
    """The format a model family reads."""
    return FAMILY_FORMAT.get(str(family or "").lower(), DEFAULT_FORMAT)


def check_format(name: str) -> Format:
    if name not in FORMATS:
        raise ValueError(f"bci_format must be one of {', '.join(FORMATS)}, got {name!r}\n"
                         f"fix: --set bci_format={DEFAULT_FORMAT}")
    return FORMATS[name]


def line_frequency(slug: str) -> float:
    """The cohort's power-line frequency: its row of ``dataio/bci.py``
    DATASET_CONFIGS when it has one, else its manifest's
    ``cohort.powerline_hz``, else 50 Hz."""
    try:
        from neuroatlas.extensions.datasets.dataio.bci import DATASET_CONFIGS

        for key in (f"{slug}_eegpt", slug):
            cfg = DATASET_CONFIGS.get(key)
            if cfg is not None and cfg.notch_freq:
                return float(cfg.notch_freq)
    except ImportError:
        pass
    try:
        from neuroatlas.benchmarking_helpers.registry.manifest import load_manifest

        hz = ((load_manifest(slug) or {}).get("cohort") or {}).get("powerline_hz")
        if hz:
            return float(hz)
    except Exception:
        pass
    return 50.0


def uses_formats(dataset: str) -> bool:
    """Whether *dataset* is read per model format: a MOABB cohort or a
    bci_cognitive cohort."""
    from neuroatlas.extensions.datasets.dataio.bci_paths import COGNITIVE_PICKLES

    if dataset in COGNITIVE_PICKLES:
        return True
    try:
        from neuroatlas.extensions.datasets.dataio.moabb_loader import MOABB_DATASETS
    except ImportError:
        return False
    return dataset in MOABB_DATASETS
