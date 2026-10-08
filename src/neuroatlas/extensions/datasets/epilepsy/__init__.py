"""Epilepsy preprocessing package.

Four dataset preprocessors that all produce the same continuous-HDF5 cache
schema (19 unipolar 10-20 channels at 256 Hz with samplewise binary and
multi-class seizure labels):

- :mod:`.chbmit_preprocessor`   — CHB-MIT (BIDS from Zenodo)
- :mod:`.siena_preprocessor`    — Siena Scalp EEG (BIDS from Zenodo)
- :mod:`.epilepsiae_preprocessor` → :mod:`.epilepsiae` sub-package
- :mod:`.tusz_preprocessor`     → :mod:`.tusz` sub-package

Shared infrastructure:

- :mod:`._common`      — canonical montage, resample, standard filters
- :mod:`.bids_index`   — BIDS scanning, ``parse_bids_events``

See ``README.md`` in this directory for the full architecture overview.
"""

# Shared constants/helpers (from _common)
from ._common import (  # noqa: F401
    BIPOLAR_MONTAGE,
    BIPOLAR_NAMES,
    CACHE_SCHEMA_TAG,
    CANONICAL_19,
    CANONICAL_IDX,
    CANONICAL_SET,
    GAP_LABEL,
    TARGET_FS,
    apply_standard_filters,
    bipolar_from_unipolar,
    resample_to,
)

# EPILEPSIAE
from .epilepsiae_preprocessor import (  # noqa: F401
    BIPOLAR_MONTAGE as EPILEPSIAE_BIPOLAR_MONTAGE,
    BIPOLAR_NAMES as EPILEPSIAE_BIPOLAR_NAMES,
    CANONICAL_19 as EPILEPSIAE_CANONICAL_19,
    CHANNEL_ALIAS as EPILEPSIAE_CHANNEL_ALIAS,
    EpilepsiAEPreprocessor,
    PATTERN_CODES,
    PATTERN_NAMES,
    SEIZURE_TYPE_CODES as EPILEPSIAE_SEIZURE_TYPE_CODES,
    SEIZURE_TYPE_NAMES as EPILEPSIAE_SEIZURE_TYPE_NAMES,
    build_samplewise_labels,
    load_origin_annotations,
    load_patient_metadata,
    load_seizure_annotations,
    merge_shards,
    normalize_channel_name,
    seizures_for_recording,
    select_canonical_channels,
)

# TUSZ
from .tusz_preprocessor import (  # noqa: F401
    SPLITS,
    TCP_MONTAGE_CHANNELS,
    UNIPOLAR_ELECTRODES,
)

# SeizeIt2
from .sz2_preprocessor import (  # noqa: F401
    SZ2_DATA_ROOT,
    SZ2_CHANNEL_ALIAS,
    build_h5_cache as sz2_build_h5_cache,
    discover_recordings as sz2_discover_recordings,
)

# SeizeIt1
from .sz1_preprocessor import (  # noqa: F401
    SZ1_DATA_ROOT,
    SZ1_CHANNEL_ALIAS,
    build_h5_cache as sz1_build_h5_cache,
    discover_recordings as sz1_discover_recordings,
)
