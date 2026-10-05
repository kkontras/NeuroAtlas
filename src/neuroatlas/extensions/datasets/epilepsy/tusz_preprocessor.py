"""TUSZ preprocessor — backward-compatible shim.

All implementation has moved to the ``tusz/`` sub-package.  This module
re-exports the full public API so existing imports keep working.
"""
from __future__ import annotations

from .tusz import (  # noqa: F401
    BIPOLAR_MONTAGE,
    CACHE_SCHEMA_TAG,
    SEIZURE_TYPE_CODES,
    SEIZURE_TYPE_NAMES,
    SPLITS,
    TARGET_FS,
    TCP_MONTAGE_CHANNELS,
    UNIPOLAR_ELECTRODES,
    bipolar_from_unipolar,
    build_cache,
    build_event_arrays,
    build_samplewise_labels,
    build_samplewise_types,
    merge_shards,
    merge_splits,
    normalize_channel_name,
    parse_csv_bi,
    parse_csv_multiclass,
    read_edf_unipolar,
    summarize_cache,
)
from .tusz import (  # noqa: F401  — private re-exports
    _CHUNK_SAMPLES,
    _copy_datasets_chunked,
    _process_one_recording,
)
from .tusz.readers import _ELECTRODE_ALIASES  # noqa: F401
from .tusz.readers import (  # noqa: F401
    discover_recordings as _discover_recordings,
    load_tuh_eeg_epilepsy_metadata,
    read_edf_unipolar as _read_edf_unipolar,
)

# Backward-compatible private alias: original code had
# ``_normalize_channel_name``. The public name is the canonical form now;
# keep the leading-underscore alias for any external import still using it.
_normalize_channel_name = normalize_channel_name
