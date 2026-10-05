"""TUSZ v2.0.3 sub-package — public re-exports.

Import from the top-level shim or from here:

    from ..tusz_preprocessor import build_cache, merge_shards, merge_splits
    # ... or equivalently ...
    from ..tusz import build_cache, merge_shards, merge_splits
"""
from .annotations import (
    SEIZURE_TYPE_CODES,
    SEIZURE_TYPE_NAMES,
    build_event_arrays,
    build_samplewise_labels,
    build_samplewise_types,
    parse_csv_bi,
    parse_csv_multiclass,
)
from .preprocessor import (
    SPLITS,
    _CHUNK_SAMPLES,
    _copy_datasets_chunked,
    _process_one_recording,
    build_cache,
    merge_shards,
    merge_splits,
    summarize_cache,
)
from .readers import (
    BIPOLAR_MONTAGE,
    CACHE_SCHEMA_TAG,
    TARGET_FS,
    TCP_MONTAGE_CHANNELS,
    UNIPOLAR_ELECTRODES,
    bipolar_from_unipolar,
    discover_recordings,
    load_tuh_eeg_epilepsy_metadata,
    normalize_channel_name,
    parse_tuh_patient_field,
    read_edf_unipolar,
)

__all__ = [
    # Readers / montage
    "UNIPOLAR_ELECTRODES",
    "BIPOLAR_MONTAGE",
    "TCP_MONTAGE_CHANNELS",
    "TARGET_FS",
    "CACHE_SCHEMA_TAG",
    "normalize_channel_name",
    "bipolar_from_unipolar",
    "read_edf_unipolar",
    "discover_recordings",
    "load_tuh_eeg_epilepsy_metadata",
    # Annotations
    "SEIZURE_TYPE_CODES",
    "SEIZURE_TYPE_NAMES",
    "parse_csv_bi",
    "parse_csv_multiclass",
    "build_samplewise_labels",
    "build_samplewise_types",
    "build_event_arrays",
    # Preprocessor
    "SPLITS",
    "build_cache",
    "merge_shards",
    "merge_splits",
    "summarize_cache",
]
