"""EPILEPSIAE preprocessor — backward-compatible shim.

All implementation has moved to the ``epilepsiae/`` sub-package.
This module re-exports the full public API so existing imports continue to work.
"""
from __future__ import annotations

# Sub-package public API
from .epilepsiae import (  # noqa: F401
    ANNOTATION_DIR,
    DATA_ROOT,
    EXCLUDED_PATIENTS,
    GAP_THRESHOLD_S,
    PATTERN_CODES,
    PATTERN_NAMES,
    SEIZURE_TYPE_CODES,
    SEIZURE_TYPE_NAMES,
    VARIANTS,
    BlockMeta,
    EpilepsiAEPreprocessor,
    PatientMeta,
    ProcessedRecording,
    SeizureEvent,
    _find_patient_sql,
    _read_vlen,
    build_samplewise_labels,
    discover_block,
    load_block_signals,
    load_origin_annotations,
    load_patient_metadata,
    load_seizure_annotations,
    merge_shards,
    normalize_channel_name,
    read_head,
    select_canonical_channels,
    seizures_for_recording,
)
from .epilepsiae.readers import CHANNEL_ALIAS, _parse_timestamp  # noqa: F401

# Shared constants (from _common — re-exported for backward compat)
from ._common import (  # noqa: F401
    BIPOLAR_MONTAGE,
    BIPOLAR_NAMES,
    CACHE_SCHEMA_TAG,
    CANONICAL_19,
    CANONICAL_IDX,
    CANONICAL_SET,
    GAP_LABEL,
    TARGET_FS,
    _BIP_A,
    _BIP_B,
    apply_standard_filters,
    bipolar_from_unipolar,
    resample_to,
)

# Private backward-compatible aliases (used by dataio/epilepsiae.py)
_apply_filters = apply_standard_filters
_resample_to = resample_to
