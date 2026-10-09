"""EPILEPSIAE sub-package — public re-exports.

Import from here instead of the individual sub-modules:

    from .epilepsiae import EpilepsiAEPreprocessor, merge_shards
    from .epilepsiae import SeizureEvent, BlockMeta, PatientMeta
"""
from .annotations import (
    SeizureEvent,
    _find_patient_sql,
    load_origin_annotations,
    load_patient_metadata,
    load_seizure_annotations,
    seizures_for_recording,
)
from .labels import (
    PATTERN_CODES,
    PATTERN_NAMES,
    SEIZURE_TYPE_CODES,
    SEIZURE_TYPE_NAMES,
    build_samplewise_labels,
)
from .preprocessor import (
    ANNOTATION_DIR,
    DATA_ROOT,
    EXCLUDED_PATIENTS,
    GAP_THRESHOLD_S,
    VARIANTS,
    EpilepsiAEPreprocessor,
    _read_vlen,
    merge_shards,
)
from .readers import (
    BlockMeta,
    PatientMeta,
    ProcessedRecording,
    discover_block,
    load_block_signals,
    normalize_channel_name,
    read_head,
    select_canonical_channels,
)

__all__ = [
    # Annotations
    "SeizureEvent",
    "load_seizure_annotations",
    "load_origin_annotations",
    "seizures_for_recording",
    "load_patient_metadata",
    # Labels
    "SEIZURE_TYPE_CODES",
    "SEIZURE_TYPE_NAMES",
    "PATTERN_CODES",
    "PATTERN_NAMES",
    "build_samplewise_labels",
    # Readers / data classes
    "BlockMeta",
    "PatientMeta",
    "ProcessedRecording",
    "read_head",
    "load_block_signals",
    "discover_block",
    "normalize_channel_name",
    "select_canonical_channels",
    # Preprocessor
    "DATA_ROOT",
    "ANNOTATION_DIR",
    "VARIANTS",
    "GAP_THRESHOLD_S",
    "EXCLUDED_PATIENTS",
    "EpilepsiAEPreprocessor",
    "merge_shards",
]
