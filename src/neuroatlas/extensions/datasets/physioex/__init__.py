"""PhysioEx data module public API."""
# The legacy dataset.py/datareader.py pair that this comment used to describe
# was removed: nothing imported it, including the datasets/ package below.
from neuroatlas.extensions.datasets.physioex.pipeline import (
    PreprocessingStep, PreprocessingPipeline, CompiledPipeline, CompiledStep,
)
from neuroatlas.extensions.datasets.physioex.steps import (
    Identity, BandpassFilter, NotchFilter, Resample,
    ZScoreNormalize, XSleepNetSpectrogram,
)
from neuroatlas.extensions.datasets.physioex.presets import get_preset, available_presets, PRESETS
from neuroatlas.extensions.datasets.physioex.base import BasePhysioDataset, SubjectSpec
from neuroatlas.extensions.datasets.physioex.cache import ChannelCache, SCHEMA_VERSION
from neuroatlas.extensions.datasets.physioex.collate import dict_collate_fn, stack_channels, is_dict_batch
from neuroatlas.extensions.datasets.physioex.readers import (
    EDFHeader, ResolvedChannel, ChannelNotAvailableError,
    DEFAULT_PREFERENCES,
    probe_edf_header, resolve_channels,
)
from neuroatlas.extensions.datasets.physioex.events import SleepEvent, map_events_to_epochs, events_to_dicts, dicts_to_events

__all__ = [
    # Pipeline
    "PreprocessingStep", "PreprocessingPipeline", "CompiledPipeline", "CompiledStep",
    # Steps
    "Identity", "BandpassFilter", "NotchFilter", "Resample",
    "ZScoreNormalize", "XSleepNetSpectrogram",
    # Presets
    "get_preset", "available_presets", "PRESETS",
    # Dataset base
    "BasePhysioDataset", "SubjectSpec",
    # Cache
    "ChannelCache", "SCHEMA_VERSION",
    # Collate
    "dict_collate_fn", "stack_channels", "is_dict_batch",
    # Readers
    "EDFHeader", "ResolvedChannel", "ChannelNotAvailableError",
    "DEFAULT_PREFERENCES",
    "probe_edf_header", "resolve_channels",
    # Events
    "SleepEvent", "map_events_to_epochs", "events_to_dicts", "dicts_to_events",
]
