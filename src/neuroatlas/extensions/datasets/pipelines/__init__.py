"""Per-model preprocessing pipelines for sleep-staging benchmarks.

Each pipeline applies:
1. High-pass filter at 0.5 Hz (artifact removal, per info.md)
2. Notch filter at 50 Hz (powerline removal, per info.md)
3. Resample to the backbone's native sampling rate

Backbone-specific amplitude scaling (÷100, clip, z-score, etc.) is
handled inside each backbone's _prepare_input() per MODEL_CONTRACTS.md.
"""
from __future__ import annotations

from typing import Callable, Dict

from neuroatlas.extensions.datasets.physioex.pipeline import PreprocessingPipeline
from neuroatlas.extensions.datasets.physioex.steps import HighPassFilter, NotchFilter, Resample


def _make_pipeline(target_fs: float) -> PreprocessingPipeline:
    """Standard pipeline: highpass 0.5 Hz + notch 50 Hz + resample."""
    return PreprocessingPipeline([
        HighPassFilter(cutoff=0.5, order=5),
        NotchFilter(freq=50.0, quality=30.0),
        Resample(target_fs=target_fs),
    ])


def biot_pipeline() -> PreprocessingPipeline:
    """BIOT: 200 Hz."""
    return _make_pipeline(200.0)


def cbramod_pipeline() -> PreprocessingPipeline:
    """CBraMod: 200 Hz."""
    return _make_pipeline(200.0)


def labram_pipeline() -> PreprocessingPipeline:
    """LaBraM: 200 Hz."""
    return _make_pipeline(200.0)


def neurolm_pipeline() -> PreprocessingPipeline:
    """NeuroLM: 200 Hz."""
    return _make_pipeline(200.0)


def sleepfm_pipeline() -> PreprocessingPipeline:
    """SleepFM: 128 Hz."""
    return _make_pipeline(128.0)


def eegpt_pipeline() -> PreprocessingPipeline:
    """EEGPT: 256 Hz."""
    return _make_pipeline(256.0)


def reve_pipeline() -> PreprocessingPipeline:
    """REVE: 200 Hz."""
    return _make_pipeline(200.0)


def legacy_pipeline() -> PreprocessingPipeline:
    """Legacy: highpass 0.5 Hz + notch 50 Hz + resample 100 Hz."""
    return _make_pipeline(100.0)


def neurorvq_pipeline() -> PreprocessingPipeline:
    """NeuroRVQ: 200 Hz."""
    return _make_pipeline(200.0)


def steegformer_pipeline() -> PreprocessingPipeline:
    """STEEGFormer: 128 Hz."""
    return _make_pipeline(128.0)


def neurogpt_pipeline() -> PreprocessingPipeline:
    """NeuroGPT: 250 Hz."""
    return _make_pipeline(250.0)


def identity_pipeline() -> PreprocessingPipeline:
    """No preprocessing — pass signal through unchanged."""
    return PreprocessingPipeline([])


def resample100_pipeline() -> PreprocessingPipeline:
    """Resample to 100 Hz only (no filtering). For supervised sleep models."""
    return PreprocessingPipeline([Resample(target_fs=100.0)])


PIPELINE_REGISTRY: Dict[str, Callable[[], PreprocessingPipeline]] = {
    "biot": biot_pipeline,
    "cbramod": cbramod_pipeline,
    "labram": labram_pipeline,
    "neurolm": neurolm_pipeline,
    "sleepfm": sleepfm_pipeline,
    "eegpt": eegpt_pipeline,
    "reve": reve_pipeline,
    "neurorvq": neurorvq_pipeline,
    "steegformer": steegformer_pipeline,
    "neurogpt": neurogpt_pipeline,
    "legacy": legacy_pipeline,
    "identity": identity_pipeline,
    "resample100": resample100_pipeline,
}
