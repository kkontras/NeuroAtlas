from __future__ import annotations

from benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_chronos(spec: CheckpointSpec):
    from .backbones.chronos import ChronosBackbone

    return ChronosBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="chronos",
        description=(
            "Chronos (Amazon) T5-based time-series foundation model with "
            "native encoder embedding. Univariate: per-channel embed + mean-pool."
        ),
        loader=_load_chronos,
        checkpoints=[
            CheckpointSpec(
                identifier="chronos_t5_tiny",
                model_family="chronos",
                variant="t5_tiny",
                source_type="huggingface",
                source_reference="amazon/chronos-t5-tiny",
                checkpoint_path="amazon/chronos-t5-tiny",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=None,
                expected_epoch_seconds=30.0,
                has_classifier_head=False,
                embedding_key="encoder_mean_pool",
                embedding_dim=256,
                wrapper_name="chronos",
                status="ready",
                notes=(
                    "Chronos-T5-tiny. Univariate model: each EEG channel is "
                    "embedded via quantile-bin tokenisation + T5 encoder, then "
                    "mean-pooled across channels."
                ),
            ),
            CheckpointSpec(
                identifier="chronos_t5_small",
                model_family="chronos",
                variant="t5_small",
                source_type="huggingface",
                source_reference="amazon/chronos-t5-small",
                checkpoint_path="amazon/chronos-t5-small",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=None,
                expected_epoch_seconds=30.0,
                has_classifier_head=False,
                embedding_key="encoder_mean_pool",
                embedding_dim=512,
                wrapper_name="chronos",
                status="ready",
                notes=(
                    "Chronos-T5-small. Univariate model: each EEG channel is "
                    "embedded via quantile-bin tokenisation + T5 encoder, then "
                    "mean-pooled across channels."
                ),
            ),
            CheckpointSpec(
                identifier="chronos_t5_base",
                model_family="chronos",
                variant="t5_base",
                source_type="huggingface",
                source_reference="amazon/chronos-t5-base",
                checkpoint_path="amazon/chronos-t5-base",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=None,
                expected_epoch_seconds=30.0,
                has_classifier_head=False,
                embedding_key="encoder_mean_pool",
                embedding_dim=768,
                wrapper_name="chronos",
                status="ready",
                notes=(
                    "Chronos-T5-base. Univariate model: each EEG channel is "
                    "embedded via quantile-bin tokenisation + T5 encoder, then "
                    "mean-pooled across channels."
                ),
            ),
            CheckpointSpec(
                identifier="chronos_t5_large",
                model_family="chronos",
                variant="t5_large",
                source_type="huggingface",
                source_reference="amazon/chronos-t5-large",
                checkpoint_path="amazon/chronos-t5-large",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=None,
                expected_epoch_seconds=30.0,
                has_classifier_head=False,
                embedding_key="encoder_mean_pool",
                embedding_dim=1024,
                wrapper_name="chronos",
                status="ready",
                notes=(
                    "Chronos-T5-large. Univariate model: each EEG channel is "
                    "embedded via quantile-bin tokenisation + T5 encoder, then "
                    "mean-pooled across channels."
                ),
            ),
        ],
    )
]
