from __future__ import annotations

from neuroatlas.benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_moment(spec: CheckpointSpec):
    from .backbones.moment import MomentBackbone

    return MomentBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="moment",
        description=(
            "MOMENT (CMU/AutonLab) time-series foundation model with native "
            "embedding head. Univariate: per-channel embed + mean-pool."
        ),
        loader=_load_moment,
        checkpoints=[
            CheckpointSpec(
                identifier="moment_small",
                model_family="moment",
                variant="small",
                source_type="huggingface",
                source_reference="AutonLab/MOMENT-1-small",
                checkpoint_path="AutonLab/MOMENT-1-small",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=None,
                expected_epoch_seconds=30.0,
                has_classifier_head=False,
                embedding_key="moment_embedding",
                embedding_dim=512,
                wrapper_name="moment",
                status="ready",
                notes=(
                    "MOMENT-1-small (T5-small encoder). Univariate model: "
                    "each EEG channel is embedded independently, then "
                    "mean-pooled across channels. Input padded/truncated "
                    "to a multiple of patch_size=8."
                ),
            ),
            CheckpointSpec(
                identifier="moment_base",
                model_family="moment",
                variant="base",
                source_type="huggingface",
                source_reference="AutonLab/MOMENT-1-base",
                checkpoint_path="AutonLab/MOMENT-1-base",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=None,
                expected_epoch_seconds=30.0,
                has_classifier_head=False,
                embedding_key="moment_embedding",
                embedding_dim=768,
                wrapper_name="moment",
                status="ready",
                notes=(
                    "MOMENT-1-base (T5-base encoder). Univariate model: "
                    "each EEG channel is embedded independently, then "
                    "mean-pooled across channels."
                ),
            ),
            CheckpointSpec(
                identifier="moment_large",
                model_family="moment",
                variant="large",
                source_type="huggingface",
                source_reference="AutonLab/MOMENT-1-large",
                checkpoint_path="AutonLab/MOMENT-1-large",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=None,
                expected_epoch_seconds=30.0,
                has_classifier_head=False,
                embedding_key="moment_embedding",
                embedding_dim=1024,
                wrapper_name="moment",
                status="ready",
                notes=(
                    "MOMENT-1-large (T5-large encoder). Univariate model: "
                    "each EEG channel is embedded independently, then "
                    "mean-pooled across channels."
                ),
            ),
        ],
    )
]
