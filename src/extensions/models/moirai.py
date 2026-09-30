from __future__ import annotations

from benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_moirai(spec: CheckpointSpec):
    from .backbones.moirai import MoiraiBackbone

    return MoiraiBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="moirai",
        description=(
            "MOIRAI (Salesforce) universal time-series foundation model. "
            "Embedding via encoder hidden states + mean-pool. "
            "Univariate: per-channel embed + mean-pool."
        ),
        loader=_load_moirai,
        checkpoints=[
            CheckpointSpec(
                identifier="moirai_small",
                model_family="moirai",
                variant="small",
                source_type="huggingface",
                source_reference="Salesforce/moirai-1.1-R-small",
                checkpoint_path="Salesforce/moirai-1.1-R-small",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=None,
                expected_epoch_seconds=30.0,
                has_classifier_head=False,
                embedding_key="encoder_mean_pool",
                embedding_dim=384,
                wrapper_name="moirai",
                status="ready",
                notes=(
                    "MOIRAI-1.1-R-small. Embedding extracted from "
                    "transformer encoder output, mean-pooled over patches. "
                    "Univariate: per-channel embed + mean-pool."
                ),
            ),
            CheckpointSpec(
                identifier="moirai_base",
                model_family="moirai",
                variant="base",
                source_type="huggingface",
                source_reference="Salesforce/moirai-1.1-R-base",
                checkpoint_path="Salesforce/moirai-1.1-R-base",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=None,
                expected_epoch_seconds=30.0,
                has_classifier_head=False,
                embedding_key="encoder_mean_pool",
                embedding_dim=768,
                wrapper_name="moirai",
                status="ready",
                notes=(
                    "MOIRAI-1.1-R-base. Embedding extracted from "
                    "transformer encoder output, mean-pooled over patches."
                ),
            ),
            CheckpointSpec(
                identifier="moirai_large",
                model_family="moirai",
                variant="large",
                source_type="huggingface",
                source_reference="Salesforce/moirai-1.1-R-large",
                checkpoint_path="Salesforce/moirai-1.1-R-large",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=None,
                expected_epoch_seconds=30.0,
                has_classifier_head=False,
                embedding_key="encoder_mean_pool",
                embedding_dim=1024,
                wrapper_name="moirai",
                status="ready",
                notes=(
                    "MOIRAI-1.1-R-large. Embedding extracted from "
                    "transformer encoder output, mean-pooled over patches."
                ),
            ),
        ],
    )
]
