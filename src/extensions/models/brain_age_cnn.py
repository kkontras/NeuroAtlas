from __future__ import annotations

from benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_brain_age_cnn(spec: CheckpointSpec):
    from .backbones.brain_age_cnn import BrainAgeCNNBackbone

    return BrainAgeCNNBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="brain_age_cnn",
        description="BrainAgeCNN_v2 feature extractor for brain age prediction.",
        loader=_load_brain_age_cnn,
        checkpoints=[
            CheckpointSpec(
                identifier="brain_age_cnn_v2_pretrained",
                model_family="brain_age_cnn",
                variant="pretrained",
                source_type="local",
                source_reference="SleepAgeBench/models/brain_age_cnn_v2.py",
                checkpoint_path=None,
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=100.0,
                expected_epoch_seconds=30.0,
                has_classifier_head=False,
                embedding_key="features",
                embedding_dim=256,
                wrapper_name="brain_age_cnn",
                status="planned",
                notes=(
                    "BrainAgeCNN_v2 4-layer CNN feature extractor from SleepAgeBench. "
                    "FeatureExtractor produces [B, 256] embeddings from [B, 1, 3000] raw input at 100Hz."
                ),
            ),
        ],
    )
]
