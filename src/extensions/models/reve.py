from __future__ import annotations

from benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_reve(spec: CheckpointSpec):
    from .backbones.reve import REVEBackbone

    return REVEBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="reve",
        description="REVE foundation encoder wrapper.",
        loader=_load_reve,
        checkpoints=[
            CheckpointSpec(
                identifier="reve_pretrained",
                model_family="reve",
                variant="pretrained",
                source_type="local",
                source_reference="brain-bzh/reve-base",
                checkpoint_path="artifacts/models/foundation/reve",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=200.0,
                expected_epoch_seconds=30.0,
                has_classifier_head=False,
                embedding_dim=512,
                wrapper_name="reve",
                status="ready",
                notes="REVE base (local). Input: (B,1,6000) at 200 Hz. Electrode positions from artifacts/models/foundation/reve-positions/.",
            ),
            CheckpointSpec(
                identifier="reve_random_init",
                model_family="reve",
                variant="random_init",
                source_type="random_init",
                source_reference="random_init",
                checkpoint_path="artifacts/models/foundation/reve",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=200.0,
                expected_epoch_seconds=30.0,
                has_classifier_head=False,
                embedding_dim=512,
                wrapper_name="reve",
                status="ready",
                notes="Random-init baseline — same REVE architecture, no pretraining. checkpoint_path locates the config; weights are not loaded. Position bank still loaded from artifacts/models/foundation/reve-positions/.",
            ),
        ],
    )
]
