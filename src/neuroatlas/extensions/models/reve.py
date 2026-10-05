from __future__ import annotations

from neuroatlas.benchmarking_helpers import CheckpointSpec, ModelSpec


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
                # Public on the hub (no longer gated) under the REVE Responsible
                # Use License; `models download` fetches the repository and the
                # position bank brain-bzh/reve-positions next to it.
                source_type="huggingface",
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
                notes="REVE base from brain-bzh/reve-base (REVE Responsible Use License v1.0: downloading or using it accepts its terms). Input: (B,1,6000) at 200 Hz. Electrode positions from brain-bzh/reve-positions, kept in artifacts/models/foundation/reve-positions/.",
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
                notes="Random-init baseline — same REVE architecture, no pretraining. Needs reve-base's config and modelling code (not its weights) and the position bank; `neuroatlas models download reve_random_init` fetches them into artifacts/models/foundation/reve{,-positions}/.",
            ),
        ],
    )
]
