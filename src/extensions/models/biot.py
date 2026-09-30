from __future__ import annotations

from benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_biot(spec: CheckpointSpec):
    from .backbones.biot import BIOTBackbone

    return BIOTBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="biot",
        description="BIOT foundation encoder wrapper.",
        loader=_load_biot,
        checkpoints=[
            CheckpointSpec(
                identifier="biot_pretrained",
                model_family="biot",
                variant="prest_16_channels",
                source_type="github_release",
                source_reference="https://github.com/ycq091044/BIOT",
                checkpoint_path="artifacts/models/foundation/EEG-PREST-16-channels.ckpt",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=200.0,
                expected_epoch_seconds=30.0,
                pretraining_datasets=("prest",),
                has_classifier_head=False,
                embedding_key="encoder_embedding",
                embedding_dim=256,
                wrapper_name="biot",
                status="ready",
                notes="Official BIOT PREST encoder checkpoint; clean-transfer candidate because it excludes SHHS.",
            )
        ],
    )
]
