from __future__ import annotations

from neuroatlas.benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_cbramod(spec: CheckpointSpec):
    from .backbones.cbramod import CBraModBackbone

    return CBraModBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="cbramod",
        description="CBraMod foundation encoder wrapper.",
        loader=_load_cbramod,
        checkpoints=[
            CheckpointSpec(
                identifier="cbramod_pretrained",
                model_family="cbramod",
                variant="pretrained",
                source_type="huggingface",
                source_reference="braindecode/cbramod-pretrained",
                checkpoint_path="braindecode/cbramod-pretrained",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=200.0,
                expected_epoch_seconds=10.0,
                pretraining_datasets=("tueg",),
                has_classifier_head=False,
                embedding_dim=200,
                wrapper_name="cbramod",
                status="ready",
                notes=(
                    "CBraMod criss-cross transformer pretrained on TUEG. ~4M params. "
                    "ACPE supports variable window length (positive multiple of 1 s). "
                    "Wrapper feeds (B, C, 2000) at 200 Hz for the 10 s default and "
                    "mean-pools the (B, C, 10, 200) encoder output to (B, 200)."
                ),
            ),
            CheckpointSpec(
                identifier="cbramod_random_init",
                model_family="cbramod",
                variant="random_init",
                source_type="random_init",
                source_reference="random_init",
                checkpoint_path=None,
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=200.0,
                expected_epoch_seconds=10.0,
                pretraining_datasets=(),
                has_classifier_head=False,
                embedding_dim=200,
                wrapper_name="cbramod",
                status="ready",
                notes="Random-init baseline — same CBraMod architecture, no pretraining.",
            ),
        ],
    )
]
