from __future__ import annotations

from neuroatlas.benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_neurolm(spec: CheckpointSpec):
    from .backbones.neurolm import NeuroLMBackbone

    return NeuroLMBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="neurolm",
        description="NeuroLM foundation encoder wrapper.",
        loader=_load_neurolm,
        checkpoints=[
            CheckpointSpec(
                identifier="neurolm_vq_pretrained",
                model_family="neurolm",
                variant="vq_tokenizer",
                source_type="huggingface",
                source_reference="Weibang/NeuroLM",
                checkpoint_path="artifacts/models/foundation/neurolm/checkpoints/VQ.pt",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=200.0,
                expected_epoch_seconds=30.0,
                pretraining_datasets=("mixed_eeg_corpus",),
                has_classifier_head=False,
                embedding_key="vq_encoder_mean_tokens",
                embedding_dim=768,
                wrapper_name="neurolm",
                status="ready",
                notes="Frozen NeuroLM VQ encoder probing path using 200-sample patch tokenization.",
            )
        ],
    )
]
