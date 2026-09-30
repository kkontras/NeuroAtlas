from __future__ import annotations

from benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_neurogpt(spec: CheckpointSpec):
    from .backbones.neurogpt import NeuroGPTBackbone

    return NeuroGPTBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="neurogpt",
        description="NeuroGPT EEG Conformer encoder (Cui et al., IEEE ISBI 2024).",
        loader=_load_neurogpt,
        checkpoints=[
            CheckpointSpec(
                identifier="neurogpt_pretrained",
                model_family="neurogpt",
                variant="encoder_pretrained",
                source_type="huggingface",
                source_reference="wenhuic/Neuro-GPT",
                checkpoint_path="artifacts/models/foundation/neurogpt/pytorch_model.bin",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=250.0,
                expected_epoch_seconds=30.0,
                pretraining_datasets=("tuh_eeg",),
                has_classifier_head=False,
                embedding_key="encoder_mean_chunks",
                embedding_dim=1080,
                wrapper_name="neurogpt",
                status="ready",
                notes=(
                    "NeuroGPT encoder-only extraction. 22-channel fixed layout "
                    "(legacy 10-20 naming). 250 Hz, 2s chunks (500 samples). "
                    "Per-channel z-score normalization (scale-invariant). "
                    "1080-dim output (27 tokens x 40 features). "
                    "Paper: Cui et al., IEEE ISBI 2024."
                ),
            ),
        ],
    )
]
