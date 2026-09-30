from __future__ import annotations

from benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_neurorvq(spec: CheckpointSpec):
    from .backbones.neurorvq import NeuroRVQBackbone

    return NeuroRVQBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="neurorvq",
        description="NeuroRVQ multi-scale EEG tokenizer foundation model.",
        loader=_load_neurorvq,
        checkpoints=[
            CheckpointSpec(
                identifier="neurorvq_eeg_pretrained",
                model_family="neurorvq",
                variant="eeg_v1",
                source_type="huggingface",
                source_reference="ntinosbarmpas/NeuroRVQ",
                checkpoint_path="artifacts/models/foundation/neurorvq/NeuroRVQ_EEG_foundation_model_v1.pt",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=200.0,
                expected_epoch_seconds=4.0,
                pretraining_datasets=("mixed_eeg_corpus",),
                has_classifier_head=False,
                embedding_key="multi_scale_concat_mean_pool",
                embedding_dim=800,
                wrapper_name="neurorvq",
                status="ready",
                notes=(
                    "NeuroRVQ EEG v1 foundation model (6M params). "
                    "4-branch inception-style multi-scale temporal convolution "
                    "feeding a 12-layer transformer. Inputs resampled to 200 Hz, "
                    "patched into 200-sample windows. Embeddings are the "
                    "concatenation of 4 branch outputs mean-pooled over tokens "
                    "(800-dim). 99-channel pretrained spatial vocabulary."
                ),
            ),
        ],
    ),
]
