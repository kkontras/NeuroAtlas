from __future__ import annotations

from neuroatlas.benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_sleep_transformer(spec: CheckpointSpec):
    from .backbones.sleep_transformer import SleepTransformerBackbone

    return SleepTransformerBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="sleep_transformer",
        description="SHHS-pretrained EEG-only SleepEnc (CoRe-Sleep single-stream variant).",
        loader=_load_sleep_transformer,
        checkpoints=[
            CheckpointSpec(
                identifier="sleep_transformer_shhs_fold0",
                model_family="sleep_transformer",
                variant="fold0",
                source_type="github_release_asset",
                source_reference="https://github.com/kkontras/NeuroAtlas/releases/download/supervised-baselines-v1/sleep_transformer_pretrained_fold0.pth.tar",
                checkpoint_path="artifacts/models/shhs/sleep_transformer_pretrained_fold0.pth.tar",
                input_kind="time_frequency",
                expected_channels=("stft_eeg",),
                expected_sampling_rate=100.0,
                expected_epoch_seconds=30.0,
                expected_sequence_length=21,
                pretraining_datasets=("shhs",),
                finetuned_datasets=("shhs",),
                has_classifier_head=True,
                embedding_key="combined",
                embedding_dim=128,
                wrapper_name="sleep_transformer",
                status="ready",
                notes="EEG-only SleepEnc trunk from Trui/unimodal_eeg_eoe_fold0 (Sleep-CoRe pretraining run).",
                runtime_overrides={
                    "stride": 1, "target_idx": "all",
                    "embedding_stride": 1, "embedding_target_idx": 10,
                    "sequential_batch_size": 128,
                },
            ),
            CheckpointSpec(
                identifier="sleep_transformer_shhs_fold0_seq1",
                model_family="sleep_transformer",
                variant="fold0_seq1",
                source_type="github_release_asset",
                source_reference="https://github.com/kkontras/NeuroAtlas/releases/download/supervised-baselines-v1/sleep_transformer_pretrained_fold0.pth.tar",
                checkpoint_path="artifacts/models/shhs/sleep_transformer_pretrained_fold0.pth.tar",
                input_kind="time_frequency",
                expected_channels=("stft_eeg",),
                expected_sampling_rate=100.0,
                expected_epoch_seconds=30.0,
                expected_sequence_length=1,
                pretraining_datasets=("shhs",),
                finetuned_datasets=("shhs",),
                has_classifier_head=True,
                embedding_key="combined",
                embedding_dim=128,
                wrapper_name="sleep_transformer",
                status="ready",
                notes="Same checkpoint as fold0 but with sequence_length=1 "
                "(independent epochs, outer transformer sees single token).",
                runtime_overrides={},
            ),
        ],
    )
]
