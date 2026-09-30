from __future__ import annotations

from neuroatlas.benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_core_sleep(spec: CheckpointSpec):
    from .backbones.core_sleep import CoreSleepBackbone

    return CoreSleepBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="core_sleep",
        description="CoRe-Sleep native-head and embedding wrapper.",
        loader=_load_core_sleep,
        checkpoints=[
            CheckpointSpec(
                identifier="core_sleep_shhs_fold0",
                model_family="core_sleep",
                variant="fold0",
                source_type="github_release_asset",
                source_reference="https://github.com/kkontras/NeuroAtlas/releases/download/supervised-baselines-v1/core_sleep_pretrained_fold0.pth.tar",
                checkpoint_path="artifacts/models/shhs/core_sleep_pretrained_fold0.pth.tar",
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
                wrapper_name="core_sleep",
                status="ready",
                notes="Existing in-repo SHHS checkpoint. Architecturally bimodal "
                "(EEG+EOG); this project supplies EEG only, and the wrapper runs true "
                "unimodal inference via skip_view='eog' — the EOG branch is never built "
                "and the EEG-only head preds['c'] / features['eeg'] is used.",
                runtime_overrides={
                    "stride": 21, "target_idx": "all",
                    "embedding_stride": 21, "embedding_target_idx": "all",
                    "sequential_batch_size": 128,
                },
            ),
            CheckpointSpec(
                identifier="core_sleep_shhs_fold0_seq1",
                model_family="core_sleep",
                variant="fold0_seq1",
                source_type="github_release_asset",
                source_reference="https://github.com/kkontras/NeuroAtlas/releases/download/supervised-baselines-v1/core_sleep_pretrained_fold0.pth.tar",
                checkpoint_path="artifacts/models/shhs/core_sleep_pretrained_fold0.pth.tar",
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
                wrapper_name="core_sleep",
                status="ready",
                notes="Same checkpoint as fold0 but with sequence_length=1 "
                "(independent epochs, no outer transformer context).",
                runtime_overrides={},
            ),
        ],
    )
]
