from __future__ import annotations

from benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_steegformer(spec: CheckpointSpec):
    from .backbones.steegformer import STEEGFormerBackbone

    return STEEGFormerBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="steegformer",
        description="ST-EEGFormer foundation encoder wrapper (Yang et al., ICLR 2026).",
        loader=_load_steegformer,
        checkpoints=[
            CheckpointSpec(
                identifier="steegformer_small",
                model_family="steegformer",
                variant="small",
                source_type="github_release_asset",
                source_reference="https://github.com/LiuyinYang1101/STEEGFormer/releases/download/ST-EEGFormer-small/checkpoint-300.pth",
                checkpoint_path="artifacts/models/foundation/steegformer/steegformer-small-checkpoint-300.pth",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=128.0,
                expected_epoch_seconds=30.0,
                pretraining_datasets=("mixed_eeg_8M",),
                has_classifier_head=False,
                embedding_dim=512,
                wrapper_name="steegformer",
                status="ready",
                notes=(
                    "ST-EEGFormer small (8 layers, 512-d). MAE pretrained on "
                    "8M+ EEG segments at 128 Hz with 142-channel vocabulary. "
                    "Patch size 16 samples (0.125 s). CLS token -> (B, 512)."
                ),
            ),
            CheckpointSpec(
                identifier="steegformer_base",
                model_family="steegformer",
                variant="base",
                source_type="github_release_asset",
                source_reference="https://github.com/LiuyinYang1101/STEEGFormer/releases/download/ST-EEGFormer-base/checkpoint-288.pth",
                checkpoint_path="artifacts/models/foundation/steegformer/steegformer-base-checkpoint-288.pth",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=128.0,
                expected_epoch_seconds=30.0,
                pretraining_datasets=("mixed_eeg_8M",),
                has_classifier_head=False,
                embedding_dim=768,
                wrapper_name="steegformer",
                status="ready",
                notes=(
                    "ST-EEGFormer base (12 layers, 768-d). MAE pretrained on "
                    "8M+ EEG segments at 128 Hz with 142-channel vocabulary. "
                    "Patch size 16 samples (0.125 s). CLS token -> (B, 768)."
                ),
            ),
            CheckpointSpec(
                identifier="steegformer_large",
                model_family="steegformer",
                variant="large",
                source_type="github_release_asset",
                source_reference="https://github.com/LiuyinYang1101/STEEGFormer/releases/download/ST-EEGFormer-large/large_weights_only_196.pth",
                checkpoint_path="artifacts/models/foundation/steegformer/steegformer-large-checkpoint-196.pth",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=128.0,
                expected_epoch_seconds=30.0,
                pretraining_datasets=("mixed_eeg_8M",),
                has_classifier_head=False,
                embedding_dim=1024,
                wrapper_name="steegformer",
                status="ready",
                notes=(
                    "ST-EEGFormer large (24 layers, 1024-d). MAE pretrained on "
                    "8M+ EEG segments at 128 Hz with 142-channel vocabulary. "
                    "Patch size 16 samples (0.125 s). CLS token -> (B, 1024). "
                    "Checkpoint is weights-only (OrderedDict, no optimizer state)."
                ),
            ),
        ],
    )
]
