from __future__ import annotations

from benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_sleepfm(spec: CheckpointSpec):
    from .backbones.sleepfm import SleepFMBackbone

    return SleepFMBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="sleepfm",
        description="SleepFM foundation encoder (Thapa et al., ICML'24 / Nat Med'25) — EEG/BAS branch of the shared multimodal SetTransformer.",
        loader=_load_sleepfm,
        checkpoints=[
            CheckpointSpec(
                identifier="sleepfm_pretrained",
                model_family="sleepfm",
                variant="model_base",
                source_type="github_release",
                source_reference="https://github.com/zou-group/sleepfm-clinical",
                checkpoint_path="artifacts/models/foundation/sleepfm/best.pt",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=128.0,
                expected_epoch_seconds=10.0,
                pretraining_datasets=("stanford_psg_14k",),
                has_classifier_head=False,
                embedding_key="temporal_pooled",
                embedding_dim=128,
                wrapper_name="sleepfm",
                status="ready",
                notes=(
                    "Single-shared SetTransformer across modalities; we feed the EEG channel into the "
                    "BAS slot and zero-pad + mask out the other 9 slots. Other modalities "
                    "(ECG/EMG/Resp) are not queried."
                ),
            )
        ],
    )
]
