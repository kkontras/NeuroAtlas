from __future__ import annotations

from neuroatlas.benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_labram(spec: CheckpointSpec):
    from .backbones.labram import LabramBackbone

    return LabramBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="labram",
        description="LaBraM foundation encoder wrapper.",
        loader=_load_labram,
        checkpoints=[
            CheckpointSpec(
                identifier="labram_pretrained",
                model_family="labram",
                variant="pretrained",
                source_type="github_release",
                source_reference="https://github.com/935963004/LaBraM",
                checkpoint_path="artifacts/models/foundation/labram-base.pth",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=200.0,
                expected_epoch_seconds=10.0,
                has_classifier_head=False,
                wrapper_name="labram",
                status="ready",
                notes=(
                    "Upstream LaBraM base checkpoint at 10 s windows. "
                    "Pretrained temporal_embedding has 16 rows (1 CLS + 15 "
                    "patches @ 1 s, 200 Hz); wrapper slices to the first 11 "
                    "rows for the 10-patch runtime. Wrapper resamples runtime "
                    "data to 200 Hz and reads fs from meta."
                ),
            )
        ],
    )
]
