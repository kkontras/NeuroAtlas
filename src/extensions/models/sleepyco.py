from __future__ import annotations

from benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_sleepyco(spec: CheckpointSpec):
    from .backbones.sleepyco import SleePyCoBackbone

    return SleePyCoBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="sleepyco",
        description="SHHS-finetuned SleePyCo: feature-pyramid CNN + Transformer head (Lee et al., ESWA 2024).",
        loader=_load_sleepyco,
        checkpoints=[
            CheckpointSpec(
                identifier="sleepyco_shhs_fold0",
                model_family="sleepyco",
                variant="fold0",
                source_type="local_artifact",
                source_reference="artifacts/models/shhs/sleepyco_pretrained_fold0.pth",
                checkpoint_path="artifacts/models/shhs/sleepyco_pretrained_fold0.pth",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=100.0,
                expected_epoch_seconds=30.0,
                expected_sequence_length=10,
                pretraining_datasets=("shhs",),
                finetuned_datasets=("shhs",),
                has_classifier_head=True,
                embedding_key="pooled",
                embedding_dim=128,
                wrapper_name="sleepyco",
                status="ready",
                notes="Upstream ckpt_fold-01.pth (SHHS freezefinetune, SL-10 numScales-3); "
                "DataParallel `module.` prefix stripped at load time.",
                runtime_overrides={
                    "stride": 1, "target_idx": -1,
                    "embedding_stride": 1, "embedding_target_idx": -1,
                    "sequential_batch_size": 256,
                },
            ),
            CheckpointSpec(
                identifier="sleepyco_shhs_fold0_seq1",
                model_family="sleepyco",
                variant="fold0_seq1",
                source_type="local_artifact",
                source_reference="artifacts/models/shhs/sleepyco_pretrained_fold0.pth",
                checkpoint_path="artifacts/models/shhs/sleepyco_pretrained_fold0.pth",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=100.0,
                expected_epoch_seconds=30.0,
                expected_sequence_length=1,
                pretraining_datasets=("shhs",),
                finetuned_datasets=("shhs",),
                has_classifier_head=True,
                embedding_key="pooled",
                embedding_dim=128,
                wrapper_name="sleepyco",
                status="ready",
                notes="Same checkpoint as fold0 but with sequence_length=1 "
                "(single 30 s epoch, no multi-epoch concatenation).",
                runtime_overrides={},
            ),
        ],
    )
]
