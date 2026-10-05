from __future__ import annotations

from neuroatlas.benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_seizure_transformer(spec: CheckpointSpec):
    from .backbones.seizure_transformer import SeizureTransformerBackbone

    return SeizureTransformerBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="seizure_transformer",
        description=(
            "SeizureTransformer (Wu et al. 2025) — U-Net + self-attention, "
            "time-step seizure detection. EpilepsyBench 2025 winner (1 FA/24h)."
        ),
        loader=_load_seizure_transformer,
        checkpoints=[
            CheckpointSpec(
                identifier="seizure_transformer_pretrained",
                model_family="seizure_transformer",
                variant="wu2025",
                # The authors publish the weights only inside their Docker image
                # (keruiwu/SeizureTransformer README: `docker pull
                # yujjio/seizure_transformer`); the download pulls model.pth
                # out of it over the registry API and checks its SHA-256.
                source_type="docker_image",
                source_reference="docker://yujjio/seizure_transformer:latest",
                checkpoint_path=(
                    "artifacts/models/supervised/seizure_transformer/model.pth"
                ),
                input_kind="time_series",
                expected_channels=("eeg",),
                expected_sampling_rate=256.0,
                # Paper trained on 60 s windows. The wrapper accepts any
                # epoch_seconds whose 256-Hz length is a multiple of 32 (the
                # encoder pools 5 stages of /2). Default to 10 s per
                # `feedback_window_10s_default` so the supervised baseline
                # benchmarks apples-to-apples with the FM 10-s probe pipeline;
                # the wrapper records `window_matches_paper=False` in metadata
                # so reports surface the deviation.
                expected_epoch_seconds=10.0,
                pretraining_datasets=("dianalund", "siena", "tusz_v2.0.3"),
                # Authors trained the released weights on Dianalund + Siena + TUSZ;
                # treat TUSZ, Siena (and their derivatives) as "home" datasets
                # where native_head_eval is the right comparison.
                finetuned_datasets=("tusz", "siena", "chbmit_siena"),
                has_classifier_head=True,
                embedding_key="trunk_mean",
                embedding_dim=512,
                wrapper_name="seizure_transformer",
                status="ready",
                notes=(
                    "Weights: model.pth from the authors' Docker image yujjio/seizure_transformer "
                    "(the only official distribution), sha256 79b14e47... "
                    "Native head emits per-sample seizure probability at 256 Hz; "
                    "wrapper reduces to per-window binary prob for the benchmarking "
                    "contract. Embedding = global-avg-pool of transformer trunk (512-d)."
                ),
            )
        ],
    )
]
