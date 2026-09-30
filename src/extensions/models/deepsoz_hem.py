from __future__ import annotations

from benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_deepsoz_hem(spec: CheckpointSpec):
    from .backbones.deepsoz_hem import DeepSOZHEMBackbone

    return DeepSOZHEMBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="deepsoz_hem",
        description=(
            "DeepSOZ-HEM (Shama et al. MICCAI 2023; SzCORE 2025 #4) — "
            "channel transformer + LSTM + MIL-pooled SOZ head, hard-example-mining "
            "variant. Pretrained on TUSZ fold-4."
        ),
        loader=_load_deepsoz_hem,
        checkpoints=[
            CheckpointSpec(
                identifier="deepsoz_hem_pretrained",
                model_family="deepsoz_hem",
                variant="amruth_sn_fold4",
                source_type="vendored",
                source_reference=(
                    "src/extensions/models/backbones/third_party/"
                    "deepsoz_hem/deepsoz_fold4.pth_4.tar"
                ),
                checkpoint_path=(
                    "src/extensions/models/backbones/third_party/"
                    "deepsoz_hem/deepsoz_fold4.pth_4.tar"
                ),
                input_kind="time_series",
                expected_channels=("eeg",),
                expected_sampling_rate=256.0,
                # Spec default mirrors the pretrain context (600 s); the wrapper
                # accepts any positive integer T at fs=256 Hz (see
                # backbones/deepsoz_hem.py — fs is architecturally hard, T is
                # not). The 10 s sweep overrides this via
                # `--expected-epoch-seconds 10` at the entrypoint.
                expected_epoch_seconds=600.0,
                pretraining_datasets=("tusz_v2.0.3", "siena"),
                # Authors trained on TUSZ + Siena; CHB-MIT explicitly excluded
                # because the channel transformer was trained on average-
                # referenced data, not bipolar pairs. native_head_eval is the
                # right comparison on TUSZ + Siena.
                finetuned_datasets=("tusz", "siena"),
                has_classifier_head=True,
                embedding_key="tx_global_mean",
                embedding_dim=256,
                wrapper_name="deepsoz_hem",
                expected_montage="unipolar_average_ref",
                status="ready",
                notes=(
                    "Vendored from amruth-sn/deepsoz-hem (GPL-3.0). Native head "
                    "emits per-second softmax logits (B,600,2); wrapper reduces "
                    "to (B,2) by max-over-time of seizure prob. Embedding = "
                    "mean-over-time of tx_encoder global-token output (256-d)."
                ),
            )
        ],
    )
]
