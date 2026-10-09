from __future__ import annotations

from neuroatlas.benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_eegpt(spec: CheckpointSpec):
    from .backbones.eegpt import EEGPTBackbone

    return EEGPTBackbone(spec)


MODEL_SPECS = [
    ModelSpec(
        slug="eegpt",
        description="EEGPT foundation encoder wrapper.",
        loader=_load_eegpt,
        checkpoints=[
            CheckpointSpec(
                identifier="eegpt_pretrained",
                model_family="eegpt",
                variant="large4e_pretrained",
                # Upstream publishes the checkpoint on a Figshare private share
                # (https://figshare.com/s/e37df4f8a907a866df4b, file
                # EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt) that only a
                # browser can open. This hub repository holds the same tensors
                # (verified bit-identical; see _checkpoint_download), and the
                # download writes them where the upstream file would go.
                source_type="huggingface",
                source_reference="eeg-telecom-paris/eegpt-large-official",
                checkpoint_path="artifacts/models/foundation/eegpt_mcae_58chs_4s_large4E.ckpt",
                input_kind="raw_timeseries",
                expected_channels=("eeg",),
                expected_sampling_rate=256.0,
                expected_epoch_seconds=10.0,
                pretraining_datasets=("mixed_dataset",),
                has_classifier_head=False,
                embedding_key="target_encoder_mean_patches",
                embedding_dim=2048,
                wrapper_name="eegpt",
                status="ready",
                notes="Official EEGPT target-encoder probing path. Weights: upstream's eegpt_mcae_58chs_4s_large4E.ckpt (Figshare share https://figshare.com/s/e37df4f8a907a866df4b, browser only), fetched from the bit-identical hub copy eeg-telecom-paris/eegpt-large-official.",
            )
        ],
    )
]
