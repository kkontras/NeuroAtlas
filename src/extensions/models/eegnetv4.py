from __future__ import annotations

from benchmarking_helpers import CheckpointSpec, ModelSpec


def _load_eegnetv4(spec: CheckpointSpec):
    from .backbones.eegnetv4 import EEGNetv4Backbone

    return EEGNetv4Backbone(spec)


# (pretraining dataset folder on HF, variant slug, matching MOABB-MI benchmark
# slug, number of classes of the classifier head). Each tuple becomes one
# CheckpointSpec pointing at its own `kwargs.pkl` + `model-params.pkl` under
# `artifacts/models/foundation/eegnetv4/<folder>/`.
_CHECKPOINTS = [
    ("EEGNetv4_AlexMI",            "alex_mi",            "alex_mi",            3),
    ("EEGNetv4_BNCI2014001",       "bnci2014_001",       "bnci2014_001",       4),
    ("EEGNetv4_BNCI2014004",       "bnci2014_004",       "bnci2014_004",       2),
    ("EEGNetv4_BNCI2015001",       "bnci2015_001",       "bnci2015_001",       2),
    ("EEGNetv4_BNCI2015004",       "bnci2015_004",       "bnci2015_004",       5),
    ("EEGNetv4_Cho2017",           "cho2017",            "cho2017",            2),
    ("EEGNetv4_Lee2019_MI",        "lee2019_mi",         "lee2019_mi",         2),
    ("EEGNetv4_Ofner2017",         "ofner2017",          "ofner2017",          7),
    ("EEGNetv4_PhysionetMI",       "physionet_mi",       "physionet_mi",       5),
    ("EEGNetv4_Schirrmeister2017", "schirrmeister2017",  "schirrmeister2017",  4),
    ("EEGNetv4_Weibo2014",         "weibo2014",          "weibo2014",          7),
    ("EEGNetv4_Zhou2016",          "zhou2016",           "zhou2016",           3),
]


def _build_checkpoints():
    specs = []
    for folder, variant, benchmark_slug, n_classes in _CHECKPOINTS:
        specs.append(
            CheckpointSpec(
                identifier=f"eegnetv4_{variant}",
                model_family="eegnetv4",
                variant=variant,
                source_type="huggingface",
                source_reference="https://huggingface.co/PierreGtch/EEGNetv4",
                checkpoint_path=f"artifacts/models/foundation/eegnetv4/{folder}",
                input_kind="raw_timeseries",
                expected_channels=("C3", "CZ", "C4"),
                expected_sampling_rate=128.0,
                expected_epoch_seconds=385.0 / 128.0,
                pretraining_datasets=(benchmark_slug,),
                finetuned_datasets=(benchmark_slug,),
                has_classifier_head=True,
                embedding_key="preclassifier_flat",
                embedding_dim=192,
                wrapper_name="eegnetv4",
                status="ready",
                notes=(
                    "Pretrained EEGNetv4 classifier (Guetschel et al., 2024) for MOABB "
                    f"motor-imagery dataset '{benchmark_slug}'. Trained on 3 central motor "
                    "channels (C3, Cz, C4), 128 Hz, 3 s windows (385 samples), with a "
                    "0.5-40 Hz bandpass. Loaded via kwargs.pkl + model-params.pkl from "
                    "PierreGtch/EEGNetv4. Feature key 'preclassifier_flat' returns the "
                    "16-filter x 12-step feature map (dim 192) from the block prior to "
                    f"the {n_classes}-class final conv classifier."
                ),
            )
        )
    return specs


MODEL_SPECS = [
    ModelSpec(
        slug="eegnetv4",
        description=(
            "EEGNetv4 (Lawhern et al. 2018) pretrained on 12 MOABB motor-imagery "
            "datasets (Guetschel et al., 2024 — PierreGtch/EEGNetv4)."
        ),
        loader=_load_eegnetv4,
        checkpoints=_build_checkpoints(),
    )
]
