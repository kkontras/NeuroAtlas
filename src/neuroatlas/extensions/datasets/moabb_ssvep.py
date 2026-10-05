"""MOABB SSVEP dataset specs (auto-discovered from MOABB)."""

from __future__ import annotations

from neuroatlas.benchmarking_helpers import DatasetSpec


def _make_factory(slug: str):
    def _create(**config):
        from .adapters.moabb_generic import MOABBBenchmarkDataModule
        from neuroatlas.benchmarking_helpers.registry.contracts import construct_datamodule

        return construct_datamodule(
            MOABBBenchmarkDataModule, {"slug": slug, **config}, dataset=slug)

    return _create


# (slug, moabb_name, n_classes, description_suffix)
_DATASETS = [
    ("chen2017singleflicker",   "Chen2017SingleFlicker",    4,  "4-class SSVEP, 32 ch"),
    ("dong2023",                "Dong2023",                40,  "40-class SSVEP, 8 ch, 59 subj"),
    ("guttmannflury2025_ssvep", "GuttmannFlury2025_SSVEP",  4,  "4-class SSVEP, 64 ch"),
    ("han2024fatigue",          "Han2024Fatigue",          32,  "32-class SSVEP fatigue, 64 ch"),
    ("kalunga2016",             "Kalunga2016",              4,  "4-class SSVEP, 8 ch"),
    ("kim2025betarange",        "Kim2025BetaRange",        40,  "40-class beta-range SSVEP, 31 ch"),
    ("lee2019_ssvep",           "Lee2019_SSVEP",            4,  "4-class SSVEP, 62 ch, 54 subj"),
    ("lee2021mobile_ssvep",     "Lee2021Mobile_SSVEP",      3,  "3-class mobile SSVEP, 73 ch"),
    ("liu2020beta",             "Liu2020BETA",             40,  "40-class BETA SSVEP, 64 ch, 70 subj"),
    ("liu2022eldbeta",          "Liu2022EldBETA",           9,  "9-class elderly BETA SSVEP, 64 ch, 100 subj"),
    ("mamem1",                  "MAMEM1",                   5,  "5-class SSVEP, 256 ch"),
    ("mamem2",                  "MAMEM2",                   5,  "5-class SSVEP, 256 ch, 5 sess"),
    ("mamem3",                  "MAMEM3",                   5,  "5-class SSVEP, 14 ch, 10 sess"),
    ("nakanishi2015",           "Nakanishi2015",           12,  "12-class SSVEP, 8 ch"),
    ("wang2016",                "Wang2016",                40,  "40-class SSVEP, 64 ch, 34 subj"),
    ("wang2021combined",        "Wang2021Combined",         4,  "4-class combined SSVEP, 31 ch"),
]

_CONFIG_DEFAULTS = {
    "fold": 0,
    "n_folds": 5,
    "batch_size": 64,
    "num_workers": 0,
    "n_jobs": 1,
}

def _attach_manifest(spec):
    """Return *spec* enriched with its dossier, when one exists.

    MOABB specs are generated from a table, so they carry no manifest and
    every tool that filters on ``paper_dataset`` skipped them -- including the
    14 BCI cohorts the paper actually reports. Only ``spec.manifest`` is
    filled in: the datamodule, defaults, task and fold behaviour are left
    exactly as generated, so attaching a dossier cannot change a run.
    """
    import dataclasses

    from neuroatlas.benchmarking_helpers.registry.manifest import (
        ManifestError,
        load_manifest,
    )

    try:
        manifest = load_manifest(spec.slug)
    except (ManifestError, Exception):
        return spec
    return dataclasses.replace(spec, manifest=manifest)


_GENERATED_SPECS = [
    DatasetSpec(
        slug=slug,
        description=f"MOABB {moabb_name} SSVEP benchmark ({desc}).",
        datamodule_cls=_make_factory(slug),
        config_defaults=dict(_CONFIG_DEFAULTS),
        metadata_keys=(
            "paradigm",
            "n_classes",
            "epoch_seconds",
            "channel_policy",
            "signal_kind",
            "fold",
        ),
        supports_folds=True,
        default_task="linear_probe",
        input_kind_signal_map={"raw_timeseries": "raw"},
    )
    for slug, moabb_name, _n_cls, desc in _DATASETS
]


DATASET_SPECS = [_attach_manifest(spec) for spec in _GENERATED_SPECS]
