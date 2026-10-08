"""MOABB c-VEP (code-modulated VEP) dataset specs."""

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
    ("castillosburstvep100",       "CastillosBurstVEP100",       2,  "2-class burst-VEP 100%, 32 ch"),
    ("castillosburstvep40",        "CastillosBurstVEP40",        2,  "2-class burst-VEP 40%, 32 ch"),
    ("castilloscvep100",           "CastillosCVEP100",           2,  "2-class c-VEP 100%, 32 ch"),
    ("castilloscvep40",            "CastillosCVEP40",            2,  "2-class c-VEP 40%, 32 ch"),
    ("martinezcagigal2023checker", "MartinezCagigal2023Checker",  2,  "2-class checker c-VEP, 16 ch"),
    ("martinezcagigal2023pary",    "MartinezCagigal2023Pary",    11,  "11-class parity c-VEP, 16 ch"),
    ("thielen2015",                "Thielen2015",                 2,  "2-class c-VEP, 64 ch"),
    ("thielen2021",                "Thielen2021",                 2,  "2-class c-VEP, 8 ch, 30 subj"),
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
        description=f"MOABB {moabb_name}: c-VEP recordings ({desc}).",
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
