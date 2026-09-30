"""MOABB resting-state dataset specs.

Hinss2021 is excluded — it has a dedicated hand-tuned config in hinss2021.py.
"""

from __future__ import annotations

from benchmarking_helpers import DatasetSpec


def _make_factory(slug: str):
    def _create(**config):
        from .adapters.moabb_generic import MOABBBenchmarkDataModule

        return MOABBBenchmarkDataModule(slug=slug, **config)

    return _create


# (slug, moabb_name, n_classes, description_suffix)
_DATASETS = [
    ("cattan2019_phmd", "Cattan2019_PHMD", 2, "2-class resting-state PHMD, 16 ch"),
    ("rodrigues2017",   "Rodrigues2017",   2, "2-class resting-state, 16 ch"),
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

    from benchmarking_helpers.registry.manifest import (
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
        description=f"MOABB {moabb_name} resting-state benchmark ({desc}).",
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
