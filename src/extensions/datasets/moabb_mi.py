"""MOABB Motor Imagery dataset specs (auto-discovered from MOABB).

Registers all MOABB MI datasets EXCEPT physionet_mi, cho2017, and lee2019_mi
which have dedicated hand-tuned configs in their own spec files.
"""

from __future__ import annotations

from benchmarking_helpers import DatasetSpec


def _make_factory(slug: str):
    """Create a lazy factory that imports the generic adapter only on use."""

    def _create(**config):
        from .adapters.moabb_generic import MOABBBenchmarkDataModule

        return MOABBBenchmarkDataModule(slug=slug, **config)

    return _create


# (slug, moabb_name, n_classes, description_suffix)
_DATASETS = [
    ("alex_mi",                "AlexMI",                3,  "3-class MI"),
    ("bnci2003_004",           "BNCI2003_004",          2,  "2-class MI, 118 ch"),
    ("bnci2014_001",           "BNCI2014_001",          4,  "4-class MI, 22 ch"),
    ("bnci2014_002",           "BNCI2014_002",          2,  "2-class MI, 15 ch"),
    ("bnci2014_004",           "BNCI2014_004",          2,  "2-class MI, 3 ch"),
    ("bnci2015_001",           "BNCI2015_001",          2,  "2-class MI, 13 ch"),
    ("bnci2015_004",           "BNCI2015_004",          5,  "5-class MI, 30 ch"),
    ("bnci2019_001",           "BNCI2019_001",          5,  "5-class MI, 61 ch"),
    ("bnci2020_001",           "BNCI2020_001",          3,  "3-class MI, 58 ch"),
    ("bnci2022_001",           "BNCI2022_001",          4,  "4-class MI, 64 ch"),
    ("bnci2024_001",           "BNCI2024_001",         10,  "10-class MI, 60 ch"),
    ("bnci2025_001",           "BNCI2025_001",         16,  "16-class MI, 67 ch"),
    ("bnci2025_002",           "BNCI2025_002",          3,  "3-class MI, 60 ch"),
    ("beetl2021_a",            "Beetl2021_A",           4,  "4-class MI, 63 ch, 3 subj"),
    ("beetl2021_b",            "Beetl2021_B",           4,  "4-class MI, 32 ch, 2 subj"),
    ("brandl2020",             "Brandl2020",            2,  "2-class MI, 63 ch"),
    ("chang2025",              "Chang2025",             3,  "3-class MI, 59 ch"),
    ("dreyer2023",             "Dreyer2023",            2,  "2-class MI, 27 ch, 87 subj"),
    ("dreyer2023a",            "Dreyer2023A",           2,  "2-class MI, 27 ch, 60 subj (healthy)"),
    ("dreyer2023b",            "Dreyer2023B",           2,  "2-class MI, 27 ch, 21 subj (BCI-naive)"),
    ("dreyer2023c",            "Dreyer2023C",           2,  "2-class MI, 27 ch, 6 subj (stroke)"),
    ("forenzo2023",            "Forenzo2023",           2,  "2-class MI, 64 ch"),
    ("gao2026",                "Gao2026",              10,  "10-class MI, 32 ch"),
    ("grossewentrup2009",      "GrosseWentrup2009",     2,  "2-class MI, 128 ch"),
    ("guttmannflury2025_me",   "GuttmannFlury2025_ME",  2,  "2-class motor execution, 64 ch"),
    ("guttmannflury2025_mi",   "GuttmannFlury2025_MI",  2,  "2-class MI, 64 ch"),
    ("hefmiich2025",           "HefmiIch2025",          2,  "2-class MI, 32 ch"),
    ("jeong2020",              "Jeong2020",            11,  "11-class MI, 60 ch"),
    ("kaya2018",               "Kaya2018",              3,  "3-class MI, 19 ch"),
    ("kumar2024",              "Kumar2024",             2,  "2-class MI, 22 ch"),
    ("liu2024",                "Liu2024",               2,  "2-class MI, 29 ch, 50 subj"),
    ("liu2025",                "Liu2025",               2,  "2-class MI, 60 ch"),
    ("ma2020",                 "Ma2020",                2,  "2-class MI, 62 ch, 15 sess"),
    ("ofner2017",              "Ofner2017",             7,  "7-class MI, 61 ch"),
    ("pressel2016",            "Pressel2016",          11,  "11-class MI, 6 ch"),
    ("rozado2015",             "Rozado2015",            2,  "2-class MI, 32 ch"),
    ("schirrmeister2017",      "Schirrmeister2017",     4,  "4-class MI, 128 ch"),
    ("shin2017a",              "Shin2017A",             2,  "2-class MI, 30 ch (paradigm A)"),
    ("shin2017b",              "Shin2017B",             2,  "2-class MI, 30 ch (paradigm B)"),
    ("stieger2021",            "Stieger2021",           4,  "4-class MI, 62 ch, up to 11 sess"),
    ("tavakolan2017",          "Tavakolan2017",         3,  "3-class MI, 32 ch"),
    ("trianaguzman2024",       "TrianaGuzman2024",      4,  "4-class MI, 17 ch"),
    ("wairagkar2018",          "Wairagkar2018",         3,  "3-class MI, 19 ch"),
    ("weibo2014",              "Weibo2014",             7,  "7-class MI, 60 ch"),
    ("yang2025",               "Yang2025",              2,  "2-class MI, 59 ch"),
    ("yi2025",                 "Yi2025",                8,  "8-class MI, 62 ch"),
    ("zhang2017",              "Zhang2017",            10,  "10-class MI, 17 ch"),
    ("zhou2016",               "Zhou2016",              3,  "3-class MI, 14 ch"),
    ("zhou2020",               "Zhou2020",              4,  "4-class MI, 41 ch, 7 sess"),
    ("zuo2025",                "Zuo2025",               2,  "2-class MI, 30 ch"),
    ("aguilerarodriguez2025",  "AguileraRodriguez2025", 4,  "4-class MI, 24 ch"),
    ("bcicomp2020_is",         "BCIComp2020IS",         5,  "5-class MI, 64 ch"),
    ("bcicomp2020_upperlimb",  "BCIComp2020UpperLimb",  3,  "3-class MI upper limb, 60 ch"),
    ("nguyen2017_l",           "Nguyen2017_L",          2,  "2-class MI (left), 60 ch"),
    ("nguyen2017_s",           "Nguyen2017_S",          3,  "3-class MI (short), 60 ch"),
    ("nguyen2017_sl",          "Nguyen2017_SL",         2,  "2-class MI (short-left), 60 ch"),
    ("nguyen2017_v",           "Nguyen2017_V",          3,  "3-class MI (visual), 60 ch"),
]

_CONFIG_DEFAULTS = {
    "fold": 0,
    "n_folds": 5,
    "batch_size": 64,
    "num_workers": 0,
    "n_jobs": 1,
    # `confound_control` is deliberately NOT here. It defaults to False in
    # MOABBBenchmarkDataModule and is reached with
    # `--set confound_control=true`, which keeps it out of every spec's
    # config_defaults -- putting it there would restate the adapter's default
    # across ~56 generated specs for no gain.
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
        description=f"MOABB {moabb_name} motor-imagery benchmark ({desc}).",
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
