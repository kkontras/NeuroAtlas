"""MOABB P300 / ERP dataset specs (auto-discovered from MOABB)."""

from __future__ import annotations

from neuroatlas.benchmarking_helpers import DatasetSpec


def _make_factory(slug: str):
    def _create(**config):
        from .adapters.moabb_generic import MOABBBenchmarkDataModule

        return MOABBBenchmarkDataModule(slug=slug, **config)

    return _create


# (slug, moabb_name, n_classes, description_suffix)
_DATASETS = [
    ("bcicomp2020_walkingerp", "BCIComp2020WalkingERP", 2,  "2-class walking ERP, 46 ch"),
    ("bi2012",                 "BI2012",                2,  "2-class P300, 16 ch"),
    ("bi2013a",                "BI2013a",               2,  "2-class P300, 16 ch"),
    ("bi2014a",                "BI2014a",               2,  "2-class P300, 16 ch, 64 subj"),
    ("bi2014b",                "BI2014b",               2,  "2-class P300, 32 ch"),
    ("bi2015a",                "BI2015a",               2,  "2-class P300, 32 ch"),
    ("bi2015b",                "BI2015b",               2,  "2-class P300, 32 ch, 4 sess"),
    ("bnci2014_008",           "BNCI2014_008",          2,  "2-class P300, 8 ch"),
    ("bnci2014_009",           "BNCI2014_009",          2,  "2-class P300, 16 ch, 3 sess"),
    ("bnci2015_003",           "BNCI2015_003",          2,  "2-class P300, 8 ch"),
    ("bnci2015_006",           "BNCI2015_006",          2,  "2-class P300, 64 ch"),
    ("bnci2015_007",           "BNCI2015_007",          2,  "2-class P300, 63 ch"),
    ("bnci2015_008",           "BNCI2015_008",          2,  "2-class P300, 63 ch"),
    ("bnci2015_009",           "BNCI2015_009",          2,  "2-class P300, 60 ch"),
    ("bnci2015_010",           "BNCI2015_010",          2,  "2-class P300, 63 ch"),
    ("bnci2015_012",           "BNCI2015_012",          2,  "2-class P300, 63 ch"),
    ("bnci2015_013",           "BNCI2015_013",          2,  "2-class P300, 64 ch, 20 sess"),
    ("bnci2016_002",           "BNCI2016_002",          2,  "2-class P300, 59 ch"),
    ("bnci2020_002",           "BNCI2020_002",          2,  "2-class P300, 30 ch"),
    ("cattan2019_vr",          "Cattan2019_VR",         2,  "2-class VR P300, 16 ch, 60 sess"),
    ("chailloux2020",          "Chailloux2020",         2,  "2-class P300, 8 ch, 7 sess"),
    ("epflp300",               "EPFLP300",              2,  "2-class P300, 32 ch"),
    ("erpcore2021",            "ErpCore2021",           1,  "ERP core, 30 ch, 40 subj"),
    ("erpcore2021_ern",        "ErpCore2021_ERN",       2,  "2-class error-related negativity, 30 ch"),
    ("erpcore2021_lrp",        "ErpCore2021_LRP",       2,  "2-class lateralized readiness potential, 30 ch"),
    ("erpcore2021_mmn",        "ErpCore2021_MMN",       2,  "2-class mismatch negativity, 30 ch"),
    ("erpcore2021_n170",       "ErpCore2021_N170",      2,  "2-class N170, 30 ch"),
    ("erpcore2021_n2pc",       "ErpCore2021_N2pc",      2,  "2-class N2pc, 30 ch"),
    ("erpcore2021_n400",       "ErpCore2021_N400",      2,  "2-class N400, 30 ch"),
    ("erpcore2021_p3",         "ErpCore2021_P3",        2,  "2-class P3, 30 ch"),
    ("guttmannflury2025_p300", "GuttmannFlury2025_P300",2,  "2-class P300, 64 ch"),
    ("huebner2017",            "Huebner2017",           2,  "2-class P300, 31 ch, 9 sess"),
    ("huebner2018",            "Huebner2018",           2,  "2-class P300, 31 ch"),
    ("kaneshiro2015",          "Kaneshiro2015",         6,  "6-class visual ERP, 124 ch"),
    ("kojima2024a",            "Kojima2024A",           2,  "2-class P300, 64 ch, 6 sess"),
    ("kojima2024b",            "Kojima2024B",           2,  "2-class P300, 64 ch, 12 sess"),
    ("lee2019_erp",            "Lee2019_ERP",           2,  "2-class ERP, 62 ch, 54 subj"),
    ("lee2021mobile_erp",      "Lee2021Mobile_ERP",     2,  "2-class mobile ERP, 73 ch"),
    ("lee2024_ac",             "Lee2024_AC",            2,  "2-class ERP (auditory-C), 25 ch"),
    ("lee2024_bs",             "Lee2024_BS",            2,  "2-class ERP (brain-speller), 31 ch"),
    ("lee2024_dl",             "Lee2024_DL",            2,  "2-class ERP (deep-learning), 31 ch"),
    ("lee2024_el",             "Lee2024_EL",            2,  "2-class ERP (elderly), 31 ch"),
    ("lee2024_tv",             "Lee2024_TV",            2,  "2-class ERP (TV), 31 ch, 30 subj"),
    ("mainsah2025_a",          "Mainsah2025_A",         2,  "2-class P300, 32 ch"),
    ("mainsah2025_b",          "Mainsah2025_B",         2,  "2-class P300, 16 ch"),
    ("mainsah2025_c",          "Mainsah2025_C",         2,  "2-class P300, 32 ch"),
    ("mainsah2025_d",          "Mainsah2025_D",         2,  "2-class P300, 32 ch"),
    ("mainsah2025_e",          "Mainsah2025_E",         2,  "2-class P300, 16 ch"),
    ("mainsah2025_f",          "Mainsah2025_F",         2,  "2-class P300, 16 ch"),
    ("mainsah2025_g",          "Mainsah2025_G",         2,  "2-class P300, 16 ch"),
    ("mainsah2025_h",          "Mainsah2025_H",         2,  "2-class P300, 16 ch"),
    ("mainsah2025_i",          "Mainsah2025_I",         2,  "2-class P300, 16 ch"),
    ("mainsah2025_j",          "Mainsah2025_J",         2,  "2-class P300, 16 ch"),
    ("mainsah2025_k",          "Mainsah2025_K",         2,  "2-class P300, 16 ch"),
    ("mainsah2025_l",          "Mainsah2025_L",         2,  "2-class P300, 16 ch"),
    ("mainsah2025_m",          "Mainsah2025_M",         2,  "2-class P300, 16 ch"),
    ("mainsah2025_n",          "Mainsah2025_N",         2,  "2-class P300, 16 ch"),
    ("mainsah2025_o",          "Mainsah2025_O",         2,  "2-class P300, 32 ch"),
    ("mainsah2025_p",          "Mainsah2025_P",         2,  "2-class P300, 32 ch"),
    ("mainsah2025_q",          "Mainsah2025_Q",         2,  "2-class P300, 32 ch"),
    ("mainsah2025_r",          "Mainsah2025_R",         2,  "2-class P300, 32 ch"),
    ("mainsah2025_s1",         "Mainsah2025_S1",        2,  "2-class P300, 32 ch"),
    ("mainsah2025_s2",         "Mainsah2025_S2",        2,  "2-class P300, 32 ch"),
    ("romanibf2025erp",        "RomaniBF2025ERP",       2,  "2-class ERP, 8 ch"),
    ("simoes2020",             "Simoes2020",            2,  "2-class P300, 8 ch, 7 sess"),
    ("sosulski2019",           "Sosulski2019",          2,  "2-class P300, 31 ch"),
    ("speier2017",             "Speier2017",            2,  "2-class P300, 32 ch, 3 sess"),
    ("zheng2020",              "Zheng2020",             2,  "2-class P300, 62 ch"),
    ("zhang2025",              "Zhang2025",             2,  "2-class P300, 57 ch, 4 sess"),
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
        description=f"MOABB {moabb_name} P300/ERP benchmark ({desc}).",
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
