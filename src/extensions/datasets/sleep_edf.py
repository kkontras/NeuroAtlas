"""sleep_edf dataset spec.

Every fact about this cohort — source, montage and why,
label modes, split grouping — lives in configs/cohorts/sleep_edf/.  This module only binds
the manifest to the datamodule callable, which is the one thing a YAML
file cannot hold.

To change a default, edit the manifest, not this file.
"""

from __future__ import annotations

from benchmarking_helpers import dataset_spec_from_manifest


def _create_sleep_edf_datamodule(**config):
    from .adapters.sleep_edf import SleepEDFBenchmarkDataModule

    runtime = dict(config)
    if "format" in runtime and "data_format" not in runtime:
        runtime["data_format"] = runtime.pop("format")
    return SleepEDFBenchmarkDataModule(**runtime)


DATASET_SPECS = [
    dataset_spec_from_manifest("sleep_edf", _create_sleep_edf_datamodule),
]
