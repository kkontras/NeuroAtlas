"""Dataset registry.

As more datasets are implemented, they get added here. Users can instantiate by
string name via ``get_dataset(name)``.
"""
from neuroatlas.extensions.datasets.physioex.datasets.hmc import HMCDataset
from neuroatlas.extensions.datasets.physioex.datasets.sleepedf import SleepEDFDataset
from neuroatlas.extensions.datasets.physioex.datasets.dcsm import DCSMDataset
from neuroatlas.extensions.datasets.physioex.datasets.mesa import MESADataset
from neuroatlas.extensions.datasets.physioex.datasets.mros import MrOSDataset
from neuroatlas.extensions.datasets.physioex.datasets.hpap import HPAPDataset
from neuroatlas.extensions.datasets.physioex.datasets.wsc import WSCDataset
from neuroatlas.extensions.datasets.physioex.datasets.mass import MASSDataset
from neuroatlas.extensions.datasets.physioex.datasets.alzheimers import AlzheimersDataset
from neuroatlas.extensions.datasets.physioex.datasets.parkinsons import ParkinsonsDataset
from neuroatlas.extensions.datasets.physioex.datasets.stages import STAGESDataset

REGISTRY = {
    "hmc": HMCDataset,
    "sleepedf": SleepEDFDataset,
    "dcsm": DCSMDataset,
    "mesa": MESADataset,
    "mros": MrOSDataset,
    "hpap": HPAPDataset,
    "wsc": WSCDataset,
    "mass": MASSDataset,
    "alzheimers": AlzheimersDataset,
    "parkinsons": ParkinsonsDataset,
    "stages": STAGESDataset,
}


def get_dataset(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def available_datasets() -> list:
    return sorted(REGISTRY.keys())


__all__ = [
    "HMCDataset",
    "SleepEDFDataset",
    "DCSMDataset",
    "MESADataset",
    "MrOSDataset",
    "HPAPDataset",
    "WSCDataset",
    "MASSDataset",
    "AlzheimersDataset",
    "ParkinsonsDataset",
    "STAGESDataset",
    "REGISTRY",
    "get_dataset",
    "available_datasets",
]
