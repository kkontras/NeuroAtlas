"""HMC benchmark adapter (thin subclass of PhysioExBenchmarkDataModule)."""
from .physioex_base import PhysioExBenchmarkDataModule


class HMCBenchmarkDataModule(PhysioExBenchmarkDataModule):
    DATASET_NAME = "hmc"
    PHYSIOEX_CLASS = "neuroatlas.extensions.datasets.physioex.datasets.hmc.HMCDataset"
    EEG_CHANNELS = ["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EEG C3-M2"]
    CHANNEL_MAP = {
        "EEG F4-M1": "F4", "EEG C4-M1": "C4",
        "EEG O2-M1": "O2", "EEG C3-M2": "C3",
    }
