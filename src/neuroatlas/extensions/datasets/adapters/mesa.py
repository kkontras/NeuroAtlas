"""MESA benchmark adapter (thin subclass of PhysioExBenchmarkDataModule)."""
from .physioex_base import PhysioExBenchmarkDataModule


class MESABenchmarkDataModule(PhysioExBenchmarkDataModule):
    DATASET_NAME = "mesa"
    PHYSIOEX_CLASS = "neuroatlas.extensions.datasets.physioex.datasets.mesa.MESADataset"
    EEG_CHANNELS = ["EEG", "EEG", "EEG"]
    CHANNEL_MAP = {
        "EEG1": "C4", "EEG2": "C3", "EEG3": "Cz",
    }
