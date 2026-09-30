"""Parkinsons night PD benchmark adapter (thin subclass of PhysioExBenchmarkDataModule)."""
from .physioex_base import PhysioExBenchmarkDataModule


class ParkinsonsNightPDBenchmarkDataModule(PhysioExBenchmarkDataModule):
    DATASET_NAME = "parkinsons_night_pd"
    PHYSIOEX_CLASS = "extensions.datasets.physioex.datasets.parkinsons.ParkinsonsDataset"
    EEG_CHANNELS = ["EEG", "EEG"]
    CHANNEL_MAP = {
        "EEG C3-A2": "C3", "EEG C4-A1": "C4",
        "EEG F3-A2": "F3", "EEG F4-A1": "F4",
        "EEG O1-A2": "O1", "EEG O2-A1": "O2",
        "EEG C3-REF": "C3", "EEG C4-REF": "C4",
        "EEG F3-REF": "F3", "EEG F4-REF": "F4",
        "EEG O1-REF": "O1", "EEG O2-REF": "O2",
    }
    DATASET_KWARGS = {"recording": "night", "group": "PD"}
