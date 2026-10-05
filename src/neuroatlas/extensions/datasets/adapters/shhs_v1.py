"""SHHS visit-1 benchmark adapter (thin subclass of PhysioExBenchmarkDataModule)."""
from .physioex_base import PhysioExBenchmarkDataModule


class SHHSv1BenchmarkDataModule(PhysioExBenchmarkDataModule):
    DATASET_NAME = "shhs_v1"
    PHYSIOEX_CLASS = "neuroatlas.extensions.datasets.physioex.datasets.shhs.SHHSDataset"
    EEG_CHANNELS = ["EEG"]
    CHANNEL_MAP = {
        "EEG": "C4",
    }
    DATASET_KWARGS = {"visit": 1}
