"""SHHS visit-2 benchmark adapter (thin subclass of PhysioExBenchmarkDataModule)."""
from .physioex_base import PhysioExBenchmarkDataModule


class SHHSv2BenchmarkDataModule(PhysioExBenchmarkDataModule):
    DATASET_NAME = "shhs_v2"
    PHYSIOEX_CLASS = "extensions.datasets.physioex.datasets.shhs.SHHSDataset"
    EEG_CHANNELS = ["EEG"]
    CHANNEL_MAP = {
        "EEG": "C4",
    }
    DATASET_KWARGS = {"visit": 2}
