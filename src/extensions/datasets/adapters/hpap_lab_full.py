"""HomePAP lab-full benchmark adapter (thin subclass of PhysioExBenchmarkDataModule)."""
from .physioex_base import PhysioExBenchmarkDataModule


class HPAPLabFullBenchmarkDataModule(PhysioExBenchmarkDataModule):
    DATASET_NAME = "hpap_lab_full"
    PHYSIOEX_CLASS = "extensions.datasets.physioex.datasets.hpap.HPAPDataset"
    EEG_CHANNELS = ["EEG", "EEG", "EEG", "EEG", "EEG", "EEG"]
    # Empty map — _strip_to_standard fallback resolves to C3, C4, F3, F4, O1, O2
    CHANNEL_MAP = {}
    DATASET_KWARGS = {"subset": "lab-full"}
