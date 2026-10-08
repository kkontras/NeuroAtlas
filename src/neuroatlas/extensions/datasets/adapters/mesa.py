"""MESA benchmark adapter (thin subclass of PhysioExBenchmarkDataModule)."""
from .physioex_base import PhysioExBenchmarkDataModule


class MESABenchmarkDataModule(PhysioExBenchmarkDataModule):
    DATASET_NAME = "mesa"
    PHYSIOEX_CLASS = "neuroatlas.extensions.datasets.physioex.datasets.mesa.MESADataset"
    EEG_CHANNELS = ["EEG", "EEG", "EEG"]
    CHANNEL_MAP = {
        "EEG1": "C4", "EEG2": "C3", "EEG3": "Cz",
    }
    # Brain age: age at the sleep exam (MESA Exam 5); mesa-sleep-0001 is
    # mesaid 1.
    AGE_TABLE = "mesa-sleep-dataset-*.csv"
    AGE_ID_COLUMN = "mesaid"
    AGE_COLUMN = "sleepage5c"
