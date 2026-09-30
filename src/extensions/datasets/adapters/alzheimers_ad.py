"""Alzheimers AD benchmark adapter (thin subclass of PhysioExBenchmarkDataModule)."""
from .physioex_base import PhysioExBenchmarkDataModule


class AlzheimersADBenchmarkDataModule(PhysioExBenchmarkDataModule):
    DATASET_NAME = "alzheimers_ad"
    PHYSIOEX_CLASS = "extensions.datasets.physioex.datasets.alzheimers.AlzheimersDataset"
    EEG_CHANNELS = [
        "EEG C4-REF", "EEG C3-REF", "EEG F4-REF", "EEG F3-REF",
        "EEG O2-REF", "EEG O1-REF", "EEG Fz-REF", "EEG Pz-REF",
    ]
    CHANNEL_MAP = {
        "EEG Fp1-REF": "Fp1", "EEG Fp2-REF": "Fp2",
        "EEG F3-REF": "F3", "EEG F4-REF": "F4",
        "EEG F7-REF": "F7", "EEG F8-REF": "F8",
        "EEG C3-REF": "C3", "EEG C4-REF": "C4",
        "EEG T7-REF": "T7", "EEG T8-REF": "T8",
        "EEG P7-REF": "P7", "EEG P8-REF": "P8",
        "EEG O1-REF": "O1", "EEG O2-REF": "O2",
        "EEG Fz-REF": "Fz", "EEG Pz-REF": "Pz",
    }
    DATASET_KWARGS = {"subset": "AD"}
