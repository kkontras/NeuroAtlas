"""STAGES MSMI benchmark adapter."""
from .physioex_base import PhysioExBenchmarkDataModule


class STAGESMSMIBenchmarkDataModule(PhysioExBenchmarkDataModule):
    DATASET_NAME = "stages_msmi"
    PHYSIOEX_CLASS = "extensions.datasets.physioex.datasets.stages.STAGESDataset"
    EEG_CHANNELS = ["EEG"] * 6
    CHANNEL_MAP = {}  # _strip_to_standard handles all site-specific variants
    DATASET_KWARGS = {"site": "MSMI"}
