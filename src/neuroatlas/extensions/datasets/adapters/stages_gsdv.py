"""STAGES GSDV benchmark adapter."""
from .physioex_base import PhysioExBenchmarkDataModule


class STAGESGSDVBenchmarkDataModule(PhysioExBenchmarkDataModule):
    DATASET_NAME = "stages_gsdv"
    PHYSIOEX_CLASS = "neuroatlas.extensions.datasets.physioex.datasets.stages.STAGESDataset"
    EEG_CHANNELS = ["EEG"] * 6
    CHANNEL_MAP = {}  # _strip_to_standard handles all site-specific variants
    DATASET_KWARGS = {"site": "GSDV"}
