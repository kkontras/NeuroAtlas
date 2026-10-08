"""STAGES benchmark adapter — all 13 sites.

The paper counts STAGES once (App. B.2), so this is the spec that runs it.
The thirteen `stages_<site>.py` siblings are the same loader pinned to one
site; they exist for per-site analysis and are not registered.
"""
from .physioex_base import PhysioExBenchmarkDataModule


class STAGESBenchmarkDataModule(PhysioExBenchmarkDataModule):
    DATASET_NAME = "stages"
    PHYSIOEX_CLASS = "neuroatlas.extensions.datasets.physioex.datasets.stages.STAGESDataset"
    EEG_CHANNELS = ["EEG"] * 6
    CHANNEL_MAP = {}  # _strip_to_standard handles all site-specific variants
    DATASET_KWARGS = {"site": None}  # None = every site
