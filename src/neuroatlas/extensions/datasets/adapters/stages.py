"""STAGES benchmark adapter — all 13 sites.

The paper counts STAGES once (App. B.2), so this is the spec that runs it.
The thirteen `stages_<site>.py` siblings are the same loader pinned to one
site; they exist for per-site analysis and are not registered.
"""
from typing import Optional

from .physioex_base import PhysioExBenchmarkDataModule


class STAGESBenchmarkDataModule(PhysioExBenchmarkDataModule):
    DATASET_NAME = "stages"
    PHYSIOEX_CLASS = "neuroatlas.extensions.datasets.physioex.datasets.stages.STAGESDataset"
    EEG_CHANNELS = ["EEG"] * 6
    CHANNEL_MAP = {}  # _strip_to_standard handles all site-specific variants
    DATASET_KWARGS = {"site": None}  # None = every site
    # Brain age: the harmonized table's age; a recording's file name is its
    # subject_code (BOGN00001; a repeat night BOGN00001_1).
    AGE_TABLE = "stages-harmonized-dataset-*.csv"
    AGE_ID_COLUMN = "subject_code"
    AGE_COLUMN = "nsrr_age"

    @staticmethod
    def age_key(recording_id: str) -> Optional[str]:
        return recording_id[:-2] if recording_id.endswith("_1") else recording_id
