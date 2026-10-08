# Vendored from moabb 1.7.2, moabb/datasets/dreyer2023.py
# Copyright (c) 2017, authors of moabb. BSD-3-Clause: see LICENSE.moabb.
#
# Adapted to moabb 1.2.0's BaseDataset (see __init__.py): the constructor takes
# no subject/session selection, download_if_missing has no force_update, and
# the METADATA block (moabb.datasets.metadata, absent in 1.2.0) is dropped.
# The download (OSF), the shared MNE-dreyer2023-data root, the legacy
# per-class root, the montage fallback and the runs read are upstream's.
# Docstrings are shortened and one comment reworded; no logic else changed.
"""
A large EEG right-left hand motor imagery dataset.
It is organized into three A, B, C datasets.
URL PATH: https://zenodo.org/record/7554429
"""

import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from mne.channels import make_standard_montage
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


_manifest_link = "https://osf.io/download/p5av2/"
_metainfo_link = "https://osf.io/download/67c9e8234f014fc76e0411ba/"

_osf_tag = "8tdk5"
_api_base_url = f"https://files.de-1.osf.io/v1/resources/{_osf_tag}/providers/osfstorage/"

# Dreyer2023 is one dataset with globally numbered subjects; the A/B/C classes
# only select subject ranges, so every class shares this root.
DATASET_FOLDER = "MNE-dreyer2023-data"


class _Dreyer2023Base(BaseDataset):
    """
    Parent class of Dreyer2023A, Dreyer2023B and Dreyer2023C.
    Should not be instantiated.
    """

    def __init__(self, all_subjects, sub_id=""):
        self.sub_id = sub_id

        if sub_id is None:
            self.sub_id = ""

        super().__init__(
            all_subjects,
            sessions_per_subject=1,
            events={"left_hand": 1, "right_hand": 2},
            code="Dreyer2023" + self.sub_id,
            interval=[0, 5],
            paradigm="imagery",
            doi="10.1038/s41597-023-02445-z",
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            A dictionary containing the raw data for the subject.
        """
        # Get the file path for the subject's data
        files_path = self.data_path(subject)
        runs = {}
        for run_id, file in enumerate(files_path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Read the subject's raw data and set the montage
                raw = read_raw_bids(bids_path=file, verbose=False)
                raw = raw.load_data()

                # Set the channel montage
                mapping = {}
                for ch in raw.ch_names:
                    if "EOG" in ch:
                        mapping[ch] = "eog"
                    elif "EMG" in ch:
                        mapping[ch] = "emg"

                raw.set_channel_types(mapping)

                # The Zenodo BIDS archive ships no electrodes.tsv sidecar, so
                # read_raw_bids leaves all EEG positions as NaN. The 27 EEG
                # channels are standard 10-20 names, so fall back to the
                # standard_1005 montage when positions are missing.
                eeg_idx = [i for i, t in enumerate(raw.get_channel_types()) if t == "eeg"]
                if any(np.isnan(raw.info["chs"][i]["loc"][:3]).any() for i in eeg_idx):
                    raw.set_montage(
                        make_standard_montage("standard_1005"), on_missing="ignore"
                    )

                # Only the two cue codes are mapped: the archive documents no
                # other event ids.
                raw.annotations.rename({"769": "left_hand", "770": "right_hand"})
                runs.update({f"{run_id}{file.task}": raw})

        sessions = {"0": runs}

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the data BIDS paths of a single subject (downloading it if needed)."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        # Download and extract the dataset
        dataset_path = self.download_by_subject(subject=subject, path=path)

        tasks = get_entity_vals(dataset_path, "task")

        bids_path_list = []
        for task in tasks:
            if "baseline" in task or "rest" in task:
                continue

            if subject == 59 and (("R5online" in task) or ("R6online" in task)):
                continue

            # Create a BIDSPath object for all the tasks
            bids_path = BIDSPath(
                subject=f"{subject:02d}",
                suffix="eeg",
                task=task,
                root=dataset_path,
                check=True,
            )
            bids_path_list.append(bids_path)

        return bids_path_list

    def download_by_subject(self, subject, path=None):
        """Download and extract the subject's files into the shared dataset root.

        A pre-existing download in the legacy per-class layout
        (``MNE-Dreyer2023A-data``, ...) is reused as-is instead of re-fetching.
        """
        root = Path(dl.get_dataset_path("Dreyer2023", path)) / DATASET_FOLDER

        # Reuse a legacy per-class download when it already has this subject
        # and the shared root does not, so existing caches keep working.
        subject_dir = f"sub-{subject:02d}"
        legacy_root = root.parent / f"MNE-{self.code}-data"
        if not (root / subject_dir).is_dir() and (legacy_root / subject_dir).is_dir():
            return legacy_root

        # checking it there is manifest file in the dataset folder.
        dl.download_if_missing(root / "dreyer2023_manifest.tsv", _manifest_link)

        manifest = pd.read_csv(root / "dreyer2023_manifest.tsv", sep="\t")

        subject_index = manifest["filename"] == f"sub-{subject:02d}.zip"

        dataset_index = ~manifest["filename"].str.contains("sub")

        manifest_subject = manifest[subject_index | dataset_index]

        manifest_subject = manifest_subject.copy()

        # no bar: the subject is counted on the item's live line
        for _, row in manifest_subject.iterrows():
            download_url = _api_base_url + row["url"].replace(
                "https://osf.io/download/", ""
            ).replace("/", "")
            dl.download_if_missing(root / row["filename"], download_url, warn_missing=False)

        for _, row in manifest_subject.iterrows():
            if row["filename"].endswith(".zip"):
                if not (root / row["filename"].replace(".zip", "")).exists():
                    with zipfile.ZipFile(root / row["filename"], "r") as zip_ref:
                        zip_ref.extractall(root)

        return root

    def get_subject_info(self, path=None):
        """Return the demographic information of the subjects (a DataFrame)."""
        path = Path(dl.get_dataset_path("Dreyer2023", path)) / DATASET_FOLDER

        # checking it there is manifest file in the dataset folder.
        dl.download_if_missing(path / "performance.csv", _metainfo_link)

        metainfo = pd.read_csv(path / "performance.csv", sep=";")

        if self.sub_id == "":
            return metainfo
        else:
            return metainfo[metainfo["SUBDATASET"] == self.sub_id].reset_index(drop=True)


class Dreyer2023A(_Dreyer2023Base):
    """Dreyer2023, sub-dataset A: subjects 1-60 (see :class:`Dreyer2023`)."""

    def __init__(self):
        super().__init__(all_subjects=list(range(1, 61)), sub_id="A")


class Dreyer2023B(_Dreyer2023Base):
    """Dreyer2023, sub-dataset B: subjects 61-81 (see :class:`Dreyer2023`)."""

    def __init__(self):
        super().__init__(all_subjects=list(range(61, 82)), sub_id="B")


class Dreyer2023C(_Dreyer2023Base):
    """Dreyer2023, sub-dataset C: subjects 82-87 (see :class:`Dreyer2023`)."""

    def __init__(self):
        super().__init__(all_subjects=list(range(82, 88)), sub_id="C")


class Dreyer2023(_Dreyer2023Base):
    """Dreyer2023 MI dataset: Dreyer2023A, B and C together, 87 subjects.

    "A large EEG database with users' profile information for motor imagery
    Brain-Computer Interface research" (Dreyer et al., Scientific Data 2023,
    doi:10.1038/s41597-023-02445-z). 27 EEG channels (10-20, left-earlobe
    reference), 3 EOG, 2 EMG, 512 Hz; left vs right hand imagery; 6 runs of 40
    trials (2 acquisition, 4 training). Trial: cross at 0 s, acoustic cue at
    2 s, arrow at 3 s, feedback from 4.25 s, end at 8 s. moabb's interval is
    [0, 5] s from the event.

    References
    ----------
    .. [1] Pillette, L., Roc, A., N'kaoua, B., & Lotte, F. (2021).
        Experimenters' influence on mental-imagery based brain-computer
        interface user training. International Journal of Human-Computer
        Studies, 149, 102603.
    .. [2] Benaroch, C., Yamamoto, M. S., Roc, A., Dreyer, P., Jeunet, C., &
        Lotte, F. (2022). When should MI-BCI feature optimization include prior
        knowledge, and which one? Brain-Computer Interfaces, 9(2), 115-128.
    """

    def __init__(self):
        super().__init__(all_subjects=list(range(1, 88)))
