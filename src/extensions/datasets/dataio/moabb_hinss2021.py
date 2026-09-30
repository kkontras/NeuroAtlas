"""MOABB-compatible dataset class for Hinss2021 (cognitive workload, 4-class).

Ported from BCI_DG_RPA/moabb_spdnet/ to live inside the EEGBenchmarks extension
tree.  The two classes here (``PreprocessedDataset`` and ``Hinss2021``) are the
minimum needed so that braindecode's ``MOABBDataset`` adapter can load the data.

Original authors: BCI_DG_RPA contributors.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import Optional

import mne
import numpy as np
import pooch

import moabb.datasets.download as dl
from moabb.datasets.base import BaseDataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zenodo helpers
# ---------------------------------------------------------------------------

def _doi_to_url(doi: str) -> Optional[str]:
    api_url = f"https://doi.org/api/handles/{doi}?type=URL"
    headers = {"Content-Type": "application/json"}
    response_data = dl.fs_issue_request("GET", api_url, headers=headers)
    if "values" in response_data:
        candidates = [
            val["data"]["value"]
            for val in response_data["values"]
            if "data" in val and isinstance(val["data"], dict) and "value" in val["data"]
        ]
        return candidates[0] if candidates else None
    return None


def _url_get_json(url: str):
    headers = {"Content-Type": "application/json"}
    return dl.fs_issue_request("GET", url, headers=headers)


# ---------------------------------------------------------------------------
# Base class (minimal port of spdnet_base.PreprocessedDataset)
# ---------------------------------------------------------------------------

class PreprocessedDataset(BaseDataset):
    """BaseDataset with optional channel picking and resampling."""

    def __init__(self, *args, channels: Optional[list] = None, srate: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.srate = srate

    def preprocess(self, raw):
        # find events from stim channels if present
        if len(mne.pick_types(raw.info, stim=True)) > 0:
            events = mne.find_events(raw, shortest_event=0, verbose=False)
        else:
            events = None

        # optional resampling
        if self.srate is not None:
            ret = raw.resample(self.srate, events=events)
            raw, events = (ret, events) if events is None else (ret[0], ret[1])

        # convert events to annotations before discarding stim channels
        if events is not None:
            rev_event_it = dict(zip(self.event_id.values(), self.event_id.keys()))
            annot = mne.annotations_from_events(events, raw.info["sfreq"], event_desc=rev_event_it)
            raw.set_annotations(annot)

        # pick channels
        if self.channels is not None:
            raw.pick_channels(self.channels)
        else:
            raw.pick_types(eeg=True)

        return raw

    def __repr__(self) -> str:
        return json.dumps({self.__class__.__name__: self.__dict__})


# ---------------------------------------------------------------------------
# Hinss2021
# ---------------------------------------------------------------------------

class Hinss2021(PreprocessedDataset):
    """Cognitive-workload EEG dataset (Zenodo 4917217).

    4 classes: easy, medium, difficult, rest.
    15 subjects, 2 sessions each.
    """

    ZENODO_JSON_API_URL = staticmethod(lambda x: f"https://zenodo.org/api/{x}")
    TASK_TO_EVENTID = dict(RS="rest", MATBeasy="easy", MATBmed="medium", MATBdiff="difficult")

    def __init__(self, interval=(0, 2), channels=None, srate=None):
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=2,
            events=dict(easy=1, medium=2, difficult=3, rest=4),
            code="Hinss2021",
            interval=interval,
            paradigm="imagery",
            doi="10.5281/zenodo.4917217",
            channels=channels,
            srate=srate,
        )

    def preprocess(self, raw):
        if len(raw.info["bads"]) > 0:
            raw.interpolate_bads()
        return super().preprocess(raw)

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        key_dest = f"MNE-{self.code:s}-data"
        path = os.path.join(dl.get_dataset_path(self.code, path), key_dest)

        url = _doi_to_url(self.doi)
        if url is None:
            raise ValueError("Could not find zenodo id based on dataset DOI!")

        zenodoid = url.split("/")[-1]
        metadata = _url_get_json(Hinss2021.ZENODO_JSON_API_URL(f"records/{zenodoid}"))

        fnames = []
        for record in metadata["files"]:
            fname = record["key"]
            fpath = os.path.join(path, fname)

            if record["type"] == "zip" and fname == f"P{subject:02d}.zip":
                if not os.path.exists(fpath):
                    pooch.retrieve(
                        record["links"]["self"],
                        record["checksum"],
                        fname,
                        path,
                        processor=pooch.Unzip(),
                        downloader=pooch.HTTPDownloader(progressbar=True),
                    )

                tasks = list(Hinss2021.TASK_TO_EVENTID.keys())
                taskpattern = "(" + "|".join(tasks) + ")"
                pattern = f"{fpath}.unzip/P{subject:02d}/S?/eeg/alldata_*.set"
                candidates = glob.glob(pattern, recursive=True)
                fnames += [c for c in candidates if re.search(f".*{taskpattern}.set", c)]

        return fnames

    def _get_single_subject_data(self, subject):
        fnames = self.data_path(subject)

        subject_data = {}
        for fn in fnames:
            meta = re.search(
                r"alldata_sbj(?P<subject>\d\d)_sess(?P<session>\d)_((?P<event>\w+))",
                os.path.basename(fn),
            )
            sid = int(meta["session"])

            if sid not in range(1, self.n_sessions + 1):
                continue

            epochs = mne.io.read_epochs_eeglab(fn, verbose=False)
            assert len(epochs.event_id) == 1
            event_id = Hinss2021.TASK_TO_EVENTID[list(epochs.event_id.keys())[0]]
            epochs.event_id = {event_id: self.event_id[event_id]}
            epochs.events[:, 2] = epochs.event_id[event_id]

            # convert to continuous raw with correct annotations
            continuous_data = np.swapaxes(epochs.get_data(), 0, 1).reshape(
                (len(epochs.info["chs"]), -1)
            )
            raw = mne.io.RawArray(data=continuous_data, info=epochs.info, verbose=False, first_samp=1)
            raw.set_montage("standard_1005")
            events = epochs.events.copy()
            evt_desc = dict(zip(epochs.event_id.values(), epochs.event_id.keys()))

            annot = mne.annotations_from_events(events, raw.info["sfreq"], event_desc=evt_desc, first_samp=1)
            raw.set_annotations(annot)

            if sid in subject_data:
                subject_data[sid][0].append(raw)
            else:
                subject_data[sid] = {0: raw}

            # discard boundary annotations
            keep = [
                i
                for i, desc in enumerate(subject_data[sid][0].annotations.description)
                if desc in self.event_id
            ]
            subject_data[sid][0].set_annotations(subject_data[sid][0].annotations[keep])

        return subject_data
