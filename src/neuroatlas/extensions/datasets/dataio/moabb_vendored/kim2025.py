# Vendored from moabb 1.7.2, moabb/datasets/ssvep_kim2025.py (identical to its
# first release, moabb 1.5.0) and build_raw_from_epochs / FIGSHARE_DL_URL from
# moabb/datasets/utils.py.
# Copyright (c) 2017, authors of moabb. BSD-3-Clause: see LICENSE.moabb.
#
# Adapted to moabb 1.2.0's BaseDataset (see __init__.py): the constructor takes
# no subject/session selection and the METADATA block is dropped. The Figshare
# download, the channel naming and the raw built from the stored epochs are
# upstream's; docstrings are shortened.
"""40-Class Beta-Range SSVEP Speller Dataset.

Kim et al. (2025), Scientific Data.
DOI: 10.1038/s41597-025-06032-2
"""

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from scipy.io import loadmat


FIGSHARE_DL_URL = "https://ndownloader.figshare.com/files/"

# Figshare file IDs for raw_eeg_ssvep_subj_NN.mat
# fmt: off
_SSVEP_FILE_IDS = {
    1: 53705183, 2: 53705180, 3: 53705123, 4: 53705168, 5: 53705171,
    6: 53705108, 7: 53705132, 8: 53705153, 9: 53705126, 10: 53705156,
    11: 53705129, 12: 53705099, 13: 53705177, 14: 53705150, 15: 53707388,
    16: 53705135, 17: 53705162, 18: 53705087, 19: 53705114, 20: 53705174,
    21: 53705075, 22: 53705117, 23: 53707391, 24: 53705078, 25: 53705165,
    26: 53705090, 27: 53705144, 28: 53705120, 29: 53707394, 30: 53705141,
    31: 53705207, 32: 53705159, 33: 53705138, 34: 53705111, 35: 53705189,
    36: 53705195, 37: 53705186, 38: 53705198, 39: 53705192, 40: 53705201,
}
# fmt: on

_EVENTS = {
    "14": 1,
    "15": 2,
    "16": 3,
    "17": 4,
    "18": 5,
    "19": 6,
    "20": 7,
    "21": 8,
    "14.2": 9,
    "15.2": 10,
    "16.2": 11,
    "17.2": 12,
    "18.2": 13,
    "19.2": 14,
    "20.2": 15,
    "21.2": 16,
    "14.4": 17,
    "15.4": 18,
    "16.4": 19,
    "17.4": 20,
    "18.4": 21,
    "19.4": 22,
    "20.4": 23,
    "21.4": 24,
    "14.6": 25,
    "15.6": 26,
    "16.6": 27,
    "17.6": 28,
    "18.6": 29,
    "19.6": 30,
    "20.6": 31,
    "21.6": 32,
    "14.8": 33,
    "15.8": 34,
    "16.8": 35,
    "17.8": 36,
    "18.8": 37,
    "19.8": 38,
    "20.8": 39,
    "21.8": 40,
}


class Kim2025BetaRange(BaseDataset):
    """40-class beta-range SSVEP speller dataset.

    33-channel EEG (31 scalp + 2 mastoid references) from 40 healthy subjects
    performing a 40-target SSVEP speller with beta-range stimulation
    (14.0-21.8 Hz, 0.2 Hz step, JFPM, 0.5*pi phase steps). 6 blocks of 40
    trials; trial = 1.5 s rest, 0.5 s cue, 5.0 s stimulation. BioSemi
    ActiveTwo at 1024 Hz. Stored epochs span [-2000, 5000] ms around stimulus
    onset; the event is placed at onset and interval=[0.0, 5.0] takes the 5 s
    of stimulation.

    References
    ----------
    .. [1] H. Kim, K. Won, M. Ahn, and S. C. Jun, "A 40-class SSVEP speller
       dataset: beta range stimulation for low-fatigue BCI applications,"
       Scientific Data, vol. 12, p. 1751, 2025.
       DOI: 10.1038/s41597-025-06032-2
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 41)),
            sessions_per_subject=6,
            events=_EVENTS,
            code="Kim2025BetaRange",
            interval=[0.0, 5.0],
            paradigm="ssvep",
            doi="10.1038/s41597-025-06032-2",
        )

    def _get_single_subject_data(self, subject):
        """Return data for one subject across all 6 blocks."""
        fname = self.data_path(subject)
        mat = loadmat(fname, squeeze_me=True, simplify_cells=True)
        eeg_struct = mat["eeg"]

        data = eeg_struct["data"]  # shape: (33, 7168, 40, 6)
        ch_names_raw = list(eeg_struct["chan_locs"])
        srate = int(eeg_struct["srate"])  # 1024
        n_classes = data.shape[2]  # 40
        n_blocks = data.shape[3]  # 6

        # Epoch window is [-2000, 5000] ms; stimulus onset is at +2000 ms
        onset_sample = int(round(2.0 * srate))  # sample 2048

        # Normalize uppercase midline channels to MNE mixed-case convention
        _midline_fix = {"CZ": "Cz", "PZ": "Pz", "OZ": "Oz", "CPZ": "CPz", "POZ": "POz"}
        ch_names = [_midline_fix.get(ch, ch) for ch in ch_names_raw]
        ch_types = _infer_ch_types(ch_names)
        event_ids = np.arange(1, n_classes + 1)

        sessions = {}
        for block_idx in range(n_blocks):
            block_data = data[:, :, :, block_idx]  # (33, 7168, 40)
            block_data = np.transpose(block_data, (2, 0, 1))  # (40, 33, 7168)
            raw = build_raw_from_epochs(
                block_data,
                ch_names,
                srate,
                event_ids,
                "standard_1005",
                ch_types=ch_types,
                onset_sample=onset_sample,
            )
            sessions[str(block_idx)] = {"0": raw}

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number: {subject}")
        file_id = _SSVEP_FILE_IDS[subject]
        url = f"{FIGSHARE_DL_URL}{file_id}"
        return dl.data_dl(url, self.code, path, force_update, verbose)


def _infer_ch_types(ch_names):
    """Infer channel types (31 EEG + 2 mastoid misc channels)."""
    mastoid_aliases = {"M1", "M2", "A1", "A2", "TP9", "TP10"}
    ch_types = ["misc" if ch.upper() in mastoid_aliases else "eeg" for ch in ch_names]
    if ch_types.count("misc") != 2 and len(ch_types) >= 2:
        # Fall back to the dataset convention where the last two channels are mastoids.
        ch_types = ["eeg"] * len(ch_names)
        ch_types[-2:] = ["misc", "misc"]
    return ch_types


def build_raw_from_epochs(
    data,
    ch_names,
    sfreq,
    event_ids,
    montage_name,
    *,
    ch_types=None,
    scale=1e-6,
    buffer_samples=50,
    onset_sample=0,
):
    """Convert (n_trials, n_channels, n_samples) epoched data to continuous Raw.

    Each trial is de-meaned and scaled to volts, a stim channel carries
    ``event_ids`` at ``onset_sample``, and ``buffer_samples`` zeros separate
    the trials (moabb's ``utils.build_raw_from_epochs``).
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(
            "data must have shape (n_trials, n_channels, n_samples), "
            f"got array with shape {data.shape}."
        )
    n_trials, n_channels, n_samples = data.shape

    if isinstance(ch_names, str):
        raise ValueError(
            "ch_names must be a sequence of channel names, not a single string."
        )
    if len(ch_names) != n_channels:
        raise ValueError(
            f"ch_names length ({len(ch_names)}) must match n_channels ({n_channels})."
        )

    if ch_types is None:
        ch_types = ["eeg"] * n_channels
    if isinstance(ch_types, str):
        raise ValueError(
            "ch_types must be a sequence of channel types, not a single string."
        )
    if len(ch_types) != n_channels:
        raise ValueError(
            f"ch_types length ({len(ch_types)}) must match n_channels ({n_channels})."
        )

    event_ids = np.asarray(event_ids)
    if event_ids.ndim != 1:
        raise ValueError(
            "event_ids must be a 1D array-like of length n_trials; "
            f"got shape {event_ids.shape}."
        )
    if len(event_ids) != n_trials:
        raise ValueError(
            f"event_ids length ({len(event_ids)}) must match n_trials ({n_trials})."
        )

    if isinstance(onset_sample, bool) or not isinstance(onset_sample, (int, np.integer)):
        raise ValueError(
            f"onset_sample must be an integer in [0, {n_samples - 1}], got "
            f"{onset_sample!r} ({type(onset_sample).__name__})."
        )
    if onset_sample < 0 or onset_sample >= n_samples:
        raise ValueError(
            f"onset_sample ({onset_sample}) must be between 0 and {n_samples - 1}."
        )

    # De-mean and scale each trial in-place (callers pass freshly created arrays)
    data = data - data.mean(axis=2, keepdims=True)
    data *= scale

    # Build stim channel
    stim = np.zeros((n_trials, 1, n_samples))
    stim[:, 0, onset_sample] = event_ids

    # Combine EEG + stim, add zero-padding buffers
    combined = np.concatenate([data, stim], axis=1)
    n_total_ch = n_channels + 1

    if buffer_samples > 0:
        buff = np.zeros((n_trials, n_total_ch, buffer_samples))
        combined = np.concatenate([buff, combined, buff], axis=2)

    # Flatten trials into continuous data: (n_trials, n_ch, n_time) -> (n_ch, n_trials*n_time)
    continuous = combined.transpose(1, 0, 2).reshape(n_total_ch, -1)

    ch_names_full = list(ch_names) + ["STI"]
    ch_types_full = list(ch_types) + ["stim"]
    info = create_info(ch_names_full, sfreq, ch_types_full)
    raw = RawArray(data=continuous, info=info, verbose=False)
    montage = make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing="ignore")
    return raw
