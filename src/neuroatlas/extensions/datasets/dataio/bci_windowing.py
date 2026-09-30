"""Patched braindecode windowing functions for BCI datasets.

Ported from BCI_DG_RPA/windows_mine.py.  The key difference from braindecode's
built-in ``create_windows_from_events`` is that ``_create_windows_from_events``
forces all trial annotation durations to 4.0 seconds before windowing.  This is
required for PhysionetMI (and used by Cho2017 / Lee2019_MI) where the raw MOABB
annotations may have inconsistent durations.

Original authors (BSD-3 license): Hubert Banville, Lukas Gemein, Simon Brandt,
David Sabbagh, Henrik Bonsmann, Ann-Kathrin Kiessner, Vytautas Jankauskas,
Dan Wilson, Maciej Sliwowski, Mohammed Fattouh.
Local patches: Georgios Zoumpourlis, Anonymous Karakai.
"""

from __future__ import annotations

import warnings

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from braindecode.datasets.base import BaseConcatDataset, WindowsDataset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_windows_from_events(
    concat_ds,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=None,
    window_stride_samples=None,
    drop_last_window=False,
    mapping=None,
    preload=False,
    drop_bad_windows=True,
    picks=None,
    reject=None,
    flat=None,
    on_missing="error",
    accepted_bads_ratio=0.0,
    n_jobs=1,
    verbose="error",
    forced_trial_duration=4.0,
):
    """Create windows based on events — with configurable trial-duration patch."""
    _check_windowing_arguments(
        trial_start_offset_samples,
        trial_stop_offset_samples,
        window_size_samples,
        window_stride_samples,
    )

    infer_mapping = mapping is None
    mapping = dict() if infer_mapping else mapping
    infer_window_size_stride = window_size_samples is None

    list_of_windows_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_windows_from_events)(
            ds,
            infer_mapping,
            infer_window_size_stride,
            trial_start_offset_samples,
            trial_stop_offset_samples,
            window_size_samples,
            window_stride_samples,
            drop_last_window,
            mapping,
            preload,
            drop_bad_windows,
            picks,
            reject,
            flat,
            on_missing,
            accepted_bads_ratio,
            verbose,
            forced_trial_duration,
        )
        for ds in concat_ds.datasets
    )
    return BaseConcatDataset(list_of_windows_ds)


def create_fixed_length_windows(
    concat_ds,
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=None,
    window_stride_samples=None,
    drop_last_window=None,
    mapping=None,
    preload=False,
    drop_bad_windows=True,
    picks=None,
    reject=None,
    flat=None,
    targets_from="metadata",
    last_target_only=True,
    on_missing="error",
    n_jobs=1,
    verbose="error",
):
    """Windower that creates sliding fixed-length windows."""
    stop_offset_samples, drop_last_window = _check_and_set_fixed_length_window_arguments(
        start_offset_samples,
        stop_offset_samples,
        window_size_samples,
        window_stride_samples,
        drop_last_window,
    )

    lengths = np.array([ds.raw.n_times for ds in concat_ds.datasets])
    if (np.diff(lengths) != 0).any() and window_size_samples is None:
        warnings.warn("Recordings have different lengths, they will not be batch-able!")
    if any(window_size_samples > lengths):
        raise ValueError(
            f"Window size {window_size_samples} exceeds trial duration {lengths.min()}."
        )

    list_of_windows_ds = Parallel(n_jobs=n_jobs)(
        delayed(_create_fixed_length_windows)(
            ds,
            start_offset_samples,
            stop_offset_samples,
            window_size_samples,
            window_stride_samples,
            drop_last_window,
            mapping,
            preload,
            drop_bad_windows,
            picks,
            reject,
            flat,
            targets_from,
            last_target_only,
            on_missing,
            verbose,
        )
        for ds in concat_ds.datasets
    )
    return BaseConcatDataset(list_of_windows_ds)


# ---------------------------------------------------------------------------
# Per-dataset workers
# ---------------------------------------------------------------------------

def _create_windows_from_events(
    ds,
    infer_mapping,
    infer_window_size_stride,
    trial_start_offset_samples,
    trial_stop_offset_samples,
    window_size_samples=None,
    window_stride_samples=None,
    drop_last_window=False,
    mapping=None,
    preload=False,
    drop_bad_windows=True,
    picks=None,
    reject=None,
    flat=None,
    on_missing="error",
    accepted_bads_ratio=0.0,
    verbose="error",
    forced_trial_duration=4.0,
):
    """Per-recording windowing with configurable trial-duration patch."""
    window_kwargs = [
        (create_windows_from_events.__name__, _get_windowing_kwargs(locals())),
    ]
    if infer_mapping:
        unique_events = np.unique(ds.raw.annotations.description)
        new_unique_events = [x for x in unique_events if x not in mapping]
        max_id_mapping = len(mapping)
        mapping.update(
            {v: k + max_id_mapping for k, v in enumerate(new_unique_events)}
        )

    events, events_id = mne.events_from_annotations(ds.raw, mapping)
    onsets = events[:, 0]

    # ------------------------------------------------------------------
    # PATCH: force all trial durations to 4.0 seconds
    # (required for PhysionetMI and used by Cho2017 / Lee2019_MI)
    # ------------------------------------------------------------------
    onset_vals = np.array([el["onset"] for el in ds.raw.annotations])
    duration_vals = np.array([el["duration"] for el in ds.raw.annotations])
    description_vals = np.array([el["description"] for el in ds.raw.annotations])

    sfreq = ds.raw.info["sfreq"]
    last_samp = ds.raw.first_samp + ds.raw.n_times

    # Drop trials that would overflow the recording after forcing the
    # target duration (typically 1-3 trials at the end of a recording).
    keep_mask = np.ones(len(onset_vals), dtype=bool)
    for trial_cnt in range(len(onset_vals)):
        forced_stop = onset_vals[trial_cnt] * sfreq + forced_trial_duration * sfreq
        if forced_stop > last_samp:
            keep_mask[trial_cnt] = False
        else:
            duration_vals[trial_cnt] = forced_trial_duration

    n_dropped = (~keep_mask).sum()
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped}/{len(onset_vals)} trials that overflow "
            f"the recording after forcing {forced_trial_duration}s duration."
        )
        onset_vals = onset_vals[keep_mask]
        duration_vals = duration_vals[keep_mask]
        description_vals = description_vals[keep_mask]

    annotations_fixed = mne.Annotations(onset_vals, duration_vals, description_vals)
    ds.raw.set_annotations(annotations_fixed)
    # ------------------------------------------------------------------

    # Re-extract events after annotation update (dropped trials removed).
    events, events_id = mne.events_from_annotations(ds.raw, mapping)
    onsets = events[:, 0]

    filtered_durations = np.array(
        [a["duration"] for a in ds.raw.annotations if a["description"] in events_id]
    )
    stops = onsets + (filtered_durations * sfreq).astype(int)

    if len(stops) > 0 and stops[-1] + trial_stop_offset_samples > last_samp:
        raise ValueError(
            '"trial_stop_offset_samples" too large. Stop of last trial '
            f"({stops[-1]}) + trial_stop_offset_samples "
            f"({trial_stop_offset_samples}) must be smaller than length of"
            f" recording ({len(ds)})."
        )

    if infer_window_size_stride:
        if window_size_samples is None:
            window_size_samples = stops[0] + trial_stop_offset_samples - (
                onsets[0] + trial_start_offset_samples
            )
            window_stride_samples = window_size_samples
        this_trial_sizes = (stops + trial_stop_offset_samples) - (
            onsets + trial_start_offset_samples
        )
        if not np.all(this_trial_sizes == window_size_samples):
            bad = this_trial_sizes != window_size_samples
            warnings.warn(
                f"Dropping {bad.sum()} additional trials with mismatched "
                f"sizes (expected {window_size_samples}, got "
                f"{np.unique(this_trial_sizes[bad])})."
            )
            onsets = onsets[~bad]
            stops = stops[~bad]
            events = events[~bad]
            filtered_durations = filtered_durations[~bad]

    description = events[:, -1]

    i_trials, i_window_in_trials, starts, stops = _compute_window_inds(
        onsets,
        stops,
        trial_start_offset_samples,
        trial_stop_offset_samples,
        window_size_samples,
        window_stride_samples,
        drop_last_window,
        accepted_bads_ratio,
    )

    events = [
        [start, window_size_samples, description[i_trials[i_start]]]
        for i_start, start in enumerate(starts)
    ]
    events = np.array(events)

    if any(np.diff(events[:, 0]) <= 0):
        raise NotImplementedError("Trial overlap not implemented.")

    description = events[:, -1]

    metadata = pd.DataFrame(
        {
            "i_window_in_trial": i_window_in_trials,
            "i_start_in_trial": starts,
            "i_stop_in_trial": stops,
            "target": description,
        }
    )

    mne_epochs = mne.Epochs(
        ds.raw,
        events,
        events_id,
        baseline=None,
        tmin=0,
        tmax=(window_size_samples - 1) / ds.raw.info["sfreq"],
        metadata=metadata,
        preload=preload,
        picks=picks,
        reject=reject,
        flat=flat,
        on_missing=on_missing,
        verbose=verbose,
    )

    if drop_bad_windows:
        mne_epochs.drop_bad()

    windows_ds = WindowsDataset(mne_epochs, ds.description)
    setattr(windows_ds, "window_kwargs", window_kwargs)
    kwargs_name = "raw_preproc_kwargs"
    if hasattr(ds, kwargs_name):
        setattr(windows_ds, kwargs_name, getattr(ds, kwargs_name))
    return windows_ds


def _create_fixed_length_windows(
    ds,
    start_offset_samples,
    stop_offset_samples,
    window_size_samples,
    window_stride_samples,
    drop_last_window,
    mapping=None,
    preload=False,
    drop_bad_windows=True,
    picks=None,
    reject=None,
    flat=None,
    targets_from="metadata",
    last_target_only=True,
    on_missing="error",
    verbose="error",
):
    """Per-recording fixed-length windowing (with AIK patches)."""
    window_kwargs = [
        (create_fixed_length_windows.__name__, _get_windowing_kwargs(locals())),
    ]
    stop = ds.raw.n_times if stop_offset_samples is None else stop_offset_samples

    if window_size_samples is None:
        window_size_samples = stop - start_offset_samples
    if window_stride_samples is None:
        window_stride_samples = window_size_samples

    stop = stop - window_size_samples + ds.raw.first_samp
    starts = np.arange(
        ds.raw.first_samp + start_offset_samples, stop + 1, window_stride_samples
    )

    if not drop_last_window and starts[-1] < stop:
        starts = np.append(starts, stop)

    target = -1 if ds.target_name is None else ds.description[ds.target_name]
    if mapping is not None:
        if isinstance(target, pd.Series):
            target = target.replace(mapping).to_list()
        else:
            target = mapping[target]

    fake_events = [[start, window_size_samples, target] for start in starts]
    fake_events = np.array(fake_events)
    description = fake_events[:, -1]
    metadata = pd.DataFrame(
        {
            "i_window_in_trial": np.arange(len(fake_events)),
            "i_start_in_trial": starts,
            "i_stop_in_trial": starts + window_size_samples,
            "target": description,
        }
    )

    if any(np.diff(fake_events[:, 0]) <= 0):
        raise NotImplementedError("Trial overlap not implemented.")

    events_id = mapping
    on_missing = "warn"
    mne_epochs = mne.Epochs(
        ds.raw,
        fake_events,
        event_id=events_id,
        baseline=None,
        tmin=0,
        tmax=(window_size_samples - 1) / ds.raw.info["sfreq"],
        metadata=metadata,
        preload=preload,
        picks=picks,
        reject=reject,
        flat=flat,
        on_missing=on_missing,
        verbose=verbose,
    )

    if drop_bad_windows:
        mne_epochs.drop_bad()

    window_kwargs.append(
        (
            WindowsDataset.__name__,
            {"targets_from": targets_from, "last_target_only": last_target_only},
        )
    )
    windows_ds = WindowsDataset(
        mne_epochs,
        ds.description,
        targets_from=targets_from,
        last_target_only=last_target_only,
    )
    setattr(windows_ds, "window_kwargs", window_kwargs)
    kwargs_name = "raw_preproc_kwargs"
    if hasattr(ds, kwargs_name):
        setattr(windows_ds, kwargs_name, getattr(ds, kwargs_name))
    return windows_ds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_window_inds(
    starts, stops, start_offset, stop_offset, size, stride, drop_last_window, accepted_bads_ratio
):
    """Compute window start and stop indices."""
    starts = np.array([starts]) if isinstance(starts, int) else starts
    stops = np.array([stops]) if isinstance(stops, int) else stops

    starts = starts + start_offset
    stops = stops + stop_offset
    if any(size > (stops - starts)):
        bads_mask = size > (stops - starts)
        min_duration = (stops - starts).min()
        if sum(bads_mask) <= accepted_bads_ratio * len(starts):
            starts = starts[np.logical_not(bads_mask)]
            stops = stops[np.logical_not(bads_mask)]
            warnings.warn(
                f"Trials {np.where(bads_mask)[0]} are being dropped as the "
                f"window size ({size}) exceeds their duration {min_duration}."
            )
        else:
            current_ratio = sum(bads_mask) / len(starts)
            raise ValueError(
                f"Window size {size} exceeds trial duration "
                f"({min_duration}) for too many trials "
                f"({current_ratio * 100}%). Set "
                f"accepted_bads_ratio to at least {current_ratio}"
                " and restart training to be able to continue."
            )

    i_window_in_trials, i_trials, window_starts = [], [], []
    for start_i, (start, stop) in enumerate(zip(starts, stops)):
        possible_starts = np.arange(start, stop, stride)
        for i_window, s in enumerate(possible_starts):
            if (s + size) <= stop:
                window_starts.append(s)
                i_window_in_trials.append(i_window)
                i_trials.append(start_i)

        if not drop_last_window:
            if window_starts[-1] + size != stop:
                window_starts.append(stop - size)
                i_window_in_trials.append(i_window_in_trials[-1] + 1)
                i_trials.append(start_i)

    window_stops = np.array(window_starts) + size
    if not (len(i_window_in_trials) == len(window_starts) == len(window_stops)):
        raise ValueError(
            f"{len(i_window_in_trials)} == {len(window_starts)} == {len(window_stops)}"
        )

    return i_trials, i_window_in_trials, window_starts, window_stops


def _check_windowing_arguments(
    trial_start_offset_samples,
    trial_stop_offset_samples,
    window_size_samples,
    window_stride_samples,
):
    assert isinstance(trial_start_offset_samples, (int, np.integer))
    assert isinstance(trial_stop_offset_samples, (int, np.integer)) or (
        trial_stop_offset_samples is None
    )
    assert isinstance(window_size_samples, (int, np.integer, type(None)))
    assert isinstance(window_stride_samples, (int, np.integer, type(None)))
    assert (window_size_samples is None) == (window_stride_samples is None)
    if window_size_samples is not None:
        assert window_size_samples > 0, "window size has to be larger than 0"
        assert window_stride_samples > 0, "window stride has to be larger than 0"


def _check_and_set_fixed_length_window_arguments(
    start_offset_samples,
    stop_offset_samples,
    window_size_samples,
    window_stride_samples,
    drop_last_window,
):
    _check_windowing_arguments(
        start_offset_samples,
        stop_offset_samples,
        window_size_samples,
        window_stride_samples,
    )

    if stop_offset_samples == 0:
        warnings.warn(
            "Meaning of `trial_stop_offset_samples`=0 has changed, use `None` "
            "to indicate end of trial/recording. Using `None`."
        )
        stop_offset_samples = None

    if start_offset_samples != 0 or stop_offset_samples is not None:
        warnings.warn(
            "Usage of offset_sample args in create_fixed_length_windows is deprecated and"
            " will be removed in future versions. Please use "
            'braindecode.preprocessing.preprocess.Preprocessor("crop", tmin, tmax)'
            " instead."
        )

    if (
        window_size_samples is not None
        and window_stride_samples is not None
        and drop_last_window is None
    ):
        raise ValueError(
            "drop_last_window must be set if both window_size_samples &"
            " window_stride_samples have also been set"
        )
    elif (
        window_size_samples is None
        and window_stride_samples is None
        and drop_last_window is False
    ):
        drop_last_window = None

    assert (window_size_samples is None) == (window_stride_samples is None) == (
        drop_last_window is None
    )

    return stop_offset_samples, drop_last_window


def _get_windowing_kwargs(windowing_func_locals):
    input_kwargs = windowing_func_locals
    input_kwargs.pop("ds")
    return {k: v for k, v in input_kwargs.items()}
