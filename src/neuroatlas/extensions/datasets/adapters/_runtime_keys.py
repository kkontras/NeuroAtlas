"""Declared answers to the runtime keys the runner offers every datamodule.

``DatasetSpec.build_config`` and ``BenchmarkRunner._pair_config`` add
``signal_kind``, ``epoch_seconds``, ``notch``/``highpass``,
``compute_recording_stats`` and ``channel_specs`` to a dataset's config
whatever the dataset is. A datamodule that has no constructor argument for one
of them declares, in ``RUNTIME_KEYS_FIXED`` / ``RUNTIME_KEYS_IGNORED`` (see
``BenchmarkDataModule``), the one value it serves or why the key does not
apply. ``construct_datamodule`` enforces the declaration. The reasons live
here so that every cohort of a kind gives the same one.
"""
from __future__ import annotations

from typing import Any, Dict

#: Every datamodule that only ever serves the raw time series.
RAW_ONLY: Dict[str, Any] = {"signal_kind": "raw"}

#: Staging-only sleep readers: the task preset names the label mode for every
#: dataset it runs on (configs/tasks/sleep_staging.json), and these serve no other.
SLEEP_STAGE_ONLY: Dict[str, Any] = {"label_mode": "sleep_stage"}

TRIAL_WINDOW = (
    "the window is the trial, cut where the paper cut it (dataio/"
    "moabb_loader.trial_window: DATASET_CONFIGS tmin/trial_duration, else the "
    "4 s after the cue for motor imagery); the runner tells the backbone that "
    "length (BenchmarkRunner._fit_checkpoint_to_window)"
)

TRIAL_NORMALISATION = (
    "BCI trials are published without per-recording statistics -- the BCI "
    "reference pipeline the published BCI embeddings came from never computed "
    "them -- so each model that normalises per recording takes its own "
    "no-statistics path (BIOT per-window q95, REVE per-window z-score, ...)"
)

#: The BCI trial readers (MOABB and the prepared-pickle cohorts).
BCI_TRIALS: Dict[str, str] = {
    "epoch_seconds": TRIAL_WINDOW,
    "compute_recording_stats": TRIAL_NORMALISATION,
}

#: Cohorts read from a file whose filtering was fixed when it was prepared.
PREPARED_FILTERS: Dict[str, str] = {
    "notch": (
        "the notch was applied when the preprocessed file was written "
        "(dataio/bci.py DATASET_CONFIGS notch_freq); a loaded pickle is not re-filtered"
    ),
    "highpass": (
        "the band-pass was applied when the preprocessed file was written "
        "(dataio/bci.py DATASET_CONFIGS fmin/fmax); a loaded pickle is not re-filtered"
    ),
}
