"""Event-Sens@FA AUC: the epilepsy benchmark's headline metric.

The paper's seizure-detection score (v2 §4.1, App. C.1 Eq. (1)-(2), App. D.1.2;
Fig. 2, Tables 3-5). Per test fold, the probe's 10 s window probabilities are
thresholded at tau = linspace(0, 1, 200); within each recording, contiguous
positive windows merge into predicted events and contiguous seizure windows
into reference events; a reference event is detected if any predicted event
overlaps it, and a predicted event overlapping none is a false alarm
(any-overlap scoring, SzCORE). Sensitivity = detected / reference events,
FA/h = false alarms / test hours (number of test windows x window length).
The sweep is trimmed to its descending branch (tau >= argmax FA/h) and the
sensitivity interpolated onto a log-spaced FA/h grid (:data:`FA_GRID`); grid
points outside the fold's reachable FA/h range are NaN.

Over folds: the fold curves are combined point-wise with ``np.nanmedian``
(the text says "mean"; the code behind every published number takes the
median) and the area under that curve against log10(FA/h) over [0.1, 100]
FA/h, divided by the width of the integrated range, is the value. The +- the
paper prints is the sample SD (ddof=1) of the per-fold AUCs (each fold's own
curve integrated the same way).

Ported verbatim from the paper's code (EEGBenchmarks-teamEpi-fmc, read-only):

* ``paper_figures/scripts/event_sens_fa.py`` -- the fold curve
  (:func:`fold_curve` is its ``event_curve_for_fold``, with its helpers);
* ``paper_figures/scripts/plot_event_sens_fa_models.py`` -- ``FA_GRID``,
  ``AUC_FAR_RANGE``, ``_curve_auc`` (:func:`curve_auc`) and the nanmedian;
* ``paper_figures/cross_dataset/scripts/add_auc_std_to_csv.py`` -- the +-
  (:func:`aggregate`).

Two properties of that code are kept on purpose, because the published numbers
depend on them: windows are taken in array order (recordings delimited by runs
of equal recording ids, events never span two recordings; consecutive windows
merge even across a gap), and the AUC divides by the width of the part of
[0.1, 100] FA/h where the curve is finite, not by the full 3 decades.

numpy only: ``neuroatlas results`` imports it to aggregate folds.
"""
from __future__ import annotations

import warnings
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

#: The FA/h grid the fold curves are sampled on (plot_event_sens_fa_models.FA_GRID).
FA_GRID = np.geomspace(0.01, 100.0, 80)
#: The FA/h range integrated over (AUC_FAR_RANGE): 0.1-100 FA/h, f in [-1, 2].
AUC_FA_RANGE = (0.1, 100.0)
#: Thresholds swept per fold (event_curve_for_fold's n_thresholds default).
N_THRESHOLDS = 200


def _events_per_recording(mask: np.ndarray, rec_starts: np.ndarray) -> np.ndarray:
    """Nx2 array of [start, end_exclusive] for True-runs, never spanning recordings."""
    if mask.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    starts = []
    ends = []
    n = rec_starts.size
    for i in range(n):
        a = int(rec_starts[i])
        b = int(rec_starts[i + 1]) if i + 1 < n else mask.size
        m = mask[a:b]
        if not m.any():
            continue
        diff = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
        starts.append(np.flatnonzero(diff == 1) + a)
        ends.append(np.flatnonzero(diff == -1) + a)
    if not starts:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([np.concatenate(starts), np.concatenate(ends)], axis=1)


def _overlap_counts(ref: np.ndarray, pred: np.ndarray) -> Tuple[int, int]:
    """Detected reference events and false-alarm predicted events, any-overlap rule."""
    if ref.shape[0] == 0:
        return 0, int(pred.shape[0])
    if pred.shape[0] == 0:
        return 0, 0
    ps = pred[:, 0]
    pe = pred[:, 1]
    pred_overlapped = np.zeros(pred.shape[0], dtype=bool)
    detected = 0
    for r0, r1 in ref:
        idx_hi = int(np.searchsorted(ps, r1, side="left"))
        if idx_hi == 0:
            continue
        idx_lo = int(np.searchsorted(pe[:idx_hi], r0, side="right"))
        if idx_lo < idx_hi:
            detected += 1
            pred_overlapped[idx_lo:idx_hi] = True
    return detected, int((~pred_overlapped).sum())


def threshold_sweep(y: np.ndarray, p: np.ndarray, rec_ids: np.ndarray, *, window_s: float,
                    n_thresholds: int = N_THRESHOLDS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(taus, sensitivity, FA/h) over ``linspace(0, 1, n_thresholds)``
    (event_sens_fa._raw_threshold_sweep)."""
    rec_change = np.concatenate(([True], rec_ids[1:] != rec_ids[:-1]))
    rec_starts = np.flatnonzero(rec_change)

    ref_events = _events_per_recording(y.astype(bool), rec_starts)
    n_ref = ref_events.shape[0]
    total_hours = y.size * window_s / 3600.0

    taus = np.linspace(0.0, 1.0, n_thresholds)
    sens = np.empty(taus.size)
    far = np.empty(taus.size)
    for i, tau in enumerate(taus):
        pred_events = _events_per_recording(p >= tau, rec_starts)
        det, fa = _overlap_counts(ref_events, pred_events)
        sens[i] = det / n_ref if n_ref else np.nan
        far[i] = fa / total_hours if total_hours > 0 else np.nan
    return taus, sens, far


def fold_curve(y, p, rec_ids, *, window_s: float, fa_grid: np.ndarray = FA_GRID,
               n_thresholds: int = N_THRESHOLDS) -> np.ndarray:
    """One fold's sensitivity at each ``fa_grid`` point, NaN where unreachable
    (event_sens_fa.event_curve_for_fold).

    *y*: 0/1 window labels, *p*: seizure probabilities, *rec_ids*: one
    recording id per window; windows in temporal order, each recording's
    windows contiguous. *window_s*: seconds per window (the windows are
    non-overlapping, so the test duration is ``len(y) * window_s``).
    """
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p).astype(np.float64)
    rec_ids = np.asarray(rec_ids)
    fa_grid = np.asarray(fa_grid, dtype=np.float64)
    assert y.shape == p.shape == rec_ids.shape, "shape mismatch"
    if y.sum() == 0:
        return np.full(fa_grid.shape, np.nan)

    _, sens, far = threshold_sweep(y, p, rec_ids, window_s=window_s, n_thresholds=n_thresholds)
    if not np.isfinite(far).any():
        return np.full(fa_grid.shape, np.nan)

    # Keep the descending branch: at very low tau a recording is one giant
    # event, so FA/h is low while sensitivity is high.
    peak_idx = int(np.nanargmax(far))
    far_branch = far[peak_idx:]
    sens_branch = sens[peak_idx:]

    # Sort the branch by FA/h for np.interp; at equal FA/h keep the highest sensitivity.
    order = np.argsort(far_branch, kind="stable")
    fa_sorted = far_branch[order]
    sens_sorted = sens_branch[order]
    uniq, inv = np.unique(fa_sorted, return_inverse=True)
    sens_at_uniq = np.zeros_like(uniq)
    np.maximum.at(sens_at_uniq, inv, sens_sorted)

    out = np.full(fa_grid.shape, np.nan)
    in_range = (fa_grid >= uniq[0]) & (fa_grid <= uniq[-1])
    out[in_range] = np.interp(fa_grid[in_range], uniq, sens_at_uniq)
    return out


def curve_auc(sens: np.ndarray, fa_grid: np.ndarray = FA_GRID,
              fa_range: Tuple[float, float] = AUC_FA_RANGE) -> float:
    """Trapezoidal area under sensitivity vs log10(FA/h) over *fa_range*,
    divided by the width of the finite part (plot_event_sens_fa_models._curve_auc).
    NaN when fewer than two finite grid points fall in the range."""
    sens = np.asarray(sens, dtype=np.float64)
    fa_grid = np.asarray(fa_grid, dtype=np.float64)
    lo, hi = fa_range
    in_range = (fa_grid >= lo) & (fa_grid <= hi) & np.isfinite(sens)
    if in_range.sum() < 2:
        return float("nan")
    x = np.log10(fa_grid[in_range])
    y = sens[in_range]
    width = x[-1] - x[0]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(y, x) / width) if width > 0 else float("nan")


def median_curve(curves: Sequence[Sequence[float]]) -> np.ndarray:
    """Point-wise ``np.nanmedian`` of the fold curves (all-NaN columns stay NaN)."""
    arr = np.asarray(curves, dtype=np.float64)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(arr, axis=0)


def aggregate(curves: Iterable[Sequence[float]], fa_grid: np.ndarray = FA_GRID
              ) -> Tuple[float, float]:
    """(AUC of the fold curves' median, sample SD of the per-fold AUCs): the
    paper's ``auc`` and ``auc_std`` (plot_event_sens_fa_radar._auc_for_model,
    add_auc_std_to_csv._per_fold_auc_std). Either is NaN when undefined: no
    curve, fewer than two finite points, fewer than two folds with an AUC."""
    arr = np.asarray([np.asarray(c, dtype=np.float64) for c in curves])
    if arr.size == 0:
        return float("nan"), float("nan")
    auc = curve_auc(median_curve(arr), fa_grid)
    if arr.shape[0] < 2:
        return auc, float("nan")
    per_fold = np.array([curve_auc(c, fa_grid) for c in arr])
    finite = per_fold[np.isfinite(per_fold)]
    sd = float(np.std(finite, ddof=1)) if finite.size >= 2 else float("nan")
    return auc, sd


def _json_float(v: float) -> Optional[float]:
    return None if v is None or not np.isfinite(v) else float(v)


def fold_metrics(y, p, rec_ids, *, window_s: float) -> dict:
    """What a seizure-detection fold records: its curve (NaN as null, so
    results.json stays strict JSON), the grid it is sampled on, and its AUC."""
    curve = fold_curve(y, p, rec_ids, window_s=window_s)
    return {
        "event_sens_fa_auc": _json_float(curve_auc(curve)),
        "event_sens_fa_curve": [_json_float(v) for v in curve],
        "event_sens_fa_grid": [float(v) for v in FA_GRID],
    }


def from_json(values: Sequence[Optional[float]]) -> np.ndarray:
    """A stored curve or grid back as floats, null as NaN."""
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
