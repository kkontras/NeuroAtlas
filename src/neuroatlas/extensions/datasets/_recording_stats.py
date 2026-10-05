"""Per-recording amplitude statistics shared across epilepsy dataio modules.

The output dict is attached to every per-window ``meta`` so backbone wrappers
that need recording-level normalisation (BIOT q95-vector path, SleepFM
recording-level z-score, REVE z-score+clip) can pull it without recomputing
per window. See MODEL_CONTRACTS.md §3 for the canonical contract.

Two implementations:
  * ``compute_recording_stats`` — original eager path. Takes a fully
    materialised (C, T) signal. Best when T is small.
  * ``compute_recording_stats_streaming`` — two-pass streaming path that
    iterates blocks one at a time. Bit-exact to the eager path
    (Welford for mean/std, histogram + bin refinement for q95) but with
    O(C × bins) extra memory instead of O(C × T). Required for the long
    epilepsiae recordings (1000+ h × 19 ch × 4 B = 30 GB unipolar array).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterator

import numpy as np


def compute_recording_stats(signals: np.ndarray) -> Dict[str, Any]:
    """Build the four amplitude stats for a per-recording ``(C, T)`` signal.

    Returns a dict with:
        recording_mean        (C,) per-channel mean
        recording_std         (C,) per-channel std
        recording_q95         (C,) per-channel 95th-percentile of |x|
        recording_q95_bipolar dict[str, float] keyed "i,j" with i<j —
            the 95th-percentile of |signals[i] - signals[j]| for every
            unordered channel pair. Only included when C > 1.

    Raises ValueError on non-2D input. Works on the **unipolar** (raw) signal
    before any montage derivation — so adapters that publish a derived
    bipolar montage should still pass the unipolar source here.
    """
    if signals.ndim != 2:
        raise ValueError(
            f"compute_recording_stats expects (C, T); got {tuple(signals.shape)}"
        )
    abs_signals = np.abs(signals)
    rec_stats: Dict[str, Any] = {
        "recording_mean": signals.mean(axis=1),
        "recording_std": signals.std(axis=1),
        "recording_q95": np.quantile(abs_signals, 0.95, axis=1),
    }
    n_ch = signals.shape[0]
    if n_ch > 1:
        bq: Dict[str, float] = {}
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                bq[f"{i},{j}"] = float(
                    np.quantile(np.abs(signals[i] - signals[j]), 0.95)
                )
        rec_stats["recording_q95_bipolar"] = bq
    return rec_stats


# Histogram parameters for streaming q95. Bin width = HIST_MAX / NBINS.
# At 0.5 µV per bin, we comfortably cover EEG amplitudes (typically <1 mV
# after preprocessing) while keeping memory tiny:
#   per-channel histogram: 200_001 int64 = 1.6 MB
#   for 19 channels + 171 bipolar pairs (at C=19): ~300 MB total — but
#   pairs are only computed when C > 1 in compute_recording_stats anyway.
_NBINS = 200_000
_HIST_MAX = 100_000.0  # µV; |EEG| over this overflows the histogram.


def compute_recording_stats_streaming(
    blocks_iter_factory: Callable[[], Iterator[np.ndarray]],
    n_channels: int,
    *,
    include_bipolar: bool = True,
) -> Dict[str, Any]:
    """Streaming equivalent of ``compute_recording_stats``.

    ``blocks_iter_factory`` is called t${HPC_CLUSTER} — once per pass — and must
    return a fresh iterator over per-block ``(C, T_block)`` float arrays
    each time. Caller is responsible for caching/loading.

    Returns the same dict as ``compute_recording_stats`` (same keys, same
    dtypes, bit-exact for mean/std, exact-to-np.quantile for q95).

    Raises:
        ValueError: if any |x| or |x_i - x_j| exceeds ``_HIST_MAX``.
    """
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1, got {n_channels}")
    C = n_channels
    pairs = [(i, j) for i in range(C) for j in range(i + 1, C)] if (C > 1 and include_bipolar) else []
    n_pairs = len(pairs)

    bin_width = _HIST_MAX / _NBINS

    # ---- Pass 1: Welford (mean, M2) + histograms over |x| and |x_i - x_j| ----
    mean_acc = np.zeros(C, dtype=np.float64)
    M2_acc = np.zeros(C, dtype=np.float64)
    n_total = 0
    # +1 bin for overflow detection (anything >= HIST_MAX lands in bin NBINS).
    hist_uni = np.zeros((C, _NBINS + 1), dtype=np.int64)
    hist_bi = np.zeros((n_pairs, _NBINS + 1), dtype=np.int64) if n_pairs else None

    for block in blocks_iter_factory():
        if block.ndim != 2 or block.shape[0] != C:
            raise ValueError(
                f"streaming stats: expected each block (C={C}, T); got {tuple(block.shape)}"
            )
        T = block.shape[1]
        if T == 0:
            continue

        # Welford parallel update (population mean / M2). Matches
        # signals.mean(axis=1) and signals.std(axis=1) bit-exact in
        # exact arithmetic; floating-point identical for typical sizes.
        block_mean = block.mean(axis=1, dtype=np.float64)
        block_var = block.var(axis=1, dtype=np.float64)  # population (ddof=0)
        new_n = n_total + T
        delta = block_mean - mean_acc
        mean_acc = mean_acc + delta * (T / new_n)
        M2_acc = M2_acc + block_var * T + (delta * delta) * (n_total * T / new_n)
        n_total = new_n

        # Histogram |x|.
        abs_b = np.abs(block)
        # bin index: floor(|x| / bin_width), clip to NBINS (overflow bucket)
        bin_idx = np.minimum((abs_b / bin_width).astype(np.int64), _NBINS)
        for c in range(C):
            np.add.at(hist_uni[c], bin_idx[c], 1)

        # Histogram |x_i - x_j|.
        if hist_bi is not None:
            for k, (i, j) in enumerate(pairs):
                diff = np.abs(block[i] - block[j])
                bin_idx_b = np.minimum((diff / bin_width).astype(np.int64), _NBINS)
                np.add.at(hist_bi[k], bin_idx_b, 1)

    if n_total == 0:
        raise ValueError("streaming stats: empty stream (no blocks / all blocks T=0)")

    # Overflow check.
    if hist_uni[:, _NBINS].sum() > 0:
        raise ValueError(
            f"streaming stats: |signal| exceeds {_HIST_MAX} µV — "
            f"increase _HIST_MAX or pre-clip."
        )
    if hist_bi is not None and hist_bi[:, _NBINS].sum() > 0:
        raise ValueError(
            f"streaming stats: bipolar |x_i - x_j| exceeds {_HIST_MAX} µV — "
            f"increase _HIST_MAX or pre-clip."
        )

    # Final mean/std.
    rec_mean = mean_acc.copy()
    rec_std = np.sqrt(M2_acc / n_total)

    # Identify the bins containing the q=0.95 sort indices for refinement.
    # np.quantile(x, 0.95, interpolation='linear'):
    #   k = 0.95 * (N - 1); below = floor(k); above = ceil(k); alpha = k - below
    #   q = (1 - alpha) * sorted[below] + alpha * sorted[above]
    k_pos = 0.95 * (n_total - 1)
    k_below = int(np.floor(k_pos))
    k_above = int(np.ceil(k_pos))
    alpha = k_pos - k_below

    def _bin_of(cumsum_row: np.ndarray, sort_idx: int) -> int:
        # Smallest bin K such that cumsum_row[K] > sort_idx. searchsorted with
        # side='right' on the cumulative count array gives exactly that.
        return int(np.searchsorted(cumsum_row, sort_idx, side="right"))

    cumsum_uni = hist_uni[:, :_NBINS].cumsum(axis=1)
    target_uni = np.empty((C, 2), dtype=np.int64)  # (lo_bin, hi_bin) per channel
    for c in range(C):
        b_below = _bin_of(cumsum_uni[c], k_below)
        b_above = _bin_of(cumsum_uni[c], k_above)
        target_uni[c, 0] = min(b_below, b_above)
        target_uni[c, 1] = max(b_below, b_above)

    if hist_bi is not None:
        cumsum_bi = hist_bi[:, :_NBINS].cumsum(axis=1)
        target_bi = np.empty((n_pairs, 2), dtype=np.int64)
        for k in range(n_pairs):
            b_below = _bin_of(cumsum_bi[k], k_below)
            b_above = _bin_of(cumsum_bi[k], k_above)
            target_bi[k, 0] = min(b_below, b_above)
            target_bi[k, 1] = max(b_below, b_above)
    else:
        target_bi = None

    # ---- Pass 2: collect samples in target bins per channel/pair ----
    bucket_uni = [[] for _ in range(C)]
    bucket_bi = [[] for _ in range(n_pairs)] if hist_bi is not None else None

    for block in blocks_iter_factory():
        abs_b = np.abs(block)
        for c in range(C):
            lo, hi = target_uni[c, 0], target_uni[c, 1]
            lo_v = lo * bin_width
            hi_v = (hi + 1) * bin_width  # inclusive of hi-th bin
            mask = (abs_b[c] >= lo_v) & (abs_b[c] < hi_v)
            if mask.any():
                bucket_uni[c].append(abs_b[c][mask].astype(np.float64))
        if bucket_bi is not None:
            for k, (i, j) in enumerate(pairs):
                lo, hi = target_bi[k, 0], target_bi[k, 1]
                lo_v = lo * bin_width
                hi_v = (hi + 1) * bin_width
                diff = np.abs(block[i] - block[j])
                mask = (diff >= lo_v) & (diff < hi_v)
                if mask.any():
                    bucket_bi[k].append(diff[mask].astype(np.float64))

    # ---- Refine: sort the bin-local samples and pick exact q95 ----
    rec_q95 = np.empty(C, dtype=np.float64)
    for c in range(C):
        in_bin = np.concatenate(bucket_uni[c]) if bucket_uni[c] else np.empty(0)
        in_bin.sort(kind="quicksort")
        lo_bin = int(target_uni[c, 0])
        n_below = int(cumsum_uni[c, lo_bin - 1]) if lo_bin > 0 else 0
        bi = k_below - n_below
        ai = k_above - n_below
        if bi < 0 or ai >= len(in_bin):
            raise RuntimeError(
                f"streaming stats: q95 refinement failed for channel {c} "
                f"(below={bi}, above={ai}, in_bin_size={len(in_bin)})"
            )
        rec_q95[c] = (1.0 - alpha) * in_bin[bi] + alpha * in_bin[ai]

    rec_stats: Dict[str, Any] = {
        "recording_mean": rec_mean,
        "recording_std": rec_std,
        "recording_q95": rec_q95,
    }

    if hist_bi is not None:
        bq: Dict[str, float] = {}
        for k, (i, j) in enumerate(pairs):
            in_bin = np.concatenate(bucket_bi[k]) if bucket_bi[k] else np.empty(0)
            in_bin.sort(kind="quicksort")
            lo_bin = int(target_bi[k, 0])
            n_below = int(cumsum_bi[k, lo_bin - 1]) if lo_bin > 0 else 0
            bi = k_below - n_below
            ai = k_above - n_below
            if bi < 0 or ai >= len(in_bin):
                raise RuntimeError(
                    f"streaming stats: bipolar q95 refinement failed for pair "
                    f"({i},{j}) (below={bi}, above={ai}, in_bin_size={len(in_bin)})"
                )
            bq[f"{i},{j}"] = float((1.0 - alpha) * in_bin[bi] + alpha * in_bin[ai])
        rec_stats["recording_q95_bipolar"] = bq

    return rec_stats


# ---------------------------------------------------------------------------
# Disk cache — one .npz per recording, model-agnostic, reusable across runs.
# ---------------------------------------------------------------------------
from pathlib import Path as _Path


def save_recording_stats(path, stats: Dict[str, Any], fingerprint: Dict[str, Any]) -> None:
    """Persist a per-recording stats dict to a single .npz.

    fingerprint is stored alongside (e.g. target_fs, total_samples) so the
    loader can refuse stale cache entries when the upstream signal changes.
    """
    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "recording_mean": np.asarray(stats["recording_mean"], dtype=np.float64),
        "recording_std": np.asarray(stats["recording_std"], dtype=np.float64),
        "recording_q95": np.asarray(stats["recording_q95"], dtype=np.float64),
    }
    bq = stats.get("recording_q95_bipolar")
    if bq is not None:
        # Preserve insertion order: emit two parallel arrays.
        keys = np.asarray(list(bq.keys()), dtype=object)
        vals = np.asarray([float(v) for v in bq.values()], dtype=np.float64)
        payload["bipolar_keys"] = keys
        payload["bipolar_vals"] = vals
    # Fingerprint as two parallel arrays (avoid pickle / object dtype headaches).
    payload["fingerprint_keys"] = np.asarray(list(fingerprint.keys()), dtype=object)
    payload["fingerprint_vals"] = np.asarray(
        [str(v) for v in fingerprint.values()], dtype=object,
    )
    np.savez(p, **payload)


def load_recording_stats(path, fingerprint: Dict[str, Any]) -> Dict[str, Any] | None:
    """Load a stats dict from disk; return None on miss or fingerprint mismatch.

    fingerprint values are stringified before comparison (matches save side).
    """
    p = _Path(path)
    if not p.exists():
        return None
    try:
        z = np.load(p, allow_pickle=True)
    except Exception:
        return None
    try:
        keys = list(z["fingerprint_keys"])
        vals = list(z["fingerprint_vals"])
        on_disk = {str(k): str(v) for k, v in zip(keys, vals)}
        want = {str(k): str(v) for k, v in fingerprint.items()}
        if on_disk != want:
            return None
        out: Dict[str, Any] = {
            "recording_mean": np.asarray(z["recording_mean"]),
            "recording_std": np.asarray(z["recording_std"]),
            "recording_q95": np.asarray(z["recording_q95"]),
        }
        if "bipolar_keys" in z.files:
            bk = list(z["bipolar_keys"])
            bv = list(z["bipolar_vals"])
            out["recording_q95_bipolar"] = {str(k): float(v) for k, v in zip(bk, bv)}
        return out
    finally:
        z.close()


# ---------------------------------------------------------------------------
# Where the readers keep them
# ---------------------------------------------------------------------------

def _legacy_stats_paths(reader: str, filename: str):
    """Where earlier versions wrote a reader's stats: ``artifacts/
    recording_stats/<reader>/`` relative to the directory a command ran in,
    and the same under a source checkout."""
    from neuroatlas._paths import checkout_root

    rel = _Path("artifacts") / "recording_stats" / reader / filename
    seen = []
    for base in (_Path.cwd(), checkout_root()):
        if base is None:
            continue
        path = (base / rel).resolve()
        if path not in seen:
            seen.append(path)
    return seen


def load_cached_recording_stats(reader: str, filename: str, fingerprint: Dict[str, Any]):
    """``(stats or None, path)`` for one recording of *reader*.

    *path* is ``<cache root>/recording_stats/<reader>/<filename>``
    (``_paths.recording_stats_dir``), where a reader saves what it computes.
    On a miss there, a file an earlier version left in ``artifacts/
    recording_stats`` (current directory, or the checkout) with a matching
    fingerprint is used and copied to *path*; otherwise the caller computes.
    """
    import shutil

    from neuroatlas._paths import recording_stats_dir

    path = recording_stats_dir(reader, filename)
    stats = load_recording_stats(path, fingerprint)
    if stats is not None:
        return stats, path
    for legacy in _legacy_stats_paths(reader, filename):
        if legacy == path.resolve():
            continue
        stats = load_recording_stats(legacy, fingerprint)
        if stats is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy, path)
            except OSError:
                pass
            return stats, path
    return None, path
