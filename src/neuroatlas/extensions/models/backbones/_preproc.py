"""Shared preprocessing helpers for backbone wrappers.

Every helper here is a single, strictly model-specific transform. Signal
cleanliness (bandpass / notch) is a dataset-preprocessor concern and is
intentionally absent. Time-window length handling is also absent: each
backbone declares its own window contract (strict fixed vs. patch-multiple)
and raises on mismatch — we never silently stretch or pad.

See AGENT_GUIDE.md §"Preprocessing responsibilities" for the split.
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from math import gcd
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Sleep datasets whose embeddings reproduce the paper's reference embeddings exactly.
# Guards in individual wrappers use this set to apply that preprocessing
# (snap_to_epoch_length, strip_zero_channels, adaptive epoch_seconds)
# without affecting other datasets.
_PAPER_PREPROC_DATASETS = frozenset({
    "dcsm", "dod", "dodh", "dodo", "isruc", "mass",
    "physionet2026", "sleep_edf_expanded", "ucddb", "wsc",
    "mros_raw_brain_age", "cfs_raw_2ch",
})

#: ``meta[i]["domain"]`` of a BCI trial. Every BCI reader stamps it (the MOABB
#: reader, physionet_mi, cho2017, lee2019_mi, hinss2021 and the cognitive
#: cohorts); :func:`is_bci_batch` reads it.
BCI_DOMAIN = "bci"


def run_names(meta: Optional[Sequence[Mapping[str, Any]]] = None,
              checkpoint: str = "") -> Tuple[str, str]:
    """``(dataset, checkpoint)`` a message about this batch names: the run the
    runner is working on, else the batch's ``meta[0]["dataset"]`` and
    *checkpoint* (the wrapper's ``spec.identifier``); ``""`` where unknown."""
    try:
        from neuroatlas.benchmarking_helpers.runtime.pair import current
        pair = current()
    except ImportError:                                  # pragma: no cover
        pair = None
    if pair is not None:
        return pair.dataset, pair.checkpoint
    dataset = ""
    if meta:
        dataset = str((meta[0] or {}).get("dataset") or "")
    return dataset, str(checkpoint or "")


def run_label(meta: Optional[Sequence[Mapping[str, Any]]] = None,
              checkpoint: str = "") -> str:
    """``hmc/reve_pretrained`` (or the one name known, or ``""``)."""
    return "/".join(part for part in run_names(meta, checkpoint) if part)


def is_bci_batch(meta: Optional[Sequence[Mapping[str, Any]]]) -> bool:
    """Whether a batch holds BCI trials.

    EEGPT, LaBraM, NeuroLM, NeuroGPT, REVE and SleepFM treat BCI trials the way
    the pipeline behind the paper's BCI embeddings did (the BCI parity report,
    ``git show 151f7f5:REPORT-BCI-Backbone-Changes.md``). That pipeline
    (probe_bci_loso.py) stamped ``meta["dataset"] = "bci"`` on every batch,
    and the wrappers keyed their BCI branches on it. The readers that replaced
    it send the cohort's slug (``bnci2014_001``, ...) as ``dataset``, so none
    of those branches ever ran -- SleepFM refused every BCI batch, EEGPT
    applied CAR and the x1000 scale of the other domains, LaBraM returned its
    200-d sleep embedding instead of the 400-d CLS + patch mean.

    The signal is now explicit and separate from the slug: the readers set
    ``meta["domain"] = "bci"``. ``dataset == "bci"`` still counts, so a batch
    built the old way keeps its meaning.
    """
    if not meta:
        return False
    first = meta[0] or {}
    return first.get("domain") == BCI_DOMAIN or first.get("dataset") == BCI_DOMAIN


# ---------------------------------------------------------------------------
# Sub-stage timing — gated by BACKBONE_TRACE env var. Each call to
# ``stage_time(tag, label)`` records a CUDA-synced elapsed-time sample.
# ``dump_stage_times()`` prints a per-stage mean in ms after N samples.
#
#   BACKBONE_TRACE=1  enables timing
#   BACKBONE_TRACE_N=20  sample count per stage before dump (default 20)
# ---------------------------------------------------------------------------

_STAGE_STATS: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
_STAGE_LIMIT = int(os.environ.get("BACKBONE_TRACE_N", "20"))
_STAGE_SKIP = int(os.environ.get("BACKBONE_TRACE_SKIP", "0"))
_STAGE_SEEN: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))


class StageTimer:
    """Context manager that records CUDA-synced elapsed time for a stage.

    Usage:
        with StageTimer("biot", "prep"):
            ...GPU or CPU work...

    No-op when BACKBONE_TRACE is unset, so hot paths pay only a check.
    """
    __slots__ = ("_tag", "_label", "_enabled", "_t0")

    def __init__(self, tag: str, label: str):
        self._tag = tag
        self._label = label
        self._enabled = bool(os.environ.get("BACKBONE_TRACE"))

    def __enter__(self):
        if not self._enabled:
            return self
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0
        seen = _STAGE_SEEN[self._tag][self._label]
        _STAGE_SEEN[self._tag][self._label] = seen + 1
        if seen < _STAGE_SKIP:
            return  # discard warmup sample (e.g. torch.compile first-call cost)
        stats = _STAGE_STATS[self._tag][self._label]
        if len(stats) < _STAGE_LIMIT:
            stats.append(dt)
            if len(stats) == _STAGE_LIMIT:
                # Dump this tag's table once it's full; cheap to call repeatedly.
                dump_stage_times(tag=self._tag)


def dump_stage_times(tag: str | None = None) -> None:
    """Pretty-print per-stage mean/count for the given tag (or all tags)."""
    keys = [tag] if tag is not None else sorted(_STAGE_STATS.keys())
    for t in keys:
        entries = _STAGE_STATS.get(t, {})
        if not entries:
            continue
        total = sum(sum(v) / len(v) for v in entries.values())
        print(f"[backbone-trace {t}] mean over first {_STAGE_LIMIT} batches (ms):")
        for label, vals in entries.items():
            mean_ms = 1000.0 * sum(vals) / len(vals)
            pct = 100.0 * (sum(vals) / len(vals)) / total if total > 0 else 0.0
            print(f"    {label:<24s} {mean_ms:8.2f} ms   ({pct:5.1f}%)   n={len(vals)}")
        print(f"    {'TOTAL':<24s} {1000.0*total:8.2f} ms", flush=True)


_MIN_PLAUSIBLE_FS = 1.0
_MAX_PLAUSIBLE_FS = 10_000.0


def resample_poly_with_fallback(
    x: torch.Tensor, src_fs: float, dst_fs: float,
    backend: str = "auto",
) -> Tuple[torch.Tensor, str]:
    """Resample along the last axis from ``src_fs`` to ``dst_fs``.

    Uses ``scipy.signal.resample_poly`` (polyphase FIR with anti-alias) when
    both rates are clean positive integers; falls back to linear interpolation
    otherwise. Does NOT change the time-axis *length* to any target — only the
    sampling-rate ratio. Callers are responsible for validating the resulting
    length against their window contract.

    Raises ``ValueError`` when ``src_fs`` is non-finite, ≤ 0, > 10 kHz, or
    when ``dst_fs`` is similarly implausible. This catches unit/metadata bugs
    (e.g. ``src_fs=0.256`` instead of ``256``).

    Returns ``(resampled_x, method_name)`` where ``method_name`` is one of
    ``"identity"``, ``"resample_poly"``, ``"linear_interp_fallback"``.
    """
    if not np.isfinite(src_fs) or src_fs < _MIN_PLAUSIBLE_FS or src_fs > _MAX_PLAUSIBLE_FS:
        raise ValueError(
            f"implausible src_fs={src_fs!r}; expected within "
            f"[{_MIN_PLAUSIBLE_FS}, {_MAX_PLAUSIBLE_FS}] Hz"
        )
    if not np.isfinite(dst_fs) or dst_fs < _MIN_PLAUSIBLE_FS or dst_fs > _MAX_PLAUSIBLE_FS:
        raise ValueError(
            f"implausible dst_fs={dst_fs!r}; expected within "
            f"[{_MIN_PLAUSIBLE_FS}, {_MAX_PLAUSIBLE_FS}] Hz"
        )

    if abs(src_fs - dst_fs) < 1e-6:
        return x, "identity"

    src_int = int(round(src_fs))
    dst_int = int(round(dst_fs))
    # Polyphase only when BOTH rates are effectively integer. Anything else
    # (e.g. Bonn's 173.5 Hz) falls through to linear interp, which preserves
    # the exact expected output length = round(T * dst_fs / src_fs). Without
    # this tighter threshold, a non-integer src_fs was silently rounded to an
    # integer and the polyphase up/down ratio produced off-by-N outputs.
    if abs(src_int - src_fs) > 0.01 or abs(dst_int - dst_fs) > 0.01:
        ratio = dst_fs / src_fs
        new_len = max(1, int(round(x.shape[-1] * ratio)))
        return (
            F.interpolate(x, size=new_len, mode="linear", align_corners=False),
            "linear_interp_fallback",
        )

    g = gcd(dst_int, src_int)
    up = dst_int // g
    down = src_int // g

    try:
        import torchaudio.functional as taF
    except ImportError:
        taF = None

    from neuroatlas import quiet

    _log = logging.getLogger(__name__)

    if taF is not None and backend != "scipy":
        try:
            resampled = taF.resample(x, orig_freq=src_int, new_freq=dst_int)
        except (AttributeError, OSError, RuntimeError) as exc:
            # `import torchaudio.functional` can succeed while the package is
            # unusable: its C extension is built against a specific CUDA
            # runtime, and when that library is absent the module loads but
            # `resample` is never bound. Resampling is a CPU operation scipy
            # does as well, so the batch is resampled there; it asks nothing
            # of the user (-v and --log say it, once per command).
            reason = (str(exc).strip().splitlines() or [""])[0]
            quiet.warn_once(
                _log, "resample: torchaudio cannot run",
                "resampling on the CPU with scipy: torchaudio cannot run here (%s: %s)",
                type(exc).__name__, reason, level=logging.INFO,
            )
        else:
            # On certain GPUs (RTX 3080 Ti / 5060 Ti / 5070 Ti) the kernel
            # silently returns NaN instead of valid samples even on finite
            # input. Detect that and resample the batch with scipy on the CPU.
            if torch.isfinite(resampled).all():
                return resampled, "torchaudio_resample"
            quiet.warn_once(
                _log, f"resample: torchaudio non-finite on {x.device}",
                "torchaudio gave non-finite samples resampling on %s; such batches are "
                "resampled on the CPU with scipy",
                x.device, level=logging.INFO,
            )

    from scipy.signal import resample_poly

    x_np = x.detach().cpu().numpy()
    resampled = resample_poly(x_np, up=up, down=down, axis=-1).astype(np.float32)
    # Sometimes scipy ALSO produces non-finite values — observed on float16
    # cached data with subtle out-of-range samples that pass the cache
    # finite-check but trip the FIR convolution. Those samples are set to 0,
    # and the command says once, at its end, in how many windows.
    nonfinite_mask = ~np.isfinite(resampled)
    if nonfinite_mask.any():
        rows = int(resampled.shape[0]) if resampled.ndim >= 2 else 1
        bad_rows = int(nonfinite_mask.reshape(rows, -1).any(axis=1).sum())
        label = run_label()
        quiet.count(
            f"resample non-finite:{label}",
            (f"{label}: " if label else "")
            + "{hit} window{s} had non-finite samples after resampling; those samples "
              "are set to 0",
            hit=bad_rows,
        )
        _log.debug("resampling: %d non-finite samples set to 0 (%d of %d windows in a batch)",
                   int(nonfinite_mask.sum()), bad_rows, rows)
        resampled[nonfinite_mask] = 0.0
    return torch.from_numpy(resampled).to(x.device), "resample_poly"


def car_reference(x: torch.Tensor) -> torch.Tensor:
    """Common average reference — subtract the mean across the channel axis.

    Input shape: ``(B, C, T)``. Returns the same shape.

    NOTE: this is the correct CAR: mean over dim=1 (channels). Taking the
    mean over dim=-1 (time) is per-channel DC removal, not CAR.
    """
    if x.ndim != 3:
        raise ValueError(f"car_reference expects (B, C, T); got shape {tuple(x.shape)}")
    return x - x.mean(dim=1, keepdim=True)


def assert_finite(x: torch.Tensor, where: str) -> None:
    """Raise ``ValueError`` if ``x`` contains any NaN or Inf."""
    if not torch.isfinite(x).all():
        n_nan = int(torch.isnan(x).sum())
        n_inf = int(torch.isinf(x).sum())
        raise ValueError(
            f"non-finite tensor at {where}: n_nan={n_nan}, n_inf={n_inf}, "
            f"shape={tuple(x.shape)}"
        )


def replace_nonfinite_with_zero(x: torch.Tensor, where: str, meta=None,
                                checkpoint: str = "") -> torch.Tensor:
    """Soft variant of :func:`assert_finite` for backbone *outputs*.

    Some TUSZ windows trigger single-window NaN blowups inside the pretrained
    stack (LaBraM/EEGPT/NeuroLM/BIoT have all been seen doing this on <0.01%
    of windows). Hard-raising at the backbone output kills the whole embedding
    run; instead this helper sets the NaN/Inf entries to 0 and the command
    says once, at its end, for how many windows of which dataset and
    checkpoint (*meta*, *checkpoint*: see :func:`run_names`; *where* is
    the wrapper's own tag, for the log).
    """
    if torch.isfinite(x).all():
        return x
    rows = int(x.shape[0]) if x.ndim >= 2 else 1
    bad_rows = int((~torch.isfinite(x)).reshape(rows, -1).any(dim=1).sum())
    dataset, ckpt = run_names(meta, checkpoint)
    from neuroatlas import quiet

    quiet.count(
        f"nonfinite embeddings:{dataset}/{ckpt}",
        (f"{dataset}: " if dataset else "") + (ckpt or "the checkpoint")
        + " gave non-finite embedding values for {hit} window{s}; those values are set to 0",
        hit=bad_rows,
    )
    logging.getLogger(__name__).debug(
        "%s: non-finite values (n_nan=%d, n_inf=%d, shape=%s) set to 0",
        where, int(torch.isnan(x).sum()), int(torch.isinf(x).sum()), tuple(x.shape),
    )
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def declared_units_from_batch(batch) -> "list[str] | None":
    """Extract the ``physical_units`` field from a ``BenchmarkBatch.meta``.

    Returns the flattened unique list of unit strings across the batch, or
    ``None`` if the batch does not carry this metadata (e.g. datasets that
    predate the field, or synthetic inputs). Strictly best-effort — never
    raises.
    """
    meta = None
    if hasattr(batch, "meta"):
        meta = batch.meta
    elif isinstance(batch, dict):
        meta = batch.get("meta")
    if not meta:
        return None
    seen: set[str] = set()
    for sample in meta:
        units = sample.get("physical_units") if isinstance(sample, dict) else None
        if not units:
            continue
        for u in units:
            if u:
                seen.add(str(u))
    return sorted(seen) if seen else None


def assert_amplitude_band(
    x: torch.Tensor,
    lo_uv: float,
    hi_uv: float,
    where: str,
    declared_units: "Iterable[str] | None" = None,  # noqa: UP037 — forward ref preserves 3.9 compat
    who: str = "",
) -> None:
    """Raise if the signal is clearly in the wrong unit; tolerate rare artifacts.

    Uses the 99th percentile (not peak) as the detector so a single-sample
    artifact spike — movement, electrode pop — does not kill a multi-hour
    shard.  The peak is still logged when it exceeds the band so silent
    contamination is visible in the logs.

    Catches unit bugs: e.g. volts passed where microvolts expected (p99 ≪ lo),
    or millivolts passed where microvolts expected (p99 ≫ hi).

    ``declared_units`` is the optional set/list of raw EDF physical-dimension
    strings for the channels in ``x`` (as carried by ``meta["physical_units"]``
    from the dataloader). Surfacing it in the error message lets the reader
    tell "artifact" (declared_units={'uV'}) from "upstream unit bug"
    (declared_units={'mV'}) without having to re-read the EDF.

    ``who`` (``stages/eegpt_pretrained``) names the run in the lines a
    tolerated batch gives -v and --log, once per command and run.
    """
    abs_x = x.detach().abs()
    # torch.quantile sorts the input and hard-caps at 2**24 (~16.78M) elements;
    # a raw (B, C, T) batch at w10s easily exceeds that (e.g. 2048×18×2560 ≈
    # 94M). Stride-subsample the flat view down to <= _Q_CAP — p99 is
    # statistically indistinguishable at this scale and the slice is a view,
    # so no extra allocation.
    _Q_CAP = 1 << 20  # 1,048,576
    flat = abs_x.to(torch.float32).reshape(-1)
    if flat.numel() > _Q_CAP:
        flat = flat[:: (flat.numel() // _Q_CAP + 1)]
    p99 = float(torch.quantile(flat, 0.99))
    peak = float(abs_x.max())
    # peak == 0 means the entire batch is flat — a genuine silent/dropout
    # window, not a unit mismatch (no V↔µV↔mV conversion produces exact zero
    # across every sample). It is embedded as is, and -v / --log say so once,
    # so a 5-hour silent stretch in one recording doesn't kill a multi-day shard.
    if peak == 0.0:
        _log_flat_batch(where, who, lo_uv, entirely=True)
        return
    # A real unit mismatch (V vs µV vs mV) scales ALL samples uniformly, so
    # if `peak` itself sits inside the valid band, the batch contains genuine
    # in-band signal and the low p99 just reflects mostly-silent windows
    # (data dropouts in long continuous EEG). Embedded as is (-v / --log).
    if lo_uv <= peak <= hi_uv and p99 < lo_uv:
        _log_flat_batch(where, who, lo_uv, entirely=False)
        return
    if p99 < lo_uv or p99 > hi_uv:
        units_suffix = ""
        if declared_units is not None:
            unique = sorted({str(u) for u in declared_units if str(u)})
            if unique:
                units_suffix = f" declared_units={unique}."
        raise ValueError(
            f"amplitude out of band at {where}: p99={p99:.4g} µV "
            f"(peak={peak:.4g} µV), expected p99 in [{lo_uv}, {hi_uv}] µV. "
            f"Likely a unit mismatch (V vs µV vs mV).{units_suffix}"
        )
    if peak > hi_uv:
        _log_spike(where, who, peak, hi_uv)


def _log_spike(where: str, who: str, peak: float, hi_uv: float) -> None:
    """A batch whose peak is above the band while 99% of it is inside: an
    artifact, embedded as is. Said once per command and run, at INFO (it asks
    nothing of the user)."""
    from neuroatlas import quiet

    quiet.warn_once(
        logging.getLogger(__name__), f"amplitude spike:{who or where}",
        "%s: a batch peaks at %.4g µV, above %.4g µV, while 99%% of its samples are in "
        "range (an artifact); it is embedded as is",
        who or where, peak, hi_uv, level=logging.INFO,
    )


def _log_flat_batch(where: str, who: str, lo_uv: float, entirely: bool) -> None:
    """A batch that is flat (every sample 0) or mostly flat (99% of its
    samples under the band's floor): data dropout, embedded as is. Said once
    per command and run, at INFO."""
    from neuroatlas import quiet

    what = ("is entirely flat (every sample 0 µV)" if entirely
            else f"is mostly flat (99% of its samples under {lo_uv:g} µV)")
    quiet.warn_once(
        logging.getLogger(__name__), f"amplitude flat:{entirely}:{who or where}",
        "%s: a batch %s; it is embedded as is",
        who or where, what, level=logging.INFO,
    )


# ---------------------------------------------------------------------------
# Unit / amplitude primitives (MODEL_CONTRACTS §0)
# ---------------------------------------------------------------------------

_UNIT_TO_UV_FACTOR: Dict[str, float] = {
    "uV": 1.0,
    "µV": 1.0,
    "μV": 1.0,
    "microvolt": 1.0,
    "microvolts": 1.0,
    "mV": 1e3,
    "millivolt": 1e3,
    "millivolts": 1e3,
    "V": 1e6,
    "volt": 1e6,
    "volts": 1e6,
}


def unit_to_uv(x: torch.Tensor, unit: Optional[str]) -> torch.Tensor:
    """Convert a tensor carrying ``unit`` amplitude into µV.

    ``unit`` is the tensor's declared unit at the point of the wrapper's
    ``_prepare_input`` — typically ``meta[i]["unit"]`` from the adapter.
    Supported values: ``"uV"`` / ``"µV"``, ``"mV"``, ``"V"`` (and
    alias spellings like ``"microvolts"``).

    Conversion factors:
        V  → µV: × 1e6
        mV → µV: × 1e3
        uV → µV: × 1 (identity)

    Raises ``ValueError`` if ``unit`` is ``None``, an empty string, or
    unknown. Adapters must publish ``meta[i]["unit"]``; a missing
    declaration is a contract bug.
    """
    if unit is None or unit == "":
        raise ValueError(
            "unit_to_uv: meta[i]['unit'] is missing. Adapters must "
            "publish the tensor unit per MODEL_CONTRACTS.md §0 "
            "(expected one of {'uV','mV','V'})."
        )
    key = str(unit).strip()
    factor = _UNIT_TO_UV_FACTOR.get(key)
    if factor is None:
        raise ValueError(
            f"unit_to_uv: unknown unit {unit!r}; expected one of "
            f"{sorted(_UNIT_TO_UV_FACTOR)}."
        )
    if factor == 1.0:
        return x
    return x * factor


# ---------------------------------------------------------------------------
# Zero-padded channel removal
# ---------------------------------------------------------------------------

_STRIP_ZERO_LOGGED: set = set()


def strip_zero_channels(
    x: torch.Tensor,
    channels: List[str],
) -> Tuple[torch.Tensor, List[str], List[int]]:
    """Remove channels that are all-zero across every sample in the batch.

    Returns ``(x_filtered, channels_filtered, kept_indices)`` — filtered
    tensor ``(B, C', T)``, matching labels, and the original indices of
    kept channels. Returns inputs unchanged when no all-zero channels
    are found. Always keeps at least one channel (degenerate fallback
    so downstream shape assumptions don't blow up).
    """
    if x.ndim != 3:
        raise ValueError(
            f"strip_zero_channels expects (B, C, T); got shape {tuple(x.shape)}"
        )
    C = x.shape[1]
    nonzero = x.abs().sum(dim=(0, 2)) > 0  # (C,)
    if nonzero.all():
        return x, list(channels), list(range(C))
    kept = nonzero.nonzero(as_tuple=True)[0]
    if len(kept) == 0:
        return x[:, :1, :], channels[:1], [0]
    kept_list = kept.tolist()
    n_stripped = C - len(kept_list)
    caller = ""
    import traceback as _tb
    for frame in _tb.extract_stack():
        if "backbones/" in frame.filename and frame.filename != __file__:
            caller = frame.filename.rsplit("/", 1)[-1].replace(".py", "")
            break
    log_key = (caller, C, n_stripped)
    if log_key not in _STRIP_ZERO_LOGGED:
        _STRIP_ZERO_LOGGED.add(log_key)
        stripped_names = [channels[i] for i in range(C) if i not in kept_list]
        logger.info(
            "[%s] stripped %d all-zero channel(s): %s",
            caller or "unknown", n_stripped, stripped_names,
        )
    return x[:, kept, :], [channels[i] for i in kept_list], kept_list


# ---------------------------------------------------------------------------
# Batch-meta validation (MODEL_CONTRACTS §3)
# ---------------------------------------------------------------------------


def read_sampling_rate(meta: Mapping[str, Any]) -> Optional[float]:
    """Return the canonical ``sampling_rate`` from a meta dict.

    Accepts ``"sampling_rate"`` (canonical per MODEL_CONTRACTS §3).
    Legacy ``"sfreq"`` is tolerated only when ``"sampling_rate"`` is
    also present and agrees. A legacy-only ``"sfreq"`` raises — adapters
    must publish the canonical key. Returns ``None`` when neither is
    present (caller falls back to inferring fs from tensor length).
    """
    canonical = meta.get("sampling_rate")
    legacy = meta.get("sfreq")
    if canonical is not None and legacy is not None:
        if abs(float(canonical) - float(legacy)) > 1e-6:
            raise ValueError(
                f"meta has conflicting sampling rates: "
                f"sampling_rate={canonical} vs sfreq={legacy}. "
                f"Adapters must publish the canonical 'sampling_rate' key only."
            )
    if canonical is not None:
        return float(canonical)
    if legacy is not None:
        raise ValueError(
            "meta[i]['sfreq'] is not accepted; adapters must publish "
            "the canonical 'sampling_rate' key per MODEL_CONTRACTS.md §3."
        )
    return None


def assert_batch_homogeneity(
    meta: Sequence[Mapping[str, Any]],
    keys: Tuple[str, ...] = ("sampling_rate", "unit", "channels"),
    *,
    where: str,
) -> None:
    """Raise ``ValueError`` if any sample's meta disagrees on ``keys``.

    Missing keys are skipped silently (backward compat) so legacy
    adapters that have not yet migrated still work — the contract demand
    only kicks in once a key is present.

    Lists/tuples are compared order-sensitively (``channels`` must match
    element-for-element per MODEL_CONTRACTS §3).
    """
    if not meta:
        return
    first = meta[0]
    for k in keys:
        if k not in first:
            continue
        ref = first[k]
        ref_cmp: Any = tuple(ref) if isinstance(ref, (list, tuple)) else ref
        for i in range(1, len(meta)):
            if k not in meta[i]:
                continue
            cur = meta[i][k]
            cur_cmp: Any = tuple(cur) if isinstance(cur, (list, tuple)) else cur
            if cur_cmp != ref_cmp:
                raise ValueError(
                    f"batch meta heterogeneity at {where}: key {k!r} "
                    f"differs — meta[0]={ref!r}, meta[{i}]={cur!r}. "
                    f"Adapters must group recordings with equal fs/unit/"
                    f"channels into homogeneous batches "
                    f"(MODEL_CONTRACTS.md §3)."
                )


def _filter_unallowed(
    keys: Sequence[str],
    allowed_literals: Set[str],
    allowed_prefixes: Tuple[str, ...],
    allowed_substrings: Tuple[str, ...],
) -> List[str]:
    out: List[str] = []
    for k in keys:
        if k in allowed_literals:
            continue
        if any(k.startswith(p) for p in allowed_prefixes):
            continue
        if any(s in k for s in allowed_substrings):
            continue
        out.append(k)
    return out


def strict_load_with_allowlist(
    model: Any,
    state: Mapping[str, Any],
    *,
    allowed_missing: Optional[Set[str]] = None,
    allowed_unexpected_literals: Optional[Set[str]] = None,
    allowed_unexpected_prefixes: Tuple[str, ...] = (),
    allowed_unexpected_substrings: Tuple[str, ...] = (),
    where: str,
) -> Dict[str, Any]:
    """Load ``state`` into ``model`` with an explicit allowlist."""
    missing_raw, unexpected_raw = model.load_state_dict(state, strict=False)
    missing_keys = list(missing_raw)
    unexpected_keys = list(unexpected_raw)

    real_missing = _filter_unallowed(
        missing_keys,
        allowed_literals=allowed_missing or set(),
        allowed_prefixes=(),
        allowed_substrings=(),
    )
    real_unexpected = _filter_unallowed(
        unexpected_keys,
        allowed_literals=allowed_unexpected_literals or set(),
        allowed_prefixes=allowed_unexpected_prefixes,
        allowed_substrings=allowed_unexpected_substrings,
    )
    if real_missing or real_unexpected:
        raise RuntimeError(
            f"weight load at {where}: unaccounted-for state_dict deltas.\n"
            f"  missing (not in allowlist): {real_missing}\n"
            f"  unexpected (not in allowlist): {real_unexpected}\n"
            "Update the wrapper's allowlist constants after verifying "
            "the checkpoint is compatible."
        )
    return {
        "missing_keys_allowed": missing_keys,
        "unexpected_keys_allowed": unexpected_keys,
        "n_state_keys_loaded": len(state) - len(unexpected_keys),
    }


# Used by the core_sleep wrapper.
def snap_to_epoch_length(
    x: torch.Tensor,
    target_sfreq: float,
    meta: list,
    max_snap_frac: float = 0.003,
) -> torch.Tensor:
    """Trim/pad resampled tensor to epoch-duration-based expected length.

    Polyphase resampling rounds non-integer source rates (e.g. 255.5 → 256),
    producing outputs a few samples off from ``epoch_seconds * target_sfreq``.
    When the mismatch is within *max_snap_frac* of the expected length,
    silently trim or zero-pad; otherwise return unchanged and let the caller's
    own window-contract check raise.
    """
    epoch_sec = meta[0].get("epoch_seconds") if meta else None
    if epoch_sec is None:
        return x
    expected_T = int(round(float(epoch_sec) * target_sfreq))
    actual_T = x.shape[-1]
    if actual_T == expected_T or expected_T <= 0:
        return x
    max_snap = max(2, int(np.ceil(expected_T * max_snap_frac)))
    if abs(actual_T - expected_T) <= max_snap:
        if actual_T > expected_T:
            x = x[..., :expected_T]
        else:
            x = F.pad(x, (0, expected_T - actual_T))
    return x



# Used by the biot wrapper.
def percentile_normalize(
    x: torch.Tensor, q: float = 0.95, eps: float = 1e-8
) -> torch.Tensor:
    """Per-channel percentile-based amplitude normalization.

    Matches BIOT's ``S[i] / (quantile(|S[i]|, q) + eps)`` paper formula
    (``MODEL_CONTRACTS.md §2``). Operates on shape ``(B, C, T)``; divides
    each channel by its ``q``-quantile of absolute values along time.

    This is scale-invariant by construction — the unit of the input is
    irrelevant to the output.
    """
    if x.ndim != 3:
        raise ValueError(
            f"percentile_normalize expects (B, C, T); got shape {tuple(x.shape)}"
        )
    abs_x = x.abs()
    # torch.quantile operates on a single axis; reshape (B, C, T) → (B*C, T),
    # compute per-row quantile, reshape back.
    flat = abs_x.reshape(-1, abs_x.shape[-1])
    q_tensor = torch.quantile(flat, q, dim=-1, keepdim=True)
    q_tensor = q_tensor.reshape(x.shape[0], x.shape[1], 1)
    return x / (q_tensor + eps)



# Used by the biot wrapper.
def clip_uv(x: torch.Tensor, clip_uv: float) -> Tuple[torch.Tensor, float]:
    """Clamp ``|x| ≤ clip_uv`` and return ``(x_clipped, fraction_clipped)``.

    Both return values are used by wrappers: the clamped tensor goes to
    the model, the clip fraction is logged to ``metadata()`` for
    observability (CBraMod applies this at 100 µV; NeuroRVQ at 500 µV).
    """
    if clip_uv <= 0:
        raise ValueError(f"clip_uv must be positive; got {clip_uv!r}")
    fraction = float((x.abs() > clip_uv).to(torch.float32).mean().item())
    return x.clamp(-clip_uv, clip_uv), fraction



# Used by the biot wrapper.
def zscore_per_recording(
    x: torch.Tensor,
    eps: float = 1e-6,
    sigma_clip: Optional[float] = None,
) -> torch.Tensor:
    """Per-recording z-score with optional symmetric σ-clip.

    Matches REVE's ``(x − µ) / max(σ, eps)`` then clamp to ``±sigma_clip``
    (``MODEL_CONTRACTS.md §2``). Statistics are computed over the full
    non-batch extent of each sample, i.e. per-sample over ``(C, T)``.

    ``sigma_clip=None`` disables the clamp (returns pure z-score).
    Set ``sigma_clip=15.0`` for REVE's documented ±15σ contract.
    """
    if x.ndim != 3:
        raise ValueError(
            f"zscore_per_recording expects (B, C, T); got shape {tuple(x.shape)}"
        )
    # Reduce over (channels, time) → per-sample mean/std.
    mu = x.mean(dim=(1, 2), keepdim=True)
    sigma = x.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(eps)
    z = (x - mu) / sigma
    if sigma_clip is not None:
        if sigma_clip <= 0:
            raise ValueError(f"sigma_clip must be positive or None; got {sigma_clip}")
        z = z.clamp(-sigma_clip, sigma_clip)
    return z

