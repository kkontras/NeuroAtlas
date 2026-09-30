"""Hypnogram reconstruction and sleep-architecture features.

The fifth verb, and the only paper result that is not a probe: Appendix C.2's
hypnogram features (Table 2) are computed *from* sleep-staging predictions, so
this runs after `probe --task sleep_staging` rather than instead of it. There
is no embedding pass and no model here.

Two steps, in order:

  reconstruct  Re-predict with each saved probe over its embedding cache and
               write per-recording epoch sequences -- predicted and
               ground-truth -- to `hypnograms.json` beside the probe results.

  features     Read those sequences and compute 34 sleep-architecture features
               per (dataset, model, fold, recording): total sleep time, stage
               proportions, transition counts, bout durations, REM latency,
               WASO. Writes a wide CSV plus a per-feature error summary.

Running it with no subcommand does both, over every dataset that has staging
results.

    python -m neuroatlas.entrypoints.hypnogram
    python -m neuroatlas.entrypoints.hypnogram --datasets dod mass
    python -m neuroatlas.entrypoints.hypnogram reconstruct \\
        --results-dir artifacts/benchmarks/dod/sleep_stage/sklearn_linear \\
        --group-by group --compute-metrics
    python -m neuroatlas.entrypoints.hypnogram features --datasets isruc --no-summary

Stage encoding is 0=W, 1=N1, 2=N2, 3=N3, 4=REM, at 30 s per epoch.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from neuroatlas.benchmarking_helpers.probes.metrics import compute_classification_metrics
from neuroatlas._paths import workspace_root

logger = logging.getLogger(__name__)

REPO = workspace_root()

EPOCH_SEC = 30
H_PER_EPOCH = EPOCH_SEC / 3600

W, N1, N2, N3, REM = 0, 1, 2, 3, 4


def _parse_csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


# ---------------------------------------------------------------------------
# Step 1 — reconstruct hypnograms from saved probes
# ---------------------------------------------------------------------------

def _load_probe(probe_path: str):
    with open(probe_path, "rb") as f:
        return pickle.load(f)


def _reconstruct_for_entry(
    entry: Dict[str, Any],
    requested_splits: Set[str],
    group_by: Optional[str],
    compute_metrics: bool,
) -> Optional[Dict[str, Any]]:
    cache_dir = Path(entry["cache_paths"]["global_cache_dir"])
    probe_path = Path(entry["cache_paths"]["probe"])
    fold = int(entry["metadata"]["fold"])
    model = entry["checkpoint_id"]
    fold_key = f"fold_{fold}"

    if not probe_path.exists():
        logger.warning("probe missing: %s", probe_path)
        return None
    if not (cache_dir / "features.npy").exists():
        logger.warning("cache missing: %s", cache_dir)
        return None

    probe = _load_probe(str(probe_path))
    features = np.load(cache_dir / "features.npy", mmap_mode="r")
    with open(cache_dir / "items.json", "r") as f:
        items = json.load(f)

    split_mask = [
        i for i, item in enumerate(items)
        if item.get("fold_assignments", {}).get(fold_key) in requested_splits
    ]
    valid_mask = [i for i in split_mask if int(items[i].get("sleep_stage", -1)) >= 0]

    if not valid_mask:
        logger.warning("no valid epochs for model=%s fold=%d", model, fold)
        return None

    idx_arr = np.array(valid_mask)
    feat_subset = np.ascontiguousarray(features[idx_arr])
    preds = probe.predict(feat_subset)

    rec_key = "recording_id"
    has_recording_id = any(rec_key in items[i] for i in valid_mask)
    if not has_recording_id:
        rec_key = "subject_id"

    by_recording: Dict[str, List[int]] = defaultdict(list)
    for local_idx, global_idx in enumerate(valid_mask):
        rid = str(items[global_idx].get(rec_key, items[global_idx].get("subject_id", "unknown")))
        by_recording[rid].append(local_idx)

    recordings = []
    for rid, local_indices in sorted(by_recording.items()):
        local_indices.sort(key=lambda li: int(items[valid_mask[li]].get("epoch_index", 0)))
        epochs = []
        for li in local_indices:
            gi = valid_mask[li]
            epochs.append({
                "epoch_index": int(items[gi].get("epoch_index", 0)),
                "predicted": int(preds[li]),
                "ground_truth": int(items[gi]["sleep_stage"]),
            })
        rec_entry: Dict[str, Any] = {rec_key: rid, "epochs": epochs}
        if rec_key == "recording_id":
            sid = items[valid_mask[local_indices[0]]].get("subject_id")
            if sid is not None:
                rec_entry["subject_id"] = str(sid)
        if group_by:
            gval = items[valid_mask[local_indices[0]]].get(group_by)
            rec_entry[group_by] = str(gval) if gval is not None else None
        recordings.append(rec_entry)

    result: Dict[str, Any] = {
        "model": model,
        "fold": fold,
        "splits": sorted(requested_splits),
        "n_epochs": len(valid_mask),
        "recordings": recordings,
    }

    if group_by and compute_metrics:
        by_group: Dict[str, Dict[str, list]] = defaultdict(lambda: {"preds": [], "labels": []})
        for li, gi in enumerate(valid_mask):
            gval = items[gi].get(group_by)
            gval_str = str(gval) if gval is not None else "<missing>"
            by_group[gval_str]["preds"].append(int(preds[li]))
            by_group[gval_str]["labels"].append(int(items[gi]["sleep_stage"]))

        metrics_by_group = {}
        for gval, data in sorted(by_group.items()):
            y_true = np.array(data["labels"])
            y_pred = np.array(data["preds"])
            metrics_by_group[gval] = compute_classification_metrics(y_true, y_pred)
            metrics_by_group[gval]["n_epochs"] = len(y_true)

        all_preds = np.array([int(preds[li]) for li in range(len(valid_mask))])
        all_labels = np.array([int(items[gi]["sleep_stage"]) for gi in valid_mask])
        metrics_by_group["__all__"] = compute_classification_metrics(all_labels, all_preds)
        metrics_by_group["__all__"]["n_epochs"] = len(valid_mask)

        result["group_by"] = group_by
        result["metrics_by_group"] = metrics_by_group

    return result


def reconstruct_results_dir(
    *,
    results_dir: Path,
    output: Path,
    models: Optional[Sequence[str]] = None,
    folds: Optional[Sequence[int]] = None,
    splits: Sequence[str] = ("test",),
    group_by: Optional[str] = None,
    compute_metrics: bool = False,
    merge: bool = True,
) -> int:
    """Re-predict every probe under *results_dir* and write hypnograms.

    Returns the number of entries written. *merge* keeps entries already in
    *output* whose (model, fold) this run did not reproduce, so a per-model
    rerun does not discard the rest.
    """
    with open(results_dir / "results.json") as f:
        all_entries = json.load(f)

    model_filter = set(models) if models else None
    fold_filter = set(int(x) for x in folds) if folds else None
    requested_splits = set(splits)

    if compute_metrics and not group_by:
        logger.warning("--compute-metrics without --group-by: overall metrics only")

    entries = [
        e for e in all_entries
        if e.get("ok", False)
        and (model_filter is None or e["checkpoint_id"] in model_filter)
        and (fold_filter is None or int(e["metadata"]["fold"]) in fold_filter)
    ]
    if not entries:
        logger.warning("no matching entries in %s", results_dir / "results.json")
        return 0

    logger.info("%s: processing %d entries", results_dir, len(entries))
    outputs: List[Dict[str, Any]] = []
    n_skipped = 0
    for entry in entries:
        try:
            result = _reconstruct_for_entry(
                entry, requested_splits, group_by, compute_metrics)
        except Exception:
            logger.warning("skipping model=%s fold=%s: %s",
                           entry.get("checkpoint_id", "?"),
                           entry.get("metadata", {}).get("fold", "?"),
                           sys.exc_info()[1])
            n_skipped += 1
            continue
        if result is not None:
            outputs.append(result)
            logger.info("model=%s fold=%d: %d recordings, %d epochs",
                        result["model"], result["fold"],
                        len(result["recordings"]), result["n_epochs"])
    if n_skipped:
        logger.warning("skipped %d entries due to errors", n_skipped)

    output.parent.mkdir(parents=True, exist_ok=True)
    if merge and output.exists():
        with open(output) as f:
            existing = json.load(f)
        fresh = {(e["model"], e["fold"]) for e in outputs}
        outputs = [e for e in existing if (e["model"], e["fold"]) not in fresh] + outputs
        logger.info("merged with existing file: %d total entries", len(outputs))

    with open(output, "w") as f:
        json.dump(outputs, f, indent=2)
    logger.info("wrote %d hypnogram entries to %s", len(outputs), output)
    return len(outputs)


# ---------------------------------------------------------------------------
# Step 2 — sleep-architecture features
# ---------------------------------------------------------------------------

#: checkpoint ids to skip. Empty: the two it used to hold (bendr, tfc)
#: are no longer in the registry, so nothing needs excluding by name.
SK_EXCLUDE: Set[str] = set()

DATASET_HYPNO_PATHS: Dict[str, str] = {
    "mass": "mass/sklearn_linear/hypnograms.json",
    "dcsm": "dcsm/sklearn_linear/hypnograms.json",
    "ucddb": "ucddb/sklearn_linear/hypnograms.json",
    "dod": "dod/sleep_stage/sklearn_linear/hypnograms.json",
    "isruc": "isruc/sleep_stage/sklearn_linear/hypnograms.json",
    "sleep_edf_expanded": "sleep_edf_expanded/sleep_stage/sklearn_linear/hypnograms.json",
    "physionet2026": "physionet2026/sklearn_linear/hypnograms.json",
    "wsc": "wsc/sklearn_linear/hypnograms.json",
}

FEATURE_NAMES = [
    "TST", "SleepEff",
    "TimeW", "TimeN1", "TimeN2", "TimeN3", "TimeR",
    "RelN1", "RelN2", "RelN3", "RelR",
    "Awakenings", "SlStCh", "ChToR", "ChToN3",
    "TimeBetweenW", "TimeBetweenR", "TimeBetweenN3",
    "WBoutDur", "RBoutDur", "N3BoutDur",
    "SleepOnsetN1", "SleepOnsetN2", "WASO", "RemLatency",
    "RelAwakenings", "RelSlStCh", "RelChToR", "RelChToN3", "RelWakeTime",
    "RelOccN1", "RelOccN2", "RelOccN3", "RelOccR",
]

META_COLS = ["dataset", "model", "fold", "recording_id", "n_epochs"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_bouts(stages: np.ndarray, target: int) -> List[Tuple[int, int]]:
    """Return list of (start_index, length) for contiguous runs of *target*."""
    mask = (stages == target).astype(np.int8)
    if mask.sum() == 0:
        return []
    diff = np.diff(mask, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), (ends - starts).tolist()))


def _sleep_period(stages: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (onset_idx, final_idx) of the sleep period, or None if all Wake."""
    non_wake = np.where(stages != W)[0]
    if len(non_wake) == 0:
        return None
    return int(non_wake[0]), int(non_wake[-1])


# ---------------------------------------------------------------------------
# Core feature computation
# ---------------------------------------------------------------------------

def compute_hypnogram_features(stages: np.ndarray, epoch_sec: int = EPOCH_SEC) -> Dict[str, float]:
    n = len(stages)
    nan = float("nan")
    h = epoch_sec / 3600

    count_w = int(np.sum(stages == W))
    count_n1 = int(np.sum(stages == N1))
    count_n2 = int(np.sum(stages == N2))
    count_n3 = int(np.sum(stages == N3))
    count_r = int(np.sum(stages == REM))

    time_in_bed = n * h
    tst = (n - count_w) * h
    sleep_eff = tst / time_in_bed if time_in_bed > 0 else nan

    time_w = count_w * h
    time_n1 = count_n1 * h
    time_n2 = count_n2 * h
    time_n3 = count_n3 * h
    time_r = count_r * h

    rel_n1 = time_n1 / tst if tst > 0 else nan
    rel_n2 = time_n2 / tst if tst > 0 else nan
    rel_n3 = time_n3 / tst if tst > 0 else nan
    rel_r = time_r / tst if tst > 0 else nan

    # -- Transitions --
    if n > 1:
        prev, cur = stages[:-1], stages[1:]
        sl_st_ch = int(np.sum(prev != cur))
        awakenings = int(np.sum((cur == W) & (prev != W)))
        ch_to_r = int(np.sum((cur == REM) & (prev != REM)))
        ch_to_n3 = int(np.sum((cur == N3) & (prev != N3)))
    else:
        sl_st_ch = awakenings = ch_to_r = ch_to_n3 = 0

    # -- Bout analysis --
    def _bout_stats(target: int):
        bouts = _get_bouts(stages, target)
        if not bouts:
            return nan, nan
        durations = [length * h for _, length in bouts]
        avg_dur = sum(durations) / len(durations)
        if len(bouts) < 2:
            avg_gap = nan
        else:
            gaps = []
            for i in range(1, len(bouts)):
                gap = (bouts[i][0] - (bouts[i - 1][0] + bouts[i - 1][1])) * h
                gaps.append(gap)
            avg_gap = sum(gaps) / len(gaps)
        return avg_gap, avg_dur

    time_between_w, w_bout_dur = _bout_stats(W)
    time_between_r, r_bout_dur = _bout_stats(REM)
    time_between_n3, n3_bout_dur = _bout_stats(N3)

    # -- Latencies --
    n1_indices = np.where(stages == N1)[0]
    n2_indices = np.where(stages == N2)[0]
    rem_indices = np.where(stages == REM)[0]

    sleep_onset_n1 = int(n1_indices[0]) * h if len(n1_indices) > 0 else nan
    sleep_onset_n2 = int(n2_indices[0]) * h if len(n2_indices) > 0 else nan

    sp = _sleep_period(stages)
    if sp is not None:
        onset_idx, final_idx = sp
        waso_epochs = np.sum(stages[onset_idx:final_idx + 1] == W)
        waso = int(waso_epochs) * h
        if len(rem_indices) > 0:
            first_rem = int(rem_indices[0])
            rem_latency = (first_rem - onset_idx) * h if first_rem >= onset_idx else nan
        else:
            rem_latency = nan
    else:
        waso = 0.0
        rem_latency = nan

    # -- Relative counts --
    rel_awakenings = awakenings / tst if tst > 0 else nan
    rel_sl_st_ch = sl_st_ch / tst if tst > 0 else nan
    rel_ch_to_r = ch_to_r / tst if tst > 0 else nan
    rel_ch_to_n3 = ch_to_n3 / tst if tst > 0 else nan
    rel_wake_time = waso / tst if tst > 0 else nan

    # -- Relative occurrence (normalized position within sleep period) --
    def _rel_occ(target: int) -> float:
        if sp is None:
            return nan
        onset_idx, final_idx = sp
        span = final_idx - onset_idx
        if span == 0:
            return nan
        idx = np.where(stages[onset_idx:final_idx + 1] == target)[0]
        if len(idx) == 0:
            return nan
        return float(np.mean(idx) / span)

    rel_occ_n1 = _rel_occ(N1)
    rel_occ_n2 = _rel_occ(N2)
    rel_occ_n3 = _rel_occ(N3)
    rel_occ_r = _rel_occ(REM)

    return {
        "TST": tst, "SleepEff": sleep_eff,
        "TimeW": time_w, "TimeN1": time_n1, "TimeN2": time_n2,
        "TimeN3": time_n3, "TimeR": time_r,
        "RelN1": rel_n1, "RelN2": rel_n2, "RelN3": rel_n3, "RelR": rel_r,
        "Awakenings": awakenings, "SlStCh": sl_st_ch,
        "ChToR": ch_to_r, "ChToN3": ch_to_n3,
        "TimeBetweenW": time_between_w, "TimeBetweenR": time_between_r,
        "TimeBetweenN3": time_between_n3,
        "WBoutDur": w_bout_dur, "RBoutDur": r_bout_dur,
        "N3BoutDur": n3_bout_dur,
        "SleepOnsetN1": sleep_onset_n1, "SleepOnsetN2": sleep_onset_n2,
        "WASO": waso, "RemLatency": rem_latency,
        "RelAwakenings": rel_awakenings, "RelSlStCh": rel_sl_st_ch,
        "RelChToR": rel_ch_to_r, "RelChToN3": rel_ch_to_n3,
        "RelWakeTime": rel_wake_time,
        "RelOccN1": rel_occ_n1, "RelOccN2": rel_occ_n2,
        "RelOccN3": rel_occ_n3, "RelOccR": rel_occ_r,
    }


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_dataset(dataset: str) -> List[Dict[str, Any]]:
    rel = DATASET_HYPNO_PATHS.get(dataset)
    if rel is None:
        logger.warning("unknown dataset: %s", dataset)
        return []
    path = REPO / "artifacts" / "benchmarks" / rel
    if not path.exists():
        logger.warning("skipping %s: %s not found", dataset, path)
        return []

    logger.info("loading %s ...", path)
    with open(path) as f:
        data = json.load(f)

    rows: List[Dict[str, Any]] = []
    for entry in data:
        model = entry["model"]
        if model in SK_EXCLUDE:
            continue
        fold = entry["fold"]

        for rec in entry["recordings"]:
            rid = rec.get("recording_id") or rec.get("subject_id", "unknown")
            epochs_sorted = sorted(rec["epochs"], key=lambda e: e["epoch_index"])
            gt = np.array([e["ground_truth"] for e in epochs_sorted], dtype=np.int8)
            pred = np.array([e["predicted"] for e in epochs_sorted], dtype=np.int8)

            gt_feats = compute_hypnogram_features(gt)
            pred_feats = compute_hypnogram_features(pred)

            row: Dict[str, Any] = {
                "dataset": dataset,
                "model": model,
                "fold": fold,
                "recording_id": str(rid),
                "n_epochs": len(epochs_sorted),
            }
            for fname in FEATURE_NAMES:
                row[f"gt_{fname}"] = gt_feats[fname]
                row[f"pred_{fname}"] = pred_feats[fname]
            rows.append(row)

    del data
    logger.info("%s: %d rows", dataset, len(rows))
    return rows


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict

    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["model"])].append(r)

    summary: List[Dict[str, Any]] = []
    for (ds, model), recs in sorted(groups.items()):
        for fname in FEATURE_NAMES:
            gt_vals = []
            pred_vals = []
            for r in recs:
                g = r[f"gt_{fname}"]
                p = r[f"pred_{fname}"]
                if not (math.isnan(g) if isinstance(g, float) else False) and \
                   not (math.isnan(p) if isinstance(p, float) else False):
                    gt_vals.append(g)
                    pred_vals.append(p)
            n_valid = len(gt_vals)
            if n_valid < 2:
                summary.append({
                    "dataset": ds, "model": model, "feature": fname,
                    "n_valid": n_valid,
                    "mae": float("nan"), "bias": float("nan"),
                    "rmse": float("nan"), "pearson_r": float("nan"),
                })
                continue
            gt_arr = np.array(gt_vals)
            pred_arr = np.array(pred_vals)
            diff = pred_arr - gt_arr
            mae = float(np.mean(np.abs(diff)))
            bias = float(np.mean(diff))
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            if np.std(gt_arr) > 0 and np.std(pred_arr) > 0:
                r_val = float(np.corrcoef(gt_arr, pred_arr)[0, 1])
            else:
                r_val = float("nan")
            summary.append({
                "dataset": ds, "model": model, "feature": fname,
                "n_valid": n_valid, "mae": mae, "bias": bias,
                "rmse": rmse, "pearson_r": r_val,
            })
    return summary


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _fmt(val: Any) -> str:
    if isinstance(val, float) and math.isnan(val):
        return ""
    if isinstance(val, float):
        return f"{val:.6g}"
    return str(val)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    cols = META_COLS[:]
    for fname in FEATURE_NAMES:
        cols.append(f"gt_{fname}")
        cols.append(f"pred_{fname}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in rows:
            w.writerow([_fmt(row[c]) for c in cols])
    logger.info("wrote %d rows to %s", len(rows), path)


def write_summary_csv(summary: List[Dict[str, Any]], path: Path) -> None:
    if not summary:
        return
    cols = ["dataset", "model", "feature", "n_valid", "mae", "bias", "rmse", "pearson_r"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in summary:
            w.writerow([_fmt(row[c]) for c in cols])
    logger.info("wrote %d rows to %s", len(summary), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_reconstruct_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results-dir", default=None,
                   help="Probe-results directory holding results.json and "
                        "probes/. Omit to derive it from --datasets.")
    p.add_argument("--models", default=None, type=_parse_csv,
                   help="Comma-separated checkpoint ids (default: all).")
    p.add_argument("--folds", default=None, type=_parse_int_list,
                   help="Comma-separated fold indices (default: all).")
    p.add_argument("--splits", default="test", type=_parse_csv,
                   help="Comma-separated splits to predict on (default: test).")
    p.add_argument("--group-by", default=None,
                   help="Metadata field to group recordings by "
                        "(subgroup, subset, group, site_id).")
    p.add_argument("--compute-metrics", action="store_true",
                   help="Also compute classification metrics per group.")
    p.add_argument("--output", default=None,
                   help="Output JSON (default: <results-dir>/hypnograms.json).")


def _add_features_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output-dir", default="artifacts/benchmarks",
                   help="Where the CSVs are written (default: %(default)s).")
    p.add_argument("--no-summary", action="store_true",
                   help="Skip the per-feature error summary.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neuroatlas.entrypoints.hypnogram",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="*", default=None, metavar="SLUG",
                        help="Dataset slugs (default: every one with staging "
                             f"results: {', '.join(sorted(DATASET_HYPNO_PATHS))}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run and exit.")
    sub = parser.add_subparsers(dest="step")
    rec = sub.add_parser("reconstruct", help="probe results -> hypnograms.json")
    _add_reconstruct_args(rec)
    rec.add_argument("--datasets", nargs="*", default=None, metavar="SLUG")
    rec.add_argument("--dry-run", action="store_true")
    fea = sub.add_parser("features", help="hypnograms.json -> feature CSVs")
    _add_features_args(fea)
    fea.add_argument("--datasets", nargs="*", default=None, metavar="SLUG")
    fea.add_argument("--dry-run", action="store_true")
    # so `hypnogram` with no subcommand still accepts either step's flags
    _add_reconstruct_args(parser)
    _add_features_args(parser)
    return parser


def _results_dir_for(dataset: str) -> Path:
    """Where this dataset's probe results live, from the known layout."""
    rel = DATASET_HYPNO_PATHS[dataset]
    return REPO / "artifacts" / "benchmarks" / Path(rel).parent


def run_reconstruct(args, datasets: Sequence[str]) -> int:
    dirs: List[Path] = ([Path(args.results_dir)] if args.results_dir
                        else [_results_dir_for(d) for d in datasets])
    failed = 0
    for d in dirs:
        out = Path(args.output) if args.output else d / "hypnograms.json"
        if args.dry_run:
            print(f"reconstruct {d} -> {out}")
            continue
        if not (d / "results.json").is_file():
            logger.warning("no results.json in %s — run `probe --task "
                           "sleep_staging` for it first", d)
            failed += 1
            continue
        reconstruct_results_dir(
            results_dir=d, output=out, models=args.models, folds=args.folds,
            splits=args.splits, group_by=args.group_by,
            compute_metrics=args.compute_metrics,
        )
    return failed


def run_features(args, datasets: Sequence[str]) -> int:
    out_dir = Path(args.output_dir)
    if args.dry_run:
        for d in datasets:
            print(f"features {_results_dir_for(d)}/hypnograms.json")
        print(f"-> {out_dir}/hypnogram_features.csv")
        return 0
    rows: List[Dict[str, Any]] = []
    for ds in datasets:
        rows.extend(process_dataset(ds))
    if not rows:
        logger.error("no hypnograms found — run the reconstruct step first")
        return 1
    write_csv(rows, out_dir / "hypnogram_features.csv")
    if not args.no_summary:
        write_summary_csv(compute_summary(rows),
                          out_dir / "hypnogram_features_summary.csv")
    logger.info("done — %d rows across %d datasets",
                len(rows), len({r["dataset"] for r in rows}))
    return 0


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    datasets = args.datasets or sorted(DATASET_HYPNO_PATHS)
    unknown = [d for d in datasets if d not in DATASET_HYPNO_PATHS]
    if unknown:
        raise SystemExit(
            f"error: no staging-results layout known for {', '.join(unknown)}.\n"
            f"Known: {', '.join(sorted(DATASET_HYPNO_PATHS))}"
        )

    step = getattr(args, "step", None)
    rc = 0
    if step in (None, "reconstruct"):
        rc |= run_reconstruct(args, datasets)
    if step in (None, "features"):
        rc |= run_features(args, datasets)
    if rc:
        sys.exit(1)


if __name__ == "__main__":
    main()
