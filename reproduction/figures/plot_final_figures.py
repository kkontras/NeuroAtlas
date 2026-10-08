"""Generate unified final benchmark figures.

Reads probe results via the data loaders in ``plot_benchmark_results`` and
writes PNG figures under ``plots/final/``.

Run:
    PYTHONPATH=src python reproduction/figures/plot_final_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_benchmark_results as pbr

REPO = pbr.REPO
OUT = REPO / "plots" / "final"
OUT.mkdir(exist_ok=True, parents=True)


def _save(fig, name: str) -> None:
    out = OUT / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Core reusable line-panel renderer
# ---------------------------------------------------------------------------

def _line_panel(
    ax,
    line_data: Dict[str, Dict[str, Tuple[float, float]]],
    line_colors: Dict[str, str],
    line_labels: Dict[str, str],
    ylabel: str,
    refs: Sequence[Tuple[float, str, dict]] = (),
    eeg_only: bool = False,
    line_widths: Dict[str, float] | None = None,
    line_styles: Dict[str, str] | None = None,
    lower_is_better: bool = False,
) -> None:
    """Render one line-plot panel (models on x, one line per variant).

    line_data:   line_key -> model_key -> (mean, std)
    line_colors: line_key -> colour
    line_labels: line_key -> display name for legend
    """
    if line_widths is None:
        line_widths = {}
    if line_styles is None:
        line_styles = {}

    all_model_set: set[str] = set()
    for ld in line_data.values():
        all_model_set |= ld.keys()

    model_all_vals: Dict[str, List[float]] = defaultdict(list)
    for ld in line_data.values():
        for mk, (m, _) in ld.items():
            model_all_vals[mk].append(m)

    eeg_models = [mk for mk in pbr.SK_MODEL_ORDER if mk in all_model_set]
    _sign = 1 if lower_is_better else -1
    eeg_models.sort(key=lambda mk: _sign * np.median(model_all_vals[mk]))

    if eeg_only:
        all_models = eeg_models
        all_pos: list[float] = list(range(len(eeg_models)))
        pos_lookup = dict(zip(all_models, all_pos))
        groups_for_draw = [eeg_models]
        sections: list[tuple[list[float], str]] = []
    else:
        tsfm_fams: List[Tuple[str, List[str], float]] = []
        for fam_name, fam_members in pbr.TSFM_FAMILIES:
            active = [mk for mk in fam_members if mk in all_model_set]
            if active:
                active.sort(key=lambda mk: _sign * np.median(model_all_vals[mk]))
                tsfm_fams.append((fam_name, active,
                                  np.median(model_all_vals[active[0]])))
        tsfm_fams.sort(key=lambda t: _sign * t[2])

        eeg_pos: list[float] = list(range(len(eeg_models)))
        eeg_span = (eeg_pos[-1] - eeg_pos[0]) if len(eeg_pos) > 1 else 1.0

        tsfm_pos: list[float] = []
        tsfm_order: list[str] = []
        n_fam = len(tsfm_fams)
        if sum(len(f[1]) for f in tsfm_fams) > 0:
            within = 0.45
            total_within = sum(len(f[1]) - 1 for f in tsfm_fams) * within
            between = ((eeg_span - total_within) / (n_fam - 1)
                       if n_fam > 1 else 0)
            between = max(between, within + 0.3)
            gap = 2.0
            pos = (eeg_pos[-1] + gap) if eeg_pos else gap
            for fi, (_, members, _) in enumerate(tsfm_fams):
                for mi, mk in enumerate(members):
                    tsfm_pos.append(pos)
                    tsfm_order.append(mk)
                    if mi < len(members) - 1:
                        pos += within
                if fi < n_fam - 1:
                    pos += between

        sup_fams: List[Tuple[str, List[str], float]] = []
        for fam_name, fam_members in pbr.SUP_FAMILIES:
            active = [mk for mk in fam_members if mk in all_model_set]
            if active:
                active.sort(
                    key=lambda mk: _sign * np.median(model_all_vals[mk]))
                sup_fams.append((fam_name, active,
                                 np.median(model_all_vals[active[0]])))
        sup_fams.sort(key=lambda t: _sign * t[2])

        sup_pos: list[float] = []
        sup_order: list[str] = []
        if sum(len(f[1]) for f in sup_fams) > 0:
            within_s = 0.45
            total_within_s = sum(len(f[1]) - 1 for f in sup_fams) * within_s
            n_sfam = len(sup_fams)
            between_s = ((eeg_span - total_within_s) / (n_sfam - 1)
                         if n_sfam > 1 else 0)
            between_s = max(between_s, within_s + 0.3)
            sup_gap = 2.0
            last = (tsfm_pos[-1] if tsfm_pos
                    else (eeg_pos[-1] if eeg_pos else -1))
            pos = last + sup_gap
            for fi, (_, members, _) in enumerate(sup_fams):
                for mi, mk in enumerate(members):
                    sup_pos.append(pos)
                    sup_order.append(mk)
                    if mi < len(members) - 1:
                        pos += within_s
                if fi < n_sfam - 1:
                    pos += between_s

        all_models = eeg_models + tsfm_order + sup_order
        all_pos = eeg_pos + tsfm_pos + sup_pos
        pos_lookup = dict(zip(all_models, all_pos))
        groups_for_draw = [eeg_models, tsfm_order, sup_order]
        sections = [(eeg_pos, "EEG FMs"), (tsfm_pos, "TS FMs"),
                    (sup_pos, "Supervised")]

    for lk in line_data:
        color = line_colors.get(lk, "#777")
        lw = line_widths.get(lk, 1.4)
        ls = line_styles.get(lk, "-")
        label = line_labels.get(lk, lk)
        ld = line_data[lk]

        first_group = True
        for group in groups_for_draw:
            xs, ys, yerr = [], [], []
            for mk in group:
                if mk in ld:
                    m, s = ld[mk]
                    xs.append(pos_lookup[mk])
                    ys.append(m)
                    yerr.append(s)
            if not xs:
                continue
            lbl = label if first_group else None
            first_group = False
            ax.errorbar(xs, ys, yerr=yerr, marker="o", ms=5, lw=lw,
                        color=color, alpha=0.8, zorder=3, label=lbl,
                        capsize=2, capthick=0.8, elinewidth=0.8,
                        linestyle=ls)
            ax.scatter(xs, ys, s=20, color=color, edgecolors="black",
                       linewidths=0.3, zorder=4)

    if not eeg_only:
        prev_end = None
        for sec_pos, sec_label in sections:
            if not sec_pos:
                continue
            if prev_end is not None:
                sep_x = (prev_end + sec_pos[0]) / 2
                ax.axvline(sep_x, color="gray", ls="--", lw=1, alpha=0.5)
            mid_x = (sec_pos[0] + sec_pos[-1]) / 2
            ax.text(mid_x, 0.97, sec_label,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, fontweight="bold")
            prev_end = sec_pos[-1]

    for y_val, ref_label, style in refs:
        pbr._hline(ax, y_val, ref_label, style)

    ax.set_xticks(all_pos)
    ax.set_xticklabels([pbr.MODEL_LABELS.get(mk, mk) for mk in all_models],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=4,
                  loc="lower left")


# ---------------------------------------------------------------------------
# Helper: build line_data from sk_sleep_metric
# ---------------------------------------------------------------------------

def _sleep_line_data(
    path_map: Dict[str, str],
    datasets: List[str],
    metric_key: str,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    populated = pbr._sk_populated_datasets(path_map, datasets)
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for ds in populated:
        ds_data = pbr.sk_sleep_metric(path_map, ds, metric_key)
        ds_line: Dict[str, Tuple[float, float]] = {}
        for mk in pbr.SK_ALL_MODEL_ORDER:
            vals = ds_data.get(mk, [])
            if vals:
                ds_line[mk] = (float(np.mean(vals)), float(np.std(vals)))
        if ds_line:
            out[ds] = ds_line
    return out


# ---------------------------------------------------------------------------
# Attention probe data loaders
# ---------------------------------------------------------------------------

ATTN_SLEEP_DIRS = {
    "mass": "mass/attention_probe",
    "dcsm": "dcsm/attention_probe",
    "wsc": "wsc/attention_probe",
    "ucddb": "ucddb/attention_probe",
    "dod": "dod/attention_probe",
    "isruc": "isruc/attention_probe",
    "sleep_edf_expanded": "sleep_edf_expanded/attention_probe",
    "physionet2026": "physionet2026/attention_probe",
}
ATTN_SLEEP_DATASETS = list(ATTN_SLEEP_DIRS.keys())


def _load_attention_probe_metric(
    dataset: str, metric_key: str,
) -> Dict[str, List[float]]:
    """Load a sleep metric from individual attention-probe JSON files.

    Returns {model: [per-fold values]}, same shape as pbr.sk_sleep_metric().
    """
    attn_dir = REPO / "artifacts/benchmarks" / ATTN_SLEEP_DIRS[dataset]
    if not attn_dir.is_dir():
        return {}
    out: Dict[str, List[float]] = {}
    for fp in sorted(attn_dir.glob("*.json")):
        with open(fp) as f:
            data = json.load(f)
        mk = data.get("model", fp.stem)
        if mk in pbr.SK_EXCLUDE:
            continue
        vals: List[float] = []
        for fold in data.get("folds", []):
            bt = (fold.get("metrics") or {}).get("best_test") or {}
            v = bt.get(metric_key)
            if v is not None:
                vals.append(float(v))
            elif metric_key == "balanced_accuracy":
                cm = bt.get("confusion_matrix")
                if cm is not None:
                    ba = pbr._balanced_accuracy_from_cm(cm)
                    if ba is not None:
                        vals.append(ba)
        if vals:
            out[mk] = vals
    return out


def _load_attention_probe_f1_per_stage(
    dataset: str,
) -> Dict[str, np.ndarray]:
    """Load per-stage F1 from attention-probe JSON files.

    Returns {model: ndarray(n_folds, 5)}, same shape as pbr.sk_sleep_f1_per_stage().
    """
    attn_dir = REPO / "artifacts/benchmarks" / ATTN_SLEEP_DIRS[dataset]
    if not attn_dir.is_dir():
        return {}
    out: Dict[str, List[List[float]]] = {}
    for fp in sorted(attn_dir.glob("*.json")):
        with open(fp) as f:
            data = json.load(f)
        mk = data.get("model", fp.stem)
        if mk in pbr.SK_EXCLUDE:
            continue
        rows: List[List[float]] = []
        for fold in data.get("folds", []):
            bt = (fold.get("metrics") or {}).get("best_test") or {}
            f1 = bt.get("f1_per_class")
            if isinstance(f1, list) and len(f1) == 5:
                rows.append(f1)
        if rows:
            out[mk] = rows
    return {k: np.asarray(v) for k, v in out.items()}


def _attn_sleep_line_data(
    datasets: List[str],
    metric_key: str,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Build line_data from attention probe results (parallel to _sleep_line_data)."""
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for ds in datasets:
        ds_data = _load_attention_probe_metric(ds, metric_key)
        ds_line: Dict[str, Tuple[float, float]] = {}
        for mk in pbr.SK_ALL_MODEL_ORDER:
            vals = ds_data.get(mk, [])
            if vals:
                ds_line[mk] = (float(np.mean(vals)), float(np.std(vals)))
        if ds_line:
            out[ds] = ds_line
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Sleep Staging Overview
# ═══════════════════════════════════════════════════════════════════════════

def plot_sleep_overview() -> None:
    metrics = [
        ("cohen_kappa", "Cohen's κ"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "macro-F1"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), constrained_layout=True)
    ds_colors = pbr.SK_DATASET_COLORS
    ds_labels = {ds: ds.upper() for ds in pbr.SK_SLEEP_DATASETS}

    for ax, (mkey, mlabel) in zip(axes, metrics):
        line_data = _sleep_line_data(
            pbr.SK_SLEEP_RESULTS_PATH, pbr.SK_SLEEP_DATASETS, mkey)
        refs: list[tuple[float, str, dict]] = []
        if mkey == "cohen_kappa":
            refs.append((0.76, "human: 0.76", pbr._HUMAN_STYLE))
        _line_panel(ax, line_data, ds_colors, ds_labels, mlabel, refs=refs)
        ax.set_title(mlabel, fontsize=13)

    fig.suptitle("Sleep staging (sklearn_linear, test)", fontsize=15)
    _save(fig, "final_sleep_overview.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1a: Sleep Staging Overview (attention probe)
# ═══════════════════════════════════════════════════════════════════════════

def plot_sleep_overview_attention() -> None:
    metrics = [
        ("cohen_kappa", "Cohen's κ"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "macro-F1"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), constrained_layout=True)
    ds_colors = pbr.SK_DATASET_COLORS
    ds_labels = {ds: ds.upper() for ds in ATTN_SLEEP_DATASETS}

    for ax, (mkey, mlabel) in zip(axes, metrics):
        line_data = _attn_sleep_line_data(ATTN_SLEEP_DATASETS, mkey)
        refs: list[tuple[float, str, dict]] = []
        if mkey == "cohen_kappa":
            refs.append((0.76, "human: 0.76", pbr._HUMAN_STYLE))
        _line_panel(ax, line_data, ds_colors, ds_labels, mlabel, refs=refs)
        ax.set_title(mlabel, fontsize=13)

    fig.suptitle("Sleep staging (attention_probe, test)", fontsize=15)
    _save(fig, "final_sleep_overview_attention.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1b: Sleep Staging — Best-of-Category Aggregation
# ═══════════════════════════════════════════════════════════════════════════

_GROUP_COLORS = {
    "Best EEG FM": "#1F77B4",
    "Best TS FM": "#D62728",
    "Best Supervised-pretraining": "#2CA02C",
    "Best Supervised-pretraining (seq1)": "#1E7A1E",
}
_GROUP_MODELS = {
    "Best EEG FM": pbr.SK_MODEL_ORDER,
    "Best TS FM": pbr.TSFM_MODEL_ORDER,
    "Best Supervised-pretraining": pbr.SUP_MODEL_ORDER,
    "Best Supervised-pretraining (seq1)": pbr.SUP_SEQ1_MODEL_ORDER,
}


def plot_sleep_best_per_category() -> None:
    metrics = [
        ("cohen_kappa", "Cohen's κ"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "macro-F1"),
    ]
    datasets = pbr._sk_populated_datasets(
        pbr.SK_SLEEP_RESULTS_PATH, pbr.SK_SLEEP_DATASETS)
    if not datasets:
        print("skip final_sleep_best_per_category.png: no data")
        return

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)),
                             constrained_layout=True)

    for ax, (mkey, mlabel) in zip(axes, metrics):
        all_data = {ds: pbr.sk_sleep_metric(
            pbr.SK_SLEEP_RESULTS_PATH, ds, mkey) for ds in datasets}

        # Rank datasets by average best-of-category for this metric
        ds_avg: Dict[str, float] = {}
        for ds in datasets:
            bests: list[float] = []
            for group_models in _GROUP_MODELS.values():
                best = max(
                    (float(np.mean(all_data[ds].get(mk, [])))
                     for mk in group_models if all_data[ds].get(mk)),
                    default=np.nan)
                if not np.isnan(best):
                    bests.append(best)
            ds_avg[ds] = float(np.mean(bests)) if bests else 0.0
        ranked_ds = sorted(datasets, key=lambda ds: -ds_avg[ds])

        x = np.arange(len(ranked_ds))
        for group_label, group_models in _GROUP_MODELS.items():
            means, stds, best_names = [], [], []
            for ds in ranked_ds:
                ds_data = all_data[ds]
                best_mean, best_std, best_mk = -np.inf, 0.0, None
                for mk in group_models:
                    vals = ds_data.get(mk, [])
                    if vals:
                        m = float(np.mean(vals))
                        if m > best_mean:
                            best_mean = m
                            best_std = float(np.std(vals))
                            best_mk = mk
                if best_mk is not None:
                    means.append(best_mean)
                    stds.append(best_std)
                    best_names.append(pbr.MODEL_LABELS.get(best_mk, best_mk))
                else:
                    means.append(np.nan)
                    stds.append(0.0)
                    best_names.append("")

            m_arr = np.asarray(means)
            s_arr = np.asarray(stds)
            valid = ~np.isnan(m_arr)
            color = _GROUP_COLORS[group_label]
            ax.plot(x[valid], m_arr[valid], marker="o", ms=6, lw=1.8,
                    color=color, alpha=0.85, label=group_label, zorder=3)
            ax.fill_between(x[valid], (m_arr - s_arr)[valid],
                            (m_arr + s_arr)[valid],
                            color=color, alpha=0.15, zorder=2)
            ax.scatter(x[valid], m_arr[valid], s=30, color=color,
                       edgecolors="black", linewidths=0.3, zorder=4)

            for xi, (yi, name) in enumerate(zip(means, best_names)):
                if not np.isnan(yi) and name:
                    ax.annotate(name, (xi, yi), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=6.5,
                                color=color, rotation=30)

        refs: list[tuple[float, str, dict]] = []
        if mkey == "cohen_kappa":
            refs.append((0.76, "human inter-rater agreement: 0.76",
                         pbr._HUMAN_STYLE))
        for y_val, ref_label, style in refs:
            pbr._hline(ax, y_val, ref_label, style)

        ax.set_xticks(x)
        ax.set_xticklabels([ds.upper() for ds in ranked_ds], fontsize=10)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel, fontsize=13)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")

    fig.suptitle("Sleep staging — best model per category (sklearn_linear)",
                 fontsize=15)
    _save(fig, "final_sleep_best_per_category.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: F1 Per Stage
# ═══════════════════════════════════════════════════════════════════════════

def plot_sleep_f1_stages() -> None:
    datasets = pbr._sk_populated_datasets(
        pbr.SK_SLEEP_RESULTS_PATH, pbr.SK_SLEEP_DATASETS)
    if not datasets:
        print("skip final_sleep_f1_per_stage.png: no data")
        return

    by_ds = {ds: pbr.sk_sleep_f1_per_stage(pbr.SK_SLEEP_RESULTS_PATH, ds)
             for ds in datasets}
    ds_colors = pbr.SK_DATASET_COLORS
    ds_labels = {ds: ds.upper() for ds in datasets}

    fig, axes = plt.subplots(5, 1, figsize=(14, 25), constrained_layout=True)
    for stage_idx, (stage, ax) in enumerate(zip(pbr.STAGE_LABELS, axes)):
        line_data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for ds in datasets:
            ds_line: Dict[str, Tuple[float, float]] = {}
            for mk in pbr.SK_ALL_MODEL_ORDER:
                mat = by_ds[ds].get(mk)
                if mat is not None:
                    ds_line[mk] = (float(mat[:, stage_idx].mean()),
                                   float(mat[:, stage_idx].std()))
            if ds_line:
                line_data[ds] = ds_line

        refs: list[tuple[float, str, dict]] = []
        human_f1 = pbr.HUMAN["sleep_f1_per_stage"].get(stage)
        if human_f1 is not None:
            refs.append(
                (human_f1, f"human: {human_f1:.2f}", pbr._HUMAN_STYLE))

        _line_panel(ax, line_data, ds_colors, ds_labels, "F1", refs=refs)
        ax.set_title(f"Stage {stage}", fontsize=13)
        ax.set_ylim(0.0, 1.0)

    fig.suptitle("Sleep staging — F1 per class (sklearn_linear, test)",
                 fontsize=15)
    _save(fig, "final_sleep_f1_per_stage.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2a: F1 Per Stage (attention probe)
# ═══════════════════════════════════════════════════════════════════════════

def plot_sleep_f1_stages_attention() -> None:
    populated = [ds for ds in ATTN_SLEEP_DATASETS
                 if _load_attention_probe_metric(ds, "cohen_kappa")]
    if not populated:
        print("skip final_sleep_f1_per_stage_attention.png: no data")
        return

    by_ds = {ds: _load_attention_probe_f1_per_stage(ds) for ds in populated}
    ds_colors = pbr.SK_DATASET_COLORS
    ds_labels = {ds: ds.upper() for ds in populated}

    fig, axes = plt.subplots(5, 1, figsize=(14, 25), constrained_layout=True)
    for stage_idx, (stage, ax) in enumerate(zip(pbr.STAGE_LABELS, axes)):
        line_data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for ds in populated:
            ds_line: Dict[str, Tuple[float, float]] = {}
            for mk in pbr.SK_ALL_MODEL_ORDER:
                mat = by_ds[ds].get(mk)
                if mat is not None:
                    ds_line[mk] = (float(mat[:, stage_idx].mean()),
                                   float(mat[:, stage_idx].std()))
            if ds_line:
                line_data[ds] = ds_line

        refs: list[tuple[float, str, dict]] = []
        human_f1 = pbr.HUMAN["sleep_f1_per_stage"].get(stage)
        if human_f1 is not None:
            refs.append(
                (human_f1, f"human: {human_f1:.2f}", pbr._HUMAN_STYLE))

        _line_panel(ax, line_data, ds_colors, ds_labels, "F1", refs=refs)
        ax.set_title(f"Stage {stage}", fontsize=13)
        ax.set_ylim(0.0, 1.0)

    fig.suptitle("Sleep staging — F1 per class (attention_probe, test)",
                 fontsize=15)
    _save(fig, "final_sleep_f1_per_stage_attention.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Clinical Metrics (placeholder)
# ═══════════════════════════════════════════════════════════════════════════

_CLINICAL_CSV = (
    REPO / "artifacts" / "benchmarks" / "hypnogram_features_summary.csv"
)
_CLINICAL_FEATURES = [
    ("RemLatency", "REM latency"),
    ("WASO", "WASO"),
    ("SlStCh", "Sleep stage changes"),
    ("Awakenings", "Awakenings"),
    ("ChToR", "REM transitions"),
]
_CLINICAL_METRICS = [
    ("mae", "MAE"),
]
_CLINICAL_DATASETS = [
    "mass", "dcsm", "wsc", "ucddb", "dod", "isruc",
    "sleep_edf_expanded", "physionet2026",
]


def _load_clinical_feature(
    feature: str, metric_col: str = "mae",
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Load a metric per dataset per model from the clinical CSV.

    Returns {dataset: {model: (value, 0.0)}} — no per-fold std available.
    """
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    with open(_CLINICAL_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["feature"] != feature:
                continue
            ds = row["dataset"]
            mk = row["model"]
            if mk in pbr.SK_EXCLUDE:
                continue
            val = float(row[metric_col])
            out.setdefault(ds, {})[mk] = (val, 0.0)
    return out


def _plot_clinical_figure(
    metric_col: str, metric_label: str, filename: str,
    lower_is_better: bool = False,
) -> None:
    if not _CLINICAL_CSV.exists():
        print(f"skip {filename}: {_CLINICAL_CSV} not found")
        return

    n = len(_CLINICAL_FEATURES)
    fig, axes = plt.subplots(n, 1, figsize=(14, 6 * n),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]
    ds_colors = pbr.SK_DATASET_COLORS
    ds_labels = {ds: ds.upper() for ds in _CLINICAL_DATASETS}

    for ax, (feat, feat_label) in zip(axes, _CLINICAL_FEATURES):
        all_ds_data = _load_clinical_feature(feat, metric_col)
        line_data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        colors: Dict[str, str] = {}
        labels: Dict[str, str] = {}
        for ds in _CLINICAL_DATASETS:
            if ds in all_ds_data:
                line_data[ds] = all_ds_data[ds]
                colors[ds] = ds_colors.get(ds, "#777")
                labels[ds] = ds_labels.get(ds, ds)
        ylabel = f"{feat_label} — {metric_label}"
        _line_panel(ax, line_data, colors, labels, ylabel,
                   lower_is_better=lower_is_better)
        ax.set_title(ylabel, fontsize=13)

    fig.suptitle(f"Clinical sleep metrics ({metric_label}, test)", fontsize=15)
    _save(fig, filename)


def plot_sleep_clinical() -> None:
    for metric_col, metric_label in _CLINICAL_METRICS:
        suffix = metric_col.replace("_", "")
        _plot_clinical_figure(
            metric_col, metric_label,
            f"final_sleep_clinical_{suffix}.png",
            lower_is_better=(metric_col == "mae"),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3b: Clinical MAE — Best per Category
# ═══════════════════════════════════════════════════════════════════════════

def plot_clinical_best_per_category() -> None:
    if not _CLINICAL_CSV.exists():
        print(f"skip final_clinical_best_per_category.png: "
              f"{_CLINICAL_CSV} not found")
        return

    n = len(_CLINICAL_FEATURES)
    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, (feat, feat_label) in zip(axes, _CLINICAL_FEATURES):
        all_ds_data = _load_clinical_feature(feat, "mae")

        populated = [ds for ds in _CLINICAL_DATASETS if ds in all_ds_data]
        if not populated:
            ax.text(0.5, 0.5, f"{feat_label}\n(no data)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="gray")
            ax.set_axis_off()
            continue

        ds_avg: Dict[str, float] = {}
        for ds in populated:
            bests: list[float] = []
            for group_models in _GROUP_MODELS.values():
                best = min(
                    (all_ds_data[ds][mk][0]
                     for mk in group_models
                     if mk in all_ds_data[ds]),
                    default=np.nan)
                if not np.isnan(best):
                    bests.append(best)
            ds_avg[ds] = float(np.mean(bests)) if bests else np.inf
        ranked_ds = sorted(populated, key=lambda ds: ds_avg[ds])

        x = np.arange(len(ranked_ds))
        for group_label, group_models in _GROUP_MODELS.items():
            means, best_names = [], []
            for ds in ranked_ds:
                ds_data = all_ds_data[ds]
                best_mean, best_mk = np.inf, None
                for mk in group_models:
                    if mk in ds_data:
                        m = ds_data[mk][0]
                        if m < best_mean:
                            best_mean = m
                            best_mk = mk
                if best_mk is not None:
                    means.append(best_mean)
                    best_names.append(
                        pbr.MODEL_LABELS.get(best_mk, best_mk))
                else:
                    means.append(np.nan)
                    best_names.append("")

            m_arr = np.asarray(means)
            valid = ~np.isnan(m_arr)
            color = _GROUP_COLORS[group_label]
            ax.plot(x[valid], m_arr[valid], marker="o", ms=6, lw=1.8,
                    color=color, alpha=0.85, label=group_label, zorder=3)
            ax.scatter(x[valid], m_arr[valid], s=30, color=color,
                       edgecolors="black", linewidths=0.3, zorder=4)

            for xi, (yi, name) in enumerate(zip(means, best_names)):
                if not np.isnan(yi) and name:
                    ax.annotate(name, (xi, yi), textcoords="offset points",
                                xytext=(0, -10), ha="center", fontsize=6.5,
                                color=color, rotation=30)

        ax.set_xticks(x)
        ax.set_xticklabels([ds.upper() for ds in ranked_ds], fontsize=10)
        ax.set_ylabel(f"{feat_label} — MAE")
        ax.set_title(f"{feat_label} — MAE (lower is better)", fontsize=13)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        "Clinical sleep metrics — best model per category (MAE, lower ↓)",
        fontsize=15)
    _save(fig, "final_clinical_best_per_category.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Scorer Agreement
# ═══════════════════════════════════════════════════════════════════════════

def plot_scorer_agreement() -> None:
    metric_key = "cohen_kappa"
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), constrained_layout=True)

    # --- DOD ---
    ax = axes[0]
    dod_scorer_paths = pbr.SK_DOD_SCORER_PATHS["dod"]
    dod_line_data: Dict[str, Dict[str, Tuple[float, float]]] = {}
    dod_colors: Dict[str, str] = {}
    dod_labels: Dict[str, str] = {}
    dod_widths: Dict[str, float] = {}

    cons_data = pbr.sk_sleep_metric(
        pbr.SK_SLEEP_RESULTS_PATH, "dod", metric_key)
    cons_line: Dict[str, Tuple[float, float]] = {}
    for mk in pbr.SK_ALL_MODEL_ORDER:
        vals = cons_data.get(mk, [])
        if vals:
            cons_line[mk] = (float(np.mean(vals)), float(np.std(vals)))
    if cons_line:
        dod_line_data["consensus"] = cons_line
        dod_colors["consensus"] = "#111111"
        dod_labels["consensus"] = "Consensus"
        dod_widths["consensus"] = 3.0

    for scorer_key, scorer_path in dod_scorer_paths.items():
        sc_data = pbr.sk_sleep_metric(
            {scorer_key: scorer_path}, scorer_key, metric_key)
        sc_line: Dict[str, Tuple[float, float]] = {}
        for mk in pbr.SK_ALL_MODEL_ORDER:
            vals = sc_data.get(mk, [])
            if vals:
                sc_line[mk] = (float(np.mean(vals)), float(np.std(vals)))
        if sc_line:
            dod_line_data[scorer_key] = sc_line
            dod_colors[scorer_key] = pbr.SK_DOD_SCORER_COLORS.get(
                scorer_key, "#777")
            dod_labels[scorer_key] = scorer_key.replace("_", " ").title()
            dod_widths[scorer_key] = 1.4

    _line_panel(ax, dod_line_data, dod_colors, dod_labels, "Cohen's κ",
                line_widths=dod_widths)
    ax.set_title("DOD — scorer agreement", fontsize=13)

    # --- ISRUC ---
    ax = axes[1]
    isruc_line_data: Dict[str, Dict[str, Tuple[float, float]]] = {}
    isruc_colors: Dict[str, str] = {}
    isruc_labels: Dict[str, str] = {}
    isruc_widths: Dict[str, float] = {}

    cons_data = pbr.sk_sleep_metric(
        pbr.SK_SLEEP_RESULTS_PATH, "isruc", metric_key)
    cons_line = {}
    for mk in pbr.SK_ALL_MODEL_ORDER:
        vals = cons_data.get(mk, [])
        if vals:
            cons_line[mk] = (float(np.mean(vals)), float(np.std(vals)))
    if cons_line:
        isruc_line_data["consensus"] = cons_line
        isruc_colors["consensus"] = "#111111"
        isruc_labels["consensus"] = "Consensus"
        isruc_widths["consensus"] = 3.0

    for scorer_key in pbr.SK_ISRUC_SCORERS:
        if scorer_key == "isruc":
            continue
        scorer_path = pbr.SK_ISRUC_SCORER_PATHS[scorer_key]
        sc_data = pbr.sk_sleep_metric(
            {scorer_key: scorer_path}, scorer_key, metric_key)
        sc_line = {}
        for mk in pbr.SK_ALL_MODEL_ORDER:
            vals = sc_data.get(mk, [])
            if vals:
                sc_line[mk] = (float(np.mean(vals)), float(np.std(vals)))
        if sc_line:
            isruc_line_data[scorer_key] = sc_line
            isruc_colors[scorer_key] = pbr.SK_DATASET_COLORS.get(
                scorer_key, "#777")
            isruc_labels[scorer_key] = (scorer_key.replace("isruc_", "")
                                        .replace("_", " ").title())
            isruc_widths[scorer_key] = 1.4

    _line_panel(ax, isruc_line_data, isruc_colors, isruc_labels, "Cohen's κ",
                line_widths=isruc_widths)
    ax.set_title("ISRUC — scorer agreement", fontsize=13)

    fig.suptitle("Scorer agreement — Cohen's κ (sklearn_linear, test)",
                 fontsize=15)
    _save(fig, "final_scorer_agreement.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Sleep Subgroups
# ═══════════════════════════════════════════════════════════════════════════

_SUBGROUP_PALETTE = [
    "#E15759", "#4E79A7", "#76B7B2", "#F28E2B", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


_SUBGROUP_PROBES = [
    ("ISRUC",
     {k: v for k, v in pbr.SK_ISRUC_SUBGROUP_PATHS.items() if k != "isruc"},
     {"isruc_SG_I_II": "SG I+II", "isruc_SG_III": "SG III"}),
    ("DOD",
     {k: v for k, v in pbr.SK_DOD_SUBGROUP_PATHS.items() if k != "dod"},
     {"dod_dodh": "DOD-H", "dod_dodo": "DOD-O"}),
    ("PhysioNet2026",
     {k: v for k, v in pbr.SK_PN2026_SITE_PATHS.items()
      if k != "physionet2026"},
     {"physionet2026_I0002": "I0002", "physionet2026_I0006": "I0006",
      "physionet2026_S0001": "S0001"}),
    ("SleepEDF",
     {k: v for k, v in pbr.SK_SLEEPEDF_SUBGROUP_PATHS.items()
      if k != "sleep_edf_expanded"},
     {"sleep_edf_expanded_cassette": "Cassette",
      "sleep_edf_expanded_telemetry": "Telemetry"}),
    ("MASS",
     {k: v for k, v in pbr.SK_MASS_SUBSET_PATHS.items() if k != "mass"},
     {"mass_SS01": "SS01", "mass_SS02": "SS02", "mass_SS03": "SS03",
      "mass_SS04": "SS04", "mass_SS05": "SS05"}),
]


def plot_sleep_subgroups() -> None:
    metric_key = "cohen_kappa"

    fig, axes = plt.subplots(len(_SUBGROUP_PROBES), 1,
                             figsize=(14, 6 * len(_SUBGROUP_PROBES)),
                             constrained_layout=True)

    for idx, (title, path_map, sg_labels) in enumerate(_SUBGROUP_PROBES):
        ax = axes[idx]

        line_data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        sg_colors: Dict[str, str] = {}
        sg_display: Dict[str, str] = {}

        for ci, (sg_key, sg_label) in enumerate(sg_labels.items()):
            if sg_key not in path_map:
                continue
            ds_data = pbr.sk_sleep_metric(path_map, sg_key, metric_key)
            grp_line: Dict[str, Tuple[float, float]] = {}
            for mk in pbr.SK_ALL_MODEL_ORDER:
                vals = ds_data.get(mk, [])
                if vals:
                    grp_line[mk] = (float(np.mean(vals)),
                                    float(np.std(vals)))
            if grp_line:
                line_data[sg_key] = grp_line
                sg_colors[sg_key] = (pbr.SK_DATASET_COLORS.get(sg_key)
                                     or _SUBGROUP_PALETTE[
                                         ci % len(_SUBGROUP_PALETTE)])
                sg_display[sg_key] = sg_label

        if line_data:
            _line_panel(ax, line_data, sg_colors, sg_display, "Cohen's κ")
            ax.set_title(f"{title} subgroups", fontsize=13)
        else:
            ax.text(0.5, 0.5, f"{title}\n(no data)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="gray")
            ax.set_axis_off()

    fig.suptitle("Sleep staging subgroups — Cohen's κ (sklearn_linear, test)",
                 fontsize=15)
    _save(fig, "final_sleep_subgroups.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Arousal Detection
# ═══════════════════════════════════════════════════════════════════════════

def plot_arousal() -> None:
    thr_key = "threshold_3.0s"
    probe = "sklearn_linear_balanced"
    datasets = {
        "mass_arousal": "MASS",
        "physionet2026_arousal": "PhysioNet2026",
    }
    ds_colors = {
        "mass_arousal": "#4C72B0",
        "physionet2026_arousal": "#DD8452",
    }

    metrics = [("auprc", "AUPRC"), ("mcc", "MCC")]
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), constrained_layout=True)

    for ax, (mkey, mlabel) in zip(axes, metrics):
        line_data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for ds_key in datasets:
            path = (REPO / "artifacts/benchmarks"
                    / pbr.SK_AROUSAL_PATHS[ds_key].format(probe=probe))
            metric_vals = pbr.sk_event_metric(path, None, thr_key, mkey)
            ds_line: Dict[str, Tuple[float, float]] = {}
            for mk in pbr.SK_ALL_MODEL_ORDER:
                vals = metric_vals.get(mk, [])
                if vals:
                    ds_line[mk] = (float(np.mean(vals)),
                                   float(np.std(vals)))
            if ds_line:
                line_data[ds_key] = ds_line

        refs: list[tuple[float, str, dict]] = []
        if mkey == "mcc":
            refs.append((0.0, "chance: 0",
                         dict(color="#999", linestyle="--", linewidth=0.8)))
        elif mkey == "auprc":
            for ds_key, ds_label in datasets.items():
                path = (REPO / "artifacts/benchmarks"
                        / pbr.SK_AROUSAL_PATHS[ds_key].format(probe=probe))
                prev = pbr.sk_event_prevalence(path, None, thr_key)
                if prev > 0:
                    refs.append((prev,
                                 f"prevalence ({ds_label}): {prev:.2f}",
                                 dict(color=ds_colors[ds_key],
                                      linestyle=":", linewidth=1.0)))

        _line_panel(ax, line_data, ds_colors, datasets, mlabel, refs=refs)
        ax.set_title(f"{mlabel} @ 3s (balanced)", fontsize=13)

    fig.suptitle("Arousal detection (sklearn_linear_balanced, test)",
                 fontsize=15)
    _save(fig, "final_arousal.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6a-bis: Arousal — Best per Category (AUPRC)
# ═══════════════════════════════════════════════════════════════════════════

def plot_arousal_best_per_category() -> None:
    thr_key = "threshold_3.0s"
    probe = "sklearn_linear_balanced"
    mkey = "auprc"
    datasets = {
        "mass_arousal": "MASS",
        "physionet2026_arousal": "PhysioNet2026",
    }
    ds_colors_map = {
        "mass_arousal": "#4C72B0",
        "physionet2026_arousal": "#DD8452",
    }

    all_data: Dict[str, Dict[str, List[float]]] = {}
    for ds_key in datasets:
        path = (REPO / "artifacts/benchmarks"
                / pbr.SK_AROUSAL_PATHS[ds_key].format(probe=probe))
        all_data[ds_key] = pbr.sk_event_metric(path, None, thr_key, mkey)

    ds_list = list(datasets.keys())
    ds_avg: Dict[str, float] = {}
    for ds in ds_list:
        bests: list[float] = []
        for group_models in _GROUP_MODELS.values():
            best = max(
                (float(np.mean(all_data[ds].get(mk, [])))
                 for mk in group_models if all_data[ds].get(mk)),
                default=np.nan)
            if not np.isnan(best):
                bests.append(best)
        ds_avg[ds] = float(np.mean(bests)) if bests else 0.0
    ranked_ds = sorted(ds_list, key=lambda ds: -ds_avg[ds])

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    x = np.arange(len(ranked_ds))

    for group_label, group_models in _GROUP_MODELS.items():
        means, stds, best_names = [], [], []
        for ds in ranked_ds:
            ds_data = all_data[ds]
            best_mean, best_std, best_mk = -np.inf, 0.0, None
            for mk in group_models:
                vals = ds_data.get(mk, [])
                if vals:
                    m = float(np.mean(vals))
                    if m > best_mean:
                        best_mean = m
                        best_std = float(np.std(vals))
                        best_mk = mk
            if best_mk is not None:
                means.append(best_mean)
                stds.append(best_std)
                best_names.append(pbr.MODEL_LABELS.get(best_mk, best_mk))
            else:
                means.append(np.nan)
                stds.append(0.0)
                best_names.append("")

        m_arr = np.asarray(means)
        s_arr = np.asarray(stds)
        valid = ~np.isnan(m_arr)
        color = _GROUP_COLORS[group_label]
        ax.plot(x[valid], m_arr[valid], marker="o", ms=6, lw=1.8,
                color=color, alpha=0.85, label=group_label, zorder=3)
        ax.fill_between(x[valid], (m_arr - s_arr)[valid],
                        (m_arr + s_arr)[valid],
                        color=color, alpha=0.15, zorder=2)
        ax.scatter(x[valid], m_arr[valid], s=30, color=color,
                   edgecolors="black", linewidths=0.3, zorder=4)

        for xi, (yi, name) in enumerate(zip(means, best_names)):
            if not np.isnan(yi) and name:
                ax.annotate(name, (xi, yi), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=6.5,
                            color=color, rotation=30)

    for ds in ranked_ds:
        path = (REPO / "artifacts/benchmarks"
                / pbr.SK_AROUSAL_PATHS[ds].format(probe=probe))
        prev = pbr.sk_event_prevalence(path, None, thr_key)
        if prev > 0:
            pbr._hline(ax, prev,
                       f"prevalence ({datasets[ds]}): {prev:.2f}",
                       dict(color=ds_colors_map[ds], linestyle=":",
                            linewidth=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels([datasets[ds] for ds in ranked_ds], fontsize=10)
    ax.set_ylabel("AUPRC")
    ax.set_title("AUPRC @ 3s (balanced)", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    fig.suptitle("Arousal — best model per category (sklearn_linear_balanced)",
                 fontsize=15)
    _save(fig, "final_arousal_best_per_category.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6b: Respiratory Event Detection
# ═══════════════════════════════════════════════════════════════════════════

def plot_respiratory_events() -> None:
    probe = "sklearn_linear_balanced"
    line_configs = {
        "pn2026_ahi_0s": ("physionet2026_respevt", "ahi_fraction",
                          "threshold_0.0s", "PN2026 AHI @ 0s"),
        "pn2026_ahi_10s": ("physionet2026_respevt", "ahi_fraction",
                           "threshold_10.0s", "PN2026 AHI @ 10s"),
        "ucddb_0s": ("ucddb_respevt", "respiratory_event_any_fraction",
                      "threshold_0.0s", "UCDDB resp-any @ 0s"),
        "ucddb_10s": ("ucddb_respevt", "respiratory_event_any_fraction",
                       "threshold_10.0s", "UCDDB resp-any @ 10s"),
    }
    line_colors = {
        "pn2026_ahi_0s": "#64B5CD", "pn2026_ahi_10s": "#3A7FA8",
        "ucddb_0s": "#C44E52", "ucddb_10s": "#8B2E31",
    }
    line_labels_map = {k: v[3] for k, v in line_configs.items()}
    line_styles = {
        "pn2026_ahi_0s": "-", "pn2026_ahi_10s": "--",
        "ucddb_0s": "-", "ucddb_10s": "--",
    }

    metrics = [("auprc", "AUPRC"), ("mcc", "MCC")]
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), constrained_layout=True)

    for ax, (mkey, mlabel) in zip(axes, metrics):
        line_data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for lk, (ds_slug, evt_field, thr_key, _) in line_configs.items():
            path = (REPO / "artifacts/benchmarks"
                    / ds_slug / probe / "results.json")
            metric_vals = pbr.sk_event_metric(path, evt_field, thr_key, mkey)
            ds_line: Dict[str, Tuple[float, float]] = {}
            for mk in pbr.SK_ALL_MODEL_ORDER:
                vals = metric_vals.get(mk, [])
                if vals:
                    ds_line[mk] = (float(np.mean(vals)),
                                   float(np.std(vals)))
            if ds_line:
                line_data[lk] = ds_line

        refs: list[tuple[float, str, dict]] = []
        if mkey == "mcc":
            refs.append((0.0, "chance: 0",
                         dict(color="#999", linestyle="--", linewidth=0.8)))
        elif mkey == "auprc":
            for lk, (ds_slug, evt_field, thr_key, display) in line_configs.items():
                if "10s" not in lk:
                    continue
                path = (REPO / "artifacts/benchmarks"
                        / ds_slug / probe / "results.json")
                prev = pbr.sk_event_prevalence(path, evt_field, thr_key)
                if prev > 0:
                    refs.append((prev,
                                 f"prevalence ({display}): {prev:.2f}",
                                 dict(color=line_colors[lk],
                                      linestyle=":", linewidth=1.0)))

        _line_panel(ax, line_data, line_colors, line_labels_map, mlabel,
                    refs=refs, line_styles=line_styles)
        ax.set_title(f"Respiratory events — {mlabel} (balanced)", fontsize=13)

    fig.suptitle("Respiratory event detection (sklearn_linear_balanced, test)",
                 fontsize=15)
    _save(fig, "final_respiratory_events.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6c: Limb Movement Detection
# ═══════════════════════════════════════════════════════════════════════════

def plot_limb_movement() -> None:
    probe = "sklearn_linear_balanced"
    thr_key = "threshold_0.0s"
    evt_field = "limb_movement_plm_fraction"
    ds_key = "physionet2026_limb"

    metrics = [("auprc", "AUPRC"), ("mcc", "MCC")]
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), constrained_layout=True)

    for ax, (mkey, mlabel) in zip(axes, metrics):
        path = (REPO / "artifacts/benchmarks"
                / ds_key / probe / "results.json")
        metric_vals = pbr.sk_event_metric(path, evt_field, thr_key, mkey)
        ds_line: Dict[str, Tuple[float, float]] = {}
        for mk in pbr.SK_ALL_MODEL_ORDER:
            vals = metric_vals.get(mk, [])
            if vals:
                ds_line[mk] = (float(np.mean(vals)),
                               float(np.std(vals)))

        line_data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        if ds_line:
            line_data["plm"] = ds_line

        refs: list[tuple[float, str, dict]] = []
        if mkey == "mcc":
            refs.append((0.0, "chance: 0",
                         dict(color="#999", linestyle="--", linewidth=0.8)))
        elif mkey == "auprc":
            prev = pbr.sk_event_prevalence(path, evt_field, thr_key)
            if prev > 0:
                refs.append((prev, f"prevalence: {prev:.2f}",
                             dict(color="#4C72B0", linestyle=":",
                                  linewidth=1.0)))

        _line_panel(ax, line_data,
                    {"plm": "#4C72B0"},
                    {"plm": "PhysioNet2026 PLM"},
                    mlabel, refs=refs)
        ax.set_title(f"Limb movement (PLM) — {mlabel} @ 0s (balanced)",
                     fontsize=13)

    fig.suptitle(
        "Limb movement detection — PLM (sklearn_linear_balanced, test)",
        fontsize=15)
    _save(fig, "final_limb_movement.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: Channel Ablation (C3-M2 vs all-channel)
# ═══════════════════════════════════════════════════════════════════════════

def plot_channel_ablation() -> None:
    variant_colors = {"all-channel": "#4C72B0", "C3-M2": "#DD8452"}
    variant_labels = {"all-channel": "All channels", "C3-M2": "C3-M2"}
    variant_styles = {"all-channel": "-", "C3-M2": "--"}
    variant_widths = {"all-channel": 1.8, "C3-M2": 1.4}
    eeg_set = set(pbr.SK_MODEL_ORDER)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

    # --- Sleep staging: Cohen's kappa ---
    ax = axes[0]
    allch_map = {"physionet2026": "physionet2026/sklearn_linear/results.json"}
    allch = pbr.sk_sleep_metric(allch_map, "physionet2026", "cohen_kappa")
    c3m2 = pbr.sk_sleep_metric(pbr.C3M2_SLEEP_PATH, "physionet2026",
                                "cohen_kappa")
    line_data: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for vk, md in [("all-channel", allch), ("C3-M2", c3m2)]:
        vl: Dict[str, Tuple[float, float]] = {}
        for mk, vals in md.items():
            if mk in eeg_set and vals:
                vl[mk] = (float(np.mean(vals)), float(np.std(vals)))
        if vl:
            line_data[vk] = vl

    _line_panel(ax, line_data, variant_colors, variant_labels, "Cohen's κ",
                eeg_only=True, line_styles=variant_styles,
                line_widths=variant_widths)
    ax.set_title("Sleep staging — Cohen's κ", fontsize=13)

    # --- Arousal: AUPRC ---
    ax = axes[1]
    thr_key = "threshold_3.0s"
    probe = "sklearn_linear_balanced"
    allch_path = (REPO / "artifacts/benchmarks"
                  / f"physionet2026_arousal/{probe}/results.json")
    c3m2_path = (REPO / "artifacts/benchmarks"
                 / pbr.C3M2_AROUSAL_PATH.format(probe=probe))
    allch_ar = pbr.sk_event_metric(allch_path, None, thr_key, "auprc")
    c3m2_ar = pbr.sk_event_metric(c3m2_path, None, thr_key, "auprc")

    line_data = {}
    for vk, md in [("all-channel", allch_ar), ("C3-M2", c3m2_ar)]:
        vl = {}
        for mk, vals in md.items():
            if mk in eeg_set and vals:
                vl[mk] = (float(np.mean(vals)), float(np.std(vals)))
        if vl:
            line_data[vk] = vl

    _line_panel(ax, line_data, variant_colors, variant_labels, "AUPRC",
                eeg_only=True, line_styles=variant_styles,
                line_widths=variant_widths)
    ax.set_title("Arousal — AUPRC @ 3s (balanced)", fontsize=13)

    fig.suptitle("PhysioNet2026 — all-channel vs C3-M2 (EEG FMs)",
                 fontsize=15)
    _save(fig, "final_channel_ablation.png")


# ═══════════════════════════════════════════════════════════════════════════
# Diagnosis helpers
# ═══════════════════════════════════════════════════════════════════════════

ISRUC_DIAG_CLASSES = ["healthy", "osa", "snoring", "affective_disorder"]
ISRUC_DIAG_CLASS_COLORS = {
    "healthy": "#4DAF4A", "osa": "#E15759",
    "snoring": "#F28E2B", "affective_disorder": "#9467BD",
}


def _sk_patient_metric_from_bt(
    path: Path, key: str, aggregation: str = "mean",
) -> Dict[str, List[float]]:
    """Extract a single scalar metric from best_test per model."""
    entries = pbr._sk_load_entries(path, "patient_classification_eval")
    entries = [e for e in entries
               if (e.get("metadata") or {}).get("aggregation") == aggregation]
    out: Dict[str, List[float]] = defaultdict(list)
    for e in entries:
        bt = (e.get("metrics") or {}).get("best_test", {}) or {}
        v = bt.get(key)
        if v is not None:
            out[e["checkpoint_id"]].append(float(v))
    return dict(out)


def _sk_patient_per_class(
    path: Path, class_idx: int, field: str = "f1_per_class",
    aggregation: str = "mean",
) -> Dict[str, List[float]]:
    """Extract one class's value from a per-class list in best_test."""
    entries = pbr._sk_load_entries(path, "patient_classification_eval")
    entries = [e for e in entries
               if (e.get("metadata") or {}).get("aggregation") == aggregation]
    out: Dict[str, List[float]] = defaultdict(list)
    for e in entries:
        bt = (e.get("metrics") or {}).get("best_test", {}) or {}
        arr = bt.get(field)
        if isinstance(arr, list) and len(arr) > class_idx:
            out[e["checkpoint_id"]].append(float(arr[class_idx]))
    return dict(out)


def _sk_patient_recall_from_cm(
    path: Path, class_idx: int, aggregation: str = "mean",
) -> Dict[str, List[float]]:
    """Compute per-class recall from the confusion matrix."""
    entries = pbr._sk_load_entries(path, "patient_classification_eval")
    entries = [e for e in entries
               if (e.get("metadata") or {}).get("aggregation") == aggregation]
    out: Dict[str, List[float]] = defaultdict(list)
    for e in entries:
        bt = (e.get("metrics") or {}).get("best_test", {}) or {}
        cm = bt.get("confusion_matrix")
        if cm and len(cm) > class_idx:
            row = cm[class_idx]
            total = sum(row)
            if total > 0:
                out[e["checkpoint_id"]].append(row[class_idx] / total)
    return dict(out)


def _metric_to_line(
    metric_vals: Dict[str, List[float]],
) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for mk in pbr.SK_ALL_MODEL_ORDER:
        vals = metric_vals.get(mk, [])
        if vals:
            out[mk] = (float(np.mean(vals)), float(np.std(vals)))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8: ISRUC Diagnosis
# ═══════════════════════════════════════════════════════════════════════════

def plot_diagnosis_isruc() -> None:
    probe = "sklearn_linear_balanced"
    path = (REPO / "artifacts/benchmarks"
            / f"isruc_diagnosis/diagnosis/healthy_osa_snoring"
            / f"{probe}/results.json")
    classes = ["healthy", "osa", "snoring"]

    if not path.exists():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5,
                "Placeholder — ISRUC diagnosis\nProbes need rerunning",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="gray")
        ax.set_axis_off()
        _save(fig, "final_diagnosis_isruc.png")
        return

    osa_idx = classes.index("osa")

    fig, axes = plt.subplots(3, 1, figsize=(14, 18), constrained_layout=True)

    # --- Panel 1: OSA Recall ---
    ax = axes[0]
    osa_recall = _sk_patient_recall_from_cm(path, osa_idx)
    line = _metric_to_line(osa_recall)
    line_data: Dict[str, Dict[str, Tuple[float, float]]] = {}
    if line:
        line_data["osa_recall"] = line
    _line_panel(ax, line_data,
                {"osa_recall": ISRUC_DIAG_CLASS_COLORS["osa"]},
                {"osa_recall": "OSA recall"},
                "Recall")
    ax.set_title("OSA Recall (balanced)", fontsize=13)
    ax.set_ylim(0.0, 1.0)

    # --- Panel 2: macro-F1 ---
    ax = axes[1]
    macro_f1 = _sk_patient_metric_from_bt(path, "macro_f1")
    line = _metric_to_line(macro_f1)
    line_data = {}
    if line:
        line_data["macro_f1"] = line
    _line_panel(ax, line_data,
                {"macro_f1": "#4C72B0"},
                {"macro_f1": "macro-F1"},
                "macro-F1")
    ax.set_title("macro-F1 (balanced)", fontsize=13)
    ax.set_ylim(0.0, 1.0)

    # --- Panel 3: Per-class F1 ---
    ax = axes[2]
    line_data = {}
    class_colors: Dict[str, str] = {}
    class_labels: Dict[str, str] = {}
    for ci, cls in enumerate(classes):
        cls_f1 = _sk_patient_per_class(path, ci, "f1_per_class")
        line = _metric_to_line(cls_f1)
        if line:
            line_data[cls] = line
            class_colors[cls] = ISRUC_DIAG_CLASS_COLORS[cls]
            class_labels[cls] = cls.replace("_", " ").title()
    _line_panel(ax, line_data, class_colors, class_labels, "F1")
    ax.set_title("Per-class F1 (balanced)", fontsize=13)
    ax.set_ylim(0.0, 1.0)

    fig.suptitle("ISRUC Diagnosis (sklearn_linear_balanced)", fontsize=15)
    _save(fig, "final_diagnosis_isruc.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9: PhysioNet Cognitive Impairment
# ═══════════════════════════════════════════════════════════════════════════

def plot_diagnosis_cognitive() -> None:
    probe = "sklearn_linear_balanced"
    path = (REPO / "artifacts/benchmarks"
            / f"physionet2026_cognitive/{probe}/results.json")

    metrics = [
        ("auroc", "AUROC"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
    ]
    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(14, 6 * len(metrics)),
                             constrained_layout=True)

    for ax, (mkey, mlabel) in zip(axes, metrics):
        metric_vals = _sk_patient_metric_from_bt(path, mkey)
        line = _metric_to_line(metric_vals)
        line_data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        if line:
            line_data["physionet2026"] = line

        refs: list[tuple[float, str, dict]] = []
        if mkey == "auroc":
            refs.append((0.5, "chance: 0.50",
                         dict(color="#999", linestyle="--", linewidth=0.8)))
        elif mkey in ("sensitivity", "specificity"):
            refs.append((0.5, "chance: 0.50",
                         dict(color="#999", linestyle="--", linewidth=0.8)))

        _line_panel(ax, line_data,
                    {"physionet2026": "#64B5CD"},
                    {"physionet2026": "PhysioNet2026"},
                    mlabel, refs=refs)
        ax.set_title(f"Cognitive impairment — {mlabel} (balanced)",
                     fontsize=13)

    fig.suptitle(
        "PhysioNet2026 cognitive impairment (sklearn_linear_balanced)",
        fontsize=15)
    _save(fig, "final_diagnosis_cognitive.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9b: Diagnosis — Best per Category
# ═══════════════════════════════════════════════════════════════════════════

_DIAG_TASKS = [
    ("ISRUC diagnosis",
     REPO / "artifacts/benchmarks"
     / "isruc_diagnosis/diagnosis/healthy_osa_snoring"
     / "sklearn_linear_balanced/results.json",
     "macro_f1", "macro-F1"),
    ("PN2026 cognitive",
     REPO / "artifacts/benchmarks"
     / "physionet2026_cognitive/sklearn_linear_balanced/results.json",
     "auroc", "AUROC"),
]


def plot_diagnosis_best_per_category() -> None:
    tasks = [(label, path, mkey, mlabel)
             for label, path, mkey, mlabel in _DIAG_TASKS if path.exists()]
    if not tasks:
        print("skip final_diagnosis_best_per_category.png: no data")
        return

    all_data: Dict[str, Dict[str, List[float]]] = {}
    task_labels: Dict[str, str] = {}
    task_metric_labels: Dict[str, str] = {}
    for label, path, mkey, mlabel in tasks:
        metric_vals = _sk_patient_metric_from_bt(path, mkey)
        all_data[label] = metric_vals
        task_labels[label] = label
        task_metric_labels[label] = mlabel

    task_keys = [t[0] for t in tasks]
    ds_avg: Dict[str, float] = {}
    for tk in task_keys:
        bests: list[float] = []
        for group_models in _GROUP_MODELS.values():
            best = max(
                (float(np.mean(all_data[tk].get(mk, [])))
                 for mk in group_models if all_data[tk].get(mk)),
                default=np.nan)
            if not np.isnan(best):
                bests.append(best)
        ds_avg[tk] = float(np.mean(bests)) if bests else 0.0
    ranked = sorted(task_keys, key=lambda tk: -ds_avg[tk])

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    x = np.arange(len(ranked))

    for group_label, group_models in _GROUP_MODELS.items():
        means, stds, best_names = [], [], []
        for tk in ranked:
            ds_data = all_data[tk]
            best_mean, best_std, best_mk = -np.inf, 0.0, None
            for mk in group_models:
                vals = ds_data.get(mk, [])
                if vals:
                    m = float(np.mean(vals))
                    if m > best_mean:
                        best_mean = m
                        best_std = float(np.std(vals))
                        best_mk = mk
            if best_mk is not None:
                means.append(best_mean)
                stds.append(best_std)
                best_names.append(pbr.MODEL_LABELS.get(best_mk, best_mk))
            else:
                means.append(np.nan)
                stds.append(0.0)
                best_names.append("")

        m_arr = np.asarray(means)
        s_arr = np.asarray(stds)
        valid = ~np.isnan(m_arr)
        color = _GROUP_COLORS[group_label]
        ax.plot(x[valid], m_arr[valid], marker="o", ms=6, lw=1.8,
                color=color, alpha=0.85, label=group_label, zorder=3)
        ax.fill_between(x[valid], (m_arr - s_arr)[valid],
                        (m_arr + s_arr)[valid],
                        color=color, alpha=0.15, zorder=2)
        ax.scatter(x[valid], m_arr[valid], s=30, color=color,
                   edgecolors="black", linewidths=0.3, zorder=4)

        for xi, (yi, name) in enumerate(zip(means, best_names)):
            if not np.isnan(yi) and name:
                ax.annotate(name, (xi, yi), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=6.5,
                            color=color, rotation=30)

    refs = [(0.5, "chance: 0.50",
             dict(color="#999", linestyle="--", linewidth=0.8))]
    for y_val, ref_label, style in refs:
        pbr._hline(ax, y_val, ref_label, style)

    tick_labels = [f"{tk}\n({task_metric_labels[tk]})" for tk in ranked]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Best model per category (balanced)", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        "Diagnosis — best model per category (sklearn_linear_balanced)",
        fontsize=15)
    _save(fig, "final_diagnosis_best_per_category.png")


# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    plot_sleep_overview()
    plot_sleep_overview_attention()
    plot_sleep_best_per_category()
    plot_sleep_f1_stages()
    plot_sleep_f1_stages_attention()
    plot_sleep_clinical()
    plot_clinical_best_per_category()
    plot_scorer_agreement()
    plot_sleep_subgroups()
    plot_arousal()
    plot_arousal_best_per_category()
    plot_respiratory_events()
    plot_limb_movement()
    plot_channel_ablation()
    plot_diagnosis_isruc()
    plot_diagnosis_cognitive()
    plot_diagnosis_best_per_category()


if __name__ == "__main__":
    main()
