#!/usr/bin/env python3
"""Generate comprehensive EEG Benchmarks summary: plots and tables."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT = Path(os.path.expandvars("${EEG_DATA_ROOT}/EEGBenchmarks/artifacts/plots/summary"))
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
MODELS = ["BENDR", "EEGPT", "NeuroLM", "LaBraM", "CBraMod"]
MODEL_COLORS = {
    "BENDR": "#1f77b4",
    "EEGPT": "#ff7f0e",
    "NeuroLM": "#2ca02c",
    "LaBraM": "#d62728",
    "CBraMod": "#9467bd",
}
TRACK_COLORS = {"A": "#4c72b0", "B": "#dd8452", "C": "#55a868", "D": "#c44e52"}

# ──────────────────────────────────────────────────────────────────────
# DATA: Motor Imagery — Track A LOSO
# ──────────────────────────────────────────────────────────────────────
mi_datasets = [
    "PhysionetMI", "Cho2017", "Lee2019_MI", "Schirrmeister2017",
    "Weibo2014", "BNCI2014_001", "Dreyer2023A", "BNCI2014_004", "BNCI2015_001"
]
mi_subjects = [103, 49, 54, 14, 10, 9, 60, 9, 12]
mi_channels = [64, 64, 20, 128, 60, 22, 27, 3, 13]

mi_trackA_loso = {
    "BENDR":   [61.7, 58.7, 50.9, 50.8, 52.3, 53.0, 60.7, 54.8, 51.4],
    "EEGPT":   [64.1, 56.1, 64.7, 63.8, 57.5, 66.7, 70.9, 61.0, 60.7],
    "NeuroLM": [73.5, 55.5, 58.4, 52.3, 49.2, 55.3, 59.8, 64.5, 60.2],
    "LaBraM":  [51.7, 55.1, 55.3, 54.1, 49.6, 54.5, 56.0, 53.0, 55.6],
    "CBraMod": [49.6, 53.5, 53.0, 51.5, 51.3, 53.1, 53.7, 55.2, 55.9],
}

mi_trackB_loso = {
    "BENDR":   [50.2, 50.5, 50.1, 49.7, 50.9, 49.7, 56.2, 53.5, 53.2],
    "EEGPT":   [61.5, 55.6, 64.6, 64.8, 58.3, 64.8, 70.5, 61.1, 59.5],
    "NeuroLM": [55.0, 51.6, 54.1, 49.8, 51.0, 54.6, 53.5, 62.3, 58.7],
    "LaBraM":  [54.5, 53.7, 56.2, 54.8, 53.0, 55.5, 57.0, 56.6, 57.4],
    "CBraMod": [49.9, 51.6, 53.2, 50.3, 48.1, 54.0, 54.2, 55.1, 55.5],
}

mi_trackA_kfold = {
    "BENDR":   [61.9, 59.0, 51.4, 50.6, 53.0, 52.8, 60.2, 55.9, 51.5],
    "EEGPT":   [63.4, 55.8, 64.4, 64.2, 59.5, 65.9, 71.0, 62.0, 61.3],
    "NeuroLM": [73.1, 56.0, 57.4, 51.3, 51.3, 55.4, 60.3, 65.4, 59.2],
    "LaBraM":  [51.8, 55.5, 54.9, 54.5, 51.1, 53.9, 56.2, 53.9, 56.9],
    "CBraMod": [50.4, 53.2, 53.3, 49.8, 49.8, 54.4, 53.3, 55.9, 56.1],
}

mi_trackB_kfold = {
    "BENDR":   [51.5, 50.1, 49.6, 49.9, 51.6, 50.2, 55.9, 52.7, 53.1],
    "EEGPT":   [60.3, 54.9, 64.1, 64.4, 58.3, 66.3, 69.9, 62.2, 60.8],
    "NeuroLM": [53.6, 51.3, 54.0, 50.0, 50.4, 54.0, 53.0, 63.2, 58.8],
    "LaBraM":  [54.5, 53.7, 56.1, 55.0, 53.2, 56.6, 56.7, 57.4, 58.0],
    "CBraMod": [50.6, 51.4, 52.8, 49.9, 48.3, 54.3, 53.6, 55.2, 56.8],
}

# Track C LOSO (tmin=1.0, wideband)
mi_trackC_loso = {
    "BENDR":   [53.3, 55.0, 50.9, 49.6, 51.1, 53.1, 52.8, 51.7, 52.4],
    "EEGPT":   [57.2, 55.7, 64.7, 61.1, 58.3, 59.1, 62.4, 61.5, 59.4],
    "NeuroLM": [60.3, 55.5, 58.4, 50.4, 48.4, 54.9, 60.8, 57.6, 58.7],
    "LaBraM":  [52.3, 55.2, 55.4, 53.4, 51.4, 53.0, 56.4, 52.4, 56.2],
    "CBraMod": [50.6, 54.3, 53.2, 51.8, 49.7, 55.4, 54.4, 54.1, 57.0],
}

# Track D LOSO (tmin=1.0, 4-40 Hz)
mi_trackD_loso = {
    "BENDR":   [50.4, 50.3, 50.1, 50.2, 50.7, 49.8, 50.6, 51.3, 51.0],
    "EEGPT":   [56.0, 54.1, 64.7, 61.9, 58.7, 58.7, 61.7, 61.2, 59.3],
    "NeuroLM": [51.1, 51.2, 54.1, 50.2, 50.2, 52.7, 52.6, 58.3, 59.1],
    "LaBraM":  [53.0, 54.5, 56.1, 54.0, 52.9, 55.4, 56.4, 56.1, 58.9],
    "CBraMod": [50.2, 52.0, 53.1, 51.6, 48.0, 54.4, 54.7, 53.9, 56.5],
}

# ──────────────────────────────────────────────────────────────────────
# DATA: SSVEP — Track A & B LOSO
# ──────────────────────────────────────────────────────────────────────
ssvep_datasets = ["Liu2020BETA", "Wang2016", "Nakanishi2015"]
ssvep_subjects = [70, 34, 9]
ssvep_channels = [64, 64, 8]
ssvep_classes = [40, 40, 12]
ssvep_chance = [2.5, 2.5, 8.3]

ssvep_trackA_loso = {
    "BENDR":   [4.4, 8.4, 9.1],
    "EEGPT":   [14.7, 23.6, 32.2],
    "NeuroLM": [5.6, 7.6, 25.3],
    "LaBraM":  [7.1, 13.1, 17.0],
    "CBraMod": [4.5, 6.1, 16.2],
}

ssvep_trackB_loso = {
    "BENDR":   [4.8, 7.0, 10.1],
    "EEGPT":   [14.8, 24.1, 34.6],
    "NeuroLM": [5.3, 9.4, 34.1],
    "LaBraM":  [9.6, 18.9, 30.2],
    "CBraMod": [4.9, 7.5, 23.4],
}

ssvep_trackA_kfold = {
    "BENDR":   [4.4, 8.2, 9.9],
    "EEGPT":   [14.7, 22.8, 32.2],
    "NeuroLM": [5.5, 7.4, 25.1],
    "LaBraM":  [7.1, 12.5, 18.3],
    "CBraMod": [4.2, 6.1, 16.4],
}

ssvep_trackB_kfold = {
    "BENDR":   [4.8, 6.7, 12.3],
    "EEGPT":   [14.7, 23.2, 33.5],
    "NeuroLM": [5.5, 9.0, 34.2],
    "LaBraM":  [9.1, 18.2, 28.2],
    "CBraMod": [4.6, 7.7, 23.6],
}

# ──────────────────────────────────────────────────────────────────────
# DATA: ERP — Track A LOSO (ROC-AUC)
# ──────────────────────────────────────────────────────────────────────
erp_datasets = ["BI2014a", "BI2015a", "BNCI2014_008", "BNCI2014_009"]
erp_subjects = [64, 43, 8, 10]
erp_channels = [16, 32, 8, 16]

erp_trackA_auc = {
    "EEGPT":   [61.0, 68.0, 53.6, 61.9],
    "NeuroLM": [58.3, 61.8, 63.7, 75.4],
    "LaBraM":  [54.7, 58.8, 59.1, 73.5],
    "CBraMod": [53.9, 56.7, 55.6, 64.7],
}

erp_trackA_f1 = {
    "EEGPT":   [49.3, 53.0, 47.4, 54.7],
    "NeuroLM": [46.0, 45.9, 51.2, 62.3],
    "LaBraM":  [45.5, 45.5, 46.9, 58.2],
    "CBraMod": [45.5, 45.5, 45.6, 49.0],
}


def build_mi_df():
    rows = []
    for track_name, track_data in [
        ("A", mi_trackA_loso), ("B", mi_trackB_loso),
        ("C", mi_trackC_loso), ("D", mi_trackD_loso),
    ]:
        for model in MODELS:
            for i, ds in enumerate(mi_datasets):
                rows.append({
                    "Paradigm": "MI", "Dataset": ds, "Model": model,
                    "Track": track_name, "Protocol": "LOSO",
                    "Accuracy": track_data[model][i],
                    "Subjects": mi_subjects[i], "Channels": mi_channels[i],
                })
    for track_name, track_data in [("A", mi_trackA_kfold), ("B", mi_trackB_kfold)]:
        for model in MODELS:
            for i, ds in enumerate(mi_datasets):
                rows.append({
                    "Paradigm": "MI", "Dataset": ds, "Model": model,
                    "Track": track_name, "Protocol": "5-Fold",
                    "Accuracy": track_data[model][i],
                    "Subjects": mi_subjects[i], "Channels": mi_channels[i],
                })
    return pd.DataFrame(rows)


def build_ssvep_df():
    rows = []
    for track_name, track_data in [("A", ssvep_trackA_loso), ("B", ssvep_trackB_loso)]:
        for model in MODELS:
            for i, ds in enumerate(ssvep_datasets):
                rows.append({
                    "Paradigm": "SSVEP", "Dataset": ds, "Model": model,
                    "Track": track_name, "Protocol": "LOSO",
                    "Accuracy": track_data[model][i],
                    "Subjects": ssvep_subjects[i], "Channels": ssvep_channels[i],
                    "Classes": ssvep_classes[i], "Chance": ssvep_chance[i],
                })
    for track_name, track_data in [("A", ssvep_trackA_kfold), ("B", ssvep_trackB_kfold)]:
        for model in MODELS:
            for i, ds in enumerate(ssvep_datasets):
                rows.append({
                    "Paradigm": "SSVEP", "Dataset": ds, "Model": model,
                    "Track": track_name, "Protocol": "5-Fold",
                    "Accuracy": track_data[model][i],
                    "Subjects": ssvep_subjects[i], "Channels": ssvep_channels[i],
                    "Classes": ssvep_classes[i], "Chance": ssvep_chance[i],
                })
    return pd.DataFrame(rows)


def build_erp_df():
    rows = []
    for model in ["EEGPT", "NeuroLM", "LaBraM", "CBraMod"]:
        for i, ds in enumerate(erp_datasets):
            rows.append({
                "Paradigm": "ERP", "Dataset": ds, "Model": model,
                "Track": "A", "Protocol": "LOSO",
                "ROC_AUC": erp_trackA_auc[model][i],
                "Macro_F1": erp_trackA_f1[model][i],
                "Subjects": erp_subjects[i], "Channels": erp_channels[i],
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# PLOT 1: Motor Imagery — Track A LOSO Heatmap
# ──────────────────────────────────────────────────────────────────────
def plot_mi_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    for ax, (title, data) in zip(axes, [
        ("Track A (FM-aligned)", mi_trackA_loso),
        ("Track B (4-40 Hz)", mi_trackB_loso),
    ]):
        mat = np.array([data[m] for m in MODELS])
        ds_labels = [f"{d}\n({s}s, {c}ch)" for d, s, c in zip(mi_datasets, mi_subjects, mi_channels)]
        im = sns.heatmap(
            mat, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn",
            xticklabels=ds_labels, yticklabels=MODELS,
            vmin=48, vmax=75, cbar_kws={"label": "Accuracy (%)"},
            linewidths=0.5,
        )
        ax.set_title(f"Motor Imagery LOSO — {title}", fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

    fig.suptitle("Motor Imagery: Frozen Linear Probe Accuracy (%) — LOSO\nBinary L/R classification, chance = 50%",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "01_mi_heatmap_loso.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 01_mi_heatmap_loso.png")


# ──────────────────────────────────────────────────────────────────────
# PLOT 2: MI Track A vs B Delta Heatmap
# ──────────────────────────────────────────────────────────────────────
def plot_mi_delta_heatmap():
    fig, ax = plt.subplots(figsize=(12, 5))
    delta = np.array([
        [mi_trackB_loso[m][i] - mi_trackA_loso[m][i] for i in range(len(mi_datasets))]
        for m in MODELS
    ])
    ds_labels = [f"{d}\n({s}s, {c}ch)" for d, s, c in zip(mi_datasets, mi_subjects, mi_channels)]
    sns.heatmap(
        delta, ax=ax, annot=True, fmt="+.1f", cmap="RdBu_r",
        xticklabels=ds_labels, yticklabels=MODELS,
        vmin=-20, vmax=5, center=0, cbar_kws={"label": "Delta (pp)"},
        linewidths=0.5,
    )
    ax.set_title("MI Preprocessing Impact: Track B - Track A (LOSO)\nNegative = Track A better (FM-aligned wideband)",
                 fontsize=14, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "02_mi_delta_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 02_mi_delta_heatmap.png")


# ──────────────────────────────────────────────────────────────────────
# PLOT 3: SSVEP Heatmap + Chance Lines
# ──────────────────────────────────────────────────────────────────────
def plot_ssvep_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (title, data) in zip(axes, [
        ("Track A (FM-aligned)", ssvep_trackA_loso),
        ("Track B (1-50 Hz)", ssvep_trackB_loso),
    ]):
        mat = np.array([data[m] for m in MODELS])
        ds_labels = [f"{d}\n({s}s, {c}ch, {cl}cls)" for d, s, c, cl in
                     zip(ssvep_datasets, ssvep_subjects, ssvep_channels, ssvep_classes)]
        sns.heatmap(
            mat, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn",
            xticklabels=ds_labels, yticklabels=MODELS,
            vmin=2, vmax=36, cbar_kws={"label": "Accuracy (%)"},
            linewidths=0.5,
        )
        ax.set_title(f"SSVEP LOSO — {title}", fontsize=13, fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

    fig.suptitle("SSVEP: Frozen Linear Probe Accuracy (%) — LOSO\nChance: 2.5% (40-class), 8.3% (12-class)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "03_ssvep_heatmap_loso.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 03_ssvep_heatmap_loso.png")


# ──────────────────────────────────────────────────────────────────────
# PLOT 4: ERP ROC-AUC Bar Chart
# ──────────────────────────────────────────────────────────────────────
def plot_erp_bars():
    fig, ax = plt.subplots(figsize=(12, 5))
    erp_models = ["EEGPT", "NeuroLM", "LaBraM", "CBraMod"]
    x = np.arange(len(erp_datasets))
    width = 0.18
    offsets = np.arange(len(erp_models)) - (len(erp_models) - 1) / 2

    for j, model in enumerate(erp_models):
        vals = erp_trackA_auc[model]
        bars = ax.bar(x + offsets[j] * width, vals, width, label=model,
                      color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="Chance (50%)")
    ax.set_xticks(x)
    ds_labels = [f"{d}\n({s}s, {c}ch)" for d, s, c in zip(erp_datasets, erp_subjects, erp_channels)]
    ax.set_xticklabels(ds_labels)
    ax.set_ylabel("ROC-AUC (%)")
    ax.set_title("ERP/P300: Frozen Linear Probe ROC-AUC (%) — Track A LOSO\nBinary Target vs NonTarget (5:1 imbalance). BENDR incompatible.",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_ylim(45, 82)
    fig.tight_layout()
    fig.savefig(OUT / "04_erp_auc_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 04_erp_auc_bars.png")


# ──────────────────────────────────────────────────────────────────────
# PLOT 5: Cross-Paradigm Model Ranking (Radar Chart)
# ──────────────────────────────────────────────────────────────────────
def plot_model_radar():
    categories = [
        "MI Track A\n(mean acc)",
        "MI Track C\n(cue-free)",
        "SSVEP Track A\n(mean acc)",
        "SSVEP Track B\n(1-50 Hz)",
        "ERP Track A\n(mean AUC)",
    ]
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    mi_a_means = {m: np.mean(mi_trackA_loso[m]) for m in MODELS}
    mi_c_means = {m: np.mean(mi_trackC_loso[m]) for m in MODELS}
    ssvep_a_means = {m: np.mean(ssvep_trackA_loso[m]) for m in MODELS}
    ssvep_b_means = {m: np.mean(ssvep_trackB_loso[m]) for m in MODELS}
    erp_means = {m: np.mean(erp_trackA_auc[m]) for m in ["EEGPT", "NeuroLM", "LaBraM", "CBraMod"]}
    erp_means["BENDR"] = 50.0  # incompatible

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for model in MODELS:
        values = [
            mi_a_means[model], mi_c_means[model],
            ssvep_a_means[model], ssvep_b_means[model],
            erp_means[model],
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=MODEL_COLORS[model])
        ax.fill(angles, values, alpha=0.05, color=MODEL_COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 75)
    ax.set_yticks([10, 20, 30, 40, 50, 60, 70])
    ax.set_yticklabels(["10%", "20%", "30%", "40%", "50%", "60%", "70%"], fontsize=8)
    ax.set_title("Cross-Paradigm Model Profile\nFrozen Linear Probe — LOSO", fontsize=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "05_model_radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 05_model_radar.png")


# ──────────────────────────────────────────────────────────────────────
# PLOT 6: MI All 4 Tracks Comparison (Box/Strip)
# ──────────────────────────────────────────────────────────────────────
def plot_mi_4tracks():
    df = build_mi_df()
    df_loso = df[df["Protocol"] == "LOSO"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = df_loso[df_loso["Model"] == model]
        sns.stripplot(data=sub, x="Track", y="Accuracy", hue="Track",
                      palette=TRACK_COLORS, ax=ax, size=7, jitter=0.15, alpha=0.7,
                      legend=False)
        means = sub.groupby("Track")["Accuracy"].mean()
        for track in means.index:
            ax.plot(list(TRACK_COLORS.keys()).index(track), means[track], "D",
                    color="black", markersize=10, zorder=5)
        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("")
        if ax != axes[0]:
            ax.set_ylabel("")

    fig.suptitle(
        "Motor Imagery LOSO: Track Comparison (9 datasets)\n"
        "A = FM-aligned, B = 4-40 Hz, C = A+tmin=1s, D = B+tmin=1s\n"
        "Dots = per-dataset accuracy, Diamonds = mean",
        fontsize=14, fontweight="bold", y=1.05,
    )
    fig.tight_layout()
    fig.savefig(OUT / "06_mi_4tracks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 06_mi_4tracks.png")


# ──────────────────────────────────────────────────────────────────────
# PLOT 7: SSVEP Track A vs B (Grouped Bars per Dataset)
# ──────────────────────────────────────────────────────────────────────
def plot_ssvep_grouped():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, (ds_idx, ds) in zip(axes, enumerate(ssvep_datasets)):
        x = np.arange(len(MODELS))
        w = 0.35
        a_vals = [ssvep_trackA_loso[m][ds_idx] for m in MODELS]
        b_vals = [ssvep_trackB_loso[m][ds_idx] for m in MODELS]
        bars_a = ax.bar(x - w / 2, a_vals, w, label="Track A", color=TRACK_COLORS["A"], edgecolor="white")
        bars_b = ax.bar(x + w / 2, b_vals, w, label="Track B", color=TRACK_COLORS["B"], edgecolor="white")
        for bar, v in zip(list(bars_a) + list(bars_b), a_vals + b_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(ssvep_chance[ds_idx], color="red", linestyle=":", linewidth=1,
                   label=f"Chance ({ssvep_chance[ds_idx]}%)")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=30, ha="right")
        ax.set_title(f"{ds}\n({ssvep_subjects[ds_idx]}s, {ssvep_channels[ds_idx]}ch, {ssvep_classes[ds_idx]}cls)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Accuracy (%)" if ds_idx == 0 else "")
        ax.legend(fontsize=8)

    fig.suptitle("SSVEP: Track A vs Track B — LOSO", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "07_ssvep_grouped_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 07_ssvep_grouped_bars.png")


# ──────────────────────────────────────────────────────────────────────
# PLOT 8: Cue-Onset Impact (Track A→C) per Model
# ──────────────────────────────────────────────────────────────────────
def plot_cue_onset_impact():
    fig, ax = plt.subplots(figsize=(10, 6))

    for m_idx, model in enumerate(MODELS):
        deltas = [mi_trackC_loso[model][i] - mi_trackA_loso[model][i] for i in range(len(mi_datasets))]
        x = np.full(len(deltas), m_idx) + np.random.uniform(-0.15, 0.15, len(deltas))
        ax.scatter(x, deltas, color=MODEL_COLORS[model], alpha=0.6, s=50, zorder=3)
        mean_d = np.mean(deltas)
        ax.plot([m_idx - 0.3, m_idx + 0.3], [mean_d, mean_d], color=MODEL_COLORS[model],
                linewidth=3, zorder=4)
        ax.text(m_idx + 0.35, mean_d, f"{mean_d:+.1f}", fontsize=10, fontweight="bold",
                va="center", color=MODEL_COLORS[model])

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, fontsize=12)
    ax.set_ylabel("Accuracy Delta (pp): Track C - Track A")
    ax.set_title("Cue-Onset Artefact Impact (tmin=0 vs tmin=1s)\nNegative = model exploited cue-onset information",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "08_cue_onset_impact.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 08_cue_onset_impact.png")


# ──────────────────────────────────────────────────────────────────────
# PLOT 9: Grand Summary — Best Model per Paradigm
# ──────────────────────────────────────────────────────────────────────
def plot_grand_summary():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # MI: Track C LOSO (cue-free, honest)
    ax = axes[0]
    for model in MODELS:
        ax.bar(model, np.mean(mi_trackC_loso[model]), color=MODEL_COLORS[model],
               edgecolor="white", linewidth=0.5)
        ax.text(MODELS.index(model), np.mean(mi_trackC_loso[model]) + 0.3,
                f"{np.mean(mi_trackC_loso[model]):.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(45, 68)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Motor Imagery\nTrack C LOSO mean\n(9 datasets, cue-free)", fontsize=13, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # SSVEP: Track B LOSO (best preprocessing)
    ax = axes[1]
    ssvep_norm = {}
    for model in MODELS:
        norm_scores = [(ssvep_trackB_loso[model][i] / ssvep_chance[i]) for i in range(3)]
        ssvep_norm[model] = np.mean(norm_scores)
    for model in MODELS:
        ax.bar(model, ssvep_norm[model], color=MODEL_COLORS[model],
               edgecolor="white", linewidth=0.5)
        ax.text(MODELS.index(model), ssvep_norm[model] + 0.05,
                f"{ssvep_norm[model]:.1f}x", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_ylabel("Accuracy / Chance")
    ax.set_title("SSVEP\nTrack B LOSO normalized\n(3 datasets)", fontsize=13, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # ERP: Track A LOSO AUC
    ax = axes[2]
    erp_models_plot = ["EEGPT", "NeuroLM", "LaBraM", "CBraMod"]
    for model in erp_models_plot:
        ax.bar(model, np.mean(erp_trackA_auc[model]), color=MODEL_COLORS[model],
               edgecolor="white", linewidth=0.5)
        ax.text(erp_models_plot.index(model), np.mean(erp_trackA_auc[model]) + 0.3,
                f"{np.mean(erp_trackA_auc[model]):.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.bar("BENDR", 50, color=MODEL_COLORS["BENDR"], edgecolor="white",
           linewidth=0.5, alpha=0.3, hatch="//")
    ax.text(4, 51, "N/A", ha="center", va="bottom", fontsize=10, color="gray")
    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(45, 78)
    ax.set_ylabel("ROC-AUC (%)")
    ax.set_title("ERP/P300\nTrack A LOSO mean AUC\n(4 datasets)", fontsize=13, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("EEG Foundation Model Benchmark — Grand Summary\nFrozen Linear Probe Evaluation",
                 fontsize=16, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "09_grand_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 09_grand_summary.png")


# ──────────────────────────────────────────────────────────────────────
# PLOT 10: Preprocessing Sensitivity — Combined MI + SSVEP
# ──────────────────────────────────────────────────────────────────────
def plot_preproc_sensitivity():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MI: mean delta B-A
    ax = axes[0]
    mi_deltas = {m: np.mean([mi_trackB_loso[m][i] - mi_trackA_loso[m][i]
                             for i in range(len(mi_datasets))]) for m in MODELS}
    colors = [MODEL_COLORS[m] for m in MODELS]
    bars = ax.barh(MODELS, [mi_deltas[m] for m in MODELS], color=colors, edgecolor="white")
    for bar, m in zip(bars, MODELS):
        v = mi_deltas[m]
        ax.text(v + (0.2 if v >= 0 else -0.2), bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f} pp", ha="left" if v >= 0 else "right", va="center",
                fontsize=11, fontweight="bold")
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Mean delta (pp): Track B - Track A")
    ax.set_title("MI: Preprocessing Sensitivity\n(4-40 Hz vs FM-aligned)", fontsize=13, fontweight="bold")
    ax.set_xlim(-6, 3)

    # SSVEP: mean delta B-A
    ax = axes[1]
    ssvep_deltas = {m: np.mean([ssvep_trackB_loso[m][i] - ssvep_trackA_loso[m][i]
                                for i in range(len(ssvep_datasets))]) for m in MODELS}
    bars = ax.barh(MODELS, [ssvep_deltas[m] for m in MODELS], color=colors, edgecolor="white")
    for bar, m in zip(bars, MODELS):
        v = ssvep_deltas[m]
        ax.text(v + (0.2 if v >= 0 else -0.2), bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f} pp", ha="left" if v >= 0 else "right", va="center",
                fontsize=11, fontweight="bold")
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Mean delta (pp): Track B - Track A")
    ax.set_title("SSVEP: Preprocessing Sensitivity\n(1-50 Hz vs FM-aligned)", fontsize=13, fontweight="bold")
    ax.set_xlim(-2, 9)

    fig.suptitle("Preprocessing Band Impact on Foundation Models",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "10_preproc_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 10_preproc_sensitivity.png")


# ──────────────────────────────────────────────────────────────────────
# PLOT 11: Per-Dataset MI Bar Chart (All models, Track C — honest)
# ──────────────────────────────────────────────────────────────────────
def plot_mi_per_dataset():
    fig, axes = plt.subplots(3, 3, figsize=(20, 16), sharey=True)
    axes_flat = axes.flatten()

    for idx, ds in enumerate(mi_datasets):
        ax = axes_flat[idx]
        vals = [mi_trackC_loso[m][idx] for m in MODELS]
        bars = ax.bar(MODELS, vals, color=[MODEL_COLORS[m] for m in MODELS],
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"{ds} ({mi_subjects[idx]}s, {mi_channels[idx]}ch)", fontsize=11, fontweight="bold")
        ax.set_ylim(45, 70)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        if idx % 3 == 0:
            ax.set_ylabel("Accuracy (%)")

    fig.suptitle("Motor Imagery Track C LOSO: Per-Dataset Breakdown (cue-free, tmin=1s)\nFrozen linear probe, binary L/R, chance = 50%",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "11_mi_per_dataset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 11_mi_per_dataset.png")


# ──────────────────────────────────────────────────────────────────────
# PLOT 12: LOSO vs 5-Fold Consistency
# ──────────────────────────────────────────────────────────────────────
def plot_loso_vs_kfold():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MI
    ax = axes[0]
    for model in MODELS:
        loso_vals = mi_trackA_loso[model]
        kfold_vals = mi_trackA_kfold[model]
        ax.scatter(loso_vals, kfold_vals, color=MODEL_COLORS[model], s=40, alpha=0.7, label=model)
    lims = [45, 78]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("LOSO Accuracy (%)")
    ax.set_ylabel("5-Fold Accuracy (%)")
    ax.set_title("MI Track A: LOSO vs 5-Fold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")

    # SSVEP
    ax = axes[1]
    for model in MODELS:
        loso_vals = ssvep_trackA_loso[model]
        kfold_vals = ssvep_trackA_kfold[model]
        ax.scatter(loso_vals, kfold_vals, color=MODEL_COLORS[model], s=60, alpha=0.7, label=model)
    lims = [2, 36]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("LOSO Accuracy (%)")
    ax.set_ylabel("5-Fold Accuracy (%)")
    ax.set_title("SSVEP Track A: LOSO vs 5-Fold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")

    fig.suptitle("Evaluation Protocol Consistency: LOSO vs 5-Fold",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "12_loso_vs_kfold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 12_loso_vs_kfold.png")


# ──────────────────────────────────────────────────────────────────────
# GENERATE ALL TABLES (CSV)
# ──────────────────────────────────────────────────────────────────────
def generate_tables():
    # Table 1: MI Track A & C LOSO
    rows = []
    for i, ds in enumerate(mi_datasets):
        for m in MODELS:
            rows.append({
                "Dataset": ds, "Subjects": mi_subjects[i], "Channels": mi_channels[i],
                "Model": m,
                "Track_A_LOSO": mi_trackA_loso[m][i],
                "Track_B_LOSO": mi_trackB_loso[m][i],
                "Track_C_LOSO": mi_trackC_loso[m][i],
                "Track_D_LOSO": mi_trackD_loso[m][i],
                "Delta_BA": mi_trackB_loso[m][i] - mi_trackA_loso[m][i],
                "Delta_CA": mi_trackC_loso[m][i] - mi_trackA_loso[m][i],
            })
    pd.DataFrame(rows).to_csv(OUT / "table_mi_all_tracks.csv", index=False)

    # Table 2: SSVEP
    rows = []
    for i, ds in enumerate(ssvep_datasets):
        for m in MODELS:
            rows.append({
                "Dataset": ds, "Subjects": ssvep_subjects[i], "Channels": ssvep_channels[i],
                "Classes": ssvep_classes[i], "Chance": ssvep_chance[i],
                "Model": m,
                "Track_A_LOSO": ssvep_trackA_loso[m][i],
                "Track_B_LOSO": ssvep_trackB_loso[m][i],
                "Delta_BA": ssvep_trackB_loso[m][i] - ssvep_trackA_loso[m][i],
                "Norm_A": ssvep_trackA_loso[m][i] / ssvep_chance[i],
                "Norm_B": ssvep_trackB_loso[m][i] / ssvep_chance[i],
            })
    pd.DataFrame(rows).to_csv(OUT / "table_ssvep.csv", index=False)

    # Table 3: ERP
    rows = []
    for i, ds in enumerate(erp_datasets):
        for m in ["EEGPT", "NeuroLM", "LaBraM", "CBraMod"]:
            rows.append({
                "Dataset": ds, "Subjects": erp_subjects[i], "Channels": erp_channels[i],
                "Model": m,
                "ROC_AUC": erp_trackA_auc[m][i],
                "Macro_F1": erp_trackA_f1[m][i],
            })
    pd.DataFrame(rows).to_csv(OUT / "table_erp.csv", index=False)

    # Table 4: Grand summary
    summary_rows = []
    for m in MODELS:
        row = {"Model": m}
        row["MI_TrackA_mean"] = np.mean(mi_trackA_loso[m])
        row["MI_TrackC_mean"] = np.mean(mi_trackC_loso[m])
        row["MI_cue_impact"] = row["MI_TrackC_mean"] - row["MI_TrackA_mean"]
        row["MI_preproc_impact"] = np.mean([mi_trackB_loso[m][i] - mi_trackA_loso[m][i]
                                            for i in range(len(mi_datasets))])
        row["SSVEP_TrackA_mean"] = np.mean(ssvep_trackA_loso[m])
        row["SSVEP_TrackB_mean"] = np.mean(ssvep_trackB_loso[m])
        row["SSVEP_preproc_impact"] = row["SSVEP_TrackB_mean"] - row["SSVEP_TrackA_mean"]
        if m in erp_trackA_auc:
            row["ERP_AUC_mean"] = np.mean(erp_trackA_auc[m])
        else:
            row["ERP_AUC_mean"] = np.nan
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT / "table_grand_summary.csv", index=False)

    print(f"  Saved 4 CSV tables to {OUT}/")


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating EEG Benchmarks Summary...")
    print()

    print("[1/12] MI heatmap (LOSO)...")
    plot_mi_heatmap()

    print("[2/12] MI delta heatmap...")
    plot_mi_delta_heatmap()

    print("[3/12] SSVEP heatmap...")
    plot_ssvep_heatmap()

    print("[4/12] ERP ROC-AUC bars...")
    plot_erp_bars()

    print("[5/12] Cross-paradigm radar...")
    plot_model_radar()

    print("[6/12] MI 4-track comparison...")
    plot_mi_4tracks()

    print("[7/12] SSVEP grouped bars...")
    plot_ssvep_grouped()

    print("[8/12] Cue-onset impact...")
    plot_cue_onset_impact()

    print("[9/12] Grand summary...")
    plot_grand_summary()

    print("[10/12] Preprocessing sensitivity...")
    plot_preproc_sensitivity()

    print("[11/12] MI per-dataset breakdown...")
    plot_mi_per_dataset()

    print("[12/12] LOSO vs 5-Fold consistency...")
    plot_loso_vs_kfold()

    print()
    print("Generating tables...")
    generate_tables()

    print()
    print(f"All outputs in: {OUT}")
    print("Done!")
