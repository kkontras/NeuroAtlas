"""Paper figures: seizure-detection foundation-model rankings across datasets.

Fig 1 (main): Slopegraph. Two panels (AUROC, Sens@1/h), lines = models,
x-axis = datasets ordered by seizure count (Siena → CHB-MIT → TUSZ → Epilepsiae).
Crossing lines = rankings swap between datasets = paper story.

Fig 2 (supplementary): Per-fold strip/box plots for the CV datasets
(CHB-MIT, Siena) showing that fold-level variance is large relative to
cross-model differences.

Outputs PDFs under artifacts/benchmarks/figures/.
"""
from __future__ import annotations

import json
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------


MODEL_DISPLAY = {
    "biot_pretrained": "BIOT",
    "biot": "BIOT",
    "bendr_pretrained": "BENDR",
    "bendr": "BENDR",
    "cbramod_pretrained": "CBraMod",
    "cbramod": "CBraMod",
    "cbramod_random_init": "CBraMod (rand)",
    "core_sleep_shhs_fold0": "CoRe-Sleep",
    "core_sleep": "CoRe-Sleep",
    "eegpt_pretrained": "EEGPT",
    "eegpt": "EEGPT",
    "labram_pretrained": "LaBraM",
    "labram": "LaBraM",
    "neurolm_vq_pretrained": "NeuroLM",
    "neurolm": "NeuroLM",
    "sleepfm_pretrained": "SleepFM",
    "sleepfm": "SleepFM",
    "reve_pretrained": "REVE",
    "reve": "REVE",
    "tfc_pretrained_sleepEEG": "TF-C",
    "tfc": "TF-C",
}


def _canonical_family(slug: str) -> str:
    """Return a canonical model-family key (so TUSZ's 'biot' matches CHB-MIT's 'biot_pretrained')."""
    s = slug.lower()
    # Keep random_init as its own family
    if s.endswith("_random_init"):
        return s
    # Strip suffixes iteratively (some slugs have multiple, e.g. tfc_pretrained_sleepEEG)
    suffixes = ("_pretrained", "_sleepeeg", "_vq", "_shhs_fold0")
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if s.endswith(suffix):
                s = s[: -len(suffix)]
                changed = True
                break
    return s


# Pretraining-domain grouping → Okabe-Ito colour palette (colour-blind safe)
PRETRAINING_DOMAIN = {
    # Epilepsy-pretrained
    "biot": ("epilepsy", "#E69F00"),     # orange
    "bendr": ("epilepsy", "#D55E00"),    # vermilion
    "eegpt": ("epilepsy", "#CC79A7"),    # pink
    # Sleep-pretrained
    "sleepfm": ("sleep", "#0072B2"),     # blue
    "tfc": ("sleep", "#56B4E9"),         # sky blue
    "core_sleep": ("sleep", "#003F5C"),  # dark blue
    # General EEG pretraining
    "cbramod": ("general", "#009E73"),           # bluish green
    "cbramod_random_init": ("general", "#555555"),  # grey (baseline)
    "labram": ("general", "#117733"),            # dark green
    "neurolm": ("general", "#88CCEE"),           # teal-ish
    "reve": ("general", "#AA4499"),              # purple
}


# Dataset metadata (ordered by seizure count — small → large)
DATASETS = [
    {"slug": "siena",   "label": "Siena\n(14 subj, 47 sz)",           "n_seizures": 47,   "cv": True},
    {"slug": "chbmit",  "label": "CHB-MIT\n(23 subj, 198 sz)",        "n_seizures": 198,  "cv": True},
    {"slug": "tusz",    "label": "TUSZ v2.0.3\n(675 subj, 4029 sz)",  "n_seizures": 4029, "cv": False},
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_chbmit_siena(dataset: str) -> Dict[str, Dict[str, Any]]:
    """Returns {family: {metric_mean, metric_std, fold_values_per_metric}}."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in sorted(glob.glob(f"artifacts/benchmarks/chbmit_siena/embeddings/{dataset}/*/results.json")):
        with open(p) as f:
            r = json.load(f)
        if "binary" not in r:
            continue
        fam = _canonical_family(r["model"])
        s = r["binary"]["summary"]
        folds = r["binary"]["folds"]
        out[fam] = {
            "slug": r["model"],
            "auroc": s["test_auroc"]["mean"],
            "auroc_std": s["test_auroc"]["std"],
            "auroc_folds": [f["test_auroc"] for f in folds],
            "sens1": s.get("test_sens_at_fpr_h_1_0", {}).get("mean"),
            "sens1_std": s.get("test_sens_at_fpr_h_1_0", {}).get("std", 0),
            "sens1_folds": [f.get("test_sens_at_fpr_h_1_0") for f in folds],
            "sens01": s.get("test_sens_at_fpr_h_0_1", {}).get("mean"),
            "sens01_std": s.get("test_sens_at_fpr_h_0_1", {}).get("std", 0),
            "sens01_folds": [f.get("test_sens_at_fpr_h_0_1") for f in folds],
        }
    return out


def _load_tusz() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in sorted(glob.glob("artifacts/benchmarks/tusz/*/results.json")):
        if "kfold_results" in p or "_old" in p:
            continue
        with open(p) as f:
            r = json.load(f)
        for res in r:
            if not res.get("ok"):
                continue
            m = res.get("metrics", {})
            if "auroc" not in m:
                continue
            fam = _canonical_family(res["checkpoint_id"])
            out[fam] = {
                "slug": res["checkpoint_id"],
                "auroc": m["auroc"],
                "auroc_std": 0,
                "auroc_folds": [m["auroc"]],
                "sens1": m.get("sensitivity_at_fpr_h_1_0"),
                "sens1_std": 0,
                "sens1_folds": [m.get("sensitivity_at_fpr_h_1_0")],
                "sens01": m.get("sensitivity_at_fpr_h_0_1"),
                "sens01_std": 0,
                "sens01_folds": [m.get("sensitivity_at_fpr_h_0_1")],
            }
    return out


def _load_all() -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {
        "siena": _load_chbmit_siena("siena"),
        "chbmit": _load_chbmit_siena("chbmit"),
        "tusz": _load_tusz(),
    }


# ---------------------------------------------------------------------------
# Fig 1: Slopegraph
# ---------------------------------------------------------------------------


def _pick_models_for_slopegraph(all_data: Dict[str, Dict]) -> List[str]:
    """Models that appear in at least 2 of 3 datasets (so they have a line)."""
    from collections import Counter
    counts = Counter()
    for ds, models in all_data.items():
        for fam in models:
            counts[fam] += 1
    return [fam for fam, n in counts.items() if n >= 2 and fam in PRETRAINING_DOMAIN]


def _stagger_label_ys(ys: List[float], min_gap: float) -> List[float]:
    """Offset labels vertically so they don't overlap.

    Sorts by y desc, walks from top down, pushing each label down if it's
    too close to the previous.
    """
    idx_sorted = sorted(range(len(ys)), key=lambda i: ys[i], reverse=True)
    placed = list(ys)  # mutable copy
    for rank, i in enumerate(idx_sorted):
        if rank == 0:
            continue
        prev_i = idx_sorted[rank - 1]
        if placed[prev_i] - placed[i] < min_gap:
            placed[i] = placed[prev_i] - min_gap
    return placed


def _plot_slopegraph(
    all_data: Dict[str, Dict],
    metric: str,
    ax: plt.Axes,
    panel_label: str,
    ylabel: str,
    ylim: Optional[Tuple[float, float]] = None,
    show_clinical: bool = False,
) -> None:
    """Draw one slopegraph panel."""
    xs = list(range(len(DATASETS)))
    ax.set_xlim(-0.3, len(DATASETS) - 1 + 0.75)  # room on the right for labels

    models = _pick_models_for_slopegraph(all_data)

    # Collect (fam, values, std) sorted by TUSZ-value (rightmost) so labels don't overlap
    model_values: List[Tuple[str, List[Optional[float]], List[float]]] = []
    for fam in models:
        vals: List[Optional[float]] = []
        stds: List[float] = []
        for d in DATASETS:
            entry = all_data[d["slug"]].get(fam)
            if entry is None:
                vals.append(None)
                stds.append(0)
            else:
                v = entry.get(metric)
                s = entry.get(f"{metric}_std", 0) or 0
                vals.append(v)
                stds.append(s)
        model_values.append((fam, vals, stds))

    # Find best-per-dataset (for star markers)
    best_per_ds: Dict[int, str] = {}
    for i in range(len(DATASETS)):
        best_val, best_fam = None, None
        for fam, vals, _ in model_values:
            v = vals[i]
            if v is None:
                continue
            if best_val is None or v > best_val:
                best_val = v
                best_fam = fam
        if best_fam is not None:
            best_per_ds[i] = best_fam

    # Draw lines and markers
    for fam, vals, stds in model_values:
        domain, colour = PRETRAINING_DOMAIN.get(fam, ("other", "#999"))
        pts_x = [x for x, v in zip(xs, vals) if v is not None]
        pts_y = [v for v in vals if v is not None]
        pts_std = [s for v, s in zip(vals, stds) if v is not None]
        pts_ds_idx = [j for j, v in enumerate(vals) if v is not None]

        ax.plot(pts_x, pts_y, "-", color=colour, linewidth=1.5, alpha=0.85, zorder=2)
        for x, y, s in zip(pts_x, pts_y, pts_std):
            if s > 0:
                ax.plot([x, x], [y - s, y + s], "-", color=colour, alpha=0.30, linewidth=5, zorder=1)
        for x, y, ds_idx in zip(pts_x, pts_y, pts_ds_idx):
            if best_per_ds.get(ds_idx) == fam:
                ax.plot(x, y, marker="*", color=colour, markersize=15,
                        markeredgecolor="black", markeredgewidth=1.0, zorder=5)
            else:
                ax.plot(x, y, marker="o", color=colour, markersize=6,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=4)

    # Label placement on the RIGHT (TUSZ column)
    # Collect (fam, rightmost_value, label_text, colour)
    rightmost_ds_idx = len(DATASETS) - 1
    label_items = []
    for fam, vals, stds in model_values:
        if vals[rightmost_ds_idx] is None:
            # Fall back to the last non-None position
            for i_back in range(rightmost_ds_idx - 1, -1, -1):
                if vals[i_back] is not None:
                    label_items.append({
                        "fam": fam,
                        "x": i_back,
                        "y_data": vals[i_back],
                        "colour": PRETRAINING_DOMAIN.get(fam, ("other", "#999"))[1],
                        "text": MODEL_DISPLAY.get(all_data[DATASETS[i_back]["slug"]].get(fam, {}).get("slug", fam), fam),
                    })
                    break
            continue
        slug = all_data[DATASETS[rightmost_ds_idx]["slug"]][fam]["slug"]
        label_items.append({
            "fam": fam,
            "x": rightmost_ds_idx,
            "y_data": vals[rightmost_ds_idx],
            "colour": PRETRAINING_DOMAIN.get(fam, ("other", "#999"))[1],
            "text": MODEL_DISPLAY.get(slug, fam),
        })

    # Compute min_gap based on ylim
    lo, hi = ylim if ylim else (0, 1)
    min_gap = (hi - lo) * 0.035  # ~3.5% of axis range
    ys_staggered = _stagger_label_ys([it["y_data"] for it in label_items], min_gap)

    for it, y_lbl in zip(label_items, ys_staggered):
        # Draw tiny leader line if label was offset
        if abs(y_lbl - it["y_data"]) > min_gap * 0.2:
            ax.plot([it["x"] + 0.05, it["x"] + 0.20], [it["y_data"], y_lbl],
                    "-", color=it["colour"], alpha=0.4, linewidth=0.6, zorder=3)
        ax.annotate(
            it["text"],
            xy=(it["x"] + 0.22, y_lbl),
            va="center", ha="left",
            fontsize=8.5, color=it["colour"], fontweight="bold",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([d["label"] for d in DATASETS], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(-0.14, 1.02, panel_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")

    if show_clinical:
        # Two literature-based reference bands:
        #   75%: real-time inpatient monitoring (Persyst PSS ~80%@0.5/h;
        #        Baumgartner 2018 review: 75-93% across commercial systems).
        #   50%: below this, >half of seizures missed — considered not useful
        #        even as a triage tool.
        # Reference: Baumgartner & Koren 2018 (Neuropediatrics), Siddiqui et
        # al. 2020 (Comput Biol Med), Ulate-Campos et al. 2016 (Seizure).
        lo_y, hi_y = ax.get_ylim()
        # "Clinically useful" band above 75%
        ax.axhspan(0.75, max(hi_y, 0.85), color="#2ca02c", alpha=0.10, zorder=0)
        ax.axhline(0.75, color="#2ca02c", linestyle="--", linewidth=1.1,
                   alpha=0.8, zorder=0)
        # "Minimum useful" band 50-75%
        ax.axhspan(0.50, 0.75, color="#f4a261", alpha=0.08, zorder=0)
        ax.axhline(0.50, color="#f4a261", linestyle=":", linewidth=1.0,
                   alpha=0.7, zorder=0)
        ax.text(-0.22, 0.76, "clinically useful\n(≥75%, Baumgartner 2018)",
                fontsize=7.2, color="#2ca02c", va="bottom", ha="right",
                fontweight="bold")
        ax.text(-0.22, 0.48, "minimum useful\n(≥50%)",
                fontsize=7.2, color="#b5651d", va="top", ha="right")


def plot_fig1(all_data: Dict[str, Dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), dpi=150)
    _plot_slopegraph(
        all_data, "auroc", axes[0],
        panel_label="A", ylabel="AUROC (threshold-free)",
        ylim=(0.45, 0.92),
    )
    _plot_slopegraph(
        all_data, "sens1", axes[1],
        panel_label="B",
        ylabel="Sens@1/h (inpatient operating point)",
        ylim=(-0.02, 0.90),
        show_clinical=True,
    )
    _plot_slopegraph(
        all_data, "sens01", axes[2],
        panel_label="C",
        ylabel="Sens@0.1/h (ambulatory operating point)",
        ylim=(-0.02, 0.90),
        show_clinical=True,
    )

    # Legend: one swatch per pretraining domain, on bottom
    domain_colours = {}
    for fam, (dom, col) in PRETRAINING_DOMAIN.items():
        if dom not in domain_colours:
            domain_colours[dom] = col

    legend_labels = [
        ("epilepsy", "Epilepsy-pretrained (BIOT, BENDR, EEGPT)"),
        ("sleep", "Sleep-pretrained (SleepFM, TF-C)"),
        ("general", "General EEG (CBraMod, LaBraM, NeuroLM, REVE)"),
    ]
    handles = []
    for dom, label in legend_labels:
        col = next((c for f, (d, c) in PRETRAINING_DOMAIN.items() if d == dom), "#999")
        handles.append(plt.Line2D([0], [0], color=col, marker="o", linewidth=2, label=label))
    handles.append(plt.Line2D([0], [0], marker="*", color="#444", linestyle="None",
                              markersize=12, markeredgecolor="black", label="Best per dataset"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig1_rankings.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig1_rankings.svg", bbox_inches="tight")
    fig.savefig(out_dir / "fig1_rankings.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out_dir}/fig1_rankings.{{pdf,svg,png}}")


# ---------------------------------------------------------------------------
# Fig 2: variance strip plot
# ---------------------------------------------------------------------------


def _plot_variance_panel(
    ax: plt.Axes,
    data: Dict[str, Dict[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
    clinical_threshold: Optional[float] = None,
    single_point: bool = False,
) -> None:
    # Sort models by mean metric (descending)
    items = [(fam, d) for fam, d in data.items() if fam in PRETRAINING_DOMAIN]
    items.sort(key=lambda x: x[1].get(metric, 0) or 0, reverse=True)

    x_positions = list(range(len(items)))
    for i, (fam, d) in enumerate(items):
        _, col = PRETRAINING_DOMAIN[fam]
        folds = d.get(f"{metric}_folds", [])
        folds = [f for f in folds if f is not None]
        if not folds:
            continue
        if single_point:
            # TUSZ: one dot, no jitter, no error bar
            ax.scatter([i], folds[:1], color=col, alpha=0.85, s=70, zorder=3,
                       edgecolor="black", linewidth=0.6)
        else:
            jitter = np.random.RandomState(42 + i).uniform(-0.1, 0.1, len(folds))
            ax.scatter([i + j for j in jitter], folds, color=col, alpha=0.55, s=22, zorder=3)
            # Mean ± std overlay
            mean = float(np.mean(folds))
            std = float(np.std(folds))
            ax.plot([i - 0.25, i + 0.25], [mean, mean], "-", color="black",
                    linewidth=1.8, zorder=4)
            ax.plot([i, i], [mean - std, mean + std], "-", color="black",
                    linewidth=1.0, alpha=0.7, zorder=4)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [MODEL_DISPLAY.get(items[i][1].get("slug", items[i][0]), items[i][0])
         for i in range(len(items))],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Literature-based clinical reference bands (Sens@1/h only)
    # See _plot_slopegraph comments for references.
    if clinical_threshold is not None:
        lo_y, hi_y = ax.get_ylim()
        ax.axhspan(0.75, max(hi_y, 0.80), color="#2ca02c", alpha=0.08, zorder=0)
        ax.axhline(0.75, color="#2ca02c", linestyle="--",
                   linewidth=0.9, alpha=0.7, zorder=0)
        ax.axhspan(0.50, 0.75, color="#f4a261", alpha=0.06, zorder=0)
        ax.axhline(0.50, color="#f4a261", linestyle=":",
                   linewidth=0.8, alpha=0.6, zorder=0)


def plot_fig2(all_data: Dict[str, Dict], out_dir: Path) -> None:
    row_specs = [
        ("chbmit", "CHB-MIT (5-fold CV)", False),
        ("siena",  "Siena (5-fold CV)",   False),
        ("tusz",   "TUSZ v2.0.3 (official test split)", True),
    ]
    fig, axes = plt.subplots(len(row_specs), 3, figsize=(15, 9), dpi=150)

    for row, (ds, title_prefix, single) in enumerate(row_specs):
        _plot_variance_panel(
            axes[row][0], all_data[ds], "auroc",
            title=f"{title_prefix} — AUROC",
            ylabel="AUROC",
            clinical_threshold=None,
            single_point=single,
        )
        _plot_variance_panel(
            axes[row][1], all_data[ds], "sens1",
            title=f"{title_prefix} — Sens@1/h (inpatient)",
            ylabel="Sens@1/h",
            clinical_threshold=0.75,
            single_point=single,
        )
        axes[row][1].set_ylim(-0.02, 0.90)
        _plot_variance_panel(
            axes[row][2], all_data[ds], "sens01",
            title=f"{title_prefix} — Sens@0.1/h (ambulatory)",
            ylabel="Sens@0.1/h",
            clinical_threshold=0.75,
            single_point=single,
        )
        axes[row][2].set_ylim(-0.02, 0.90)
        # Label clinical threshold bands on first row only to avoid clutter
        if row == 0:
            for col in (1, 2):
                axes[row][col].text(0.02, 0.76, "≥75% clinically useful (Baumgartner 2018)",
                                    transform=axes[row][col].get_yaxis_transform(),
                                    fontsize=7, color="#2ca02c", va="bottom",
                                    fontweight="bold")
                axes[row][col].text(0.02, 0.51, "≥50% minimum useful",
                                    transform=axes[row][col].get_yaxis_transform(),
                                    fontsize=7, color="#b5651d", va="bottom")

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig2_variance.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig2_variance.svg", bbox_inches="tight")
    fig.savefig(out_dir / "fig2_variance.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {out_dir}/fig2_variance.{{pdf,svg,png}}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,  # embed TrueType (not Type 3) for camera-ready PDFs
        "ps.fonttype": 42,
    })

    all_data = _load_all()

    # Quick sanity summary
    for ds, models in all_data.items():
        print(f"{ds}: {len(models)} models → {sorted(models.keys())}")

    out_dir = Path("artifacts/benchmarks/figures")
    plot_fig1(all_data, out_dir)
    plot_fig2(all_data, out_dir)


if __name__ == "__main__":
    main()
