"""Generate benchmark-result figures across every dataset × probe combination.

Reads per-fold results from ``artifacts/benchmarks/<slug>/.../results.json``
and writes PNG plots under ``plots/``. Coverage:

    sleep staging (linear_probe_eval):
        mass, dcsm, wsc, ucddb, dod, isruc  (κ / macro-F1 / F1-per-stage /
        spider)
    arousal detection (arousal_detection):
        mass_arousal                         (threshold sweep + 3s bar panel)
    respiratory-event detection (respiratory_event_detection):
        ucddb_respevt                        (per-event-type sweeps + 10s bars)
    patient classification (patient_classification_eval):
        dod_osa                              (AUROC / AUPRC / MCC / macro-F1)
    per-scorer inter-rater κ (linear_probe_eval against each expert):
        dod/scorer_1..5, isruc/scorer_1..2   (one figure per dataset)

Datasets whose results.json has zero successful rows (``ok == False`` for all
entries) are skipped silently so the script stays green as new probes come
online. Run:

    PYTHONPATH=src python reproduction/figures/plot_benchmark_results.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "plots"
OUT.mkdir(exist_ok=True, parents=True)
TABLE_OUT = REPO / "tables"
TABLE_OUT.mkdir(exist_ok=True, parents=True)


SLEEP_DATASETS = ["mass", "dcsm", "wsc", "ucddb", "dod", "isruc"]
DATASET_COLORS = {
    "mass":  "#4C72B0",
    "dcsm":  "#DD8452",
    "wsc":   "#55A467",
    "ucddb": "#C44E52",
    "dod":   "#8172B2",
    "isruc": "#CCB974",
}

# Per-dataset results file for the default sleep-staging probe.  Most datasets
# write directly under ``<slug>/linear/``; DOD/ISRUC add an extra probe-type
# folder (``sleep_stage``) because they also run per-scorer probes.
DATASET_RESULTS_PATH = {
    "mass":  "mass/linear/results.json",
    "dcsm":  "dcsm/linear/results.json",
    "wsc":   "wsc/linear/results.json",
    "ucddb": "ucddb/linear/results.json",
    "dod":   "dod/sleep_stage/linear/results.json",
    "isruc": "isruc/sleep_stage/linear/results.json",
}

# Per-dataset expert-scorer probes.  Empty for datasets that have no
# per-scorer ground truth.
DATASET_SCORER_SLUGS = {
    "dod":   [f"scorer_{i}" for i in range(1, 6)],
    "isruc": [f"scorer_{i}" for i in range(1, 3)],
}

MODEL_ORDER = [
    "reve_pretrained",
    "neurolm_vq_pretrained",
    "neurorvq_eeg_pretrained",
    "sleepfm_pretrained",
    "cbramod_pretrained",
    "labram_pretrained",
    "biot_pretrained",
    "steegformer_small",
    "steegformer_base",
    "neurogpt_pretrained",
    "eegpt_pretrained",
]
MODEL_LABELS = {
    "reve_pretrained": "reve",
    "neurolm_vq_pretrained": "neurolm",
    "neurorvq_eeg_pretrained": "neurorvq",
    "sleepfm_pretrained": "sleepfm",
    "cbramod_pretrained": "cbramod",
    "cbramod_random_init": "cbramod_rand",
    "labram_pretrained": "labram",
    "biot_pretrained": "biot",
    "steegformer_small": "steegformer",
    "steegformer_base": "steegformer-b",
    "neurogpt_pretrained": "neurogpt",
    "eegpt_pretrained": "eegpt",
}
_cmap = plt.get_cmap("tab10")
MODEL_COLORS = {mk: _cmap(i) for i, mk in enumerate(MODEL_ORDER)}
MODEL_COLORS["cbramod_random_init"] = "#888888"

STAGE_LABELS = ["W", "N1", "N2", "N3", "REM"]

AROUSAL_THRESHOLDS = [0.0, 1.0, 3.0, 5.0, 7.0, 9.0]
AROUSAL_CLINICAL = 3.0

RESP_THRESHOLDS = [0.0, 1.0, 3.0, 5.0, 7.0, 10.0]
RESP_CLINICAL = 10.0
RESP_EVENT_TYPES = [
    ("Any respiratory event", "respiratory_event_any_fraction"),
    ("Apnea (any subtype)",    "apnea_any_fraction"),
    ("Hypopnea (any subtype)", "hypopnea_any_fraction"),
    ("Periodic breathing (CSR)", "periodic_breathing_fraction"),
]


# ---------------------------------------------------------------------------
# sklearn_linear probe configuration
# ---------------------------------------------------------------------------

SK_EXCLUDE = {"bendr_pretrained", "tfc_pretrained_sleepEEG"}

SK_MODEL_ORDER = MODEL_ORDER + ["cbramod_random_init"]

TSFM_MODEL_ORDER = [
    "chronos_t5_tiny", "chronos_t5_small", "chronos_t5_base", "chronos_t5_large",
    "moment_small", "moment_base", "moment_large",
    "moirai_small", "moirai_base", "moirai_large",
    "timesfm_200m", "timesfm_500m",
]
TSFM_MODEL_LABELS = {
    "chronos_t5_tiny":  "chronos-t", "chronos_t5_small": "chronos-s",
    "chronos_t5_base":  "chronos-b", "chronos_t5_large": "chronos-l",
    "moment_small": "moment-s", "moment_base": "moment-b", "moment_large": "moment-l",
    "moirai_small": "moirai-s", "moirai_base": "moirai-b", "moirai_large": "moirai-l",
    "timesfm_200m": "timesfm-200m", "timesfm_500m": "timesfm-500m",
}
TSFM_MODEL_COLORS = {
    "chronos_t5_tiny":  "#E8A838", "chronos_t5_small": "#D4790E",
    "chronos_t5_base":  "#C05A10", "chronos_t5_large": "#9B3D12",
    "moment_small": "#D45B7A", "moment_base": "#B5365A", "moment_large": "#8E1E40",
    "moirai_small": "#8CBF3F", "moirai_base": "#5F9A20", "moirai_large": "#3D7010",
    "timesfm_200m": "#6A5ACD", "timesfm_500m": "#483D8B",
}
TSFM_MODEL_SET = frozenset(TSFM_MODEL_ORDER)
MODEL_LABELS.update(TSFM_MODEL_LABELS)
MODEL_COLORS.update(TSFM_MODEL_COLORS)

SUP_MODEL_ORDER = [
    "core_sleep_shhs_fold0",
    "sleepyco_shhs_fold0",
    "sleep_transformer_shhs_fold0",
]
SUP_MODEL_LABELS = {
    "core_sleep_shhs_fold0": "CoreSleep",
    "sleepyco_shhs_fold0": "SleepyCo",
    "sleep_transformer_shhs_fold0": "SleepTransf",
}
SUP_MODEL_COLORS = {
    "core_sleep_shhs_fold0": "#2CA02C",
    "sleepyco_shhs_fold0": "#17BECF",
    "sleep_transformer_shhs_fold0": "#BCBD22",
}
SUP_SEQ1_MODEL_ORDER = [
    "core_sleep_shhs_fold0_seq1",
    "sleepyco_shhs_fold0_seq1",
    "sleep_transformer_shhs_fold0_seq1",
]
SUP_SEQ1_MODEL_LABELS = {
    "core_sleep_shhs_fold0_seq1": "CoreSleep-s1",
    "sleepyco_shhs_fold0_seq1": "SleepyCo-s1",
    "sleep_transformer_shhs_fold0_seq1": "SleepTransf-s1",
}
SUP_SEQ1_MODEL_COLORS = {
    "core_sleep_shhs_fold0_seq1": "#1E7A1E",
    "sleepyco_shhs_fold0_seq1": "#0F8EA0",
    "sleep_transformer_shhs_fold0_seq1": "#9A9B1C",
}
SUP_FAMILIES = [
    ("CoreSleep", ["core_sleep_shhs_fold0", "core_sleep_shhs_fold0_seq1"]),
    ("SleepyCo",  ["sleepyco_shhs_fold0", "sleepyco_shhs_fold0_seq1"]),
    ("SleepTransf", ["sleep_transformer_shhs_fold0",
                     "sleep_transformer_shhs_fold0_seq1"]),
]
SUP_MODEL_SET = frozenset(SUP_MODEL_ORDER + SUP_SEQ1_MODEL_ORDER)
MODEL_LABELS.update(SUP_MODEL_LABELS)
MODEL_LABELS.update(SUP_SEQ1_MODEL_LABELS)
MODEL_COLORS.update(SUP_MODEL_COLORS)
MODEL_COLORS.update(SUP_SEQ1_MODEL_COLORS)

SK_ALL_MODEL_ORDER = (SK_MODEL_ORDER + TSFM_MODEL_ORDER
                      + SUP_MODEL_ORDER + SUP_SEQ1_MODEL_ORDER)

TSFM_FAMILIES = [
    ("chronos", ["chronos_t5_tiny", "chronos_t5_small", "chronos_t5_base",
                  "chronos_t5_large"]),
    ("moment",  ["moment_small", "moment_base", "moment_large"]),
    ("moirai",  ["moirai_small", "moirai_base", "moirai_large"]),
    ("timesfm", ["timesfm_200m", "timesfm_500m"]),
]


def _is_tsfm(checkpoint_id: str) -> bool:
    return checkpoint_id in TSFM_MODEL_SET


def _is_sup(checkpoint_id: str) -> bool:
    return checkpoint_id in SUP_MODEL_SET


def _model_group(checkpoint_id: str) -> str:
    if _is_tsfm(checkpoint_id):
        return "tsfm"
    if _is_sup(checkpoint_id):
        return "sup"
    return "eeg"

SK_SLEEP_DATASETS = ["mass", "dcsm", "wsc", "ucddb", "dod", "isruc",
                     "sleep_edf_expanded", "physionet2026"]
SK_SLEEP_RESULTS_PATH = {
    "mass":              "mass/sklearn_linear/results.json",
    "dcsm":              "dcsm/sklearn_linear/results.json",
    "wsc":               "wsc/sklearn_linear/results.json",
    "ucddb":             "ucddb/sklearn_linear/results.json",
    "dod":               "dod/sleep_stage/sklearn_linear/results.json",
    "isruc":             "isruc/sleep_stage/sklearn_linear/results.json",
    "sleep_edf_expanded":"sleep_edf_expanded/sleep_stage/sklearn_linear/results.json",
    "physionet2026":     "physionet2026/sklearn_linear/results.json",
}

SK_DOD_SUBGROUPS = ["dod", "dod_dodh", "dod_dodo"]
SK_DOD_SUBGROUP_PATHS = {
    "dod":      "dod/sleep_stage/sklearn_linear/results.json",
    "dod_dodh": "dod_dodh/sleep_stage/sklearn_linear/results.json",
    "dod_dodo": "dod_dodo/sleep_stage/sklearn_linear/results.json",
}

SK_DOD_SCORERS = [f"scorer_{i}" for i in range(1, 6)]
SK_DOD_SCORER_DATASETS = ["dod", "dod_dodh", "dod_dodo"]
SK_DOD_SCORER_PATHS: Dict[str, Dict[str, str]] = {
    ds: {f"scorer_{i}": f"{ds}/scorer_{i}/sklearn_linear/results.json"
         for i in range(1, 6)}
    for ds in ["dod", "dod_dodh", "dod_dodo"]
}
SK_DOD_SCORER_COLORS = {
    "scorer_1": "#3B7DD8",
    "scorer_2": "#E8913A",
    "scorer_3": "#4DAF4A",
    "scorer_4": "#E15759",
    "scorer_5": "#9467BD",
}

SK_SLEEPEDF_SUBGROUPS = ["sleep_edf_expanded", "sleep_edf_expanded_cassette",
                         "sleep_edf_expanded_telemetry"]
SK_SLEEPEDF_SUBGROUP_PATHS = {
    "sleep_edf_expanded":          "sleep_edf_expanded/sleep_stage/sklearn_linear/results.json",
    "sleep_edf_expanded_cassette": "sleep_edf_expanded_cassette/sklearn_linear/results.json",
    "sleep_edf_expanded_telemetry":"sleep_edf_expanded_telemetry/sklearn_linear/results.json",
}
SK_SLEEPEDF_INTERVENTION_PATH = "sleep_edf_expanded_intervention/sklearn_linear/results.json"

SK_DATASET_COLORS = {
    **DATASET_COLORS,
    "sleep_edf_expanded":          "#937860",
    "physionet2026":               "#64B5CD",
    "dod_dodh":                    "#A06AB4",
    "dod_dodo":                    "#6AAB9C",
    "sleep_edf_expanded_cassette": "#E07B54",
    "sleep_edf_expanded_telemetry":"#76A963",
    "sleep_edf_expanded_intervention": "#C4A24D",
}

SK_MASS_SUBSETS = ["mass", "mass_SS01", "mass_SS02", "mass_SS03",
                   "mass_SS04", "mass_SS05"]
SK_MASS_SUBSET_PATHS = {
    "mass":      "mass/sklearn_linear/results.json",
    "mass_SS01": "mass/sklearn_linear_SS01/results.json",
    "mass_SS02": "mass/sklearn_linear_SS02/results.json",
    "mass_SS03": "mass/sklearn_linear_SS03/results.json",
    "mass_SS04": "mass/sklearn_linear_SS04/results.json",
    "mass_SS05": "mass/sklearn_linear_SS05/results.json",
}
SK_DATASET_COLORS.update({
    "mass_SS01": "#2A5599",
    "mass_SS02": "#5B8DBE",
    "mass_SS03": "#8CB8D9",
    "mass_SS04": "#3D7A5F",
    "mass_SS05": "#B8604A",
})

SK_ISRUC_SCORERS = ["isruc", "isruc_scorer_1", "isruc_scorer_2"]
SK_ISRUC_SCORER_PATHS = {
    "isruc":          "isruc/sleep_stage/sklearn_linear/results.json",
    "isruc_scorer_1": "isruc/scorer_1/sklearn_linear/results.json",
    "isruc_scorer_2": "isruc/scorer_2/sklearn_linear/results.json",
}
SK_DATASET_COLORS.update({
    "isruc_scorer_1": "#D4A844",
    "isruc_scorer_2": "#A89244",
})

SK_ISRUC_SUBGROUPS = ["isruc", "isruc_SG_I_II", "isruc_SG_III"]
SK_ISRUC_SUBGROUP_PATHS = {
    "isruc":          "isruc/sleep_stage/sklearn_linear/results.json",
    "isruc_SG_I_II":  "isruc/sleep_stage/sklearn_linear_SG_I_II/results.json",
    "isruc_SG_III":   "isruc/sleep_stage/sklearn_linear_SG_III/results.json",
}
SK_DATASET_COLORS.update({
    "isruc_SG_I_II": "#C4985A",
    "isruc_SG_III":  "#8A6E3E",
})

SK_PN2026_SITES = ["physionet2026", "physionet2026_I0002",
                    "physionet2026_I0006", "physionet2026_S0001"]
SK_PN2026_SITE_PATHS = {
    "physionet2026":        "physionet2026/sklearn_linear/results.json",
    "physionet2026_I0002":  "physionet2026/sklearn_linear_I0002/results.json",
    "physionet2026_I0006":  "physionet2026/sklearn_linear_I0006/results.json",
    "physionet2026_S0001":  "physionet2026/sklearn_linear_S0001/results.json",
}
SK_DATASET_COLORS.update({
    "physionet2026_I0002": "#3A9BDC",
    "physionet2026_I0006": "#E8913A",
    "physionet2026_S0001": "#4DAF4A",
})

SK_AROUSAL_PATHS = {
    "mass_arousal":         "mass_arousal/{probe}/results.json",
    "physionet2026_arousal":"physionet2026_arousal/{probe}/results.json",
}
SK_AROUSAL_COLORS = {"mass_arousal": "#4C72B0", "physionet2026_arousal": "#DD8452"}

LIMB_THRESHOLDS = [0.0, 0.5, 1.0, 3.0, 5.0]
LIMB_CLINICAL = 3.0
LIMB_EVENT_TYPES = [
    ("Limb movement (any)", "limb_movement_any_fraction"),
    ("Limb movement (PLM)", "limb_movement_plm_fraction"),
]

SK_RESP_UCDDB_EVENT_TYPES = RESP_EVENT_TYPES
SK_RESP_PN2026_EVENT_TYPES = [
    ("Any respiratory event", "respiratory_event_any_fraction"),
    ("Apnea (any subtype)",   "apnea_any_fraction"),
    ("Hypopnea",              "hypopnea_fraction"),
    ("RERA",                  "rera_fraction"),
]

PROBE_VARIANTS = [
    ("sklearn_linear",          "non-balanced"),
    ("sklearn_linear_balanced", "balanced"),
]
PROBE_VARIANT_COLORS = {"non-balanced": "#4C72B0", "balanced": "#DD8452"}


# ---------------------------------------------------------------------------
# Literature-derived reference values
#
# Human = expert inter-rater agreement (or best-non-human ceiling where no
# direct human benchmark exists).  Clinical = utility threshold below which the
# community does not consider automated scoring diagnostically meaningful.
# ---------------------------------------------------------------------------
HUMAN = {
    "sleep_kappa": 0.76,      # Rosenberg & Van Hout 2013 (AASM ISR)
    "sleep_macro_f1": 0.76,   # Magalang 2013 SAGIC ~0.78
    "sleep_f1_per_stage": {"W": 0.93, "N1": 0.46, "N2": 0.88, "N3": 0.75, "REM": 0.85},
    # Arousal — PhysioNet 2018 winner as the effective automated ceiling.
    "arousal_auprc": 0.54,    # Howe-Patterson et al. 2018
    "arousal_mcc": 0.50,      # Drakatos 2019; κ≈MCC on binary labels
    # Respiratory — EEG-only plateau; AUROC with full PSG.
    "resp_auprc_any": 0.45,   # Mostafa 2019 / Urtnasan 2018 EEG-only plateau
    "resp_auroc": 0.88,       # Ruehland 2009 AHI ICC → expert AUROC
    "resp_mcc": 0.30,         # Whitney 1998 hypopnea κ ≈ 0.31
}
CLINICAL = {
    "sleep_kappa": 0.65,      # Landis-Koch "substantial"; USleep/SleepEEGNet norm
    "sleep_macro_f1": 0.65,
    "arousal_mcc": 0.15,
    "arousal_auprc_mult": 3.0,  # ≈ 3 × base rate
    "resp_auroc": 0.70,       # conventional "fair-to-good diagnostic"
    "resp_mcc": 0.10,
    "resp_auprc_mult": 3.0,   # ≈ 3 × base rate
}

_HUMAN_STYLE = dict(color="#333333", linestyle=(0, (5, 2)), linewidth=1.0)
_CLINICAL_STYLE = dict(color="#C06A1A", linestyle=(0, (3, 1, 1, 1)), linewidth=1.0)


def _hline(ax, y: float, label: str, style: Dict[str, object]) -> None:
    """Draw one horizontal reference line with an inline right-edge label."""
    ax.axhline(y, **style)
    xmax = ax.get_xlim()[1]
    ax.annotate(
        label,
        xy=(xmax, y),
        xytext=(-2, 2), textcoords="offset points",
        ha="right", va="bottom", fontsize=8,
        color=style["color"],
    )


def _draw_refs(ax, *, human=None, clinical=None,
               human_label="human", clinical_label="clinical") -> None:
    """Draw up to two reference lines with small inline labels."""
    if human is not None:
        _hline(ax, human, f"{human_label}: {human:.2f}", _HUMAN_STYLE)
    if clinical is not None:
        _hline(ax, clinical, f"{clinical_label}: {clinical:.2f}", _CLINICAL_STYLE)


def _load_json(path: Path) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _sleep_results_path(dataset: str) -> Path:
    rel = DATASET_RESULTS_PATH.get(dataset, f"{dataset}/linear/results.json")
    return REPO / "artifacts/benchmarks" / rel


def _load_linear_probe_entries(path: Path) -> List[Dict]:
    """Return only successful linear-probe rows from a results.json."""
    if not path.exists():
        return []
    return [
        e for e in _load_json(path)
        if e.get("evaluation_mode") == "linear_probe_eval" and e.get("ok")
    ]


def sleep_metric(dataset: str, key: str) -> Dict[str, List[float]]:
    """Return {checkpoint_id: [per-fold mean]} for a sleep-staging metric."""
    entries = _load_linear_probe_entries(_sleep_results_path(dataset))
    out: Dict[str, List[float]] = defaultdict(list)
    for e in entries:
        v = e.get("metrics", {}).get(key)
        if isinstance(v, dict) and "mean" in v:
            out[e["checkpoint_id"]].append(v["mean"])
    return dict(out)


def sleep_f1_per_stage(dataset: str) -> Dict[str, np.ndarray]:
    """Return {checkpoint_id: (n_folds, 5) F1-per-stage matrix}."""
    entries = _load_linear_probe_entries(_sleep_results_path(dataset))
    bucket: Dict[str, List[List[float]]] = defaultdict(list)
    for e in entries:
        bt = e.get("metrics", {}).get("best_test", {})
        f1 = bt.get("f1_per_class")
        if isinstance(f1, list) and len(f1) == 5:
            bucket[e["checkpoint_id"]].append(f1)
    return {k: np.asarray(v) for k, v in bucket.items()}


def _populated_sleep_datasets() -> List[str]:
    """Subset of ``SLEEP_DATASETS`` with at least one successful fold."""
    return [ds for ds in SLEEP_DATASETS if sleep_metric(ds, "cohen_kappa")]


def arousal_metric(thr_key: str, key: str) -> Dict[str, List[float]]:
    path = REPO / "artifacts/benchmarks/mass_arousal/linear/results.json"
    data = _load_json(path)
    out: Dict[str, List[float]] = defaultdict(list)
    for e in data:
        if not e.get("ok"):
            continue
        bt = e.get("metrics", {}).get(thr_key, {}).get("best_test", {})
        if key in bt:
            out[e["checkpoint_id"]].append(bt[key])
    return dict(out)


def arousal_prevalence(thr_key: str) -> float:
    path = REPO / "artifacts/benchmarks/mass_arousal/linear/results.json"
    data = _load_json(path)
    vals = []
    for e in data:
        if not e.get("ok"):
            continue
        bt = e.get("metrics", {}).get(thr_key, {}).get("best_test", {})
        cm = bt.get("confusion_matrix")
        if cm and len(cm) == 2:
            tn, fp = cm[0]; fn, tp = cm[1]; t = tn + fp + fn + tp
            if t:
                vals.append((tp + fn) / t)
    return float(np.mean(vals)) if vals else 0.0


def resp_metric(thr_key: str, field: str, key: str) -> Dict[str, List[float]]:
    path = REPO / "artifacts/benchmarks/ucddb_respevt/linear/results.json"
    data = _load_json(path)
    out: Dict[str, List[float]] = defaultdict(list)
    for e in data:
        if not e.get("ok"):
            continue
        fld = (e.get("metrics") or {}).get(field, {}) or {}
        if not isinstance(fld, dict):
            continue
        tv = fld.get(thr_key) or {}
        bt = tv.get("best_test", {}) if isinstance(tv, dict) else {}
        if key in bt:
            out[e["checkpoint_id"]].append(bt[key])
    return dict(out)


def resp_prevalence(thr_key: str, field: str) -> float:
    path = REPO / "artifacts/benchmarks/ucddb_respevt/linear/results.json"
    data = _load_json(path)
    vals = []
    for e in data:
        if not e.get("ok"):
            continue
        fld = (e.get("metrics") or {}).get(field, {}) or {}
        tv = fld.get(thr_key) if isinstance(fld, dict) else None
        if not tv or "best_test" not in tv:
            continue
        bt = tv["best_test"]
        cm = bt.get("confusion_matrix")
        if cm and len(cm) == 2:
            tn, fp = cm[0]; fn, tp = cm[1]; t = tn + fp + fn + tp
            if t:
                vals.append((tp + fn) / t)
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# sklearn_linear data extraction
# ---------------------------------------------------------------------------

def _sk_load_entries(path: Path, eval_mode: str) -> List[Dict]:
    if not path.exists():
        return []
    return [
        e for e in _load_json(path)
        if e.get("evaluation_mode") == eval_mode
        and e.get("ok")
        and e.get("checkpoint_id") not in SK_EXCLUDE
    ]


def _sk_results_path(path_map: Dict[str, str], dataset: str) -> Path:
    return REPO / "artifacts/benchmarks" / path_map[dataset]


def _balanced_accuracy_from_cm(cm) -> float | None:
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        return None
    row_sums = cm.sum(axis=1)
    if np.any(row_sums == 0):
        return None
    return float(np.mean(np.diag(cm) / row_sums))


def sk_sleep_metric(path_map: Dict[str, str], dataset: str,
                    key: str) -> Dict[str, List[float]]:
    entries = _sk_load_entries(
        _sk_results_path(path_map, dataset), "linear_probe_eval")
    out: Dict[str, List[float]] = defaultdict(list)
    for e in entries:
        v = e.get("metrics", {}).get(key)
        if isinstance(v, dict) and "mean" in v:
            out[e["checkpoint_id"]].append(v["mean"])
        elif key == "balanced_accuracy":
            bt = e.get("metrics", {}).get("best_test", {})
            cm = bt.get("confusion_matrix")
            if cm is not None:
                ba = _balanced_accuracy_from_cm(cm)
                if ba is not None:
                    out[e["checkpoint_id"]].append(ba)
    return dict(out)


def sk_sleep_f1_per_stage(path_map: Dict[str, str],
                          dataset: str) -> Dict[str, np.ndarray]:
    entries = _sk_load_entries(
        _sk_results_path(path_map, dataset), "linear_probe_eval")
    bucket: Dict[str, List[List[float]]] = defaultdict(list)
    for e in entries:
        bt = e.get("metrics", {}).get("best_test", {})
        f1 = bt.get("f1_per_class")
        if isinstance(f1, list) and len(f1) == 5:
            bucket[e["checkpoint_id"]].append(f1)
    return {k: np.asarray(v) for k, v in bucket.items()}


def sk_event_metric(path: Path, event_field: str | None,
                    thr_key: str, key: str) -> Dict[str, List[float]]:
    entries = _sk_load_entries(path, "arousal_detection" if event_field is None
                              else "respiratory_event_detection")
    out: Dict[str, List[float]] = defaultdict(list)
    for e in entries:
        if event_field is None:
            tv = (e.get("metrics") or {}).get(thr_key) or {}
        else:
            fld = (e.get("metrics") or {}).get(event_field, {}) or {}
            if not isinstance(fld, dict):
                continue
            tv = fld.get(thr_key) or {}
        bt = tv.get("best_test", {}) if isinstance(tv, dict) else {}
        if key in bt:
            out[e["checkpoint_id"]].append(bt[key])
    return dict(out)


def sk_event_prevalence(path: Path, event_field: str | None,
                        thr_key: str) -> float:
    entries = _sk_load_entries(path, "arousal_detection" if event_field is None
                              else "respiratory_event_detection")
    vals = []
    for e in entries:
        if event_field is None:
            tv = (e.get("metrics") or {}).get(thr_key) or {}
        else:
            fld = (e.get("metrics") or {}).get(event_field, {}) or {}
            tv = fld.get(thr_key) if isinstance(fld, dict) else None
        if not tv or "best_test" not in tv:
            continue
        bt = tv["best_test"]
        cm = bt.get("confusion_matrix")
        if cm and len(cm) == 2:
            tn, fp = cm[0]; fn, tp = cm[1]; t = tn + fp + fn + tp
            if t:
                vals.append((tp + fn) / t)
    return float(np.mean(vals)) if vals else 0.0


def sk_patient_metric(path: Path,
                      metric_keys: Tuple[str, ...],
                      aggregation: str = "mean") -> Tuple[
        Dict[str, Dict[str, List[float]]], List[float]]:
    entries = _sk_load_entries(path, "patient_classification_eval")
    entries = [e for e in entries
               if (e.get("metadata") or {}).get("aggregation") == aggregation]
    per_model: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    prevalences: List[float] = []
    for e in entries:
        bt = (e.get("metrics") or {}).get("best_test", {}) or {}
        for key in metric_keys:
            if key in bt and bt[key] is not None:
                per_model[e["checkpoint_id"]][key].append(float(bt[key]))
        cm = bt.get("confusion_matrix")
        if cm and len(cm) == 2:
            tn, fp = cm[0]; fn, tp = cm[1]; total = tn + fp + fn + tp
            if total:
                prevalences.append((tp + fn) / total)
    return {k: dict(v) for k, v in per_model.items()}, prevalences


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save(fig, name: str) -> None:
    out = OUT / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Plot 1 + 2: sleep-staging grouped bars
# ---------------------------------------------------------------------------

def plot_sleep_grouped_bar(metric_key: str, ylabel: str, filename: str) -> None:
    datasets = _populated_sleep_datasets()
    data = {ds: sleep_metric(ds, metric_key) for ds in datasets}
    n_models = len(MODEL_ORDER)
    n_ds = len(datasets)
    bar_w = 0.8 / n_ds
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, ds in enumerate(datasets):
        means, stds = [], []
        for mk in MODEL_ORDER:
            vals = data[ds].get(mk, [])
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            else:
                means.append(np.nan)
                stds.append(0.0)
        offset = (i - (n_ds - 1) / 2) * bar_w
        ax.bar(
            x + offset, means, bar_w,
            yerr=stds, capsize=2.5,
            label=ds.upper(),
            color=DATASET_COLORS[ds], edgecolor="black", linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Sleep staging — {ylabel} (5-fold, test)")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="dataset", loc="lower left")
    _draw_refs(ax, human=HUMAN[metric_key.replace("cohen_kappa", "sleep_kappa")
                                  .replace("macro_f1", "sleep_macro_f1")],
               clinical=CLINICAL[metric_key.replace("cohen_kappa", "sleep_kappa")
                                            .replace("macro_f1", "sleep_macro_f1")])
    _save(fig, filename)


# ---------------------------------------------------------------------------
# Plot 3: sleep-staging F1-per-stage heatmaps
# ---------------------------------------------------------------------------

def plot_sleep_f1_per_stage() -> None:
    # Pre-load f1-per-class for each dataset once.
    datasets = _populated_sleep_datasets()
    by_ds = {ds: sleep_f1_per_stage(ds) for ds in datasets}
    x = np.arange(len(datasets))
    ds_labels = [ds.upper() for ds in datasets]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for stage_idx, stage in enumerate(STAGE_LABELS):
        ax = axes.flat[stage_idx]
        for mk in MODEL_ORDER:
            means, stds = [], []
            for ds in datasets:
                mat = by_ds[ds].get(mk)
                if mat is None:
                    means.append(np.nan); stds.append(0.0)
                else:
                    means.append(float(mat[:, stage_idx].mean()))
                    stds.append(float(mat[:, stage_idx].std()))
            means = np.asarray(means); stds = np.asarray(stds)
            ax.plot(
                x, means, marker="o", color=MODEL_COLORS[mk],
                label=MODEL_LABELS[mk], linewidth=1.8,
            )
            ax.fill_between(x, means - stds, means + stds, color=MODEL_COLORS[mk], alpha=0.1)
        ax.set_xticks(x)
        ax.set_xticklabels(ds_labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("F1")
        ax.set_title(f"Stage {stage}")
        ax.grid(True, alpha=0.3)
        # Per-stage human-agreement reference line.
        _draw_refs(ax, human=HUMAN["sleep_f1_per_stage"][stage], human_label="human")
    # Hide the 6th subplot; use it for a single shared legend.
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", fontsize=11, title="model",
                     title_fontsize=12, frameon=False)
    fig.suptitle("Sleep staging — F1 per class across datasets (test, fold-mean ± std)",
                 fontsize=14)
    _save(fig, "sleep_f1_per_stage.png")


# ---------------------------------------------------------------------------
# Plot 4 + 6: threshold sweep line plot
# ---------------------------------------------------------------------------

def plot_threshold_sweep(
    *,
    thresholds: List[float],
    per_model_per_thr: Dict[str, Dict[float, Tuple[float, float]]],
    title: str,
    ylabel: str,
    ref_vline: float | None,
    ref_hline_label: str | None,
    ref_hline: float | None,
    out: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for mk in MODEL_ORDER + ["cbramod_random_init"]:
        if mk not in per_model_per_thr:
            continue
        values = per_model_per_thr[mk]
        xs, ys, ses = [], [], []
        for t in thresholds:
            if t in values:
                m, s = values[t]
                xs.append(t); ys.append(m); ses.append(s)
        if not xs:
            continue
        xs = np.array(xs); ys = np.array(ys); ses = np.array(ses)
        ax.plot(xs, ys, marker="o", color=MODEL_COLORS[mk],
                label=MODEL_LABELS[mk], linewidth=1.8)
        ax.fill_between(xs, ys - ses, ys + ses, color=MODEL_COLORS[mk], alpha=0.12)
    if ref_vline is not None:
        ax.axvline(ref_vline, color="black", linestyle="--", lw=1,
                   label=f"AASM = {ref_vline:g} s")
    if ref_hline is not None:
        ax.axhline(ref_hline, color="tab:gray", linestyle=":", lw=1,
                   label=ref_hline_label or "reference")
    ax.set_xlabel("event-duration threshold (s of event per epoch)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    _save(fig, out)


def plot_arousal_threshold_sweep() -> None:
    per_model: Dict[str, Dict[float, Tuple[float, float]]] = defaultdict(dict)
    for t in AROUSAL_THRESHOLDS:
        thr_key = f"threshold_{t}s"
        vals = arousal_metric(thr_key, "auprc")
        for mk, v in vals.items():
            per_model[mk][t] = (float(np.mean(v)), float(np.std(v)))
    base = arousal_prevalence("threshold_0.0s")
    # Clinical AUPRC bar at the AASM-3s threshold: 3× base rate at thr=3.0s.
    base_at_3 = arousal_prevalence(f"threshold_{AROUSAL_CLINICAL}s")
    clinical = CLINICAL["arousal_auprc_mult"] * base_at_3
    human = HUMAN["arousal_auprc"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for mk in MODEL_ORDER + ["cbramod_random_init"]:
        if mk not in per_model:
            continue
        xs, ys, ses = [], [], []
        for t in AROUSAL_THRESHOLDS:
            if t in per_model[mk]:
                m, s = per_model[mk][t]
                xs.append(t); ys.append(m); ses.append(s)
        xs = np.array(xs); ys = np.array(ys); ses = np.array(ses)
        ax.plot(xs, ys, marker="o", color=MODEL_COLORS[mk],
                label=MODEL_LABELS[mk], linewidth=1.8)
        ax.fill_between(xs, ys - ses, ys + ses, color=MODEL_COLORS[mk], alpha=0.12)
    ax.axvline(AROUSAL_CLINICAL, color="black", linestyle="--", lw=1,
               label=f"AASM = {AROUSAL_CLINICAL:g} s")
    ax.set_xlabel("event-duration threshold (s of event per epoch)")
    ax.set_ylabel("AUPRC")
    ax.set_title("MASS arousal detection — AUPRC vs threshold (class-weighted probe)")
    ax.grid(True, alpha=0.3)
    _draw_refs(ax, human=human, clinical=clinical,
               human_label="human/PN2018", clinical_label="clinical @ 3s")
    ax.legend(loc="best", fontsize=9, ncol=2)
    _save(fig, "arousal_auprc_vs_threshold.png")


# ---------------------------------------------------------------------------
# Plot 5 + parts of 8: 3-panel bar (AUPRC / AUROC / MCC) at one threshold
# ---------------------------------------------------------------------------

def _bar_panel(
    ax, vals_by_model: Dict[str, List[float]], baseline: float | None,
    ylabel: str, sort_key: Dict[str, float] | None = None, ylim: Tuple[float, float] | None = None,
):
    ordering = sorted(vals_by_model.keys(),
                      key=(lambda m: -(sort_key or vals_by_model).get(m, [0]).__getitem__(0)
                           if isinstance((sort_key or vals_by_model).get(m), list)
                           else -(sort_key or {m: np.mean(vals_by_model[m])}).get(m, 0.0)))
    if sort_key is not None:
        # sort_key is dict of float mean values
        ordering = sorted(vals_by_model.keys(), key=lambda m: -sort_key.get(m, -np.inf))
    else:
        ordering = sorted(vals_by_model.keys(),
                          key=lambda m: -float(np.mean(vals_by_model[m])))
    means = [float(np.mean(vals_by_model[m])) for m in ordering]
    stds = [float(np.std(vals_by_model[m])) for m in ordering]
    colors = [MODEL_COLORS.get(m, "#777") for m in ordering]
    xs = np.arange(len(ordering))
    ax.bar(xs, means, color=colors, yerr=stds, capsize=3, edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in ordering], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if baseline is not None:
        ax.axhline(baseline, color="black", linestyle="--", lw=1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.3)


def _paired_bar_panel(
    ax,
    vals_by_variant: Dict[str, Dict[str, List[float]]],
    baseline: float | None,
    ylabel: str,
    ylim: Tuple[float, float] | None = None,
) -> None:
    all_models = set()
    for v in vals_by_variant.values():
        all_models |= v.keys()
    first_variant = next(iter(vals_by_variant.values()))
    _sort = lambda m: (-float(np.mean(first_variant[m]))
                       if m in first_variant else -np.inf)
    eeg_models = sorted(
        [m for m in all_models if not _is_tsfm(m) and not _is_sup(m)],
        key=_sort)
    tsfm_models = sorted([m for m in all_models if _is_tsfm(m)], key=_sort)
    sup_models = sorted([m for m in all_models if _is_sup(m)], key=_sort)
    ordering = eeg_models + tsfm_models + sup_models
    variants = list(vals_by_variant.keys())
    n_v = len(variants)
    bar_w = 0.8 / n_v
    n_eeg = len(eeg_models)
    n_tsfm = len(tsfm_models)
    n_sup = len(sup_models)
    gap = 0.7
    groups = []
    pos = 0
    if n_eeg:
        groups.append(np.arange(n_eeg) + pos)
        pos += n_eeg + (gap if (n_tsfm or n_sup) else 0)
    if n_tsfm:
        groups.append(np.arange(n_tsfm) + pos)
        pos += n_tsfm + (gap if n_sup else 0)
    if n_sup:
        groups.append(np.arange(n_sup) + pos)
    x = np.concatenate(groups) if groups else np.arange(len(ordering))
    for vi, vname in enumerate(variants):
        vdata = vals_by_variant[vname]
        means = [float(np.mean(vdata[m])) if m in vdata else np.nan
                 for m in ordering]
        stds = [float(np.std(vdata[m])) if m in vdata else 0.0
                for m in ordering]
        offset = (vi - (n_v - 1) / 2) * bar_w
        bars = ax.bar(x + offset, means, bar_w, yerr=stds, capsize=2.5,
                      label=vname, color=PROBE_VARIANT_COLORS.get(vname, "#777"),
                      edgecolor="black", linewidth=0.4)
        for bar, mk in zip(bars, ordering):
            if _is_tsfm(mk):
                bar.set_hatch("///")
            elif _is_sup(mk):
                bar.set_hatch("\\\\")
    if n_eeg and (n_tsfm or n_sup):
        ax.axvline(n_eeg - 0.5 + gap / 2, color="#aaa", linestyle=":",
                   lw=0.8, zorder=0)
    if n_tsfm and n_sup:
        ax.axvline(n_eeg + (gap if n_eeg else 0) + n_tsfm - 0.5 + gap / 2,
                   color="#aaa", linestyle=":", lw=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in ordering],
                       rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if baseline is not None:
        ax.axhline(baseline, color="black", linestyle="--", lw=1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.3)


def plot_arousal_bar_3s() -> None:
    thr_key = f"threshold_{AROUSAL_CLINICAL}s"
    auprc = arousal_metric(thr_key, "auprc")
    auroc = arousal_metric(thr_key, "auroc")
    mcc = arousal_metric(thr_key, "mcc")
    base = arousal_prevalence(thr_key)

    # rank by AUPRC mean
    sort_key = {m: float(np.mean(v)) for m, v in auprc.items()}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    _bar_panel(axes[0], auprc, baseline=base, ylabel="AUPRC", sort_key=sort_key)
    axes[0].set_title(f"AUPRC  (base rate = {base:.2f})")
    _draw_refs(axes[0], human=HUMAN["arousal_auprc"],
               clinical=CLINICAL["arousal_auprc_mult"] * base,
               human_label="human/PN2018", clinical_label="clinical")

    _bar_panel(axes[1], auroc, baseline=0.5, ylabel="AUROC", sort_key=sort_key, ylim=(0.4, 1.0))
    axes[1].set_title("AUROC  (chance = 0.5)")
    # No reliable human AUROC reference for arousal — omit.

    _bar_panel(axes[2], mcc, baseline=0.0, ylabel="MCC", sort_key=sort_key)
    axes[2].set_title("MCC  (chance = 0)")
    _draw_refs(axes[2], human=HUMAN["arousal_mcc"],
               clinical=CLINICAL["arousal_mcc"],
               human_label="human (κ)", clinical_label="clinical")

    fig.suptitle(f"MASS arousal @ threshold = {AROUSAL_CLINICAL:g} s (AASM arousal ≥3 s)", fontsize=14)
    _save(fig, "arousal_metrics_bar_3s.png")


# ---------------------------------------------------------------------------
# Plot 6 + 7: respiratory threshold sweeps (2×2 panels per event type)
# ---------------------------------------------------------------------------

def plot_resp_sweep(metric_key: str, ylabel: str, out: str, *, show_baseline: bool = True) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, (label, field) in zip(axes.flat, RESP_EVENT_TYPES):
        per_model: Dict[str, Dict[float, Tuple[float, float]]] = defaultdict(dict)
        for t in RESP_THRESHOLDS:
            thr_key = f"threshold_{t}s"
            vals = resp_metric(thr_key, field, metric_key)
            for mk, v in vals.items():
                per_model[mk][t] = (float(np.mean(v)), float(np.std(v)))
        for mk in MODEL_ORDER + ["cbramod_random_init"]:
            if mk not in per_model:
                continue
            xs, ys, ses = [], [], []
            for t in RESP_THRESHOLDS:
                if t in per_model[mk]:
                    m, s = per_model[mk][t]
                    xs.append(t); ys.append(m); ses.append(s)
            if not xs:
                continue
            xs = np.array(xs); ys = np.array(ys); ses = np.array(ses)
            ax.plot(xs, ys, marker="o", color=MODEL_COLORS[mk],
                    label=MODEL_LABELS[mk], linewidth=1.5)
            ax.fill_between(xs, ys - ses, ys + ses, color=MODEL_COLORS[mk], alpha=0.1)
        # per-panel baseline
        base = resp_prevalence(f"threshold_{RESP_CLINICAL}s", field)
        if show_baseline and base > 0:
            ax.axhline(base, color="tab:gray", linestyle=":", lw=1,
                       label=f"base rate @10s: {base:.3f}")
        ax.axvline(RESP_CLINICAL, color="black", linestyle="--", lw=1)
        ax.set_title(label)
        ax.set_xlabel("threshold (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        # Reference lines: applicable for "any" and "hypopnea-any" panels; apnea
        # and CSR subtypes are too rare / different in character.
        apply_refs = field in ("respiratory_event_any_fraction", "hypopnea_any_fraction")
        if apply_refs:
            if metric_key == "auprc":
                _draw_refs(ax, human=HUMAN["resp_auprc_any"],
                           clinical=CLINICAL["resp_auprc_mult"] * base if base > 0 else None,
                           human_label="human (EEG-only)",
                           clinical_label="clinical")
            elif metric_key == "mcc":
                _draw_refs(ax, human=HUMAN["resp_mcc"],
                           clinical=CLINICAL["resp_mcc"],
                           human_label="human (κ)",
                           clinical_label="clinical")
        if ax is axes[0, 0]:
            ax.legend(fontsize=8, loc="best", ncol=2)
    fig.suptitle(f"UCDDB respiratory events — {ylabel} vs threshold", fontsize=14)
    _save(fig, out)


# ---------------------------------------------------------------------------
# Plot 8: respiratory 4×3 grid at threshold 10s
# ---------------------------------------------------------------------------

def plot_resp_bar_10s() -> None:
    thr_key = f"threshold_{RESP_CLINICAL}s"
    fig, axes = plt.subplots(4, 3, figsize=(15, 15), constrained_layout=True)
    for row, (label, field) in enumerate(RESP_EVENT_TYPES):
        auprc = resp_metric(thr_key, field, "auprc")
        auroc = resp_metric(thr_key, field, "auroc")
        mcc = resp_metric(thr_key, field, "mcc")
        base = resp_prevalence(thr_key, field)

        # rank by AUPRC mean (same ordering across 3 cols)
        sort_key = {m: float(np.mean(v)) for m, v in auprc.items()}

        _bar_panel(axes[row, 0], auprc, baseline=base, ylabel="AUPRC", sort_key=sort_key)
        axes[row, 0].set_title(f"{label}\nAUPRC  (base rate = {base:.3f})")
        _bar_panel(axes[row, 1], auroc, baseline=0.5, ylabel="AUROC", sort_key=sort_key, ylim=(0.3, 1.0))
        axes[row, 1].set_title("AUROC  (chance = 0.5)")
        _bar_panel(axes[row, 2], mcc, baseline=0.0, ylabel="MCC", sort_key=sort_key)
        axes[row, 2].set_title("MCC  (chance = 0)")
        # Reference lines — informative only for broad categories.
        apply_refs = field in ("respiratory_event_any_fraction", "hypopnea_any_fraction")
        if apply_refs:
            _draw_refs(axes[row, 0],
                       human=HUMAN["resp_auprc_any"],
                       clinical=CLINICAL["resp_auprc_mult"] * base if base > 0 else None,
                       human_label="human (EEG-only)",
                       clinical_label="clinical")
            _draw_refs(axes[row, 1],
                       human=HUMAN["resp_auroc"],
                       clinical=CLINICAL["resp_auroc"],
                       human_label="human (full PSG)",
                       clinical_label="clinical")
            _draw_refs(axes[row, 2],
                       human=HUMAN["resp_mcc"],
                       clinical=CLINICAL["resp_mcc"],
                       human_label="human (κ)",
                       clinical_label="clinical")
    fig.suptitle(f"UCDDB respiratory events @ threshold = {RESP_CLINICAL:g} s (AASM floor)",
                 fontsize=14)
    _save(fig, "respiratory_metrics_at_10s.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_sleep_kappa_spider() -> None:
    """Radar / spider plot of Cohen's κ across datasets, one polygon per model."""
    datasets = _populated_sleep_datasets()
    data = {ds: sleep_metric(ds, "cohen_kappa") for ds in datasets}
    n_axes = len(datasets)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection="polar"))
    ax.set_theta_offset(np.pi / 2)   # first axis at top
    ax.set_theta_direction(-1)        # clockwise
    ax.set_rlabel_position(180 / n_axes)

    # Radial grid & reference circles.
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#555")
    ax.grid(True, alpha=0.3)

    # Reference rings.
    human = HUMAN["sleep_kappa"]
    clinical = CLINICAL["sleep_kappa"]
    theta_dense = np.linspace(0, 2 * np.pi, 361)
    ax.plot(theta_dense, np.full_like(theta_dense, human),
            color=_HUMAN_STYLE["color"], linestyle=_HUMAN_STYLE["linestyle"],
            linewidth=_HUMAN_STYLE["linewidth"], label=f"human κ = {human:.2f}")
    ax.plot(theta_dense, np.full_like(theta_dense, clinical),
            color=_CLINICAL_STYLE["color"], linestyle=_CLINICAL_STYLE["linestyle"],
            linewidth=_CLINICAL_STYLE["linewidth"], label=f"clinical κ = {clinical:.2f}")

    # Model polygons.
    for mk in MODEL_ORDER:
        values = []
        for ds in datasets:
            v = data[ds].get(mk, [])
            values.append(float(np.mean(v)) if v else np.nan)
        if all(np.isnan(values)):
            continue
        # Replace NaN with 0 for plotting but mark the axis tick.
        plotted = [v if not np.isnan(v) else 0 for v in values]
        plotted_closed = plotted + plotted[:1]
        ax.plot(angles_closed, plotted_closed,
                marker="o", color=MODEL_COLORS[mk], linewidth=1.8,
                label=MODEL_LABELS[mk])
        ax.fill(angles_closed, plotted_closed, color=MODEL_COLORS[mk], alpha=0.08)

    ax.set_xticks(angles)
    ax.set_xticklabels([ds.upper() for ds in datasets], fontsize=11)
    ax.set_title("Sleep staging — Cohen's κ across datasets", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05), fontsize=9,
              title="model", title_fontsize=10)
    _save(fig, "sleep_kappa_spider.png")


# ---------------------------------------------------------------------------
# Per-scorer inter-rater κ (DOD 5-scorer, ISRUC 2-scorer)
# ---------------------------------------------------------------------------

def _per_scorer_kappa(dataset: str) -> Tuple[List[str], Dict[str, Dict[str, List[float]]]]:
    """Return (label_order, {checkpoint: {label: [per-fold κ]}}) for a dataset.

    ``label_order`` starts with "consensus" and then one entry per populated
    expert scorer.  Datasets with no scorer runs or no successful folds return
    an empty order — the caller skips rendering.
    """
    sources: List[Tuple[str, Path]] = [("consensus", _sleep_results_path(dataset))]
    for slug in DATASET_SCORER_SLUGS.get(dataset, []):
        sources.append((slug, REPO / "artifacts/benchmarks" / dataset / slug / "linear/results.json"))

    populated_labels: List[str] = []
    by_label: Dict[str, Dict[str, List[float]]] = {}
    for label, path in sources:
        entries = _load_linear_probe_entries(path)
        if not entries:
            continue
        bucket: Dict[str, List[float]] = defaultdict(list)
        for e in entries:
            v = e.get("metrics", {}).get("cohen_kappa")
            if isinstance(v, dict) and "mean" in v:
                bucket[e["checkpoint_id"]].append(v["mean"])
        if bucket:
            populated_labels.append(label)
            by_label[label] = dict(bucket)
    return populated_labels, by_label


def _plot_per_scorer_kappa(dataset: str, filename: str) -> None:
    labels, by_label = _per_scorer_kappa(dataset)
    if not labels:
        print(f"skip {filename}: no per-scorer results for {dataset}")
        return

    # Keep only models that have at least one κ in at least one source, in
    # the canonical MODEL_ORDER.
    all_models = set().union(*(by_label[l].keys() for l in labels))
    models = [mk for mk in MODEL_ORDER if mk in all_models]
    if not models:
        print(f"skip {filename}: no known checkpoints in {dataset}")
        return

    n_labels = len(labels)
    bar_w = 0.8 / n_labels
    x = np.arange(len(models))
    cmap = plt.get_cmap("viridis", max(n_labels, 2))

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(models) + 2), 5.5))
    for i, label in enumerate(labels):
        means, stds = [], []
        for mk in models:
            vals = by_label[label].get(mk, [])
            means.append(float(np.mean(vals)) if vals else np.nan)
            stds.append(float(np.std(vals)) if vals else 0.0)
        offset = (i - (n_labels - 1) / 2) * bar_w
        color = "#333333" if label == "consensus" else cmap(i / max(n_labels - 1, 1))
        ax.bar(
            x + offset, means, bar_w,
            yerr=stds, capsize=2.5,
            label=label,
            color=color, edgecolor="black", linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=20, ha="right")
    ax.set_ylabel("Cohen's κ")
    ax.set_title(f"{dataset.upper()} sleep staging — κ against each expert scorer")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="label source", loc="lower left", fontsize=8, ncol=2)
    _draw_refs(ax, human=HUMAN["sleep_kappa"], clinical=CLINICAL["sleep_kappa"])
    _save(fig, filename)


def plot_dod_per_scorer_kappa() -> None:
    _plot_per_scorer_kappa("dod", "dod_per_scorer_kappa.png")


def plot_isruc_per_scorer_kappa() -> None:
    _plot_per_scorer_kappa("isruc", "isruc_per_scorer_kappa.png")


# ---------------------------------------------------------------------------
# Patient classification — DOD OSA
# ---------------------------------------------------------------------------

def _patient_metric_values(
    path: Path, metric_keys: Tuple[str, ...]
) -> Tuple[Dict[str, Dict[str, List[float]]], List[float]]:
    """Return ({checkpoint: {metric: [per-fold]}}, [per-fold positive rate]).

    ``metric_keys`` are read from ``metrics.best_test``.  Positive rate is
    derived per fold from the ``best_test.confusion_matrix`` of *any* row in
    that fold — patient prevalence is fold-level, so computing it once per
    (checkpoint, fold) and then deduplicating would just give noise; instead
    we average over every present confusion matrix and treat that as the
    dataset base rate.
    """
    if not path.exists():
        return {}, []
    data = _load_json(path)
    per_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    prevalences: List[float] = []
    for e in data:
        if e.get("evaluation_mode") != "patient_classification_eval" or not e.get("ok"):
            continue
        bt = (e.get("metrics") or {}).get("best_test", {}) or {}
        for key in metric_keys:
            if key in bt and bt[key] is not None:
                per_model[e["checkpoint_id"]][key].append(float(bt[key]))
        cm = bt.get("confusion_matrix")
        if cm and len(cm) == 2:
            tn, fp = cm[0]; fn, tp = cm[1]; total = tn + fp + fn + tp
            if total:
                prevalences.append((tp + fn) / total)
    return {k: dict(v) for k, v in per_model.items()}, prevalences


def plot_patient_classification_bar() -> None:
    path = REPO / "artifacts/benchmarks/dod_osa/linear/results.json"
    metric_keys = ("auroc", "auprc", "mcc", "macro_f1")
    per_model, prevalences = _patient_metric_values(path, metric_keys)
    if not per_model:
        print("skip dod_osa_patient_metrics.png: no patient-classification results")
        return

    base_rate = float(np.mean(prevalences)) if prevalences else 0.0
    vals_by_metric = {
        key: {mk: scores[key] for mk, scores in per_model.items() if scores.get(key)}
        for key in metric_keys
    }
    # Rank everything by AUPRC mean so the four panels stay visually aligned.
    auprc = vals_by_metric["auprc"]
    sort_key = {m: float(np.mean(v)) for m, v in auprc.items()}

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
    panel_specs = [
        ("auroc",    "AUROC",    0.5,           "AUROC  (chance = 0.5)",            (0.3, 1.0)),
        ("auprc",    "AUPRC",    base_rate,    f"AUPRC  (base rate = {base_rate:.2f})", None),
        ("mcc",      "MCC",      0.0,           "MCC  (chance = 0)",                 None),
        ("macro_f1", "macro-F1", 0.5,           "macro-F1  (random = 0.5)",          (0.0, 1.0)),
    ]
    for ax, (key, ylabel, baseline, title, ylim) in zip(axes, panel_specs):
        if not vals_by_metric[key]:
            ax.axis("off")
            ax.set_title(f"{title}\n(no data)")
            continue
        _bar_panel(ax, vals_by_metric[key],
                   baseline=baseline, ylabel=ylabel,
                   sort_key=sort_key, ylim=ylim)
        ax.set_title(title)

    fig.suptitle("DOD OSA patient classification — subject-level metrics (5-fold, test)",
                 fontsize=14)
    _save(fig, "dod_osa_patient_metrics.png")


def plot_isruc_diagnosis_bar() -> None:
    """Binary (has_diagnosis) + multi-class (diagnosis) side-by-side for ISRUC."""
    label_modes = [
        ("has_diagnosis", "Binary: Has Diagnosis", ("macro_f1",)),
        ("diagnosis",     "Multi-class: Diagnosis", ("macro_f1",)),
    ]
    aggregations = ["mean"]
    agg_labels = {"mean": "Mean", "mean_std": "Mean+Std"}
    agg_colors = {"mean": "#4C72B0", "mean_std": "#DD8452"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, (lm, title, metric_keys) in zip(axes, label_modes):
        path = REPO / f"artifacts/benchmarks/isruc_diagnosis/{lm}/linear/results.json"
        if not path.exists():
            ax.set_title(f"{title}\n(no data)")
            ax.axis("off")
            continue
        data = _load_json(path)
        ok_entries = [
            e for e in data
            if e.get("evaluation_mode") == "patient_classification_eval" and e.get("ok")
        ]
        if not ok_entries:
            ax.set_title(f"{title}\n(no data)")
            ax.axis("off")
            continue

        models = []
        for mk in MODEL_ORDER:
            if any(e["checkpoint_id"] == mk for e in ok_entries):
                models.append(mk)

        x = np.arange(len(models))
        bar_w = 0.35
        for i, agg in enumerate(aggregations):
            means, stds = [], []
            for mk in models:
                vals = [
                    e["metrics"]["best_test"]["macro_f1"]
                    for e in ok_entries
                    if e["checkpoint_id"] == mk
                    and (e.get("metadata") or {}).get("aggregation") == agg
                    and "best_test" in (e.get("metrics") or {})
                    and "macro_f1" in e["metrics"]["best_test"]
                ]
                means.append(float(np.mean(vals)) if vals else np.nan)
                stds.append(float(np.std(vals)) if vals else 0.0)
            offset = (i - 0.5) * bar_w
            bars = ax.bar(
                x + offset, means, bar_w, yerr=stds, capsize=4,
                label=agg_labels[agg], color=agg_colors[agg],
                edgecolor="black", linewidth=0.4,
            )
            for bar, val in zip(bars, means):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold",
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=20, ha="right")
        ax.set_ylabel("Test macro-F1")
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.legend(title="Aggregation", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3, lw=0.8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "ISRUC Diagnosis Probes — Linear Probe (subject-level, 5-fold)",
        fontsize=14, fontweight="bold",
    )
    _save(fig, "isruc_diagnosis_probes.png")


ISRUC_DIAGNOSIS_CLASSES = ["healthy", "osa", "snoring", "affective_disorder"]


# ---------------------------------------------------------------------------
# Confusion matrix helpers
# ---------------------------------------------------------------------------

def _sum_confusion_matrices(entries: List[Dict]) -> np.ndarray | None:
    """Sum best_test confusion matrices across folds into one aggregate."""
    cms = []
    for e in entries:
        cm = (e.get("metrics") or {}).get("best_test", {}).get("confusion_matrix")
        if cm is not None:
            cms.append(np.asarray(cm, dtype=float))
    if not cms:
        return None
    return sum(cms)


def _plot_cm(ax, cm: np.ndarray, labels: List[str], title: str) -> None:
    """Draw a normalised confusion matrix heatmap on *ax*."""
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="equal")
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for i in range(n):
        for j in range(n):
            count = int(cm[i, j])
            pct = cm_norm[i, j]
            color = "white" if pct > 0.5 else "black"
            ax.text(j, i, f"{count}\n{pct:.0%}", ha="center", va="center",
                    fontsize=7, color=color)
    return im


def plot_sleep_confusion_matrices() -> None:
    """One confusion matrix per (model, dataset) for sleep staging."""
    datasets = _populated_sleep_datasets()
    models = []
    # Determine which models have data in at least one dataset.
    for mk in MODEL_ORDER:
        for ds in datasets:
            entries = _load_linear_probe_entries(_sleep_results_path(ds))
            if any(e["checkpoint_id"] == mk for e in entries):
                models.append(mk)
                break
    if not models or not datasets:
        print("skip sleep_confusion_matrices.png: no data")
        return

    n_rows = len(models)
    n_cols = len(datasets)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.0 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    for row, mk in enumerate(models):
        for col, ds in enumerate(datasets):
            ax = axes[row, col]
            entries = [
                e for e in _load_linear_probe_entries(_sleep_results_path(ds))
                if e["checkpoint_id"] == mk
            ]
            cm = _sum_confusion_matrices(entries)
            if cm is None or cm.shape != (5, 5):
                ax.axis("off")
                ax.set_title(f"{MODEL_LABELS[mk]}\n{ds.upper()}", fontsize=10)
                continue
            title = f"{MODEL_LABELS[mk]} — {ds.upper()}"
            _plot_cm(ax, cm, STAGE_LABELS, title)

    fig.suptitle(
        "Sleep Staging — Confusion Matrices (summed across folds, test)",
        fontsize=14, fontweight="bold",
    )
    _save(fig, "sleep_confusion_matrices.png")


def plot_isruc_diagnosis_confusion_matrices() -> None:
    """One confusion matrix per model for ISRUC multi-class diagnosis."""
    path = REPO / "artifacts/benchmarks/isruc_diagnosis/diagnosis/linear/results.json"
    if not path.exists():
        print("skip isruc_diagnosis_confusion_matrices.png: no results")
        return
    data = _load_json(path)
    ok_entries = [
        e for e in data
        if e.get("evaluation_mode") == "patient_classification_eval" and e.get("ok")
    ]

    models = [mk for mk in MODEL_ORDER if any(e["checkpoint_id"] == mk for e in ok_entries)]
    aggregations = ["mean"]
    agg_labels_map = {"mean": "Mean", "mean_std": "Mean+Std"}
    if not models:
        print("skip isruc_diagnosis_confusion_matrices.png: no models")
        return

    n_rows = len(models)
    n_cols = len(aggregations)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4.0 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    class_labels = [c.replace("_", " ").title() for c in ISRUC_DIAGNOSIS_CLASSES]
    for row, mk in enumerate(models):
        for col, agg in enumerate(aggregations):
            ax = axes[row, col]
            entries = [
                e for e in ok_entries
                if e["checkpoint_id"] == mk
                and (e.get("metadata") or {}).get("aggregation") == agg
            ]
            cm = _sum_confusion_matrices(entries)
            n_cls = len(ISRUC_DIAGNOSIS_CLASSES)
            if cm is None or cm.shape != (n_cls, n_cls):
                ax.axis("off")
                ax.set_title(f"{MODEL_LABELS[mk]} ({agg_labels_map[agg]})", fontsize=10)
                continue
            title = f"{MODEL_LABELS[mk]} ({agg_labels_map[agg]})"
            _plot_cm(ax, cm, class_labels, title)

    fig.suptitle(
        "ISRUC Diagnosis — Confusion Matrices (summed across folds, test)",
        fontsize=14, fontweight="bold",
    )
    _save(fig, "isruc_diagnosis_confusion_matrices.png")


def plot_isruc_diagnosis_f1_per_class() -> None:
    """Per-class F1 for multi-class diagnosis, one subplot per class."""
    path = REPO / "artifacts/benchmarks/isruc_diagnosis/diagnosis/linear/results.json"
    if not path.exists():
        print("skip isruc_diagnosis_f1_per_class.png: no results")
        return
    data = _load_json(path)
    ok_entries = [
        e for e in data
        if e.get("evaluation_mode") == "patient_classification_eval" and e.get("ok")
    ]
    if not ok_entries:
        print("skip isruc_diagnosis_f1_per_class.png: no successful entries")
        return

    n_classes = len(ISRUC_DIAGNOSIS_CLASSES)
    aggregations = ["mean"]
    agg_labels = {"mean": "Mean", "mean_std": "Mean+Std"}

    models = []
    for mk in MODEL_ORDER:
        if any(e["checkpoint_id"] == mk for e in ok_entries):
            models.append(mk)
    if not models:
        print("skip isruc_diagnosis_f1_per_class.png: no known models")
        return

    fig, axes = plt.subplots(1, n_classes, figsize=(4.5 * n_classes, 5),
                             constrained_layout=True)
    if n_classes == 1:
        axes = [axes]

    bar_w = 0.8 / len(aggregations)
    x = np.arange(len(models))

    for cls_idx, cls_name in enumerate(ISRUC_DIAGNOSIS_CLASSES):
        ax = axes[cls_idx]
        for agg_i, agg in enumerate(aggregations):
            means, stds = [], []
            for mk in models:
                fold_f1s = []
                for e in ok_entries:
                    if (e["checkpoint_id"] == mk
                            and (e.get("metadata") or {}).get("aggregation") == agg):
                        f1_list = (e.get("metrics") or {}).get("best_test", {}).get("f1_per_class", [])
                        if cls_idx < len(f1_list):
                            fold_f1s.append(f1_list[cls_idx])
                means.append(float(np.mean(fold_f1s)) if fold_f1s else np.nan)
                stds.append(float(np.std(fold_f1s)) if fold_f1s else 0.0)

            offset = (agg_i - (len(aggregations) - 1) / 2) * bar_w
            ax.bar(
                x + offset, means, bar_w, yerr=stds, capsize=3,
                label=agg_labels[agg],
                color=["#4C72B0", "#DD8452"][agg_i],
                edgecolor="black", linewidth=0.4,
            )
            for xi, val in zip(x + offset, means):
                if not np.isnan(val):
                    ax.text(xi, val + 0.02, f"{val:.2f}",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                           rotation=20, ha="right")
        ax.set_ylabel("F1")
        ax.set_title(cls_name.replace("_", " ").title())
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if cls_idx == 0:
            ax.legend(title="Aggregation", fontsize=8)

    fig.suptitle(
        "ISRUC Diagnosis — F1 per class (subject-level, 5-fold, test)",
        fontsize=14, fontweight="bold",
    )
    _save(fig, "isruc_diagnosis_f1_per_class.png")


# ═══════════════════════════════════════════════════════════════════════════
# sklearn_linear plots — EEG foundation models
# ═══════════════════════════════════════════════════════════════════════════


def _sk_populated_datasets(path_map: Dict[str, str],
                           datasets: List[str]) -> List[str]:
    return [ds for ds in datasets
            if sk_sleep_metric(path_map, ds, "cohen_kappa")]


# ---------------------------------------------------------------------------
# SK sleep staging: violin, dataset-grouped bar, grouped bar, F1-per-stage,
# spider
# ---------------------------------------------------------------------------


def _pastel(color, factor=0.4):
    import matplotlib.colors as mcolors
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * factor)


def _plot_sk_sleep_violin(
    metric_key: str, ylabel: str, filename: str,
    path_map: Dict[str, str], datasets: List[str], title_prefix: str,
) -> None:
    populated = _sk_populated_datasets(path_map, datasets)
    if not populated:
        print(f"skip {filename}: no data")
        return

    data = {ds: sk_sleep_metric(path_map, ds, metric_key) for ds in populated}

    model_vals: Dict[str, List[float]] = {}
    for mk in SK_ALL_MODEL_ORDER:
        vals = []
        for ds in populated:
            ds_data = data[ds].get(mk, [])
            if ds_data:
                vals.append(float(np.mean(ds_data)))
        if vals:
            model_vals[mk] = vals

    eeg_models = [mk for mk in SK_MODEL_ORDER if mk in model_vals]
    eeg_models.sort(key=lambda mk: -np.median(model_vals[mk]))

    tsfm_fams: List[Tuple[str, List[str], float]] = []
    for fam_name, fam_members in TSFM_FAMILIES:
        active = [mk for mk in fam_members if mk in model_vals]
        if active:
            active.sort(key=lambda mk: -np.median(model_vals[mk]))
            tsfm_fams.append((fam_name, active, np.median(model_vals[active[0]])))
    tsfm_fams.sort(key=lambda t: -t[2])

    sup_models = [mk for mk in SUP_MODEL_ORDER if mk in model_vals]
    sup_models.sort(key=lambda mk: -np.median(model_vals[mk]))

    eeg_pos = list(range(len(eeg_models)))
    eeg_span = (eeg_pos[-1] - eeg_pos[0]) if len(eeg_pos) > 1 else 1.0

    tsfm_pos: List[float] = []
    tsfm_order: List[str] = []
    n_tsfm = sum(len(f[1]) for f in tsfm_fams)
    n_fam = len(tsfm_fams)
    if n_tsfm > 0:
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

    sup_pos: List[float] = []
    if sup_models:
        last = tsfm_pos[-1] if tsfm_pos else (eeg_pos[-1] if eeg_pos else -1)
        sup_gap = 2.0
        for si, mk in enumerate(sup_models):
            sup_pos.append(last + sup_gap + si)

    all_models = eeg_models + tsfm_order + sup_models
    all_pos = eeg_pos + tsfm_pos + sup_pos
    all_data = [model_vals[mk] for mk in all_models]

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(all_models) + 2), 5),
                           constrained_layout=True)

    for i, (mk, pi, vals) in enumerate(zip(all_models, all_pos, all_data)):
        base_color = MODEL_COLORS.get(mk, "#777")
        face = (_pastel(base_color, 0.45) if not _is_tsfm(mk)
                and not _is_sup(mk) else base_color)

        if len(vals) >= 3:
            vp = ax.violinplot([vals], positions=[pi], widths=0.7,
                               showmeans=False, showmedians=False,
                               showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(face)
                body.set_edgecolor("black")
                body.set_linewidth(0.6)
                body.set_alpha(0.75)

        rng = np.random.default_rng(hash(mk) % 2**32)
        jitter = rng.uniform(-0.1, 0.1, len(vals))
        ax.scatter(pi + jitter, vals, color=base_color, s=22, zorder=5,
                   edgecolors="black", linewidths=0.4, alpha=0.85)
        ax.scatter(pi, np.mean(vals), color="white", s=50, zorder=6,
                   edgecolors="black", linewidths=1.0, marker="D")

    _sections = [(eeg_pos, "EEG FMs"), (tsfm_pos, "TS FMs"),
                 (sup_pos, "Supervised")]
    prev_end = None
    for sec_pos, sec_label in _sections:
        if not sec_pos:
            continue
        if prev_end is not None:
            sep_x = (prev_end + sec_pos[0]) / 2
            ax.axvline(sep_x, color="gray", ls="--", lw=1, alpha=0.5)
        mid_x = (sec_pos[0] + sec_pos[-1]) / 2
        ax.text(mid_x, 0.97, sec_label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, fontweight="bold")
        prev_end = sec_pos[-1]

    ax.set_xticks(all_pos)
    ax.set_xticklabels([MODEL_LABELS.get(mk, mk) for mk in all_models],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} — {ylabel} (sklearn_linear, test)",
                 fontsize=13, pad=12)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, filename)


def _plot_sk_sleep_lines(
    metric_key: str, ylabel: str, filename: str,
    path_map: Dict[str, str], datasets: List[str],
    ds_colors: Dict[str, str], title_prefix: str,
) -> None:
    populated = _sk_populated_datasets(path_map, datasets)
    if not populated:
        print(f"skip {filename}: no data")
        return

    data = {ds: sk_sleep_metric(path_map, ds, metric_key) for ds in populated}

    model_ds_vals: Dict[str, Dict[str, float]] = {}
    model_ds_std: Dict[str, Dict[str, float]] = {}
    model_all_vals: Dict[str, List[float]] = {}
    for mk in SK_ALL_MODEL_ORDER:
        per_ds: Dict[str, float] = {}
        per_ds_s: Dict[str, float] = {}
        for ds in populated:
            ds_data = data[ds].get(mk, [])
            if ds_data:
                per_ds[ds] = float(np.mean(ds_data))
                per_ds_s[ds] = float(np.std(ds_data))
        if per_ds:
            model_ds_vals[mk] = per_ds
            model_ds_std[mk] = per_ds_s
            model_all_vals[mk] = list(per_ds.values())

    eeg_models = [mk for mk in SK_MODEL_ORDER if mk in model_all_vals]
    eeg_models.sort(key=lambda mk: -np.median(model_all_vals[mk]))

    tsfm_fams: List[Tuple[str, List[str], float]] = []
    for fam_name, fam_members in TSFM_FAMILIES:
        active = [mk for mk in fam_members if mk in model_all_vals]
        if active:
            active.sort(key=lambda mk: -np.median(model_all_vals[mk]))
            tsfm_fams.append((fam_name, active,
                              np.median(model_all_vals[active[0]])))
    tsfm_fams.sort(key=lambda t: -t[2])

    eeg_pos = list(range(len(eeg_models)))
    eeg_span = (eeg_pos[-1] - eeg_pos[0]) if len(eeg_pos) > 1 else 1.0

    tsfm_pos: List[float] = []
    tsfm_order: List[str] = []
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

    sup_models_lines = [mk for mk in SUP_MODEL_ORDER if mk in model_all_vals]
    sup_models_lines.sort(key=lambda mk: -np.median(model_all_vals[mk]))

    sup_pos: List[float] = []
    if sup_models_lines:
        last = tsfm_pos[-1] if tsfm_pos else (eeg_pos[-1] if eeg_pos else -1)
        sup_gap = 2.0
        for si, mk in enumerate(sup_models_lines):
            sup_pos.append(last + sup_gap + si)

    all_models = eeg_models + tsfm_order + sup_models_lines
    all_pos = eeg_pos + tsfm_pos + sup_pos
    pos_lookup = dict(zip(all_models, all_pos))

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(all_models) + 2), 5),
                           constrained_layout=True)

    for ds in populated:
        color = ds_colors.get(ds, "#777")
        for group, label_once in [(eeg_models, True), (tsfm_order, False),
                                  (sup_models_lines, False)]:
            xs, ys, yerr = [], [], []
            for mk in group:
                v = model_ds_vals.get(mk, {}).get(ds)
                if v is not None:
                    xs.append(pos_lookup[mk])
                    ys.append(v)
                    yerr.append(model_ds_std.get(mk, {}).get(ds, 0.0))
            if not xs:
                continue
            lbl = ds.upper() if label_once else None
            ax.errorbar(xs, ys, yerr=yerr, marker="o", ms=5, lw=1.4,
                        color=color, alpha=0.8, zorder=3, label=lbl,
                        capsize=2, capthick=0.8, elinewidth=0.8)
            ax.scatter(xs, ys, s=20, color=color, edgecolors="black",
                       linewidths=0.3, zorder=4)

    _sections = [(eeg_pos, "EEG FMs"), (tsfm_pos, "TS FMs"),
                 (sup_pos, "Supervised")]
    prev_end = None
    for sec_pos, sec_label in _sections:
        if not sec_pos:
            continue
        if prev_end is not None:
            sep_x = (prev_end + sec_pos[0]) / 2
            ax.axvline(sep_x, color="gray", ls="--", lw=1, alpha=0.5)
        mid_x = (sec_pos[0] + sec_pos[-1]) / 2
        ax.text(mid_x, 0.97, sec_label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, fontweight="bold")
        prev_end = sec_pos[-1]

    ax.set_xticks(all_pos)
    ax.set_xticklabels([MODEL_LABELS.get(mk, mk) for mk in all_models],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} — {ylabel} (sklearn_linear, test)",
                 fontsize=13, pad=12)
    ax.grid(True, axis="y", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=4,
              loc="lower left")
    _save(fig, filename)


def _plot_sk_sleep_byds(
    metric_key: str, ylabel: str, filename: str,
    path_map: Dict[str, str], datasets: List[str],
    colors: Dict[str, str], title_prefix: str,
) -> None:
    populated = _sk_populated_datasets(path_map, datasets)
    if not populated:
        print(f"skip {filename}: no data")
        return
    data = {ds: sk_sleep_metric(path_map, ds, metric_key) for ds in populated}

    n_ds = len(populated)
    fig, ax = plt.subplots(figsize=(max(16, 2.5 * n_ds), 5.5))

    ds_spacing = 1.0
    seen_labels: set = set()

    for di, ds in enumerate(populated):
        ds_data = data[ds]
        eeg_ranked = sorted(
            [(mk, ds_data[mk]) for mk in SK_MODEL_ORDER if mk in ds_data],
            key=lambda t: -float(np.mean(t[1])))
        tsfm_ranked = sorted(
            [(mk, ds_data[mk]) for mk in TSFM_MODEL_ORDER if mk in ds_data],
            key=lambda t: -float(np.mean(t[1])))
        sup_ranked = sorted(
            [(mk, ds_data[mk]) for mk in SUP_MODEL_ORDER if mk in ds_data],
            key=lambda t: -float(np.mean(t[1])))

        all_bars = eeg_ranked + tsfm_ranked + sup_ranked
        n_bars = len(all_bars)
        if n_bars == 0:
            continue
        group_width = 0.85
        bar_w = group_width / max(n_bars + 0.3, 1)
        gap_at_1 = len(eeg_ranked)
        gap_at_2 = len(eeg_ranked) + len(tsfm_ranked)

        for bi, (mk, vals) in enumerate(all_bars):
            extra = (bar_w * 0.4 * (1 if bi >= gap_at_1 else 0)
                     + bar_w * 0.4 * (1 if bi >= gap_at_2 else 0))
            offset = (bi - (n_bars - 1) / 2) * bar_w + extra
            m = float(np.mean(vals))
            s = float(np.std(vals))
            hatch = ("///" if _is_tsfm(mk) else
                     "\\\\" if _is_sup(mk) else None)
            lbl_key = MODEL_LABELS.get(mk, mk)
            lbl = lbl_key if lbl_key not in seen_labels else None
            if lbl:
                seen_labels.add(lbl_key)
            ax.bar(di * ds_spacing + offset, m, bar_w, yerr=s, capsize=1.5,
                   color=MODEL_COLORS.get(mk, "#777"), edgecolor="black",
                   lw=0.3, hatch=hatch, label=lbl)

    ax.set_xticks([i * ds_spacing for i in range(n_ds)])
    ax.set_xticklabels([ds.upper() for ds in populated], fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} — {ylabel} (sklearn_linear, test)",
                 fontsize=13)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ref_key = metric_key.replace("cohen_kappa", "sleep_kappa").replace(
        "macro_f1", "sleep_macro_f1")
    _draw_refs(ax, human=HUMAN.get(ref_key), clinical=CLINICAL.get(ref_key))
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=5,
              loc="lower left")
    _save(fig, filename)


def _plot_sk_sleep_byds_mixed(
    metric_key: str, ylabel: str, filename: str,
    path_map: Dict[str, str], datasets: List[str],
    colors: Dict[str, str], title_prefix: str,
) -> None:
    populated = _sk_populated_datasets(path_map, datasets)
    if not populated:
        print(f"skip {filename}: no data")
        return
    data = {ds: sk_sleep_metric(path_map, ds, metric_key) for ds in populated}

    n_ds = len(populated)
    fig, ax = plt.subplots(figsize=(max(16, 2.5 * n_ds), 5.5))

    ds_spacing = 1.0
    seen_labels: set = set()

    for di, ds in enumerate(populated):
        ds_data = data[ds]
        ranked = sorted(
            [(mk, ds_data[mk]) for mk in SK_ALL_MODEL_ORDER if mk in ds_data],
            key=lambda t: -float(np.mean(t[1])))

        n_bars = len(ranked)
        if n_bars == 0:
            continue
        group_width = 0.85
        bar_w = group_width / max(n_bars, 1)

        for bi, (mk, vals) in enumerate(ranked):
            offset = (bi - (n_bars - 1) / 2) * bar_w
            m = float(np.mean(vals))
            s = float(np.std(vals))
            hatch = ("///" if _is_tsfm(mk) else
                     "\\\\" if _is_sup(mk) else None)
            lbl_key = MODEL_LABELS.get(mk, mk)
            lbl = lbl_key if lbl_key not in seen_labels else None
            if lbl:
                seen_labels.add(lbl_key)
            ax.bar(di * ds_spacing + offset, m, bar_w, yerr=s, capsize=1.5,
                   color=MODEL_COLORS.get(mk, "#777"), edgecolor="black",
                   lw=0.3, hatch=hatch, label=lbl)

    ax.set_xticks([i * ds_spacing for i in range(n_ds)])
    ax.set_xticklabels([ds.upper() for ds in populated], fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} — {ylabel} (sklearn_linear, test) "
                 "[ranked mixed]", fontsize=13)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ref_key = metric_key.replace("cohen_kappa", "sleep_kappa").replace(
        "macro_f1", "sleep_macro_f1")
    _draw_refs(ax, human=HUMAN.get(ref_key), clinical=CLINICAL.get(ref_key))
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=5,
              loc="lower left")
    _save(fig, filename)


def _plot_sk_sleep_grouped_bar(
    metric_key: str, ylabel: str, filename: str,
    path_map: Dict[str, str], datasets: List[str],
    colors: Dict[str, str], title_prefix: str,
) -> None:
    populated = _sk_populated_datasets(path_map, datasets)
    if not populated:
        print(f"skip {filename}: no data")
        return
    data = {ds: sk_sleep_metric(path_map, ds, metric_key) for ds in populated}
    n_eeg = len(SK_MODEL_ORDER)
    n_tsfm = len(TSFM_MODEL_ORDER)
    n_sup = len(SUP_MODEL_ORDER)
    gap = 0.7
    x_eeg = np.arange(n_eeg)
    x_tsfm = np.arange(n_tsfm) + n_eeg + gap
    x_sup = np.arange(n_sup) + n_eeg + gap + n_tsfm + gap
    x = np.concatenate([x_eeg, x_tsfm, x_sup])
    n_ds = len(populated)
    bar_w = 0.8 / max(n_ds, 1)

    fig, ax = plt.subplots(figsize=(max(14, 1.2 * len(SK_ALL_MODEL_ORDER)), 5))
    for grp_models, grp_x, hatch in [
        (SK_MODEL_ORDER, x_eeg, None),
        (TSFM_MODEL_ORDER, x_tsfm, "///"),
        (SUP_MODEL_ORDER, x_sup, "\\\\"),
    ]:
        for i, ds in enumerate(populated):
            means, stds = [], []
            for mk in grp_models:
                vals = data[ds].get(mk, [])
                if vals:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))
                else:
                    means.append(np.nan)
                    stds.append(0.0)
            offset = (i - (n_ds - 1) / 2) * bar_w
            ax.bar(
                grp_x + offset, means, bar_w,
                yerr=stds, capsize=2.5,
                label=ds.upper() if hatch is None else None,
                color=colors.get(ds, "#777"), edgecolor="black", linewidth=0.4,
                hatch=hatch,
            )
    ax.axvline(n_eeg - 0.5 + gap / 2, color="#aaa", linestyle=":", lw=0.8,
               zorder=0)
    ax.axvline(n_eeg + gap + n_tsfm - 0.5 + gap / 2, color="#aaa",
               linestyle=":", lw=0.8, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in SK_ALL_MODEL_ORDER],
                       rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix} — {ylabel} (sklearn_linear, test)")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="dataset", loc="lower left", fontsize=8, ncol=2)
    ref_key = metric_key.replace("cohen_kappa", "sleep_kappa").replace(
        "macro_f1", "sleep_macro_f1")
    _draw_refs(ax, human=HUMAN.get(ref_key), clinical=CLINICAL.get(ref_key))
    _save(fig, filename)


def plot_sk_sleep_grouped_bar(metric_key: str, ylabel: str,
                              filename: str) -> None:
    _plot_sk_sleep_grouped_bar(
        metric_key, ylabel, filename,
        SK_SLEEP_RESULTS_PATH, SK_SLEEP_DATASETS, SK_DATASET_COLORS,
        "Sleep staging",
    )


def plot_sk_dod_subgroup(metric_key: str, ylabel: str,
                         filename: str) -> None:
    _plot_sk_sleep_grouped_bar(
        metric_key, ylabel, filename,
        SK_DOD_SUBGROUP_PATHS, SK_DOD_SUBGROUPS, SK_DATASET_COLORS,
        "DOD subgroups — sleep staging",
    )


def plot_sk_dod_scorers(ds: str, metric_key: str, ylabel: str,
                        filename: str) -> None:
    ds_label = {"dod": "DOD", "dod_dodh": "DOD-H",
                "dod_dodo": "DOD-O"}.get(ds, ds)
    _plot_sk_sleep_grouped_bar(
        metric_key, ylabel, filename,
        SK_DOD_SCORER_PATHS[ds], SK_DOD_SCORERS, SK_DOD_SCORER_COLORS,
        f"{ds_label} scorers — sleep staging",
    )


def plot_sk_sleepedf_subgroup(metric_key: str, ylabel: str,
                              filename: str) -> None:
    _plot_sk_sleep_grouped_bar(
        metric_key, ylabel, filename,
        SK_SLEEPEDF_SUBGROUP_PATHS, SK_SLEEPEDF_SUBGROUPS, SK_DATASET_COLORS,
        "SleepEDF subgroups — sleep staging",
    )


def plot_sk_sleep_f1_per_stage() -> None:
    datasets = _sk_populated_datasets(SK_SLEEP_RESULTS_PATH, SK_SLEEP_DATASETS)
    if not datasets:
        print("skip sk_sleep_f1_per_stage.png: no data")
        return
    by_ds = {ds: sk_sleep_f1_per_stage(SK_SLEEP_RESULTS_PATH, ds)
             for ds in datasets}
    x = np.arange(len(datasets))
    ds_labels = [ds.upper() for ds in datasets]

    n_stages = len(STAGE_LABELS)
    n_cols = 3
    n_rows = (n_stages + n_cols) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows),
                             constrained_layout=True)
    for stage_idx, stage in enumerate(STAGE_LABELS):
        ax = axes.flat[stage_idx]
        for mk in SK_ALL_MODEL_ORDER:
            means, stds = [], []
            for ds in datasets:
                mat = by_ds[ds].get(mk)
                if mat is None:
                    means.append(np.nan); stds.append(0.0)
                else:
                    means.append(float(mat[:, stage_idx].mean()))
                    stds.append(float(mat[:, stage_idx].std()))
            m_arr = np.asarray(means); s_arr = np.asarray(stds)
            ls = "--" if _is_tsfm(mk) else (":" if _is_sup(mk) else "-")
            mrk = "s" if _is_tsfm(mk) else ("^" if _is_sup(mk) else "o")
            ax.plot(x, m_arr, marker=mrk, color=MODEL_COLORS.get(mk, "#777"),
                    label=MODEL_LABELS.get(mk, mk), linewidth=1.8,
                    linestyle=ls)
            ax.fill_between(x, m_arr - s_arr, m_arr + s_arr,
                            color=MODEL_COLORS.get(mk, "#777"), alpha=0.1)
        ax.set_xticks(x)
        ax.set_xticklabels(ds_labels, fontsize=8, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("F1")
        ax.set_title(f"Stage {stage}")
        ax.grid(True, alpha=0.3)
        _draw_refs(ax, human=HUMAN["sleep_f1_per_stage"].get(stage),
                   human_label="human")
    for idx in range(n_stages, n_rows * n_cols):
        legend_ax = axes.flat[idx]
        legend_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", fontsize=10,
                         title="model", title_fontsize=11, frameon=False)
    fig.suptitle("Sleep staging — F1 per class (sklearn_linear, test)",
                 fontsize=14)
    _save(fig, "sk_sleep_f1_per_stage.png")


def plot_sk_sleep_kappa_spider() -> None:
    datasets = _sk_populated_datasets(SK_SLEEP_RESULTS_PATH, SK_SLEEP_DATASETS)
    if not datasets:
        print("skip sk_sleep_kappa_spider.png: no data")
        return
    data = {ds: sk_sleep_metric(SK_SLEEP_RESULTS_PATH, ds, "cohen_kappa")
            for ds in datasets}
    n_axes = len(datasets)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9),
                           subplot_kw=dict(projection="polar"))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(180 / n_axes)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                       fontsize=8, color="#555")
    ax.grid(True, alpha=0.3)

    human = HUMAN["sleep_kappa"]
    clinical = CLINICAL["sleep_kappa"]
    theta_dense = np.linspace(0, 2 * np.pi, 361)
    ax.plot(theta_dense, np.full_like(theta_dense, human),
            **_HUMAN_STYLE, label=f"human κ = {human:.2f}")
    ax.plot(theta_dense, np.full_like(theta_dense, clinical),
            **_CLINICAL_STYLE, label=f"clinical κ = {clinical:.2f}")

    all_values: Dict[str, List[float]] = {}
    for mk in SK_ALL_MODEL_ORDER:
        values = []
        for ds in datasets:
            v = data[ds].get(mk, [])
            values.append(float(np.mean(v)) if v else np.nan)
        if all(np.isnan(v) for v in values):
            continue
        all_values[mk] = values
        plotted = [v if not np.isnan(v) else 0 for v in values]
        plotted_closed = plotted + plotted[:1]
        ls = "--" if _is_tsfm(mk) else (":" if _is_sup(mk) else "-")
        mrk = "s" if _is_tsfm(mk) else ("^" if _is_sup(mk) else "o")
        ax.plot(angles_closed, plotted_closed, marker=mrk,
                color=MODEL_COLORS.get(mk, "#777"), linewidth=1.8,
                label=MODEL_LABELS.get(mk, mk), linestyle=ls)
        _alpha = 0.04 if _is_tsfm(mk) else (0.06 if _is_sup(mk) else 0.08)
        ax.fill(angles_closed, plotted_closed,
                color=MODEL_COLORS.get(mk, "#777"), alpha=_alpha)

    for ds_idx, ds in enumerate(datasets):
        best_mk, best_val = None, -np.inf
        for mk, vals in all_values.items():
            v = vals[ds_idx]
            if not np.isnan(v) and v > best_val:
                best_val = v
                best_mk = mk
        if best_mk is not None:
            ax.annotate(
                f"{best_val:.2f}",
                xy=(angles[ds_idx], best_val),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, fontweight="bold",
                color=MODEL_COLORS.get(best_mk, "#333"),
            )

    ax.set_xticks(angles)
    ax.set_xticklabels([ds.upper() for ds in datasets], fontsize=10)
    ax.set_title("Sleep staging — Cohen's κ (sklearn_linear)",
                 fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9,
              title="model", title_fontsize=10)
    _save(fig, "sk_sleep_kappa_spider.png")


def _load_intervention_csv() -> Dict[str, Dict[str, List[float]]]:
    """Load intervention results from CSV. Returns {metric: {model: [values]}}."""
    csv_path = (REPO / "artifacts/benchmarks"
                / "sleep_edf_expanded_intervention/sklearn_linear/results.csv")
    if not csv_path.exists():
        return {}
    import csv
    metrics_out: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["checkpoint_id"]
            if cid in SK_EXCLUDE:
                continue
            if row.get("status") != "ok":
                continue
            for col, mkey in [
                ("test_macro_f1", "macro_f1"),
                ("test_cohen_kappa", "cohen_kappa"),
                ("test_accuracy", "accuracy"),
            ]:
                val = row.get(col)
                if val:
                    metrics_out[mkey][cid].append(float(val))
    # Also pull auroc/auprc/mcc from JSON best_test (available there)
    json_path = (REPO / "artifacts/benchmarks"
                 / SK_SLEEPEDF_INTERVENTION_PATH)
    for e in _sk_load_entries(json_path, "linear_probe_eval"):
        bt = e.get("metrics", {}).get("best_test", {})
        cid = e["checkpoint_id"]
        for mkey in ("auroc", "auprc", "mcc"):
            if mkey in bt and bt[mkey] is not None:
                metrics_out[mkey][cid].append(float(bt[mkey]))
    return dict(metrics_out)


def plot_sk_sleepedf_intervention() -> None:
    all_metrics = _load_intervention_csv()
    if not all_metrics:
        print("skip sk_sleepedf_intervention.png: no data")
        return
    metric_keys = ["auroc", "auprc", "mcc", "macro_f1", "cohen_kappa",
                   "accuracy"]
    metric_labels = ["AUROC", "AUPRC", "MCC", "macro-F1", "Cohen's κ",
                     "Accuracy"]
    baselines = [0.5, None, 0.0, 0.5, 0.0, 0.5]

    fig, axes = plt.subplots(1, len(metric_keys),
                             figsize=(4.5 * len(metric_keys), 5),
                             constrained_layout=True)
    for ax, mkey, mlabel, base in zip(axes, metric_keys, metric_labels,
                                      baselines):
        vals = all_metrics.get(mkey, {})
        if not vals:
            ax.axis("off")
            ax.set_title(f"{mlabel} (no data)")
            continue
        _sk = lambda m: -float(np.mean(vals[m]))
        eeg_sorted = sorted(
            [m for m in vals if not _is_tsfm(m) and not _is_sup(m)], key=_sk)
        tsfm_sorted = sorted([m for m in vals if _is_tsfm(m)], key=_sk)
        sup_sorted = sorted([m for m in vals if _is_sup(m)], key=_sk)
        ordering = eeg_sorted + tsfm_sorted + sup_sorted
        means = [float(np.mean(vals[m])) for m in ordering]
        stds = [float(np.std(vals[m])) for m in ordering]
        colors = [MODEL_COLORS.get(m, "#777") for m in ordering]
        hatches = [("///" if _is_tsfm(m) else
                    "\\\\" if _is_sup(m) else None) for m in ordering]
        n_e = len(eeg_sorted)
        n_t = len(tsfm_sorted)
        n_s = len(sup_sorted)
        gap = 0.7
        groups = []
        pos = 0
        if n_e:
            groups.append(np.arange(n_e) + pos)
            pos += n_e + (gap if (n_t or n_s) else 0)
        if n_t:
            groups.append(np.arange(n_t) + pos)
            pos += n_t + (gap if n_s else 0)
        if n_s:
            groups.append(np.arange(n_s) + pos)
        xs = np.concatenate(groups) if groups else np.arange(len(ordering))
        bars = ax.bar(xs, means, color=colors, yerr=stds, capsize=3,
                      edgecolor="black", linewidth=0.4)
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)
        ax.set_xticks(xs)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in ordering],
                           rotation=25, ha="right")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        if base is not None:
            ax.axhline(base, color="black", linestyle="--", lw=1)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("SleepEDF Telemetry — Drug Use Detection (sklearn_linear)",
                 fontsize=14)
    _save(fig, "sk_sleepedf_intervention.png")


def plot_sk_mass_subsets(metric_key: str, ylabel: str,
                         filename: str) -> None:
    _plot_sk_sleep_grouped_bar(
        metric_key, ylabel, filename,
        SK_MASS_SUBSET_PATHS, SK_MASS_SUBSETS, SK_DATASET_COLORS,
        "MASS subsets — sleep staging",
    )


def plot_sk_isruc_scorers(metric_key: str, ylabel: str,
                          filename: str) -> None:
    _plot_sk_sleep_grouped_bar(
        metric_key, ylabel, filename,
        SK_ISRUC_SCORER_PATHS, SK_ISRUC_SCORERS, SK_DATASET_COLORS,
        "ISRUC scorers — sleep staging",
    )


def plot_sk_isruc_subgroups(metric_key: str, ylabel: str,
                            filename: str) -> None:
    _plot_sk_sleep_grouped_bar(
        metric_key, ylabel, filename,
        SK_ISRUC_SUBGROUP_PATHS, SK_ISRUC_SUBGROUPS, SK_DATASET_COLORS,
        "ISRUC subject groups — sleep staging",
    )


def plot_sk_pn2026_sites(metric_key: str, ylabel: str,
                         filename: str) -> None:
    _plot_sk_sleep_grouped_bar(
        metric_key, ylabel, filename,
        SK_PN2026_SITE_PATHS, SK_PN2026_SITES, SK_DATASET_COLORS,
        "PhysioNet2026 sites — sleep staging",
    )


# ---------------------------------------------------------------------------
# SK arousal detection
# ---------------------------------------------------------------------------

def plot_sk_arousal_sweep() -> None:
    datasets = list(SK_AROUSAL_PATHS.keys())
    probes = [("sklearn_linear", "non-balanced"),
              ("sklearn_linear_balanced", "balanced")]
    fig, axes = plt.subplots(len(datasets), len(probes),
                             figsize=(10 * len(probes), 5.5 * len(datasets)),
                             constrained_layout=True, squeeze=False)
    for row, ds in enumerate(datasets):
        for col, (probe, label) in enumerate(probes):
            ax = axes[row, col]
            path = REPO / "artifacts/benchmarks" / SK_AROUSAL_PATHS[ds].format(
                probe=probe)
            per_model: Dict[str, Dict[float, Tuple[float, float]]] = (
                defaultdict(dict))
            for t in AROUSAL_THRESHOLDS:
                thr_key = f"threshold_{t}s"
                vals = sk_event_metric(path, None, thr_key, "auprc")
                for mk, v in vals.items():
                    per_model[mk][t] = (float(np.mean(v)), float(np.std(v)))
            for mk in SK_ALL_MODEL_ORDER:
                if mk not in per_model:
                    continue
                xs, ys, ses = [], [], []
                for t in AROUSAL_THRESHOLDS:
                    if t in per_model[mk]:
                        m, s = per_model[mk][t]
                        xs.append(t); ys.append(m); ses.append(s)
                if not xs:
                    continue
                xs_a = np.array(xs); ys_a = np.array(ys); ses_a = np.array(ses)
                ls = "--" if _is_tsfm(mk) else "-"
                mrk = "s" if _is_tsfm(mk) else ("^" if _is_sup(mk) else "o")
                ax.plot(xs_a, ys_a, marker=mrk,
                        color=MODEL_COLORS.get(mk, "#777"),
                        label=MODEL_LABELS.get(mk, mk), linewidth=1.8,
                        linestyle=ls)
                ax.fill_between(xs_a, ys_a - ses_a, ys_a + ses_a,
                                color=MODEL_COLORS.get(mk, "#777"), alpha=0.12)
            ax.axvline(AROUSAL_CLINICAL, color="black", linestyle="--", lw=1)
            ax.set_xlabel("threshold (s)")
            ax.set_ylabel("AUPRC")
            ax.set_title(f"{ds} — {label}")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="best", ncol=2)
    fig.suptitle("Arousal detection — AUPRC vs threshold (sklearn_linear)",
                 fontsize=14)
    _save(fig, "sk_arousal_auprc_vs_threshold.png")


def plot_sk_arousal_bar_3s() -> None:
    datasets = list(SK_AROUSAL_PATHS.keys())
    probes = [("sklearn_linear", "non-balanced"),
              ("sklearn_linear_balanced", "balanced")]
    metric_keys = ["auprc", "auroc", "mcc", "macro_f1"]
    metric_labels = ["AUPRC", "AUROC", "MCC", "macro-F1"]

    fig, axes = plt.subplots(len(datasets), len(metric_keys),
                             figsize=(5 * len(metric_keys),
                                      4.5 * len(datasets)),
                             constrained_layout=True, squeeze=False)
    thr_key = f"threshold_{AROUSAL_CLINICAL}s"
    for row, ds in enumerate(datasets):
        variant_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        for probe, label in probes:
            path = REPO / "artifacts/benchmarks" / SK_AROUSAL_PATHS[ds].format(
                probe=probe)
            by_metric: Dict[str, Dict[str, List[float]]] = {}
            for mkey in metric_keys:
                by_metric[mkey] = sk_event_metric(path, None, thr_key, mkey)
            variant_data[label] = by_metric

        for col, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
            ax = axes[row, col]
            vals_by_variant = {}
            for label, by_metric in variant_data.items():
                if by_metric[mkey]:
                    vals_by_variant[label] = by_metric[mkey]
            if not vals_by_variant:
                ax.axis("off")
                ax.set_title(f"{ds} — {mlabel} (no data)")
                continue
            base = (0.5 if mkey in ("auroc", "macro_f1") else
                    0.0 if mkey == "mcc" else None)
            _paired_bar_panel(ax, vals_by_variant, baseline=base, ylabel=mlabel)
            ax.set_title(f"{ds} — {mlabel}")
            if col == 0 and row == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Arousal @ {AROUSAL_CLINICAL}s — sklearn_linear", fontsize=14)
    _save(fig, "sk_arousal_bar_3s.png")


# ---------------------------------------------------------------------------
# SK limb movement detection
# ---------------------------------------------------------------------------

def plot_sk_limb_sweep() -> None:
    probes = [("sklearn_linear", "non-balanced"),
              ("sklearn_linear_balanced", "balanced")]
    fig, axes = plt.subplots(len(LIMB_EVENT_TYPES), len(probes),
                             figsize=(10 * len(probes),
                                      5.5 * len(LIMB_EVENT_TYPES)),
                             constrained_layout=True, squeeze=False)
    for row, (evt_label, evt_field) in enumerate(LIMB_EVENT_TYPES):
        for col, (probe, label) in enumerate(probes):
            ax = axes[row, col]
            path = REPO / f"artifacts/benchmarks/physionet2026_limb/{probe}/results.json"
            per_model: Dict[str, Dict[float, Tuple[float, float]]] = (
                defaultdict(dict))
            for t in LIMB_THRESHOLDS:
                thr_key = f"threshold_{t}s"
                vals = sk_event_metric(path, evt_field, thr_key, "auprc")
                for mk, v in vals.items():
                    per_model[mk][t] = (float(np.mean(v)), float(np.std(v)))
            for mk in SK_ALL_MODEL_ORDER:
                if mk not in per_model:
                    continue
                xs, ys, ses = [], [], []
                for t in LIMB_THRESHOLDS:
                    if t in per_model[mk]:
                        m, s = per_model[mk][t]
                        xs.append(t); ys.append(m); ses.append(s)
                if not xs:
                    continue
                xs_a = np.array(xs); ys_a = np.array(ys); ses_a = np.array(ses)
                ls = "--" if _is_tsfm(mk) else "-"
                mrk = "s" if _is_tsfm(mk) else ("^" if _is_sup(mk) else "o")
                ax.plot(xs_a, ys_a, marker=mrk,
                        color=MODEL_COLORS.get(mk, "#777"),
                        label=MODEL_LABELS.get(mk, mk), linewidth=1.8,
                        linestyle=ls)
                ax.fill_between(xs_a, ys_a - ses_a, ys_a + ses_a,
                                color=MODEL_COLORS.get(mk, "#777"), alpha=0.12)
            ax.axvline(LIMB_CLINICAL, color="black", linestyle="--", lw=1)
            ax.set_xlabel("threshold (s)")
            ax.set_ylabel("AUPRC")
            ax.set_title(f"{evt_label} — {label}")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="best", ncol=2)
    fig.suptitle("Limb movement — AUPRC vs threshold (sklearn_linear)",
                 fontsize=14)
    _save(fig, "sk_limb_auprc_vs_threshold.png")


def plot_sk_limb_bar_05s() -> None:
    probes = [("sklearn_linear", "non-balanced"),
              ("sklearn_linear_balanced", "balanced")]
    metric_keys = ["auprc", "auroc", "mcc", "macro_f1"]
    metric_labels = ["AUPRC", "AUROC", "MCC", "macro-F1"]
    limb_thr = 0.5
    thr_key = f"threshold_{limb_thr}s"

    fig, axes = plt.subplots(len(LIMB_EVENT_TYPES), len(metric_keys),
                             figsize=(5 * len(metric_keys),
                                      4.5 * len(LIMB_EVENT_TYPES)),
                             constrained_layout=True, squeeze=False)
    for row, (evt_label, evt_field) in enumerate(LIMB_EVENT_TYPES):
        variant_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        for probe, label in probes:
            path = REPO / f"artifacts/benchmarks/physionet2026_limb/{probe}/results.json"
            by_metric: Dict[str, Dict[str, List[float]]] = {}
            for mkey in metric_keys:
                by_metric[mkey] = sk_event_metric(
                    path, evt_field, thr_key, mkey)
            variant_data[label] = by_metric

        for col, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
            ax = axes[row, col]
            vals_by_variant = {}
            for label, by_metric in variant_data.items():
                if by_metric[mkey]:
                    vals_by_variant[label] = by_metric[mkey]
            if not vals_by_variant:
                ax.axis("off")
                ax.set_title(f"{evt_label} — {mlabel} (no data)")
                continue
            base = (0.5 if mkey in ("auroc", "macro_f1") else
                    0.0 if mkey == "mcc" else None)
            _paired_bar_panel(ax, vals_by_variant, baseline=base, ylabel=mlabel)
            ax.set_title(f"{evt_label} — {mlabel}")
            if col == 0 and row == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Limb movement @ {limb_thr}s — sklearn_linear",
                 fontsize=14)
    _save(fig, "sk_limb_bar_05s.png")


# ---------------------------------------------------------------------------
# SK respiratory event detection
# ---------------------------------------------------------------------------

def plot_sk_resp_sweep(ds_slug: str, ds_label: str,
                       event_types: List[Tuple[str, str]],
                       filename: str) -> None:
    probes = [("sklearn_linear", "non-balanced"),
              ("sklearn_linear_balanced", "balanced")]
    n_evt = min(len(event_types), 4)
    n_cols = 2
    n_rows = (n_evt + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(10 * n_cols, 5 * n_rows),
                             constrained_layout=True, squeeze=False)
    for evt_idx, (evt_label, evt_field) in enumerate(event_types[:4]):
        ax = axes.flat[evt_idx]
        for probe, plabel in probes:
            path = REPO / f"artifacts/benchmarks/{ds_slug}/{probe}/results.json"
            ls = "-" if plabel == "non-balanced" else "--"
            per_model: Dict[str, Dict[float, Tuple[float, float]]] = (
                defaultdict(dict))
            for t in RESP_THRESHOLDS:
                thr_key = f"threshold_{t}s"
                vals = sk_event_metric(path, evt_field, thr_key, "auprc")
                for mk, v in vals.items():
                    per_model[mk][t] = (float(np.mean(v)), float(np.std(v)))
            for mk in SK_ALL_MODEL_ORDER:
                if mk not in per_model:
                    continue
                xs, ys, ses = [], [], []
                for t in RESP_THRESHOLDS:
                    if t in per_model[mk]:
                        m, s = per_model[mk][t]
                        xs.append(t); ys.append(m); ses.append(s)
                if not xs:
                    continue
                xs_a = np.array(xs); ys_a = np.array(ys)
                lbl = (MODEL_LABELS.get(mk, mk)
                       if plabel == "non-balanced" else None)
                mrk = "s" if _is_tsfm(mk) else ("^" if _is_sup(mk) else "o")
                lw = 1.2 if (_is_tsfm(mk) or _is_sup(mk)) else 1.5
                ax.plot(xs_a, ys_a, marker=mrk,
                        color=MODEL_COLORS.get(mk, "#777"),
                        label=lbl, linewidth=lw, linestyle=ls)
        ax.axvline(RESP_CLINICAL, color="black", linestyle="--", lw=1)
        ax.set_title(evt_label)
        ax.set_xlabel("threshold (s)")
        ax.set_ylabel("AUPRC")
        ax.grid(True, alpha=0.3)
        if evt_idx == 0:
            ax.legend(fontsize=7, loc="best", ncol=2)
    for idx in range(n_evt, n_rows * n_cols):
        axes.flat[idx].axis("off")
    fig.suptitle(f"{ds_label} respiratory events — AUPRC sweep "
                 f"(solid=non-balanced, dashed=balanced)", fontsize=14)
    _save(fig, filename)


def plot_sk_resp_bar_10s(ds_slug: str, ds_label: str,
                         event_types: List[Tuple[str, str]],
                         filename: str) -> None:
    probes = [("sklearn_linear", "non-balanced"),
              ("sklearn_linear_balanced", "balanced")]
    metric_keys = ["auprc", "auroc", "mcc", "macro_f1"]
    metric_labels = ["AUPRC", "AUROC", "MCC", "macro-F1"]
    thr_key = f"threshold_{RESP_CLINICAL}s"
    n_evt = min(len(event_types), 4)

    fig, axes = plt.subplots(n_evt, len(metric_keys),
                             figsize=(5 * len(metric_keys), 4.5 * n_evt),
                             constrained_layout=True, squeeze=False)
    for row, (evt_label, evt_field) in enumerate(event_types[:4]):
        variant_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        for probe, label in probes:
            path = REPO / f"artifacts/benchmarks/{ds_slug}/{probe}/results.json"
            by_metric: Dict[str, Dict[str, List[float]]] = {}
            for mkey in metric_keys:
                by_metric[mkey] = sk_event_metric(
                    path, evt_field, thr_key, mkey)
            variant_data[label] = by_metric

        for col, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
            ax = axes[row, col]
            vals_by_variant = {}
            for label, by_metric in variant_data.items():
                if by_metric[mkey]:
                    vals_by_variant[label] = by_metric[mkey]
            if not vals_by_variant:
                ax.axis("off")
                ax.set_title(f"{evt_label}\n{mlabel} (no data)")
                continue
            base = (0.5 if mkey in ("auroc", "macro_f1") else
                    0.0 if mkey == "mcc" else None)
            _paired_bar_panel(ax, vals_by_variant, baseline=base, ylabel=mlabel)
            ax.set_title(f"{evt_label}\n{mlabel}")
            if col == 0 and row == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"{ds_label} respiratory @ {RESP_CLINICAL}s — sklearn_linear",
                 fontsize=14)
    _save(fig, filename)


# ---------------------------------------------------------------------------
# SK diagnosis / patient classification
# ---------------------------------------------------------------------------

def plot_sk_dod_osa_bar() -> None:
    probes = [("sklearn_linear", "non-balanced"),
              ("sklearn_linear_balanced", "balanced")]
    metric_keys = ("auroc", "auprc", "mcc", "macro_f1")
    metric_labels = ["AUROC", "AUPRC", "MCC", "macro-F1"]

    variant_data: Dict[str, Tuple[Dict, List[float]]] = {}
    for probe, label in probes:
        path = REPO / f"artifacts/benchmarks/dod_osa/{probe}/results.json"
        per_model, prevs = sk_patient_metric(path, metric_keys)
        variant_data[label] = (per_model, prevs)

    fig, axes = plt.subplots(1, len(metric_keys),
                             figsize=(5 * len(metric_keys), 5),
                             constrained_layout=True)
    for col, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[col]
        vals_by_variant: Dict[str, Dict[str, List[float]]] = {}
        for label, (per_model, _) in variant_data.items():
            v = {mk: scores[mkey] for mk, scores in per_model.items()
                 if scores.get(mkey)}
            if v:
                vals_by_variant[label] = v
        if not vals_by_variant:
            ax.axis("off")
            ax.set_title(f"{mlabel} (no data)")
            continue
        base = (0.5 if mkey in ("auroc", "macro_f1") else
                0.0 if mkey == "mcc" else None)
        _paired_bar_panel(ax, vals_by_variant, baseline=base, ylabel=mlabel)
        ax.set_title(mlabel)
        if col == 0:
            ax.legend(fontsize=8)
    fig.suptitle("DOD OSA patient classification — sklearn_linear", fontsize=14)
    _save(fig, "sk_dod_osa_patient_metrics.png")


def plot_sk_pn2026_cognitive_bar() -> None:
    probes = [("sklearn_linear", "non-balanced"),
              ("sklearn_linear_balanced", "balanced")]
    metric_keys = ("auroc", "auprc", "mcc", "macro_f1")
    metric_labels = ["AUROC", "AUPRC", "MCC", "macro-F1"]

    variant_data: Dict[str, Tuple[Dict, List[float]]] = {}
    for probe, label in probes:
        path = REPO / f"artifacts/benchmarks/physionet2026_cognitive/{probe}/results.json"
        per_model, prevs = sk_patient_metric(path, metric_keys)
        variant_data[label] = (per_model, prevs)

    fig, axes = plt.subplots(1, len(metric_keys),
                             figsize=(5 * len(metric_keys), 5),
                             constrained_layout=True)
    for col, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[col]
        vals_by_variant: Dict[str, Dict[str, List[float]]] = {}
        for label, (per_model, _) in variant_data.items():
            v = {mk: scores[mkey] for mk, scores in per_model.items()
                 if scores.get(mkey)}
            if v:
                vals_by_variant[label] = v
        if not vals_by_variant:
            ax.axis("off")
            ax.set_title(f"{mlabel} (no data)")
            continue
        base = (0.5 if mkey in ("auroc", "macro_f1") else
                0.0 if mkey == "mcc" else None)
        _paired_bar_panel(ax, vals_by_variant, baseline=base, ylabel=mlabel)
        ax.set_title(mlabel)
        if col == 0:
            ax.legend(fontsize=8)
    fig.suptitle("PhysioNet2026 cognitive impairment — sklearn_linear",
                 fontsize=14)
    _save(fig, "sk_pn2026_cognitive_patient_metrics.png")


def plot_sk_isruc_diagnosis_bar() -> None:
    label_modes = [
        ("has_diagnosis", "Binary: Has Diagnosis"),
        ("diagnosis",     "Multi-class: Diagnosis"),
    ]
    probes = [("sklearn_linear", "non-balanced"),
              ("sklearn_linear_balanced", "balanced")]
    aggregations = ["mean"]
    agg_labels = {"mean": "Mean", "mean_std": "Mean+Std"}
    agg_colors = {"mean": "#4C72B0", "mean_std": "#DD8452"}

    fig, axes = plt.subplots(len(label_modes), len(probes),
                             figsize=(6 * len(probes),
                                      5 * len(label_modes)),
                             constrained_layout=True, squeeze=False)
    for row, (lm, title) in enumerate(label_modes):
        for col, (probe, plabel) in enumerate(probes):
            ax = axes[row, col]
            path = (REPO / f"artifacts/benchmarks/isruc_diagnosis/{lm}"
                    f"/{probe}/results.json")
            if not path.exists():
                ax.set_title(f"{title} ({plabel})\n(no data)")
                ax.axis("off")
                continue
            data = _load_json(path)
            ok_entries = [
                e for e in data
                if e.get("evaluation_mode") == "patient_classification_eval"
                and e.get("ok")
                and e.get("checkpoint_id") not in SK_EXCLUDE
            ]
            if not ok_entries:
                ax.set_title(f"{title} ({plabel})\n(no data)")
                ax.axis("off")
                continue

            models = [mk for mk in SK_ALL_MODEL_ORDER
                      if any(e["checkpoint_id"] == mk for e in ok_entries)]
            eeg_m = [m for m in models
                     if not _is_tsfm(m) and not _is_sup(m)]
            tsfm_m = [m for m in models if _is_tsfm(m)]
            sup_m = [m for m in models if _is_sup(m)]
            models = eeg_m + tsfm_m + sup_m
            n_e, n_t, n_s = len(eeg_m), len(tsfm_m), len(sup_m)
            _gap = 0.7
            _groups = []
            _pos = 0
            if n_e:
                _groups.append(np.arange(n_e) + _pos)
                _pos += n_e + (_gap if (n_t or n_s) else 0)
            if n_t:
                _groups.append(np.arange(n_t) + _pos)
                _pos += n_t + (_gap if n_s else 0)
            if n_s:
                _groups.append(np.arange(n_s) + _pos)
            x = (np.concatenate(_groups) if _groups
                 else np.arange(len(models)))
            bar_w = 0.35
            for i, agg in enumerate(aggregations):
                means, stds = [], []
                for mk in models:
                    vals = [
                        e["metrics"]["best_test"]["macro_f1"]
                        for e in ok_entries
                        if e["checkpoint_id"] == mk
                        and (e.get("metadata") or {}).get("aggregation") == agg
                        and "best_test" in (e.get("metrics") or {})
                        and "macro_f1" in e["metrics"]["best_test"]
                    ]
                    means.append(float(np.mean(vals)) if vals else np.nan)
                    stds.append(float(np.std(vals)) if vals else 0.0)
                offset = (i - (len(aggregations) - 1) / 2) * bar_w
                bars = ax.bar(x + offset, means, bar_w, yerr=stds, capsize=4,
                              label=agg_labels[agg], color=agg_colors[agg],
                              edgecolor="black", linewidth=0.4)
                for bar, mk in zip(bars, models):
                    if _is_tsfm(mk):
                        bar.set_hatch("///")
                    elif _is_sup(mk):
                        bar.set_hatch("\\\\")
            if n_e and (n_t or n_s):
                ax.axvline(n_e - 0.5 + _gap / 2, color="#aaa", linestyle=":",
                           lw=0.8, zorder=0)
            if n_t and n_s:
                ax.axvline(n_e + (_gap if n_e else 0) + n_t - 0.5 + _gap / 2,
                           color="#aaa", linestyle=":", lw=0.8, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                               rotation=20, ha="right")
            ax.set_ylabel("Test macro-F1")
            ax.set_title(f"{title} ({plabel})")
            ax.set_ylim(0, 1.0)
            ax.legend(title="Aggregation", fontsize=8)
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3, lw=0.8)
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ISRUC Diagnosis — sklearn_linear", fontsize=14,
                 fontweight="bold")
    _save(fig, "sk_isruc_diagnosis_probes.png")


ISRUC_DIAGNOSIS_CLASSES_SK = ["healthy", "osa", "snoring",
                              "affective_disorder"]


def plot_sk_isruc_diagnosis_f1_per_class() -> None:
    probes = [("sklearn_linear", "non-balanced"),
              ("sklearn_linear_balanced", "balanced")]
    aggregations = ["mean"]
    agg_labels = {"mean": "Mean", "mean_std": "Mean+Std"}

    n_classes = len(ISRUC_DIAGNOSIS_CLASSES_SK)
    fig, axes = plt.subplots(len(probes), n_classes,
                             figsize=(4.5 * n_classes, 5 * len(probes)),
                             constrained_layout=True, squeeze=False)

    for p_row, (probe, plabel) in enumerate(probes):
        path = (REPO / "artifacts/benchmarks/isruc_diagnosis/diagnosis"
                f"/{probe}/results.json")
        if not path.exists():
            for ax in axes[p_row]:
                ax.axis("off")
            continue
        data = _load_json(path)
        ok_entries = [
            e for e in data
            if e.get("evaluation_mode") == "patient_classification_eval"
            and e.get("ok")
            and e.get("checkpoint_id") not in SK_EXCLUDE
        ]
        models = [mk for mk in SK_ALL_MODEL_ORDER
                  if any(e["checkpoint_id"] == mk for e in ok_entries)]
        eeg_m = [m for m in models
                 if not _is_tsfm(m) and not _is_sup(m)]
        tsfm_m = [m for m in models if _is_tsfm(m)]
        sup_m = [m for m in models if _is_sup(m)]
        models = eeg_m + tsfm_m + sup_m
        if not models:
            for ax in axes[p_row]:
                ax.axis("off")
            continue

        n_e, n_t, n_s = len(eeg_m), len(tsfm_m), len(sup_m)
        _gap = 0.7
        _groups = []
        _pos = 0
        if n_e:
            _groups.append(np.arange(n_e) + _pos)
            _pos += n_e + (_gap if (n_t or n_s) else 0)
        if n_t:
            _groups.append(np.arange(n_t) + _pos)
            _pos += n_t + (_gap if n_s else 0)
        if n_s:
            _groups.append(np.arange(n_s) + _pos)
        x = (np.concatenate(_groups) if _groups
             else np.arange(len(models)))
        bar_w = 0.8 / max(len(aggregations), 1)
        for cls_idx, cls_name in enumerate(ISRUC_DIAGNOSIS_CLASSES_SK):
            ax = axes[p_row, cls_idx]
            for agg_i, agg in enumerate(aggregations):
                means, stds = [], []
                for mk in models:
                    fold_f1s = []
                    for e in ok_entries:
                        if (e["checkpoint_id"] == mk
                                and (e.get("metadata") or {}).get(
                                    "aggregation") == agg):
                            f1_list = (e.get("metrics") or {}).get(
                                "best_test", {}).get("f1_per_class", [])
                            if cls_idx < len(f1_list):
                                fold_f1s.append(f1_list[cls_idx])
                    means.append(float(np.mean(fold_f1s))
                                 if fold_f1s else np.nan)
                    stds.append(float(np.std(fold_f1s))
                                if fold_f1s else 0.0)
                offset = (agg_i - (len(aggregations) - 1) / 2) * bar_w
                bars = ax.bar(x + offset, means, bar_w, yerr=stds, capsize=3,
                              label=agg_labels[agg],
                              color=["#4C72B0", "#DD8452"][agg_i],
                              edgecolor="black", linewidth=0.4)
                for bar, mk in zip(bars, models):
                    if _is_tsfm(mk):
                        bar.set_hatch("///")
                    elif _is_sup(mk):
                        bar.set_hatch("\\\\")
            if n_e and (n_t or n_s):
                ax.axvline(n_e - 0.5 + _gap / 2, color="#aaa", linestyle=":",
                           lw=0.8, zorder=0)
            if n_t and n_s:
                ax.axvline(n_e + (_gap if n_e else 0) + n_t - 0.5 + _gap / 2,
                           color="#aaa", linestyle=":", lw=0.8, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                               rotation=20, ha="right")
            ax.set_ylabel("F1")
            ax.set_title(f"{cls_name.replace('_', ' ').title()} ({plabel})")
            ax.set_ylim(0, 1.0)
            ax.grid(True, axis="y", alpha=0.3)
            if cls_idx == 0:
                ax.legend(title="Aggregation", fontsize=8)

    fig.suptitle("ISRUC Diagnosis — F1 per class (sklearn_linear)",
                 fontsize=14, fontweight="bold")
    _save(fig, "sk_isruc_diagnosis_f1_per_class.png")


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX table generation
# ═══════════════════════════════════════════════════════════════════════════


def _write_tex(name: str, content: str) -> None:
    out = TABLE_OUT / name
    out.write_text(content)
    print(f"wrote {out}")


def _fmt_val(mean: float, std: float, bold: bool = False) -> str:
    if np.isnan(mean):
        return "---"
    s = f"{mean:.2f} \\pm {std:.2f}"
    if bold:
        return f"\\bm{{{s}}}"
    return s


def _build_table(
    rows: List[str],
    cols: List[str],
    data: Dict[str, Dict[str, Tuple[float, float]]],
    caption: str,
    label: str,
    row_header: str = "Model",
    bold_max: bool = True,
) -> str:
    col_best: Dict[str, float] = {}
    if bold_max:
        for c in cols:
            vals = [data.get(r, {}).get(c, (np.nan, 0))[0] for r in rows]
            valid = [v for v in vals if not np.isnan(v)]
            col_best[c] = max(valid) if valid else np.nan

    n_cols = len(cols)
    col_fmt = "l" + "c" * n_cols
    caption_safe = caption.replace("—", "---")
    lines = [
        f"% Requires: \\usepackage{{booktabs, graphicx, bm}}",
        f"\\begin{{table}}[ht]",
        f"\\centering",
        f"\\caption{{{caption_safe}}}",
        f"\\label{{{label}}}",
        f"\\resizebox{{\\textwidth}}{{!}}{{%",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        f"\\toprule",
        f"{row_header} & " + " & ".join(
            [c.replace("_", "\\_") for c in cols]) + " \\\\",
        f"\\midrule",
    ]
    midrule_after: set = set()
    for idx in range(len(rows) - 1):
        if _model_group(rows[idx]) != _model_group(rows[idx + 1]):
            midrule_after.add(idx)
    for idx, r in enumerate(rows):
        cells = []
        for c in cols:
            m, s = data.get(r, {}).get(c, (np.nan, 0.0))
            bold = (bold_max and not np.isnan(m)
                    and abs(m - col_best.get(c, np.nan)) < 1e-9)
            cells.append(f"${_fmt_val(m, s, bold)}$")
        rname = MODEL_LABELS.get(r, r).replace("_", "\\_")
        lines.append(f"{rname} & " + " & ".join(cells) + " \\\\")
        if idx in midrule_after:
            lines.append("\\midrule")
    lines += [
        f"\\bottomrule",
        f"\\end{{tabular}}}}",
        f"\\end{{table}}",
    ]
    return "\n".join(lines)


def table_sk_sleep_staging() -> None:
    for metric_key, metric_name, fname in [
        ("cohen_kappa", "Cohen's $\\kappa$", "sk_sleep_kappa.tex"),
        ("macro_f1", "Macro-F1", "sk_sleep_macro_f1.tex"),
        ("balanced_accuracy", "Balanced Accuracy", "sk_sleep_bal_acc.tex"),
    ]:
        datasets = _sk_populated_datasets(SK_SLEEP_RESULTS_PATH,
                                          SK_SLEEP_DATASETS)
        data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for mk in SK_ALL_MODEL_ORDER:
            row: Dict[str, Tuple[float, float]] = {}
            for ds in datasets:
                vals = sk_sleep_metric(SK_SLEEP_RESULTS_PATH, ds, metric_key
                                       ).get(mk, [])
                if vals:
                    row[ds] = (float(np.mean(vals)), float(np.std(vals)))
                else:
                    row[ds] = (np.nan, 0.0)
            data[mk] = row
        _write_tex(fname, _build_table(
            SK_ALL_MODEL_ORDER, datasets, data,
            f"Sleep staging — {metric_name} (sklearn\\_linear, test)",
            f"tab:{fname.replace('.tex', '')}",
        ))


def table_sk_dod_subgroups() -> None:
    for metric_key, metric_name, fname in [
        ("cohen_kappa", "Cohen's $\\kappa$", "sk_dod_subgroup_kappa.tex"),
        ("macro_f1", "Macro-F1", "sk_dod_subgroup_macro_f1.tex"),
        ("balanced_accuracy", "Balanced Accuracy", "sk_dod_subgroup_bal_acc.tex"),
    ]:
        datasets = _sk_populated_datasets(SK_DOD_SUBGROUP_PATHS,
                                          SK_DOD_SUBGROUPS)
        data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for mk in SK_ALL_MODEL_ORDER:
            row: Dict[str, Tuple[float, float]] = {}
            for ds in datasets:
                vals = sk_sleep_metric(SK_DOD_SUBGROUP_PATHS, ds, metric_key
                                       ).get(mk, [])
                row[ds] = (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)
            data[mk] = row
        _write_tex(fname, _build_table(
            SK_ALL_MODEL_ORDER, datasets, data,
            f"DOD subgroups — {metric_name} (sklearn\\_linear, test)",
            f"tab:{fname.replace('.tex', '')}",
        ))


def table_sk_dod_scorers() -> None:
    ds_labels = {"dod": "DOD", "dod_dodh": "DOD-H", "dod_dodo": "DOD-O"}
    for ds in SK_DOD_SCORER_DATASETS:
        ds_label = ds_labels.get(ds, ds)
        for metric_key, metric_name, suffix in [
            ("cohen_kappa", "Cohen's $\\kappa$", "kappa"),
            ("macro_f1", "Macro-F1", "macro_f1"),
            ("balanced_accuracy", "Balanced Accuracy", "bal_acc"),
        ]:
            fname = f"sk_{ds}_scorer_{suffix}.tex"
            datasets = _sk_populated_datasets(SK_DOD_SCORER_PATHS[ds],
                                              SK_DOD_SCORERS)
            data: Dict[str, Dict[str, Tuple[float, float]]] = {}
            for mk in SK_ALL_MODEL_ORDER:
                row: Dict[str, Tuple[float, float]] = {}
                for sc in datasets:
                    vals = sk_sleep_metric(SK_DOD_SCORER_PATHS[ds], sc,
                                           metric_key).get(mk, [])
                    row[sc] = (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)
                data[mk] = row
            _write_tex(fname, _build_table(
                SK_ALL_MODEL_ORDER, datasets, data,
                f"{ds_label} scorers --- {metric_name} (sklearn\\_linear, test)",
                f"tab:{fname.replace('.tex', '')}",
            ))


def table_sk_sleepedf_subgroups() -> None:
    for metric_key, metric_name, fname in [
        ("cohen_kappa", "Cohen's $\\kappa$", "sk_sleepedf_subgroup_kappa.tex"),
        ("macro_f1", "Macro-F1", "sk_sleepedf_subgroup_macro_f1.tex"),
        ("balanced_accuracy", "Balanced Accuracy", "sk_sleepedf_subgroup_bal_acc.tex"),
    ]:
        datasets = _sk_populated_datasets(SK_SLEEPEDF_SUBGROUP_PATHS,
                                          SK_SLEEPEDF_SUBGROUPS)
        data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for mk in SK_ALL_MODEL_ORDER:
            row: Dict[str, Tuple[float, float]] = {}
            for ds in datasets:
                vals = sk_sleep_metric(SK_SLEEPEDF_SUBGROUP_PATHS, ds,
                                       metric_key).get(mk, [])
                row[ds] = (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)
            data[mk] = row
        _write_tex(fname, _build_table(
            SK_ALL_MODEL_ORDER, datasets, data,
            f"SleepEDF subgroups — {metric_name} (sklearn\\_linear, test)",
            f"tab:{fname.replace('.tex', '')}",
        ))


def table_sk_sleepedf_intervention() -> None:
    all_metrics = _load_intervention_csv()
    if not all_metrics:
        print("skip sk_sleepedf_intervention.tex: no data")
        return
    metric_keys = ["auroc", "auprc", "mcc", "macro_f1", "cohen_kappa",
                   "accuracy"]
    col_labels = ["AUROC", "AUPRC", "MCC", "Macro-F1", "Cohen's $\\kappa$",
                  "Accuracy"]
    data: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for mk in SK_ALL_MODEL_ORDER:
        row: Dict[str, Tuple[float, float]] = {}
        for mkey, clabel in zip(metric_keys, col_labels):
            vals = all_metrics.get(mkey, {}).get(mk, [])
            row[clabel] = (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)
        data[mk] = row
    _write_tex("sk_sleepedf_intervention.tex", _build_table(
        [mk for mk in SK_ALL_MODEL_ORDER if mk in data and any(
            not np.isnan(data[mk][c][0]) for c in col_labels)],
        col_labels, data,
        "SleepEDF Telemetry --- Drug Use Detection (sklearn\\_linear, test)",
        "tab:sk_sleepedf_intervention",
    ))


def table_sk_mass_subsets() -> None:
    for metric_key, metric_name, fname in [
        ("cohen_kappa", "Cohen's $\\kappa$", "sk_mass_subset_kappa.tex"),
        ("macro_f1", "Macro-F1", "sk_mass_subset_macro_f1.tex"),
        ("balanced_accuracy", "Balanced Accuracy", "sk_mass_subset_bal_acc.tex"),
    ]:
        datasets = _sk_populated_datasets(SK_MASS_SUBSET_PATHS,
                                          SK_MASS_SUBSETS)
        data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for mk in SK_ALL_MODEL_ORDER:
            row: Dict[str, Tuple[float, float]] = {}
            for ds in datasets:
                vals = sk_sleep_metric(SK_MASS_SUBSET_PATHS, ds, metric_key
                                       ).get(mk, [])
                row[ds] = (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)
            data[mk] = row
        _write_tex(fname, _build_table(
            SK_ALL_MODEL_ORDER, datasets, data,
            f"MASS subsets --- {metric_name} (sklearn\\_linear, test)",
            f"tab:{fname.replace('.tex', '')}",
        ))


def table_sk_isruc_scorers() -> None:
    for metric_key, metric_name, fname in [
        ("cohen_kappa", "Cohen's $\\kappa$", "sk_isruc_scorer_kappa.tex"),
        ("macro_f1", "Macro-F1", "sk_isruc_scorer_macro_f1.tex"),
        ("balanced_accuracy", "Balanced Accuracy", "sk_isruc_scorer_bal_acc.tex"),
    ]:
        datasets = _sk_populated_datasets(SK_ISRUC_SCORER_PATHS,
                                          SK_ISRUC_SCORERS)
        data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for mk in SK_ALL_MODEL_ORDER:
            row: Dict[str, Tuple[float, float]] = {}
            for ds in datasets:
                vals = sk_sleep_metric(SK_ISRUC_SCORER_PATHS, ds, metric_key
                                       ).get(mk, [])
                row[ds] = (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)
            data[mk] = row
        _write_tex(fname, _build_table(
            SK_ALL_MODEL_ORDER, datasets, data,
            f"ISRUC scorers --- {metric_name} (sklearn\\_linear, test)",
            f"tab:{fname.replace('.tex', '')}",
        ))


def table_sk_isruc_subgroups() -> None:
    for metric_key, metric_name, fname in [
        ("cohen_kappa", "Cohen's $\\kappa$", "sk_isruc_subgroup_kappa.tex"),
        ("macro_f1", "Macro-F1", "sk_isruc_subgroup_macro_f1.tex"),
        ("balanced_accuracy", "Balanced Accuracy", "sk_isruc_subgroup_bal_acc.tex"),
    ]:
        datasets = _sk_populated_datasets(SK_ISRUC_SUBGROUP_PATHS,
                                          SK_ISRUC_SUBGROUPS)
        data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for mk in SK_ALL_MODEL_ORDER:
            row: Dict[str, Tuple[float, float]] = {}
            for ds in datasets:
                vals = sk_sleep_metric(SK_ISRUC_SUBGROUP_PATHS, ds, metric_key
                                       ).get(mk, [])
                row[ds] = (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)
            data[mk] = row
        _write_tex(fname, _build_table(
            SK_ALL_MODEL_ORDER, datasets, data,
            f"ISRUC subject groups --- {metric_name} (sklearn\\_linear, test)",
            f"tab:{fname.replace('.tex', '')}",
        ))


def table_sk_pn2026_sites() -> None:
    for metric_key, metric_name, fname in [
        ("cohen_kappa", "Cohen's $\\kappa$", "sk_pn2026_site_kappa.tex"),
        ("macro_f1", "Macro-F1", "sk_pn2026_site_macro_f1.tex"),
        ("balanced_accuracy", "Balanced Accuracy", "sk_pn2026_site_bal_acc.tex"),
    ]:
        datasets = _sk_populated_datasets(SK_PN2026_SITE_PATHS,
                                          SK_PN2026_SITES)
        data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for mk in SK_ALL_MODEL_ORDER:
            row: Dict[str, Tuple[float, float]] = {}
            for ds in datasets:
                vals = sk_sleep_metric(SK_PN2026_SITE_PATHS, ds, metric_key
                                       ).get(mk, [])
                row[ds] = (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)
            data[mk] = row
        _write_tex(fname, _build_table(
            SK_ALL_MODEL_ORDER, datasets, data,
            f"PhysioNet2026 sites --- {metric_name} (sklearn\\_linear, test)",
            f"tab:{fname.replace('.tex', '')}",
        ))


def _table_event_detection(
    ds_slug: str, ds_label: str, event_field: str | None,
    event_label: str, thr: float, probes: List[Tuple[str, str]],
    metric_keys: List[str], metric_labels: List[str], fname: str,
    caption: str, label: str,
) -> None:
    cols = []
    for _, plabel in probes:
        for ml in metric_labels:
            cols.append(f"{ml} ({plabel})" if len(probes) > 1 else ml)

    data: Dict[str, Dict[str, Tuple[float, float]]] = {}
    thr_key = f"threshold_{thr}s"
    for mk in SK_ALL_MODEL_ORDER:
        row: Dict[str, Tuple[float, float]] = {}
        ci = 0
        for probe, plabel in probes:
            path = REPO / f"artifacts/benchmarks/{ds_slug}/{probe}/results.json"
            for mkey, ml in zip(metric_keys, metric_labels):
                cname = cols[ci]; ci += 1
                vals = sk_event_metric(path, event_field, thr_key, mkey
                                       ).get(mk, [])
                row[cname] = (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)
        data[mk] = row

    active_models = [mk for mk in SK_ALL_MODEL_ORDER
                     if any(not np.isnan(data.get(mk, {}).get(c, (np.nan,))[0])
                            for c in cols)]
    if not active_models:
        print(f"skip {fname}: no data")
        return
    _write_tex(fname, _build_table(active_models, cols, data,
                                   caption, label))


def table_sk_arousal() -> None:
    probes = [("sklearn_linear", "nb"), ("sklearn_linear_balanced", "bal")]
    mkeys = ["auprc", "auroc", "mcc", "macro_f1"]
    mlabels = ["AUPRC", "AUROC", "MCC", "Macro-F1"]
    for ds, ds_label in SK_AROUSAL_PATHS.items():
        slug = ds
        nice = ds.replace("_", " ").title()
        _table_event_detection(
            slug, nice, None, "arousal", AROUSAL_CLINICAL,
            probes, mkeys, mlabels,
            f"sk_arousal_{slug}.tex",
            f"{nice} Arousal @ {AROUSAL_CLINICAL}s (sklearn\\_linear)",
            f"tab:sk_arousal_{slug}",
        )


def table_sk_limb() -> None:
    probes = [("sklearn_linear", "nb"), ("sklearn_linear_balanced", "bal")]
    mkeys = ["auprc", "auroc", "mcc", "macro_f1"]
    mlabels = ["AUPRC", "AUROC", "MCC", "Macro-F1"]
    limb_thr = 0.5
    for evt_label, evt_field in LIMB_EVENT_TYPES:
        short = evt_field.replace("_fraction", "")
        _table_event_detection(
            "physionet2026_limb", "PN2026", evt_field, evt_label,
            limb_thr, probes, mkeys, mlabels,
            f"sk_limb_{short}.tex",
            f"Limb Movement ({evt_label}) @ {limb_thr}s (sklearn\\_linear)",
            f"tab:sk_limb_{short}",
        )


def table_sk_resp() -> None:
    probes = [("sklearn_linear", "nb"), ("sklearn_linear_balanced", "bal")]
    mkeys = ["auprc", "auroc", "mcc", "macro_f1"]
    mlabels = ["AUPRC", "AUROC", "MCC", "Macro-F1"]
    for ds_slug, ds_label, event_types in [
        ("ucddb_respevt", "UCDDB", SK_RESP_UCDDB_EVENT_TYPES),
        ("physionet2026_respevt", "PN2026", SK_RESP_PN2026_EVENT_TYPES),
    ]:
        for evt_label, evt_field in event_types:
            short = evt_field.replace("_fraction", "")
            _table_event_detection(
                ds_slug, ds_label, evt_field, evt_label, RESP_CLINICAL,
                probes, mkeys, mlabels,
                f"sk_resp_{ds_slug}_{short}.tex",
                f"{ds_label} Respiratory — {evt_label} @ {RESP_CLINICAL}s "
                f"(sklearn\\_linear)",
                f"tab:sk_resp_{ds_slug}_{short}",
            )


def table_sk_dod_osa() -> None:
    probes = [("sklearn_linear", "nb"), ("sklearn_linear_balanced", "bal")]
    metric_keys = ("auroc", "auprc", "mcc", "macro_f1")
    metric_labels = ["AUROC", "AUPRC", "MCC", "Macro-F1"]
    cols = []
    for _, plabel in probes:
        for ml in metric_labels:
            cols.append(f"{ml} ({plabel})")

    data: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for mk in SK_ALL_MODEL_ORDER:
        row: Dict[str, Tuple[float, float]] = {}
        ci = 0
        for probe, plabel in probes:
            path = REPO / f"artifacts/benchmarks/dod_osa/{probe}/results.json"
            per_model, _ = sk_patient_metric(path, metric_keys)
            for mkey, ml in zip(metric_keys, metric_labels):
                cname = cols[ci]; ci += 1
                vals = per_model.get(mk, {}).get(mkey, [])
                row[cname] = (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)
        data[mk] = row

    active = [mk for mk in SK_ALL_MODEL_ORDER
              if any(not np.isnan(data.get(mk, {}).get(c, (np.nan,))[0])
                     for c in cols)]
    _write_tex("sk_dod_osa.tex", _build_table(
        active, cols, data,
        "DOD OSA Patient Classification (sklearn\\_linear, test)",
        "tab:sk_dod_osa",
    ))


def table_sk_pn2026_cognitive() -> None:
    probes = [("sklearn_linear", "nb"), ("sklearn_linear_balanced", "bal")]
    metric_keys = ("auroc", "auprc", "mcc", "macro_f1")
    metric_labels = ["AUROC", "AUPRC", "MCC", "Macro-F1"]
    cols = []
    for _, plabel in probes:
        for ml in metric_labels:
            cols.append(f"{ml} ({plabel})")

    data: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for mk in SK_ALL_MODEL_ORDER:
        row: Dict[str, Tuple[float, float]] = {}
        ci = 0
        for probe, plabel in probes:
            path = REPO / f"artifacts/benchmarks/physionet2026_cognitive/{probe}/results.json"
            per_model, _ = sk_patient_metric(path, metric_keys)
            for mkey, ml in zip(metric_keys, metric_labels):
                cname = cols[ci]; ci += 1
                vals = per_model.get(mk, {}).get(mkey, [])
                row[cname] = (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)
        data[mk] = row

    active = [mk for mk in SK_ALL_MODEL_ORDER
              if any(not np.isnan(data.get(mk, {}).get(c, (np.nan,))[0])
                     for c in cols)]
    _write_tex("sk_pn2026_cognitive.tex", _build_table(
        active, cols, data,
        "PhysioNet2026 Cognitive Impairment (sklearn\\_linear, test)",
        "tab:sk_pn2026_cognitive",
    ))


def table_sk_isruc_diagnosis() -> None:
    label_modes = [
        ("has_diagnosis", "Binary: Has Diagnosis"),
        ("diagnosis", "Multi-class: Diagnosis"),
    ]
    probes = [("sklearn_linear", "nb"), ("sklearn_linear_balanced", "bal")]
    aggregations = [("mean", "mean")]

    for lm, lm_title in label_modes:
        cols = []
        for _, plabel in probes:
            for _, agg_label in aggregations:
                cols.append(f"F1 ({plabel}, {agg_label})")

        data: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for mk in SK_ALL_MODEL_ORDER:
            row: Dict[str, Tuple[float, float]] = {}
            ci = 0
            for probe, plabel in probes:
                path = (REPO / f"artifacts/benchmarks/isruc_diagnosis/{lm}"
                        f"/{probe}/results.json")
                if not path.exists():
                    for _, agg_label in aggregations:
                        row[cols[ci]] = (np.nan, 0.0); ci += 1
                    continue
                entries = _load_json(path)
                ok = [e for e in entries
                      if e.get("evaluation_mode") == "patient_classification_eval"
                      and e.get("ok")
                      and e.get("checkpoint_id") not in SK_EXCLUDE]
                for agg, agg_label in aggregations:
                    cname = cols[ci]; ci += 1
                    vals = [e["metrics"]["best_test"]["macro_f1"]
                            for e in ok
                            if e["checkpoint_id"] == mk
                            and (e.get("metadata") or {}).get("aggregation") == agg
                            and "best_test" in (e.get("metrics") or {})
                            and "macro_f1" in e["metrics"]["best_test"]]
                    row[cname] = (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, 0.0)
            data[mk] = row

        active = [mk for mk in SK_ALL_MODEL_ORDER
                  if any(not np.isnan(data.get(mk, {}).get(c, (np.nan,))[0])
                         for c in cols)]
        if not active:
            print(f"skip sk_isruc_{lm}.tex: no data")
            continue
        _write_tex(f"sk_isruc_{lm}.tex", _build_table(
            active, cols, data,
            f"ISRUC {lm_title} — Macro-F1 (sklearn\\_linear, test)",
            f"tab:sk_isruc_{lm}",
        ))


# ═══════════════════════════════════════════════════════════════════════════
# C3-M2 single-channel ablation — all-channel vs C3-M2 (EEG FM only)
# ═══════════════════════════════════════════════════════════════════════════

C3M2_SLEEP_PATH = {"physionet2026": "physionet2026_c3m2/sklearn_linear/results.json"}
C3M2_AROUSAL_PATH = "physionet2026_c3m2_arousal/{probe}/results.json"
C3M2_RESP_PATH = "physionet2026_c3m2_respevt/{probe}/results.json"
C3M2_LIMB_PATH = "physionet2026_c3m2_limb/{probe}/results.json"
C3M2_COGNITIVE_PATH = "physionet2026_c3m2_cognitive/{probe}/results.json"

C3M2_VARIANT_COLORS = {"all-channel": "#4C72B0", "C3-M2": "#DD8452"}
C3M2_EEG_MODELS = SK_MODEL_ORDER


def _c3m2_bar_panel(
    ax,
    vals_by_variant: Dict[str, Dict[str, List[float]]],
    baseline: float | None,
    ylabel: str,
    ylim: Tuple[float, float] | None = None,
) -> None:
    eeg_set = set(C3M2_EEG_MODELS)
    all_models = set()
    for v in vals_by_variant.values():
        all_models |= v.keys()
    first_variant = next(iter(vals_by_variant.values()))
    ordering = sorted(
        [m for m in all_models if m in eeg_set],
        key=lambda m: (-float(np.mean(first_variant[m]))
                       if m in first_variant else -np.inf))
    variants = list(vals_by_variant.keys())
    n_v = len(variants)
    bar_w = 0.8 / n_v
    x = np.arange(len(ordering))
    for vi, vname in enumerate(variants):
        vdata = vals_by_variant[vname]
        means = [float(np.mean(vdata[m])) if m in vdata else np.nan
                 for m in ordering]
        stds = [float(np.std(vdata[m])) if m in vdata else 0.0
                for m in ordering]
        offset = (vi - (n_v - 1) / 2) * bar_w
        ax.bar(x + offset, means, bar_w, yerr=stds, capsize=2.5,
               label=vname, color=C3M2_VARIANT_COLORS.get(vname, "#777"),
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in ordering],
                       rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if baseline is not None:
        ax.axhline(baseline, color="black", linestyle="--", lw=1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.3)


def _c3m2_sweep_panel(ax, allch_per_model, c3m2_per_model, thresholds):
    for mk in C3M2_EEG_MODELS:
        color = MODEL_COLORS.get(mk, "#777")
        for per_model, ls, alpha, lbl_suffix in [
            (allch_per_model, "-", 1.0, ""),
            (c3m2_per_model, "--", 0.75, " (C3-M2)"),
        ]:
            if mk not in per_model:
                continue
            xs, ys, ses = [], [], []
            for t in thresholds:
                if t in per_model[mk]:
                    m, s = per_model[mk][t]
                    xs.append(t); ys.append(m); ses.append(s)
            if not xs:
                continue
            xs_a = np.array(xs); ys_a = np.array(ys); ses_a = np.array(ses)
            lbl = MODEL_LABELS.get(mk, mk) + lbl_suffix if ls == "-" else None
            ax.plot(xs_a, ys_a, marker="o", color=color, linestyle=ls,
                    linewidth=1.5, alpha=alpha, label=lbl)
            ax.fill_between(xs_a, ys_a - ses_a, ys_a + ses_a,
                            color=color, alpha=0.08 * alpha)


def plot_c3m2_sleep_staging() -> None:
    allch_map = {"physionet2026": "physionet2026/sklearn_linear/results.json"}
    metrics = [("cohen_kappa", "Cohen's κ"),
               ("macro_f1", "macro-F1"),
               ("balanced_accuracy", "Balanced accuracy")]
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(5 * len(metrics), 5),
                             constrained_layout=True)
    for ax, (mkey, mlabel) in zip(axes, metrics):
        allch = sk_sleep_metric(allch_map, "physionet2026", mkey)
        c3m2 = sk_sleep_metric(C3M2_SLEEP_PATH, "physionet2026", mkey)
        eeg_set = set(C3M2_EEG_MODELS)
        allch_eeg = {mk: v for mk, v in allch.items() if mk in eeg_set}
        c3m2_eeg = {mk: v for mk, v in c3m2.items() if mk in eeg_set}
        variants = {}
        if allch_eeg:
            variants["all-channel"] = allch_eeg
        if c3m2_eeg:
            variants["C3-M2"] = c3m2_eeg
        if not variants:
            ax.axis("off")
            ax.set_title(f"{mlabel} (no data)")
            continue
        _c3m2_bar_panel(ax, variants, baseline=None, ylabel=mlabel)
        ax.set_title(mlabel)
        if ax is axes[0]:
            ax.legend(fontsize=9)
    fig.suptitle("PhysioNet2026 — All-channel vs C3-M2 (sleep staging)",
                 fontsize=14)
    _save(fig, "c3m2_pn2026_sleep_staging.png")


def plot_c3m2_arousal() -> None:
    probes = [("sklearn_linear", "non-balanced"),
              ("sklearn_linear_balanced", "balanced")]
    metric_keys = ["auprc", "auroc", "mcc", "macro_f1"]
    metric_labels = ["AUPRC", "AUROC", "MCC", "macro-F1"]
    thr_key = f"threshold_{AROUSAL_CLINICAL}s"
    eeg_set = set(C3M2_EEG_MODELS)

    n_rows = len(probes)
    n_cols = len(metric_keys)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.5 * n_rows),
                             constrained_layout=True, squeeze=False)
    for row, (probe, plabel) in enumerate(probes):
        allch_path = REPO / f"artifacts/benchmarks/physionet2026_arousal/{probe}/results.json"
        c3m2_path = REPO / "artifacts/benchmarks" / C3M2_AROUSAL_PATH.format(probe=probe)
        for col, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
            ax = axes[row, col]
            allch = sk_event_metric(allch_path, None, thr_key, mkey)
            c3m2 = sk_event_metric(c3m2_path, None, thr_key, mkey)
            variants: Dict[str, Dict[str, List[float]]] = {}
            allch_eeg = {mk: v for mk, v in allch.items() if mk in eeg_set}
            c3m2_eeg = {mk: v for mk, v in c3m2.items() if mk in eeg_set}
            if allch_eeg:
                variants["all-channel"] = allch_eeg
            if c3m2_eeg:
                variants["C3-M2"] = c3m2_eeg
            if not variants:
                ax.axis("off")
                ax.set_title(f"{plabel} — {mlabel} (no data)")
                continue
            base = (0.5 if mkey in ("auroc", "macro_f1") else
                    0.0 if mkey == "mcc" else None)
            _c3m2_bar_panel(ax, variants, baseline=base, ylabel=mlabel)
            ax.set_title(f"{plabel} — {mlabel}")
            if col == 0 and row == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"PhysioNet2026 arousal @ {AROUSAL_CLINICAL}s — "
                 f"all-channel vs C3-M2", fontsize=14)
    _save(fig, "c3m2_pn2026_arousal_bar.png")


def plot_c3m2_resp() -> None:
    probes = [("sklearn_linear", "non-balanced"),
              ("sklearn_linear_balanced", "balanced")]
    metric_keys = ["auprc", "auroc", "mcc", "macro_f1"]
    metric_labels = ["AUPRC", "AUROC", "MCC", "macro-F1"]
    thr_key = f"threshold_{RESP_CLINICAL}s"
    eeg_set = set(C3M2_EEG_MODELS)
    event_types = SK_RESP_PN2026_EVENT_TYPES

    n_evt = min(len(event_types), 4)
    for probe, plabel in probes:
        allch_path = REPO / f"artifacts/benchmarks/physionet2026_respevt/{probe}/results.json"
        c3m2_path = REPO / "artifacts/benchmarks" / C3M2_RESP_PATH.format(probe=probe)
        fig, axes = plt.subplots(n_evt, len(metric_keys),
                                 figsize=(5 * len(metric_keys), 4.5 * n_evt),
                                 constrained_layout=True, squeeze=False)
        for row, (evt_label, evt_field) in enumerate(event_types[:4]):
            for col, (mkey, mlabel) in enumerate(zip(metric_keys,
                                                     metric_labels)):
                ax = axes[row, col]
                allch = sk_event_metric(allch_path, evt_field, thr_key, mkey)
                c3m2 = sk_event_metric(c3m2_path, evt_field, thr_key, mkey)
                variants: Dict[str, Dict[str, List[float]]] = {}
                allch_eeg = {mk: v for mk, v in allch.items()
                             if mk in eeg_set}
                c3m2_eeg = {mk: v for mk, v in c3m2.items()
                            if mk in eeg_set}
                if allch_eeg:
                    variants["all-channel"] = allch_eeg
                if c3m2_eeg:
                    variants["C3-M2"] = c3m2_eeg
                if not variants:
                    ax.axis("off")
                    ax.set_title(f"{evt_label}\n{mlabel} (no data)")
                    continue
                base = (0.5 if mkey in ("auroc", "macro_f1") else
                        0.0 if mkey == "mcc" else None)
                _c3m2_bar_panel(ax, variants, baseline=base, ylabel=mlabel)
                ax.set_title(f"{evt_label}\n{mlabel}")
                if col == 0 and row == 0:
                    ax.legend(fontsize=8)
        sfx = "bal" if "balanced" in probe else "unwt"
        fig.suptitle(f"PhysioNet2026 respiratory @ {RESP_CLINICAL}s "
                     f"({plabel}) — all-channel vs C3-M2", fontsize=14)
        _save(fig, f"c3m2_pn2026_resp_bar_{sfx}.png")


def plot_c3m2_limb() -> None:
    probes = [("sklearn_linear", "non-balanced"),
              ("sklearn_linear_balanced", "balanced")]
    metric_keys = ["auprc", "auroc", "mcc", "macro_f1"]
    metric_labels = ["AUPRC", "AUROC", "MCC", "macro-F1"]
    limb_thr = 0.5
    thr_key = f"threshold_{limb_thr}s"
    eeg_set = set(C3M2_EEG_MODELS)

    for probe, plabel in probes:
        allch_path = REPO / f"artifacts/benchmarks/physionet2026_limb/{probe}/results.json"
        c3m2_path = REPO / "artifacts/benchmarks" / C3M2_LIMB_PATH.format(probe=probe)
        fig, axes = plt.subplots(len(LIMB_EVENT_TYPES), len(metric_keys),
                                 figsize=(5 * len(metric_keys),
                                          4.5 * len(LIMB_EVENT_TYPES)),
                                 constrained_layout=True, squeeze=False)
        for row, (evt_label, evt_field) in enumerate(LIMB_EVENT_TYPES):
            for col, (mkey, mlabel) in enumerate(zip(metric_keys,
                                                     metric_labels)):
                ax = axes[row, col]
                allch = sk_event_metric(allch_path, evt_field, thr_key, mkey)
                c3m2 = sk_event_metric(c3m2_path, evt_field, thr_key, mkey)
                variants: Dict[str, Dict[str, List[float]]] = {}
                allch_eeg = {mk: v for mk, v in allch.items()
                             if mk in eeg_set}
                c3m2_eeg = {mk: v for mk, v in c3m2.items()
                            if mk in eeg_set}
                if allch_eeg:
                    variants["all-channel"] = allch_eeg
                if c3m2_eeg:
                    variants["C3-M2"] = c3m2_eeg
                if not variants:
                    ax.axis("off")
                    ax.set_title(f"{evt_label}\n{mlabel} (no data)")
                    continue
                base = (0.5 if mkey in ("auroc", "macro_f1") else
                        0.0 if mkey == "mcc" else None)
                _c3m2_bar_panel(ax, variants, baseline=base, ylabel=mlabel)
                ax.set_title(f"{evt_label}\n{mlabel}")
                if col == 0 and row == 0:
                    ax.legend(fontsize=8)
        sfx = "bal" if "balanced" in probe else "unwt"
        fig.suptitle(f"PhysioNet2026 limb movement @ {limb_thr}s "
                     f"({plabel}) — all-channel vs C3-M2", fontsize=14)
        _save(fig, f"c3m2_pn2026_limb_bar_{sfx}.png")


def plot_c3m2_cognitive_bar() -> None:
    probes = [("sklearn_linear", "non-balanced"),
              ("sklearn_linear_balanced", "balanced")]
    metric_keys = ("auroc", "auprc", "mcc", "macro_f1")
    metric_labels = ["AUROC", "AUPRC", "MCC", "macro-F1"]
    eeg_set = set(C3M2_EEG_MODELS)

    fig, axes = plt.subplots(len(probes), len(metric_keys),
                             figsize=(5 * len(metric_keys),
                                      4.5 * len(probes)),
                             constrained_layout=True, squeeze=False)
    for row, (probe, plabel) in enumerate(probes):
        allch_path = REPO / f"artifacts/benchmarks/physionet2026_cognitive/{probe}/results.json"
        c3m2_path = REPO / "artifacts/benchmarks" / C3M2_COGNITIVE_PATH.format(probe=probe)
        allch_data, _ = sk_patient_metric(allch_path, metric_keys)
        c3m2_data, _ = sk_patient_metric(c3m2_path, metric_keys)
        for col, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
            ax = axes[row, col]
            allch_vals = {mk: scores[mkey]
                         for mk, scores in allch_data.items()
                         if scores.get(mkey) and mk in eeg_set}
            c3m2_vals = {mk: scores[mkey]
                         for mk, scores in c3m2_data.items()
                         if scores.get(mkey) and mk in eeg_set}
            variants: Dict[str, Dict[str, List[float]]] = {}
            if allch_vals:
                variants["all-channel"] = allch_vals
            if c3m2_vals:
                variants["C3-M2"] = c3m2_vals
            if not variants:
                ax.axis("off")
                ax.set_title(f"{plabel} — {mlabel} (no data)")
                continue
            base = (0.5 if mkey in ("auroc", "macro_f1") else
                    0.0 if mkey == "mcc" else None)
            _c3m2_bar_panel(ax, variants, baseline=base, ylabel=mlabel)
            ax.set_title(f"{plabel} — {mlabel}")
            if col == 0 and row == 0:
                ax.legend(fontsize=8)
    fig.suptitle("PhysioNet2026 cognitive impairment — all-channel vs C3-M2",
                 fontsize=14)
    _save(fig, "c3m2_pn2026_cognitive_bar.png")


# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    # --- sklearn_linear EEG FM plots ---
    for mkey, ylabel, suffix in [
        ("cohen_kappa", "Cohen's κ", "kappa"),
        ("macro_f1", "macro-F1", "macro_f1"),
        ("balanced_accuracy", "Balanced accuracy", "bal_acc"),
    ]:
        plot_sk_sleep_grouped_bar(mkey, ylabel, f"sk_sleep_{suffix}.png")
        _plot_sk_sleep_byds(
            mkey, ylabel, f"sk_sleep_{suffix}_byds.png",
            SK_SLEEP_RESULTS_PATH, SK_SLEEP_DATASETS,
            SK_DATASET_COLORS, "Sleep staging")
        _plot_sk_sleep_byds_mixed(
            mkey, ylabel, f"sk_sleep_{suffix}_byds_mixed.png",
            SK_SLEEP_RESULTS_PATH, SK_SLEEP_DATASETS,
            SK_DATASET_COLORS, "Sleep staging")
        _plot_sk_sleep_violin(
            mkey, ylabel, f"sk_sleep_{suffix}_violin.png",
            SK_SLEEP_RESULTS_PATH, SK_SLEEP_DATASETS, "Sleep staging")
        _plot_sk_sleep_lines(
            mkey, ylabel, f"sk_sleep_{suffix}_lines.png",
            SK_SLEEP_RESULTS_PATH, SK_SLEEP_DATASETS,
            SK_DATASET_COLORS, "Sleep staging")
        plot_sk_dod_subgroup(mkey, ylabel, f"sk_dod_subgroup_{suffix}.png")
        for ds in SK_DOD_SCORER_DATASETS:
            plot_sk_dod_scorers(ds, mkey, ylabel,
                                f"sk_{ds}_scorer_{suffix}.png")
        plot_sk_sleepedf_subgroup(mkey, ylabel,
                                  f"sk_sleepedf_subgroup_{suffix}.png")
        plot_sk_mass_subsets(mkey, ylabel, f"sk_mass_subset_{suffix}.png")
        plot_sk_isruc_scorers(mkey, ylabel, f"sk_isruc_scorer_{suffix}.png")
        plot_sk_isruc_subgroups(mkey, ylabel,
                                f"sk_isruc_subgroup_{suffix}.png")
        plot_sk_pn2026_sites(mkey, ylabel,
                             f"sk_pn2026_site_{suffix}.png")
    plot_sk_sleep_f1_per_stage()
    plot_sk_sleep_kappa_spider()
    plot_sk_sleepedf_intervention()

    plot_sk_arousal_sweep()
    plot_sk_arousal_bar_3s()

    plot_sk_limb_sweep()
    plot_sk_limb_bar_05s()

    plot_sk_resp_sweep("ucddb_respevt", "UCDDB", SK_RESP_UCDDB_EVENT_TYPES,
                       "sk_resp_ucddb_auprc_sweep.png")
    plot_sk_resp_sweep("physionet2026_respevt", "PhysioNet2026",
                       SK_RESP_PN2026_EVENT_TYPES,
                       "sk_resp_pn2026_auprc_sweep.png")
    plot_sk_resp_bar_10s("ucddb_respevt", "UCDDB", SK_RESP_UCDDB_EVENT_TYPES,
                         "sk_resp_ucddb_bar_10s.png")
    plot_sk_resp_bar_10s("physionet2026_respevt", "PhysioNet2026",
                         SK_RESP_PN2026_EVENT_TYPES,
                         "sk_resp_pn2026_bar_10s.png")

    plot_sk_dod_osa_bar()
    plot_sk_pn2026_cognitive_bar()
    plot_sk_isruc_diagnosis_bar()
    plot_sk_isruc_diagnosis_f1_per_class()

    # --- C3-M2 ablation comparison (PhysioNet2026 only) ---
    plot_c3m2_sleep_staging()
    plot_c3m2_arousal()
    plot_c3m2_resp()
    plot_c3m2_limb()
    plot_c3m2_cognitive_bar()

    # --- LaTeX tables ---
    table_sk_sleep_staging()
    table_sk_dod_subgroups()
    table_sk_dod_scorers()
    table_sk_sleepedf_subgroups()
    table_sk_sleepedf_intervention()
    table_sk_mass_subsets()
    table_sk_isruc_scorers()
    table_sk_isruc_subgroups()
    table_sk_pn2026_sites()
    table_sk_arousal()
    table_sk_limb()
    table_sk_resp()
    table_sk_dod_osa()
    table_sk_pn2026_cognitive()
    table_sk_isruc_diagnosis()


if __name__ == "__main__":
    main()
