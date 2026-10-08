#!/usr/bin/env python3
"""Generate NeurIPS-style LaTeX tables for EEG Foundation Model Benchmark.

All tables fit within NeurIPS single-column text width (~5.5in).
Tracks are sub-rows within each dataset instead of extra columns.
"""

import os
from pathlib import Path
import numpy as np

OUT = Path(os.path.expandvars("${EEG_DATA_ROOT}/EEGBenchmarks/artifacts/tables"))
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["BENDR", "EEGPT", "NeuroLM", "LaBraM", "CBraMod"]

# ──────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────

mi_info = [
    # (name, subj, ch, trials_total, classes, sfreq, country)
    ("PhysionetMI",       103, 64,  4300,  2, 160,  "US"),
    ("Cho2017",            49, 64,  9800,  2, 512,  "KR"),
    ("Lee2019\\_MI",       54, 20, 10800,  2, 1000, "KR"),
    ("Schirrmeister2017",  14, 128, 9000,  2, 500,  "DE"),
    ("Weibo2014",          10, 60,  1600,  2, 200,  "CN"),
    ("BNCI2014\\_001",      9, 22,  1300,  2, 250,  "AT"),
    ("Dreyer2023A",        60, 27,  4800,  2, 512,  "FR"),
    ("BNCI2014\\_004",      9,  3,  6500,  2, 250,  "AT"),
    ("BNCI2015\\_001",     12, 13,  2400,  2, 512,  "AT"),
]

mi_trackA = {
    "BENDR":   [61.7, 58.7, 50.9, 50.8, 52.3, 53.0, 60.7, 54.8, 51.4],
    "EEGPT":   [64.1, 56.1, 64.7, 63.8, 57.5, 66.7, 70.9, 61.0, 60.7],
    "NeuroLM": [73.5, 55.5, 58.4, 52.3, 49.2, 55.3, 59.8, 64.5, 60.2],
    "LaBraM":  [51.7, 55.1, 55.3, 54.1, 49.6, 54.5, 56.0, 53.0, 55.6],
    "CBraMod": [49.6, 53.5, 53.0, 51.5, 51.3, 53.1, 53.7, 55.2, 55.9],
}
mi_trackA_std = {
    "BENDR":   [9.5, 12.0, 5.1, 2.3, 3.4, 3.5, 5.1, 3.9, 2.2],
    "EEGPT":   [10.4, 7.5, 11.9, 9.1, 11.2, 8.2, 8.7, 9.3, 6.0],
    "NeuroLM": [12.0, 9.6, 9.2, 2.0, 4.4, 3.0, 6.4, 6.8, 5.4],
    "LaBraM":  [7.4, 8.9, 6.9, 3.2, 4.8, 3.4, 4.5, 1.4, 6.7],
    "CBraMod": [6.2, 9.9, 5.5, 3.2, 3.0, 4.7, 5.1, 3.3, 9.7],
}
mi_trackB = {
    "BENDR":   [50.2, 50.5, 50.1, 49.7, 50.9, 49.7, 56.2, 53.5, 53.2],
    "EEGPT":   [61.5, 55.6, 64.6, 64.8, 58.3, 64.8, 70.5, 61.1, 59.5],
    "NeuroLM": [55.0, 51.6, 54.1, 49.8, 51.0, 54.6, 53.5, 62.3, 58.7],
    "LaBraM":  [54.5, 53.7, 56.2, 54.8, 53.0, 55.5, 57.0, 56.6, 57.4],
    "CBraMod": [49.9, 51.6, 53.2, 50.3, 48.1, 54.0, 54.2, 55.1, 55.5],
}
mi_trackC = {
    "BENDR":   [53.3, 55.0, 50.9, 49.6, 51.1, 53.1, 52.8, 51.7, 52.4],
    "EEGPT":   [57.2, 55.7, 64.7, 61.1, 58.3, 59.1, 62.4, 61.5, 59.4],
    "NeuroLM": [60.3, 55.5, 58.4, 50.4, 48.4, 54.9, 60.8, 57.6, 58.7],
    "LaBraM":  [52.3, 55.2, 55.4, 53.4, 51.4, 53.0, 56.4, 52.4, 56.2],
    "CBraMod": [50.6, 54.3, 53.2, 51.8, 49.7, 55.4, 54.4, 54.1, 57.0],
}
mi_trackC_std = {
    "BENDR":   [6.6, 11.0, 5.2, 2.2, 3.6, 1.9, 3.4, 2.6, 2.0],
    "EEGPT":   [9.8, 6.7, 11.9, 8.0, 11.2, 6.8, 10.2, 9.7, 8.7],
    "NeuroLM": [9.3, 8.9, 9.3, 2.2, 3.5, 4.7, 6.7, 5.4, 6.0],
    "LaBraM":  [7.3, 9.3, 6.9, 3.6, 4.0, 3.4, 4.2, 2.2, 7.3],
    "CBraMod": [6.5, 9.7, 5.4, 2.9, 2.4, 5.4, 5.5, 3.4, 11.7],
}

ssvep_info = [
    ("Liu2020BETA",    70, 64, 11200, 40, 250, "CN"),
    ("Wang2016",       34, 64,  8160, 40, 250, "CN"),
    ("Nakanishi2015",   9,  8,  1620, 12, 256, "US"),
]
ssvep_chance = [2.5, 2.5, 8.3]

ssvep_trackA = {
    "BENDR":   [4.4, 8.4, 9.1],
    "EEGPT":   [14.7, 23.6, 32.2],
    "NeuroLM": [5.6, 7.6, 25.3],
    "LaBraM":  [7.1, 13.1, 17.0],
    "CBraMod": [4.5, 6.1, 16.2],
}
ssvep_trackA_std = {
    "BENDR":   [2.1, 3.7, 1.9],
    "EEGPT":   [7.1, 8.4, 10.8],
    "NeuroLM": [3.7, 3.8, 10.7],
    "LaBraM":  [3.8, 7.2, 4.9],
    "CBraMod": [1.8, 2.5, 5.2],
}
ssvep_trackB = {
    "BENDR":   [4.8, 7.0, 10.1],
    "EEGPT":   [14.8, 24.1, 34.6],
    "NeuroLM": [5.3, 9.4, 34.1],
    "LaBraM":  [9.6, 18.9, 30.2],
    "CBraMod": [4.9, 7.5, 23.4],
}
ssvep_trackB_std = {
    "BENDR":   [2.6, 3.0, 2.5],
    "EEGPT":   [7.2, 8.3, 10.7],
    "NeuroLM": [3.1, 5.9, 17.0],
    "LaBraM":  [5.4, 10.3, 9.5],
    "CBraMod": [1.9, 3.7, 7.6],
}

erp_info = [
    ("BI2014a",       64, 16, 1188, 2, 512, "FR"),
    ("BI2015a",       43, 32, 2574, 2, 512, "FR"),
    ("BNCI2014\\_008",  8,  8, 4200, 2, 256, "AT"),
    ("BNCI2014\\_009", 10, 16,  576, 2, 256, "AT"),
]

erp_auc = {
    "EEGPT":   [61.0, 68.0, 53.6, 61.9],
    "NeuroLM": [58.3, 61.8, 63.7, 75.4],
    "LaBraM":  [54.7, 58.8, 59.1, 73.5],
    "CBraMod": [53.9, 56.7, 55.6, 64.7],
}
erp_auc_std = {
    "EEGPT":   [5.7, 7.0, 2.1, 4.6],
    "NeuroLM": [6.2, 5.5, 5.5, 6.4],
    "LaBraM":  [3.0, 3.8, 4.8, 5.9],
    "CBraMod": [2.9, 4.8, 4.8, 6.1],
}
erp_f1 = {
    "EEGPT":   [49.3, 53.0, 47.4, 54.7],
    "NeuroLM": [46.0, 45.9, 51.2, 62.3],
    "LaBraM":  [45.5, 45.5, 46.9, 58.2],
    "CBraMod": [45.5, 45.5, 45.6, 49.0],
}


def _fmt(v, s=None):
    """Format value with optional tiny std."""
    base = f"{v:.1f}"
    if s is not None:
        return base + r"{\tiny$\pm$" + f"{s:.1f}" + "}"
    return base


def _bold_row(vals, stds=None):
    """Format a row of values, bolding the best (highest)."""
    best = int(np.argmax(vals))
    out = []
    for i, v in enumerate(vals):
        s = _fmt(v, stds[i] if stds else None)
        if i == best:
            s = r"\textbf{" + s + "}"
        out.append(s)
    return out


def _delta_fmt(v):
    """Format a delta value with sign and color hint."""
    return f"{v:+.1f}"


# ──────────────────────────────────────────────────────────────────
# TABLE 1: Dataset overview (compact, 8 columns)
# ──────────────────────────────────────────────────────────────────
def generate_table1_datasets():
    lines = [r"""\begin{table}[t]
\centering
\caption{\textbf{Dataset overview.} All 16 datasets grouped by BCI paradigm.
MI: binary left/right hand. SSVEP: 12--40 frequency classes. ERP: binary
target/non-target with 5:1 class imbalance.
$N$\,=\,subjects, $C$\,=\,channels, $K$\,=\,classes, $T$\,=\,total trials.}
\label{tab:datasets}
\vskip 0.1in
\small
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
\textbf{Paradigm} & \textbf{Dataset} & $N$ & $C$ & $K$ & $T$ & \textbf{Hz} & \textbf{Origin} \\
\midrule"""]

    for i, (name, subj, ch, trials, cls, sfreq, country) in enumerate(mi_info):
        p = r"\multirow{9}{*}{\rotatebox[origin=c]{90}{MI}}" if i == 0 else ""
        lines.append(f"{p} & {name} & {subj} & {ch} & {cls} & {trials:,} & {sfreq} & {country} \\\\")
    lines.append(r"\midrule")

    for i, (name, subj, ch, trials, cls, sfreq, country) in enumerate(ssvep_info):
        p = r"\multirow{3}{*}{\rotatebox[origin=c]{90}{SSVEP}}" if i == 0 else ""
        lines.append(f"{p} & {name} & {subj} & {ch} & {cls} & {trials:,} & {sfreq} & {country} \\\\")
    lines.append(r"\midrule")

    for i, (name, subj, ch, trials, cls, sfreq, country) in enumerate(erp_info):
        p = r"\multirow{4}{*}{\rotatebox[origin=c]{90}{ERP}}" if i == 0 else ""
        lines.append(f"{p} & {name} & {subj} & {ch} & {cls} & {trials:,} & {sfreq} & {country} \\\\")

    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# TABLE 2: Foundation model specifications (6 columns)
# ──────────────────────────────────────────────────────────────────
def generate_table2_models():
    return r"""\begin{table}[t]
\centering
\caption{\textbf{Foundation model specifications.} All models evaluated via
frozen linear probe (LogReg, max\_iter=1000). Two preprocessing groups
share the same resampled data: BENDR/EEGPT at 256\,Hz and
NeuroLM/LaBraM/CBraMod at 200\,Hz.}
\label{tab:models}
\vskip 0.1in
\small
\begin{tabular}{@{}lccll@{}}
\toprule
\textbf{Model} & \textbf{Emb.\ dim} & \textbf{Sfreq} & \textbf{Track~A band} & \textbf{Pretraining data} \\
\midrule
BENDR   & 2048 & 256\,Hz & 0.5--70\,Hz  & TUH-EEG (clinical)  \\
EEGPT   & 2048 & 256\,Hz & 0.5--70\,Hz  & Large-scale EEG     \\
NeuroLM &  768 & 200\,Hz & 0.1--75\,Hz  & Multi-domain EEG    \\
LaBraM  &  400 & 200\,Hz & 0.1--75\,Hz  & Large-scale EEG     \\
CBraMod &  200 & 200\,Hz & 0.1--75\,Hz  & Channel-aware EEG   \\
\bottomrule
\end{tabular}
\end{table}"""


# ──────────────────────────────────────────────────────────────────
# TABLE 3: MI results — stacked tracks as sub-rows (8 columns)
# ──────────────────────────────────────────────────────────────────
def generate_table3_mi():
    lines = [r"""\begin{table}[t]
\centering
\caption{\textbf{Motor imagery results} (LOSO, accuracy~\%, chance\,=\,50).
Binary left/right hand classification with frozen linear probe.
Track~A: FM-aligned wideband.
Track~C: FM-aligned with tmin\,=\,1\,s (cue-onset artefact removed).
$\Delta$: Track~C\,$-$\,Track~A.
Best per row in \textbf{bold}.}
\label{tab:mi}
\vskip 0.1in
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrccccc@{}}
\toprule
\textbf{Dataset} & \textbf{Track} & $N$ & BENDR & EEGPT & NeuroLM & LaBraM & CBraMod \\
\midrule"""]

    for i, (name, subj, ch, trials, cls, sfreq, country) in enumerate(mi_info):
        a_vals = _bold_row(
            [mi_trackA[m][i] for m in MODELS],
            [mi_trackA_std[m][i] for m in MODELS],
        )
        c_vals = _bold_row(
            [mi_trackC[m][i] for m in MODELS],
            [mi_trackC_std[m][i] for m in MODELS],
        )
        d_vals = [_delta_fmt(mi_trackC[m][i] - mi_trackA[m][i]) for m in MODELS]

        nrows = 3
        lines.append(
            f"\\multirow{{{nrows}}}{{*}}{{{name}}}"
            f" & A & \\multirow{{{nrows}}}{{*}}{{{subj}}}"
            f" & " + " & ".join(a_vals) + r" \\"
        )
        lines.append(f" & C & & " + " & ".join(c_vals) + r" \\")
        lines.append(
            r" & $\Delta$ & & "
            + " & ".join(d_vals)
            + r" \\"
        )
        if i < len(mi_info) - 1:
            lines.append(r"\addlinespace[2pt]")

    # Mean row
    lines.append(r"\midrule")
    a_means = _bold_row([np.mean(mi_trackA[m]) for m in MODELS])
    c_means = _bold_row([np.mean(mi_trackC[m]) for m in MODELS])
    d_means = [_delta_fmt(np.mean(mi_trackC[m]) - np.mean(mi_trackA[m])) for m in MODELS]
    lines.append(r"\multirow{3}{*}{\textbf{Mean}} & A & \multirow{3}{*}{320} & " + " & ".join(a_means) + r" \\")
    lines.append(r" & C & & " + " & ".join(c_means) + r" \\")
    lines.append(r" & $\Delta$ & & " + " & ".join(d_means) + r" \\")

    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# TABLE 4: MI preprocessing band impact (compact delta table)
# ──────────────────────────────────────────────────────────────────
def generate_table4_mi_preproc():
    lines = [r"""\begin{table}[t]
\centering
\caption{\textbf{MI preprocessing sensitivity} (LOSO, $\Delta$ in pp).
$\Delta_\text{band}$: Track~B\,(4--40\,Hz) $-$ Track~A\,(FM-aligned).
$\Delta_\text{cue}$: Track~C\,(tmin=1\,s) $-$ Track~A\,(tmin=0\,s).
Negative $\Delta_\text{band}$\,=\,model needs wideband.
Negative $\Delta_\text{cue}$\,=\,model exploited cue-onset artefact.}
\label{tab:mi_preproc}
\vskip 0.1in
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
& BENDR & EEGPT & NeuroLM & LaBraM & CBraMod \\
\midrule"""]

    # Per-dataset band deltas
    for i, (name, subj, ch, trials, cls, sfreq, country) in enumerate(mi_info):
        d = [_delta_fmt(mi_trackB[m][i] - mi_trackA[m][i]) for m in MODELS]
        lines.append(f"{name} & " + " & ".join(d) + r" \\")

    lines.append(r"\midrule")
    band_means = [_delta_fmt(np.mean([mi_trackB[m][i] - mi_trackA[m][i] for i in range(9)])) for m in MODELS]
    cue_means = [_delta_fmt(np.mean([mi_trackC[m][i] - mi_trackA[m][i] for i in range(9)])) for m in MODELS]
    lines.append(r"\textbf{Mean} $\Delta_\text{band}$ & " + " & ".join(band_means) + r" \\")
    lines.append(r"\textbf{Mean} $\Delta_\text{cue}$ & " + " & ".join(cue_means) + r" \\")

    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# TABLE 5: SSVEP results — stacked tracks (8 columns)
# ──────────────────────────────────────────────────────────────────
def generate_table5_ssvep():
    lines = [r"""\begin{table}[t]
\centering
\caption{\textbf{SSVEP results} (LOSO, accuracy~\%).
Multi-class frequency classification. Chance: 2.5\% (40-class), 8.3\%
(12-class). Track~A: FM-aligned. Track~B: 1--50\,Hz. $\Delta$: B\,$-$\,A.
Best per row in \textbf{bold}.}
\label{tab:ssvep}
\vskip 0.1in
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrrccccc@{}}
\toprule
\textbf{Dataset} & \textbf{Track} & $N$ & $K$ & BENDR & EEGPT & NeuroLM & LaBraM & CBraMod \\
\midrule"""]

    for i, (name, subj, ch, trials, cls, sfreq, country) in enumerate(ssvep_info):
        a_vals = _bold_row(
            [ssvep_trackA[m][i] for m in MODELS],
            [ssvep_trackA_std[m][i] for m in MODELS],
        )
        b_vals = _bold_row(
            [ssvep_trackB[m][i] for m in MODELS],
            [ssvep_trackB_std[m][i] for m in MODELS],
        )
        d_vals = [_delta_fmt(ssvep_trackB[m][i] - ssvep_trackA[m][i]) for m in MODELS]

        lines.append(
            f"\\multirow{{3}}{{*}}{{{name}}}"
            f" & A & \\multirow{{3}}{{*}}{{{subj}}} & \\multirow{{3}}{{*}}{{{cls}}}"
            f" & " + " & ".join(a_vals) + r" \\"
        )
        lines.append(f" & B & & & " + " & ".join(b_vals) + r" \\")
        lines.append(r" & $\Delta$ & & & " + " & ".join(d_vals) + r" \\")
        if i < len(ssvep_info) - 1:
            lines.append(r"\addlinespace[2pt]")

    # Mean delta
    lines.append(r"\midrule")
    d_means = [_delta_fmt(np.mean([ssvep_trackB[m][i] - ssvep_trackA[m][i] for i in range(3)])) for m in MODELS]
    lines.append(r"\textbf{Mean} & $\Delta$ & & & " + " & ".join(d_means) + r" \\")

    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# TABLE 6: ERP/P300 results (8 columns)
# ──────────────────────────────────────────────────────────────────
def generate_table6_erp():
    erp_models = ["EEGPT", "NeuroLM", "LaBraM", "CBraMod"]
    lines = [r"""\begin{table}[t]
\centering
\caption{\textbf{ERP/P300 results} (LOSO, Track~A).
Binary target vs.\ non-target (5:1 imbalance). Primary metric: ROC-AUC\,(\%).
Accuracy $\approx$83\% for all models (trivial majority-class baseline).
BENDR is incompatible with 1\,s trials.
Best per row in \textbf{bold}.}
\label{tab:erp}
\vskip 0.1in
\small
\begin{tabular}{@{}lrrr|cccc@{}}
\toprule
\textbf{Dataset} & $N$ & $C$ & $T$/subj & EEGPT & NeuroLM & LaBraM & CBraMod \\
\midrule
\multicolumn{8}{@{}l}{\textit{ROC-AUC (\%)}} \\
\midrule"""]

    for i, (name, subj, ch, trials, cls, sfreq, country) in enumerate(erp_info):
        vals = _bold_row(
            [erp_auc[m][i] for m in erp_models],
            [erp_auc_std[m][i] for m in erp_models],
        )
        lines.append(f"{name} & {subj} & {ch} & {trials:,} & " + " & ".join(vals) + r" \\")

    lines.append(r"\midrule")
    auc_means = _bold_row([np.mean(erp_auc[m]) for m in erp_models])
    lines.append(r"\textbf{Mean} & 125 & & & " + " & ".join(auc_means) + r" \\")

    lines.append(r"""\midrule
\multicolumn{8}{@{}l}{\textit{Macro-F1 (\%)}} \\
\midrule""")

    for i, (name, subj, ch, trials, cls, sfreq, country) in enumerate(erp_info):
        vals = _bold_row([erp_f1[m][i] for m in erp_models])
        lines.append(f"{name} & {subj} & {ch} & {trials:,} & " + " & ".join(vals) + r" \\")

    lines.append(r"\midrule")
    f1_means = _bold_row([np.mean(erp_f1[m]) for m in erp_models])
    lines.append(r"\textbf{Mean} & 125 & & & " + " & ".join(f1_means) + r" \\")

    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# TABLE 7: Grand cross-paradigm summary (7 columns)
# ──────────────────────────────────────────────────────────────────
def generate_table7_grand():
    lines = [r"""\begin{table}[t]
\centering
\caption{\textbf{Cross-paradigm summary} (LOSO).
MI: Track~C accuracy\,(\%, cue-free). SSVEP: Track~B accuracy\,(\%).
ERP: Track~A ROC-AUC\,(\%).
$\Delta_\text{band}$: task-specific band $-$ FM-aligned (MI: 4--40\,Hz; SSVEP: 1--50\,Hz).
$\Delta_\text{cue}$: tmin=1\,s $-$ tmin=0\,s.}
\label{tab:grand}
\vskip 0.1in
\small
\begin{tabular}{@{}lccc|cc|c@{}}
\toprule
& \multicolumn{3}{c|}{\textbf{Performance}} & \multicolumn{2}{c|}{$\Delta_\text{band}$} & $\Delta_\text{cue}$ \\
\textbf{Model} & MI & SSVEP & ERP & MI & SSVEP & MI \\
\midrule"""]

    erp_set = {"EEGPT", "NeuroLM", "LaBraM", "CBraMod"}
    for m in MODELS:
        mi_c = np.mean(mi_trackC[m])
        ssvep_b = np.mean(ssvep_trackB[m])
        erp_v = f"{np.mean(erp_auc[m]):.1f}" if m in erp_set else "---"
        d_mi = np.mean([mi_trackB[m][i] - mi_trackA[m][i] for i in range(9)])
        d_ssvep = np.mean([ssvep_trackB[m][i] - ssvep_trackA[m][i] for i in range(3)])
        d_cue = np.mean([mi_trackC[m][i] - mi_trackA[m][i] for i in range(9)])
        lines.append(f"{m} & {mi_c:.1f} & {ssvep_b:.1f} & {erp_v} & {d_mi:+.1f} & {d_ssvep:+.1f} & {d_cue:+.1f} \\\\")

    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# TABLE 8: Preprocessing pipeline specs (6 columns)
# ──────────────────────────────────────────────────────────────────
def generate_table8_preprocessing():
    return r"""\begin{table}[t]
\centering
\caption{\textbf{Preprocessing tracks.} All tracks apply CAR,
per-dataset notch (50/60\,Hz), and V\,$\to$\,$\mu$V scaling.
Band notation: BENDR/EEGPT\,/\,NeuroLM/LaBraM/CBraMod.}
\label{tab:preproc}
\vskip 0.1in
\small
\begin{tabular}{@{}cllcc@{}}
\toprule
\textbf{Track} & \textbf{Band (MI)} & \textbf{Band (SSVEP)} & \textbf{tmin} & \textbf{Duration} \\
\midrule
A & 0.5--70\,/\,0.1--75\,Hz & 0.5--70\,/\,0.1--75\,Hz & 0\,s & 4\,s \\
B & 4--40\,Hz              & 1--50\,Hz                & 0\,s & 4\,s \\
C & 0.5--70\,/\,0.1--75\,Hz & ---                      & 1\,s & 3\,s \\
D & 4--40\,Hz              & ---                      & 1\,s & 3\,s \\
\bottomrule
\end{tabular}
\end{table}"""


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tables = {
        "table1_datasets.tex":     generate_table1_datasets(),
        "table2_models.tex":       generate_table2_models(),
        "table3_mi.tex":           generate_table3_mi(),
        "table4_mi_preproc.tex":   generate_table4_mi_preproc(),
        "table5_ssvep.tex":        generate_table5_ssvep(),
        "table6_erp.tex":          generate_table6_erp(),
        "table7_grand.tex":        generate_table7_grand(),
        "table8_preprocessing.tex": generate_table8_preprocessing(),
    }

    for fname, content in tables.items():
        (OUT / fname).write_text(content)
        print(f"  Saved {fname}")

    combined = [
        r"% EEG Foundation Model Benchmark — LaTeX Tables (NeurIPS style)",
        r"% Generated by reproduction/figures/generate_latex_tables.py",
        r"% Requires: booktabs, multirow",
        "",
    ]
    for fname, content in tables.items():
        combined.append(f"% === {fname} ===")
        combined.append(content)
        combined.append("")
    (OUT / "all_tables.tex").write_text("\n".join(combined))
    print(f"\n  Combined: all_tables.tex")
    print(f"  Output:   {OUT}")
