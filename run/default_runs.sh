#!/usr/bin/env bash
#
# Every experiment NeuroAtlas reports, one command per line.
#
# The reference for "what was actually run". Each line is complete and
# copy-pasteable, and the settings that DIFFER between experiments are written
# out rather than left to a default a reader would have to go looking for.
#
# Constant across every dataset, so left at the default:
#   preprocessing   0.5 Hz highpass + powerline notch, then resample to each
#                   model's training rate (Sec. 3). cohort.powerline_hz in the
#                   manifest gives the notch frequency.
#   backbone        frozen. No fine-tuning anywhere in the benchmark.
#   solver          LBFGS.
#   precision       fp32.
#   folds           patient-level 5-fold, except BCI, which is LOSO.
#
# What varies, and is therefore explicit below (Appendix C):
#   window/stride   10 s non-overlapping for epilepsy (C.1); 30 s epochs for
#                   sleep and brain age (C.2, C.3); per-paradigm trial windows
#                   for BCI (C.4), taken from each dataset's manifest.
#   probe           logistic regression, except brain age, which is ridge.
#   C / alpha       epilepsy grid-searches C on a validation fold; sleep and
#                   BCI use C=1.0; brain age nested-CVs alpha internally.
#   class weight    balanced for epilepsy, sleep events/diagnosis and BCI;
#                   unweighted for sleep staging (C.2).
#   aggregation     none for epoch-level tasks; mean-pooled per recording for
#                   diagnosis and brain age.
#   selection       AUPRC for binary seizure detection, MAE for brain age.
#
# --models is omitted, meaning every checkpoint in the registry. Add
# --models eeg_fm | ts_fm | supervised for one of the paper's three groups
# (src/neuroatlas/configs/model_groups.yaml).
#
# Protocols follow the paper's Appendix C; see src/neuroatlas/configs/tasks/ for the defaults.
#
set -euo pipefail


# ==========================================================================
# 1. FETCH -- obtain every corpus
# Roughly half need credentials or a manual download; `fetch` prints
# what to request and where to put it, and transfers nothing without
# --download.
# ==========================================================================
python -m neuroatlas.entrypoints.fetch --dataset bonn
python -m neuroatlas.entrypoints.fetch --dataset chbmit
python -m neuroatlas.entrypoints.fetch --dataset epilepsiae
python -m neuroatlas.entrypoints.fetch --dataset helsinki_neonatal
python -m neuroatlas.entrypoints.fetch --dataset nmt
python -m neuroatlas.entrypoints.fetch --dataset siena
python -m neuroatlas.entrypoints.fetch --dataset sz1
python -m neuroatlas.entrypoints.fetch --dataset sz2
python -m neuroatlas.entrypoints.fetch --dataset tuab
python -m neuroatlas.entrypoints.fetch --dataset tusz
python -m neuroatlas.entrypoints.fetch --dataset cfs
python -m neuroatlas.entrypoints.fetch --dataset dcsm
python -m neuroatlas.entrypoints.fetch --dataset dod
python -m neuroatlas.entrypoints.fetch --dataset hmc
python -m neuroatlas.entrypoints.fetch --dataset hpap_lab_full
python -m neuroatlas.entrypoints.fetch --dataset isruc
python -m neuroatlas.entrypoints.fetch --dataset mass
python -m neuroatlas.entrypoints.fetch --dataset mesa
python -m neuroatlas.entrypoints.fetch --dataset mros
python -m neuroatlas.entrypoints.fetch --dataset physionet2026
python -m neuroatlas.entrypoints.fetch --dataset shhs
python -m neuroatlas.entrypoints.fetch --dataset sleep_edf_expanded
python -m neuroatlas.entrypoints.fetch --dataset stages
python -m neuroatlas.entrypoints.fetch --dataset ucddb
python -m neuroatlas.entrypoints.fetch --dataset wsc
python -m neuroatlas.entrypoints.fetch --dataset cfs
python -m neuroatlas.entrypoints.fetch --dataset isruc
python -m neuroatlas.entrypoints.fetch --dataset mros
python -m neuroatlas.entrypoints.fetch --dataset physionet2026
python -m neuroatlas.entrypoints.fetch --dataset sleep_edf_expanded
python -m neuroatlas.entrypoints.fetch --dataset wsc
python -m neuroatlas.entrypoints.fetch --dataset bi2013a
python -m neuroatlas.entrypoints.fetch --dataset bi2014a
python -m neuroatlas.entrypoints.fetch --dataset bnci2014_001
python -m neuroatlas.entrypoints.fetch --dataset bnci2014_004
python -m neuroatlas.entrypoints.fetch --dataset bnci2014_008
python -m neuroatlas.entrypoints.fetch --dataset bnci2015_001
python -m neuroatlas.entrypoints.fetch --dataset dreyer2023
python -m neuroatlas.entrypoints.fetch --dataset epflp300
python -m neuroatlas.entrypoints.fetch --dataset erpcore2021_n170
python -m neuroatlas.entrypoints.fetch --dataset kim2025betarange
python -m neuroatlas.entrypoints.fetch --dataset liu2024
python -m neuroatlas.entrypoints.fetch --dataset nakanishi2015
python -m neuroatlas.entrypoints.fetch --dataset shin2017a
python -m neuroatlas.entrypoints.fetch --dataset weibo2014

# ==========================================================================
# 2. PREPARE -- none of the 45 needs a build step
# Every cohort reads its raw corpus directly; the MOABB cohorts are
# epoched by their reader as `embed` loads them. Two kinds of optional
# build exist and are not listed here: the epilepsy HDF5 caches that
# speed up large sweeps (`prepare --dataset tusz`), and dataio/bci.py's
# pickles for five MI cohorts (`prepare --dataset bnci2014_001`), which
# `embed` does not read. (The paper's BCI embeddings were extracted from
# per-model pickles of an earlier pipeline; see the MI cohort manifests'
# known_issues.)
# ==========================================================================

# ==========================================================================
# 3. EMBED -- frozen backbones, one pass per (dataset, model)
# ==========================================================================

# Epilepsy: 10 s windows, no overlap (C.1). --expected-epoch-seconds tells
# each backbone the window it is handed, so one pretrained on 30 s epochs
# (BIOT) does not reject a 10 s one; every epilepsy launcher set it.
python -m neuroatlas.entrypoints.embed --models all --dataset bonn --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all --dataset chbmit --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all --dataset epilepsiae --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all --dataset helsinki_neonatal --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all --dataset nmt --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all --dataset siena --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all --dataset sz1 --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all --dataset sz2 --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all --dataset tuab --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all --dataset tusz --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10

# Sleep and brain age: 30 s epochs (C.2, C.3). Cohorts whose readers are
# built on fixed 30 s scoring epochs take no window arguments -- passing
# them was a TypeError -- so only the configurable ones are told.
python -m neuroatlas.entrypoints.embed --models all --dataset cfs --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all --dataset dcsm
python -m neuroatlas.entrypoints.embed --models all --dataset dod
python -m neuroatlas.entrypoints.embed --models all --dataset hmc --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all --dataset hpap_lab_full --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all --dataset isruc
python -m neuroatlas.entrypoints.embed --models all --dataset mass
python -m neuroatlas.entrypoints.embed --models all --dataset mesa --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all --dataset mros --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all --dataset physionet2026
python -m neuroatlas.entrypoints.embed --models all --dataset shhs
python -m neuroatlas.entrypoints.embed --models all --dataset sleep_edf_expanded
python -m neuroatlas.entrypoints.embed --models all --dataset stages --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all --dataset ucddb
python -m neuroatlas.entrypoints.embed --models all --dataset wsc
python -m neuroatlas.entrypoints.embed --models all --dataset cfs --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all --dataset isruc
python -m neuroatlas.entrypoints.embed --models all --dataset mros --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all --dataset physionet2026
python -m neuroatlas.entrypoints.embed --models all --dataset sleep_edf_expanded
python -m neuroatlas.entrypoints.embed --models all --dataset wsc

# BCI (C.4). Trial windows come from the manifest, and two axes
# are crossed over every cohort.
#
#   confound_control  an MI trial opens with a visual cue, so its
#                     first second holds the evoked response and the
#                     eye movement towards it. On holds the band to
#                     4-40 Hz and starts the window 1 s in. The gap
#                     between off and on is how much of a score came
#                     from the cue rather than the imagery. Motor
#                     imagery only -- for P300, ERP and SSVEP the
#                     cue-locked response IS the signal, and `embed`
#                     refuses the flag there.
#   pooling           what comes back for each window, the window
#                     itself unchanged: `mean` is one vector
#                     averaged over that window's patch tokens,
#                     `per_patch` keeps that window's tokens apart.
#
# Each combination is its own cache cell, so none overwrites another.

#   --- confound_control off, pooling mean ------------------------
python -m neuroatlas.entrypoints.embed --models all --dataset bi2013a --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset bi2014a --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset bnci2014_001 --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset bnci2014_004 --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset bnci2014_008 --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset bnci2015_001 --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset dreyer2023 --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset epflp300 --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset erpcore2021_n170 --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset kim2025betarange --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset liu2024 --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset nakanishi2015 --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset shin2017a --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset weibo2014 --pooling mean

#   --- confound_control off, pooling per_patch ------------------------
python -m neuroatlas.entrypoints.embed --models all --dataset bi2013a --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset bi2014a --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset bnci2014_001 --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset bnci2014_004 --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset bnci2014_008 --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset bnci2015_001 --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset dreyer2023 --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset epflp300 --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset erpcore2021_n170 --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset kim2025betarange --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset liu2024 --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset nakanishi2015 --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset shin2017a --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset weibo2014 --pooling per_patch

#   --- confound_control ON, pooling mean ------------------------
python -m neuroatlas.entrypoints.embed --models all --dataset bnci2014_001 --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset bnci2014_004 --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset bnci2015_001 --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset dreyer2023 --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset liu2024 --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset shin2017a --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset weibo2014 --set confound_control=true --pooling mean

#   --- confound_control ON, pooling per_patch ------------------------
python -m neuroatlas.entrypoints.embed --models all --dataset bnci2014_001 --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset bnci2014_004 --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset bnci2015_001 --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset dreyer2023 --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset liu2024 --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset shin2017a --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all --dataset weibo2014 --set confound_control=true --pooling per_patch

# ==========================================================================
# 4. PROBE -- reads the embeddings above; refuses if they are absent
# A probe finds the embeddings by a key computed from the dataset config,
# so each probe repeats its embed line's --set flags, spelled the same
# (window_s=10, not 10.0: the key hashes the values as written).
# --expected-epoch-seconds belongs to the checkpoint and is not part of
# the key, so it stays on the embed line.
# ==========================================================================

# --- Epilepsy: seizure detection, AUPRC-selected C (C.1) ------------------
python -m neuroatlas.entrypoints.probe --models all --dataset bonn --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all --dataset chbmit --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all --dataset epilepsiae --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all --dataset helsinki_neonatal --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all --dataset nmt --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all --dataset siena --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all --dataset sz1 --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all --dataset sz2 --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all --dataset tuab --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all --dataset tusz --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc

# --- Sleep axis 1: epoch-wise staging, unweighted loss (C.2) --------------
python -m neuroatlas.entrypoints.probe --models all --dataset cfs --task sleep_staging --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.probe --models all --dataset dcsm --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all --dataset dod --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all --dataset hmc --task sleep_staging --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.probe --models all --dataset hpap_lab_full --task sleep_staging --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.probe --models all --dataset isruc --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all --dataset mass --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all --dataset mesa --task sleep_staging --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.probe --models all --dataset mros --task sleep_staging --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.probe --models all --dataset physionet2026 --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all --dataset shhs --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all --dataset sleep_edf_expanded --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all --dataset stages --task sleep_staging --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.probe --models all --dataset ucddb --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all --dataset wsc --task sleep_staging

# --- Sleep axis 2: hypnogram features (C.2, Table 2) ------------------
#     Derived from the staging predictions above -- TST, REM latency,
#     WASO, bout durations, transition rates. No embedding pass and no
#     model: `reconstruct` re-predicts with each saved probe to get
#     per-recording epoch sequences, then `features` turns those into
#     the 34 features. Runs after the staging probes above, which is
#     why it is last in the file.

# --- Sleep axis 3: event detection, balanced loss (C.2) --------------------
python -m neuroatlas.entrypoints.probe --models all --dataset mass --task mass_arousal --probe-type linear --class-weight balanced --tune-c 1.0
python -m neuroatlas.entrypoints.probe --models all --dataset physionet2026 --task arousal --probe-type linear --class-weight balanced --tune-c 1.0
python -m neuroatlas.entrypoints.probe --models all --dataset physionet2026 --task limb --probe-type linear --class-weight balanced --tune-c 1.0
python -m neuroatlas.entrypoints.probe --models all --dataset physionet2026 --task respiratory --probe-type linear --class-weight balanced --tune-c 1.0
python -m neuroatlas.entrypoints.probe --models all --dataset ucddb --task respiratory --probe-type linear --class-weight balanced --tune-c 1.0

# --- Sleep axis 4: recording-level diagnosis, mean-pooled (C.2) ------------
python -m neuroatlas.entrypoints.probe --models all --dataset dod --task osa --probe-type linear --class-weight balanced --tune-c 1.0 --aggregation mean
python -m neuroatlas.entrypoints.probe --models all --dataset isruc --task pathology --probe-type linear --class-weight balanced --tune-c 1.0 --aggregation mean
python -m neuroatlas.entrypoints.probe --models all --dataset physionet2026 --task cognitive --probe-type linear --class-weight balanced --tune-c 1.0 --aggregation mean

# --- Brain age: ridge on mean-pooled epochs, nested-CV alpha (C.3) --------
#     A task over the sleep cohorts that carry age labels, not a
#     separate domain. The brain_age task fits the ridge and selects
#     alpha itself, so no --probe-type or --tune-c here.
python -m neuroatlas.entrypoints.probe --models all --dataset cfs --task brain_age --set window_s=30 --set stride_s=30 --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all --dataset isruc --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all --dataset mros --task brain_age --set window_s=30 --set stride_s=30 --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all --dataset physionet2026 --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all --dataset sleep_edf_expanded --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all --dataset wsc --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all --dataset cfs --task brain_age --set window_s=30 --set stride_s=30 --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all --dataset isruc --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all --dataset mros --task brain_age --set window_s=30 --set stride_s=30 --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all --dataset physionet2026 --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all --dataset sleep_edf_expanded --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all --dataset wsc --task brain_age --set label_mode=age --aggregation mean

# --- BCI: LOSO, balanced logistic regression (C.4) ------------------------
#     n_folds=loso is one fold per subject, whatever the cohort
#     size (dataio/bci.py:resolve_n_folds). Cohorts run from 8 to
#     100 subjects, so one setting covers all of them.

#   --- confound_control off, pooling mean ------------------------
python -m neuroatlas.entrypoints.probe --models all --dataset bi2013a --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset bi2014a --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset bnci2014_001 --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset bnci2014_004 --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset bnci2014_008 --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset bnci2015_001 --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset dreyer2023 --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset epflp300 --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset erpcore2021_n170 --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset kim2025betarange --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset liu2024 --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset nakanishi2015 --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset shin2017a --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset weibo2014 --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000

#   --- confound_control off, pooling per_patch ------------------------
python -m neuroatlas.entrypoints.probe --models all --dataset bi2013a --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset bi2014a --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset bnci2014_001 --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset bnci2014_004 --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset bnci2014_008 --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset bnci2015_001 --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset dreyer2023 --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset epflp300 --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset erpcore2021_n170 --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset kim2025betarange --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset liu2024 --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset nakanishi2015 --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset shin2017a --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset weibo2014 --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000

#   --- confound_control ON, pooling mean ------------------------
python -m neuroatlas.entrypoints.probe --models all --dataset bnci2014_001 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset bnci2014_004 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset bnci2015_001 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset dreyer2023 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset liu2024 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset shin2017a --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset weibo2014 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000

#   --- confound_control ON, pooling per_patch ------------------------
python -m neuroatlas.entrypoints.probe --models all --dataset bnci2014_001 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset bnci2014_004 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset bnci2015_001 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset dreyer2023 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset liu2024 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset shin2017a --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset weibo2014 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000

# ==========================================================================
# 5. HYPNOGRAM -- sleep-architecture features from the staging probes
# The one paper result that is not a probe (C.2, Table 2). It reads the
# probe output written by section 4, so it runs last.
# ==========================================================================
python -m neuroatlas.entrypoints.hypnogram --datasets dcsm
python -m neuroatlas.entrypoints.hypnogram --datasets dod
python -m neuroatlas.entrypoints.hypnogram --datasets isruc
python -m neuroatlas.entrypoints.hypnogram --datasets mass
python -m neuroatlas.entrypoints.hypnogram --datasets physionet2026
python -m neuroatlas.entrypoints.hypnogram --datasets sleep_edf_expanded
python -m neuroatlas.entrypoints.hypnogram --datasets ucddb
python -m neuroatlas.entrypoints.hypnogram --datasets wsc

# ==========================================================================
# All of the paper's datasets are now registered.
#
# The registry holds 43 paper cohorts for 42 paper datasets: DREAMER is split
# into dreamer_valence and dreamer_arousal, which is how its pickles, its
# embeddings and its probes are stored, and how the paper reports it.
#
# EEGMat, ArithmeticTask and DREAMER read preprocessed pickles rather than a
# downloadable corpus -- `fetch` has nothing to do for them. Point
# --set preprocessed_path=<file> at the pickle, or place it where
# dataio/bci.py's PREPROCESSED_SEARCH_PATHS looks.

python -m neuroatlas.entrypoints.embed --models all --dataset dreamer_valence --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset dreamer_arousal --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset eegmat --pooling mean
python -m neuroatlas.entrypoints.embed --models all --dataset arithmetic_task --pooling mean

python -m neuroatlas.entrypoints.probe --models all --dataset dreamer_valence --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset dreamer_arousal --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset eegmat --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all --dataset arithmetic_task --pooling mean --set n_folds=loso --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000

# Brain age (App. B.3) is reported on 10 cohorts: CFS, HomePAP, ISRUC, MESA,
# MrOS, PhysioNet 2026, SHHS, SleepEDF SC, STAGES, WSC. Six run above -- the
# six whose dossier declares an `age` label mode.
#
# The other four (HomePAP, MESA, SHHS, STAGES) are all registered and all run
# sleep staging, but each declares `sleep_stage` only: their loaders carry no
# demographics, so there is no age to regress on. For SHHS that means joining
# the NSRR shhs1-dataset-*.csv on patient id; the other three need the
# equivalent from their own NSRR metadata. It is a loader change plus an
# `age` entry in labels.modes, not a flag.
# ==========================================================================
