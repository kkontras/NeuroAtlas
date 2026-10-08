#!/usr/bin/env bash
#
# Every experiment NeuroAtlas reports, one command per line.
#
# The record of what the paper ran. Each line is complete and copy-pasteable,
# and the settings that differ between experiments are written out rather
# than left to a default. `neuroatlas show <benchmark>` prints the same lines,
# spelled `neuroatlas <verb>`; `neuroatlas run` and `neuroatlas submit` run
# them.
#
# Constant across every dataset, so left at the default:
#   preprocessing   0.5 Hz highpass + powerline notch, then resample to each
#                   model's training rate (Sec. 3), at each dataset's
#                   powerline frequency.
#   backbone        frozen. No fine-tuning anywhere in the benchmark.
#   solver          LBFGS.
#   precision       fp32.
#   folds           patient-level 5-fold, except BCI, which is LOSO.
#
# What varies, and is therefore explicit below (Appendix C):
#   window/stride   10 s non-overlapping for epilepsy (C.1); 30 s epochs for
#                   sleep and brain age (C.2, C.3); per-paradigm trial windows
#                   for BCI (C.4), set per dataset.
#   probe           logistic regression, except brain age, which is ridge.
#   C / alpha       epilepsy chooses C on validation AUPRC (C.1); sleep
#                   staging chooses it on validation Cohen's kappa (C.2);
#                   BCI chooses it on the validation subject's Cohen's
#                   kappa (C.4); sleep events and diagnosis use C = 1;
#                   brain age chooses alpha by nested cross-validation.
#   class weight    balanced for epilepsy and sleep events; unweighted for
#                   sleep staging, diagnosis and BCI.
#   aggregation     none for epoch-level tasks; one mean embedding per
#                   subject for diagnosis and brain age (per recording for
#                   ISRUC, SHHS and WSC brain age).
#   selection       AUPRC for seizure detection, Cohen's kappa for sleep
#                   staging, MAE for brain age.
#
# Models: each line starts from `--models all`, every checkpoint, and removes
# by name the families its benchmark does not evaluate (`-name`):
#   epilepsy        the sleep-staging sequence models (sleep_transformer,
#                   sleepyco, core_sleep), which take sequences of 30 s
#                   epochs, and the motor-imagery EEGNet checkpoints
#                   (eegnetv4);
#   sleep, brain age
#                   eegnetv4 and the seizure models (deepsoz_hem,
#                   seizure_transformer);
#   BCI             the sleep-staging sequence models and the seizure models.
# `neuroatlas list benchmarks -v` lists the same exclusions, with the reason.
# --models eeg_fm | ts_fm | supervised selects one of the paper's groups.
#
set -euo pipefail


# ==========================================================================
# 1. FETCH -- where to get every dataset
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
# the bci_cognitive datasets' raw data (EEGMat from PhysioNet, ArithmeticTask
# from OSF; DREAMER on request from Zenodo), built in section 2
python -m neuroatlas.entrypoints.fetch --dataset arithmetic_task
python -m neuroatlas.entrypoints.fetch --dataset dreamer_arousal
python -m neuroatlas.entrypoints.fetch --dataset dreamer_valence
python -m neuroatlas.entrypoints.fetch --dataset eegmat

# ==========================================================================
# 2. PREPARE -- the files the four bci_cognitive datasets are read from
# DREAMER (valence, arousal), EEGMat and ArithmeticTask are read from two
# files each (no filtering, confound filtering), built from their raw data.
# Every other dataset is read as obtained; the MOABB datasets are epoched by their reader as
# `embed` loads them. Five epilepsy datasets (bonn, epilepsiae, sz1, tuab,
# tusz) have an optional faster-reading copy (`neuroatlas data prepare
# <dataset>`), not listed here.
# ==========================================================================
python -m neuroatlas.entrypoints.prepare --dataset arithmetic_task
python -m neuroatlas.entrypoints.prepare --dataset dreamer_arousal
python -m neuroatlas.entrypoints.prepare --dataset dreamer_valence
python -m neuroatlas.entrypoints.prepare --dataset eegmat

# ==========================================================================
# 3. EMBED -- frozen backbones, one pass per (dataset, model)
# ==========================================================================

# Epilepsy: 10 s windows, no overlap (C.1). --expected-epoch-seconds tells
# each backbone the window it is handed, so one pretrained on 30 s epochs
# (BIOT) accepts a 10 s one.
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset bonn --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset chbmit --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset epilepsiae --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset helsinki_neonatal --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset nmt --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset siena --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset sz1 --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset sz2 --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset tuab --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset tusz --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10

# Sleep and brain age: 30 s epochs (C.2, C.3). Readers built on fixed 30 s
# scoring epochs take no window arguments; only the others are told.
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset cfs --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset dcsm
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset dod
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset hmc --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset hpap_lab_full --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset isruc
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset mass
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset mesa --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset mros --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset physionet2026
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset shhs
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset sleep_edf_expanded
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset stages --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset ucddb
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset wsc
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset cfs --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset hpap_lab_full --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset isruc
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset mesa --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset mros --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset physionet2026
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset sleep_edf_expanded
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset stages --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset wsc

# BCI (C.4). Each dataset has its own trial window, and each model reads
# the trials in its own format: 0.1-75 Hz at 200 Hz with a 50 Hz notch;
# EEGPT 0.5-70 Hz at 256 Hz, ST-EEGFormer and SleepFM 0.1-64 Hz at 128 Hz,
# both with a notch at the dataset's line frequency. Two settings are
# crossed over the datasets.
#
#   confound_control  Confound filtering (App. D.6), on in each BCI
#                     benchmark's default; the model's rate and notch stay.
#                     Motor imagery: 4-40 Hz and the trial from 1 s after
#                     the cue, so the score cannot come from the cue's
#                     evoked response or the eye movement towards it. ERP:
#                     0.5-40 Hz. SSVEP: the model's high-pass and no
#                     low-pass, which keeps the stimulus harmonics. Off is
#                     "no filtering": the model's own band and the whole
#                     trial.
#   pooling           `mean` averages a trial's patch tokens into one
#                     vector. `per_patch` concatenates them.
#
# Each combination has its own cache, so none overwrites another. The
# paper calls confound_control on "confound filtering" and off "no
# filtering", pooling mean "mean-pool" and per_patch "token flattening".
# `neuroatlas show bci_motor_imagery` names the figure each reproduces.

#   --- confound_control on, pooling mean ------------------------
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2013a --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2014a --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_001 --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_004 --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_008 --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2015_001 --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreyer2023 --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset epflp300 --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset erpcore2021_n170 --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset kim2025betarange --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset liu2024 --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset nakanishi2015 --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset shin2017a --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset weibo2014 --set confound_control=true --pooling mean

#   --- confound_control on, pooling per_patch ------------------------
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2013a --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2014a --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_001 --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_004 --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_008 --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2015_001 --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreyer2023 --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset epflp300 --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset erpcore2021_n170 --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset kim2025betarange --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset liu2024 --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset nakanishi2015 --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset shin2017a --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset weibo2014 --set confound_control=true --pooling per_patch

#   --- confound_control off, pooling mean ------------------------
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2013a --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2014a --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_001 --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_004 --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_008 --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2015_001 --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreyer2023 --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset epflp300 --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset erpcore2021_n170 --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset kim2025betarange --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset liu2024 --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset nakanishi2015 --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset shin2017a --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset weibo2014 --set confound_control=false --pooling mean

#   --- confound_control off, pooling per_patch ------------------------
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2013a --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2014a --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_001 --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_004 --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_008 --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2015_001 --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreyer2023 --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset epflp300 --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset erpcore2021_n170 --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset kim2025betarange --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset liu2024 --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset nakanishi2015 --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset shin2017a --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset weibo2014 --set confound_control=false --pooling per_patch

# ==========================================================================
# 4. PROBE -- reads the embeddings above; refuses if they are absent
# A probe finds the embeddings by a key computed from the dataset config,
# so each probe repeats its embed line's --set flags, spelled the same
# (window_s=10, not 10.0: the key hashes the values as written).
# --expected-epoch-seconds belongs to the checkpoint and is not part of
# the key, so it stays on the embed line.
# ==========================================================================

# --- Epilepsy: seizure detection, AUPRC-selected C (C.1) ------------------
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset bonn --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset chbmit --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset epilepsiae --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset helsinki_neonatal --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset nmt --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset siena --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset sz1 --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset sz2 --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset tuab --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-eegnetv4 --dataset tusz --task seizure_detection --set window_s=10 --set stride_s=10 --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc

# --- Sleep axis 1: epoch-wise staging, unweighted loss (C.2) --------------
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset cfs --task sleep_staging --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset dcsm --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset dod --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset hmc --task sleep_staging --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset hpap_lab_full --task sleep_staging --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset isruc --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset mass --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset mesa --task sleep_staging --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset mros --task sleep_staging --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset physionet2026 --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset shhs --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset sleep_edf_expanded --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset stages --task sleep_staging --set window_s=30 --set stride_s=30
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset ucddb --task sleep_staging
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset wsc --task sleep_staging

# --- Sleep axis 2: hypnogram features (C.2, Table 2) ------------------
#     Derived from the staging predictions above -- TST, REM latency,
#     WASO, bout durations, transition rates. No embedding pass and no
#     model: `reconstruct` re-predicts with each saved probe to get
#     per-recording epoch sequences, then `features` turns those into
#     the 34 features. Runs after the staging probes above, which is
#     why it is last in the file.

# --- Sleep axis 3: event detection, balanced loss (C.2) --------------------
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset mass --task mass_arousal --probe-type linear --class-weight balanced --tune-c 1.0
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset physionet2026 --task arousal --probe-type linear --class-weight balanced --tune-c 1.0
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset physionet2026 --task limb --probe-type linear --class-weight balanced --tune-c 1.0
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset physionet2026 --task respiratory --probe-type linear --class-weight balanced --tune-c 1.0
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset ucddb --task respiratory --probe-type linear --class-weight balanced --tune-c 1.0

# --- Sleep axis 4: subject-level diagnosis, mean-pooled, unweighted (C.2) ---
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset dod --task osa --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset isruc --task pathology --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset physionet2026 --task cognitive --aggregation mean

# --- Brain age: ridge on mean-pooled epochs, nested-CV alpha (C.3) --------
#     Runs on the sleep datasets that carry participant ages. The
#     brain_age task fits the ridge and selects alpha itself, so no
#     --probe-type or --tune-c here.
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset cfs --task brain_age --set window_s=30 --set stride_s=30 --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset hpap_lab_full --task brain_age --set window_s=30 --set stride_s=30 --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset isruc --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset mesa --task brain_age --set window_s=30 --set stride_s=30 --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset mros --task brain_age --set window_s=30 --set stride_s=30 --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset physionet2026 --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset shhs --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset sleep_edf_expanded --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset stages --task brain_age --set window_s=30 --set stride_s=30 --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset wsc --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset cfs --task brain_age --set window_s=30 --set stride_s=30 --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset hpap_lab_full --task brain_age --set window_s=30 --set stride_s=30 --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset isruc --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset mesa --task brain_age --set window_s=30 --set stride_s=30 --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset mros --task brain_age --set window_s=30 --set stride_s=30 --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset physionet2026 --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset shhs --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset sleep_edf_expanded --task brain_age --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset stages --task brain_age --set window_s=30 --set stride_s=30 --set label_mode=age --aggregation mean
python -m neuroatlas.entrypoints.probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset wsc --task brain_age --set label_mode=age --aggregation mean

# --- BCI: LOSO, logistic regression, C on validation kappa (C.4) ---------
#     n_folds=loso is one fold per subject, whatever the number of
#     subjects (8 to 100 across the datasets): fold k tests the k-th subject
#     in ascending order, and one other subject, drawn with seed 42, picks
#     C from 0.001 to 100 on its Cohen's kappa.

#   --- confound_control on, pooling mean ------------------------
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2013a --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2014a --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_001 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_004 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_008 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2015_001 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreyer2023 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset epflp300 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset erpcore2021_n170 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset kim2025betarange --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset liu2024 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset nakanishi2015 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset shin2017a --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset weibo2014 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000

#   --- confound_control on, pooling per_patch ------------------------
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2013a --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2014a --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_001 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_004 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_008 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2015_001 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreyer2023 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset epflp300 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset erpcore2021_n170 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset kim2025betarange --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset liu2024 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset nakanishi2015 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset shin2017a --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset weibo2014 --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000

#   --- confound_control off, pooling mean ------------------------
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2013a --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2014a --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_001 --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_004 --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_008 --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2015_001 --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreyer2023 --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset epflp300 --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset erpcore2021_n170 --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset kim2025betarange --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset liu2024 --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset nakanishi2015 --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset shin2017a --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset weibo2014 --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000

#   --- confound_control off, pooling per_patch ------------------------
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2013a --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bi2014a --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_001 --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_004 --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2014_008 --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset bnci2015_001 --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreyer2023 --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset epflp300 --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset erpcore2021_n170 --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset kim2025betarange --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset liu2024 --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset nakanishi2015 --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset shin2017a --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset weibo2014 --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000

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
# BCI cognitive state (C.4): DREAMER (valence and arousal, two entries of
# one dataset), EEGMat and ArithmeticTask, read from the files section 2
# builds, in each model's format. Confound filtering (confound_control=true)
# reads each cohort's 4-40 Hz file (<name>_preprocessed_trackD_<format>.pkl),
# no filtering the file in the format's own band.

#   --- confound_control on, pooling mean ------------------------
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset arithmetic_task --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_arousal --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_valence --set confound_control=true --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset eegmat --set confound_control=true --pooling mean

#   --- confound_control on, pooling per_patch ------------------------
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset arithmetic_task --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_arousal --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_valence --set confound_control=true --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset eegmat --set confound_control=true --pooling per_patch

#   --- confound_control off, pooling mean ------------------------
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset arithmetic_task --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_arousal --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_valence --set confound_control=false --pooling mean
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset eegmat --set confound_control=false --pooling mean

#   --- confound_control off, pooling per_patch ------------------------
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset arithmetic_task --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_arousal --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_valence --set confound_control=false --pooling per_patch
python -m neuroatlas.entrypoints.embed --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset eegmat --set confound_control=false --pooling per_patch

#   --- confound_control on, pooling mean ------------------------
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset arithmetic_task --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_arousal --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_valence --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset eegmat --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000

#   --- confound_control on, pooling per_patch ------------------------
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset arithmetic_task --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_arousal --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_valence --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset eegmat --set confound_control=true --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000

#   --- confound_control off, pooling mean ------------------------
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset arithmetic_task --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_arousal --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_valence --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset eegmat --set confound_control=false --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000

#   --- confound_control off, pooling per_patch ------------------------
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset arithmetic_task --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_arousal --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset dreamer_valence --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000
python -m neuroatlas.entrypoints.probe --models all,-sleep_transformer,-sleepyco,-core_sleep,-deepsoz_hem,-seizure_transformer --dataset eegmat --set confound_control=false --pooling per_patch --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000

# Brain age runs on ten datasets: CFS, HomePAP, ISRUC, MESA, MrOS,
# PhysioNet 2026, SHHS, Sleep-EDF Expanded (its Sleep Cassette subjects),
# STAGES and WSC. HomePAP, MESA, SHHS and STAGES take each participant's age
# from the study's NSRR dataset table, which `neuroatlas data download`
# fetches with the recordings.
# ==========================================================================
