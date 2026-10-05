# Datasets

A review table over every cohort manifest. Each row links to the one file
that holds every fact about that cohort, and `cohort.yaml` is the source of
truth — if this table and a manifest disagree, the manifest is right. The
rows and the known issues below are generated from the manifests (`source`
is `acquisition.kind`, `+N unused` counts `labels.also_implemented`, `frozen`
means the folds come from a manifest file).

50 datasets have a manifest; 43 of them are in the paper's evaluation
(`neuroatlas list datasets`). The 138 other datasets the registry can load
(`neuroatlas list datasets --all`, 188 in all) are MOABB datasets generated
from `MOABB_DATASETS` in `dataio/moabb_loader.py`; none is in the paper.

## bci (22)

| dataset | paper | source | montage | window s | label modes | one label per | folds | grouping |
|---|---|---|---|---|---|---|---|---|
| [`arithmetic_task`](arithmetic_task/cohort.yaml) | yes | internal | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bi2013a`](bi2013a/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bi2014a`](bi2014a/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bnci2014_001`](bnci2014_001/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bnci2014_004`](bnci2014_004/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bnci2014_008`](bnci2014_008/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bnci2015_001`](bnci2015_001/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`cho2017`](cho2017/cohort.yaml) | — | moabb | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`dreamer_arousal`](dreamer_arousal/cohort.yaml) | yes | manual | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`dreamer_valence`](dreamer_valence/cohort.yaml) | yes | manual | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`dreyer2023`](dreyer2023/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`eegmat`](eegmat/cohort.yaml) | yes | physionet | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`epflp300`](epflp300/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`erpcore2021_n170`](erpcore2021_n170/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`hinss2021`](hinss2021/cohort.yaml) | — | moabb | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`kim2025betarange`](kim2025betarange/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`lee2019_mi`](lee2019_mi/cohort.yaml) | — | moabb | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`liu2024`](liu2024/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`nakanishi2015`](nakanishi2015/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`physionet_mi`](physionet_mi/cohort.yaml) | — | moabb | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`shin2017a`](shin2017a/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`weibo2014`](weibo2014/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |

## epilepsy (11)

| dataset | paper | source | montage | window s | label modes | one label per | folds | grouping |
|---|---|---|---|---|---|---|---|---|
| [`aub_med`](aub_med/cohort.yaml) | — | mendeley | bipolar | 10.0 | `binary` | window | 6 <br><sub>derived</sub> | patient |
| [`bonn`](bonn/cohort.yaml) | yes | url | unipolar | — | `binary_s_vs_rest` <br><sub>+8 unused</sub> | recording | 5 <br><sub>derived</sub> | clip |
| [`chbmit`](chbmit/cohort.yaml) | yes | zenodo | bipolar | 10.0 | `binary` <br><sub>+1 unused</sub> | window | 5 <br><sub>frozen</sub> | patient |
| [`epilepsiae`](epilepsiae/cohort.yaml) | yes | manual | bipolar | 10.0 | `binary` <br><sub>+2 unused</sub> | window | 5 <br><sub>frozen</sub> | patient |
| [`helsinki_neonatal`](helsinki_neonatal/cohort.yaml) | yes | zenodo | unipolar | 10.0 | `binary` | window | 5 <br><sub>derived</sub> | patient |
| [`nmt`](nmt/cohort.yaml) | yes | manual | unipolar | 10.0 | `binary` | recording | 5 <br><sub>derived</sub> | patient |
| [`siena`](siena/cohort.yaml) | yes | zenodo | bipolar | 10.0 | `binary` | window | 5 <br><sub>derived</sub> | patient |
| [`sz1`](sz1/cohort.yaml) | yes | internal | unipolar | 10.0 | `binary` | window | 5 <br><sub>frozen</sub> | patient |
| [`sz2`](sz2/cohort.yaml) | yes | internal | unipolar | 10.0 | `binary` | window | 5 <br><sub>frozen</sub> | patient |
| [`tuab`](tuab/cohort.yaml) | yes | tuh | unipolar | 10.0 | `binary` | recording | 5 <br><sub>derived</sub> | patient |
| [`tusz`](tusz/cohort.yaml) | yes | tuh | bipolar | 10.0 | `binary` <br><sub>+6 unused</sub> | window | 5 <br><sub>frozen</sub> | patient |

## sleep (17)

| dataset | paper | source | montage | window s | label modes | one label per | folds | grouping |
|---|---|---|---|---|---|---|---|---|
| [`cfs`](cfs/cohort.yaml) | yes | nsrr | — | — | `sleep_stage`, `age` | sleep_stage: window, age: recording | 5 <br><sub>derived</sub> | patient |
| [`dcsm`](dcsm/cohort.yaml) | yes | manual | — | — | `sleep_stage` | window | 5 <br><sub>frozen</sub> | patient |
| [`dod`](dod/cohort.yaml) | yes | zenodo | — | — | `sleep_stage`, `osa_group` | sleep_stage: window, osa_group: recording | 5 <br><sub>frozen</sub> | patient |
| [`hmc`](hmc/cohort.yaml) | yes | physionet | — | — | `sleep_stage` | window | 5 <br><sub>frozen</sub> | patient |
| [`hpap_lab_full`](hpap_lab_full/cohort.yaml) | yes | nsrr | — | — | `sleep_stage` | window | 5 <br><sub>frozen</sub> | patient |
| [`isruc`](isruc/cohort.yaml) | yes | manual | — | — | `sleep_stage`, `diagnosis`, `has_diagnosis`, `age` <br><sub>+1 unused</sub> | sleep_stage: window, diagnosis: recording, has_diagnosis: recording, age: recording | 5 <br><sub>frozen</sub> | patient |
| [`mass`](mass/cohort.yaml) | yes | manual | — | — | `sleep_stage` | window | 5 <br><sub>frozen</sub> | patient |
| [`mesa`](mesa/cohort.yaml) | yes | nsrr | — | — | `sleep_stage` | window | 5 <br><sub>frozen</sub> | patient |
| [`mros`](mros/cohort.yaml) | yes | nsrr | — | — | `sleep_stage`, `age` | sleep_stage: window, age: recording | 5 <br><sub>derived</sub> | patient |
| [`parkinson`](parkinson/cohort.yaml) | — | internal | — | — | `group` | recording | 5 <br><sub>derived</sub> | patient |
| [`physionet2026`](physionet2026/cohort.yaml) | yes | manual | — | — | `sleep_stage`, `ci_label`, `age` <br><sub>+1 unused</sub> | sleep_stage: window, ci_label: recording, age: recording | 5 <br><sub>frozen</sub> | patient |
| [`shhs`](shhs/cohort.yaml) | yes | nsrr | — | — | `sleep_stage` | window | — | patient |
| [`sleep_edf`](sleep_edf/cohort.yaml) | — | physionet | — | — | `sleep_stage` | window | — | patient |
| [`sleep_edf_expanded`](sleep_edf_expanded/cohort.yaml) | yes | physionet | — | — | `sleep_stage`, `condition`, `age` <br><sub>+1 unused</sub> | sleep_stage: window, condition: recording, age: recording | 5 <br><sub>frozen</sub> | patient |
| [`stages`](stages/cohort.yaml) | yes | nsrr | — | — | `sleep_stage` | window | 5 <br><sub>frozen</sub> | patient |
| [`ucddb`](ucddb/cohort.yaml) | yes | physionet | — | — | `sleep_stage` <br><sub>+1 unused</sub> | window | 5 <br><sub>frozen</sub> | patient |
| [`wsc`](wsc/cohort.yaml) | yes | nsrr | — | — | `sleep_stage`, `age` <br><sub>+1 unused</sub> | sleep_stage: window, age: recording | 5 <br><sub>frozen</sub> | patient |

## Needs a decision

### Montage conflicts

Sources disagree; the manifest keeps the DatasetSpec value so
behaviour is unchanged.

- **siena** — resolved to `bipolar`; DatasetSpec.config_defaults=bipolar, embed_dataset._DEFAULT_MONTAGE=unipolar, description=unipolar
- **tusz** — resolved to `bipolar`; DatasetSpec.config_defaults=bipolar, embed_dataset._DEFAULT_MONTAGE=bipolar, description=unipolar

### Licence still TBD

Carried over from an earlier review; the manifests have no licence field
to check it against.

- **epilepsiae**
- **mass**

### Known issues

- **bi2013a** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **bi2014a** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **bnci2014_001** — BCI preprocessing differs from the paper's. The paper's BCI embeddings were extracted from per-model pickles (trackA: each model's own passband and rate, e.g. 0.1-75 Hz / 200 Hz / 50 Hz notch for the LaBraM-format pickle; trackD: 4-40 Hz and 1-4 s). `run` epochs MOABB at 128 Hz with the paradigm's 4-40 Hz band and no notch, so off and on confound control differ here only in the start of the window, not in the band. The 4 s trial window is the paper's.
- **bnci2014_001** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **bnci2014_004** — BCI preprocessing differs from the paper's. The paper's BCI embeddings were extracted from per-model pickles (trackA: each model's own passband and rate, e.g. 0.1-75 Hz / 200 Hz / 50 Hz notch for the LaBraM-format pickle; trackD: 4-40 Hz and 1-4 s). `run` epochs MOABB at 128 Hz with the paradigm's 4-40 Hz band and no notch, so off and on confound control differ here only in the start of the window, not in the band. The 4 s trial window is the paper's.
- **bnci2014_004** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **bnci2014_008** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **bnci2015_001** — BCI preprocessing differs from the paper's. The paper's BCI embeddings were extracted from per-model pickles (trackA: each model's own passband and rate, e.g. 0.1-75 Hz / 200 Hz / 50 Hz notch for the LaBraM-format pickle; trackD: 4-40 Hz and 1-4 s). `run` epochs MOABB at 128 Hz with the paradigm's 4-40 Hz band and no notch, so off and on confound control differ here only in the start of the window, not in the band. The 4 s trial window is the paper's.
- **bnci2015_001** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **bonn** — Folds are not subject-independent: the corpus ships no subject identifiers, so the split is over clips. Numbers are not comparable with the subject-disjoint cohorts — see splits.derivation.
- **bonn** — Needs an author decision -- the paper split Bonn finer than this. App. D.1.3 and its epilepsy probe both used stratified 5-fold CV over the 1,000 10-s segments (two per 23.6-s clip), so a clip's two segments could fall in train and test (770 of the 1,000 test segments had the other half of their clip in train or val); these folds keep a clip's segments together. The paper's exact segment assignment IS recoverable (the probe's seeded StratifiedKFold over the clip-major segment order): --set split_mode=paper_segments reproduces it, frozen in folds_paper_segments.json and checked by re-probing the paper's own CBraMod embedding to its saved predictions (<= 2.6e-10). Evidence: fixwork/R1_siena_bonn/REPORT.md.
- **cfs** — Needs an author decision -- two pre-restructure protocols disagree on CFS sleep-staging folds. This reader's age-binned folds are the brain-age protocol (05fd2a8 datasets/cfs: one cache for both tasks); the c-tuned staging probes (artifacts/probes_perC_sleep3/cfs_bp_2ch) used 3c6721c configs/folds/cfs_bp_2ch.json, a subject round-robin (seed 42). The two share about a fifth of each test fold.
- **chbmit** — No channel map: src/neuroatlas/configs/channel_maps/chbmit.yaml does not exist, unlike siena/tuab/tusz. Vocabulary-strict wrappers therefore receive the raw bipolar labels.
- **chbmit** — Window size disagrees between embed_chbmit_siena.py (30 s) and embed_chbmit_siena_sharded.py (2 s / 1 s) — audit OQ-11.4.
- **dod** — The multi-scorer evaluation needs the dreem-learning-evaluation scorers, which dod_fm_probe.sh pointed at /tmp/dreem-learning- evaluation/scorers -- a clone someone made by hand, with no default that survives a reboot. Pass --set expert_scorers_root=<path>.
- **dreyer2023** — BCI preprocessing differs from the paper's. The paper's BCI embeddings were extracted from per-model pickles (trackA: each model's own passband and rate, e.g. 0.1-75 Hz / 200 Hz / 50 Hz notch for the LaBraM-format pickle; trackD: 4-40 Hz and 1-4 s). `run` epochs MOABB at 128 Hz with the paradigm's 4-40 Hz band and no notch, so off and on confound control differ here only in the start of the window, not in the band. The 4 s trial window is the paper's.
- **dreyer2023** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **epflp300** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **epilepsiae** — default_task is `linear_probe`, but this is a continuously labelled cohort (one label per 10 s window) like chbmit, siena, sz1/sz2, tusz, aub_med and helsinki_neonatal — all of which use `seizure_detection`. The three cohorts that legitimately use linear_probe (bonn, nmt, tuab) are all recording-level. So Epilepsiae is scored WITHOUT the event-level metrics the other continuous cohorts get. run/configs/benchmark_epilepsiae_test.json agrees with the spec, so this is what was actually run — changing it is a scientific decision, not a bug fix. Needs a maintainer.
- **erpcore2021_n170** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **hmc** — Never loaded. The 4 EEG channels and their 10-20 names come from adapters/hmc.py, the data_root from configs/dataset_paths_sofia.yaml; neither has been checked against the corpus.
- **hmc** — The fold manifest's ids (SN003, SN009, ...) have not been matched against the subject ids the loader discovers. A mismatch raises rather than falling back to a random split (physioex_base _resolve_splits), so it will be loud, but it has not been seen to pass.
- **hpap_lab_full** — Never loaded. HomePAP's corpus is not on the machine this was written on; data_root comes from configs/dataset_paths_sofia.yaml and the channel handling from adapters/hpap_lab_full.py, neither checked.
- **hpap_lab_full** — The manifest ids are recording ids (homepap-lab-full-1600003), so `grouping: patient` holds only if HomePAP has one lab-full recording per subject. Unverified -- if it does not, the folds are recording-level and could leak a subject across splits.
- **hpap_lab_full** — App. B.3 uses this cohort for brain age, but the loader carries no demographics, so there is no `age` label mode here and brain age cannot run on it. Adding it means joining the NSRR homepap-baseline-dataset-*.csv on subject id -- the same work SHHS needs.
- **kim2025betarange** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **liu2024** — BCI preprocessing differs from the paper's. The paper's BCI embeddings were extracted from per-model pickles (trackA: each model's own passband and rate, e.g. 0.1-75 Hz / 200 Hz / 50 Hz notch for the LaBraM-format pickle; trackD: 4-40 Hz and 1-4 s). `run` epochs MOABB at 128 Hz with the paradigm's 4-40 Hz band and no notch, so off and on confound control differ here only in the start of the window, not in the band. The 4 s trial window is the paper's.
- **liu2024** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **mesa** — Never loaded. MESA's corpus is not on the machine this was written on.
- **mesa** — 2056 recordings makes this the benchmark's largest sleep cohort; the default batch_size of 8 and num_workers of 0 are the physioex base defaults, not a considered choice for a corpus this size.
- **mesa** — App. B.3 uses this cohort for brain age, but the loader carries no demographics, so there is no `age` label mode here and brain age cannot run on it. Adding it means joining the NSRR mesa-sleep-dataset-*.csv on subject id -- the same work SHHS needs.
- **mros** — Needs an author decision -- two pre-restructure protocols disagree on MrOS sleep-staging folds. This reader's age-binned folds are the brain-age protocol (05fd2a8 datasets/mros); the c-tuned staging probes (artifacts/probes_perC_mros) used 3c6721c configs/folds/mros_bp_2ch.json, a subject round-robin (seed 42). The two share about a fifth of each test fold.
- **nakanishi2015** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **nmt** — Needs an author decision -- the paper's text and its NMT numbers disagree. App. D.1.3 gives the official train/eval split (2,232 / 185), which is what split_mode=official (the default) does. The published numbers came from the paper's epilepsy probe, which ran 5-fold CV over 2,414 recordings (3c6721c configs/folds/nmt.json).
- **shhs** — Needs an author decision -- which SHHS folds the paper's foundation-model staging numbers used is not established. This reader runs one fixed split (SleepTransformer's data_split_eval.mat, as before the restructure); the pre-restructure combined probes (artifacts/probes_combined_perC/shhs_combined) used five folds from configs/folds/shhs_combined.json over 8,444 recordings, which this reader does not read (it indexes the CoRe-Sleep patient .mat lists).
- **shin2017a** — BCI preprocessing differs from the paper's. The paper's BCI embeddings were extracted from per-model pickles (trackA: each model's own passband and rate, e.g. 0.1-75 Hz / 200 Hz / 50 Hz notch for the LaBraM-format pickle; trackD: 4-40 Hz and 1-4 s). `run` epochs MOABB at 128 Hz with the paradigm's 4-40 Hz band and no notch, so off and on confound control differ here only in the start of the window, not in the band. The 4 s trial window is the paper's.
- **shin2017a** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **siena** — Needs an author decision -- the published Siena numbers were not drawn on these folds. The paper's epilepsy probe saw subject id 'raw' for every window (a path-parsing bug in its embeddings), fell back to grouping by recording, and split Siena's 40 recordings, not its 14 patients: 30 of the 33 (fold, test patient) pairs also have recordings in train or val (fold 0: all 6; 5 of them in train), and 40,419 of the 49,285 test windows (283 of 312 seizure windows) belong to such patients. These folds are patient-level, as the paper states (App. B.1, C.1), so they do not reproduce its Siena numbers; --set split_mode=paper_recordings does (frozen in folds_paper_recordings.json, checked against all 26 paper prediction files). Evidence: fixwork/R1_siena_bonn/REPORT.md.
- **siena** — The paper-era Siena embeddings hold 40 of the release's 41 recordings: sub-14 run-03 (14,642 s, one 83 s focal seizure) is missing from them, and their labels mark no seizure window in sub-14 run-02 although its events list one (17,540-17,581 s). The published Siena numbers are therefore over 45 of the release's 47 seizures. split_mode=paper_recordings leaves run-03 out of every fold.
- **stages** — Never loaded. STAGES' corpus is not on the machine this was written on.
- **stages** — The loader defaults to recording='first', so a subject's repeat `_1` recording is excluded. The fold manifests are keyed on base subject ids, which matches, but this has not been run to confirm.
- **stages** — App. B.3 uses STAGES for brain age, but the loader surfaces no demographics, so there is no `age` label mode and brain age cannot run on it. Demographics exist in stages-harmonized-dataset-0.3.0.csv, which the loader reads for other purposes but does not publish.
- **tuab** — Needs an author decision -- the paper's text and its TUAB numbers disagree. App. D.1.3 says the official train/eval split with the train subjects stratified into 5 folds, which is what split_mode=official (the default) does. The published numbers (e.g. REVE AUROC 0.898) came from the paper's epilepsy probe, which ran 5-fold CV over all 2,993 sessions grouped by session file (3c6721c configs/folds/tuab.json), so a patient's sessions could sit in train and test.
- **tusz** — Montage conflict: the description says unipolar, the spec and embed_dataset._DEFAULT_MONTAGE both say bipolar. Kept bipolar to preserve behaviour — audit OQ-11.2.
- **weibo2014** — BCI preprocessing differs from the paper's. The paper's BCI embeddings were extracted from per-model pickles (trackA: each model's own passband and rate, e.g. 0.1-75 Hz / 200 Hz / 50 Hz notch for the LaBraM-format pickle; trackD: 4-40 Hz and 1-4 s). `run` epochs MOABB at 128 Hz with the paradigm's 4-40 Hz band and no notch, so off and on confound control differ here only in the start of the window, not in the band. The 4 s trial window is the paper's.
- **weibo2014** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
