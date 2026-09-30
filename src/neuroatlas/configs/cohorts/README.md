# Datasets

A review table over every cohort manifest. Each row links to the one file
that holds every fact about that cohort, and `cohort.yaml` is the source of
truth — if this table and a manifest disagree, the manifest is right.

46 datasets have a manifest. The ~150 MOABB datasets the
registry can additionally load are generated from `MOABB_DATASETS` in
`dataio/moabb_loader.py` and are not in the paper.

## bci (18)

| dataset | paper | source | montage | window s | label modes | one label per | folds | grouping |
|---|---|---|---|---|---|---|---|---|
| [`bi2013a`](bi2013a/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bi2014a`](bi2014a/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bnci2014_001`](bnci2014_001/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bnci2014_004`](bnci2014_004/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bnci2014_008`](bnci2014_008/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`bnci2015_001`](bnci2015_001/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`cho2017`](cho2017/cohort.yaml) | — | moabb | — | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
| [`dreyer2023`](dreyer2023/cohort.yaml) | yes | moabb | as-published | — | `trial_class` | trial | 5 <br><sub>derived</sub> | patient |
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
| [`bonn`](bonn/cohort.yaml) | yes | manual | unipolar | — | `binary_s_vs_rest` <br><sub>+8 unused</sub> | recording | 5 <br><sub>derived</sub> | clip |
| [`chbmit`](chbmit/cohort.yaml) | yes | zenodo | bipolar | 10.0 | `binary` <br><sub>+1 unused</sub> | window | 5 <br><sub>frozen</sub> | patient |
| [`epilepsiae`](epilepsiae/cohort.yaml) | yes | manual | bipolar | 10.0 | `binary` <br><sub>+2 unused</sub> | window | 5 <br><sub>derived</sub> | patient |
| [`helsinki_neonatal`](helsinki_neonatal/cohort.yaml) | yes | zenodo | unipolar | 10.0 | `binary` | window | 5 <br><sub>derived</sub> | patient |
| [`nmt`](nmt/cohort.yaml) | yes | manual | unipolar | 10.0 | `binary` | recording | 5 <br><sub>derived</sub> | patient |
| [`siena`](siena/cohort.yaml) | yes | zenodo | bipolar | 10.0 | `binary` | window | 5 <br><sub>derived</sub> | patient |
| [`sz1`](sz1/cohort.yaml) | yes | internal | unipolar | 10.0 | `binary` | window | 5 <br><sub>frozen</sub> | patient |
| [`sz2`](sz2/cohort.yaml) | yes | internal | unipolar | 10.0 | `binary` | window | 5 <br><sub>derived</sub> | patient |
| [`tuab`](tuab/cohort.yaml) | yes | tuh | unipolar | 10.0 | `binary` | recording | 5 <br><sub>derived</sub> | patient |
| [`tusz`](tusz/cohort.yaml) | yes | tuh | bipolar | 10.0 | `binary` <br><sub>+6 unused</sub> | window | 5 <br><sub>derived</sub> | patient |

## sleep (17)

| dataset | paper | source | montage | window s | label modes | one label per | folds | grouping |
|---|---|---|---|---|---|---|---|---|
| [`cfs`](cfs/cohort.yaml) | yes | nsrr | — | — | `sleep_stage`, `age` | sleep_stage: window, age: recording | 5 <br><sub>derived</sub> | patient |
| [`dcsm`](dcsm/cohort.yaml) | yes | manual | — | — | `sleep_stage` | window | 5 <br><sub>frozen</sub> | patient |
| [`dod`](dod/cohort.yaml) | yes | manual | — | — | `sleep_stage`, `osa_group` | sleep_stage: window, osa_group: recording | 5 <br><sub>frozen</sub> | patient |
| [`hmc`](hmc/cohort.yaml) | yes | physionet | — | — | `sleep_stage` | window | 5 <br><sub>frozen</sub> | patient |
| [`hpap_lab_full`](hpap_lab_full/cohort.yaml) | yes | nsrr | — | — | `sleep_stage` | window | 5 <br><sub>frozen</sub> | patient |
| [`isruc`](isruc/cohort.yaml) | yes | manual | — | — | `sleep_stage`, `diagnosis`, `has_diagnosis`, `age` <br><sub>+1 unused</sub> | sleep_stage: window, diagnosis: recording, has_diagnosis: recording, age: recording | 5 <br><sub>frozen</sub> | patient |
| [`mass`](mass/cohort.yaml) | yes | manual | — | — | `sleep_stage` | window | 5 <br><sub>frozen</sub> | patient |
| [`mesa`](mesa/cohort.yaml) | yes | nsrr | — | — | `sleep_stage` | window | 5 <br><sub>frozen</sub> | patient |
| [`mros`](mros/cohort.yaml) | yes | nsrr | — | — | `sleep_stage`, `age` | sleep_stage: window, age: recording | 5 <br><sub>derived</sub> | patient |
| [`parkinson`](parkinson/cohort.yaml) | — | internal | — | — | `group` | recording | 5 <br><sub>derived</sub> | patient |
| [`physionet2026`](physionet2026/cohort.yaml) | yes | physionet | — | — | `sleep_stage`, `ci_label`, `age` <br><sub>+1 unused</sub> | sleep_stage: window, ci_label: recording, age: recording | 5 <br><sub>frozen</sub> | patient |
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

- **epilepsiae**
- **mass**

### Known issues

- **bi2013a** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **bi2014a** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **bnci2014_001** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **bnci2014_004** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **bnci2014_008** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **bnci2015_001** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **bonn** — Folds are not subject-independent: the corpus ships no subject identifiers, so the split is over clips. Numbers are not comparable with the subject-disjoint cohorts — see splits.derivation.
- **chbmit** — No channel map: src/neuroatlas/configs/channel_maps/chbmit.yaml does not exist, unlike siena/tuab/tusz. Vocabulary-strict wrappers therefore receive the raw bipolar labels.
- **chbmit** — Window size disagrees between embed_chbmit_siena.py (30 s) and embed_chbmit_siena_sharded.py (2 s / 1 s) — audit OQ-11.4.
- **dod** — The multi-scorer evaluation needs the dreem-learning-evaluation scorers, which dod_fm_probe.sh pointed at /tmp/dreem-learning- evaluation/scorers -- a clone someone made by hand, with no default that survives a reboot. Pass --set expert_scorers_root=<path>.
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
- **liu2024** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **mesa** — Never loaded. MESA's corpus is not on the machine this was written on.
- **mesa** — 2056 recordings makes this the benchmark's largest sleep cohort; the default batch_size of 8 and num_workers of 0 are the physioex base defaults, not a considered choice for a corpus this size.
- **mesa** — App. B.3 uses this cohort for brain age, but the loader carries no demographics, so there is no `age` label mode here and brain age cannot run on it. Adding it means joining the NSRR mesa-sleep-dataset-*.csv on subject id -- the same work SHHS needs.
- **nakanishi2015** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **shin2017a** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
- **stages** — Never loaded. STAGES' corpus is not on the machine this was written on.
- **stages** — The loader defaults to recording='first', so a subject's repeat `_1` recording is excluded. The fold manifests are keyed on base subject ids, which matches, but this has not been run to confirm.
- **stages** — App. B.3 uses STAGES for brain age, but the loader surfaces no demographics, so there is no `age` label mode and brain age cannot run on it. Demographics exist in stages-harmonized-dataset-0.3.0.csv, which the loader reads for other purposes but does not publish.
- **tusz** — Montage conflict: the description says unipolar, the spec and embed_dataset._DEFAULT_MONTAGE both say bipolar. Kept bipolar to preserve behaviour — audit OQ-11.2.
- **weibo2014** — cohort.powerline_hz is a placeholder: MOABB does not expose the recording site and nothing reads this field yet.
