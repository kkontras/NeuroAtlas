# 4. Benchmarks

A benchmark is one protocol of the paper. It fixes what is predicted, on which datasets, and how
the result is scored, so that every model is measured the same way. There are twelve, in four
domains: sleep, brain age, epilepsy and brain-computer interfaces (BCI). This chapter describes
them domain by domain. [Running benchmarks](running.md) shows how to run one.

## 4.1 Exploring the benchmarks

The command line describes every benchmark itself:

```bash
neuroatlas list benchmarks
neuroatlas show sleep_stage
```

`list benchmarks` shows the twelve with their domain, headline metric, quick dataset, datasets and
variants. The quick dataset is the one `run` and `check` use by default. Add `--dataset full` to
use all of a benchmark's datasets. Every listing takes `--format table|csv|md|json`.

`show` explains one benchmark. It says what the headline metric is computed over, what a fold is,
what ± means, which other metrics are reported, and which datasets and variants exist. It ends
with the `embed` and `probe` commands that `run` executes:

```text
$ neuroatlas show sleep_stage
sleep_stage: Sleep staging  (sleep, App. C.2)
  Which sleep stage (W, N1, N2, N3, REM) is each 30 s epoch?

headline    kappa: Cohen's kappa over 30 s epochs, 5 stages (W, N1, N2, N3,
            REM), each fold's test subjects pooled, unscored epochs left out;
            higher is better; chance 0
folds       5 subject-level folds; the probe's C is chosen from 0.001-100 on
            validation Cohen's kappa, unweighted loss; ± = population SD over
            the folds
...
datasets    15; the quick dataset (--dataset single): sleep_edf_expanded
  cfs
  ...
the commands `neuroatlas run` runs (all datasets, default variant):
  neuroatlas embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset cfs --set window_s=30 --set stride_s=30
  ...
  neuroatlas probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset wsc --task sleep_staging
```

These are the same commands as in `run/default_runs.sh`, the record of every experiment in the
paper.

Each benchmark reports one headline metric, the first number in its results table, and a few
others beside it. The ± after the headline is a standard deviation over folds. Every SD except
epilepsy's is the population SD (ddof=0). Each benchmark also leaves out the checkpoints the paper
does not evaluate on it ([Models a benchmark leaves out](models.md#33-models-a-benchmark-leaves-out)).

## 4.2 Sleep

The six sleep benchmarks read the night in 30 s epochs. `sleep_stage` asks which sleep stage (W,
N1, N2, N3, REM) each epoch is. Three event benchmarks ask whether an epoch contains a
microarousal (`sleep_arousal`), an apnea or hypopnea (`sleep_respiratory`), or a periodic limb
movement (`sleep_limb`). `sleep_diagnosis` works on each subject's mean embedding, all of its
30 s epochs averaged, and asks for obstructive sleep apnea against healthy (DOD), cognitive
impairment (PhysioNet 2026), or which disorder (ISRUC's mixed cohort). `sleep_hypnogram` asks how
well sleep-architecture features (total sleep time, stage proportions, WASO, REM latency,
fragmentation) computed from the predicted hypnograms match those from the true ones.

| Benchmark | Headline | Scored over | Folds and ± |
|---|---|---|---|
| `sleep_stage` | Cohen's κ, 5 stages. C chosen from 0.001-100 on validation κ, unweighted loss. | 30 s epochs, each fold's test subjects pooled | 5 subject-level folds, SD over folds |
| `sleep_arousal` | AUPRC. An epoch is positive if it holds more than 3 s of arousal. | 30 s epochs | 5 subject-level folds, SD over folds |
| `sleep_respiratory` | AUPRC. More than 10 s of apnea or hypopnea (RERAs not counted). | 30 s epochs | the same |
| `sleep_limb` | AUPRC. More than 0.5 s of periodic limb movement. | 30 s epochs | the same |
| `sleep_diagnosis` | AUROC. Logistic regression at C = 1, unweighted loss. `n/a` on ISRUC, whose diagnosis has more than two classes. | subjects, one mean embedding each | 5 subject-level folds, SD over folds |
| `sleep_hypnogram` | Mean Pearson r between hypnogram features from the predicted and the scored hypnograms | test recordings of all folds pooled | one value, no ± |

| Benchmark | Datasets | Quick dataset |
|---|---|---|
| `sleep_stage` | CFS, DCSM, DOD, HMC, HomePAP, ISRUC, MASS, MESA, MrOS, PhysioNet 2026, SHHS, Sleep-EDF Expanded, STAGES, UCDDB, WSC | Sleep-EDF Expanded |
| `sleep_arousal` | MASS (subset SS01, the only one with arousal annotations), PhysioNet 2026 | MASS |
| `sleep_respiratory` | PhysioNet 2026, UCDDB | UCDDB |
| `sleep_limb` | PhysioNet 2026 | PhysioNet 2026 |
| `sleep_diagnosis` | DOD, PhysioNet 2026, ISRUC | DOD |
| `sleep_hypnogram` | DCSM, DOD, ISRUC, MASS, PhysioNet 2026, Sleep-EDF Expanded, UCDDB, WSC | Sleep-EDF Expanded |

For the event benchmarks, a guess scores the share of positive test epochs, and the results table
shows it in a `chance` column. `sleep_hypnogram` fits no probe of its own. It predicts every
epoch again with the saved sleep staging probes and computes 34 sleep features per recording, so
run it after `sleep_stage`:

```bash
neuroatlas run sleep_stage -m biot_pretrained
neuroatlas run sleep_hypnogram -m biot_pretrained
```

## 4.3 Brain age

`brain_age` asks how old the sleeper is. A ridge regressor predicts the age from each subject's
mean embedding, and the brain-age gap (predicted minus true age) is the clinical readout. The
benchmark reuses the 30 s sleep embeddings rather than reading the recordings again, so on
Sleep-EDF, after `sleep_stage`, it takes about a minute.

| Benchmark | Headline | Scored over | Folds and ± |
|---|---|---|---|
| `brain_age` | MAE in years. Ridge regression, alpha by nested 4-fold CV. | subjects (recordings for ISRUC, SHHS and WSC), one mean embedding each | 5 subject-level folds stratified on age, SD over folds |

Brain age runs on ten datasets: CFS, HomePAP, ISRUC, MESA, MrOS, PhysioNet 2026, SHHS, Sleep-EDF
Expanded, STAGES and WSC. The quick dataset is Sleep-EDF Expanded. Brain age has its own folds and
keeps each dataset's published protocol:

| Dataset | Protocol |
|---|---|
| Sleep-EDF Expanded | its 78 Sleep Cassette subjects in age-stratified folds (seed 42), not the sleep-staging folds |
| ISRUC, WSC | scored per recording, with a subject's recordings in one fold |
| PhysioNet 2026 | 100 healthy and 100 impaired age-matched subjects held out of every fold; the ridge trains on healthy subjects only |
| CFS, MrOS | their age-stratified subject folds |
| SHHS | scored per recording, each with the participant's age at that visit, on its five sleep-staging folds |
| HomePAP, MESA, STAGES | five age-stratified subject folds (seed 42) over the recordings of their sleep-staging embeddings, one per participant |

For HomePAP, MESA and STAGES, each recording's age comes from the study's NSRR table, joined on
the participant id. A recording whose participant has no age in the table is left out, and `-v`
names it.

| Dataset | Table | Age column | Id |
|---|---|---|---|
| HomePAP | `homepap-baseline-dataset-0.2.0.csv` | `age` (baseline visit) | `nsrrid` (`homepap-lab-full-1600001` is 1600001) |
| MESA | `mesa-sleep-dataset-0.8.0.csv` | `sleepage5c` (sleep exam) | `mesaid` (`mesa-sleep-0001` is 1) |
| STAGES | `stages-harmonized-dataset-0.3.0.csv` | `nsrr_age` | `subject_code` (the file name, `BOGN00001`) |

The other datasets keep the ages in the embedding cache, which needs `xlrd` for Sleep-EDF and
`openpyxl` for ISRUC when the cache is built ([Caches](advanced.md#73-caches)).

## 4.4 Epilepsy

`epilepsy` asks whether a 10 s EEG window is part of a seizure. For TUAB, NMT and Bonn, which
label whole recordings rather than seizures, it asks whether the recording is abnormal.

| Benchmark | Headline | Scored over | Folds and ± |
|---|---|---|---|
| `epilepsy` | Event-level Sens@FA AUC over 0.1-100 false alarms per hour. C chosen on validation AUPRC. `n/a` on Bonn, TUAB and NMT. | seizure events | 5 patient-level folds, sample SD (ddof=1) of the per-fold AUCs |

The ten datasets are Bonn, CHB-MIT, EPILEPSIAE, Helsinki Neonatal, NMT, Siena, SeizeIT1, SeizeIT2,
TUAB and TUSZ. The quick dataset is Siena. On Bonn, whose recordings have no seizure events, a
fold's result line shows the AUROC instead. The headline is the AUC of the folds'
median curve, so it is not the mean of the per-fold values ([Per-fold values](results.md#62-per-fold-values)).

## 4.5 Brain-computer interfaces

The four BCI benchmarks classify single trials and test every model on people the probe never
saw (leave-one-subject-out). `bci_motor_imagery` asks which movement the person is imagining.
`bci_erp` asks whether an EEG segment is the response to a target stimulus (P300, N170).
`bci_ssvep` asks which flickering target the person is looking at. `bci_cognitive` covers emotion
(DREAMER valence and arousal) and mental arithmetic (EEGMat, ArithmeticTask).

| Benchmark | Datasets | Quick dataset |
|---|---|---|
| `bci_motor_imagery` | BNCI2014_001, BNCI2014_004, BNCI2015_001, Dreyer2023, Liu2024, Shin2017A, Weibo2014 | BNCI2014_001 |
| `bci_erp` | BI2013a, BI2014a, BNCI2014_008, EPFLP300, ErpCore2021_N170 | BI2013a |
| `bci_ssvep` | Kim2025BetaRange, Nakanishi2015 | Nakanishi2015 |
| `bci_cognitive` | DREAMER valence, DREAMER arousal, EEGMat, ArithmeticTask | EEGMat |

| Benchmark | Headline | Scored over | Folds and ± |
|---|---|---|---|
| `bci_*` | Balanced accuracy. Logistic regression, unweighted loss, C from 0.001 to 100 chosen on the validation subject's Cohen's kappa. | one held-out subject's trials | LOSO: fold k tests the k-th subject in ascending order, one other subject (drawn with seed 42) validates; SD over held-out subjects |

Chance is 1 over the number of classes, and the results table shows it in a `chance` column. The
paper's figures plot the rescaling (BA - 1/C) / (1 - 1/C) * 0.5 + 0.5 of balanced accuracy
(App. C.4, Eq. 4). It puts chance at 0.5 for every number of classes C.

### BCI variants

The four BCI benchmarks run confound filtering by default (App. D.6). Each model reads the trials
in its own format:

| Models | Band | Rate | Notch |
|---|---|---|---|
| EEGPT | 0.5-70 Hz | 256 Hz | the dataset's line frequency |
| ST-EEGFormer, SleepFM | 0.1-64 Hz | 128 Hz | the dataset's line frequency |
| every other model | 0.1-75 Hz | 200 Hz | 50 Hz |

The notch covers the line frequency and its harmonics, and every trial is average-referenced.
Confound filtering keeps the model's rate and notch and changes the band, each paradigm its own
way:

| Paradigm | Confound filtering (the default) | No filtering (`--variant no_filtering`) |
|---|---|---|
| Motor imagery | 4-40 Hz, the trial from 1 s to 4 s after the cue | the model's band, the trial from the cue to 4 s after it |
| ERP | 0.5-40 Hz, the cohort's own trial window | the model's band, the cohort's own trial window |
| SSVEP | the model's high-pass and no low-pass, the cohort's own trial window | the model's band, the cohort's own trial window |
| Cognitive | 4-40 Hz, the dataset's file of that band ([built from the raw data](data.md#26-files-built-from-the-raw-data)) | the model's band, the dataset's other file |

For motor imagery, dropping the first second after the cue means the score cannot come from the
cue's evoked response or the eye movement towards it. SSVEP has no low-pass so that the stimulus
harmonics stay in.

Each benchmark has four variants, the two filterings crossed with the two ways of turning a
trial's patch tokens into one vector:

| Variant | Filtering | Embedding | Paper (motor imagery) |
|---|---|---|---|
| `default` | confound filtering | patch tokens averaged | Fig. 26a, confound-filtering bars, and Fig. 26b |
| `token_flattening` | confound filtering | patch tokens concatenated in order | Fig. 5a, confound-filtering bars, and Fig. 5b |
| `no_filtering` | no filtering | patch tokens averaged | Fig. 26a, no-filtering bars |
| `no_filtering_token_flattening` | no filtering | patch tokens concatenated in order | Fig. 5a, no-filtering bars |

Select a variant with `--variant`, as in
`neuroatlas run bci_motor_imagery -m all_fm --variant token_flattening`. Its results go to their
own folder. `neuroatlas show <benchmark>` names the figure of each variant for the other
paradigms.

Other names are accepted: `confound_filtering` and `confound_control` for `default`,
`confound_filtering_token_flattening` and `confound_control_per_patch` for `token_flattening`, and
`per_patch` for `no_filtering_token_flattening`. A results folder written under one of these names
reads as that variant.

`results` reads each result as the variant it was made with, which every result records. A result
made without confound filtering is a `no_filtering` result, also when it is in the default's
folder.
