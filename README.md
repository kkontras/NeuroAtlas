# NeuroAtlas

The largest public benchmark to date for evaluating EEG foundation models:
**47 datasets, ~260,000 hours** of recordings, covering clinical EEG (epilepsy
and sleep medicine), brain-computer interfaces, and a newly introduced
**brain-age estimation** task.

## Why NeuroAtlas

EEG foundation models (FMs) promise unified representations that transfer to
many downstream tasks, but progress has been hard to measure. Published
evaluations differ in their datasets (both for pretraining and downstream
tests), in their EEG-specific preprocessing, and in their reliance on generic
machine-learning metrics that obscure clinical relevance. NeuroAtlas
addresses all three problems in a single harness:

- **Coverage.** Multiple datasets per task across four domains (epilepsy,
  sleep, BCI, brain age), so within-domain rankings can be measured rather
  than read off a single dataset.
- **Bespoke clinical metrics.** Beyond accuracy and macro-F1, NeuroAtlas
  reports evaluations that reflect downstream clinical utility:
  - *Epilepsy*: event-level (any-overlap and TAES) decision metrics, not
    only window-level accuracy.
  - *Sleep*: hypnogram-derived features (sleep efficiency, WASO, REM
    latency, fragmentation, ...) on top of per-epoch staging.
  - *Brain age*: the brain-age gap (predicted minus chronological age) as
    the principal clinical readout.
- **Comparable baselines.** Each FM is evaluated head-to-head against
  task-specific supervised baselines and against generic time-series FMs
  that were neither designed for EEG nor pretrained on it.
- **Generalizability without fine-tuning.** Frozen-backbone linear probing
  is the default protocol, isolating representation quality from end-to-end
  training tricks.

NeuroAtlas reports three high-level findings (see the paper):

1. EEG-specific FMs do not consistently outperform generic time-series FMs.
2. Standard ML metrics are insufficient to assess clinical utility; the
   bespoke metrics above frequently change rankings.
3. Model rankings vary substantially within a single domain across datasets,
   so single-dataset benchmarking can be misleading.

The takeaway: out of the box, current FMs are not yet a unified EEG model.
NeuroAtlas exposes this gap and provides the testbed of datasets and metrics
for the next generation of unified EEG foundation models.

## Domains and datasets

### Epilepsy

Eleven cohorts. Eight are continuously-labelled and span neonatal (Helsinki),
pediatric (CHB-MIT), adult seizure monitoring (Siena, SeizeIT1, AUB-Med), and
large-scale clinical archives (TUSZ, Epilepsiae, SeizeIT2). Three additional
cohorts (TUAB, NMT, Bonn) provide recording-level abnormality labels rather
than continuous seizure annotations and contribute ~2,500 abnormal recordings
to the recording-level evaluation only; of these, only Bonn's 200 ictal clips
carry explicit seizure annotation, while TUAB and NMT label broader pathology
including epileptiform activity, slowing, and artifacts. Splits are
patient-level throughout.

| Continuously-labelled | Recording-level only |
|---|---|
| Helsinki Neonatal Seizure, CHB-MIT, Siena, SeizeIT1, AUB-Med, TUSZ, Epilepsiae, SeizeIT2 | TUAB, NMT, Bonn |

### Sleep

Fifteen cohorts of polysomnography (PSG) recordings, spanning lab and clinical
PSG and at-home PSG, with multiple clinical populations including children
(CFS), people with cognitive impairment (PhysioNet 2026), and people with
obstructive sleep apnea (DOD, UCDDB). Several cohorts also carry event-level
annotations: MASS for microarousals; PhysioNet 2026 for microarousals,
respiratory events, and limb movements; UCDDB for respiratory events.

| Lab and clinical PSG | At-home PSG |
|---|---|
| Cleveland Family Study (CFS), DCSM, Dreem Open Datasets (DOD), Haaglanden Medisch Centrum (HMC), HomePAP, ISRUC-Sleep, MASS, PhysioNet 2026 (PN2026), Sleep-EDF Expanded, STAGES, UCDDB, Wisconsin Sleep Cohort (WSC) | MESA, MrOS, SHHS, Sleep-EDF |

### Brain age and neurodegenerative diseases

Ten cohorts of PSG recordings carrying age labels (~15,000 patients, ~193k
hours): CFS, HomePAP, ISRUC, MESA, MrOS, PhysioNet 2026, SHHS, Sleep-EDF
Cassette subset (SC), STAGES, WSC. PhysioNet 2026 additionally includes
patients with cognitive impairment alongside healthy controls.

### BCI

A diverse set of paradigms across 509 participants and approximately 159
hours of EEG, with substantial heterogeneity in channel configurations,
sampling rates, and recording designs (event-related trials versus continuous
sessions).

| Paradigm | Datasets |
|---|---|
| Motor imagery (MI) | BNCI2014_001, BNCI2014_004, BNCI2015_001, Weibo2014, Dreyer2023, Shin2017A, Liu2024 |
| ERP | BI2013a, BI2014a, EPFLP300, BNCI2014_008, ErpCore2021_N170 |
| SSVEP | Nakanishi2015, Kim2025BetaRange |
| Cognitive and affective | DREAMER (valence / arousal), EEGMat, ArithmeticTask |

MI datasets provide multi-second trials with moderate sample sizes per
subject; ERP (P300 and N170) datasets yield high-density event-locked
epochs; SSVEP datasets consist of short (~2-6 s) frequency-tagged trials.

## Foundation models supported

**EEG-specific FMs** (pretrained on EEG):
`biot`, `brainbert`, `cbramod`, `core_sleep`, `eegnet-v4`, `eegpt`, `labram`,
`neurogpt`, `neurolm`, `neurorvq`, `physioex`, `reve`, `sjepa`, `sleepfm`,
`sleepgpt`, `sleepyco`, `sleep_transformer`.

**Generic time-series FMs** (no EEG-focused architecture, no EEG
pretraining): `chronos`, `lag-llama`, `moirai`, `moment`, `timemoe`,
`timesfm`.

**Task-specific supervised baselines:** `brain_age_cnn`,
`seizure_transformer`, `steegformer`, `deepsoz_hem`.

## Tasks and evaluation modes

- `linear_probe`: frozen-backbone sklearn / GPU-torch linear probe (default).
- `native_head_eval`: drop in the FM's pretrained classifier head where one
  exists.
- `seizure_detection`: patient-grouped probe with event-level metrics
  (any-overlap sensitivity, FA-per-hour, TAES) on top of window-level
  scores.
- `arousal_detection`, `respiratory_event_detection`: event-level sleep
  metrics.
- `patient_classification`: per-patient aggregation (e.g., abnormal versus
  normal EEG).
- `brain_age`: regression head with brain-age-gap reporting and hypnogram-
  conditional analysis where staging is available.
- `attention_probe`, `attention_probe_patient`: attention-pooled probes for
  variable-length windows.

## Quickstart

```bash
pip install "neuroatlas[fm]"          # install torch for your CUDA version first
neuroatlas config init --data-root /data/eeg

neuroatlas list benchmarks            # what can be run
neuroatlas data status sleep_stage    # which datasets are here, and where they were looked for
neuroatlas data download sleep_edf_expanded
neuroatlas models download biot_pretrained
neuroatlas check sleep_stage -m biot_pretrained     # one real batch, in seconds
neuroatlas run sleep_stage -m biot_pretrained       # embed, probe over 5 folds
neuroatlas results sleep_stage
```

For a whole suite on a cluster, `neuroatlas submit <benchmark> -m all_fm
--out runs/x` writes one HTCondor or SLURM job per dataset × model and
`neuroatlas status --out runs/x` tracks them; `neuroatlas leaderboard`
ranks across benchmarks. From Python, `neuroatlas.api` has the same verbs.

- [docs/user_guide.md](docs/user_guide.md) -- every step, with real output
- [docs/cli.md](docs/cli.md) -- every command and flag (generated from the code)

Each benchmark names a protocol the paper ran: `neuroatlas show <benchmark>`
prints the `embed` and `probe` command lines it expands to, and those are
the lines of `run/default_runs.sh`, the record of every experiment in the
paper. The five verbs underneath -- `fetch`, `prepare`, `embed`, `probe`,
`hypnogram` -- remain available as `neuroatlas <verb>`.

## Repository layout

```
src/neuroatlas/
├── cli/                    # the `neuroatlas` command, one module per command
├── api.py                  # the same verbs from Python
├── catalog.py, selectors.py, config.py, data.py, models.py,
│   check.py, run.py, submit.py, results.py
├── benchmarking_helpers/   # engine: registry, runner, cache, probes, channel maps
├── entrypoints/            # the verbs: fetch / prepare / embed / probe / hypnogram
├── extensions/
│   ├── datasets/           # dataset specs, adapters, readers, corpus builders
│   ├── models/backbones/   # foundation-model wrappers
│   └── tasks/              # linear probe, seizure detection, brain age, ...
└── configs/                # shipped with the package
    ├── benchmarks/         # one file per benchmark (neuroatlas list benchmarks)
    ├── cohorts/            # one directory per corpus: identity, labels, splits, defaults
    ├── channel_maps/       # per-cohort electrode maps for each model family
    ├── folds/              # frozen train/val/test splits
    ├── tasks/              # probe presets (--task <name>)
    └── model_groups.yaml   # the paper's model groups and the -m aliases

run/default_runs.sh         # every experiment in the paper, one line each
docs/                       # user guide and command reference
reproduction/               # a minimal end-to-end notebook
artifacts/                  # gitignored: weights, caches and results of a checkout
```

Settings, tokens and -- outside a checkout -- weights, caches and results
live under `~/.neuroatlas` (`$NEUROATLAS_HOME`).

## Reproducibility notes

- All probes use deterministic seeds via
  `neuroatlas.benchmarking_helpers.seed_everything`.
- Embeddings are cached on first extraction and reused across probe variants.
  Cache layout is documented in
  `src/neuroatlas/benchmarking_helpers/runtime/cache.py`.
- Channel-map YAMLs in `src/neuroatlas/configs/channel_maps/` define per-dataset channel
  translations for each foundation-model family. They are required to
  reproduce the cross-dataset evaluation.

## License

MIT; see `LICENSE`. The vendored DeepSOZ and Seizure-Transformer code keep their upstream licences (`third_party/*/LICENSE*`).

## Citation

Citation block to follow.
