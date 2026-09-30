# NeuroAtlas

The largest public benchmark to date for evaluating EEG foundation models:
**47 datasets, ~260,000 hours** of recordings, covering clinical EEG (epilepsy
and sleep medicine), brain-computer interfaces, and a newly introduced
**brain-age estimation** task.

Anonymous release for double-blind review.

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
# 1. Set the data and cache roots (point at your local dataset mirrors)
export EEG_DATA_ROOT=/path/to/eeg_datasets
export EEG_CACHE_ROOT=/path/to/embedding_cache
export PY_ENV_ROOT=/path/to/python_env  # optional

# 2. Install
pip install -e .

# 3. Hugging Face token for gated model weights (never commit it)
mkdir -p .secrets && printf '%s' 'hf_xxx' > .secrets/hf_token && chmod 600 .secrets/hf_token
# .secrets/ is gitignored. The loader resolves a token from
# $NEUROATLAS_HF_TOKEN_FILE, then .secrets/hf_token, then $HF_TOKEN.

# 4. Run a smoke probe
neuroatlas probe --dataset sleep_edf_expanded --models cbramod --set n_folds=1
```

There is one entrypoint per verb, not per dataset: `fetch`, `prepare`,
`embed`, `probe`, and `hypnogram` for the one paper result computed from
probe output rather than from embeddings. `run/default_runs.sh` lists every
experiment in the paper as a single command each.

## Repository layout

```
src/
├── benchmarking_helpers/   # Probe, runner, cache, metrics, channel-map utilities
├── entrypoints/            # fetch / prepare / embed / probe / hypnogram
└── extensions/
    ├── datasets/           # Dataset specs, adapters, dataio and physioex readers
    ├── models/backbones/   # Foundation-model wrappers
    └── tasks/              # Task definitions (linear probe, regression, ...)

run/
├── launch.sh               # <verb> --dataset D --models M, over a whole domain
├── default_runs.sh         # every experiment in the paper, one line each
└── _launch_sets.py         # which datasets and models each bundle expands to

artifacts/                  # gitignored, not in a fresh clone
├── models/foundation/      # EEG-FM weights; most are fetched from HuggingFace
│                           # on first use, but REVE is a local artifact
├── models/shhs/            # the supervised sleep baselines (CoRe-Sleep,
│                           # SleepTransformer, SleePyCo) -- local, no auto-fetch
└── models/supervised/      # Seizure-Transformer -- local, no auto-fetch

src/neuroatlas/configs/cohorts/            # one directory per corpus: identity,
                            # labels, splits, runtime defaults
src/neuroatlas/configs/channel_maps/       # per-cohort channel-vocabulary maps
src/neuroatlas/configs/folds/              # frozen train/val/test splits
src/neuroatlas/configs/tasks/              # probe presets (--task <name>)
tests/                      # Unit and integration tests (pytest)
```

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

MIT. See `LICENSE` (placeholder for camera-ready).

## Citation

Anonymous submission. Citation block will be added after de-anonymization.
