# NeuroAtlas

A benchmark for evaluating EEG foundation models: **43 datasets** in the
evaluation (`neuroatlas list datasets`), covering clinical EEG (epilepsy and
sleep medicine), brain-computer interfaces, and a newly introduced
**brain-age estimation** task. This repository is the benchmark harness: the
`neuroatlas` command and its Python API.

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

The 43 datasets are 10 epilepsy, 15 sleep and 18 BCI datasets; brain age
reuses six of the sleep cohorts. They form 12 benchmarks
(`neuroatlas list benchmarks`). A cohort marked *planned* is named by the
paper but not run here yet; `neuroatlas show <benchmark>` says why.

### Epilepsy

Ten cohorts in the `epilepsy` benchmark, scored by window-level AUROC with
event-level metrics beside it. Seven are continuously labelled (a label per
10 s window); three label whole recordings as normal or abnormal. Folds are
patient-level, except Bonn, which ships no subject identifiers (its folds
are over clips). AUB-Med is planned: readable, but the paper reports no
result for it.

| Continuously labelled | Recording-level |
|---|---|
| Helsinki Neonatal, CHB-MIT, Siena, SeizeIT1, SeizeIT2, TUSZ, Epilepsiae | TUAB, NMT, Bonn |

### Sleep

Fifteen polysomnography (PSG) cohorts in `sleep_stage`, five-class staging
of 30 s epochs:

| Lab and clinical PSG | At-home PSG |
|---|---|
| Cleveland Family Study (CFS), DCSM, Dreem Open Datasets (DOD), HMC, HomePAP (lab, full montage), ISRUC-Sleep, MASS, PhysioNet 2026, STAGES, UCDDB, Wisconsin Sleep Cohort (WSC) | MESA, MrOS, SHHS |

Sleep-EDF Expanded is one dataset with two subsets: Sleep Cassette, recorded
at home, and Sleep Telemetry, recorded in hospital.

Five more sleep benchmarks reuse some of these cohorts:
`sleep_hypnogram` (hypnogram features from the staging results: DCSM, DOD,
ISRUC, MASS, PhysioNet 2026, Sleep-EDF, UCDDB, WSC), `sleep_diagnosis`
(DOD obstructive sleep apnea, PhysioNet 2026 cognitive impairment, ISRUC
pathology), `sleep_arousal` (MASS, PhysioNet 2026), `sleep_respiratory`
(PhysioNet 2026, UCDDB) and `sleep_limb` (PhysioNet 2026).

### Brain age

Six cohorts in `brain_age`, a ridge regressor on each subject's mean
embedding (each recording's, for ISRUC and WSC), scored by MAE in years:
CFS, ISRUC, MrOS, PhysioNet 2026, Sleep-EDF Expanded and WSC. It reads the
same 30 s embeddings as sleep staging, and each cohort keeps the paper's own
split: on Sleep-EDF the Sleep Cassette subset only (78 subjects), in
age-stratified subject folds, not the sleep-staging folds. Four more
cohorts the paper uses are planned, because their readers carry no
participant ages yet: SHHS, MESA, HomePAP and STAGES.

### BCI

Eighteen datasets in four benchmarks, all scored leave-one-subject-out.
DREAMER counts twice, once per label (valence, arousal).

| Benchmark | Datasets |
|---|---|
| `bci_motor_imagery` | BNCI2014_001, BNCI2014_004, BNCI2015_001, Dreyer2023, Liu2024, Shin2017A, Weibo2014 |
| `bci_erp` | BI2013a, BI2014a, BNCI2014_008, EPFLP300, ErpCore2021_N170 |
| `bci_ssvep` | Nakanishi2015, Kim2025BetaRange |
| `bci_cognitive` | DREAMER valence, DREAMER arousal, EEGMat, ArithmeticTask |

Fourteen of them come through MOABB. The four cognitive ones are read from
preprocessed files the authors hold; see the user guide.

## Models

44 checkpoints of 19 model families, plus one planned
(`neuroatlas list models`, `neuroatlas list aliases`). Each group below is
also an alias for `-m`.

**EEG foundation models** (`all_fm`, 12 checkpoints): `biot`, `cbramod`,
`eegpt`, `labram`, `neurogpt`, `neurolm`, `neurorvq`, `reve`, `sleepfm`,
`steegformer` (small, base, large).

**General time-series foundation models** (`all_ts`, 10 checkpoints):
`chronos` (T5 tiny, small, base, large), `moirai` (small, base, large),
`moment` (small, base, large).

**Supervised baselines**, trained with labels on other EEG data
(`all_supervised`, 20 checkpoints): `core_sleep`, `deepsoz_hem`, `eegnetv4`
(12 checkpoints, one per training dataset), `seizure_transformer`,
`sleep_transformer`, `sleepyco`. A brain-age CNN (`brain_age_cnn`) is
planned.

**Untrained baselines** (`all_random`): `cbramod_random_init` and
`reve_random_init`, the same architectures with random weights.

## Tasks

A benchmark names the probe task that scores it (`neuroatlas list tasks`):
`linear_probe` (sleep staging, BCI), `seizure_detection` (epilepsy: a
balanced logistic regression with event-level metrics), `brain_age` (ridge
regression with nested cross-validation), `patient_classification` (sleep
diagnosis), `arousal_detection` and `respiratory_event_detection` (sleep
events). The registry also has `attention_probe`, `attention_probe_patient`,
`lstm_probe` and `native_head_eval`, which no benchmark uses.

## Install

Python 3.10 or newer; 3.11 is what the paper used and what is tested.
Install from a clone. The distribution is named `neuroatlas-bench` (a built
wheel is `neuroatlas_bench-<version>-py3-none-any.whl`); the command and the
import stay `neuroatlas`. It is not on PyPI yet. The PyPI name `neuroatlas`
belongs to an unrelated project: do not install it. The GitHub repository
is not public yet (2026-10-05); until it is, cloning needs access from the
authors.

```bash
git clone https://github.com/kkontras/NeuroAtlas.git && cd NeuroAtlas
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

pip install torch torchvision torchaudio   # first; see the note below
pip install -r requirements-fm.txt         # the paper's exact versions
pip install -e ".[fm]"                     # the neuroatlas command
pip check                                  # No broken requirements found.
```

In a fresh venv this took 4 min for torch, 2 min for `requirements-fm.txt`
and seconds for the rest; the venv is 6.5 GB.

**PyTorch.** `pip install torch` currently gives a CUDA 13 build (2.14.1+cu130
on 2026-10-02), with kernels for GPUs of compute capability 7.5 and newer
only (Turing onwards). `nvidia-smi` must report CUDA 13.0 or higher. For an
older GPU or driver, install a CUDA 12 build from pytorch.org instead.

**Optional extras**, after the block above:

```bash
pip install -e ".[ts]"                       # Chronos
pip install --no-deps "momentfm==0.1.4"      # MOMENT
pip install -e ".[fm,bci]"                   # the 14 MOABB BCI datasets: see below
pip install --no-deps "moabb==1.2.0"
```

- `momentfm` declares old pins of transformers, numpy and huggingface-hub;
  installed with `--no-deps` it runs with this stack. From then on
  `pip check` lists those three pins, and later installs print a pip
  "dependency conflicts" error about them. Both are expected.
- The BCI datasets stay on the paper's numpy 1.26.4. The `bci` extra holds
  MOABB's dependencies that fit this stack; MOABB itself goes in with
  `--no-deps`, as 1.2.0, the last release on numpy<2. `pip check` then also
  reports its declared caps (scikit-learn<1.6, urllib3<2, seaborn<0.13),
  which are harmless: BNCI2014_001 and BNCI2014_004 trials and labels come
  out bit-identical to moabb 1.7.2's. Dreyer2023 and Kim2025BetaRange,
  newer than moabb 1.2.0, ship inside the package.
- Moirai needs its own environment (Python 3.10, torch 2.4.1):
  `requirements-tsfm.txt` has the recipe.

`neuroatlas models status` says `package missing`, with the install line,
for a model whose package is not installed.

## Quickstart

On an open dataset, Sleep-EDF Expanded (8.1 GB), with one model:

```bash
neuroatlas config init --data-root /data/eeg
neuroatlas data download sleep_edf_expanded --mirror aws   # PhysioNet's copy on AWS
neuroatlas data status sleep_stage
neuroatlas models download biot_pretrained
neuroatlas check sleep_stage -m biot_pretrained      # one real batch through the model
neuroatlas run sleep_stage -m biot_pretrained        # embed, then probe 5 folds
neuroatlas results sleep_stage
```

Measured on 2026-10-02, on a machine with an RTX 4500 Ada and 28 cores that
other jobs were also using: the download took 18 min from the AWS copy (at
the ~100 KB/s the tester saw from physionet.org, 8 GB takes about a day);
`check` 10 s; extracting the embeddings 29 min (17 min on an idle GPU, in
the tester's run); the five probe folds, CPU work, 60 min. BIOT scored
0.656 ± 0.017 balanced accuracy over the 5 folds, against 0.657 ± 0.016 in
the authors' run.

`neuroatlas show sleep_stage` explains the benchmark and prints the `embed`
and `probe` commands `run` executes; they are equivalent to the lines of
`run/default_runs.sh`, the record of every experiment in the paper. The five
verbs underneath -- `fetch`, `prepare`, `embed`, `probe`, `hypnogram` --
remain available as `neuroatlas <verb>`.

For a whole suite on a cluster, `neuroatlas submit <benchmark> -m all_fm
--out runs/x` writes one HTCondor or SLURM job per dataset × model, and
`neuroatlas status --out runs/x` tracks them; `neuroatlas leaderboard` ranks
across benchmarks. From Python, `neuroatlas.api` has the same verbs.

- [docs/user_guide.md](docs/user_guide.md) -- every step, with real output
- [docs/cli.md](docs/cli.md) -- every command and flag (generated from the code)

## Repository layout

```
src/neuroatlas/
├── cli/                    # the `neuroatlas` command, one module per command
├── api.py                  # the same verbs from Python
├── catalog.py, selectors.py, config.py, _paths.py, data.py, models.py,
│   check.py, run.py, submit.py, results.py
├── benchmarking_helpers/   # engine: registry, runner, cache, probes, channel maps
├── entrypoints/            # the verbs: fetch / prepare / embed / probe / hypnogram
├── extensions/
│   ├── datasets/           # dataset adapters, readers, corpus builders
│   ├── models/backbones/   # model wrappers (third_party/: vendored code)
│   └── tasks/              # linear probe, seizure detection, brain age, ...
└── configs/                # shipped with the package
    ├── benchmarks/         # one file per benchmark (neuroatlas list benchmarks)
    ├── cohorts/            # one directory per corpus: identity, labels, splits (folds.json), defaults
    ├── channel_maps/       # per-cohort electrode maps for each model family
    ├── folds/              # frozen train/val/test splits
    ├── tasks/              # probe presets (--task <name>)
    └── model_groups.yaml   # the paper's model groups and the -m aliases

run/default_runs.sh         # every experiment in the paper, one line each
docs/                       # user guide and command reference
reproduction/               # a minimal end-to-end notebook
requirements-fm.txt         # the paper's exact versions
requirements-tsfm.txt       # the separate Moirai environment
```

Settings and tokens live under `~/.neuroatlas` (`$NEUROATLAS_HOME`), and so,
by default, do weights, caches and results, however the package was
installed. The user guide lists every location the tool reads and writes
(["Where things live"](docs/user_guide.md#where-things-live)), including
those outside `$NEUROATLAS_HOME`: the raw data, the MOABB folder and the
Hugging Face cache.

## Reproducibility notes

- Probes are seeded (`neuroatlas.benchmarking_helpers.seed_everything`).
- Embeddings are extracted once per dataset × model and reused by every
  fold, probe and benchmark that reads the same windows; the user guide
  describes the cache and its exceptions.
- The channel maps in `src/neuroatlas/configs/channel_maps/` decide which
  electrodes each model family gets on each dataset. They are part of the
  protocol.
- Folds come from a frozen file where the paper's split is known
  (`configs/cohorts/<dataset>/folds.json`, `configs/folds/`), otherwise from
  a seeded splitter. Where a cohort's folds are not the ones behind the
  published numbers, its `cohort.yaml` says so under
  `provenance.known_issues`.
- The same embeddings probed on two machines agree to about 1e-3 (BIOT ×
  Sleep-EDF fold 0: 0.6624 on one machine, 0.6619 on another).
- CoRe-Sleep on Sleep-EDF, extracted afresh, gives the paper's numbers:
  sleep staging κ 0.8197 (published 0.8196), macro-F1 0.7690 (0.7689),
  balanced accuracy 0.7621; brain-age MAE per fold 11.73, 8.61, 11.20,
  10.42, 10.00.

## License

MIT; see `LICENSE`. Three vendored components keep their upstream
licences. In `src/neuroatlas/extensions/models/backbones/third_party/`:
DeepSOZ (`deepsoz_hem/LICENSE`, GPL-3.0, with its checkpoint; whether it
can stay inside an MIT package is not settled) and Seizure-Transformer
(`seizure_transformer/LICENSE.upstream`, MIT). In
`src/neuroatlas/extensions/datasets/dataio/moabb_vendored/`: MOABB's
Dreyer2023 and Kim2025BetaRange readers (`LICENSE.moabb`, BSD-3-Clause).

## Citation

Citation block to follow.
