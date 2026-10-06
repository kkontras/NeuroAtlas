# NeuroAtlas

A benchmark for EEG foundation models across clinical EEG (epilepsy, sleep
medicine, brain-age estimation) and brain-computer interfaces: 42 datasets
(`neuroatlas list datasets` shows 43 entries, because DREAMER appears once
per label), 44 model checkpoints, frozen-backbone probing with clinical
metrics. This repository is the benchmark harness: the `neuroatlas` command
and its Python API. Paper: [arXiv:2605.14698](https://arxiv.org/abs/2605.14698).

- [Install](#install)
- [The commands](#the-commands) and a [cheat sheet](#command-cheat-sheet) of every command
- [Test it by hand](#test-it-by-hand): one pass through every command, with
  the output to expect
- [Run each benchmark](#run-each-benchmark): one block per benchmark, with
  its datasets and how to get them
- [What is in the benchmark](#why-neuroatlas): domains, datasets, models,
  tasks
- [docs/user_guide.md](docs/user_guide.md): every step in detail, with real
  output; [docs/cli.md](docs/cli.md): every command and flag (generated from
  the code)

## Install

From a clone, with Python 3.11 (3.10 or newer works; 3.11 is what the
paper used). The repository is private for now: cloning needs access from
the authors.

```bash
git clone https://github.com/kkontras/NeuroAtlas.git && cd NeuroAtlas
conda create -n neuroatlas python=3.11 -y && conda activate neuroatlas
pip install -e ".[fm]" -c requirements-fm.txt
```

Conda only provides Python here; every package comes from pip, pinned by
`requirements-fm.txt`. A plain venv works the same: replace the second line
with `python3.11 -m venv .venv && source .venv/bin/activate`.

That is the whole install for the EEG foundation models, the supervised
baselines and every sleep, epilepsy and brain-age dataset. `[fm]` lists
everything a run imports; `-c requirements-fm.txt` pins each package to the
version the paper used. In a fresh environment it took 5 minutes; the
environment is 6.3 GB, most of it
PyTorch.

**Everything else** (the 14 MOABB BCI datasets, Chronos and MOMENT) is two
more lines:

```bash
pip install -e ".[fm,bci,ts]" -c requirements-fm.txt
pip install --no-deps "moabb==1.2.0" "momentfm==0.1.4"
```

The second line is separate because both packages declare old pins that
would downgrade the stack: moabb 1.2.0 (the last release on the paper's
numpy 1.26.4) declares scikit-learn<1.6, urllib3<2 and seaborn<0.13;
momentfm declares transformers==4.33.3, numpy==1.25.2 and
huggingface-hub==0.24.0. Both run with this stack (BNCI2014_001 and
BNCI2014_004 trials come out bit-identical to moabb 1.7.2's), so they go in
without their dependencies, and `pip check` lists those declared pins;
that is expected. Dreyer2023 and Kim2025BetaRange, newer than moabb 1.2.0,
ship inside the package. Moirai needs its own environment (Python 3.10,
torch 2.4.1): `requirements-tsfm.txt` has the recipe.

**The name.** The package is `neuroatlas-bench` (a built wheel is
`neuroatlas_bench-<version>-py3-none-any.whl`); the command and the import
are `neuroatlas`. It is not on PyPI yet, and the PyPI project called
`neuroatlas` is an unrelated one: do not `pip install neuroatlas`.

**PyTorch.** pip installs the default build, currently CUDA 13 (2.14.1+cu130
on 2026-10-02), with kernels for GPUs of compute capability 7.5 and newer
(Turing onwards); `nvidia-smi` must report CUDA 13.0 or higher. For an older
GPU or driver, install a CUDA 12 build from pytorch.org before the line
above; pip keeps it.

`neuroatlas models status` says `package missing`, with the install line,
for a model whose package is not installed.

## The commands

Everything goes through one command, `neuroatlas <command>`; `neuroatlas
<command> --help` shows its options, and `-v`, `--log FILE` and `--online`
work with every command.

| Command | What it does |
|---|---|
| `config init / show / set / unset / path` | Where data, caches, results and weights live (`~/.neuroatlas/config.yaml`). |
| `config token hf / github / nsrr` | Save a Hugging Face, GitHub or NSRR token: asked for without being shown, stored `chmod 600`. |
| `list benchmarks / datasets / models / aliases / tasks` | What exists. |
| `show <benchmark>` | Explain a benchmark and print the `embed` and `probe` commands it runs. |
| `data status / download / prepare` | Is each dataset here; download the ones that can be downloaded; build an optional cache. |
| `models status / download` | Are a selection's weights here; download them. |
| `check <benchmark> -m MODELS` | Push one real batch through each dataset × model pair, then stop. Run it before any long job. |
| `run <benchmark> -m MODELS` | Run a benchmark on this machine: extract embeddings where missing, probe every fold, write `results.json`. |
| `submit <benchmark> -m MODELS --out DIR` | Write HTCondor or SLURM jobs for a whole benchmark (one per dataset × model). |
| `status --out DIR` | How each submitted job is doing. |
| `results <benchmark>` | Mean ± std over the folds, folds done / expected, normalised score. |
| `rescore <benchmark>` | Recompute every result's metrics from the test predictions each fold saved; nothing is fitted. |
| `fetch`, `prepare`, `embed`, `probe`, `hypnogram` | The verbs underneath `run`, for running one step by hand. |

Model selections (`-m`): a checkpoint id (`biot_pretrained`), a family
(`biot`), a comma list, or an alias: `all_fm` (12 EEG foundation models),
`all_ts` (10 time-series foundation models), `all_supervised` (20 supervised
baselines), `all_random` (2 untrained baselines), `all`. Dataset selections
(`--dataset`): `single` (the benchmark's one quick dataset, the default for
`run` and `check`), `full`, or a comma list of dataset names.

Downloads are off by default for every command except `data download`,
`models download` and `fetch`; `--online` allows them for one command.

## Command cheat sheet

Every command, in the order you would use them. The examples use open data
(Sleep-EDF, Siena, BNCI2014_001); `neuroatlas <command> --help` has the
options.

**Is it installed**
```bash
neuroatlas --version                 # neuroatlas 0.1.0
neuroatlas --help                    # every command
```

**Paths**
```bash
neuroatlas config init --data-root ~/eeg/data --cache-root ~/eeg/cache \
    --output-root ~/eeg/results --models-root ~/eeg/models
neuroatlas config show               # every setting, and where its value came from
neuroatlas config path               # where config.yaml is
neuroatlas config set sleep_edf_expanded.data_root /path/to/existing/sleep-edf   # use a copy you already have
neuroatlas config unset sleep_edf_expanded.data_root
neuroatlas config token hf          # Hugging Face token, asked for without being shown: gated weights (REVE)
neuroatlas config token github      # GitHub token: the CoRe-Sleep and SleepTransformer weights (private release assets)
neuroatlas config token nsrr        # NSRR token: the NSRR sleep cohorts (CFS, MESA, MrOS, ...)
```

**What exists**
```bash
neuroatlas list benchmarks           # 12 benchmarks: datasets, task, headline metric
neuroatlas list datasets             # the 43 dataset entries: domain, access, size, benchmarks
neuroatlas list models               # 44 checkpoints: family, group, input rate/window, embedding size
neuroatlas list models --benchmark epilepsy   # the checkpoints one benchmark evaluates
neuroatlas list aliases              # all_fm, all_ts, all_supervised, all_random, all
neuroatlas list tasks                # probe tasks and their presets
neuroatlas show sleep_stage          # what it measures, and the exact embed/probe commands it runs
```

**Datasets**
```bash
neuroatlas data status                                     # every dataset: found / missing / how to get it
neuroatlas data status sleep_stage epilepsy bci_motor_imagery
neuroatlas data download sleep_edf_expanded --dry-run      # what it would fetch, and where
neuroatlas data download sleep_edf_expanded --mirror aws   # 8.1 GB
neuroatlas data download siena                             # 4.7 GB, Zenodo
neuroatlas data download bnci2014_001                      # MOABB (needs the BCI install lines)
neuroatlas data prepare --list                             # optional speed-up caches; none is required
neuroatlas data prepare bonn --dry-run                        # e.g. Bonn's fast-path cache
```
Credentialed datasets (NSRR, TUH) and manual ones print where to get them
and the folder layout to use.

**Weights**
```bash
neuroatlas models status all_fm                            # found / auto / hub (cached) / manual / package missing
neuroatlas models download biot_pretrained,cbramod_pretrained,deepsoz_hem_pretrained
neuroatlas models download all_fm
```

**Check before any long job** (one real batch per dataset × model)
```bash
neuroatlas check sleep_stage -m biot_pretrained
neuroatlas check epilepsy --dataset siena -m cbramod_pretrained
neuroatlas check bci_motor_imagery --dataset bnci2014_001 -m biot_pretrained,cbramod_pretrained
neuroatlas check sleep_stage -m all_fm --dataset full --format json   # machine-readable
```

**Run whole benchmarks** (extract embeddings where missing, probe every fold)
```bash
neuroatlas run sleep_stage -m biot_pretrained --dry-run     # the plan: datasets, folds, runs, data state
neuroatlas run bci_motor_imagery --dataset bnci2014_001 -m biot_pretrained --debug   # fold 0 only
neuroatlas run sleep_stage -m biot_pretrained               # ~40 min
neuroatlas run brain_age -m biot_pretrained                 # reuses the sleep embeddings
neuroatlas run sleep_hypnogram -m biot_pretrained           # from the staging results
neuroatlas run epilepsy --dataset siena -m cbramod_pretrained
neuroatlas run sleep_stage -m biot_pretrained               # again: reuses each fold's saved predictions, fits nothing
neuroatlas run sleep_stage -m biot_pretrained --reprobe     # fit every fold again
```

**The steps underneath, by hand** (`show` prints them for any benchmark,
dataset and model)
```bash
neuroatlas show epilepsy --dataset siena -m cbramod_pretrained
neuroatlas embed --models cbramod_pretrained --dataset siena --set window_s=10 --set stride_s=10 --expected-epoch-seconds 10
neuroatlas probe --models cbramod_pretrained --dataset siena --task seizure_detection --set window_s=10 --set stride_s=10 \
    --probe-type linear --class-weight balanced --tune-c 0.001,0.01,0.1,1,10,100 --selection-metric auprc

neuroatlas embed --models biot_pretrained --dataset sleep_edf_expanded
neuroatlas probe --models biot_pretrained --dataset sleep_edf_expanded --task sleep_staging

neuroatlas embed --models biot_pretrained --dataset bnci2014_001 --pooling mean
neuroatlas probe --models biot_pretrained --dataset bnci2014_001 --pooling mean --set n_folds=loso \
    --probe-type linear --class-weight balanced --tune-c 1.0 --max-iter 1000

neuroatlas hypnogram --datasets sleep_edf_expanded         # finds the staging results by itself
neuroatlas embed --list-datasets                           # every dataset the verbs accept
```

**Results**
```bash
neuroatlas results sleep_stage       # mean ± std over folds, folds done/expected, normalised score
neuroatlas results brain_age
neuroatlas results epilepsy -v       # -v also shows why failed folds failed
neuroatlas results sleep_stage --format csv
neuroatlas rescore sleep_stage       # recompute every result's metrics from its fold's saved test predictions
neuroatlas rescore epilepsy --dataset siena -m cbramod_pretrained
```

**Cluster** (`submit` writes the job files; it submits nothing)
```bash
neuroatlas submit sleep_stage -m all_fm --out runs/sleep_fm                    # HTCondor
neuroatlas submit sleep_stage -m all_fm --out runs/sleep_fm --backend slurm    # SLURM
neuroatlas status --out runs/sleep_fm
```

**Python**
```python
from neuroatlas import api
api.benchmarks()
api.data_status("sleep_stage")
api.check("sleep_stage", models="biot_pretrained")
api.results("sleep_stage")
api.rescore("sleep_stage")           # = rescore
```

With any command: `-v` shows more, `--log run.log` keeps everything, and
`--online` allows downloads (off except in the download commands).

## Test it by hand

One pass through every command, on open data, with the output to expect.
The outputs and timings below are from a fresh install on 2026-10-05 (RTX
4500 Ada, 28 cores shared with other jobs); your numbers should agree to
about 1e-3.

**What you need:** Linux, an NVIDIA GPU (compute capability 7.5 or newer
with the default PyTorch build; a CPU works for `check` but extraction is
slow), about 25 GB of disk (6.5 GB for the environment, 8.1 GB Sleep-EDF,
4.7 GB Siena, 2.1 GB of MOABB data, about 3 GB of embeddings), and about 2
hours, most of it the Sleep-EDF download and its embedding extraction.

**1. Install** (see [Install](#install) for the details), then:

```bash
neuroatlas --version            # neuroatlas 0.1.0
neuroatlas --help               # the command list above
```

**2. Tell it where things go.** Pick four folders; they are created as
needed.

```bash
neuroatlas config init --data-root ~/eeg/data --cache-root ~/eeg/cache \
    --output-root ~/eeg/results --models-root ~/eeg/models
neuroatlas config show
```

`config init` prints `wrote ~/.neuroatlas/config.yaml` and the four roots.
If you already have a dataset somewhere else, point at it instead of
downloading it, e.g. `neuroatlas config set sleep_edf_expanded.data_root
/path/to/sleep-edf`; a misspelt key is refused with a suggestion (exit 2).

**3. Look around.**

```bash
neuroatlas list benchmarks      # 12 benchmarks, their datasets and headline metric
neuroatlas list models          # 44 checkpoints and their families
neuroatlas show sleep_stage     # what the benchmark measures, and the commands `run` executes
```

**4. Get data.**

```bash
neuroatlas data status sleep_stage epilepsy bci_motor_imagery
neuroatlas data download sleep_edf_expanded --mirror aws   # 8.1 GB; 18 min from PhysioNet's AWS copy
neuroatlas data download siena                              # 4.7 GB from Zenodo
neuroatlas data download bnci2014_001                       # MOABB; needs the BCI install lines
neuroatlas data status sleep_edf_expanded siena bnci2014_001
```

The last command should report `found (197/197 files)` for Sleep-EDF and
`found` for the other two. `data status` shows, for every dataset, where it
looked and how `data download` would get it (automatic, with a token, or
instructions for the manual and credentialed ones).

**5. Get weights.**

```bash
neuroatlas models status all_fm
neuroatlas models download biot_pretrained,cbramod_pretrained,deepsoz_hem_pretrained
```

`models status` gives one line per checkpoint: `found`, `auto` (downloads
on request), `hub (cached)`, `manual` (with instructions) or `package
missing` (with the install line). DeepSOZ is GPL-3.0 and is fetched from its
upstream repository, never shipped with the package.

**6. Check before running.** Each pair pushes one real batch through the
model and stops; this takes seconds per pair.

```bash
neuroatlas check sleep_stage -m biot_pretrained
neuroatlas check epilepsy --dataset siena -m cbramod_pretrained
neuroatlas check bci_motor_imagery --dataset bnci2014_001 -m biot_pretrained,cbramod_pretrained
```

The last one ends like this:

```
dataset       model               data                 weights       channel_map  forward           time
bnci2014_001  biot_pretrained     found (18/18 files)  found         none         (64, 256) finite  29.1s
bnci2014_001  cbramod_pretrained  found (18/18 files)  hub (cached)  none         (64, 200) finite  19.8s
2 pairs: 2 ok, 0 error, 0 skipped, 0 n/a (48.9 s)
```

A pair that did not run says why on an indented line under it, starting with
`error:` (it failed), `skipped:` (data or weights missing; a `fix:` line
below gives the command that fetches them) or `n/a:` (the channel map rules
it out). Every other message starts the same way, on stderr: `error:` (the
command or a pair failed or was refused), `warning:` (it goes on, but you
should know) or `note:`, with the remedy on its own `fix:` line:

```
$ neuroatlas check epilepsy --dataset siena -m neurorvq_eeg_pretrained
...
siena    neurorvq_eeg_pretrained  found (41/41 files)  auto     applied (pass-through)  -        0.0s
      skipped: weights not downloaded
        fix: neuroatlas models download neurorvq_eeg_pretrained
```

A long message from a library (a CUDA out-of-memory error, an HTTP error) is
cut to its first sentence, `(-v: full message)`; `-v` and a `--log FILE`
keep it whole. Colour only on a terminal, and never with `NO_COLOR` set.

**7. A one-fold run** (`--debug` = fold 0 only), and a plan without running:

```bash
neuroatlas run bci_motor_imagery --dataset bnci2014_001 -m biot_pretrained --debug   # ~1 min
neuroatlas run epilepsy --dataset siena -m cbramod_pretrained --dry-run
```

Every `run` ends with a line like `5 runs: 5 ok, 0 failed, 0 n/a` and the
folder its results went to.

**8. Full runs.** Embeddings are extracted once per dataset × model and
reused by every fold and by every benchmark that reads the same windows.

```bash
neuroatlas run sleep_stage -m biot_pretrained                    # ~40 min: extraction on the GPU, then 5 probe folds
neuroatlas run brain_age -m biot_pretrained                      # ~1 min: reuses the sleep embeddings
neuroatlas run sleep_hypnogram -m biot_pretrained                # ~20 s: from the staging results
neuroatlas run epilepsy --dataset siena -m cbramod_pretrained    # ~4 min
```

**9. Read the results.**

```bash
neuroatlas results sleep_stage
neuroatlas results brain_age
neuroatlas results epilepsy
neuroatlas rescore sleep_stage                     # seconds: every metric recomputed from the folds' saved test predictions ("rescored, same")
neuroatlas run sleep_stage -m biot_pretrained      # seconds: no fold is fitted again ("note: reused saved predictions for 5 folds")
```

What the fresh install gave (± is the standard deviation over the folds;
for the epilepsy headline it is the SD of the per-fold AUCs):

| Benchmark | Dataset | Model | Metric | Result | Reference |
|---|---|---|---|---|---|
| `sleep_stage` | Sleep-EDF | BIOT | balanced accuracy | 0.657 ± 0.014 (5/5 folds) | 0.657 (the authors' run) |
| `brain_age` | Sleep-EDF SC | BIOT | MAE, years | 11.984 ± 1.690 (5/5) | 11.98 ± 1.69 (paper's brain-age table) |
| `epilepsy` | Siena | CBraMod | event Sens@FA AUC (window AUROC 0.813) | 0.496 ± 0.159 (5/5) | 0.51 ± 0.09, on the paper's recording-level folds (see note) |

Siena's published numbers were drawn on folds of recordings, not patients
(see the Siena entry under `provenance.known_issues` in
`configs/cohorts/siena/cohort.yaml`); `run` uses the patient-level folds
the paper describes, so its Siena numbers are not the published ones.

**10. Reproduce a paper number with a supervised baseline.** The
CoRe-Sleep weights are a release asset of this (private) repository, so the
download needs a GitHub token that can read it:

```bash
neuroatlas config token github                # paste a GitHub token that can read this repository
neuroatlas models download core_sleep_shhs_fold0
neuroatlas run brain_age -m core_sleep_shhs_fold0       # a few minutes: extracts CoRe-Sleep on Sleep-EDF
neuroatlas results brain_age                            # core_sleep_shhs_fold0  10.392 ± 1.074; paper 10.39 ± 1.07
```

**11. Cluster jobs.** `submit` writes the job files and prints the command
that submits them; it submits nothing itself.

```bash
neuroatlas submit sleep_stage -m all_fm --out runs/sleep_fm               # HTCondor; --backend slurm for SLURM
condor_submit runs/sleep_fm/jobs.job                                      # the line submit printed
neuroatlas status --out runs/sleep_fm                                     # done / running / idle / held / failed per job
neuroatlas results sleep_stage                                            # once jobs are done
```

`submit` skips, and lists, the pairs that cannot run here (missing data or
weights, a channel map that rules the model out); `--force` writes the
data- and weights-blocked ones anyway. `--time`, `--memory`, `--gpus` and
`--extra` set the job resources.

**12. From Python.** The same verbs, returning the CLI's JSON rows:

```python
from neuroatlas import api

api.benchmarks()                       # = neuroatlas list benchmarks
api.data_status("sleep_stage")         # = neuroatlas data status sleep_stage
api.check("sleep_stage", models="biot_pretrained")
api.results("sleep_stage")             # = neuroatlas results sleep_stage
```

**13. The steps underneath.** `neuroatlas show <benchmark>` prints the
`embed` and `probe` commands `run` executes (the lines of
`run/default_runs.sh`, the record of every experiment in the paper); you
can run them yourself, e.g. to probe existing embeddings with other
settings.

**If something goes wrong,** rerun the command with `-v --log run.log` and
send the command and `run.log`. Exit codes: 0 done, 1 something failed, 2
the command was refused (a bad option, missing data or weights, a model the
benchmark does not include).

## Run each benchmark

One block per benchmark. Each runs on the benchmark's quick dataset (the
first one named, and the default of `--dataset`); `--dataset full` runs every
dataset whose data is on this machine, and `--dataset a,b` names some.
`neuroatlas show <benchmark>` explains the benchmark and prints the exact
`embed` and `probe` commands. `--debug` probes fold 0 only: a quick first
pass before the full run.

How each dataset is obtained (`neuroatlas data status <benchmark>` says it per
dataset, and `neuroatlas data download <dataset>` does it or prints how):
**open** downloads with `data download`; **NSRR** needs a token
(`neuroatlas config token nsrr`) and the cohort's approval on sleepdata.org;
**TUH** needs the TUH EEG corpus credentials; **manual** prints where to get it
and the folder layout; **authors** are files the authors hold. Point at a copy
you already have with `neuroatlas config set <dataset>.<key> DIR` (`data
status` names the key).

Pick the models once, e.g. `M=biot_pretrained,cbramod_pretrained,reve_pretrained`,
or an alias (`all_fm`, `all_ts`, `all_supervised`); `neuroatlas models download $M`.

### Sleep staging: `sleep_stage`
Five-stage staging of 30 s epochs; balanced accuracy. Quick: **Sleep-EDF
Expanded** (open, 8.1 GB). Open: UCDDB (1.3 GB), HMC (15.7 GB), DOD (58 GB).
NSRR: CFS, HomePAP, MESA, MrOS, STAGES, WSC. Manual: DCSM, ISRUC, MASS,
PhysioNet 2026. SHHS reads the CoRe-Sleep preprocessed version (ask the
authors).
```bash
neuroatlas data download sleep_edf_expanded --mirror aws
neuroatlas check sleep_stage -m $M
neuroatlas run sleep_stage -m $M --debug
neuroatlas run sleep_stage -m $M
neuroatlas results sleep_stage
```

### Hypnograms: `sleep_hypnogram`
Sleep-architecture features (sleep efficiency, WASO, REM latency, ...)
reconstructed from the staging results; Pearson r with the scored hypnogram.
Run `sleep_stage` first on the same datasets.
```bash
neuroatlas run sleep_hypnogram -m $M
neuroatlas results sleep_hypnogram
```

### Brain age: `brain_age`
Age from each subject's mean embedding (ridge regression, nested CV); MAE in
years. It reuses the 30 s sleep embeddings. Quick: **Sleep-EDF Expanded**
(Sleep Cassette subjects). NSRR: CFS, MrOS, WSC. Manual: ISRUC, PhysioNet 2026.
```bash
neuroatlas check brain_age -m $M
neuroatlas run brain_age -m $M
neuroatlas results brain_age
```

### Sleep diagnosis: `sleep_diagnosis`
From one night's mean embedding; balanced accuracy. Quick: **DOD** (open,
58 GB; obstructive sleep apnea vs healthy). Manual: PhysioNet 2026 (cognitive
impairment), ISRUC (pathology).
```bash
neuroatlas data download dod
neuroatlas check sleep_diagnosis -m $M
neuroatlas run sleep_diagnosis -m $M
neuroatlas results sleep_diagnosis
```

### Respiratory events: `sleep_respiratory`
Does a 30 s epoch contain an apnea or hypopnea; macro-F1. Quick: **UCDDB** (open, 1.3 GB).
Manual: PhysioNet 2026.
```bash
neuroatlas data download ucddb --mirror aws
neuroatlas check sleep_respiratory -m $M
neuroatlas run sleep_respiratory -m $M
neuroatlas results sleep_respiratory
```

### Arousals: `sleep_arousal`
Does a 30 s epoch contain a microarousal; macro-F1. Quick: **MASS** (manual,
granted on request). Manual: PhysioNet 2026.
```bash
neuroatlas data download mass                    # prints where to get it and the layout
neuroatlas config set mass.data_root DIR
neuroatlas check sleep_arousal -m $M
neuroatlas run sleep_arousal -m $M
neuroatlas results sleep_arousal
```

### Limb movements: `sleep_limb`
Does a 30 s epoch contain a (periodic) limb movement; macro-F1. **PhysioNet
2026** only (manual, 230 GB).
```bash
neuroatlas config set physionet2026.data_root DIR
neuroatlas check sleep_limb -m $M
neuroatlas run sleep_limb -m $M
neuroatlas results sleep_limb
```

### Epilepsy: `epilepsy`
Is a 10 s window part of a seizure; headline: the event-level Sens@FA AUC
(area under the folds' median curve of seizure-event sensitivity against false
alarms per hour, 0.1-100 FA/h), with window-level AUROC beside it. TUAB, NMT
and Bonn label whole recordings (normal/abnormal): no events, so their headline
is n/a and AUROC is the number to read. Quick: **Siena** (open, 4.5 GB). Open:
CHB-MIT (21.7 GB), Helsinki neonatal (4.3 GB), Bonn (3 MB). TUH: TUSZ, TUAB.
Manual: EPILEPSIAE (licensed), NMT. Authors: SeizeIT1, SeizeIT2.
```bash
neuroatlas data download siena
neuroatlas check epilepsy -m $M
neuroatlas run epilepsy -m $M
neuroatlas results epilepsy
```

### Motor imagery: `bci_motor_imagery`
Leave-one-subject-out; balanced accuracy. All seven come through MOABB (open;
the BCI install lines are needed): quick **BNCI2014_001** (0.8 GB),
BNCI2014_004, BNCI2015_001, Dreyer2023, Liu2024, Shin2017A, Weibo2014.
`--variant per_patch` probes the per-patch embeddings instead of the pooled ones.
```bash
neuroatlas data download bnci2014_001
neuroatlas check bci_motor_imagery -m $M
neuroatlas run bci_motor_imagery -m $M --debug
neuroatlas run bci_motor_imagery -m $M
neuroatlas results bci_motor_imagery
```

### ERP: `bci_erp`
Target vs non-target; leave-one-subject-out; balanced accuracy. MOABB: quick
**BI2013a** (1.4 GB), BI2014a, BNCI2014_008 (0.2 GB), EPFLP300,
ErpCore2021_N170.
```bash
neuroatlas data download bi2013a
neuroatlas check bci_erp -m $M
neuroatlas run bci_erp -m $M
neuroatlas results bci_erp
```

### SSVEP: `bci_ssvep`
Which flickering target the person looks at; leave-one-subject-out; balanced
accuracy. MOABB: quick
**Nakanishi2015** (0.14 GB), Kim2025BetaRange (9 GB).
```bash
neuroatlas data download nakanishi2015
neuroatlas check bci_ssvep -m $M
neuroatlas run bci_ssvep -m $M
neuroatlas results bci_ssvep
```

### Cognitive state: `bci_cognitive`
Emotion (DREAMER valence and arousal) and mental arithmetic (EEGMat, the
arithmetic task); leave-one-subject-out; balanced accuracy.
DREAMER (valence, arousal), EEGMat and the arithmetic task are read from
preprocessed files the authors hold (`data status bci_cognitive` names each
file); ask the authors, then point at them.
```bash
neuroatlas data status bci_cognitive
neuroatlas config set eegmat.preprocessed_path FILE
neuroatlas run bci_cognitive --dataset eegmat -m $M
neuroatlas results bci_cognitive
```

**Everything at once.** `neuroatlas run <benchmark> --dataset full -m all_fm`
runs every dataset you have with every EEG foundation model; on a cluster,
`neuroatlas submit <benchmark> --dataset full -m all_fm --out runs/x` writes
one job per dataset and model, and `neuroatlas status --out runs/x` follows
them.

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

Ten cohorts in the `epilepsy` benchmark, scored by the event-level
Sens@FA AUC (the paper's headline: the area under the folds' median curve of
seizure-event sensitivity against false alarms per hour, 0.1-100 FA/h), with
window-level AUROC, AUPRC and balanced accuracy beside it. Seven are continuously labelled (a label per
10 s window); three label whole recordings as normal or abnormal. Folds are
patient-level, except Bonn, which ships no subject identifiers (its folds
are over clips). AUB-Med is planned: readable, but the paper reports no
result for it.

The sleep-staging sequence models SleepTransformer, SleePyCo and CoRe-Sleep
are not part of the epilepsy benchmark: `-m all` (or `all_supervised`) leaves
them out there, and naming one is refused. The paper's epilepsy figures and
tables do show CoRe-Sleep and SleepTransformer; those cells are not rerun.

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
planned. The last three are not part of the epilepsy benchmark.

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
requirements-fm.txt         # the paper's exact versions (a pip constraints file)
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
  10.42, 10.00, which `results` prints as 10.392 ± 1.074 (the paper's
  table: 10.39 ± 1.07).

## License

MIT; see `LICENSE`. Two vendored components keep their upstream licences:
Seizure-Transformer's architecture, in
`src/neuroatlas/extensions/models/backbones/third_party/seizure_transformer/`
(`LICENSE.upstream`, MIT), and MOABB's Dreyer2023 and Kim2025BetaRange
readers, in `src/neuroatlas/extensions/datasets/dataio/moabb_vendored/`
(`LICENSE.moabb`, BSD-3-Clause).

DeepSOZ-HEM's code and checkpoint are GPL-3.0 and are not part of the
package. `neuroatlas models download deepsoz_hem_pretrained` fetches them,
with their licence, from upstream (github.com/amruth-sn/deepsoz-hem, commit
a7c13bd) into the models root, each file checked against its recorded
SHA-256; the wrapper runs that copy.

## Citation

```bibtex
@article{kontras2026neuroatlas,
  title   = {NeuroAtlas: Benchmarking Foundation Models for Clinical EEG and Brain-Computer Interfaces},
  author  = {Kontras, Konstantinos and Osselaer, Trui and Mouslech, Stylianos G. and
             Karaiskou, Angeliki-Ilektra and Gagliardi, Guido and Strypsteen, Thomas and
             Badiei, Mohammad Hossein and Rani, Anku and Vanmarcke, Maarten and
             Bhagubai, Miguel and Ekbote, Chanakya and Hwang, Jaedong and
             Chatzichristos, Christos and Liang, Paul Pu and De Vos, Maarten},
  journal = {arXiv preprint arXiv:2605.14698},
  year    = {2026}
}
```
