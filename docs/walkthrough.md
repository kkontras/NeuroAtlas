# Walkthrough

This page runs every `neuroatlas` command once, on open data, and shows
what to expect. It takes an afternoon. Most of that time is the Sleep-EDF
download, its extraction and the sleep-staging probes.

Extraction times below are from an RTX 4500 Ada GPU. Probe times are from
4 CPU cores, since the probes run on the CPU. The [user guide](user_guide.md)
covers every command in more depth.

## What you need

- Linux and an NVIDIA GPU with compute capability 7.5 or newer. A CPU is
  enough for `check`, but extraction is slow on it.
- About 25 GB of disk: 6.5 GB for the environment, 8.1 GB for Sleep-EDF
  Expanded, 4.5 GB for Siena, 2.1 GB of MOABB data and about 3 GB of
  embeddings.

## 1. Install

Follow the [README](../README.md#install), including the BCI line for the
MOABB datasets. Then check the command:

```bash
neuroatlas --version
neuroatlas --help
```

From a clone, `--version` prints the version and the commit, such as
`neuroatlas 0.1.0 (1c13d9e)`.

## 2. Choose your folders

Pick one project folder. The tool creates `data`, `cache`, `results` and
`models` inside it.

```bash
neuroatlas config init ~/neuroatlas
neuroatlas config show
```

`config init` prints the four folders it wrote to `~/.neuroatlas/config.yaml`.

If you already have a dataset somewhere else, point the tool at it instead
of downloading it again:

```bash
neuroatlas config set sleep_edf_expanded.data_root /path/to/sleep-edf
```

A misspelt key is refused, and the error gives the corrected command.

## 3. Look around

```bash
neuroatlas list benchmarks
neuroatlas list models
neuroatlas show sleep_stage
```

`list benchmarks` shows the 12 benchmarks with their domain, headline
metric, quick dataset, datasets and variants. The quick dataset is the one
`run` and `check` use by default. Add `--dataset full` to use all of a
benchmark's datasets.

`list models` shows the 44 checkpoints with their family, group, where the
weights come from, and the sampling rate and window each model was built
for. A benchmark cuts its own windows (30 s epochs for sleep, 10 s windows
for epilepsy, the trial for BCI) and resamples them to each model's rate.

`show` explains one benchmark. It says what the headline number is computed
over, what a fold is, what ± means, and which other metrics are reported.
For BCI it also says which variant reproduces which figure of the paper.
Try `neuroatlas show epilepsy` and `neuroatlas show bci_motor_imagery` too.

## 4. Download data

```bash
neuroatlas data status sleep_stage epilepsy bci_motor_imagery
neuroatlas data download sleep_edf_expanded --mirror aws
neuroatlas data download siena
neuroatlas data download bnci2014_001
neuroatlas data status sleep_edf_expanded siena bnci2014_001
```

The first command lists every dataset of the three benchmarks. It shows
where each dataset is served from, whether it is here, and how
`data download` gets it. A dataset that is not here gets a `fix:` line with
the command that fetches it.

Sleep-EDF Expanded (8.1 GB) takes about 18 min from PhysioNet's AWS copy.
Siena (4.5 GB) comes from Zenodo. BNCI2014_001 comes through MOABB.

Each download prints one line per file and ends with one line per dataset,
such as `[1/1] bonn: downloaded, 5 files, 3.2 MB (1s)`. The last command
then reports:

```
dataset             host       state                  download
sleep_edf_expanded  PhysioNet  found (197/197 files)  downloadable
siena               Zenodo     found (41/41 files)    downloadable
bnci2014_001        MOABB      found (18 files)       downloadable

3 datasets: 3 found
```

Sleep-EDF and Siena know how many files a complete copy has. MOABB keeps
no such list, so BNCI2014_001 shows a count only.

## 5. Download weights

```bash
neuroatlas models status biot_pretrained,cbramod_pretrained,deepsoz_hem_pretrained
neuroatlas models download biot_pretrained,cbramod_pretrained,deepsoz_hem_pretrained
```

`models status` shows each checkpoint's state, such as `found`,
`downloadable` or `in Hugging Face cache`. While a checkpoint downloads, a
progress line shows the bytes and the time left. Each checkpoint then ends
in one line:

```
downloading 3 checkpoints
[1/3] biot_pretrained: downloaded, 14 MB (2s)
[2/3] cbramod_pretrained: in Hugging Face cache
[3/3] deepsoz_hem_pretrained: downloaded, 5.6 MB (1s)
3 checkpoints: 3 ready
```

DeepSOZ-HEM is GPL-3.0. Its code and checkpoint are fetched from the
upstream repository and are never part of the package.

## 6. Check before running

`check` loads one batch of real data for each pair of dataset and
checkpoint and pushes it through the model. It fits nothing and writes no
results. Each pair takes seconds to a minute.

```bash
neuroatlas check sleep_stage -m biot_pretrained
neuroatlas check epilepsy --dataset siena -m cbramod_pretrained,biot_pretrained
neuroatlas check bci_motor_imagery --dataset bnci2014_001 -m biot_pretrained
```

The second command prints:

```
dataset  checkpoint          data                 weights                channel map                   forward pass      time
siena    cbramod_pretrained  found (41/41 files)  in Hugging Face cache  applied (labels as recorded)  (64, 200) finite  7s
siena    biot_pretrained     found (41/41 files)  found                  applied                       (64, 256) finite  1s

2 pairs: 2 ok, 0 failed, 0 skipped (8s)
```

`forward pass` is the shape of one batch's output, windows by embedding
size. `finite` means no value is NaN or infinite.

A pair that cannot run says why on a line under its row. `skipped` means
its data or weights are missing, and a `fix:` line gives the command that
fetches them. `ruled out` means the dataset's channel map excludes the
model, so there is nothing to fix. `error` means it ran and failed.

The same rule holds for every message. It starts with `error:`, `warning:`
or `note:`, and the remedy is on its own `fix:` line. A long error from a
library is cut to its first sentence. Add `-v` to see library output and
full messages, and `--log FILE` to keep them.

## 7. Plan, then run one fold

```bash
neuroatlas run epilepsy --dataset siena -m cbramod_pretrained --dry-run
neuroatlas run bci_motor_imagery --dataset bnci2014_001 -m biot_pretrained --debug
```

`--dry-run` runs nothing. It prints the plan in seconds: each dataset, its
folds, how many probe fits, whether the data is here, and the exact
`embed` and `probe` commands.

`--debug` runs the whole pipeline on fold 0 only. It extracts the
embeddings or finds them in the cache, fits the probe and writes
`results.json`. The BCI run takes 1-2 min on a CPU. With the embeddings
already cached, and output sent to a file, it prints:

```
$ neuroatlas embed --dataset bnci2014_001 --set confound_control=true --pooling mean --set n_folds=loso --folds 0 --models biot_pretrained
embedding 1 checkpoint on bnci2014_001: 1 run
[1/1] bnci2014_001 biot_pretrained: embedding
[1/1] bnci2014_001 biot_pretrained: loading the data 22% (2/9 subjects, 0m 09s, ~0m 34s left)
...
[1/1] bnci2014_001 biot_pretrained: already extracted (51s)
embed: 1 ok, 0 failed (cache: ~/neuroatlas/cache)

$ neuroatlas probe --dataset bnci2014_001 --set confound_control=true --pooling mean --set n_folds=loso --probe-type linear --tune-c 0.001,0.01,0.1,1,10,100 --seeds 0 --max-iter 1000 --folds 0 --models biot_pretrained --output-root ~/neuroatlas/results/bci_motor_imagery/bnci2014_001
probing 1 checkpoint on bnci2014_001: 1 run, one line each as it finishes
[1/1] bnci2014_001 biot_pretrained fold 0: ok, bal_acc 0.257 (9s)
probe: 1 ok, 0 failed (results: ~/neuroatlas/results/bci_motor_imagery/bnci2014_001)

1 run (checkpoint x fold): 1 ok, 0 failed
results: ~/neuroatlas/results/bci_motor_imagery; `neuroatlas results bci_motor_imagery` summarises them
```

A run is one probe fit, of one checkpoint on one fold. A `--debug` run is
1 run. A full Sleep-EDF run of one checkpoint is 5. A full BNCI2014_001 run
is 9, one fold per held-out subject.

## 8. Full runs

Embeddings are extracted once per dataset and checkpoint. Every fold, and
every benchmark that reads the same windows, reuses them.

```bash
neuroatlas run sleep_stage -m biot_pretrained
neuroatlas run brain_age -m biot_pretrained
neuroatlas run sleep_hypnogram -m biot_pretrained
neuroatlas run epilepsy --dataset siena -m cbramod_pretrained
```

| Command | Time |
|---|---|
| `sleep_stage` | about 3 h: extraction on the GPU, then five probe folds of about 34 min on 4 CPU cores |
| `brain_age` | about a minute, reusing the sleep embeddings |
| `sleep_hypnogram` | about 20 s, from the staging results |
| `epilepsy` on Siena | about 4 min |

On a terminal, each step shows a progress line from its first second. For
an extraction you see the data loading, the weights loading, then
`embedding 50% (10/20 batches, 1m 49s, ~1m 30s left)` and the cache being
written. A probe fold reads the embeddings, cuts its split, fits and
scores. Sleep staging tries six values of C, and its progress line counts
them without a time estimate, since a large C takes much longer than a
small one.

Each extraction ends in a result line with the number of windows it
embedded, or `already extracted`. Each probe fold ends with its headline
value, under the same label as the `results` column:

```
[1/5] sleep_edf_expanded biot_pretrained fold 0: ok, kappa 0.777 (33m 55s)
```

A fold scored from its saved predictions says `reused`. In a pipe or a
cluster log there is no progress line. Extraction then prints a line per
tenth of its work, and a probe fold prints its result line only.

## 9. Read the results

```bash
neuroatlas results sleep_stage
neuroatlas results brain_age
neuroatlas results epilepsy
neuroatlas results epilepsy -v
```

Each table starts by saying what its numbers are:

```
sleep_stage: Cohen's kappa over 30 s epochs, 5 stages (W, N1, N2, N3, REM), each fold's test subjects pooled, unscored epochs left out; higher is better; chance 0
folds: 5 subject-level folds; the probe's C is chosen from 0.001-100 on validation Cohen's kappa, unweighted loss; ± = population SD over the folds
also: bal_acc = balanced accuracy, macro_F1 = macro-F1
dataset             model            kappa  ±      folds  bal_acc  macro_F1
sleep_edf_expanded  biot_pretrained  0.758  0.016  5/5    0.669    0.675
```

The first number after the model is the headline, the paper's metric for
that benchmark. `folds` is the folds done out of the protocol's folds. It
reads `1/5` after a `--debug` run, and `LOSO 1/9` on a
leave-one-subject-out dataset. Where a guess scores differently on each
dataset, a `chance` column gives that level. For BCI it is 1 over the
number of classes.

`results epilepsy -v` adds each fold's value and the C it chose. Each value
is that fold's own Sens@FA AUC. The headline is the AUC of the folds'
median curve, so it is not the mean of the five.

### Saved predictions

```bash
neuroatlas rescore sleep_stage
neuroatlas run sleep_stage -m biot_pretrained
```

Every probe saves its fold's test predictions. `rescore` recomputes every
metric from them in seconds. Running `sleep_stage` again also takes
seconds, because each fold's saved predictions are reused and nothing is
fitted.

## 10. A supervised baseline

The supervised baselines run like any other model. CoRe-Sleep on brain age:

```bash
neuroatlas models download core_sleep_shhs_fold0
neuroatlas run brain_age -m core_sleep_shhs_fold0
neuroatlas results brain_age
```

The run takes a few minutes, most of it extracting CoRe-Sleep on Sleep-EDF.
`results` then shows `core_sleep_shhs_fold0` at 10.392 ± 1.074 years.

## 11. Cluster jobs

```bash
neuroatlas submit sleep_stage -m all_fm --out runs/sleep_fm
condor_submit runs/sleep_fm/jobs.job
neuroatlas status --out runs/sleep_fm
neuroatlas results sleep_stage
```

`submit` writes the job files and prints the command that queues them. It
queues nothing itself. Use `--backend slurm` for SLURM. `status` shows each
job as done, running, idle, held or failed. Run `results` once the jobs are
done.

`submit` writes no job for a pair that cannot run here. It counts them as
skipped (missing data or weights, with a `fix:` line) or ruled out by the
channel map. Add `-v` to list each pair, and `--force` to write the
skipped ones anyway. `--time`, `--memory`, `--gpus` and `--extra` set the
job resources.

## 12. From Python

```python
from neuroatlas import api

api.benchmarks()
api.data_status("sleep_stage")
api.check("sleep_stage", "biot_pretrained")
api.results("sleep_stage")
```

Each call returns a pandas DataFrame with the rows of the command's
`--format json`.

## 13. The steps underneath

`neuroatlas show <benchmark>` prints the `embed` and `probe` commands that
`run` executes. They are the lines of `run/default_runs.sh`, the record of
every experiment in the paper. Run them yourself to probe existing
embeddings with other settings.

## If something goes wrong

Run the command again with `-v --log run.log`. Then open an issue at
https://github.com/kkontras/NeuroAtlas/issues with the command and
`run.log` attached.

Exit codes are 0 when the command succeeded, 1 when something failed, and 2
when the command was refused, for example for a bad option, missing data
or weights, or a model the benchmark does not include.
