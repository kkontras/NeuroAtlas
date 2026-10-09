# 7. Advanced use

`run` executes steps that you can also call yourself: `embed`, `probe` and, for hypnograms,
`hypnogram`. This chapter covers those steps, how folds are made, what is cached and reused, the
environment variables the tool reads, and the Python API.

## 7.1 The individual steps

`neuroatlas show <benchmark>` prints the `embed` and `probe` commands that `run` executes. They
are the lines of `run/default_runs.sh`, the record of every experiment in the paper. Run them
yourself to probe embeddings with other settings. For example, this extracts ISRUC's embeddings
with BIOT once, then fits the dataset's own task and the `sex` task on them:

```bash
neuroatlas embed --dataset isruc -m biot_pretrained
neuroatlas probe --dataset isruc -m biot_pretrained
neuroatlas probe --dataset isruc -m biot_pretrained --task sex
```

`embed` extracts embeddings of one dataset with frozen models. `probe` fits probes on them, for
any dataset, checkpoint and task. `hypnogram` rebuilds hypnograms from the sleep staging probes and
computes the sleep features, as in `neuroatlas hypnogram --datasets sleep_edf_expanded`. The
[command reference](../reference/individual-steps.md) lists their options.

`embed` and `probe` know nothing about benchmarks. They run whatever `--models` names, including
the checkpoints a benchmark leaves out. `run` picks the right task for each benchmark, so you only
need tasks when you call `probe` yourself. `neuroatlas list tasks` shows the 11 tasks the
benchmarks run. Add `-v` to see all 20.

A flag that a task does not use is refused before anything runs, never silently ignored. For
example, the diagnosis probe always fits the published unweighted logistic regression at C = 1:

```text
$ neuroatlas probe --dataset dod --task osa --aggregation mean --class-weight balanced --tune-c 1.0 --models biot_pretrained
error: the osa task does not apply --class-weight or --tune-c: it fits an unweighted logistic regression at C = 1 on each subject's mean embedding (the published diagnosis probe)
  fix: neuroatlas probe --dataset dod --task osa --aggregation mean --models biot_pretrained
```

## 7.2 Folds

A fold never puts one subject on both sides of the split. Bonn has no subject identifiers, so its
folds are over clips. Most datasets have five folds, and BCI uses one fold per subject (LOSO).
SHHS's five folds are over its 8444 recordings, with a participant's two visits in the same fold.
Brain age has its own folds ([Brain age](benchmarks.md#43-brain-age)).

Where the paper's split is a file, it ships with the package and `run` reads it. This covers
CHB-MIT, SeizeIT1, SeizeIT2, EPILEPSIAE, TUSZ and most sleep datasets. The other datasets derive
their folds with a seeded splitter, which gives the same folds on every machine.

On the epilepsy datasets, `embed` and `probe` accept `--set folds_manifest=none` to derive folds
with the dataset's own splitter instead of the shipped file. `--set strict_folds=false` runs on
the subjects that the data and the file share when they differ. Both change the benchmark's
folds.

## 7.3 Caches

Extraction is the expensive part, so it happens once. For most datasets there is one cache per
dataset, checkpoint and windowing, in `<cache root>/<dataset>/<checkpoint>/all/<key>/`. It holds
every window once, and the folds are assigned when probing. Every fold, probe and benchmark that
reads the same windows uses it. `--debug` extracts the whole dataset, so the full run reuses it,
and `brain_age` reuses the 30 s embeddings of `sleep_stage`.

Some datasets cache each fold and split separately, in
`<cache root>/<dataset>/<checkpoint>/{train,val,test}/<key>/`:

- HMC, MESA, STAGES and HomePAP
- the four `bci_cognitive` datasets
- TUAB after `data prepare tuab`
- CHB-MIT with `--set backend=hdf5`
- any run whose stride differs from its window

For these, `--debug` extracts fold 0 and the full run extracts the other folds. TUSZ joins this
group only with `--set split_mode=official`, the dataset's own train, dev and eval split. By
default it uses the paper's five patient-level folds over all 675 patients from one cache.

The cache key changes with anything that changes the windows: window length, stride, channels,
preprocessing, or a non-default epilepsy montage. Probe settings do not change it. If `probe`
finds no cache under its key but one under another key, its error lists the settings that differ.

> [!WARNING]
> Brain age needs the subjects' ages in the cache. A Sleep-EDF cache built without `xlrd`, or an
> ISRUC cache built without `openpyxl`, holds no ages. `brain_age` then stops, names the cache
> folder, and asks you to delete it and run again. Installing the package afterwards does not
> repair the cache.

HomePAP, MESA and STAGES keep no ages in the cache: `brain_age` reads them from the study's NSRR
table each time it probes.

## 7.4 Saved predictions are reused

Every probe saves each fold's test predictions. A second `run` or `probe` with nothing changed
does not fit the fold again. It recomputes the metrics from the saved file. Two Sleep-EDF folds
that took 26 min to an hour each to fit were scored again in 8 s:

```text
$ neuroatlas run sleep_stage -m biot_pretrained --folds 0,1
...
[1/2] sleep_edf_expanded biot_pretrained fold 0: ok, reused, kappa 0.777 (0s)
[2/2] sleep_edf_expanded biot_pretrained fold 1: ok, reused, kappa 0.765 (0s)
probe: 2 ok, 0 failed (results: <output root>/sleep_stage/sleep_edf_expanded)

2 runs (checkpoint x fold): 2 ok, 0 failed; results.json also keeps 3 results of earlier runs
note: 2 folds scored from their saved predictions (same settings, weights and embeddings), not fitted again; --reprobe fits them again
  fix: neuroatlas run sleep_stage -m biot_pretrained --folds 0,1 --reprobe
```

BCI saves less time. The embed step still loads every MOABB recording to find its cache (34 s for
BNCI2014_001) before the probe reuses the fold in about 2 s.

A fold's saved predictions are reused when all of these match:

- the probe, task and dataset settings, including the fold
- the seeds, `--pooling` and the weights file
- the embeddings, by the size and modification time of the cache files

Extracting the embeddings again therefore means a new probe. Weights are compared by their file
path, not their content. A change in how a metric is computed takes effect, because the metrics
are recomputed. A change in how a probe is fitted is not detected, so run with `--reprobe` after
one. When saved predictions exist but are not used, a `note:` says why.

`--reprobe` on `run`, `probe` or `submit` fits every fold again and overwrites its files.

## 7.5 Environment variables

| Variable | Effect |
|---|---|
| `NEUROATLAS_HOME` | Folder of `config.yaml` and the token files. The default is `~/.neuroatlas`. |
| `EEG_DATA_ROOT`, `EEG_CACHE_ROOT`, `NEUROATLAS_OUTPUT_ROOT`, `NEUROATLAS_MODELS_ROOT` | Override the `data_root`, `cache_root`, `output_root` and `models_root` settings ([The project folder](configuration.md#11-the-project-folder)). |
| `MNE_DATA` | The MOABB folder ([MOABB datasets](data.md#24-moabb-datasets)). |
| `NSRR_TOKEN` | The NSRR token. |
| `HF_TOKEN`, `NEUROATLAS_HF_TOKEN_FILE` | The Hugging Face token, or the file that holds it. |
| `GITHUB_TOKEN`, `GH_TOKEN` | The GitHub token. |
| `HF_HUB_CACHE`, `HF_HOME` | The Hugging Face cache, for the models that are not kept under the models root. |
| `NO_COLOR` | No colours, also on a terminal. |

`submit` copies the folder variables, `MNE_DATA`, `HF_HOME` and `PYTHONPATH` into every job
script when they are set, never a token.

## 7.6 Using it from Python

The main commands are also Python functions, in `neuroatlas.api`:

```python
from neuroatlas import api

api.benchmarks()                                     # list benchmarks
api.models("all_fm")                                 # models status all_fm
api.data_status("sleep_stage")                       # data status
api.plan("sleep_stage", "all_fm", datasets="full")   # run --dry-run
api.check("sleep_stage", "biot_pretrained")          # check
api.run_benchmark("sleep_stage", "biot_pretrained", debug=True)
api.results("sleep_stage")                           # results
api.rescore("sleep_stage")                           # rescore
```

Each function returns a pandas DataFrame with the columns of the command's `--format json`.
Missing numbers are `None`, or NaN in a float column. `results()` takes `variant=`.
`run_benchmark()` returns the results of that call only and takes `debug=`, `limit_batches=`,
`cache_root=`, `output_root=`, `online=` and `reprobe=`. `rescore()` takes `datasets=`,
`models=`, `output_root=` and `variant=`.

Importing `neuroatlas.api` reads the same settings file and does not import torch. No API function
downloads. `check()` and `run_benchmark()` run offline unless you pass `online=True` to
`run_benchmark()`. The API leaves `HF_HUB_OFFLINE` alone, because the Hugging Face library reads it
once per session. The weights check before extraction still refuses a hub model that is not
cached.
