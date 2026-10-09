<!-- Generated from the command-line parsers by `python -m neuroatlas.cli._gendocs`. Do not edit by hand. -->
# Individual steps

`run` executes these steps for you. Call them yourself to try other settings. See [The individual steps](../guide/advanced.md#71-the-individual-steps).

## embed

Extract the embeddings of one dataset with frozen models and save them to the cache. `neuroatlas run` does this for you. Use it directly for a dataset outside the benchmarks, or to change its settings.

Usage: `neuroatlas embed [options]`

```bash
neuroatlas embed --dataset dod -m biot_pretrained,labram_pretrained
neuroatlas embed --dataset hmc -m reve_pretrained --set window_s=30 --set stride_s=30
neuroatlas embed --dataset shhs -m labram_pretrained --embed-chunk 0/4    # chunk 0 of 4
neuroatlas embed --list-datasets --paper-only
```

| Option | Description | Default |
|---|---|---|
| `--dataset DATASET` | Dataset name (see --list-datasets). |  |
| `-m`, `--models MODELS` | Checkpoint ids, families, groups or an alias such as all_fm, separated by commas. | all |
| `--set KEY=VALUE` | Change a dataset setting, as in --set window_s=10. Can be repeated. `neuroatlas embed --dataset NAME --help` lists the settings. |  |
| `--checkpoint ID=PATH` | Load the weights of checkpoint ID from PATH. Can be repeated. |  |
| `--folds FOLDS` | Folds to extract, separated by commas. Most datasets share one set of embeddings across folds. HMC, MESA, STAGES, HomePAP, the four bci_cognitive datasets, TUAB and CHB-MIT when read from an HDF5 file, and runs with a stride other than the window need every fold the probe will read. | the dataset's fold setting, usually 0 |
| `--data-root DIR` | Read the dataset from DIR instead of its configured folder. |  |
| `--batch-size N` | Windows per batch. | the dataset's own |
| `--num-workers N` | Data loader workers. 0 reads in the main process. | the dataset's own, else the CPUs this job may use minus one, at most 16 |
| `--cache-root DIR` | Where to write the embeddings. | the cache_root setting |
| `--limit-batches N` | Stop each extraction after N batches, for a smoke test. The embeddings go to `<cache root>/_limited`, apart from complete ones. |  |
| `--embed-chunk K/N` | Extract only chunk K of N of the subjects, as in 0/4, to split the work across jobs. The chunks are merged when first read. |  |
| `--expected-epoch-seconds S` | Window length in seconds that each checkpoint is told to expect. The epilepsy benchmark passes 10. | the checkpoint's own |
| `--no-recording-norm` | Skip the per-recording normalisation, such as the z-score of REVE. For ablations, not the benchmark protocol. |  |
| `--no-amplitude-scale` | Skip the fixed amplitude scaling, such as x1000 for EEGPT. For ablations, not the benchmark protocol. |  |
| `--pooling {mean,per_patch}` | What to keep for each window. mean averages the patch tokens into one vector, per_patch keeps them all. The two are cached apart. | `mean` |
| `--seed N` | Random seed. | `42` |
| `--dry-run` | Print the resolved settings as JSON and exit. |  |
| `--list-datasets` | List every dataset name and exit. |  |
| `--paper-only` | With --list-datasets, only list the datasets of the paper. |  |

> [!NOTE]
> Run `neuroatlas embed --help` for the list of datasets, tasks and models, and `neuroatlas embed --dataset hmc --help` for the `--set` keys of one dataset.

## probe

Fit probes on extracted embeddings, for any dataset, checkpoint and task. `neuroatlas run` does this for you. Use it directly to try another task or other probe settings.

Usage: `neuroatlas probe [options]`

```bash
neuroatlas probe --dataset isruc -m biot_pretrained
neuroatlas probe --dataset isruc -m biot_pretrained --task sex
neuroatlas probe --dataset mass --task mass_arousal --dry-run
```

| Option | Description | Default |
|---|---|---|
| `--config FILE` | Run a saved JSON config file as written, instead of --dataset. |  |
| `--dataset DATASET` | Dataset name, as listed by `neuroatlas list datasets --all`. |  |
| `-m`, `--models MODELS` | Checkpoint ids, families, groups or an alias such as all_fm, separated by commas. | all |
| `--task TASK` | Task to fit, as listed by `neuroatlas list tasks`. | the dataset's own task |
| `--set KEY=VALUE` | Change a dataset setting. Use the same --set values as the `embed` that made the embeddings. Can be repeated. |  |
| `--checkpoint ID=PATH` | Load the weights of checkpoint ID from PATH. Can be repeated. |  |
| `--folds FOLDS` | Folds to fit, as in 0,1,2 or 0-4. | all |
| `--data-root DIR` | Read the dataset from DIR instead of its configured folder. |  |
| `--batch-size N` | Windows per batch. | the dataset's own |
| `--num-workers N` | Data loader workers. | the dataset's own, else the CPUs this job may use minus one, at most 16 |
| `--probe-type {linear,sklearn_linear,nonlinear}` | Probe model. linear and sklearn_linear are both the logistic regression of the benchmarks. nonlinear is an MLP. | `linear` |
| `--hidden-dims N,N` | Hidden layer sizes for --probe-type nonlinear. | `256,128` |
| `--max-iter N` | Iteration limit of the solver. The BCI benchmarks use 1000, and seizure detection keeps its 500 unless this is given. | `10000` |
| `--class-weight {balanced}` | Weight each class by its inverse frequency. | unweighted, unless the task sets it |
| `--selection-metric METRIC` | Validation metric that picks the seed. Seizure detection also ranks C by it, from auprc (its default), auroc or event_sens_fa_auc. Other logistic regression probes choose C on validation Cohen's kappa. | `macro_f1` |
| `--tune-c C,C,...` | C values to try for the logistic regression, separated by commas. A task without a logistic regression refuses it. |  |
| `--aggregation NAME[,NAME]` | How subject-level tasks combine the windows of a subject, as in mean or mean,mean_std. Each value is a separate run. |  |
| `--seeds N,N,...` | Probe seeds for --seed-mode shared, separated by commas. | `0,1,2` |
| `--seed-mode {fold,shared}` | How probe seeds are chosen. fold uses the fold number as the seed. shared fits each of --seeds on every fold. | `fold` |
| `--pooling {mean,per_patch}` | Which embeddings to probe. It must match the --pooling of `embed`. | `mean` |
| `--seed N` | Random seed. | `42` |
| `--output-root DIR` | Where to write results.json and the fold probes. Each dataset and task gets its own folder. | the output_root setting |
| `--cache-root DIR` | Where to read the embeddings. | the cache_root setting |
| `--reprobe` | Fit every fold again. Without it, a fold with saved predictions from the same settings, weights and embeddings is only rescored. |  |
| `--extract-only` | Only extract the embeddings, as `neuroatlas embed` does. |  |
| `--embed-chunk K/N` | With --extract-only, extract only chunk K of N of the subjects, as in 0/4. |  |
| `--no-recording-norm` | Probe the embeddings made with `embed --no-recording-norm`. |  |
| `--no-amplitude-scale` | Probe the embeddings made with `embed --no-amplitude-scale`. |  |
| `--dry-run` | Print the resolved settings as JSON and exit. |  |
| `--list-tasks` | List the tasks and exit. |  |

> [!NOTE]
> Run `neuroatlas probe --help` for the list of datasets, tasks and models, and `neuroatlas probe --dataset hmc --help` for the `--set` keys of one dataset.

## hypnogram

Rebuild hypnograms from the sleep staging probes and compute 34 sleep features per recording, such as total sleep time and REM latency. Run it after `neuroatlas run sleep_stage`. Without a step, it runs reconstruct and then features.

Usage: `neuroatlas hypnogram [{reconstruct,features}] [options]`

```bash
neuroatlas hypnogram --datasets sleep_edf_expanded
neuroatlas hypnogram --datasets dod mass
```

| Command | Description |
|---|---|
| [`hypnogram reconstruct`](#hypnogram-reconstruct) | Rebuild hypnograms.json from the sleep staging results. |
| [`hypnogram features`](#hypnogram-features) | Compute the sleep features from hypnograms.json. |

| Option | Description | Default |
|---|---|---|
| `--datasets [DATASET ...]` | Datasets to process. | dcsm, dod, isruc, mass, physionet2026, sleep_edf_expanded, ucddb, wsc |
| `--dry-run` | Print what would run, then exit. |  |
| `--results-dir RESULTS_DIR` | Folder with the sleep staging results.json and probes/. | found from --datasets |
| `--models MODELS` | Checkpoint ids, separated by commas. | all |
| `--folds FOLDS` | Fold indices, separated by commas. | all |
| `--splits SPLITS` | Splits to predict, separated by commas. | `test` |
| `--group-by GROUP_BY` | Metadata field to group recordings by, such as subgroup, subset, group or site_id. |  |
| `--compute-metrics` | Also compute classification metrics per group. |  |
| `--output OUTPUT` | Output JSON file. | `<results-dir>/hypnograms.json` |
| `--output-dir OUTPUT_DIR` | Where to write the feature CSV files. | the output root |
| `--no-summary` | Skip the per-feature error summary. |  |

```text
It looks for the sleep staging results of a dataset in these folders, in order:
  <output root>/sleep_stage/<dataset>/          from `neuroatlas run sleep_stage`
  <output root>/sleep_stage/<dataset>/<model>/  from the jobs of `neuroatlas submit`
  <output root>/<dataset>/sleep_staging/        from `neuroatlas probe --task sleep_staging`
--results-dir gives the folder directly. Stages are coded 0=W, 1=N1, 2=N2, 3=N3
and 4=REM, in 30 s epochs.
```

### hypnogram reconstruct

Predict every epoch again with the saved sleep staging probes, and write the predicted and true hypnogram of each recording to hypnograms.json next to the results.

Usage: `neuroatlas hypnogram reconstruct [options]`

```bash
neuroatlas hypnogram reconstruct --datasets dod --group-by group --compute-metrics
```

| Option | Description | Default |
|---|---|---|
| `--results-dir RESULTS_DIR` | Folder with the sleep staging results.json and probes/. | found from --datasets |
| `--models MODELS` | Checkpoint ids, separated by commas. | all |
| `--folds FOLDS` | Fold indices, separated by commas. | all |
| `--splits SPLITS` | Splits to predict, separated by commas. | `test` |
| `--group-by GROUP_BY` | Metadata field to group recordings by, such as subgroup, subset, group or site_id. |  |
| `--compute-metrics` | Also compute classification metrics per group. |  |
| `--output OUTPUT` | Output JSON file. | `<results-dir>/hypnograms.json` |
| `--datasets [DATASET ...]` | Datasets to process. | dcsm, dod, isruc, mass, physionet2026, sleep_edf_expanded, ucddb, wsc |
| `--dry-run` | Print what would run, then exit. |  |

### hypnogram features

Compute 34 sleep features per recording from hypnograms.json, and write them to hypnogram_features.csv with a per-feature error summary.

Usage: `neuroatlas hypnogram features [options]`

```bash
neuroatlas hypnogram features --datasets isruc --no-summary
```

| Option | Description | Default |
|---|---|---|
| `--output-dir OUTPUT_DIR` | Where to write the feature CSV files. | the output root |
| `--no-summary` | Skip the per-feature error summary. |  |
| `--datasets [DATASET ...]` | Datasets to process. | dcsm, dod, isruc, mass, physionet2026, sleep_edf_expanded, ucddb, wsc |
| `--dry-run` | Print what would run, then exit. |  |
