# Command reference

This page lists every `neuroatlas` command and its options. Run
`neuroatlas <command> --help` to see the same text in the terminal. The
[user guide](user_guide.md) shows how the commands fit together.

This page is generated from the command-line parsers by
`python -m neuroatlas.cli._gendocs`. Do not edit it by hand.

## Global options

These options work with every command, before or after the command name.

Usage: `neuroatlas [-v] [--log FILE] [--online] <command> [options]`

| Option | Description |
|---|---|
| `-v`, `--verbose` | Show more detail, library log lines and full tracebacks. |
| `--log FILE` | Also write all output, log lines and tracebacks to FILE. |
| `--online` | Allow downloads. Only the commands that download are online by default. |
| `-V`, `--version` | Print the version, and the commit when run from a git clone. |

Every command exits with status 0 when it succeeds, 1 when a run fails and 2 when the command line is wrong.

## Commands

| Group | Command | Description |
|---|---|---|
| Setup | [`config`](#config) | Set where data, caches, results and model weights are stored. |
| Explore | [`list`](#list) | List the benchmarks, datasets, models, aliases or tasks. |
|  | [`show`](#show) | Describe a benchmark and print the commands it runs. |
| Data and weights | [`data`](#data) | Check, download and prepare datasets. |
|  | [`models`](#models) | Check and download model weights. |
| Run | [`check`](#check) | Test each dataset and model on one batch before a long run. |
|  | [`run`](#run) | Run a benchmark on this machine. |
|  | [`submit`](#submit) | Write HTCondor or SLURM jobs for a benchmark. |
|  | [`status`](#status) | Show the state of the jobs that `submit` wrote. |
| Results | [`results`](#results) | Summarise a benchmark's results. |
|  | [`rescore`](#rescore) | Recompute the metrics from saved test predictions. |
| Individual steps | [`embed`](#embed) | Extract embeddings of one dataset with frozen models. |
|  | [`probe`](#probe) | Fit probes on extracted embeddings. |
|  | [`hypnogram`](#hypnogram) | Compute hypnograms and sleep features from sleep staging results. |

## Setup

### config

Set where NeuroAtlas reads datasets and writes caches, results and model weights. The settings are saved in ~/.neuroatlas/config.yaml, or in $NEUROATLAS_HOME/config.yaml when that variable is set.

Usage: `neuroatlas config <action>`

| Command | Description |
|---|---|
| [`config init`](#config-init) | Write the settings file. |
| [`config show`](#config-show) | Print each setting, whether its folder exists and where the value comes from. |
| [`config set`](#config-set) | Change one setting. |
| [`config unset`](#config-unset) | Remove one setting, so it goes back to its default. |
| [`config path`](#config-path) | Print the path of the settings file. |
| [`config token`](#config-token) | Save a Hugging Face, GitHub or NSRR token. |

#### config init

Write the settings file. With a project folder DIR, everything goes to DIR/data, DIR/cache, DIR/results and DIR/models. A `--*-root` option moves one of them.

Usage: `neuroatlas config init [DIR] [options]`

```bash
neuroatlas config init ~/neuroatlas
neuroatlas config init ~/neuroatlas --data-root /data/eeg    # datasets elsewhere
```

| Option | Description | Default |
|---|---|---|
| `DIR` | Project folder. Its sub-folders are created when needed. |  |
| `--data-root DIR` | Folder with the raw datasets, one sub-folder per dataset. | DIR/data |
| `--cache-root DIR` | Folder for saved embeddings and prepared datasets, which can grow large. | DIR/cache, else $NEUROATLAS_HOME/artifacts/embedding_cache |
| `--output-root DIR` | Folder for the probe results. | DIR/results, else $NEUROATLAS_HOME/artifacts/benchmarks |
| `--models-root DIR` | Folder for model weights. | DIR/models, else $NEUROATLAS_HOME/artifacts/models |
| `--dataset-path DATASET.KEY=PATH` | Folder of a dataset kept elsewhere, as in ucddb.data_root=/mnt/ucddb. Can be repeated. |  |
| `--force` | Overwrite an existing settings file. |  |

#### config show

Print each setting, whether its folder exists and where the value comes from.

Usage: `neuroatlas config show`

```bash
neuroatlas config show
```

#### config set

Change one setting. KEY is data_root, cache_root, output_root, models_root or a dataset folder such as chbmit.bids_root. `neuroatlas data status DATASET` shows the key a dataset reads.

Usage: `neuroatlas config set key value`

```bash
neuroatlas config set ucddb.data_root /mnt/ucddb
neuroatlas config set cache_root /scratch/neuroatlas_cache
```

| Option | Description | Default |
|---|---|---|
| `key` | Setting name, such as data_root or ucddb.data_root. |  |
| `value` | New value, usually a folder. |  |

#### config unset

Remove one setting, so it goes back to its default.

Usage: `neuroatlas config unset key`

```bash
neuroatlas config unset ucddb.data_root
```

| Option | Description | Default |
|---|---|---|
| `key` | Setting name. |  |

#### config path

Print the path of the settings file.

Usage: `neuroatlas config path`

```bash
neuroatlas config path
```

#### config token

Save an access token for Hugging Face (hf), GitHub (github) or NSRR (nsrr). Type it when asked, or pipe it in as in `neuroatlas config token hf < file`. It is saved to `$NEUROATLAS_HOME/<name>_token`, readable only by you, and never printed.

Usage: `neuroatlas config token {github,hf,nsrr} [options]`

```bash
neuroatlas config token nsrr
neuroatlas config token hf --remove
```

| Option | Description | Default |
|---|---|---|
| `{github,hf,nsrr}` | hf for gated Hugging Face repositories, github when GitHub refuses a download, or nsrr for the NSRR sleep datasets. |  |
| `--remove` | Delete the saved token. |  |

## Explore

### list

List the benchmarks, datasets, models, aliases or tasks. This reads only the tables that ship with NeuroAtlas. Use `neuroatlas data status` and `neuroatlas models status` to see what is on this machine.

Usage: `neuroatlas list <what>`

| Command | Description |
|---|---|
| [`list benchmarks`](#list-benchmarks) | List each benchmark with its task, datasets and headline metric. |
| [`list datasets`](#list-datasets) | List the paper's datasets with their domain, access, size and benchmarks. |
| [`list models`](#list-models) | List the checkpoints with their family, group, input rate, window and embedding size. |
| [`list aliases`](#list-aliases) | List the names that select groups of checkpoints with -m. |
| [`list tasks`](#list-tasks) | List the probe tasks and the settings each benchmark runs them with. |

#### list benchmarks

List each benchmark with its task, datasets and headline metric.

Usage: `neuroatlas list benchmarks [options]`

```bash
neuroatlas list benchmarks
neuroatlas list benchmarks -v
```

| Option | Description | Default |
|---|---|---|
| `--grep TEXT` | Only show rows that mention TEXT (case-insensitive). |  |
| `-v`, `--verbose` | Also show how to get each dataset and the models a benchmark leaves out. |  |
| `--format {table,csv,md,json}` | Output format. In csv, md and json the notes under a row go to a `note` column. | `table` |

#### list datasets

List the paper's datasets with their domain, access, size and benchmarks.

Usage: `neuroatlas list datasets [options]`

```bash
neuroatlas list datasets --grep sleep
neuroatlas list datasets --all --format csv
```

| Option | Description | Default |
|---|---|---|
| `--grep TEXT` | Only show rows that mention TEXT (case-insensitive). |  |
| `-v`, `--verbose` | Also show the web page or MOABB name of each dataset. |  |
| `--format {table,csv,md,json}` | Output format. In csv, md and json the notes under a row go to a `note` column. | `table` |
| `--all` | Also list the datasets that can be read but are not in the paper. |  |

#### list models

List the checkpoints with their family, group, input rate, window and embedding size.

Usage: `neuroatlas list models [selector] [options]`

```bash
neuroatlas list models all_fm
neuroatlas list models --benchmark epilepsy
```

| Option | Description | Default |
|---|---|---|
| `selector` | Only show these checkpoints, given as an alias, group, family or ids such as all_fm, baseline or reve. |  |
| `--grep TEXT` | Only show rows that mention TEXT (case-insensitive). |  |
| `-v`, `--verbose` | Also show where the weights of each checkpoint come from. |  |
| `--format {table,csv,md,json}` | Output format. In csv, md and json the notes under a row go to a `note` column. | `table` |
| `--benchmark NAME` | Leave out the models this benchmark does not evaluate. |  |

#### list aliases

List the names that select groups of checkpoints with -m.

Usage: `neuroatlas list aliases [options]`

```bash
neuroatlas list aliases -v
```

| Option | Description | Default |
|---|---|---|
| `--grep TEXT` | Only show rows that mention TEXT (case-insensitive). |  |
| `-v`, `--verbose` | Also list the checkpoints of each alias. |  |
| `--format {table,csv,md,json}` | Output format. In csv, md and json the notes under a row go to a `note` column. | `table` |

#### list tasks

List the probe tasks and the settings each benchmark runs them with.

Usage: `neuroatlas list tasks [options]`

```bash
neuroatlas list tasks
```

| Option | Description | Default |
|---|---|---|
| `--grep TEXT` | Only show rows that mention TEXT (case-insensitive). |  |
| `-v`, `--verbose` | Also list the tasks no benchmark runs, and the kind of each. |  |
| `--format {table,csv,md,json}` | Output format. In csv, md and json the notes under a row go to a `note` column. | `table` |

### show

Describe one benchmark and print the `embed` and `probe` commands that `run` executes for it. Use it to see the metrics, folds, datasets and variants of a benchmark.

Usage: `neuroatlas show benchmark [options]`

```bash
neuroatlas show sleep_stage
neuroatlas show epilepsy --dataset single -m cbramod_pretrained
```

| Option | Description | Default |
|---|---|---|
| `benchmark` | Benchmark name, as listed by `neuroatlas list benchmarks`. |  |
| `--dataset single\|full\|NAMES` | Datasets to use. Give single for the benchmark's quick dataset, full for all its datasets, or dataset names separated by commas. | `full` |
| `--variant VARIANT` | Benchmark variant to describe. | `default` |
| `-m`, `--models MODELS` | Put this model selection into the printed commands. |  |

## Data and weights

### data

Check which datasets are on this machine, download them, and build optional faster-to-read copies.

Usage: `neuroatlas data <action>`

| Command | Description |
|---|---|
| [`data status`](#data-status) | Show which datasets are on this machine. |
| [`data download`](#data-download) | Download datasets, or print how to get them. |
| [`data prepare`](#data-prepare) | Build the files a bci_cognitive dataset is read from, or a faster copy of an epilepsy dataset. |

#### data status

Show which datasets are on this machine, where they were looked for and how to get the missing ones.

Usage: `neuroatlas data status [targets ...] [options]`

```bash
neuroatlas data status
neuroatlas data status sleep_stage
neuroatlas data status siena -v
```

| Option | Description | Default |
|---|---|---|
| `targets` | Benchmarks, domains (epilepsy, sleep, brain_age, bci or all) or dataset names, separated by spaces or commas. | all |
| `-v`, `--verbose` | Add columns for each path, the setting that moves it and the channel map, and explain every value. |  |
| `--format {table,csv,md,json}` | Output format. In csv, md and json the notes under a row go to a `note` column. | `table` |

#### data download

Download datasets into the folder each one is read from. MOABB datasets go to $MNE_DATA. For datasets that need an account or a request, it prints where to get them.

Usage: `neuroatlas data download datasets [datasets ...] [options]`

```bash
neuroatlas data download sleep_edf_expanded --mirror aws
neuroatlas data download siena --first 5    # a quick test
neuroatlas data download cfs --dry-run
```

| Option | Description | Default |
|---|---|---|
| `datasets` | Dataset names, as listed by `neuroatlas list datasets`. |  |
| `--dry-run` | Print the plan and run every check (token, disk space, tools), but download nothing. |  |
| `--mirror {physionet,aws}` | Where to get PhysioNet datasets. aws is PhysioNet's open-data copy on AWS, which is usually much faster. | `physionet` |
| `--keep-archive` | Keep downloaded .zip files after unpacking them. |  |
| `--first N` | Download only the first N recordings, for a quick test. `check` works on them, but `run` needs the whole dataset. Works for PhysioNet, NSRR and file-by-file Zenodo downloads. |  |

#### data prepare

Build the two files that dreamer_valence, dreamer_arousal, eegmat or arithmetic_task are read from. Run it once before `run`. For bonn, epilepsiae, sz1, tuab and tusz it builds an optional faster-to-read copy. Download the raw data first. The files go to `<cache root>/prepared`.

Usage: `neuroatlas data prepare [dataset] [options]`

```bash
neuroatlas data prepare --list
neuroatlas data prepare bonn --dry-run
neuroatlas data prepare tusz --shard 0/8
```

| Option | Description | Default |
|---|---|---|
| `dataset` | Dataset name. --list shows the datasets with a build step. |  |
| `--set KEY=VALUE` | Change a dataset setting, as in raw_root=/data. |  |
| `--dest PATH` | Where to write the copy. | `<cache root>/prepared` |
| `--shard K/N` | Build only shard K of N, to split the work across cluster jobs. |  |
| `--dry-run` | Print what would be built, without building it. |  |
| `--list` | List the datasets that have a build step. |  |

### models

Check which model weights are on this machine, and download the missing ones.

Usage: `neuroatlas models <action>`

| Command | Description |
|---|---|
| [`models status`](#models-status) | Show which checkpoints have their weights here. |
| [`models download`](#models-download) | Download the weights a selection is missing. |

#### models status

Show which checkpoints have their weights on this machine. It does not use the network.

Usage: `neuroatlas models status [selector] [options]`

```bash
neuroatlas models status
neuroatlas models status all_fm -v
```

| Option | Description | Default |
|---|---|---|
| `selector` | Checkpoints to check, given as an alias, group, family or ids. | every checkpoint |
| `-m`, `--models SELECTOR` | The same selection, given with -m as for `run`. |  |
| `-v`, `--verbose` | Show where each file is expected, and explain every state. |  |
| `--format {table,csv,md,json}` | Output format. In csv, md and json the notes under a row go to a `note` column. | `table` |

#### models download

Download the weights that a selection of checkpoints is missing. It exits with status 1 if one could not be downloaded.

Usage: `neuroatlas models download [selector] [options]`

```bash
neuroatlas models download biot_pretrained
neuroatlas models download all_fm
```

| Option | Description | Default |
|---|---|---|
| `selector` | Checkpoints to download, given as an alias, group, family or ids such as all_fm. |  |
| `-m`, `--models SELECTOR` | The same selection, given with -m as for `run`. |  |

`neuroatlas fetch` and `neuroatlas prepare` are other names for `neuroatlas data download` and `neuroatlas data prepare`.

## Run

### check

Run one batch of real data through each pair of dataset and checkpoint, then stop. Use it before a long run to catch missing data, missing weights and channel problems. It exits with status 1 if a pair fails or if no pair could be checked.

Usage: `neuroatlas check benchmark -m MODELS [options]`

```bash
neuroatlas check sleep_stage -m biot_pretrained
neuroatlas check epilepsy -m all_fm --dataset full
```

| Option | Description | Default |
|---|---|---|
| `benchmark` | Benchmark name, as listed by `neuroatlas list benchmarks`. |  |
| `-m`, `--models MODELS` | Checkpoint ids, families, groups or an alias such as all_fm, separated by commas. Put `-` before a name to remove it, as in all_fm,-reve. An alias or group skips the models a benchmark does not evaluate. |  |
| `--dataset single\|full\|NAMES` | Datasets to use. Give single for the benchmark's quick dataset, full for all its datasets, or dataset names separated by commas. | `single` |
| `--variant VARIANT` | Benchmark variant. `neuroatlas show BENCHMARK` lists the variants. | `default` |
| `--num-workers N` | Data loader workers. 0 reads in the main process, which is fastest for a single batch. | 0 |
| `--strict` | Also exit with status 1 if a pair was skipped or invalid. |  |
| `--format {table,csv,md,json}` | Output format. In csv, md and json the notes under a row go to a `note` column. | `table` |

### run

Run a benchmark on this machine. For each dataset it extracts the embeddings that are not cached yet, fits the probes on every fold and writes the results to `<output root>/<benchmark>/<dataset>/`.

Usage: `neuroatlas run benchmark -m MODELS [options]`

```bash
neuroatlas run sleep_stage -m biot_pretrained --debug    # fold 0 only
neuroatlas run sleep_stage -m biot_pretrained            # all folds
neuroatlas run epilepsy -m all_fm --dataset full --dry-run
neuroatlas run bci_motor_imagery -m all_fm --variant token_flattening
```

| Option | Description | Default |
|---|---|---|
| `benchmark` | Benchmark name, as listed by `neuroatlas list benchmarks`. |  |
| `-m`, `--models MODELS` | Checkpoint ids, families, groups or an alias such as all_fm, separated by commas. Put `-` before a name to remove it, as in all_fm,-reve. An alias or group skips the models a benchmark does not evaluate. |  |
| `--dataset single\|full\|NAMES` | Datasets to use. Give single for the benchmark's quick dataset, full for all its datasets, or dataset names separated by commas. | `single` |
| `--variant VARIANT` | Benchmark variant. `neuroatlas show BENCHMARK` lists the variants. | `default` |
| `--dry-run` | Print the datasets, checkpoints, folds and runs, and whether the data is here, then stop. |  |
| `--debug` | Run fold 0 only, as a quick test. Most datasets still extract the embeddings of every fold, which a later full run reuses. HMC, MESA, STAGES, HomePAP, the four bci_cognitive datasets, TUAB and CHB-MIT when read from an HDF5 file, and runs with a stride other than the window extract fold 0 only. |  |
| `--folds LIST` | Only run these folds, as in 0,1 or 0-4. | all |
| `--limit-batches N` | Stop each extraction after N batches, for a smoke test. Its embeddings and results go to `_limited` folders under the cache and output roots. A fold may end up with no test subjects. |  |
| `--skip-embed` | Only fit the probes. Fail where embeddings are missing. |  |
| `--reprobe` | Fit every fold again. Without it, a fold with saved predictions from the same settings, weights and embeddings is only rescored. |  |
| `--num-workers N` | Data loader workers for extraction. Some datasets set their own. | the CPUs this job may use minus one, at most 16 |
| `--cache-root DIR` | Where to save and read embeddings. | the cache_root setting |
| `--output-root DIR` | Where to write results. | the output_root setting |
| `--checkpoint-override ID.checkpoint_path=PATH` | Load the weights of checkpoint ID from PATH, as in biot_pretrained.checkpoint_path=/my/weights.ckpt. ID must be in the -m selection. Needs its own --cache-root, because embeddings are cached by checkpoint id. |  |
| `--per-model-output` | Write results to `<dataset>/<checkpoint>/`, as the jobs of `submit` do. |  |
| `--allow-partial` | Run on a dataset that `data status` reports as partial or empty. Without it, run refuses, because the results would cover only some subjects. |  |
| `--format {table,csv,md,json}` | Output format. In csv, md and json the notes under a row go to a `note` column. | `table` |

### submit

Write one job per dataset and checkpoint, and an HTCondor or SLURM file that queues them. Nothing is queued until you run the command printed on the last line. Pairs whose data or weights are missing on this machine, or that the channel map rules out, get no job.

Usage: `neuroatlas submit benchmark -m MODELS --out DIR [options]`

```bash
neuroatlas submit sleep_stage -m all_fm --out jobs/sleep_stage
neuroatlas submit sleep_stage -m all_fm --out jobs/sleep_stage --mode retry
neuroatlas submit epilepsy -m all_fm --out jobs/epilepsy --backend slurm --partition gpu --time 12:00:00
```

| Option | Description | Default |
|---|---|---|
| `benchmark` | Benchmark name, as listed by `neuroatlas list benchmarks`. |  |
| `-m`, `--models MODELS` | Checkpoint ids, families, groups or an alias such as all_fm, separated by commas. Put `-` before a name to remove it, as in all_fm,-reve. An alias or group skips the models a benchmark does not evaluate. |  |
| `--dataset single\|full\|NAMES` | Datasets to use. Give single for the benchmark's quick dataset, full for all its datasets, or dataset names separated by commas. | `full` |
| `--variant VARIANT` | Benchmark variant. `neuroatlas show BENCHMARK` lists the variants. | `default` |
| `--out DIR` | Folder for the job files and logs. |  |
| `--backend {condor,slurm}` | Scheduler to write the jobs for. | `condor` |
| `--mode {cached,retry,all}` | Which jobs to queue. cached queues the jobs that never ran, retry also the failed, partial and exited ones, and all every job. A job still in the queue is never queued twice. | `cached` |
| `--output-root DIR` | Where the jobs write results. | the output_root setting |
| `--force` | Also write jobs for pairs whose data or weights are missing here. The channel map still applies. |  |
| `--reprobe` | Make the jobs fit every fold again, as `run --reprobe` does. Use with --mode all to rerun finished jobs. |  |
| `-v`, `--verbose` | List every pair without a job, and why. |  |
| `--format {table,csv,md,json}` | Output format. In csv, md and json the notes under a row go to a `note` column. | `table` |

Resources per job:

| Option | Description | Default |
|---|---|---|
| `--gpus N` | GPUs per job. | `1` |
| `--cpus N` | CPUs per job. | `4` |
| `--memory SIZE` | Memory per job, with a unit such as 32G or 1500M. | `32G` |
| `--time H:MM:SS` | Wall time per job. It sets --time for SLURM, and +RequestWalltime and +MaxRuntime in seconds for HTCondor. | 24:00:00 |
| `--walltime SECONDS` | The same wall time, in seconds or as H:MM:SS. |  |
| `--gpu-capability auto\|none\|X.Y` | Lowest GPU compute capability a job may run on. auto is the lowest the installed PyTorch supports. For SLURM it is only printed as a hint. | `auto` |
| `--partition PARTITION` | SLURM partition. |  |
| `--no-mem` | Leave --mem out of SLURM jobs, for sites that forbid it. |  |
| `--requirements REQUIREMENTS` | HTCondor requirements expression. |  |
| `--walltime-attr NAME` | HTCondor job attribute that holds the wall time. Can be repeated. | RequestWalltime and MaxRuntime |
| `--extra LINE` | Add a raw line to the job file. For SLURM it follows #SBATCH, as in `--extra --account=myproject`. Can be repeated. |  |

### status

Count the jobs that `submit` wrote by state, such as done, failed, running or missing. Run it where you submitted, so that it can ask the scheduler (condor_q or squeue).

Usage: `neuroatlas status --out OUT [options]`

```bash
neuroatlas status --out jobs/sleep_stage
neuroatlas status --out jobs/sleep_stage -v
```

| Option | Description | Default |
|---|---|---|
| `--out OUT` | The folder `submit` wrote. |  |
| `-v`, `--verbose` | Show one row per job. |  |
| `--no-scheduler` | Do not ask condor_q or squeue. Use only the files. |  |
| `--format {table,csv,md,json}` | Output format. In csv, md and json the notes under a row go to a `note` column. | `table` |

```text
job states:
  done                 every fold succeeded
  partial              some folds failed or are missing
  failed               no fold succeeded
  exited N             the job stopped with exit status N before writing results
  stopped              the job started, then left the queue without an exit status
  removed              the job was removed from the queue
  running, idle, held  as the scheduler reports them
  missing              the job never ran
  waiting              the job needs the results of another benchmark first
```

## Results

### results

Summarise the results of a benchmark, with one row per dataset, variant and checkpoint. Each row shows the mean and spread of the headline metric over the folds, the number of folds, the chance level and the other metrics.

Usage: `neuroatlas results benchmark [paths ...] [options]`

```bash
neuroatlas results sleep_stage
neuroatlas results bci_motor_imagery --variant token_flattening
neuroatlas results epilepsy --format csv
```

| Option | Description | Default |
|---|---|---|
| `benchmark` | Benchmark name, as listed by `neuroatlas list benchmarks`. |  |
| `paths` | results.json files, globs or folders to read. | every results.json under `<output root>/<benchmark>/` |
| `--variant VARIANT` | Only show this variant. | every variant |
| `--output-root DIR` | Results folder to read. | the output_root setting |
| `-v`, `--verbose` | Show the error message of each failed fold. |  |
| `--format {table,csv,md,json}` | Output format. In csv, md and json the notes under a row go to a `note` column. | `table` |

### rescore

Recompute the metrics of every result from the test predictions its probes saved, and rewrite results.json. Nothing is fitted again. Use it when a metric was fixed or added after a run.

Usage: `neuroatlas rescore benchmark [options]`

```bash
neuroatlas rescore epilepsy
neuroatlas rescore sleep_stage --dataset dod -m biot_pretrained
```

| Option | Description | Default |
|---|---|---|
| `benchmark` | Benchmark name, as listed by `neuroatlas list benchmarks`. |  |
| `--dataset NAMES` | Only these datasets, separated by commas. | every dataset with results |
| `-m`, `--models MODELS` | Checkpoint ids, families, groups or an alias such as all_fm, separated by commas. Put `-` before a name to remove it, as in all_fm,-reve. An alias or group skips the models a benchmark does not evaluate. | every checkpoint with results |
| `--variant VARIANT` | Only this variant. | every variant |
| `--output-root DIR` | Results folder. | the output_root setting |
| `--format {table,csv,md,json}` | Output format. In csv, md and json the notes under a row go to a `note` column. | `table` |

## Individual steps

### embed

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

Run `neuroatlas embed --help` for the list of datasets, tasks and models, and `neuroatlas embed --dataset hmc --help` for the `--set` keys of one dataset.

### probe

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

Run `neuroatlas probe --help` for the list of datasets, tasks and models, and `neuroatlas probe --dataset hmc --help` for the `--set` keys of one dataset.

### hypnogram

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

#### hypnogram reconstruct

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

#### hypnogram features

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
