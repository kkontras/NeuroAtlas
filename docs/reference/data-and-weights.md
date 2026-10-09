<!-- Generated from the command-line parsers by `python -m neuroatlas.cli._gendocs`. Do not edit by hand. -->
# Data and weights

These commands check, download and prepare the datasets and the model weights. See [Data](../guide/data.md) and [Models](../guide/models.md).

## data

Check which datasets are on this machine, download them, and build optional faster-to-read copies.

Usage: `neuroatlas data <action>`

| Command | Description |
|---|---|
| [`data status`](#data-status) | Show which datasets are on this machine. |
| [`data download`](#data-download) | Download datasets, or print how to get them. |
| [`data prepare`](#data-prepare) | Build the files a bci_cognitive dataset is read from, or a faster copy of an epilepsy dataset. |

### data status

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

### data download

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

### data prepare

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

## models

Check which model weights are on this machine, and download the missing ones.

Usage: `neuroatlas models <action>`

| Command | Description |
|---|---|
| [`models status`](#models-status) | Show which checkpoints have their weights here. |
| [`models download`](#models-download) | Download the weights a selection is missing. |

### models status

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

### models download

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

> [!NOTE]
> `neuroatlas fetch` and `neuroatlas prepare` are other names for `neuroatlas data download` and `neuroatlas data prepare`.
