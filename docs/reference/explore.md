<!-- Generated from the command-line parsers by `python -m neuroatlas.cli._gendocs`. Do not edit by hand. -->
# Explore

These commands describe the benchmarks, datasets, models and tasks without running anything. [Benchmarks](../guide/benchmarks.md) explains what they show.

## list

List the benchmarks, datasets, models, aliases or tasks. This reads only the tables that ship with NeuroAtlas. Use `neuroatlas data status` and `neuroatlas models status` to see what is on this machine.

Usage: `neuroatlas list <what>`

| Command | Description |
|---|---|
| [`list benchmarks`](#list-benchmarks) | List each benchmark with its task, datasets and headline metric. |
| [`list datasets`](#list-datasets) | List the paper's datasets with their domain, access, size and benchmarks. |
| [`list models`](#list-models) | List the checkpoints with their family, group, input rate, window and embedding size. |
| [`list aliases`](#list-aliases) | List the names that select groups of checkpoints with -m. |
| [`list tasks`](#list-tasks) | List the probe tasks and the settings each benchmark runs them with. |

### list benchmarks

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

### list datasets

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

### list models

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

### list aliases

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

### list tasks

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

## show

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
