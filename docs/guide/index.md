# User guide

This guide takes you from an empty machine to a benchmark's scores. You set the package up once,
download datasets and weights, check that everything fits, run a benchmark on your machine or on
a cluster, and read the results. If you have not installed the package yet, start with
[Installation](../getting-started/installation.md). For a shorter tour that runs every command
once on open data, see the [walkthrough](../getting-started/walkthrough.md). Every flag is listed
in the [command reference](../reference/index.md), which is generated from the code.

The chapters follow the order in which you will need them:

1. [Configuration](configuration.md): the project folder, settings and tokens.
2. [Data](data.md): which datasets you have and how to get the others.
3. [Models](models.md): checkpoints, groups and weights.
4. [Benchmarks](benchmarks.md): the twelve benchmarks, by domain.
5. [Running benchmarks](running.md): `check`, `run`, and jobs on a cluster.
6. [Results](results.md): the results tables, saved predictions and `rescore`.
7. [Advanced use](advanced.md): the individual steps, folds, caches, environment variables and
   the Python API.
8. [Troubleshooting](troubleshooting.md): what to do when a command fails.

In the sample outputs, paths are shortened to a name in angle brackets, such as `<data root>` or
`<output root>`, and `...` marks lines left out.

## Words used in this guide

Every run answers one question: which benchmark, with which checkpoints, on which datasets. The
tool uses the same words in its tables, messages and flags.

| Word | Meaning |
|---|---|
| benchmark | One protocol of the paper, such as `sleep_stage` or `epilepsy`. It fixes what is predicted, on which datasets, and how it is scored. There are 12 (`neuroatlas list benchmarks`). |
| dataset | One collection of recordings, by its name, such as `sleep_edf_expanded`. The paper evaluates 42 datasets. `neuroatlas list datasets` shows 43 entries because DREAMER's valence and arousal labels are listed separately. |
| quick dataset | The one small dataset of a benchmark that `run` and `check` use by default (`--dataset single`). Use `--dataset full` for all of a benchmark's datasets. |
| variant | Another protocol of the same benchmark, selected with `--variant`. Its results are kept apart from the default's. |
| checkpoint | One set of model weights, such as `biot_pretrained`. Select checkpoints with `-m`. |
| family | The architecture behind several checkpoints, such as `reve`. |
| alias, group | A name for a set of checkpoints, such as `all_fm`. |
| embeddings | The vectors a frozen model produces for each window of EEG. Models are never fine-tuned. |
| probe | The linear model fitted on the embeddings. Logistic regression, or ridge regression for brain age. |
| fold | One split of a dataset's subjects into train, validation and test. |
| run | One checkpoint on one fold, which is one probe fit. |
| LOSO | Leave-one-subject-out, the BCI protocol. Each fold tests one subject. |
| channel map | For each dataset, which electrodes each model family receives. |

A pair of dataset and checkpoint that gives no result is marked with one of four words:

| Word | Meaning |
|---|---|
| skipped | Its data or weights are not on this machine. The `fix:` line tells you how to get them. |
| ruled out | The dataset's channel map excludes this model. It never runs there. |
| invalid | The dataset's channel map has no entry for the model's family. The pair is reported and not run. |
| failed | It ran and raised an error. |

A number that does not exist is shown as `n/a` (and `null` in JSON), never as zero. Examples are
the spread of a single fold, or the event-level seizure metric on datasets without seizure events.

## How the tool talks to you

Results, tables, and the start and result lines of long steps go to stdout. Messages go to
stderr.

Long steps show a progress line that is rewritten in place on a terminal:

```text
[1/1] bonn cbramod_pretrained: embedding 50% (8/16 batches, 1m 01s, ~1m 02s left)
```

When the step finishes, the progress line is replaced by a result line:

```text
[1/1] bonn cbramod_pretrained: ok, 1,000 windows (2m 09s)
```

Result lines are ordinary output, so pipes, cluster logs and `--log` files keep them. The
progress line appears on a terminal only.

Messages start with a word that tells you how serious they are. `error:` means the command, or
one pair in it, failed or was refused. `warning:` means the command continues, but you should
read the message before you use the results. `note:` is information. When there is something to
do, the command is on its own `fix:` line below the message:

```text
missing data          fix: neuroatlas data download <dataset>
a dataset elsewhere   fix: neuroatlas config set <dataset>.<key> DIR
missing weights       fix: neuroatlas models download <checkpoint>
```

Library output, such as MNE's filter reports or Hugging Face progress bars, is hidden by default.
Add `-v` to see it, or `--log FILE` to keep it in a file. A long error from a library is cut to
its first sentence and ends with `(-v: full message)`. Colours appear on a terminal only, and
never when `NO_COLOR` is set.
