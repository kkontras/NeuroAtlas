<!-- Generated from the command-line parsers by `python -m neuroatlas.cli._gendocs`. Do not edit by hand. -->
# Command reference

This section lists every `neuroatlas` command and its options, one page per
group of commands. Run `neuroatlas <command> --help` to see the same text in
the terminal. The [user guide](../guide/index.md) shows how the commands fit
together.

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
| [Setup](setup.md) | [`config`](setup.md#config) | Set where data, caches, results and model weights are stored. |
| [Explore](explore.md) | [`list`](explore.md#list) | List the benchmarks, datasets, models, aliases or tasks. |
|  | [`show`](explore.md#show) | Describe a benchmark and print the commands it runs. |
| [Data and weights](data-and-weights.md) | [`data`](data-and-weights.md#data) | Check, download and prepare datasets. |
|  | [`models`](data-and-weights.md#models) | Check and download model weights. |
| [Run](run.md) | [`check`](run.md#check) | Test each dataset and model on one batch before a long run. |
|  | [`run`](run.md#run) | Run a benchmark on this machine. |
|  | [`submit`](run.md#submit) | Write HTCondor or SLURM jobs for a benchmark. |
|  | [`status`](run.md#status) | Show the state of the jobs that `submit` wrote. |
| [Results](results.md) | [`results`](results.md#results) | Summarise a benchmark's results. |
|  | [`rescore`](results.md#rescore) | Recompute the metrics from saved test predictions. |
| [Individual steps](individual-steps.md) | [`embed`](individual-steps.md#embed) | Extract embeddings of one dataset with frozen models. |
|  | [`probe`](individual-steps.md#probe) | Fit probes on extracted embeddings. |
|  | [`hypnogram`](individual-steps.md#hypnogram) | Compute hypnograms and sleep features from sleep staging results. |
