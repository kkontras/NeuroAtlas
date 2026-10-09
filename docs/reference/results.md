<!-- Generated from the command-line parsers by `python -m neuroatlas.cli._gendocs`. Do not edit by hand. -->
# Results

These commands summarise the results and recompute their metrics from the saved predictions. See [Results](../guide/results.md).

## results

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

## rescore

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
