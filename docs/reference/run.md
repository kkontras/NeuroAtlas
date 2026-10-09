<!-- Generated from the command-line parsers by `python -m neuroatlas.cli._gendocs`. Do not edit by hand. -->
# Run

These commands test a benchmark, run it on this machine, or write it as cluster jobs and follow them. See [Running benchmarks](../guide/running.md).

## check

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

## run

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

## submit

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

## status

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
