# 5. Running benchmarks

A benchmark run does two things for each dataset. It extracts embeddings with each frozen
checkpoint, and then fits a probe on each fold. `check` tests that a run can work before you spend
hours on it, `run` executes it on this machine, and `submit` writes the same work as jobs for an
HTCondor or SLURM cluster. A typical session looks like this:

```bash
neuroatlas check sleep_stage -m biot_pretrained
neuroatlas run sleep_stage -m all_fm --dataset full --dry-run
neuroatlas run sleep_stage -m biot_pretrained --debug
neuroatlas run sleep_stage -m biot_pretrained
```

## 5.1 Checking before a long run

`check` pushes one real batch through each pair of dataset and checkpoint, with exactly the code a
run uses, and then stops. It trains, caches and downloads nothing. On a CPU machine that has
BIOT's and LaBraM's weights, but not CBraMod's or REVE's, and no `momentfm` package:

```text
$ neuroatlas check sleep_stage -m biot_pretrained,labram_pretrained,cbramod_pretrained,reve_pretrained,moment_small
dataset             checkpoint          data                   weights          channel map  forward pass      time
sleep_edf_expanded  biot_pretrained     found (197/197 files)  found            applied      (32, 256) finite  12s
sleep_edf_expanded  labram_pretrained   found (197/197 files)  found            applied      (32, 200) finite  14s
sleep_edf_expanded  cbramod_pretrained  found (197/197 files)  downloadable     applied      -                 0s
      skipped: weights not downloaded
sleep_edf_expanded  reve_pretrained     found (197/197 files)  downloadable     applied      -                 0s
      skipped: weights not downloaded
sleep_edf_expanded  moment_small        found (197/197 files)  package missing  applied      -                 0s
      skipped: momentfm is not installed
        fix: pip install --no-deps "momentfm==0.1.4"

5 pairs: 2 ok, 0 failed, 3 skipped (27s)
...
note: 2 checkpoints skipped for missing weights
  fix: neuroatlas models download cbramod_pretrained,reve_pretrained
```

The `forward pass` column is the shape of one batch's output, windows by embedding size.
`(32, 256) finite` means 32 embeddings of 256 numbers, none NaN or infinite. A batch where every
window gets the same embedding, or one with non-finite values, counts as failed.

The `channel map` column is `applied` when the dataset's map renames the electrodes for this
model, `applied (labels as recorded)` when the model receives the recorded names unchanged, and
`none` when the dataset has no map. `ruled out` and `invalid` mean what the
[words of this guide](index.md#words-used-in-this-guide) say. A pair that did not run says why on
the line under it. A pair the channel map excludes is ruled out before any data is read:

```text
$ neuroatlas check epilepsy --dataset bonn -m biot_pretrained
...
      ruled out: by the bonn channel map (not run on Bonn's single channel: BIOT reads 16 bipolar pairs, and Bonn's one channel (labelled FZ) is in none of them)
```

`check` exits with 0 when at least one pair went through and none failed. It exits with 1 when a
pair failed or when nothing could be checked. With `--strict`, a skipped pair also gives 1.
`--format json` and `csv` put the lines under each row in a `note` field.

## 5.2 Planning a run

For each dataset, `run` executes `neuroatlas embed`, which does nothing for embeddings already in
the cache, and then `neuroatlas probe` over the folds. `--dry-run` runs nothing. It prints the
task, the checkpoints, the folds, the number of runs, the state of the data, and the exact
commands:

```text
$ neuroatlas run sleep_stage -m biot_pretrained --dry-run
benchmark    dataset             task           checkpoints  folds          runs  data                   ruled out
sleep_stage  sleep_edf_expanded  sleep_staging  1            0, 1, 2, 3, 4  5     found (197/197 files)  0

  runs  one probe fit per checkpoint and fold

the commands it runs:
  neuroatlas embed --dataset sleep_edf_expanded --folds 0,1,2,3,4 --models biot_pretrained
  neuroatlas probe --dataset sleep_edf_expanded --task sleep_staging --models biot_pretrained --output-root <output root>/sleep_stage/sleep_edf_expanded
```

If a dataset's data is missing, the plan warns that the run would fail on it and prints the `fix:`
lines. BCI plans show the folds as `LOSO (N)`, one fold per subject.

## 5.3 Running on this machine

`run --debug` runs the whole pipeline on fold 0 only, and `run` without it runs every fold. Here
is `brain_age`, which reuses the 30 s Sleep-EDF embeddings that `sleep_stage` already extracted.
It took 72 s on a CPU:

```text
$ neuroatlas run brain_age -m biot_pretrained

$ neuroatlas embed --dataset sleep_edf_expanded --folds 0,1,2,3,4 --models biot_pretrained
embedding 1 checkpoint on sleep_edf_expanded: 1 run
[1/1] sleep_edf_expanded biot_pretrained: embedding
[1/1] sleep_edf_expanded biot_pretrained: already extracted
embed: 1 ok, 0 failed (cache: <cache root>)

$ neuroatlas probe --dataset sleep_edf_expanded --task brain_age --set label_mode=age --aggregation mean --models biot_pretrained --output-root <output root>/brain_age/sleep_edf_expanded
probing 1 checkpoint on sleep_edf_expanded: 5 runs, one line each as it finishes
[1/5] sleep_edf_expanded biot_pretrained fold 0: ok, MAE(years) 14.151 (16s)
...
[5/5] sleep_edf_expanded biot_pretrained fold 4: ok, MAE(years) 11.754 (13s)
probe: 5 ok, 0 failed (results: <output root>/brain_age/sleep_edf_expanded)

5 runs (checkpoint x fold): 5 ok, 0 failed
results: <output root>/brain_age; `neuroatlas results brain_age` summarises them
```

Each fold's result line carries that fold's headline value, under the same label as the `results`
column. Examples are `kappa 0.777` for sleep staging, `bal_acc 0.257` for BCI and
`Sens@FA_AUC(event) 0.428` on Siena. On Bonn, whose recordings have no seizure events, the line
shows `AUROC 0.993` instead. A fold scored from saved predictions says `ok, reused`
([Saved predictions are reused](advanced.md#74-saved-predictions-are-reused)).

If a pair fails, you see it at once in one line, for example
`error: sleep_edf_expanded/biot_pretrained (fold 2): CUDA out of memory.` The full message is in
`results.json` and in the `--log` file.

The closing lines count this run's work. They also say how many pairs the channel map ruled out,
and how many earlier results `results.json` still holds. [Results](results.md) explains where the
results go and how to read them.

Before extraction, `run` checks every checkpoint's weights. A checkpoint whose weights are missing
is reported by name and counted as failed, and the others run. An interrupted extraction resumes
from the batches it already finished.

`run` exits with 0 if every run succeeded, 1 if any failed, and 2 for a usage error such as an
unknown name, a bad flag value or a refused partial dataset.

## 5.4 Progress

Every step shows what it is doing from its first second. Extraction goes through loading the
data, loading the weights, reading the recordings, embedding the batches and writing the cache. A
probe fold goes through reading the embeddings, cutting the fold's split, fitting and scoring.
While fitting, the progress line counts what the probe tries: the C values of the grid for sleep
staging, epilepsy and BCI, the seeds for the sleep events and diagnosis, and the ridge's alpha
values for brain age. Sleep staging shows no time left while it fits, because a large C takes many
times longer than a small one.

Off a terminal, for example in a pipe or a cluster job's output file, there is no progress line.
`embed` prints a line for every tenth of its work instead, at most one every 5 s:

```text
embedding 1 checkpoint on bonn: 1 run
[1/1] bonn cbramod_pretrained: embedding
[1/1] bonn cbramod_pretrained: embedding 12% (2/16 batches, 0m 14s, ~1m 44s left)
...
[1/1] bonn cbramod_pretrained: ok, 1,000 windows (2m 09s)
embed: 1 ok, 0 failed (cache: <cache root>)
```

A probe fold prints only its result line, because a LOSO probe over many checkpoints can have
thousands of folds.

## 5.5 How long it takes

Measured on Sleep-EDF with BIOT:

| Step | Time |
|---|---|
| extraction, idle RTX 4500 Ada | 17 min |
| sleep-staging probe, one fold, 4 CPU cores | about 34 min |
| the same, 8 cores of an i7-9700K | 26 min |
| the same, 2 cores of an i7-9700K | 1 h 06 min |
| all five staging folds, 4 cores | about 3 h |
| `brain_age` on cached embeddings | about a minute |

Sleep staging fits six values of C in every fold, which is why it is the slowest probe.
`--dry-run` does not estimate times.

## 5.6 Run options

| Option | Effect |
|---|---|
| `--debug` | Fold 0 only. `results` then shows `1/5` folds, or `LOSO 1/9` for BCI. |
| `--folds 0,2` | These folds only. A fold the dataset does not have is a usage error. |
| `--limit-batches N` | A smoke test. Each extraction stops after N batches, in its own cache under `<cache root>/_limited`. Results go to `<output root>/_limited/<benchmark>/<dataset>/`, and the closing line gives the `results` command that reads them. |
| `--num-workers N` | Worker processes that read data during extraction. The default is the number of CPUs this job may use, minus one, at most 16. |
| `--skip-embed` | Probe only. A pair without embeddings fails. |
| `--reprobe` | Fit every fold again instead of reusing saved predictions. |
| `--cache-root`, `--output-root` | Other folders for this run. |
| `--checkpoint-override biot_pretrained.checkpoint_path=/my.ckpt` | Other weights for this run. The file must exist. `--cache-root` is required too, because the cache is keyed by checkpoint id, not by weights. |
| `--allow-partial` | Run on a dataset that `data status` calls `partial` or `empty`. Without it `run` refuses, since the results would cover only some subjects. |
| `--variant NAME` | Another protocol of the benchmark. Results go to `<dataset>/<variant>/`. |
| `--per-model-output` | Results in `<dataset>/<model>/`, as cluster jobs write them. |

With `--limit-batches`, too few subjects can leave a fold with nothing to test. On a LOSO BCI
dataset every fold then fails.

## 5.7 Running on a cluster

`submit` writes one job per dataset and checkpoint, each a `neuroatlas run` of that pair. It
queues nothing itself: its last line prints the command that does. `--dataset` defaults to
`full`.

=== "HTCondor"

    ```bash
    neuroatlas submit sleep_stage -m all_fm --out runs/sleep_stage
    condor_submit runs/sleep_stage/jobs.job
    neuroatlas status --out runs/sleep_stage
    ```

=== "SLURM"

    ```bash
    neuroatlas submit sleep_stage -m all_fm --out runs/ss --backend slurm \
        --partition gpu --cpus 24 --no-mem --extra --account=myproject
    sbatch runs/ss/jobs.sbatch
    neuroatlas status --out runs/ss
    ```

Each job writes its own `<output root>/<benchmark>/<dataset>/<model>/results.json`, so jobs never
share a file. Next to it the job's shell keeps a `job_status.json` with its state, exit code, host
and times. A job that dies before Python starts still leaves a record. `sleep_hypnogram` gets one
job per dataset.

Jobs inherit this machine's settings (the folder variables, `MNE_DATA`, `HF_HOME` and
`PYTHONPATH`, never a token) and run offline. So `submit` checks every pair first. A pair whose
data or weights are missing, or that the channel map rules out, gets no job and is counted with
the reason. Add `-v` to list each one. `--force` writes jobs for the pairs with missing data or
weights anyway, never for ruled-out ones. On a machine with only Sleep-EDF and UCDDB, and BIOT's
weights but not REVE's:

```text
$ neuroatlas submit sleep_stage -m biot_pretrained,reve_pretrained --out runs/ss
2 jobs: 2 to queue (--mode cached), 2 missing
28 pairs without a job: 26 data missing, 2 weights not downloaded (-v lists them)
note: 28 pairs were skipped for missing data or weights; --force writes their jobs anyway (they fail where those are missing)
  fix: neuroatlas data download cfs dcsm dod hmc hpap_lab_full isruc mass mesa mros physionet2026 shhs stages wsc
  fix: neuroatlas models download reve_pretrained
...
queue them: condor_submit <current folder>/runs/ss/jobs.job
```

A `GPUs:` line before the last one gives the lowest GPU compute capability the installed PyTorch
supports (7.5 for the CUDA 13 build). HTCondor jobs require it. SLURM has no standard attribute for
it, so it is a comment there.

The `--out` folder holds `jobs.json` (what was planned and skipped), one script per job in
`jobs/`, `jobs.txt` (the scripts this submit queues), `jobs.job` or `jobs.sbatch`, and `logs/`.

## 5.8 Following the jobs

`status` asks `condor_q` or `squeue` when they are available, so run it where you submitted. It
also reads the HTCondor event logs and the jobs' own files. `--no-scheduler` uses the files only.
Failures are listed with their host and log, and `-v` gives one row per job.

| Verdict | Meaning |
|---|---|
| `done` | Results written, every fold succeeded. |
| `partial` | Some folds succeeded, some failed. |
| `failed` | Results written but no fold succeeded, or the job exited 0 without results. |
| `exited N` | The job ended with code N before writing results. 128+N means it was killed by signal N. |
| `stopped` | It started, then left the queue without an exit code (killed, evicted or a node failure). |
| `removed` | Removed from the queue. |
| `running`, `idle`, `held` | Still in the scheduler's queue. |
| `missing` | Never ran. Every job starts here. |
| `waiting` | A hypnogram job whose staging results are not there yet. |

Run `neuroatlas results <benchmark>` once the jobs are done. It reads the jobs' results like those
of `run`.

## 5.9 Submitting again

By default (`--mode cached`) `submit` queues only `missing` jobs, so running it again never
repeats work. `--mode retry` also takes failed, partial, exited, stopped and removed jobs.
`--mode all` takes everything. A job still in the queue is never queued twice. Jobs reuse saved
predictions like `run` does. `submit --reprobe` writes jobs that fit every fold again.

## 5.10 Job resources

| Option | Default | Notes |
|---|---|---|
| `--gpus` | 1 | |
| `--cpus` | 4 | |
| `--memory` | `32G` | needs a unit, such as `32G` or `1500M` |
| `--time H:MM:SS` or `--walltime SECONDS` | 24 h | SLURM gets `--time`. HTCondor gets `+RequestWalltime` and `+MaxRuntime`, or the names in `--walltime-attr`. |
| `--gpu-capability` | `auto` | requires a GPU the installed PyTorch supports |
| `--partition`, `--no-mem` | | SLURM only |
| `--requirements EXPR` | | HTCondor only |
| `--extra LINE` | | adds a raw line to the submit file, for example `--extra --account=x` |

`submit` warns about a flag the chosen backend ignores. It also warns when the Python or a
`PYTHONPATH` entry is under `/tmp`, `/var/tmp` or `/dev/shm`, since other machines cannot see it.

If one host keeps failing your jobs, keep them off it:

```bash
neuroatlas submit sleep_stage -m all_fm --out runs/ss \
    --requirements 'Machine =!= "host.example.org"'
```

`condor_submit -dry-run` checks the submit file's syntax, not your pool's policy.
