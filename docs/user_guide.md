# Running NeuroAtlas with the `neuroatlas` command

From an empty machine to a leaderboard: set the tool up once, get datasets
and model weights, check that everything fits, run a benchmark here or on a
cluster, and read the scores. Every flag is in [cli.md](cli.md).

Every run answers one question: **benchmark × models × datasets**.

- **Benchmark** -- what is predicted and how it is scored: `sleep_stage`,
  `brain_age`, `epilepsy`, `bci_motor_imagery`, ... (`neuroatlas list benchmarks`).
- **Models** -- checkpoints, by id, family or alias: `-m all_fm`,
  `-m reve,biot_pretrained`.
- **Datasets** -- `--dataset single` (one dataset, a quick check), `full`
  (every dataset of the benchmark) or a comma list.

The steps are always the same, and each command says what is missing and
which command fixes it:

    config init → list → data → models → check → run  (or submit → status) → results / leaderboard

## Words you will meet

| | |
|---|---|
| **checkpoint** | One set of pretrained weights, e.g. `biot_pretrained`. "Model" below means a checkpoint. |
| **alias** | A name for a group of checkpoints: `all_fm` (EEG foundation models), `all_ts` (general time-series models), `all_supervised`, `all_random` (untrained baselines), `all`. |
| **embedding** | What a frozen model turns one window of EEG into. Models are never fine-tuned. |
| **embedding cache** | Where embeddings are saved: computed once per dataset × model, reused by every fold, probe and benchmark. |
| **probe** | A linear classifier (ridge regressor for brain age) trained on the embeddings. Its score is the model's score. |
| **fold** | One patient-level split into train and test. Most benchmarks use 5 fixed folds; BCI is leave-one-subject-out. |
| **channel map** | Per dataset: which electrodes each model gets. It can mark a model *skip* when the dataset lacks what it needs; that pair is **n/a**, never a failure. |
| **normalised score** | `(score − dummy) / (1 − dummy)`: 0 is a trivial guess, 1 is perfect. |

## 1. Install and set up once

```bash
# PyTorch for your CUDA version first (pytorch.org), then
pip install "neuroatlas[fm]"            # or, from a checkout: pip install -e ".[fm]"
pip install "neuroatlas[bci]"           # the MOABB BCI cohorts, if you need them

neuroatlas config init --data-root /data/eeg
neuroatlas config show
```

The package carries its own tables (cohort manifests, channel maps, folds,
benchmark definitions); a checkout is not needed. `config init` writes
`~/.neuroatlas/config.yaml` (`$NEUROATLAS_HOME` moves it):

| setting | what | default |
|---|---|---|
| `data_root` | raw datasets, one sub-folder each | required |
| `cache_root` | embeddings; can grow large | `<workspace>/artifacts/embedding_cache` |
| `output_root` | results | `<workspace>/artifacts/benchmarks` |
| `models_root` | model weights | `<workspace>/artifacts/models` |

`<workspace>` is the checkout when you run from one, else `~/.neuroatlas`.
Change one setting with `neuroatlas config set cache_root /scratch/cache`,
and point at a dataset kept elsewhere with
`neuroatlas config set ucddb.data_root /mnt/ucddb`. An environment variable
(`EEG_DATA_ROOT`, `EEG_CACHE_ROOT`, `NEUROATLAS_OUTPUT_ROOT`,
`NEUROATLAS_MODELS_ROOT`) beats the file, so a cluster job can redirect one
run without touching it.

Tokens, only for gated downloads, go next to the config, `chmod 600`:
`~/.neuroatlas/hf_token` (Hugging Face) and `~/.neuroatlas/nsrr_token`
(sleepdata.org/token). `config show` says where they are, never what.

**Offline by default.** Only `data download`, `models download`, `fetch
--download` and `data prepare` use the network; everything else, and every
cluster job, runs with downloads off, so a missing file fails at once and
names the command that fetches it. `--online` lifts that for one run.

## 2. See what exists

```bash
neuroatlas list benchmarks          # what each predicts, datasets, headline metric, dummy
neuroatlas show sleep_stage         # one benchmark explained, and the exact commands it runs
neuroatlas list datasets --grep sleep
neuroatlas list models all_fm
neuroatlas list aliases
neuroatlas list tasks
```

Every listing takes `--format table|csv|md|json`.

Each benchmark is a protocol the paper ran, by name: `neuroatlas show
<benchmark>` prints the `embed` and `probe` command lines it expands to, and
those are the lines of `run/default_runs.sh`, the record of what the paper
ran.

## 3. Get the datasets

```bash
neuroatlas data status sleep_stage        # found (N files) / partial / missing / not configured
neuroatlas data download ucddb --dry-run  # the exact commands
neuroatlas data download ucddb
```

`data status` resolves each dataset's path exactly as a run will, lists the
folder it looked in for each one it did not find, and prints the `config
set` line that points at a copy you already have. `-v` shows every path.

| access | what `data download` does |
|---|---|
| PhysioNet (Sleep-EDF, UCDDB, HMC) | recursive `wget` of the release, straight into the folder the reader expects; resumes |
| Zenodo (CHB-MIT, Siena, Helsinki) | downloads the archive and unpacks it where the reader expects |
| NSRR (SHHS, WSC, CFS, MrOS, MESA, ...) | the official `nsrr` tool (`gem install --user-install nsrr irb`), token on its prompt, after a free-disk check. `NEUROATLAS_NSRR_DATASETS=wsc,shhs` refuses cohorts your token is not approved for, up front |
| MOABB (the BCI cohorts) | MOABB's own download, into `$MNE_DATA` |
| TUH, internal, manual | prints what to request and where to put it |

The token is fed to the tool's prompt on stdin: never on the command line
(visible in `ps`), never in the environment, never printed.

`neuroatlas data prepare <dataset>` builds a cache for the few datasets that
have a build step -- required for the MOABB motor-imagery cohorts, an
optional speed-up for the large epilepsy ones -- and refuses until the raw
corpus is there.

## 4. Get the models

```bash
neuroatlas models status all_fm
neuroatlas models download all_fm
```

| state | meaning |
|---|---|
| `found` | weights on disk |
| `auto` | `models download` fetches them (GitHub release, Hugging Face, Figshare) |
| `hub` / `hub (cached)` | a Hugging Face model loaded through its cache |
| `manual` | fetch by hand; the note says from where (REVE: accept its licence on huggingface.co) |
| `nothing needed` | an untrained baseline |

`-v` shows where each is expected: under the models root.

## 5. Check before spending GPU hours

```bash
neuroatlas check sleep_stage -m biot_pretrained,labram_pretrained,reve_pretrained
```

One real batch through each dataset × model pair, with exactly the code a
run uses, then it stops: nothing trained, cached or downloaded. From this
repository's own test on Sleep-EDF:

```
dataset             model               data               weights       channel_map  forward
sleep_edf_expanded  biot_pretrained     found (394 files)  found         applied      (32, 256) finite
sleep_edf_expanded  labram_pretrained   found (394 files)  found         applied      (32, 200) finite
sleep_edf_expanded  cbramod_pretrained  found (394 files)  hub (cached)  applied      (32, 200) finite
sleep_edf_expanded  reve_pretrained     found (394 files)  manual        applied      -
    ↳ weights manual: forward skipped (from brain-bzh/reve-base)
pairs: 4   forward passes: 3   errors: 0
```

`(32, 256) finite` is a batch of 32 windows turned into 32 embeddings of
256 numbers, none NaN or infinite; a constant embedding is flagged too.
Exit status 1 if any pair errored.

## 6. Run a benchmark

```bash
neuroatlas run sleep_stage -m all_fm --dataset single --dry-run   # the plan
neuroatlas run sleep_stage -m biot_pretrained --debug             # fold 0: a quick try
neuroatlas run sleep_stage -m all_fm --dataset full               # the real thing
```

For each dataset, `run` calls `embed` (a no-op for pairs already in the
cache) and then `probe` over the folds. Results go to
`<output root>/<benchmark>/<dataset>/`: `results.json`, `results.csv`,
`summary.md`.

- **The cache is the expensive part, and it is shared.** Embeddings are keyed
  by dataset and checkpoint, not by fold or probe, so `--debug` extracts in
  full and the real run reuses it; so do other benchmarks on the same
  dataset (brain age reuses sleep staging's).
- **`--limit-batches N`** is a smoke test: extraction stops after N batches,
  in its own `<cache root>/_limited`, so a truncated cache is never read as
  a complete one.
- **Interrupted extraction** resumes where it stopped.
- **Other weights:** `--checkpoint-override biot_pretrained.checkpoint_path=/my.ckpt`
  requires `--cache-root`, since the cache is keyed by checkpoint id.
- **Exit status** 0 if every run succeeded, 1 if any failed, 2 for a usage error.

`--variant` picks one of a benchmark's other cells (BCI: `per_patch`,
`confound_control`; `neuroatlas show` lists them).

## 7. Run on a cluster

```bash
neuroatlas submit sleep_stage -m all_fm --out runs/sleep_stage            # HTCondor
neuroatlas submit sleep_stage -m all_fm --out runs/ss --backend slurm \
    --partition gpu --cpus 24 --no-mem
condor_submit runs/sleep_stage/jobs.job
neuroatlas status --out runs/sleep_stage
```

One job per dataset × model, each a `neuroatlas run` of that pair writing
`<output root>/<benchmark>/<dataset>/<model>/results.json`, so jobs never
share a file. Jobs inherit this machine's settings and run offline: download
data and weights first. Pairs the channel map rules out get no job.

| verdict | meaning |
|---|---|
| done | every fold has a result |
| partial | some folds succeeded, some failed |
| failed | results written, none succeeded; the job's log says why |
| missing | no results yet -- every job, right after submit |
| waiting | (hypnograms) the staging results it reads are not there yet |

`--mode cached` (default) queues only jobs with no results, so re-running
submit never repeats work; `--mode retry` adds failed and partial ones.
Resources: `--gpus 1 --cpus 4 --memory 32G` by default; SLURM `--time`,
`--partition`, `--no-mem`; HTCondor `--requirements`, `--walltime`;
`--extra LINE` for anything site-specific.

## 8. Read the results

```bash
neuroatlas results sleep_stage
neuroatlas results sleep_stage runs/old/**/results.json --format md
neuroatlas leaderboard --suite single
```

`results` gives one row per dataset × model: the headline metric's mean and
standard deviation over the folds that succeeded, how many did, the
normalised score and the secondary metrics. `leaderboard` ranks models on
each dataset, averages the ranks within a benchmark, then across
benchmarks (`mean_rank` 1 = best everywhere).

**n/a is never zero**: a model the channel map rules out, a spread over one
fold, a model missing from some dataset of a suite (listed, not ranked). A
result recorded twice -- a quick `run` and later a cluster job -- counts
once, the newest.

`results --reference` puts the paper's numbers beside yours, from
`configs/reference/<benchmark>.csv`; no benchmark ships that table yet.

## From Python

```python
from neuroatlas import api

api.plan("sleep_stage", "all_fm", datasets="full")     # = run --dry-run
api.check("sleep_stage", "biot_pretrained")             # = check
api.run_benchmark("sleep_stage", "biot_pretrained", debug=True)
api.results("sleep_stage")                              # = results, a DataFrame
api.leaderboard(suite="single")["global"]
```

Importing `neuroatlas.api` applies the same settings file.

## Where things live

| what | where |
|---|---|
| settings, tokens | `~/.neuroatlas/` |
| datasets | `<data root>/...`, one folder each (`data status -v`) |
| weights | `<models root>/...`; hub models in the Hugging Face cache |
| embeddings | `<cache root>/<dataset>/<checkpoint>/all/<key>/features.npy` |
| results of `run` | `<output root>/<benchmark>/<dataset>/results.json` |
| results of cluster jobs | `<output root>/<benchmark>/<dataset>/<model>/results.json` |
| job files | the `--out` folder: `jobs.json`, `jobs/`, `jobs.job` or `jobs.sbatch`, `logs/` |

## Known gaps

- No `configs/reference/*.csv` ships yet, so `results --reference` has
  nothing to compare against.
- The default dataset folders are the authors' layout (e.g.
  `data/Guido/hmc/physionet.org/files/hmc-sleep-staging/1.1/recordings`);
  `data download` lands in them and `data status` looks there, so the two
  agree, but the tree is not tidy.
- The CoRe-Sleep and SleepTransformer baselines are release assets of a
  repository that is not public yet: without a GitHub token `models
  download` gets a 404.
- `check` builds each pair's datamodule afresh; on a cohort whose reader
  indexes every window up front (CHB-MIT: 232k) that takes minutes.
