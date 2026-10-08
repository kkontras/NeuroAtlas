# User guide

This guide takes you from an empty machine to a benchmark's scores. You
install the package, set it up once, download datasets and weights, check
that everything fits, run a benchmark on your machine or on a cluster, and
read the results.

For a shorter tour that runs every command once on open data, see
[walkthrough.md](walkthrough.md). Every flag is listed in [cli.md](cli.md),
which is generated from the code.

In the sample outputs below, paths are shortened to a name in angle
brackets, such as `<data root>` or `<output root>`, and `...` marks lines
left out.

## Contents

1. [Install](#1-install)
2. [Settings and tokens](#2-settings-and-tokens)
3. [See what exists](#3-see-what-exists)
4. [Get the datasets](#4-get-the-datasets)
5. [Get the model weights](#5-get-the-model-weights)
6. [Check before a long run](#6-check-before-a-long-run)
7. [Run a benchmark](#7-run-a-benchmark)
8. [Run on a cluster](#8-run-on-a-cluster)
9. [Read the results](#9-read-the-results)
10. [Use it from Python](#10-use-it-from-python)
11. [Where files go](#11-where-files-go)

## Words used in this guide

Every run answers one question: which **benchmark**, with which
**checkpoints**, on which **datasets**. The tool uses the same words in its
tables, messages and flags.

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

A pair of dataset and checkpoint that gives no result is marked with one
of four words.

| Word | Meaning |
|---|---|
| skipped | Its data or weights are not on this machine. The `fix:` line tells you how to get them. |
| ruled out | The dataset's channel map excludes this model. It never runs there. |
| invalid | The dataset's channel map has no entry for the model's family. The pair is reported and not run. |
| failed | It ran and raised an error. |

A number that does not exist is shown as `n/a` (and `null` in JSON), never
as zero. Examples are the spread of a single fold, or the event-level
seizure metric on datasets without seizure events.

## How the tool talks to you

Results, tables, and the start and result lines of long steps go to
stdout. Messages go to stderr.

Long steps show a progress line that is rewritten in place on a terminal:

```
[1/1] bonn cbramod_pretrained: embedding 50% (8/16 batches, 1m 01s, ~1m 02s left)
```

When the step finishes, the progress line is replaced by a result line:

```
[1/1] bonn cbramod_pretrained: ok, 1,000 windows (2m 09s)
```

Result lines are ordinary output, so pipes, cluster logs and `--log` files
keep them. The progress line appears on a terminal only.

Messages start with a word that tells you how serious they are.

| First word | Meaning |
|---|---|
| `error:` | The command, or one pair in it, failed or was refused. |
| `warning:` | The command continues. Read it before you use the results. |
| `note:` | Information. |

When there is something to do, the command is on its own `fix:` line
below the message. For example:

```
missing data          fix: neuroatlas data download <dataset>
a dataset elsewhere   fix: neuroatlas config set <dataset>.<key> DIR
missing weights       fix: neuroatlas models download <checkpoint>
```

Library output, such as MNE's filter reports or Hugging Face progress bars,
is hidden by default. Add `-v` to see it, or `--log FILE` to keep it in a
file. A long error from a library is cut to its first sentence and ends with
`(-v: full message)`. Colours appear on a terminal only, and never when
`NO_COLOR` is set.

## 1. Install

The [README](../README.md#install) has the full recipe. In short, use
Python 3.11 (3.10 or newer works) in a clone of the repository:

```bash
conda create -n neuroatlas python=3.11 -y
conda activate neuroatlas
pip install -e ".[fm]" -c requirements-fm.txt
```

The constraints file pins every package to the version the paper used.
PyTorch's default build targets CUDA 13.

The distribution is called `neuroatlas-bench` and is installed from the
repository. The command and the Python import are both `neuroatlas`. Do not
`pip install neuroatlas`. That PyPI project is unrelated.

Some models and datasets need more packages:

| To run | Install |
|---|---|
| EEG foundation models, supervised baselines, sleep and epilepsy datasets | `[fm]` |
| Chronos | `pip install -e ".[fm,ts]" -c requirements-fm.txt` |
| MOMENT | the line above, then `pip install --no-deps "momentfm==0.1.4"` |
| Moirai | a separate environment from `requirements-tsfm.txt` (Python 3.10, torch 2.4.1) |
| the 14 MOABB BCI datasets | `pip install -e ".[fm,bci]" -c requirements-fm.txt`, then `pip install --no-deps "moabb==1.2.0"` |

`--no-deps` keeps the old pins that moabb 1.2.0 and momentfm declare from
downgrading the rest of the stack. Both run correctly with it. `pip check`
will still list those declared pins.

The `[fm]` extra includes `xlrd` and `openpyxl`, which read the ages of
Sleep-EDF and ISRUC subjects for brain age. Dreyer2023 and Kim2025BetaRange
are newer than moabb 1.2.0, so their readers ship inside the package.

Check the install:

```bash
neuroatlas --version
```

From a clone this prints the version and the commit, for example
`neuroatlas 0.1.0 (1c13d9e)`.

## 2. Settings and tokens

### The four folders

Choose one project folder. Everything the tool keeps goes into it:

```bash
neuroatlas config init ~/neuroatlas
neuroatlas config show
```

This creates four sub-folders and writes their paths to
`~/.neuroatlas/config.yaml` (set `$NEUROATLAS_HOME` to keep that file
elsewhere):

| Setting | Holds | Folder | Environment variable |
|---|---|---|---|
| `data_root` | raw datasets | `~/neuroatlas/data` | `EEG_DATA_ROOT` |
| `cache_root` | embeddings and prepared files (can grow large) | `~/neuroatlas/cache` | `EEG_CACHE_ROOT` |
| `output_root` | results | `~/neuroatlas/results` | `NEUROATLAS_OUTPUT_ROOT` |
| `models_root` | model weights | `~/neuroatlas/models` | `NEUROATLAS_MODELS_ROOT` |

To put one of them somewhere else, add its option, for example
`neuroatlas config init ~/neuroatlas --data-root /shared/eeg`.

To change one folder later, run `neuroatlas config set cache_root DIR`.
`neuroatlas config unset cache_root` returns it to the default.

An environment variable overrides the file. This lets a cluster job
redirect one run without touching your settings. `submit` copies the
variables that are set into every job script.

### A dataset you already have

Every dataset lives in its own folder, `<data root>/<dataset>`. If you
already have a copy somewhere else, point the tool at it:

```bash
neuroatlas config set ucddb.data_root /mnt/ucddb
```

The name of the setting depends on the dataset:

| Setting | Datasets |
|---|---|
| `data_root` | most datasets |
| `bids_root` | `chbmit`, `siena` |
| `raw_root` | `helsinki_neonatal`, `nmt`, `tuab`, `tusz` |
| `raw_dir` | `bonn` |
| `preprocessed_path` | the four `bci_cognitive` datasets |

MOABB datasets use one shared folder instead (see [MOABB](#moabb-datasets)).
You do not need to remember the table. `data status` prints the setting for
every dataset it does not find, and `config set` refuses a wrong key and
suggests the right one:

```
$ neuroatlas config set siena.data_root /tmp
error: siena has no path key 'data_root' (did you mean bids_root?); its keys: bids_root, cache_root
  fix: neuroatlas config set siena.bids_root /tmp
```

A misspelt top-level setting in `config.yaml` stops every command except
`config`, and the error names the `config unset` command that removes it.

### What `config show` prints

`config show` lists each folder, whether it exists and is writable, and
what set it (an environment variable, the config file or the default). It
also shows the MOABB folder, the dataset paths you set, and where each
token was found. It never prints a token.

```
$ neuroatlas config show
config file: <home>/config.yaml  [found]
home:        <home>  ($NEUROATLAS_HOME; the defaults below are under it)
package:     checkout <clone>
folders  [exists?] [set by: an environment variable, the config file, or the default]
  data root    <data root>  [exists, writable] [config file]
  cache root   <cache root>  [exists, writable] [config file]
  output root  <output root>  [exists, writable] [config file]
  models root  <models root>  [exists, writable] [config file]
  MOABB data   <MOABB folder>  [exists, writable] [$MNE_DATA]
dataset paths
  sleep_edf_expanded.data_root  <Sleep-EDF folder>  [exists, writable]
...
benchmarks: 12 defined, all valid
```

### Tokens

You need a token only for the NSRR sleep datasets. Save it once:

```bash
neuroatlas config token nsrr
```

The command asks for the token without showing it and stores it next to
the config file, readable only by you. `--remove` deletes it. `config show`
warns if others can read a token file.

| Token | Used for | Looked for in this order |
|---|---|---|
| NSRR (sleepdata.org/token) | the NSRR sleep datasets | `$NSRR_TOKEN`, `$NEUROATLAS_HOME/nsrr_token` |
| Hugging Face | gated Hugging Face repositories | `$HF_TOKEN`, the file named by `$NEUROATLAS_HF_TOKEN_FILE`, `$NEUROATLAS_HOME/hf_token`, `<checkout>/.secrets/hf_token`, `~/.cache/huggingface/token` |
| GitHub | GitHub release downloads | `$GITHUB_TOKEN`, `$GH_TOKEN`, `$NEUROATLAS_HOME/github_token` |

No checkpoint needs a Hugging Face token. Every Hugging Face repository the
models use is public. If a gated repository ever refuses a download, the
error tells you to accept its terms on huggingface.co and then run
`neuroatlas config token hf`.

### Downloads happen only when you ask

Only `data download`, `data prepare`, `models download` and
`fetch --download` use the network. Every other command, and every cluster
job, runs with downloads switched off. A missing checkpoint then fails at
once, by name, with the command that fetches it, instead of stalling a job
halfway.

To allow downloads for one command, add `--online`:

```bash
neuroatlas --online run sleep_stage -m biot_pretrained
```

With `--online`, `run` fetches a missing checkpoint itself. `check` never
downloads anything. The same rule covers MOABB. Offline, a MOABB dataset
that is not fully downloaded stops the run at its first read and names
`neuroatlas data download <dataset>`.

Global options such as `--online`, `--log FILE` and `-v` can go before or
after the command. A mistyped command or option gets the corrected command
as its fix:

```
$ neuroatlas chek epilepsy -m biot_pretrained
error: unknown command 'chek'
  fix: neuroatlas check epilepsy -m biot_pretrained
```

## 3. See what exists

```bash
neuroatlas list benchmarks
neuroatlas show sleep_stage
neuroatlas list datasets --grep sleep
neuroatlas list models all_fm
neuroatlas list aliases
neuroatlas list tasks
```

Every listing takes `--format table|csv|md|json`.

`list models` shows each checkpoint's family and group, where its weights
come from, the sampling rate and window it was built for, and the size of
its embedding. `list tasks` shows the 11 tasks the benchmarks run. Add `-v`
to see all 20. `run` picks the right task for each benchmark, so you only
need tasks when you call `probe` yourself.

### One benchmark in detail

`show` explains a benchmark. It says what the headline metric is computed
over, what a fold is, what ± means, which other metrics are reported, and
which datasets and variants exist. It ends with the `embed` and `probe`
commands that `run` executes.

```
$ neuroatlas show sleep_stage
sleep_stage: Sleep staging  (sleep, App. C.2)
  Which sleep stage (W, N1, N2, N3, REM) is each 30 s epoch?

headline    kappa: Cohen's kappa over 30 s epochs, 5 stages (W, N1, N2, N3,
            REM), each fold's test subjects pooled, unscored epochs left out;
            higher is better; chance 0
folds       5 subject-level folds; the probe's C is chosen from 0.001-100 on
            validation Cohen's kappa, unweighted loss; ± = population SD over
            the folds
...
datasets    15; the quick dataset (--dataset single): sleep_edf_expanded
  cfs
  ...
the commands `neuroatlas run` runs (all datasets, default variant):
  neuroatlas embed --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset cfs --set window_s=30 --set stride_s=30
  ...
  neuroatlas probe --models all,-eegnetv4,-deepsoz_hem,-seizure_transformer --dataset wsc --task sleep_staging
```

These are the same commands as in `run/default_runs.sh`, the record of
every experiment in the paper.

### Headline metrics

| Benchmark | Headline | Scored over | Folds and ± |
|---|---|---|---|
| `sleep_stage` | Cohen's κ, 5 stages. C chosen from 0.001-100 on validation κ, unweighted loss. | 30 s epochs, each fold's test subjects pooled | 5 subject-level folds, SD over folds |
| `sleep_arousal` | AUPRC. An epoch is positive if it holds more than 3 s of arousal. | 30 s epochs | 5 subject-level folds, SD over folds |
| `sleep_respiratory` | AUPRC. More than 10 s of apnea or hypopnea (RERAs not counted). | 30 s epochs | the same |
| `sleep_limb` | AUPRC. More than 0.5 s of periodic limb movement. | 30 s epochs | the same |
| `sleep_diagnosis` | AUROC. Logistic regression at C = 1, unweighted loss. `n/a` on ISRUC, whose diagnosis has more than two classes. | subjects, one mean embedding each | 5 subject-level folds, SD over folds |
| `sleep_hypnogram` | Mean Pearson r between hypnogram features from the predicted and the scored hypnograms | test recordings of all folds pooled | one value, no ± |
| `brain_age` | MAE in years. Ridge regression, alpha by nested 4-fold CV. | subjects (recordings for ISRUC and WSC), one mean embedding each | 5 subject-level folds stratified on age, SD over folds |
| `epilepsy` | Event-level Sens@FA AUC over 0.1-100 false alarms per hour. C chosen on validation AUPRC. `n/a` on Bonn, TUAB and NMT. | seizure events | 5 patient-level folds, sample SD (ddof=1) of the per-fold AUCs |
| `bci_*` | Balanced accuracy. Logistic regression at C = 1, balanced loss. | one held-out subject's trials | LOSO, SD over held-out subjects |

Every SD except epilepsy's is the population SD (ddof=0). For BCI, the
paper's figures plot the rescaling (BA - 1/C) / (1 - 1/C) * 0.5 + 0.5 of
balanced accuracy (App. C.4, Eq. 4). It puts chance at 0.5 for every number
of classes C.

### BCI variants

The four BCI benchmarks run confound filtering by default (App. D.6). Each
paradigm is filtered its own way:

| Paradigm | Confound filtering (the default) | No filtering (`--variant no_filtering`) |
|---|---|---|
| Motor imagery | 4-40 Hz, the trial from 1 s to 4 s after the cue | 4-40 Hz, the trial from the cue to 4 s after it |
| ERP | 0.5-40 Hz, the cohort's own trial window | 1-30 Hz, the cohort's own trial window |
| SSVEP | 1 Hz high-pass and no low-pass, the cohort's own trial window | 1-50 Hz, the cohort's own trial window |
| Cognitive | 4-40 Hz, the dataset's file of that band ([built from the raw data](#files-built-from-the-raw-data)) | 0.1-64 Hz, the dataset's other file |

For motor imagery, dropping the first second after the cue means the score
cannot come from the cue's evoked response or the eye movement towards it.
SSVEP has no low-pass so that the stimulus harmonics stay in. The MOABB
datasets are cut at 128 Hz (SSVEP at 256 Hz), with the average reference
and no notch filter.

Each benchmark has four variants, the two filterings crossed with the two
ways of turning a trial's patch tokens into one vector:

| Variant | Filtering | Embedding | Paper (motor imagery) |
|---|---|---|---|
| `default` | confound filtering | patch tokens averaged | Fig. 26a, confound-filtering bars, and Fig. 26b |
| `token_flattening` | confound filtering | patch tokens concatenated in order | Fig. 5a, confound-filtering bars, and Fig. 5b |
| `no_filtering` | no filtering | patch tokens averaged | Fig. 26a, no-filtering bars |
| `no_filtering_token_flattening` | no filtering | patch tokens concatenated in order | Fig. 5a, no-filtering bars |

`neuroatlas show <benchmark>` names the figure of each variant for the
other paradigms. Other names are accepted: `confound_filtering` and
`confound_control` for `default`, `confound_filtering_token_flattening`
and `confound_control_per_patch` for `token_flattening`, and `per_patch`
for `no_filtering_token_flattening`. A results folder written under one of
these names reads as that variant.

`results` reads each result as the variant it was made with, which every
result records. A result made without confound filtering is a
`no_filtering` result, also when it is in the default's folder.

### Selecting checkpoints

`-m` works the same in `run`, `check`, `submit`, `list models` and
`models`. Give an alias, a group, a family, a checkpoint id, or a comma
list of them. Prefix a term with `-` to remove it:

```bash
neuroatlas run sleep_stage -m all_fm,-reve
```

A family selects its trained checkpoints, not its untrained baseline.
Unknown names are refused with exit status 2.

| Alias | Group | Checkpoints |
|---|---|---|
| `all_fm` | `eeg_fm` | 12 EEG foundation models |
| `all_ts` | `ts_fm` | 10 general time-series foundation models |
| `all_supervised` | `supervised` | 20 models trained with labels on other EEG data |
| `all_random` | `baseline` | 2 untrained baselines |
| `all` | | every checkpoint (44) |

### Models a benchmark leaves out

Each benchmark runs the checkpoints the paper evaluates on it. The others
are left out of `-m all` and every alias, and naming one is refused.

| Benchmark | Left out | Why |
|---|---|---|
| `epilepsy` | `sleep_transformer`, `sleepyco`, `core_sleep` | sleep-staging sequence models that take sequences of 30 s epochs, not 10 s windows |
| `epilepsy` | `eegnetv4` | motor-imagery checkpoints, evaluated on motor imagery only |
| sleep benchmarks and `brain_age` | `eegnetv4` | the same |
| sleep benchmarks and `brain_age` | `deepsoz_hem`, `seizure_transformer` | seizure-detection models, evaluated on epilepsy only |
| BCI benchmarks | `sleep_transformer`, `sleepyco`, `core_sleep` | sleep-staging sequence models |
| BCI benchmarks | `deepsoz_hem`, `seizure_transformer` | seizure-detection models |

`list benchmarks` and `show` list the same exclusions.
`neuroatlas list models --benchmark epilepsy` lists the 26 checkpoints
epilepsy evaluates. Naming a left-out checkpoint fails like this:

```
$ neuroatlas run epilepsy -m sleepyco_shhs_fold0 --dry-run
error: sleepyco_shhs_fold0 is not part of the epilepsy benchmark (sleepyco: sleep-staging sequence models that take sequences of 30 s epochs, not the benchmark's 10 s windows)
  fix: neuroatlas list models --benchmark epilepsy
```

`embed` and `probe` know nothing about benchmarks. They run whatever
`--models` names, which is why `show` writes the exclusions into each line.

## 4. Get the datasets

```bash
neuroatlas data status sleep_stage
neuroatlas data download sleep_edf_expanded --dry-run
neuroatlas data download sleep_edf_expanded --mirror aws
```

### Is it here?

`data status` takes benchmarks, domains or dataset names. It looks for
each dataset exactly where a run will.

```
$ neuroatlas data status epilepsy
dataset            host                    state                  download
bonn               web                     found (500/500 files)  downloadable
chbmit             Zenodo                  missing                downloadable
epilepsiae         campus-technologies.de  missing                manual
helsinki_neonatal  Zenodo                  found (79/79 files)    downloadable
nmt                dll.seecs.nust.edu.pk   missing                manual
siena              Zenodo                  found (41/41 files)    downloadable
sz1                the authors             missing                from the authors
sz2                the authors             missing                from the authors
tuab               TUH                     missing                credentialed
tusz               TUH                     missing                credentialed

10 datasets: 7 missing, 3 found
...
not here; where each was looked for, and the setting that points it at a copy elsewhere:
  chbmit      $DATA/chbmit      chbmit.bids_root
  epilepsiae  $DATA/epilepsiae  epilepsiae.data_root
  ...
  tusz        $DATA/tusz        tusz.raw_root
$DATA = <data root>
```

Under the table, the tool explains the words it used and prints the `fix:`
lines that apply. Add `-v` to see every path and setting, and whether a
channel map ships for the dataset.

The `state` column:

| State | Meaning |
|---|---|
| `found (n/N files)` | Every expected file is there. Shows `found (n files)` when the total is not known. |
| `partial (n/N files)` | Fewer files than a complete copy, for example after a download that stopped part-way. `run` refuses it unless you pass `--allow-partial`. |
| `empty` | The folder exists but holds no recordings. `run` refuses it too. |
| `missing` | The folder does not exist. |
| `not downloaded` | A MOABB dataset that is not in the MOABB folder. |
| `not prepared` | The preprocessed file this dataset is read from is not there. |
| `prepared` | The preprocessed file is there. |
| `not configured` | No data root is set. Run `neuroatlas config set data_root DIR`. |
| `no path` | No folder is recorded for this dataset, so it cannot be checked. |
| `unknown` | A package its reader needs is not installed. The line under the row gives the install command. |

The `download` column:

| Download | Meaning |
|---|---|
| `downloadable` | `data download` fetches it. No account needed. |
| `credentialed` | You need an approved account or a signed agreement first. For NSRR, `data download` then fetches it with your token. TUH data you download yourself. |
| `manual` | There is no download API. `data download` tells you where to request it and where to put it. |
| `from the authors` | Not public. Ask the authors. `data download` tells you where to put it. |

### Download

Run `data download` with one or more dataset names. Add `--dry-run` first
to see the plan. It makes every check a real run makes and transfers
nothing.

| Host | Datasets | What happens |
|---|---|---|
| PhysioNet | Sleep-EDF Expanded (8.1 GB), UCDDB (1.3 GB), HMC (15.7 GB), EEGMat (0.18 GB) | `wget` from physionet.org, resumable. Add `--mirror aws` to use PhysioNet's open copy on AWS, which is much faster. |
| Zenodo | CHB-MIT (21.7 GB), Siena (4.5 GB), Helsinki Neonatal (4.3 GB), DOD (58.1 GB) | Resumable and md5-checked. Zip files are unpacked and then deleted. Keep them with `--keep-archive`. You need about twice the size free while a zip exists. |
| web | Bonn (3 MB), ArithmeticTask (0.7 GB) | Bonn: five zip files from the University of Bonn's site. ArithmeticTask: one zip from OSF holding one zip per experiment. Both md5-checked and unpacked. |
| NSRR | CFS, HomePAP, MESA, MrOS, SHHS (375 GB), STAGES, WSC | The official `nsrr` tool, after a check for 50 GB of free disk (for SHHS, its 375 GB). Needs your NSRR token. Each study's `datasets/` folder comes with the recordings: its participant tables, which give brain age the participants' ages. |
| MOABB | the 14 MOABB BCI datasets | MOABB's own download into the MOABB folder. Needs the BCI install lines from [Install](#1-install). |
| TUH, and sites without an API | TUAB, TUSZ, DCSM, ISRUC, MASS, NMT, EPILEPSIAE, PhysioNet 2026, DREAMER | Prints where to request the data and where to put it. |
| the authors | SeizeIT1, SeizeIT2 | Prints where to put the files once you have them. |

`data download shhs` fetches the EDFs of both SHHS visits (8444
recordings), NSRR's annotations and the dataset tables. The reader takes
the C4-A1 EEG, resampled to 100 Hz and band-passed 0.3-40 Hz, and scores
each 30 s epoch from the annotations: stage 4 counts as N3, unscored and
movement epochs are left out, and wake beyond the largest sleep stage is
trimmed from the ends of the night. Ages come from the `shhs1-dataset` and
`shhs2-dataset` tables, each participant's age at that visit.

Measured download times: Sleep-EDF Expanded from the AWS copy took 18 min
(399 files, 8.2 GB). UCDDB from physionet.org came at about 100 KB/s and
took 3 h 35 min for 1.3 GB, so use `--mirror aws`. Helsinki Neonatal from
Zenodo took 13 min. BNCI2014_004 from MOABB took 1 min.

Each download prints a header, one line per file, and a result line:

```
$ neuroatlas data download bonn
downloading 1 dataset: bonn
[1/1] bonn: downloading from www.ukbonn.de into <data root>/bonn
  [1/5] z.zip: 591 kB in 0s, md5 ok
  ...
[1/1] bonn: downloaded, 5 files, 3.2 MB (1s)
```

Run the same command again and it skips the files that are already
complete. Stop a download with Ctrl-C and the next run continues where it
stopped. A download refused before it started (no token, not enough disk,
a missing tool) exits with status 2. A transfer that failed exits with 1.

To try a dataset before the full download, fetch only its first few
recordings:

```bash
neuroatlas data download ucddb --first 2
```

This is enough for `check` to read real data. `run` needs the whole
dataset. `--first` works for PhysioNet, NSRR and file-by-file Zenodo
downloads.

The `nsrr` tool is a Ruby gem. Install Ruby with its development headers
(`ruby-dev` or `ruby-devel`), then install the gem and put its folder on
your `PATH`:

```bash
gem install --user-install nsrr irb
export PATH="$(ruby -e 'print Gem.user_dir')/bin:$PATH"
```

### Where each dataset goes

Every dataset has its own folder, `<data root>/<dataset>`. This is where
`data download` puts it and where `run` looks for it. MOABB datasets are
the exception and use the MOABB folder. `data status -v` shows every path.

### MOABB datasets

MOABB keeps all its datasets in one folder. The tool picks it in this
order:

1. `$MNE_DATA`
2. `MNE_DATA` in MNE's config file, `~/.mne/mne-python.json` (read, never written)
3. `<data root>/mne_data`
4. `~/mne_data`

`data download`, `data status` and `run` all use the same folder.
`config show` prints which one and why. To move it, set
`export MNE_DATA=/your/mne_data`.

`run` reads the MOABB recordings directly and cuts the trials itself,
with each paradigm's band and sampling rate and each dataset's trial
window. By default that is confound filtering: for motor imagery 4-40 Hz
at 128 Hz, average reference, no notch filter, and the trial from 1 s to
4 s after the cue ([BCI variants](#bci-variants) lists the others). There
is nothing to prepare first.

`list datasets --all` also shows MOABB datasets outside the paper. Most of
them need a newer moabb than 1.2.0, which needs numpy 2. The tool refuses
those and says so.

### Optional faster copies

No dataset needs a build step. Five epilepsy datasets (Bonn, EPILEPSIAE,
SeizeIT1, TUAB and TUSZ) can be converted into a copy that is faster to
read:

```bash
neuroatlas data prepare --list
neuroatlas data prepare tuab
```

The copy goes to `<cache root>/prepared/`. `data prepare` refuses to start
until the raw data is there. `--dry-run` prints the build command,
`--dest` writes elsewhere and `--shard K/N` builds one shard.

### Files built from the raw data

The four `bci_cognitive` datasets are read from files that `data prepare`
builds from their raw data:

```bash
neuroatlas data download eegmat          # PhysioNet, 180 MB
neuroatlas data prepare eegmat
```

ArithmeticTask downloads the same way (OSF, 0.7 GB). DREAMER is on Zenodo,
where access is granted on request: `data download dreamer_valence` says
where to ask and where to put `DREAMER.mat`. `dreamer_valence` and
`dreamer_arousal` read the same file, so put a copy or a link in each
folder.

`data prepare` writes two files per dataset into `<cache root>/prepared/<dataset>/`,
both at 128 Hz with a 50 Hz notch and the average reference:
`<name>_preprocessed_trackD_steegformer.pkl`, filtered 4-40 Hz, which the
default (confound filtering) reads, and `<name>_preprocessed_steegformer.pkl`,
filtered 0.1-64 Hz, which the `no_filtering` variants read. EEGMat and
DREAMER are cut into 4 s windows, ArithmeticTask into 1 s windows.

`data status` shows `not prepared` until the files are built, with the
command that builds them. A file whose recorded band is the other
filtering's is refused, with the command that runs the variant it belongs
to. `preprocessed_path` points at a file built elsewhere.

## 5. Get the model weights

```bash
neuroatlas models status all_fm
neuroatlas models download all_fm
```

`models status` shows where each checkpoint's weights come from and
whether they are here:

```
$ neuroatlas models status all_fm
checkpoint               weights from    state
biot_pretrained          GitHub release  found
cbramod_pretrained       Hugging Face    in Hugging Face cache
...
steegformer_base         GitHub release  downloadable
...
note: 3 checkpoints not here can be downloaded
  fix: neuroatlas models download steegformer_base,steegformer_large,steegformer_small
```

| State | Meaning |
|---|---|
| `found` | The weights are on this machine. |
| `downloadable` | `models download` fetches them. A half-finished download is noted under the row. |
| `in Hugging Face cache` | Already in Hugging Face's local cache. Nothing to download. |
| `manual` | Fetch them by hand. The line under the row says from where. |
| `no weights needed` | An untrained baseline. |
| `package missing` | A Python package the model needs is not installed. The line under the row gives the install command. |

`models download` fetches what is missing and shows the bytes as they
arrive. It exits with status 1 unless every checkpoint ends up ready.

```
$ neuroatlas models download biot_pretrained,cbramod_pretrained,deepsoz_hem_pretrained
downloading 3 checkpoints
[1/3] biot_pretrained: downloading from GitHub release
[1/3] biot_pretrained: downloaded, 14 MB (2s)
[2/3] cbramod_pretrained: downloading from Hugging Face
[2/3] cbramod_pretrained: downloaded, 39 MB (3s)
[3/3] deepsoz_hem_pretrained: downloading from GitHub repository
  GPL-3.0: code and weights are not part of neuroatlas; `models download` fetches them from github.com/amruth-sn/deepsoz-hem at commit a7c13bd
[3/3] deepsoz_hem_pretrained: downloaded, 5.6 MB (1s)
3 checkpoints: 3 ready
```

A few checkpoints need a word of explanation:

- Downloading REVE accepts the REVE Responsible Use License.
- EEGPT's upstream share on Figshare works only in a browser, so the
  weights come from a bit-identical copy on Hugging Face.
- Seizure-Transformer's weights are pulled from the authors' Docker image,
  without Docker. The download streams up to 3.4 GB of image layers and
  keeps the 168 MB model.
- DeepSOZ-HEM's code and checkpoint are GPL-3.0, so the package does not
  include them. `models download deepsoz_hem_pretrained` fetches
  `baselines.py`, `deepsoz_fold4.pth_4.tar` and the upstream `LICENSE` from
  github.com/amruth-sn/deepsoz-hem at commit a7c13bd. Each file is checked
  against its recorded SHA-256 and stored in
  `<models root>/foundation/deepsoz_hem/`.
- CoRe-Sleep and SleepTransformer are release assets of this repository.

`models status -v` adds the path where each checkpoint is expected.

## 6. Check before a long run

`check` pushes one real batch through each pair of dataset and checkpoint,
with exactly the code a run uses, and then stops. It trains, caches and
downloads nothing.

```bash
neuroatlas check sleep_stage -m biot_pretrained,labram_pretrained,cbramod_pretrained,reve_pretrained,moment_small
```

On a CPU machine that has BIOT's and LaBraM's weights, but not CBraMod's
or REVE's, and no `momentfm` package:

```
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

Read the table like this:

- `forward pass` is the shape of one batch's output, windows by embedding
  size. `(32, 256) finite` means 32 embeddings of 256 numbers, none NaN or
  infinite. A batch where every window gets the same embedding, or one
  with non-finite values, counts as failed.
- `channel map` is `applied` when the dataset's map renames the electrodes
  for this model, `applied (labels as recorded)` when the model receives
  the recorded names unchanged, and `none` when the dataset has no map.
  `ruled out` and `invalid` mean what the table at the top of this guide
  says.
- A pair that did not run says why on the line under it.

A pair the channel map excludes is ruled out before any data is read:

```
$ neuroatlas check epilepsy --dataset bonn -m biot_pretrained
...
      ruled out: by the bonn channel map (not run on Bonn's single channel: BIOT reads 16 bipolar pairs, and Bonn's one channel (labelled FZ) is in none of them)
```

`check` exits with 0 when at least one pair went through and none failed.
It exits with 1 when a pair failed or when nothing could be checked. With
`--strict`, a skipped pair also gives 1. `--format json` and `csv` put the
lines under each row in a `note` field.

## 7. Run a benchmark

```bash
neuroatlas run sleep_stage -m all_fm --dataset full --dry-run
neuroatlas run sleep_stage -m biot_pretrained --debug
neuroatlas run sleep_stage -m biot_pretrained
```

The first command prints the plan. The second runs the whole pipeline on
fold 0 only. The third runs every fold.

### The plan

For each dataset, `run` executes `neuroatlas embed`, which does nothing for
embeddings already in the cache, and then `neuroatlas probe` over the
folds. `--dry-run` prints the task, the checkpoints, the folds, the number
of runs, the state of the data, and the exact commands:

```
$ neuroatlas run sleep_stage -m biot_pretrained --dry-run
benchmark    dataset             task           checkpoints  folds          runs  data                   ruled out
sleep_stage  sleep_edf_expanded  sleep_staging  1            0, 1, 2, 3, 4  5     found (197/197 files)  0

  runs  one probe fit per checkpoint and fold

the commands it runs:
  neuroatlas embed --dataset sleep_edf_expanded --folds 0,1,2,3,4 --models biot_pretrained
  neuroatlas probe --dataset sleep_edf_expanded --task sleep_staging --models biot_pretrained --output-root <output root>/sleep_stage/sleep_edf_expanded
```

If a dataset's data is missing, the plan warns that the run would fail on
it and prints the `fix:` lines. BCI plans show the folds as `LOSO (N)`,
one fold per subject.

### A whole run

Here is `brain_age`, which reuses the 30 s Sleep-EDF embeddings that
`sleep_stage` already extracted. It took 72 s on a CPU:

```
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

Each fold's result line carries that fold's headline value, under the
same label as the `results` column. Examples are `kappa 0.777` for sleep
staging, `bal_acc 0.316` for BCI and `Sens@FA_AUC(event) 0.428` on Siena.
On Bonn, whose recordings have no seizure events, the line shows
`AUROC 0.993` instead. A fold scored from saved predictions says
`ok, reused`.

If a pair fails, you see it at once in one line, for example
`error: sleep_edf_expanded/biot_pretrained (fold 2): CUDA out of memory.`
The full message is in `results.json` and in the `--log` file.

The closing lines count this run's work. They also say how many pairs the
channel map ruled out, and how many earlier results `results.json` still
holds.

### Progress

Every step shows what it is doing from its first second. Extraction goes
through loading the data, loading the weights, reading the recordings,
embedding the batches and writing the cache. A probe fold goes through
reading the embeddings, cutting the fold's split, fitting and scoring.

While fitting, the progress line counts what the probe tries:

| Benchmark | Counts |
|---|---|
| sleep staging, epilepsy | the C values of the grid |
| BCI, sleep events, diagnosis | the seeds |
| brain age | the ridge's alpha values |

Sleep staging shows no time left while it fits, because a large C takes
many times longer than a small one.

Off a terminal, for example in a pipe or a cluster job's output file,
there is no progress line. `embed` prints a line for every tenth of its
work instead, at most one every 5 s:

```
embedding 1 checkpoint on bonn: 1 run
[1/1] bonn cbramod_pretrained: embedding
[1/1] bonn cbramod_pretrained: embedding 12% (2/16 batches, 0m 14s, ~1m 44s left)
...
[1/1] bonn cbramod_pretrained: ok, 1,000 windows (2m 09s)
embed: 1 ok, 0 failed (cache: <cache root>)
```

A probe fold prints only its result line, because a LOSO probe over many
checkpoints can have thousands of folds.

### How long it takes

Measured on Sleep-EDF with BIOT:

| Step | Time |
|---|---|
| extraction, idle RTX 4500 Ada | 17 min |
| sleep-staging probe, one fold, 4 CPU cores | about 34 min |
| the same, 8 cores of an i7-9700K | 26 min |
| the same, 2 cores of an i7-9700K | 1 h 06 min |
| all five staging folds, 4 cores | about 3 h |
| `brain_age` on cached embeddings | about a minute |

Sleep staging fits six values of C in every fold, which is why it is the
slowest probe. `--dry-run` does not estimate times.

### Results on disk

Results go to `<output root>/<benchmark>/<dataset>/`. A variant adds
`/<variant>`. The folder holds:

- `results.json`, one record per checkpoint and fold. `neuroatlas results`
  reads it.
- `probes/`, one folder per checkpoint and fold with the fitted probe, its
  test predictions (`predictions.npz`) and the record it wrote
  (`result.json`).

A later run adds to `results.json` and replaces the folds it runs again. A
successful retry also removes the earlier failed record of that fold. Each
record notes when it was written (`metadata.written_at`).

### What is reused

Extraction is the expensive part, so it happens once.

For most datasets there is one cache per dataset, checkpoint and windowing,
in `<cache root>/<dataset>/<checkpoint>/all/<key>/`. It holds every window
once, and the folds are assigned when probing. Every fold, probe and
benchmark that reads the same windows uses it. `--debug` extracts the
whole dataset, so the full run reuses it, and `brain_age` reuses the 30 s
embeddings of `sleep_stage`.

Some datasets cache each fold and split separately, in
`<cache root>/<dataset>/<checkpoint>/{train,val,test}/<key>/`:

- HMC, MESA, STAGES and HomePAP
- the four `bci_cognitive` datasets
- TUAB after `data prepare tuab`
- CHB-MIT with `--set backend=hdf5`
- any run whose stride differs from its window

For these, `--debug` extracts fold 0 and the full run extracts the other
folds. TUSZ joins this group only with `--set split_mode=official`, the
dataset's own train, dev and eval split. By default it uses the paper's
five patient-level folds over all 675 patients from one cache.

The cache key changes with anything that changes the windows: window
length, stride, channels, preprocessing, or a non-default epilepsy
montage. Probe settings do not change it. If `probe` finds no cache under
its key but one under another key, its error lists the settings that
differ.

Brain age needs the subjects' ages in the cache. A Sleep-EDF cache built
without `xlrd`, or an ISRUC cache built without `openpyxl`, holds no ages.
`brain_age` then stops, names the cache folder, and asks you to delete it
and run again. Installing the package afterwards does not repair the cache.
HomePAP, MESA and STAGES keep no ages in the cache: `brain_age` reads them
from the study's NSRR table each time it probes.

### Saved predictions are reused

Every probe saves each fold's test predictions. A second `run` or `probe`
with nothing changed does not fit the fold again. It recomputes the
metrics from the saved file. Two Sleep-EDF folds that took 26 min to an
hour each to fit were scored again in 8 s:

```
$ neuroatlas run sleep_stage -m biot_pretrained --folds 0,1
...
[1/2] sleep_edf_expanded biot_pretrained fold 0: ok, reused, kappa 0.777 (0s)
[2/2] sleep_edf_expanded biot_pretrained fold 1: ok, reused, kappa 0.765 (0s)
probe: 2 ok, 0 failed (results: <output root>/sleep_stage/sleep_edf_expanded)

2 runs (checkpoint x fold): 2 ok, 0 failed; results.json also keeps 3 results of earlier runs
note: 2 folds scored from their saved predictions (same settings, weights and embeddings), not fitted again; --reprobe fits them again
  fix: neuroatlas run sleep_stage -m biot_pretrained --folds 0,1 --reprobe
```

BCI saves less time. The embed step still loads every MOABB recording to
find its cache (34 s for BNCI2014_001) before the probe reuses the fold in
about 2 s.

A fold's saved predictions are reused when all of these match:

- the probe, task and dataset settings, including the fold
- the seeds, `--pooling` and the weights file
- the embeddings, by the size and modification time of the cache files

Extracting the embeddings again therefore means a new probe. Weights are
compared by their file path, not their content. A change in how a metric
is computed takes effect, because the metrics are recomputed. A change in
how a probe is fitted is not detected, so run with `--reprobe` after one.
When saved predictions exist but are not used, a `note:` says why.

`--reprobe` on `run`, `probe` or `submit` fits every fold again and
overwrites its files.

### Folds

A fold never puts one subject on both sides of the split. Bonn has no
subject identifiers, so its folds are over clips. Most datasets have five
folds, and BCI uses one fold per subject (LOSO). SHHS's five folds are over
its 8444 recordings, with a participant's two visits in the same fold.

Where the paper's split is a file, it ships with the package and `run`
reads it. This covers CHB-MIT, SeizeIT1, SeizeIT2, EPILEPSIAE, TUSZ and
most sleep datasets. The other datasets derive their folds with a seeded
splitter, which gives the same folds on every machine.

Brain age has its own folds and keeps each dataset's published protocol:

- Sleep-EDF uses its 78 Sleep Cassette subjects in age-stratified folds
  (seed 42), not the sleep-staging folds.
- ISRUC and WSC are scored per recording, with a subject's recordings in
  one fold.
- PhysioNet 2026 holds 100 healthy and 100 impaired age-matched subjects
  out of every fold, and the ridge trains on healthy subjects only.
- CFS and MrOS use their age-stratified subject folds.
- SHHS is scored per recording, each with the participant's age at that
  visit, on its five sleep-staging folds.
- HomePAP, MESA and STAGES use five age-stratified subject folds (seed 42)
  over the recordings of their sleep-staging embeddings, one per
  participant. Each recording's age comes from the study's NSRR table,
  joined on the participant id:

  | Dataset | Table | Age column | Id |
  |---|---|---|---|
  | HomePAP | `homepap-baseline-dataset-0.2.0.csv` | `age` (baseline visit) | `nsrrid` (`homepap-lab-full-1600001` is 1600001) |
  | MESA | `mesa-sleep-dataset-0.8.0.csv` | `sleepage5c` (sleep exam) | `mesaid` (`mesa-sleep-0001` is 1) |
  | STAGES | `stages-harmonized-dataset-0.3.0.csv` | `nsrr_age` | `subject_code` (the file name, `BOGN00001`) |

  A recording whose participant has no age in the table is left out, and
  `-v` names it.

Brain age runs on ten datasets: CFS, HomePAP, ISRUC, MESA, MrOS,
PhysioNet 2026, SHHS, Sleep-EDF Expanded, STAGES and WSC.

#### Changing the folds

On the epilepsy datasets, `embed` and `probe` accept
`--set folds_manifest=none` to derive folds with the dataset's own splitter
instead of the shipped file. `--set strict_folds=false` runs on the
subjects that the data and the file share when they differ. Neither result
is comparable with the paper's.

### Run options

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

With `--limit-batches`, too few subjects can leave a fold with nothing to
test. On a LOSO BCI dataset every fold then fails.

A flag that a task does not use is refused before anything runs, never
silently ignored. For example, the diagnosis probe always fits the
published unweighted logistic regression at C = 1:

```
$ neuroatlas probe --dataset dod --task osa --aggregation mean --class-weight balanced --tune-c 1.0 --models biot_pretrained
error: the osa task does not apply --class-weight or --tune-c: it fits an unweighted logistic regression at C = 1 on each subject's mean embedding (the published diagnosis probe)
  fix: neuroatlas probe --dataset dod --task osa --aggregation mean --models biot_pretrained
```

Before extraction, `run` checks every checkpoint's weights. A checkpoint
whose weights are missing is reported by name and counted as failed, and
the others run. An interrupted extraction resumes from the batches it
already finished.

`run` exits with 0 if every run succeeded, 1 if any failed, and 2 for a
usage error such as an unknown name, a bad flag value or a refused partial
dataset.

## 8. Run on a cluster

```bash
neuroatlas submit sleep_stage -m all_fm --out runs/sleep_stage
condor_submit runs/sleep_stage/jobs.job
neuroatlas status --out runs/sleep_stage
```

For SLURM, add `--backend slurm` and submit with `sbatch`:

```bash
neuroatlas submit sleep_stage -m all_fm --out runs/ss --backend slurm \
    --partition gpu --cpus 24 --no-mem --extra --account=myproject
sbatch runs/ss/jobs.sbatch
```

`submit` writes one job per dataset and checkpoint, each a `neuroatlas run`
of that pair. It queues nothing itself. Its last line prints the command
that does. `--dataset` defaults to `full`.

Each job writes its own
`<output root>/<benchmark>/<dataset>/<model>/results.json`, so jobs never
share a file. Next to it the job's shell keeps a `job_status.json` with its
state, exit code, host and times. A job that dies before Python starts
still leaves a record. `sleep_hypnogram` gets one job per dataset.

Jobs inherit this machine's settings (the folder variables, `MNE_DATA`,
`HF_HOME` and `PYTHONPATH`, never a token) and run offline. So `submit`
checks every pair first. A pair whose data or weights are missing, or that
the channel map rules out, gets no job and is counted with the reason. Add
`-v` to list each one. `--force` writes jobs for the pairs with missing data
or weights anyway, never for ruled-out ones.

On a machine with only Sleep-EDF and UCDDB, and BIOT's weights but not
REVE's:

```
$ neuroatlas submit sleep_stage -m biot_pretrained,reve_pretrained --out runs/ss
2 jobs: 2 to queue (--mode cached), 2 missing
28 pairs without a job: 26 data missing, 2 weights not downloaded (-v lists them)
note: 28 pairs were skipped for missing data or weights; --force writes their jobs anyway (they fail where those are missing)
  fix: neuroatlas data download cfs dcsm dod hmc hpap_lab_full isruc mass mesa mros physionet2026 shhs stages wsc
  fix: neuroatlas models download reve_pretrained
...
queue them: condor_submit <current folder>/runs/ss/jobs.job
```

A `GPUs:` line before the last one gives the lowest GPU compute capability
the installed PyTorch supports (7.5 for the CUDA 13 build). HTCondor jobs
require it. SLURM has no standard attribute for it, so it is a comment
there.

The `--out` folder holds `jobs.json` (what was planned and skipped), one
script per job in `jobs/`, `jobs.txt` (the scripts this submit queues),
`jobs.job` or `jobs.sbatch`, and `logs/`.

### Following the jobs

`status` asks `condor_q` or `squeue` when they are available, so run it
where you submitted. It also reads the HTCondor event logs and the jobs'
own files. `--no-scheduler` uses the files only. Failures are listed with
their host and log, and `-v` gives one row per job.

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

### Submitting again

By default (`--mode cached`) `submit` queues only `missing` jobs, so
running it again never repeats work. `--mode retry` also takes failed,
partial, exited, stopped and removed jobs. `--mode all` takes everything.
A job still in the queue is never queued twice. Jobs reuse saved
predictions like `run` does. `submit --reprobe` writes jobs that fit every
fold again.

### Resources

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

`submit` warns about a flag the chosen backend ignores. It also warns when
the Python or a `PYTHONPATH` entry is under `/tmp`, `/var/tmp` or
`/dev/shm`, since other machines cannot see it.

If one host keeps failing your jobs, keep them off it:

```bash
neuroatlas submit sleep_stage -m all_fm --out runs/ss \
    --requirements 'Machine =!= "host.example.org"'
```

`condor_submit -dry-run` checks the submit file's syntax, not your pool's
policy.

## 9. Read the results

```bash
neuroatlas results sleep_stage
neuroatlas results sleep_stage -v
neuroatlas results sleep_stage 'runs/old/**/results.json' --format md
```

`results` reads every `results.json` under `<output root>/<benchmark>/`,
from local runs and cluster jobs alike. You can also name files, folders
or quoted globs. It prints one row per dataset, variant and checkpoint.

```
$ neuroatlas results sleep_stage
sleep_stage: Cohen's kappa over 30 s epochs, 5 stages (W, N1, N2, N3, REM), each fold's test subjects pooled, unscored epochs left out; higher is better; chance 0
folds: 5 subject-level folds; the probe's C is chosen from 0.001-100 on validation Cohen's kappa, unweighted loss; ± = population SD over the folds
also: bal_acc = balanced accuracy, macro_F1 = macro-F1
dataset             model            kappa  ±      folds  bal_acc  macro_F1
sleep_edf_expanded  biot_pretrained  0.758  0.016  5/5    0.669    0.675
```

The first lines state what the headline is computed over, what a fold is,
what ± means, and the names of the other columns. Then:

- The first number is the headline's mean over the folds that succeeded.
  The second is its SD, `n/a` with one fold.
- `folds` is the folds done out of the protocol's folds, such as `5/5`,
  `1/5` after `--debug`, or `LOSO 1/9`. A warning tells you when a mean
  covers fewer folds than the protocol, because it is not comparable with a
  full run.
- `chance` appears where a guess scores differently per dataset. For BCI it
  is 1 over the number of classes. For the sleep-event benchmarks it is
  the share of positive test epochs.
- A row whose headline is `n/a` gives the reason on the line under it.

`--variant NAME` shows one variant's rows. A `variant` column appears once
a variant other than the default has results.

### Per-fold values

`-v` adds every fold's value under each row, with the C the probe chose
where it picks C from a grid:

```
$ neuroatlas results epilepsy -v
...
dataset  model               Sens@FA_AUC(event)  ±      folds  AUROC(window)  AUPRC(window)  bal_acc  MCC    event_F1  sens@1FA/h
siena    cbramod_pretrained  0.496               0.159  5/5    0.813          0.122          0.621    0.146  0.140     0.118
      by fold: 0: 0.428 (C 0.01), 1: 0.516 (C 10), 2: 0.641 (C 0.001), 3: 0.566 (C 0.01), 4: 0.227 (C 0.001)
```

On epilepsy each fold's value is that fold's own Sens@FA AUC. The headline
is the AUC of the folds' median curve, so it is not the mean of the
per-fold values. `-v` also shows why failed folds failed and how many folds
are still to run.

In JSON a missing number is `null`. Every row has the fields `benchmark`,
`metric`, `variant`, `status`, `chance`, `C`, `per_fold`, `n_folds`,
`n_expected`, `n_failed`, `failures`, `errors` and `note`.

### Duplicates

A result recorded twice counts once, for example from a quick `run` and a
later cluster job of the same dataset, variant, model, fold and task. The
newest file wins, and a warning says how many duplicates were dropped. Two
variants of the same model and fold are two results.

`results` exits with 0 when there is something to show, 1 when there are
no results, and 2 for a usage error.

### Saved predictions and `rescore`

Every probe writes each fold's test predictions to `predictions.npz` in
the fold's probe folder,
`<output root>/<benchmark>/<dataset>/probes/<dataset>/<model>/<key>/`.
Every metric a result records is computed from this file. A metric added
or corrected later reaches old results without probing again:

```bash
neuroatlas rescore sleep_stage
neuroatlas rescore epilepsy --dataset siena -m cbramod_pretrained
```

```
$ neuroatlas rescore sleep_stage --dataset sleep_edf_expanded -m biot_pretrained
dataset             model            fold  result          kappa before  after
sleep_edf_expanded  biot_pretrained  0     rescored, same  0.777328      0.777328
...
sleep_edf_expanded  biot_pretrained  4     rescored, same  0.752377      0.752377

5 rescored from saved predictions: 0 changed, 5 the same; 1 results.json file rewritten
```

`rescore` fits nothing. It recomputes each record's metrics, keeps its
metadata, adds `metadata.rescored_at`, and rewrites `results.json`. A
record whose fold has no saved predictions is listed as `skipped`, with
the command that probes it again. `rescore` takes `--dataset`, `-m`,
`--variant`, `--output-root` and `--format` like `results`.

### The predictions file

Load it with `np.load("predictions.npz")`. It contains no pickles.
`help(neuroatlas.predictions)` describes it in full.

| Entry | Contents |
|---|---|
| `format`, `dataset`, `checkpoint_id`, `task`, `fold` | what made the file. `fold` is empty for a dataset with one fixed split. |
| `y_true`, `y_pred` | the true and predicted class, or value, for each test row |
| `y_proba`, `classes` | class probabilities (rows by classes) and the class of each column |
| `y_score` | for binary tasks, the positive class's score, which AUROC, AUPRC and Sens@FA read |
| `subject_id`, `recording_id`, `session_id`, `epoch_index`, `trial_idx`, `window_start_s` | the row identifiers that apply |
| `window_s`, `threshold` | epilepsy only: seconds per window, and the decision threshold tuned on validation |
| `<group>/<column>` | further probes of the fold, such as one per arousal threshold, or every ridge alpha for brain age |
| `seed_y_pred`, `seed_y_score` | every seed's predictions, with `probe --seed-mode shared` |
| `info` | JSON with what the probe chose on validation, the settings the metrics read, and what the probe was given |

A row is what the metric counts: a 30 s epoch for the sleep tasks, a 10 s
window for epilepsy, a trial for BCI, and a subject or recording for
diagnosis and brain age. A Sleep-EDF staging fold takes about 1.8 MB.

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score
from neuroatlas import predictions

z = np.load("predictions.npz")
cohen_kappa_score(z["y_true"], z["y_pred"])   # one fold, by hand
predictions.score("predictions.npz")          # every metric the task records
```

## 10. Use it from Python

```python
from neuroatlas import api

api.benchmarks()                                     # list benchmarks
api.models("all_fm")                                 # models status all_fm
api.data_status("sleep_stage")                       # data status
api.plan("sleep_stage", "all_fm", datasets="full")   # run --dry-run
api.check("sleep_stage", "biot_pretrained")          # check
api.run_benchmark("sleep_stage", "biot_pretrained", debug=True)
api.results("sleep_stage")                           # results
api.rescore("sleep_stage")                           # rescore
```

Each function returns a pandas DataFrame with the columns of the command's
`--format json`. Missing numbers are `None`, or NaN in a float column.
`results()` takes `variant=`. `run_benchmark()` returns the results of
that call only and takes `debug=`, `limit_batches=`, `cache_root=`,
`output_root=`, `online=` and `reprobe=`. `rescore()` takes `datasets=`,
`models=`, `output_root=` and `variant=`.

Importing `neuroatlas.api` reads the same settings file and does not
import torch. No API function downloads. `check()` and `run_benchmark()`
run offline unless you pass `online=True` to `run_benchmark()`. The API
leaves `HF_HUB_OFFLINE` alone, because the Hugging Face library reads it
once per session. The weights check before extraction still refuses a hub
model that is not cached.

## 11. Where files go

Everything goes under `$NEUROATLAS_HOME` (`~/.neuroatlas`) unless you set
other folders.

| What | Where |
|---|---|
| settings and tokens | `$NEUROATLAS_HOME/config.yaml`, `hf_token`, `nsrr_token`, `github_token` |
| raw datasets | `<data root>/<dataset>` |
| MOABB datasets | the MOABB folder, one subfolder per dataset |
| optional faster copies | `<cache root>/prepared/` |
| weights | `<models root>/foundation/` (DeepSOZ-HEM in `foundation/deepsoz_hem/`), `<models root>/shhs/` (CoRe-Sleep, SleepTransformer, SleePyCo), `<models root>/supervised/` (Seizure-Transformer) |
| Hugging Face models | Chronos, MOMENT and Moirai in `<models root>/foundation/huggingface_cache`. The others in the Hugging Face cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`). |
| embeddings | `<cache root>/<dataset>/<checkpoint>/all/<key>/`, or per fold and split for [some datasets](#what-is-reused) |
| `--limit-batches` runs | `<cache root>/_limited/` and `<output root>/_limited/<benchmark>/<dataset>/` |
| results of `run` | `<output root>/<benchmark>/<dataset>/results.json` and `probes/` |
| one fold's probe | `<output root>/<benchmark>/<dataset>/probes/<dataset>/<model>/<key>/`, with `predictions.npz`, `result.json`, `probe.pkl` and `metadata.json` |
| results of cluster jobs | `<output root>/<benchmark>/<dataset>/<model>/results.json` and `job_status.json` |
| hypnograms | `<output root>/sleep_hypnogram/<dataset>/`, with `hypnograms.json`, `hypnogram_features.csv`, `hypnogram_features_summary.csv` and `results.json` |
| cluster job files | the `--out` folder |
| `check` | a temporary folder, deleted when it ends |
| epilepsy recording statistics | `<cache root>/recording_stats/` |
