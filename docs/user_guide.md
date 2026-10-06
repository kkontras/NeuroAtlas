# Running NeuroAtlas with the `neuroatlas` command

From an empty machine to a benchmark's scores: install, set the tool up
once, get datasets and model weights, check that everything fits, run a
benchmark here or on a cluster, and read the scores. Every flag is in [cli.md](cli.md),
which is generated from the code.

Every run answers one question: **benchmark × models × datasets**.

- **Benchmark** -- what is predicted and how it is scored: `sleep_stage`,
  `brain_age`, `epilepsy`, `bci_motor_imagery`, ... (`neuroatlas list benchmarks`).
- **Models** -- checkpoints, by id, family or alias: `-m all_fm`,
  `-m reve,biot_pretrained`.
- **Datasets** -- `--dataset single` (the benchmark's one quick dataset),
  `full` (every dataset of the benchmark) or a comma list of dataset names.

The steps are always the same, and each command says what is missing and
which command fixes it:

    config init → list → data → models → check → run  (or submit → status) → results

## Words you will meet

One word per idea, the same in the tables, the JSON and the Python API.

| word | meaning |
|---|---|
| **benchmark** | One protocol of the paper: what is predicted, on which datasets, how it is scored. 12 of them (`list benchmarks`). |
| **dataset** | One corpus, by its name (`sleep_edf_expanded`, `chbmit`). The paper's evaluation has 43 (`list datasets`). The column is `dataset` everywhere. |
| **suite** | Which of a benchmark's datasets a command uses: `single` (its one quick dataset) or `full` (all of them). `run` and `check` default to `single`; `submit` and `show` to `full`. |
| **variant** | Another cell of the same benchmark (`--variant per_patch`); `show` lists them. Its results are its own rows in `results`, never merged with the default's. |
| **checkpoint** | One set of weights, e.g. `biot_pretrained`. "Model" means a checkpoint; the column is `model` (`checkpoint` in `list models` and `models status`). |
| **family** | The architecture behind checkpoints, e.g. `reve`. In `-m` a family means its trained checkpoints, not its untrained baseline. |
| **alias**, **group** | A name for a set of checkpoints: `all_fm` (= group `eeg_fm`), `all_ts` (`ts_fm`), `all_supervised` (`supervised`), `all_random` (= `baseline`, the untrained ones), `all` (every ready checkpoint). |
| **access** | How a dataset is obtained: `physionet`, `zenodo`, `url`, `nsrr`, `moabb`, `tuh`, `manual`, `internal`. The same word in `list datasets` and `data status`. |
| **data root**, **cache root**, **output root**, **models root** | The four folders the tool reads and writes; see [Settings](#2-settings-and-tokens). |
| **prepared file** | A file `data prepare` builds from a raw corpus, under `<cache root>/prepared/`. |
| **embedding** | What a frozen model turns one window of EEG into. Models are never fine-tuned. |
| **embedding cache** | Where embeddings are saved, keyed by dataset, checkpoint and the windowing; see [What is reused](#what-is-reused). |
| **probe** | A linear model trained on the embeddings (logistic regression; ridge for brain age). Its score is the model's score. |
| **predictions** | What a probe predicts on a fold's test rows, saved as `predictions.npz` in its probe folder; every metric is computed from them, and `rescore` recomputes them. See [Saved predictions and `rescore`](#saved-predictions-and-rescore). |
| **fold** | One split of a dataset's subjects into train, validation and test, never sharing a subject (Bonn, which ships no subject ids, splits by clip). Most datasets have 5 folds, from a frozen file or a seeded splitter; BCI is leave-one-subject-out (`LOSO (N)`, one fold per subject); SHHS ships one fixed split (`fixed split`). |
| **channel map** | Per dataset: which electrodes each model family gets, under which names. It can mark a family `skip`; that pair is **n/a**. A family the map has no entry for is **invalid**: `check`, `run` and `submit` report the pair and never run it. |
| **n/a** | Not applicable, never zero: a pair the channel map skips, a spread over one fold, a model missing from part of a suite. `null` in JSON. |
| **dummy** | What a trivial predictor scores (0.2 balanced accuracy for 5 sleep stages). Epilepsy's event-level headline has none: a constant predictor's score depends on the split. |
| **normalized** | `(score − dummy) / (1 − dummy)`: 0 is the trivial guess, 1 is perfect. Only for benchmarks with a fixed dummy and a higher-is-better metric. |
| **offline** | Downloads switched off; see [the offline rule](#the-offline-rule). |

## 1. Install

The full recipe, with what each step does to the environment, is in the
[README](../README.md#install). In short, Python 3.10 or newer (3.11 tested),
in a clone of the repository:

```bash
conda create -n neuroatlas python=3.11 -y && conda activate neuroatlas   # or: python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[fm]" -c requirements-fm.txt     # the paper's exact versions; torch's default build is CUDA 13
```

The distribution is named `neuroatlas-bench` and is not on PyPI yet; the
command and the import stay `neuroatlas`. The PyPI project named
`neuroatlas` is an unrelated one: do not install it.

What each model family and dataset needs beyond `[fm]`:

| for | install | note |
|---|---|---|
| EEG foundation models, supervised baselines, sleep and epilepsy datasets | `[fm]` | includes `xlrd` (Sleep-EDF ages) and `openpyxl` (ISRUC ages) |
| Chronos | `pip install -e ".[fm,ts]" -c requirements-fm.txt` | |
| MOMENT | `pip install --no-deps "momentfm==0.1.4"` | `pip check` then lists momentfm's three declared pins; expected |
| Moirai | a separate environment: `requirements-tsfm.txt` | Python 3.10, torch 2.4.1 |
| the 14 MOABB BCI datasets | `pip install -e ".[fm,bci]" -c requirements-fm.txt`, then `pip install --no-deps "moabb==1.2.0"` | stays on numpy 1.26.4; `pip check` lists moabb's declared caps, which are harmless; Dreyer2023 and Kim2025BetaRange, newer than moabb 1.2.0, ship inside the package |
| all of the above but Moirai | `pip install -e ".[fm,bci,ts]" -c requirements-fm.txt`, then `pip install --no-deps "moabb==1.2.0" "momentfm==0.1.4"` | |

`neuroatlas models status` reports a model whose package is missing as
`package missing`, with the install line.

## 2. Settings and tokens

```bash
neuroatlas config init --data-root /data/eeg
neuroatlas config show
```

`config init` writes `~/.neuroatlas/config.yaml` with all four roots spelled
out, so nothing moves later. `$NEUROATLAS_HOME` replaces `~/.neuroatlas`.
The defaults depend on `$NEUROATLAS_HOME` only: not on the working
directory, and not on whether the package is an editable install or a wheel.

| setting | what | environment variable | default |
|---|---|---|---|
| `data_root` | raw datasets | `EEG_DATA_ROOT` | none: `config init` requires `--data-root` |
| `cache_root` | embeddings and prepared files; can grow large | `EEG_CACHE_ROOT` | `$NEUROATLAS_HOME/artifacts/embedding_cache` |
| `output_root` | results | `NEUROATLAS_OUTPUT_ROOT` | `$NEUROATLAS_HOME/artifacts/benchmarks` |
| `models_root` | model weights | `NEUROATLAS_MODELS_ROOT` | `$NEUROATLAS_HOME/artifacts/models` |

- `neuroatlas config set cache_root /scratch/cache` changes one root;
  `config unset KEY` returns it to its default. `set` stores absolute paths.
- An environment variable beats the file, so a cluster job can redirect one
  run without touching it. A relative value in a variable is taken against
  the current directory; a relative path typed into the file by hand is
  taken against the file's folder. `submit` writes the variables that are
  set into every job script.
- A dataset kept outside its default folder:
  `neuroatlas config set ucddb.data_root /mnt/ucddb`. The key differs per
  dataset; `data status` prints it for every dataset it does not find, and
  `config set` refuses a dataset or key that nothing reads, with the keys
  that dataset does read. The raw-data keys today: `data_root` for most datasets;
  `bids_root` for `chbmit` and `siena`; `raw_root` for `helsinki_neonatal`,
  `nmt`, `tuab` and `tusz`; `raw_dir` for `bonn`; `preprocessed_path` for the
  four `bci_cognitive` datasets. MOABB datasets take no key (see
  [MOABB](#moabb-the-14-bci-datasets)).
- An unknown top-level setting in `config.yaml` (a misspelt `cahce_root`)
  stops every command except `config`, which warns and names the
  `config unset` that removes it. An unknown dataset key only warns: it has
  no effect.

`config show` prints each root with its state (exists, writable or
read-only, missing) and where its value came from (`$VARIABLE`, `file` or
`default`), the MOABB folder, the offline rule, and where each token was
found -- never its value.

**Tokens** go next to the config file, `chmod 600`: `neuroatlas config token hf` (or `github`, `nsrr`) asks for one without showing it and saves it there; `--remove` deletes it. `config show` warns
about a token file others can read. Only gated downloads need one.

| token | for | looked for, in this order |
|---|---|---|
| Hugging Face | gated model repositories | `$HF_TOKEN`, the file named by `$NEUROATLAS_HF_TOKEN_FILE`, `$NEUROATLAS_HOME/hf_token`, `<checkout>/.secrets/hf_token`, `~/.cache/huggingface/token` |
| NSRR (sleepdata.org/token) | the NSRR sleep cohorts | `$NSRR_TOKEN`, `$NEUROATLAS_HOME/nsrr_token` |
| GitHub | release assets of a private repository | `$GITHUB_TOKEN`, `$GH_TOKEN`, `$NEUROATLAS_HOME/github_token` |

On 2026-10-02 none of the Hugging Face repositories the registry names was
gated, so no model needed a Hugging Face token. The GitHub token is needed
for CoRe-Sleep and SleepTransformer (see [Known gaps](#known-gaps)).

### The offline rule

Only `data download`, `data prepare`, `models download` and the legacy
`fetch --download` and `prepare` use the network. Every other command, and
every cluster job, runs with downloads switched off
(`NEUROATLAS_OFFLINE=1`, `HF_HUB_OFFLINE=1`), so a missing checkpoint fails
at once, by name, with the command that fetches it, instead of stalling a
job halfway. `neuroatlas --online <command>` lifts that for one invocation;
global options (`--online`, `--log FILE`, `-v`) may stand before or after
the command. On screen every command shows its own output, progress bars
and warnings; the libraries' INFO lines (each model's loading banner, the
loader's per-dataset line, weight reports) and Python warnings appear only
with `-v`, and a closing `note:` line says how many were left out. `--log FILE`
keeps all of it.

Every message that is not ordinary output starts with what it is, on stderr:

| first word | meaning |
|---|---|
| `error:` | the command, or one pair or row of it, failed or was refused (exit 1, or 2 for a refusal or a usage error) |
| `warning:` | the command goes on, but results may be affected |
| `note:` | information, neither of the two |

The remedy, when there is one, is on its own line below, the command to run
first: `  fix: neuroatlas models download neurorvq_eeg_pretrained`. A mistyped
command, option or name gets the corrected command as its fix
(`neuroatlas chek epilepsy` -> `fix: neuroatlas check epilepsy`). A long
message from a library (a CUDA out-of-memory error, an HTTP error, a list of
channel labels) is cut to its first sentence, `(-v: full message)`; `-v` and
the `--log FILE` keep it whole. Colour (red `error:`, yellow `warning:`) only
on a terminal, and never with `NO_COLOR` set or in a `--log` file. With `--online`, `run` fetches a missing Hugging Face or
release checkpoint itself; `check` still downloads nothing.

The rule covers MOABB too: offline, a MOABB dataset that is not all in the
MOABB folder stops the run at its first read, naming
`neuroatlas data download <dataset>`, instead of being fetched from MOABB's
servers mid-run. With `--online`, MOABB fetches it.

## 3. See what exists

```bash
neuroatlas list benchmarks          # what each predicts, datasets, headline metric, dummy
neuroatlas show sleep_stage         # one benchmark explained, and the commands it runs
neuroatlas list datasets --grep sleep
neuroatlas list models all_fm
neuroatlas list aliases             # -v lists each alias's members
neuroatlas list tasks
```

Every listing takes `--format table|csv|md|json`. In JSON a missing number
is `null`, and the lines a table prints under a row are its `note` field.

`neuroatlas show <benchmark>` prints the `embed` and `probe` command lines
the benchmark expands to. They are equivalent to the lines of
`run/default_runs.sh`, the record of what the paper ran, spelled
`neuroatlas <verb>` instead of `python -m neuroatlas.entrypoints.<verb>`.
Without `-m` they name
`--models all`, every ready checkpoint (44) -- on epilepsy
`--models all,-sleep_transformer,-sleepyco,-core_sleep` (38), see below.

**Selecting models** (`-m`, the same in `run`, `check`, `submit`, `list
models` and `models`): an alias or group, a family, a checkpoint id, or a
comma list of them, read left to right; `-name` removes what the terms
before it selected (`-m all_fm,-reve`; a family removes every checkpoint of
it). Unknown and planned names are refused (exit 2).

**Models a benchmark leaves out.** A benchmark can declare model families
it does not evaluate (`excluded_models` in its catalog file; `show` and
`list benchmarks` print them). Epilepsy leaves out the sleep-staging
sequence models SleepTransformer, SleePyCo and CoRe-Sleep (families
`sleep_transformer`, `sleepyco`, `core_sleep`, their `_seq1` checkpoints
included): their wrappers take only sequences of 30 s epochs and refuse the
10 s windows. On epilepsy `run`, `check`, `submit`, `show` and the Python
API leave them out of `-m all`, `all_supervised` and every other alias or
group, and say so under the table; naming one, as a family or a checkpoint
id, is refused (exit 2):

```
$ neuroatlas run epilepsy -m sleepyco_shhs_fold0 --dry-run
error: sleepyco_shhs_fold0 is not part of the epilepsy benchmark (sleepyco: sleep-staging sequence models, built on sequences of 30 s sleep epochs; their wrappers refuse the benchmark's 10 s windows)
  fix: neuroatlas list models --benchmark epilepsy
```

`neuroatlas list models --benchmark epilepsy` lists the 38 it evaluates;
`results epilepsy` does not summarise results of the three (a line names
the checkpoints it left out). The verbs (`embed`, `probe`) know no benchmark: they run
what `--models` names, which is why `show` spells the scope out.

## 4. Get the datasets

```bash
neuroatlas data status sleep_stage                        # benchmarks, domains or dataset names
neuroatlas data download sleep_edf_expanded --dry-run     # the plan, and every check
neuroatlas data download sleep_edf_expanded --mirror aws
```

`data status` resolves each dataset's folder exactly as a run will. Its
columns: `access`; `state`; `download`, what `data download` would do
(`automatic`, `with NSRR token`, `instructions`, `from the authors`,
`refused`); and `map`, whether a channel map ships for the dataset. Under
the table it counts the states, lists, for each dataset it did not find, the
folder it looked in and the setting that points it at your copy
(`neuroatlas config set <setting> DIR`), and defines `$DATA` (your data
root); one `note:` names the NSRR cohorts that cannot be downloaded without
a token. `-v` adds the path and the key as columns; `--format json` always
has them.

| state | meaning |
|---|---|
| `found (n/N files)` | every expected file is there (`found (n files)` when the total is not known) |
| `partial (n/N files)` | some files, fewer than expected: an interrupted download? `run` refuses it unless `--allow-partial` |
| `empty` | the folder exists but holds no recordings; `run` refuses it too |
| `missing` | the folder does not exist |
| `not downloaded` | a MOABB dataset that is not in the MOABB folder yet |
| `not prepared` | the reader needs a prepared file that is not there (the raw data may be) |
| `prepared` | that prepared file is there |
| `not configured` | no data root set: `neuroatlas config init --data-root DIR` |
| `no path` | the manifest names no folder for it |
| `unknown` | a prepared file's state could not be checked: a package is missing, and the note names it |

`data status` lists only the states that occur, each with its meaning.

### What `data download` does, by access

| access | datasets | what happens |
|---|---|---|
| `physionet` | Sleep-EDF Expanded (8.1 GB), UCDDB (1.3 GB), HMC (15.7 GB), EEGMat (0.18 GB; the benchmark does not read these raw files, see `data prepare`) | `wget -r -N -c -nv` from physionet.org into the folder the reader expects: one line per file, resumable. `--mirror aws` fetches the same files from PhysioNet's open-data copy on AWS instead: no account, one line per file, a partial file resumes, md5 checked where the copy publishes one. |
| `zenodo` | CHB-MIT (21.7 GB), Siena (4.5 GB), Helsinki neonatal (4.3 GB), DOD (58.1 GB) | in Python: resumable, md5-checked, one line per file; a zip is unpacked where the reader expects it and then deleted (`--keep-archive` keeps it). Needs about twice the size free while the zip exists. |
| `url` | Bonn (3 MB) | the five zips from the university's site, md5-checked, unpacked |
| `nsrr` | CFS, HomePAP, MESA, MrOS, STAGES, WSC | the official `nsrr` tool, after a free-disk check (50 GB). The token is fed to the tool's prompt on stdin: never on the command line, never in the environment, never printed. `NEUROATLAS_NSRR_DATASETS=wsc,cfs` refuses, up front, cohorts your token is not approved for. SHHS is refused: its reader needs the CoRe-Sleep preprocessed version, which no script here builds. |
| `moabb` | the 14 MOABB BCI datasets | MOABB's own download into the MOABB folder, one line per subject. Needs the `[bci]` extra and `pip install --no-deps "moabb==1.2.0"` ([Install](#1-install)); without MOABB it is refused (exit 2). |
| `tuh`, `manual` | TUAB, TUSZ, DCSM, DREAMER (2), Epilepsiae, ISRUC, MASS, NMT, PhysioNet 2026 | prints where to request it, what the reader expects, and where to put it (exit 0) |
| `internal` | ArithmeticTask, SeizeIT1, SeizeIT2 | the same: not public, obtain it from the authors |

Measured transfers: Sleep-EDF Expanded from the AWS copy, 399 files and 8.2
GB in 18 min; from physionet.org the tester saw about 100 KB/s (UCDDB, 1.3
GB, 3 h 35 min). EEGMat from the AWS copy, 76 files in 2 min. Helsinki from
Zenodo, 85 files in 13 min. BNCI2014_004 from MOABB, 9 subjects in 1 min.

A download refused before anything ran (no token, not enough disk, a
missing tool, SHHS) exits 2; a transfer that failed exits 1. `--dry-run`
makes the same checks as a real run and transfers nothing. Running a
finished download again checks each file and says `already complete`.

The `nsrr` tool is a Ruby gem: `gem install --user-install nsrr irb`. Not
verified here: the tester found that it needs Ruby's development headers to
build, and that the gem's `bin` folder must be on `PATH`.

The default folders under the data root follow the authors' layout
(`$DATA/data/sleep-edf-database-expanded-1.0.0`,
`$DATA/data/stvincent_ucddb/files/ucddb/1.0.0`, ...); the Zenodo cohorts
go to `$DATA/<dataset>/` (`$DATA/chbmit/BIDS_CHB-MIT`,
`$DATA/siena/BIDS_Siena`, `$DATA/helsinki_neonatal`). `data download` and
`data status` agree on them; `data status -v` shows each.

### MOABB (the 14 BCI datasets)

MOABB keeps its data in one folder, outside the data root's per-dataset
keys. NeuroAtlas resolves it once, in this order: `$MNE_DATA`; `MNE_DATA`
in MNE's own config file (`~/.mne/mne-python.json`, read, never written);
`<data root>/mne_data`; `~/mne_data`. It exports the result as `MNE_DATA`,
with `MNE_DONTWRITE_HOME=true`, so `data download`, `data prepare`,
`data status` and `run` use the same folder and MNE never writes its config
file. `config show` prints the folder and where it came from. To move it,
`export MNE_DATA=/your/mne_data`.

Inside that folder, moabb 1.2.0 keeps each dataset in its own subfolder
(BNCI2014_001: `MNE-bnci-data/database/data-sets/001-2014`); moabb 1.6 and
later put it under `NEMAR/<id>`. Each BCI manifest names both, and
`data status` checks both and, for a dataset not downloaded yet, names the
one the installed moabb will use.

`run` and `embed` read the MOABB recordings directly and epoch them
themselves, with the paradigm's band and rate (motor imagery: 4–40 Hz at
128 Hz, average reference, no notch) and each trial cut where the paper cut
it (motor imagery: the 4 s after the cue; `--variant confound_control`:
1–4 s). Nothing needs preparing first (see [data prepare](#data-prepare)).
Most of the other MOABB datasets `list datasets --all` registers (none of
them in the paper) are newer than moabb 1.2.0: they fail with a message
saying so.

### `data prepare`

```bash
neuroatlas data prepare --list            # which datasets have a build step
neuroatlas data prepare tuab
```

No dataset needs a build step. Ten have an optional one. Five epilepsy
cohorts (Bonn, Epilepsiae, SeizeIT1, TUAB, TUSZ) read their raw corpus
without it; the prepared files are a faster path. The five MOABB
motor-imagery cohorts BNCI2014_001, BNCI2014_004, BNCI2015_001, Shin2017A
and Weibo2014 can have a pickle built (4–40 Hz, 100 Hz), but nothing a run
does reads it: `data status`, `check` and `submit` judge them by their raw
data, and `run` epochs the MOABB recordings itself. The other 33 need
nothing: `data prepare` says so and exits 0.

`data prepare` refuses (exit 2) until the raw data is there, and writes
under `<cache root>/prepared/` (BNCI2014_004: a 30 MB pickle in 77 s).
`--dry-run` prints the builder command, `--dest` puts the file elsewhere,
`--set KEY=VALUE` overrides a dataset setting, `--shard K/N` builds one
shard (epilepsy builders).

The four `bci_cognitive` datasets read preprocessed files the authors hold;
no script here builds them. `data status` says which file and the
`preprocessed_path` key to point at it.

## 5. Get the models

```bash
neuroatlas models status all_fm
neuroatlas models download all_fm
```

| state | meaning |
|---|---|
| `found` | the weights are on disk |
| `auto` | not here; `models download` fetches them (GitHub release, files of a GitHub repository at a pinned commit, Hugging Face, Google Drive, a Docker image); a half-finished download says `incomplete: missing ...` in its note |
| `hub` | a Hugging Face model read from the hub cache; not cached yet |
| `hub (cached)` | ... already in the cache |
| `manual` | fetch by hand; the note says from where and where to put it |
| `nothing needed` | an untrained baseline that needs no files |
| `package missing` | the Python package the wrapper imports is not installed; the note has the install line and the weights' own state |
| `planned` | the wrapper is not ready (`brain_age_cnn_v2_pretrained`) |

`models download` fetches what a selection lacks and exits 1 unless every
model ends up usable; a model whose package is missing still gets its
weights. Some notes worth knowing:

- REVE: downloading it accepts the REVE Responsible Use License.
- EEGPT: the upstream Figshare share is browser-only; it comes from a
  bit-identical copy on Hugging Face.
- Seizure-Transformer: pulled out of the authors' Docker image without
  Docker (it streams up to 3.4 GB of image layers and keeps the 168 MB
  model).
- DeepSOZ-HEM: its model code and checkpoint are GPL-3.0, so the package
  does not carry them. `models download deepsoz_hem_pretrained` fetches
  `baselines.py`, `deepsoz_fold4.pth_4.tar` and upstream's `LICENSE` from
  github.com/amruth-sn/deepsoz-hem at commit a7c13bd (5.5 MB), each checked
  against its recorded SHA-256, into `<models root>/foundation/deepsoz_hem/`;
  the wrapper checks the code and checkpoint again and imports that
  `baselines.py`. The status note names the licence in every state.
- CoRe-Sleep and SleepTransformer: release assets of a private repository;
  without a GitHub token that can read it, the download fails.

`-v` shows where each checkpoint is expected: a path under the models root,
or, for a hub model, its repository id; hub models are cached as
[Where things live](#where-things-live) says.

## 6. Check before spending GPU hours

```bash
neuroatlas check sleep_stage -m biot_pretrained,labram_pretrained,cbramod_pretrained,reve_pretrained,moment_small
```

One real batch through each dataset × model pair, with exactly the code a
run uses, then it stops: nothing trained, cached or downloaded. On Sleep-EDF:

```
dataset             model               data                   weights          channel_map  forward           time
sleep_edf_expanded  biot_pretrained     found (197/197 files)  found            applied      (32, 256) finite  2.8s
sleep_edf_expanded  labram_pretrained   found (197/197 files)  auto             applied      -                 0.0s
      skipped: weights not downloaded
        fix: neuroatlas models download labram_pretrained
sleep_edf_expanded  cbramod_pretrained  found (197/197 files)  hub (cached)     applied      (32, 200) finite  1.0s
sleep_edf_expanded  reve_pretrained     found (197/197 files)  hub (cached)     applied      (32, 512) finite  6.1s
sleep_edf_expanded  moment_small        found (197/197 files)  package missing  applied      -                 0.0s
      skipped: momentfm is not installed
        fix: pip install --no-deps "momentfm==0.1.4"
5 pairs: 3 ok, 0 error, 2 skipped, 0 n/a (9.9 s)
```

- `data` and `weights` are the states of sections 4 and 5. A pair that did
  not run says why on the line under it: `skipped:` (data or weights not
  ready, with the `fix:` that gets them; a fix the table already gave is not
  repeated), `n/a:` (the channel map rules the pair out) or `error:` (it
  failed: the message's first sentence, all of it with `-v`).
- `forward`: `(32, 256) finite` is a batch of 32 windows turned into 32
  embeddings of 256 numbers, none NaN or infinite. `constant` (every window
  embedded identically) and `N non-finite` are errors.
- `channel_map`, decided before any data is read:

  | value | meaning |
  |---|---|
  | `applied` | the dataset's map has an entry for this family: the labels are renamed before the model sees a batch |
  | `applied (pass-through)` | ... an entry that passes the labels through unchanged; the wrapper resolves them |
  | `none` | the dataset has no map: the model receives the dataset's own labels |
  | `n/a (skip)` | the map marks this family `skip`: the pair is not run, with the map's reason |
  | `invalid` | the map fails validation, or has no entry for this family: an error here; `run` and `submit` leave the pair out and report it as invalid |

- `time` is the pair's wall time; the footer adds them up. A cohort that
  indexes every window first takes longer (CHB-MIT: about 20 s for two
  models).
- With `--format json` or `csv` the lines under a row are each row's `note`
  (joined with `; `), and a failed pair's whole message is its `error`
  field. What a model prints while it loads (REVE's `flash_attn not found`)
  is a log line: `-v` shows it, `--log` keeps it, the output stays clean.

Exit status: 0 when at least one pair went through and none errored; 1 if a
pair errored, or if nothing could be checked (every pair skipped); with
`--strict`, also 1 if any pair was skipped. `--num-workers N` reads the one
batch with loader workers (default 0, in-process).

## 7. Run a benchmark

```bash
neuroatlas run sleep_stage -m all_fm --dataset full --dry-run   # the plan
neuroatlas run sleep_stage -m biot_pretrained --debug           # fold 0 only
neuroatlas run sleep_stage -m biot_pretrained                   # every fold
```

For each dataset, `run` prints and executes `neuroatlas embed ...` (a no-op
for pairs already in the cache), then `neuroatlas probe ...` over the folds.
`--dry-run` shows the plan: datasets, task, models, folds (`0, 1, 2, 3, 4`,
`LOSO (9)`, `fixed split`), runs, data state, n/a pairs, invalid pairs (an
`invalid` column, when there are any, and a line naming them and the map file
to fix), and the exact commands.

The quickstart's run, with the embeddings already extracted (paths
shortened):

```
$ neuroatlas run sleep_stage -m biot_pretrained

$ neuroatlas embed --dataset sleep_edf_expanded --models biot_pretrained
[extract-only] global cache already exists at <cache root>/sleep_edf_expanded/biot_pretrained/all/644ef899f1fb53b6
embed: 1 ok, 0 failed (cache: <cache root>)

$ neuroatlas probe --dataset sleep_edf_expanded --task sleep_staging --models biot_pretrained --output-root <output root>/sleep_stage/sleep_edf_expanded
[probe] fitting linear classification probe on train=265039 val=95642 test=96971 with 1 seed(s)
...
probe: 5 ok, 0 failed (results: <output root>/sleep_stage/sleep_edf_expanded)

5 runs: 5 ok, 0 failed, 0 n/a
results: <output root>/sleep_stage  (`neuroatlas results sleep_stage`)
```

A pair that fails says so the moment it does, in one line (`error:
sleep_edf_expanded/biot_pretrained (fold 2): CUDA out of memory. (-v: full
message)`); the whole message is in `results.json` and in `--log`.

How long it takes, measured on Sleep-EDF with BIOT: extraction 29 min on an
RTX 4500 Ada another job was also using (17 min on an idle GPU, in the
tester's run); then the probe, which is CPU work, 5 to 12 min per fold on a
28-core machine depending on how busy it was, 20 min on an 8-core one.
`--dry-run` does not estimate times.

Results go to `<output root>/<benchmark>/<dataset>/` (a variant adds
`/<variant>`): `results.json` (one record per model and fold),
`results.csv`, `results.md` and `summary.md` (the same table), and
`probes/`: per model and fold, the fitted probe, its test predictions
(`predictions.npz`, [Saved predictions and `rescore`](#saved-predictions-and-rescore))
and the row it recorded (`result.json`). `results.json` keeps earlier folds and
models: a later run adds to it, and replaces a fold it runs again. A new
result for a fold also drops that fold's earlier failed rows, so a retry
that succeeds leaves no failure behind. Each row records when it was
written (`metadata.written_at`).

The closing lines count this run's work, not the file's:
`N runs: N ok, M failed, K n/a` (and `J invalid` when the channel map has no
entry for some model: those are not run, do not count as failed, and a
`warning:` names them and the map to edit), plus how many earlier results
the file also keeps, then where the results are and the `neuroatlas results`
line that reads them. When some folds were not fitted because their saved
predictions were reused, a `note:` says how many
([Saved predictions are reused](#saved-predictions-are-reused)).

### What is reused

Extraction is the expensive part, so it happens once and is shared:

- **Most datasets** (the sleep cohorts, the MOABB BCI datasets, the
  epilepsy cohorts): one cache per dataset × checkpoint × windowing,
  `<cache root>/<dataset>/<checkpoint>/all/<key>/`, holding every window
  once, with the folds assigned when probing. Every fold, probe and
  benchmark that reads the same windows uses it: `--debug` extracts in full
  and the real run reuses it, and `brain_age` reuses `sleep_stage`'s 30 s
  embeddings.
- **The cohorts that cache per fold and split**,
  `<cache root>/<dataset>/<checkpoint>/{train,val,test}/<key>/`: HMC, MESA,
  STAGES and HomePAP (the PhysioEx-format sleep cohorts), SHHS, the four
  `bci_cognitive` datasets, TUAB once `data prepare tuab` has built its H5
  files (its fast path), CHB-MIT with `--set backend=hdf5`, and any run with
  a stride other than the window. A fold's caches serve that fold only:
  `run` hands the embed step every fold its probe reads, so `--debug`
  extracts fold 0 and the real run then extracts the others. TUSZ joins
  them only with `--set split_mode=official` (the corpus's own
  train/dev/eval split, through `embed` and `probe`); by default it runs
  the paper's 5-fold patient-level split over all 675 patients from one
  `all/` cache.
- The key changes with anything that changes the windows (window length,
  stride, channels, preprocessing, an epilepsy montage other than the
  cohort's default); a probe setting does not change it. When `probe` finds
  no cache under its key but one under another, its error lists the inputs
  that differ.
- Caches written by earlier versions: BCI embeddings made before the
  models' BCI-specific code ran (it is keyed on the trials' domain now) are
  under another key and are extracted again. A CoRe-Sleep cache from before
  it wrote one row per epoch is refused with a row-count message that says
  to delete the folder and extract again.

Brain age has its own folds, by subject, and keeps each cohort's published
protocol (`cohort_protocols` in `configs/tasks/brain_age.json`): on
Sleep-EDF the Sleep Cassette subset only (78 subjects), split into
age-stratified folds (seed 42), not the sleep-staging folds; on ISRUC and
WSC one score per recording, a subject's recordings kept in one fold; on
PhysioNet 2026, 100 healthy and 100 impaired age-matched subjects held out
of every fold and the ridge trained on healthy subjects only. CFS and MrOS
keep their readers' age-stratified folds. It still reads the same
embeddings, and picks the ridge's alpha by nested cross-validation from
0.1 to 100.

Where a cohort's folds are frozen in `configs/cohorts/<dataset>/folds.json`
(CHB-MIT, SeizeIT1, SeizeIT2, EPILEPSIAE, TUSZ and most sleep cohorts), they
are the paper's splits. On the epilepsy readers, `embed` and `probe` take
`--set folds_manifest=none` to derive folds with the reader's own splitter
instead, and `--set strict_folds=false` to run on the subjects the data and
the file share when they differ; neither result is comparable with the
paper's.

Brain age needs ages in the cache. A Sleep-EDF cache built without `xlrd`
(or ISRUC without `openpyxl`) holds no ages; `brain_age` stops with the
cache's folder and says to delete it and run again. Installing the package
afterwards does not repair an existing cache.

### Saved predictions are reused

Every probe keeps each fold's test predictions
([Saved predictions and `rescore`](#saved-predictions-and-rescore)), so a
second `run` (or `probe`) does not fit a fold again when nothing it depends
on has changed: it recomputes the fold's metrics from the saved file and
says so. On the quickstart's two folds, 580 s the first time, 6 to 10 s the
second (paths shortened):

```
$ neuroatlas run sleep_stage -m biot_pretrained --folds 0,1
...
$ neuroatlas probe --dataset sleep_edf_expanded --task sleep_staging --folds 0,1 --models biot_pretrained --output-root <output root>/sleep_stage/sleep_edf_expanded
probing 1 model(s) on sleep_edf_expanded: 2 run(s), one line each as it finishes
[1/2] sleep_edf_expanded biot_pretrained fold 0: ok (reused) (0s)
[2/2] sleep_edf_expanded biot_pretrained fold 1: ok (reused) (0s)
probe: 2 ok, 0 failed (results: <output root>/sleep_stage/sleep_edf_expanded)

2 runs: 2 ok, 0 failed, 0 n/a
note: reused saved predictions for 2 folds; --reprobe probes again
```

(BCI cohorts save less: building their datamodule reads and filters every
MOABB recording, in the embed step and again in the probe step, before any
fold is reused -- about 70 s of BNCI2014_001's 116 s.)

A fold's file is reused when it is in the fold's probe folder -- whose name
is a key of the probe settings, the task settings and the dataset settings,
fold included -- and was made with the same seeds, `--pooling` and weights
file (`--checkpoint-override`), from the same embeddings: a cache under this
run's cache root whose files have the size and modification time they had
when the probe read them. A cache extracted again (deleted and rebuilt, or
re-embedded after a fix) therefore means a new probe. The weights are
compared by their file's path, not its content; weights replaced in place
also leave the old embeddings in the cache, and extracting those again is
what makes the probe run again. A change to how a metric is computed
applies (the metrics are recomputed); a change to how a probe is fitted is
not seen: after one, run with `--reprobe`. Whenever a file is
there but not reused, a `note:` says why (`saved predictions not reused,
probing again: the embeddings they were fitted on were extracted again or
are gone (...)`). The row restored has the metrics recomputed with today's
code and `metadata.reused_predictions: true`; the probe step does
not load the backbone, and its progress line reads `ok (reused)`.

`--reprobe` (on `run`, `probe` and `submit`) fits every fold again and
overwrites its files. Results from before this version have no file, and
are fitted again as before.

### Run options

- **`--debug`**: fold 0 only (results say `1/5` folds). Its extraction is
  reused by the full run, except on the cohorts that cache per fold and
  split ([What is reused](#what-is-reused)), where the full run extracts the
  other folds.
- **`--folds 0,2`**: these folds only; a fold the dataset does not have is a
  usage error.
- **`--limit-batches N`**: a smoke test. Each extraction stops after N
  batches, in its own cache, `<cache root>/_limited`, so a truncated cache
  is never read as a complete one; the results go to
  `<output root>/_limited/<benchmark>/<dataset>/`, and the closing line
  names that folder and the command that reads it
  (`neuroatlas results <benchmark> --output-root <output root>/_limited`). Too
  few subjects can leave a fold with nothing to test on: on a
  leave-one-subject-out BCI dataset every fold then fails.
- **`--num-workers N`**: loader workers for extraction. Default: the CPUs
  this job may use (CPU affinity, cgroup quota, SLURM or HTCondor's
  allocation) minus one, at most 16, unless the dataset pins its own. The
  probe caps its BLAS threads at 8.
- **`--skip-embed`**: probe only; a pair with no embeddings fails.
- **`--reprobe`**: fit every fold again instead of reusing its saved
  predictions ([Saved predictions are reused](#saved-predictions-are-reused)).
- **`--cache-root`, `--output-root`**: other roots for this run.
- **`--checkpoint-override biot_pretrained.checkpoint_path=/my.ckpt`**:
  other weights for this run. Only `checkpoint_path` for now; the file must
  exist, and `--cache-root` is required, since the cache is keyed by
  checkpoint id rather than by weights.
- **`--allow-partial`**: run on a dataset `data status` calls `partial` or
  `empty`. Without it `run` refuses (exit 2): the results would silently
  cover only the subjects that arrived.
- **`--variant`**: another cell of the benchmark (BCI: `per_patch`,
  `confound_control`); its results go to `<dataset>/<variant>/` and stay
  apart from the default's in `results`.
- **`--per-model-output`**: results in `<dataset>/<model>/`, what cluster
  jobs use.

Before extraction, `run` checks every model's weights; a model whose weights
are not here is reported by name and counted as failed, and the others run.
An interrupted extraction resumes: finished batches are kept, and where the
reader's order is fixed they are skipped, otherwise read again but not
re-embedded (the progress bar says `efficient` or `re-walking`).

Exit status: 0 if every run succeeded, 1 if any failed, 2 for a usage
error (an unknown name, a bad flag value, a refused partial dataset). A
tool's own status (wget, nsrr) is reported as 1, never passed through.

## 8. Run on a cluster

```bash
neuroatlas submit sleep_stage -m all_fm --out runs/sleep_stage            # HTCondor
neuroatlas submit sleep_stage -m all_fm --out runs/ss --backend slurm \
    --partition gpu --cpus 24 --no-mem --extra --account=myproject
condor_submit runs/sleep_stage/jobs.job                                    # or: sbatch runs/ss/jobs.sbatch
neuroatlas status --out runs/sleep_stage
```

`submit` writes one job per dataset × model, each a `neuroatlas run` of that
pair, and queues nothing: its last line prints the command that does.
`--dataset` defaults to `full`. Each job writes
`<output root>/<benchmark>/<dataset>/<model>/results.json`, so jobs never
share a file, and next to it a `job_status.json` its shell keeps (running,
then exit code, host, times), so a job that dies before Python starts still
leaves a record. A `sleep_hypnogram` job is one per dataset, over all its
models.

Jobs inherit this machine's settings (the root variables, `MNE_DATA`,
`HF_HOME`, `PYTHONPATH`, never a token) and run offline. So `submit`
checks first: a pair whose data or weights are not here, or whose dataset's
channel map skips it (`n/a (channel map)`), has no entry for its family
(`invalid (channel map)`) or fails to load, gets no job and is counted as
skipped with the reason (`-v` lists each; `--force` writes the data- or
weights-blocked ones anyway, never the ones the channel map rules out). A
job of an interpreter under `/tmp` is warned about: another machine will not
find it.

On a machine with only Sleep-EDF, BIOT's weights and not LaBraM's:

```
$ neuroatlas submit sleep_stage -m biot_pretrained,labram_pretrained --out runs/ss
1 job: 1 queued (mode cached), 1 missing
29 pairs skipped: 28 data missing, 1 weights auto (-v lists them)
  --force writes jobs for the 29 blocked by data or weights
GPUs: compute capability >= 7.0 (the lowest torch 2.8.0+cu128 supports), a requirement in jobs.job
queue them: condor_submit /.../runs/ss/jobs.job  (-dry-run checks the file's syntax only, not the pool's policy)
$ neuroatlas status --out runs/ss
sleep_stage (full, default): 1 job: 1 missing; 29 pairs skipped
```

The `--out` folder holds `jobs.json` (what was planned and skipped),
`jobs/<job>.sh` (one script per job), `jobs.txt` (the scripts this submit
queues), `jobs.job` or `jobs.sbatch`, and `logs/`.

| verdict | meaning |
|---|---|
| `done` | results written, every fold succeeded |
| `partial` | some folds succeeded, some failed |
| `failed` | results written, none succeeded; or the job exited 0 without results |
| `exited N` | the job ended with code N before writing results (128+N: killed by signal N) |
| `stopped` | it started, then left the queue without recording an exit (killed, evicted, node failure) |
| `removed` | removed from the queue |
| `running`, `idle`, `held` | still in the scheduler's queue |
| `missing` | never ran: every job, right after `submit` |
| `waiting` | a hypnogram job whose staging results are not there yet |

`status` asks `condor_q` or `squeue` when it is on the machine (run it where
you submitted), reads the HTCondor event logs in `<out>/logs`, then the
jobs' own files; `--no-scheduler` uses only the files. It prints each
failure with its host and log; `-v` gives one row per job.

`--mode cached` (default) queues only `missing` jobs, so re-running
`submit` never repeats work; `--mode retry` also takes `failed`, `partial`,
`exited`, `stopped` and `removed` ones; `--mode all` takes everything. A job
still in the queue (`running`, `idle`, `held`) is never queued again. A job
reuses the saved predictions of folds already probed with the same inputs
([Saved predictions are reused](#saved-predictions-are-reused));
`submit --reprobe` writes jobs that fit every fold again (`run --reprobe`),
with `--mode all` to queue jobs that already finished.

Resources per job: `--gpus 1 --cpus 4 --memory 32G` by default. `--memory`
needs a unit (`32G`, `1500M`). The wall time is one setting for both
backends, `--time H:MM:SS` or `--walltime SECONDS`, default 24 h: SLURM gets
`--time`, HTCondor `+RequestWalltime` and `+MaxRuntime` (`--walltime-attr`
picks other attribute names). `--gpu-capability auto` (default) requires a
GPU the installed torch has kernels for (7.5 for the CUDA 13 build): an
HTCondor requirement, a comment for SLURM, which has no standard attribute.
SLURM: `--partition`, `--no-mem`; HTCondor: `--requirements EXPR`.
`--extra LINE` adds a raw line to the submit file; a value that starts with
`-` works as `--extra --account=x`. A flag the chosen backend ignores is
warned about.

**Site notes (ESAT HTCondor).** The pool refuses a job without
`+RequestWalltime`; `submit` writes it. Old GPUs cannot run the CUDA 13
torch; the default `--gpu-capability auto` keeps jobs off them. Some nodes
fail jobs for reasons no attribute shows (crashes inside BLAS, a shared
disk they cannot write); when `status` shows the same host failing, keep
jobs off it with `--requirements 'Machine =!= "host.esat.kuleuven.be"'`.
`condor_submit -dry-run` checks the file's syntax only, not the pool's
policy.

## 9. Read the results

```bash
neuroatlas results sleep_stage
neuroatlas results sleep_stage 'runs/old/**/results.json' --format md
```

```
$ neuroatlas results sleep_stage
sleep_stage: balanced accuracy, higher is better; mean ± std over folds; normalized: 0 = chance, 1 = perfect
dataset             model            bal_acc  ±      folds  normalized  kappa  macro_F1
sleep_edf_expanded  biot_pretrained  0.657    0.014  5/5    0.571       0.749  0.661
```

`results` reads every `results.json` under `<output root>/<benchmark>/`
(the files of a local `run` and of cluster jobs alike), or the files,
folders and quoted globs you name. One row per dataset × variant × model; a
`variant` column appears once a variant other than the default has results
(`run --variant per_patch` writes to `<dataset>/per_patch/`), and
`--variant NAME` shows that variant's rows only:

- The first two numbers are the headline metric over the folds that
  succeeded: its mean, under the metric's name (`bal_acc`, `AUROC(window)`,
  `MAE(years)`, ...), and `±`, the population standard deviation (numpy's
  default); with one fold it is n/a. Columns are named by metric in the
  table; `--format json` and `csv` keep the metric keys (`balanced_accuracy`,
  `auroc`, ...) and the fields `mean` and `std`.
- `folds`: succeeded / the protocol's folds, e.g. `5/5`, or `1/5` after
  `--debug`. A `warning:` says when a mean covers fewer folds than the
  protocol and is not comparable with a full run.
- `normalized` (0 = dummy, 1 = perfect), only for benchmarks with a fixed
  dummy; then the secondary metrics.
- A row whose headline is `n/a` says why on the line under it (`n/a: the
  channel map skips this model`, or the reason the metric does not apply).
  `-v` also shows why its failed folds failed (`error: folds 0, 1: <the
  message each recorded> [code]`); without `-v`, a `warning:` counts them.
- In JSON a missing number is `null`, and every row also has `benchmark`,
  `metric`, `variant`, `status`, `n_folds`, `n_expected`, `n_failed`,
  `failures`, `errors` and `note`.

A result recorded twice -- a quick `run` and later a cluster job of the same
dataset, variant, model, fold and task -- counts once: the newest file wins,
and a `warning:` says how many duplicates it dropped. Two variants of the same
model and fold are two results, not a duplicate. The same embeddings probed
on two machines can differ by about 1e-3 (BIOT × Sleep-EDF fold 0: 0.6624
and 0.6619), so the newest copy is not always the same number.

`results --reference` would put the paper's numbers beside yours, from
`configs/reference/<benchmark>.csv`; no benchmark ships that table yet, so
it exits 1 saying so.

Exit status: 0 with something to show; 1 when there are no results; 2 for
a usage error.

### Saved predictions and `rescore`

Every probe writes, for each fold, its test predictions to `predictions.npz`
in the fold's probe folder,
`<output root>/<benchmark>/<dataset>/probes/<dataset>/<model>/<key>/`,
beside the row it recorded (`result.json`); `results.json` has the file's
path under `cache_paths.predictions`. The metrics each row records are
computed from that file by its task's own scoring function, so a metric
fixed or added after a run reaches it without probing again:

```bash
neuroatlas rescore sleep_stage                     # every result of the benchmark
neuroatlas rescore epilepsy --dataset siena -m cbramod_pretrained
```

```
$ neuroatlas rescore sleep_stage --dataset sleep_edf_expanded -m biot_pretrained
dataset             model            fold  result          bal_acc before  after
sleep_edf_expanded  biot_pretrained  0     rescored, same  0.66228         0.66228
sleep_edf_expanded  biot_pretrained  1     rescored, same  0.650876        0.650876

2 rescored from saved predictions: 0 changed, 2 the same; 1 results.json file rewritten
```

`rescore` recomputes each row's metrics from its file, merges them into the
row as a run does (a metric the task computes is replaced, one it does not
is kept), keeps the row's metadata and adds `metadata.rescored_at`, and
rewrites `results.json` and the tables beside it; it fits nothing.
`result` reads `rescored, same` or `rescored, changed`, with the headline
before and after: on files the current version wrote it is always `same`.
A row probed before predictions were saved has nothing to rescore and is
listed with the command that probes it again:

```
skipped: sleep_edf_expanded/biot_pretrained folds 2, 3, 4: probed before predictions were saved
  fix: neuroatlas run sleep_stage --dataset sleep_edf_expanded -m biot_pretrained --reprobe
```

It takes `--dataset`, `-m`, `--variant`, `--output-root` and `--format` as
`results` does; `--format json` adds, per row, its file and that `fix`.
Seizure predictions saved in their first layout (2026-10-06, before this
version) are read too; their rescore keeps the validation choices (C,
threshold) from `results.json`.

**The file.** `np.load("predictions.npz")` reads it (no pickles); the full
description is in `neuroatlas/predictions.py`.

| entry | what |
|---|---|
| `format`, `dataset`, `checkpoint_id`, `task`, `fold` | text: `neuroatlas.predictions/1`, what made it (`fold` is empty for a cohort with one fixed split) |
| `y_true` | per test row: the true class, or value (an age in years) |
| `y_pred` | the predicted class, or value |
| `y_proba`, `classes` | class probabilities (float32, rows × classes) and the class of each column |
| `y_score` | binary tasks: the positive class's score (float64), what AUROC, AUPRC and the event Sens@FA read |
| `subject_id`, `recording_id`, `session_id`, `epoch_index`, `trial_idx`, `window_start_s` | the rows' ids, those they have |
| `window_s`, `threshold` | epilepsy: seconds per window, and the decision threshold tuned on validation |
| `<group>/<column>` | further probes of the fold: arousal one per threshold (`threshold_1.0s/`), respiratory and limb events one per field and threshold (`apnea_any_fraction/threshold_1.0s/`), brain age one per estimator besides the headline (`ridge_subject/alpha_y_pred`: every ridge alpha's test predictions; `epoch_regression/`: the epoch-level rows; `holdout/`: a held-out cohort) |
| `seed_y_pred`, `seed_y_score` | every seed's predictions, under `probe --seed-mode shared` |
| `info` | JSON: `fit`, what the probe chose on validation (seed, C, alpha, threshold) and the validation scores it chose by -- a rescore keeps them; `score`, settings the metrics read; `groups`; and what the probe was given (seeds, pooling, probe and task settings, dataset settings, weights file, embedding caches with their files' sizes and times) |

The rows are what the metrics count: 30 s epochs (sleep staging, arousal,
respiratory and limb events, with their recording and epoch index -- what a
hypnogram is built from), 10 s windows (epilepsy), trials (BCI; the fold's
held-out subject is in `subject_id`), subjects or recordings (diagnosis,
brain age). A Sleep-EDF staging fold is about 1.9 MB.

```python
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from neuroatlas import predictions

z = np.load("predictions.npz")
balanced_accuracy_score(z["y_true"], z["y_pred"])     # one fold, by hand
predictions.score("predictions.npz")                  # every metric the task records
```

## 10. From Python

```python
from neuroatlas import api

api.benchmarks()                                        # = list benchmarks
api.models("all_fm")                                    # = models status all_fm
api.data_status("sleep_stage")                          # = data status
api.plan("sleep_stage", "all_fm", datasets="full")      # = run --dry-run
api.check("sleep_stage", "biot_pretrained")             # = check
api.run_benchmark("sleep_stage", "biot_pretrained", debug=True)
api.results("sleep_stage")                              # = results
api.rescore("sleep_stage")                              # = rescore
```

Each returns a pandas DataFrame. Importing `neuroatlas.api` applies the same
settings file and does not import torch. It exports the roots and
`MNE_DATA` from that file; a variable you set afterwards in the same
session (`os.environ["MNE_DATA"] = ...`) is yours, and wins as it would on
the command line.

The columns are those of the command's `--format json` (`dataset`,
`model`, `not_applicable`, `note`, ...): `models()` has
`models status -v`'s `checkpoint`, `source`, `state`, `path` and `note`,
and `data_status()` has `data status`'s fields, `download` and `map`
(`yes`/`no`) among them.
Missing numbers are `None` (NaN in a float column). `results()` takes
`variant=` as the command takes `--variant`.

`run_benchmark(...)` returns the results of this run's datasets, models and
variant only, and takes `debug=`, `limit_batches=`, `cache_root=`,
`output_root=`, `online=` and `reprobe=` (not `num_workers`). `rescore(...)`
takes `datasets=`, `models=`, `output_root=` and `variant=`, rewrites the
results as the command does, and returns one row per result: `result`
(`same`, `changed`, `skipped`, `failed`), `before`, `after`, `predictions`
and, for a skipped row, `fix`.

The offline rule holds in Python too: no API function downloads. `check()`
and `run_benchmark()` run with downloads switched off
(`NEUROATLAS_OFFLINE=1` for the duration of the call, unless you set it
yourself), so a model whose weights are not here is refused by name and the
others run. `run_benchmark(..., online=True)` is `neuroatlas --online run`:
it fetches a missing checkpoint. The API leaves `HF_HUB_OFFLINE` alone (the
Hugging Face library reads it once per Python session); the weights check
before extraction already refuses a hub model that is not cached.

## Where things live

Everything the tool writes, by default under `$NEUROATLAS_HOME`
(`~/.neuroatlas`), except where noted.

| what | where |
|---|---|
| settings, tokens | `$NEUROATLAS_HOME/config.yaml`, `hf_token`, `nsrr_token`, `github_token` |
| raw datasets | `<data root>/...`, one folder each; `data status -v` shows each. Zenodo cohorts under `<data root>/<dataset>/`. |
| MOABB raw data | the MOABB folder (`config show`; [MOABB](#moabb-the-14-bci-datasets)): one subfolder per dataset under moabb 1.2.0 (e.g. `MNE-bnci-data/...`), `NEMAR/<id>/` under moabb 1.6+ |
| prepared files | `<cache root>/prepared/<Name>/...` (optional: epilepsy fast paths, BCI pickles) |
| weights | `<models root>/foundation/...` (DeepSOZ-HEM's code and checkpoint in `foundation/deepsoz_hem/`), `<models root>/shhs/...` (CoRe-Sleep, SleepTransformer, SleePyCo; the STFT normalisation CoRe-Sleep and SleepTransformer read ships with the package), `<models root>/supervised/...` (Seizure-Transformer) |
| hub models | Chronos, MOMENT, Moirai: `<models root>/foundation/huggingface_cache`; the other Hugging Face models (CBraMod, a hub-cached REVE): the Hugging Face cache, `$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub` |
| embeddings | `<cache root>/<dataset>/<checkpoint>/all/<key>/` (`features.npy`, `labels.npy`, `items.json`, `metadata.json`); per fold and split for some cohorts ([What is reused](#what-is-reused)) |
| `--limit-batches` | `<cache root>/_limited/...` and `<output root>/_limited/<benchmark>/<dataset>/` |
| results of `run` | `<output root>/<benchmark>/<dataset>/`: `results.json`, `results.csv`, `results.md`, `summary.md`, `probes/` |
| a fold's probe | `<output root>/<benchmark>/<dataset>/probes/<dataset>/<model>/<key>/`: `predictions.npz` (its test predictions), `result.json` (the row it recorded), `probe.pkl` and `metadata.json` (the fitted linear probe) |
| results of cluster jobs | `<output root>/<benchmark>/<dataset>/<model>/results.json` and `job_status.json` |
| hypnograms | `<output root>/sleep_hypnogram/<dataset>/`: `hypnograms.json`, `hypnogram_features.csv`, `hypnogram_features_summary.csv`, `results.json` |
| job files | the `--out` folder: `jobs.json`, `jobs/<job>.sh`, `jobs.txt`, `jobs.job` or `jobs.sbatch`, `logs/` |
| `check` | a temporary folder, deleted when it ends |
| epilepsy recording statistics | `<cache root>/recording_stats/<reader>/` (CHB-MIT, Siena, Helsinki, TUSZ, TUAB, NMT, Epilepsiae readers) |

`config show` names the four roots and the MOABB folder; an earlier version
kept weights, caches and results inside the source checkout, and
`config show` prints the `config set` line that keeps using them.

## Known gaps

What does not work, or not as the rest of this guide would suggest, as of
2026-10-05. Most are decisions for the authors.

**Install and access**

- Not on PyPI yet (as `neuroatlas-bench`), and the GitHub repository is not
  public yet: install from a clone you have access to.
- Moirai cannot share the main environment (`requirements-tsfm.txt`).
- CoRe-Sleep and SleepTransformer weights are release assets of a private
  repository: `models download` needs a GitHub token that can read it.

**Data**

- SHHS: the reader needs the CoRe-Sleep preprocessed copy, which no script
  here builds; `data download shhs` is refused.
- `bci_cognitive` cannot run without the authors' preprocessed files
  (DREAMER, EEGMat, ArithmeticTask).
- The default folders under the data root follow the authors' layout
  (`$DATA/data/Guido/hmc/...`); `data download` and `data status` agree on
  them, but the tree is not tidy.

**Runs**

- BCI: `run`'s preprocessing (MOABB epoched at 128 Hz, 4–40 Hz, no notch)
  is neither of the paper's two (per-model pickles; or 4–40 Hz, 1–4 s).
- BCI: whether REVE should normalise BCI trials per batch is undecided, and
  LaBraM's BCI embeddings are 400-d where the registry says 200.
- Folds that are not the published numbers' (see each `cohort.yaml`,
  `provenance.known_issues`): Siena (published split by recording), TUAB and
  NMT (published from 5-fold CV; the official split here), Bonn (published
  split by segment), CFS and MrOS staging (two fold protocols), SHHS (where
  the published split came from is unknown).
- CoRe-Sleep runs the paper's path (an all-zero EOG through the bimodal
  model); its EEG-only path is not offered.
- SleePyCo, CoRe-Sleep and SleepTransformer are not part of the epilepsy
  benchmark (they take only 30 s epochs). The paper's epilepsy results do
  show CoRe-Sleep and SleepTransformer (Figures 2b and 8, Tables 3 and 4,
  App. D.1.3); those cells cannot be rerun with the command.
- The sleep channel maps have no entry for EEGNetv4, DeepSOZ and
  Seizure-Transformer (some lack more, e.g. HMC's): those pairs are `invalid`.
- The epilepsy cache key leaves out `normalize`, `label_mode` and
  `overlap_threshold`; TUAB ignores `--set montage`; Bonn accepts unknown
  `--set` keys.
- Brain age for SHHS, MESA, HomePAP and STAGES is planned: their readers
  carry no ages.

**Results**

- No `configs/reference/*.csv` ships, so `results --reference` has nothing
  to compare against.
