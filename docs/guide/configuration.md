# 1. Configuration

NeuroAtlas keeps what it reads and writes in four folders: the raw datasets, a cache of
embeddings, the results and the model weights. You choose them once. They are saved in a settings
file that every command, and every cluster job, reads.

## 1.1 The project folder

The simplest setup is one project folder that holds all four:

```bash
neuroatlas config init ~/neuroatlas
neuroatlas config show
```

This creates four sub-folders and writes their paths to `~/.neuroatlas/config.yaml`. Set
`$NEUROATLAS_HOME` to keep that file elsewhere.

| Setting | Holds | Folder | Environment variable |
|---|---|---|---|
| `data_root` | raw datasets | `~/neuroatlas/data` | `EEG_DATA_ROOT` |
| `cache_root` | embeddings and prepared files (can grow large) | `~/neuroatlas/cache` | `EEG_CACHE_ROOT` |
| `output_root` | results | `~/neuroatlas/results` | `NEUROATLAS_OUTPUT_ROOT` |
| `models_root` | model weights | `~/neuroatlas/models` | `NEUROATLAS_MODELS_ROOT` |

To put one of them somewhere else, add its option, for example
`neuroatlas config init ~/neuroatlas --data-root /shared/eeg`. To change one folder later, run
`neuroatlas config set cache_root DIR`. `neuroatlas config unset cache_root` returns it to the
default.

An environment variable overrides the file. This lets a cluster job redirect one run without
touching your settings, and `submit` copies the variables that are set into every job script.

## 1.2 Datasets you already have

Every dataset lives in its own folder, `<data root>/<dataset>`. If you already have a copy
somewhere else, point the tool at it instead of downloading it again:

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

MOABB datasets use one shared folder instead (see [MOABB datasets](data.md#24-moabb-datasets)).
You do not need to remember the table. `data status` prints the setting for every dataset it does
not find, and `config set` refuses a wrong key and suggests the right one:

```text
$ neuroatlas config set siena.data_root /tmp
error: siena has no path key 'data_root' (did you mean bids_root?); its keys: bids_root, cache_root
  fix: neuroatlas config set siena.bids_root /tmp
```

A misspelt top-level setting in `config.yaml` stops every command except `config`, and the error
names the `config unset` command that removes it.

## 1.3 Checking the settings

`config show` lists each folder, whether it exists and is writable, and what set it: an
environment variable, the config file or the default. It also shows the MOABB folder, the dataset
paths you set, and where each token was found. It never prints a token.

```text
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

## 1.4 Tokens

You need a token only for the NSRR sleep datasets. Get it from sleepdata.org/token and save it
once:

```bash
neuroatlas config token nsrr
```

The command asks for the token without showing it and stores it next to the config file, readable
only by you. `--remove` deletes it. `config show` warns if others can read a token file.

| Token | Used for | Looked for in this order |
|---|---|---|
| NSRR | the NSRR sleep datasets | `$NSRR_TOKEN`, `$NEUROATLAS_HOME/nsrr_token` |
| Hugging Face | gated Hugging Face repositories | `$HF_TOKEN`, the file named by `$NEUROATLAS_HF_TOKEN_FILE`, `$NEUROATLAS_HOME/hf_token`, `<checkout>/.secrets/hf_token`, `~/.cache/huggingface/token` |
| GitHub | GitHub release downloads | `$GITHUB_TOKEN`, `$GH_TOKEN`, `$NEUROATLAS_HOME/github_token` |

No checkpoint needs a Hugging Face token, because every Hugging Face repository the models use is
public. If a gated repository ever refuses a download, the error tells you to accept its terms on
huggingface.co and then run `neuroatlas config token hf`.

## 1.5 Downloads happen only when you ask

Only `data download`, `data prepare`, `models download` and `fetch --download` use the network.
Every other command, and every cluster job, runs with downloads switched off. A missing checkpoint
then fails at once, by name, with the command that fetches it, instead of stalling a job halfway.

To allow downloads for one command, add `--online`:

```bash
neuroatlas --online run sleep_stage -m biot_pretrained
```

With `--online`, `run` fetches a missing checkpoint itself. `check` never downloads anything. The
same rule covers MOABB: offline, a MOABB dataset that is not fully downloaded stops the run at its
first read and names `neuroatlas data download <dataset>`.

Global options such as `--online`, `--log FILE` and `-v` can go before or after the command. A
mistyped command or option gets the corrected command as its fix:

```text
$ neuroatlas chek epilepsy -m biot_pretrained
error: unknown command 'chek'
  fix: neuroatlas check epilepsy -m biot_pretrained
```

## 1.6 Where files go

Everything goes under `$NEUROATLAS_HOME` (`~/.neuroatlas`) unless you set other folders. This
table lists every file the tool writes and where to find it.

| What | Where |
|---|---|
| settings and tokens | `$NEUROATLAS_HOME/config.yaml`, `hf_token`, `nsrr_token`, `github_token` |
| raw datasets | `<data root>/<dataset>` |
| MOABB datasets | the MOABB folder, one subfolder per dataset |
| optional faster copies | `<cache root>/prepared/` |
| weights | `<models root>/foundation/` (DeepSOZ-HEM in `foundation/deepsoz_hem/`), `<models root>/shhs/` (CoRe-Sleep, SleepTransformer, SleePyCo), `<models root>/supervised/` (Seizure-Transformer) |
| Hugging Face models | Chronos, MOMENT and Moirai in `<models root>/foundation/huggingface_cache`. The others in the Hugging Face cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`). |
| embeddings | `<cache root>/<dataset>/<checkpoint>/all/<key>/`, or per fold and split for [some datasets](advanced.md#73-caches) |
| `--limit-batches` runs | `<cache root>/_limited/` and `<output root>/_limited/<benchmark>/<dataset>/` |
| results of `run` | `<output root>/<benchmark>/<dataset>/results.json` and `probes/` |
| one fold's probe | `<output root>/<benchmark>/<dataset>/probes/<dataset>/<model>/<key>/`, with `predictions.npz`, `result.json`, `probe.pkl` and `metadata.json` |
| results of cluster jobs | `<output root>/<benchmark>/<dataset>/<model>/results.json` and `job_status.json` |
| hypnograms | `<output root>/sleep_hypnogram/<dataset>/`, with `hypnograms.json`, `hypnogram_features.csv`, `hypnogram_features_summary.csv` and `results.json` |
| cluster job files | the `--out` folder |
| `check` | a temporary folder, deleted when it ends |
| epilepsy recording statistics | `<cache root>/recording_stats/` |
