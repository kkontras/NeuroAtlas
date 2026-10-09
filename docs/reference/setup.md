<!-- Generated from the command-line parsers by `python -m neuroatlas.cli._gendocs`. Do not edit by hand. -->
# Setup

`config` sets where NeuroAtlas reads datasets and writes caches, results and model weights. [Configuration](../guide/configuration.md) explains the settings.

## config

Set where NeuroAtlas reads datasets and writes caches, results and model weights. The settings are saved in ~/.neuroatlas/config.yaml, or in $NEUROATLAS_HOME/config.yaml when that variable is set.

Usage: `neuroatlas config <action>`

| Command | Description |
|---|---|
| [`config init`](#config-init) | Write the settings file. |
| [`config show`](#config-show) | Print each setting, whether its folder exists and where the value comes from. |
| [`config set`](#config-set) | Change one setting. |
| [`config unset`](#config-unset) | Remove one setting, so it goes back to its default. |
| [`config path`](#config-path) | Print the path of the settings file. |
| [`config token`](#config-token) | Save a Hugging Face, GitHub or NSRR token. |

### config init

Write the settings file. With a project folder DIR, everything goes to DIR/data, DIR/cache, DIR/results and DIR/models. A `--*-root` option moves one of them.

Usage: `neuroatlas config init [DIR] [options]`

```bash
neuroatlas config init ~/neuroatlas
neuroatlas config init ~/neuroatlas --data-root /data/eeg    # datasets elsewhere
```

| Option | Description | Default |
|---|---|---|
| `DIR` | Project folder. Its sub-folders are created when needed. |  |
| `--data-root DIR` | Folder with the raw datasets, one sub-folder per dataset. | DIR/data |
| `--cache-root DIR` | Folder for saved embeddings and prepared datasets, which can grow large. | DIR/cache, else $NEUROATLAS_HOME/artifacts/embedding_cache |
| `--output-root DIR` | Folder for the probe results. | DIR/results, else $NEUROATLAS_HOME/artifacts/benchmarks |
| `--models-root DIR` | Folder for model weights. | DIR/models, else $NEUROATLAS_HOME/artifacts/models |
| `--dataset-path DATASET.KEY=PATH` | Folder of a dataset kept elsewhere, as in ucddb.data_root=/mnt/ucddb. Can be repeated. |  |
| `--force` | Overwrite an existing settings file. |  |

### config show

Print each setting, whether its folder exists and where the value comes from.

Usage: `neuroatlas config show`

```bash
neuroatlas config show
```

### config set

Change one setting. KEY is data_root, cache_root, output_root, models_root or a dataset folder such as chbmit.bids_root. `neuroatlas data status DATASET` shows the key a dataset reads.

Usage: `neuroatlas config set key value`

```bash
neuroatlas config set ucddb.data_root /mnt/ucddb
neuroatlas config set cache_root /scratch/neuroatlas_cache
```

| Option | Description | Default |
|---|---|---|
| `key` | Setting name, such as data_root or ucddb.data_root. |  |
| `value` | New value, usually a folder. |  |

### config unset

Remove one setting, so it goes back to its default.

Usage: `neuroatlas config unset key`

```bash
neuroatlas config unset ucddb.data_root
```

| Option | Description | Default |
|---|---|---|
| `key` | Setting name. |  |

### config path

Print the path of the settings file.

Usage: `neuroatlas config path`

```bash
neuroatlas config path
```

### config token

Save an access token for Hugging Face (hf), GitHub (github) or NSRR (nsrr). Type it when asked, or pipe it in as in `neuroatlas config token hf < file`. It is saved to `$NEUROATLAS_HOME/<name>_token`, readable only by you, and never printed.

Usage: `neuroatlas config token {github,hf,nsrr} [options]`

```bash
neuroatlas config token nsrr
neuroatlas config token hf --remove
```

| Option | Description | Default |
|---|---|---|
| `{github,hf,nsrr}` | hf for gated Hugging Face repositories, github when GitHub refuses a download, or nsrr for the NSRR sleep datasets. |  |
| `--remove` | Delete the saved token. |  |
