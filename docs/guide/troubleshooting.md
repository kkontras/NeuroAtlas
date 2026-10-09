# 8. Troubleshooting

Most problems show up as a message with a `fix:` line under it, and the fix is the command to run
([How the tool talks to you](index.md#how-the-tool-talks-to-you)). This chapter shows how to get
more detail when that is not enough, and what to do about the problems that come up most often.

## 8.1 Getting more detail

Run the command again with `-v --log run.log`. `-v` shows library output and full messages, and
`--log` keeps everything in a file. If that does not explain it, open an issue at
[github.com/kkontras/NeuroAtlas/issues](https://github.com/kkontras/NeuroAtlas/issues) with the
command and `run.log` attached.

Exit codes are 0 when the command succeeded, 1 when something failed, and 2 when the command was
refused, for example for a bad option, missing data or weights, or a model the benchmark does not
include.

## 8.2 Common problems

### A pair is skipped

Its data or weights are not on this machine. The `fix:` line gives the `data download` or
`models download` command that gets them. If you already have the dataset somewhere else, point
the tool at it with `config set` ([Datasets you already have](configuration.md#12-datasets-you-already-have)).

### The run refuses a partial or empty dataset

`data status` calls a dataset `partial` when it has fewer files than a complete copy, for example
after a download that stopped part-way. Run the same `data download` again: it skips the files
that are complete and fetches the rest. `--allow-partial` runs on the files you have, but the
results then cover only some subjects.

### A checkpoint is not part of the benchmark

Each benchmark runs only the checkpoints the paper evaluates on it, and naming another one is
refused. `neuroatlas list models --benchmark <benchmark>` lists the ones it accepts
([Models a benchmark leaves out](models.md#33-models-a-benchmark-leaves-out)).

### A run stops offline

Every command except `data download`, `data prepare`, `models download` and `fetch --download`
runs offline. A missing checkpoint fails at once with the command that fetches it, and a MOABB
dataset that is not fully downloaded stops the run at its first read. Download what is missing, or
add `--online` to let `run` fetch a missing checkpoint itself
([Downloads happen only when you ask](configuration.md#15-downloads-happen-only-when-you-ask)).

### A Hugging Face download is refused

No checkpoint needs a Hugging Face token. If a gated repository ever refuses a download, accept its
terms on huggingface.co and then run `neuroatlas config token hf`.

### Brain age stops and asks you to delete a cache

The cache was built without the package that reads the subjects' ages: `xlrd` for Sleep-EDF or
`openpyxl` for ISRUC. Install it, delete the cache folder the message names, and run again
([Caches](advanced.md#73-caches)).

### `probe` finds no embeddings

`probe` reads the cache under a key made from the window settings. Use the same `--set` values as
the `embed` that made the embeddings. If a cache exists under another key, the error lists the
settings that differ.

### The GPU is not supported

The default PyTorch build needs a GPU with compute capability 7.5 or higher and a CUDA 13 driver.
On older hardware, install a CUDA 12 build of PyTorch first
([Installation](../getting-started/installation.md#gpu)).

### Cluster jobs fail on one host

Keep the jobs off that host with `--requirements` ([Job resources](running.md#510-job-resources)).
`neuroatlas status --out DIR` lists each failure with its host and log.
