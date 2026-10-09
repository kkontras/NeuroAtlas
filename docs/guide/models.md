# 3. Models

A model enters a benchmark as a checkpoint, one set of weights. It is never fine-tuned: the
benchmark extracts embeddings with the frozen model and fits a linear probe on them. This chapter
describes the 44 checkpoints, how to select them on the command line, and how to get their
weights.

## 3.1 Groups

The checkpoints fall into four groups. Each group has an alias that selects all of its
checkpoints:

| Alias | Group | Checkpoints | Models |
|---|---|---|---|
| `all_fm` | `eeg_fm` | 12 EEG foundation models | BIOT, CBraMod, EEGPT, LaBraM, Neuro-GPT, NeuroLM, NeuroRVQ, REVE, SleepFM, ST-EEGFormer (S/B/L) |
| `all_ts` | `ts_fm` | 10 general time-series foundation models | Chronos-T5 (T/S/B/L), Moirai (S/B/L), MOMENT (S/B/L) |
| `all_supervised` | `supervised` | 20 models trained with labels on other EEG data | CoRe-Sleep, SleepTransformer, SleePyCo, DeepSOZ-HEM, Seizure-Transformer, EEGNet (12 checkpoints) |
| `all_random` | `baseline` | 2 untrained baselines | CBraMod, REVE |
| `all` | | every checkpoint (44) | |

`neuroatlas list models` lists every checkpoint, and `neuroatlas list models all_fm` one group.
For each checkpoint it shows the family and group, where the weights come from, the sampling rate
and window the model was built for, and the size of its embedding. A benchmark cuts its own
windows (30 s epochs for sleep, 10 s windows for epilepsy, the trial for BCI) and resamples them
to each model's rate. `neuroatlas list aliases` lists the aliases.

## 3.2 Selecting checkpoints

`-m` works the same in `run`, `check`, `submit`, `list models` and `models`. Give an alias, a
group, a family, a checkpoint id, or a comma list of them. Prefix a term with `-` to remove it:

```bash
neuroatlas run sleep_stage -m all_fm,-reve
```

A family selects its trained checkpoints, not its untrained baseline. Unknown names are refused
with exit status 2.

## 3.3 Models a benchmark leaves out

Each benchmark runs the checkpoints the paper evaluates on it. The others are left out of
`-m all` and every alias, and naming one is refused.

| Benchmark | Left out | Why |
|---|---|---|
| `epilepsy` | `sleep_transformer`, `sleepyco`, `core_sleep` | sleep-staging sequence models that take sequences of 30 s epochs, not 10 s windows |
| `epilepsy` | `eegnetv4` | motor-imagery checkpoints, evaluated on motor imagery only |
| sleep benchmarks and `brain_age` | `eegnetv4` | the same |
| sleep benchmarks and `brain_age` | `deepsoz_hem`, `seizure_transformer` | seizure-detection models, evaluated on epilepsy only |
| BCI benchmarks | `sleep_transformer`, `sleepyco`, `core_sleep` | sleep-staging sequence models |
| BCI benchmarks | `deepsoz_hem`, `seizure_transformer` | seizure-detection models |

`list benchmarks` and `show` list the same exclusions.
`neuroatlas list models --benchmark epilepsy` lists the 26 checkpoints epilepsy evaluates. Naming
a left-out checkpoint fails like this:

```text
$ neuroatlas run epilepsy -m sleepyco_shhs_fold0 --dry-run
error: sleepyco_shhs_fold0 is not part of the epilepsy benchmark (sleepyco: sleep-staging sequence models that take sequences of 30 s epochs, not the benchmark's 10 s windows)
  fix: neuroatlas list models --benchmark epilepsy
```

`embed` and `probe` know nothing about benchmarks. They run whatever `--models` names, which is
why `show` writes the exclusions into each command it prints.

## 3.4 Getting the weights

`models status` shows where each checkpoint's weights come from and whether they are here, and
`models download` fetches the ones that are missing:

```bash
neuroatlas models status all_fm
neuroatlas models download all_fm
```

```text
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

`models status -v` adds the path where each checkpoint is expected.

`models download` shows the bytes as they arrive. It exits with status 1 unless every checkpoint
ends up ready.

```text
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

## 3.5 Notes on some checkpoints

Downloading REVE accepts the REVE Responsible Use License.

EEGPT's upstream share on Figshare works only in a browser, so the weights come from a
bit-identical copy on Hugging Face.

Seizure-Transformer's weights are pulled from the authors' Docker image, without Docker. The
download streams up to 3.4 GB of image layers and keeps the 168 MB model.

DeepSOZ-HEM's code and checkpoint are GPL-3.0, so the package does not include them.
`models download deepsoz_hem_pretrained` fetches `baselines.py`, `deepsoz_fold4.pth_4.tar` and
the upstream `LICENSE` from [github.com/amruth-sn/deepsoz-hem](https://github.com/amruth-sn/deepsoz-hem)
at commit a7c13bd. Each file is checked against its recorded SHA-256 and stored in
`<models root>/foundation/deepsoz_hem/`.

CoRe-Sleep and SleepTransformer are release assets of this repository.
