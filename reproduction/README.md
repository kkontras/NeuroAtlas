# Minimal reproduction

Two copies of the same notebook live here:

| File | What it is |
|---|---|
| `neuroatlas_minimal_repro.ipynb` | the notebook to run yourself, with no outputs stored |
| `neuroatlas_minimal_repro.executed.ipynb` | the same notebook after a complete run, with every result and all six figures saved inside it |

Open the executed one on GitHub if you want to see what the pipeline produces without
downloading 30 GB of EEG first. It was run on the full Sleep-EDF Cassette subset and on
CHB-MIT, so the numbers and plots in it are real rather than illustrative.

`neuroatlas_minimal_repro.ipynb` walks through the NeuroAtlas evaluation pipeline one stage
at a time, on a small enough slice of data to run on a single machine. Nothing is hidden
behind a wrapper script: each stage is a short cell whose output you can inspect before
moving on.

The stages are read raw EDF, map channel names into each model's vocabulary, pick models,
extract frozen embeddings, fit a linear probe on the training subjects, and turn the
predictions on held out subjects into the metrics the paper reports.

That last stage happens three times, because the three tasks report very different things:

| Task | Dataset | Reported |
|---|---|---|
| Sleep staging | Sleep-EDF Expanded | per epoch accuracy and Cohen's kappa, then a reconstructed hypnogram (with a plot) and the clinical features derived from it |
| Brain age | Sleep-EDF Expanded, Cassette subset | mean absolute error and the brain age gap |
| Seizure detection | CHB-MIT | event level sensitivity and false alarms per hour over whole recordings |

## Which models you can actually run

Most of the model weights used in the paper are not distributed with this repository. They
live in a directory called `artifacts/` that is excluded from version control, so a fresh
clone does not have them and there is no way to download them from here.

This notebook is therefore built around models whose weights it can obtain on its own,
either downloading from Hugging Face the first time you run it or reading a file that is
small enough to be committed. Section 4 sorts every model in the benchmark into those
categories and prints the result, so you can see what is and is not available to you before
anything fails.

The consequence is that the EEG side of the comparison is CBraMod alone. REVE, NeuroLM,
LaBraM, BIOT and EEGPT need local weights. If you have them, put them under
`artifacts/models/` and add their identifiers to `MODELS` in section 4; nothing else
changes.

| Role | Checkpoint | Weights |
|---|---|---|
| EEG foundation model | `cbramod_pretrained` | Hugging Face |
| Untrained control, same architecture | `cbramod_random_init` | none needed |
| Generic time series model | `moment_small` | Hugging Face |
| Generic time series model | `chronos_t5_small` | Hugging Face |
| Supervised sleep baseline | `core_sleep_shhs_fold0` | `artifacts/`, optional |
| Supervised seizure baseline | `seizure_transformer_pretrained` | `artifacts/`, optional |

The last two are checked for per file and skipped with an explanation when absent, so the
notebook runs either way. The check tests for the Git LFS pointer magic rather than file
size, because CoRe-Sleep's normalisation stats are a legitimate 2.5 kB npz that a size
threshold misreads as a stub.

The contrast between the EEG model and the two generic ones is the paper's central
comparison. The random initialised control is what separates "this architecture suits the
task" from "the pretraining helped".

## Install

```bash
pip install -e ".[notebook]"
```

or explicitly:

```bash
pip install numpy scipy pandas scikit-learn matplotlib torch \
            edfio h5py pyyaml xlrd transformers safetensors momentfm chronos-forecasting \
            easydict
```

One Python 3.11 kernel. Two absences are deliberate. There is no `mne`, `braindecode` or
`physioex`, because both Sleep-EDF tasks read raw EDF through `edfio` and no PhysioEx model
is used. There is also no jax, uni2ts or gluonts: `requirements-tsfm.txt` describes a
second environment, but that split exists for moirai, timesfm and lag-llama, whereas MOMENT
and Chronos are ordinary torch plus transformers.

`xlrd` is needed because Sleep-EDF ships its age metadata as a legacy `.xls` file that
pandas cannot open without it.

## Downloading the data

Two public datasets, about 30 GB total, no credentials and no data use agreements.

```bash
# Sleep-EDF Expanded, about 8 GB, covers the sleep staging and brain age sections
wget -r -N -c -np -nH --cut-dirs=1 -P "$(dirname "$SLEEPEDF_ROOT")" \
     https://physionet.org/files/sleep-edfx/1.0.0/

# CHB-MIT, about 22 GB, covers the seizure detection section
python -m neuroatlas.entrypoints.fetch --dataset chbmit --download   # Zenodo record 10259996
```

Point the notebook at them with `SLEEPEDF_ROOT` and `CHBMIT_ROOT`. Without them it falls
back to `data/sleep-edf-database-expanded-1.0.0` and `chbmit_cache/raw` under the
repository root.

CHB-MIT is here because the paper's headline epilepsy cohort, TUSZ, is released under a
Temple University data use agreement that has to be signed and approved, so it cannot
appear in a notebook anyone can run. CHB-MIT is open and exercises the same task and the
same event level metrics. The absolute numbers are not comparable to the paper's TUSZ rows.
The protocol is.

## How subjects are divided into train and test

Fold 0 only, read verbatim from `src/neuroatlas/configs/folds/` rather than recomputed. That is the only
way a reproduction lands on the paper's partitions. The adapters used here (`chbmit`,
`sleep_edf_expanded`, `sleepedf_raw_brain_age`) accept a `folds_manifest` key that makes
them take frozen subject lists from disk instead of running their own k-fold at import
time. `benchmarking_helpers/fold_manifest.py` is the reader, and it handles both on-disk
layouts.

Using one fold saves less than it looks. Fold 0's train, validation and test splits
together cover the whole cohort, so reading, preprocessing and embedding are unchanged.
Only the probe fitting gets cheaper.

### Why the same person must never appear on both sides

Manifests list recordings, not subjects, and Sleep-EDF Cassette records two nights per
subject. A split that is disjoint over recordings can still put night 1 of a subject in
train and night 2 of the same subject in test, which quietly inflates everything
downstream.

An earlier Sleep-EDF manifest here had exactly that defect: 55 of its 78 subjects were spread
over more than one fold, and in fold 0, 16 of the 26 test subjects also appear in train.
The notebook therefore uses `sleep_edf_expanded.json`, which is clean, and prints the
comparison rather than hiding it. Run any manifest you add through
`fold_manifest.check_subject_grouping` before trusting numbers computed from it.

The CHB-MIT manifest also carries per fold window counts, so the notebook can assert
232,291 train, 44,468 validation and 77,065 test windows after preprocessing and fail
loudly if your copy of the data diverges.

## How long it takes to run

About two and a half minutes end to end on one GPU at the defaults, measured at 154 s.
Embedding 37,665 sleep epochs is 22 s of that; most of the rest is reading EDF files. It is
meant to stay that way, so that it is something you run while reading it.

`SUBJECT_LIMIT` (section 5) keeps six subjects per split and `LIMIT_BATCHES` (section 7d)
keeps 40 batches of CHB-MIT windows. Raising them costs time roughly linearly: full fold 0
is about ten minutes for sleep and around two hours for CHB-MIT, which is 353,824 windows.
Neither is needed to follow the pipeline, and the defaults are the intended scale.

## What differs from the paper

The paper averages folds 0 to 4 over seeds 0, 1 and 2, so loop `FOLD` over `range(5)` and
average to match it.

Section 8 of the notebook lists every difference from the published protocol in one place.
The two worth knowing before you start: CBraMod was pretrained on 10 second windows while
sleep is scored in 30 second epochs, and the notebook handles that by embedding three
consecutive windows and concatenating, whereas the harness applies the per checkpoint
`runtime_overrides` in the registry. And the seizure section sets `balance="none"` so that
false alarms per hour is measured against real elapsed time, where the harness default is a
weighted sampler.

## Changing the notebook

```bash
python reproduction/build_notebook.py
```

Edit `build_notebook.py`, not the `.ipynb`. A notebook cell's `source` is a list of lines
that each have to keep their trailing newline, and hand-editing the JSON tends to collapse
a whole cell onto one line.
