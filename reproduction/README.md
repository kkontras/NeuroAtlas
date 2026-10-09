# Pipeline walkthrough

`neuroatlas_minimal_repro.ipynb` walks through what `neuroatlas run` does, one stage per
cell. It runs on a slice of Sleep-EDF Expanded and CHB-MIT and takes a few minutes on one
GPU.

By default it uses fold 0 and six subjects per split, so its numbers show how each stage
works. The paper's numbers come from the benchmark commands:

```bash
neuroatlas run sleep_stage -m cbramod_pretrained
neuroatlas run brain_age -m cbramod_pretrained
neuroatlas run epilepsy --dataset chbmit -m cbramod_pretrained
```

The notebook ships without outputs. Run it to see them.

## What it covers

Every task goes through the same stages. Read the raw EDF, map the channel names to each
model, extract frozen embeddings, fit a linear probe on the training subjects, and compute
the metrics on the test subjects.

| Task | Dataset | Reported |
|---|---|---|
| Sleep staging | Sleep-EDF Expanded | accuracy and Cohen's kappa, then a hypnogram and five clinical features |
| Brain age | Sleep-EDF Expanded | mean absolute error and the brain age gap |
| Seizure detection | CHB-MIT | event-level sensitivity and false alarms per hour over whole recordings |

The notebook embeds with CBraMod. It also defines MOMENT, Chronos and an untrained CBraMod
in `MODELS`, and section 4 prints the state of each checkpoint's weights.

## Install

Use Python 3.11 and run these lines in a clone of the repository.

```bash
pip install -e ".[ts]" -c requirements-fm.txt
pip install --no-deps "momentfm==0.1.4"
pip install matplotlib -c requirements-fm.txt
```

The constraints file pins every package to the version the paper used. `--no-deps` stops
momentfm from downgrading transformers and numpy.

## Get the weights and the data

```bash
neuroatlas config init ~/neuroatlas                       # once, to set the project folder
neuroatlas models download cbramod_pretrained
neuroatlas data download sleep_edf_expanded --mirror aws   # 8.1 GB
neuroatlas data download chbmit                             # 21.7 GB
```

The notebook reads both datasets from the folders `data download` puts them in. To read a
copy somewhere else, set `SLEEPEDF_ROOT` or `CHBMIT_ROOT`. Without CHB-MIT, the notebook
skips the seizure section.

CHB-MIT is one of the paper's ten epilepsy datasets, and it is open.

## Run it

Start Jupyter in `reproduction/` and run the cells in order. To start it elsewhere, set
`NEUROATLAS_ROOT` to your clone.

In section 5, `SUBJECT_LIMIT` keeps six subjects per split. In section 7d, `LIMIT_BATCHES`
keeps 40 batches of CHB-MIT windows. To run all of fold 0, set both to `None`, or export
`NEUROATLAS_SUBJECT_LIMIT=none` and `NEUROATLAS_LIMIT_BATCHES=none` before you start
Jupyter. Fold 0 of CHB-MIT has 353,824 windows, so a full run takes much longer.

## Folds

The notebook reads fold 0 from the frozen fold files that ship with the package.
`neuroatlas run` reads the same files. Fold 0's train, validation and test splits together
cover the whole dataset, so using one fold makes only the probe fitting cheaper.
`SUBJECT_LIMIT` is what keeps the notebook fast.

Sleep-EDF Cassette records two nights per subject, so a split of recordings can still put
one subject on both sides. Section 1 checks that no subject is on both sides of the
benchmark's folds. It then moves one recording across to show what the check catches.

## What `neuroatlas run` does differently

Section 8 of the notebook compares the two stage by stage. In short, `neuroatlas run` uses
all five folds and every subject, embeds each 30 s epoch whole with CBraMod, picks the
probe's C from a wider grid, and scores seizure detection by the event-level Sens@FA AUC.

## Change the notebook

Edit `build_notebook.py`, not the `.ipynb`, then regenerate the notebook.

```bash
python reproduction/build_notebook.py
```

A notebook cell's `source` is a list of lines that each keep their trailing newline.
Editing the JSON by hand tends to collapse a cell onto one line.
