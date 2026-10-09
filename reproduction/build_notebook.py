#!/usr/bin/env python3
"""Regenerate ``neuroatlas_minimal_repro.ipynb`` from this cell list.

The notebook is generated rather than hand-edited so that its ``source`` arrays
stay well formed. Each line of a notebook cell must keep its trailing newline,
and hand-editing the JSON tends to collapse a whole cell onto one line.

Edit the ``md(...)`` and ``code(...)`` blocks below, then run:

    python reproduction/build_notebook.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "neuroatlas_minimal_repro.ipynb"

cells = []


def _lines(src):
    return (src.strip("\n") + "\n").splitlines(keepends=True)


def _cell_id(kind):
    """Stable per-cell id. nbformat 4.5 requires one, and deriving it from the
    position keeps regenerated notebooks diffable instead of churning ids."""
    return f"{kind}-{len(cells):03d}"


def md(src):
    cells.append({"cell_type": "markdown", "id": _cell_id("md"),
                  "metadata": {}, "source": _lines(src)})


def code(src):
    cells.append({"cell_type": "code", "id": _cell_id("code"), "execution_count": None,
                  "metadata": {}, "outputs": [], "source": _lines(src)})


# =============================================================== title
md(r"""
# NeuroAtlas: the pipeline, step by step

This notebook walks through what `neuroatlas run` does, one stage per cell. It runs on a
slice of Sleep-EDF Expanded and CHB-MIT and takes a few minutes on one GPU.

By default it uses fold 0 and six subjects per split. Its numbers show how each stage
works. The paper's numbers come from the benchmark commands:

```bash
neuroatlas run sleep_stage -m cbramod_pretrained
neuroatlas run brain_age -m cbramod_pretrained
neuroatlas run epilepsy --dataset chbmit -m cbramod_pretrained
```

Every task goes through the same six steps.

    EDF -> labelled epochs -> channel mapping -> frozen embeddings -> linear probe -> metrics

Only the last step differs between tasks.

| Task | Dataset | Reported |
|---|---|---|
| Sleep staging | Sleep-EDF Expanded | accuracy and kappa, then a hypnogram and clinical features |
| Brain age | Sleep-EDF Expanded | mean absolute error and the brain age gap |
| Seizure detection | CHB-MIT | event sensitivity and false alarms per hour |

You need two open datasets, about 30 GB in total, and CBraMod's weights. Section 0 shows
how to get them. `SUBJECT_LIMIT` in section 5 and `LIMIT_BATCHES` in section 7d set the
size of the slice. Section 8 lists what `neuroatlas run` does differently at full scale.
""")

# =============================================================== 0. env
md(r"""
## 0. Setting up

Use one Python 3.11 kernel in a clone of the repository. Install the package with the
time-series models, MOMENT and matplotlib.

```bash
pip install -e ".[ts]" -c requirements-fm.txt
pip install --no-deps "momentfm==0.1.4"
pip install matplotlib -c requirements-fm.txt
```

The constraints file pins every package to the version the paper used. `--no-deps` stops
momentfm from downgrading transformers and numpy. The cell below checks that every package
the notebook imports is installed.
""")

code(r'''
import importlib
import platform
import sys
import warnings
from pathlib import Path

REQUIRED = ["numpy", "scipy", "pandas", "sklearn", "matplotlib", "torch",
            "edfio", "h5py", "yaml", "xlrd", "transformers", "safetensors",
            "momentfm", "chronos", "easydict"]
PIP_NAMES = {"sklearn": "scikit-learn", "yaml": "pyyaml", "chronos": "chronos-forecasting"}

print(f"python {platform.python_version()}")
# The environment's name only, so a shared copy of the notebook carries no home
# directory.
print(f"kernel {Path(sys.executable).parent.parent.name}\n")

# tqdm prints an IProgress warning that quotes its install path; this notebook
# uses no widgets.
warnings.filterwarnings("ignore", message=".*IProgress.*")

missing = []
for name in REQUIRED:
    try:
        module = importlib.import_module(name)
        print(f"  ok      {name:14s} {getattr(module, '__version__', 'installed')}")
    except Exception as exc:
        print(f"  absent  {name:14s} ({type(exc).__name__})")
        missing.append(name)

if missing:
    raise SystemExit("Missing: " + ", ".join(PIP_NAMES.get(name, name) for name in missing)
                     + ". Install them with the three lines in the cell above.")
print("\nEnvironment is complete.")
''')

md(r"""
### Where your data is

Start the notebook from `reproduction/`, or set `NEUROATLAS_ROOT` to your clone.

The notebook reads each dataset from the folder `neuroatlas data download` puts it in.
Run `neuroatlas data status -v` to see these folders. To read a copy somewhere else, set
`SLEEPEDF_ROOT` or `CHBMIT_ROOT` before you start the kernel.
""")

code(r'''
import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("NEUROATLAS_ROOT", Path.cwd().parent)).resolve()
if not (REPO_ROOT / "src" / "neuroatlas" / "configs" / "folds").is_dir():
    raise SystemExit(f"{REPO_ROOT} is not a NeuroAtlas checkout. Set NEUROATLAS_ROOT.")
sys.path.insert(0, str(REPO_ROOT / "src"))

import neuroatlas.api   # applies ~/.neuroatlas/config.yaml: the data root, each dataset's folder
from neuroatlas.data import raw_location


def dataset_folder(variable, dataset):
    """$variable when set, else the folder `neuroatlas data download` puts the dataset in."""
    if os.environ.get(variable):
        return Path(os.environ[variable]).expanduser()
    _, path, _ = raw_location(dataset)
    if path is None:
        raise SystemExit(f"No data root is set, so {dataset}'s folder is not known: run "
                         f"`neuroatlas config init --data-root DIR`, or set {variable}.")
    return path


SLEEPEDF_ROOT = dataset_folder("SLEEPEDF_ROOT", "sleep_edf_expanded")
CHBMIT_ROOT = dataset_folder("CHBMIT_ROOT", "chbmit")

WORK = REPO_ROOT / "artifacts" / "minimal_repro"
WORK.mkdir(parents=True, exist_ok=True)

FOLD = 0        # fold 0 of the frozen fold files (section 1)
SEED = 0

# Sleep is scored in 30 s epochs, which is the clinical convention and what the
# Sleep-EDF hypnograms annotate. Seizure detection uses 10 s windows with a 10 s
# stride, the epilepsy benchmark's setting; see the note in section 1.
EPOCH_SECONDS = 30.0
CHBMIT_WINDOW_SECONDS = 10.0
# The unipolar montage, so the supervised seizure baseline (pretrained on the
# 19-channel unipolar 10-20 montage) and the foundation models see the same input.
# The epilepsy benchmark reads CHB-MIT in its bipolar montage (section 8).
CHBMIT_MONTAGE = "unipolar"

def show(path):
    """Relative to the repository when possible, so a shared copy of this
    notebook carries no home directory."""
    try:
        return f"<repo>/{path.relative_to(REPO_ROOT)}"
    except ValueError:
        return f"<external>/{path.name}"

for label, path in [("repository", REPO_ROOT), ("Sleep-EDF", SLEEPEDF_ROOT),
                    ("CHB-MIT", CHBMIT_ROOT), ("outputs", WORK)]:
    state = "found" if path.exists() else "missing"
    print(f"  {label:11s} {show(path):46s} [{state}]")
''')

md(r"""
### Getting the data

Download both datasets with the tool.

```bash
neuroatlas data download sleep_edf_expanded --mirror aws   # 8.1 GB, sections 2 to 7c
neuroatlas data download chbmit                             # 21.7 GB, section 7d
```

CHB-MIT is one of the paper's ten epilepsy datasets, and it is open. The download is the
SzCORE BIDS conversion on Zenodo (record 10259996), which is what the benchmark reads. If
CHB-MIT is missing, the notebook skips section 7d.
""")

# =============================================================== 1. folds
md(r"""
## 1. Who trains, who is tested

Every number below depends on which subjects the model learns from and which it is tested
on. So the split comes first, before any EEG file is opened.

The notebook does not compute the splits. It reads the frozen fold files that ship with
the package, the same files `neuroatlas run` reads. Each file holds five folds, and this
notebook uses fold 0 throughout.
""")

code(r'''
from neuroatlas.benchmarking_helpers.registry.fold_manifest import (
    SPLIT_RULE,
    check_subject_grouping,
    load_fold_split,
)

SLEEP_MANIFEST = "sleep_edf_expanded"
CHBMIT_MANIFEST = "chbmit"

sleep_split = load_fold_split(SLEEP_MANIFEST, FOLD)
chbmit_split = load_fold_split(CHBMIT_MANIFEST, FOLD)

print(f"How fold k is read from a file of five partitions:\n  {SPLIT_RULE}\n")
for split in (sleep_split, chbmit_split):
    print(split.summary())
    print(f"    read from {split.source_path.relative_to(REPO_ROOT)}")
''')

md(r"""
### Check one: is a subject on both sides of the split?

Fold files list recordings, not subjects. Sleep-EDF Cassette has two nights per subject.
So a split can keep recordings apart and still put night 1 of a subject in training and
night 2 in test.

The cell below checks the folds the benchmark uses. No subject is on both sides. It then
moves one recording across to show what the check catches.
""")

code(r'''
# Sleep-EDF recording ids look like SC4001E0 or ST7011J0, where the two digits after the
# three character prefix identify the subject and the next digit is the night.
recording_to_subject = lambda recording_id: recording_id[:5]

report = check_subject_grouping(sleep_split, recording_to_subject)
print(f"{SLEEP_MANIFEST}: clean")
print(f"  {report['n_recordings']} recordings from {report['n_subjects']} subjects, "
      f"no subject on both sides of any split\n")

# What the check looks like when a split IS leaky. Take one subject who has two
# recordings in test and move the second into train -- the split stays disjoint over
# recordings, which is exactly what makes this failure easy to miss.
from collections import Counter
from dataclasses import replace

nights = Counter(recording_to_subject(r) for r in sleep_split.test)
two_nights = next((s for s, n in nights.items() if n > 1), None)
if two_nights is None:
    print("\nNo subject has two nights in this fold's test split, so there is nothing to show.")
else:
    moved = [r for r in sleep_split.test if recording_to_subject(r) == two_nights][-1]
    leaky = replace(sleep_split,
                    train=list(sleep_split.train) + [moved],
                    test=[r for r in sleep_split.test if r != moved])
    leaky_report = check_subject_grouping(leaky, recording_to_subject, raise_on_leak=False)
    shared = leaky_report["overlaps"]["train&test"]
    print(f"\nthe same split with {moved} moved from test to train:")
    print(f"  recordings are still disjoint, but {len(shared)} subject is now in both: {shared}")
    print("  check_subject_grouping(..., raise_on_leak=True) would stop the run here")
''')

md(r"""
### What the CHB-MIT split holds

The CHB-MIT fold file records how many windows each split holds in a complete copy. A
partial download or another window length gives other counts.
""")

code(r'''
if not chbmit_split.stats:
    raise SystemExit("The CHB-MIT fold file has no window counts.")

print(f"CHB-MIT fold 0, expected counts at {CHBMIT_WINDOW_SECONDS:g} s windows "
      f"with {CHBMIT_WINDOW_SECONDS:g} s stride:\n")
print(f"  {'split':6s} {'subjects':>8s} {'windows':>10s} {'seizure':>8s}  positive rate")
for name, stats in chbmit_split.stats.items():
    rate = 100 * stats["n_pos_windows"] / stats["n_windows"]
    print(f"  {name:6s} {stats['n_subjects']:8d} {stats['n_windows']:10,d} "
          f"{stats['n_pos_windows']:8,d}  {rate:.3f} percent")

EXPECTED_CHBMIT_WINDOWS = {k: v["n_windows"] for k, v in chbmit_split.stats.items()}
print("\nNote how rare the positive class is. This is why section 7 reports event level "
      "sensitivity and\nfalse alarms per hour rather than accuracy, which would sit above "
      "99 percent for a model that\nnever predicts a seizure at all.")
''')

# =============================================================== 2. preprocessing
md(r"""
## 2. From EDF to labelled epochs

Sleep-EDF ships two files per recording, the PSG and the hypnogram. The hypnogram lists
stage annotations with an onset and a duration, and one annotation can cover many epochs.
Preprocessing turns the pair into 30 second epochs with one label each.

The code below makes four decisions.

1. Stages 3 and 4 merge into N3 (AASM). Movement and unscored epochs are dropped.
2. Epochs are 30 seconds long.
3. Cassette recordings include hours of wake before and after the night, and kept whole,
   Wake would dominate. The notebook keeps 30 minutes of wake before the first sleep
   epoch and after the last one. `neuroatlas run` crops Sleep-EDF the same way.
4. The signals are already in microvolts, so nothing is scaled. The models still get the
   unit, because several of them normalise by it.

The notebook reads both subsets, Cassette and Telemetry. The fold file covers all 197
recordings, and reading one subset would test on part of the split.
""")

code(r'''
import re

import numpy as np

STAGE_LABELS = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # merged into N3 per AASM
    "Sleep stage R": 4,
}
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def find_recordings(root, subsets=("sleep-cassette", "sleep-telemetry"), limit=None):
    """Pair each PSG file with its hypnogram. Returns a list of dicts.

    Both subsets by default: the fold file covers all 197 recordings, 153
    Cassette and 44 Telemetry, and reading only one of them would quietly
    evaluate on a subset of the split it claims to use.
    """
    folders = [Path(root) / s for s in subsets]
    missing = [f for f in folders if not f.is_dir()]
    if missing:
        raise FileNotFoundError(f"Expected {missing[0]}. Did the download finish?")

    found = []
    for folder in folders:
     for psg_path in sorted(folder.glob("*-PSG.edf")):
         match = re.match(r"(S[CT]\d)(\d{2})(\d)([A-Z])", psg_path.name)
         if match is None:
             continue
         prefix, subject, night, _ = match.groups()
         hypnograms = sorted(folder.glob(f"{prefix}{subject}{night}*-Hypnogram.edf"))
         if not hypnograms:
             continue
         found.append({
             "recording_id": psg_path.name.split("-")[0],
             "subject_id": f"{prefix}{subject}",
             "night": int(night),
             "psg_path": psg_path,
             "hypnogram_path": hypnograms[0],
         })
    return found[:limit] if limit else found


def load_epochs(recording, channels=("EEG Fpz-Cz", "EEG Pz-Oz"),
                epoch_seconds=EPOCH_SECONDS, crop_wake_minutes=30):
    """Read one recording into (epochs, labels, sampling_rate).

    epochs has shape (n_epochs, n_channels, n_samples) in microvolts.
    """
    import edfio

    psg = edfio.read_edf(str(recording["psg_path"]))
    hypnogram = edfio.read_edf(str(recording["hypnogram_path"]))

    by_label = {signal.label: signal for signal in psg.signals}
    for name in channels:
        if name not in by_label:
            raise KeyError(f"{recording['recording_id']} has no channel {name!r}. "
                           f"Available: {sorted(by_label)}")

    traces = [by_label[name].data.astype(np.float32) for name in channels]
    sampling_rate = float(psg.signals[0].sampling_frequency)
    samples_per_epoch = int(round(sampling_rate * epoch_seconds))
    n_samples = len(traces[0])

    # Expand each annotation into the 30 s epochs it covers.
    scored = []
    for annotation in hypnogram.annotations:
        label = STAGE_LABELS.get(annotation.text)
        if label is None:
            continue
        start = int(round(annotation.onset * sampling_rate))
        length = int(round(annotation.duration * sampling_rate))
        for offset in range(0, length, samples_per_epoch):
            begin = start + offset
            if begin + samples_per_epoch <= n_samples:
                scored.append((begin, label))

    if not scored:
        return (np.empty((0, len(channels), samples_per_epoch), np.float32),
                np.empty(0, np.int64), sampling_rate)

    # Trim the long wake stretches at either end of the night.
    if crop_wake_minutes is not None:
        asleep = [i for i, (_, label) in enumerate(scored) if label != 0]
        if asleep:
            margin = int(crop_wake_minutes * 60 // epoch_seconds)
            scored = scored[max(0, asleep[0] - margin):asleep[-1] + margin + 1]

    epochs = np.empty((len(scored), len(channels), samples_per_epoch), np.float32)
    labels = np.empty(len(scored), np.int64)
    for i, (begin, label) in enumerate(scored):
        for c, trace in enumerate(traces):
            epochs[i, c] = trace[begin:begin + samples_per_epoch]
        labels[i] = label
    return epochs, labels, sampling_rate
''')

md(r"""
Run it on one recording first. If Wake is 80 percent of the epochs, the cropping did not
work.
""")

code(r'''
import collections

recordings = find_recordings(SLEEPEDF_ROOT)
by_subset = collections.Counter(r["subject_id"][:2] for r in recordings)
print(f"Found {len(recordings)} recordings "
      f"from {len({r['subject_id'] for r in recordings})} subjects "
      f"({by_subset['SC']} Cassette, {by_subset['ST']} Telemetry).")

# The fold file lists every recording it expects, so say plainly whether this
# matches. A silent shortfall here is the difference between running the split
# the file describes and running some subset of it.
expected = set(sleep_split.train) | set(sleep_split.val) | set(sleep_split.test)
present = {r["recording_id"][:7] for r in recordings}
if expected - present:
    print(f"  WARNING: {len(expected - present)} recordings in the fold file are "
          f"not on disk, e.g. {sorted(expected - present)[:4]}")
else:
    print(f"  all {len(expected)} recordings the fold file names are present\n")

example = recordings[0]
# The two EEG derivations Sleep-EDF provides. Section 3 covers how these names get
# translated into each model's own channel vocabulary.
EXAMPLE_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz")
epochs, labels, sampling_rate = load_epochs(example, channels=EXAMPLE_CHANNELS)

print(f"Recording {example['recording_id']}, subject {example['subject_id']}, "
      f"night {example['night']}")
print(f"  sampling rate     {sampling_rate:g} Hz")
print(f"  epochs            {epochs.shape[0]} of {EPOCH_SECONDS:g} s "
      f"({epochs.shape[0] * EPOCH_SECONDS / 3600:.1f} h scored)")
print(f"  array shape       {epochs.shape}  (epochs, channels, samples)")
print(f"  amplitude         {epochs.min():.0f} to {epochs.max():.0f} microvolts\n")
print("  stage distribution")
for value, count in zip(*np.unique(labels, return_counts=True)):
    bar = "#" * int(40 * count / len(labels))
    print(f"    {STAGE_NAMES[value]:4s} {count:5d}  {100 * count / len(labels):5.1f}%  {bar}")
''')

md(r"""
Now look at what the model gets, one 30 second epoch per stage.

Wake is fast and low in amplitude. N3 is slow, high-amplitude delta. N2 sits between them,
with spindles and K complexes. N1 is the hard class. It is a transition that looks like
slightly slower Wake, and expert scorers disagree on it more than on any other stage.
""")

code(r'''
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})
STAGE_COLOURS = {0: "tab:gray", 1: "tab:orange", 2: "tab:blue",
                 3: "tab:purple", 4: "tab:red"}

figure, axes = plt.subplots(5, 1, figsize=(10, 7), sharex=True, sharey=True)
seconds = np.arange(epochs.shape[-1]) / sampling_rate
for stage, axis in enumerate(axes):
    where = np.where(labels == stage)[0]
    if len(where) == 0:
        axis.set_visible(False)
        continue
    axis.plot(seconds, epochs[where[len(where) // 2], 0],   # a middle example, channel 0
              linewidth=0.6, color=STAGE_COLOURS[stage])
    axis.set_ylabel(f"{STAGE_NAMES[stage]}\n(uV)")
    axis.text(0.995, 0.90, f"{len(where)} epochs", transform=axis.transAxes,
              ha="right", va="top", fontsize=8, color="0.35")
axes[-1].set_xlabel("seconds within the epoch")
axes[0].set_xlim(0, EPOCH_SECONDS)
figure.suptitle(f"{example['recording_id']}, channel {EXAMPLE_CHANNELS[0]}: "
                f"one 30 s epoch per sleep stage", y=0.995)
figure.tight_layout()
plt.show()
''')

# =============================================================== 3. channel mapping
md(r"""
## 3. Matching electrode names to each model

Each model was pretrained with its own electrode names, and they do not agree. Sleep-EDF
records `EEG Fpz-Cz` and `EEG Pz-Oz`. LaBraM expects `FPZ` and `PZ`, REVE expects `Fpz`
and `Pz`, and CBraMod takes the bipolar labels as they are. A wrong mapping does not
crash. The model gets a channel it has never seen, and the embeddings get worse without
a warning.

So the mapping lives in data, not code. Each dataset has one file,
`src/neuroatlas/configs/channel_maps/<dataset>.yaml`, with a `per_model` block for each
model family.
""")

code(r'''
import yaml   # ships with the conda/pip base in practice; part of pyyaml

channel_map_path = REPO_ROOT / "src" / "neuroatlas" / "configs" / "channel_maps" / "sleep_edf_expanded.yaml"
channel_map = yaml.safe_load(channel_map_path.read_text())

print(f"{channel_map_path.relative_to(REPO_ROOT)}\n")
print(f"  channels available in the dataset : {channel_map['channels_available']}")
print(f"  channels the benchmark uses       : {channel_map['channels_used']}\n")

CHANNELS = list(channel_map["channels_used"])
PER_MODEL = channel_map["per_model"]

print("  translation per model family")
for family, mapping in sorted(PER_MODEL.items()):
    if isinstance(mapping, str):
        # A bare string is a directive rather than a mapping. "skip" means this
        # model cannot be evaluated on this dataset at all, usually because its
        # channel vocabulary has nothing in common with what the dataset records.
        print(f"    {family:18s} directive: {mapping}")
    else:
        rendered = ", ".join(f"{src} -> {dst}" for src, dst in mapping.items())
        print(f"    {family:18s} {rendered}")


def channels_for(family, channels=CHANNELS):
    """Dataset channel names translated into one model family's vocabulary."""
    mapping = PER_MODEL.get(family)
    if mapping is None:
        return list(channels)          # channel agnostic, pass the names through
    if isinstance(mapping, str):
        if mapping == "skip":
            raise ValueError(
                f"{channel_map['dataset']} marks {family!r} as 'skip': this model is "
                f"not evaluated on this dataset. Choose another model.")
        raise ValueError(f"Unrecognised directive {mapping!r} for {family!r}.")
    return [mapping.get(name, name) for name in channels]


print(f"\n  cbramod sees {channels_for('cbramod')} where the EDF says {CHANNELS}")
print(f"  moment sees  {channels_for('moment')}, unchanged, because a generic time "
      f"series\n               model has no electrode vocabulary to map onto")
skipped = [f for f, m in PER_MODEL.items() if m == "skip"]
if skipped:
    print(f"\n  marked 'skip' for this dataset: {', '.join(skipped)}")
''')

# =============================================================== 4. models
md(r"""
## 4. Choosing models

The paper compares three kinds of model.

* EEG foundation models, pretrained on large EEG corpora.
* Generic time-series models, never trained on EEG. If they keep up, EEG pretraining is
  not what makes the difference.
* An untrained control, with the same architecture and random weights. It tells a good
  architecture apart from useful pretraining.

The cell below asks the tool which checkpoints are ready on this machine, as
`neuroatlas models status` does. Fetch any other with
`neuroatlas models download <checkpoint>`.
""")

code(r'''
from neuroatlas import api
from neuroatlas.benchmarking_helpers import checkpoint_registry

registry = checkpoint_registry()
by_id = registry if isinstance(registry, dict) else {c.identifier: c for c in registry}

# The states `neuroatlas models status` reports, one row per checkpoint.
status = api.models("all").set_index("checkpoint")
READY = {"found", "in Hugging Face cache", "no weights needed"}

print(f"{len(status)} checkpoints, by the state of their weights on this machine:\n")
print(status["state"].value_counts().to_string())
print(f"\nready: {', '.join(i for i in status.index if status.at[i, 'state'] in READY)}")
print("\n`neuroatlas models download <checkpoint>` fetches the weights of any other.")
''')

md(r"""
The notebook defines four checkpoints in `MODELS`. CBraMod is the EEG foundation model,
MOMENT and Chronos are the generic ones, and CBraMod with random weights is the control.
The walkthrough embeds with CBraMod, set by `MODEL_FOR_WALKTHROUGH` in section 5.

The cell below also lists two supervised baselines, CoRe-Sleep and Seizure-Transformer. It
prints each checkpoint's role, embedding size and state.
""")

code(r'''
import torch

MODELS = {
    "cbramod_pretrained":  ("cbramod", "CBraModBackbone", "EEG foundation model"),
    "cbramod_random_init": ("cbramod", "CBraModBackbone", "untrained control"),
    "moment_small":        ("moment", "MomentBackbone", "generic time series model"),
    "chronos_t5_small":    ("chronos", "ChronosBackbone", "generic time series model"),
}

# Task specific supervised baselines, listed with the state of their weights
# (`neuroatlas models download <checkpoint>` fetches them).
SUPERVISED = {
    "core_sleep_shhs_fold0": (
        "core_sleep", "CoreSleepBackbone", "supervised sleep baseline"),
    "seizure_transformer_pretrained": (
        "seizure_transformer", "SeizureTransformerBackbone", "supervised seizure baseline"),
}

# CBraMod's pretraining window. A 30 s sleep epoch is embedded as three of these.
CBRAMOD_WINDOW_SECONDS = 10.0

AVAILABLE_SUPERVISED = {}
for identifier in SUPERVISED:
    state = status.at[identifier, "state"]
    if state in READY:
        AVAILABLE_SUPERVISED[identifier] = SUPERVISED[identifier]
    print(f"  supervised  {identifier:32s} {state}")

# Everything the notebook can instantiate, keyed the same way.
ALL_MODELS = {**MODELS, **SUPERVISED}

print()
for identifier, (_, _, role) in {**MODELS, **AVAILABLE_SUPERVISED}.items():
    spec = by_id[identifier]
    print(f"  {identifier:32s} {role:26s} dim={spec.embedding_dim:<5} "
          f"{status.at[identifier, 'state']}")


def load_backbone(identifier):
    """Instantiate a backbone. Weights download on the first call and are then cached."""
    module_name, class_name, _ = ALL_MODELS[identifier]
    module = importlib.import_module(
        f"neuroatlas.extensions.models.backbones.{module_name}")
    return getattr(module, class_name)(by_id[identifier])


print(f"\nDevice: {'cuda' if torch.cuda.is_available() else 'cpu'}")
''')

# =============================================================== 5. embeddings
md(r"""
## 5. Extracting embeddings

The protocol is frozen-backbone linear probing. Each epoch goes through the model once,
the notebook keeps the output vector, and the weights never change. This measures the
representation, not the training recipe, and the expensive step runs once per model.

Each batch carries the signal and the metadata the model needs. Models use
`sampling_rate` to resample, `unit` to normalise and `channels` to look up per-electrode
embeddings. This is where the mapping from section 3 comes in.

CBraMod was pretrained on 10 second windows. The notebook cuts each 30 second epoch into
three such windows and concatenates the three vectors. `neuroatlas run` gives CBraMod the
whole 30 second epoch and gets one embedding per epoch, as in App. C.2 of the paper.
""")

code(r'''
def make_batch(epoch_array, family, sampling_rate):
    """Wrap epochs as the batch dictionary a backbone expects."""
    tensor = torch.from_numpy(np.ascontiguousarray(epoch_array)).float()
    n = tensor.shape[0]
    meta = [{
        "dataset": "sleep_edf_expanded",
        "sampling_rate": float(sampling_rate),
        "unit": "uV",
        "channels": channels_for(family),
        "epoch_seconds": tensor.shape[-1] / sampling_rate,
    } for _ in range(n)]
    return {"signals": {"eeg": tensor},
            "label": torch.zeros(n, dtype=torch.long),   # unused, required by the contract
            "meta": meta}


def embed(backbone, family, epoch_array, sampling_rate, batch_size=64,
          window_seconds=None):
    """Embed epochs, optionally in fixed sub windows that are then concatenated."""
    if window_seconds is not None:
        samples = int(round(window_seconds * sampling_rate))
        n_windows = epoch_array.shape[-1] // samples
        if n_windows < 1:
            raise ValueError(f"Epoch is shorter than the model's {window_seconds:g} s window.")
        pieces = [embed(backbone, family,
                        epoch_array[..., i * samples:(i + 1) * samples],
                        sampling_rate, batch_size)
                  for i in range(n_windows)]
        return np.concatenate(pieces, axis=1)

    out = []
    for start in range(0, epoch_array.shape[0], batch_size):
        chunk = epoch_array[start:start + batch_size]
        out.append(np.asarray(backbone.extract_embeddings(
            make_batch(chunk, family, sampling_rate))))
    return np.concatenate(out, axis=0) if out else np.empty((0, 0), np.float32)


# Sanity check on the single recording loaded in section 2, before the full pass.
probe_backbone = load_backbone("cbramod_pretrained")
sample = embed(probe_backbone, "cbramod", epochs[:8], sampling_rate,
               window_seconds=CBRAMOD_WINDOW_SECONDS)
print(f"8 epochs of {EPOCH_SECONDS:g} s at {sampling_rate:g} Hz")
print(f"  input  {epochs[:8].shape}")
print(f"  output {sample.shape}   "
      f"({int(EPOCH_SECONDS // CBRAMOD_WINDOW_SECONDS)} windows x "
      f"{sample.shape[1] // int(EPOCH_SECONDS // CBRAMOD_WINDOW_SECONDS)} dims)")
print(f"  finite {np.isfinite(sample).all()}")
''')

md(r"""
Now run the full pass over fold 0.

This is the slow cell. Fold 0's three splits together cover the whole cohort, so every
subject gets embedded. `SUBJECT_LIMIT` keeps the first N subjects of each split, so you
can run the whole notebook quickly first. Set it to `None` to embed all of fold 0.
""")

code(r'''
import time

# Number of subjects kept per split. None means the whole of fold 0.
# Override without editing the notebook: export NEUROATLAS_SUBJECT_LIMIT=none
_limit = os.environ.get("NEUROATLAS_SUBJECT_LIMIT", "6").strip().lower()
SUBJECT_LIMIT = None if _limit in ("none", "0", "all", "") else int(_limit)
MODEL_FOR_WALKTHROUGH = "cbramod_pretrained"


AVAILABLE_SUBJECTS = {r["subject_id"] for r in recordings}


def subjects_in(split_name, split=sleep_split, limit=SUBJECT_LIMIT):
    """Subjects of one split, in the fold file's order, restricted to what is on disk.

    The fold file covers 197 recordings across the Cassette and Telemetry subsets.
    If you downloaded only one subset, or are working from a partial copy, the
    missing subjects are dropped here rather than silently producing an empty
    split further down.
    """
    ordered, seen = [], set()
    for recording_id in getattr(split, split_name):
        subject = recording_to_subject(recording_id)
        if subject in seen or subject not in AVAILABLE_SUBJECTS:
            continue
        seen.add(subject)
        ordered.append(subject)
    return ordered[:limit] if limit else ordered


wanted = {name: set(subjects_in(name)) for name in ("train", "val", "test")}
selected = [r for r in recordings
            if any(r["subject_id"] in group for group in wanted.values())]

print(f"Subjects per split: "
      + ", ".join(f"{k}={len(v)}" for k, v in wanted.items())
      + f"  ({len(selected)} recordings)")
if SUBJECT_LIMIT:
    print(f"SUBJECT_LIMIT={SUBJECT_LIMIT}, so this is a demonstration slice, "
          f"not the full fold.\n")

backbone = load_backbone(MODEL_FOR_WALKTHROUGH)
family = MODELS[MODEL_FOR_WALKTHROUGH][0]

features, stage_labels, subject_of_epoch, recording_of_epoch = [], [], [], []
per_recording = {}
started = time.time()

for i, recording in enumerate(selected, 1):
    epoch_array, epoch_labels, rate = load_epochs(recording, channels=CHANNELS)
    if epoch_array.shape[0] == 0:
        print(f"  [{i}/{len(selected)}] {recording['recording_id']} has no scored epochs")
        continue
    vectors = embed(backbone, family, epoch_array, rate,
                    window_seconds=CBRAMOD_WINDOW_SECONDS)
    features.append(vectors)
    stage_labels.append(epoch_labels)
    subject_of_epoch += [recording["subject_id"]] * len(epoch_labels)
    recording_of_epoch += [recording["recording_id"]] * len(epoch_labels)
    per_recording[recording["recording_id"]] = {
        "subject_id": recording["subject_id"], "n_epochs": len(epoch_labels)}
    print(f"  [{i}/{len(selected)}] {recording['recording_id']}  "
          f"{len(epoch_labels):5d} epochs -> {vectors.shape}")

X = np.concatenate(features).astype(np.float32)
y = np.concatenate(stage_labels)
subject_of_epoch = np.array(subject_of_epoch)
recording_of_epoch = np.array(recording_of_epoch)

print(f"\nEmbedding matrix {X.shape}, labels {y.shape}, "
      f"{len(per_recording)} recordings, {time.time() - started:.0f} s")
''')

md(r"""
Do these vectors hold what the linear probe is about to look for? Project them onto
their first two principal components and colour each point by its true stage.

Clear structure is good evidence that the probe has something to work with. Overlap is
weak evidence against it, since the plot shows two directions out of 600.
""")

code(r'''
from sklearn.decomposition import PCA

rng = np.random.default_rng(SEED)
sample = rng.choice(len(X), size=min(6000, len(X)), replace=False)
centred = X[sample] - X[sample].mean(axis=0)
projected = PCA(n_components=2, random_state=SEED).fit_transform(centred)

figure, axis = plt.subplots(figsize=(6.4, 5.2))
for stage in range(5):
    mask = y[sample] == stage
    if mask.sum():
        axis.scatter(projected[mask, 0], projected[mask, 1], s=3, alpha=0.35,
                     color=STAGE_COLOURS[stage],
                     label=f"{STAGE_NAMES[stage]} ({mask.sum()})")
axis.set_xlabel("first principal component")
axis.set_ylabel("second principal component")
axis.set_title(f"{MODEL_FOR_WALKTHROUGH} embeddings\n{len(sample)} epochs sampled "
               f"from {X.shape[1]} dimensions")
axis.legend(markerscale=4, frameon=False, fontsize=8)
figure.tight_layout()
plt.show()
''')

# =============================================================== 6. probe
md(r"""
## 6. Fitting the probe

The probe is a multinomial logistic regression on the frozen embeddings. Three details
matter more than the choice of classifier.

1. Split by subject, never by epoch. Neighbouring 30 second epochs of one night are nearly
   identical. A random epoch split puts them on both sides, and accuracy goes up without
   meaning anything.
2. Fit the standardisation on the training set only. Otherwise test statistics leak in.
3. Choose the regularisation on the validation set. The test set is used once, at the end.
""")

code(r'''
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

in_split = {name: np.isin(subject_of_epoch, sorted(group))
            for name, group in wanted.items()}
for name, mask in in_split.items():
    print(f"  {name:5s} {mask.sum():6d} epochs from "
          f"{len(set(subject_of_epoch[mask]))} subjects")

overlap = set(subject_of_epoch[in_split["train"]]) & set(subject_of_epoch[in_split["test"]])
assert not overlap, f"Subject leakage between train and test: {overlap}"
print("\n  no subject appears in both train and test")

scaler = StandardScaler().fit(X[in_split["train"]])
Xs = {name: scaler.transform(X[mask]) for name, mask in in_split.items()}
ys = {name: y[mask] for name, mask in in_split.items()}

from sklearn.metrics import cohen_kappa_score

best = None
print("\n  selecting regularisation strength on validation")
for C in (0.001, 0.01, 0.1, 1.0):
    model = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    model.fit(Xs["train"], ys["train"])
    kappa = cohen_kappa_score(ys["val"], model.predict(Xs["val"]))
    print(f"    C={C:<7g} validation kappa {kappa:.4f}")
    if best is None or kappa > best[0]:
        best = (kappa, C, model)

val_kappa, best_C, probe = best
print(f"\n  chosen C={best_C:g} (validation kappa {val_kappa:.4f})")
''')

md(r"""
### The test set, once

`predicted` is the only place where test labels meet a prediction. All of section 7 comes
from it.
""")

code(r'''
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

predicted = probe.predict(Xs["test"])
truth = ys["test"]

accuracy = accuracy_score(truth, predicted)
macro_f1 = f1_score(truth, predicted, average="macro")
kappa = cohen_kappa_score(truth, predicted)

print(f"Sleep staging on {len(truth)} held out epochs "
      f"from {len(set(subject_of_epoch[in_split['test']]))} subjects\n")
print(f"  accuracy        {accuracy:.4f}")
print(f"  macro F1        {macro_f1:.4f}")
print(f"  Cohen's kappa   {kappa:.4f}\n")
present = sorted(set(truth) | set(predicted))
print(classification_report(truth, predicted, labels=present,
                            target_names=[STAGE_NAMES[i] for i in present],
                            digits=3, zero_division=0))
print("Confusion matrix, rows are true stages and columns predicted:\n")
print("        " + "".join(f"{STAGE_NAMES[i]:>7s}" for i in present))
for row, i in zip(confusion_matrix(truth, predicted, labels=present), present):
    print(f"  {STAGE_NAMES[i]:5s} " + "".join(f"{v:7d}" for v in row))
''')

md(r"""
Here is the same matrix as a picture. Each row is normalised, so it shows where the
epochs of one true stage ended up. A perfect model gives a bright diagonal.

Follow the N1 row. Its errors spread into Wake, N2 and REM instead of one neighbour. That
means the representation does not separate N1 at all, which is worse than confusing two
adjacent stages.
""")

code(r'''
matrix = confusion_matrix(truth, predicted, labels=present).astype(float)
normalised = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)

figure, axis = plt.subplots(figsize=(5.4, 4.6))
image = axis.imshow(normalised, cmap="magma", vmin=0, vmax=1)
axis.set_xticks(range(len(present)), [STAGE_NAMES[i] for i in present])
axis.set_yticks(range(len(present)), [STAGE_NAMES[i] for i in present])
axis.set_xlabel("predicted stage")
axis.set_ylabel("true stage")
axis.set_title("Confusion matrix, each row normalised")
axis.grid(False)
for r in range(len(present)):
    for c in range(len(present)):
        axis.text(c, r, f"{normalised[r, c]:.2f}", ha="center", va="center", fontsize=8,
                  color="white" if normalised[r, c] < 0.6 else "black")
figure.colorbar(image, ax=axis, fraction=0.046, label="fraction of the true stage")
figure.tight_layout()
plt.show()
''')

# =============================================================== 7a. hypnogram
md(r"""
## 7. The numbers that matter

Most sleep papers stop at per-epoch accuracy. The paper goes further. A model can be
accurate per epoch and still produce a hypnogram no clinician would accept. What matters
clinically is the whole night, such as how long the patient slept, how broken up the sleep
was and when REM arrived. Two models with the same accuracy can disagree on all three.

### 7a. Reconstruct the hypnogram

Predictions come back in extraction order. Group them by recording and you have the
hypnogram.
""")

code(r'''
test_recordings = recording_of_epoch[in_split["test"]]

hypnograms = {}
for recording_id in dict.fromkeys(test_recordings):     # preserves order
    mask = test_recordings == recording_id
    hypnograms[recording_id] = {"true": truth[mask], "predicted": predicted[mask]}

print(f"Reconstructed {len(hypnograms)} hypnograms from the test split:\n")
for recording_id, night in hypnograms.items():
    agreement = (night["true"] == night["predicted"]).mean()
    hours = len(night["true"]) * EPOCH_SECONDS / 3600
    print(f"  {recording_id}  {len(night['true']):5d} epochs "
          f"({hours:4.1f} h)  per epoch agreement {agreement:.3f}")
''')

md(r"""
Read a hypnogram as a staircase over time, with Wake at the top, N3 at the bottom and REM
highlighted. With the truth above the prediction, the failure modes are easy to see. Look
for REM read as N2, and for extra transitions that break up the night.
""")

code(r'''
import matplotlib.pyplot as plt

# Plot order puts deep sleep at the bottom and REM just below Wake, the usual convention.
PLOT_ROW = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}          # stage -> row
PLOT_TICKS = ["N3", "N2", "N1", "REM", "W"]

def draw_hypnogram(axis, stages, title, colour):
    rows = np.array([PLOT_ROW[int(s)] for s in stages])
    hours = np.arange(len(stages)) * EPOCH_SECONDS / 3600
    axis.step(hours, rows, where="post", linewidth=0.9, color=colour)
    rem = np.where(stages == 4)[0]
    for i in rem:                                   # highlight REM bouts
        axis.axvspan(i * EPOCH_SECONDS / 3600, (i + 1) * EPOCH_SECONDS / 3600,
                     color="tab:red", alpha=0.18, linewidth=0)
    axis.set_yticks(range(5), PLOT_TICKS)
    axis.set_ylim(-0.5, 4.5)
    axis.set_xlim(0, hours[-1] if len(hours) else 1)
    axis.set_ylabel("stage")
    axis.set_title(title, loc="left", fontsize=10)
    axis.grid(axis="x", alpha=0.25, linewidth=0.5)

shown = next(iter(hypnograms))
night = hypnograms[shown]

figure, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True)
draw_hypnogram(axes[0], night["true"], f"{shown}: expert scoring", "tab:blue")
draw_hypnogram(axes[1], night["predicted"],
               f"{shown}: {MODEL_FOR_WALKTHROUGH} linear probe "
               f"(per epoch agreement {(night['true'] == night['predicted']).mean():.3f})",
               "tab:orange")
axes[1].set_xlabel("hours from start of the scored recording")
figure.suptitle("Hypnogram, expert against predicted (red shading marks REM)",
                fontsize=11, y=0.99)
figure.tight_layout()
plt.show()
''')

md(r"""
### 7b. Clinical features

These are the features a sleep report contains. The notebook writes them out so you can
see what each one measures.

| Feature | Definition |
|---|---|
| TST | time in any stage other than Wake |
| Sleep efficiency | TST over the scored recording |
| WASO | Wake after the first sleep epoch |
| REM latency | time from sleep onset to the first REM epoch |
| Awakenings | transitions from sleep into Wake |

What matters is the error against the expert's hypnogram, not the value itself. That
error is the clinical readout, and it can rank models differently from plain accuracy.
`neuroatlas run sleep_hypnogram` computes 34 such features.
""")

code(r'''
import pandas as pd


def hypnogram_features(stages, epoch_seconds=EPOCH_SECONDS):
    """Clinical summary of one night. Times are in minutes."""
    stages = np.asarray(stages)
    per_epoch = epoch_seconds / 60.0
    asleep = stages != 0
    total = len(stages) * per_epoch

    if not asleep.any():
        return {"TST": 0.0, "SleepEff": 0.0, "WASO": 0.0,
                "RemLatency": np.nan, "Awakenings": 0}

    onset = int(np.argmax(asleep))
    after_onset = stages[onset:]
    rem = np.where(after_onset == 4)[0]
    wake_transitions = int(np.sum((stages[:-1] != 0) & (stages[1:] == 0)))
    return {
        "TST": float(asleep.sum() * per_epoch),
        "SleepEff": float(100.0 * asleep.sum() * per_epoch / total),
        "WASO": float((after_onset == 0).sum() * per_epoch),
        "RemLatency": float(rem[0] * per_epoch) if len(rem) else np.nan,
        "Awakenings": wake_transitions,
    }


rows = []
for recording_id, night in hypnograms.items():
    expert = hypnogram_features(night["true"])
    model = hypnogram_features(night["predicted"])
    for feature in expert:
        rows.append({"recording_id": recording_id, "feature": feature,
                     "expert": expert[feature], "predicted": model[feature],
                     "error": model[feature] - expert[feature]})

features_frame = pd.DataFrame(rows)
wide = features_frame.pivot(index="recording_id", columns="feature",
                            values=["expert", "predicted"])
print("Per night values, expert against predicted:\n")
print(wide.round(1).to_string())

summary = (features_frame.groupby("feature")
           .agg(mean_absolute_error=("error", lambda s: s.abs().mean()),
                bias=("error", "mean"))
           .reindex(["TST", "SleepEff", "WASO", "RemLatency", "Awakenings"]))
print("\nAgreement across the test nights "
      "(minutes, except SleepEff in percent and Awakenings as a count):\n")
print(summary.round(2).to_string())
print("\nBias is signed, so it tells you the direction of the systematic error: "
      "a positive\nWASO bias means the model reports the patient as more awake than the "
      "expert did.")
print("\nWatch for features where the absolute value of the bias equals the mean absolute")
print("error. Then the error points the same way in every night, a systematic artefact")
print("rather than noise. REM latency and the awakening count are prone to this. A first")
print("occurrence or a count of transitions defines them, so one misclassified epoch moves")
print("them a long way. One spurious REM epoch in the first minutes sets REM latency to")
print("nearly zero, however good the rest of the night is. Clinical scoring rules avoid")
print("this with a minimum bout length. The definitions above leave it out on purpose,")
print("because the fragility is the point. A model can look good per epoch and still be")
print("unusable for what a sleep report contains.")
''')

# =============================================================== 7c. brain age
md(r"""
### 7c. Brain age

The same embeddings serve a second task without another extraction. That is why they are
cached. Predict chronological age from the night's EEG, then report the brain age gap,
the predicted age minus the true age. A gap that stays positive means the brain looks
older than it is.

Two things differ from staging. Age is known per recording, so the notebook averages the
epoch embeddings of each night. The target is continuous, so the probe is a ridge
regression and the metric is the mean absolute error in years.
""")

code(r'''
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score


def load_ages(root):
    """Read both subject tables into {subject_id: age}, keyed like recording ids.

    The two ship in different shapes: SC-subjects.xls is one row per recording
    with tidy column names, ST-subjects.xls is one row per subject under a
    two-line header.
    """
    ages = {}
    cassette = pd.read_excel(Path(root) / "SC-subjects.xls")
    for _, row in cassette.iterrows():
        ages.setdefault(f"SC4{int(row['subject']):02d}", float(row["age"]))

    telemetry = pd.read_excel(Path(root) / "ST-subjects.xls", header=1)
    for _, row in telemetry.iterrows():
        try:
            nr, age = int(row["Nr"]), float(row["Age"])
        except (TypeError, ValueError, KeyError):
            continue                      # the stray header/blank rows
        ages.setdefault(f"ST7{nr:02d}", age)
    return ages


ages = load_ages(SLEEPEDF_ROOT)

# One pooled vector per recording, the mean over its epochs. `neuroatlas run brain_age`
# pools one mean vector per subject.
pooled, pooled_age, pooled_subject = [], [], []
for recording_id in dict.fromkeys(recording_of_epoch):
    subject = per_recording[recording_id]["subject_id"]
    if subject not in ages:
        continue
    pooled.append(X[recording_of_epoch == recording_id].mean(axis=0))
    pooled_age.append(ages[subject])
    pooled_subject.append(subject)

pooled = np.vstack(pooled)
pooled_age = np.asarray(pooled_age, dtype=np.float64)
pooled_subject = np.asarray(pooled_subject)

is_train = np.isin(pooled_subject, sorted(wanted["train"] | wanted["val"]))
is_test = np.isin(pooled_subject, sorted(wanted["test"]))
print(f"Pooled to {pooled.shape[0]} recordings, {pooled.shape[1]} dimensions")
print(f"  train and validation {is_train.sum()} recordings")
print(f"  test                 {is_test.sum()} recordings")
print(f"  age range            {pooled_age.min():.0f} to {pooled_age.max():.0f} years\n")

age_scaler = StandardScaler().fit(pooled[is_train])
regressor = Ridge(alpha=10.0, random_state=SEED).fit(
    age_scaler.transform(pooled[is_train]), pooled_age[is_train])

predicted_age = regressor.predict(age_scaler.transform(pooled[is_test]))
true_age = pooled_age[is_test]
gap = predicted_age - true_age

print(f"  mean absolute error   {mean_absolute_error(true_age, predicted_age):.2f} years")
print(f"  r2                    {r2_score(true_age, predicted_age):.3f}")
print(f"  brain age gap         {gap.mean():+.2f} years (sd {gap.std():.2f})\n")
for subject, actual, estimate in zip(pooled_subject[is_test], true_age, predicted_age):
    print(f"    {subject}  true {actual:5.1f}  predicted {estimate:5.1f}  "
          f"gap {estimate - actual:+5.1f}")
''')

md(r"""
Plot predicted against true age. The dashed diagonal is a perfect prediction, and each
vertical line is one night's gap.

Watch for a flat cloud, with predictions close to the training mean whatever the true age.
A regressor does this when the features carry no age information, and the MAE can still
look normal. The dotted line marks the training mean. If the points follow it more closely
than the diagonal, the model learned the cohort average and nothing else. A negative r2
says the same.
""")

code(r'''
figure, axis = plt.subplots(figsize=(5.6, 5.2))
limits = [min(true_age.min(), predicted_age.min()) - 4,
          max(true_age.max(), predicted_age.max()) + 4]
axis.plot(limits, limits, color="0.5", linestyle="--", linewidth=1,
          label="perfect prediction")
axis.axhline(pooled_age[is_train].mean(), color="tab:orange", linewidth=1, linestyle=":",
             label="mean age of the training set")
for actual, estimate in zip(true_age, predicted_age):
    axis.plot([actual, actual], [actual, estimate], color="tab:blue", alpha=0.3,
              linewidth=0.8)
axis.scatter(true_age, predicted_age, s=42, alpha=0.85, color="tab:blue",
             edgecolor="white", linewidth=0.5, zorder=3)
axis.set_xlim(limits)
axis.set_ylim(limits)
axis.set_xlabel("chronological age (years)")
axis.set_ylabel("predicted age (years)")
axis.set_title(f"Brain age on {int(is_test.sum())} held out nights\n"
               f"MAE {mean_absolute_error(true_age, predicted_age):.1f} y,  "
               f"r2 {r2_score(true_age, predicted_age):.2f},  "
               f"mean gap {gap.mean():+.1f} y")
axis.legend(frameon=False, fontsize=8, loc="upper left")
figure.tight_layout()
plt.show()
''')

md(r"""
With a handful of subjects these statistics are noise, and a ridge fitted on a few nights
predicts close to the training mean. Here the point is the mechanics. Set `SUBJECT_LIMIT`
to `None` for values you can interpret.
""")

# =============================================================== 7d. epilepsy
md(r"""
### 7d. Seizure detection over whole recordings

Window-level metrics mislead most in epilepsy. Seizures take up well under one percent of
a recording, so a model that never predicts a seizure scores above 99 percent accuracy.
Clinicians ask two other questions.

* Did you catch it? Any-overlap sensitivity counts a seizure as detected if any positive
  prediction overlaps it.
* How often do you raise a false alarm? False alarms per hour over the full recording
  decide whether a detector is usable.

Both are event-level metrics. The predictions have to stay in order along the timeline, so
this section works on whole recordings.

CHB-MIT comes as BIDS. Parsing BIDS teaches nothing about the protocol, so the package's
reader does that one step. Everything after it is in the cells.
""")

code(r'''
from neuroatlas.extensions.datasets.adapters.chbmit import CHBMITBenchmarkDataModule

# Look for the BIDS tree itself, not just the download directory: a download that
# is still running, or was interrupted before unzipping, leaves the directory in
# place with no usable data in it. `neuroatlas data download chbmit` unpacks the
# tree into BIDS_CHB-MIT/ under the dataset's folder.
BIDS_ROOT = CHBMIT_ROOT if CHBMIT_ROOT.name == "BIDS_CHB-MIT" else CHBMIT_ROOT / "BIDS_CHB-MIT"
RUN_EPILEPSY = any(BIDS_ROOT.glob("sub-*")) if BIDS_ROOT.is_dir() else False

if not RUN_EPILEPSY:
    print(f"No CHB-MIT BIDS tree with sub-* directories under {show(BIDS_ROOT)}.")
    print("Get it with:  neuroatlas data download chbmit   (21.7 GB)")
    print("Skipping section 7d.")
else:
    chbmit = CHBMITBenchmarkDataModule(
        backend="bids",
        bids_root=str(BIDS_ROOT),
        fold=FOLD,
        n_folds=5,
        window_s=CHBMIT_WINDOW_SECONDS,
        stride_s=CHBMIT_WINDOW_SECONDS,
        montage=CHBMIT_MONTAGE,
        label_mode="binary",
        overlap_threshold=0.0,
        balance="none",          # keep the true class ratio; see the note below
        batch_size=64,
        num_workers=2,
        folds_manifest="chbmit",
        strict_folds=True,
    )
    print("CHB-MIT module ready.")
    fold_source = str(chbmit.metadata.get("fold_source", ""))
    print(f"  split taken from {fold_source.replace(str(REPO_ROOT), '<repo>')}")
    print(f"  label space      {chbmit.metadata.get('canonical_label_space')}")
    print(f"  window           {chbmit.metadata.get('epoch_seconds')} s")
''')

md(r"""
Two of the settings above matter. `balance="none"` keeps the real class ratio of the
windows, so false alarms per hour are counted against real time. `strict_folds=True` stops
the run if the fold file names a subject your download lacks. A partial cohort gives
numbers you cannot compare with anything.
""")

code(r'''
def event_metrics(true_windows, predicted_windows,
                  window_seconds=CHBMIT_WINDOW_SECONDS):
    """Any overlap sensitivity and false alarms per hour, from window level labels.

    Both sequences are laid out along the recording timeline. Consecutive positive
    windows are merged into events before matching, which is what makes this event
    level rather than window level.
    """
    def to_events(binary):
        events, start = [], None
        for i, value in enumerate(list(binary) + [0]):
            if value and start is None:
                start = i
            elif not value and start is not None:
                events.append((start, i))
                start = None
        return events

    true_events = to_events(true_windows)
    predicted_events = to_events(predicted_windows)

    detected = sum(
        any(p_start < t_end and t_start < p_end
            for p_start, p_end in predicted_events)
        for t_start, t_end in true_events)
    false_alarms = sum(
        not any(t_start < p_end and p_start < t_end
                for t_start, t_end in true_events)
        for p_start, p_end in predicted_events)

    hours = len(true_windows) * window_seconds / 3600.0
    return {
        "n_seizures": len(true_events),
        "n_detected": detected,
        "any_overlap_sensitivity": detected / len(true_events) if true_events else np.nan,
        "n_false_alarms": false_alarms,
        "false_alarms_per_hour": false_alarms / hours if hours else np.nan,
        "hours": hours,
    }


# A worked example on a synthetic recording, so the accounting is verifiable by eye.
demo_true = np.array([0] * 20 + [1, 1, 1] + [0] * 30 + [1, 1] + [0] * 25)
demo_pred = np.array([0] * 21 + [1, 1] + [0] * 40 + [1] + [0] * 15)
print("Worked example on a synthetic recording:")
print(f"  true seizures at windows      {[(a, b) for a, b in [(20, 23), (53, 55)]]}")
for key, value in event_metrics(demo_true, demo_pred).items():
    print(f"  {key:26s} {value}")
print("\n  The first seizure overlaps a prediction, so it is detected. The second does "
      "not.\n  The stray prediction near the end overlaps no seizure, so it is one false "
      "alarm.")
''')

code(r'''
if RUN_EPILEPSY:
    seizure_backbone = load_backbone(MODEL_FOR_WALKTHROUGH)

    def embed_loader(loader, limit_batches=None):
        vectors, labels, groups = [], [], []
        for i, batch in enumerate(loader):
            if limit_batches and i >= limit_batches:
                break
            vectors.append(np.asarray(seizure_backbone.extract_embeddings(batch)))
            labels.append(np.asarray(batch["label"]).reshape(-1))
            groups += [m.get("recording_id", m.get("subject_id", "?"))
                       for m in batch["meta"]]
        return (np.concatenate(vectors), np.concatenate(labels), np.array(groups))

    # Batches of CHB-MIT windows kept per split. None means the whole of fold 0,
    # which is 353,824 windows and roughly two hours. Override the same way as
    # SUBJECT_LIMIT: export NEUROATLAS_LIMIT_BATCHES=none
    _batches = os.environ.get("NEUROATLAS_LIMIT_BATCHES", "40").strip().lower()
    LIMIT_BATCHES = None if _batches in ("none", "0", "all", "") else int(_batches)
    print("Embedding CHB-MIT windows. This is the slow part.\n")
    Xtr, ytr, _ = embed_loader(chbmit.train_dataloader(), LIMIT_BATCHES)
    Xte, yte, groups_te = embed_loader(chbmit.test_dataloader(), LIMIT_BATCHES)
    print(f"  train {Xtr.shape}, {int(ytr.sum())} seizure windows")
    print(f"  test  {Xte.shape}, {int(yte.sum())} seizure windows")

    seizure_scaler = StandardScaler().fit(Xtr)
    detector = LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=SEED)
    detector.fit(seizure_scaler.transform(Xtr), ytr)

    scores = detector.predict_proba(seizure_scaler.transform(Xte))[:, 1]
    print(f"\n  fitted on {len(ytr)} windows, scoring {len(yte)} test windows")
else:
    print("Section 7d skipped, CHB-MIT is not present.")
''')

md(r"""
The decision threshold is the last free parameter, and it trades one metric against the
other. Lower it and you catch more seizures but raise more false alarms. A single
operating point hides this, so the sweep below shows the whole curve.
""")

code(r'''
if RUN_EPILEPSY:
    from sklearn.metrics import average_precision_score, roc_auc_score

    print(f"  window level AUROC   {roc_auc_score(yte, scores):.4f}")
    print(f"  average precision    {average_precision_score(yte, scores):.4f}")
    print(f"  (positive rate in test: {100 * yte.mean():.3f} percent)\n")

    print(f"  {'threshold':>9s} {'sensitivity':>12s} {'FA/hour':>9s} {'detected':>9s}")
    sweep = []
    for threshold in (0.5, 0.7, 0.9, 0.95, 0.99):
        metrics = event_metrics(yte, (scores >= threshold).astype(int))
        sweep.append({"threshold": threshold, **metrics})
        print(f"  {threshold:9.2f} {metrics['any_overlap_sensitivity']:12.3f} "
              f"{metrics['false_alarms_per_hour']:9.2f} "
              f"{metrics['n_detected']:4d}/{metrics['n_seizures']:<4d}")

    headline = event_metrics(yte, (scores >= 0.5).astype(int))
    print(f"\n  At threshold 0.5, over {headline['hours']:.1f} h of recording:")
    print(f"    any overlap sensitivity  {headline['any_overlap_sensitivity']:.3f}")
    print(f"    false alarms per hour    {headline['false_alarms_per_hour']:.2f}")
    print("\n  These two numbers are the axes of the paper's headline epilepsy metric. The "
          "Event-Sens@FA AUC\n  integrates sensitivity over 0.1 to 100 false alarms per hour "
          "(App. C.1). Window level\n  accuracy would look excellent here whatever the model "
          "does. That is why the paper\n  reports event level metrics.")
''')

md(r"""
The top panel shows the detector's score along the recording, with the annotated seizures
shaded. This is what event-level means in practice. For each shaded region, you ask
whether any detection touched it. Then you count the detections that landed outside all
of them.

The bottom panel shows the tradeoff, one point per threshold. Where to operate on it is a
clinical choice, not a modelling one. That is why a single operating point hides what a
reader needs.
""")

code(r'''
if RUN_EPILEPSY:
    figure, axes = plt.subplots(2, 1, figsize=(11, 5.6),
                                gridspec_kw={"height_ratios": [2, 1.5]})

    hours = np.arange(len(scores)) * CHBMIT_WINDOW_SECONDS / 3600
    axes[0].plot(hours, scores, linewidth=0.7, color="tab:blue", label="seizure score")
    axes[0].axhline(0.5, color="tab:red", linestyle="--", linewidth=1, label="threshold 0.5")

    labelled = False
    run_start = None
    for i, value in enumerate(list(yte) + [0]):        # shade the annotated seizures
        if value and run_start is None:
            run_start = i
        elif not value and run_start is not None:
            axes[0].axvspan(run_start * CHBMIT_WINDOW_SECONDS / 3600,
                            i * CHBMIT_WINDOW_SECONDS / 3600,
                            color="tab:red", alpha=0.25, linewidth=0,
                            label=None if labelled else "annotated seizure")
            labelled = True
            run_start = None

    axes[0].set_ylabel("score")
    axes[0].set_xlabel("hours into the held out recording")
    axes[0].set_xlim(0, hours[-1] if len(hours) else 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Detector score along the recording")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right", ncol=3)

    sweep_frame = pd.DataFrame(sweep)
    axes[1].plot(sweep_frame["false_alarms_per_hour"],
                 sweep_frame["any_overlap_sensitivity"], marker="o", color="tab:purple")
    for _, row in sweep_frame.iterrows():
        axes[1].annotate(f"{row['threshold']:.2f}",
                         (row["false_alarms_per_hour"], row["any_overlap_sensitivity"]),
                         textcoords="offset points", xytext=(6, -4), fontsize=8)
    axes[1].set_xlabel("false alarms per hour")
    axes[1].set_ylabel("any overlap\nsensitivity")
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].set_title("Operating points, labelled by decision threshold")
    figure.tight_layout()
    plt.show()
else:
    print("Section 7d was skipped, so there is nothing to plot here.")
''')

# =============================================================== 8. caveats
md(r"""
## 8. What `neuroatlas run` does at full scale

This notebook keeps every stage small enough to read. The benchmark commands run the same
stages at the paper's scale and with the paper's settings.

| Stage | This notebook | `neuroatlas run` |
|---|---|---|
| Folds and subjects | fold 0, `SUBJECT_LIMIT` subjects per split, `LIMIT_BATCHES` batches of CHB-MIT windows | all five folds, every subject |
| CBraMod on a 30 s epoch | three 10 s windows, concatenated | the whole epoch, one embedding (App. C.2) |
| Sleep staging probe | C from 0.001 to 1, chosen on validation kappa | C from 0.001 to 100, chosen per fold on validation kappa |
| Hypnogram features | 5 | 34, in `neuroatlas run sleep_hypnogram` |
| Brain age | the sleep-staging fold 0, one vector per recording, ridge alpha 10 | Sleep Cassette subjects in age-stratified folds, one mean embedding per subject, alpha chosen by nested cross-validation from 0.1 to 100 |
| Seizure detection | unipolar montage, C = 1, five thresholds | bipolar montage, C chosen per fold from six values on validation AUPRC, Event-Sens@FA AUC over 0.1 to 100 false alarms per hour (App. C.1) |

To run all of fold 0 here, set `SUBJECT_LIMIT` and `LIMIT_BATCHES` to `None`. You can also
export `NEUROATLAS_SUBJECT_LIMIT=none` and `NEUROATLAS_LIMIT_BATCHES=none` before you start
the kernel.

### Going further

* Edit `build_notebook.py`, not the `.ipynb`. Then run `python reproduction/build_notebook.py`
  to regenerate the notebook.
* `neuroatlas show <benchmark>` prints the `embed` and `probe` commands `neuroatlas run`
  executes.
* `run/default_runs.sh` lists every experiment the paper reports.
* Run `check_subject_grouping` from section 1 on any fold file you add.
""")

# =============================================================== write
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1))
n_code = sum(c["cell_type"] == "code" for c in cells)
print(f"wrote {OUT} ({len(cells)} cells: {n_code} code, {len(cells) - n_code} markdown)")
