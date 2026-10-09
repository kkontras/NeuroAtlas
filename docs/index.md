---
hide:
  - navigation
  - toc
---

# NeuroAtlas documentation

NeuroAtlas is a Python package and command-line tool, `neuroatlas`, for benchmarking EEG
foundation models. It downloads the datasets and model weights, extracts embeddings with a frozen
model, fits a probe on every fold and reports the results. It covers 12 benchmarks in sleep,
epilepsy, brain age and brain-computer interfaces, with 42 datasets and 44 model checkpoints.

## Installation

Install the package from PyPI, with Python 3.11:

```bash
pip install neuroatlas-bench
```

[Installation](getting-started/installation.md) covers conda and venv, GPUs, and the extra
packages for BCI datasets and time-series models.

## A first run

Choose a project folder, download one dataset and one model, and run a benchmark:

```bash
neuroatlas config init ~/neuroatlas
neuroatlas data download sleep_edf_expanded --mirror aws
neuroatlas models download biot_pretrained
neuroatlas run sleep_stage -m biot_pretrained --debug
neuroatlas results sleep_stage
```

## Contents

[Getting started](getting-started/installation.md)
:   Installation, a quick start, and a walkthrough of every command on open data.

[User guide](guide/index.md)
:   Configuration, data, models, the benchmarks, running them locally or on a cluster, and
    reading the results.

[Command reference](reference/index.md)
:   Every `neuroatlas` command with its options and examples.

[Citation](citation.md)
:   How to cite NeuroAtlas in your work.
