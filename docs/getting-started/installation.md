# Installation

NeuroAtlas requires Python 3.10 or newer and runs on Linux.

## Install the package

Install the package from PyPI into a new environment:

=== "conda"

    ```bash
    conda create -n neuroatlas python=3.11 -y
    conda activate neuroatlas
    pip install neuroatlas-bench
    ```

=== "venv"

    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install neuroatlas-bench
    ```

The install takes about 5 minutes and 6.5 GB of disk space. To install exactly the package
versions used in the paper, add the constraints file:

```bash
pip install neuroatlas-bench -c https://raw.githubusercontent.com/kkontras/NeuroAtlas/v0.1.1/requirements-fm.txt
```

 The distribution is called `neuroatlas-bench`; the
command and the Python import are both `neuroatlas`.

> [!WARNING]
> Do not `pip install neuroatlas`. The package on PyPI called `neuroatlas` is unrelated to this
> project.

To work on the code, or to run `run/default_runs.sh` and the notebook in `reproduction/`, install
from a clone instead:

```bash
git clone https://github.com/kkontras/NeuroAtlas.git
cd NeuroAtlas
pip install -e . -c requirements-fm.txt
```

Check the install:

```bash
neuroatlas --version
```

It prints `neuroatlas 0.1.0`. From a clone it also prints the commit.

## GPU

The default PyTorch build needs a GPU with compute capability 7.5 or higher and a CUDA 13 driver.
On older hardware, install a CUDA 12 build of PyTorch first. The probes themselves run on the CPU.

## Extra packages

Some models and datasets need more packages:

| To run | Install |
|---|---|
| EEG foundation models, supervised baselines, sleep and epilepsy datasets | the base install |
| Chronos | `pip install "neuroatlas-bench[ts]"` |
| MOMENT | the line above, then `pip install --no-deps "momentfm==0.1.4"` |
| Moirai | a separate environment from `requirements-tsfm.txt` (Python 3.10, torch 2.4.1) |
| the 14 MOABB BCI datasets | `pip install "neuroatlas-bench[bci]"`, then `pip install --no-deps "moabb==1.2.0"` |

=== "BCI datasets (MOABB)"

    ```bash
    pip install "neuroatlas-bench[bci]"
    pip install --no-deps "moabb==1.2.0"
    ```

=== "Chronos and MOMENT"

    ```bash
    pip install "neuroatlas-bench[ts]"
    pip install --no-deps "momentfm==0.1.4"
    ```

=== "Moirai"

    Moirai needs an older PyTorch and has its own environment (Python 3.10, torch 2.4.1),
    described in `requirements-tsfm.txt`.

`--no-deps` keeps the old pins that moabb 1.2.0 and momentfm declare from downgrading the rest of
the stack. Both run correctly with it. `pip check` will still list those declared pins.

The base install includes `xlrd` and `openpyxl`, which read the ages of Sleep-EDF and ISRUC
subjects for brain age. Dreyer2023 and Kim2025BetaRange are newer than moabb 1.2.0, so their
readers ship inside the package.

## Next

[Quick start](quick-start.md) sets up a project folder and runs a first benchmark.
