# Installation

We tested the code on Linux with Python 3.11. Python 3.10 or newer works.

## Install the package

Clone the repository and install it into a new environment:

=== "conda"

    ```bash
    git clone https://github.com/kkontras/NeuroAtlas.git
    cd NeuroAtlas
    conda create -n neuroatlas python=3.11 -y
    conda activate neuroatlas
    pip install -e ".[fm]" -c requirements-fm.txt
    ```

=== "venv"

    ```bash
    git clone https://github.com/kkontras/NeuroAtlas.git
    cd NeuroAtlas
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[fm]" -c requirements-fm.txt
    ```

`requirements-fm.txt` pins every dependency to the version used in the paper. The install takes
about 5 minutes and 6.5 GB of disk space.

The distribution is called `neuroatlas-bench` and is installed from the repository. The command
and the Python import are both `neuroatlas`.

> [!WARNING]
> Do not `pip install neuroatlas`. The package on PyPI called `neuroatlas` is unrelated to this
> project.

Check the install:

```bash
neuroatlas --version
```

From a clone this prints the version and the commit, for example `neuroatlas 0.1.0 (1c13d9e)`.

## GPU

The default PyTorch build needs a GPU with compute capability 7.5 or higher and a CUDA 13 driver.
On older hardware, install a CUDA 12 build of PyTorch first. The probes themselves run on the CPU.

## Extra packages

Some models and datasets need more packages:

| To run | Install |
|---|---|
| EEG foundation models, supervised baselines, sleep and epilepsy datasets | `[fm]` |
| Chronos | `pip install -e ".[fm,ts]" -c requirements-fm.txt` |
| MOMENT | the line above, then `pip install --no-deps "momentfm==0.1.4"` |
| Moirai | a separate environment from `requirements-tsfm.txt` (Python 3.10, torch 2.4.1) |
| the 14 MOABB BCI datasets | `pip install -e ".[fm,bci]" -c requirements-fm.txt`, then `pip install --no-deps "moabb==1.2.0"` |

=== "BCI datasets (MOABB)"

    ```bash
    pip install -e ".[fm,bci]" -c requirements-fm.txt
    pip install --no-deps "moabb==1.2.0"
    ```

=== "Chronos and MOMENT"

    ```bash
    pip install -e ".[fm,ts]" -c requirements-fm.txt
    pip install --no-deps "momentfm==0.1.4"
    ```

=== "Moirai"

    Moirai needs an older PyTorch and has its own environment (Python 3.10, torch 2.4.1),
    described in `requirements-tsfm.txt`.

`--no-deps` keeps the old pins that moabb 1.2.0 and momentfm declare from downgrading the rest of
the stack. Both run correctly with it. `pip check` will still list those declared pins.

The `[fm]` extra includes `xlrd` and `openpyxl`, which read the ages of Sleep-EDF and ISRUC
subjects for brain age. Dreyer2023 and Kim2025BetaRange are newer than moabb 1.2.0, so their
readers ship inside the package.

## Next

[Quick start](quick-start.md) sets up a project folder and runs a first benchmark.
