# Package layout

Four parts: the `neuroatlas` command, a stable benchmarking engine, the
extensions that plug into it, and the original verbs.

## Main areas

- `cli/` and the top-level modules
  The command. `cli/` has one module per command (argument parsing and
  output only); the work is in `catalog.py` (benchmarks), `selectors.py`
  (`-m`), `config.py` and `_paths.py` (settings and where files go),
  `data.py`, `models.py`, `check.py`, `run.py`, `submit.py`,
  `results.py` and `rescore.py`; `api.py` exposes the same verbs to Python.
  `predictions.py` defines the file every probe fold saves its test
  predictions in, and scores it. `cli/_gendocs.py` writes `docs/cli.md`
  from the parsers.
- `benchmarking_helpers/`
  The engine. `registry/` holds the contracts, spec discovery and fold
  manifests; `runtime/` the embedding cache, the runner, resource sizing and
  the determinism controls; `probes/` the sklearn probes and their metrics;
  `channels/` the per-cohort channel maps. Importing it does not pull torch.
- `extensions/`
  Everything cohort- or model-specific, and the default place to add new
  benchmark-facing functionality. Model wrappers under
  `extensions/models/backbones/` (vendored upstream code in `third_party/`;
  DeepSOZ-HEM's GPL-3.0 code and checkpoint are fetched into the models root
  by `models download`, not vendored -- `_checkpoint_download.GITHUB_COMMIT_FILES`
  pins them),
  dataset adapters under `extensions/datasets/adapters/`, low-level dataset
  I/O under `extensions/datasets/dataio/` (`moabb_vendored/` holds the two
  MOABB readers moabb 1.2.0 lacks, Dreyer2023 and Kim2025BetaRange, under
  their BSD-3 licence), corpus builders under
  `extensions/datasets/preprocessors/`, the shared epilepsy layer under
  `extensions/datasets/epilepsy/`, and the probe tasks under
  `extensions/tasks/`.
- `entrypoints/`
  The five verbs -- `fetch`, `prepare`, `embed`, `probe`, `hypnogram` --
  which build a config and call the runner. `neuroatlas run` calls `embed`
  and `probe` exactly as they are called by hand.
- `configs/`
  The shipped tables: benchmarks, cohort manifests, channel maps, folds,
  task presets, model groups. See `configs/README.md`.

## How to read it

Start at `benchmarking_helpers/registry/contracts.py` for the dataclasses
everything speaks, then `runtime/runner.py` for the execution flow. Read
`extensions/datasets/`, `extensions/models/` and `extensions/tasks/` to see
what is currently exposed.

## Adding something

1. Add or update the cohort dossier under `src/neuroatlas/configs/cohorts/<slug>/`.
2. Add or reuse the implementation behind it under `extensions/`. A new
   model family joins `-m all` on every benchmark; where it should not run,
   list it under that benchmark's `excluded_models`
   (`configs/benchmarks/<name>.yaml`, as epilepsy does for the sleep-staging
   sequence models), and give that benchmark's lines of `run/default_runs.sh`
   the same `--models` (`neuroatlas show <name>` prints it).
3. Update the root README and `docs/user_guide.md` if the public surface
   changed, and regenerate `docs/cli.md` if a parser did
   (`python -m neuroatlas.cli._gendocs`; `--check` says whether it is current).
4. Run the tests (`pytest tests`). They run offline, in a temporary home
   with its own roots, and fail if a test touches the real
   `~/.neuroatlas`. `NEUROATLAS_TEST_MODELS_ROOT=<dir>` lends them weights
   (the tests that load real models then run instead of skipping),
   `NEUROATLAS_TEST_DATA_ROOT=<dir>` lends data, and
   `NEUROATLAS_TEST_ONLINE=1` allows downloads.
