# Package layout

Three packages: a stable benchmarking engine, the extensions that plug into
it, and the command-line verbs.

## Main areas

- `benchmarking_helpers/`
  The engine. `registry/` holds the contracts, spec discovery and fold
  manifests; `runtime/` the embedding cache, the runner and the determinism
  controls; `probes/` the sklearn probes and their metrics; `channels/` the
  per-cohort channel maps. Importing it does not pull torch.
- `extensions/`
  Everything cohort- or model-specific, and the default place to add new
  benchmark-facing functionality. Model wrappers under
  `extensions/models/backbones/`, low-level dataset I/O under
  `extensions/datasets/dataio/`, corpus builders under
  `extensions/datasets/preprocessors/`, and the shared epilepsy
  preprocessing layer under `extensions/datasets/epilepsy/`.
- `entrypoints/`
  The five verbs -- `fetch`, `prepare`, `embed`, `probe`, `hypnogram` --
  which build a config and call the runner. Nothing else belongs here.

## How to read it

Start at `benchmarking_helpers/registry/contracts.py` for the dataclasses
everything speaks, then `runtime/runner.py` for the execution flow. Read
`extensions/datasets/`, `extensions/models/` and `extensions/tasks/` to see
what is currently exposed.

## Adding something

1. Add or update the cohort dossier under `configs/cohorts/<slug>/`.
2. Add or reuse the implementation behind it under `extensions/`.
3. Update the root README if the public surface changed.
