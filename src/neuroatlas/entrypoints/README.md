# Entrypoints

This package contains thin command entrypoints for the benchmark system.

## Use This Package For

- config-driven benchmark entrypoints
- preprocessing entrypoints
- reporting or aggregation entrypoints
- wrapper-facing entrypoints that should stay stable over time

## Design Rule

Keep orchestration logic out of entrypoint modules when it belongs in the reusable
benchmark architecture:

- discovery belongs in `benchmarking_helpers/registry/discovery.py`
- run orchestration belongs in `benchmarking_helpers/runtime/runner.py`
- evaluation behavior belongs in `extensions/tasks/`

Entrypoint modules should mainly:

- parse arguments
- build config
- call the runner or a focused helper

`run/default_runs.sh` calls the modules of this package
(`python -m neuroatlas.entrypoints.<verb>`); `neuroatlas <verb>` runs the same
code.
