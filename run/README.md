# Run Scripts

`run/` contains thin entrypoint wrappers around the benchmark entrypoints.

## Layout

```text
run/
  sleep/    contributor-facing wrappers
  condor/   cluster submission examples
  configs/  benchmark configs consumed by the wrappers
```

## Rule

Wrappers should only:

- resolve paths
- choose the Python interpreter
- call an entrypoint module with arguments

Benchmark orchestration logic belongs in:

- `src/benchmarking_helpers/`
- `src/extensions/tasks/`

## Current direction

The repository is moving toward generic benchmark entrypoints plus config-driven runs.
Keep adding wrapper scripts only when there is real user convenience or cluster
integration value.
