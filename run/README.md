# run/

- `default_runs.sh` -- every experiment the paper reports, one verb
  invocation per line. It is the record of what was run, and the benchmark
  catalog (`src/neuroatlas/configs/benchmarks/`) is held to it by a test:
  `neuroatlas show <benchmark>` prints lines equivalent to its lines for
  that benchmark, spelled `neuroatlas <verb>` instead of
  `python -m neuroatlas.entrypoints.<verb>`.
- `launch.sh`, `_launch_sets.py` -- the shell launcher that preceded the
  `neuroatlas` command. `neuroatlas run <benchmark>` (on this machine) and
  `neuroatlas submit <benchmark>` (HTCondor or SLURM jobs) do what it did,
  per benchmark rather than per verb; it is kept for existing job scripts.

To run anything, use the command -- see `docs/user_guide.md`.
