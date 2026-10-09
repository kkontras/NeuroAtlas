# run/

`default_runs.sh` is every experiment the paper reports, one command per line,
spelled `python -m neuroatlas.entrypoints.<verb>`: where to get each dataset,
then extracting the embeddings, fitting the probes and building the hypnogram
features. Its header lists
the settings that are the same everywhere and those each line spells out.

`neuroatlas show <benchmark>` prints the same lines for one benchmark, spelled
`neuroatlas <verb>`, and the test suite checks that the two agree.
`neuroatlas run <benchmark>` runs them on this machine and
`neuroatlas submit <benchmark>` writes them as cluster jobs; see
[the user guide](../docs/guide/index.md).
