# Benchmarks

One file per benchmark: what is predicted, on which datasets, how a run is
invoked, and how it is scored. `neuroatlas list benchmarks` prints them,
`neuroatlas show <benchmark>` explains one and `neuroatlas run <benchmark>`
executes one.

Every command a benchmark expands to is equivalent to a line of
`run/default_runs.sh`, the record of what the paper ran: `show` spells it
`neuroatlas <verb>` where the script has `python -m
neuroatlas.entrypoints.<verb>`. The development test suite
(`tests/test_benchmark_catalog.py`, not shipped) fails if the two drift
apart. So a file here never invents a protocol -- it names one.

## Fields

Read by `src/neuroatlas/catalog.py`, which refuses a file that breaks these
rules.

    schema_version                  1 (not checked)
    name, title, domain, question   what it is, in words (required, with metrics.headline)
    paper                           where the paper reports it (App. C.x)
    task                            probe task preset; omitted = each dataset's own
    embed / probe                   extra arguments to `embed` / `probe`, one string
    datasets                        the full suite: a slug, or {slug, task, note, embed};
                                    a dataset's own `embed` replaces the benchmark's
    single                          the one-dataset quick suite; one of `datasets`
    planned                         cohorts the paper names that nothing here runs yet,
                                    each {name, reason}
    excluded_models                 model families the benchmark does not evaluate,
                                    each {families: [...], reason}: an alias or group
                                    leaves them out, naming one is refused (exit 2),
                                    and `show` spells `--models all,-<family>,...`
                                    for the verbs
    metrics.headline                the number `results` summarises over folds
    metrics.higher_is_better        false for errors such as MAE (default true)
    metrics.dummy                   what a trivial predictor scores; a number,
                                    {slug: number}, or omitted when it depends on
                                    the split (normalised scores are then n/a)
    metrics.secondary               reported beside the headline
    metrics.tolerance               how far from the paper's number still counts as
                                    reproduced (`results --reference`); the paper's
                                    numbers would live in ../reference/<benchmark>.csv,
                                    and no benchmark ships one yet
    variants                        other cells the paper also runs, each with a
                                    description, its own embed/probe arguments and
                                    an optional dataset subset (`default` is implicit)
    derived_from                    a benchmark computed from another's output
                                    (hypnograms), with no embed or probe of its own

Every `--set KEY=VALUE` of a dataset's embed line is copied onto its probe
line, so the probe looks up the cache the embed wrote: the cache key hashes
the dataset settings, and even `10` against `10.0` is a different key. A
probe that sets the same key to another value is refused when the catalog
loads.
