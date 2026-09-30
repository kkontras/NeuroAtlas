# Benchmarks

One file per benchmark: what is predicted, on which datasets, how a run is
invoked, and how it is scored. `neuroatlas list benchmarks` prints them and
`neuroatlas run <benchmark>` executes one.

Every command a benchmark expands to is a line of `run/default_runs.sh`, the
record of what the paper ran; `tests/test_benchmark_catalog.py` fails if the
two drift apart. So a file here never invents a protocol -- it names one.

## Fields

    name, title, domain, question   what it is, in words
    task                            probe task preset; omitted = each dataset's own
    embed / probe                   extra arguments to `embed` / `probe`, one string
    datasets                        the full suite: a slug, or {slug, task, note}
    single                          the one-dataset quick suite
    planned                         cohorts the paper names that nothing here runs yet,
                                    each with the reason
    metrics.headline                the number a leaderboard ranks on
    metrics.higher_is_better        false for errors such as MAE
    metrics.dummy                   what a trivial predictor scores; a number,
                                    {slug: number}, or omitted when it depends on
                                    the split (normalised scores are then n/a)
    metrics.secondary               reported beside the headline
    metrics.tolerance               how far from the paper's number still counts as
                                    reproduced (`results --reference`); the paper's
                                    numbers live in ../reference/<benchmark>.csv
    variants                        other cells the paper also runs, each with its
                                    own embed/probe arguments and optional dataset subset
    derived_from                    a benchmark computed from another's output
                                    (hypnograms), with no embed or probe of its own
