# Benchmarks

One file per benchmark: what is predicted, on which datasets, how a run is
invoked, and how it is scored. `neuroatlas list benchmarks` prints them,
`neuroatlas show <benchmark>` explains one and `neuroatlas run <benchmark>`
executes one.

Every command a benchmark expands to is equivalent to a line of
`run/default_runs.sh`, the record of what the paper ran: `show` spells it
`neuroatlas <verb>` where the script has `python -m
neuroatlas.entrypoints.<verb>`, and the two agree line for line. So a file
here never invents a protocol: it names one.

## Fields

Read by `src/neuroatlas/catalog.py`, which refuses a file that breaks these
rules.

    schema_version                  1 (not checked)
    name, title, domain, question   what it is, in words (required, with metrics.headline)
    paper                           where the paper reports it (App. C.x)
    task                            probe task preset; omitted = each dataset's own
    embed / probe                   extra arguments to `embed` / `probe`, one string
    datasets                        all datasets (`--dataset full`): a name, or
                                    {slug, task, note, embed};
                                    a dataset's own `embed` replaces the benchmark's
    single                          the quick dataset (`--dataset single`); one of
                                    `datasets`
    planned                         datasets the paper names that this benchmark does
                                    not run, each {name, reason}
    excluded_models                 model families the benchmark does not evaluate,
                                    each {families: [...], reason}: an alias or group
                                    leaves them out, naming one is refused (exit 2),
                                    and `show` spells `--models all,-<family>,...`
                                    for the verbs
    metrics.headline                the number `results` summarises over folds: a key
                                    of the metric registry (src/neuroatlas/metrics_info.py),
                                    which holds its column label, meaning, direction
                                    and chance level
    metrics.higher_is_better        false for errors such as MAE (default true); must
                                    agree with the registry
    metrics.secondary               reported beside the headline (registry keys too)
    metrics.at                      for a task that fits several probes per fold, the
                                    one every number of a row is read from:
                                    `field@<seconds>s` (ahi_fraction@10.0s) or
                                    `<seconds>s` (3.0s); never another threshold's
    metrics.unit                    one scored item: "30 s epoch", "subject", "trial"
    metrics.describe                the headline in one line (unit, classes or
                                    threshold, pooling); the `results` title and `show`
    metrics.fold                    what a fold is (and how the probe is fitted)
    metrics.spread                  what ± is over, and which SD (population or sample);
                                    omitted for one pooled value (no ± column)
    metrics.note                    one more line under the title (BCI: Eq. 4)
    metrics.chance                  the headline's chance level when the registry's
                                    does not hold: a number, prevalence or 1/C
                                    (`dummy` is read as the same key)
    default_variant                 {description, paper}: the default variant in the
                                    paper's words, and the figure it reproduces
    variants                        other cells the paper also runs, each with a
                                    description, the paper figure it reproduces
                                    (`paper`), other names it answers to (`aliases`:
                                    --variant takes them, and results folders named
                                    after them are read as the variant's), its
                                    own embed/probe arguments and an optional dataset
                                    subset (`default` is implicit)
    derived_from                    a benchmark computed from another's output
                                    (hypnograms), with no embed or probe of its own

Every `--set KEY=VALUE` of a dataset's embed line is copied onto its probe
line, so the probe looks up the cache the embed wrote: the cache key hashes
the dataset settings, and even `10` against `10.0` is a different key. A
probe that sets the same key to another value is refused when the catalog
loads.
