# configs/

Everything a run reads that is not code. Five kinds, and they are not
interchangeable.

## `cohorts/` — one directory per corpus, and the registry

`cohorts/<slug>/cohort.yaml` is everything factual about one corpus: what it
is, what its labels mean, how its
folds are grouped, and the runtime defaults its loader takes. It is not
documentation — `dataset_spec_from_manifest` reads it to build the
DatasetSpec, and a spec module in `src/` is three lines binding it to a
datamodule. Delete a directory here and that corpus stops existing.

`cohorts/_schema.yaml` describes the format; `cohorts/README.md` is a
review table over all of them.

Some cohorts also keep their `folds.json` here, beside their cohort.yaml.

## `folds/` — frozen train/val/test splits

35 manifests. A cohort with one here takes its folds from the file; a cohort
without derives them from a seeded splitter, which is reproducible only
until someone touches the splitter.

Two shapes, both handled by `benchmarking_helpers/fold_manifest.py`:
`explicit` gives `train`/`val`/`test` per fold, `partition` gives fold *k*'s
test group and derives `val = k+1`, `train = the rest`.

Not every file here has a dataset: `alzheimers_*` and `parkinsons_*` are
cohorts outside the paper's 42, and the thirteen `stages_*` are per-site
splits whose union is `stages.json`, the one the registered cohort uses.

**A manifest is not automatically read.** These eight cohorts name one in
their dossier that their adapter never loads, so the file records what the
splitter once produced rather than feeding the run:

    dcsm  dod  isruc  mass  physionet2026  sz1  ucddb  wsc

Check that list before assuming a file here affects a run. To clear one,
confirm the adapter's derived folds match the file, then teach the adapter to
call `load_fold_split`.

A manifest may live in either of two places. The reader tries
`cohorts/<slug>/folds.json` first and falls back to `folds/<slug>.json`, so
when a cohort has both, the copy under `cohorts/` is the one that counts.

## `channel_maps/` — per-dataset, per-model electrode mappings

48 files, each naming which of a corpus's channels feed a model and under
what name. Resolved by filename: `<slug>.yaml`, then the underscore-collapsed
spelling. **A miss returns an empty map rather than raising**, so a cohort
whose file does not match its slug silently gets no mapping at all —
the internal test suite guards exactly that.

The epilepsy maps carry a `REFERENCE ONLY` header: the runtime loader is not
wired for them, wrappers read `meta['channels']` directly, and they fail
validation against REVE's vocabulary because they name bipolar derivations.

## `tasks/` — probe presets

What `probe --task <name>` resolves. A preset names the task implementation
and may pin the probe (`C`, class weighting) and dataset defaults, so a
protocol lives in one reviewable file instead of being retyped per command.
`sleep_staging.json` is the shape to copy.

## `model_groups.yaml` — the paper's model taxonomy

The three groups of Appendix A.2, so `--models eeg_fm` means what the paper
means. Every family listed is registered and every registered family is
listed: 10 + 3 + 7 = the 20 in the checkpoint registry.

## Not tracked

`dataset_paths_sofia.yaml` and `embeddings_paths_sofia.yaml` are gitignored.
They inventory where corpora and embeddings sit on one cluster — read by no
code, useful only to whoever has that filesystem, and full of internal
paths. They stay on disk for the people who need them.
