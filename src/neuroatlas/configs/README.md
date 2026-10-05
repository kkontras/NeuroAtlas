# configs/

Everything a run reads that is not code. Six kinds, and they are not
interchangeable.

## `cohorts/` — one directory per corpus, and the registry

`cohorts/<slug>/cohort.yaml` is everything factual about one corpus: what it
is, what its labels mean, how its
folds are grouped, and the runtime defaults its loader takes. It is not
documentation — `dataset_spec_from_manifest` reads it to build the
DatasetSpec, and a spec module in `src/` is three lines binding it to a
datamodule. Delete a directory here and that corpus stops existing.

`cohorts/_schema.yaml` describes the format; `cohorts/README.md` is a
review table over all 50 of them.

Thirteen cohorts also keep their `folds.json` here, beside their cohort.yaml:
chbmit, dcsm, dod, epilepsiae, isruc, mass, physionet2026,
sleep_edf_expanded, sz1, sz2, tusz, ucddb, wsc.

## `folds/` — frozen train/val/test splits

26 manifests here, plus the thirteen `cohorts/<slug>/folds.json`. A cohort
with one takes its folds from the file; a cohort without derives them from a
seeded splitter, which is reproducible only until someone touches the
splitter.

Two shapes, both handled by `benchmarking_helpers/registry/fold_manifest.py`:
`explicit` gives `train`/`val`/`test` per fold, `partition` gives fold *k*'s
test group and derives `val = k+1`, `train = the rest`.

Not every file here has a dataset: `alzheimers_*` and `parkinsons_*` are
cohorts outside the paper's 43, the `shhs_*` files are SHHS visit splits,
and the thirteen `stages_*` are per-site splits whose union is
`stages.json`, the one the registered cohort uses.

**A manifest is not automatically read.** These seven cohorts name one in
their dossier that their adapter never loads, so the file records what the
splitter once produced rather than feeding the run:

    dcsm  dod  isruc  mass  physionet2026  ucddb  wsc

For all seven the record is the paper's: the files equal the fold
assignments stored in the paper's own embedding caches, and today's
splitters, run on the record lists those caches hold, reproduce them fold for
fold (pinned, without data, by the internal test suite). Check that
list before assuming a file here affects a run. To clear one, confirm the
adapter's derived folds match the file, then teach the adapter to call
`load_fold_split`.

A manifest may live in either of two places. The reader tries
`cohorts/<slug>/folds.json` first and falls back to `folds/<slug>.json`, so
when a cohort has both, the copy under `cohorts/` is the one that counts.

## `channel_maps/` — per-dataset, per-model electrode mappings

52 files, each naming which of a corpus's channels feed a model and under
what name. Resolved by filename: `<slug>.yaml`, then the underscore-collapsed
spelling. **A miss returns no map rather than raising**, so a cohort whose
file does not match its slug silently runs without one (`check` reports
`none`: the model gets the dataset's own labels).

The runner applies a map before each wrapper sees a batch
(`benchmarking_helpers/channels/channel_map.py`). Every model family a map
covers has an entry: a rename, `mode: label_pass_through` (the labels reach
the wrapper unchanged and it resolves them), or `skip` (the pair is not run
and reported n/a, with the map's note as the reason). A family with no entry
is an invalid pair, refused before any data is read. `check`'s
`channel_map` column shows which applies to each pair.

## `benchmarks/` — the catalog

One file per benchmark: what is predicted, on which datasets, how a run is
invoked and scored. `benchmarks/README.md` lists the fields.

## `tasks/` — probe presets

What `probe --task <name>` resolves (`neuroatlas list tasks`). A preset
names the task implementation and may pin the probe (`C`, class weighting)
and dataset defaults, so a protocol lives in one reviewable file instead of
being retyped per command. `sleep_staging.json` is the shape to copy.

## `model_groups.yaml` — the paper's model taxonomy

The three groups of Appendix A.2, so `--models eeg_fm` means what the paper
means. Every family listed is registered and every registered family is
listed: 10 + 3 + 7 = the 20 in the checkpoint registry.

## Not here

`dataset_paths_sofia.yaml` and `embeddings_paths_sofia.yaml`, inventories of
where corpora and embeddings sit on one cluster, are gitignored under the
repository's top-level `configs/`. No code reads them.
