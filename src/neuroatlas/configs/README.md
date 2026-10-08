# configs/

Everything a run reads that is not code. Six kinds, and they are not
interchangeable.

## `cohorts/`: one directory per dataset

`cohorts/<dataset>/cohort.yaml` holds every fact about one dataset: what it
is, what its labels mean, how its folds are grouped, and its reader's
defaults. The package builds the dataset from this file; delete a directory
here and neuroatlas no longer knows that dataset.

`cohorts/_schema.yaml` describes the format; `cohorts/README.md` is a table
of all 50.

Thirteen datasets also keep their `folds.json` here, beside their cohort.yaml:
chbmit, dcsm, dod, epilepsiae, isruc, mass, physionet2026,
sleep_edf_expanded, sz1, sz2, tusz, ucddb, wsc.

## `folds/`: frozen train/val/test splits

26 files here, plus the thirteen `cohorts/<dataset>/folds.json`. A dataset
with a file takes its folds from it; a dataset without one derives them with
its reader's seeded splitter, so the same data always gives the same folds.

Two shapes: `explicit` gives `train`/`val`/`test` per fold; `partition` gives
fold *k*'s test group and derives `val = k+1`, `train = the rest`.

Not every file here is read: `alzheimers_*` and `parkinsons_*` belong to
datasets neuroatlas does not list, the `shhs_*` files are SHHS visit splits,
and the thirteen `stages_*` are per-site splits whose union is `stages.json`,
the one the stages dataset reads.

For dcsm, dod, isruc, mass, physionet2026, ucddb and wsc the reader derives
the folds with its seeded splitter, and the file holds the same assignment,
frozen, fold for fold: the paper's, as stored with its embeddings.

A fold file may live in either of two places. The reader tries
`cohorts/<dataset>/folds.json` first and falls back to `folds/<dataset>.json`,
so when a dataset has both, the copy under `cohorts/` is the one that counts.

## `channel_maps/`: per-dataset, per-model electrode mappings

35 files, each naming which of a dataset's channels feed a model and under
what name. A map is found by file name: `<dataset>.yaml`, or the same name
without underscores (`sleepedf.yaml` for `sleep_edf`). `check` shows `none`
for a dataset without one: its models get the dataset's own channel labels.

A map is applied before each model sees a batch. Every model family a map
covers has an entry: a rename, `mode: label_pass_through` (the labels reach
the model unchanged and it resolves them), or `skip` (the pair is not run
and reported ruled out, with the map's note as the reason). A family with no
entry is an invalid pair, refused before any data is read. `check`'s
`channel_map` column shows which applies to each pair.

## `benchmarks/`: the catalog

One file per benchmark: what is predicted, on which datasets, how a run is
invoked and scored. `benchmarks/README.md` lists the fields.

## `tasks/`: probe presets

What `probe --task <name>` resolves (`neuroatlas list tasks`). A preset
names the task implementation and may pin the probe (`C`, class weighting)
and dataset defaults, so a protocol lives in one reviewable file instead of
being retyped per command. `sleep_staging.json` is the shape to copy.

## `model_groups.yaml`: the paper's model taxonomy

The three groups of Appendix A.2, so `--models eeg_fm` means what the paper
means. Every family with a checkpoint is in exactly one group: 10 + 3 + 7 =
20 families (`neuroatlas list models`).
