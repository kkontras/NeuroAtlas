# Dataset extensions

Everything cohort-specific. Three layers plus four support packages, 43.5k
lines across 198 modules, so this map is worth reading before adding to it.

## The three layers

A cohort is normally two files that share its slug, plus its manifest:

    src/neuroatlas/configs/cohorts/<slug>/cohort.yaml   every fact, and the datamodule to build
    adapters/<slug>.py                   splits, folds, batch assembly
    dataio/<slug>.py                     files, signals, annotations

Every *fact* about a cohort -- source, montage, label modes, split
grouping -- lives in the manifest. **To change a default, edit the manifest.**
The manifest also names the datamodule::

    spec:
      datamodule: neuroatlas.extensions.datasets.adapters.ucddb.UCDDBBenchmarkDataModule

and `_manifest_specs.py` builds the dataset from it, resolving the path on
first construction. 34 datasets are bound this way and have no Python module
of their own.

A cohort keeps its own `<slug>.py` only when the binding is not a constant:
`tusz` chooses its adapter from `backend`, `sleep_edf` wraps the callable, and
the five `moabb_*` modules generate one spec per MOABB dataset.

    datasets with a manifest   50   (43 of them in the paper)
    MOABB-generated           138   (readable, not in the paper)
    total                     188

## Support packages

| package | files | lines | what it is |
|---|---|---|---|
| `dataio/` | 38 | 16,074 | low-level readers, plus private helpers (`_edf_units`, `_bids_participants`, `_eeg_channel_discovery`, `_stage_resample`) |
| `adapters/` | 60 | 12,267 | datamodules; `base.py`, `physioex_base.py`, `moabb_generic.py` and `_loader_adapters.py` are shared bases, not cohorts |
| `epilepsy/` | 21 | 6,418 | shared epilepsy preprocessing, used by 30 modules across all three layers |
| `physioex/` | 32 | 5,983 | a home-grown pipeline/steps/cache framework (**not** the PyPI `physioex`, which is a separate declared dependency) serving 4 registered cohorts: hmc, mesa, hpap_lab_full, stages |
| `preprocessors/` | 7 | 1,043 | corpus builders reached by `prepare` and by direct import |
| `pipelines/` | 1 | 107 | the pipeline registry `physioex_base` resolves against |

## One naming trap

**Backend suffixes are not a convention.** Five datasets have two readers,
under three spellings: `chbmit`/`chbmit_bids`, `epilepsiae`/
`epilepsiae_cached`, `sz1`/`sz1_edf`, `sz2`/`sz2_edf`, `tusz`/`tusz_edf`.
Siena's one reader is `siena_bids`.

`tuh` names the corpus family (the source of TUAB and TUSZ in their
manifests), not a reader: TUSZ's reader is `tusz.py`, TUAB's `tuab.py`.

## Adding a dataset

1. Write `src/neuroatlas/configs/cohorts/<slug>/cohort.yaml` -- that is where the facts go,
   including `spec.datamodule`.
2. Add `adapters/<slug>.py`, reusing a base where one fits.
3. Add `dataio/<slug>.py` only if no existing reader covers the format.

No fourth step: declaring `spec.datamodule` is what registers the cohort.
Write a `<slug>.py` only if the binding cannot be a constant.
