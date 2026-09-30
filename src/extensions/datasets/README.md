# Dataset extensions

Everything cohort-specific. Three layers plus four support packages, 43.5k
lines across 198 modules, so this map is worth reading before adding to it.

## The three layers

A cohort is normally two files that share its slug, plus its manifest:

    configs/cohorts/<slug>/cohort.yaml   every fact, and the datamodule to build
    adapters/<slug>.py                   splits, folds, batch assembly
    dataio/<slug>.py                     files, signals, annotations

Every *fact* about a cohort -- source, montage, label modes, split
grouping -- lives in the manifest. **To change a default, edit the manifest.**
The manifest also names the datamodule::

    spec:
      datamodule: extensions.datasets.adapters.ucddb.UCDDBBenchmarkDataModule

and `_manifest_specs.py` builds the registry spec from it, resolving the path
on first construction. 30 cohorts are bound this way and have no Python
module of their own.

A cohort keeps its own `<slug>.py` only when the binding is not a constant:
`tusz` chooses its adapter from `backend`, `sleep_edf` wraps the callable, and
the five `moabb_*` modules generate one spec per MOABB dataset.

    registered cohorts     46   (one dossier each, all with a spec)
    MOABB-generated specs 138   (loadable, not in the paper)
    total discovered      184

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

**Backend suffixes are not a convention.** Six cohorts have two readers, under
three different spellings: `chbmit`/`chbmit_bids`, `epilepsiae`/
`epilepsiae_cached`, `siena_bids` (the plain `dataio/siena.py` was deleted),
`sz1`/`sz1_edf`, `sz2`/`sz2_edf`, `tusz`/`tusz_edf`.

TUSZ's cache-backed reader used to be called `tuh`, which was misleading --
TUH is the corpus family and TUAB is also TUH, with its own `tuab.py`. It is
now `tusz`, matching its slug. `tuh` still appears as a *source* name (the
fetch kind, and `source: tuh` in the TUAB and TUSZ manifests), which is
correct: that is the corpus, not the reader.

## What is here but cannot run

Checked by import-reachability and by asking the registry directly; these are
kept deliberately, not overlooked.

- **21 adapters with no dossier and no spec** (266 lines): `stages_*` (13
  per-site), `shhs_v1`, `shhs_v2`, `alzheimers_ad/hc`, `parkinsons_*` (4).
  Each has a fold manifest under `configs/folds/` but no
  `configs/cohorts/<slug>/`, so `load_dataset_spec` raises *"Unsupported
  benchmark dataset"*. They are per-site and per-arm variants from earlier
  experiments.
- **`adapters/precomputed_embeddings.py`** (767 lines): no spec binds it, and
  probing from cached embeddings goes through `require_cached_embeddings` and
  the runtime cache instead. It is kept because `tasks/brain_age.py` still
  computes brain-age gap from `split_payloads.get("holdout")`, and this is the
  only datamodule implementing `holdout_eval_groups`.

## Adding a cohort

1. Write `configs/cohorts/<slug>/cohort.yaml` -- that is where the facts go,
   including `spec.datamodule`.
2. Add `adapters/<slug>.py`, reusing a base where one fits.
3. Add `dataio/<slug>.py` only if no existing reader covers the format.

No fourth step: declaring `spec.datamodule` is what registers the cohort.
Write a `<slug>.py` only if the binding cannot be a constant.
