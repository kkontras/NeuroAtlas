"""TUSZ (TUH EEG Seizure Corpus) dataset spec for the benchmarking registry.

Every fact about this cohort — source, montage and why, label modes,
split grouping — lives in ``configs/cohorts/tusz/``.  This module only binds the
manifest to the datamodule callable, which is the one thing a YAML file cannot
hold.

TUSZ is one dataset with two loaders.  It used to be registered twice, as
``tusz`` and ``tusz_edf_direct``, so the same corpus appeared as two cohorts in
every listing.  The loader is now chosen with ``--set backend=hdf5|edf``; the
manifest's ``backends`` block says what each one reads and what it cannot do.

To change a default, edit the manifest, not this file.
"""

from __future__ import annotations

from benchmarking_helpers import dataset_spec_from_manifest, load_manifest


def _create_tusz_datamodule(**config):
    """Build the TUSZ datamodule for the requested ``backend``.

    The two loaders do not take the same arguments — the EDF one has no cache
    and no k-fold — so the keys the chosen backend cannot accept are dropped
    here rather than passed on to fail inside a constructor.  Dropping is safe
    because both classes also declare ``**kwargs``, which would otherwise
    swallow a stale ``cache_root`` in silence.
    """
    backends = load_manifest("tusz")["backends"]
    name = config.pop("backend", "hdf5")
    if name not in backends:
        raise ValueError(
            f"Unknown TUSZ backend {name!r}. Expected one of "
            f"{', '.join(sorted(backends))} — see configs/cohorts/tusz/cohort.yaml."
        )
    for key in backends[name].get("unsupported") or ():
        config.pop(key, None)
    config = {**(backends[name].get("defaults") or {}), **config}

    if name == "edf":
        from .adapters.tusz_edf import TUSZEDFDirectDataModule

        return TUSZEDFDirectDataModule(**config)

    from .adapters.tusz import TUSZDataModule

    return TUSZDataModule(**config)


DATASET_SPECS = [
    dataset_spec_from_manifest("tusz", _create_tusz_datamodule),
]
