"""Every cohort whose manifest declares its own datamodule.

These used to be one module per cohort, 24 lines each, differing only in a
slug and an adapter class -- 30 files to carry two strings. A cohort that
needs nothing more than that now says so in its manifest::

    spec:
      datamodule: extensions.datasets.adapters.ucddb.UCDDBBenchmarkDataModule

and this module builds the spec for it, the same way ``pipeline.preprocessor
.module`` and ``PHYSIOEX_CLASS`` already name code from config elsewhere in
this package.

A cohort still gets its own module when the binding is not a constant --
`tusz` picks its adapter from ``backend``, `sleep_edf` wraps the callable,
and the ``moabb_*`` modules generate a spec per MOABB dataset. Those declare
no ``spec.datamodule`` and are discovered as before.

The import is deferred to first construction, so reading the registry stays
free of torch and of every adapter's dependencies.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, List

from benchmarking_helpers import DatasetSpec, dataset_spec_from_manifest
from benchmarking_helpers.registry.manifest import available_manifests, load_manifest


def _datamodule_factory(dotted: str) -> Callable[..., Any]:
    """Return a callable that imports *dotted* on first use and constructs it."""
    def _create(**config: Any) -> Any:
        module_path, _, attr = dotted.rpartition(".")
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:  # a missing optional dependency, usually
            raise ImportError(
                f"Could not import {module_path!r}, named by spec.datamodule "
                f"in its cohort manifest: {exc}"
            ) from exc
        try:
            cls = getattr(module, attr)
        except AttributeError as exc:
            raise AttributeError(
                f"{module_path!r} has no {attr!r}, named by spec.datamodule "
                f"in its cohort manifest."
            ) from exc
        return cls(**config)

    _create.__name__ = f"create_{dotted.rsplit('.', 1)[-1]}"
    _create.__qualname__ = _create.__name__
    return _create


def _build() -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    for slug in available_manifests():
        dotted = ((load_manifest(slug).get("spec") or {}).get("datamodule"))
        if not dotted:
            continue                      # has its own module, or is not a cohort
        specs.append(dataset_spec_from_manifest(slug, _datamodule_factory(dotted)))
    return specs


DATASET_SPECS = _build()
