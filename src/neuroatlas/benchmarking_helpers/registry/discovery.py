from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .contracts import CheckpointSpec, DatasetSpec, ModelSpec, TaskSpec


def _collect_specs(package_name: str, attr_name: str):
    package = importlib.import_module(package_name)
    specs = []
    for module_info in pkgutil.iter_modules(package.__path__, prefix=f"{package_name}."):
        module = importlib.import_module(module_info.name)
        values = getattr(module, attr_name, None)
        if values is None:
            continue
        if isinstance(values, (list, tuple)):
            specs.extend(values)
        else:
            specs.append(values)
    return specs


@lru_cache(maxsize=1)
def dataset_specs() -> List[DatasetSpec]:
    return sorted(_collect_specs("neuroatlas.extensions.datasets", "DATASET_SPECS"), key=lambda spec: spec.slug)


@lru_cache(maxsize=1)
def model_specs() -> List[ModelSpec]:
    return sorted(_collect_specs("neuroatlas.extensions.models", "MODEL_SPECS"), key=lambda spec: spec.slug)


@lru_cache(maxsize=1)
def task_specs() -> List[TaskSpec]:
    return sorted(_collect_specs("neuroatlas.extensions.tasks", "TASK_SPECS"), key=lambda spec: spec.slug)


# Slugs that were registered once and are not any more, each with the exact
# command that replaces it.  Without this a retired slug fails as a bare
# "unsupported dataset", which reads like a typo rather than a rename.
RETIRED_SLUGS = {
    "tusz_edf_direct": (
        "tusz_edf_direct was never a separate cohort — it is TUSZ v2.0.3 read "
        "from raw EDF instead of the HDF5 cache. Use "
        "`--dataset tusz --set backend=edf`, or in a config file, the key "
        '"tusz" with "backend": "edf".'
    ),
}


def load_dataset_spec(dataset_name: str) -> DatasetSpec:
    key = dataset_name.lower()
    for spec in dataset_specs():
        if spec.slug.lower() == key:
            return spec
    if key in RETIRED_SLUGS:
        raise KeyError(RETIRED_SLUGS[key])
    raise KeyError(f"Unsupported benchmark dataset {dataset_name!r}.")


def load_model_spec(model_name: str) -> ModelSpec:
    key = model_name.lower()
    for spec in model_specs():
        if spec.slug.lower() == key:
            return spec
    raise KeyError(f"Unsupported benchmark model family {model_name!r}.")


def load_task_spec(task_name: str) -> TaskSpec:
    key = task_name.lower()
    for spec in task_specs():
        if spec.slug.lower() == key:
            return spec
    raise KeyError(f"Unsupported benchmark task {task_name!r}.")


def _matches_filter(spec: CheckpointSpec, model_names: Optional[Iterable[str]]) -> bool:
    if not model_names:
        return True
    allowed = {name.lower() for name in model_names}
    return spec.model_family.lower() in allowed or spec.identifier.lower() in allowed


def _localise(value: Optional[str]) -> Optional[str]:
    """Anchor a checkpoint path written relative to the checkout.

    Registry entries say ``artifacts/models/...`` or ``src/neuroatlas/...``,
    which only resolved when the process happened to start in the checkout.
    Weights go under the models root (``$NEUROATLAS_MODELS_ROOT`` or
    ``<workspace>/artifacts/models``); vendored files under the installed
    package. Hub ids (``amazon/chronos-t5-base``) and absolute paths are left
    as they are.
    """
    from neuroatlas import _paths

    if not value or Path(value).is_absolute():
        return value
    if value.startswith("artifacts/models/"):
        return str(_paths.models_dir(value[len("artifacts/models/"):]))
    if value.startswith("src/neuroatlas/"):
        return str(_paths.PACKAGE_ROOT / value[len("src/neuroatlas/"):])
    return value


def checkpoint_registry() -> List[CheckpointSpec]:
    import dataclasses

    specs: List[CheckpointSpec] = []
    for model_spec in model_specs():
        for spec in model_spec.checkpoints:
            specs.append(dataclasses.replace(
                spec,
                checkpoint_path=_localise(spec.checkpoint_path),
                source_reference=_localise(spec.source_reference)))
    return sorted(specs, key=lambda spec: (spec.model_family, spec.identifier))


def load_checkpoint_registry(model_names: Optional[Sequence[str]] = None) -> List[CheckpointSpec]:
    return [spec for spec in checkpoint_registry() if _matches_filter(spec, model_names)]


def load_backbone(spec: CheckpointSpec):
    model_spec = load_model_spec(spec.model_family)
    return model_spec.loader(spec)


def evaluate_task(task_name: str, **kwargs):
    """Run a registered task's evaluator.

    Was its own nine-line module, `tasks.py`, which did nothing the registry
    could not.
    """
    task = load_task_spec(task_name)
    return task.evaluator(**kwargs)
