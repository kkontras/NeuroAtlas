def ensure_dataloader_sharing_strategy() -> None:
    """Put torch's DataLoader workers on the file_system sharing strategy.

    Called when a run starts, not at import. It is a worker concern, and
    importing torch to set it made `probe --help` -- and anything else that
    only wanted to read the registry -- pay for the GPU stack. Probing is
    CPU-only by design, so that cost bought nothing. Idempotent.
    """
    try:
        import torch.multiprocessing as _mp

        _mp.set_sharing_strategy("file_system")
    except (ImportError, RuntimeError):
        pass


from .registry.contracts import (
    BenchmarkBatch,
    BenchmarkDataModule,
    BenchmarkFailure,
    BenchmarkResult,
    CheckpointSpec,
    DatasetSpec,
    EmbeddingPayload,
    ModelSpec,
    TaskSpec,
)
from .runtime.cache import build_cache_key, cache_exists, load_embedding_payload, save_embedding_payload, save_probe_payload
from .registry.discovery import (
    checkpoint_registry,
    dataset_specs,
    load_backbone,
    load_checkpoint_registry,
    load_dataset_spec,
    load_model_spec,
    load_task_spec,
    model_specs,
    task_specs,
)
from .registry.manifest import (
    ManifestError,
    available_manifests,
    dataset_spec_from_manifest,
    fold_manifest_path,
    load_manifest,
)

__all__ = [
    "BenchmarkBatch",
    "CHECKPOINT_REGISTRY",
    "BenchmarkDataModule",
    "BenchmarkFailure",
    "BenchmarkResult",
    "CheckpointSpec",
    "DatasetSpec",
    "EmbeddingPayload",
    "ModelSpec",
    "TaskSpec",
    "BenchmarkRunner",
    "checkpoint_registry",
    "dataset_specs",
    "model_specs",
    "task_specs",
    "load_backbone",
    "load_checkpoint_registry",
    "load_dataset_spec",
    "load_model_spec",
    "load_task_spec",
    "build_cache_key",
    "cache_exists",
    "load_embedding_payload",
    "save_embedding_payload",
    "save_probe_payload",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "ProbeResult",
    "train_linear_probe",
    "train_probe",
    "dataloader_worker_init_fn",
    "seed_everything",
    "ManifestError",
    "available_manifests",
    "dataset_spec_from_manifest",
    "fold_manifest_path",
    "load_manifest",
]


# `.reproducibility` imports torch -- it seeds torch's RNGs and sets the
# cudnn flags -- and importing it here made every reader of the registry pay
# for the GPU stack, including `probe --help`, on a path that is CPU-only by
# design. Resolved on first use instead (PEP 562), so the names stay
# available and the cost is paid only by a caller that wants one.
#
# The probes and the runner (sklearn, scipy: over a second) are resolved the
# same way: reading the registry -- `data status`, `list`, `--help` -- needs
# neither.
_LAZY = {
    "dataloader_worker_init_fn": "runtime.reproducibility",
    "seed_everything": "runtime.reproducibility",
    "compute_classification_metrics": "probes.metrics",
    "compute_regression_metrics": "probes.metrics",
    "ProbeResult": "probes.probe",
    "train_linear_probe": "probes.probe",
    "train_probe": "probes.probe",
    "BenchmarkRunner": "runtime.runner",
}


def __getattr__(name: str):
    if name == "CHECKPOINT_REGISTRY":
        # The built registry, not a module attribute, so it cannot go in
        # _LAZY. Came from the internal_helpers shim package, which built it
        # eagerly at import and so pulled in torch for anything that merely
        # wanted to read a spec. Built on first use here instead.
        from .registry.discovery import checkpoint_registry

        value = checkpoint_registry()
        globals()[name] = value
        return value
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value      # cache: the cost is paid at most once
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    # CHECKPOINT_REGISTRY is built rather than re-exported, so it is not
    # in _LAZY and has to be named here to stay discoverable.
    return sorted(set(globals()) | set(_LAZY) | {"CHECKPOINT_REGISTRY"})
