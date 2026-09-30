from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypedDict


class BenchmarkBatch(TypedDict, total=False):
    """Normalized batch contract emitted by benchmark dataset adapters.

    Required keys:
    - ``signals``: normalized model inputs keyed by modality
    - ``label``: batch-aligned labels
    - ``meta``: one metadata dict per sample

    Optional keys:
    - ``raw_batch``: original loader payload for wrappers that need it

    Per-sample ``meta[i]`` schema (see ``MODEL_CONTRACTS.md``):

    - ``dataset``: dataset slug (str).
    - ``subject_id``: subject identifier (int | str).
    - ``epoch_index``: per-recording epoch index (int).
    - ``sampling_rate``: tensor sample rate in Hz at the point of
      collation (float). Canonical key. Legacy adapters may still emit
      ``sfreq``; wrappers read both via ``_read_sampling_rate``.
    - ``unit``: amplitude unit the tensor carries, one of
      ``"uV"`` | ``"mV"`` | ``"V"``. Wrappers convert to µV internally
      via ``unit_to_uv`` in ``_preproc.py``. Missing key → default
      ``"uV"`` with a one-shot DeprecationWarning.
    - ``channels``: list of per-channel labels in tensor order
      (list[str]). Required by wrappers that care about channel identity
      (CBraMod, EEGPT, LaBraM, REVE, NeuroLM, NeuroRVQ).
    - ``reference``: optional informational string (e.g. ``"CAR"``,
      ``"bipolar"``, ``"ear_ref"``, ``"laplacian"``). No wrapper
      currently applies or removes a re-reference — see
      ``MODEL_CONTRACTS.md`` §2.

    Batch homogeneity: all samples in a single batch must share the same
    ``sampling_rate``, ``unit``, and ``channels`` layout. Adapters
    guarantee this; wrappers validate via ``assert_batch_homogeneity``.
    """

    signals: Mapping[str, Any]
    label: Any
    meta: List[Dict[str, Any]]
    raw_batch: Any


def chunk_records(records: Sequence, chunk_idx: int, n_chunks: int) -> list:
    """Split *records* by ``subject_id`` into *n_chunks* and return chunk *chunk_idx*."""
    sorted_records = sorted(records, key=lambda r: r.subject_id)
    n = len(sorted_records)
    chunk_size = n // n_chunks
    remainder = n % n_chunks
    start = chunk_idx * chunk_size + min(chunk_idx, remainder)
    end = start + chunk_size + (1 if chunk_idx < remainder else 0)
    return sorted_records[start:end]


@dataclass
class BenchmarkDataModule:
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embed_chunk: Optional[Tuple[int, int]] = None

    def cache_context(self, purpose: str = "default") -> Dict[str, Any]:
        return dict(self.metadata)

    def supports_global_embedding_cache(self) -> bool:
        return False

    def full_embedding_dataloader(self):
        raise NotImplementedError

    def split_global_embedding_payload(self, payload: "EmbeddingPayload") -> Dict[str, "EmbeddingPayload"]:
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError


@dataclass(frozen=True)
class CheckpointSpec:
    identifier: str
    model_family: str
    variant: str
    source_type: str
    source_reference: str
    checkpoint_path: Optional[str]
    input_kind: str
    expected_channels: Sequence[str]
    expected_sampling_rate: Optional[float]
    expected_epoch_seconds: float
    expected_bandpass: Optional[Tuple[float, float]] = None
    expected_notch: bool = False
    expected_sequence_length: int = 1
    expected_montage: Optional[str] = None
    pretraining_datasets: Sequence[str] = field(default_factory=tuple)
    finetuned_datasets: Sequence[str] = field(default_factory=tuple)
    has_classifier_head: bool = False
    embedding_key: str = "embedding"
    embedding_dim: Optional[int] = None
    wrapper_name: str = ""
    status: str = "planned"
    notes: str = ""
    runtime_overrides: Dict[str, Any] = field(default_factory=dict)
    """Opaque per-run knobs consumed by the wrapper (not the runner).

    Recognised keys (all optional; defaults preserve paper-faithful
    behaviour):

    - ``apply_amplitude_scale`` (bool, default ``True``):
      if ``False``, skip fixed-formula model-specific amplitude
      transforms (EEGPT ×1000, LaBraM /100, CBraMod clip+/100,
      NeuroLM /100, NeuroRVQ ±500-clip). Scale-dependent wrappers
      log a WARNING because embeddings may be off-distribution.
    - ``apply_recording_normalization`` (bool, default ``True``):
      if ``False``, skip statistics-based normalization transforms
      (BENDR per-sequence min-max, BIOT per-channel 95-percentile,
      REVE z-score + σ-clip). Meaningful only for scale-invariant
      models.

    See ``MODEL_CONTRACTS.md``.
    """

    def evaluation_mode(self, dataset_name: str) -> str:
        if self.has_classifier_head and dataset_name.lower() in {
            dataset.lower() for dataset in self.finetuned_datasets
        }:
            return "native_head_eval"
        return "linear_probe_eval"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkFailure:
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class BenchmarkResult:
    checkpoint_id: str
    dataset_name: str
    evaluation_mode: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    cache_paths: Dict[str, str] = field(default_factory=dict)
    failure: Optional[BenchmarkFailure] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.failure is None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "checkpoint_id": self.checkpoint_id,
            "dataset_name": self.dataset_name,
            "evaluation_mode": self.evaluation_mode,
            "metrics": self.metrics,
            "cache_paths": self.cache_paths,
            "metadata": self.metadata,
            "ok": self.ok,
        }
        if self.failure is not None:
            payload["failure"] = self.failure.to_dict()
        return payload

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkResult":
        failure = None
        if "failure" in d and d["failure"]:
            f = d["failure"]
            failure = BenchmarkFailure(code=f.get("code", ""), message=f.get("message", ""), details=f.get("details", {}))
        return cls(
            checkpoint_id=d["checkpoint_id"],
            dataset_name=d["dataset_name"],
            evaluation_mode=d["evaluation_mode"],
            metrics=d.get("metrics", {}),
            cache_paths=d.get("cache_paths", {}),
            failure=failure,
            metadata=d.get("metadata", {}),
        )


@dataclass
class EmbeddingPayload:
    features: Any
    labels: Any
    metadata: List[Dict[str, Any]]


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    description: str
    datamodule_cls: Callable[..., BenchmarkDataModule]
    config_defaults: Dict[str, Any] = field(default_factory=dict)
    metadata_keys: Sequence[str] = field(default_factory=tuple)
    supports_folds: bool = False
    default_task: str = "linear_probe"
    input_kind_signal_map: Dict[str, str] = field(default_factory=dict)
    notch_freq: Optional[float] = None
    manifest: Dict[str, Any] = field(default_factory=dict)
    """The dataset's manifest (``src/neuroatlas/configs/cohorts/<slug>/cohort.yaml``), when it has one.

    The single home for per-dataset facts: cohort, acquisition, signal,
    label modes, split grouping.  Only ``runtime_defaults`` and the ``spec``
    block feed the fields above; the rest is provenance that reporting and
    ``fetch`` read, and that the generated README renders.  Empty for specs not
    yet migrated.
    """

    def build_config(self, config: Dict[str, Any], checkpoint: Optional[CheckpointSpec] = None) -> Dict[str, Any]:
        runtime = dict(self.config_defaults)
        runtime.update(config)
        if self.notch_freq is not None:
            runtime.setdefault("notch", self.notch_freq)
            runtime.setdefault("highpass", 0.5)
        if checkpoint is not None:
            signal_kind = self.input_kind_signal_map.get(checkpoint.input_kind)
            if signal_kind is not None:
                runtime.setdefault("signal_kind", signal_kind)
            runtime.setdefault("epoch_seconds", checkpoint.expected_epoch_seconds)
            # ESAT-8: sleep datasets always use 30 s epochs regardless of
            # model spec (which may be 10 s or 4 s for the pretrain window).
            from neuroatlas.extensions.models.backbones._preproc import _ESAT_DATASETS
            if self.slug in _ESAT_DATASETS:
                runtime["epoch_seconds"] = 30.0
            # Resampling is wrapper-side per MODEL_CONTRACTS.md §3.
            # target_sfreq is no longer propagated to datasets.
        return runtime

    def create_datamodule(self, config: Dict[str, Any], checkpoint: Optional[CheckpointSpec] = None) -> BenchmarkDataModule:
        runtime = self.build_config(config, checkpoint=checkpoint)
        return self.datamodule_cls(**runtime)

    def expand_runs(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        runtime = dict(config)
        folds = runtime.pop("folds", None)
        if self.supports_folds and folds is not None:
            return [{**runtime, "fold": int(fold)} for fold in folds]
        return [runtime]


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    description: str
    loader: Callable[[CheckpointSpec], Any]
    checkpoints: Sequence[CheckpointSpec] = field(default_factory=tuple)


@dataclass(frozen=True)
class TaskSpec:
    slug: str
    description: str
    evaluator: Callable[..., BenchmarkResult]

