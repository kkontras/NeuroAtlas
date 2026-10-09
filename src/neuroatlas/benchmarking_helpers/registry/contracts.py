from __future__ import annotations

import inspect
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypedDict

logger = logging.getLogger(__name__)


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

    # The runner offers every datamodule the same runtime keys -- signal_kind
    # and epoch_seconds from the checkpoint, notch/highpass from the spec's
    # notch_freq, compute_recording_stats for the models that normalise per
    # recording, channel_specs from a channel map -- and not each of them
    # means something for every dataset.  A datamodule that does not take one
    # as a constructor argument says so here, and `construct_datamodule`
    # holds it to that: a key declared below is handled as declared, and any
    # other key the constructor does not take is an error that names it.
    # Nothing is dropped without a recorded reason.

    #: key -> the one value this datamodule serves. Any other value is an
    #: error; the matching value is consumed (the constructor never sees it).
    RUNTIME_KEYS_FIXED: ClassVar[Mapping[str, Any]] = {}
    #: key -> why it does not apply here. Dropped, logged once at debug level.
    RUNTIME_KEYS_IGNORED: ClassVar[Mapping[str, str]] = {}
    #: True when the dataset, not the model, fixes the window length -- a BCI
    #: trial. ``metadata["epoch_seconds"]`` is then the length of the windows
    #: the loaders yield, and the runner tells the backbone that length
    #: (``expected_epoch_seconds``) instead of the model's pretraining window:
    #: the BCI counterpart of ``--expected-epoch-seconds 10`` on epilepsy.
    FIXED_WINDOW: ClassVar[bool] = False

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
            # _PAPER_PREPROC_DATASETS (sleep) always use 30 s epochs, whatever the
            # checkpoint's own window (10 s or 4 s for some pretraining windows).
            from neuroatlas.extensions.models.backbones._preproc import _PAPER_PREPROC_DATASETS
            if self.slug in _PAPER_PREPROC_DATASETS:
                runtime["epoch_seconds"] = 30.0
            # Resampling is wrapper-side per MODEL_CONTRACTS.md §3.
            # target_sfreq is no longer propagated to datasets.
        return runtime

    def create_datamodule(self, config: Dict[str, Any], checkpoint: Optional[CheckpointSpec] = None) -> BenchmarkDataModule:
        runtime = self.build_config(config, checkpoint=checkpoint)
        if isinstance(self.datamodule_cls, type):
            return construct_datamodule(self.datamodule_cls, runtime, dataset=self.slug)
        # A factory picks the class (lazily) and hands it to construct_datamodule.
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
    #: The task's metrics from one fold's saved test predictions
    #: (``neuroatlas.predictions``): the evaluator records what this returns
    #: on the file it writes, and ``neuroatlas rescore`` calls it again.
    score: Optional[Callable[..., Dict[str, Any]]] = None


class DatamoduleConfigError(TypeError):
    """A runtime key the dataset's datamodule neither takes nor declares.

    A ``TypeError`` because that is what the bare constructor call raised
    before, so callers that caught one keep working.
    """


#: The keys the runner itself adds to a dataset config (`DatasetSpec.build_config`
#: and `BenchmarkRunner._pair_config`); named in the error so a reader can tell
#: a key they typed from one the engine offered.
RUNNER_OFFERED_KEYS = (
    "signal_kind", "epoch_seconds", "notch", "highpass",
    "compute_recording_stats", "channel_specs",
)

_LOGGED_DROPS: set = set()


def construct_datamodule(cls: type, config: Mapping[str, Any], *, dataset: str) -> Any:
    """Construct *cls* from *config*, holding *config* to the class's contract.

    Every key in *config* must be one of:

    * a parameter of ``cls.__init__`` (or ``cls`` takes ``**kwargs``);
    * in ``cls.RUNTIME_KEYS_FIXED``, with exactly the value declared there;
    * in ``cls.RUNTIME_KEYS_IGNORED``, which records why it does not apply.

    A fixed key with another value raises ``ValueError``.  Any other key
    raises :class:`DatamoduleConfigError` naming it -- so a typo in ``--set``
    and a key the runner offers that this datamodule was never taught about
    fail the same clear way, instead of as ``__init__() got an unexpected
    keyword argument`` from deep inside a run.
    """
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):  # nothing to introspect: pass through
        return cls(**config)
    variadic = (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    named = {name for name, p in params.items() if p.kind not in variadic}
    takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    fixed = getattr(cls, "RUNTIME_KEYS_FIXED", None) or {}
    ignored = getattr(cls, "RUNTIME_KEYS_IGNORED", None) or {}

    kwargs: Dict[str, Any] = {}
    unknown: List[str] = []
    for key, value in config.items():
        if key in named:
            kwargs[key] = value
        elif key in fixed:
            if value != fixed[key]:
                raise ValueError(
                    f"{dataset}: {key}={value!r} is not available; "
                    f"{dataset} serves only {key}={fixed[key]!r}"
                )
        elif key in ignored:
            if (cls, key) not in _LOGGED_DROPS:
                _LOGGED_DROPS.add((cls, key))
                logger.debug("%s: %s does not take %s (dropped): %s",
                             dataset, cls.__name__, key, ignored[key])
        elif takes_kwargs:
            kwargs[key] = value
        else:
            unknown.append(key)
    if unknown:
        # the keys a user sets (the runner's own are set for each checkpoint)
        takes = sorted(k for k in named if k not in RUNNER_OFFERED_KEYS and k != "self")
        raise DatamoduleConfigError(unknown_settings_text(dataset, unknown, takes))
    return cls(**kwargs)


def unknown_settings_text(dataset: str, unknown: Sequence[str], takes: Sequence[str]) -> str:
    """``siena takes no setting 'windw_s'; its settings: ...``, and the
    command to run instead: the user's own with the closest setting, else
    the dataset's help (which lists them)."""
    import difflib

    names = ", ".join(repr(k) for k in unknown)
    text = (f"{dataset} takes no setting{'s' if len(unknown) > 1 else ''} {names}; its "
            f"settings: {', '.join(takes) or 'none'}")
    fix = None
    close = difflib.get_close_matches(str(unknown[0]), list(takes), n=1)
    if close and len(unknown) == 1:
        fix = _command_with_setting_renamed(str(unknown[0]), close[0])
    if fix is None:
        fix = f"neuroatlas embed --dataset {dataset} --help (its settings)"
    return f"{text}\nfix: {fix}"


def _command_with_setting_renamed(old: str, new: str) -> Optional[str]:
    """The command being run with ``--set old=V`` written ``--set new=V``, or
    None when no command line is known or it has no such --set."""
    try:
        from neuroatlas import cli
    except Exception:       # pragma: no cover - the CLI is part of the package
        return None
    line = list(getattr(cli, "_COMMAND_LINE", None) or [])
    for i, token in enumerate(line):
        for head in (f"{old}=", f"--set={old}="):
            if token.startswith(head) and (head.startswith("--set=") or
                                           (i and line[i - 1] == "--set")):
                line[i] = token.replace(f"{old}=", f"{new}=", 1)
                import shlex

                return "neuroatlas " + " ".join(shlex.quote(t) for t in line)
    return None

