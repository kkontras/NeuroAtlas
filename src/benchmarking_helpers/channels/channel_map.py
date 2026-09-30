"""Dataset × model channel-map layer (opt-in per dataset).

YAML schema (see ``MODEL_CONTRACTS.md`` §1 "Per-dataset channel
configuration"):

    dataset: <slug>
    channels_available: [...]           # physical channels in the dataset
    channels_used:      [...]           # subset the benchmark selects
    per_model:
      <model_family>:
        <source_label>: <target_label>
        ...
      <other_model_family>: skip        # explicitly unsupported pair
    notes:
      <model_family>: "justification string"
      ...

Rules enforced at load time:

- Every ``(dataset, model)`` that the benchmark will actually run must
  have a ``per_model[<model>]`` entry — either a dict mapping every
  label in ``channels_used`` onto a valid target in the model's
  vocabulary, or the string ``"skip"``.
- Any rename where source != target must have a matching
  ``notes[<model>]`` string, forcing the author to justify
  anatomical approximations in writing.
- Target labels must exist in the model's vocabulary (see
  ``channel_vocabularies.py``). Unknown targets raise with a suggestion
  (via ``difflib``).

Runtime behaviour:

- The loader returns ``None`` if the yaml file does not exist — the
  layer is opt-in and the runner falls back to wrapper-side alias maps.
- When a file exists, the runner wraps every dataloader so that
  ``meta[i]["channels"]`` is rewritten with resolved labels *before*
  the wrapper sees it.
- ``per_model[<model>] == "skip"`` → runner aborts that ``(dataset,
  model)`` combination with a clear failure; no wrapper load, no
  embedding.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .channel_vocabularies import accepts_label, suggest_close_matches


_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs" / "channel_maps"


@dataclass(frozen=True)
class ChannelMap:
    """Parsed, validated channel-map config for one dataset."""

    dataset: str
    channels_available: List[str]
    channels_used: List[str]
    per_model: Dict[str, Union[Dict[str, str], str]]
    notes: Dict[str, str]

    def is_skip(self, model_family: str) -> bool:
        entry = self.per_model.get(model_family)
        return isinstance(entry, str) and entry.strip().lower() == "skip"

    def has_entry(self, model_family: str) -> bool:
        return model_family in self.per_model

    def mapping_for(self, model_family: str) -> Dict[str, str]:
        entry = self.per_model.get(model_family)
        if not isinstance(entry, dict):
            raise ValueError(
                f"channel_map[{self.dataset!r}]: no dict mapping for model "
                f"{model_family!r} (found {entry!r})."
            )
        return dict(entry)


def _configs_dir() -> Path:
    return _CONFIGS_DIR


def _default_yaml_path(dataset: str) -> Path:
    # Normalize slug: files are named with underscores collapsed (e.g.
    # "sleep_edf" → "sleepedf.yaml") OR with the exact slug, supporting
    # both forms.
    candidates = [
        _configs_dir() / f"{dataset}.yaml",
        _configs_dir() / f"{dataset.replace('_', '')}.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_channel_map(
    dataset: str, *, base_dir: Optional[Path] = None
) -> Optional[ChannelMap]:
    """Read, validate, and return the channel map for ``dataset``.

    Returns ``None`` if the yaml file does not exist (layer is opt-in).
    Raises ``ValueError`` on malformed config, missing ``notes[<model>]``
    for a rename, or target label outside the destination model's
    vocabulary.
    """
    if base_dir is None:
        path = _default_yaml_path(dataset)
    else:
        path = Path(base_dir) / f"{dataset}.yaml"
    if not path.exists():
        return None

    import yaml  # local import — yaml is a soft dep at the channel-map layer

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"channel map {path}: top-level must be a mapping")

    declared_dataset = data.get("dataset")
    if declared_dataset is None:
        raise ValueError(f"channel map {path}: missing 'dataset' field")
    channels_available = list(data.get("channels_available") or [])
    channels_used = list(data.get("channels_used") or [])
    per_model_raw = data.get("per_model") or {}
    notes = dict(data.get("notes") or {})

    if not isinstance(per_model_raw, Mapping):
        raise ValueError(f"channel map {path}: 'per_model' must be a mapping")
    per_model: Dict[str, Union[Dict[str, str], str]] = {}
    for model, entry in per_model_raw.items():
        if isinstance(entry, str):
            if entry.strip().lower() != "skip":
                raise ValueError(
                    f"channel map {path}: per_model[{model!r}]={entry!r} is a "
                    f"string but not the sentinel 'skip'; use a dict or 'skip'."
                )
            per_model[model] = "skip"
            continue
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"channel map {path}: per_model[{model!r}] must be a dict or "
                f"'skip', got {type(entry).__name__}"
            )

        # Validate completeness.
        missing = [label for label in channels_used if label not in entry]
        if missing:
            raise ValueError(
                f"channel map {path}: per_model[{model!r}] is missing "
                f"entries for {missing!r} (must cover every label in "
                f"channels_used)."
            )
        extras = [label for label in entry if label not in channels_used]
        if extras:
            raise ValueError(
                f"channel map {path}: per_model[{model!r}] has entries "
                f"{extras!r} not in channels_used."
            )

        # Validate targets against the model's vocabulary + force notes[<model>]
        # on any non-trivial rename.
        has_rename = any(src != dst for src, dst in entry.items())
        if has_rename and model not in notes:
            raise ValueError(
                f"channel map {path}: per_model[{model!r}] contains "
                f"rename(s) but notes[{model!r}] is missing — every "
                f"rename must have a written justification."
            )
        for src, dst in entry.items():
            if not isinstance(dst, str) or not dst:
                raise ValueError(
                    f"channel map {path}: per_model[{model!r}][{src!r}] must "
                    f"be a non-empty string, got {dst!r}"
                )
            if not accepts_label(model, dst):
                suggestions = suggest_close_matches(model, dst)
                hint = (
                    f" did you mean {suggestions!r}?" if suggestions else ""
                )
                raise ValueError(
                    f"channel map {path}: target {dst!r} not in {model!r}'s "
                    f"vocabulary.{hint}"
                )
        per_model[model] = {str(k): str(v) for k, v in entry.items()}

    return ChannelMap(
        dataset=str(declared_dataset),
        channels_available=channels_available,
        channels_used=channels_used,
        per_model=per_model,
        notes=notes,
    )


class ChannelMapSkip(RuntimeError):
    """Raised when a ``(dataset, model)`` combination is marked ``skip``."""


def resolve_labels(
    cmap: ChannelMap, model_family: str, labels: Sequence[str]
) -> List[str]:
    """Apply ``cmap``'s per-model rename to ``labels``.

    Raises:
        ChannelMapSkip: if ``cmap.per_model[model_family] == "skip"``.
        ValueError: if ``model_family`` has no entry, or if any label in
            ``labels`` is not covered by the mapping.
    """
    if cmap.is_skip(model_family):
        raise ChannelMapSkip(
            f"({cmap.dataset!r}, {model_family!r}) explicitly skipped by "
            f"dataset config; remove from benchmark matrix or remove the "
            f"skip marker."
        )
    if not cmap.has_entry(model_family):
        raise ValueError(
            f"channel map {cmap.dataset!r}: no mapping for model "
            f"{model_family!r}."
        )
    mapping = cmap.mapping_for(model_family)
    resolved: List[str] = []
    for label in labels:
        if label not in mapping:
            raise ValueError(
                f"channel map {cmap.dataset!r}: label {label!r} has no "
                f"per-model target for {model_family!r} (available: "
                f"{sorted(mapping)!r})."
            )
        resolved.append(mapping[label])
    return resolved


def rewrite_meta_channels(
    meta: Sequence[Mapping[str, Any]],
    cmap: ChannelMap,
    model_family: str,
) -> List[Dict[str, Any]]:
    """Return a new meta list with ``channels`` rewritten via resolve_labels.

    Preserves other keys untouched. Used by the runner to wrap dataloader
    batches before the wrapper sees them.
    """
    if not meta:
        return list(meta)
    out: List[Dict[str, Any]] = []
    for entry in meta:
        new_entry = dict(entry)
        labels = new_entry.get("channels")
        if labels is not None:
            new_entry["channels"] = resolve_labels(cmap, model_family, labels)
        out.append(new_entry)
    return out
