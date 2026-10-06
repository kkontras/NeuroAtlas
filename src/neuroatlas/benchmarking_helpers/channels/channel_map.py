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
      <third_model_family>:
        mode: label_pass_through        # labels unchanged; the wrapper resolves them
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
  (via ``difflib``). A ``label_pass_through`` entry writes no targets, so
  it is not checked against that vocabulary: the wrapper's own resolver
  decides, and ``neuroatlas check`` exercises it with a real batch.
- A model family with no entry at all is an *invalid* pair
  (``ChannelMap.state_for``), reported before any data is read -- not a
  failure halfway through a run.

``pair_state`` gives the one word each (dataset, model) pair is reported
with -- ``applied`` / ``none`` / ``skip`` / ``invalid``, defined in
``CHANNEL_MAP_STATES``.

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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Union

from .channel_vocabularies import accepts_label, suggest_close_matches
from neuroatlas._paths import configs_dir


_CONFIGS_DIR = configs_dir("channel_maps")


@dataclass(frozen=True)
class ChannelMap:
    """Parsed, validated channel-map config for one dataset."""

    dataset: str
    channels_available: List[str]
    channels_used: List[str]
    per_model: Dict[str, Union[Dict[str, str], str]]
    notes: Dict[str, str]
    pass_through: FrozenSet[str] = frozenset()   # families declared label_pass_through
    # Per-model montages (optional). A dataset that can serve more than one
    # montage names them (`montages: {unipolar: [...], bipolar: [...]}`), the
    # one a family gets unless it says otherwise (`montage:`), and the families
    # that need another (`model_montage: {biot: bipolar}`): each model gets the
    # derivation it was pretrained on. Without `montages`, every family gets
    # channels_used, as before.
    montage: Optional[str] = None
    montages: Dict[str, List[str]] = field(default_factory=dict)
    model_montage: Dict[str, str] = field(default_factory=dict)

    def montage_for(self, model_family: str) -> Optional[str]:
        """The montage this family is given, or None when the map names none."""
        if not self.montages:
            return None
        return self.model_montage.get(model_family, self.montage)

    def channels_for(self, model_family: str) -> List[str]:
        """The channel labels this family receives, in order."""
        montage = self.montage_for(model_family)
        if montage is None:
            return list(self.channels_used)
        return list(self.montages[montage])

    def is_skip(self, model_family: str) -> bool:
        entry = self.per_model.get(model_family)
        return isinstance(entry, str) and entry.strip().lower() == "skip"

    def has_entry(self, model_family: str) -> bool:
        return model_family in self.per_model

    def state_for(self, model_family: str):
        """``(state, detail)``: ``applied`` / ``skip`` / ``invalid`` (no entry);
        see ``CHANNEL_MAP_STATES``."""
        if self.is_skip(model_family):
            return "skip", self.notes.get(model_family, "")
        if not self.has_entry(model_family):
            # what is wrong, then (its own line) the fix: neuroatlas.cli._msg
            return "invalid", (
                f"the {self.dataset} channel map has no entry for the {model_family} family\n"
                f"fix: add one to configs/channel_maps/{self.dataset}.yaml: a mapping, "
                f"`mode: label_pass_through` or `skip`")
        if model_family in self.pass_through:
            return "applied", "pass-through"
        return "applied", ""

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

    montages = {str(k): [str(c) for c in (v or [])] for k, v in (data.get("montages") or {}).items()}
    default_montage = data.get("montage")
    model_montage = {str(k): str(v) for k, v in (data.get("model_montage") or {}).items()}
    if montages:
        if default_montage not in montages:
            raise ValueError(
                f"channel map {path}: 'montage' must name one of montages "
                f"{sorted(montages)!r}, got {default_montage!r}")
        bad = {m: v for m, v in model_montage.items() if v not in montages}
        if bad:
            raise ValueError(f"channel map {path}: model_montage names unknown montages {bad!r}; "
                             f"declared: {sorted(montages)!r}")
        undocumented = [m for m in model_montage if m not in notes]
        if undocumented:
            raise ValueError(f"channel map {path}: model_montage for {undocumented!r} has no "
                             f"notes[<model>] saying why")
        channels_used = list(montages[default_montage])
    elif model_montage or default_montage:
        raise ValueError(f"channel map {path}: 'montage'/'model_montage' need 'montages'")

    def _labels_for(model: str) -> List[str]:
        if not montages:
            return channels_used
        return montages[model_montage.get(model, default_montage)]

    if not isinstance(per_model_raw, Mapping):
        raise ValueError(f"channel map {path}: 'per_model' must be a mapping")
    per_model: Dict[str, Union[Dict[str, str], str]] = {}
    pass_through_models: set = set()
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
        pass_through = "mode" in entry
        if pass_through:
            # `mode: label_pass_through` -- the wrapper takes the used channels
            # under the dataset's own names and resolves them itself (CBraMod is
            # index-based; LaBraM, EEGPT, NeuroLM, ... carry their own alias and
            # bipolar tables). The map is the identity over channels_used, and
            # the targets are not checked against the config-time vocabulary
            # below: no target was written here, so there is no typo to catch,
            # and the wrapper's resolver -- not that conservative subset -- is
            # what decides. `neuroatlas check` runs a real batch through it.
            # non_eeg_drop_prefixes documents what the reader already left
            # out of channels_used.
            mode = entry.get("mode")
            if mode != "label_pass_through":
                raise ValueError(
                    f"channel map {path}: per_model[{model!r}] has mode {mode!r}; "
                    f"the only mode is 'label_pass_through'."
                )
            unknown = set(entry) - {"mode", "non_eeg_drop_prefixes"}
            if unknown:
                raise ValueError(
                    f"channel map {path}: per_model[{model!r}] mixes mode with "
                    f"entries {sorted(unknown)!r}; use one or the other."
                )
            entry = {label: label for label in _labels_for(model)}

        # Validate completeness, against the montage this family is given.
        labels = _labels_for(model)
        missing = [label for label in labels if label not in entry]
        if missing:
            raise ValueError(
                f"channel map {path}: per_model[{model!r}] is missing "
                f"entries for {missing!r} (must cover every label of its "
                f"montage)."
            )
        extras = [label for label in entry if label not in labels]
        if extras:
            raise ValueError(
                f"channel map {path}: per_model[{model!r}] has entries "
                f"{extras!r} not in its montage's channels."
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
            if not pass_through and not accepts_label(model, dst):
                suggestions = suggest_close_matches(model, dst)
                hint = (
                    f" did you mean {suggestions!r}?" if suggestions else ""
                )
                raise ValueError(
                    f"channel map {path}: target {dst!r} not in {model!r}'s "
                    f"vocabulary.{hint}"
                )
        per_model[model] = {str(k): str(v) for k, v in entry.items()}
        if pass_through:
            pass_through_models.add(str(model))

    return ChannelMap(
        dataset=str(declared_dataset),
        channels_available=channels_available,
        channels_used=channels_used,
        per_model=per_model,
        notes=notes,
        pass_through=frozenset(pass_through_models),
        montage=str(default_montage) if montages else None,
        montages=montages,
        model_montage=model_montage,
    )


# What the `channel_map` column of `neuroatlas check` says
# about one (dataset, model family) pair. The four values, defined once:
CHANNEL_MAP_STATES = {
    "applied": "the dataset has a channel map with an entry for this model: its "
               "labels are renamed (or passed through unchanged, `pass-through`) "
               "before the model sees a batch",
    "none": "the dataset has no channel map: the model receives the dataset's own "
            "channel labels and resolves them itself",
    "skip": "the map marks this model `skip`: the pair is not run and reported "
            "n/a, with the map's note as the reason",
    "invalid": "the map fails validation, or has no entry for this model: the pair "
               "cannot run until the map is fixed",
}


def pair_state(dataset: str, model_family: str, *, base_dir: Optional[Path] = None):
    """``(state, detail)`` for one pair; *state* is a key of ``CHANNEL_MAP_STATES``.

    Never raises: a map that fails to load is ``invalid`` with the reason, so a
    caller can report it against the pair instead of stopping. Callers decide
    before any data is read.
    """
    try:
        cmap = load_channel_map(dataset, base_dir=base_dir)
    except (ValueError, KeyError) as exc:
        return "invalid", str(exc)
    if cmap is None:
        return "none", ""
    return cmap.state_for(model_family)


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
