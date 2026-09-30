"""Turn what a user types after ``-m`` / ``--models`` into checkpoint ids.

    all_fm                  an alias from model_groups.yaml
    eeg_fm                  a group name (the paper's taxonomy)
    reve                    a model family: every ready checkpoint of it
    reve_pretrained         one checkpoint
    all_fm,chronos_t5_base  any mix, comma-separated

Only *ready* checkpoints are selected by an alias, a group or a family;
naming a planned checkpoint by id is an error that says it is planned.
Duplicates collapse, order is kept, and an unknown name fails with the
closest matches before any work starts.
"""
from __future__ import annotations

import difflib
from functools import lru_cache
from typing import Any, Dict, List, Sequence

from neuroatlas import _paths


class SelectionError(ValueError):
    pass


@lru_cache(maxsize=1)
def model_groups() -> Dict[str, Any]:
    import yaml

    return yaml.safe_load(_paths.configs_dir("model_groups.yaml").read_text())


def _registry():
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    return checkpoint_registry()


def group_of(family: str) -> str:
    for name, group in model_groups()["groups"].items():
        if family in group["families"]:
            return name
    return "-"


def is_baseline(spec) -> bool:
    return spec.source_type == "random_init"


def alias_members(alias: str, specs=None) -> List[str]:
    """Checkpoint ids an alias selects."""
    specs = _registry() if specs is None else specs
    rule = model_groups()["aliases"][alias]
    ready = [s for s in specs if s.status == "ready"]
    if "group" in rule:
        families = set(model_groups()["groups"][rule["group"]]["families"])
        return [s.identifier for s in ready if s.model_family in families and not is_baseline(s)]
    if "source_type" in rule:
        return [s.identifier for s in ready if s.source_type == rule["source_type"]]
    return [s.identifier for s in ready]


def expand_models(expr: str | Sequence[str]) -> List[str]:
    """What ``--models`` hands the runner: aliases and groups become checkpoint
    ids; a family or an id stays exactly as typed, after checking it exists.

    The runner matches a family itself, and per-model overrides
    (``--checkpoint``, ``--expected-epoch-seconds``) are keyed by the name the
    user typed, so keeping it keeps those working unchanged.
    """
    names = [n.strip() for n in (expr.split(",") if isinstance(expr, str) else expr) if n.strip()]
    groups, aliases = model_groups()["groups"], model_groups()["aliases"]
    out: List[str] = []
    for name in names:
        picked = resolve_models([name])          # validates; raises on unknown / planned
        if name.lower() not in aliases and name.lower() not in groups:
            picked = [name]
        out.extend(p for p in picked if p not in out)
    return out


def resolve_models(expr: str | Sequence[str]) -> List[str]:
    """Checkpoint ids for a selector expression. Raises SelectionError."""
    names = [n.strip() for n in (expr.split(",") if isinstance(expr, str) else expr) if n.strip()]
    if not names:
        raise SelectionError("no models selected")
    specs = _registry()
    by_id = {s.identifier: s for s in specs}
    families: Dict[str, List[Any]] = {}
    for s in specs:
        families.setdefault(s.model_family, []).append(s)
    aliases = model_groups()["aliases"]
    groups = model_groups()["groups"]

    out: List[str] = []
    for name in names:
        key = name.lower()
        if key in aliases:
            picked = alias_members(key, specs)
        elif key in groups:
            fams = set(groups[key]["families"])
            picked = [s.identifier for s in specs
                      if s.status == "ready" and s.model_family in fams and not is_baseline(s)]
        elif key in families:
            picked = [s.identifier for s in families[key] if s.status == "ready"]
            if not picked:
                raise SelectionError(f"{name}: no ready checkpoint (all are "
                                     f"{', '.join(sorted({s.status for s in families[key]}))})")
        elif key in by_id:
            spec = by_id[key]
            if spec.status != "ready":
                raise SelectionError(f"{name} is {spec.status}, not ready: {spec.notes or 'no note'}")
            picked = [spec.identifier]
        else:
            vocabulary = [*aliases, *groups, *families, *by_id]
            close = difflib.get_close_matches(key, vocabulary, n=3, cutoff=0.6)
            hint = f" Did you mean {', '.join(close)}?" if close else ""
            raise SelectionError(f"unknown model {name!r}.{hint} "
                                 f"See `neuroatlas list models` and `neuroatlas list aliases`.")
        out.extend(p for p in picked if p not in out)
    return out
