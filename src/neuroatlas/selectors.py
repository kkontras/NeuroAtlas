"""Turn what a user types after ``-m`` / ``--models`` into checkpoint ids.

    all_fm                  an alias from model_groups.yaml
    eeg_fm                  a group name (the paper's taxonomy)
    baseline                the untrained baselines (= all_random): the group
                            `list models` shows for them
    reve                    a model family: its ready *trained* checkpoints
    reve_pretrained         one checkpoint
    all_fm,chronos_t5_base  any mix, comma-separated
    all,-sleepyco           ``-name`` removes what the terms before it selected

Only *ready* checkpoints are selected by an alias, a group or a family;
a registry entry that is not ready is not part of the release, and naming
it is an error.
An untrained (random-init) baseline is selected only by name, by
``all_random`` / ``baseline`` or by ``all``: ``neuroatlas run -m reve`` is
REVE, not REVE and its random-weight control. (The original verbs keep the
runner's meaning of a family name -- see :func:`expand_models`.) Duplicates
collapse, order is kept, and an unknown name fails with the closest matches
before any work starts.

A benchmark can leave model families out (``excluded_models`` in its
catalog file; :meth:`neuroatlas.catalog.Benchmark.select_models`): there an
alias or a group skips them and naming one is an error that says why.
"""
from __future__ import annotations

import difflib
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence

from neuroatlas import _paths


class SelectionError(ValueError):
    """A -m selection that names nothing, or something not ready. Its text is
    what is wrong, then a ``fix:`` line; ``suggest`` maps a mistyped name to
    the one it is close to (see :class:`neuroatlas.catalog.CatalogError`)."""

    def __init__(self, message: str = "", suggest: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.suggest = dict(suggest or {})


class NotInBenchmark(SelectionError):
    """A model named explicitly that the benchmark leaves out."""


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


def _names(expr: str | Sequence[str]) -> List[str]:
    return [n.strip() for n in (expr.split(",") if isinstance(expr, str) else expr) if n.strip()]


def _removal(name: str) -> str | None:
    """The name a ``-name`` term removes, or None for a term that adds."""
    if not name.startswith("-"):
        return None
    key = name[1:].strip()
    if not key:
        raise SelectionError("`-` needs a name after it\nfix: -m all,-reve (every ready "
                             "checkpoint but REVE's)")
    return key


def _removed_ids(key: str, specs) -> List[str]:
    """Every checkpoint id ``-key`` removes: what an alias or a group selects,
    *every* checkpoint of a family (its random-init control too: ``all,-reve``
    is no REVE at all), or the one id."""
    key = key.lower()
    groups, aliases = model_groups()["groups"], model_groups()["aliases"]
    if key in aliases or key in groups:
        return resolve_models([key])
    family = [s.identifier for s in specs if s.model_family == key]
    if family:
        return family
    if any(s.identifier == key for s in specs):
        return [key]
    resolve_models([key])                        # raises: unknown, with the closest names
    return []


def expand_models(expr: str | Sequence[str]) -> List[str]:
    """What the original verbs' ``--models`` hands the runner: aliases and
    groups become checkpoint ids; a family or an id stays exactly as typed,
    after checking it exists.

    The runner matches a family itself -- every checkpoint of it, the
    random-init control included, which is what the paper's launchers rely
    on when they pass a list of families -- and per-model overrides
    (``--checkpoint``, ``--expected-epoch-seconds``) are keyed by the name the
    user typed, so keeping it keeps those working unchanged. The new
    commands (run, check, submit) resolve with :func:`resolve_models` and
    hand the verbs ids, so there a family means its trained checkpoints.

    A ``-name`` term removes what came before it (see :func:`resolve_models`);
    a family kept as typed that loses a checkpoint to it is spelled out as
    its remaining checkpoints.
    """
    names = _names(expr)
    groups, aliases = model_groups()["groups"], model_groups()["aliases"]
    specs = _registry()
    out: List[str] = []
    for name in names:
        removed = _removal(name)
        if removed is not None:
            drop = set(_removed_ids(removed, specs))
            kept: List[str] = []
            for entry in out:
                members = [s.identifier for s in specs if s.model_family == entry.lower()]
                if members and drop.intersection(members):
                    kept.extend(m for m in members if m not in drop and m not in kept)
                elif entry.lower() not in drop and entry not in kept:
                    kept.append(entry)
            out = kept
            continue
        picked = resolve_models([name])          # validates; raises on unknown / planned
        if name.lower() not in aliases and name.lower() not in groups:
            picked = [name]
        out.extend(p for p in picked if p not in out)
    if names and not out:
        raise SelectionError(f"{','.join(names)!r} selects no checkpoint: its `-` terms remove "
                             f"everything before them")
    return out


def resolve_models(expr: str | Sequence[str], *,
                   exclude: Optional[Mapping[str, str]] = None, scope: str = "") -> List[str]:
    """Checkpoint ids for a selector expression. Raises SelectionError.

    Terms are read left to right; ``-name`` removes what the terms before it
    selected (``all,-sleepyco``, ``all_fm,-reve``): an alias's or a group's
    members, every checkpoint of a family, or one id.

    *exclude* -- ``{family: reason}``, the model families a benchmark does not
    evaluate (its ``excluded_models``), *scope* naming it ("the epilepsy
    benchmark"). An alias or a group leaves those families out; naming one
    of them, as a family or a checkpoint id, is a SelectionError that says
    why.
    """
    names = _names(expr)
    if not names:
        raise SelectionError("no checkpoint selected\nfix: neuroatlas list aliases (the "
                             "names -m takes)")
    specs = _registry()
    by_id = {s.identifier: s for s in specs}
    families: Dict[str, List[Any]] = {}
    for s in specs:
        families.setdefault(s.model_family, []).append(s)
    aliases = model_groups()["aliases"]
    groups = model_groups()["groups"]
    exclude = {f.lower(): why for f, why in (exclude or {}).items()}

    out: List[str] = []
    took_away = False
    for name in names:
        removed = _removal(name)
        if removed is not None:
            drop = set(_removed_ids(removed, specs))
            took_away = took_away or any(i in drop for i in out)
            out = [i for i in out if i not in drop]
            continue
        key = name.lower()
        named = False                        # a family or an id, not a set of them
        if key in aliases:
            picked = alias_members(key, specs)
        elif key in groups:
            fams = set(groups[key]["families"])
            picked = [s.identifier for s in specs
                      if s.status == "ready" and s.model_family in fams and not is_baseline(s)]
        elif key in families:
            named = True
            ready = [s for s in families[key] if s.status == "ready"]
            # a family means its trained checkpoints; its random-init control
            # only when it has nothing else
            picked = [s.identifier for s in ready if not is_baseline(s)] or \
                [s.identifier for s in ready]
            if not picked and key not in exclude:
                raise SelectionError(f"{name} is not a model family of this release\n"
                                     f"fix: neuroatlas list models")
        elif key in by_id:
            named = True
            spec = by_id[key]
            if spec.status != "ready" and spec.model_family not in exclude:
                raise SelectionError(f"{name} is not a checkpoint of this release\n"
                                     f"fix: neuroatlas list models")
            picked = [spec.identifier]
        else:
            ready = [s for s in specs if s.status == "ready"]
            vocabulary = [*aliases, *groups, *dict.fromkeys(s.model_family for s in ready),
                          *(s.identifier for s in ready)]
            close = difflib.get_close_matches(key, vocabulary, n=1, cutoff=0.6)
            hint = f" (did you mean {', '.join(close)}?)" if close else ""
            raise SelectionError(f"no checkpoint, family, group or alias {name!r}{hint}\n"
                                 f"fix: neuroatlas list models\n"
                                 f"fix: neuroatlas list aliases",
                                 suggest={name: close[0]} if close else None)
        if exclude:
            family = key if key in families else by_id[key].model_family if key in by_id else None
            if named and family in exclude:
                raise NotInBenchmark(f"{name} is not part of {scope or 'this benchmark'} "
                                     f"({family}: {exclude[family]})")
            kept = [p for p in picked if by_id[p].model_family not in exclude]
            took_away = took_away or len(kept) < len(picked)
            picked = kept
        out.extend(p for p in picked if p not in out)
    if not out and took_away:
        where = f" in {scope}" if scope else ""
        bench = scope.split()[1] if scope.startswith("the ") and len(scope.split()) > 2 else None
        raise SelectionError(f"{','.join(names)!r} selects no checkpoint{where}\n"
                             + (f"fix: neuroatlas list models --benchmark {bench}" if bench
                                else "fix: neuroatlas list aliases"))
    return out
