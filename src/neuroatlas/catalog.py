"""The benchmark catalog: ``neuroatlas/configs/benchmarks/*.yaml``.

A benchmark names a protocol the paper ran -- which datasets, which task,
which ``embed`` and ``probe`` arguments, how it is scored -- and expands into
the exact verb invocations that run it. Those invocations are the lines of
``run/default_runs.sh``; the catalog adds no protocol of its own, it gives the
paper's a name you can type.

    bench = load("sleep_stage")
    for step in bench.steps(suite="single"):
        print(step.command(models="all_fm"))
    # neuroatlas embed --models all_fm --dataset sleep_edf_expanded --set window_s=30 ...
    # neuroatlas probe --models all_fm --dataset sleep_edf_expanded --task sleep_staging
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from neuroatlas import _paths

DEFAULT_VARIANT = "default"
SUITES = ("single", "full")


class CatalogError(ValueError):
    """A benchmark file that does not say what it must."""


@dataclass(frozen=True)
class DatasetEntry:
    slug: str
    task: Optional[str] = None      # overrides the benchmark's task for this cohort
    note: Optional[str] = None
    embed: Optional[Tuple[str, ...]] = None   # replaces the benchmark's embed args


@dataclass(frozen=True)
class Metrics:
    headline: str
    higher_is_better: bool = True
    dummy: Any = None               # number, {slug: number}, or None
    secondary: Tuple[str, ...] = ()
    tolerance: Optional[float] = None   # |ours - paper| that still counts as reproduced

    def dummy_for(self, slug: str) -> Optional[float]:
        if isinstance(self.dummy, dict):
            value = self.dummy.get(slug)
            return None if value is None else float(value)
        return None if self.dummy is None else float(self.dummy)


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    embed: Tuple[str, ...]
    probe: Tuple[str, ...]
    datasets: Optional[Tuple[str, ...]] = None   # None = the benchmark's own


@dataclass(frozen=True)
class Step:
    """One verb invocation: ``neuroatlas <verb> <argv>``."""
    verb: str                        # embed | probe | hypnogram
    dataset: str
    argv: Tuple[str, ...]

    def command(self, models: Optional[str] = None) -> str:
        extra = ("--models", models) if models and self.verb != "hypnogram" else ()
        return " ".join(["neuroatlas", self.verb, *map(shlex.quote, (*extra, *self.argv))])


@dataclass(frozen=True)
class Benchmark:
    name: str
    title: str
    domain: str
    question: str
    paper: Optional[str]
    task: Optional[str]
    embed: Tuple[str, ...]
    probe: Tuple[str, ...]
    datasets: Tuple[DatasetEntry, ...]
    single: Optional[str]
    planned: Tuple[Dict[str, str], ...]
    metrics: Metrics
    variants: Dict[str, Variant] = field(default_factory=dict)
    derived_from: Optional[str] = None

    # -- suites ---------------------------------------------------------------
    def suite(self, which: str = "full") -> List[DatasetEntry]:
        """``single``, ``full``, or a comma list of slugs from this benchmark."""
        if which == "full":
            return list(self.datasets)
        if which == "single":
            return [e for e in self.datasets if e.slug == self.single]
        wanted = [s.strip() for s in which.split(",") if s.strip()]
        known = {e.slug: e for e in self.datasets}
        unknown = [s for s in wanted if s not in known]
        if unknown:
            import difflib

            hints = []
            for slug in unknown:
                close = difflib.get_close_matches(slug, known, n=1)
                planned = [p for p in self.planned if p["name"] == slug]
                if planned:
                    hints.append(f"{slug} (planned: {planned[0]['reason']})")
                else:
                    hints.append(f"{slug} (did you mean {close[0]}?)" if close else slug)
            raise CatalogError(
                f"{self.name} does not include {', '.join(hints)}. "
                f"Its datasets: {', '.join(known) or 'none yet'}; or use --dataset single|full.")
        return [known[s] for s in wanted]

    # -- variants -------------------------------------------------------------
    def variant(self, name: str = DEFAULT_VARIANT) -> Variant:
        if name == DEFAULT_VARIANT:
            return Variant(DEFAULT_VARIANT, "the paper's headline protocol", self.embed, self.probe)
        if name not in self.variants:
            options = ", ".join([DEFAULT_VARIANT, *self.variants])
            raise CatalogError(f"{self.name} has no variant {name!r}; it has: {options}")
        return self.variants[name]

    def variant_names(self) -> List[str]:
        return [DEFAULT_VARIANT, *self.variants]

    # -- expansion ------------------------------------------------------------
    def steps(self, suite: str = "full", variant: str = DEFAULT_VARIANT) -> List[Step]:
        """Every verb invocation this benchmark runs, embed before probe."""
        chosen = self.variant(variant)
        entries = self.suite(suite)
        if chosen.datasets is not None:
            entries = [e for e in entries if e.slug in chosen.datasets]
        if self.derived_from:
            return [Step("hypnogram", e.slug, ("--datasets", e.slug)) for e in entries]
        out: List[Step] = []
        for e in entries:
            embed = chosen.embed if e.embed is None or chosen.name != DEFAULT_VARIANT else e.embed
            out.append(Step("embed", e.slug, ("--dataset", e.slug, *embed)))
        for e in entries:
            task = e.task or self.task
            head = ("--dataset", e.slug) + (("--task", task) if task else ())
            out.append(Step("probe", e.slug, head + chosen.probe))
        return out

    def task_for(self, slug: str) -> Optional[str]:
        for e in self.datasets:
            if e.slug == slug:
                return e.task or self.task
        return self.task


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

_REQUIRED = ("name", "title", "domain", "question", "metrics")


def _args(value: Any, where: str) -> Tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, str):
        raise CatalogError(f"{where} must be one string of arguments, got {type(value).__name__}")
    return tuple(shlex.split(value))


def _parse(data: Dict[str, Any], source: str) -> Benchmark:
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise CatalogError(f"{source}: missing {', '.join(missing)}")
    entries = []
    for item in data.get("datasets") or []:
        if isinstance(item, str):
            entries.append(DatasetEntry(item))
        elif isinstance(item, dict) and "slug" in item:
            embed = _args(item["embed"], f"{source} {item['slug']}.embed") if "embed" in item else None
            entries.append(DatasetEntry(item["slug"], item.get("task"), item.get("note"), embed))
        else:
            raise CatalogError(f"{source}: a dataset is a slug or {{slug, task, note}}, got {item!r}")
    slugs = [e.slug for e in entries]
    if len(set(slugs)) != len(slugs):
        raise CatalogError(f"{source}: a dataset is listed twice")
    single = data.get("single")
    if single is not None and single not in slugs:
        raise CatalogError(f"{source}: single suite {single!r} is not one of its datasets")

    m = data["metrics"] or {}
    if "headline" not in m:
        raise CatalogError(f"{source}: metrics.headline is required")
    metrics = Metrics(m["headline"], bool(m.get("higher_is_better", True)),
                      m.get("dummy"), tuple(m.get("secondary") or ()),
                      float(m["tolerance"]) if m.get("tolerance") is not None else None)

    variants = {}
    for name, v in (data.get("variants") or {}).items():
        if name == DEFAULT_VARIANT:
            raise CatalogError(f"{source}: `{DEFAULT_VARIANT}` is the implicit variant")
        subset = v.get("datasets")
        if subset is not None and not set(subset) <= set(slugs):
            raise CatalogError(f"{source}: variant {name} names datasets outside the benchmark")
        variants[name] = Variant(name, " ".join(str(v.get("description", "")).split()),
                                 _args(v.get("embed"), f"{source} {name}.embed"),
                                 _args(v.get("probe"), f"{source} {name}.probe"),
                                 tuple(subset) if subset is not None else None)
    planned = tuple({"name": str(p["name"]), "reason": str(p.get("reason", ""))}
                    for p in (data.get("planned") or []))
    return Benchmark(
        name=data["name"], title=data["title"], domain=data["domain"],
        question=" ".join(str(data["question"]).split()), paper=data.get("paper"),
        task=data.get("task"),
        embed=_args(data.get("embed"), f"{source} embed"),
        probe=_args(data.get("probe"), f"{source} probe"),
        datasets=tuple(entries), single=single, planned=planned, metrics=metrics,
        variants=variants, derived_from=data.get("derived_from"),
    )


@lru_cache(maxsize=1)
def catalog() -> Dict[str, Benchmark]:
    """Every benchmark, by name, in the order `list benchmarks` shows them."""
    import yaml

    out: Dict[str, Benchmark] = {}
    for path in sorted(_paths.configs_dir("benchmarks").glob("*.yaml")):
        bench = _parse(yaml.safe_load(path.read_text()) or {}, path.name)
        if bench.name != path.stem:
            raise CatalogError(f"{path.name}: name {bench.name!r} must match the file name")
        out[bench.name] = bench
    for bench in out.values():
        if bench.derived_from and bench.derived_from not in out:
            raise CatalogError(f"{bench.name}: derived_from {bench.derived_from!r} is not a benchmark")
    return out


def load(name: str) -> Benchmark:
    benches = catalog()
    if name in benches:
        return benches[name]
    import difflib

    close = difflib.get_close_matches(name, benches, n=3)
    hint = f" Did you mean {', '.join(close)}?" if close else ""
    raise CatalogError(f"no benchmark {name!r}.{hint} See `neuroatlas list benchmarks`.")


def benchmarks_using(slug: str) -> List[str]:
    return [b.name for b in catalog().values() if any(e.slug == slug for e in b.datasets)]
