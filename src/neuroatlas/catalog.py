"""The benchmark catalog: ``neuroatlas/configs/benchmarks/*.yaml``.

A benchmark names a protocol the paper ran -- which datasets, which task,
which ``embed`` and ``probe`` arguments, how it is scored -- and expands into
the exact verb invocations that run it. Each invocation has the same
arguments as a line of ``run/default_runs.sh`` (which spells the verb
``python -m neuroatlas.entrypoints.<verb> --models all``, or, for a benchmark
that leaves model families out, ``--models all,-<family>,...``:
:meth:`Benchmark.models_arg`; `neuroatlas <verb>` runs the same code); the
catalog adds no protocol of its own, it gives the paper's a name you can
type. ``tests/test_benchmark_catalog.py`` holds the two to the same set of
commands, --models included.

    bench = load("sleep_stage")
    for step in bench.steps(suite="single"):
        print(step.command(models="all_fm"))
    # neuroatlas embed --models all_fm --dataset sleep_edf_expanded
    # neuroatlas probe --models all_fm --dataset sleep_edf_expanded --task sleep_staging

A probe step carries every ``--set`` of its embed step (``probe_args``): the
probe must resolve the dataset exactly as the embed did, or it computes a
different embedding-cache key and finds nothing.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from neuroatlas import _paths

DEFAULT_VARIANT = "default"
SUITES = ("single", "full")


def _suite_fix(given: str, value: str, benchmark: str) -> str:
    """The command being run with ``--dataset <value>`` in place of *given*;
    `show` (which lists the datasets) when it is not known."""
    from neuroatlas.cli import corrected_command

    return (corrected_command({given: value})
            or corrected_command({f"--dataset={given}": f"--dataset={value}"})
            or f"neuroatlas show {benchmark} (lists its datasets)")


class CatalogError(ValueError):
    """A benchmark file that does not say what it must, or a name that is not
    in the catalog. Its text is what is wrong, then a ``fix:`` line
    (:mod:`neuroatlas.cli._msg`); ``suggest`` maps a mistyped word to the
    one it is close to, so the command can print the corrected command."""

    def __init__(self, message: str = "", suggest: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.suggest = dict(suggest or {})


@dataclass(frozen=True)
class DatasetEntry:
    slug: str
    task: Optional[str] = None      # overrides the benchmark's task for this cohort
    note: Optional[str] = None
    embed: Optional[Tuple[str, ...]] = None   # replaces the benchmark's embed args


@dataclass(frozen=True)
class Metrics:
    """How a benchmark is scored. What each metric key means is the metric
    registry's (:mod:`neuroatlas.metrics_info`); this says which keys, where
    in a fold's metrics they are, and the words that put them in context."""
    headline: str
    higher_is_better: bool = True
    # The headline's chance level when the registry's does not hold: a number,
    # "prevalence", "1/C"; None takes the registry's (the YAML key was `dummy`).
    chance: Any = None
    secondary: Tuple[str, ...] = ()
    unit: Optional[str] = None          # one scored item: "30 s epoch", "subject", "trial"
    # Where every metric of a row is in a fold's metrics, for a task that fits
    # several probes per fold: ("threshold_3.0s",), ("ahi_fraction",
    # "threshold_10.0s"). The YAML spells it `at: ahi_fraction@10.0s` / `at: 3.0s`.
    at: Tuple[str, ...] = ()
    describe: Optional[str] = None      # the headline in one line: unit, classes, pooling
    fold: Optional[str] = None          # what a fold is
    spread: Optional[str] = None        # what ± is over; None: one value, no ±
    note: Optional[str] = None          # one more line under the title

    def at_text(self) -> Optional[str]:
        """``at`` as the YAML spells it (the JSON ``at`` field)."""
        if not self.at:
            return None
        *field_, threshold = self.at
        seconds = threshold[len("threshold_"):-1] if threshold.startswith("threshold_") else threshold
        return "@".join([*field_, f"{seconds}s"])

    def at_seconds(self) -> Optional[float]:
        """The event threshold the headline is read at, in seconds."""
        if not self.at or not self.at[-1].startswith("threshold_"):
            return None
        try:
            return float(self.at[-1][len("threshold_"):-1])
        except ValueError:
            return None

    def event_words(self) -> Optional[str]:
        """What the threshold counts, in the title's words (``scored arousal``,
        ``scored apnea or hypopnea``), from ``describe``; else from the field."""
        import re

        match = re.search(r"more than [\d.]+ s of (.+?) in the epoch", " ".join(
            str(self.describe or "").split()))
        if match:
            return match.group(1)
        field_ = self.at[0] if len(self.at) > 1 else None
        return EVENT_WORDS.get(field_ or "", None)

    def at_words(self) -> Optional[str]:
        """``3 s of scored arousal``: the headline's threshold in words, for
        messages (machine formats keep :meth:`at_text`)."""
        seconds = self.at_seconds()
        if seconds is None:
            return self.at_text()
        event = self.event_words()
        return f"{seconds:g} s" + (f" of {event}" if event else "")


#: An event field's words, when a benchmark's ``describe`` does not give them.
EVENT_WORDS = {"ahi_fraction": "scored apnea or hypopnea",
               "limb_movement_plm_fraction": "scored periodic limb movement",
               "arousal_fraction": "scored arousal"}


def parse_at(text: Any, where: str = "") -> Tuple[str, ...]:
    """``ahi_fraction@10.0s`` -> ``("ahi_fraction", "threshold_10.0s")``,
    ``3s`` -> ``("threshold_3.0s",)``: the keys an event task files a probe
    under (``threshold_<float seconds>s``, as respiratory_event_detection and
    arousal_detection write them)."""
    if text in (None, ""):
        return ()
    field_, _, threshold = str(text).rpartition("@")
    seconds = threshold.strip()
    if seconds.endswith("s"):
        seconds = seconds[:-1]
    try:
        value = float(seconds)
    except ValueError:
        raise CatalogError(f"{where}metrics.at is [field@]<seconds>s, e.g. ahi_fraction@10.0s "
                           f"or 3.0s, got {text!r}") from None
    return (*([field_.strip()] if field_.strip() else []), f"threshold_{value}s")


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    embed: Tuple[str, ...]
    probe: Tuple[str, ...]
    datasets: Optional[Tuple[str, ...]] = None   # None = the benchmark's own
    aliases: Tuple[str, ...] = ()       # earlier names, still accepted and read
    paper: Optional[str] = None         # which paper figure it reproduces


@dataclass(frozen=True)
class ModelExclusion:
    """Model families a benchmark does not evaluate, and why."""
    families: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class Step:
    """One verb invocation: ``neuroatlas <verb> <argv>``."""
    verb: str                        # prepare | embed | probe | hypnogram
    dataset: str
    argv: Tuple[str, ...]

    def command(self, models: Optional[str] = None) -> str:
        extra = ("--models", models) if models and self.verb in ("embed", "probe") else ()
        return " ".join(["neuroatlas", self.verb, *map(shlex.quote, (*extra, *self.argv))])


def prepare_required(slug: str) -> bool:
    """Whether *slug* must be built by `prepare` before `embed` can read it.

    True for a MOABB cohort (acquisition kind ``moabb``) with a declared
    ``pipeline.preprocessor`` that its manifest does not mark
    ``required: false``. No paper cohort is one today: the five MI cohorts
    with a builder (bnci2014_001, bnci2014_004, bnci2015_001, shin2017a,
    weibo2014) load through the MOABB reader like the other nine, and their
    pickle is optional. Other cohorts with a builder (the epilepsy HDF5
    caches) only use it to go faster.
    """
    from neuroatlas.benchmarking_helpers.registry.discovery import load_dataset_spec

    manifest = load_dataset_spec(slug).manifest or {}
    kind = (manifest.get("acquisition") or {}).get("kind")
    pipeline = manifest.get("pipeline") or {}
    return (kind == "moabb" and isinstance(pipeline.get("preprocessor"), dict)
            and pipeline.get("required", True) is not False)


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
    excluded_models: Tuple[ModelExclusion, ...] = ()
    # what the default variant is, in the words of the paper (BCI: "no
    # filtering, mean-pool") and the figure it reproduces
    default_description: Optional[str] = None
    default_paper: Optional[str] = None

    # -- metrics --------------------------------------------------------------
    def metric_info(self, key: str):
        """*key* as this benchmark means it (:func:`metrics_info.info`)."""
        from neuroatlas import metrics_info

        return metrics_info.info(key, self)

    def chance(self):
        """The headline's chance level: a number, ``"prevalence"``, ``"1/C"``
        or None -- the YAML's ``metrics.chance``, else the registry's."""
        if self.metrics.chance is not None:
            return self.metrics.chance
        return self.metric_info(self.metrics.headline).chance

    def loso_datasets(self, variant: str = DEFAULT_VARIANT) -> List[str]:
        """The datasets this benchmark probes leave-one-subject-out
        (``--set n_folds=loso``): one fold per subject."""
        if self.derived_from:
            return []
        return [e.slug for e in self.datasets
                if any(k == "n_folds" and v.strip().lower() == "loso"
                       for k, v in _set_pairs(self.probe_args(e, variant)))]

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

            hints, suggest = [], {}
            for slug in unknown:
                close = difflib.get_close_matches(slug, known, n=1)
                planned = [p for p in self.planned if p["name"] == slug]
                if planned:
                    hints.append(f"{slug} (planned: {planned[0]['reason']})")
                elif close:
                    hints.append(f"{slug} (did you mean {close[0]}?)")
                    suggest[slug] = close[0]
                else:
                    hints.append(slug)
            kept = [s for s in wanted if s in known]
            fix = _suite_fix(which, ",".join(kept) or "full", self.name)
            raise CatalogError(
                f"the {self.name} benchmark has no dataset {', '.join(hints)}; its datasets: "
                f"{', '.join(known)}\nfix: {fix}",
                suggest=suggest)
        return [known[s] for s in wanted]

    # -- variants -------------------------------------------------------------
    def variant(self, name: str = DEFAULT_VARIANT) -> Variant:
        """The variant *name* -- or an earlier name of it (``per_patch`` is
        ``token_flattening``) -- under its current name."""
        if name == DEFAULT_VARIANT:
            return Variant(DEFAULT_VARIANT,
                           self.default_description or "the paper's headline protocol",
                           self.embed, self.probe, paper=self.default_paper)
        name = self.variant_folders().get(name, name)
        if name not in self.variants:
            import difflib

            close = difflib.get_close_matches(name, list(self.variant_folders()), n=1)
            hint = f" (did you mean {close[0]}?)" if close else ""
            options = ", ".join([DEFAULT_VARIANT, *self.variants])
            raise CatalogError(f"the {self.name} benchmark has no variant {name!r}{hint}; "
                               f"its variants: {options}\n"
                               f"fix: neuroatlas show {self.name} (explains each)",
                               suggest={name: close[0]} if close else None)
        return self.variants[name]

    def variant_names(self) -> List[str]:
        return [DEFAULT_VARIANT, *self.variants]

    def variant_folders(self) -> Dict[str, str]:
        """``{folder or name: variant}``: each variant's name and its earlier
        names, so results written under an earlier name still read as the
        variant's (``results.variant_of``)."""
        out = {name: name for name in self.variant_names()}
        for v in self.variants.values():
            out.update({alias: v.name for alias in v.aliases})
        return out

    # -- models ---------------------------------------------------------------
    def excluded_families(self) -> Dict[str, str]:
        """``{family: reason}`` for the model families this benchmark does not
        evaluate: its ``excluded_models``, and those of the benchmark it is
        derived from."""
        out: Dict[str, str] = {}
        if self.derived_from:
            out.update(load(self.derived_from).excluded_families())
        for exclusion in self.excluded_models:
            out.update({f: exclusion.reason for f in exclusion.families})
        return out

    def select_models(self, expr) -> List[str]:
        """The checkpoint ids *expr* (``-m``) selects for this benchmark:
        ``selectors.resolve_models``, with the families the benchmark leaves
        out dropped from an alias or a group, and refused (SelectionError, a
        usage error) when named."""
        from neuroatlas import selectors

        excluded = self.excluded_families()
        if excluded:
            known = {s.model_family for s in selectors._registry()}
            unknown = sorted(set(excluded) - known)
            if unknown:
                raise CatalogError(f"{self.name}: excluded_models names "
                                   f"{', '.join(unknown)}, which no checkpoint has as its family")
        try:
            return selectors.resolve_models(expr, exclude=excluded,
                                            scope=f"the {self.name} benchmark")
        except selectors.NotInBenchmark as exc:
            raise selectors.NotInBenchmark(
                f"{exc}\nfix: neuroatlas list models --benchmark {self.name}") from None

    def left_out(self, expr) -> List[str]:
        """The checkpoint ids *expr* selects outside this benchmark that the
        benchmark leaves out (an alias or a group taking an excluded family)."""
        from neuroatlas import selectors

        if not self.excluded_families():
            return []
        kept = set(self.select_models(expr))
        return [i for i in selectors.resolve_models(expr) if i not in kept]

    def left_out_note(self, expr) -> Optional[str]:
        """The line that says which checkpoints of *expr* this benchmark left
        out, and why; None when it left out none."""
        from neuroatlas import selectors

        ids = self.left_out(expr)
        if not ids:
            return None
        by_id = {s.identifier: s for s in selectors._registry()}
        families = list(dict.fromkeys(by_id[i].model_family for i in ids))
        n = f"{len(ids)} checkpoint{'s' if len(ids) != 1 else ''}"
        return f"left out {', '.join(families)} ({n}): not in the {self.name} benchmark"

    def models_arg(self, expr: str = "all") -> str:
        """*expr* as the verbs' ``--models`` must spell it for this benchmark:
        the verbs know no benchmark, so each left-out family that *expr*
        would otherwise select is removed by name (``all`` on epilepsy:
        ``all,-sleep_transformer,-sleepyco,-core_sleep``). Raises as
        :meth:`select_models` does."""
        from neuroatlas import selectors

        self.select_models(expr)
        excluded = self.excluded_families()
        if not excluded:
            return expr
        by_id = {s.identifier: s for s in selectors._registry()}
        taken = {by_id[i].model_family for i in selectors.resolve_models(expr)}
        return ",".join([expr, *(f"-{f}" for f in excluded if f in taken)])

    # -- expansion ------------------------------------------------------------
    def embed_args(self, entry: DatasetEntry, variant: str = DEFAULT_VARIANT) -> Tuple[str, ...]:
        chosen = self.variant(variant)
        return chosen.embed if entry.embed is None or chosen.name != DEFAULT_VARIANT else entry.embed

    def probe_args(self, entry: DatasetEntry, variant: str = DEFAULT_VARIANT) -> Tuple[str, ...]:
        """The probe's arguments: the dataset's ``--set`` overrides from the
        embed line, then the benchmark's probe arguments.

        A probe reads the cache the embed wrote, and the cache key is computed
        from the dataset config. So every ``--set`` the embed is given, the
        probe must be given too, with the same spelling: the key hashes the
        values as JSON, so even ``window_s=10`` against the manifest's
        ``10.0`` is a different key (F-082). They are copied here rather than
        written twice in the YAML, so the two cannot drift apart. Flags that
        belong to the checkpoint, not the dataset (``--expected-epoch-seconds``),
        stay on the embed: they are not part of the key.
        """
        chosen = self.variant(variant)
        embed_sets = _set_pairs(self.embed_args(entry, variant))
        probe_sets = dict(_set_pairs(chosen.probe))
        copied: List[str] = []
        for key, value in embed_sets:
            if key in probe_sets:
                if probe_sets[key] != value:
                    raise CatalogError(
                        f"{self.name} ({chosen.name}) {entry.slug}: embed sets {key}={value} "
                        f"but probe sets {key}={probe_sets[key]}; the probe would look "
                        "for embeddings the embed never wrote")
                continue
            copied += ["--set", f"{key}={value}"]
        task = entry.task or self.task
        head = ("--dataset", entry.slug) + (("--task", task) if task else ())
        return head + tuple(copied) + chosen.probe

    def steps(self, suite: str = "full", variant: str = DEFAULT_VARIANT, *,
              prepare: bool = False) -> List[Step]:
        """Every verb invocation this benchmark runs, embed before probe.

        ``prepare=True`` puts a ``prepare`` step first for each dataset that
        needs one (``prepare_required``), as `run/default_runs.sh` section 2
        does. It is off by default because `run` and `submit` act only on the
        embed and probe steps and check the prepared data themselves.
        """
        chosen = self.variant(variant)
        entries = self.suite(suite)
        if chosen.datasets is not None:
            entries = [e for e in entries if e.slug in chosen.datasets]
        if self.derived_from:
            return [Step("hypnogram", e.slug, ("--datasets", e.slug)) for e in entries]
        out: List[Step] = []
        if prepare:
            out += [Step("prepare", e.slug, ("--dataset", e.slug))
                    for e in entries if prepare_required(e.slug)]
        for e in entries:
            out.append(Step("embed", e.slug, ("--dataset", e.slug, *self.embed_args(e, variant))))
        for e in entries:
            out.append(Step("probe", e.slug, self.probe_args(e, variant)))
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


def _set_pairs(argv: Tuple[str, ...]) -> List[Tuple[str, str]]:
    """The ``--set key=value`` pairs in *argv*, in order (``--set=k=v`` too)."""
    out: List[Tuple[str, str]] = []
    it = iter(range(len(argv)))
    for i in it:
        arg = argv[i]
        if arg == "--set" and i + 1 < len(argv):
            pair = argv[i + 1]
            next(it, None)
        elif arg.startswith("--set="):
            pair = arg[len("--set="):]
        else:
            continue
        key, _, value = pair.partition("=")
        out.append((key.strip(), value))
    return out


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
    # `chance` was called `dummy`; a benchmark file written before reads the same
    chance = m.get("chance", m.get("dummy"))
    if not (chance is None or chance in ("prevalence", "1/C")
            or (isinstance(chance, (int, float)) and not isinstance(chance, bool))):
        raise CatalogError(f"{source}: metrics.chance is a number, prevalence or 1/C, "
                           f"got {chance!r}")
    text = lambda key: " ".join(str(m[key]).split()) if m.get(key) not in (None, "") else None
    metrics = Metrics(m["headline"], bool(m.get("higher_is_better", True)),
                      chance, tuple(m.get("secondary") or ()),
                      unit=text("unit"), at=parse_at(m.get("at"), f"{source}: "),
                      describe=text("describe"), fold=text("fold"), spread=text("spread"),
                      note=text("note"))

    variants = {}
    taken = {DEFAULT_VARIANT}
    for name, v in (data.get("variants") or {}).items():
        if name == DEFAULT_VARIANT:
            raise CatalogError(f"{source}: `{DEFAULT_VARIANT}` is the implicit variant")
        subset = v.get("datasets")
        if subset is not None and not set(subset) <= set(slugs):
            raise CatalogError(f"{source}: variant {name} names datasets outside the benchmark")
        aliases = tuple(str(a) for a in (v.get("aliases") or ()))
        if taken & {name, *aliases}:
            raise CatalogError(f"{source}: variant name {sorted(taken & {name, *aliases})[0]!r} "
                               f"is used twice")
        taken |= {name, *aliases}
        variants[name] = Variant(name, " ".join(str(v.get("description", "")).split()),
                                 _args(v.get("embed"), f"{source} {name}.embed"),
                                 _args(v.get("probe"), f"{source} {name}.probe"),
                                 tuple(subset) if subset is not None else None,
                                 aliases=aliases,
                                 paper=" ".join(str(v["paper"]).split()) if v.get("paper") else None)
    default = data.get("default_variant") or {}
    planned = tuple({"name": str(p["name"]), "reason": str(p.get("reason", ""))}
                    for p in (data.get("planned") or []))
    exclusions = []
    for item in data.get("excluded_models") or []:
        families = item.get("families") if isinstance(item, dict) else None
        reason = " ".join(str(item.get("reason") or "").split()) if isinstance(item, dict) else ""
        if not (isinstance(families, list) and families
                and all(isinstance(f, str) for f in families) and reason):
            raise CatalogError(f"{source}: an excluded_models entry is {{families: [family, ...], "
                               f"reason: why}}, got {item!r}")
        exclusions.append(ModelExclusion(tuple(families), reason))
    return Benchmark(
        name=data["name"], title=data["title"], domain=data["domain"],
        question=" ".join(str(data["question"]).split()), paper=data.get("paper"),
        task=data.get("task"),
        embed=_args(data.get("embed"), f"{source} embed"),
        probe=_args(data.get("probe"), f"{source} probe"),
        datasets=tuple(entries), single=single, planned=planned, metrics=metrics,
        variants=variants, derived_from=data.get("derived_from"),
        excluded_models=tuple(exclusions),
        default_description=(" ".join(str(default["description"]).split())
                             if default.get("description") else None),
        default_paper=" ".join(str(default["paper"]).split()) if default.get("paper") else None,
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
        if not bench.derived_from:
            for variant in bench.variant_names():      # raises on an embed/probe --set conflict
                for e in bench.datasets:
                    bench.probe_args(e, variant)
    return out


def load(name: str) -> Benchmark:
    benches = catalog()
    if name in benches:
        return benches[name]
    import difflib

    close = difflib.get_close_matches(name, benches, n=1)
    hint = f" (did you mean {', '.join(close)}?)" if close else ""
    raise CatalogError(f"no benchmark {name!r}{hint}\nfix: neuroatlas list benchmarks",
                       suggest={name: close[0]} if close else None)


def benchmarks_using(slug: str) -> List[str]:
    return [b.name for b in catalog().values() if any(e.slug == slug for e in b.datasets)]
