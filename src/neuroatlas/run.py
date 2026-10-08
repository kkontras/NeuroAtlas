"""Run a benchmark: plan it, then execute its steps -- ``embed`` then
``probe`` per dataset -- in this process.

Every step is the catalog's verb invocation (the paper's command line) plus
the few arguments a run adds: the model selection, where results go, and,
for a quick try, fold 0 only. The verbs do the work exactly as they do when
called by hand.

Results land in ``<output_root>/<benchmark>/<dataset>/`` (a variant other
than the default adds ``/<variant>``): the runner's results.json, which
``neuroatlas results`` reads.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neuroatlas import _paths
from neuroatlas.catalog import CatalogError


@dataclass
class DatasetPlan:
    benchmark: str
    dataset: str
    task: str
    models: List[str]
    skipped: List[str]            # not applicable (channel map)
    folds: List[Any]
    seeds: str
    data: str
    output: Path
    embed_argv: List[str]
    probe_argv: List[str]
    notes: List[str] = field(default_factory=list)
    output_base: Optional[Path] = None      # the output root `output` sits under
    # models the dataset's channel map has no entry for (state `invalid`):
    # never run, reported as invalid -- not n/a, and not a failure
    invalid: List[str] = field(default_factory=list)
    # moved under <output root>/_limited for a --limit-batches run
    limited: bool = False
    # why the dataset's channel map does not load (nothing runs there)
    map_error: Optional[str] = None
    # the data is not here (`data status`: missing, empty, not prepared, ...):
    # `run` skips the dataset, with *data_fix* (the fix lines) and where it looked
    data_missing: bool = False
    data_sample: bool = False        # only `data download --first N`'s recordings
    data_fix: Optional[str] = None
    data_path: Optional[str] = None

    @property
    def n_runs(self) -> int:
        return len(self.models) * max(1, len(self.folds))


def result_dir(benchmark: str, dataset: str, variant: str = "default",
               output_root: Optional[Path] = None) -> Path:
    base = Path(output_root) if output_root else _paths.output_dir()
    out = base / benchmark / dataset
    return out if variant == "default" else out / variant


#: The fix of a channel map that does not load: the package's own files are
#: damaged, and reinstalling it restores them.
REINSTALL = "pip install --force-reinstall neuroatlas"


def channel_map(dataset: str):
    """(map or None, error message or None). A map that fails validation is
    reported against its dataset instead of stopping a whole suite."""
    from neuroatlas.benchmarking_helpers.channels.channel_map import load_channel_map
    from neuroatlas.cli import _msg

    try:
        return load_channel_map(dataset), None
    except (ValueError, KeyError) as exc:
        return None, (f"the {dataset} channel map shipped with the package does not load: "
                      f"{_msg.brief(str(exc).split(': ', 1)[-1])}")


def map_states(cmap, specs) -> Tuple[List[str], Dict[str, str]]:
    """The checkpoints a loaded channel map rules out, decided before any data
    is read (``ChannelMap.state_for``, the states `check` reports):
    ``(skipped, invalid)`` -- the ids the map marks ``skip`` (n/a), and
    {id: family} for those whose family it has no entry for (``invalid``).
    Neither is ever run. No map: nothing is ruled out."""
    skipped: List[str] = []
    invalid: Dict[str, str] = {}
    if cmap is None:
        return skipped, invalid
    for s in specs:
        state, _ = cmap.state_for(s.model_family)
        if state == "skip":
            skipped.append(s.identifier)
        elif state == "invalid":
            invalid[s.identifier] = s.model_family
    return skipped, invalid


def _and(words: List[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def family_text(families: List[str]) -> str:
    """``the neurogpt family``, ``the neurogpt and steegformer families``."""
    return f"the {_and(list(families))} {'family' if len(families) == 1 else 'families'}"


def ids_text(ids: Dict[str, str]) -> str:
    """Checkpoint ids for a message: themselves when few, else by family
    (``steegformer (3 checkpoints)``); JSON keeps every id."""
    if len(ids) <= 3:
        return ", ".join(ids)
    groups: Dict[str, List[str]] = {}
    for ident, family in ids.items():
        groups.setdefault(family, []).append(ident)
    return ", ".join(f"{f} ({len(m)} checkpoints)" if len(m) > 1 else m[0]
                     for f, m in groups.items())


def invalid_fix(models: str, families: List[str], fallback: str) -> str:
    """The command being run with these families taken out of its -m
    selection; *fallback* (a ``neuroatlas run`` line) when it is not known."""
    from neuroatlas.cli import command_with_value, without_families

    return (command_with_value(("-m", "--models"), without_families(models, families))
            or fallback)


def invalid_note(dataset: str, invalid: Dict[str, str], models: str = "all",
                 fallback: Optional[str] = None) -> str:
    """The message a plan carries for its invalid pairs: which and why, then
    the command without their families on its own ``fix:`` line
    (:mod:`neuroatlas.cli._msg`)."""
    from neuroatlas.cli import without_families

    families = sorted(set(invalid.values()))
    fix = invalid_fix(models, families, fallback or
                      f"neuroatlas run <benchmark> -m {without_families(models, families)}")
    return (f"invalid, not run: {ids_text(invalid)} (the {dataset} channel map has no entry "
            f"for {family_text(families)})\nfix: {fix}")


#: `data status` states that mean the data a run reads is not here at all:
#: `run` skips such a dataset (it never starts a step that cannot read it).
DATA_NOT_HERE = ("missing", "empty", "not prepared", "not configured", "not downloaded")


def is_sample(state: Any) -> bool:
    """A `data status` state of a dataset of which only the first recordings
    are here (``sample (first 3 recordings)``, `data download --first N`)."""
    return str(state).startswith("sample")


class PlanError(CatalogError):
    """A run that cannot be planned as asked, e.g. a fold the dataset does not
    have. A usage error: the command layer prints a CatalogError as
    ``error: ...`` and exits 2, and this is one."""


def _check_folds(dataset: str, step_argv: List[str], wanted: List[Any],
                 given: str = "") -> Optional[str]:
    """Why *wanted* (the user's ``--folds`` *given*) is not a set of this
    dataset's folds, with the command that asks for folds it has; None if it is."""
    available = _folds(dataset, list(step_argv))
    if not available or not all(isinstance(f, int) for f in available):
        return None                              # unknown count, or a single split
    bad = [f for f in wanted if f not in available]
    if not bad:
        return None
    lo, hi = min(available), max(available)
    span = f"{lo}-{hi}" if available == list(range(lo, hi + 1)) else ", ".join(map(str, available))
    return (f"{dataset}: fold {', '.join(map(str, bad))} does not exist; "
            f"it has {len(available)} {'fold' if len(available) == 1 else 'folds'}: {span}\n"
            f"fix: {_with_folds(given, span.replace(' ', ''))}")


def _with_folds(given: str, folds: str) -> str:
    """The command being run with ``--folds <folds>`` in place of *given*
    (from the command line, when known)."""
    from neuroatlas.cli import corrected_command

    return (corrected_command({given: folds}) if given else None) or f"--folds {folds}"


def _folds(dataset: str, probe_argv: List[str]) -> List[Any]:
    from neuroatlas.entrypoints import probe

    try:
        cfg = probe.build_config(probe.build_parser(probe_argv).parse_args(probe_argv))
        block = cfg["datasets"][dataset]
        return list(block.get("folds") or [block.get("fold", "-")])
    except SystemExit:
        return ["? (not counted)"]
    except ModuleNotFoundError as exc:           # LOSO needs moabb to count the subjects
        return [f"? (needs {exc.name} to count)"]
    except Exception:
        return ["? (not counted)"]


#: Dataset settings that decide which folds exist (beside the fold list).
_FOLD_STRUCTURE = ("n_folds",)


def embed_argv(embed: List[str], probe_argv: List[str], folds: List[Any]) -> List[str]:
    """The embed step of a run: the catalog's embed line, plus the folds the
    probe will read (and the fold count they are numbered under).

    Embeddings cached per fold and split -- a cohort that cuts its train
    split per fold (TUAB's H5 fast path, CHB-MIT's HDF5 backend, the
    PhysioEx sleep cohorts), or any cohort run with stride_s != window_s --
    serve only the fold they were extracted for. The embed step used to
    extract the manifest's fold 0 alone, so the probe of fold 1 found
    nothing. Where one cache serves every fold (the global ``all/<key>``
    layout of most cohorts) the runner stops after the first fold
    (``runner._serves_every_fold``), so the extra folds cost nothing there.
    """
    if not folds or not all(isinstance(f, int) for f in folds):
        return list(embed)
    from neuroatlas import catalog

    out = list(embed)
    have = {k for k, _ in catalog._set_pairs(tuple(out))}
    for key, value in catalog._set_pairs(tuple(probe_argv)):
        if key in _FOLD_STRUCTURE and key not in have:
            out += ["--set", f"{key}={value}"]
    return out + ["--folds", ",".join(str(f) for f in folds)]


def staging_results(dataset: str, output_root: Path) -> List[Path]:
    """The folders holding *dataset*'s sleep-staging results.json under
    *output_root*: ``run``'s <root>/sleep_stage/<dataset>/, ``submit``'s
    per-model <root>/sleep_stage/<dataset>/<model>/, the probe verb's
    <root>/<dataset>/sleep_staging/ -- the search the hypnogram verb makes."""
    from neuroatlas.entrypoints import hypnogram as hyp

    return hyp.staging_results_dirs(dataset, root=output_root)


def _run_line(benchmark: str, models: str, suite: str, variant: str,
              invalid: Dict[str, str]) -> str:
    """``neuroatlas run`` of this plan without the invalid pairs' families:
    the fix line when the command being run is not known (the Python API)."""
    from neuroatlas.cli import without_families

    families = sorted(set(invalid.values()))
    return " ".join(["neuroatlas run", benchmark, "-m", without_families(models, families),
                     *(["--dataset", suite] if suite != "single" else []),
                     *(["--variant", variant] if variant != "default" else [])])


def plan(benchmark: str, models: str, suite: str = "single", variant: str = "default", *,
         debug: bool = False, output_root: Optional[Path] = None,
         folds: Optional[str] = None, per_model: bool = False) -> List[DatasetPlan]:
    """``per_model`` puts each model's results in <dataset>/<model>/: cluster
    jobs, one per (dataset, model), never write the same results.json."""
    from neuroatlas import catalog, data
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry
    from neuroatlas.cli import _msg

    bench = catalog.load(benchmark)
    variant = bench.variant(variant).name       # an earlier name: the variant's (W1)
    ids = bench.select_models(models)
    by_id = {s.identifier: s for s in checkpoint_registry()}
    specs = [by_id[i] for i in ids]
    steps = bench.steps(suite, variant)
    plans: List[DatasetPlan] = []

    if bench.derived_from:
        source = catalog.load(bench.derived_from)
        base = Path(output_root) if output_root else _paths.output_dir()
        for step in steps:
            found = bool(staging_results(step.dataset, base))
            state = f"{source.name} results found" if found else f"no {source.name} results"
            plans.append(DatasetPlan(bench.name, step.dataset, "hypnogram", ids, [], ["all"],
                                     "-", state, result_dir(bench.name, step.dataset, variant, output_root),
                                     [], [], output_base=base))
            if not found:
                plans[-1].notes += _msg.lines(
                    "warning", f"the hypnograms are built from the {source.name} predictions, "
                               f"which are not in the results folder",
                    f"neuroatlas run {source.name} --dataset {step.dataset} -m {models}")
        return plans

    embeds = {s.dataset: list(s.argv) for s in steps if s.verb == "embed"}
    if folds:
        from neuroatlas.entrypoints.probe import _parse_int_list

        try:
            wanted = _parse_int_list(folds)
        except ValueError:
            raise PlanError(f"--folds wants fold numbers such as 0,1 or 0-4, got {folds!r}\n"
                            f"fix: {_with_folds(folds, '0')}") from None
        problems = [why for why in (_check_folds(s.dataset, list(s.argv), wanted, folds)
                                    for s in steps if s.verb == "probe") if why]
        if problems:
            raise PlanError("\n".join(problems))
    base = Path(output_root) if output_root else _paths.output_dir()
    probe_steps = [s for s in steps if s.verb == "probe"]
    # the command's live line meanwhile (a report's on stderr, a terminal only)
    from neuroatlas import progress

    item = progress.current().phase("planning", total=len(probe_steps), unit="datasets")
    for step in probe_steps:
        item.update(note=step.dataset)
        slug = step.dataset
        cmap, map_error = channel_map(slug)
        skipped, invalid = map_states(cmap, specs)
        runnable = [i for i in ids if i not in skipped and i not in invalid]
        out = result_dir(bench.name, slug, variant, output_root)
        if per_model and len(runnable) == 1:
            out = out / runnable[0]
        probe_argv = list(step.argv)
        if folds:
            probe_argv += ["--folds", folds]
        elif debug:
            probe_argv += ["--folds", "0"]
        fold_list = _folds(slug, probe_argv)
        st = data.status(slug)
        uncounted = [str(f) for f in fold_list if str(f).startswith("? (needs ")]
        seeds = "per fold"
        plans.append(DatasetPlan(bench.name, slug, bench.task_for(slug) or "default",
                                 [] if map_error else runnable,
                                 skipped, fold_list, seeds, st.state, out,
                                 embed_argv(embeds[slug], probe_argv, fold_list),
                                 probe_argv, output_base=base, invalid=list(invalid)))
        if map_error:
            plans[-1].map_error = map_error
            plans[-1].notes += _msg.lines("error", f"{map_error}; not run", REINSTALL)
        if uncounted:
            package = uncounted[0][len("? (needs "):].split(" ", 1)[0]
            plans[-1].notes += _msg.lines("warning", f"the fold count of {slug} needs {package}",
                                          data.MOABB_INSTALL if package == "moabb" else
                                          f"pip install {package}")
        if invalid:
            plans[-1].notes += _msg.lines("warning", invalid_note(
                slug, invalid, models, _run_line(bench.name, models, suite, variant, invalid)))
        if is_sample(st.state):
            # `data download --first N`: enough for `check`, not for a run
            # (its folds would be built from a few recordings)
            plans[-1].data_missing = plans[-1].data_sample = True
            plans[-1].data_fix = f"neuroatlas data download {slug}"
            plans[-1].data_path = str(st.path) if st.path else None
            plans[-1].notes += _msg.lines(
                "warning", f"data {st.state}: `run` needs the whole dataset, and skips this "
                           f"one until it is here", plans[-1].data_fix)
        elif not st.found and st.state != "fetched on first use":
            from neuroatlas.check import data_fix

            fix = data_fix(slug, st)
            missing = st.state.split(" (")[0] in DATA_NOT_HERE
            plans[-1].data_missing = missing
            plans[-1].data_fix = fix
            plans[-1].data_path = str(st.path) if st.path else None
            plans[-1].notes += _msg.lines(
                "warning", f"data {st.state}: "
                           + ("`run` skips this dataset until it is here" if missing
                              else "the run would fail on this dataset"), fix)
        item.update(advance=1)
    return plans


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def _call(verb: str, argv: List[str]) -> int:
    import importlib

    from neuroatlas import progress

    module = importlib.import_module(f"neuroatlas.entrypoints.{verb}")
    print(f"\n$ neuroatlas {verb} {' '.join(argv)}", flush=True)
    try:
        # importing the engine takes seconds before the step's own header:
        # its starting line meanwhile (a terminal only)
        with progress.starting(verb):
            module.main(argv)
    except SystemExit as exc:
        if exc.code in (None, 0):
            return 0
        if isinstance(exc.code, str):
            from neuroatlas.cli import _msg

            _msg.error(exc.code)
            return 1
        return int(exc.code)
    return 0


def read_results(path: Path) -> List[Dict[str, Any]]:
    try:
        return json.loads((path / "results.json").read_text())
    except (OSError, ValueError):
        return []


def _checkpoint_paths(extra: Optional[List[str]]) -> Dict[str, str]:
    """``--checkpoint ID=PATH`` pairs among the extra verb arguments."""
    out: Dict[str, str] = {}
    args = list(extra or [])
    for i, a in enumerate(args):
        if a == "--checkpoint" and i + 1 < len(args) and "=" in args[i + 1]:
            ident, _, path = args[i + 1].partition("=")
            out[ident.strip()] = path.strip()
    return out


def weights_problems(model_ids: List[str], checkpoint_paths: Optional[Dict[str, str]] = None
                     ) -> Dict[str, str]:
    """Models whose weights this run cannot load, with why (and the fix on a
    ``fix:`` line) -- the same test ``check`` makes (``models.status``)
    before it forwards a batch.

    Without it a manual model with no weights (REVE, gated on Hugging Face)
    died deep in transformers (``'NoneType' object has no attribute
    'endswith'``) after the job had already started.
    """
    import dataclasses

    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    by_id = {s.identifier: s for s in checkpoint_registry()}
    out: Dict[str, str] = {}
    for ident in model_ids:
        spec = by_id.get(ident)
        if spec is None:
            continue
        path = (checkpoint_paths or {}).get(ident)
        if path:
            spec = dataclasses.replace(spec, checkpoint_path=path, status="ready")
        problem = weights_problem_for(spec)
        if problem:
            out[ident] = problem
    return out


def weights_problem_for(spec) -> Optional[str]:
    """Why this checkpoint's weights cannot be loaded here, with the fix on a
    ``fix:`` line, or None: ready, or fetched on first use (--online)."""
    from neuroatlas import config
    from neuroatlas import models as model_state

    from neuroatlas.cli import command_with

    fetchable = set() if config.is_offline() else {"hub", "auto"}
    st = model_state.status(spec)
    if st.ready or st.state in fetchable:
        return None
    what, fix = model_state.weights_problem(st) or (f"weights {st.state}", None)
    if st.state in ("auto", "hub"):
        # a second remedy, on its own line: fetch them as the run needs them
        online = command_with("--online")
        if online:
            fix = f"{fix}\nfix: {online} (downloads them as the run needs them)"
    return what + (f"\nfix: {fix}" if fix else "")


def _release_gpu() -> None:
    """Hand back cached GPU memory between the embed and probe steps: the
    probe is CPU-only, and on a shared node another job can use it."""
    import gc

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def limit(plans: List[DatasetPlan], cache_root: Optional[Path] = None) -> Path:
    """A --limit-batches run's folders: the cache root it reads and writes
    (<cache root>/_limited) is returned, and each plan's results move under
    <output root>/_limited. A truncated cache must never be read as a complete
    one, and the results it gives must never land beside real ones (F-061).
    Plans already moved stay where they are, so `run --dry-run` can show the
    folders before execute() applies the same move."""
    from neuroatlas import config

    base = Path(cache_root) if cache_root else (config.get("cache_root") or _paths.artifacts_dir("embedding_cache"))
    for p in plans:
        if p.limited or p.output_base is None or p.task == "hypnogram":
            continue
        try:
            p.output = p.output_base / "_limited" / p.output.relative_to(p.output_base)
            # the results root these results are read from now
            # (`neuroatlas results ... --output-root <it>`)
            p.output_base = p.output_base / "_limited"
        except ValueError:
            p.output = p.output / "_limited"
        p.limited = True
    return Path(base) / "_limited"


def execute(plans: List[DatasetPlan], *, cache_root: Optional[Path] = None,
            limit_batches: Optional[int] = None, skip_embed: bool = False,
            extra_embed: Optional[List[str]] = None,
            extra_probe: Optional[List[str]] = None,
            num_workers: Optional[int] = None, reprobe: bool = False) -> Dict[str, Any]:
    """Run the plans. Returns counts of ok / failed / skipped runs, and of the
    (dataset, checkpoint) pairs the channel map rules out (n/a) or has no
    entry for (invalid).

    ``num_workers``: data-loader workers for the embed step (default: the
    CPUs this job may use, minus one, at most 16; see
    benchmarking_helpers/runtime/resources.py).

    ``reprobe``: fit every fold again (``probe --reprobe``). Without it a fold
    whose saved predictions match is rescored, not fitted; ``reused`` counts
    those (of the ``ok`` ones).
    """
    import contextlib

    from neuroatlas import config
    from neuroatlas.cli import _msg

    # "kept": records already in results.json that this run did not write;
    # "n/a": (dataset, checkpoint) pairs the channel map rules out (never run);
    # "invalid": pairs the channel map has no entry for (never run);
    # "skipped": runs not started because their data or weights are not here;
    # "reused": ok results recomputed from a fold's saved predictions
    summary = {"ok": 0, "failed": 0, "skipped": 0, "n/a": 0, "invalid": 0, "kept": 0,
               "reused": 0, "datasets_failed": [], "datasets_skipped": []}
    ruled_out = set()
    if reprobe:
        extra_probe = [*(extra_probe or []), "--reprobe"]
    if limit_batches is not None:
        cache_root = limit(plans, cache_root)
    cache_args = ["--cache-root", str(cache_root)] if cache_root else []
    worker_args = ["--num-workers", str(int(num_workers))] if num_workers is not None else []
    extra_embed = [*worker_args, *(extra_embed or [])]
    extra_probe = [*worker_args, *(extra_probe or [])]
    ckpt_paths = _checkpoint_paths(extra_embed)

    for p in plans:
        if p.task == "hypnogram":
            status = _run_hypnogram(p)
            summary["ok" if status == 0 else "failed"] += 1
            continue
        if not p.models:
            if p.map_error:
                _msg.error(f"{p.dataset}: not run: {p.map_error}", REINSTALL)
                summary["datasets_failed"].append(p.dataset)
                summary["failed"] += 1
            else:
                _msg.note(f"{p.dataset}: nothing to run: its channel map rules out "
                          f"{', '.join([*p.skipped, *p.invalid])}")
                ruled_out.update((p.dataset, m) for m in p.skipped)
                summary["invalid"] += len(p.invalid)
            continue
        # pairs the channel map rules out, whatever becomes of the others
        ruled_out.update((p.dataset, m) for m in p.skipped)
        summary["invalid"] += len(p.invalid)
        runnable = list(p.models)
        if p.data_sample:
            # a run on a few recordings is not the benchmark: refused, with or
            # without --skip-embed (the cache would hold the sample's embeddings)
            inner = str(p.data)[len("sample ("):-1] if str(p.data).endswith(")") else ""
            _msg.error(f"{p.dataset}: not run: what is here is a sample"
                       + (f" ({inner})" if inner else "")
                       + ", from `data download --first N`; `run` needs the whole dataset",
                       p.data_fix)
            summary["skipped"] += len(runnable) * max(1, len(p.folds))
            summary["datasets_skipped"].append(p.dataset)
            continue
        if p.data_missing and not skip_embed and not (str(p.data).startswith("not downloaded")
                                                      and not config.is_offline()):
            # Nothing to read: say so once, with the fix, instead of starting
            # an embed step that fails on the missing files. (A MOABB dataset
            # not downloaded yet is fetched by the run itself under --online;
            # --skip-embed probes what the cache holds, as asked.)
            where = f" (looked in {p.data_path})" if p.data_path else ""
            _msg.error(f"{p.dataset}: not run: its data is {p.data}{where}", p.data_fix)
            summary["skipped"] += len(runnable) * max(1, len(p.folds))
            summary["datasets_skipped"].append(p.dataset)
            continue
        if not skip_embed:
            # The embed step loads every model's weights; a model whose
            # weights are not here is skipped now, by name, not inside the job.
            blocked = weights_problems(runnable, ckpt_paths)
            for ident, why in blocked.items():
                _msg.error(f"{p.dataset}/{ident}: not run: {why}")
                summary["skipped"] += max(1, len(p.folds))
            runnable = [m for m in runnable if m not in blocked]
            if not runnable:
                continue
        models = ["--models", ",".join(runnable)]
        n_runs = len(runnable) * max(1, len(p.folds))
        ctx = contextlib.nullcontext()
        if limit_batches is not None:
            from neuroatlas.extensions.tasks.linear_probe import limit_batches as _limit

            ctx = _limit(limit_batches)
        with ctx:
            if not skip_embed:
                rc = _call("embed", [*p.embed_argv, *models, *cache_args, *(extra_embed or [])])
                _release_gpu()
                if rc:
                    summary["datasets_failed"].append(p.dataset)
                    summary["failed"] += n_runs
                    continue
            p.output.mkdir(parents=True, exist_ok=True)
            prior = {_row_id(r) for r in read_results(p.output)}
            rc = _call("probe", [*p.probe_argv, *models, *cache_args,
                                 "--output-root", str(p.output), *(extra_probe or [])])
        # Count what this run did, not what the file holds: results.json keeps
        # earlier folds and models, so a 1-fold --debug re-run must not
        # report the four folds an earlier run wrote, and a row this run left
        # untouched (an earlier attempt's) is not this run's either. A failure
        # recorded before any fold has no fold, and is this run's all the same.
        # (Rows, not the file's mtime: on NFS a quick rewrite can keep it.)
        on_disk = read_results(p.output)
        rows = on_disk
        wanted = set(runnable)
        planned = {str(f) for f in p.folds}
        by_fold = not any(str(f).startswith(("?", "-")) for f in p.folds)

        def _this_run(r: Dict[str, Any]) -> bool:
            fold = (r.get("metadata") or {}).get("fold")
            return (r.get("checkpoint_id") in wanted and _row_id(r) not in prior
                    and (not by_fold or fold is None or str(fold) in planned))

        mine = [r for r in rows if _this_run(r)]
        summary["kept"] += len(on_disk) - len(mine)
        invalid_here = set(p.invalid)
        for r in mine:
            word = outcome(r)
            if word == "ok":
                summary["ok"] += 1
                if (r.get("metadata") or {}).get("reused_predictions"):
                    summary["reused"] += 1
            elif word == "ruled out":
                ruled_out.add((p.dataset, r.get("checkpoint_id")))
            elif word == "invalid":
                # a pair, counted once (the plan may have counted it already)
                if r.get("checkpoint_id") not in invalid_here:
                    invalid_here.add(r.get("checkpoint_id"))
                    summary["invalid"] += 1
            elif word == "skipped":
                summary["skipped"] += 1                 # data or weights not here
            else:
                summary["failed"] += 1
        if rc and not mine:
            summary["datasets_failed"].append(p.dataset)
            summary["failed"] += n_runs
    summary["n/a"] = len(ruled_out)
    return summary


def outcome(row: Dict[str, Any]) -> str:
    """``ok``, ``ruled out``, ``invalid``, ``skipped`` or ``failed``: what a
    result row of results.json counts as in a run's closing line, in the
    runner's own words (:func:`runner.outcome_word`: data_missing and
    weights_missing are skipped, channel_map_invalid invalid)."""
    from types import SimpleNamespace

    from neuroatlas.benchmarking_helpers.runtime.runner import outcome_word

    code = (row.get("failure") or {}).get("code")
    return outcome_word(SimpleNamespace(ok=bool(row.get("ok")),
                                        failure=SimpleNamespace(code=code)))


def _row_id(row: Dict[str, Any]) -> str:
    """A result row's identity for telling this run's rows from the ones
    already in results.json: the whole record, so a re-run's row counts as
    this run's even when it lands on the same key."""
    return json.dumps(row, sort_keys=True, default=str)


def _run_hypnogram(p: DatasetPlan) -> int:
    """Hypnograms from the staging probes, then their features, into the
    benchmark's own folder, with a results.json the results command reads."""
    import numpy as np

    from neuroatlas import catalog
    from neuroatlas.entrypoints import hypnogram as hyp

    source = catalog.load(catalog.load(p.benchmark).derived_from)
    root = p.output_base if p.output_base is not None else p.output.parents[1]
    # `run` writes <root>/sleep_stage/<ds>/results.json, `submit` one folder
    # per model under it; both are read (it used to look at the first only).
    sources = staging_results(p.dataset, root)
    if not sources:
        from neuroatlas.cli import _msg

        where = ", ".join(str(c) for c in hyp.candidate_results_dirs(p.dataset, root))
        _msg.error(f"{p.dataset}: no {source.name} results in {where}",
                   f"neuroatlas run {source.name} --dataset {p.dataset} -m "
                   f"{','.join(p.models) if p.models else 'all'}")
        return 1
    p.output.mkdir(parents=True, exist_ok=True)
    hypno = p.output / "hypnograms.json"
    for i, src in enumerate(sources):
        print(f"\n$ neuroatlas hypnogram reconstruct --results-dir {src} --output {hypno}",
              flush=True)
        # The first folder starts the file afresh; the others add to it.
        hyp.reconstruct_results_dir(results_dir=src, output=hypno, models=p.models or None,
                                    compute_metrics=True, merge=i > 0)
    rows = hyp.rows_from_hypnograms(hypno, p.dataset)
    if not rows:
        from neuroatlas.cli import _msg

        _msg.error(f"{p.dataset}: no hypnogram could be rebuilt from the {source.name} "
                   f"results: the warnings above name the checkpoints whose saved probe or "
                   f"embeddings are missing",
                   f"neuroatlas run {source.name} --dataset {p.dataset} -m "
                   f"{','.join(p.models) if p.models else 'all'} --reprobe")
        return 1
    hyp.write_csv(rows, p.output / "hypnogram_features.csv")
    summary = hyp.compute_summary(rows)
    hyp.write_summary_csv(summary, p.output / "hypnogram_features_summary.csv")
    results = []
    for model in sorted({r["model"] for r in summary}):
        feats = [r for r in summary if r["model"] == model]
        rs = [r["pearson_r"] for r in feats if r["pearson_r"] == r["pearson_r"]]
        results.append({
            "checkpoint_id": model, "dataset_name": p.dataset, "evaluation_mode": "hypnogram",
            "ok": bool(rs),
            "metrics": {"pearson_r": float(np.mean(rs)) if rs else float("nan"),
                        "per_feature": {r["feature"]: {k: r[k] for k in ("mae", "bias", "rmse", "pearson_r", "n_valid")}
                                        for r in feats}},
            "metadata": {"fold": "all", "task_name": "hypnogram"},
        })
    (p.output / "results.json").write_text(json.dumps(results, indent=2, default=float))
    return 0
