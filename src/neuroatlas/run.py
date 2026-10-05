"""Run a benchmark: plan it, then execute its steps -- ``embed`` then
``probe`` per dataset -- in this process.

Every step is the catalog's verb invocation (the paper's command line) plus
the few arguments a run adds: the model selection, where results go, and,
for a quick try, fold 0 only. The verbs do the work exactly as they do when
called by hand.

Results land in ``<output_root>/<benchmark>/<dataset>/`` (a variant other
than the default adds ``/<variant>``): the runner's results.json,
results.csv and summary.md, which ``neuroatlas results`` reads.
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

    @property
    def n_runs(self) -> int:
        return len(self.models) * max(1, len(self.folds))


def result_dir(benchmark: str, dataset: str, variant: str = "default",
               output_root: Optional[Path] = None) -> Path:
    base = Path(output_root) if output_root else _paths.output_dir()
    out = base / benchmark / dataset
    return out if variant == "default" else out / variant


def channel_map(dataset: str):
    """(map or None, error message or None). A map that fails validation is
    reported against its dataset instead of stopping a whole suite."""
    from neuroatlas.benchmarking_helpers.channels.channel_map import load_channel_map

    try:
        return load_channel_map(dataset), None
    except (ValueError, KeyError) as exc:
        return None, f"channel map invalid: {str(exc).split(': ', 1)[-1]}"


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


def invalid_note(dataset: str, invalid: Dict[str, str]) -> str:
    """The line a plan carries for its invalid pairs: which, why, the fix."""
    from neuroatlas.benchmarking_helpers.channels.channel_map import _default_yaml_path

    families = sorted(set(invalid.values()))
    return (f"invalid, not run: {', '.join(invalid)} -- the channel map has no entry for "
            f"{'family' if len(families) == 1 else 'families'} {', '.join(families)}; add a "
            f"mapping, `mode: label_pass_through` or `skip` to "
            f"configs/channel_maps/{_default_yaml_path(dataset).name}")


class PlanError(CatalogError):
    """A run that cannot be planned as asked, e.g. a fold the dataset does not
    have. A usage error: the command layer prints a CatalogError as
    ``error: ...`` and exits 2, and this is one."""


def _check_folds(dataset: str, step_argv: List[str], wanted: List[Any]) -> Optional[str]:
    """Why *wanted* is not a set of this dataset's folds, or None if it is."""
    available = _folds(dataset, list(step_argv))
    if not available or not all(isinstance(f, int) for f in available):
        return None                              # unknown count, or a single split
    bad = [f for f in wanted if f not in available]
    if not bad:
        return None
    lo, hi = min(available), max(available)
    span = f"{lo}-{hi}" if available == list(range(lo, hi + 1)) else ", ".join(map(str, available))
    return (f"{dataset}: fold {', '.join(map(str, bad))} does not exist; "
            f"it has {len(available)} fold(s): {span}")


def _folds(dataset: str, probe_argv: List[str]) -> List[Any]:
    from neuroatlas.entrypoints import probe

    try:
        cfg = probe.build_config(probe.build_parser(probe_argv).parse_args(probe_argv))
        block = cfg["datasets"][dataset]
        return list(block.get("folds") or [block.get("fold", "-")])
    except SystemExit:
        return ["?"]
    except Exception as exc:                     # e.g. LOSO needs moabb to count subjects
        return [f"? ({type(exc).__name__})"]


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


def plan(benchmark: str, models: str, suite: str = "single", variant: str = "default", *,
         debug: bool = False, output_root: Optional[Path] = None,
         folds: Optional[str] = None, per_model: bool = False) -> List[DatasetPlan]:
    """``per_model`` puts each model's results in <dataset>/<model>/: cluster
    jobs, one per (dataset, model), never write the same results.json."""
    from neuroatlas import catalog, data, selectors
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    bench = catalog.load(benchmark)
    ids = selectors.resolve_models(models)
    by_id = {s.identifier: s for s in checkpoint_registry()}
    specs = [by_id[i] for i in ids]
    steps = bench.steps(suite, variant)
    plans: List[DatasetPlan] = []

    if bench.derived_from:
        source = catalog.load(bench.derived_from)
        base = Path(output_root) if output_root else _paths.output_dir()
        for step in steps:
            state = "staging results found" if staging_results(step.dataset, base) else \
                f"needs `neuroatlas run {source.name} --dataset {step.dataset}` first"
            plans.append(DatasetPlan(bench.name, step.dataset, "hypnogram", ids, [], ["all"],
                                     "-", state, result_dir(bench.name, step.dataset, variant, output_root),
                                     [], [], output_base=base))
        return plans

    embeds = {s.dataset: list(s.argv) for s in steps if s.verb == "embed"}
    if folds:
        from neuroatlas.entrypoints.probe import _parse_int_list

        try:
            wanted = _parse_int_list(folds)
        except ValueError:
            raise PlanError(f"--folds wants fold numbers such as 0,1 or 0-4, got {folds!r}") from None
        problems = [why for why in (_check_folds(s.dataset, list(s.argv), wanted)
                                    for s in steps if s.verb == "probe") if why]
        if problems:
            raise PlanError("; ".join(problems))
    base = Path(output_root) if output_root else _paths.output_dir()
    for step in (s for s in steps if s.verb == "probe"):
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
        seeds = "per fold"
        plans.append(DatasetPlan(bench.name, slug, bench.task_for(slug) or "default",
                                 [] if map_error else runnable,
                                 skipped, fold_list, seeds, st.state, out,
                                 embed_argv(embeds[slug], probe_argv, fold_list),
                                 probe_argv, output_base=base, invalid=list(invalid)))
        if map_error:
            plans[-1].notes.append(f"{map_error}; not run")
        if invalid:
            plans[-1].notes.append(invalid_note(slug, invalid))
        if not st.found and st.state != "fetched on first use":
            plans[-1].notes.append(f"data {st.state}: `neuroatlas data status {slug}`")
    return plans


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def _call(verb: str, argv: List[str]) -> int:
    import importlib

    module = importlib.import_module(f"neuroatlas.entrypoints.{verb}")
    print(f"\n$ neuroatlas {verb} {' '.join(argv)}", flush=True)
    try:
        module.main(argv)
    except SystemExit as exc:
        if exc.code in (None, 0):
            return 0
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
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
    """Models whose weights this run cannot load, with why -- the same test
    ``check`` makes (``models.status``) before it forwards a batch.

    Without it a manual model with no weights (REVE, gated on Hugging Face)
    died deep in transformers (``'NoneType' object has no attribute
    'endswith'``) after the job had already started.
    """
    import dataclasses

    from neuroatlas import config
    from neuroatlas import models as model_state
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    by_id = {s.identifier: s for s in checkpoint_registry()}
    fetchable = set() if config.is_offline() else {"hub", "auto"}
    out: Dict[str, str] = {}
    for ident in model_ids:
        spec = by_id.get(ident)
        if spec is None:
            continue
        path = (checkpoint_paths or {}).get(ident)
        if path:
            spec = dataclasses.replace(spec, checkpoint_path=path, status="ready")
        st = model_state.status(spec)
        if st.ready or st.state in fetchable:
            continue
        why = f"weights {st.state}"
        if st.notes:
            why += f" ({'; '.join(st.notes)})"
        if st.state in ("auto", "hub"):
            why += f"; `neuroatlas models download {ident}`, or pass --online"
        elif st.state == "manual":
            why += f"; see `neuroatlas models status {ident}`"
        out[ident] = why
    return out


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


def execute(plans: List[DatasetPlan], *, cache_root: Optional[Path] = None,
            limit_batches: Optional[int] = None, skip_embed: bool = False,
            extra_embed: Optional[List[str]] = None,
            extra_probe: Optional[List[str]] = None,
            num_workers: Optional[int] = None) -> Dict[str, Any]:
    """Run the plans. Returns counts of ok / failed / n/a results.

    ``num_workers``: data-loader workers for the embed step (default: the
    CPUs this job may use, minus one, at most 16; see
    benchmarking_helpers/runtime/resources.py).
    """
    import contextlib

    from neuroatlas import config

    # "kept": records already in results.json that this run did not write;
    # "invalid": pairs the channel map has no entry for (never run)
    summary = {"ok": 0, "failed": 0, "n/a": 0, "invalid": 0, "kept": 0, "datasets_failed": []}
    if limit_batches is not None:
        # A truncated cache must never be read as a complete one, and the
        # results it gives must never land beside real ones (F-061).
        base = Path(cache_root) if cache_root else (config.get("cache_root") or _paths.artifacts_dir("embedding_cache"))
        cache_root = Path(base) / "_limited"
        for p in plans:
            if p.output_base is not None and p.task != "hypnogram":
                try:
                    p.output = p.output_base / "_limited" / p.output.relative_to(p.output_base)
                    # the results root these results are read from now
                    # (`neuroatlas results ... --output-root <it>`)
                    p.output_base = p.output_base / "_limited"
                except ValueError:
                    p.output = p.output / "_limited"
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
            reason = "; ".join(p.notes) or f"channel map skips {', '.join(p.skipped)}"
            print(f"\n{p.dataset}: nothing to run ({reason})")
            if any(n.startswith("channel map invalid") for n in p.notes):
                summary["datasets_failed"].append(p.dataset)
                summary["failed"] += 1
            else:
                summary["n/a"] += len(p.skipped)
                summary["invalid"] += len(p.invalid)
            continue
        # pairs the channel map rules out, whatever becomes of the others
        summary["n/a"] += len(p.skipped)
        summary["invalid"] += len(p.invalid)
        runnable = list(p.models)
        if not skip_embed:
            # The embed step loads every model's weights; a model whose
            # weights are not here fails now, by name, not inside the job.
            blocked = weights_problems(runnable, ckpt_paths)
            for ident, why in blocked.items():
                print(f"\n{p.dataset}/{ident}: not run: {why}", file=sys.stderr)
                summary["failed"] += max(1, len(p.folds))
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
        for r in mine:
            code = (r.get("failure") or {}).get("code")
            if r.get("ok"):
                summary["ok"] += 1
            elif code == "channel_map_skip":
                summary["n/a"] += 1
            else:
                summary["failed"] += 1
        if rc and not mine:
            summary["datasets_failed"].append(p.dataset)
            summary["failed"] += n_runs
    return summary


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
        where = ", ".join(str(c) for c in hyp.candidate_results_dirs(p.dataset, root))
        print(f"\n{p.dataset}: no staging results in {where}; run `neuroatlas run {source.name} "
              f"--dataset {p.dataset}` first", file=sys.stderr)
        return 1
    p.output.mkdir(parents=True, exist_ok=True)
    hypno = p.output / "hypnograms.json"
    for i, src in enumerate(sources):
        print(f"\n$ hypnogram reconstruct --results-dir {src} --output {hypno}", flush=True)
        # The first folder starts the file afresh; the others add to it.
        hyp.reconstruct_results_dir(results_dir=src, output=hypno, models=p.models or None,
                                    compute_metrics=True, merge=i > 0)
    rows = hyp.rows_from_hypnograms(hypno, p.dataset)
    if not rows:
        print(f"{p.dataset}: no hypnograms reconstructed", file=sys.stderr)
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
