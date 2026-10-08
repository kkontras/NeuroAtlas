"""Push one real batch through each (dataset, model) pair, then stop.

The dataset config is built by ``embed``'s own ``build_config`` from the
benchmark's embed arguments, and the datamodule and backbone by the runner's
``_prepare_pair`` -- so a pair that passes here fails later only for reasons a
single batch cannot show. Nothing is trained, cached or downloaded.
"""
from __future__ import annotations

import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PairCheck:
    dataset: str
    model: str
    data: str
    weights: str
    channel_map: str
    forward: str = "-"
    notes: List[str] = field(default_factory=list)
    error: bool = False
    seconds: Optional[float] = None      # wall time of this pair's check
    #: The whole error message of a failed forward pass (the JSON ``error``
    #: field); the table note carries its first sentence.
    message: Optional[str] = None


def _error_note(exc: BaseException) -> List[str]:
    """The lines under a failed pair: ``error:`` and the message's first
    sentence (whole with -v); `--format json` carries the whole message."""
    from neuroatlas.cli import _msg

    return _msg.lines("error", _msg.brief(_msg.exception_text(exc)))


def data_fix(slug: str, st) -> str:
    """The commands that bring a dataset here: download it (or ask
    `data download` how to get it), or point at a copy elsewhere -- each on its
    own ``fix:`` line (pass it as the *fix* of a message)."""
    from neuroatlas import data
    from neuroatlas.cli import _msg

    setting = getattr(st, "setting", None)
    what = "FILE" if setting and setting.endswith("_path") else "DIR"
    try:
        acq = data.acquisition(slug)
    except KeyError:                    # a dataset without a manifest: nothing more to say
        acq = {}
    if acq.get("prepared_only"):
        # the authors' preprocessed file: the manifest says what to ask for
        _, _, fixes = _msg.split(data.manifest_text(acq["prepared_only"]))
        if fixes:
            return f"\n{_msg.FIX}: ".join(fixes)
    handler = getattr(st, "handler", None)
    elsewhere = f"neuroatlas config set {setting} {what}" if setting else None
    if handler == "refused":
        # `data download` refuses it (what the host serves is not what the
        # benchmark reads): only a copy from elsewhere helps
        where = f"neuroatlas data status {slug} says what to get"
        return f"{elsewhere} ({where})" if elsewhere else f"neuroatlas data status {slug}"
    first = {
        "internal": f"neuroatlas data download {slug} (says what to ask the authors for)",
        "manual": f"neuroatlas data download {slug} (says where to get it)",
    }.get(handler, f"neuroatlas data download {slug}")
    fixes = [first]
    if handler == "nsrr" and data.NO_NSRR_TOKEN in getattr(st, "notes", []):
        fixes.insert(0, "neuroatlas config token nsrr (first)")
    if elsewhere:
        fixes.append(f"{elsewhere} ("
                     + ("where you put it" if handler in ("internal", "manual")
                        else "a copy elsewhere") + ")")
    return f"\n{_msg.FIX}: ".join(fixes)


#: What the `channel_map` column of `check` says for a map's pass-through
#: entry (``state_for``'s detail ``pass-through``).
_DETAIL_WORDS = {"pass-through": "channels as recorded"}


def _dataset_config(slug: str, embed_argv, model: str, num_workers: Optional[int] = None):
    from neuroatlas.entrypoints import embed

    # One batch is read: loader workers would only add start-up time (each
    # forks the process), so check reads in-process unless told otherwise.
    argv = [*embed_argv, "--models", model, "--num-workers", str(num_workers or 0)]
    return embed.build_config(embed.build_parser(argv).parse_args(argv))


def _first_batch(datamodule):
    """A batch from the loader extraction reads -- in order.

    Not ``train_dataloader``: the epilepsy readers draw train windows through
    a weighted random sampler, so its first batch decoded ~50 whole
    recordings (CHB-MIT: ~110 s per pair) where an in-order batch decodes
    one.
    """
    if datamodule.supports_global_embedding_cache():
        loader = datamodule.full_embedding_dataloader()
    else:
        loader = None
        for name in ("val_dataloader", "test_dataloader"):
            try:
                candidate = getattr(datamodule, name)()
                if len(candidate):
                    loader = candidate
                    break
            except Exception:
                continue
        if loader is None:
            loader = datamodule.train_dataloader()
    return next(iter(loader))


def _verbose_check(benchmark: Optional[str], ident: str, slug: str) -> str:
    """The `check` of this one pair with -v: the model's own warnings."""
    return (f"neuroatlas -v check {benchmark or '<benchmark>'} -m {ident} --dataset {slug} "
            f"(the model's own warnings)")


def check_pair(slug: str, spec, embed_argv, data_status, model_status,
               num_workers: Optional[int] = None, *, benchmark: Optional[str] = None,
               models: Optional[str] = None) -> PairCheck:
    import numpy as np

    from neuroatlas import models as model_weights
    from neuroatlas import progress
    from neuroatlas.benchmarking_helpers.channels.channel_map import load_channel_map
    from neuroatlas.cli import _msg
    from neuroatlas.benchmarking_helpers.runtime.runner import BenchmarkRunner

    pc = PairCheck(slug, spec.identifier, data_status.state,
                   model_weights.state_word(model_status.state), "none")
    try:
        cmap = load_channel_map(slug)
    except (ValueError, KeyError) as exc:
        from neuroatlas.run import REINSTALL

        pc.channel_map, pc.error = "unreadable", True
        pc.notes += _msg.lines("error", f"the {slug} channel map shipped with the package does "
                                        f"not load: "
                               + _msg.brief(str(exc).split(": ", 1)[-1]), REINSTALL)
        return pc
    if cmap is not None:
        # The pair's map state, decided before any data is read (values:
        # channel_map.CHANNEL_MAP_STATES). A family the map has no entry for
        # is `invalid` here rather than `applied` and a failure on the batch.
        state, detail = cmap.state_for(spec.model_family)
        if state == "skip":
            pc.channel_map = _msg.RULED_OUT
            # the same sentence as `run --dry-run`: which map, then its reason
            reason = _msg.brief(detail) if detail else ""
            reason = reason[:1].lower() + reason[1:]
            pc.notes += _msg.lines(_msg.RULED_OUT, f"by the {slug} channel map"
                                   + (f" ({reason.rstrip('.')})" if reason else ""))
            return pc
        if state == "invalid":
            # no entry for the family: not checked, and not a failure
            from neuroatlas.run import family_text, invalid_fix

            pc.channel_map = "invalid"
            fix = invalid_fix(models or spec.identifier, [spec.model_family],
                              f"neuroatlas check {benchmark or '<benchmark>'} -m "
                              f"{models or spec.identifier},-{spec.model_family} "
                              f"--dataset {slug}")
            pc.notes += _msg.lines("invalid", f"not checked (the {slug} channel map has no "
                                              f"entry for {family_text([spec.model_family])})",
                                   fix)
            return pc
        detail = _DETAIL_WORDS.get(detail, detail)
        pc.channel_map = f"applied ({detail})" if detail else "applied"

    # a sample (`data download --first N`) is what check is for: it is read
    sample = str(data_status.state).startswith("sample")
    if not data_status.found and data_status.state != "fetched on first use" and not sample:
        pc.notes += _msg.lines(_msg.SKIPPED, f"data {data_status.state}"
                               + (f" ({data_status.path})" if data_status.path else ""),
                               data_fix(slug, data_status))
        return pc
    problem = model_weights.weights_problem(model_status)
    if problem:
        pc.notes += _msg.lines(_msg.SKIPPED, *problem)
        return pc

    try:
        config = _dataset_config(slug, embed_argv, spec.identifier, num_workers)
        with tempfile.TemporaryDirectory(prefix="neuroatlas-check-") as tmp:
            config["benchmark"]["cache_root"] = tmp
            config["benchmark"]["output_root"] = tmp
            runner = BenchmarkRunner(config)
            dataset_name, run_config = runner._dataset_runs()[0]
            spec = runner._apply_spec_overrides(spec)
            run_config = runner._pair_config(dict(run_config), cmap, spec)
            run_config.pop("channel_map_name", None)
            datamodule, backbone = runner._prepare_pair(dataset_name, run_config, spec, cmap)
            # the pair's live line: loading the data, loading weights, then
            item = progress.current()
            item.phase("reading one batch")
            batch = _first_batch(datamodule)
            item.phase("running one batch through the model")
            emb = np.asarray(backbone.extract_embeddings(batch))
        n_bad = int((~np.isfinite(emb)).sum())
        shape = "(" + ", ".join(map(str, emb.shape)) + ")"
        if n_bad:
            pc.forward, pc.error = f"{shape} {n_bad} non-finite", True
            pc.notes += _msg.lines("error", f"{n_bad} NaN or infinite values in one batch of "
                                            f"{slug} embeddings from {spec.identifier}",
                                   _verbose_check(benchmark, spec.identifier, slug))
        elif emb.ndim >= 2 and emb.shape[0] > 1 and np.allclose(emb, emb[:1]):
            pc.forward = f"{shape} constant"
            pc.notes += _msg.lines("error", f"every window of the batch gave the same "
                                            f"embedding: {spec.identifier} received a "
                                            f"constant input from {slug}",
                                   _verbose_check(benchmark, spec.identifier, slug))
            pc.error = True
        else:
            pc.forward = f"{shape} finite"
    except StopIteration:
        pc.forward, pc.error = "no batch", True
        pc.notes += _msg.lines("error", f"no window to read: every window of {slug} was "
                                        f"filtered out",
                               f"neuroatlas data status {slug}")
    except Exception as exc:
        pc.forward, pc.error = "failed", True
        pc.notes += _error_note(exc)
        pc.message = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    if pc.error and sample:
        pc.notes += _sample_note(slug, data_status)
    return pc


def _sample_note(slug: str, data_status) -> List[str]:
    """Under a pair that failed on a sample: what the sample is, and a larger one."""
    from neuroatlas.cli import _msg

    match = re.search(r"first (\d+)", str(data_status.state))
    n = int(match.group(1)) if match else 0
    return _msg.lines("note", f"{slug} here is its first {_msg.plural(n, 'recording')} "
                              f"only; fold 0 needs a train, a validation and a test subject "
                              f"among them",
                      f"neuroatlas data download {slug} --first {max(20, 2 * n)}")


def check(benchmark: str, models: str, suite: str = "single",
          variant: str = "default", num_workers: Optional[int] = None) -> List[PairCheck]:
    """``num_workers``: loader workers for the one batch (default 0: read in
    this process, which is fastest for a single batch)."""
    from neuroatlas import catalog, data
    from neuroatlas import models as model_state
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    from neuroatlas import progress

    # the command's live line (stderr, a terminal only) meanwhile
    progress.current().phase("reading the checkpoint list")
    bench = catalog.load(benchmark)
    if bench.derived_from:
        bench = catalog.load(bench.derived_from)
    by_id = {s.identifier: s for s in checkpoint_registry()}
    specs = [by_id[i] for i in bench.select_models(models)]
    import sys

    from neuroatlas.benchmarking_helpers.runtime.runner import _Ticker

    out: List[PairCheck] = []
    steps = [st for st in bench.steps(suite, variant) if st.verb == "embed"]
    total = len(steps) * len(specs)
    for step in steps:
        progress.current().phase(f"checking where {step.dataset} is")
        ds = data.status(step.dataset)
        for spec in specs:
            started = time.monotonic()
            # A live "checking k/N" line on a terminal (stderr, cleared before
            # the table) through the pair's phases -- loading the data,
            # loading weights, one batch -- so a minute of loading models
            # does not look stuck.
            with _Ticker(f"checking {len(out) + 1}/{total}: {step.dataset} {spec.identifier}",
                         verb="starting", stream=sys.stderr):
                pc = check_pair(step.dataset, spec, list(step.argv), ds,
                                model_state.status(spec), num_workers=num_workers,
                                benchmark=bench.name, models=models)
            pc.seconds = time.monotonic() - started
            out.append(pc)
    return out
