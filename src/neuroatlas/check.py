"""Push one real batch through each (dataset, model) pair, then stop.

The dataset config is built by ``embed``'s own ``build_config`` from the
benchmark's embed arguments, and the datamodule and backbone by the runner's
``_prepare_pair`` -- so a pair that passes here fails later only for reasons a
single batch cannot show. Nothing is trained, cached or downloaded.
"""
from __future__ import annotations

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
    """The commands that bring a dataset here: download it, or point at a copy."""
    fix = f"neuroatlas data download {slug}"
    setting = getattr(st, "setting", None)
    return fix + (f", or neuroatlas config set {setting} DIR" if setting else "")


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


def check_pair(slug: str, spec, embed_argv, data_status, model_status,
               num_workers: Optional[int] = None) -> PairCheck:
    import numpy as np

    from neuroatlas import models as model_weights
    from neuroatlas.benchmarking_helpers.channels.channel_map import load_channel_map
    from neuroatlas.cli import _msg
    from neuroatlas.benchmarking_helpers.runtime.runner import BenchmarkRunner

    pc = PairCheck(slug, spec.identifier, data_status.state, model_status.state, "none")
    try:
        cmap = load_channel_map(slug)
    except (ValueError, KeyError) as exc:
        pc.channel_map, pc.error = "invalid", True
        pc.notes += _msg.lines("error", "channel map: " + _msg.brief(str(exc).split(": ", 1)[-1]))
        return pc
    if cmap is not None:
        # The pair's map state, decided before any data is read (values:
        # channel_map.CHANNEL_MAP_STATES). A family the map has no entry for
        # is `invalid` here rather than `applied` and a failure on the batch.
        state, detail = cmap.state_for(spec.model_family)
        if state == "skip":
            pc.channel_map = "n/a (skip)"
            pc.notes += _msg.lines("n/a", _msg.brief(detail) if detail
                                   else "the channel map skips this model")
            return pc
        if state == "invalid":
            pc.channel_map, pc.error = "invalid", True
            pc.notes += _msg.lines("error", detail)
            return pc
        pc.channel_map = f"applied ({detail})" if detail else "applied"

    if not data_status.found and data_status.state != "fetched on first use":
        pc.notes += _msg.lines("skipped", f"data {data_status.state}"
                               + (f" ({data_status.path})" if data_status.path else ""),
                               data_fix(slug, data_status))
        return pc
    problem = model_weights.weights_problem(model_status)
    if problem:
        pc.notes += _msg.lines("skipped", *problem)
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
            batch = _first_batch(datamodule)
            emb = np.asarray(backbone.extract_embeddings(batch))
        n_bad = int((~np.isfinite(emb)).sum())
        shape = "(" + ", ".join(map(str, emb.shape)) + ")"
        if n_bad:
            pc.forward, pc.error = f"{shape} {n_bad} non-finite", True
            pc.notes += _msg.lines("error", f"{n_bad} non-finite values in the embeddings")
        elif emb.ndim >= 2 and emb.shape[0] > 1 and np.allclose(emb, emb[:1]):
            pc.forward = f"{shape} constant"
            pc.notes += _msg.lines("error", "every window embedded identically: "
                                            "the input may be empty or flat")
            pc.error = True
        else:
            pc.forward = f"{shape} finite"
    except StopIteration:
        pc.forward, pc.error = "no batch", True
        pc.notes += _msg.lines("error", "the loader yielded nothing: no windows survived "
                                        "the dataset's filters")
    except Exception as exc:
        pc.forward, pc.error = "error", True
        pc.notes += _error_note(exc)
        pc.message = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    return pc


def check(benchmark: str, models: str, suite: str = "single",
          variant: str = "default", num_workers: Optional[int] = None) -> List[PairCheck]:
    """``num_workers``: loader workers for the one batch (default 0: read in
    this process, which is fastest for a single batch)."""
    from neuroatlas import catalog, data
    from neuroatlas import models as model_state
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

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
        ds = data.status(step.dataset)
        for spec in specs:
            started = time.monotonic()
            # A live "checking k/N" line on a terminal (stderr, cleared before
            # the table), so a minute of loading models does not look stuck.
            with _Ticker(f"checking {len(out) + 1}/{total}: {step.dataset} {spec.identifier}",
                         verb="loading and running one batch", stream=sys.stderr,
                         start_line_off_tty=False):
                pc = check_pair(step.dataset, spec, list(step.argv), ds,
                                model_state.status(spec), num_workers=num_workers)
            pc.seconds = time.monotonic() - started
            out.append(pc)
    return out
