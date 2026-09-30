"""Push one real batch through each (dataset, model) pair, then stop.

The dataset config is built by ``embed``'s own ``build_config`` from the
benchmark's embed arguments, and the datamodule and backbone by the runner's
``_prepare_pair`` -- so a pair that passes here fails later only for reasons a
single batch cannot show. Nothing is trained, cached or downloaded.
"""
from __future__ import annotations

import tempfile
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


def _dataset_config(slug: str, embed_argv, model: str):
    from neuroatlas.entrypoints import embed

    argv = [*embed_argv, "--models", model]
    return embed.build_config(embed.build_parser(argv).parse_args(argv))


def _first_batch(datamodule):
    if datamodule.supports_global_embedding_cache():
        loader = datamodule.full_embedding_dataloader()
    else:
        loader = datamodule.train_dataloader()
    return next(iter(loader))


def check_pair(slug: str, spec, embed_argv, data_status, model_status) -> PairCheck:
    import numpy as np

    from neuroatlas.benchmarking_helpers.channels.channel_map import load_channel_map
    from neuroatlas.benchmarking_helpers.runtime.runner import BenchmarkRunner

    pc = PairCheck(slug, spec.identifier, data_status.state, model_status.state, "none")
    try:
        cmap = load_channel_map(slug)
    except (ValueError, KeyError) as exc:
        pc.channel_map, pc.error = "invalid", True
        pc.notes.append(str(exc).split(": ", 1)[-1])
        return pc
    if cmap is not None:
        if cmap.is_skip(spec.model_family):
            pc.channel_map = "n/a (skip)"
            note = cmap.notes.get(spec.model_family)
            pc.notes.append(f"not applicable: {note}" if note else "not applicable to this dataset")
            return pc
        pc.channel_map = "applied"

    if not data_status.found and data_status.state != "fetched on first use":
        pc.notes.append(f"data {data_status.state}: forward skipped"
                        + (f" ({data_status.path})" if data_status.path else ""))
        return pc
    load_weights = model_status.ready
    if not load_weights:
        pc.notes.append(f"weights {model_status.state}: forward skipped"
                        + (f" ({'; '.join(model_status.notes)})" if model_status.notes else "")
                        + (f"; `neuroatlas models download {spec.identifier}`"
                           if model_status.state in ("auto", "hub") else ""))
        return pc

    try:
        config = _dataset_config(slug, embed_argv, spec.identifier)
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
        elif emb.ndim >= 2 and emb.shape[0] > 1 and np.allclose(emb, emb[:1]):
            pc.forward = f"{shape} constant"
            pc.notes.append("every window embedded identically: the input may be empty or flat")
            pc.error = True
        else:
            pc.forward = f"{shape} finite"
    except StopIteration:
        pc.forward, pc.error = "no batch", True
        pc.notes.append("the loader yielded nothing: no windows survived the dataset's filters")
    except Exception as exc:
        pc.forward, pc.error = "error", True
        pc.notes.append(f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}")
    return pc


def check(benchmark: str, models: str, suite: str = "single",
          variant: str = "default") -> List[PairCheck]:
    from neuroatlas import catalog, data, selectors
    from neuroatlas import models as model_state
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    bench = catalog.load(benchmark)
    if bench.derived_from:
        bench = catalog.load(bench.derived_from)
    by_id = {s.identifier: s for s in checkpoint_registry()}
    specs = [by_id[i] for i in selectors.resolve_models(models)]
    out: List[PairCheck] = []
    for step in bench.steps(suite, variant):
        if step.verb != "embed":
            continue
        ds = data.status(step.dataset)
        for spec in specs:
            out.append(check_pair(step.dataset, spec, list(step.argv), ds, model_state.status(spec)))
    return out
