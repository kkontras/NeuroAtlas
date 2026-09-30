"""Cluster jobs for a benchmark: one per (dataset, model), each a plain
``neuroatlas run`` of that pair, plus an HTCondor submit file or a SLURM array.

    <out>/jobs.json         what was planned, one entry per job
    <out>/jobs/<id>.sh      each job's script
    <out>/jobs.txt          the scripts still to queue under --mode
    <out>/jobs.job | jobs.sbatch
    <out>/logs/

Each job writes ``<output_root>/<benchmark>/<dataset>/<model>/results.json``,
so no two jobs ever share a results file. Jobs run offline: download data
and weights before queueing (`data download`, `models download`).

Nothing is generated for a pair the channel map rules out; it is counted as
skipped.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuroatlas import _paths

# environment a job inherits from the machine that wrote it
_PASS_ENV = ("NEUROATLAS_HOME", "EEG_DATA_ROOT", "EEG_CACHE_ROOT", "NEUROATLAS_OUTPUT_ROOT",
             "NEUROATLAS_MODELS_ROOT", "MNE_DATA",
             "HF_HOME", "HF_HUB_CACHE")

VERDICTS = ("done", "partial", "failed", "missing", "waiting", "skipped")


@dataclass
class Job:
    id: str
    benchmark: str
    dataset: str
    model: str
    variant: str
    result_dir: str
    script: str
    depends_on: Optional[str] = None     # a benchmark whose results must exist first


def _job_id(bench: str, dataset: str, model: str, variant: str) -> str:
    parts = [bench, dataset, model] + ([variant] if variant != "default" else [])
    return "__".join(parts)


def verdict(job: Job, expected_folds: Optional[int] = None) -> str:
    from neuroatlas.results import _records

    path = Path(job.result_dir) / "results.json"
    if job.depends_on and not path.is_file():
        src = Path(job.result_dir).parents[1] / job.depends_on / job.dataset
        if not any(src.rglob("results.json")):
            return "waiting"
    if not path.is_file():
        return "missing"
    recs = [r for r in _records(path) if r.model == job.model or job.model == "*"]
    if not recs:
        return "missing"
    ok = [r for r in recs if r.ok]
    if not ok:
        return "failed"
    failed = [r for r in recs if not r.ok and r.failure != "channel_map_skip"]
    if failed or (expected_folds and len(ok) < expected_folds):
        return "partial"
    return "done"


def _script(job_cmd: List[str], env: Dict[str, str]) -> str:
    lines = ["#!/usr/bin/env bash", "# written by `neuroatlas submit`; runs one (dataset, model) pair",
             "set -euo pipefail"]
    lines += [f"export {k}={shlex.quote(v)}" for k, v in sorted(env.items())]
    lines += ["export NEUROATLAS_OFFLINE=1 HF_HUB_OFFLINE=1",
              "exec " + " ".join(shlex.quote(c) for c in job_cmd)]
    return "\n".join(lines) + "\n"


def plan_jobs(benchmark: str, models: str, suite: str, variant: str, out: Path,
              output_root: Optional[Path] = None) -> Dict[str, Any]:
    from neuroatlas import catalog, run as runmod, selectors
    from neuroatlas.benchmarking_helpers.registry.discovery import checkpoint_registry

    bench = catalog.load(benchmark)
    ids = selectors.resolve_models(models)
    by_id = {s.identifier: s for s in checkpoint_registry()}
    env = {k: os.environ[k] for k in _PASS_ENV if os.environ.get(k)}
    python = sys.executable
    jobs: List[Job] = []
    skipped: List[Dict[str, str]] = []
    invalid: List[Dict[str, str]] = []
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    root_args = ["--output-root", str(output_root)] if output_root else []

    datasets = [s.dataset for s in bench.steps(suite, variant) if s.verb in ("embed", "hypnogram")]
    for dataset in datasets:
        if bench.derived_from:
            # one CPU job per dataset, over every model's staging results
            job_models = [("*", ",".join(ids))]
        else:
            cmap, map_error = runmod.channel_map(dataset)
            if map_error:
                invalid.append({"dataset": dataset, "error": map_error})
                continue
            job_models = []
            for i in ids:
                if cmap is not None and cmap.is_skip(by_id[i].model_family):
                    skipped.append({"dataset": dataset, "model": i})
                else:
                    job_models.append((i, i))
        for label, selector in job_models:
            jid = _job_id(bench.name, dataset, label if label != "*" else "all", variant)
            rdir = runmod.result_dir(bench.name, dataset, variant, output_root)
            if label != "*":
                rdir = rdir / label
            cmd = [python, "-m", "neuroatlas", "run", bench.name, "-m", selector,
                   "--dataset", dataset, "--variant", variant, *root_args]
            if label != "*":
                cmd.append("--per-model-output")
            script = out / "jobs" / f"{jid}.sh"
            script.write_text(_script(cmd, env))
            script.chmod(0o755)
            jobs.append(Job(jid, bench.name, dataset, label, variant, str(rdir), str(script),
                            depends_on=bench.derived_from))
    return {"benchmark": bench.name, "suite": suite, "variant": variant, "models": ids,
            "jobs": jobs, "skipped": skipped, "invalid": invalid}


def select(jobs: List[Job], mode: str) -> List[Job]:
    """cached: jobs with no results yet. retry: those plus failed and partial.
    all: every job. A job waiting on another benchmark is never queued."""
    wanted = {"cached": {"missing"}, "retry": {"missing", "failed", "partial"},
              "all": {"missing", "failed", "partial", "done"}}[mode]
    return [j for j in jobs if verdict(j) in wanted]


def write_condor(out: Path, queued: List[Job], *, gpus: int, cpus: int, memory: str,
                 requirements: Optional[str], walltime: Optional[int],
                 extra: List[str]) -> Path:
    (out / "jobs.txt").write_text("".join(f"{j.script}\n" for j in queued))
    mem_mb = _memory_mb(memory)
    lines = [
        "# written by `neuroatlas submit`",
        "universe       = vanilla",
        "executable     = $(script)",
        "getenv         = False",
        f"request_cpus   = {cpus}",
        f"request_memory = {mem_mb}",
        f"request_gpus   = {gpus}" if gpus else "",
        f"requirements   = {requirements}" if requirements else "",
        f"+MaxRuntime    = {walltime}" if walltime else "",
        *extra,
        f"log            = {out}/logs/$(Cluster).log",
        f"output         = {out}/logs/$(script_name).out",
        f"error          = {out}/logs/$(script_name).err",
        "script_name    = $Fn(script)",
        f"queue script from {out}/jobs.txt",
    ]
    path = out / "jobs.job"
    path.write_text("\n".join(l for l in lines if l) + "\n")
    return path


def write_slurm(out: Path, queued: List[Job], *, gpus: int, cpus: int, memory: str,
                time: str, partition: Optional[str], no_mem: bool, extra: List[str]) -> Path:
    (out / "jobs.txt").write_text("".join(f"{j.script}\n" for j in queued))
    lines = ["#!/usr/bin/env bash", "# written by `neuroatlas submit`",
             "#SBATCH --job-name=neuroatlas",
             f"#SBATCH --array=0-{max(len(queued) - 1, 0)}",
             f"#SBATCH --cpus-per-task={cpus}",
             f"#SBATCH --gpus={gpus}" if gpus else "",
             "" if no_mem else f"#SBATCH --mem={memory}",
             f"#SBATCH --time={time}",
             f"#SBATCH --partition={partition}" if partition else "",
             *[f"#SBATCH {e}" for e in extra],
             f"#SBATCH --output={out}/logs/%A_%a.out",
             f"#SBATCH --error={out}/logs/%A_%a.err",
             "set -euo pipefail",
             f'script=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {out}/jobs.txt)',
             'exec "$script"']
    path = out / "jobs.sbatch"
    path.write_text("\n".join(l for l in lines if l) + "\n")
    return path


def _memory_mb(memory: str) -> int:
    text = memory.strip().upper()
    for suffix, factor in (("TB", 1024 * 1024), ("GB", 1024), ("G", 1024), ("MB", 1), ("M", 1),
                           ("T", 1024 * 1024)):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * factor)
    return int(float(text))


def save_manifest(out: Path, planned: Dict[str, Any], backend: str, mode: str) -> Path:
    payload = {**planned, "backend": backend, "mode": mode,
               "jobs": [asdict(j) for j in planned["jobs"]]}
    path = out / "jobs.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_manifest(out: Path) -> Dict[str, Any]:
    path = out / "jobs.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist: run `neuroatlas submit ... --out {out}` first")
    data = json.loads(path.read_text())
    data["jobs"] = [Job(**j) for j in data["jobs"]]
    return data
