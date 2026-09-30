"""``neuroatlas submit`` writes cluster jobs for a benchmark; ``neuroatlas
status --out DIR`` says how each is doing."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Optional

from neuroatlas.cli._table import add_format_arg, render


def build_submit_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neuroatlas submit",
        description="Write one job per (dataset, model) -- each a `neuroatlas run` of that "
                    "pair -- and an HTCondor or SLURM file to queue them. Nothing is queued "
                    "here; the last line prints the command that does. Jobs run offline.")
    p.add_argument("benchmark")
    p.add_argument("-m", "--models", required=True)
    p.add_argument("--dataset", default="full", metavar="single|full|SLUGS",
                   help="Default: full -- this is for the whole suite.")
    p.add_argument("--variant", default="default")
    p.add_argument("--out", required=True, type=Path, help="Folder for the job files and logs.")
    p.add_argument("--backend", choices=["condor", "slurm"], default="condor")
    p.add_argument("--mode", choices=["cached", "retry", "all"], default="cached",
                   help="cached: only jobs with no results yet (default; re-running submit "
                        "never repeats finished work). retry: also failed and partial ones. "
                        "all: everything.")
    p.add_argument("--output-root", type=Path, default=None, help="Results root for the jobs.")
    r = p.add_argument_group("resources (per job)")
    r.add_argument("--gpus", type=int, default=1)
    r.add_argument("--cpus", type=int, default=4)
    r.add_argument("--memory", default="32G")
    r.add_argument("--time", default="24:00:00", help="SLURM wall time (default 24:00:00).")
    r.add_argument("--partition", default=None, help="SLURM partition.")
    r.add_argument("--no-mem", action="store_true",
                   help="SLURM: leave --mem out (some sites forbid it).")
    r.add_argument("--requirements", default=None, help="HTCondor requirements expression.")
    r.add_argument("--walltime", type=int, default=None, help="HTCondor +MaxRuntime, in seconds.")
    r.add_argument("--extra", action="append", default=[], metavar="LINE",
                   help="A raw line for the submit file (SLURM: after #SBATCH). Repeatable.")
    add_format_arg(p)
    return p


def submit_main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas import submit as sub

    args = build_submit_parser().parse_args(argv)
    out = args.out.resolve()
    planned = sub.plan_jobs(args.benchmark, args.models, args.dataset, args.variant, out,
                            args.output_root.resolve() if args.output_root else None)
    jobs = planned["jobs"]
    verdicts = Counter(sub.verdict(j) for j in jobs)
    queued = sub.select(jobs, args.mode)
    if args.backend == "condor":
        path = sub.write_condor(out, queued, gpus=args.gpus, cpus=args.cpus, memory=args.memory,
                                requirements=args.requirements, walltime=args.walltime,
                                extra=args.extra)
        queue_cmd = f"condor_submit {path}"
    else:
        path = sub.write_slurm(out, queued, gpus=args.gpus, cpus=args.cpus, memory=args.memory,
                               time=args.time, partition=args.partition, no_mem=args.no_mem,
                               extra=args.extra)
        queue_cmd = f"sbatch {path}"
    sub.save_manifest(out, planned, args.backend, args.mode)
    parts = [f"skipped {len(planned['skipped'])}"] + [f"{k} {v}" for k, v in sorted(verdicts.items())]
    print("jobs: " + "   ".join(parts) + f"   queued (mode={args.mode}) {len(queued)}")
    if not queued:
        print("nothing to queue: every job has results for this mode. "
              "`neuroatlas status --out " + str(args.out) + "` shows them.")
        return
    waiting = verdicts.get("waiting")
    if waiting:
        print(f"{waiting} jobs wait for another benchmark's results and were not queued; "
              f"run this submit again once they exist")
    print(f"submit with:  {queue_cmd}")


# --------------------------------------------------------------------------

def build_status_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neuroatlas status",
                                description="One verdict per job of a `submit`: done, partial, "
                                            "failed, missing, waiting; skipped pairs are counted.")
    p.add_argument("--out", required=True, type=Path, help="The folder `submit` wrote.")
    p.add_argument("-v", "--verbose", action="store_true", help="One row per job.")
    add_format_arg(p)
    return p


def status_main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas import submit as sub

    args = build_status_parser().parse_args(argv)
    try:
        manifest = sub.load_manifest(args.out.resolve())
    except FileNotFoundError as exc:
        raise SystemExit(f"error: {exc}") from None
    rows = [{"job": j.id, "dataset": j.dataset, "model": j.model, "verdict": sub.verdict(j)}
            for j in manifest["jobs"]]
    counts = Counter(r["verdict"] for r in rows)
    if args.verbose or args.format != "table":
        render(rows, ["dataset", "model", "verdict"], args.format)
    if args.format == "table":
        parts = [f"{v} {counts[v]}" for v in sub.VERDICTS if counts.get(v)]
        parts.append(f"skipped {len(manifest['skipped'])}")
        print(f"{manifest['benchmark']} ({manifest['suite']}, {manifest['variant']}): "
              + "   ".join(parts))
        if counts.get("failed") or counts.get("partial"):
            print(f"logs: {args.out}/logs; re-queue with `neuroatlas submit ... --mode retry`")
