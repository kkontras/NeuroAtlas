"""``neuroatlas submit`` writes cluster jobs for a benchmark; ``neuroatlas
status --out DIR`` says how each is doing."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from neuroatlas.cli import MODELS_HELP, Parser, _msg
from neuroatlas.cli._table import add_format_arg, render

# options whose value is a raw submit-file line: it may itself start with "-"
# (`--extra --account=x`), which argparse would take for an option
_RAW_VALUE_OPTIONS = ("--extra",)


def _memory(text: str) -> int:
    from neuroatlas.submit import parse_memory

    try:
        return parse_memory(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _walltime(text: str) -> int:
    from neuroatlas.submit import parse_walltime

    try:
        return parse_walltime(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _clock(text: str) -> int:
    """--time: H:MM:SS (or D-HH:MM:SS). A bare number is refused: SLURM reads it
    as minutes, HTCondor as seconds."""
    from neuroatlas.submit import parse_walltime

    if str(text).strip().isdigit():
        raise argparse.ArgumentTypeError(
            f"{text!r} is ambiguous (SLURM reads minutes, HTCondor seconds): write H:MM:SS, "
            f"or --walltime SECONDS")
    try:
        return parse_walltime(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _capability(text: str) -> str:
    import re

    if text in ("auto", "none") or re.fullmatch(r"\d+\.\d", text):
        return text
    raise argparse.ArgumentTypeError(f"{text!r}: write auto, none or a capability like 7.5")


def build_submit_parser() -> argparse.ArgumentParser:
    p = Parser(
        prog="neuroatlas submit",
        description="Write one job per (dataset, model) -- each a `neuroatlas run` of that "
                    "pair -- and an HTCondor or SLURM file to queue them. Nothing is queued "
                    "here; the last line prints the command that does. Jobs run offline, so "
                    "a pair whose data or weights are not on this machine, or that its "
                    "dataset's channel map rules out, gets no job: it is listed as skipped "
                    "with the reason.")
    p.add_argument("benchmark")
    p.add_argument("-m", "--models", required=True, help=MODELS_HELP)
    p.add_argument("--dataset", default="full", metavar="single|full|SLUGS",
                   help="Default: full -- this is for the whole suite.")
    p.add_argument("--variant", default="default")
    p.add_argument("--out", required=True, type=Path, help="Folder for the job files and logs.")
    p.add_argument("--backend", choices=["condor", "slurm"], default="condor")
    p.add_argument("--mode", choices=["cached", "retry", "all"], default="cached",
                   help="cached: only jobs that never ran (default; re-running submit never "
                        "repeats finished work). retry: also failed, partial and exited ones. "
                        "all: everything. A job still in the queue (running, idle, held) is "
                        "never queued again.")
    p.add_argument("--output-root", type=Path, default=None, help="Results root for the jobs.")
    p.add_argument("--force", action="store_true",
                   help="Write jobs for pairs whose data or weights are not found here "
                        "(the channel map is still obeyed).")
    p.add_argument("-v", "--verbose", action="store_true", help="List every skipped pair.")
    r = p.add_argument_group("resources (per job)")
    r.add_argument("--gpus", type=int, default=1)
    r.add_argument("--cpus", type=int, default=4)
    r.add_argument("--memory", type=_memory, default="32G", metavar="SIZE",
                   help="With a unit: 32G, 1500M (default 32G).")
    r.add_argument("--time", type=_clock, default=None, metavar="H:MM:SS",
                   help="Wall time per job, both backends (default 24:00:00). SLURM: --time; "
                        "HTCondor: +RequestWalltime and +MaxRuntime, in seconds.")
    r.add_argument("--walltime", type=_walltime, default=None, metavar="SECONDS",
                   help="The same wall time, in seconds (or H:MM:SS).")
    r.add_argument("--gpu-capability", type=_capability, default="auto", metavar="auto|none|X.Y",
                   help="Least GPU compute capability a job may land on. auto (default): the "
                        "lowest the installed torch has kernels for. HTCondor: a requirement; "
                        "SLURM has no standard attribute, so it is printed as a hint.")
    r.add_argument("--partition", default=None, help="SLURM partition.")
    r.add_argument("--no-mem", action="store_true",
                   help="SLURM: leave --mem out (some sites forbid it).")
    r.add_argument("--requirements", default=None, help="HTCondor requirements expression.")
    r.add_argument("--walltime-attr", action="append", default=None, metavar="NAME",
                   help="HTCondor: the job attribute(s) the wall time goes in (default "
                        "RequestWalltime and MaxRuntime; repeatable).")
    r.add_argument("--extra", action="append", default=[], metavar="LINE",
                   help="A raw line for the submit file (SLURM: after #SBATCH, e.g. "
                        "`--extra --account=myproject`). Repeatable.")
    add_format_arg(p)
    return p


def _join_raw_values(argv: List[str]) -> List[str]:
    """`--extra --account=x` -> `--extra=--account=x`."""
    out, i = [], 0
    while i < len(argv):
        tok = argv[i]
        if tok in _RAW_VALUE_OPTIONS and i + 1 < len(argv):
            out.append(f"{tok}={argv[i + 1]}")
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def _warn(msg: str, fix: Optional[str] = None) -> None:
    _msg.warning(msg, fix)


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def submit_main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas import submit as sub

    parser = build_submit_parser()
    args = parser.parse_args(_join_raw_values(list(sys.argv[1:] if argv is None else argv)))
    if args.time and args.walltime and args.time != args.walltime:
        parser.error("--time and --walltime are the same setting; give one")
    walltime = args.time or args.walltime or sub.parse_walltime("24:00:00")
    if args.gpus < 0 or args.cpus < 1:
        parser.error("--gpus must be >= 0 and --cpus >= 1")

    # F-053: a flag the chosen backend has no use for is said so, not dropped silently
    other = {"condor": [("--partition", args.partition), ("--no-mem", args.no_mem)],
             "slurm": [("--requirements", args.requirements),
                       ("--walltime-attr", args.walltime_attr)]}[args.backend]
    for flag, value in other:
        if value:
            _warn(f"{flag} has no effect with --backend {args.backend}")

    for p in sub.node_local_paths():
        _warn(f"jobs run {p}, on this machine's local disk: other machines will not find it",
              "run submit from a Python environment on a shared filesystem")

    out = args.out.resolve()
    previous: Dict = {}
    try:
        previous = sub.load_manifest(out)
    except (FileNotFoundError, ValueError, TypeError, KeyError):
        pass
    planned = sub.plan_jobs(args.benchmark, args.models, args.dataset, args.variant, out,
                            args.output_root.resolve() if args.output_root else None,
                            force=args.force)
    jobs = planned["jobs"]
    queue = sub.queue_snapshot(args.backend, out, previous.get("batches"))
    states = sub.job_states(jobs, queue)
    verdicts = Counter(s.kind for s in states.values())
    queued = sub.select(jobs, args.mode, states)

    capability, how = None, ""
    if args.gpus and args.gpu_capability == "auto":
        capability, how = sub.torch_min_capability()
        if capability is None:
            _warn(f"--gpus {args.gpus} but {how}: no GPU-capability requirement written")
    elif args.gpus and args.gpu_capability != "none":
        capability, how = args.gpu_capability, "--gpu-capability"

    batch = None
    if args.backend == "condor":
        path = sub.write_condor(out, queued, gpus=args.gpus, cpus=args.cpus, memory=args.memory,
                                requirements=args.requirements, walltime=walltime,
                                extra=args.extra, gpu_capability=capability,
                                walltime_attrs=tuple(args.walltime_attr or
                                                     ("RequestWalltime", "MaxRuntime")))
        queue_cmd = f"condor_submit {path}"
    else:
        path = sub.write_slurm(out, queued, gpus=args.gpus, cpus=args.cpus, memory=args.memory,
                               time=sub.slurm_time(walltime), partition=args.partition,
                               no_mem=args.no_mem, extra=args.extra, gpu_capability=capability)
        queue_cmd = f"sbatch {path}"
        if queued:
            batch = (sub.slurm_batch_name(out, queued), [j.id for j in queued])
    sub.save_manifest(out, planned, args.backend, args.mode, batch)

    order = [v for v in sub.VERDICTS if v != "skipped"]
    parts = [f"{len(queued)} queued (mode {args.mode})",
             *sub.count_parts(list(states.values()), order)]
    print(f"{_plural(len(jobs), 'job', 'jobs')}: {', '.join(parts)}")
    skipped = planned["skipped"]
    if skipped:
        blocked = [s for s in skipped if sub.forceable(s)]
        print(sub.skip_summary(skipped) + ("" if args.verbose else " (-v lists them)"))
        if args.verbose:
            for s in skipped:
                kind = "n/a" if s.get("reason", "").startswith("n/a") else \
                    "warning" if s.get("reason", "").startswith("invalid") else "skipped"
                detail = _msg.split(s.get("detail") or "")[1][:1]
                print(f"  {kind}: {s['dataset']}/{s['model']}: {s.get('reason', '')}"
                      + (f" ({detail[0]})" if detail else ""))
        if blocked and not args.force:
            print(f"  --force writes jobs for the {len(blocked)} blocked by data or weights")
    for bad in planned.get("invalid", []):
        _msg.warning(f"{bad['dataset']}: no jobs: {bad['error']}")
    from neuroatlas import catalog

    left_out = catalog.load(args.benchmark).left_out_note(args.models)
    if left_out:
        _msg.note(left_out)
    if capability and args.gpus:
        where = f"a requirement in {path.name}" if args.backend == "condor" else \
            "a hint only (SLURM has no standard attribute)"
        # "torch 2.8.0+cu128 is built for sm_70 ... sm_120" -> the version alone
        source = (f"the lowest {how.split(' is built for')[0]} supports"
                  if " is built for" in how else how)
        print(f"GPUs: compute capability >= {capability} ({source}), {where}")
    active = {k: verdicts[k] for k in sub.ACTIVE if verdicts.get(k)}
    if active:
        n = sum(active.values())
        print(f"{_plural(n, 'job is', 'jobs are')} still in the queue "
              f"({', '.join(f'{k} {v}' for k, v in active.items())}), not queued again")
    if not queue.ok:
        _msg.note(f"the scheduler was not asked ({queue.note}): a running job is known only "
                  f"by its status file")
    waiting = verdicts.get("waiting")
    if waiting:
        print(f"{_plural(waiting, 'job waits', 'jobs wait')} for another benchmark's results, "
              f"not queued; run this submit again once they exist")
    if not queued:
        if not jobs:
            _msg.warning("nothing to queue: no pair can run here")
        else:
            print(f"nothing to queue in mode {args.mode}; neuroatlas status --out {args.out} "
                  f"shows every job")
        return
    print(f"queue them: {queue_cmd}" + (
        "  (-dry-run checks the file's syntax only, not the pool's policy)"
        if args.backend == "condor" else ""))


# --------------------------------------------------------------------------

def build_status_parser() -> argparse.ArgumentParser:
    p = Parser(
        prog="neuroatlas status",
        description="One verdict per job of a `submit`: done, partial, failed (results, none "
                    "ok), exited N (died before writing results), stopped (started, then "
                    "vanished from the queue without an exit), removed, running, idle "
                    "(queued, not started), held, missing (never ran), waiting; skipped "
                    "pairs are counted. The scheduler (condor_q / squeue) is asked when it "
                    "is on this machine; run status where you submitted.")
    p.add_argument("--out", required=True, type=Path, help="The folder `submit` wrote.")
    p.add_argument("-v", "--verbose", action="store_true", help="One row per job.")
    p.add_argument("--no-scheduler", action="store_true",
                   help="Do not ask condor_q / squeue; decide from the files only.")
    add_format_arg(p)
    return p


_FAILED = ("failed", "exited", "stopped", "removed", "partial")


def status_main(argv: Optional[List[str]] = None) -> None:
    from neuroatlas import submit as sub

    parser = build_status_parser()
    args = parser.parse_args(argv)
    out = args.out.resolve()
    try:
        manifest = sub.load_manifest(out)
    except FileNotFoundError as exc:
        _msg.error(str(exc))
        raise SystemExit(2) from None
    backend = manifest.get("backend", "condor")
    queue = sub.queue_snapshot(backend, out, manifest.get("batches"),
                               ask_scheduler=not args.no_scheduler)
    states = sub.job_states(manifest["jobs"], queue)
    rows = []
    for j in manifest["jobs"]:
        st = states[j.id]
        rows.append({"job": j.id, "dataset": j.dataset, "model": j.model, "verdict": st.verdict,
                     "detail": st.detail or None, "host": st.host,
                     "scheduler_id": st.scheduler_id,
                     "log": st.log if st.kind in _FAILED + ("running",) else None})
    counts = Counter(states[j.id].kind for j in manifest["jobs"])
    if args.verbose or args.format != "table":
        cols = ["dataset", "model", "verdict", "detail", "host", "log"]
        if args.format != "table":
            cols = ["job"] + cols + ["scheduler_id"]
        render(rows, cols, args.format)
    if args.format != "table":
        return
    parts = sub.count_parts(list(states.values()))
    print(f"{manifest['benchmark']} ({manifest['suite']}, {manifest['variant']}): "
          f"{_plural(len(manifest['jobs']), 'job', 'jobs')}: {', '.join(parts) or 'none'}; "
          f"{_plural(len(manifest['skipped']), 'pair', 'pairs')} skipped")
    if not queue.ok and not args.no_scheduler:
        _msg.note(f"the scheduler was not asked ({queue.note}): a running job is known only "
                  f"by its status file")
    if not args.verbose:
        bad = [r for r in rows if r["verdict"].split()[0] in _FAILED + ("held",)]
        for r in bad[:20]:
            print(f"  error: {r['dataset']}/{r['model']}: {r['verdict']}"
                  + (f" ({r['detail']})" if r["detail"] else "")
                  + (f"\n    log: {r['log']}" if r["log"] else ""))
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more (-v lists every job)")
    if any(counts.get(k) for k in _FAILED):
        print(f"re-queue the failed ones: neuroatlas submit {manifest['benchmark']} ... "
              f"--out {args.out} --mode retry")
    if counts.get("held"):
        print("held jobs stay in the queue until released or removed (by job id, "
              "never by user name)")
