"""``neuroatlas submit`` writes cluster jobs for a benchmark; ``neuroatlas
status --out DIR`` says how each is doing."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from neuroatlas.cli import MODELS_HELP, LinesFormatter, Parser, _msg
from neuroatlas.cli._table import NOT_APPLICABLE, add_format_arg, render

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
    from neuroatlas.cli.check import BENCHMARK_HELP, DATASET_HELP, VARIANT_HELP

    p = Parser(
        prog="neuroatlas submit",
        description="Write one job per dataset and checkpoint, and an HTCondor or SLURM file "
                    "that queues them. Nothing is queued until you run the command printed on "
                    "the last line. Pairs whose data or weights are missing on this machine, "
                    "or that the channel map rules out, get no job.")
    p.add_argument("benchmark", help=BENCHMARK_HELP)
    p.add_argument("-m", "--models", required=True, help=MODELS_HELP)
    p.add_argument("--dataset", default="full", metavar="single|full|NAMES",
                   help=f"{DATASET_HELP} (default: full).")
    p.add_argument("--variant", default="default", help=VARIANT_HELP)
    p.add_argument("--out", required=True, type=Path, metavar="DIR",
                   help="Folder for the job files and logs.")
    p.add_argument("--backend", choices=["condor", "slurm"], default="condor",
                   help="Scheduler to write the jobs for (default: condor).")
    p.add_argument("--mode", choices=["cached", "retry", "all"], default="cached",
                   help="Which jobs to queue (default: cached). cached queues the jobs that "
                        "never ran, retry also the failed, partial and exited ones, and all "
                        "every job. A job still in the queue is never queued twice.")
    p.add_argument("--output-root", type=Path, default=None, metavar="DIR",
                   help="Where the jobs write results (default: the output_root setting).")
    p.add_argument("--force", action="store_true",
                   help="Also write jobs for pairs whose data or weights are missing here. "
                        "The channel map still applies.")
    p.add_argument("--reprobe", action="store_true",
                   help="Make the jobs fit every fold again, as `run --reprobe` does. Use "
                        "with --mode all to rerun finished jobs.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="List every pair without a job, and why.")
    r = p.add_argument_group("resources per job")
    r.add_argument("--gpus", type=int, default=1, metavar="N", help="GPUs per job (default: 1).")
    r.add_argument("--cpus", type=int, default=4, metavar="N", help="CPUs per job (default: 4).")
    r.add_argument("--memory", type=_memory, default="32G", metavar="SIZE",
                   help="Memory per job, with a unit such as 32G or 1500M (default: 32G).")
    r.add_argument("--time", type=_clock, default=None, metavar="H:MM:SS",
                   help="Wall time per job (default: 24:00:00). It sets --time for SLURM, and "
                        "+RequestWalltime and +MaxRuntime in seconds for HTCondor.")
    r.add_argument("--walltime", type=_walltime, default=None, metavar="SECONDS",
                   help="The same wall time, in seconds or as H:MM:SS.")
    r.add_argument("--gpu-capability", type=_capability, default="auto", metavar="auto|none|X.Y",
                   help="Lowest GPU compute capability a job may run on (default: auto). auto "
                        "is the lowest the installed PyTorch supports. For SLURM it is only "
                        "printed as a hint.")
    r.add_argument("--partition", default=None, help="SLURM partition.")
    r.add_argument("--no-mem", action="store_true",
                   help="Leave --mem out of SLURM jobs, for sites that forbid it.")
    r.add_argument("--requirements", default=None, help="HTCondor requirements expression.")
    r.add_argument("--walltime-attr", action="append", default=None, metavar="NAME",
                   help="HTCondor job attribute that holds the wall time (default: "
                        "RequestWalltime and MaxRuntime). Can be repeated.")
    r.add_argument("--extra", action="append", default=[], metavar="LINE",
                   help="Add a raw line to the job file. For SLURM it follows #SBATCH, as in "
                        "`--extra --account=myproject`. Can be repeated.")
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
        _warn(f"the jobs would run {p}, which is on this machine's local disk: other machines "
              f"will not find it",
              "install neuroatlas in a Python environment on a shared filesystem, and run "
              "submit from it")

    out = args.out.resolve()
    previous: Dict = {}
    try:
        previous = sub.load_manifest(out)
    except (FileNotFoundError, ValueError, TypeError, KeyError):
        pass
    planned = sub.plan_jobs(args.benchmark, args.models, args.dataset, args.variant, out,
                            args.output_root.resolve() if args.output_root else None,
                            force=args.force, reprobe=args.reprobe)
    jobs = planned["jobs"]
    queue = sub.queue_snapshot(args.backend, out, previous.get("batches"))
    states = sub.job_states(jobs, queue)
    verdicts = Counter(s.kind for s in states.values())
    queued = sub.select(jobs, args.mode, states)

    capability, how = None, ""
    if args.gpus and args.gpu_capability == "auto":
        capability, how = sub.torch_min_capability()
        if capability is None:
            _warn(f"--gpus {args.gpus} but {how}: no GPU-capability requirement written, so a "
                  f"job may land on a GPU torch cannot use",
                  "--gpu-capability X.Y sets one by hand (e.g. 7.5)")
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
    parts = [f"{len(queued)} to queue (--mode {args.mode})",
             *sub.count_parts(list(states.values()), order)]
    print(f"{_plural(len(jobs), 'job', 'jobs')}: {', '.join(parts)}")
    skipped = planned["skipped"]
    if skipped:
        blocked = [s for s in skipped if sub.forceable(s)]
        print(sub.skip_summary(skipped) + ("" if args.verbose else " (-v lists them)"))
        if args.verbose:
            rows = [{"dataset": s["dataset"], "model": s["model"],
                     "outcome": _skip_outcome(s), "reason": sub._reason_word(s.get("reason", "")),
                     "detail": (_msg.split(s.get("detail") or "")[1][:1]
                                or [NOT_APPLICABLE])[0]}
                    for s in skipped]
            render(rows, ["dataset", "model", "outcome", "reason", "detail"], "table",
                   labels={"model": "checkpoint"})
        if blocked and not args.force:
            _skipped_fixes(blocked)
    for bad in planned.get("invalid", []):
        from neuroatlas.run import REINSTALL

        _msg.warning(f"{bad['dataset']}: no jobs: {bad['error']}", REINSTALL)
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
                  f"by its status file; run submit where you queue jobs to see the queue")
    waiting = verdicts.get("waiting")
    if waiting:
        source = catalog.load(args.benchmark).derived_from or "another benchmark"
        _msg.note(f"{_plural(waiting, 'job waits', 'jobs wait')} for the {source} results, "
                  f"so {'it is' if waiting == 1 else 'they are'} not queued yet; run this "
                  f"submit again once those exist",
                  f"neuroatlas submit {source} -m {args.models} --out DIR (first)")
    if not queued:
        if not jobs:
            _msg.warning("nothing to queue: no pair can run here (the lines above say why)")
        else:
            print(f"nothing to queue with --mode {args.mode}; `neuroatlas status --out "
                  f"{args.out}` shows every job")
        return
    print(f"queue them: {queue_cmd}")


def _skip_outcome(skip: dict) -> str:
    """skipped (data or weights missing) | ruled out | invalid, for one pair
    without a job."""
    reason = str(skip.get("reason", ""))
    if reason.startswith(("n/a", "ruled out")):
        return _msg.RULED_OUT
    if reason.startswith("invalid"):
        return "invalid"
    return _msg.SKIPPED


def _skipped_fixes(blocked: List[dict]) -> None:
    """What brings the pairs skipped for data or weights back: the downloads,
    by name; or --force."""
    datasets = list(dict.fromkeys(s["dataset"] for s in blocked
                                  if str(s.get("reason", "")).startswith("data")))
    weights = list(dict.fromkeys(s["model"] for s in blocked
                                 if str(s.get("reason", "")).startswith("weights not")))
    fixes = ([f"neuroatlas data download {' '.join(datasets)}"] if datasets else []) + (
        [f"neuroatlas models download {','.join(weights)}"] if weights else [])
    text = (f"{_plural(len(blocked), 'pair was', 'pairs were')} skipped for missing data or "
            f"weights; --force writes their jobs anyway (they fail where those are missing)")
    _msg.note(text + "".join(f"\nfix: {f}" for f in fixes))


# --------------------------------------------------------------------------

#: What each state `status` reports means (its --help and docs/cli.md).
JOB_STATES = {
    "done": "every fold succeeded",
    "partial": "some folds failed or are missing",
    "failed": "no fold succeeded",
    "exited N": "the job stopped with exit status N before writing results",
    "stopped": "the job started, then left the queue without an exit status",
    "removed": "the job was removed from the queue",
    "running, idle, held": "as the scheduler reports them",
    "missing": "the job never ran",
    "waiting": "the job needs the results of another benchmark first",
}


def _states_text() -> str:
    width = max(len(state) for state in JOB_STATES)
    return "job states:\n" + "\n".join(f"  {state:<{width}}  {meaning}"
                                       for state, meaning in JOB_STATES.items())


def build_status_parser() -> argparse.ArgumentParser:
    p = Parser(
        prog="neuroatlas status",
        description="Count the jobs that `submit` wrote by state, such as done, failed, "
                    "running or missing. Run it where you submitted, so that it can ask the "
                    "scheduler (condor_q or squeue).",
        epilog=_states_text(),
        formatter_class=LinesFormatter)
    p.add_argument("--out", required=True, type=Path, help="The folder `submit` wrote.")
    p.add_argument("-v", "--verbose", action="store_true", help="Show one row per job.")
    p.add_argument("--no-scheduler", action="store_true",
                   help="Do not ask condor_q or squeue. Use only the files.")
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
        shown = rows if args.format != "table" else \
            [{**r, "verdict": sub.verdict_word(r["verdict"])} for r in rows]
        render(shown, cols, args.format, labels={"model": "checkpoint", "verdict": "state"})
    if args.format != "table":
        return
    parts = sub.count_parts(list(states.values()))
    print(f"{manifest['benchmark']} ({_suite_text(manifest)}, variant {manifest['variant']}): "
          f"{_plural(len(manifest['jobs']), 'job', 'jobs')}: {', '.join(parts) or 'none'}"
          + (f"; {_plural(len(manifest['skipped']), 'pair', 'pairs')} without a job"
             if manifest["skipped"] else ""))
    if not queue.ok and not args.no_scheduler:
        _msg.note(f"the scheduler was not asked ({queue.note}): a running job is known only "
                  f"by its status file; run status where you submitted to see the queue")
    if not args.verbose:
        failed = [r for r in rows if r["verdict"].split()[0] in _FAILED]
        for r in failed[:20]:
            _msg.error(f"{r['dataset']}/{r['model']}: {r['verdict']}"
                       + (f" ({r['detail']})" if r["detail"] else "")
                       + (f"\nits log: {r['log']}" if r["log"] else ""))
        if len(failed) > 20:
            _msg.note(f"{len(failed) - 20} more failed jobs are not shown",
                      f"neuroatlas status --out {args.out} -v (one row per job)")
    if any(counts.get(k) for k in _FAILED):
        _msg.note("the failed jobs can be queued again",
                  _retry_command(manifest, args.out))
    if counts.get("held"):
        held = [r for r in rows if r["verdict"].split()[0] == "held"]
        ids = " ".join(r["scheduler_id"] for r in held[:20] if r["scheduler_id"])
        unknown = sum(1 for r in held[:20] if not r["scheduler_id"])
        release = "condor_release" if manifest.get("backend", "condor") == "condor" \
            else "scontrol release"
        _msg.note(f"{_plural(len(held), 'job is', 'jobs are')} held: the scheduler keeps "
                  f"{'it' if len(held) == 1 else 'them'} in the queue until released or "
                  f"removed (by job id, never by user name: that would act on all your jobs)"
                  + (f"; no scheduler id is recorded for {unknown} of them" if unknown else "")
                  + "".join(f"\n{r['dataset']}/{r['model']}: {r['detail'] or 'held'}"
                            for r in held[:20]),
                  f"{release} {ids}" if ids else None)


def _suite_text(manifest: dict) -> str:
    """The datasets a submit covered, in words: ``all datasets``, ``the
    quick dataset``, or their names."""
    suite = str(manifest.get("suite", "full"))
    return {"full": "all datasets", "single": "the quick dataset"}.get(suite, suite)


def _retry_command(manifest: dict, out: Path) -> str:
    """The submit that re-queues a folder's failed jobs, with the selection
    it was written for (a jobs.json from before the selector was kept: the
    checkpoint ids)."""
    import shlex

    selector = manifest.get("selector") or ",".join(manifest.get("models") or []) or "MODELS"
    parts = ["neuroatlas", "submit", manifest["benchmark"], "-m", selector,
             "--dataset", str(manifest.get("suite", "full"))]
    if manifest.get("variant", "default") != "default":
        parts += ["--variant", manifest["variant"]]
    if manifest.get("backend", "condor") != "condor":
        parts += ["--backend", manifest["backend"]]
    parts += ["--out", str(out), "--mode", "retry"]
    return " ".join(shlex.quote(p) for p in parts)
