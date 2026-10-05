"""Cluster jobs for a benchmark: one per (dataset, model), each a plain
``neuroatlas run`` of that pair, plus an HTCondor submit file or a SLURM array.

    <out>/jobs.json         what was planned, one entry per job, and what was skipped
    <out>/jobs/<id>.sh      each job's script
    <out>/jobs.txt          the scripts still to queue under --mode
    <out>/jobs.job | jobs.sbatch
    <out>/logs/

Each job writes ``<output_root>/<benchmark>/<dataset>/<model>/results.json``,
so no two jobs ever share a results file. Next to it the job script keeps
``job_status.json`` -- ``running`` when it starts, then its exit code, host and
log -- written by the shell, so a job that dies before Python writes anything
(a crash, a missing interpreter, a signal) still leaves a record.

Jobs run offline: download data and weights before queueing (`data download`,
`models download`). A pair that cannot run -- data or weights missing, a
channel map that skips it, has no entry for its family (invalid) or fails to
load -- gets no job; it is listed as skipped with the reason.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neuroatlas import _paths

# environment a job inherits from the machine that wrote it (locations only:
# no token is ever written into a job script -- jobs run offline)
_PASS_ENV = ("NEUROATLAS_HOME", "EEG_DATA_ROOT", "EEG_CACHE_ROOT", "NEUROATLAS_OUTPUT_ROOT",
             "NEUROATLAS_MODELS_ROOT", "MNE_DATA",
             "HF_HOME", "HF_HUB_CACHE",
             # so the job imports the same neuroatlas as the submitter (a checkout
             # on PYTHONPATH rather than an install)
             "PYTHONPATH", "PYTHONDONTWRITEBYTECODE")


def node_local_paths() -> List[str]:
    """The interpreter or PYTHONPATH entries under /tmp-like folders, which a
    job on another machine will not find."""
    local = ("/tmp/", "/var/tmp/", "/dev/shm/")
    paths = [sys.executable] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    return [p for p in paths if p.startswith(local)]


MARKER = "job_status.json"

# Verdicts, in the order `status` prints them. "exited N" carries the code.
VERDICTS = ("done", "partial", "failed", "exited", "stopped", "removed",
            "running", "idle", "held", "missing", "waiting", "skipped")
# A job in one of these is still the scheduler's: never queue it again.
ACTIVE = ("running", "idle", "held")
# What each --mode queues. "exited N", "stopped" and "removed" are failures
# that left no results, so retry takes them.
_MODES = {"cached": {"missing"},
          "retry": {"missing", "failed", "partial", "exited", "stopped", "removed"},
          "all": {"missing", "failed", "partial", "exited", "stopped", "removed", "done"}}


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
    depends_dir: Optional[str] = None    # ... and where they are
    models: Optional[str] = None         # the job's -m selection (several for a derived job)


@dataclass
class JobState:
    verdict: str                          # one of VERDICTS ("exited" as "exited N")
    detail: str = ""
    host: Optional[str] = None
    log: Optional[str] = None
    scheduler_id: Optional[str] = None

    @property
    def kind(self) -> str:
        return self.verdict.split()[0]


def _job_id(bench: str, dataset: str, model: str, variant: str) -> str:
    parts = [bench, dataset, model] + ([variant] if variant != "default" else [])
    return "__".join(parts)


def submit_dir(job: Job) -> Path:
    return Path(job.script).parents[1]


def tag(out: Path) -> str:
    """Names this submit folder's jobs in the scheduler's queue."""
    return "na-" + hashlib.sha1(str(Path(out).resolve()).encode()).hexdigest()[:10]


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------

def _staging_ready(job: Job) -> Tuple[bool, str]:
    """A derived job (hypnograms) needs the staging results of every one of
    its models, each with at least one successful fold."""
    from neuroatlas.results import _records

    if job.depends_dir:
        src = Path(job.depends_dir)
    else:                                               # manifests written before depends_dir
        src = Path(job.result_dir).parents[1] / job.depends_on / job.dataset
    files = sorted(src.rglob("results.json")) if src.is_dir() else []
    ok_models = {r.model for f in files for r in _records(f) if r.ok}
    wanted = [m for m in (job.models or "").split(",") if m] if job.model == "*" else [job.model]
    if not files:
        return False, f"no {job.depends_on} results in {src}"
    if not wanted:
        return bool(ok_models), f"no successful {job.depends_on} fold in {src}"
    lacking = [m for m in wanted if m not in ok_models]
    if lacking:
        return False, f"no successful {job.depends_on} fold yet for {', '.join(lacking)}"
    return True, ""


def read_marker(job: Job) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads((Path(job.result_dir) / MARKER).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _from_results(job: Job, expected_folds: Optional[int]) -> Optional[str]:
    from neuroatlas.results import _records

    path = Path(job.result_dir) / "results.json"
    if not path.is_file():
        return None
    recs = [r for r in _records(path) if r.model == job.model or job.model == "*"]
    if not recs:
        return None
    ok = [r for r in recs if r.ok]
    if not ok:
        return "failed"
    failed = [r for r in recs if not r.ok and r.failure != "channel_map_skip"]
    if failed or (expected_folds and len(ok) < expected_folds):
        return "partial"
    return "done"


def job_log(job: Job, backend: Optional[str] = None, sched_id: Optional[str] = None) -> Optional[str]:
    """The job's stderr file, where its error is: HTCondor names it after the
    job, SLURM after the array task."""
    out = submit_dir(job)
    if backend is None:
        backend = "slurm" if (out / "jobs.sbatch").is_file() and not (out / "jobs.job").is_file() \
            else "condor"
    if backend == "condor":
        return str(out / "logs" / f"{job.id}.err")
    return str(out / "logs" / f"{sched_id}.err") if sched_id else None


def job_state(job: Job, expected_folds: Optional[int] = None,
              queue: Optional["QueueSnapshot"] = None) -> JobState:
    """What became of one job, from (in this order) the scheduler's queue, the
    HTCondor event log, the job's own status file and its results.

    ``queue`` is a :class:`QueueSnapshot`; without one (or when the scheduler
    could not be asked) only the files decide.
    """
    marker = read_marker(job)
    host = (marker or {}).get("host") or None
    sched_id = (marker or {}).get("scheduler_id") or None
    log = job_log(job, queue.backend if queue else None, sched_id)

    # 1. still the scheduler's?
    entry = queue.find(job, sched_id) if queue else None
    if entry:
        return JobState(entry.state, entry.detail, host, log, entry.id)
    event = queue.events.get(job.id) if queue else None
    queue_known = bool(queue and queue.ok)
    if not queue_known and event and event.state in ACTIVE:
        # no scheduler to ask: the event log is the best evidence there is
        return JobState(event.state, event.detail + " (from the HTCondor log)", host, log, event.id)

    # 2. results, unless the last attempt started after them and failed
    path = Path(job.result_dir) / "results.json"
    from_results = _from_results(job, expected_folds)
    started = (marker or {}).get("started")
    stale = bool(from_results and marker and isinstance(started, (int, float))
                 and marker.get("state") == "exited" and marker.get("exit_code") not in (0, None)
                 and path.stat().st_mtime < started)
    if from_results and not stale:
        return JobState(from_results, "", host, log, sched_id)

    # 3. the job's own record, then the event log
    if marker and marker.get("state") == "exited":
        code = marker.get("exit_code")
        detail = f"on {host}" if host else ""
        if stale:
            detail += (", " if detail else "") + "older results kept"
        if code == 0:
            return JobState("failed", "exited 0 without writing results", host, log, sched_id)
        if isinstance(code, int) and code > 128:
            detail += (", " if detail else "") + f"signal {code - 128}"
        if event and event.state == "removed" and event.id == sched_id:
            detail += (", " if detail else "") + "removed from the queue"
        return JobState(f"exited {code}", detail, host, log, sched_id)
    if event and event.state in ("exited", "removed"):
        return JobState(event.verdict, event.detail + " (from the HTCondor log)", host, log, event.id)
    if marker and marker.get("state") == "running":
        if queue_known:
            return JobState("stopped", "started but recorded no exit and is not in the queue "
                                       "(killed, evicted, or the node failed)", host, log, sched_id)
        return JobState("running", "per the job's status file; the scheduler was not asked",
                        host, log, sched_id)
    if event and event.state in ACTIVE and queue_known:
        return JobState("stopped", f"the HTCondor log's last word is '{event.state}' "
                                   f"({event.detail}), but it is not in the queue", host, log, event.id)

    if job.depends_on:
        ready, why = _staging_ready(job)
        if not ready:
            return JobState("waiting", why)
    return JobState("missing")


def verdict(job: Job, expected_folds: Optional[int] = None,
            queue: Optional["QueueSnapshot"] = None) -> str:
    return job_state(job, expected_folds, queue).verdict


def select(jobs: List[Job], mode: str, states: Optional[Dict[str, JobState]] = None) -> List[Job]:
    """cached: jobs with no results yet. retry: those plus failed, partial and
    exited ones. all: every job. A job still in the queue (running, idle,
    held) or waiting on another benchmark is never queued."""
    wanted = _MODES[mode]
    states = states if states is not None else {j.id: job_state(j) for j in jobs}
    return [j for j in jobs if states[j.id].kind in wanted]


# --------------------------------------------------------------------------
# the scheduler's view
# --------------------------------------------------------------------------

@dataclass
class QueueEntry:
    id: str
    job: Optional[str]
    state: str                  # running | idle | held | exited | removed
    detail: str = ""
    verdict: str = ""


@dataclass
class QueueSnapshot:
    """The scheduler's queue as `status` saw it, plus (HTCondor) the last event
    of each job in this folder's event logs."""
    backend: str
    ok: bool                                            # the scheduler answered
    by_job: Dict[str, QueueEntry] = field(default_factory=dict)
    by_id: Dict[str, QueueEntry] = field(default_factory=dict)
    events: Dict[str, QueueEntry] = field(default_factory=dict)
    note: str = ""

    def find(self, job: Job, sched_id: Optional[str]) -> Optional[QueueEntry]:
        entry = self.by_job.get(job.id)
        if entry is None and sched_id:
            entry = self.by_id.get(str(sched_id))
        return entry


_CONDOR_STATUS = {"1": "idle", "2": "running", "3": "removed", "4": "exited", "5": "held",
                  "6": "running", "7": "held"}


def _run_quiet(cmd: List[str], timeout: int = 60) -> Optional[str]:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _condor_queue(snap: QueueSnapshot, out: Path, batches: Dict[str, List[str]]) -> None:
    text = _run_quiet(["condor_q", "-constraint", "NeuroatlasJob =!= undefined",
                       "-af:t", "ClusterId", "ProcId", "JobStatus", "NeuroatlasJob",
                       "NeuroatlasTag", "HoldReason"])
    if text is None:
        snap.note = "condor_q is not available here" if shutil.which("condor_q") is None \
            else "condor_q failed"
        return
    snap.ok = True
    mine = tag(out)
    for line in text.splitlines():
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        cid, pid, status, jid, jtag = cols[:5]
        state = _CONDOR_STATUS.get(status, "idle")
        if state not in ACTIVE:
            continue
        reason = cols[5] if len(cols) > 5 and cols[5] != "undefined" else ""
        entry = QueueEntry(f"{cid}.{pid}", jid, state,
                           f"{cid}.{pid}" + (f": {reason}" if state == "held" and reason else ""))
        snap.by_id[entry.id] = entry
        if jtag == mine:
            snap.by_job[jid] = entry


def _slurm_queue(snap: QueueSnapshot, out: Path, batches: Dict[str, List[str]]) -> None:
    user = os.environ.get("USER") or ""
    text = _run_quiet(["squeue", "-h", "-r", "-u", user, "-o", "%i|%j|%K|%T|%R"])
    if text is None:
        snap.note = "squeue is not available here" if shutil.which("squeue") is None \
            else "squeue failed"
        return
    snap.ok = True
    for line in text.splitlines():
        cols = line.strip().split("|")
        if len(cols) < 5:
            continue
        sid, name, index, status, reason = cols[:5]
        status = status.upper()
        state = "running" if status in ("RUNNING", "COMPLETING", "CONFIGURING", "STAGE_OUT") else \
            "held" if status in ("SUSPENDED", "STOPPED") or "held" in reason.lower() else \
            "idle" if status in ("PENDING", "REQUEUED", "REQUEUE_HOLD", "RESIZING") else None
        if state is None:
            continue
        jobs = batches.get(name)
        jid = jobs[int(index)] if jobs and index.isdigit() and int(index) < len(jobs) else None
        entry = QueueEntry(sid, jid, state, sid + (f": {reason}" if state != "running" and reason else ""))
        snap.by_id[sid] = entry
        if jid:
            snap.by_job[jid] = entry


_EVENT = re.compile(r"^(\d{3}) \((\d+)\.(\d+)\.\d+\) (\S+ \S+) (.*)$")


def _event_blocks(text: str):
    """(code, 'cluster.proc', time, first line rest, body lines) per event."""
    for block in text.split("\n...\n"):
        lines = block.strip("\n").splitlines()
        if lines and lines[-1].strip() == "...":
            lines = lines[:-1]
        if not lines:
            continue
        m = _EVENT.match(lines[0])
        if m:
            code, cid, pid, when, rest = m.groups()
            yield code, f"{int(cid)}.{int(pid)}", when, rest, lines[1:]


def condor_events(logs: Path) -> Dict[str, QueueEntry]:
    """The last event of each job, from the HTCondor event logs our submit
    files name. `job_ad_information_attrs = NeuroatlasJob` makes HTCondor
    write the job's id into the log (as a 028 event after each event), so
    this works without a scheduler to ask, from any machine."""
    events = []
    names: Dict[str, str] = {}
    for path in sorted(logs.glob("*.log")) if logs.is_dir() else []:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for code, sid, when, rest, body in _event_blocks(text):
            for line in body:
                key, _, value = line.strip().partition(" = ")
                if key == "NeuroatlasJob":
                    names[sid] = value.strip().strip('"')
            if code != "028":
                events.append((code, sid, when, rest, body))

    last: Dict[str, Tuple[str, QueueEntry]] = {}
    for code, sid, when, rest, body in events:
        jid = names.get(sid)
        if not jid:
            continue
        entry: Optional[QueueEntry] = None
        if code in ("000", "013", "004", "022"):        # submitted, released, evicted, disconnected
            entry = QueueEntry(sid, jid, "idle", sid)
        elif code == "001":
            host = re.search(r"alias=([^&>]+)", rest)
            entry = QueueEntry(sid, jid, "running", sid + (f" on {host.group(1)}" if host else ""))
        elif code == "012":
            reason = body[0].strip() if body else ""
            entry = QueueEntry(sid, jid, "held", sid + (f": {reason}" if reason else ""))
        elif code == "005":
            text = "\n".join(body)
            rv = re.search(r"return value (\d+)", text)
            sig = re.search(r"signal (\d+)", text)
            n = int(rv.group(1)) if rv else 128 + int(sig.group(1)) if sig else None
            entry = QueueEntry(sid, jid, "exited", sid, f"exited {n}" if n is not None else "exited ?")
            if n == 0:
                entry.verdict = "failed"
                entry.detail = f"{sid} exited 0 without writing results"
        elif code == "009":
            entry = QueueEntry(sid, jid, "removed", f"{sid} removed from the queue", "removed")
        if entry is not None and (jid not in last or when >= last[jid][0]):
            last[jid] = (when, entry)
    return {jid: e for jid, (_, e) in last.items()}


def queue_snapshot(backend: str, out: Path, batches: Optional[Dict[str, List[str]]] = None,
                   ask_scheduler: bool = True) -> QueueSnapshot:
    snap = QueueSnapshot(backend, False)
    if ask_scheduler:
        (_condor_queue if backend == "condor" else _slurm_queue)(snap, Path(out), batches or {})
    if backend == "condor":
        snap.events = condor_events(Path(out) / "logs")
    return snap


def count_parts(states: List[JobState], include=VERDICTS) -> List[str]:
    """['done 3', 'exited 2 (code 1, 127)', 'running 1'] in VERDICTS order."""
    from collections import Counter

    kinds = Counter(s.kind for s in states)
    parts = []
    for v in include:
        if not kinds.get(v):
            continue
        text = f"{v} {kinds[v]}"
        if v == "exited":
            codes = sorted({s.verdict.split(" ", 1)[1] for s in states if s.kind == "exited"},
                           key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else 0, c))
            text += f" (exit code{'s' if len(codes) > 1 else ''} {', '.join(codes)})"
        parts.append(text)
    return parts


def job_states(jobs: List[Job], queue: Optional[QueueSnapshot] = None,
               expected_folds: Optional[int] = None) -> Dict[str, JobState]:
    return {j.id: job_state(j, expected_folds, queue) for j in jobs}


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

_WRAPPER = r"""
_na_job=__JOB__
_na_dir=__DIR__
_na_submit=__SUBMIT__
_na_started=$(date +%s)
_na_host=$(hostname 2>/dev/null | tr -cd 'A-Za-z0-9._-')
_na_sched=$(printf %s "${NEUROATLAS_SCHED_ID:-}" | tr -cd 'A-Za-z0-9._-')
_na_child=
_na_status() {   # state exit_code finished
  mkdir -p "$_na_dir" 2>/dev/null || return 0
  local tmp="$_na_dir/.job_status.$$.tmp"
  printf '{"job": "%s", "state": "%s", "exit_code": %s, "host": "%s", "scheduler_id": "%s", "submit_dir": %s, "started": %s, "finished": %s}\n' \
    "$_na_job" "$1" "$2" "${_na_host:-unknown}" "$_na_sched" "$_na_submit" "$_na_started" "$3" \
    > "$tmp" 2>/dev/null && mv -f "$tmp" "$_na_dir/job_status.json" 2>/dev/null
  return 0
}
_na_signal() {   # forward a signal to the run, then exit 128+N
  trap - TERM INT HUP
  if [ -n "$_na_child" ]; then kill -s "$2" "$_na_child" 2>/dev/null; wait "$_na_child" 2>/dev/null; fi
  exit $((128 + $1))
}
trap '_na_status exited $? $(date +%s)' EXIT
trap '_na_signal 15 TERM' TERM; trap '_na_signal 2 INT' INT; trap '_na_signal 1 HUP' HUP
_na_status running null null
"""


def _script(job_cmd: List[str], env: Dict[str, str], job_id: str, result_dir: Path,
            out: Path) -> str:
    """The job: run the command, and record in <result_dir>/job_status.json that
    it started and how it ended -- from the shell, so the record survives a
    Python that never starts, a crash, or a signal (forwarded to the run)."""
    q = shlex.quote
    if not re.fullmatch(r"[A-Za-z0-9._-]+", job_id):
        raise ValueError(f"job id {job_id!r} has characters a job file cannot carry")
    lines = ["#!/usr/bin/env bash",
             "# written by `neuroatlas submit`; runs one (dataset, model) pair",
             "set -uo pipefail"]
    lines += [f"export {k}={q(v)}" for k, v in sorted(env.items())]
    lines += ["export NEUROATLAS_OFFLINE=1 HF_HUB_OFFLINE=1"]
    wrapper = (_WRAPPER.replace("__JOB__", q(job_id)).replace("__DIR__", q(str(result_dir)))
               .replace("__SUBMIT__", q(json.dumps(str(out)))))
    lines += wrapper.strip("\n").splitlines()
    lines += ["", " ".join(q(c) for c in job_cmd) + " &",
              "_na_child=$!",
              'wait "$_na_child"',
              "exit $?"]
    return "\n".join(lines) + "\n"


#: The skip reason of a pair whose dataset's channel map has no entry for the
#: model's family (`check`'s `invalid`). Like an n/a pair it never gets a
#: job, not even with --force.
INVALID_PAIR = "invalid (channel map)"


def forceable(skip: Dict[str, str]) -> bool:
    """Whether --force would write a job for this skipped pair: data or
    weights missing, yes; a channel map that rules it out, never."""
    return not str(skip.get("reason", "")).startswith(("n/a", "invalid"))


def _ready(data_state, model_state) -> Optional[str]:
    """Why a pair cannot run offline, or None."""
    if data_state is not None and not data_state.found and data_state.state != "fetched on first use":
        return f"data {data_state.state}"
    if model_state is not None and not model_state.ready:
        return f"weights {model_state.state}"
    return None


def plan_jobs(benchmark: str, models: str, suite: str, variant: str, out: Path,
              output_root: Optional[Path] = None, *, force: bool = False) -> Dict[str, Any]:
    """Write a script per runnable (dataset, model). A pair whose data or
    weights are not on this machine is skipped with its reason (``force``
    writes it anyway); a pair the channel map rules out -- ``skip`` (n/a) or
    no entry for the family (invalid) -- or a dataset whose map does not
    load, never gets a job."""
    from neuroatlas import catalog, data, models as model_state, run as runmod, selectors
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
    # Always name the results root: the job must write where `status` looks,
    # whatever the node's defaults would be.
    root = Path(output_root) if output_root else _paths.output_dir()
    root_args = ["--output-root", str(root)]
    weights = {i: model_state.status(by_id[i]) for i in ids}

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
            ds_state = data.status(dataset)
            ruled_out, invalid_pairs = runmod.map_states(cmap, [by_id[i] for i in ids])
            job_models = []
            for i in ids:
                if i in ruled_out:
                    skipped.append({"dataset": dataset, "model": i, "reason": "n/a (channel map)"})
                    continue
                if i in invalid_pairs:
                    # no map entry for the family: as `check` and `run` say, invalid
                    skipped.append({"dataset": dataset, "model": i, "reason": INVALID_PAIR,
                                    "detail": cmap.state_for(invalid_pairs[i])[1]})
                    continue
                why = _ready(ds_state, weights[i])
                if why and not force:
                    skipped.append({"dataset": dataset, "model": i, "reason": why})
                    continue
                job_models.append((i, i))
        for label, selector in job_models:
            jid = _job_id(bench.name, dataset, label if label != "*" else "all", variant)
            rdir = runmod.result_dir(bench.name, dataset, variant, root)
            if label != "*":
                rdir = rdir / label
            cmd = [python, "-m", "neuroatlas", "run", bench.name, "-m", selector,
                   "--dataset", dataset, "--variant", variant, *root_args]
            if label != "*":
                cmd.append("--per-model-output")
            script = out / "jobs" / f"{jid}.sh"
            # replace, never rewrite in place: bash reads a running script as it
            # goes, so a job of an earlier submit still running it must keep its copy
            tmp = script.with_name(f".{script.name}.{os.getpid()}.tmp")
            tmp.write_text(_script(cmd, env, jid, rdir, out))
            tmp.chmod(0o755)
            os.replace(tmp, script)
            depends_dir = (str(runmod.result_dir(bench.derived_from, dataset, "default", root))
                           if bench.derived_from else None)
            jobs.append(Job(jid, bench.name, dataset, label, variant, str(rdir), str(script),
                            depends_on=bench.derived_from, depends_dir=depends_dir,
                            models=selector))
    return {"benchmark": bench.name, "suite": suite, "variant": variant, "models": ids,
            "jobs": jobs, "skipped": skipped, "invalid": invalid}


def skip_summary(skipped: List[Dict[str, str]]) -> str:
    """'skipped 171: data missing 165, weights manual 6'."""
    from collections import Counter

    counts = Counter(s.get("reason", "n/a (channel map)") for s in skipped)
    text = f"skipped {len(skipped)}"
    if counts:
        text += ": " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return text


# --------------------------------------------------------------------------
# resources and submit files
# --------------------------------------------------------------------------

_MEM_UNITS = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}


def parse_memory(text: str) -> int:
    """'32G' -> 32768 (MB). A unit (K, M, G or T, optionally followed by B) is
    required: a bare number means MB to HTCondor and something else to SLURM.
    Raises ValueError with the reason."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT])I?B?\s*", str(text).upper())
    if not m:
        if re.fullmatch(r"\s*\d+(\.\d+)?\s*", str(text)):
            raise ValueError(f"{text!r} has no unit: write {text}G or {text}M")
        raise ValueError(f"{text!r} is not a memory size: write e.g. 32G or 1500M")
    mb = float(m.group(1)) * _MEM_UNITS[m.group(2)]
    if mb < 1:
        raise ValueError(f"{text!r} is not a positive memory size")
    return int(-(-mb // 1))                                 # round up to whole MB


def _memory_mb(memory) -> int:
    return memory if isinstance(memory, int) else parse_memory(memory)


def parse_walltime(text: str) -> int:
    """Seconds from '7200', '2:00:00', '1-00:00:00' or '90:00' (MM:SS as SLURM)."""
    s = str(text).strip()
    days = 0
    if "-" in s:
        d, _, s = s.partition("-")
        if not d.isdigit():
            raise ValueError(f"{text!r} is not a wall time")
        days = int(d)
    parts = s.split(":")
    if not all(p.isdigit() for p in parts) or not 1 <= len(parts) <= 3:
        raise ValueError(f"{text!r} is not a wall time: write seconds or H:MM:SS")
    nums = [int(p) for p in parts]
    if len(nums) == 1:
        secs = nums[0] if not days else nums[0] * 3600        # SLURM: D-H
    elif len(nums) == 2:
        secs = nums[0] * 60 + nums[1]
    else:
        secs = nums[0] * 3600 + nums[1] * 60 + nums[2]
    secs += days * 86400
    if secs <= 0:
        raise ValueError(f"{text!r} is not a positive wall time")
    return secs


def slurm_time(seconds: int) -> str:
    d, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return (f"{d}-" if d else "") + f"{h:02d}:{m:02d}:{s:02d}"


def torch_min_capability() -> Tuple[Optional[str], str]:
    """The lowest GPU compute capability the installed torch has kernels for,
    from the architectures it was built for ('sm_75 sm_80 ...' -> '7.5').
    Read from the build, so it works on a submit host without a GPU.
    Returns (capability or None, how it was found)."""
    try:
        import torch
    except ImportError:
        return None, "torch is not installed here"
    flags = None
    try:
        flags = torch._C._cuda_getArchFlags()               # the build's list, GPU or not
    except Exception:
        pass
    archs = flags.split() if flags else list(getattr(torch.cuda, "get_arch_list", lambda: [])())
    caps = []
    for a in archs:
        m = re.fullmatch(r"(?:sm|compute)_(\d+)(\d)[a-z]?", a)
        if m:
            caps.append((int(m.group(1)), int(m.group(2))))
    if not caps:
        return None, f"torch {torch.__version__} has no CUDA kernels (a CPU build?)"
    lo = min(caps)
    return f"{lo[0]}.{lo[1]}", f"torch {torch.__version__} is built for {' '.join(archs)}"


def write_condor(out: Path, queued: List[Job], *, gpus: int, cpus: int, memory,
                 requirements: Optional[str], walltime: Optional[int],
                 extra: List[str], gpu_capability: Optional[str] = None,
                 walltime_attrs: Tuple[str, ...] = ("RequestWalltime", "MaxRuntime")) -> Path:
    (out / "jobs.txt").write_text("".join(f"{j.script}\n" for j in queued))
    mem_mb = _memory_mb(memory)
    reqs = [f"({requirements})"] if requirements else []
    if gpus and gpu_capability:
        reqs.append(f"(GPUs_Capability >= {gpu_capability})")
    lines = [
        "# written by `neuroatlas submit`",
        "universe       = vanilla",
        "executable     = $(script)",
        "getenv         = False",
        f"request_cpus   = {cpus}",
        f"request_memory = {mem_mb}",
        f"request_gpus   = {gpus}" if gpus else "",
        f"requirements   = {' && '.join(reqs)}" if reqs else "",
        *[f"+{attr} = {walltime}" for attr in walltime_attrs if walltime],
        # names each job in the queue and in every event of the log, for `status`
        f'+NeuroatlasTag = "{tag(out)}"',
        '+NeuroatlasJob = "$(script_name)"',
        "job_ad_information_attrs = NeuroatlasJob",
        'environment    = "NEUROATLAS_SCHED_ID=$(Cluster).$(Process)"',
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


def slurm_batch_name(out: Path, queued: List[Job]) -> str:
    digest = hashlib.sha1("\n".join(j.id for j in queued).encode()).hexdigest()[:6]
    return f"{tag(out)}-{digest}"


def write_slurm(out: Path, queued: List[Job], *, gpus: int, cpus: int, memory,
                time: str, partition: Optional[str], no_mem: bool, extra: List[str],
                gpu_capability: Optional[str] = None) -> Path:
    """The scripts are listed inside the batch file, which sbatch copies at
    submission: a later submit to the same folder cannot change what a pending
    array task runs, and `status` maps task N back to its job."""
    (out / "jobs.txt").write_text("".join(f"{j.script}\n" for j in queued))
    mem = _memory_mb(memory)
    mem_arg = f"{mem // 1024}G" if mem % 1024 == 0 else f"{mem}M"
    name = slurm_batch_name(out, queued)
    extra = [e[len("#SBATCH"):].strip() if e.startswith("#SBATCH") else e for e in extra]
    lines = ["#!/usr/bin/env bash", "# written by `neuroatlas submit`",
             f"#SBATCH --job-name={name}",
             f"#SBATCH --array=0-{max(len(queued) - 1, 0)}",
             f"#SBATCH --cpus-per-task={cpus}",
             f"#SBATCH --gpus={gpus}" if gpus else "",
             "" if no_mem else f"#SBATCH --mem={mem_arg}",
             f"#SBATCH --time={time}",
             f"#SBATCH --partition={partition}" if partition else "",
             *[f"#SBATCH {e}" for e in extra],
             f"#SBATCH --output={out}/logs/%A_%a.out",
             f"#SBATCH --error={out}/logs/%A_%a.err",
             (f"# the installed torch needs GPU compute capability >= {gpu_capability}; "
              f"ask for such a GPU with your site's --constraint/--gres") if gpus and gpu_capability else "",
             "set -euo pipefail",
             "scripts=(",
             *[f"  {shlex.quote(j.script)}" for j in queued],
             ")",
             'export NEUROATLAS_SCHED_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"',
             'exec "${scripts[$SLURM_ARRAY_TASK_ID]}"']
    path = out / "jobs.sbatch"
    path.write_text("\n".join(l for l in lines if l) + "\n")
    return path


def save_manifest(out: Path, planned: Dict[str, Any], backend: str, mode: str,
                  batch: Optional[Tuple[str, List[str]]] = None) -> Path:
    """``batch`` (SLURM): the array's job name and the job of each task index;
    earlier batches of this folder are kept so `status` can still map them."""
    path = out / "jobs.json"
    batches: Dict[str, List[str]] = {}
    try:
        batches = dict(json.loads(path.read_text()).get("batches") or {})
    except (OSError, ValueError):
        pass
    if batch:
        batches[batch[0]] = batch[1]
    payload = {**planned, "backend": backend, "mode": mode, "tag": tag(out),
               "batches": batches, "jobs": [asdict(j) for j in planned["jobs"]]}
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_manifest(out: Path) -> Dict[str, Any]:
    path = out / "jobs.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist: run `neuroatlas submit ... --out {out}` first")
    data = json.loads(path.read_text())
    known = set(Job.__dataclass_fields__)
    data["jobs"] = [Job(**{k: v for k, v in j.items() if k in known}) for j in data["jobs"]]
    data.setdefault("batches", {})
    return data
