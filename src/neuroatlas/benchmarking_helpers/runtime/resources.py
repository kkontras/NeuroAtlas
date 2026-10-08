"""How many CPUs this process may really use, and what to do with them.

``os.cpu_count()`` counts the machine, not the job: on a 64-core node where a
scheduler gave the job 4 cores (through CPU affinity or a cgroup quota) it
still says 64, so a loader sized from it starts 63 workers that fight over 4
cores, and BLAS starts 64 threads per call. Everything that sizes a worker or
thread pool asks :func:`available_cpus` instead.
"""
from __future__ import annotations

import contextlib
import math
import os
from pathlib import Path
from typing import Iterator, Optional

# Loader workers beyond this rarely help: a window loader is bound by EDF
# decoding per recording, and each worker holds its own decoded recordings.
MAX_DEFAULT_WORKERS = 16

# BLAS threads for the probes. A probe is a sequence of small dense solves
# (lbfgs over a few hundred features); past a handful of threads OpenBLAS
# spends its time synchronising, and on a busy node it spin-waits against the
# other jobs. Measured on deanston (28 cores): see the F-073 note in the
# report that introduced this constant.
MAX_PROBE_THREADS = 8


def _cgroup_cpu_limit() -> Optional[float]:
    """The cgroup CPU quota in cores, or None when there is none."""
    # cgroup v2
    try:
        text = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if text and text[0] != "max":
            return float(text[0]) / float(text[1])
    except (OSError, ValueError, IndexError, ZeroDivisionError):
        pass
    # cgroup v1
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def available_cpus() -> int:
    """CPUs this process may use: affinity, cgroup quota and scheduler hints.

    The smallest of: the CPU affinity mask (what SLURM and most container
    runtimes set), the cgroup v1/v2 quota, ``SLURM_CPUS_PER_TASK``, and inside
    an HTCondor job the ``OMP_NUM_THREADS`` HTCondor exports from
    ``request_cpus``. Never less than 1.
    """
    counts = []
    try:
        counts.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        counts.append(os.cpu_count() or 1)
    quota = _cgroup_cpu_limit()
    if quota is not None:
        counts.append(max(1, math.floor(quota)))
    for var in ("SLURM_CPUS_PER_TASK",):
        try:
            counts.append(int(os.environ[var]))
        except (KeyError, ValueError):
            pass
    if os.environ.get("_CONDOR_SCRATCH_DIR"):
        try:
            counts.append(int(os.environ["OMP_NUM_THREADS"]))
        except (KeyError, ValueError):
            pass
    return max(1, min(c for c in counts if c > 0))


def default_num_workers() -> int:
    """Data-loader workers when nobody said: one core for the main process,
    the rest for loading, at most :data:`MAX_DEFAULT_WORKERS`."""
    return max(0, min(available_cpus() - 1, MAX_DEFAULT_WORKERS))


def probe_threads() -> int:
    return max(1, min(available_cpus(), MAX_PROBE_THREADS))


@contextlib.contextmanager
def limit_probe_threads(n: Optional[int] = None) -> Iterator[int]:
    """Cap BLAS/OpenMP threads (numpy, scipy, sklearn, torch) inside the block.

    ``threadpoolctl`` covers every BLAS and OpenMP runtime loaded in the
    process, including the second OpenBLAS that scipy wheels bundle; torch's
    intra-op pool is set separately because it does not go through them.
    """
    n = int(n or probe_threads())
    torch_prev = None
    try:
        import torch

        torch_prev = torch.get_num_threads()
        torch.set_num_threads(n)
    except Exception:
        torch_prev = None
    try:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:                       # sklearn depends on it; be safe anyway
            yield n
            return
        with threadpool_limits(limits=n):
            yield n
    finally:
        if torch_prev is not None:
            try:
                import torch

                torch.set_num_threads(torch_prev)
            except Exception:
                pass
