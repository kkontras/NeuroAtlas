"""What a long operation is doing, and that it is moving.

Every command that works for more than a few seconds -- probing, embedding,
downloading data or weights -- prints the same three kinds of line:

``header``
    what it is about to do, at once::

        embedding 12 model(s) on siena: 12 run(s)

``live line``
    while one item works, on a terminal only: one line, rewritten in place
    every second or two and cleared when the item ends. It names the phase
    the item is in, and when it is known how far along it is::

        [3/12] siena eegpt_pretrained: loading weights (0m 12s)
        [3/12] siena eegpt_pretrained: embedding 45% (352/793 batches, 1m 05s, ~1m 20s left)
        [2/3] neurogpt_pretrained: downloading 45% (143 MB of 318 MB, 12.1 MB/s, 0m 12s, ~0m 14s left)

    Off a terminal (a pipe, a cluster job's output file) there is no live
    line: a start line instead (``[3/12] siena eegpt_pretrained: embedding``),
    then one line per tenth of the work, or every few minutes when the total
    is not known.

``result line``
    one per item when it ends, whatever the log level::

        [3/12] siena eegpt_pretrained: ok, 50,749 windows (6m 07s)

Header and result lines are ordinary output: stdout, and a ``--log`` file.
The live line and the off-terminal progress lines go to the terminal (or the
job's output) only, never into ``--log``; a machine format never sees them.
The live line erases itself with an escape code only where colour is allowed
(a terminal, ``NO_COLOR`` unset, ``TERM`` not ``dumb``) and with spaces
elsewhere, and it is cut to the terminal's width, so it never wraps.

Code deep inside an item -- the extraction loop, a download -- reports to it
through :func:`current`, which does nothing outside one. While a live line is
on screen, anything else written to stdout or stderr (a warning, a library's
print) first clears it, so every message starts on a clean line.

Standard library only: the engine, the downloaders and the CLI all import it.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

# --------------------------------------------------------------------------
# numbers as the lines show them
# --------------------------------------------------------------------------


def clock(seconds: float) -> str:
    """``1m 05s`` (``4h 47m`` from an hour on): the time on a live line."""
    minutes, secs = divmod(int(max(0.0, seconds)), 60)
    if minutes >= 60:
        return f"{minutes // 60}h {minutes % 60:02d}m"
    return f"{minutes}m {secs:02d}s"


def duration(seconds: float) -> str:
    """``4s``, ``2m 05s`` or ``1h 15m``: an item's time on its result line."""
    minutes, secs = divmod(int(round(max(0.0, seconds))), 60)
    if minutes >= 60:
        return f"{minutes // 60}h {minutes % 60:02d}m"
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def size(n: float) -> str:
    """``318 MB``, ``1.2 GB``: bytes in decimal units, as downloads count them."""
    n = float(n)
    for unit in ("B", "kB", "MB", "GB"):
        if abs(n) < 1000:
            if unit == "B":
                return f"{n:.0f} B"
            return f"{n:.1f} {unit}" if abs(n) < 10 else f"{n:.0f} {unit}"
        n /= 1000
    return f"{n:.1f} TB" if abs(n) < 10 else f"{n:.0f} TB"


def result_line(done: int, total: int, where: str, what: str,
                seconds: Optional[float] = None) -> str:
    """``[3/12] siena eegpt_pretrained: ok, 50,749 windows (6m 07s)``; without
    *seconds*, no time (``[1/3] labram_pretrained: found``)."""
    line = f"[{done}/{total}] {where}: {what}"
    return line if seconds is None else f"{line} ({duration(seconds)})"


def say(line: str) -> None:
    """Print a header or result line: stdout (and so --log), at once."""
    print(line, flush=True)


# --------------------------------------------------------------------------
# where a line may go
# --------------------------------------------------------------------------

#: Return to the start of the line and erase it (where escape codes are allowed).
CLEAR_LINE = "\r\x1b[K"


def terminal(stream):
    """The stream itself under a ``--log`` tee (``_stream``) or an output
    guard (``_inner``): live lines are for the screen only."""
    for _ in range(8):
        own = getattr(stream, "__dict__", None) or {}
        inner = own.get("_inner") or own.get("_stream")
        if inner is None:
            return stream
        stream = inner
    return stream


def isatty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def escapes_allowed(stream) -> bool:
    """A terminal that takes escape codes: the same rule as colour."""
    try:
        from neuroatlas.cli._msg import use_color
    except Exception:                                   # pragma: no cover
        return False
    return use_color(stream)


def _width(stream) -> int:
    """The terminal's width; $COLUMNS, else 100, when it does not say (a
    pseudo-terminal can report 0)."""
    try:
        columns = os.get_terminal_size(stream.fileno()).columns
    except (AttributeError, ValueError, OSError):
        columns = 0
    if columns <= 0:
        try:
            columns = int(os.environ.get("COLUMNS", ""))
        except ValueError:
            columns = 100
    return max(20, columns)


# --------------------------------------------------------------------------
# bytes on disk, for tools that report nothing themselves
# --------------------------------------------------------------------------

class DiskBytes:
    """Bytes under some folders, beyond what was there at the start: what a
    tool that reports nothing (MOABB, nsrr, a hub download) has written so far.

    Cheap or nothing: a walk that took *t* seconds is not repeated for 20 *t*
    seconds, and after a walk longer than ``give_up`` seconds (a huge tree on
    a slow filesystem) it stops walking and reports None."""

    def __init__(self, *roots, give_up: float = 5.0):
        self.roots = [Path(r) for r in roots if r]
        self.give_up = give_up
        self._next = 0.0
        self._last: Optional[int] = None
        self._off = False
        self.baseline = self._walk() or 0

    def _walk(self) -> Optional[int]:
        started = time.monotonic()
        total = 0
        for root in self.roots:
            if root.is_file():
                total += root.stat().st_size
                continue
            for folder, _, files in os.walk(root):
                for name in files:
                    try:
                        total += os.lstat(os.path.join(folder, name)).st_size
                    except OSError:
                        pass
                if time.monotonic() - started > self.give_up:
                    self._off = True
                    return None
        took = time.monotonic() - started
        self._next = time.monotonic() + max(1.0, 20 * took)
        return total

    def __call__(self) -> Optional[int]:
        if self._off:
            return None
        if time.monotonic() >= self._next:
            seen = self._walk()
            if seen is not None:
                self._last = max(0, seen - self.baseline)
        return self._last


# --------------------------------------------------------------------------
# one item's progress
# --------------------------------------------------------------------------

_STACK: List["Progress"] = []


class _Null:
    """What :func:`current` returns outside an item: every call does nothing."""

    counts: Dict[str, int] = {}
    total = None
    phase_text = None

    def add_total(self, *args, **kwargs):
        return self

    def phase(self, *args, **kwargs):
        return self

    def update(self, *args, **kwargs):
        return self

    def count(self, *args, **kwargs):
        return self

    def watch(self, *args, **kwargs):
        return self

    def relabel(self, *args, **kwargs):
        return self


_NULL = _Null()


def current():
    """The item running now (the innermost), or a stand-in that ignores every call.
    A command's starting line (:func:`starting`) is not an item."""
    for item in reversed(_STACK):
        if not item.until_output:
            return item
    return _NULL


def starting(command: str) -> "Progress":
    """Around a whole command, on a terminal: "neuroatlas embed: starting (0m
    03s)" while it imports and plans, from its first second until it prints
    anything (its header), when it is erased for good. Nothing off a
    terminal."""
    return Progress(f"neuroatlas {command}", verb="starting", phases=False,
                    start_line_off_tty=False, until_output=True)


def make_way() -> None:
    """Clear the live line, if one is on screen, before something else is written."""
    for item in list(_STACK):
        item._make_way()


class LineSafe:
    """A stream that clears the live line before each write: for a writer
    that holds the stream from before any item started (the console's
    logging handler), which the stdout/stderr guard cannot see."""

    def __init__(self, stream):
        self._stream = stream

    def write(self, text):
        if text:
            make_way()
        return self._stream.write(text)

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _Guard:
    """sys.stdout or sys.stderr while a live line is on screen: whatever else
    is written first clears the live line, so it starts on a clean line (the
    next tick draws the live line again)."""

    def __init__(self, inner, item: "Progress"):
        self._inner = inner
        self._item = item

    def write(self, text):
        if text and os.getpid() == self._item._pid:      # not from a forked loader worker
            self._item._make_way()
        return self._inner.write(text)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class Progress:
    """One item's progress: a context manager around the item's work.

    ``label`` names the item (``[3/12] siena eegpt_pretrained``); ``verb`` is
    what it shows before any phase (``probing``, ``embedding``,
    ``downloading``). With ``phases=False`` the line keeps saying the verb
    whatever the code inside reports (the probe's line).

    Inside, :meth:`phase` names what it is doing now -- with a total and a
    unit when it is countable -- :meth:`update` moves it on, :meth:`watch`
    gives it a callable to poll (bytes on disk), and :meth:`count` adds to
    the tallies its result line reads (``windows``, ``bytes``, ...).
    """

    def __init__(self, label: str, *, verb: str = "working", stream=None, every: float = 1.0,
                 start_line_off_tty: bool = True, phases: bool = True, steps: int = 10,
                 quiet_every: float = 300.0, min_gap: float = 5.0,
                 until_output: bool = False):
        self.label, self.verb, self.every = label, verb, every
        # a command's "starting" line: drawn from its first second on, and
        # gone for good at the command's first output (see starting())
        self.until_output = until_output
        self._retired = False
        self.stream = terminal(stream if stream is not None else sys.stdout)
        self.tty = isatty(self.stream)
        self.escapes = escapes_allowed(self.stream)
        self.start_line_off_tty = start_line_off_tty
        self.phases = phases
        self.steps = max(1, int(steps))
        self.quiet_every = quiet_every
        # off a terminal, a tenth's line at most this often: a phase that
        # takes a second prints none
        self.min_gap = min_gap
        self.counts: Dict[str, int] = {}
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._guards: List[tuple] = []
        self._shown = 0                    # characters of the live line on screen (0: none)
        self._last_draw = 0.0
        self._started = time.monotonic()
        self._reset_phase(None)

    # -- state -------------------------------------------------------------

    def _reset_phase(self, text: Optional[str], *, total=None, done: float = 0,
                     unit: Optional[str] = None, approx: bool = False, eta: bool = True):
        self._phase = text
        self._total = total if total else None
        self._total_unknown = False
        self._eta = eta
        self._done = done or 0
        self._start_done = self._done
        self._unit = unit
        self._approx = approx
        self._note: Optional[str] = None
        self._phase_started = time.monotonic()
        # (time, done) at the first move: the rate and the time left count
        # from there, not from the phase's start (a loader's first batch, a
        # connection, take their time once)
        self._first: Optional[tuple] = None
        self._poll: Optional[Callable[[], Optional[float]]] = None
        self._polled: Optional[float] = None
        self._next_step = self._step_of(self._done) + 1
        self._last_line = time.monotonic()

    def _step_of(self, done) -> int:
        if not self._total:
            return 0
        return int(self.steps * min(1.0, float(done) / float(self._total)))

    def phase(self, text: str, *, total: Optional[float] = None, done: float = 0,
              unit: Optional[str] = None, approx: bool = False, eta: bool = True) -> "Progress":
        """Start a phase: ``loading weights``; ``embedding`` with ``total=793,
        unit="batches"``; ``downloading`` with ``unit="bytes"``. ``approx``:
        the total is an estimate (shown as ``~1.1 GB``). ``eta=False``: no
        time left (units of very different sizes, git's objects)."""
        if not self.phases:
            return self
        with self._lock:
            self._reset_phase(text, total=total, done=done, unit=unit, approx=approx, eta=eta)
        self._draw(force=True)
        return self

    def update(self, done: Optional[float] = None, *, advance: float = 0,
               total: Optional[float] = None, carried: float = 0,
               note: Optional[str] = None) -> "Progress":
        """Move the phase on: to *done*, or by *advance*. ``carried``: how much
        of that was there before (a file already complete, the part of one a
        resume skips): it counts toward the percent, not toward the rate or
        the time left. ``note``: a word on what it is at (``z.zip 2/5``),
        shown first in the brackets."""
        if not self.phases:
            return self
        with self._lock:
            if total:
                self._total = total
            self._done = (done if done is not None else self._done) + advance
            self._start_done += carried
            if note is not None:
                self._note = note or None
            self._mark_first(self._done)
        self._after_update()
        return self

    def add_total(self, n: Optional[float]) -> "Progress":
        """Add *n* to the phase's total (files that start one by one); an
        unknown size (None) makes the whole total unknown."""
        if self.phases:
            with self._lock:
                if not n:
                    self._total, self._total_unknown = None, True
                elif not self._total_unknown:
                    self._total = (self._total or 0) + n
        return self

    def _mark_first(self, done: float) -> None:
        if self._first is None and done > self._start_done:
            self._first = (time.monotonic(), done)

    def watch(self, poll: Callable[[], Optional[float]]) -> "Progress":
        """Poll *poll* (bytes so far, or None) on each tick: the phase's
        progress when its unit is bytes, else shown beside it."""
        if self.phases:
            with self._lock:
                self._poll = poll
        return self

    def count(self, key: str, n: int = 1) -> "Progress":
        """Add *n* to a tally the result line reads (``windows``, ``bytes``, ...)."""
        with self._lock:                     # downloads count from several threads
            self.counts[key] = self.counts.get(key, 0) + int(n)
        return self

    @property
    def total(self) -> Optional[float]:
        """The phase's total, when it has one."""
        return self._total

    @property
    def phase_text(self) -> Optional[str]:
        """The phase the item is in (None before the first)."""
        return self._phase

    def relabel(self, label: str) -> "Progress":
        """Name the item anew (a fold within it, a file within a dataset)."""
        with self._lock:
            self.label = label
        self._draw(force=True)
        return self

    # -- text --------------------------------------------------------------

    def text(self, now: Optional[float] = None) -> str:
        """The live line now, e.g. ``[3/12] siena eegpt: embedding 45% (352/793
        batches, 1m 05s, ~1m 20s left)``."""
        now = time.monotonic() if now is None else now
        with self._lock:
            what = self._phase or self.verb
            elapsed = now - (self._phase_started if self._phase else self._started)
            total, unit, approx = self._total, self._unit, self._approx
            done = self._done
            polled = self._polled
            if unit == "bytes" and polled is not None:
                done = max(done, polled)
                polled = None
            moved = done - self._start_done
            note = self._note
            first = self._first
        # the pace since the first move, once there has been a second; before
        # that, over the whole phase, if it has moved enough to say (a 1-byte
        # file appearing is not a rate)
        if first is not None and done > first[1] and now > first[0]:
            pace = (done - first[1]) / (now - first[0])
        elif moved > 0 and elapsed > 0 and (not total or moved >= 0.01 * total):
            pace = moved / elapsed
        else:
            pace = 0.0
        parts: List[str] = [note] if note else []
        head = what
        if total and done > total * 1.05:
            total = None                    # it was not the whole (an estimate, a missing size)
        if total and unit:
            frac = min(1.0, float(done) / float(total))
            pct = int(frac * 100)
            if done < total:
                pct = min(pct, 99)
            head = f"{what} {pct}%"
            if unit == "bytes":
                parts.append(f"{size(done)} of {'~' if approx else ''}{size(total)}")
            else:
                parts.append(f"{int(done):,}/{'~' if approx else ''}{int(total):,} {unit}")
        elif unit == "bytes" and done:
            head = f"{what} {size(done)}"
        elif unit and done:
            parts.append(f"{int(done):,} {unit}")
        if polled:
            parts.append(f"{size(polled)} so far")
        running = not total or done < total
        if unit == "bytes" and pace > 0 and elapsed >= 1 and running:
            parts.append(f"{size(pace)}/s")
        parts.append(clock(elapsed))
        if total and unit and pace > 0 and done < total and elapsed >= 3 and self._eta:
            parts.append(f"~{clock((total - done) / pace)} left")
        return f"{self.label}: {head} ({', '.join(parts)})"

    # -- output ------------------------------------------------------------

    def _active(self) -> bool:
        # only the innermost item draws: two live lines would fight
        return bool(_STACK) and _STACK[-1] is self

    def _clear_text(self) -> str:
        if self.escapes:
            return CLEAR_LINE
        return "\r" + " " * self._shown + "\r"

    def _write(self, text: str) -> None:
        if os.getpid() != self._pid:          # a forked loader worker never writes the line
            return
        try:
            self.stream.write(text)
            self.stream.flush()
        except (OSError, ValueError):                   # a closed or broken terminal
            pass

    def _draw(self, force: bool = False) -> None:
        if not self.tty or self._stop.is_set() or self._retired or not self._active():
            return
        now = time.monotonic()
        if not force and now - self._last_draw < 0.25:
            return
        line = self.text(now)
        width = _width(self.stream) - 1
        if len(line) > width:
            line = line[: max(0, width - 3)] + "..."
        with self._lock:
            prefix = self._clear_text() if self._shown else ""
            self._write(prefix + line)
            self._shown = len(line)
            self._last_draw = now

    def _make_way(self) -> None:
        """Clear the live line before someone else writes."""
        if not self.tty or os.getpid() != self._pid:     # not from a forked loader worker
            return
        with self._lock:
            if self.until_output:
                self._retired = True
            if self._shown:
                self._write(self._clear_text())
                self._shown = 0

    def _line(self, text: str) -> None:
        """A whole line, off a terminal."""
        with self._lock:
            self._write(text + "\n")
            self._last_line = time.monotonic()

    def _effective_done(self) -> float:
        polled = self._polled if self._unit == "bytes" else None
        return max(self._done, polled or 0)

    def _after_update(self) -> None:
        if self.tty:
            self._draw()
            return
        with self._lock:
            done = self._effective_done()
            step = self._step_of(done)
            due = (bool(self._total) and step >= self._next_step and done < self._total
                   and time.monotonic() - self._last_line >= self.min_gap)
            if due:
                self._next_step = step + 1
        if due and self._active():
            self._line(self.text())

    def _poll_now(self) -> None:
        poll = self._poll
        if poll is None:
            return
        try:
            value = poll()
        except Exception:                               # a watch never breaks the work
            value = None
        with self._lock:
            if poll is self._poll:
                self._polled = value
                if value is not None and self._unit == "bytes":
                    self._mark_first(value)

    def _tick(self) -> None:
        while not self._stop.wait(self.every):
            self._poll_now()
            if self.tty:
                self._draw(force=True)
            elif self._active():
                # off a terminal: a line now and then when there are no
                # tenths to count (an unknown total), so a log shows life
                with self._lock:
                    quiet = (not self._total and self._phase is not None
                             and time.monotonic() - self._last_line >= self.quiet_every)
                if quiet:
                    self._line(self.text())
                elif self._total and self._unit == "bytes":
                    self._after_update()

    def __enter__(self) -> "Progress":
        _STACK.append(self)
        self._started = time.monotonic()
        self._reset_phase(None)
        if self.tty:
            # at once: a starting line too (importing the engine holds the
            # interpreter for seconds, so a ticking thread may not get a turn)
            self._draw(force=True)
            # anything else printed meanwhile clears the live line first
            for name in ("stdout", "stderr"):
                inner = getattr(sys, name)
                guard = _Guard(inner, self)
                setattr(sys, name, guard)
                self._guards.append((name, guard, inner))
        elif self.start_line_off_tty:
            self._line(f"{self.label}: {self.verb}")
        if self.tty or self.phases:
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        for name, guard, inner in reversed(self._guards):
            if getattr(sys, name) is guard:
                setattr(sys, name, inner)
        self._guards.clear()
        if self.tty:
            with self._lock:
                # cleared for the result line that follows
                self._write(CLEAR_LINE if self.escapes else
                            "\r" + " " * self._shown + "\r")
                self._shown = 0
        if self in _STACK:
            _STACK.remove(self)
        return False

    @property
    def seconds(self) -> float:
        """Time since the item started."""
        return time.monotonic() - self._started
