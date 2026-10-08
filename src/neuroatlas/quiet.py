"""What would flood the terminal, kept to a line.

A long run repeats itself: a backbone meets the same missing statistic on
every batch, a probe's solver stops at ``max_iter`` fold after fold, a reader
skips the same kind of recording in every split. Each is said once per
command:

* :func:`warn_once` -- the first time, as a warning; every later time at
  DEBUG (``-v`` and ``--log``);
* :func:`count` -- nothing at the time (DEBUG), then one line at the end of
  the command with the count (``4 of 15 probe fits ...``), from
  :func:`summarize`, which the ``neuroatlas`` command calls on its way out.

Libraries that print on their own -- MNE (to stdout: the "FIR filter
parameters" block of every recording it filters), transformers and
huggingface_hub (to stderr) -- are routed through our logging instead
(:func:`route_libraries`): their lines are then hidden on screen unless
``-v``, counted in the command's closing "N log lines not shown" note, and
kept by ``--log``. :func:`printed_to_log` does the same for code that
``print()``\\ s (REVE's remote code says "flash_attn not found" at load).

Work a command would repeat -- every model and fold of a BCI `run` loaded
and filtered the same cohort again -- can be kept for the command's length
in a :func:`per_command` store.

Standard library only: the CLI imports it before anything heavy.
"""
from __future__ import annotations

import contextlib
import importlib.abc
import logging
import os
import sys
from typing import Callable, Dict, Iterable, List, Optional, Union

logger = logging.getLogger("neuroatlas.repeats")

# --------------------------------------------------------------------------
# once per command
# --------------------------------------------------------------------------

#: message key -> times it came up in this command
_SEEN: Dict[str, int] = {}


def once(key: str) -> bool:
    """True the first time *key* comes up in this command, False after."""
    seen = _SEEN.get(key, 0)
    _SEEN[key] = seen + 1
    return seen == 0


def warn_once(log: logging.Logger, key: str, message: str, *args,
              level: int = logging.WARNING) -> bool:
    """Log *message* at *level* the first time *key* comes up in this
    command, at DEBUG every later time. True when it was the first."""
    first = once(key)
    log.log(level if first else logging.DEBUG, message, *args)
    return first


class _Count:
    """One end-of-command line: how many of how many."""

    def __init__(self, text: str, fix: Optional[str], level: int):
        self.text, self.fix, self.level = text, fix, level
        self.hit: Union[int, set] = 0
        self.of: Union[int, set] = 0

    @staticmethod
    def _add(have, more):
        if isinstance(more, int):
            return (len(have) if isinstance(have, set) else have) + more
        items = set(have) if isinstance(have, set) else set()
        return items | set(more)

    @staticmethod
    def _n(value) -> int:
        return len(value) if isinstance(value, set) else int(value)


_COUNTS: Dict[str, _Count] = {}


def count(key: str, text: str, *, hit: Union[int, Iterable] = 1,
          of: Union[int, Iterable] = 0, fix: Optional[str] = None,
          level: int = logging.WARNING) -> None:
    """Add to a count said once, at the end of the command: *text* with
    ``{hit}`` and ``{of}`` filled in (``{hit} of {of} probe fits stopped at
    max_iter``). *hit* and *of* are numbers to add, or items (recording
    paths) counted once however often they come up. A count whose *hit*
    stays 0 says nothing."""
    entry = _COUNTS.get(key)
    if entry is None:
        entry = _COUNTS[key] = _Count(text, fix, level)
    entry.hit = entry._add(entry.hit, hit)
    entry.of = entry._add(entry.of, of)


def summarize() -> List[str]:
    """Say each count once (a log record at its level, with its ``fix:``),
    and forget them; returns the lines said."""
    said = []
    for entry in list(_COUNTS.values()):
        hit, of = entry._n(entry.hit), entry._n(entry.of)
        if not hit:
            continue
        text = entry.text.format(hit=hit, of=of, s="" if hit == 1 else "s")
        logger.log(entry.level, "%s", text + (f"\nfix: {entry.fix}" if entry.fix else ""))
        said.append(text)
    _COUNTS.clear()
    return said


def reset() -> None:
    """A new command: nothing said yet, nothing kept from the last one."""
    _SEEN.clear()
    _COUNTS.clear()
    for store in _PER_COMMAND:
        store.clear()


# --------------------------------------------------------------------------
# work one command repeats
# --------------------------------------------------------------------------

#: Stores whose contents live for one ``neuroatlas`` command (see per_command).
_PER_COMMAND: List[dict] = []
_COMMAND = {"active": False}


def per_command(store: dict) -> dict:
    """Register *store* as kept for one command only: emptied when a
    command starts and ends. What `run` would otherwise redo for every model
    and fold (a BCI cohort's load) goes in one; outside a command (the API,
    a test) :func:`in_command` is False and nothing should be reused."""
    _PER_COMMAND.append(store)
    return store


def in_command() -> bool:
    """Whether a ``neuroatlas`` command is running in this process."""
    return _COMMAND["active"]


def begin_command() -> None:
    reset()
    _COMMAND["active"] = True


def end_command() -> None:
    _COMMAND["active"] = False
    reset()


# --------------------------------------------------------------------------
# libraries that print on their own
# --------------------------------------------------------------------------

#: logger -> the module whose import gives it its own console handler
LIBRARIES = {
    "mne": "mne",
    "transformers": "transformers.utils.logging",
    "huggingface_hub": "huggingface_hub.utils.logging",
}

#: Top-level logger names that are this package's own (the vendored readers
#: log under physioex.*; one builder under its script's name).
OWN_LOGGERS = ("neuroatlas", "physioex", "preprocess_sz1")


def is_library(record: logging.LogRecord) -> bool:
    """A record from a third-party library (not ours, not a Python warning)."""
    top = record.name.split(".", 1)[0]
    return record.name not in ("root", "py.warnings") and top not in OWN_LOGGERS


class _AfterImport(importlib.abc.MetaPathFinder):
    """Runs a callback right after a module is first imported: the libraries
    set up their console handler at import, after the command started."""

    def __init__(self, hooks: Dict[str, Callable]):
        self.hooks = dict(hooks)

    def find_spec(self, fullname, path, target=None):
        hook = self.hooks.get(fullname)
        if hook is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        spec.loader = _Then(spec.loader, hook)
        return spec


class _Then(importlib.abc.Loader):
    def __init__(self, loader, hook):
        self._loader, self._hook = loader, hook

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        try:
            self._hook(module)
        except Exception:                               # never break an import
            pass

    def __getattr__(self, name):                        # get_source, resource readers, ...
        return getattr(self._loader, name)


class _Routed(logging.Logger):
    """A library's logger while routed: it propagates, whatever the library
    sets later -- MNE turns propagation off again in each module that
    imports its logger (``mne/utils/mixin.py``, imported lazily, mid-run)."""

    @property
    def propagate(self):
        return True

    @propagate.setter
    def propagate(self, value):
        pass


class _Routing:
    """The libraries' loggers while a command runs, and how to put them back."""

    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.saved: Dict[str, tuple] = {}
        self.finder: Optional[_AfterImport] = None
        # MNE's level the user chose (the variable; MNE's own config file is
        # read by MNE itself on every call)
        self.mne_level_chosen = "MNE_LOGGING_LEVEL" in os.environ

    def route(self, name: str, module=None) -> None:
        lib = logging.getLogger(name)
        if name == "transformers" and module is not None:
            # its handler is added on first use: add it now, to take it off
            configure = getattr(module, "_configure_library_root_logger", None)
            if callable(configure):
                configure()
        if name not in self.saved:
            self.saved[name] = (list(lib.handlers), lib.propagate, lib.level, type(lib))
        for h in list(lib.handlers):
            # its own console handler (MNE: stdout; the others: stderr); a
            # file the user asked the library for stays
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                lib.removeHandler(h)
        if type(lib) is logging.Logger:
            lib.__class__ = _Routed
        else:
            lib.propagate = True
        if name == "mne" and not self.mne_level_chosen:
            # its INFO lines are made, to be counted and kept by --log (not
            # its DEBUG, even with -v: noise)
            lib.setLevel(logging.INFO)

    def undo(self) -> None:
        if self.finder is not None and self.finder in sys.meta_path:
            sys.meta_path.remove(self.finder)
        for name, (handlers, propagate, level, cls) in self.saved.items():
            lib = logging.getLogger(name)
            if type(lib) is _Routed:
                lib.__class__ = cls
            for h in handlers:
                if h not in lib.handlers:
                    lib.addHandler(h)
            lib.propagate, lib.level = propagate, level
        self.saved.clear()


def route_libraries(verbose: bool) -> Callable[[], None]:
    """Route MNE, transformers and huggingface_hub through our logging for
    one command; returns what puts them back.

    Those imported later are routed as they are imported; a forked loader
    worker inherits the routing. ``MNE_LOGGING_LEVEL`` is deliberately not
    set: MNE's ``verbose`` decorator reads it on every call, so WARNING there
    would drop the lines rather than route them, and neither the count nor
    ``--log`` would have them (a user's own setting is kept)."""
    routing = _Routing(verbose)
    later = {}
    for name, module in LIBRARIES.items():
        if module in sys.modules:
            routing.route(name, sys.modules[module])
        else:
            later[module] = (lambda m, _name=name: routing.route(_name, m))
    if later:
        routing.finder = _AfterImport(later)
        sys.meta_path.insert(0, routing.finder)
    return routing.undo


# --------------------------------------------------------------------------
# code that prints
# --------------------------------------------------------------------------

class _LogLines:
    """A stdout whose lines become log records."""

    def __init__(self, log: logging.Logger, level: int):
        self._log, self._level, self._buffer = log, level, ""

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._log.log(self._level, "%s", line.rstrip())
        return len(text)

    def flush(self):
        if self._buffer.strip():
            self._log.log(self._level, "%s", self._buffer.rstrip())
        self._buffer = ""

    def isatty(self):
        return False


@contextlib.contextmanager
def printed_to_log(log: logging.Logger, level: int = logging.INFO):
    """What the code inside prints to stdout becomes log lines (-v, --log)."""
    lines = _LogLines(log, level)
    with contextlib.redirect_stdout(lines):
        try:
            yield
        finally:
            lines.flush()
