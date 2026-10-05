"""The ``neuroatlas`` command.

    neuroatlas [-v] [--log FILE] [--online] <command> [options]

Each command is a module under ``neuroatlas.cli`` (or, for the original five
verbs, ``neuroatlas.entrypoints``) with a ``main(argv)``; this module only
picks one, prepares the process for it, and turns its outcome into an exit
status. Commands are imported only when chosen, so ``neuroatlas --help`` does
not pay for torch.

Before any command runs:

* ``~/.neuroatlas/config.yaml`` is applied (:func:`neuroatlas.config.apply_to_environ`);
* downloads are switched off, except for the commands whose job is to
  download -- a run that needs a missing file fails at once and names the
  command that fetches it, instead of stalling halfway through a cluster job.
  ``--online`` lifts that for one invocation.

The global options may stand before or after the command. ``-v`` means one
thing everywhere: show more -- the command's own detail (members, paths,
reasons), library log lines and warnings, and tracebacks.

Exit status: 0 success, 1 a run failed, 2 a usage error (unknown command,
bad flag, a path or setting that is missing). A tool's own exit status
(wget, nsrr, ...) never escapes: anything else is reported as 1.
"""
from __future__ import annotations

import argparse
import difflib
import importlib
import logging
import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from neuroatlas import __version__


@dataclass(frozen=True)
class Command:
    module: str
    summary: str
    # Is this command's job to fetch from the network? Those run online.
    downloads: Callable[[Sequence[str]], bool] = lambda argv: False
    # Does it print a report (a table)? Then library INFO lines and warnings
    # go only to --log, unless -v: they would bury the report.
    report: Callable[[Sequence[str]], bool] = lambda argv: False


_always = lambda argv: True       # noqa: E731

COMMANDS = {
    "config": Command("neuroatlas.cli.config",
                      "Set up and inspect where data, caches, results and weights live.",
                      report=_always),
    "list": Command("neuroatlas.cli.listing",
                    "What exists: benchmarks, datasets, models, aliases, tasks.",
                    report=_always),
    "show": Command("neuroatlas.cli.show",
                    "Explain one benchmark and print the commands it runs.",
                    report=_always),
    "data": Command("neuroatlas.cli.data",
                    "Is each dataset here; download it; build its cache.",
                    downloads=lambda argv: argv[:1] in (["download"], ["prepare"]),
                    report=lambda argv: argv[:1] == ["status"]),
    "models": Command("neuroatlas.cli.models",
                      "Are a selection's weights here; download them.",
                      downloads=lambda argv: argv[:1] == ["download"],
                      report=lambda argv: argv[:1] == ["status"]),
    "check": Command("neuroatlas.cli.check",
                     "Push one real batch through each dataset x model pair, then stop.",
                     report=_always),
    "run": Command("neuroatlas.cli.run",
                   "Run a benchmark here: embed where missing, probe, write results.",
                   report=lambda argv: "--dry-run" in argv),
    "submit": Command("neuroatlas.cli.submit_cmd",
                      "Write cluster jobs (HTCondor or SLURM) for a whole benchmark.",
                      report=_always),
    "status": Command("neuroatlas.cli.status_cmd",
                      "How each job of a `submit` is doing.",
                      report=_always),
    "results": Command("neuroatlas.cli.results_cmd",
                       "Summarise a benchmark's results: mean, spread, normalised score.",
                       report=_always),
    "fetch": Command("neuroatlas.entrypoints.fetch",
                     "Obtain a dataset's raw corpus, or say exactly how to.",
                     downloads=lambda argv: "--download" in argv),
    "prepare": Command("neuroatlas.entrypoints.prepare",
                       "Build a dataset's optional cache (no benchmark needs one).",
                       # MOABB builders download the corpus as they epoch it.
                       downloads=lambda argv: True),
    "embed": Command("neuroatlas.entrypoints.embed",
                     "Extract frozen-backbone embeddings for a dataset."),
    "probe": Command("neuroatlas.entrypoints.probe",
                     "Fit a probe on extracted embeddings."),
    "hypnogram": Command("neuroatlas.entrypoints.hypnogram",
                         "Reconstruct hypnograms and sleep-architecture features."),
}

#: Names that are no longer commands, with the command a user who types one
#: wants instead: the did-you-mean of the unknown-command error, for names
#: too unlike any command for difflib to find it.
REMOVED = {"leaderboard": "results"}

USAGE = "usage: neuroatlas [-v] [--log FILE] [--online] <command> [options]"

#: The -m / --models help of the commands that take a benchmark.
MODELS_HELP = ("An alias (all_fm, all_ts, ...), group, family or checkpoint ids, "
               "comma-separated; `-name` removes one (all,-reve). A family the benchmark "
               "leaves out (`neuroatlas show <benchmark>`) is skipped by an alias and refused "
               "by name.")

# (flags, help) of the options every command takes, before or after its name.
GLOBAL_OPTIONS = [
    ("-v, --verbose", "show more: the command's own detail (members, paths, reasons), "
                      "library log lines and warnings, and full tracebacks"),
    ("--log FILE", "also write everything printed, every log line and every traceback, "
                   "to FILE"),
    ("--online", "allow downloads for this run (off by default, except for the "
                 "commands that exist to download)"),
    ("-V, --version", "print the version"),
]


def global_options_text(width: int = 78) -> str:
    """The global options as an aligned, wrapped block (``--help`` and cli.md)."""
    pad = max(len(flags) for flags, _ in GLOBAL_OPTIONS) + 4
    lines = []
    for flags, text in GLOBAL_OPTIONS:
        body = textwrap.wrap(text, width - pad) or [""]
        lines.append(f"  {flags:<{pad - 2}}{body[0]}")
        lines += [" " * pad + more for more in body[1:]]
    return "\n".join(lines)


def _help() -> str:
    width = max(len(name) for name in COMMANDS)
    return "\n".join([
        USAGE,
        "",
        "commands:",
        *(f"  {name:<{width}}  {cmd.summary}" for name, cmd in COMMANDS.items()),
        "",
        "options (before or after the command):",
        global_options_text(),
        "",
        "Start with `neuroatlas config init --data-root DIR`.",
        "Run `neuroatlas <command> --help` for a command's options.",
    ])


# --------------------------------------------------------------------------
# what every command's parser shares
# --------------------------------------------------------------------------

class UsageError(Exception):
    """The user asked for something invalid: printed as ``error: ...``, exit 2."""


_SYNOPSIS = re.compile(r"^``[^`]*``\s*(?:--|—)\s*")


def describe(doc: Optional[str]) -> Optional[str]:
    """A parser description from a docstring: its first paragraph, whole,
    without reST markup and without a leading ``neuroatlas x ...`` -- synopsis
    (the usage line already shows it)."""
    if not doc:
        return doc
    first = textwrap.dedent(doc).strip().split("\n\n", 1)[0]
    text = " ".join(first.split())
    stripped = _SYNOPSIS.sub("", text)
    if stripped != text and stripped:
        text = stripped[0].upper() + stripped[1:]
    text = text.replace("``", "`")
    return text if text.endswith((".", "?", "!", ":")) else text + "."


class _SubParsers(argparse._SubParsersAction):
    def add_parser(self, name, **kwargs):
        # a sub-action's --help says what it does, not nothing
        if kwargs.get("description") is None and kwargs.get("help"):
            kwargs["description"] = kwargs["help"]
        return super().add_parser(name, **kwargs)


class Parser(argparse.ArgumentParser):
    """argparse, plus: descriptions from docstrings (:func:`describe`), a
    "did you mean" for every mistyped choice or flag, and unknown arguments
    reported by the sub-command that received them, not by its parent."""

    def __init__(self, *args, **kwargs):
        if kwargs.get("description"):
            kwargs["description"] = describe(kwargs["description"])
        super().__init__(*args, **kwargs)
        self.register("action", "parsers", _SubParsers)

    def _check_value(self, action, value):
        if action.choices is not None and value not in action.choices:
            choices = [str(c) for c in action.choices]
            close = difflib.get_close_matches(str(value), choices, n=1, cutoff=0.6)
            hint = f" -- did you mean {close[0]!r}?" if close else ""
            raise argparse.ArgumentError(
                action, f"invalid choice: {value!r}{hint} (choose from {', '.join(choices)})")

    def _leaf(self, namespace) -> "argparse.ArgumentParser":
        parser = self
        while True:
            sub = next((a for a in parser._actions
                        if isinstance(a, argparse._SubParsersAction)), None)
            name = getattr(namespace, sub.dest, None) if sub is not None else None
            if name not in getattr(sub, "choices", {}):
                return parser
            parser = sub.choices[name]

    def parse_args(self, args=None, namespace=None):
        namespace, extras = self.parse_known_args(args, namespace)
        if extras:
            leaf = self._leaf(namespace)
            leaf.error(unrecognized_message(leaf, extras))
        return namespace


def unrecognized_message(parser: argparse.ArgumentParser, extras: Sequence[str]) -> str:
    known = [s for s in parser._option_string_actions if s.startswith("--")]
    hints = []
    for extra in extras:
        flag = extra.split("=", 1)[0]
        if flag.startswith("-"):
            close = difflib.get_close_matches(flag, known, n=1, cutoff=0.6)
            if close:
                hints.append((flag, close[0]))
    if len(hints) == 1:
        hint = f" -- did you mean {hints[0][1]}?"
    else:
        hint = "".join(f"; {flag}: did you mean {close}?" for flag, close in hints)
    return f"unrecognized arguments: {' '.join(extras)}{hint}"


def command_parser(name: str) -> argparse.ArgumentParser:
    """The argparse parser of a command (for cli.md and option routing)."""
    module = importlib.import_module(COMMANDS[name].module)
    if hasattr(module, "build_parser"):
        return module.build_parser()
    # thin aliases (submit_cmd, results_cmd, ...) re-export a main
    owner = importlib.import_module(module.main.__module__)
    builder = {"submit_main": "build_submit_parser", "status_main": "build_status_parser",
               "results_main": "build_results_parser"}[module.main.__name__]
    return getattr(owner, builder)()


def _leaf_for(parser: argparse.ArgumentParser, tokens: Sequence[str]) -> argparse.ArgumentParser:
    """The sub-parser ``tokens`` would reach (``list models ...`` -> models)."""
    tokens = list(tokens)
    while True:
        sub = next((a for a in parser._actions if isinstance(a, argparse._SubParsersAction)), None)
        if sub is None:
            return parser
        name = next((t for t in tokens if t in sub.choices), None)
        if name is None:
            return parser
        parser, tokens = sub.choices[name], tokens[tokens.index(name) + 1:]


# --------------------------------------------------------------------------
# the process around a command
# --------------------------------------------------------------------------

class _Tee:
    """Write to a stream and a log file at once."""

    def __init__(self, stream, log):
        self._stream, self._log = stream, log

    def write(self, text):
        self._stream.write(text)
        self._log.write(text)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _usage_error(message: str) -> "SystemExit":
    print(f"neuroatlas: {message}", file=sys.stderr)
    return SystemExit(2)


TRACEBACK_LOGGER = "neuroatlas.traceback"


def _setup_logging(verbose: bool, report: bool, console_stream, log) -> List[logging.Handler]:
    """Console: INFO (DEBUG with -v); for a report command WARNING and no
    library warnings unless -v. The --log file always gets everything at
    INFO, plus every traceback."""
    fmt = logging.Formatter("%(levelname)s: %(message)s")
    root = logging.getLogger()
    for h in [h for h in root.handlers if getattr(h, "_neuroatlas", False)]:
        root.removeHandler(h)
    # third-party libraries stay at INFO even with -v (their DEBUG is noise);
    # our own loggers go to DEBUG with -v
    root.setLevel(logging.INFO)
    logging.getLogger("neuroatlas").setLevel(logging.DEBUG if verbose else logging.NOTSET)
    logging.getLogger(TRACEBACK_LOGGER).setLevel(logging.DEBUG)

    console = logging.StreamHandler(console_stream)
    console.setFormatter(fmt)
    if verbose:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.WARNING if report else logging.INFO)
        console.addFilter(lambda r: r.name != TRACEBACK_LOGGER)
        if report:
            console.addFilter(lambda r: r.name != "py.warnings")
    handlers = [console]
    if log is not None:
        to_file = logging.StreamHandler(log)
        to_file.setFormatter(fmt)
        to_file.setLevel(logging.DEBUG)
        handlers.append(to_file)
    for h in handlers:
        h._neuroatlas = True
        root.addHandler(h)
    if report:
        logging.captureWarnings(True)
    return handlers


def _take_global_options(rest: List[str], state: dict) -> List[str]:
    """Remove --online and --log FILE from anywhere before ``--``."""
    out: List[str] = []
    i = 0
    while i < len(rest):
        token = rest[i]
        if token == "--":
            out += rest[i:]
            break
        if token == "--online":
            state["online"] = True
        elif token == "--log":
            if i + 1 >= len(rest):
                raise _usage_error("--log needs a file")
            state["log"] = rest[i + 1]
            i += 1
        elif token.startswith("--log="):
            state["log"] = token.split("=", 1)[1]
            if not state["log"]:
                raise _usage_error("--log needs a file")
        else:
            out.append(token)
        i += 1
    return out


def _route_verbose(name: str, rest: List[str], state: dict) -> List[str]:
    """``-v`` after the command: the command's own flag where it has one
    (and then it also means verbose logging); otherwise the global one. A
    global ``-v`` is handed to the command too, so both places mean the same."""
    try:
        leaf = _leaf_for(command_parser(name), rest)
        has_v = "-v" in leaf._option_string_actions
    except Exception:                                  # a parser we cannot build: leave it be
        return rest
    end = rest.index("--") if "--" in rest else len(rest)
    head, tail = rest[:end], rest[end:]
    if any(t in ("-v", "--verbose") for t in head):
        state["verbose"] = True
        if not has_v:
            head = [t for t in head if t not in ("-v", "--verbose")]
    elif state["verbose"] and has_v:
        head = head + ["-v"]
    return head + tail


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    state = {"verbose": False, "online": False, "log": None}
    while argv and argv[0].startswith("-"):
        flag = argv.pop(0)
        if flag in ("-h", "--help"):
            print(_help())
            return
        if flag in ("-V", "--version"):
            print(f"neuroatlas {__version__}")
            return
        if flag in ("-v", "--verbose"):
            state["verbose"] = True
        elif flag == "--online":
            state["online"] = True
        elif flag == "--log" or flag.startswith("--log="):
            state["log"] = flag.split("=", 1)[1] if "=" in flag else (argv.pop(0) if argv else None)
            if not state["log"]:
                raise _usage_error("--log needs a file")
        else:
            close = difflib.get_close_matches(flag, ["--verbose", "--log", "--online", "--version",
                                                     "--help"], n=1)
            hint = f" -- did you mean {close[0]}?" if close else ""
            raise _usage_error(f"unknown option {flag!r}{hint}\n\n{USAGE}")

    if not argv:
        print(_help())
        raise SystemExit(2)

    name, rest = argv[0], argv[1:]
    command = COMMANDS.get(name)
    if command is None:
        close = difflib.get_close_matches(name, COMMANDS, n=1) or (
            [REMOVED[name]] if name in REMOVED else [])
        hint = f" -- did you mean `{close[0]}`?" if close else ""
        raise _usage_error(f"unknown command {name!r}{hint}\n\n{USAGE}")
    rest = _take_global_options(rest, state)

    from neuroatlas import catalog, config, selectors

    saved = sys.stdout, sys.stderr
    log = None
    handlers: List[logging.Handler] = []
    levels = {name: logging.getLogger(name or None).level
              for name in ("", "neuroatlas", TRACEBACK_LOGGER)}
    try:
        # `config` loads the file leniently: it is how a bad file gets fixed.
        config.apply_to_environ(strict=name != "config")
        if not (state["online"] or command.downloads(rest)):
            config.set_offline()
        rest = _route_verbose(name, rest, state)
        if state["log"]:
            log = open(state["log"], "a", encoding="utf-8")
        handlers = _setup_logging(state["verbose"], command.report(rest), sys.stderr, log)
        if log is not None:
            sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
        importlib.import_module(command.module).main(rest)
    except config.ConfigError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from None
    except (catalog.CatalogError, selectors.SelectionError, UsageError) as exc:
        # a benchmark, dataset or model name that does not exist, a bad value: a usage error
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except SystemExit as exc:
        if isinstance(exc.code, str):
            # SystemExit("error: ...") from a verb: print it here, where it can
            # be captured, and exit 1 as Python itself would.
            print(exc.code, file=sys.stderr)
            raise SystemExit(1) from None
        if isinstance(exc.code, int) and exc.code not in (0, 1, 2, 130):
            # a tool's own status (wget 8, nsrr 3): not part of our contract
            print(f"neuroatlas: a step exited with status {exc.code}; reporting it as 1 "
                  f"(a run failed)", file=sys.stderr)
            raise SystemExit(1) from None
        raise
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        if state["verbose"]:
            raise
        logging.getLogger(TRACEBACK_LOGGER).debug("uncaught", exc_info=True)
        where = f"in {state['log']}" if state["log"] else "with -v or --log FILE"
        print(f"error: {type(exc).__name__}: {exc}\n(the traceback is {where})", file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        root = logging.getLogger()
        for h in handlers:
            root.removeHandler(h)
        for logger_name, level in levels.items():
            logging.getLogger(logger_name).setLevel(level)
        logging.captureWarnings(False)
        sys.stdout, sys.stderr = saved
        if log is not None:
            log.close()
