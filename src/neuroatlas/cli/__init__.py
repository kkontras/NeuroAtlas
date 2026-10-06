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


#: The words after ``neuroatlas`` of the command being run (set by
#: :func:`main`), so a usage error can print the corrected command.
_COMMAND_LINE: List[str] = []
#: mistyped word -> the word it is close to, gathered while parsing
_SUGGEST: dict = {}


def corrected_command(replacements: dict) -> Optional[str]:
    """The command line being run with each mistyped word replaced, or None
    when it is not known or a word is not in it."""
    if not _COMMAND_LINE or not replacements:
        return None
    import shlex

    out, done = [], set()
    for token in _COMMAND_LINE:
        flag, eq, value = token.partition("=")
        items = token.split(",")
        if token in replacements and token not in done:
            out.append(replacements[token])
            done.add(token)
        elif eq and flag in replacements and flag not in done:
            out.append(f"{replacements[flag]}={value}")
            done.add(flag)
        elif len(items) > 1 and any(i in replacements and i not in done for i in items):
            # one name of a comma list (-m biot_pretraind,reve)
            fixed = []
            for item in items:
                if item in replacements and item not in done:
                    fixed.append(replacements[item])
                    done.add(item)
                else:
                    fixed.append(item)
            out.append(",".join(fixed))
        else:
            out.append(token)
    if len(done) < len(replacements):
        return None
    return "neuroatlas " + " ".join(shlex.quote(t) for t in out)


class ErrorParser(argparse.ArgumentParser):
    """argparse whose errors read like every other message: ``error: ...``,
    then the corrected command (a mistyped flag or choice) or ``<prog>
    --help`` on a ``fix:`` line, instead of the whole usage block; exit 2.
    The verbs (embed, probe, ...) use it as it is; :class:`Parser` adds the
    rest."""

    def error(self, message):
        from neuroatlas.cli import _msg

        suggestions = dict(_SUGGEST)
        _SUGGEST.clear()
        fix = corrected_command(suggestions)
        if fix is None:
            fix = f"{self.prog} --help"
            if suggestions:      # not run through `neuroatlas`: name the word instead
                message += " (did you mean " + ", ".join(suggestions.values()) + "?)"
        _msg.error(message, fix)
        raise SystemExit(2)


class Parser(ErrorParser):
    """argparse, plus: descriptions from docstrings (:func:`describe`), a
    "did you mean" for every mistyped choice or flag -- printed as the
    corrected command -- unknown arguments reported by the sub-command that
    received them, not by its parent, and errors as ``error: ...`` with a
    ``fix:`` line instead of the whole usage block (:class:`ErrorParser`)."""

    def __init__(self, *args, **kwargs):
        if kwargs.get("description"):
            kwargs["description"] = describe(kwargs["description"])
        super().__init__(*args, **kwargs)
        self.register("action", "parsers", _SubParsers)

    def _check_value(self, action, value):
        if action.choices is not None and value not in action.choices:
            choices = [str(c) for c in action.choices]
            close = difflib.get_close_matches(str(value), choices, n=1, cutoff=0.6)
            if close:
                _SUGGEST[str(value)] = close[0]
            text = f"invalid choice {value!r} (choose from {', '.join(choices)})"
            # a sub-command's own name has no flag to name it by
            raise argparse.ArgumentError(
                None if isinstance(action, argparse._SubParsersAction) else action, text)

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
    """``unrecognized argument: --datset siena``; a close flag is remembered
    for the corrected command (:meth:`Parser.error`)."""
    known = [s for s in parser._option_string_actions if s.startswith("--")]
    for extra in extras:
        flag = extra.split("=", 1)[0]
        if flag.startswith("-"):
            close = difflib.get_close_matches(flag, known, n=1, cutoff=0.6)
            if close:
                _SUGGEST[flag] = close[0]
    return f"unrecognized argument{'s' if len(extras) > 1 else ''}: {' '.join(extras)}"


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
    """Write to a stream and a log file at once; the log gets no colour."""

    def __init__(self, stream, log):
        self._stream, self._log = stream, log

    def write(self, text):
        from neuroatlas.cli._msg import ANSI

        self._stream.write(text)
        self._log.write(ANSI.sub("", text))
        return len(text)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _usage_error(message: str, fix: str = "neuroatlas --help") -> "SystemExit":
    from neuroatlas.cli import _msg

    _msg.error(message, fix)
    return SystemExit(2)


TRACEBACK_LOGGER = "neuroatlas.traceback"


class _HiddenCount(logging.Handler):
    """Counts what the console did not show (library INFO lines, Python
    warnings), so a command can say so in one line at the end."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.count = 0

    def emit(self, record):
        if record.levelno < logging.WARNING or record.name == "py.warnings":
            self.count += 1


def _setup_logging(verbose: bool, report: bool, console_stream, log) -> List[logging.Handler]:
    """Console: WARNING and no library warnings, for every command, unless
    -v (then DEBUG and everything): the per-model INFO lines of a long run
    (backbone banners, loader lines, weight reports) are for -v and the log.
    The --log file always gets everything at INFO, plus every traceback.
    Both say ``error:`` and ``warning:`` as every other message does
    (:class:`neuroatlas.cli._msg.LogFormatter`); only the console is coloured."""
    from neuroatlas.cli import _msg

    root = logging.getLogger()
    for h in [h for h in root.handlers if getattr(h, "_neuroatlas", False)]:
        root.removeHandler(h)
    # third-party libraries stay at INFO even with -v (their DEBUG is noise);
    # our own loggers go to DEBUG with -v
    root.setLevel(logging.INFO)
    logging.getLogger("neuroatlas").setLevel(logging.DEBUG if verbose else logging.NOTSET)
    logging.getLogger(TRACEBACK_LOGGER).setLevel(logging.DEBUG)

    console = logging.StreamHandler(console_stream)
    console.setFormatter(_msg.LogFormatter(color=_msg.use_color(console_stream)))
    # the whole text of a message the screen shows shortened: for the log
    # (with -v the screen shows it whole in the first place)
    console.addFilter(lambda r: r.name != _msg.FULL_LOGGER)
    if verbose:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.WARNING)
        console.addFilter(lambda r: r.name != TRACEBACK_LOGGER)
        console.addFilter(lambda r: r.name != "py.warnings")
    handlers = [console]
    if not verbose:
        hidden = _HiddenCount()
        hidden.addFilter(lambda r: r.name not in (TRACEBACK_LOGGER, _msg.FULL_LOGGER))
        handlers.append(hidden)
    if log is not None:
        to_file = logging.StreamHandler(log)
        to_file.setFormatter(_msg.LogFormatter(color=False))
        to_file.setLevel(logging.DEBUG)
        handlers.append(to_file)
    for h in handlers:
        h._neuroatlas = True
        root.addHandler(h)
    logging.captureWarnings(True)
    return handlers


_LOG_FIX = "neuroatlas --log FILE <command> ..."


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
                raise _usage_error("--log needs a file", _LOG_FIX)
            state["log"] = rest[i + 1]
            i += 1
        elif token.startswith("--log="):
            state["log"] = token.split("=", 1)[1]
            if not state["log"]:
                raise _usage_error("--log needs a file", _LOG_FIX)
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
    import shlex

    from neuroatlas.cli import _msg

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
                raise _usage_error("--log needs a file", _LOG_FIX)
        else:
            close = difflib.get_close_matches(flag, ["--verbose", "--log", "--online", "--version",
                                                     "--help"], n=1)
            fixed = " ".join(["neuroatlas", close[0], *(shlex.quote(t) for t in argv)]) \
                if close else "neuroatlas --help"
            raise _usage_error(f"unknown option {flag!r}", fixed)

    if not argv:
        print(_help())
        raise SystemExit(2)

    name, rest = argv[0], argv[1:]
    command = COMMANDS.get(name)
    if command is None:
        close = difflib.get_close_matches(name, COMMANDS, n=1) or (
            [REMOVED[name]] if name in REMOVED else [])
        fixed = " ".join(["neuroatlas", close[0], *(shlex.quote(t) for t in rest)]) \
            if close else "neuroatlas --help"
        raise _usage_error(f"unknown command {name!r}", fixed)
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
        _msg.set_verbose(state["verbose"])
        _COMMAND_LINE[:] = [name, *rest]
        if state["log"]:
            log = open(state["log"], "a", encoding="utf-8")
        handlers = _setup_logging(state["verbose"], command.report(rest), sys.stderr, log)
        if log is not None:
            sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
        importlib.import_module(command.module).main(rest)
    except config.ConfigError as exc:
        _msg.error(exc.code if isinstance(exc.code, str) else str(exc))
        raise SystemExit(2) from None
    except (catalog.CatalogError, selectors.SelectionError, UsageError) as exc:
        # a benchmark, dataset or model name that does not exist, a bad value:
        # a usage error; a mistyped name gets the corrected command as its fix
        corrected = corrected_command(getattr(exc, "suggest", None) or {})
        if corrected:
            _, lines, _ = _msg.split(str(exc))
            _msg.error("\n".join(lines), corrected)
        else:
            _msg.error(str(exc))
        raise SystemExit(2) from None
    except SystemExit as exc:
        if isinstance(exc.code, str):
            # SystemExit("error: ...") from a verb: print it here, where it can
            # be captured, and exit 1 as Python itself would.
            _msg.error(exc.code)
            raise SystemExit(1) from None
        if isinstance(exc.code, int) and exc.code not in (0, 1, 2, 130):
            # a tool's own status (wget 8, nsrr 3): not part of our contract
            _msg.error(f"a step exited with status {exc.code}")
            raise SystemExit(1) from None
        raise
    except KeyboardInterrupt:
        print(file=sys.stderr)
        _msg.error("interrupted")
        raise SystemExit(130) from None
    except BrokenPipeError:
        # `neuroatlas results ... | head`: the reader stopped reading; not an error
        import os

        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.__stdout__.fileno())
        except (OSError, ValueError, AttributeError):
            pass
        raise SystemExit(0) from None
    except Exception as exc:
        if state["verbose"]:
            raise
        logging.getLogger(TRACEBACK_LOGGER).debug("uncaught", exc_info=True)
        text = _msg.exception_text(exc)
        head = _msg.first_sentence(text)
        whole = " ".join(head.split()) == " ".join(text.split())
        if state["log"]:
            where = f"traceback in {state['log']}"
            if not whole:
                logging.getLogger(_msg.FULL_LOGGER).info("full message: %s", text)
        else:
            where = "-v: traceback" if whole else "-v: full message and traceback"
        _msg.error(f"{head} ({where})")
        raise SystemExit(1) from None
    finally:
        root = logging.getLogger()
        hidden = sum(h.count for h in handlers if isinstance(h, _HiddenCount))
        if hidden and not command.report(rest):
            where = f"they are in {state['log']}" if state["log"] else "-v shows them"
            _msg.note(f"{_msg.plural(hidden, 'log line')} not shown: {where}", file=saved[1])
        for h in handlers:
            root.removeHandler(h)
        for logger_name, level in levels.items():
            logging.getLogger(logger_name).setLevel(level)
        logging.captureWarnings(False)
        _msg.set_verbose(False)
        _COMMAND_LINE.clear()
        sys.stdout, sys.stderr = saved
        if log is not None:
            log.close()
