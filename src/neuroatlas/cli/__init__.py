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

Exit status: 0 success, 1 a run failed, 2 a usage error (unknown command,
bad flag, a path or setting that is missing).
"""
from __future__ import annotations

import difflib
import importlib
import logging
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from neuroatlas import __version__


@dataclass(frozen=True)
class Command:
    module: str
    summary: str
    # Is this command's job to fetch from the network? Those run online.
    downloads: Callable[[Sequence[str]], bool] = lambda argv: False


COMMANDS = {
    "config": Command("neuroatlas.cli.config",
                      "Set up and inspect where data, caches, results and weights live."),
    "list": Command("neuroatlas.cli.listing",
                    "What exists: benchmarks, datasets, models, aliases, tasks."),
    "show": Command("neuroatlas.cli.show",
                    "Explain one benchmark and print the commands it runs."),
    "fetch": Command("neuroatlas.entrypoints.fetch",
                     "Obtain a dataset's raw corpus, or say exactly how to.",
                     downloads=lambda argv: "--download" in argv),
    "prepare": Command("neuroatlas.entrypoints.prepare",
                       "Build a dataset's cache (required for BCI, optional elsewhere).",
                       # MOABB builders download the corpus as they epoch it.
                       downloads=lambda argv: True),
    "embed": Command("neuroatlas.entrypoints.embed",
                     "Extract frozen-backbone embeddings for a dataset."),
    "probe": Command("neuroatlas.entrypoints.probe",
                     "Fit a probe on extracted embeddings."),
    "hypnogram": Command("neuroatlas.entrypoints.hypnogram",
                         "Reconstruct hypnograms and sleep-architecture features."),
}

USAGE = "usage: neuroatlas [-v] [--log FILE] [--online] <command> [options]"


def _help() -> str:
    width = max(len(name) for name in COMMANDS)
    return "\n".join([
        USAGE,
        "",
        "commands:",
        *(f"  {name:<{width}}  {cmd.summary}" for name, cmd in COMMANDS.items()),
        "",
        "options:",
        "  -v, --verbose  more logging",
        "  --log FILE     also write everything printed to FILE",
        "  --online       allow downloads for this run (off by default, except",
        "                 for the commands that exist to download)",
        "  -V, --version  print the version",
        "",
        "Start with `neuroatlas config init --data-root DIR`.",
        "Run `neuroatlas <command> --help` for a command's options.",
    ])


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


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    verbose, online, log_path = False, False, None
    while argv and argv[0].startswith("-"):
        flag = argv.pop(0)
        if flag in ("-h", "--help"):
            print(_help())
            return
        if flag in ("-V", "--version"):
            print(f"neuroatlas {__version__}")
            return
        if flag in ("-v", "--verbose"):
            verbose = True
        elif flag == "--online":
            online = True
        elif flag == "--log" or flag.startswith("--log="):
            log_path = flag.split("=", 1)[1] if "=" in flag else (argv.pop(0) if argv else None)
            if not log_path:
                raise _usage_error("--log needs a file")
        else:
            raise _usage_error(f"unknown option {flag!r}\n\n{USAGE}")

    if not argv:
        print(_help())
        raise SystemExit(2)

    name, rest = argv[0], argv[1:]
    command = COMMANDS.get(name)
    if command is None:
        close = difflib.get_close_matches(name, COMMANDS, n=1)
        hint = f" -- did you mean `{close[0]}`?" if close else ""
        raise _usage_error(f"unknown command {name!r}{hint}\n\n{USAGE}")

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")
    if log_path:
        log = open(log_path, "a", encoding="utf-8")
        sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
        logging.getLogger().addHandler(logging.StreamHandler(log))

    from neuroatlas import catalog, config, selectors

    try:
        config.apply_to_environ()
        if not (online or command.downloads(rest)):
            config.set_offline()
        importlib.import_module(command.module).main(rest)
    except config.ConfigError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from None
    except (catalog.CatalogError, selectors.SelectionError) as exc:
        # a benchmark, dataset or model name that does not exist: a usage error
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except SystemExit as exc:
        if isinstance(exc.code, str):
            # SystemExit("error: ...") from a verb: print it here, where it can
            # be captured, and exit 1 as Python itself would.
            print(exc.code, file=sys.stderr)
            raise SystemExit(1) from None
        raise
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130) from None
