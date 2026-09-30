"""The ``neuroatlas`` command.

Each subcommand is one of the entrypoint verbs, handed the rest of the command
line untouched, so ``neuroatlas probe --dataset dod`` is exactly
``python -m neuroatlas.entrypoints.probe --dataset dod``. Verbs are imported
only when chosen: ``neuroatlas --help`` must not pay for torch.
"""
from __future__ import annotations

import importlib
import sys
from typing import List, Optional

from neuroatlas import __version__

# verb -> (module, one-line summary)
VERBS = {
    "fetch": ("neuroatlas.entrypoints.fetch",
              "Obtain a dataset's raw corpus, or say exactly how to."),
    "prepare": ("neuroatlas.entrypoints.prepare",
                "Build a dataset's cache (required for BCI, optional elsewhere)."),
    "embed": ("neuroatlas.entrypoints.embed",
              "Extract frozen-backbone embeddings for a dataset."),
    "probe": ("neuroatlas.entrypoints.probe",
              "Fit a probe on extracted embeddings."),
    "hypnogram": ("neuroatlas.entrypoints.hypnogram",
                  "Reconstruct hypnograms and sleep-architecture features."),
}


def _usage() -> str:
    width = max(len(v) for v in VERBS)
    lines = [
        "usage: neuroatlas <command> [options]",
        "",
        "commands:",
        *(f"  {verb:<{width}}  {summary}" for verb, (_, summary) in VERBS.items()),
        "",
        "Run `neuroatlas <command> --help` for a command's options.",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return
    if argv[0] in ("-V", "--version"):
        print(f"neuroatlas {__version__}")
        return
    verb, rest = argv[0], argv[1:]
    if verb not in VERBS:
        import difflib

        close = difflib.get_close_matches(verb, VERBS, n=1)
        hint = f" -- did you mean `{close[0]}`?" if close else ""
        print(f"neuroatlas: unknown command {verb!r}{hint}\n\n{_usage()}", file=sys.stderr)
        raise SystemExit(2)
    module = importlib.import_module(VERBS[verb][0])
    module.main(rest)
