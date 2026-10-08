"""Write docs/cli.md from the command's own parsers, so the reference cannot
drift from the code.

    python -m neuroatlas.cli._gendocs            # rewrite docs/cli.md
    python -m neuroatlas.cli._gendocs --check    # exit 1 if it is stale

Descriptions, options and defaults come from the parsers. What is written
here by hand: the groups the commands are listed in, and the examples (each
one is parsed by the command's own parser in the tests, so it stays a real
command line).
"""
from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from neuroatlas import _paths

#: The sections of the page, in order, and the commands in each.
GROUPS: List[Tuple[str, List[str]]] = [
    ("Setup", ["config"]),
    ("Explore", ["list", "show"]),
    ("Data and weights", ["data", "models"]),
    ("Run", ["check", "run", "submit", "status"]),
    ("Results", ["results", "rescore"]),
    ("Individual steps", ["embed", "probe", "hypnogram"]),
]

#: Commands that are other names for a sub-action: one line, no section.
OTHER_NAMES = {"fetch": "data download", "prepare": "data prepare"}

#: Typical command lines, by command or sub-action. A `# ...` comment is
#: shown and ignored when the example is parsed.
EXAMPLES: Dict[str, List[str]] = {
    "config init": [
        "neuroatlas config init ~/neuroatlas",
        "neuroatlas config init ~/neuroatlas --data-root /data/eeg    # datasets elsewhere",
    ],
    "config show": ["neuroatlas config show"],
    "config set": [
        "neuroatlas config set ucddb.data_root /mnt/ucddb",
        "neuroatlas config set cache_root /scratch/neuroatlas_cache",
    ],
    "config unset": ["neuroatlas config unset ucddb.data_root"],
    "config path": ["neuroatlas config path"],
    "config token": [
        "neuroatlas config token nsrr",
        "neuroatlas config token hf --remove",
    ],
    "list benchmarks": [
        "neuroatlas list benchmarks",
        "neuroatlas list benchmarks -v",
    ],
    "list datasets": [
        "neuroatlas list datasets --grep sleep",
        "neuroatlas list datasets --all --format csv",
    ],
    "list models": [
        "neuroatlas list models all_fm",
        "neuroatlas list models --benchmark epilepsy",
    ],
    "list aliases": ["neuroatlas list aliases -v"],
    "list tasks": ["neuroatlas list tasks"],
    "show": [
        "neuroatlas show sleep_stage",
        "neuroatlas show epilepsy --dataset single -m cbramod_pretrained",
    ],
    "data status": [
        "neuroatlas data status",
        "neuroatlas data status sleep_stage",
        "neuroatlas data status siena -v",
    ],
    "data download": [
        "neuroatlas data download sleep_edf_expanded --mirror aws",
        "neuroatlas data download siena --first 5    # a quick test",
        "neuroatlas data download cfs --dry-run",
    ],
    "data prepare": [
        "neuroatlas data prepare --list",
        "neuroatlas data prepare bonn --dry-run",
        "neuroatlas data prepare tusz --shard 0/8",
    ],
    "models status": [
        "neuroatlas models status",
        "neuroatlas models status all_fm -v",
    ],
    "models download": [
        "neuroatlas models download biot_pretrained",
        "neuroatlas models download all_fm",
    ],
    "check": [
        "neuroatlas check sleep_stage -m biot_pretrained",
        "neuroatlas check epilepsy -m all_fm --dataset full",
    ],
    "run": [
        "neuroatlas run sleep_stage -m biot_pretrained --debug    # fold 0 only",
        "neuroatlas run sleep_stage -m biot_pretrained            # all folds",
        "neuroatlas run epilepsy -m all_fm --dataset full --dry-run",
        "neuroatlas run bci_motor_imagery -m all_fm --variant token_flattening",
    ],
    "submit": [
        "neuroatlas submit sleep_stage -m all_fm --out jobs/sleep_stage",
        "neuroatlas submit sleep_stage -m all_fm --out jobs/sleep_stage --mode retry",
        "neuroatlas submit epilepsy -m all_fm --out jobs/epilepsy --backend slurm "
        "--partition gpu --time 12:00:00",
    ],
    "status": [
        "neuroatlas status --out jobs/sleep_stage",
        "neuroatlas status --out jobs/sleep_stage -v",
    ],
    "results": [
        "neuroatlas results sleep_stage",
        "neuroatlas results bci_motor_imagery --variant token_flattening",
        "neuroatlas results epilepsy --format csv",
    ],
    "rescore": [
        "neuroatlas rescore epilepsy",
        "neuroatlas rescore sleep_stage --dataset dod -m biot_pretrained",
    ],
    "embed": [
        "neuroatlas embed --dataset dod -m biot_pretrained,labram_pretrained",
        "neuroatlas embed --dataset hmc -m reve_pretrained --set window_s=30 --set stride_s=30",
        "neuroatlas embed --dataset shhs -m labram_pretrained --embed-chunk 0/4    # chunk 0 of 4",
        "neuroatlas embed --list-datasets --paper-only",
    ],
    "probe": [
        "neuroatlas probe --dataset isruc -m biot_pretrained",
        "neuroatlas probe --dataset isruc -m biot_pretrained --task sex",
        "neuroatlas probe --dataset mass --task mass_arousal --dry-run",
    ],
    "hypnogram": [
        "neuroatlas hypnogram --datasets sleep_edf_expanded",
        "neuroatlas hypnogram --datasets dod mass",
    ],
    "hypnogram reconstruct": [
        "neuroatlas hypnogram reconstruct --datasets dod --group-by group --compute-metrics",
    ],
    "hypnogram features": [
        "neuroatlas hypnogram features --datasets isruc --no-summary",
    ],
}

#: Text shown after the options of a command.
NOTES: Dict[str, str] = {
    "embed": "Run `neuroatlas embed --help` for the list of datasets, tasks and models, and "
             "`neuroatlas embed --dataset hmc --help` for the `--set` keys of one dataset.\n",
    "probe": "Run `neuroatlas probe --help` for the list of datasets, tasks and models, and "
             "`neuroatlas probe --dataset hmc --help` for the `--set` keys of one dataset.\n",
}

#: Commands whose --help ends with text worth showing here. The others end
#: with the long dataset lists of the terminal help.
SHOW_EPILOG = ("status", "hypnogram")

INTRO = """\
# Command reference

This page lists every `neuroatlas` command and its options. Run
`neuroatlas <command> --help` to see the same text in the terminal. The
[user guide](user_guide.md) shows how the commands fit together.

This page is generated from the command-line parsers by
`python -m neuroatlas.cli._gendocs`. Do not edit it by hand.
"""

_DEFAULT_IN_PARENS = re.compile(r"\s*\(default: ([^()]*)\)")
_DEFAULT_SENTENCE = re.compile(r"\s*Default: ([^.]*)\.\s*$")
_CODE = re.compile(r"(`[^`]*`)")


def _escape(text: str) -> str:
    """A help text as Markdown: `<` and `>` outside code spans as entities
    (`<output root>` would read as an HTML tag), and `|` escaped for a table."""
    parts = _CODE.split(text)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            part = part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts[i] = part.replace("|", "\\|")
    return "".join(parts)


def _flat(text: Optional[str]) -> str:
    return " ".join((text or "").split())


def split_default(action: argparse.Action) -> Tuple[str, str]:
    """(description, default) of an option. The default is the one its help
    states, as ``(default: X)`` or a closing ``Default: X.``, which is taken
    out of the text. Else it is the parser's own default when that is a value."""
    text = _flat(action.help)
    stated = None
    match = _DEFAULT_IN_PARENS.search(text)
    if match:
        stated = match.group(1)
        text = text[:match.start()] + text[match.end():]
    else:
        match = _DEFAULT_SENTENCE.search(text)
        if match:
            stated = match.group(1)
            text = text[:match.start()]
    text = re.sub(r"\s+([.,])(?=\s|$)", r"\1", text).strip()
    if text and not text.endswith((".", "?", "!", ")")):
        text += "."
    value = action.default
    literal = (isinstance(value, (str, int, float)) and not isinstance(value, bool)
               and action.nargs != 0 and value is not argparse.SUPPRESS)
    if stated is None:
        return text, f"`{value}`" if literal else ""
    return text, f"`{stated}`" if literal and stated == str(value) else _escape(stated)


def _metavar(action: argparse.Action) -> str:
    formatter = argparse.HelpFormatter("neuroatlas")
    if not action.option_strings:
        return formatter._metavar_formatter(action, action.dest)(1)[0]
    if action.nargs == 0:
        return ""
    return formatter._format_args(action, action.dest.upper())


def _option_cell(action: argparse.Action) -> str:
    metavar = _metavar(action)
    if not action.option_strings:
        return f"`{metavar}`".replace("|", "\\|")
    flags = list(action.option_strings)
    if metavar:
        flags[-1] = f"{flags[-1]} {metavar}"
    return ", ".join(f"`{f}`" for f in flags).replace("|", "\\|")


def _shown(action: argparse.Action) -> bool:
    return not (isinstance(action, (argparse._HelpAction, argparse._SubParsersAction))
                or action.help == argparse.SUPPRESS)


def _subparsers(parser: argparse.ArgumentParser) -> Optional[argparse._SubParsersAction]:
    return next((a for a in parser._actions if isinstance(a, argparse._SubParsersAction)), None)


def usage_line(title: str, parser: argparse.ArgumentParser) -> str:
    """``neuroatlas run benchmark -m MODELS [options]``: the positionals, the
    required options, then [options] when there are others."""
    formatter = argparse.HelpFormatter("neuroatlas")
    parts = ["neuroatlas", *title.split()]
    sub = _subparsers(parser)
    optional = False
    for action in parser._actions:
        if action is sub or isinstance(action, argparse._HelpAction):
            continue
        if not action.option_strings:
            parts.append(formatter._format_args(action, action.dest))
        elif action.required:
            parts.append(f"{action.option_strings[0]} {_metavar(action)}".strip())
        elif action.help != argparse.SUPPRESS:
            optional = True
    if sub is not None:
        word = sub.metavar or "{" + ",".join(sub.choices) + "}"
        parts.append(word if sub.required or not optional else f"[{word}]")
    if optional:
        parts.append("[options]")
    return " ".join(parts)


def options_tables(parser: argparse.ArgumentParser) -> List[str]:
    """A table of the options, positionals first, and one more table for
    each named group of options (the resources of `submit`)."""
    out: List[str] = []
    main_rows: List[argparse.Action] = []
    groups: List[Tuple[str, List[argparse.Action]]] = []
    for group in parser._action_groups:
        actions = [a for a in group._group_actions if _shown(a)]
        if not actions:
            continue
        if group.title in ("positional arguments", "options", "optional arguments"):
            main_rows += actions
        else:
            groups.append((group.title, actions))
    main_rows.sort(key=lambda a: bool(a.option_strings))
    for title, actions in [("", main_rows), *groups]:
        if not actions:
            continue
        if title:
            out.append(f"{title[0].upper()}{title[1:]}:\n")
        rows = ["| Option | Description | Default |", "|---|---|---|"]
        for action in actions:
            text, default = split_default(action)
            rows.append(f"| {_option_cell(action)} | {_escape(text)} | {default} |")
        out.append("\n".join(rows) + "\n")
    return out


def _anchor(title: str) -> str:
    return title.replace(" ", "-")


def section(title: str, parser: argparse.ArgumentParser, level: str) -> List[str]:
    """One command or sub-action: its description, usage line, examples and
    options, then its sub-actions one heading level down."""
    out = [f"{level} {title}\n"]
    if parser.description:
        out.append(_escape(_flat(parser.description)) + "\n")
    out.append(f"Usage: `{usage_line(title, parser)}`\n")
    if EXAMPLES.get(title):
        out.append("```bash\n" + "\n".join(EXAMPLES[title]) + "\n```\n")
    sub = _subparsers(parser)
    if sub is not None:
        helps = {a.dest: a.help for a in sub._choices_actions}
        rows = ["| Command | Description |", "|---|---|"]
        for name in sub.choices:
            full = f"{title} {name}"
            rows.append(f"| [`{full}`](#{_anchor(full)}) | {_escape(_flat(helps.get(name)))} |")
        out.append("\n".join(rows) + "\n")
    out += options_tables(parser)
    if title in SHOW_EPILOG and parser.epilog:
        out.append("```text\n" + parser.epilog.strip("\n") + "\n```\n")
    if title in NOTES:
        out.append(NOTES[title])
    if sub is not None:
        for name, child in sub.choices.items():
            out += section(f"{title} {name}", child, level + "#")
    return out


def _global_options() -> List[str]:
    from neuroatlas.cli import GLOBAL_OPTIONS, USAGE

    rows = ["| Option | Description |", "|---|---|"]
    for flags, text in GLOBAL_OPTIONS:
        cell = ", ".join(f"`{f.strip()}`" for f in flags.split(","))
        rows.append(f"| {cell} | {_escape(text)} |")
    return [
        "## Global options\n",
        "These options work with every command, before or after the command name.\n",
        f"Usage: `{USAGE.split(': ', 1)[1]}`\n",
        "\n".join(rows) + "\n",
        "Every command exits with status 0 when it succeeds, 1 when a run fails and 2 "
        "when the command line is wrong.\n",
    ]


def _check_groups(commands) -> None:
    placed = [name for _, names in GROUPS for name in names] + list(OTHER_NAMES)
    missing = [name for name in commands if name not in placed]
    unknown = [name for name in placed if name not in commands]
    if missing or unknown:
        raise SystemExit(f"error: _gendocs.GROUPS does not match the commands "
                         f"(missing: {missing}, unknown: {unknown})")


def render() -> str:
    from neuroatlas.cli import COMMANDS, command_parser

    _check_groups(COMMANDS)
    out = [INTRO]
    out += _global_options()
    out.append("## Commands\n")
    rows = ["| Group | Command | Description |", "|---|---|---|"]
    for group, names in GROUPS:
        for i, name in enumerate(names):
            rows.append(f"| {group if i == 0 else ''} | [`{name}`](#{_anchor(name)}) | "
                        f"{_escape(COMMANDS[name].summary)} |")
    out.append("\n".join(rows) + "\n")
    for group, names in GROUPS:
        out.append(f"## {group}\n")
        for name in names:
            out += section(name, command_parser(name), "###")
        if group == "Data and weights":
            others = " and ".join(f"`neuroatlas {n}`" for n in OTHER_NAMES)
            targets = " and ".join(f"`neuroatlas {t}`" for t in OTHER_NAMES.values())
            out.append(f"{others} are other names for {targets}.\n")
    return "\n".join(out)


def examples() -> Iterator[Tuple[str, List[str]]]:
    """(command or sub-action, the words after ``neuroatlas``) of every example."""
    for title, lines in EXAMPLES.items():
        for line in lines:
            words = shlex.split(line, comments=True)
            if words[:1] != ["neuroatlas"]:
                raise ValueError(f"an example that is not a neuroatlas command: {line}")
            yield title, words[1:]


def target() -> Path:
    root = _paths.checkout_root()
    if root is None:
        raise SystemExit("error: run this from a source checkout")
    return root / "docs" / "cli.md"


def main(argv: List[str] | None = None) -> None:
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--check", action="store_true")
    opts = args.parse_args(argv)
    text = render()
    path = target()
    if opts.check:
        if not path.is_file() or path.read_text() != text:
            print(f"{path} is stale: run `python -m neuroatlas.cli._gendocs`", file=sys.stderr)
            raise SystemExit(1)
        print(f"{path} is current")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
