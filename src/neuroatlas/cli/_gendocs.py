"""Write the command reference, docs/reference/, from the command's own
parsers, so the reference cannot drift from the code.

    python -m neuroatlas.cli._gendocs            # rewrite docs/reference/
    python -m neuroatlas.cli._gendocs --check    # exit 1 if a page is stale

The reference is one page per group of commands, plus index.md with the
global options and the table of commands. Descriptions, options and defaults
come from the parsers. What is written here by hand: the groups, a line that
opens each group's page, and the examples (each one is parsed by the
command's own parser in the tests, so it stays a real command line).
"""
from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from neuroatlas import _paths

#: The pages of the reference, in order, and the commands on each.
GROUPS: List[Tuple[str, List[str]]] = [
    ("Setup", ["config"]),
    ("Explore", ["list", "show"]),
    ("Data and weights", ["data", "models"]),
    ("Run", ["check", "run", "submit", "status"]),
    ("Results", ["results", "rescore"]),
    ("Individual steps", ["embed", "probe", "hypnogram"]),
]

#: The line that opens each group's page, with the user guide chapter to read.
GROUP_INTROS: Dict[str, str] = {
    "Setup": "`config` sets where NeuroAtlas reads datasets and writes caches, results and "
             "model weights. [Configuration](../guide/configuration.md) explains the settings.",
    "Explore": "These commands describe the benchmarks, datasets, models and tasks without "
               "running anything. [Benchmarks](../guide/benchmarks.md) explains what they show.",
    "Data and weights": "These commands check, download and prepare the datasets and the model "
                        "weights. See [Data](../guide/data.md) and [Models](../guide/models.md).",
    "Run": "These commands test a benchmark, run it on this machine, or write it as cluster "
           "jobs and follow them. See [Running benchmarks](../guide/running.md).",
    "Results": "These commands summarise the results and recompute their metrics from the saved "
               "predictions. See [Results](../guide/results.md).",
    "Individual steps": "`run` executes these steps for you. Call them yourself to try other "
                        "settings. See [The individual steps]"
                        "(../guide/advanced.md#71-the-individual-steps).",
}

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
             "`neuroatlas embed --dataset hmc --help` for the `--set` keys of one dataset.",
    "probe": "Run `neuroatlas probe --help` for the list of datasets, tasks and models, and "
             "`neuroatlas probe --dataset hmc --help` for the `--set` keys of one dataset.",
}

#: Commands whose --help ends with text worth showing here. The others end
#: with the long dataset lists of the terminal help.
SHOW_EPILOG = ("status", "hypnogram")

#: The first line of every page: hidden on the site, seen by whoever edits it.
GENERATED = ("<!-- Generated from the command-line parsers by "
             "`python -m neuroatlas.cli._gendocs`. Do not edit by hand. -->\n")

INTRO = """\
# Command reference

This section lists every `neuroatlas` command and its options, one page per
group of commands. Run `neuroatlas <command> --help` to see the same text in
the terminal. The [user guide](../guide/index.md) shows how the commands fit
together.
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


def page_name(group: str) -> str:
    """The file of a group's page, as in ``data-and-weights.md``."""
    return group.lower().replace(" ", "-") + ".md"


def _note(text: str) -> str:
    """A note box: an admonition on the site, a note on GitHub."""
    return "> [!NOTE]\n> " + text + "\n"


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
        out.append(_note(NOTES[title]))
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


def pages() -> Dict[str, str]:
    """Every page of the reference, by file name: index.md, then one page per group."""
    from neuroatlas.cli import COMMANDS, command_parser

    _check_groups(COMMANDS)
    index = [GENERATED + INTRO]
    index += _global_options()
    index.append("## Commands\n")
    rows = ["| Group | Command | Description |", "|---|---|---|"]
    for group, names in GROUPS:
        page = page_name(group)
        for i, name in enumerate(names):
            cell = f"[{group}]({page})" if i == 0 else ""
            rows.append(f"| {cell} | [`{name}`]({page}#{_anchor(name)}) | "
                        f"{_escape(COMMANDS[name].summary)} |")
    index.append("\n".join(rows) + "\n")
    out = {"index.md": "\n".join(index)}
    for group, names in GROUPS:
        text = [GENERATED + f"# {group}\n", GROUP_INTROS[group] + "\n"]
        for name in names:
            text += section(name, command_parser(name), "##")
        if group == "Data and weights":
            others = " and ".join(f"`neuroatlas {n}`" for n in OTHER_NAMES)
            targets = " and ".join(f"`neuroatlas {t}`" for t in OTHER_NAMES.values())
            text.append(_note(f"{others} are other names for {targets}."))
        out[page_name(group)] = "\n".join(text)
    return out


def render() -> str:
    """All pages as one text, index.md first (for searching them in tests)."""
    return "\n".join(pages().values())


def examples() -> Iterator[Tuple[str, List[str]]]:
    """(command or sub-action, the words after ``neuroatlas``) of every example."""
    for title, lines in EXAMPLES.items():
        for line in lines:
            words = shlex.split(line, comments=True)
            if words[:1] != ["neuroatlas"]:
                raise ValueError(f"an example that is not a neuroatlas command: {line}")
            yield title, words[1:]


def target() -> Path:
    """The folder of the reference pages, docs/reference/ of the checkout."""
    root = _paths.checkout_root()
    if root is None:
        raise SystemExit("error: run this from a source checkout")
    return root / "docs" / "reference"


def stale(folder: Optional[Path] = None) -> List[Path]:
    """The pages that differ from what the parsers give, are missing, or are
    no longer generated. Empty when the reference is current."""
    folder = folder or target()
    wanted = pages()
    out = [folder / name for name, text in wanted.items()
           if not (folder / name).is_file() or (folder / name).read_text() != text]
    out += sorted(p for p in folder.glob("*.md") if p.name not in wanted)
    return out


def write(folder: Optional[Path] = None) -> List[Path]:
    """Write every page, and remove the pages that are no longer generated
    (the folder holds the generated pages only). Returns the pages written."""
    folder = folder or target()
    folder.mkdir(parents=True, exist_ok=True)
    wanted = pages()
    for path in folder.glob("*.md"):
        if path.name not in wanted:
            path.unlink()
    for name, text in wanted.items():
        (folder / name).write_text(text)
    return [folder / name for name in wanted]


def main(argv: List[str] | None = None) -> None:
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--check", action="store_true")
    opts = args.parse_args(argv)
    folder = target()
    if opts.check:
        out_of_date = stale(folder)
        if out_of_date:
            names = ", ".join(str(p) for p in out_of_date)
            print(f"stale: {names}; run `python -m neuroatlas.cli._gendocs`", file=sys.stderr)
            raise SystemExit(1)
        print(f"{folder} is current")
        return
    written = write(folder)
    print(f"wrote {len(written)} pages to {folder}")


if __name__ == "__main__":
    main()
