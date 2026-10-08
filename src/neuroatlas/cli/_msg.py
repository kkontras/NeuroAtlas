"""How ``neuroatlas`` talks to its user: three kinds of message, one shape.

The message contract -- every command, every module that prints:

1. **Three channels only.** Results and tables go to stdout. Progress goes to
   stderr (a live line on a terminal; a start and a result line off it).
   Messages go to stderr as ``error:`` / ``warning:`` / ``note:``, the remedy
   on its own ``fix:`` line. Nothing else reaches the terminal without
   ``-v``: no library log lines, no tqdm bar, no ``print`` from task,
   dataset or model code.
2. **Every number says what it is:** the metric, its unit (30 s epoch, 10 s
   window, seizure event, subject, trial, recording), the classes or the
   threshold, what a fold is and what ± is over.
3. **One word per concept,** the same in every table, message and flag (the
   glossary below).
4. **Alive within a second:** a step that can take more than ~2 s shows a
   live line naming what it does, to what, with a count when one is known,
   and ends with one result line with its time (``12s``, ``2m 05s``:
   :func:`neuroatlas.progress.duration`).
5. **Every message says what happened, why, and what to do,** one form per
   remedy::

       missing data          fix: neuroatlas data download <dataset>
       a dataset elsewhere   fix: neuroatlas config set <dataset>.<key> DIR
       missing weights       fix: neuroatlas models download <checkpoint>

   Plain words: no internal file path, module, YAML key or class name in
   user text unless the user must act on it; no em dash or arrow as a
   separator. A ``fix:`` that names a neuroatlas command names a real one
   (``tests/test_fix_Z2_fix_commands.py`` parses every one).

The words (the glossary; tables, messages and flags use these and no
synonyms): *checkpoint* (one model's weights; ``-m`` selects checkpoints),
*family*, *dataset* (never cohort or corpus), *benchmark*, *quick dataset*
(``--dataset single``) and *all datasets* (``--dataset full``), *fold*,
*LOSO*, *variant*, *embeddings*, *probe*, *run* (one checkpoint on one
fold), and for a pair or a run that did not produce a result: *skipped*
(its data or weights are missing here), *ruled out* (the dataset's channel
map excludes the model), *invalid* (the channel map has no entry for the
model's family), *failed* (it ran and broke). Weights are *found*,
*downloadable*, *in Hugging Face cache* or *manual*; a dataset is
*downloadable*, *credentialed*, *manual* or *from the authors*.

Every message that is not ordinary output starts with the word that says
what it is, in lower case, and goes to stderr::

    error: <what went wrong>     the command, or one pair or row of it, failed or was refused
    warning: <what is off>       the command goes on, but results may be affected
    note: <information>          neither (used sparingly)

A message with a remedy gives it on its own line, the command to run first
when there is one::

    error: no weights for neurorvq_eeg_pretrained
      fix: neuroatlas models download neurorvq_eeg_pretrained

Ordinary output -- tables, progress, summaries -- has no prefix and goes to
stdout. The logging formatter (:class:`LogFormatter`) uses the same words.
The lines under a table row start with the same words, or with
``skipped:`` / ``ruled out:`` / ``invalid:`` (:data:`ROW_KINDS`).

Exception texts written for a user have the same shape: the first line says
what is wrong, a line starting ``fix:`` says what to do (:func:`compose`).
Whoever prints one renders it with :func:`format` (or :func:`say`), which
recognises both parts. Texts from elsewhere -- a CUDA out-of-memory error,
an HTTP error, a wrapper's list of channel labels -- are cut to their first
sentence in tables and summaries (:func:`brief`); ``-v`` shows them whole,
and so does a ``--log`` file.

Colour (red error, yellow warning, dim note) only when the stream is a
terminal and ``NO_COLOR`` is not set; never in a --log file or a machine
format.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from typing import List, Optional, Sequence, Tuple

KINDS = ("error", "warning", "note")
#: The words a line under a table row starts with (beside the three kinds):
#: ``skipped`` (data or weights missing here), ``ruled out`` (the channel map
#: excludes the model), ``invalid`` (the channel map has no entry for its family).
ROW_KINDS = ("error", "warning", "note", "skipped", "ruled out", "invalid")
#: Not run because the dataset's channel map excludes the model: the word
#: every table and message uses (`check`, `run`, `submit`, `results`).
RULED_OUT = "ruled out"
#: Not run because the data or the weights are not here.
SKIPPED = "skipped"
#: Not run because the dataset's channel map has no entry for the family.
INVALID = "invalid"
FIX = "fix"
#: Where a shortened message's full text goes (the --log file only, unless -v).
FULL_LOGGER = "neuroatlas.full"
#: Appended to a message :func:`brief` shortened.
MORE = "(-v: full message)"
#: The longest first sentence :func:`brief` keeps whole.
LIMIT = 200

_state = {"verbose": False}

_COLORS = {"error": "\033[1;31m", "warning": "\033[1;33m", "note": "\033[2m",
           "skipped": "\033[33m", "ruled out": "\033[2m", "invalid": "\033[33m", "n/a": "\033[2m",
           FIX: "\033[1m"}
_RESET = "\033[0m"
ANSI = re.compile(r"\r?\x1b\[[0-9;?]*[A-Za-z]")   # colours, and the clear-line code

# `n/a:` is still read (results.json files written before `ruled out`)
_PREFIX = re.compile(r"^\s*(?:neuroatlas(?: [\w-]+)?: )?"
                     r"(error|warning|note|skipped|ruled out|invalid|n/a):\s*", re.IGNORECASE)
_FIX_LINE = re.compile(r"^\s*fix:\s*", re.IGNORECASE)
# a sentence ends at . ! ? followed by a space and a capital, a quote or a bracket
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[`'\"])")


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

def set_verbose(flag: bool) -> None:
    """-v: show every message whole (see :func:`brief`)."""
    _state["verbose"] = bool(flag)


def verbose() -> bool:
    return _state["verbose"]


def use_color(stream) -> bool:
    """Colour only on a terminal, and never with NO_COLOR set (no-color.org)."""
    if "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def paint(kind: str, text: str, color: bool) -> str:
    return f"{_COLORS[kind]}{text}{_RESET}" if color and kind in _COLORS else text


# --------------------------------------------------------------------------
# composing and reading a message
# --------------------------------------------------------------------------

def compose(what: str, fix: Optional[str] = None) -> str:
    """A message as an exception carries it: ``what``, then ``fix: ...`` on its
    own line. :func:`format` renders it with its kind."""
    return what if not fix else f"{what}\n{FIX}: {fix}"


def split(text: str) -> Tuple[Optional[str], List[str], List[str]]:
    """``(kind or None, lines, fixes)`` of a message: a leading ``error:`` (or
    ``warning:``, ``note:``, ``skipped:``, ``n/a:``, or the ``neuroatlas run:
    error:`` argparse writes) is taken off, ``fix:`` lines are set apart."""
    text = str(text or "").strip("\n")
    kind = None
    match = _PREFIX.match(text)
    if match:
        kind, text = match.group(1).lower(), text[match.end():]
    lines, fixes = [], []
    for line in text.splitlines():
        if _FIX_LINE.match(line):
            fixes.append(_FIX_LINE.sub("", line, count=1).strip())
        elif line.strip() or lines:
            lines.append(line.rstrip())
    while lines and not lines[-1].strip():
        lines.pop()
    return kind, lines, fixes


def strip_kind(text: str) -> str:
    """The message without its leading kind word."""
    match = _PREFIX.match(str(text or ""))
    return str(text)[match.end():] if match else str(text)


def first_sentence(text: str, limit: int = LIMIT) -> str:
    """The first sentence of the first line, at most *limit* characters (cut at
    a word, with ``...``)."""
    lines = [line for line in str(text or "").strip().splitlines() if line.strip()]
    line = " ".join(lines[0].split()) if lines else ""
    match = _SENTENCE_END.search(line)
    head = line[:match.start()] if match else line
    if len(head) > limit:
        head = head[:limit].rsplit(" ", 1)[0].rstrip(",;:") + " ..."
    return head


def brief(text: str, *, keep_fix: bool = True, hint: bool = True) -> str:
    """A long text, for a table or a summary: its first sentence, with
    ``(-v: full message)`` when that dropped something (``hint=False``: not,
    for a machine format that carries the whole text in another field); its
    ``fix:`` lines are kept. With -v, the whole text. The whole text also goes
    to the --log file."""
    text = str(text or "")
    kind, lines, fixes = split(text)
    if verbose():
        return text.strip("\n")
    body = "\n".join(lines)
    head = first_sentence(body)
    if " ".join(head.split()) != " ".join(body.split()):
        logging.getLogger(FULL_LOGGER).info("full message: %s", text.strip("\n"))
        if hint:
            head = f"{head} {MORE}"
    if kind:
        head = f"{kind}: {head}"
    return compose(head, fixes[0] if keep_fix and fixes else None) + "".join(
        f"\n{FIX}: {f}" for f in (fixes[1:] if keep_fix else []))


def exception_text(exc: BaseException) -> str:
    """What an exception says, for a user: its message, with the type in front
    when the type says something the message does not (``KeyError: 'x'``,
    ``OutOfMemoryError: CUDA out of memory.``) -- not for the plain
    ValueError / RuntimeError / OSError texts our own code raises."""
    message = str(exc).strip()
    plain = isinstance(exc, (OSError, ImportError)) or type(exc) in (ValueError, RuntimeError)
    if not message:
        return type(exc).__name__
    return message if plain else f"{type(exc).__name__}: {message}"


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def format(kind: str, text: str, fix: Optional[str] = None, *, color: bool = False,
           indent: str = "") -> str:
    """``kind: what`` -- further lines indented two spaces, then each ``fix:``.

    *text* may carry its own kind (which *kind* replaces) and its own fix
    lines; *fix* adds one more."""
    _, lines, fixes = split(text)
    if fix:
        # one remedy per line: a fix of several lines ("A\nfix: B") is that many
        fixes += [f[len(FIX) + 2:] if f.startswith(FIX + ": ") else f
                  for f in fix.splitlines() if f.strip()]
    lines = lines or [""]
    out = [f"{indent}{paint(kind, kind + ':', color)} {lines[0]}".rstrip()]
    out += [f"{indent}  {line}" for line in lines[1:]]
    out += [f"{indent}  {paint(FIX, FIX + ':', color)} {f}" for f in fixes]
    return "\n".join(out)


def lines(kind: str, text: str, fix: Optional[str] = None) -> List[str]:
    """The lines of a message under a table row (no colour: the table adds it)."""
    return format(kind, text, fix).splitlines()


def say(kind: str, text: str, fix: Optional[str] = None, *, file=None) -> None:
    """Print one message to stderr (or *file*)."""
    stream = file if file is not None else sys.stderr
    if stream is not sys.stdout:
        # what a table printed so far lands first: through a pipe stdout is
        # buffered and stderr is not, and a message that says "the lines
        # above" must come after them
        try:
            sys.stdout.flush()
        except (AttributeError, ValueError, OSError):
            pass
    escapes = use_color(stream)     # a terminal that takes escape codes (not NO_COLOR)
    print((CLEAR_LINE if escapes else "") + format(kind, text, fix, color=escapes),
          file=stream, flush=True)


def error(text: str, fix: Optional[str] = None, *, file=None) -> None:
    say("error", text, fix, file=file)


def warning(text: str, fix: Optional[str] = None, *, file=None) -> None:
    say("warning", text, fix, file=file)


def note(text: str, fix: Optional[str] = None, *, file=None) -> None:
    say("note", text, fix, file=file)


def paint_row_note(line: str, color: bool) -> str:
    """Colour the leading word of a line under a table row."""
    if not color:
        return line
    stripped = line.lstrip()
    pad = line[: len(line) - len(stripped)]
    for kind in (*ROW_KINDS, FIX):
        if stripped.startswith(kind + ":"):
            return pad + paint(kind, kind + ":", True) + stripped[len(kind) + 1:]
    return line


def counts(total: int, noun: str, parts: Sequence[Tuple[str, int]], *,
           keep_zero: Sequence[str] = ()) -> str:
    """A footer: ``3 pairs: 2 ok, 1 failed, 0 skipped`` (zero counts only for
    the states in *keep_zero*)."""
    shown = [f"{n} {state}" for state, n in parts if n or state in keep_zero]
    return f"{total} {noun}" + (f": {', '.join(shown)}" if shown else "")


def plural(n: int, one: str, many: Optional[str] = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def legend(entries: Sequence[Tuple[str, str]], indent: str = "  ") -> List[str]:
    """The lines under a table that explain the values that are not
    self-explanatory: ``  downloadable   models download can fetch them``,
    one per (value, meaning), the meanings aligned."""
    if not entries:
        return []
    width = max(len(word) for word, _ in entries)
    return [f"{indent}{word:<{width}}  {meaning}".rstrip() for word, meaning in entries]


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------

_PY_WARNING = re.compile(r"^(?P<path>.*?):(?P<line>\d+): (?P<category>\w+): (?P<message>.*)$", re.S)


# Return to the start of the line and erase it (a terminal only).
CLEAR_LINE = "\r\x1b[K"


class LogFormatter(logging.Formatter):
    """Log records in the same words: ``error: ...``, ``warning: ...``; INFO
    and DEBUG lines (shown with -v, kept in --log) as they are. A Python
    warning reads ``warning: <message> (<Category>, <file>:<line>)``."""

    def __init__(self, color: bool = False, clear_line: bool = False):
        super().__init__("%(message)s")
        self.color = color
        # On a terminal, start each record on a clean line: a live progress
        # line ("probing (1m 05s)") may be on screen without a newline.
        self.clear = CLEAR_LINE if clear_line else ""

    def format(self, record: logging.LogRecord) -> str:
        return self.clear + self._format(record)

    def _format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if record.name == "py.warnings":
            match = _PY_WARNING.match(text.strip())
            if match:
                message = match.group("message").split("\n")[0].strip()
                text = (f"{message} ({match.group('category')}, "
                        f"{os.path.basename(match.group('path'))}:{match.group('line')})")
        if record.levelno >= logging.ERROR:
            kind = "error"
        elif record.levelno >= logging.WARNING:
            kind = "warning"
        else:
            return text
        return format(kind, text, color=self.color)
