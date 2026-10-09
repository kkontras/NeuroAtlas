"""The run the runner is working on, for messages said deep inside it.

A backbone, a reader or a probe that warns about what it meets can name the
dataset, the checkpoint and the fold it met it on without having them passed
down: the runner sets them for the length of one run (:func:`working_on`),
and :func:`where` / :func:`current` read them back. Outside a run (the API,
a test) nothing is set and they return what the caller gives as default.
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Iterator, NamedTuple, Optional


class Pair(NamedTuple):
    dataset: str
    checkpoint: str
    fold: Optional[Any] = None


_CURRENT: "contextvars.ContextVar[Optional[Pair]]" = contextvars.ContextVar(
    "neuroatlas_pair", default=None)


@contextlib.contextmanager
def working_on(dataset: str, checkpoint: str, fold: Optional[Any] = None) -> Iterator[Pair]:
    """The run on *dataset* with *checkpoint* (and *fold*, when it has one)."""
    token = _CURRENT.set(Pair(str(dataset), str(checkpoint), fold))
    try:
        yield _CURRENT.get()
    finally:
        _CURRENT.reset(token)


def current() -> Optional[Pair]:
    """The run being worked on, or None outside one."""
    return _CURRENT.get()


def where(default: str = "", *, fold: bool = True) -> str:
    """``siena/biot_pretrained fold 2`` (``siena/biot_pretrained`` with
    ``fold=False`` or for a run with no fold), or *default* outside a run."""
    pair = _CURRENT.get()
    if pair is None:
        return default
    text = f"{pair.dataset}/{pair.checkpoint}"
    if fold and pair.fold is not None:
        text += f" fold {pair.fold}"
    return text
