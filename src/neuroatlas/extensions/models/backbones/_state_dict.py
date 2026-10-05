"""Strict state-dict loading whose error names every mismatch on one line.

``torch.nn.Module.load_state_dict(strict=True)`` puts the summary on the first
line and the missing / unexpected keys on the following ones, and a report that
keeps only the first line (``check``'s table) then shows nothing useful:
``Error(s) in loading state_dict for _BIOTEncoder:``. :func:`load_strict`
raises with the whole diagnosis -- and the installed library versions, since
a mismatch with a known-good checkpoint usually means the architecture changed
under it -- in the first line.
"""
from __future__ import annotations

from typing import Iterable, Mapping


def _versions(packages: Iterable[str]) -> str:
    from importlib import metadata

    out = []
    for name in packages:
        try:
            out.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            out.append(f"{name} not installed")
    return ", ".join(out)


def describe_mismatch(module, state: Mapping) -> str:
    """'' when *state* loads into *module* exactly, else what differs."""
    own = module.state_dict()
    missing = sorted(k for k in own if k not in state)
    unexpected = sorted(k for k in state if k not in own)
    shapes = sorted(f"{k} {tuple(state[k].shape)} vs model {tuple(own[k].shape)}"
                    for k in own if k in state and hasattr(state[k], "shape")
                    and tuple(state[k].shape) != tuple(own[k].shape))
    parts = []
    if missing:
        parts.append(f"missing {len(missing)}: {', '.join(missing)}")
    if unexpected:
        parts.append(f"unexpected {len(unexpected)}: {', '.join(unexpected)}")
    if shapes:
        parts.append(f"shape mismatch {len(shapes)}: {', '.join(shapes)}")
    return "; ".join(parts)


def load_strict(module, state: Mapping, what: str, packages: Iterable[str] = ("torch",)) -> None:
    """load_state_dict(strict=True), raising with every mismatch on one line."""
    problem = describe_mismatch(module, state)
    if problem:
        raise RuntimeError(
            f"{what} does not match the model built by {_versions(packages)}: {problem}")
    module.load_state_dict(state, strict=True)
