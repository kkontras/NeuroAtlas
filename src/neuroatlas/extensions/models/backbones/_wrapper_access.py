"""Reaching the nn.Module inside a backbone wrapper.

Wrappers are not themselves modules and disagree on where they keep theirs.
Any code that needs the real module -- a contract check, a parameter count,
a precision sweep -- needs this, so it is here rather than inside whichever
caller happened to need it first.
"""
from __future__ import annotations

import torch.nn as nn


#: Where each wrapper keeps its module, in the order they are tried.
_INNER_ATTRS = ("model", "_model", "_module", "_embed_model", "_model_inner")


def inner_module(backbone) -> nn.Module:
    """The ``nn.Module`` inside a :class:`BenchmarkBackbone` wrapper.

    Wrappers are not themselves modules and disagree on the attribute name:
    ``self.model`` (most), ``self._model`` (LaBraM, S-JEPA), ``self._module``
    (Moirai), ``self._embed_model`` (MOMENT), ``self._model_inner`` (Lag-Llama).
    A wrapper may also override the ``trainable_module`` property when the module
    is nested out of reach — Chronos keeps it at ``pipeline.model``.

    Note that LaBraM and S-JEPA build their inner module *lazily* on the first
    batch, so call this only after a warm-up forward.
    """
    declared = getattr(type(backbone), "trainable_module", None)
    if declared is not None:
        module = backbone.trainable_module
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"{type(backbone).__name__}.trainable_module returned "
                f"{type(module).__name__}, expected nn.Module"
            )
        return module

    for attr in _INNER_ATTRS:
        module = getattr(backbone, attr, None)
        if isinstance(module, nn.Module):
            return module

    candidates = {
        name: value
        for name, value in vars(backbone).items()
        if isinstance(value, nn.Module)
    }
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    if not candidates:
        raise AttributeError(
            f"{type(backbone).__name__}: no nn.Module found. Probed "
            f"{list(_INNER_ATTRS)} and scanned instance attributes. If the module "
            f"is built lazily, run a warm-up forward first; if it is nested, add a "
            f"'trainable_module' property to the wrapper."
        )
    raise AttributeError(
        f"{type(backbone).__name__}: ambiguous inner module, found "
        f"{sorted(candidates)}. Add a 'trainable_module' property to the wrapper "
        f"to say which one fine-tuning should adapt."
    )


# ---------------------------------------------------------------------------
# Name classification
# ---------------------------------------------------------------------------
