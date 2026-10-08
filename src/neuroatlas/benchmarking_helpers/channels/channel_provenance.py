"""Name-only channel-mapping provenance for backbones.

This module answers, for a given ``(model_family, input_channels)`` pair,
*which* model target slots get **filled** (and from which input channel),
which are **zero-inserted** (a fixed slot with no real source), and which
input channels are **dropped** — using ONLY channel-name strings (no
signals, no model weights, no GPU).

This module holds only the vocabulary -- the two dataclasses below. The
provenance itself is produced by each backbone's own name-mapping code (the
same functions the wrapper runs at inference), so the view cannot drift from
runtime behaviour. A backbone that supports introspection exposes

    def describe_channel_mapping(channels: list[str]) -> ChannelMapping

and is called directly; nine currently do. A by-family dispatcher used to
live here for the audit tree, which is gone, so it was removed rather than
left as an unreachable entry point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# Slot status vocabulary:
#   filled        — a real input channel feeds this target slot
#   derived       — slot synthesised from two inputs (e.g. BIOT A-B bipolar)
#   approx        — anatomical approximation (single electrode for a pair, or
#                   bipolar value placed at a monopolar anchor)
#   zero_inserted — fixed target slot with NO source → all-zero at runtime
#   dropped       — input channel not used by the model (status on a slot
#                   whose ``target`` is the dropped input label)
_STATUS = ("filled", "derived", "approx", "zero_inserted", "dropped")


@dataclass(frozen=True)
class ChannelSlot:
    target: str            # model slot/target label (or input label, agnostic)
    source: Optional[str]  # input channel feeding it; None if zero-inserted
    status: str            # one of _STATUS
    note: str = ""         # optional reason / alias-kind


@dataclass(frozen=True)
class ChannelMapping:
    model_family: str
    layout_kind: str             # "fixed" | "variable" | "agnostic"
    slots: List[ChannelSlot] = field(default_factory=list)
    dropped: List[ChannelSlot] = field(default_factory=list)
    error: Optional[str] = None  # set if the real mapping would RAISE here

    @property
    def zero_inserted(self) -> List[str]:
        return [s.target for s in self.slots if s.status == "zero_inserted"]

    @property
    def filled(self) -> List[ChannelSlot]:
        return [s for s in self.slots if s.status != "zero_inserted"]
