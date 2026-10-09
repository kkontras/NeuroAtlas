"""Shared EDF physical-dimension → µV scale table.

Hoisted from ``chbmit_bids.py`` once TUSZ became the third call site. Keeping
one table means every EDF-reading path handles the rare ``mV``/``V`` declarations
identically.
"""
from __future__ import annotations


def edf_unit_to_uv_scale(unit: str) -> float:
    """Return the multiplier that converts EDF-declared samples to µV."""
    u = unit.lower().replace("μ", "u").strip()
    if u in {"uv", "microvolt", "microvolts"}:
        return 1.0
    if u in {"mv", "millivolt", "millivolts"}:
        return 1e3
    if u in {"v", "volt", "volts"}:
        return 1e6
    raise ValueError(
        f"Unsupported EDF physical dimension {unit!r}; expected uV/mV/V. "
        "If the dataset genuinely uses other units, extend edf_unit_to_uv_scale."
    )
