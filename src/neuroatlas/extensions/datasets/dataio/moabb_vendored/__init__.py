"""MOABB dataset classes the pinned moabb does not have, vendored.

The BCI stack runs moabb 1.2.0, the last release that accepts numpy<2 (the
paper's numpy 1.26.4); it is installed with ``--no-deps``. Two of the paper's
14 MOABB cohorts arrived later:

=================  =================  =======================================
class              first in moabb     vendored from
=================  =================  =======================================
Dreyer2023 (+A/B/C)  1.4.0 (2025-11)  moabb 1.7.2 ``moabb/datasets/dreyer2023.py``
Kim2025BetaRange   1.5.0 (2026-03)    moabb 1.7.2 ``moabb/datasets/ssvep_kim2025.py``
                                      (+ ``utils.build_raw_from_epochs``)
=================  =================  =======================================

Both are moabb code, BSD-3-Clause (``LICENSE.moabb`` beside this file, which
the licence requires to travel with it). The changes are only what moabb
1.2.0's ``BaseDataset`` needs: its constructor (no ``selected_subjects`` /
``selected_sessions`` / ``return_all_modalities``), its ``download_if_missing``
(no ``force_update``), and no ``METADATA`` block (moabb 1.2.0 has no
``moabb.datasets.metadata``). Downloading, file layout and the raw each
subject yields are upstream's, so data fetched by a newer moabb is found, and
the reverse.

:func:`dataset_class` prefers the installed moabb's own class, so with a
moabb that has them these copies are never used.
"""

from __future__ import annotations

from typing import Optional

#: class name -> (module, first moabb release that ships it)
VENDORED = {
    "Dreyer2023": ("dreyer2023", "1.4.0"),
    "Dreyer2023A": ("dreyer2023", "1.4.0"),
    "Dreyer2023B": ("dreyer2023", "1.4.0"),
    "Dreyer2023C": ("dreyer2023", "1.4.0"),
    "Kim2025BetaRange": ("kim2025", "1.5.0"),
}


def dataset_class(name: str) -> Optional[type]:
    """The MOABB dataset class *name*: the installed moabb's, else the vendored copy.

    None when neither has it. Importing moabb is left to the caller's error
    handling: without moabb this raises ImportError.
    """
    import moabb.datasets as upstream

    cls = getattr(upstream, name, None)
    if cls is not None:
        return cls
    if name in VENDORED:
        import importlib

        module = importlib.import_module(f"{__name__}.{VENDORED[name][0]}")
        return getattr(module, name)
    return None
