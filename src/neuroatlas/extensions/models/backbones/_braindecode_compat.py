"""Compatibility shims for importing braindecode on hosts with a broken torchaudio.

On some anonorg compute nodes (observed on `anonhost.anonorg.example.invalid`), the
installed `torchaudio` .so has an undefined symbol mismatch against the local
libstdc++:

    OSError: .../torchaudio/lib/libtorchaudio.so: undefined symbol:
             _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11...

`braindecode` imports `torchaudio.functional` unconditionally at module load
(`braindecode/modules/filter.py`) even though the wrappers we use
(CBraMod, LaBraM, …) never exercise the audio path.

This module injects a minimal `torchaudio` stub into `sys.modules` when the
real package can't be loaded, so `import braindecode.models` succeeds.  Call
``ensure_torchaudio_stub()`` before the first ``from braindecode.X import …``
in each wrapper that needs braindecode.

This is the extracted, shareable
version.
"""
from __future__ import annotations

import sys


def ensure_torchaudio_stub() -> bool:
    """If torchaudio can't be imported, install a minimal stub.  Returns True
    if the stub was injected (so callers can record provenance in metadata)."""
    # Unconditionally try the real torchaudio — only fall through to the stub
    # when the import genuinely fails. Previously the second branch below was
    # guarded by `if not need_stub`, which meant a fresh process (torchaudio
    # not yet in sys.modules) skipped the real-import attempt and went
    # straight to stub installation, masking the working torchaudio on hosts
    # where it loads fine (e.g. anonhost) and breaking downstream
    # `torchaudio.functional.resample` callers.
    try:
        import torchaudio as _ta  # noqa: F401
        return False
    except (ImportError, OSError):
        import types

        _ta = types.ModuleType("torchaudio")
        _ta.__path__ = []  # make it a package

        _taf = types.ModuleType("torchaudio.functional")
        _taf.fftconvolve = None
        _taf.filtfilt = None

        _tat = types.ModuleType("torchaudio.transforms")
        _tat.Resample = type("Resample", (), {"__init__": lambda *a, **kw: None})

        _ta.functional = _taf
        _ta.transforms = _tat

        sys.modules["torchaudio"] = _ta
        sys.modules["torchaudio.functional"] = _taf
        sys.modules["torchaudio.transforms"] = _tat
        return True
