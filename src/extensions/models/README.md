# Model Extensions

This is the default entrypoint for new benchmark model families and checkpoints.

Edit here when you are:

- adding a model-family spec
- declaring checkpoint specs
- wiring a backbone loader

Canonical pattern:

- `src/extensions/models/<slug>.py`
- `src/extensions/models/backbones/<slug>.py`

Do not add model-facing benchmark code outside `extensions/models/`.

## Backbone wrapper contract

Every file under `backbones/` must follow this contract. See `AGENT_GUIDE.md §7.1` for the owner-of-each-concern split, and `AGENT_GUIDE_ONMODELS.md` for the per-model pretraining recipes.

1. **Model-specific transforms only.** Inside `_prepare_input`, do fs resampling, amplitude scaling, amplitude clip (when tied to the scale), channel adapter, and reference change. Do not bandpass / notch filter — that is the dataset preprocessor's job. Do not silently resize the time-window length.

2. **Explicit window contract.** State in the module docstring what window sizes the backbone accepts (strict fixed or patch-multiple). If the caller passes a non-matching length, raise `ValueError` with a message naming the accepted sizes. Never stretch via `F.interpolate`, never silently pad.

3. **Strict weight load with inline allowlist.** Replace any `load_state_dict(..., strict=False)` with a helper that calls `strict=False` and then raises unless `(missing - allowed_missing)` and `(unexpected - allowed_unexpected)` are both empty. Populate the allowlist from a one-time audit on the pretrained checkpoint.

4. **First-call `logger.info` banner** (guarded by `self._banner_logged`): one line summarising `fs`, window contract, scale, reference, and target-channel count. Printed once per backbone instance.

5. **Provenance via `metadata()` + `_last_*`.** Populate per-batch reports (channel mapping, resample method, clip fraction, scale applied, channels actually used) into `self._last_*` dicts; merge them into `metadata()` via `**super().metadata()`. The existing task wiring (`linear_probe.py`, `seizure_detection.py`, …) already carries this into `BenchmarkResult.metadata`.

6. **Use the shared helpers** in `backbones/_preproc.py`:
   - `resample_poly_with_fallback(x, src_fs, dst_fs)` — fs conversion only.
   - `car_reference(x)` — channel-axis mean; correct CAR.
   - `assert_finite(x, where)` — NaN/Inf guard with descriptive message.
   - `assert_amplitude_band(x, lo_uv, hi_uv, where)` — unit-sanity guard (catches V↔µV↔mV mixups).

7. **No silent fallbacks.** If batch metadata is missing required fields (e.g. channel names), raise — do not switch to a passthrough mode. If a channel-match ratio drops below threshold, raise. If a weight NaN would be patched, raise instead and have the caller fix the checkpoint.

See any migrated backbone (`cbramod.py`, `labram.py`, `eegpt.py`) for a reference implementation of this contract.
