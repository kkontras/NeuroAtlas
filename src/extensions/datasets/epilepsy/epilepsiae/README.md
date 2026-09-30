# EPILEPSIAE preprocessor

Converts the three surface variants of the EPILEPSIAE corpus (`surf30`, `surfPA`, `surfCO`) into the standard continuous-HDF5 cache.  EPILEPSIAE is the only one of the four epilepsy datasets in this package that ships as a proprietary binary format rather than EDF or BIDS, so it has its own sub-package.

- **Raw data layout.**  Each recording is a sequence of `*.head` / `*.data` block pairs with a timestamp in the `.head`.  Adjacent blocks may have gaps; we concatenate them end-to-end and fill gaps with zeros marked by `GAP_LABEL=255` in `samplewise_label`.
- **Annotations.**  Seizure times are in `Annotation/Seizure_annotations.csv`; the origin electrode (when known) is in `Annotation/Annotations_origin.csv`; patient demographics are in per-patient SQL dumps under `Annotation/Metadata/`.

## Files

| File             | What it owns                                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| `readers.py`     | `.head` / `.data` binary reader, channel-name aliases, canonical-channel selection, `BlockMeta`. Amplitude scaling: `conversion_factor` is extracted in `read_head()` and applied in `load_block_signals()` (`signals * conversion_factor`). Multi-center acquisition means per-patient factors are sometimes miscalibrated — enable `EPILEPSIAE_AUTO_RESCALE=1` at Condor submit time to auto-detect V→µV (×1e6) or mV→µV (×1e3) when `abs_max` falls below the µV band. One warning per rescaled block; bit-identical when the env is unset. |
| `annotations.py` | `SeizureEvent`, CSV parsers, onset/offset resolution, SQL metadata parser                        |
| `labels.py`      | `build_samplewise_labels` — resolves datetime seizure annotations to sample indices              |
| `preprocessor.py`| `EpilepsiAEPreprocessor` orchestrator (two-pass layout + stream-write) and `merge_shards`        |
| `__init__.py`    | Re-exports of everything public                                                                  |

## Key design points

- **Two-pass build.**  Pass 1 reads only `.head` files to compute the total output size and label arrays.  Pass 2 streams each block through `read → resample → filter → write` without ever holding more than one block in memory.  This keeps peak RAM bounded regardless of recording length.
- **Inter-block gaps.**  Gaps ≥ `GAP_THRESHOLD_S` (2 s) are inserted as zero-filled regions marked with `GAP_LABEL=255` in both `samplewise_label` and `samplewise_type`.  The dataio layer excludes windows that fall entirely inside a gap.
- **Channel normalisation.**  The three variants use inconsistent channel naming (titlecase `Fp1`, 10-10 names `T7/T8/P7/P8`, mixed caps).  `readers.CHANNEL_ALIAS` collapses them all to the canonical uppercase 10-20 set before channel selection.
- **Sharding.**  `EpilepsiAEPreprocessor.build_cache(shard_index=N, num_shards=K)` processes a deterministic patient subset, writing `shard_NNN_of_KKK.h5`.  `merge_shards` then concatenates shards into the final cache and fixes up event recording indices.
- **Seizure taxonomy (4-class).**  `SEIZURE_TYPE_CODES = {"UC": 0, "SP": 1, "CP": 2, "SG": 3}` — **deliberately different from TUSZ** (13-class).  Cross-dataset classification heads must map between them explicitly.
- **Unit mismatch (opt-in rescale).**  Per-patient `conversion_factor` values in the `.head` files occasionally export data in V or mV instead of µV. The downstream `assert_amplitude_band` in the model wrappers then raises. Set `EPILEPSIAE_AUTO_RESCALE=1` in the Condor job environment to rescale detected mis-scaled blocks (×1e6 for V, ×1e3 for mV) with a per-block warning log. Off by default — strict behaviour preserved for reproducibility.

## Public API

```python
from extensions.datasets.epilepsy.epilepsiae import (
    EpilepsiAEPreprocessor, merge_shards,        # preprocessor.py
    SeizureEvent, load_seizure_annotations,      # annotations.py
    seizures_for_recording, load_patient_metadata,
    BlockMeta, PatientMeta, read_head,           # readers.py
    discover_block, load_block_signals,
    normalize_channel_name, select_canonical_channels,
    build_samplewise_labels,                     # labels.py
    SEIZURE_TYPE_CODES, SEIZURE_TYPE_NAMES,
    PATTERN_CODES, PATTERN_NAMES,
)
```

Or equivalently via the backward-compat shim at the parent package:

```python
from extensions.datasets.epilepsy.epilepsiae_preprocessor import (
    EpilepsiAEPreprocessor, ...
)
```

The shim exists only to keep the pre-split import paths working for `dataio/epilepsiae.py`, `adapters/epilepsiae.py`, the CLI entrypoints, and the test suite.  New code should import from the sub-package directly.

## Related reading

- `../README.md` — parent package overview and shared cache schema.

## Persistent per-recording embed cache (`--cache-root`)

The on-the-fly dataset (`dataio/epilepsiae.py`) re-reads and re-filters
`.data` blocks on every LRU miss. Single blocks can exceed 500 MB (many
hours of signal) and `scipy.filtfilt` over 10 M samples × 19 channels is
tens of seconds. At 2 s / 1 s overlap this dominates embedding
wall-clock. The solution is a one-shot preprocess into a compact,
memmap-friendly layout:

```
<cache-root>/<subject_id>/<rec_id>/
  signals.npy      # (19, T) float16 at target_fs, filtered unipolar
  labels.npy       # (T,)   uint8 — samplewise binary seizure label
  types.npy        # (T,)   uint8 — samplewise seizure type (0 = bckg/gap)
  meta.json        # rec_id, subject_id, variant, fs, gap_regions, demographics
  events.json      # seizure events with pattern codes
  .schema_v1       # written last; marks completion
```

Build (sharded, CPU-only, on Condor):

```
condor_submit <the job file, not shipped>     # 50 shards
```

Embed automatically uses the cache when `--cache-root` is passed:

```
python -m entrypoints.embed_epilepsiae \
    --model biot_pretrained --cache-root <cache-root> ...
```

The cached runtime Dataset (`dataio/epilepsiae_cached.py`) is signature-
compatible with the EDF Dataset (same `__getitem__` return dict, same
convenience properties), so every existing entrypoint / adapter keeps
working. Float16 storage quantises to ≈0.1 µV at 100 µV — below sensor
noise; no observable effect on downstream probes. The `.schema_v1`
marker lets preprocess resume mid-run.
