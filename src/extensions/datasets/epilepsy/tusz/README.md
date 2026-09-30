# TUSZ v2.0.3 preprocessor

Converts the Temple University Hospital Seizure Corpus (EDF + TUSZ annotations) into the standard continuous-HDF5 cache.  TUSZ is the largest dataset we ship; building it on a single machine would take many hours, so the pipeline is explicitly split-aware and shardable for Condor.

- **Raw data layout.**  `{raw_root}/{split}/{subject}/{session}/{montage_dir}/{stem}.edf` with matching `{stem}.csv_bi` (binary seizure labels on the TERM whole-recording channel) and `{stem}.csv` (per-channel multi-class seizure types).
- **Splits.**  TUSZ ships with its own `train` / `dev` / `eval` split.  We keep each one intact *and* produce an `all` merge on top (needed for subject-disjoint k-fold).
- **Sibling metadata.**  TUSZ itself has no age / sex / diagnosis fields.  We join against the separate `tuh_eeg_epilepsy` corpus if its root is supplied (see `readers.load_tuh_eeg_epilepsy_metadata`).
- **Voltage units.**  `read_edf_unipolar` queries `pyedflib.EdfReader.getPhysicalDimension` per channel, eagerly scales to µV via the shared `dataio/_edf_units.edf_unit_to_uv_scale` table (×1 / ×1e3 / ×1e6 for uV/mV/V), and persists the raw declarations into a `physical_units` dataset of shape `(n_recordings, 19)` (byte-strings, ordered as `UNIPOLAR_ELECTRODES`; missing electrodes are the empty string).  `TUSZContinuousDataset` forwards the per-recording row into `meta["physical_units"]` on every window.  Legacy caches that predate this field are read with an implicit-uV fallback, matching the prior behaviour.

## Files

| File             | What it owns                                                                                    |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| `readers.py`     | Montage constants, channel-name normalisation, EDF → 19-unipolar resampler, TUH-Epilepsy join   |
| `annotations.py` | 13-class seizure taxonomy, `.csv` / `.csv_bi` parsers, samplewise label builders, event arrays  |
| `preprocessor.py`| `build_cache`, `merge_shards`, `merge_splits`, `summarize_cache`                                |
| `__init__.py`    | Re-exports of everything public                                                                 |

## Three-step build pipeline

1. **Shard build** (`build_cache`) — per-split HDF5 file, optionally sharded via `shard_index / num_shards` for Condor parallelism.  One worker reads a subset of EDFs, resamples them to 256 Hz, writes them end-to-end.
2. **Shard merge** (`merge_shards`) — combines `tusz_<ver>_<split>_<tag>_shardNNN.h5` files into `tusz_<ver>_<split>_<tag>.h5`.  Recording offsets, `event_recording_idx` and `event_channel_tcp` are shifted to stay globally consistent.
3. **Split merge** (`merge_splits`) — concatenates `train`, `dev`, `eval` into `tusz_<ver>_all_<tag>.h5` and adds a `split_labels` dataset that records which split each recording came from.  This is what the k-fold dataio reads.

Pipeline commands:

```bash
# One shard on Condor
python -m extensions.datasets.preprocessors.preprocess_tusz \
    --raw-root /anonorg/.../TUSZ/v2.0.3/edf \
    --cache-root /anonorg/.../caches/tusz \
    --split train --shard-index $CONDOR_TASK_ID --num-shards 16

# Merge shards per-split, then merge splits
python -m entrypoints.merge_tusz_shards \
    --cache-root /anonorg/.../caches/tusz --splits train dev eval --merge-splits
```

## Key design points

- **Channel normalisation.**  TUSZ labels are inconsistent (`EEG FP1-REF`, `T7-LE`, `Fp1`, etc.).  `normalize_channel_name` strips the `EEG ` prefix and the `-REF`/`-LE`/`-AR`/`-AVG`/`-A1`/`-A2` reference suffix, then applies the 10-10 → 10-20 alias map (`T7→T3`, `T8→T4`, `P7→T5`, `P8→T6`).
- **Per-channel variable fs.**  Individual channels in the same EDF can have different sampling rates.  Channels are resampled **individually** to 256 Hz before concatenation.
- **Missing channels are zero-filled.**  Some TUSZ recordings are missing a few electrodes.  They are zero-filled; the count goes to `n_missing_electrodes` so a downstream filter can exclude them.
- **Label building has a specificity rule.**  A sample inside both a generic `seiz` annotation and a specific type (`fnsz`, `gnsz`, ...) ends up labelled with the specific type — the generic `seiz` code is only written where the array is still zero.  See `annotations.build_samplewise_types`.
- **13-class seizure taxonomy.**  `SEIZURE_TYPE_CODES` goes `bckg=0, seiz=1, fnsz=2, ..., nesz=12`.  **Deliberately different from EPILEPSIAE's 4-class taxonomy** — cross-dataset models must map between them explicitly.
- **Two bipolar montages.**  We keep both the 18-pair `BIPOLAR_MONTAGE` (no ear electrodes) and the 22-pair `TCP_MONTAGE_CHANNELS` (with `A1`/`A2`) because TUSZ annotations reference the TCP channel names.  The dataio layer uses `BIPOLAR_MONTAGE`; event arrays store `event_channel_tcp` as an index into `TCP_MONTAGE_CHANNELS`.
- **Bipolar axis convention.**  This sub-package's `bipolar_from_unipolar` is **time-first** (`(T, 19) → (T, 18)`) because the dataio consumes `(T, C)` arrays.  The shared `_common.bipolar_from_unipolar` is channel-first (`(..., 19, T) → (..., 18, T)`).  Both names coexist on purpose — see the parent `README.md`.
- **Memory-bounded `pos_fraction`.**  For caches larger than ~50 M samples, the final `pos_fraction` attribute is computed by a chunked scan of `samplewise_label` instead of `np.sum(ds[:])`, which would otherwise read the entire array back from disk.

## Public API

```python
from extensions.datasets.epilepsy.tusz import (
    build_cache, merge_shards, merge_splits, summarize_cache,          # preprocessor.py
    UNIPOLAR_ELECTRODES, BIPOLAR_MONTAGE, TCP_MONTAGE_CHANNELS,        # readers.py
    TARGET_FS, CACHE_SCHEMA_TAG, SPLITS,
    normalize_channel_name, bipolar_from_unipolar, read_edf_unipolar,
    discover_recordings, load_tuh_eeg_epilepsy_metadata,
    SEIZURE_TYPE_CODES, SEIZURE_TYPE_NAMES,                            # annotations.py
    parse_csv_bi, parse_csv_multiclass,
    build_samplewise_labels, build_samplewise_types, build_event_arrays,
)
```

Or equivalently via the backward-compat shim at the parent package:

```python
from extensions.datasets.epilepsy.tusz_preprocessor import (
    build_cache, ...
)
```

## Related reading

- `../README.md` — parent package overview and shared cache schema.
