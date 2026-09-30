# Epilepsy preprocessing

This package converts four scalp-EEG seizure-monitoring datasets into a single uniform on-disk format so that a downstream model can train on any of them without special-casing.

All four preprocessors write the **same continuous-HDF5 cache schema** (19 unipolar 10-20 channels at 256 Hz with samplewise binary and multi-class seizure labels).  A dataio layer (`extensions/datasets/dataio/*.py`) slides windows across the cache and serves PyTorch tensors; a benchmark adapter (`extensions/datasets/adapters/*.py`) wraps that in a `LightningDataModule` with train / val / test splits.

## Datasets covered

| Dataset        | Source                      | Channels         | Annotation style                     | Filtering (see § "Filter policy" below)                                    | Preprocessor entrypoint                           |
| -------------- | --------------------------- | ---------------- | ------------------------------------ | -------------------------------------------------------------------------- | ------------------------------------------------- |
| **EPILEPSIAE** | Proprietary `.head`/`.data` | 19 unipolar      | `Seizure_annotations.csv` + SQL meta | in-cache: 0.5 Hz HP + **50 Hz** notch (EU mains)                           | `epilepsiae.EpilepsiAEPreprocessor.build_cache`   |
| **TUSZ**       | EDF + `.csv_bi`/`.csv`      | 19 unipolar      | per-channel, 13-class                | in-cache: 0.5 Hz HP + **60 Hz** notch (US mains, Temple University)        | `tusz.build_cache`                                |
| **TUAB**       | EDF                         | 19 unipolar      | file-level binary (normal/abnormal)  | in-cache: 0.5 Hz HP + **60 Hz** notch (US mains, Temple University)        | `tuab_preprocessor.build_cache`                   |
| **CHB-MIT**    | BIDS (Zenodo)               | 18 bipolar       | BIDS `_events.tsv`                   | in-cache: 0.5 Hz HP + **60 Hz** notch (US mains, Boston Children's)        | `chbmit_preprocessor.build_h5_cache_bids`         |
| **Siena**      | BIDS (Zenodo)               | 19 unipolar      | BIDS `_events.tsv`                   | **upstream** 0.1–70 Hz Butterworth BP + 50 Hz Butterworth notch (Detti 2020) — none added in-repo | `siena_preprocessor.build_h5_cache_bids`          |
| **Bonn**       | `.TXT` clips                | single-channel   | segment-level multi-class            | **upstream** 0.53–40 Hz BP at A/D (Andrzejak 2001) — none added in-repo    | `bonn_preprocessor.build_h5_cache`                |
| **AUB-MED**    | EDF (Mendeley)              | 21 → 19 unipolar | `Seizure_times.py` clock-time dict   | **upstream** ≈0.1 Hz HP (1.6 s TC) + 70 Hz LP + 50 Hz notch — none added   | *none — direct-EDF in `dataio/aub_med.py`*        |

### Filter policy

Every dataset goes through one of three filter paths: *in-cache* (we run
`apply_standard_filters` once at cache-build time), *upstream* (the data
provider already applied filtering and we don't double up), or *raw*
(neither; the dataset docstring must say so explicitly).  The shared
utility `_common.apply_standard_filters(signals, fs, notch_hz, axis)`
applies a 0.5 Hz Butterworth-4 zero-phase highpass and a Q=30 IIR notch
at `notch_hz`.  Pass `notch_hz=50.0` for EU recordings and `notch_hz=60.0`
for US recordings; pass `axis=0` for time-first `(T, C)` arrays (TUSZ /
TUAB readers) and keep the default `axis=1` for channel-first `(C, T)`
arrays (EPILEPSIAE / CHB-MIT readers).  Backbone wrappers must not
re-filter per batch — see `AGENT_GUIDE_ONMODELS.md`.

*CHB-MIT writes the 18-channel bipolar montage as-is (no unipolar step), so its cache has `n_channels=18` rather than 19.  Everything else in the schema matches.*

*AUB-MED goes straight from raw EDFs to the dataloader with no HDF5
cache — there is no preprocessor module for it.  The alias map and
500 → 256 Hz resample live inline in `dataio/aub_med.py`.*

### EPILEPSIAE per-patient unit mismatch

Epilepsiae is multi-center and a small number of patients' `.head` files export a wrong `conversion_factor`, so the output of `signals * conversion_factor` lands in V or mV instead of µV. Downstream `assert_amplitude_band` in the model wrappers (BENDR/LaBraM/CBraMod/EEGPT) then raises. Set `EPILEPSIAE_AUTO_RESCALE=1` on the Condor job to rescale detected mis-scaled blocks (×1e6 for V, ×1e3 for mV) with a single warning per block; leave it unset for strict behaviour (the default). See `epilepsiae/readers.py`.

## Related: abnormal-classification datasets

The TUAB integration under this folder shares the abnormal-classification
pattern with the NMT dataset.  NMT is not epilepsy-specific, but it
lives under the same dataio/adapter family.  See
`src/extensions/datasets/dataio/nmt.py` for the lazy-EDF
loader (200 → 256 Hz resample on load, no HDF5 cache).

## Directory layout

```
epilepsy/
├── README.md                       # (this file)
├── __init__.py                     # re-exports the public API
├── _common.py                      # shared constants + signal helpers (see below)
├── bids_index.py                   # BIDS scanning + events.tsv parsing
│
├── chbmit_preprocessor.py          # CHB-MIT BIDS → HDF5 (flat, ~360 loc)
├── siena_preprocessor.py           # Siena BIDS  → HDF5 (flat, ~350 loc)
│
├── epilepsiae_preprocessor.py      # thin shim → epilepsiae/  sub-package
├── epilepsiae/                     # see epilepsiae/README.md
│   ├── readers.py                  #   .head/.data binary reader + channel norm
│   ├── annotations.py              #   CSV + SQL metadata parsers
│   ├── labels.py                   #   samplewise label builder
│   └── preprocessor.py             #   EpilepsiAEPreprocessor + merge_shards
│
├── tusz_preprocessor.py            # thin shim → tusz/  sub-package
└── tusz/                           # see tusz/README.md
    ├── readers.py                  #   EDF → 19-unipolar + channel norm
    ├── annotations.py              #   .csv / .csv_bi parsers + label builders
    └── preprocessor.py             #   build_cache + merge_shards + merge_splits
```

The two **shim** files (`epilepsiae_preprocessor.py`, `tusz_preprocessor.py`) only re-export their sub-packages.  They exist because downstream code (`dataio/`, `adapters/`, entrypoints, tests) imports dozens of symbols from the old flat path; keeping the shim means none of those imports change.

## Shared infrastructure

### `_common.py` — constants and signal helpers

Every preprocessor that outputs the unipolar schema shares these:

| Symbol                 | What                                                                           |
| ---------------------- | ------------------------------------------------------------------------------ |
| `CANONICAL_19`         | Tuple of 19 standard 10-20 electrode names                                     |
| `CANONICAL_SET`, `CANONICAL_IDX` | Frozenset / `{name: idx}` mapping for fast lookup                    |
| `BIPOLAR_MONTAGE`      | 18 bipolar pairs (TCP montage) as tuple of `(anode, cathode)` pairs            |
| `BIPOLAR_NAMES`        | 18 string labels `"FP1-F7"`, `"F7-T3"`, ...                                    |
| `TARGET_FS` = `256`    | Target sampling rate                                                           |
| `CACHE_SCHEMA_TAG`     | `"256hz_continuous_unipolar19"` — stored in every cache's root attrs          |
| `GAP_LABEL` = `255`    | Sentinel samplewise label for inter-recording gaps (EPILEPSIAE only)           |
| `bipolar_from_unipolar(w)` | `(..., 19, T) → (..., 18, T)` via pairwise subtraction                     |
| `resample_to(sig, src, tgt)`   | Polyphase resample (`scipy.signal.resample_poly`)                      |
| `apply_standard_filters(sig, fs, notch_hz=50.0, axis=1)` | 0.5 Hz Butterworth HP + `notch_hz` IIR notch (zero-phase); pass `notch_hz=60.0` for US mains (TUAB, TUSZ, CHB-MIT) and `axis=0` for time-first `(T, C)` arrays |

> **Note on `bipolar_from_unipolar` axis convention.**  The shared copy in `_common.py` uses the **channel-first** convention `(..., 19, T) → (..., 18, T)`.  The TUSZ dataio layer requires a **time-first** version `(T, 19) → (T, 18)`; that copy lives in `tusz/readers.py` with the same name.  The two are deliberate duplicates — forcing a single API would require a transpose on every call in the hot path.

### `bids_index.py` — BIDS discovery

Parses only JSON sidecars and `_events.tsv` files — no EDF reads — to build a lightweight recording index used by the CHB-MIT and Siena direct-BIDS loaders.  Exports `find_bids_root()` and `parse_bids_events()`, both also used during preprocessing of those two datasets.

## Cache schema (`CACHE_SCHEMA_TAG="256hz_continuous_unipolar19"`)

Each preprocessor writes an HDF5 file with this shape.  All four caches are interchangeable from the dataio layer's perspective (except CHB-MIT, which stores 18 bipolar channels instead of 19 unipolar and is read by a dedicated dataio).

### Root attributes

| Attribute        | Type         | Value                                                              |
| ---------------- | ------------ | ------------------------------------------------------------------ |
| `fs`             | int          | `256`                                                              |
| `schema`         | str          | `"continuous"`                                                     |
| `schema_tag`     | str          | `"256hz_continuous_unipolar19"`                                    |
| `channels`       | list[str]    | 19 canonical names (or 18 bipolar for CHB-MIT)                     |
| `bipolar_montage`| list[str]    | `["FP1-F7", "F7-T3", ...]`                                         |
| `corpus`         | str          | `"epilepsiae"`, `"TUSZ"`, ...                                      |
| `n_subjects`     | int          |                                                                    |
| `n_recordings`   | int          |                                                                    |
| `total_samples`  | int          |                                                                    |
| `pos_fraction`   | float        | Fraction of samples labelled as seizure                            |

TUSZ and EPILEPSIAE also add `seizure_type_names`, `pattern_names`, and dataset-specific counts.

### Required datasets

| Dataset             | Shape             | Dtype   | Meaning                                                                    |
| ------------------- | ----------------- | ------- | -------------------------------------------------------------------------- |
| `signals`           | `(T, 19)` (or 18) | float32 | All recordings concatenated along time                                     |
| `samplewise_label`  | `(T,)`            | uint8   | `0` = background, `1` = seizure, `255` = gap (EPILEPSIAE only)             |
| `samplewise_type`   | `(T,)`            | uint8   | Multi-class seizure type code (0 = bckg; see each dataset for the taxonomy) |
| `recording_offsets` | `(n_rec + 1,)`    | int64   | `offsets[i]:offsets[i+1]` is the sample span for recording `i`            |
| `recording_ids`     | `(n_rec,)`        | vlen str|                                                                            |
| `subject_ids`       | `(n_rec,)`        | vlen str|                                                                            |
| `durations_s`       | `(n_rec,)`        | float32 |                                                                            |

### Event datasets (per-seizure, flat)

Built from the original annotations so downstream code can query events without scanning `samplewise_label`:

| Dataset                | Shape     | Dtype | Meaning                                               |
| ---------------------- | --------- | ----- | ----------------------------------------------------- |
| `event_start_s`        | `(n_ev,)` | float32 | Onset in seconds relative to the recording start     |
| `event_stop_s`         | `(n_ev,)` | float32 | Offset in seconds                                    |
| `event_type`           | `(n_ev,)` | uint8  | Dataset-specific seizure type code                   |
| `event_recording_idx`  | `(n_ev,)` | int32  | Index into `recording_ids`                            |

EPILEPSIAE adds `event_pattern`, `event_classification`, `event_vigilance`, `event_onset_electrode`.  TUSZ adds `event_channel_tcp`.

### Per-recording metadata (dataset-specific)

Each preprocessor adds what it has: `ages`, `genders` / `sexes`, `hospitals`, `variant` (EPILEPSIAE), `session_ids` and `epilepsy_diagnosis` (TUSZ), etc.  Always optional from the dataio side — see `dataio/{dataset}.py` for the fields used.

## Building caches

Each preprocessor has a CLI entrypoint under `entrypoints`:

```bash
# EPILEPSIAE — one shard, one patient
python -m extensions.datasets.preprocessors.preprocess_epilepsiae \
    --data-root ${EEG_DATA_ROOT}/Epillepsie \
    --cache-root /anonorg/.../caches/epilepsiae \
    --shard 0 --num-shards 32

# EPILEPSIAE — merge shards into the final cache
python -m extensions.datasets.preprocessors.preprocess_epilepsiae \
    --cache-root /anonorg/.../caches/epilepsiae \
    --merge

# TUSZ — per-split, sharded
python -m extensions.datasets.preprocessors.preprocess_tusz \
    --raw-root /anonorg/.../TUSZ/v2.0.3/edf --cache-root /anonorg/.../caches/tusz \
    --split train --shard-index 0 --num-shards 16

# TUSZ — merge shards and then merge splits for k-fold
python -m entrypoints.merge_tusz_shards \
    --cache-root /anonorg/.../caches/tusz --splits train dev eval

# CHB-MIT / Siena — single-process, downloads from Zenodo on demand
python -m extensions.datasets.epilepsy.chbmit_preprocessor \
    --raw-dir /anonorg/.../bids_chbmit --output /anonorg/.../caches/chbmit.h5 --download
```

For Condor fan-out the EPILEPSIAE and TUSZ preprocessors both support `--shard-index N --num-shards K`; the corresponding merge command stitches the shards back together.

## Adding a new dataset

1. Reuse the schema — write `(T, 19)` float32 `signals` at 256 Hz, `samplewise_label`, `recording_offsets`, and the required per-recording arrays.
2. Import canonical channel constants and `bipolar_from_unipolar` from `_common` rather than redefining.
3. If the source is BIDS, reuse `bids_index.find_bids_root` and `bids_index.parse_bids_events`.
4. Add a dataio in `extensions/datasets/dataio/<name>.py` (see `tusz.py` / `epilepsiae.py` for the pattern).
5. Add an adapter in `extensions/datasets/adapters/<name>.py` exposing a `BenchmarkDataModule`.
6. Register the new dataset in `extensions/datasets/<name>.py` (`DATASET_SPECS`).

## Review / tests

- The primary test files are the internal test suite, the internal test suite, the internal test suite.
- the internal test suite and the internal test suite verify that all dataset specs registered here obey the `extensions/__init__.py` contracts (schema tag, expected label modes, etc.).
- `python -m entrypoints.validate_repo` does a full import / registration health check.
