"""Generic MOABB-MI preprocessor.

Takes ``--slug <name>`` and dispatches to ``DATASET_CONFIGS[slug]`` via
``load_and_preprocess()``. Supports the 7 broad-sweep MI datasets
(schirrmeister2017, shin2017a, weibo2014, bnci2014_001, dreyer2023a,
bnci2014_004, bnci2015_001) and any variant suffix (default / ``_bendr`` /
``_labram``).

Output file lands under the first repo-local path in
``PREPROCESSED_SEARCH_PATHS[slug]`` (see ``dataio/bci.py``), which
``load_preprocessed_dataset`` and ``probe_bci_loso`` already know about.

Usage:
    PYTHONPATH=src python -m neuroatlas.extensions.datasets.preprocessors.preprocess_moabb_mi \\
        --slug bnci2014_001 --n-jobs 8

    # Variant (BENDR-aligned wideband):
    PYTHONPATH=src python -m neuroatlas.extensions.datasets.preprocessors.preprocess_moabb_mi \\
        --slug bnci2014_001_bendr --n-jobs 8
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from neuroatlas.extensions.datasets.dataio.bci import (
    DATASET_CONFIGS,
    PREPROCESSED_SEARCH_PATHS,
    load_and_preprocess,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--slug",
        required=True,
        help="Dataset slug in DATASET_CONFIGS (e.g. bnci2014_001, "
             "bnci2014_001_bendr, bnci2014_001_labram).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pkl path (default: first path in PREPROCESSED_SEARCH_PATHS[slug]).",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--per-subject", action="store_true",
                        help="Process one subject at a time to reduce peak RAM. "
                             "Required for large datasets (Cho2017, Lee2019_MI) "
                             "with wideband resampling.")
    args = parser.parse_args(argv)

    if args.slug not in DATASET_CONFIGS:
        raise SystemExit(f"Unknown slug {args.slug!r}. Known: {sorted(DATASET_CONFIGS)}")

    cfg = DATASET_CONFIGS[args.slug]

    if args.output:
        output_file = Path(args.output)
    else:
        paths = PREPROCESSED_SEARCH_PATHS.get(args.slug)
        if not paths:
            raise SystemExit(f"No preprocessed path registered for {args.slug!r}.")
        output_file = Path(paths[0])

    if output_file.exists() and not args.force:
        print(f"Output already exists: {output_file}\nUse --force to overwrite.")
        return 0

    print(f"Slug: {args.slug}")
    print(f"  moabb_name:     {cfg.moabb_name}")
    print(f"  subjects:       {len(cfg.subjects)}")
    print(f"  channels:       {len(cfg.channels)}")
    print(f"  fmin={cfg.fmin} fmax={cfg.fmax} notch={cfg.notch_freq} "
          f"sfreq={cfg.resample_sfreq} car={cfg.use_car}")
    print(f"  tmin={cfg.tmin} tmax={cfg.tmax}")
    print(f"  targets:        {list(cfg.targets)}")
    print(f"  output:         {output_file}")
    print(f"  n_jobs={args.n_jobs}\n")

    t0 = time.time()

    if args.per_subject:
        # Process one subject at a time to keep peak RAM bounded.
        # Each subject's braindecode dataset is built, trials extracted,
        # then discarded before the next subject starts.
        print(f"  [per-subject mode] Processing {len(cfg.subjects)} subjects one at a time\n")
        subject_data: dict = {}
        for si, sid in enumerate(cfg.subjects):
            try:
                ws = load_and_preprocess(cfg, subject_ids=[sid], n_jobs=args.n_jobs)
            except Exception as exc:
                print(f"  skip subject {sid}: {type(exc).__name__}: {exc}")
                continue
            trials, labels = [], []
            for ds in ws.datasets:
                for j in range(len(ds)):
                    x, y, _ = ds[j]
                    trials.append(np.asarray(x, dtype=np.float32))
                    labels.append(int(y))
            if trials:
                subject_data[sid] = {"trials": trials, "labels": labels}
            print(f"  [{si+1}/{len(cfg.subjects)}] subject {sid}: {len(trials)} trials")
            del ws, trials, labels
        print(f"\nPreprocessing done in {time.time() - t0:.1f}s")
    else:
        windows_dataset = load_and_preprocess(cfg, n_jobs=args.n_jobs)
        print(f"Preprocessing done in {time.time() - t0:.1f}s")
        print(f"Total recording-level datasets: {len(windows_dataset.datasets)}")

        subject_data = {}
        for ds in windows_dataset.datasets:
            subj = ds.description["subject"]
            if subj not in subject_data:
                subject_data[subj] = {"trials": [], "labels": []}
            for j in range(len(ds)):
                x, y, _ = ds[j]
                subject_data[subj]["trials"].append(np.asarray(x, dtype=np.float32))
                subject_data[subj]["labels"].append(int(y))

    sorted_subjects = sorted(subject_data.keys())
    n_subjects = len(sorted_subjects)
    data_raw = np.empty(n_subjects, dtype=object)
    condition = np.empty(n_subjects, dtype=object)
    subject_name = np.array(sorted_subjects, dtype=int)
    for i, subj in enumerate(sorted_subjects):
        data_raw[i] = np.stack(subject_data[subj]["trials"], axis=0)
        condition[i] = np.array(subject_data[subj]["labels"], dtype=int)

    print(f"  {n_subjects} subjects")
    counts = [len(condition[i]) for i in range(n_subjects)]
    print(f"  trials per subject: min={min(counts)} max={max(counts)} mean={np.mean(counts):.1f}")
    if n_subjects > 0:
        print(f"  trial shape: {data_raw[0].shape} (n_trials, n_channels, n_timesamples)")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as fh:
        pickle.dump(
            {"data_raw": data_raw, "condition": condition, "subject_name": subject_name},
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {output_file} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
