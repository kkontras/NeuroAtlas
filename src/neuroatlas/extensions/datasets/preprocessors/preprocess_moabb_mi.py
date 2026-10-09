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
import logging
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

logger = logging.getLogger(__name__)


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
        raise SystemExit(f"error: no motor-imagery build for {args.slug!r}; the builds: "
                         f"{', '.join(sorted(DATASET_CONFIGS))}")

    cfg = DATASET_CONFIGS[args.slug]

    if args.output:
        output_file = Path(args.output)
    else:
        paths = PREPROCESSED_SEARCH_PATHS.get(args.slug)
        if not paths:
            raise SystemExit(f"error: {args.slug}: no place to write its prepared file\n"
                             f"fix: neuroatlas data prepare {args.slug} --dest FILE")
        output_file = Path(paths[0])

    if output_file.exists() and not args.force:
        logger.info("%s: already built at %s (--force builds it again)",
                    args.slug, output_file)
        return 0

    # what is built, for -v and --log (`data prepare` shows its own lines)
    logger.info("%s: MOABB dataset %s, %d subjects, %d channels", args.slug,
                cfg.moabb_name, len(cfg.subjects), len(cfg.channels))
    logger.info("%s: band %s-%s Hz, notch %s Hz, resampled to %s Hz, common average "
                "reference %s, trial %s to %s s, classes %s", args.slug, cfg.fmin, cfg.fmax,
                cfg.notch_freq, cfg.resample_sfreq, "on" if cfg.use_car else "off",
                cfg.tmin, cfg.tmax, ", ".join(map(str, cfg.targets)))
    logger.info("%s: writing %s (%d jobs)", args.slug, output_file, args.n_jobs)

    t0 = time.time()

    if args.per_subject:
        # Process one subject at a time to keep peak RAM bounded.
        # Each subject's braindecode dataset is built, trials extracted,
        # then discarded before the next subject starts.
        logger.info("%s: one subject at a time (%d subjects)", args.slug, len(cfg.subjects))
        subject_data: dict = {}
        for si, sid in enumerate(cfg.subjects):
            try:
                ws = load_and_preprocess(cfg, subject_ids=[sid], n_jobs=args.n_jobs)
            except Exception as exc:
                from neuroatlas.cli._msg import exception_text

                logger.warning("%s: subject %s left out of the prepared file: %s",
                               args.slug, sid, exception_text(exc))
                continue
            trials, labels = [], []
            for ds in ws.datasets:
                for j in range(len(ds)):
                    x, y, _ = ds[j]
                    trials.append(np.asarray(x, dtype=np.float32))
                    labels.append(int(y))
            if trials:
                subject_data[sid] = {"trials": trials, "labels": labels}
            logger.info("%s: [%d/%d] subject %s: %d trials", args.slug, si + 1,
                        len(cfg.subjects), sid, len(trials))
            del ws, trials, labels
        logger.info("%s: subjects read and filtered in %.1fs", args.slug, time.time() - t0)
    else:
        windows_dataset = load_and_preprocess(cfg, n_jobs=args.n_jobs)
        logger.info("%s: %d recordings read and filtered in %.1fs", args.slug,
                    len(windows_dataset.datasets), time.time() - t0)

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

    counts = [len(condition[i]) for i in range(n_subjects)]
    logger.info("%s: %d subjects, %d to %d trials each (mean %.1f)", args.slug, n_subjects,
                min(counts), max(counts), np.mean(counts))
    if n_subjects > 0:
        logger.info("%s: trials of %d channels x %d samples", args.slug,
                    data_raw[0].shape[1], data_raw[0].shape[2])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as fh:
        pickle.dump(
            {"data_raw": data_raw, "condition": condition, "subject_name": subject_name},
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info("%s: wrote %s (%.1f MB)", args.slug, output_file, size_mb)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main() or 0)
