"""Adapter that exposes an *already materialised* external embedding cache.

Use case: a collaborator has run a linear-probe sleep-stage benchmark and
cached foundation-model embeddings to disk with the same layout this repo
writes (``features.npy + labels.npy + items.json + metadata.json``). Each
row of ``items.json`` carries a per-subject ``age`` float, so we can reuse
those embeddings to run the ``brain_age`` regression task without invoking
any backbone — we just swap the cached label for ``age`` and build our own
stratified-by-age subject-level folds at split time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from neuroatlas.benchmarking_helpers.registry.contracts import EmbeddingPayload
from .base import BenchmarkDataModule


class _EmptyLoader:
    """No-op iterable stand-in for full_embedding_dataloader().

    ``brain_age.py`` and ``linear_probe.py`` evaluate
    ``datamodule.full_embedding_dataloader()`` eagerly as a positional
    argument to ``_extract_or_load_embeddings``. The hook path skips
    iteration entirely — this stub exists so construction doesn't crash.
    """

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[Any]:
        return iter(())


def _stratified_subject_splits(
    subjects: Sequence[str],
    ages_by_subject: Dict[str, float],
    fold: int,
    n_folds: int,
    seed: int,
    age_bins: Sequence[float],
) -> Tuple[List[str], List[str], List[str]]:
    """Mirror of sleepedf_raw_brain_age._subject_splits with parameterised bins.

    Outer StratifiedKFold (by age bin) picks the test subjects; an inner
    StratifiedKFold on the remaining subjects picks val. Deterministic for
    fixed (seed, age_bins, subjects).
    """
    from neuroatlas.benchmarking_helpers.registry.splits import (
        make_subject_kfold,
    )

    subjects_arr = list(subjects)
    subject_ages = np.array([ages_by_subject[s] for s in subjects_arr], dtype=float)
    bins = np.asarray(age_bins, dtype=float)
    age_bin_ids = np.digitize(subject_ages, bins) - 1

    # Shared rule: benchmarking_helpers/splits. Age bins are a harsher
    # stratification than a binary label -- one sparse decade can hold
    # fewer subjects than folds -- so passing n_folds straight to
    # StratifiedKFold raised on exactly the cohorts it mattered for.
    outer, _stratified, n_folds = make_subject_kfold(
        n_folds, age_bin_ids, seed=seed)
    outer_splits = list(outer.split(subjects_arr, age_bin_ids))

    train_val_idx, test_idx = outer_splits[fold % n_folds]
    test_subjects = [subjects_arr[i] for i in test_idx]

    train_val_subjects = [subjects_arr[i] for i in train_val_idx]
    train_val_bins = age_bin_ids[train_val_idx]
    inner, _inner_strat, inner_folds = make_subject_kfold(
        n_folds, train_val_bins, seed=seed)
    val_fold = (fold + 1) % inner_folds
    inner_splits = list(inner.split(train_val_subjects, train_val_bins))
    inner_train_idx, inner_val_idx = inner_splits[val_fold % len(inner_splits)]
    train_subjects = [train_val_subjects[i] for i in inner_train_idx]
    val_subjects = [train_val_subjects[i] for i in inner_val_idx]
    return train_subjects, val_subjects, test_subjects


def _subjects_with_mean_ages_from_metadata(
    metadata_list: Sequence[Dict[str, Any]],
    label_field: str,
) -> Tuple[List[str], Dict[str, float]]:
    """Extract unique subjects with their mean label value across all rows.

    For longitudinal datasets (WSC) a subject has multiple visits at different
    ages — per-row ages vary within a subject and that is expected. The mean
    is used only to pick a stratification bin; row-level ages remain the
    regression labels.
    """
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in metadata_list:
        subj = str(row["subject_id"])
        age = float(row[label_field])
        totals[subj] = totals.get(subj, 0.0) + age
        counts[subj] = counts.get(subj, 0) + 1
    ages_by_subject = {s: totals[s] / counts[s] for s in totals}
    subjects = sorted(ages_by_subject)
    return subjects, ages_by_subject


def _discover_hash_subdir(
    checkpoint_root: Path,
    override: Optional[str] = None,
) -> Optional[Path]:
    """Return the single complete cache subdir under ``<checkpoint_root>/all/``.

    Returns None if no complete subdir exists (caller treats as "cache not
    ready yet"). Raises if multiple complete subdirs exist without an
    ``override`` to disambiguate.
    """
    all_dir = checkpoint_root / "all"
    if not all_dir.is_dir():
        return None

    if override is not None:
        candidate = all_dir / override
        if (candidate / "features.npy").exists() and (candidate / "labels.npy").exists():
            return candidate
        return None

    complete = [
        p for p in all_dir.iterdir()
        if p.is_dir()
        and (p / "features.npy").exists()
        and (p / "labels.npy").exists()
    ]
    if not complete:
        return None
    if len(complete) > 1:
        names = ", ".join(sorted(p.name for p in complete))
        raise RuntimeError(
            f"Ambiguous cache: multiple complete subdirs under {all_dir}: {names}. "
            f"Set checkpoint_resolver={{'<checkpoint_id>': '<hash>'}} to disambiguate."
        )
    return complete[0]


class PrecomputedEmbeddingDataModule(BenchmarkDataModule):
    """BenchmarkDataModule that loads an external embedding cache verbatim.

    The cache must follow the ``features.npy + labels.npy + items.json``
    layout written by ``neuroatlas.benchmarking_helpers.runtime.cache``. Each
    row in ``items.json`` must carry ``subject_id`` and a configurable
    ``label_field`` (default ``"age"``) that is constant within a subject.

    The task's ``_extract_or_load_embeddings`` short-circuit calls
    ``precomputed_embedding_cache_dir`` before any hash-based lookup, so
    this datamodule bypasses the backbone entirely. Sleep-stage labels
    baked into ``labels.npy`` are discarded — we rewrite labels from
    ``items.json[<label_field>]`` inside ``split_global_embedding_payload``.
    """

    def __init__(
        self,
        *,
        name: str,
        dataset_slug_in_cache: str,
        external_cache_root: str,
        fold: int = 0,
        n_folds: int = 5,
        drop_unscored: bool = False,
        label_field: str = "age",
        split_seed: int = 42,
        age_bins: Sequence[float] = (0, 35, 50, 65, 80, 200),
        aggregation_group: str = "recording_id",
        cv_filter: Optional[Dict[str, Any]] = None,
        train_filter: Optional[Dict[str, Any]] = None,
        label_lookup_table: Optional[str] = None,
        label_lookup_join: Sequence[str] = ("subject_id",),
        label_lookup_values: Optional[Dict[Any, Any]] = None,
        label_lookup_origin: Optional[str] = None,
        holdout_eval_groups: Optional[Sequence[Dict[str, Any]]] = None,
        holdout_eval_size_per_group: Optional[int] = None,
        holdout_eval_match_field: str = "age",
        holdout_eval_match_tolerance: Optional[float] = None,
        holdout_eval_seed: int = 1729,
        checkpoint_resolver: Optional[Dict[str, str]] = None,
        checkpoint: Any = None,
        **_: Any,
    ) -> None:
        self._external_cache_root = Path(external_cache_root)
        self._dataset_slug_in_cache = str(dataset_slug_in_cache)
        self._fold = int(fold)
        self._n_folds = int(n_folds)
        self._drop_unscored = bool(drop_unscored)
        self._label_field = str(label_field)
        self._split_seed = int(split_seed)
        self._age_bins = list(age_bins)
        self._aggregation_group = str(aggregation_group)
        # ``cv_filter`` restricts the whole CV pool (e.g. cassette-only for
        # Sleep-EDF SC); ``train_filter`` restricts only the training split.
        self._cv_filter = self._normalize_filter(cv_filter)
        self._train_filter = self._normalize_filter(train_filter)
        # Labels can be recovered from an external table rather than trusted
        # from the cache — the sleep-staging cache this reads was written for a
        # classification task, so its ``age`` values are incidental.
        self._label_lookup_table = str(label_lookup_table) if label_lookup_table else None
        self._label_lookup_join: Tuple[str, ...] = tuple(label_lookup_join)
        self._label_lookup: Optional[Dict[Tuple[Any, ...], Any]] = None
        self._label_lookup_source: Optional[str] = None
        if label_lookup_values is not None:
            # Built in memory by a dataset spec from the dataset's own metadata
            # (e.g. ISRUC's Details_*.xlsx, Sleep-EDF's SC-subjects.xls) via the
            # loaders main already ships, so no label file has to travel with the
            # repo. Same key convention as a table: stringified join values.
            self._label_lookup = {
                (tuple(str(x) for x in k) if isinstance(k, tuple) else (str(k),)): v
                for k, v in dict(label_lookup_values).items()
                if v is not None
            }
            origin = f" from {label_lookup_origin}" if label_lookup_origin else ""
            self._label_lookup_source = f"in-memory{origin} ({len(self._label_lookup)} keys)"
        elif self._label_lookup_table:
            self._label_lookup = self._load_label_lookup()
            self._label_lookup_source = f"table:{self._label_lookup_table}"

        # Age-matched holdout cohort, reserved before the CV split so it never
        # enters any train/val/test fold and is identical across outer folds.
        # Used for the brain-age-gap study: train on one group (via cv_filter),
        # then compare the brain-age gap between two matched groups.
        self._holdout_eval_groups: Optional[List[Dict[str, List[Any]]]] = None
        if holdout_eval_groups:
            grps = [self._normalize_filter(g) for g in holdout_eval_groups]
            if len(grps) != 2:
                raise ValueError(
                    "holdout_eval_groups supports exactly 2 groups to age-match; "
                    f"got {len(grps)}."
                )
            self._holdout_eval_groups = grps
        self._holdout_eval_size_per_group = (
            int(holdout_eval_size_per_group) if holdout_eval_size_per_group else None
        )
        if self._holdout_eval_groups and not self._holdout_eval_size_per_group:
            raise ValueError(
                "holdout_eval_size_per_group must be set when holdout_eval_groups is."
            )
        self._holdout_eval_match_field = str(holdout_eval_match_field)
        self._holdout_eval_match_tolerance = (
            float(holdout_eval_match_tolerance)
            if holdout_eval_match_tolerance is not None else None
        )
        self._holdout_eval_seed = int(holdout_eval_seed)
        self._checkpoint_resolver = dict(checkpoint_resolver) if checkpoint_resolver else {}

        meta: Dict[str, Any] = {
            "fold": self._fold,
            "n_folds": self._n_folds,
            "dataset_slug_in_cache": self._dataset_slug_in_cache,
            "external_cache_root": str(self._external_cache_root),
            "drop_unscored": self._drop_unscored,
            "label_field": self._label_field,
            "split_seed": self._split_seed,
            "age_bins": list(self._age_bins),
            "aggregation_group": self._aggregation_group,
            "cv_filter": self._cv_filter,
            "train_filter": self._train_filter,
            "label_lookup_table": self._label_lookup_table,
            "label_lookup_join": (
                list(self._label_lookup_join) if self._label_lookup is not None else None
            ),
            "label_lookup_source": self._label_lookup_source,
            "holdout_eval_groups": self._holdout_eval_groups,
            "holdout_eval_size_per_group": self._holdout_eval_size_per_group,
            "holdout_eval_match_field": self._holdout_eval_match_field,
            "holdout_eval_match_tolerance": self._holdout_eval_match_tolerance,
            "holdout_eval_seed": self._holdout_eval_seed,
        }
        super().__init__(name=str(name), metadata=meta)

    # -- filters and label lookup -------------------------------------------
    #
    # Ported from the pre-merge EEGBenchmarks adapter. Both knobs are required
    # to reproduce the published brain-age protocol: ``cv_filter`` selects the
    # cassette-only SC cohort out of the full staging cache, and the label
    # lookup supplies ages from PhysioBank's own table rather than whatever the
    # sleep-staging extractor happened to record.

    @staticmethod
    def _normalize_filter(
        filt: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, List[Any]]]:
        """Normalize each filter entry to a list of allowed values."""
        if not filt:
            return None
        normalized: Dict[str, List[Any]] = {}
        for field, value in dict(filt).items():
            if isinstance(value, (list, tuple, set)):
                normalized[str(field)] = list(value)
            else:
                normalized[str(field)] = [value]
        return normalized

    @staticmethod
    def _row_matches(row: Dict[str, Any], filt: Optional[Dict[str, List[Any]]]) -> bool:
        if not filt:
            return True
        return all(row.get(field) in allowed for field, allowed in filt.items())

    def _load_label_lookup(self) -> Dict[Tuple[Any, ...], Any]:
        """Read a CSV/XLS lookup table and return ``dict[(join_keys,) -> label]``.

        Join keys are stringified on both sides. Without that, pandas' auto-typing
        turns an integer id into ``10119.0`` while ``items.json`` holds
        ``"10119"``, and the join silently loses every row.
        """
        path = Path(self._label_lookup_table)
        if not path.exists():
            raise FileNotFoundError(f"label_lookup_table not found: {path}")
        try:
            import pandas as pd  # local import — only when this knob is set
        except ImportError as e:
            raise ImportError("label_lookup_table requires pandas.") from e
        df = (
            pd.read_excel(path)
            if path.suffix.lower() in (".xls", ".xlsx")
            else pd.read_csv(path)
        )
        for c in self._label_lookup_join:
            if c not in df.columns:
                continue
            if pd.api.types.is_integer_dtype(df[c]):
                df[c] = df[c].astype(str)
            elif pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].apply(
                    lambda v: str(int(v)) if float(v).is_integer() else str(v)
                )
            else:
                df[c] = df[c].astype(str)
        missing = [c for c in self._label_lookup_join if c not in df.columns]
        if missing:
            raise ValueError(
                f"label_lookup_table {path} is missing join columns {missing!r}; "
                f"have {list(df.columns)!r}"
            )
        if self._label_field not in df.columns:
            raise ValueError(
                f"label_lookup_table {path} is missing label column "
                f"{self._label_field!r}; have {list(df.columns)!r}"
            )
        lookup: Dict[Tuple[Any, ...], Any] = {}
        for _, row in df.iterrows():
            lookup[tuple(str(row[c]) for c in self._label_lookup_join)] = row[
                self._label_field
            ]
        print(
            f"[precomputed] Loaded label_lookup_table {path.name}: {len(lookup)} "
            f"entries; join={list(self._label_lookup_join)}, "
            f"label_field={self._label_field!r}",
            flush=True,
        )
        return lookup

    def _apply_label_lookup(self, metadata_list: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Override ``items[i][label_field]`` from the lookup where keys match.

        Returns ``(n_overridden, n_unmatched)``. Unmatched rows keep whatever the
        cache held, and are dropped later if that value is missing.
        """
        if not self._label_lookup:
            return 0, 0
        n_overridden = n_unmatched = 0
        for item in metadata_list:
            raw_key = tuple(item.get(c) for c in self._label_lookup_join)
            if any(k is None for k in raw_key):
                n_unmatched += 1
                continue
            value = self._label_lookup.get(tuple(str(k) for k in raw_key))
            if value is None:
                n_unmatched += 1
                continue
            item[self._label_field] = value
            n_overridden += 1
        print(
            f"[precomputed] label_lookup applied: overrode {n_overridden}/"
            f"{len(metadata_list)} item rows; {n_unmatched} unmatched "
            f"(kept original {self._label_field}).",
            flush=True,
        )
        return n_overridden, n_unmatched

    def _select_holdout_subjects(
        self, metadata_list: Sequence[Dict[str, Any]]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Pick K age-matched pairs across the two holdout groups.

        Returns ``(holdout_subject_ids, group_label_per_subject)``, where each
        label is a stable string such as ``"ci_label=0"`` so the evaluator can
        split predictions by group.

        Greedy nearest-neighbour matching on ``holdout_eval_match_field``:
        every (group-A, group-B) subject pair is ranked by the absolute
        difference in their mean match value, and pairs are accepted in that
        order as long as neither subject is already used (and the difference is
        within ``holdout_eval_match_tolerance``, if set) until K are chosen.
        Equal differences are ordered by a draw from ``holdout_eval_seed``, so
        the cohort is deterministic and does not depend on the fold. Same
        algorithm as the pre-merge EEGBenchmarks adapter.
        """
        if not (self._holdout_eval_groups and self._holdout_eval_size_per_group):
            return [], {}
        match_field = self._holdout_eval_match_field
        K = self._holdout_eval_size_per_group
        tol = self._holdout_eval_match_tolerance

        per_subj_match: Dict[str, List[float]] = {}
        per_subj_group: Dict[str, int] = {}
        for row in metadata_list:
            subj = str(row["subject_id"])
            v = row.get(match_field)
            if v is None:
                continue
            for gi, gfilt in enumerate(self._holdout_eval_groups):
                if self._row_matches(row, gfilt):
                    if subj in per_subj_group and per_subj_group[subj] != gi:
                        per_subj_group[subj] = -1  # straddles both groups: ambiguous
                    else:
                        per_subj_group[subj] = gi
                    per_subj_match.setdefault(subj, []).append(float(v))
                    break

        a_subj = sorted(s for s, g in per_subj_group.items() if g == 0)
        b_subj = sorted(s for s, g in per_subj_group.items() if g == 1)
        a_val = {s: float(np.mean(per_subj_match[s])) for s in a_subj}
        b_val = {s: float(np.mean(per_subj_match[s])) for s in b_subj}

        rng = np.random.default_rng(self._holdout_eval_seed)
        candidates: List[Tuple[float, float, str, str]] = []
        for sa in a_subj:
            for sb in b_subj:
                d = abs(a_val[sa] - b_val[sb])
                if tol is not None and d > tol:
                    continue
                candidates.append((d, float(rng.random()), sa, sb))
        candidates.sort()

        used_a: set = set()
        used_b: set = set()
        chosen: List[Tuple[str, str, float]] = []
        for d, _jitter, sa, sb in candidates:
            if sa in used_a or sb in used_b:
                continue
            chosen.append((sa, sb, d))
            used_a.add(sa)
            used_b.add(sb)
            if len(chosen) == K:
                break
        if len(chosen) < K:
            raise ValueError(
                f"Could not find {K} matched pairs (got {len(chosen)}). Relax "
                f"holdout_eval_match_tolerance or lower holdout_eval_size_per_group. "
                f"Group sizes: a={len(a_subj)}, b={len(b_subj)}."
            )

        def _label(group: Dict[str, List[Any]]) -> str:
            key = next(iter(group))
            return f"{key}={group[key][0]}"

        a_label = _label(self._holdout_eval_groups[0])
        b_label = _label(self._holdout_eval_groups[1])
        holdout_ids: List[str] = []
        labels: Dict[str, str] = {}
        for sa, sb, _d in chosen:
            holdout_ids.append(sa)
            labels[sa] = a_label
            holdout_ids.append(sb)
            labels[sb] = b_label
        deltas = [d for _, _, d in chosen]
        print(
            f"[precomputed] Holdout-eval: {K} pairs reserved ({a_label} vs {b_label}); "
            f"max |d {match_field}|={max(deltas):.2f}, mean={float(np.mean(deltas)):.2f}.",
            flush=True,
        )
        return holdout_ids, labels

    def supports_global_embedding_cache(self) -> bool:
        return True

    def cache_context(self, purpose: str = "default") -> Dict[str, Any]:
        return {
            "fold": self._fold,
            "n_folds": self._n_folds,
            "dataset_slug_in_cache": self._dataset_slug_in_cache,
            "external_cache_root": str(self._external_cache_root),
            "drop_unscored": self._drop_unscored,
            "label_field": self._label_field,
            "split_seed": self._split_seed,
            "age_bins": list(self._age_bins),
            "aggregation_group": self._aggregation_group,
        }

    def full_embedding_dataloader(self) -> _EmptyLoader:
        return _EmptyLoader()

    def train_dataloader(self):
        raise RuntimeError(
            "PrecomputedEmbeddingDataModule only supports the global-embedding-cache path."
        )

    def val_dataloader(self):
        raise RuntimeError(
            "PrecomputedEmbeddingDataModule only supports the global-embedding-cache path."
        )

    def test_dataloader(self):
        raise RuntimeError(
            "PrecomputedEmbeddingDataModule only supports the global-embedding-cache path."
        )

    def precomputed_embedding_cache_dir(
        self,
        *,
        checkpoint_id: str,
        split: str,
        purpose: str = "default",
    ) -> Optional[Path]:
        checkpoint_root = (
            self._external_cache_root / self._dataset_slug_in_cache / checkpoint_id
        )
        resolved = _discover_hash_subdir(
            checkpoint_root,
            override=self._checkpoint_resolver.get(checkpoint_id),
        )
        if resolved is None:
            # Returning None is the documented contract — the runner's
            # ``_precomputed_cache_available`` uses this as a boolean probe, so
            # raising here would break its ability to fall back to a real
            # backbone. Log loudly instead; the hard guard lives in the task,
            # which is where an empty payload would otherwise surface as
            # "axis 1 is out of bounds for array of dimension 1".
            all_dir = checkpoint_root / "all"
            inflight = []
            if all_dir.is_dir():
                inflight = [
                    p.name for p in all_dir.iterdir()
                    if p.is_dir() and (
                        (p / "progress.json").exists() or any(p.glob("*.tmp"))
                    )
                ]
            suffix = (
                f" — {len(inflight)} extraction(s) still in flight "
                f"({', '.join(sorted(inflight)[:3])}); wait for the staging run "
                f"to finalize before probing"
                if inflight else ""
            )
            print(
                f"[precomputed] {self.name}: no finalized embedding cache for "
                f"checkpoint {checkpoint_id!r} under {all_dir}{suffix}.",
                flush=True,
            )
            return None
        self._record_source_provenance(resolved)
        return resolved

    def _record_source_provenance(self, cache_dir: Optional[Path]) -> None:
        """Surface the source run's provenance in this run's metadata.

        Probing a cache means the preprocessing that produced it is invisible in
        this run's own config — which is exactly the variable a preprocessing
        ablation is measuring. Lift ``filter_provenance`` (and the source
        checkpoint) out of the cache's ``metadata.json`` so each result row
        records which filtering produced its embeddings.
        """
        if cache_dir is None:
            return
        meta_path = Path(cache_dir) / "metadata.json"
        if not meta_path.exists():
            return
        try:
            src = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(src, dict):
            return
        for key in ("filter_provenance", "checkpoint_id", "embedding_key"):
            if key in src:
                self.metadata[f"source_{key}"] = src[key]
        self.metadata["source_cache_dir"] = str(cache_dir)

    def split_global_embedding_payload(self, payload) -> Dict[str, EmbeddingPayload]:
        """Split the global payload into train/val/test (+ an optional holdout).

        Order matches the pre-merge EEGBenchmarks adapter:

        1. drop rows with non-finite features;
        2. apply the label lookup, then drop rows whose label is still missing;
        3. reserve the age-matched holdout cohort, if configured, from the full
           labelled set -- before any filtering, since it needs both groups;
        4. assign folds over every remaining (non-holdout) subject;
        5. apply ``cv_filter`` to train/val/test and ``train_filter`` to
           train/val as row masks. They restrict what each split *contains*,
           not which subjects take part in fold assignment.

        The holdout split is never touched by ``cv_filter``/``train_filter``:
        the point is to score one trained model on a controlled cohort spanning
        both groups.
        """
        features = np.asarray(payload.features)
        metadata_list = list(payload.metadata)
        n_rows = len(metadata_list)
        if features.shape[0] != n_rows:
            raise ValueError(
                f"features ({features.shape[0]}) and metadata ({n_rows}) row counts disagree."
            )

        def _keep(mask: np.ndarray) -> None:
            nonlocal features, metadata_list, n_rows
            keep_idx = np.flatnonzero(mask)
            features = features[keep_idx]
            metadata_list = [metadata_list[i] for i in keep_idx]
            n_rows = len(metadata_list)

        finite_mask = np.isfinite(features).all(axis=1)
        if not finite_mask.all():
            print(
                f"[precomputed] Dropped {int((~finite_mask).sum())} / {n_rows} rows with "
                f"non-finite (NaN/inf) features before fold split.",
                flush=True,
            )
            _keep(finite_mask)

        # Labels first: the stratified split, the holdout matching and the
        # filters all need the final label values.
        self._apply_label_lookup(metadata_list)

        labelled_mask = np.array(
            [row.get(self._label_field) is not None for row in metadata_list], dtype=bool
        )
        if not labelled_mask.all():
            groups = sorted({
                str(row.get("subject_id"))
                for row, ok in zip(metadata_list, labelled_mask) if not ok
            })
            print(
                f"[precomputed] Dropped {int((~labelled_mask).sum())} / {n_rows} rows with "
                f"no {self._label_field} ({len(groups)} subjects): "
                f"{groups[:10]}{' ...' if len(groups) > 10 else ''}",
                flush=True,
            )
            _keep(labelled_mask)

        # Reserve the holdout cohort before the CV split so its subjects never
        # appear in any train/val/test fold.
        holdout_subject_ids, holdout_group_labels = self._select_holdout_subjects(
            metadata_list
        )
        holdout_set = set(holdout_subject_ids)
        in_cv_pool = np.array(
            [str(row.get("subject_id")) not in holdout_set for row in metadata_list],
            dtype=bool,
        )
        cv_metadata = [row for row, ok in zip(metadata_list, in_cv_pool) if ok]

        subjects, ages_by_subject = _subjects_with_mean_ages_from_metadata(
            cv_metadata, self._label_field
        )
        train_subs, val_subs, test_subs = _stratified_subject_splits(
            subjects,
            ages_by_subject,
            fold=self._fold,
            n_folds=self._n_folds,
            seed=self._split_seed,
            age_bins=self._age_bins,
        )
        train_set, val_set, test_set = set(train_subs), set(val_subs), set(test_subs)
        for label, overlap in (
            ("train/val", train_set & val_set),
            ("train/test", train_set & test_set),
            ("val/test", val_set & test_set),
        ):
            if overlap:
                raise RuntimeError(f"Subject overlap in {label} split: {sorted(overlap)!r}")

        subject_to_split: Dict[str, str] = {}
        for s in train_subs:
            subject_to_split[s] = "train"
        for s in val_subs:
            subject_to_split[s] = "val"
        for s in test_subs:
            subject_to_split[s] = "test"

        row_splits = np.empty(n_rows, dtype=object)
        new_labels = np.empty(n_rows, dtype=np.float64)
        stages = np.empty(n_rows, dtype=np.int64)
        for i, row in enumerate(metadata_list):
            row_splits[i] = subject_to_split.get(str(row["subject_id"]), "")
            new_labels[i] = float(row[self._label_field])
            stage = row.get("sleep_stage")
            stages[i] = int(stage) if stage is not None else 0

        cv_filter_mask: Optional[np.ndarray] = None
        if self._cv_filter:
            cv_filter_mask = np.array(
                [self._row_matches(row, self._cv_filter) for row in metadata_list],
                dtype=bool,
            )
            if not (cv_filter_mask & in_cv_pool).any():
                raise ValueError(
                    f"cv_filter {self._cv_filter!r} excluded every CV-pool row of "
                    f"{self._dataset_slug_in_cache}. Check the field names against "
                    f"items.json."
                )
        train_filter_mask: Optional[np.ndarray] = None
        if self._train_filter:
            train_filter_mask = np.array(
                [self._row_matches(row, self._train_filter) for row in metadata_list],
                dtype=bool,
            )

        def _payload(idx: np.ndarray, extra: Optional[Dict[str, str]] = None) -> EmbeddingPayload:
            split_meta: List[Dict[str, Any]] = []
            for i in idx:
                row = dict(metadata_list[i])
                if extra is not None:
                    row["holdout_group"] = extra.get(str(row.get("subject_id")), "")
                if self._aggregation_group and self._aggregation_group != "subject_id":
                    group_key = row.get(self._aggregation_group)
                    if group_key is None:
                        raise KeyError(
                            f"aggregation_group={self._aggregation_group!r} missing from "
                            f"metadata row {i}: {row!r}"
                        )
                    row["original_subject_id"] = row.get("subject_id")
                    row["subject_id"] = str(group_key)
                split_meta.append(row)
            return EmbeddingPayload(
                features=features[idx], labels=new_labels[idx], metadata=split_meta,
            )

        result: Dict[str, EmbeddingPayload] = {}
        for split_name in ("train", "val", "test"):
            mask = row_splits == split_name
            if cv_filter_mask is not None:
                pre = int(mask.sum())
                mask &= cv_filter_mask
                print(
                    f"[precomputed] cv_filter={self._cv_filter!r} on {split_name}: "
                    f"kept {int(mask.sum())}/{pre} rows.",
                    flush=True,
                )
            if train_filter_mask is not None and split_name in ("train", "val"):
                pre = int(mask.sum())
                mask &= train_filter_mask
                print(
                    f"[precomputed] train_filter={self._train_filter!r} on {split_name}: "
                    f"kept {int(mask.sum())}/{pre} rows.",
                    flush=True,
                )
            if self._drop_unscored:
                mask &= stages != -1
            result[split_name] = _payload(np.flatnonzero(mask))

        if holdout_subject_ids:
            holdout_idx = np.flatnonzero(~in_cv_pool)
            result["holdout"] = _payload(holdout_idx, extra=holdout_group_labels)
            print(
                f"[precomputed] Holdout payload: {len(holdout_idx)} rows from "
                f"{len(holdout_subject_ids)} subjects.",
                flush=True,
            )
        return result
