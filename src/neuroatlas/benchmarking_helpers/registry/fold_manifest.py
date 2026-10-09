"""Unified reader for the frozen fold manifests.

Looks in ``src/neuroatlas/configs/cohorts/<slug>/folds.json`` first (a migrated dataset dossier),
then ``src/neuroatlas/configs/folds/<slug>.json`` (the historical location).

The manifests are the source of truth for train/val/test subject assignments.
They exist in two on-disk shapes, both handled here:

**explicit** — ``folds[k] = {"train": [...], "val": [...], "test": [...]}``
    plus an optional ``stats`` block.  Used by the epilepsy-side manifests
    (``schema_version`` 1 or 2): chbmit, tusz, siena, helsinki, sz1, sz2, ...

**partition** — ``folds[k] = [subject_ids]``, where each entry is the *test*
    partition for fold ``k``.  Used by the unversioned sleep-side manifests:
    shhs_*, stages_*, mesa, isruc, ...  Explicit splits are derived
    with the rule documented in those files and implemented identically in
    ``entrypoints/probe_from_embeddings._load_fold_split``::

        test  = folds[k]
        val   = folds[(k + 1) % n_folds]
        train = the remaining folds

A partition manifest may additionally carry a materialized ``splits`` block.
No manifest shipped here does, but when present it is used as a cross-check and a
mismatch raises, so the derivation cannot silently drift from the frozen file.

This module exists because ``patient_splits.load_folds`` accepts only
``schema_version == 1`` *and* expects ``folds`` to be a list, so it currently
rejects or crashes on every manifest in ``src/neuroatlas/configs/folds/``.  Prefer this reader
for new code paths.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
from neuroatlas._paths import configs_dir

_DEFAULT_CONFIGS = configs_dir("folds")

SPLIT_RULE = "test=folds[k]; val=folds[(k+1) % n]; train=rest"


@dataclass(frozen=True)
class FoldSplit:
    """Explicit subject-level train/val/test assignment for a single fold."""

    dataset: str
    fold: int
    n_folds: int
    train: List[str]
    val: List[str]
    test: List[str]
    shape: str
    source_path: Path
    stats: Optional[Dict] = None

    @property
    def n_subjects(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def as_dict(self) -> Dict[str, List[str]]:
        return {"train": list(self.train), "val": list(self.val), "test": list(self.test)}

    def summary(self) -> str:
        return (
            f"{self.dataset} fold {self.fold}/{self.n_folds} [{self.shape}]: "
            f"{len(self.train)} train / {len(self.val)} val / {len(self.test)} test "
            f"subjects ({self.n_subjects} total)"
        )


def _folds_container(payload: Dict) -> Dict:
    """Return the per-fold mapping, tolerating the ``subject_folds`` alias."""
    key = "subject_folds" if "subject_folds" in payload else "folds"
    if key not in payload:
        raise ValueError(f"manifest has neither 'folds' nor 'subject_folds': keys={sorted(payload)}")
    folds = payload[key]
    if isinstance(folds, list):
        folds = {str(i): f for i, f in enumerate(folds)}
    if not isinstance(folds, dict):
        raise ValueError(f"unsupported 'folds' type: {type(folds).__name__}")
    return folds


def _assert_disjoint(dataset: str, fold: int, train: Sequence[str], val: Sequence[str],
                     test: Sequence[str]) -> None:
    tr, va, te = set(train), set(val), set(test)
    for a_name, a, b_name, b in (("train", tr, "val", va), ("train", tr, "test", te),
                                 ("val", va, "test", te)):
        overlap = a & b
        if overlap:
            raise ValueError(
                f"{dataset} fold {fold}: {a_name}/{b_name} overlap on {sorted(overlap)[:8]} "
                f"({len(overlap)} subjects) — manifest is leaky"
            )


def _derive_from_partition(folds: Dict, fold: int, n_folds: int) -> Dict[str, List[str]]:
    """Apply SPLIT_RULE. Mirrors probe_from_embeddings._load_fold_split exactly."""
    test_fold = fold % n_folds
    val_fold = (fold + 1) % n_folds
    train: List[str] = []
    for i in range(n_folds):
        if i not in (test_fold, val_fold):
            train.extend(folds[str(i)])
    return {"train": train, "val": list(folds[str(val_fold)]), "test": list(folds[str(test_fold)])}


def _manifest_candidates(dataset: str, configs_root=None) -> List[Path]:
    """Where a cohort's fold manifest may live, most specific first.

    An explicit *configs_root* bypasses the search entirely, which is what
    the tests use.
    """
    if configs_root is not None:
        return [Path(configs_root) / f"{dataset}.json"]
    return [
        configs_dir("cohorts", dataset, "folds.json"),
        _DEFAULT_CONFIGS / f"{dataset}.json",
    ]


def _resolve_manifest_path(dataset: str, configs_root=None) -> Optional[Path]:
    """The first candidate that exists, or None."""
    return next((c for c in _manifest_candidates(dataset, configs_root) if c.exists()), None)


def load_fold_split(
    dataset: str,
    fold: int = 0,
    configs_root: Path | str | None = None,
) -> FoldSplit:
    """Load one fold's explicit subject lists from ``src/neuroatlas/configs/folds/<dataset>.json``.

    Resolution order (first hit wins):

    1. ``src/neuroatlas/configs/cohorts/<dataset>/folds.json`` — the dataset's own dossier, for
       cohorts already migrated to a manifest.
    2. ``src/neuroatlas/configs/folds/<dataset>.json`` — the historical location.

    Both are supported so datasets can migrate one at a time; an explicit
    ``configs_root`` bypasses (1) entirely and is used by the tests.

    Args:
        dataset: Manifest slug, e.g. ``"chbmit"`` or ``"sleep_edf_expanded"``.
        fold: Zero-based fold index.
        configs_root: Override the manifest directory. When given, only that
            directory is searched.

    Raises:
        FileNotFoundError: No manifest for ``dataset`` in any searched location.
        ValueError: Malformed manifest, out-of-range fold, leaky split, or a
            derived split that disagrees with a materialized ``splits`` block.
    """
    path = _resolve_manifest_path(dataset, configs_root)
    if path is None:
        candidates = _manifest_candidates(dataset, configs_root)
        searched = "\n  ".join(str(c) for c in candidates)
        available = sorted(p.stem for p in _DEFAULT_CONFIGS.glob("*.json"))
        available += sorted(
            p.parent.name for p in configs_dir("cohorts").glob("*/folds.json")
        )
        raise FileNotFoundError(
            f"No fold manifest for {dataset!r}. Searched:\n  {searched}\n"
            f"Available: {', '.join(sorted(set(available))) or '(none)'}"
        )

    payload = json.loads(path.read_text())
    if payload.get("dataset") not in (None, dataset):
        raise ValueError(
            f"manifest dataset mismatch: requested {dataset!r}, file declares "
            f"{payload['dataset']!r} ({path})"
        )

    folds = _folds_container(payload)
    n_folds = int(payload.get("n_folds", len(folds)))
    if not 0 <= fold < n_folds:
        raise ValueError(f"{dataset} has {n_folds} folds (0 to {n_folds - 1}); there is no "
                         f"fold {fold}")

    entry = folds.get(str(fold))
    if entry is None:
        raise ValueError(f"{dataset}: fold key {str(fold)!r} absent; have {sorted(folds)}")

    stats = None
    if isinstance(entry, dict):
        shape = "explicit"
        missing = {"train", "val", "test"} - set(entry)
        if missing:
            raise ValueError(f"{dataset} fold {fold}: explicit entry missing {sorted(missing)}")
        split = {k: list(entry[k]) for k in ("train", "val", "test")}
        stats = entry.get("stats")
    elif isinstance(entry, list):
        shape = "partition"
        split = _derive_from_partition(folds, fold, n_folds)
        materialized = payload.get("splits")
        if isinstance(materialized, dict) and str(fold) in materialized:
            frozen = materialized[str(fold)]
            for key in ("train", "val", "test"):
                if key in frozen and set(frozen[key]) != set(split[key]):
                    raise ValueError(
                        f"{dataset} fold {fold}: derived {key} disagrees with the manifest's "
                        f"materialized 'splits' block. Rule used: {SPLIT_RULE}. "
                        f"derived_n={len(split[key])} frozen_n={len(frozen[key])}"
                    )
            split = {k: list(frozen.get(k, split[k])) for k in ("train", "val", "test")}
    else:
        raise ValueError(f"{dataset} fold {fold}: unsupported entry type {type(entry).__name__}")

    _assert_disjoint(dataset, fold, **split)
    return FoldSplit(
        dataset=dataset,
        fold=fold,
        n_folds=n_folds,
        shape=shape,
        source_path=path,
        stats=stats,
        **split,
    )


def available_manifests(configs_root: Path | str | None = None) -> List[str]:
    """List manifest slugs found under the configs root."""
    root = Path(configs_root) if configs_root is not None else _DEFAULT_CONFIGS
    return sorted(p.stem for p in root.glob("*.json"))


def check_subject_grouping(
    split: FoldSplit,
    subject_of: "Callable[[str], str]",
    raise_on_leak: bool = True,
) -> Dict[str, object]:
    """Verify a split is disjoint at *subject* level, not just at recording level.

    Manifests list recording ids.  When a cohort has several recordings per
    subject (e.g. Sleep-EDF Cassette's two nights per subject), a split that is
    disjoint over recordings can still put the same subject in train and test.
    An earlier Sleep-EDF manifest here had exactly this defect and was removed,
    so any new manifest should be run through this check.

    Args:
        split: The fold to validate.
        subject_of: Maps a recording id to its subject id.  For Sleep-EDF
            Cassette: ``lambda r: r[:5]`` (``SC4<ss>``).
        raise_on_leak: Raise ``ValueError`` when a subject spans two splits.

    Returns:
        A report dict with the per-pair overlapping subject ids.
    """
    groups = {name: {subject_of(r) for r in ids}
              for name, ids in (("train", split.train), ("val", split.val), ("test", split.test))}
    overlaps = {
        f"{a}&{b}": sorted(groups[a] & groups[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    leaked = {k: v for k, v in overlaps.items() if v}
    report = {
        "dataset": split.dataset,
        "fold": split.fold,
        "n_recordings": split.n_subjects,
        "n_subjects": len(set().union(*groups.values())),
        "overlaps": overlaps,
        "clean": not leaked,
    }
    if leaked and raise_on_leak:
        detail = "; ".join(f"{k}: {v[:8]}{'...' if len(v) > 8 else ''} ({len(v)})"
                           for k, v in leaked.items())
        raise ValueError(
            f"{split.dataset} fold {split.fold} leaks at subject level — {detail}. "
            f"Folds were assigned per recording without grouping recordings by subject. "
            f"Pass raise_on_leak=False to proceed anyway (results are not patient-independent)."
        )
    return report


def recording_indices_for_split(
    subject_ids_per_recording: Sequence[str],
    split: FoldSplit,
    strict: bool = True,
) -> Dict[str, List[int]]:
    """Map a subject-level :class:`FoldSplit` onto recording indices.

    Adapters index their data per *recording*, while manifests are per *subject*.
    This is the bridge, replacing the runtime ``StratifiedKFold`` calls so that
    frozen splits are used verbatim.

    Args:
        subject_ids_per_recording: Subject id for each recording, in adapter order.
        split: The fold to apply.
        strict: If True, raise when the manifest names subjects absent from the
            data, or when recordings belong to no split.  Set False to tolerate a
            partial download (a warning-free subset), which makes results
            incomparable to the paper.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` of recording indices.
    """
    observed = set(subject_ids_per_recording)
    assigned = {"train": set(split.train), "val": set(split.val), "test": set(split.test)}
    manifest_subjects = assigned["train"] | assigned["val"] | assigned["test"]

    if strict:
        missing = manifest_subjects - observed
        if missing:
            raise ValueError(
                f"{split.dataset} fold {split.fold}: {_some(missing)} of the benchmark's "
                f"folds {'is' if len(missing) == 1 else 'are'} not in the data: the data is "
                f"incomplete, or names its subjects differently. The folds name subjects such "
                f"as {', '.join(sorted(manifest_subjects)[:3])}; the data "
                f"{', '.join(sorted(observed)[:3]) or 'has none'}."
                f"\nfix: neuroatlas data status {split.dataset}"
            )
        unassigned = observed - manifest_subjects
        if unassigned:
            raise ValueError(
                f"{split.dataset} fold {split.fold}: {_some(unassigned)} in the data "
                f"{'is' if len(unassigned) == 1 else 'are'} in none of the benchmark's "
                f"folds: the folder holds subjects the benchmark does not use."
                f"\nfix: neuroatlas data status {split.dataset}"
            )

    out: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for idx, sid in enumerate(subject_ids_per_recording):
        for name, members in assigned.items():
            if sid in members:
                out[name].append(idx)
                break
    return out


def _some(subjects) -> str:
    """``3 subjects (P01, P07, P09)``, ``12 subjects (P01, P02, P03 and 9 more)``."""
    ids = sorted(str(s) for s in subjects)
    shown = ", ".join(ids[:3]) + (f" and {len(ids) - 3} more" if len(ids) > 3 else "")
    return f"{len(ids)} subject{'' if len(ids) == 1 else 's'} ({shown})"


def fold_source_label(split: FoldSplit) -> str:
    """Where a split came from, relative to the shipped configs
    (``cohorts/sz1/folds.json``), so that recording it in a datamodule's
    metadata -- which per-split embedding caches hash -- does not tie a cache
    key to the directory the package is installed in."""
    try:
        return str(Path(split.source_path).resolve().relative_to(configs_dir().resolve()))
    except ValueError:
        return Path(split.source_path).name


def recording_splits_from_manifest(
    dataset: str,
    fold: int,
    n_folds: int,
    subject_ids_per_recording: Sequence[str],
    *,
    strict: bool = True,
) -> "tuple[List[int], List[int], List[int], FoldSplit]":
    """One fold of a cohort's frozen manifest, as recording indices.

    The call a manifest-driven adapter makes in place of its own splitter: the
    published folds come from the file, so a change to the reader's subject
    discovery, its stratification labels or the shared splitter cannot move
    them.

    Raises when the requested fold count is not the manifest's: ``n_folds=7``
    over a 5-fold manifest has no published answer, and quietly serving fold
    ``k % 5`` would put the paper's name on a different experiment. The
    adapters take ``folds_manifest=None`` (``--set folds_manifest=none``) to
    derive ``n_folds`` folds with their own splitter instead.

    Returns ``(train, val, test, split)``: recording indices in reader order,
    and the :class:`FoldSplit`, for provenance (:func:`fold_source_label`).
    """
    split = load_fold_split(dataset, fold)
    if int(n_folds) != split.n_folds:
        fix = None
        try:
            from neuroatlas.cli import corrected_command

            fix = (corrected_command({f"n_folds={n_folds}": f"n_folds={split.n_folds}"})
                   or corrected_command({f"num_folds={n_folds}": f"num_folds={split.n_folds}"}))
        except Exception:
            fix = None
        raise ValueError(
            f"{dataset} has {split.n_folds} folds in the benchmark, not {n_folds}. With --set folds_manifest=none, `neuroatlas embed` and "
            f"`neuroatlas probe` cut {n_folds} folds with the dataset's own splitter instead "
            f"(not the benchmark's folds)." + (f"\nfix: {fix}" if fix else "")
        )
    try:
        recs = recording_indices_for_split(
            [str(s) for s in subject_ids_per_recording], split, strict=strict,
        )
    except ValueError as exc:
        text, _, fix = str(exc).partition("\nfix: ")
        raise ValueError(
            f"{text} With --set strict_folds=false, `neuroatlas embed` and `neuroatlas "
            f"probe` run on the subjects the data and the folds share, each in its fold's "
            f"role (not the benchmark's folds)." + (f"\nfix: {fix}" if fix else "")
        ) from exc
    return recs["train"], recs["val"], recs["test"], split


# --------------------------------------------------------------------------
# The whole-manifest view, merged here from patient_splits.py so that
# both readers share one search order and one parser.
# --------------------------------------------------------------------------

import hashlib
from dataclasses import field
from typing import Literal, Set

Split = Literal["train", "val", "test"]
_SPLITS: tuple = ("train", "val", "test")

_SUPPORTED_SCHEMA_VERSIONS = (None, 1, 2)

#: Fold-container keys, in preference order. ``subject_folds`` wins, matching
#: ``probe_from_embeddings._load_fold_split``: ``shhs_combined`` ships both a
#: coarse ``patient_folds`` and the subject-level ``subject_folds``, and the
#: embedding store is keyed by subject id. No manifest currently carries both
#: ``folds`` and ``subject_folds``, so this ordering changes no existing result.
_CONTAINER_KEYS = ("subject_folds", "folds")

PARTITION_RULE = "test=folds[k]; val=folds[(k+1) % n]; train=rest"


@dataclass
class FoldsManifest:
    dataset: str
    n_folds: int
    n_subjects: int
    generator: str
    subject_seizure_flag: Dict[str, bool]
    folds: List[Dict]
    source_path: Path
    #: ``"explicit"`` or ``"partition[<container key>]"`` — recorded in run
    #: artifacts so a result can always be traced back to how its split was derived.
    convention: str = "explicit"
    #: sha256 of the manifest file, for provenance in results.json.
    sha256: str = ""

    def split_subjects(self, fold: int, split: Split) -> Set[str]:
        return set(self.ordered_split_ids(fold, split))

    def ordered_split_ids(self, fold: int, split: Split) -> List[str]:
        """Ids in manifest order.

        Order is load-bearing for the probe path: features are stacked in this
        order and per-subject predictions are sliced back out by the resulting
        boundaries. Within a fold the order is the manifest's own; for the
        partition schema, ``train`` concatenates the contributing groups in
        ascending fold index — the same order
        ``probe_from_embeddings._load_fold_split`` produced.
        """
        if not 0 <= fold < self.n_folds:
            raise ValueError(f"fold {fold} out of range [0, {self.n_folds})")
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
        return list(self.folds[fold][split])

    def all_subjects(self) -> Set[str]:
        """Every id assigned to any split of any fold.

        Falls back to the union over folds when the manifest carries no
        ``subject_seizure_flag`` (only sz1 does).
        """
        if self.subject_seizure_flag:
            return set(self.subject_seizure_flag.keys())
        return {
            s
            for fold in self.folds
            for split in _SPLITS
            for s in fold[split]
        }

    @property
    def n_ids(self) -> int:
        """Number of distinct ids across all folds (see module docstring)."""
        return len(self.all_subjects())

    def assert_covers(self, observed_subjects: Set[str]) -> None:
        manifest_subjects = self.all_subjects()
        if manifest_subjects != observed_subjects:
            missing = manifest_subjects - observed_subjects
            extra = observed_subjects - manifest_subjects
            raise RuntimeError(
                f"FoldsManifest({self.dataset}): subject set mismatch with cache. "
                f"in_manifest_not_in_cache={sorted(missing)} "
                f"in_cache_not_in_manifest={sorted(extra)}. "
                f"Re-freeze the manifest from the cohort's splitter, or rebuild "
                f"the cache to match it."
            )

    def stats(self, fold: int, split: Split) -> Dict:
        entry = self.folds[fold]
        if "stats" not in entry:
            raise KeyError(
                f"FoldsManifest({self.dataset}) fold {fold} carries no 'stats' block "
                f"(only the schema_version 1/2 epilepsy manifests do)."
            )
        return entry["stats"][split]


def _fold_entries(payload: Dict, path: Path) -> tuple[List, str]:
    """Return the raw per-fold entries in fold order, plus the container key used."""
    for key in _CONTAINER_KEYS:
        raw = payload.get(key)
        if raw:
            break
    else:
        raise ValueError(
            f"{path}: no fold container found (looked for {list(_CONTAINER_KEYS)}); "
            f"top-level keys are {sorted(payload)}"
        )

    if isinstance(raw, dict):
        try:
            ordered_keys = sorted(raw.keys(), key=int)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: {key!r} is keyed by non-integer strings: {sorted(raw)[:8]}"
            ) from exc
        return [raw[k] for k in ordered_keys], key
    if isinstance(raw, list):
        return list(raw), key
    raise ValueError(f"{path}: {key!r} must be a dict or list, got {type(raw).__name__}")


def _explicit_from_partition(parts: Sequence[Sequence[str]], path: Path) -> List[Dict]:
    """Apply ``PARTITION_RULE`` to test-group lists.

    Byte-identical to ``probe_from_embeddings._load_fold_split``, which produced
    every published probe number.
    """
    n = len(parts)
    if n < 3:
        raise ValueError(
            f"{path}: partition schema needs >= 3 folds to leave a non-empty train "
            f"split under '{PARTITION_RULE}', got n_folds={n}"
        )
    folds: List[Dict] = []
    for k in range(n):
        val_idx = (k + 1) % n
        folds.append(
            {
                "train": [s for j in range(n) if j not in (k, val_idx) for s in parts[j]],
                "val": list(parts[val_idx]),
                "test": list(parts[k]),
            }
        )
    return folds


def _assert_matches_materialised_splits(
    folds: List[Dict], payload: Dict, path: Path
) -> None:
    """Cross-check derived splits against a manifest's own ``splits`` block."""
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        return
    for k, derived in enumerate(folds):
        stored = splits.get(str(k))
        if not isinstance(stored, dict):
            continue
        for split in _SPLITS:
            if split not in stored:
                continue
            if set(stored[split]) != set(derived[split]):
                raise ValueError(
                    f"{path}: fold {k} '{split}' disagrees with the manifest's own "
                    f"'splits' block under '{PARTITION_RULE}'. The manifest is "
                    f"internally inconsistent and must be regenerated."
                )


def _validate(folds: List[Dict], dataset: str, path: Path) -> None:
    seen_test: Set[str] = set()
    for fi, fold in enumerate(folds):
        if not isinstance(fold, dict):
            raise ValueError(
                f"{path}: fold {fi} is {type(fold).__name__}, expected a "
                f"train/val/test mapping"
            )
        missing = [s for s in _SPLITS if s not in fold]
        if missing:
            raise ValueError(f"{path}: fold {fi} missing split(s) {missing}")

        train, val, test = (set(fold[s]) for s in _SPLITS)
        for a, b, name_a, name_b in (
            (train, val, "train", "val"),
            (train, test, "train", "test"),
            (val, test, "val", "test"),
        ):
            shared = a & b
            if shared:
                raise ValueError(
                    f"{path}: fold {fi} leaks {len(shared)} id(s) between "
                    f"{name_a} and {name_b}: {sorted(shared)[:8]}. Training on this "
                    f"split would produce an invalid result; regenerate the manifest."
                )

        reused = seen_test & test
        if reused:
            raise ValueError(
                f"{path}: fold {fi} reuses {len(reused)} test id(s) already tested in "
                f"an earlier fold: {sorted(reused)[:8]}"
            )
        seen_test |= test


def load_folds(dataset: str, configs_root: Path | str | None = None) -> FoldsManifest:
    """The whole manifest, with fold order and provenance.

    Same file and same search order as :func:`load_fold_split`, which returns
    one fold's three lists instead. Two views of one file, not two formats:
    they used to resolve paths differently, so a cohort with a copy in both
    locations could be read one way when probing and another when finetuning.
    """
    path = _resolve_manifest_path(dataset, configs_root)
    if path is None:
        raise FileNotFoundError(
            f"Folds manifest not found for {dataset!r}. Expected "
            f"src/neuroatlas/configs/cohorts/{dataset}/folds.json or src/neuroatlas/configs/folds/{dataset}.json."
        )
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)

    schema_version = payload.get("schema_version")
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema_version={schema_version!r} in {path} "
            f"(supported: {_SUPPORTED_SCHEMA_VERSIONS})"
        )
    if payload["dataset"] != dataset:
        raise ValueError(
            f"Manifest dataset mismatch: requested {dataset!r}, file has "
            f"{payload['dataset']!r}"
        )

    entries, container_key = _fold_entries(payload, path)
    n_folds = int(payload.get("n_folds", len(entries)))
    if len(entries) != n_folds:
        raise ValueError(
            f"{path}: n_folds={n_folds} but {container_key!r} has {len(entries)} entries"
        )

    if all(isinstance(e, dict) for e in entries):
        folds = entries
        convention = "explicit"
    elif all(isinstance(e, (list, tuple)) for e in entries):
        folds = _explicit_from_partition(entries, path)
        convention = f"partition[{container_key}]"
        _assert_matches_materialised_splits(folds, payload, path)
    else:
        kinds = sorted({type(e).__name__ for e in entries})
        raise ValueError(
            f"{path}: {container_key!r} mixes fold entry types {kinds}; expected all "
            f"dicts (explicit) or all lists (partition)"
        )

    _validate(folds, dataset, path)

    seizure_flag = {
        s: bool(v) for s, v in (payload.get("subject_seizure_flag") or {}).items()
    }
    manifest = FoldsManifest(
        dataset=payload["dataset"],
        n_folds=n_folds,
        n_subjects=int(payload.get("n_subjects", 0)),
        generator=str(payload.get("generator", "")),
        subject_seizure_flag=seizure_flag,
        folds=folds,
        source_path=path,
        convention=convention,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    if not manifest.n_subjects:
        manifest.n_subjects = manifest.n_ids
    return manifest
