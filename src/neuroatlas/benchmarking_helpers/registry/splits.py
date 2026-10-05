"""One rule for turning a requested fold count into a splitter that works.

Every adapter cross-validates at the subject level, and each one used to build
its ``StratifiedKFold`` by hand.  That produced three different behaviours for
the same situation — asking for more folds than the data can stratify:

* ``bonn``, ``tuab``, ``helsinki_neonatal`` quietly **reduced** the fold count
  to the minority-class size, so ``--set n_folds=40`` silently ran 7 folds;
* ``siena``, ``sz1``, ``sz2`` capped at the *subject* count and then handed
  that to ``StratifiedKFold``, which has a stricter requirement, so they
  **raised** ``n_splits=14 cannot be greater than the number of members in
  each class``;
* ``chbmit``, ``epilepsiae``, ``tusz``, ``parkinson`` and the brain-age
  adapters did not cap at all and **raised** even sooner.

That third behaviour is what makes leave-one-subject-out impossible to ask
for: LOSO is just ``n_folds = n_subjects``, and stratification cannot survive
one subject per fold.

The rule: keep the fold count the caller asked for, and drop stratification
only where scikit-learn itself refuses it.  Reducing the folds answers a
question nobody asked; dropping stratification still answers this one.

"Where scikit-learn refuses" is the whole of it.  ``StratifiedKFold`` raises
only when *every* class has fewer members than folds; a class that is merely
sparse (fewer members than folds, while another class has enough) is split
anyway, with a warning.  Every splitter the published folds were drawn with --
the adapters' own ``StratifiedKFold`` calls, the paper's epilepsy probe
(``probe_sz1_sklearn._build_patient_splits``) and the brain-age folds -- had
exactly that behaviour.  A first version of this module fell back to ``KFold``
as soon as *any* class was sparse, which moved the folds of every cohort with
one sparse stratum (all five ISRUC brain-age test folds; EPILEPSIAE, whose
four seizure-free patients are a class of four).  That stricter rule is still
available as ``sparse_classes="fall_back"``; nothing uses it by default.

The invariant that makes this safe to adopt everywhere:

    whenever scikit-learn accepts the labels, this returns exactly the
    splitter the adapter built before -- same class, same n_splits, same
    seed, so the same folds.

Only the cases that used to crash behave differently.  The internal test suite
pins that.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence, Tuple

__all__ = ["effective_n_splits", "make_subject_kfold", "describe_split",
           "why_not_stratified"]


def effective_n_splits(requested: int, n_samples: int) -> int:
    """Fold count that ``KFold`` can actually produce for *n_samples*.

    Capped at one fold per sample (which is leave-one-out) and floored at 2,
    because a single fold has no held-out set.
    """
    if n_samples < 2:
        raise ValueError(
            f"cross-validation needs at least 2 subjects, got {n_samples}"
        )
    return max(2, min(int(requested), int(n_samples)))


def _can_stratify(labels: Optional[Sequence], n_splits: int) -> bool:
    """Every class has at least ``n_splits`` members (no sparse class).

    This is the strict condition ``sparse_classes="fall_back"`` requires;
    scikit-learn's own (and the default here) is weaker, see
    :func:`make_subject_kfold`.

    A single class is deliberately allowed.  It looks like a case for plain
    ``KFold`` -- there is nothing to stratify by -- but the two do not produce
    the same folds, and several cohorts are single-class in practice: CHB-MIT
    stratifies on "does this patient have any seizure", and in a seizure
    corpus every patient does.  Falling back to ``KFold`` there would silently
    move every fold in a dataset that had been splitting fine for years.
    ``StratifiedKFold`` handles one class without complaint as long as it has
    at least ``n_splits`` members, which is the same condition as for two.
    """
    if labels is None:
        return False
    counts = Counter(labels)
    if not counts:
        return False
    return min(counts.values()) >= n_splits


def why_not_stratified(labels: Optional[Sequence], n_splits: int) -> str:
    """The reason stratification was skipped, for the log line."""
    if labels is None:
        return "no labels given"
    counts = Counter(labels)
    if not counts:
        return "no labels"
    largest = max(counts.values())
    if largest < n_splits:
        return (f"every class has fewer subjects than the {n_splits} folds "
                f"(the largest has {largest})")
    smallest = min(counts.values())
    return (f"the smallest class has {smallest} subject"
            f"{'' if smallest == 1 else 's'}, fewer than the {n_splits} folds")


def make_subject_kfold(
    n_splits: int,
    labels: Optional[Sequence] = None,
    *,
    seed: int = 42,
    n_samples: Optional[int] = None,
    sparse_classes: str = "stratify",
) -> Tuple[object, bool, int]:
    """Return ``(splitter, stratified, n_splits)`` for a subject-level split.

    Args:
        n_splits: folds the caller asked for. Reduced only when it exceeds the
            number of subjects, where it becomes leave-one-subject-out.
        labels: per-subject stratification labels, or None for no attempt.
        seed: ``random_state``; every splitter here shuffles.
        n_samples: subject count, when *labels* is None or a different length.
        sparse_classes: what a class with fewer members than folds does.
            ``"stratify"`` (the default) keeps ``StratifiedKFold`` whenever
            scikit-learn accepts the labels -- it only refuses when *every*
            class is smaller than the fold count, and merely warns otherwise
            -- which is what an adapter that called ``StratifiedKFold``
            directly got, and so what every published fold was drawn with.
            ``"fall_back"`` drops to ``KFold`` as soon as any class is
            smaller than the fold count; it reproduces no published fold.

    The returned flag says whether stratification survived, so the caller can
    record it rather than leave a reader guessing which one ran.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    total = n_samples if n_samples is not None else (len(labels) if labels is not None else None)
    if total is None:
        raise ValueError("give either labels or n_samples")
    n_splits = effective_n_splits(n_splits, total)

    if sparse_classes not in ("fall_back", "stratify"):
        raise ValueError(f"sparse_classes must be 'fall_back' or 'stratify', got {sparse_classes!r}")
    stratifiable = _can_stratify(labels, n_splits)
    if not stratifiable and sparse_classes == "stratify" and labels is not None:
        counts = Counter(labels)
        stratifiable = bool(counts) and max(counts.values()) >= n_splits
    if stratifiable:
        return StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=seed), True, n_splits
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed), False, n_splits


def describe_split(stratified: bool, n_splits: int, requested: int,
                   labels: Optional[Sequence] = None) -> str:
    """One line for the log, so which splitter ran is never a guess."""
    how = ("stratified" if stratified
           else f"unstratified ({why_not_stratified(labels, n_splits)})")
    if n_splits != requested:
        return (f"{n_splits}-fold {how}; reduced from the requested {requested} "
                f"— that is one subject per fold, i.e. leave-one-subject-out")
    return f"{n_splits}-fold {how}"
