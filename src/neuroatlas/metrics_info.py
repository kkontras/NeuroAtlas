"""What every metric a benchmark reports is: the metric registry.

One entry per metric key (the key a task writes in results.json): the short
label a table heads its column with, the metric's name in a sentence, a
one-line meaning, what one scored item is (the unit: a 30 s epoch, a 10 s
window, a seizure event, a subject, a trial, a recording), the direction, and
chance -- a number, :data:`PREVALENCE` (AUPRC: the share of positive test
items), :data:`ONE_OVER_C` (balanced accuracy: 1 over the number of classes)
or None (no chance level the table can state).

A benchmark changes an entry in two ways: its YAML's ``metrics.unit`` is the
unit of every metric that does not name its own (an event-level metric always
counts seizure events), and :data:`_OVERRIDES` holds what one benchmark means
differently by the same key -- brain age's ``pearson_r`` is the correlation of
predicted and true age across test subjects, the hypnogram's is a mean over
features. ``results``, ``show``, ``list benchmarks`` and ``rescore`` read this
module; :func:`info` is the one lookup.

    from neuroatlas import catalog, metrics_info
    bench = catalog.load("sleep_diagnosis")
    metrics_info.info("auroc", bench).meaning
    # 'area under the ROC curve of the predicted probability of the positive
    #  class, over the test subjects'
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Union

#: chance is the share of positive test items (what AUPRC scores at random)
PREVALENCE = "prevalence"
#: chance is 1/C, C the number of classes (balanced accuracy at random)
ONE_OVER_C = "1/C"

Chance = Union[float, str, None]


@dataclass(frozen=True)
class MetricInfo:
    key: str
    label: str                  # the column label in a table
    name: str                   # the metric in a sentence
    meaning: str                # one line; "{units}" becomes e.g. "30 s epochs"
    unit: Optional[str] = None  # one scored item; None: the benchmark's metrics.unit
    value_unit: str = ""        # what the number is in ("years"), "" for a ratio
    higher_is_better: bool = True
    chance: Chance = None

    def describe(self) -> str:
        """The meaning, with the unit filled in."""
        units = _plural(self.unit) if self.unit else "items"
        return self.meaning.format(units=units)

    def chance_text(self) -> Optional[str]:
        """``chance 0``, ``chance = the test prevalence``, ``chance = 1/C``, or
        None when no chance level is stated."""
        return chance_text(self.chance)


def chance_text(chance: Chance) -> Optional[str]:
    if chance is None:
        return None
    if chance == PREVALENCE:
        return "chance = the test prevalence"
    if chance == ONE_OVER_C:
        return "chance = 1/C (C classes)"
    return f"chance {float(chance):g}"


def _plural(unit: str) -> str:
    return unit if unit.endswith("s") else unit + "s"


_BASE: Dict[str, MetricInfo] = {m.key: m for m in (
    MetricInfo("cohen_kappa", "kappa", "Cohen's kappa",
               "agreement of predicted and true classes beyond chance agreement, over the "
               "test {units} (0 = chance, 1 = perfect)", chance=0.0),
    MetricInfo("balanced_accuracy", "bal_acc", "balanced accuracy",
               "the mean over the classes of each class's recall, over the test {units}",
               chance=ONE_OVER_C),
    MetricInfo("macro_f1", "macro_F1", "macro-F1",
               "the mean over the classes of each class's F1 score, over the test {units}"),
    MetricInfo("accuracy", "acc", "accuracy",
               "the share of the test {units} classified correctly"),
    MetricInfo("auroc", "AUROC", "AUROC",
               "area under the ROC curve of the predicted probability of the positive "
               "class, over the test {units}", chance=0.5),
    MetricInfo("auprc", "AUPRC", "AUPRC",
               "area under the precision-recall curve (average precision) of the predicted "
               "probability of the positive class, over the test {units}",
               chance=PREVALENCE),
    MetricInfo("mcc", "MCC", "Matthews correlation",
               "correlation of the predicted and the true class, over the test {units}",
               chance=0.0),
    MetricInfo("event_sens_fa_auc", "Sens@FA_AUC(event)", "event-level Sens@FA AUC",
               "event-level sensitivity averaged over 0.1-100 false alarms per hour on a "
               "log axis (App. C.1, Eq. 2): seizures detected, at every alarm budget",
               unit="seizure event"),
    MetricInfo("ovlp_f1", "event_F1", "event-level F1",
               "F1 of the predicted against the annotated seizure events, a seizure "
               "detected when a predicted event overlaps it", unit="seizure event"),
    MetricInfo("sensitivity_at_fpr_h_1_0", "sens@1FA/h", "sensitivity at 1 FA/h",
               "the share of seizures detected at the threshold that gives 1 false alarm "
               "per hour", unit="seizure event"),
    MetricInfo("sensitivity_at_fpr_h_0_1", "sens@0.1FA/h", "sensitivity at 0.1 FA/h",
               "the share of seizures detected at the threshold that gives 0.1 false alarm "
               "per hour", unit="seizure event"),
    MetricInfo("mae", "MAE", "MAE", "mean absolute error of the prediction, over the "
               "test {units}", higher_is_better=False),
    MetricInfo("pearson_r", "r", "Pearson r",
               "Pearson correlation of the predicted and the true value, across the test "
               "{units}"),
    MetricInfo("r2", "R2", "R2", "coefficient of determination of the prediction, across "
               "the test {units}"),
)}

#: {benchmark: {metric key: fields that benchmark means differently}}
_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "brain_age": {
        "mae": {"label": "MAE(years)", "name": "MAE in years", "value_unit": "years",
                "meaning": "mean absolute error of the predicted age, in years, over the "
                           "test {units}"},
        "pearson_r": {"label": "r(age)", "name": "Pearson r of age",
                      "meaning": "Pearson correlation of the predicted and the true age, "
                                 "across the test {units}"},
        "r2": {"label": "R2(age)", "name": "R2 of age",
               "meaning": "coefficient of determination of the predicted age, across the "
                          "test {units}"},
    },
    "sleep_hypnogram": {
        "pearson_r": {"label": "mean_r", "name": "mean Pearson r",
                      "meaning": "the mean over the hypnogram features of each feature's "
                                 "Pearson r between its value from the predicted and from "
                                 "the scored hypnogram, across the test {units}"},
    },
    "epilepsy": {
        "auroc": {"label": "AUROC(window)"},
        "auprc": {"label": "AUPRC(window)"},
    },
}


def known(key: str) -> bool:
    return key in _BASE


def info(key: str, bench=None) -> MetricInfo:
    """The registry entry of *key* as *bench* (a catalog.Benchmark, or None)
    means it. An unknown key is a KeyError: a benchmark may not report a
    metric nobody has described."""
    base = _BASE[key]
    if bench is None:
        return base
    fields: Dict[str, Any] = {}
    unit = getattr(getattr(bench, "metrics", None), "unit", None)
    if base.unit is None and unit:
        fields["unit"] = unit
    fields.update(_OVERRIDES.get(getattr(bench, "name", bench), {}).get(key, {}))
    return replace(base, **fields) if fields else base


def label(key: str, bench=None) -> str:
    """The column label of *key*; the key itself when the registry does not
    know it (a results file may carry metrics no benchmark reports)."""
    return info(key, bench).label if key in _BASE else key
