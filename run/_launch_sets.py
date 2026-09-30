"""Resolve a dataset or model selector into the slugs it names.

`run/launch.sh` used to hardcode its per-domain dataset lists, which is exactly
how such lists drift: they were missing aub_med, parkinson, sleep_edf, six
brain-age variants and four BCI cohorts by the time anyone checked. Both axes
now come from the registry, so a dataset added to the registry is launchable
the same day.

  datasets <domain> [--paper-only]   one slug per line
  models   <group>                   one family per line
  probes   <slug> <selector>         one probe argument-string per line

domain: epilepsy | sleep | brain_age | bci | all
group : eeg_fm | ts_fm | supervised | all, a family name, or a comma list
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def in_the_paper(manifest: dict) -> bool:
    """Is this spec part of what the paper evaluates?

    Not the same question as "is it one of the 42". Brain age re-labels ten
    sleep cohorts (Appendix B.3), so its specs are paper_dataset:false to keep
    the headline count honest -- but the evaluation is in the paper. A spec
    that names the cohort it re-labels is in scope; only cohorts the paper
    never touches are out.
    """
    return bool(manifest.get("paper_dataset")) or "paper_cohort" in manifest


def datasets(domain: str, paper_only: bool) -> list[str]:
    from neuroatlas.benchmarking_helpers.registry.discovery import dataset_specs

    out = []
    for spec in sorted(dataset_specs(), key=lambda s: s.slug):
        manifest = spec.manifest or {}
        got = manifest.get("domain")
        if not got:
            continue                      # MOABB-generated, no dossier
        if paper_only and not in_the_paper(manifest):
            continue
        # Brain age is a task, not a domain: Appendix B.3 re-labels sleep
        # cohorts rather than adding data, so `--dataset brain_age` means
        # the sleep cohorts whose adapter exposes an `age` label mode.
        if domain == "brain_age":
            if "age" in ((manifest.get("labels") or {}).get("modes") or {}):
                out.append(spec.slug)
            continue
        if domain in ("all", got):
            out.append(spec.slug)
    return out


def models(group: str) -> list[str]:
    import yaml

    cfg = yaml.safe_load((REPO / "src" / "neuroatlas" / "configs" / "model_groups.yaml").read_text())
    groups = cfg["groups"]
    if group == "all":
        return sorted({f for g in groups.values() for f in g["families"]})
    if group in groups:
        return sorted(groups[group]["families"])
    # a family name, or a comma-separated list of them
    named = [g.strip() for g in group.split(",") if g.strip()]
    known = {f for g in groups.values() for f in g["families"]}
    unknown = [n for n in named if n not in known]
    if unknown:
        raise SystemExit(
            f"error: unknown model group or family: {', '.join(unknown)}\n"
            f"groups: {', '.join(sorted(groups))}, all\n"
            f"families: {', '.join(sorted(known))}"
        )
    return named


# The paper's sleep evaluation has four axes (App. C): epoch-wise staging,
# hypnogram features, event detection, and recording-level diagnosis. Only the
# first and last are launched per (dataset, task) here.
#
# The diagnosis axis is OSA and cognitive impairment, with ISRUC as the cohort
# with other disorders (Fig. 3d). It is NOT Parkinson's or Alzheimer's -- those
# cohorts exist in the registry but the paper does not evaluate them.
#
# Each entry is a shipped preset in src/neuroatlas/configs/tasks/, which already carries the
# right label mode and the subject-level aggregation.
DIAGNOSIS_TASKS = {
    "dod": ["osa"],                     # OSA vs healthy
    "physionet2026": ["cognitive"],     # cognitive impairment
    "isruc": ["pathology"],             # the mixed-disorder cohort
}

# Event detection: microarousals, respiratory events, periodic limb movements.
EVENT_TASKS = {
    "mass": ["mass_arousal"],
    "physionet2026": ["arousal", "limb", "respiratory"],
    "ucddb": ["respiratory"],
}


def probes(slug: str, selector: str) -> list[str]:
    """Probe argument strings for one dataset.

    selector:
      default    the dataset's own default task only
      diagnosis  recording-level diagnosis (OSA, cognitive impairment, ISRUC)
      events     microarousal / respiratory / limb-movement detection
      all        every axis above
    """
    out = []
    if selector in ("default", "all"):
        out.append("")
    if selector in ("diagnosis", "all"):
        out += [f"--task {t}" for t in DIAGNOSIS_TASKS.get(slug, [])]
    if selector in ("events", "all"):
        out += [f"--task {t}" for t in EVENT_TASKS.get(slug, [])]
    return out


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    what, selector = argv[0], argv[1]
    if what == "datasets":
        for slug in datasets(selector, "--paper-only" in argv):
            print(slug)
    elif what == "models":
        print(",".join(models(selector)))
    elif what == "probes":
        for line in probes(selector, argv[2] if len(argv) > 2 else "default"):
            print(line)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
