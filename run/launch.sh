#!/usr/bin/env bash
#
# Every launch in NeuroAtlas, in one place.
#
# This replaces ~80 shell wrappers that were each the same script with a
# different dataset slug. They existed because every dataset used to have its
# own entrypoint; now there are four verbs and the per-dataset facts live in
# configs/cohorts/<slug>/cohort.yaml, so a launcher has nothing left to carry.
#
# Usage
#   run/launch.sh <verb> [options]
#
#   verb        fetch | prepare | embed | probe
#
#   --dataset   one slug, or a domain: epilepsy | sleep | brain_age | bci | all
#               (default: all)
#   --models    a group -- eeg_fm | ts_fm | supervised | all -- or a family
#               name, or a comma-separated list. Default: every checkpoint.
#   --axis      which probes to run, for `probe` (default: default)
#                 default    each dataset's own task
#                 diagnosis  recording-level diagnosis: OSA, cognitive
#                            impairment, and ISRUC's mixed-disorder cohort
#                 events     microarousals, respiratory events, limb movements
#                 all        every axis above
#   --paper-only  restrict to what the paper evaluates
#   --dry-run     print the commands, run nothing
#
# Examples
#   run/launch.sh probe  --dataset ucddb --models biot,labram
#   run/launch.sh embed  --dataset epilepsy --models eeg_fm
#   run/launch.sh probe  --dataset sleep --axis diagnosis
#   run/launch.sh probe  --dataset sleep --axis all --paper-only
#   run/launch.sh fetch  --dataset all --dry-run
#
# Everything dataset-specific -- montage, subsets, folds, window, label mode --
# comes from the manifest. Override any of it with EXTRA="--set key=value".
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${EEGBENCHMARKS_PYTHON:-python}"
EXTRA="${EXTRA:-}"

VERB=""; SELECT="all"; MODELS=""; AXIS="default"
PAPER_ONLY="${PAPER_ONLY:-}"; DRY_RUN="${DRY_RUN:-}"

usage() { sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[[ $# -eq 0 ]] && usage 0
VERB="$1"; shift
case "$VERB" in fetch|prepare|embed|probe) ;; -h|--help) usage 0 ;;
    *) echo "unknown verb: $VERB" >&2; usage 2 ;; esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)    SELECT="$2"; shift 2 ;;
        --models)     MODELS="$2"; shift 2 ;;
        --axis)       AXIS="$2";   shift 2 ;;
        --paper-only) PAPER_ONLY=1; shift ;;
        --dry-run)    DRY_RUN=1;    shift ;;
        -h|--help)    usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 2 ;;
    esac
done

sets() { PYTHONPATH="${REPO}/src" "$PYTHON_BIN" "${REPO}/run/_launch_sets.py" "$@"; }

# --dataset takes either a domain or a single slug; a domain expands from the
# registry, so a dataset added there is launchable the same day.
case "$SELECT" in
    epilepsy|sleep|brain_age|bci|all)
        args=("$SELECT"); [[ -n "$PAPER_ONLY" ]] && args+=(--paper-only)
        DATASETS="$(sets datasets "${args[@]}" | tr '\n' ' ')" ;;
    *)  DATASETS="$SELECT" ;;
esac
[[ -z "${DATASETS// }" ]] && { echo "no datasets selected: --dataset $SELECT" >&2; exit 2; }

# Resolve a model group once, up front, so an unknown one fails before any job
# starts rather than on the last dataset of a long sweep.
[[ -n "$MODELS" ]] && MODELS="$(sets models "$MODELS")"

run() {
    echo "+ $*"
    [[ -n "$DRY_RUN" ]] && return 0
    PYTHONPATH="${REPO}/src" "$@"
}

for ds in $DATASETS; do
    model_args=(); [[ -n "$MODELS" ]] && model_args=(--models "$MODELS")
    case "$VERB" in
        fetch)
            run "$PYTHON_BIN" -m entrypoints.fetch --dataset "$ds" ;;
        prepare)
            # most cohorts need none; `prepare` says so and exits non-zero
            run "$PYTHON_BIN" -m entrypoints.prepare --dataset "$ds" \
                ${EXTRA:+$EXTRA} || true ;;
        embed)
            run "$PYTHON_BIN" -m entrypoints.embed --dataset "$ds" \
                "${model_args[@]}" ${EXTRA:+$EXTRA} ;;
        probe)
            while IFS= read -r task_args; do
                # shellcheck disable=SC2086  -- task_args is a flag string
                run "$PYTHON_BIN" -m entrypoints.probe --dataset "$ds" \
                    $task_args "${model_args[@]}" ${EXTRA:+$EXTRA}
            done < <(sets probes "$ds" "$AXIS") ;;
    esac
done
