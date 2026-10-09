# 6. Results

Every run, on this machine or on a cluster, writes its results to the output folder, and
`neuroatlas results` gathers them into one table per benchmark. Each probe also saves its test
predictions, so a metric can be recomputed later without fitting anything again.

## 6.1 The results table

```bash
neuroatlas results sleep_stage
neuroatlas results sleep_stage -v
neuroatlas results sleep_stage 'runs/old/**/results.json' --format md
```

`results` reads every `results.json` under `<output root>/<benchmark>/`, from local runs and
cluster jobs alike. You can also name files, folders or quoted globs. It prints one row per
dataset, variant and checkpoint:

```text
$ neuroatlas results sleep_stage
sleep_stage: Cohen's kappa over 30 s epochs, 5 stages (W, N1, N2, N3, REM), each fold's test subjects pooled, unscored epochs left out; higher is better; chance 0
folds: 5 subject-level folds; the probe's C is chosen from 0.001-100 on validation Cohen's kappa, unweighted loss; ± = population SD over the folds
also: bal_acc = balanced accuracy, macro_F1 = macro-F1
dataset             model            kappa  ±      folds  bal_acc  macro_F1
sleep_edf_expanded  biot_pretrained  0.758  0.016  5/5    0.669    0.675
```

The first lines state what the headline is computed over, what a fold is, what ± means, and the
names of the other columns. In the table, the first number after the model is the headline's mean
over the folds that succeeded, and the second is its SD, `n/a` with one fold.

`folds` is the folds done out of the protocol's folds, such as `5/5`, `1/5` after `--debug`, or
`LOSO 1/9`. A warning tells you when a mean covers fewer folds than the protocol, because it is
not comparable with a full run.

A `chance` column appears where a guess scores differently per dataset. For BCI it is 1 over the
number of classes. For the sleep-event benchmarks it is the share of positive test epochs. A row
whose headline is `n/a` gives the reason on the line under it.

`--variant NAME` shows one variant's rows. A `variant` column appears once a variant other than
the default has results.

## 6.2 Per-fold values

`-v` adds every fold's value under each row, with the C the probe chose where it picks C from a
grid:

```text
$ neuroatlas results epilepsy -v
...
dataset  model               Sens@FA_AUC(event)  ±      folds  AUROC(window)  AUPRC(window)  bal_acc  MCC    event_F1  sens@1FA/h
siena    cbramod_pretrained  0.496               0.159  5/5    0.813          0.122          0.621    0.146  0.140     0.118
      by fold: 0: 0.428 (C 0.01), 1: 0.516 (C 10), 2: 0.641 (C 0.001), 3: 0.566 (C 0.01), 4: 0.227 (C 0.001)
```

On epilepsy each fold's value is that fold's own Sens@FA AUC. The headline is the AUC of the
folds' median curve, so it is not the mean of the per-fold values. `-v` also shows why failed
folds failed and how many folds are still to run.

In JSON a missing number is `null`. Every row has the fields `benchmark`, `metric`, `variant`,
`status`, `chance`, `C`, `per_fold`, `n_folds`, `n_expected`, `n_failed`, `failures`, `errors` and
`note`.

## 6.3 Duplicates

A result recorded twice counts once, for example from a quick `run` and a later cluster job of
the same dataset, variant, model, fold and task. The newest file wins, and a warning says how many
duplicates were dropped. Two variants of the same model and fold are two results.

`results` exits with 0 when there is something to show, 1 when there are no results, and 2 for a
usage error.

## 6.4 Results on disk

`run` writes to `<output root>/<benchmark>/<dataset>/`. A variant adds `/<variant>`. The folder
holds `results.json`, one record per checkpoint and fold, which `neuroatlas results` reads. Next
to it, `probes/` holds one folder per checkpoint and fold with the fitted probe, its test
predictions (`predictions.npz`) and the record it wrote (`result.json`).

A later run adds to `results.json` and replaces the folds it runs again. A successful retry also
removes the earlier failed record of that fold. Each record notes when it was written
(`metadata.written_at`). Cluster jobs write one `results.json` per checkpoint instead
([Running on a cluster](running.md#57-running-on-a-cluster)).

## 6.5 Rescoring saved predictions

Every probe writes each fold's test predictions to `predictions.npz` in the fold's probe folder,
`<output root>/<benchmark>/<dataset>/probes/<dataset>/<model>/<key>/`. Every metric a result
records is computed from this file, so a metric added or corrected later reaches old results
without probing again:

```bash
neuroatlas rescore sleep_stage
neuroatlas rescore epilepsy --dataset siena -m cbramod_pretrained
```

```text
$ neuroatlas rescore sleep_stage --dataset sleep_edf_expanded -m biot_pretrained
dataset             model            fold  result          kappa before  after
sleep_edf_expanded  biot_pretrained  0     rescored, same  0.777328      0.777328
...
sleep_edf_expanded  biot_pretrained  4     rescored, same  0.752377      0.752377

5 rescored from saved predictions: 0 changed, 5 the same; 1 results.json file rewritten
```

`rescore` fits nothing. It recomputes each record's metrics, keeps its metadata, adds
`metadata.rescored_at`, and rewrites `results.json`. A record whose fold has no saved predictions
is listed as `skipped`, with the command that probes it again. `rescore` takes `--dataset`, `-m`,
`--variant`, `--output-root` and `--format` like `results`.

## 6.6 The predictions file

Load it with `np.load("predictions.npz")`. It contains no pickles, and
`help(neuroatlas.predictions)` describes it in full.

| Entry | Contents |
|---|---|
| `format`, `dataset`, `checkpoint_id`, `task`, `fold` | what made the file. `fold` is empty for a dataset with one fixed split. |
| `y_true`, `y_pred` | the true and predicted class, or value, for each test row |
| `y_proba`, `classes` | class probabilities (rows by classes) and the class of each column |
| `y_score` | for binary tasks, the positive class's score, which AUROC, AUPRC and Sens@FA read |
| `subject_id`, `recording_id`, `session_id`, `epoch_index`, `trial_idx`, `window_start_s` | the row identifiers that apply |
| `window_s`, `threshold` | epilepsy only: seconds per window, and the decision threshold tuned on validation |
| `<group>/<column>` | further probes of the fold, such as one per arousal threshold, or every ridge alpha for brain age |
| `seed_y_pred`, `seed_y_score` | every seed's predictions, with `probe --seed-mode shared` |
| `info` | JSON with what the probe chose on validation, the settings the metrics read, and what the probe was given |

A row is what the metric counts: a 30 s epoch for the sleep tasks, a 10 s window for epilepsy, a
trial for BCI, and a subject or recording for diagnosis and brain age. A Sleep-EDF staging fold
takes about 1.8 MB.

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score
from neuroatlas import predictions

z = np.load("predictions.npz")
cohen_kappa_score(z["y_true"], z["y_pred"])   # one fold, by hand
predictions.score("predictions.npz")          # every metric the task records
```
