# Task Extensions

This is the default entrypoint for new evaluation workflows.

Edit here when you are:

- adding a new task
- changing task-specific metrics
- implementing task-specific aggregation or summarization

Canonical pattern:

- `src/neuroatlas/extensions/tasks/<slug>.py`

If a workflow can be expressed as a task, put it here instead of creating a
separate orchestration pipeline.

A task saves each fold's test predictions in the common format
(`neuroatlas/predictions.py`: `predictions.new`, `predictions.finalize`) and
registers `TaskSpec(..., score=...)`, the one function that computes its
metrics from that file. The evaluator records `score()` of the file it has
just written, so `neuroatlas rescore` reproduces those metrics exactly and a
fixed metric reaches old results without probing again.
