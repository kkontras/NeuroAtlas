# Task Extensions

This is the default entrypoint for new evaluation workflows.

Edit here when you are:

- adding a new task
- changing task-specific metrics
- implementing task-specific aggregation or summarization

Canonical pattern:

- `src/extensions/tasks/<slug>.py`

If a workflow can be expressed as a task, put it here instead of creating a
separate orchestration pipeline.
