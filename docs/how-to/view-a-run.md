# View a run's metrics and samples

```bash
uv run python -m llm_training.explain.runs runs/tinystories/<run-id>
# writes runs/tinystories/<run-id>/run-view.html; open it in a browser
uv run python -m llm_training.explain.runs runs/<a> runs/<b> --output compare.html
```

The command turns a next-token run directory into one HTML page. The page needs no server
and loads nothing from the network. It shows:

- training and validation loss, the learning rate (with the planned schedule for the whole
  budget), and the gradient norm with the clipping threshold, on one shared axis of updates
  or tokens;
- when each evaluation, checkpoint and new best happened, and where a resumed run took over;
- the latest sample at or before the update under the pointer, the final `sample.json`
  included, so you can watch the text improve;
- the runs table, `evaluation.json`, and every monitored evaluation as tables.

The [training loop guide](../training-loop.md#evaluation-metrics) explains how to read each
number. The page reads only the run's JSON and YAML records, never a checkpoint. It computes
the planned schedule with the trainer's own `learning_rate_at` and checks it against the rate
each update recorded; the TinyStories run matches at all 14,155 updates. Several run
directories overlay, colored by run, with a warning when their losses don't compare
(different tokenizers, prompt masking, validation tokens, monitoring recipes or monitored
subset sizes) or when a run and its resumed child are both given. The child already holds the
parent's history. Every record is read from the run directory itself: a symlinked record is
refused, so a page can't embed a file from elsewhere.

The page is a snapshot: rerun the command to refresh a run that is still training. It covers
next-token runs only; GRPO runs are not supported yet ([backlog](../backlog.md)). Like the
[architecture explorer](../architecture-explorer.md), it lives in `explain/`, outside every
resume fingerprint.
