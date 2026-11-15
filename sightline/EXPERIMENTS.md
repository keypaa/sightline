# EXPERIMENTS

One entry per experiment, newest last. An experiment changes one thing. Negative results get the same entry as positive ones. Entries are never deleted; a later entry may refer to an earlier one.

## Entry template

```
## EXP-NNNN: <short title>

Date: YYYY-MM-DD
Milestone and gate: Mn / Gn
Question: what this run is meant to answer.
Hypothesis and pass rule: written and frozen before the run (link the DECISIONS file).
The one change: what differs from the previous experiment.
Config: experiments/<name>/config.<ext> (immutable after launch)
Run records: experiments/<name>/runs/<run_id>/ (commit, seeds, versions, backend, dtypes, metrics, GPU-hours)
Result: numbers with intervals, controls included.
Interpretation: what the result does and does not show.
Decision: what happens next, and the DECISIONS file it created, if any.
```

## Log

(empty: the first entry is written when the first experiment runs, expected in WP10 of M0)
