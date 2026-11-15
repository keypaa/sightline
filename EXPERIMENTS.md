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

## EXP-0001: WP10 32-example overfit (G0)

Date: 2026-11-15
Milestone and gate: M0 / G0
Question: can the frozen Qwen3-1.7B + K3 MoonViT-V2 + trained projector reach 99% exact match on 32 scenes within 500 steps, and do the blank/shuffled/swapped controls collapse to chance?
Hypothesis and pass rule: DECISIONS 0025 — 32/32 within 500 steps on at least 2 of 3 seeds; controls <= 20%; vision-minus-blind >= 79 points.
The one change: train the WP7 projector on the 32 WP9 scenes (8 colors x 4 shapes).
Config: `experiments/wp10_overfit/run.json`.
Run records: `experiments/wp10_overfit/metrics.jsonl`.
Result: 3/3 seeds pass; steps_to_99 125/125/150; controls 0% / 6.25% / 6.25%; vision-minus-blind 100%; answer loss gap 15.66 / 15.80 / 15.72. Modal A10G, 1.086 GPU-hours.
Interpretation: the overfit is vision-driven; the controls collapse to chance or below; the projector can fit the 32 scenes.
Decision: G0 sign-off proposed in DECISIONS/0027; no M1 until approved.
