# DECISION-0032: G1 thresholds and evaluation

Date: 2026-11-15
Status: accepted (frozen before the M1c run)
Work package: M1c

## Context
G1 requires paired evidence that the frozen retrofit uses the image on held-out synthetic tasks. The thresholds must be frozen before the first run.

## Decision
- Held-out test: 500+ items per task (color, shape, count) from the M1a test split.
- Conditions: correct, blank, mismatched, no-tokens, untrained projector.
- Metric: exact match (greedy decoding) per condition; correct minus the best of blank/mismatched/no-tokens.
- G1 pass: correct minus the best control >= 30 points on at least 2 of 3 tasks, McNemar p < 0.01, and all 3 seeds positive; the control conditions within the 95% CI of chance.
- Seeds: 0, 1, 2; the M1b checkpoints.
- The evaluation uses the same chat template and decoding as the training; the exact-match parser.

## Implementation note (added 2026-11-15, before the first M1c run)

Two points where the code had to interpret the frozen text:

1. **"Controls within the 95% CI of chance" is implemented one-sided.** The literal
   two-sided reading fails a control that scores 0/500, but a collapsed control is the
   desired outcome (AGENTS.md: "controls collapse to near chance"). So a control passes
   when the lower bound of its Wilson interval is at or below chance: only a control
   significantly *above* chance fails. Wilson score intervals, z = 1.96.
2. **Chance is derived from the data, not hard-coded.** `1 / (distinct answers in the
   test split)`: color 1/8 = 0.125, shape 1/4, count 1/5. Color has 8 options, so a
   hard-coded 1/6 would have been wrong.

The untrained-projector condition is reported as a diagnostic and is deliberately not
one of the controls that the pass criteria minimise over, because the frozen text names
blank, mismatched and no-tokens.

## Consequences
The M1c evaluation is frozen; the G1 report uses these thresholds. The one-sided
reading and the runtime-derived chance rates are part of that frozen definition.

## Evidence
- Main document section 9.4, G1 thresholds. [V]
- M1a test set: 512 color, 512 shape, 500 count. [V]
- M1b training: 2048 steps, 3 seeds. [V]
- `COLORS` has 8 entries and `SHAPES` has 4 (`src/sightline/data/synthetic.py`). [V]
- Chance rates derived at runtime by `tools/run_m1.py:chance_rates`. [V]

## Open items
none