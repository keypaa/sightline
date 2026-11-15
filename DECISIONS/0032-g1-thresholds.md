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

## Evidence
- Main document section 9.4, G1 thresholds. [V]
- M1a test set: 512 color, 512 shape, 500 count. [V]
- M1b training: 2048 steps, 3 seeds. [V]

## Consequences
The M1c evaluation is frozen; the G1 report uses these thresholds.

## Open items
none