# DECISION-0006: Memorization controls

Date: 2026-11-15
Status: accepted
Work package: WP0

## Context
Low loss or high exact match on a small set is not evidence of vision. The 32-example overfit counts for G0 only if the controls collapse to near chance.

## Options considered
1. Rely on exact match alone.
2. Balanced answers, held-out compositions, and blank/shuffled/swapped controls with frozen thresholds.
3. Hold-out only, without counterfactual controls.

## Decision
Balance answers given the question so text alone scores at chance. For the 32-example overfit, use one question template whose answers vary only by image. After the overfit, evaluate blank, shuffled and swapped images; the overfit counts only if those controls collapse to near chance. Report the per-token answer loss gap between the real and blank images on held-out compositions.

## Evidence
- M0 breakdown, 2b decision 6 and the memorization controls list. [V]
- Main document section 9.1: correct, blank, mismatched, no-tokens and untrained-projector conditions. [V]

## Consequences
Question text never uniquely identifies the answer. The control thresholds are frozen in DECISIONS/ before the WP10 run.

## Open items
- [U] Numeric control thresholds (what "near chance" means for each task). Owner: implementation agent. Closing step: a DECISIONS file before the first WP10 run.