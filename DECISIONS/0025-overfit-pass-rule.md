# DECISION-0025: 32-example overfit pass rule and control thresholds

Date: 2026-11-15
Status: accepted (frozen before the first run)
Work package: WP10

## Context
G0 requires a 32-example overfit to reach 99% exact match within 500 steps, and the overfit counts only if the blank, shuffled and swapped controls collapse to near chance. The thresholds must be frozen before the first run.

## Decision
- **Overfit pass:** 32/32 (100%, which is the only way to reach >=99% on 32 examples) exact match on the training set within 500 steps, on at least 2 of 3 seeds (0, 1, 2).
- **Exact match:** greedy decoding, max 8 new tokens, answer parsed as `strip().lower().strip(" .!")` and compared to the ground-truth color.
- **Controls:** blank, shuffled (cyclic image shift) and swapped (offset-16 image shift) exact match each <= 20% (<= 6/32). Chance is 12.5%; 20% is within the 95% CI of chance (about 1.1% to 23.9% for 32 trials).
- **Vision-minus-blind:** real exact match minus blank exact match >= 79 points (99% - 20%).
- **Answer loss gap:** report the mean per-token answer loss with the blank image minus the real image on the 32 examples; no pass threshold (the controls are the pass criterion).
- **Overfit hyperparameters:** LR 1e-3, warmup 0, 500 steps, micro-batch 8 (4 gradient-accumulation steps per optimizer step), weight decay 0.0, betas (0.9, 0.95), gradient clip 1.0, bf16 autocast, fp32 projector and optimizer state.
- **Seeds:** 0, 1, 2; the pass is at least two of three.

## Evidence
- Chance = 1/8 because the 32 answers are balanced 4 per color (DECISION 0024). [V]
- The 95% CI of chance for 32 trials is about 1.1% to 23.9%; 20% is within it. [I, arithmetic]
- The 99% threshold on 32 examples requires 32/32. [I, arithmetic]

## Consequences
The overfit run on Modal T4 uses these thresholds. If the overfit fails, the debugging ladder in the M0 breakdown applies before changing hyperparameters; any change to these thresholds needs a new DECISIONS entry before the next run.

## Open items
none