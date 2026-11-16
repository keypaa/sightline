# M1b seed 0: postmortem

Run: 2048 steps, micro-batch 16, RTX PRO 6000, 2026-11-15.
Record: `experiments/m1b_train/seed0/`.

## Outcome

| metric | value |
| --- | --- |
| elapsed (train + eval) | 5186 s (1.44 h) |
| peak GPU memory | 32361.2 MB |
| peak RSS | 9526.4 MB |
| final loss | 5.02 |
| loss every 100 steps | 18.02, 20.35, 21.58, 13.36, 13.34, 14.46, 10.70, 10.67, 4.07, 6.23, 6.84, 0.80, 2.41, 4.02, 0.45, 1.40, 3.26, 0.35, 1.18, 3.09, 0.33 |
| G1 | **fail** |

Held-out test, exact match: color 1.4% (chance 12.5%), shape 0.0% (chance 25%),
count 0.8% (chance 20%). Colour's best control, the mismatched image, scored
12.1% — chance.

## What the debug run showed

`--eval-split train --debug-generations 20`, sampled evenly:

| task | correct on train |
| --- | --- |
| color | 0 / 7 |
| shape | 1 / 7 |
| count | 5 / 6 |

So the model does not reproduce training answers for colour or shape. That rules
out memorisation, which would score near 100% on train, and it rules out a
generalisation gap, because there is no fit to generalise from. Count reached
5/6, so information does pass through the frozen reader and the projector can
move the output distribution.

## Reading the loss

The loss fell from 18 to a final 5.02, but it oscillates between 0.33 and 4.0
through the last thousand steps. A loss near 2 nats is a confused model, not a
converged one; the low samples are lucky batches, not a plateau. **Training did
not converge.**

## What is ruled out

- **The vision path is broken.** Answers reference image content, vary with the
  image, and count works on train.
- **The decode is broken.** `max_new_tokens=8` produces real words, not empty
  strings or think blocks.
- **Memorisation.** Train accuracy is 0/7 and 1/7, not high.
- **An evaluation alignment bug.** The eval path shares its prompt, splice and
  feature construction with training.

## Leading hypothesis

Gradients must reach the projector through 28 frozen bf16 layers. If that signal
is small, the projector barely moves and a low LR with random init leaves it near
its initial behaviour — which looks exactly like "confident, plausible, wrong".
This is a hypothesis. It is testable by logging the projector's gradient norm per
step, which `clip_grad_norm_` already computes.

## The gradient-norm check (2026-11-16, A10G, 50 steps)

| metric | value |
| --- | --- |
| mean gradient norm, last 25 steps | **829.4** |
| max gradient norm, last 25 steps | 1264.4 |
| loss at step 10 | 22.29 |
| loss at step 50 | 16.07 |
| mean loss, last 25 steps | 20.56 |
| step time | 9.69 s |

The projector's gradient is not weak. It is enormous, and `grad_clip=1.0` rescales
it by roughly 800–1000× on every step. So the hypothesis "the signal does not
reach the projector" is false, and an LR sweep is not the highest-value next
experiment.

The more striking number is the loss: 20.6 mean against `ln(151936) = 11.9` for a
uniform guess. A randomly initialised projector spliced into 256 image positions
makes the reader produce logits far more extreme than a uniform prior. That is a
plausible root cause for both the high gradients and the failed fine-tuning: the
output scale of the projector is wrong at init, not the learning rate.

## Next

1. Scale the projector output to the reader's embedding scale (DECISIONS entry
   required), then re-run this 50-step check. Success criteria: initial loss near
   or below 11.9 and a gradient norm in single digits.
2. Only if that fails, sweep LR.
3. Re-run 2048 steps once the loss converges rather than oscillates.