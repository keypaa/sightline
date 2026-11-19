# M1b seed 0: held-out evaluation at LR 1e-3

Run 2026-11-19 on an RTX PRO 6000, 2048 steps, micro-batch 16, `--lr 1e-3`,
seed 0. Record: `experiments/m1b_train/seed0/`.
Checkpoints: `artifacts/m1b/seed0/projector.pt`.

This supersedes the 2026-11-15 seed-0 run, which was trained at 3e-4 and failed
G1 everywhere. That run was undertrained, not architectural.

## Held-out results

| task | split | correct | chance | best control | vision − blind | McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| color | val | 0.859 | 0.125 | 0.000 | **+0.734** | 7.0e-133 |
| color | test | 0.855 | 0.125 | 0.000 | **+0.730** | 2.8e-132 |
| shape | val | 0.266 | 0.25 | 0.000 | +0.016 | 2.3e-41 |
| shape | test | 0.260 | 0.25 | 0.000 | +0.010 | 1.8e-40 |
| count | val | 0.953 | 0.25 | 0.000 | **+0.703** | 2.5e-147 |
| count | test | 0.318 | 0.20 | 0.000 | **+0.118** | 2.7e-48 |

Controls are 0.000 everywhere and `controls_within_chance` is true, so the
criteria on controls are met. Train-split accuracy was 1.000 on all three, so
these numbers are real generalisation gaps, not fit.

## G1 on the test split

`g1_pass: true`. Colour clears the 30-point bar by a wide margin, count clears
it by 1.8 points, shape fails at +1.0. Two of three tasks pass, which is the
frozen criterion.

## Reading it

**Colour is genuinely learned.** 85.5% on unseen images against 12.5% chance,
with all controls at zero. This is the first real evidence in the project that
the frozen retrofit transmits visual information. It is not a small-sample
effect: 512 colour test examples, bootstrap 95% interval [0.824, 0.885].

**Shape is not learned at all.** 26.0% on test is chance, despite the model
scoring 1.000 on the training subset. Shape is pure memorisation. The probe
showed shape is linearly decodable from the features, so the information is
available and the projector is not using it.

**Count generalises on val but not on test.** 95.3% on val against 31.8% on
test. The likely reason is visible in the data generator: train and val use
counts 1–4, while test uses 1–5. Count 5 is never trained. But that alone
cannot explain 31.8%: if counts 1–4 held at their val rate, the test score
would be near 0.76. So counts 1–4 also degrade on the test split, which needs
its own explanation. [U]

**Count's test margin is thin.** +31.8 points clears the threshold by 1.8. A
small change in the test composition or the parsing could flip it. It should not
be treated as a comfortable pass.

## Open items

- [U] Why count degrades from 0.953 on val to 0.318 on test, beyond the unseen
  count 5. Owner: implementation agent. Closing step: per-count accuracy on both
  splits.
- [U] Why shape generalises not at all while colour does. Owner: implementation
  agent. Closing step: compare a shape-only training run against colour-only.
- [U] Whether count 5 belongs in the test split at all, given train and val stop
  at 4. That is a data design decision and needs a DECISIONS entry.
- [U] Whether seeds 1 and 2 reproduce this at 1e-3. Closing step: two more runs.

## Next

1. Seeds 1 and 2 at `--lr 1e-3 --micro-batch 16`, ~1.1 h each, then the G1
   report over all three.
2. Per-count accuracy on val and test to explain the count gap.
3. Shape-only control to localise the shape failure.