# G1: three-seed verdict on held-out M1 test data

Runs 2026-11-19/20 on an RTX PRO 6000, 2048 steps, micro-batch 16, LR 1e-3,
seeds 0, 1 and 2 with real seeding (DECISIONS 0037). Frozen thresholds:
DECISIONS 0032. Parser and prompts frozen for these runs: DECISIONS 0038.

## Held-out test results

| task | seed 0 | seed 1 | seed 2 | mean | chance | best control |
| --- | --- | --- | --- | --- | --- | --- |
| color | 0.8125 | 0.8320 | 0.7480 | **0.798** | 0.125 | 0.000 |
| shape | 0.8066 | 0.8359 | 0.7891 | **0.811** | 0.25 | 0.000 |
| count | 0.0000 | 0.3420 | 0.0000 | 0.114 | 0.20 | 0.000 |

Vision minus best control, per seed:

| task | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- |
| color | +0.688 | +0.707 | +0.623 |
| shape | +0.557 | +0.586 | +0.539 |
| count | +0.000 | +0.142 | +0.000 |

Every control is 0.000 on every task and seed, `controls_within_chance` is true
throughout, and McNemar p < 1e-114 wherever the model is above chance.

## Verdict

**G1: PASS.**

- At least 2 of 3 tasks clear the 30-point bar on every seed: colour and shape,
  on all three.
- McNemar p < 0.01 on all of those comparisons.
- All three seeds positive.
- All controls at zero, comfortably inside the 95% CI of chance.

The per-seed verdicts are `g1_pass: true` for seeds 0, 1 and 2 independently.

## What is actually established

**The retrofit transmits visual information.** Colour 0.798 and shape 0.811
against chance rates of 0.125 and 0.25, with blank, mismatched and no-tokens all
at zero. The spread across independent initialisations is 3.6 points for colour
and 1.9 for shape, so this is a stable capability rather than a lucky run.

**The controls are behaving.** A blank image, a mismatched image and a text-only
prompt all score zero. If the model were answering from the question or from a
prior, those would sit at their chance rates. They do not.

**Count is not a capability.** 0.114 mean against a 0.20 chance rate, with two
seeds at exactly zero. Its per-answer pattern is erratic rather than low:
seed 1 scored 0.74 / 0.00 / 0.25 / 0.72 / 0.00 for counts 1–5. That is not a
model that cannot count; it is a model whose count responses are keyed to
something other than the number of shapes.

**Count fails for a phrasing reason, not a missing class.** Seed 0's val split
scored counts 1–4 at 0.969 / 0.797 / 0.617 / 0.203 and test scored every count
at 0.000, including count 1. `--debug-logits` shows teacher-forced top-1 at the
answer position is `'The'` at 0.905 — the model writes prose for
`"Count the shapes."` but digits for `"How many shapes do you see?"`. The three
splits use three different templates, so test measures generalisation to an
unseen phrasing.

## Caveats on this pass

- **The parser understates shape.** One debug example produced
  `'diamond\n```json\n{\n  "'` with `diamond` at 0.998, scored wrong. Shape's
  0.811 is therefore a floor. Not fixed here, because changing a frozen scoring
  rule after a run would break comparability (DECISIONS 0038).
- **Magenta is a weak class.** 0.078 / 0.469 / 0.219 across seeds, against
  0.64–1.00 for the other seven colours. The colour average conceals it.
- **Train accuracy is 1.000 on all three tasks.** These numbers are generalisation
  gaps, not fit; do not compare them against the training subset.
- **`metrics.jsonl` for seed 0 contains a stale event** from the superseded 3e-4
  run on its first line. It is preserved deliberately and does not affect the
  evaluation.

## What G1 does not say

G1 asks whether the frozen retrofit uses the image. It does, on colour and shape,
robustly. It does not say the system is a good general-purpose describer: one of
three tasks fails outright, one colour class is near chance, and the count
behaviour is unstable in a way that suggests a shortcut rather than counting.

## Next

1. Re-score all three seeds under a first-line parser and report both numbers.
2. A magenta-only probe to find out why that class lags.
3. A count run with one question template across all splits, to separate
  "cannot count" from "cannot read this phrasing".
4. Only then consider M2a.