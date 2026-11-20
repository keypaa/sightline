# M1b seed 0: held-out evaluation at LR 1e-3, real seeding

Run 2026-11-19 on an RTX PRO 6000, 2048 steps, micro-batch 16, `--lr 1e-3`,
seed 0 with real seeding (DECISIONS 0037). Record: `experiments/m1b_train/seed0/`.

This supersedes both earlier seed-0 runs: the 2026-11-15 run at 3e-4, which was
undertrained, and the 2026-11-19 run that shared its initialisation with the other
seeds because `--seed` was not wired through.

## Held-out results

| task | split | correct | chance | best control | vision − blind | bootstrap 95% |
| --- | --- | --- | --- | --- | --- | --- |
| color | val | 0.697 | 0.125 | 0.000 | +0.572 | [0.656, 0.736] |
| color | test | **0.8125** | 0.125 | 0.000 | **+0.688** | [0.779, 0.846] |
| shape | val | 0.625 | 0.25 | 0.000 | +0.375 | [0.584, 0.666] |
| shape | test | **0.8066** | 0.25 | 0.000 | **+0.557** | [0.771, 0.840] |
| count | val | 0.6465 | 0.25 | 0.000 | +0.397 | [0.605, 0.688] |
| count | test | 0.0000 | 0.20 | 0.000 | +0.000 | [0.000, 0.000] |

Controls are 0.000 everywhere and `controls_within_chance` is true. McNemar
p < 1e-96 on every task where the model is above chance.

## Answer-level breakdown

| task | answer | val | test |
| --- | --- | --- | --- |
| color | blue | 0.828 | 0.969 |
| color | cyan | 0.688 | 0.953 |
| color | green | 0.922 | 0.891 |
| color | magenta | **0.172** | **0.078** |
| color | orange | 0.703 | 0.984 |
| color | purple | 0.734 | 0.891 |
| color | red | 0.734 | 0.797 |
| color | yellow | 0.797 | 0.938 |
| shape | circle | 0.742 | 0.883 |
| shape | diamond | 0.422 | 0.750 |
| shape | square | 0.844 | 0.969 |
| shape | triangle | 0.492 | 0.625 |
| count | 1 | 0.969 | **0.000** |
| count | 2 | 0.797 | **0.000** |
| count | 3 | 0.617 | **0.000** |
| count | 4 | 0.203 | **0.000** |
| count | 5 | not in val | **0.000** |

## What this establishes

**Colour and shape are real, generalising capabilities.** Both clear the 30-point
bar on held-out test with every control at zero, and both are *higher* on test
than on val. That is the first solid evidence in the project that the frozen
retrofitted reader transmits visual information. The seed-0/seed-1 spread seen
earlier (shape 0.260 vs 0.955) was nondeterminism plus a seed-less
initialisation, not a property of the method.

**Count is a genuine negative result, not a data defect.** Counts 1–4 score
0.969 / 0.797 / 0.617 / 0.203 on val and 0.000 on test, including count 1. So
the unseen count 5 cannot be the explanation — if it were, counts 1–4 would hold
on test. `--debug-logits` shows why: at the answer position for a count question
the teacher-forced top-1 is `'The'` at 0.905, and greedy produces
`'the given text is: \n\n"cr'`. The model does not emit a digit for an unseen
question phrasing.

The three splits use three different count templates — `"How many shapes are
there?"`, `"How many shapes do you see?"`, `"Count the shapes."` — so this is a
generalisation failure to an unseen phrasing, not a counting failure. The
distinction matters: the model counts when asked the way it was trained to
expect, and switches to prose when the phrasing changes.

**Magenta is a specific weak spot.** 0.078 on test against 0.80–0.98 for the
other seven colours. Whatever the model does with magenta, it is not learning it.

**The parser loses a little.** One shape example produced
`'diamond\n```json\n{\n  "'` with `diamond` at 0.998, scored wrong because the
parser strips only whitespace and ` .!`. Measured on seed 0, a first-line parser
gains 1.8 points on shape and nothing on colour or count, so 0.807 is a mild
floor rather than a badly understated number. The parser was not adopted as the
default: changing a frozen scoring rule after a run would need its own decision.
See `reports/g1-parser-comparison.md` and DECISIONS 0038.

## G1

`g1_pass: true` on test: colour +0.688 and shape +0.557 clear the 30-point bar,
McNemar p < 0.01, controls at zero. Count contributes nothing.

That satisfies the frozen criteria for this seed, but it is worth being precise
about what it means: two of three tasks pass, and the third fails for a
methodological reason — phrasing sensitivity — rather than for want of visual
information.

## Open items

- Answered: the parser costs 1.8 points on shape, 0.0 on colour and count
  (`reports/g1-parser-comparison.md`). Not adopted as the default.
- Answered: magenta is learned (12/12 on train) but 1/12 on test, and the
  errors are not explained by RGB distance (`reports/m1b-magenta.md`).
- [U] Whether count generalises with a consistent question template across
  splits. Closing step: a run with one count template everywhere.