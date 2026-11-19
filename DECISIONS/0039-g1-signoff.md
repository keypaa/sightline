# DECISION-0039: G1 sign-off

Date: 2026-11-19
Status: accepted
Work package: M1c

## Context

G1 requires paired evidence that the frozen retrofit uses the image on held-out
synthetic tasks. The thresholds were frozen in DECISIONS 0032 before the first
run. Three seeds were trained at LR 1e-3 with real seeding (DECISIONS 0037),
2048 steps, micro-batch 16, and evaluated on the held-out test split with the
parser and prompts frozen (DECISIONS 0038).

## Evidence

- Report: `reports/g1.md`, generated from the run records by
  `tools/g1_report.py --split test`.
- Records: `experiments/m1b_train/seed{0,1,2}/` with `run.json` and
  `metrics.jsonl`.
- Colour: 0.812 / 0.832 / 0.748 against chance 0.125. [V]
- Shape: 0.807 / 0.836 / 0.789 against chance 0.25. [V]
- Count: 0.000 / 0.342 / 0.000 against chance 0.20. [V]
- Controls: blank, mismatched and no-tokens all 0.000 on every task and seed;
  `controls_within_chance` true throughout. [V]
- McNemar p < 1e-114 on every task above chance; bootstrap 95% lower bound
  > 0 on every one of those comparisons. [V]
- All three seeds pass independently: colour and shape each clear the 30-point
  bar on all three seeds. [V]
- Checkpoints (gitignored, held outside the repo) at step 2048 with projector,
  optimizer, scheduler, RNG and step: [V]
  - seed0 `99cc1cd75e6efb0c8abf8748baf607ce63e2c03742544fa41239a97a5a9b1d8f`
  - seed1 `6d418c3201f9641f3b0736258d364202ec2123e21ef04dce6ed478ff5ab66591`
  - seed2 `abb047d6fdd8c180217993c8c97ea97164b8c9819bd3a2563bd01d416a10eb0f`

## Decision

G1 passes on the frozen criteria: at least 2 of 3 tasks clear 30 points, McNemar
p < 0.01, all three seeds positive, and all controls at chance.

Approved by Keylhan Paumard--André on 19/11/2026.

## What this does and does not establish

It establishes that the frozen retrofit transmits visual information: colour and
shape are read from held-out images at roughly 0.8 against chance rates of 0.125
and 0.25, with the blind controls at zero. The spread across independent
initialisations is 3.6 points for colour and 1.9 for shape, so it is stable.

It does **not** establish a general-purpose describer. One of three tasks fails
outright, one colour class is near chance, and the count behaviour is erratic
rather than merely low. Details and the reasoning are in `reports/m1b-g1.md`.

## Consequences

M1 is complete. M2a may start, with the open items below as its first work
packages.

## Open items

- [U] The exact-match parser loses correct shape answers to trailing markdown
  (`'diamond\n```json\n{\n  "'` with diamond at 0.998). Shape's 0.807 is a
  floor. Owner: implementation agent. Closing step: re-score all three seeds
  under a first-line parser and report both numbers.
- [U] Magenta sits at 0.078 / 0.469 / 0.219 while the other seven colours reach
  0.64-1.00. Owner: implementation agent. Closing step: a magenta-only probe.
- [U] Count fails on an unseen question phrasing while scoring 0.969/0.797/0.617/
  0.203 on val for counts 1-4. Owner: implementation agent. Closing step: one
  count template across all splits.
- [U] `experiments/m1b_train/seed0/metrics.jsonl` holds a stale event from the
  superseded 3e-4 run on its first line. Owner: implementation agent. Closing
  step: note it in the report, or move it to `metrics.3e-4.jsonl`.
