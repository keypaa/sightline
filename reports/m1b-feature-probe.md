# M1b: linear probe on frozen K3 features

Run 2026-11-18 on an A10G, `modal/run_tool.py -- probe_features --per-task 256 --steps 600 --shuffle-control`.
Code: `tools/probe_features.py`. Commit `3941435`.

## Question

Do the frozen K3 features contain the answer in a form a simple decoder can
read? If yes, the failure is in the projector-to-reader path. If no, extraction
or pooling is at fault and no training run could have worked.

## Method

For each image the K3 tower produces 256 patch vectors of width 4096. We
mean-pool over the token axis to one 4096-wide vector, standardise it, and fit a
single linear layer on 256 train-split examples with Adam for 600 steps. The
classifier is scored on 256 held-out val-split examples, both with their true
labels and with permuted labels as a leakage control.

Chance rates come from the classes actually present in the sampled subsets.

## Result

| task | classes | chance | train | val | shuffled val |
| --- | --- | --- | --- | --- | --- |
| color | 8 | 0.125 | 1.000 | **1.000** | 0.109 |
| shape | 4 | 0.25 | 1.000 | **1.000** | 0.262 |
| count | 4 | 0.25 | 1.000 | **1.000** | 0.277 |

Encoding took ~103 s; the whole run took 133 s.

## Reading it

A single linear layer over mean-pooled frozen features achieves **100% on held-out
images for all three tasks**. The permuted-label control lands at chance for each
(0.109 vs 0.125, 0.262 vs 0.25, 0.277 vs 0.25), so this is not leakage, duplication
between splits, or a probe reading image identity instead of the label. [V]

Conclusion: **the answers are present in the frozen features, in a form that is
linearly decodable after mean pooling.** The envelope is not empty. The failure
is downstream, in how the projector injects those features into the frozen
reader and how that path is trained.

## Caveat

Mean pooling makes colour unusually easy: the average colour of a synthetic
image is roughly the colour of the shape on it. That shortcut is not available
to the projector, which sees per-patch features. Shape and count at 100% after
pooling are less trivial, but the probe is an upper bound on accessibility, not
a claim about what per-patch decoding requires. [I]

## Secondary finding: two bugs in the probe itself

The first run reported 16 classes and a 6.25% chance rate for every task; the
label-table loop had collected all three tasks' answers. The second run's shuffled
control gave 0.508 for count against a stated 0.25, exposing that `--per-task 256`
took the *first* 256 of an answer-ordered split — only counts 1 and 2. The third
fix replaced a fixed stride with a seeded shuffle after a test showed stride 8
over an 8-periodic list aliases onto a single class. Each bug is recorded in
`docs/measurement-pitfalls.md` and `zenno/memory/findings.md`. [V]

## What remains open

- [U] Why a 25M-parameter projector cannot use information a 4096×8 linear layer
  reads perfectly. Owner: implementation agent. Closing step: a full run at the
  learning rate that converges, with train exact-match logged, not just loss.
- [U] Whether the 256-token splice dilutes the signal. Closing step: compare a
  single pooled visual token against the current 256-token injection.