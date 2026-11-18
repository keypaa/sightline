# DECISION-0036: single-token visual injection (test-only ablation)

Date: 2026-11-18
Status: accepted
Work package: M1b

## Context

A linear probe on mean-pooled frozen K3 features reads colour, shape and count
at 100% on held-out images, with shuffled-label controls at chance
(`reports/m1b-feature-probe.md`). The answers are therefore present in the
features. Yet the 25M-parameter projector cannot use them: a 300-step run at
LR 1e-3 reached a final loss of 0.023 but only 10/20 teacher-forced top-1
correct on train, with colour 2/7 and shape 2/7.

One candidate mechanism is dilution: the projector produces 256 visual vectors
that replace 256 `<|image_pad|>` positions, and the frozen reader must find the
signal across all of them. The linear probe succeeded by collapsing those same
256 vectors into one.

## Decision

- Add a **test-only** `--visual-tokens 1` mode that mean-pools the K3 features
  into a single 4096-wide vector and splices it into one placeholder position.
  The shipped `--visual-tokens 256` behaviour is unchanged and remains the
  default.
- The placeholder count and pooling are carried in one `VisualSpec` object. No
  prompt, splice or run-record field hard-codes 256 any more.
- The ablation is evaluated at 300 steps, LR 1e-3, on the train split with
  `--debug-logits`, then compared with the existing 256-token result at the same
  settings (teacher-forced top-1 10/20; colour 2/7, shape 2/7, count 6/6).

## Interpretation

- **Positive** (colour and shape clearly above 2/7): the 256-token splice is a
  bottleneck. The 2048-step confirmation run is then redesigned around
  single-token injection, with a further DECISIONS entry.
- **Negative** (no clear improvement): the splice is not the bottleneck; the
  default 256-token design stays, and the 2048-step confirmation runs as-is at
  LR 1e-3. This is only suggestive, since one token may be too narrow a channel
  to carry three tasks at once.

This is an ablation, not a change to the frozen design in DECISIONS 0027. The
canvas stays 448×448 and the tower stays unmodified.

## Evidence

- Probe: 100% held-out on all three tasks, shuffles at chance. [V]
- 300-step, LR 1e-3, 256-token result: loss 0.023, teacher-forced 10/20. [V]
- `VisualSpec` unit tests pin the placeholder count and pooled shape. [V]

## Open items

- [U] Whether the single-token channel can carry all three tasks. Closing step:
  this ablation.
- [U] If positive, how the 2048-step run and the G1 evaluation change. Closing
  step: a new DECISIONS entry before that run.