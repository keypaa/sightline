# DECISION-0042: M2a split ratios

Date: 2026-11-18
Status: accepted
Work package: M2a

## Context

The data table in the project document requires train and validation splits for
M2a but does not fix the ratio. The split has to be by hash and reproducible
(`docs/measurement-pitfalls.md`: the M1 splits are ordered by answer, so slicing
the first N is a biased sample).

## Decision

- **80 / 10 / 10** across train, val and test.
- Assignment is a seeded shuffle of the records (`random.Random(seed)`), with the
  seed recorded in the manifest. The same seed and inputs give the same split.
- Splits are disjoint by construction; a test asserts no id appears twice.
- The test split is held out for G2 and is never used for early stopping.

## Rationale

80/10/10 is the conventional default and needs no justification beyond being
unremarkable, which is the point: a split ratio chosen to make a number look
better is a defect. The M2a iteration set is small enough that a 60/20/20 would
cost training data for no benefit.

## Consequences

At 20k conversations: 16,000 train, 2,000 val, 2,000 test.

## Open items

- [U] Whether the test split should be enlarged for G2, where the reproduction
  needs held-out data per task. Closing step: measure the per-task counts in the
  test split once the build runs.