# DECISION-0038: keep the frozen parser and prompt for the G1 runs

Date: 2026-11-19
Status: accepted
Work package: M1c

## Context

Seed 0 at LR 1e-3 (with real seeding, DECISIONS 0037) produced, on held-out test:

| task | correct | chance | best control |
| --- | --- | --- | --- |
| color | 0.8125 | 0.125 | 0.000 |
| shape | 0.8066 | 0.25 | 0.000 |
| count | 0.0000 | 0.20 | 0.000 |

Two separate weaknesses surfaced while explaining the count result:

1. **Count does not generalise to an unseen question phrasing.** At the answer
   position for a test count question, teacher-forced top-1 is `'The'` at 0.905
   and greedy produces `'the given text is: \n\n"cr'`. The model never emits a
   digit. Train, val and test use different templates by design; val
   (`"How many shapes do you see?"`) scores 0.646, test (`"Count the shapes."`)
  scores 0.000. So the capability is phrasing-specific.
2. **The exact-match parser loses some correct shape answers.** One debug example
  produced `'diamond\n```json\n{\n  "'` with `diamond` at 0.998. The parser
  strips only whitespace and ` .!`, so trailing markdown scores a correct answer
  as wrong.

## Decision

- **Do not change the data split.** The earlier hypothesis that the unseen count 5
  explained the count failure is refuted: counts 1-4 score 0.969, 0.797, 0.617,
  0.203 on val and 0.000 on test. Removing count 5 would not have fixed it.
- **Do not change the parser or the question templates for these runs.** Seed 0
  has already been scored under the frozen rules in DECISIONS 0032. Changing the
  parser now would make seed 0 incomparable with seeds 1 and 2, and loosening a
  frozen criterion after seeing a failing result is exactly what the evidence
  rules forbid.
- The count result stands as a **negative finding about the method**, not as a
  data defect: the retrofit learns phrasing-specific behaviour.
- The parser weakness is recorded as an open item. If it is addressed, all three
  seeds are re-scored under the revised parser, with the revision documented.

## Consequences

G1 for these runs is judged on colour and shape, both of which clear the 30-point
bar on test with all controls at zero. Count is a known failure of the method as
run.

## Evidence

- Seed 0 test report with `answer_breakdown` (2026-11-19). [V]
- Seed 0 val report with `answer_breakdown` (2026-11-19). [V]
- `--debug-logits 8` on test: count teacher-forced top-1 `'The'` at 0.905. [V]
- `COLORS` has 8 entries, `SHAPES` has 4; magenta 0.078 on test. [V]

## Open items

- [U] How many shape answers the parser loses to trailing content. **Answered**
  (2026-11-20): 1.8 points on shape, 0.0 on colour and count, measured on seed 0
  with the identical checkpoint on two cards. `reports/g1-parser-comparison.md`.
  The parser is not adopted as the default; `exact` stands.
- [U] Why magenta is at 0.078 when the other seven colours sit at 0.80-0.98.
  Owner: implementation agent. Closing step: a magenta-only probe.