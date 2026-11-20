# Why magenta lags: train versus test

Seed 0 checkpoint, `--debug-logits 12 --filter-answer magenta`, on the A10G via
`modal/run_tool.py`. Magenta scores 0.078 on the signed test run, against
0.80-0.98 for the other seven colours.

## Train split

All 12 sampled magenta examples are answered correctly. The greedy output is
`'magenta'` every time, and the teacher-forced top-1 token is `'m'` at
probability 1.0 — the model emits the first token of the word and then spells it
out. **The class is learned.**

## Test split

1 of 12 correct. The other 11 answers are:

| answer | count |
| --- | --- |
| green | 3 |
| blue | 3 |
| cyan | 2 |
| diamond | 1 |
| human | 1 |
| to | 1 |

Confidence on the wrong answers is often high — `'blue'` at 0.901 and 0.762,
`'green'` at 0.554, `'cyan'` at 0.665.

## Reading it

Magenta is not an unlearned class. The model answers it perfectly on training
images and confidently wrong on unseen ones, and the wrong answers are
concentrated on **blue, cyan and green** — the colours nearest magenta in RGB
space. So the magenta region the projector learns is narrow: it covers the
training images and collapses onto its neighbours on anything new.

That is a generalisation failure with a direction, not random noise, and it
explains the per-class number without needing a capacity story.

The `'diamond'`, `'human'` and `'to'` answers are a separate matter: on a colour
question the model occasionally emits a shape word or a chat token. That is the
same phrasing/format instability seen in the count task, and it is a reason to
treat single-example debug output as indicative rather than definitive.

## A defect in the debug output

The `teacher_forced_top1` column compares only the **first token** against the
whole answer, so `'m'` is scored as wrong when the greedy output is correctly
`'magenta'`. The "teacher-forced top-1 correct: 0/12" line is therefore
meaningless for multi-token answers. It does not affect any evaluation — the
eval path compares full decoded strings — but it makes the debug output
misleading, and it should compare the first token of the answer instead.

## Evidence

- Train: 12/12 greedy correct, first token `'m'` at 1.0. [V]
- Test: 1/12 correct, wrong answers as tabulated. [V]
- Signed test accuracy for magenta: 0.078. [V]

## Open items

- [U] Whether the magenta region widens with more data or a different projector
  init. Owner: implementation agent. Closing step: a magenta-focused training
  run, or a per-class learning curve.
- [U] The debug output's teacher-forced column. Owner: implementation agent.
  Closing step: compare against the answer's first token.