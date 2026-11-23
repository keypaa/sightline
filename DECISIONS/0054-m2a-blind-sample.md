# DECISIONS 0054: M2a blind-reader sample

Date: 2026-11-23
Status: accepted
Work package: M2a

## Decision

1. Blind **500 real claims**, `--sample 500 --sample-seed 0`, so the blind set is
   the same draw `judge` used for its 500 real claims. The two layers can then
   be paired claim-by-claim instead of compared as two unrelated rates.
2. The blind reader follows `DECISIONS/0053` (`opencode/hy4-flash-free`, no image).
3. `blind` now supports `--sample` and resumes: rows with `error` are dropped
   and only claims without a successful answer are re-asked. This matters
   because the verifier quota tripped once during the 20+20 probe. **[V]**
4. The blind-answerability threshold stays `[U]` per `DECISIONS/0049`. We inspect
   the distribution first; setting a bar before seeing it would be a guess.

## Why paired

A blind rate over a different set of questions would measure question
difficulty as much as answer leakage. Using the judged 500 controls for that
and makes the vision-minus-blind comparison meaningful.
