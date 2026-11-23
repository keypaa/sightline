# DECISIONS 0053: M2a blind reader model

Date: 2026-11-23
Status: accepted
Work package: M2a
Owner instruction: use `opencode/hy4-flash-free` for the blind reader.

## Context

`zenno/specs/2026-11-21-m2a-corpus-schemas.md` names `opencode/hy4-flash-free`
for both the verifier and the blind reader: the same model, different jobs, no
shared state, and neither is the annotator (`Qwen/Qwen3.6-27B-FP8`). In the
blind role it receives the question and answer text only — no image.

`src/sightline/m2a/verify.py` instead defaulted to
`kilo/z-ai/glm-5-turbo:free`, an undocumented divergence from the spec. No
DECISIONS record named the blind model. AGENTS.md requires a stop on a source
conflict and a DECISIONS entry for an undeclared choice.

## Decision

1. The blind reader is `opencode/hy4-flash-free`, via the opencode CLI, with no
   image attached.
2. It remains a different *role* from the verifier: separate archived sessions,
   no image, and its rows carry `blind_model` so the two cannot be confused.
3. The verifier stays `opencode/hy4-flash-free` too. This does not put a model
   in both the annotating and judging roles.

## Consequence

The blind rate measures whether the *question text alone* gives away the
answer. It is not a measure of verifier quality. The threshold stays `[U]` in
`DECISIONS/0049` until the distribution is inspected.
