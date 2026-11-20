# M2a annotator pilot

Two runs, 2026-11-20 and 2026-11-21, `tools/annotate_pilot.py`, annotator
`kilo/nvidia/nemotron-3.5-super-vl:free`, images from the M1 synthetic set.
Records: `experiments/m2a_pilot/` (v1) and `experiments/m2a_pilot_v2/` (v2).

Pass criteria, fixed before the runs: parse rate 30/30, grounded >= 0.90, token
count recorded for every example.

## Verdict

**Not green.** Parse rate and token logging pass; grounding does not.

## v1 (before the grounding fix)

| metric | value |
| --- | --- |
| parsed as JSON | 30 / 30 |
| multi-turn | 30 / 30, mean 2.73 turns |
| grounded | 20 / 30 (0.667) — but the count check was broken |
| input tokens, mean | 9,953 |
| projected for 40k | ~398M |

## v2 (grounding fixed, determinism measured)

| metric | value |
| --- | --- |
| parsed as JSON | **30 / 30** |
| multi-turn | 30 / 30, mean 2.63 turns |
| grounded | **26 / 30 = 0.867** (bar 0.90) |
| input tokens, mean | **3,249** |
| projected for 40k | **~130M** |
| token repeat on one image | 3,118 / 3,118 / 3,118 — deterministic |

Grounding by task: colour 10/10, shape 9/10, count 7/10.

## Why it is not green

The four misses are real, not measurement errors:

- **shape 19**: expected `square`, the annotator answered `'Purple.'` and never
  mentions the shape — it answered a colour question instead.
- **count 20, 21, 22**: expected `1`; the annotator says `'A red circle.'` and
  `'Red.'`, which implies one without ever stating a count. For a conversation
  meant to teach the target, the target is absent.

So on 13% of images the annotator produces a plausible conversation that does not
cover the attribute being asked about. That is a **prompt** problem: the prompt
asks for grounded turns but does not require the target attribute to appear.

## The cost number is unstable, and that is the bigger finding

The same 30 images, same annotator, two runs:

| | v1 | v2 |
| --- | --- | --- |
| input tokens per image | 3,118 / 4,365 / 11,790 / **16,126** | **3,118** (one 7,043 outlier) |
| mean | 9,953 | 3,249 |

Within a run the count is deterministic (3,118 three times). Across runs, the
same image cost 16,126 and then 3,118. The provider's image tokenisation is
therefore **provider-side and variable**, not a property of the image.

That means any cost projection carries a 3× uncertainty: **130M to 398M input
tokens for 40k conversations**. Section 16's cost check cannot be written until
that is understood, which is why I am not proceeding to it.

## Consequences

- The annotator is viable on contract (30/30 JSON, 2.6 turns) and adequate on
  quality (0.867), but not yet green on the pre-declared bar.
- The fix is a prompt change, not a model change: require the target attribute
  to be stated explicitly in an assistant turn.
- The cost projection is unusable until the provider's tokenisation variance is
  explained.

## Open items

- [U] Why the same image costs 16,126 tokens in one run and 3,118 in another.
  Owner: implementation agent. Closing step: repeat the pilot twice in one
  session and compare per-image counts.
- [U] The prompt does not force the target attribute into the conversation.
  Closing step: add the requirement, re-run, require grounded >= 0.90.
- [U] Section 16's cost check. Blocked on the two above.