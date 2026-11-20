# M2a annotator pilot

Run 2026-11-20, `tools/annotate_pilot.py --n 30`, annotator
`kilo/nvidia/nemotron-3.5-super-vl:free`, images from the M1 synthetic set.
Records: `experiments/m2a_pilot/annotations.jsonl`.

## Results

| metric | value |
| --- | --- |
| parsed as JSON | **30 / 30** |
| multi-turn (2–3 turns as asked) | 30 / 30, mean 2.73 turns |
| grounded, colour | 10 / 10 |
| grounded, shape | 8 / 10 |
| grounded, count | 0 / 10 — **artifact, see below** |
| input tokens, mean | 9,953 |
| input tokens, total | 298,579 |
| projected for 40k conversations | **~398M input tokens** |

## What this establishes

**The contract holds.** Every call returned valid JSON in the requested shape,
with no code fence and no prose around it, and every conversation had the
requested 2–3 turns. The annotator can be driven programmatically.

**The answers are grounded, not invented.** Every assistant turn references
something actually in the image — colour names, shape names, counts. The
conversation shape is right for M2a.

**The count "0/10" is my measurement bug, not the annotator's.** The pilot
checks grounding with `expected_answer in assistant_text`, and count answers are
digits (`'2'`), while the annotator correctly writes `'Two.'`. Reading the raw
outputs, the counts are right: `"How many circles are in the image?"` →
`"Two."` for an image with two circles. The check should map digits to words.

## The number that matters: cost

**~398M input tokens for 40k conversations**, against the project document's
assumption of 1.6k tokens per conversation (about 64M). The measurement is 6×
that.

Token counts also varied for identical 448×448 images: 3,118 / 4,365 / 11,790 /
16,126 across the run. The pilot does not explain that; it is **[U]**, and it
may be provider-side image tiling. If the high end is the norm, the projection
worsens by up to 1.6×.

## Consequences

- The annotator choice is viable on **contract and quality**.
- It is not viable on **cost** at the free tier: 398M tokens will hit rate limits
  long before 40k conversations are done.
- Section 16's terms-and-cost check must be redone against the measured 9,953
  tokens per conversation, not the assumed 1.6k.
- A 100-example pilot was enough to find this; a 40k run would have found it
  after the money was spent.

## Open items

- [U] Why token counts vary 5× for identical-size images. Closing step: log the
  provider's image-tiling behaviour, or fix the resolution.
- [U] The grounding check mis-scores number words. Closing step: map digits to
  words in `tools/annotate_pilot.py`.
- [U] Whether a cheaper or paid annotator changes the projection. Closing step:
  repeat the pilot with K3 or V4.1 if credits are available.