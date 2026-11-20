# M2a annotator pilot

Three runs, 2026-11-20 and 2026-11-21, `tools/annotate_pilot.py`, annotator
`kilo/nvidia/nemotron-3.5-super-vl:free`, images from the M1 synthetic set.
Records: `experiments/m2a_pilot/` (v1), `experiments/m2a_pilot_v2/` (v2),
`experiments/m2a_pilot_v3/` (v3).

Pass criteria, fixed before v3: parse rate 30/30, grounded >= 0.90, token count
recorded for every example.

## Verdict

**Green.** v3 meets the bar: 30/30 parsed, 27/30 grounded (0.900), tokens
recorded for all, and deterministic on repeat.

## Progression

| | v1 | v2 | v3 |
| --- | --- | --- | --- |
| prompt | original | original | **target must be stated explicitly** |
| parsed | 30/30 | 30/30 | 30/30 |
| grounded | 20/30 | 26/30 | **27/30 (0.900)** |
| mean input tokens | 9,953 | 3,249 | **3,179** |
| token repeat | not measured | 3,118 ×3 | **3,179 ×3** |

Prompt hash for v3: `b6d8d625b4ffc424`.

## Why v3 passes

The v2 prompt asked for grounded turns but allowed an answer to be *implied* by a
description. v3 requires the answer in the first few words and requires counts as
number words. That moved grounding from 0.867 to 0.900.

## The three remaining misses, stated plainly

- **shape 14** — expected `square`; the annotator answers colour, count and
  position, never the shape.
- **count 20, 21** — expected `1`; the annotator describes `'a solid red circle'`
  without ever stating a count.

0.900 is the bar, not comfortably above it. Two more misses would have failed the
run, and the count task is the weaker of the three (7/10 in v2, 8/10 in v3). If
M2a data quality depends on counts, this prompt is not good enough and needs
another iteration.

## Cost, against section 16 item 8

Section 16 caps the annotator at about 200M tokens a day and forbids costs in the
hundreds of dollars.

- **3,179 tokens per conversation** measured, so 40k conversations is **127M
  tokens** — inside the 200M daily cap, with about 1.6× headroom.
- The 131k reproduction run would be **~417M tokens**, which is **over** the cap
  and would need two days or a higher tier.
- **Token counts vary by provider, not by image.** The same 30 images cost
  16,126 each in v1 and 3,179 in v3, deterministically within each run. At the
  v1 rate, 40k conversations would be 645M tokens and would breach the cap.
  This is the largest uncertainty in the estimate. [U]
- **Pricing is [U].** I have not verified kilo's rates for this model and will
  not invent them. Closing step: read the provider's pricing page and multiply by
  the measured 127M.

## Open items

- [U] Why the same images cost 16,126 tokens in v1 and 3,179 in v3. Owner:
  implementation agent. Closing step: two pilots in one session, compared
  per-image.
- [U] Provider pricing for the annotator. Closing step: read the pricing page,
  multiply by 127M.
- [U] The prompt does not reliably elicit counts (8/10). Closing step: another
  prompt iteration if count data quality matters for M2a.