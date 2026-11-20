# Parser comparison and cross-hardware variance

Runs 2026-11-19/20. Seed 0 checkpoint evaluated on the held-out test split
under both answer parsers, on an A10G (`modal/eval_recheck.py`, throwaway
harness), against the signed run on the RTX PRO 6000.

## Results

| task | PRO 6000, exact (signed) | A10G, exact | A10G, first-line |
| --- | --- | --- | --- |
| color | 0.8125 | 0.8105 | 0.8105 |
| shape | 0.8066 | 0.7988 | **0.8164** |
| count | 0.0000 | 0.0020 | 0.0020 |

## What this shows

**The parser changes only shape, and by 1.8 points.** Colour is identical to
four decimal places between parsers (0.810546875 both), and count is identical.
Shape goes 0.7988 → 0.8164, which matches the hand-found case: a correct
`'diamond'` followed by a markdown code fence, scored wrong by exact match.

**`exact` did not reproduce the signed numbers exactly.** Colour 0.8105 vs
0.8125, shape 0.7988 vs 0.8066, count 0.002 vs 0.000. The differences are
0.2–0.8 points, and colour matching to four decimals shows this is not a code
drift — it is bf16 numerics differing between sm_86 and sm_120, flipping a
handful of near-tie argmax decisions across 512 examples.

**That makes the hardware noise the same order as the parser effect.** The
parser buys 1.76 points on shape; the card changes the answer by up to 0.78.
The parser effect is real and correctly signed, but it is small, and it does not
change any verdict.

## Consequences

- The signed G1 numbers stand. Shape's 0.8066 is a fair measurement, not a
  badly understated one; the parser is worth about 2 points and nothing more.
- **`exact` remains the default and the recorded rule.** Switching to
  `first-line` would buy ~2 points on one task at the cost of a new frozen
  definition, which is not a good trade.
- **Do not re-run seeds 1 and 2 for the parser on the A10G.** The effect is
  smaller than the cross-hardware noise, so the comparison would be measuring
  the card, not the parser. If it is worth doing, it is worth doing on the
  PRO 6000.
- Recorded for future calibration, as **[I]** from a single unreplicated
  comparison: bf16 evaluation at n=512 moved by up to 0.8 points between sm_86
  and sm_120. Treat it as an order-of-magnitude warning, not a constant:
  **do not compare small accuracy deltas across cards.**

## Evidence

- Seed 0 checkpoint sha256 `99cc1cd7…`, identical file on both machines. [V]
- A10G run: `modal/eval_recheck.py --seeds 0 --parsers exact,first-line`, exit 0. [V]
- Signed run: `reports/g1.md`, RTX PRO 6000. [V]
- The sm_86/sm_120 attribution and the size of the noise floor are inference from
  one comparison, not a controlled measurement. [I]

## Open items

- [U] Whether the parser is worth adopting for M2, where answers will be longer
  and trailing content more likely. Owner: implementation agent. Closing step: a
  PRO 6000 comparison once M2 data exists.