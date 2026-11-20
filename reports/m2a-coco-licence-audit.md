# COCO 2017 licence audit

Run 2026-11-18, `tools/build_m2a.py index` and `filter` on
`annotations/instances_train2017.json` (118,287 images). No GPU was used.

## Result

| | count | share |
| --- | --- | --- |
| images in the file | 118,287 | 100% |
| **kept** | **30,836** | **26.1%** |
| dropped | 87,451 | 73.9% |

Drop reasons:

| reason | count | share of drops |
| --- | --- | --- |
| non-commercial | 81,459 | 93.1% |
| no-derivatives | 5,992 | 6.9% |

Licences among the survivors:

| licence | count |
| --- | --- |
| Attribution License (CC BY 2.0) | 19,470 |
| Attribution-ShareAlike License (CC BY-SA 2.0) | 10,901 |
| No known copyright restrictions | 465 |

## What this means for M2a

**COCO alone supports the iteration set, but not the reproduction run.**

- The 20k iteration target fits: 30,836 usable images is enough for one
  conversation each, with about 10k spare.
- The 131k G2 reproduction does **not** fit on one conversation per image. It
  would need either roughly four conversations per image, or additional sources.

**The headline licence was misleading by a factor of four.** COCO's card says
CC BY 4.0; that covers the annotations. Three quarters of the photographs are
non-commercial or no-derivatives. A pipeline that trusted the card would have
ingested 118k images it had no right to use.

## After DECISIONS 0043 (non-commercial allowed)

With the policy changed to drop only no-derivatives and missing:

| | count | share |
| --- | --- | --- |
| kept | **80,111** | **67.7%** |
| dropped (all no-derivatives) | 38,176 | 32.3% |

Drop breakdown: 5,992 `Attribution-NoDerivs` and **32,184
`Attribution-NonCommercial-NoDerivs`**. The second group carries both restrictions,
so it is excluded under either policy — this is why an early estimate of 112,295
was wrong (see the correction in `DECISIONS/0043`).

Survivors: NC-SA 32,878, CC BY 19,470, NC 16,397, BY-SA 10,901, no-known 465.

## Consequences for the source decision

- COCO covers the 20k and 40k iteration targets, and 80,111 images against the
  131k reproduction leaves it **1.64× short**.
- The reproduction therefore needs either multiple conversations per image or a
  second source. That decision belongs before M2a ends, not at G2.
- `DECISIONS/0041` said the drop rate was open; it is now closed at 67.7% under
  the signed policy in `DECISIONS/0043`.

## Evidence

- `experiments/m2a_build/index.jsonl`, `filtered.jsonl`, `dropped.jsonl`. [V]
- Licence names read from the annotations file, not from the dataset card. [V]
- The first run of this gate dropped 0 of 118,287 because the patterns missed
  COCO's `Attribution-NonCommercial License` naming; fixed in commit `290e431`
  and covered by a test using the eight real names. [V]

## Open items

- [U] Whether the reproduction run uses multiple conversations per image or a
  second source. Owner: implementation agent. Closing step: a DECISIONS entry
  once the iteration run shows how many conversations per image are useful.