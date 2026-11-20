# DECISION-0043: whether non-commercial images may enter the M2a corpus

Date: 2026-11-19
Status: accepted
Work package: M2a entry criteria

## Why this needs its own record

`DECISIONS/0041` set the licence gate to drop non-commercial (NC) and
no-derivatives (ND) images. That was a conservative default chosen before the
data was measured. The COCO audit then showed the choice governs **three quarters
of the available corpus**, so it is not a detail of implementation — it decides
whether M2a has 30k images or 112k, and whether the 131k reproduction is
reachable at all.

This record exists so the decision is made once, with the licence text in front
of it, rather than re-argued per run.

## The question

May images licensed CC BY-NC be used in the M2a training corpus?

## What the licence says

Read from the CC BY-NC 2.0 legal code (creativecommons.org/licenses/by-nc/2.0/legalcode),
not recalled. **[V]**

- **Section 3** grants a *worldwide, royalty-free, non-exclusive, perpetual*
  licence to reproduce the work, to create and reproduce derivative works, and to
  distribute copies.
- **Section 4(b)** is the only material restriction: "You may not exercise any of
  the rights granted to You in Section 3 above in any manner that is **primarily
  intended for or directed toward commercial advantage or private monetary
  compensation**."
- **Section 4(a)** requires keeping the copyright notice and crediting the
  original author. COCO records no author name, only `flickr_url` and `coco_url`;
  those are the attribution route and are now kept in the index.
- **Section 2** preserves fair use and other limitations on the copyright owner's
  rights.
- **Section 7** terminates the licence automatically on breach.

So NC does not restrict *use*. It restricts exercising the granted rights in a
manner primarily directed at commercial advantage. Research with no revenue is
the paradigm case the licence permits.

**Not legal advice.** Creative Commons states it is not a law firm and provides no
legal services. This is a reading of the licence text. If the corpus's legal basis
becomes material — a release, a product, a company — take advice.

## The measured supply

Full `instances_train2017.json` (118,287 images), run 2026-11-18.
Detail in `reports/m2a-coco-licence-audit.md`. **[V]**

| policy | usable | share |
| --- | --- | --- |
| drop NC and ND (current) | 30,836 | 26.1% |
| **allow NC, drop ND** | **112,295** | **94.9%** |

Drops under the current policy: 81,459 non-commercial (93.1% of drops) and 5,992
no-derivatives (6.9%). Survivors are CC BY 2.0 (19,470), CC BY-SA 2.0 (10,901)
and no-known-restrictions (465).

### Supply against the plan

| target | need | current (drop NC+ND) | allow NC, drop ND |
| --- | --- | --- | --- |
| M2a iteration, 20k | 20,000 | enough (30,836) | enough |
| M2a iteration, 40k | 40,000 | short by 9,164 | enough |
| G2 reproduction, 131k | 131,072 | short by 4.3× | short by ~51k (1.64×) |

## Correction to the figures above

The table above was written with **112,295** usable images for the allow-NC
policy. That number was **wrong**, and the signed decision was made on it. The
measured value is **80,111 (67.7%)**.

The error: I computed the allow-NC case as "118,287 minus the 5,992 no-derivatives
images", but the 5,992 was counted *after* non-commercial images had already been
removed. The licence `Attribution-NonCommercial-NoDerivs License` carries **both**
restrictions, and there are **32,184** of them. They are no-derivatives images and
must be dropped under either policy.

Corrected breakdown of the 38,176 drops (32.3% of the file):

| licence | count |
| --- | --- |
| Attribution-NoDerivs License (ND only) | 5,992 |
| Attribution-NonCommercial-NoDerivs License (NC **and** ND) | 32,184 |

Survivors under allow-NC, with the measured counts:

| licence | count |
| --- | --- |
| Attribution-NonCommercial-ShareAlike License | 32,878 |
| Attribution License (CC BY 2.0) | 19,470 |
| Attribution-NonCommercial License | 16,397 |
| Attribution-ShareAlike License | 10,901 |
| No known copyright restrictions | 465 |
| **total** | **80,111** |

The decision itself is unaffected — allowing NC still beats the conservative
policy by 49,275 images (2.6×) — but the reproduction is **1.64× out of reach**,
not 17% short. That changes how much a second source is needed, so it matters.

`flickr_url` is present on all 80,111 kept records after re-running `index`. [V]

## Costs of allowing NC

1. **The corpus becomes non-commercial only, permanently.** NC is irrevocable for
   the granted scope. If Sightline becomes a product, a hosted service, or is
   built on commercially, those images cannot come along.
2. **If the corpus is published, the restriction travels.** Every downstream user
   inherits NC, which reduces who can reuse what we release. This is the largest
   cost for a dataset intended to be reused.
3. **Attribution remains mandatory** (s.4(a)) and is already implemented via
   `flickr_url`.

For a research repository with no revenue, cost 1 is near zero today. Costs 1 and
2 are about what Sightline is for, not about the next run.

## Why ND stays excluded under either policy

NoDerivs sits in the licence name and removes the derivative-works grant that
section 3 otherwise gives. Training pipelines resize, crop and re-encode, which a
strict reading treats as creating a derivative. 5,992 images (5.1%) carry it, and
arguing the edge is not worth the exposure.

## Other sources, if more volume is needed

| source | volume | licence | what it costs |
| --- | --- | --- | --- |
| PixMo (Ai2) | ~717k caption rows | ODC-BY 1.0 on annotations | images are **URLs to third-party hosts**; per-image rights unknown; some content generated by Claude |
| The Cauldron (HF M4) | 1.88M rows, 169 GB | CC BY 4.0 on the collection | **50 sub-datasets, each with its own licence**; per-subset review required |
| Flickr30k | 31,783 | non-commercial research only | same NC question, and small |

They trade volume risk for provenance risk. COCO remains the best-behaved source
because its licence is recorded per image, which is what makes the gate
automatable.

## Recommendation

**Allow NC, drop ND.**

- The 131k target moves from 4.3× out of reach to ~17% short. That is the
  difference between "COCO cannot support the reproduction" and "COCO nearly
  does".
- The cost is a constraint on commercial futures and on downstream reuse, not on
  the research itself. For a learning repository that is likely acceptable, but
   it is the owner's call, not the agent's.

## Decision

**Option B: allow non-commercial images, drop no-derivatives and missing.**

- [ ] **Option A** — drop NC and ND. Corpus 30,836 (26.1%). Supports the 20k
  iteration only. `DECISIONS/0041` stands unchanged.
- [x] **Option B** — allow NC, drop ND. Corpus **80,111 (67.7%)** after the
  correction above. Supports the 40k iteration; the 131k reproduction is 1.64×
  out of reach. `DECISIONS/0041` is amended.

Signed: Keylhan Paumard--André  Date: 2026-11-19

Signed at the owner's instruction ("sign it please") on the recommendation above.
The owner may amend this record; the gate follows it.

## Review trigger

Revisit this decision if any of the following happens:

- Sightline becomes a product, a hosted service, or is used commercially.
- The corpus is published for third-party reuse.
- A second image source is added, which changes the licence mix.

## Evidence

- CC BY-NC 2.0 legal code, sections 2, 3, 4(a), 4(b), 7. [V]
- COCO terms of use: annotations CC BY 4.0, images under Flickr terms, copyright
  not owned by the consortium. [V]
- Licence audit: 118,287 in, 30,836 kept, 87,451 dropped. [V]
- `flickr_url` retained in `src/sightline/m2a/coco_index.py` for s.4(a). [V]