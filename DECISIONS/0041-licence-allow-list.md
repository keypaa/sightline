# DECISION-0041: licence allow-list for M2a image sources

Date: 2026-11-17
Status: accepted
Work package: M2a entry criteria

## Context

Section 16 item 7 requires a per-source licence table and forbids any source
entering without a recorded licence. COCO's terms page states that the
annotations are CC BY 4.0 but that the consortium **does not own the copyright of
the images**, and that use of the images must abide by the Flickr terms. Each
image therefore carries its own licence id in the annotations file. [V:
cocodataset.org/#termsofuse]

Any gate that trusts the dataset-level licence would admit non-commercial and
no-derivatives photographs.

## Decision

- The gate reads the licence **name** from the annotations file, per image.
- **Drop** any name carrying `nc` (non-commercial) or `nd` (no-derivatives), and
  drop a missing name. Reasons are recorded as `non-commercial`,
  `no-derivatives`, `missing-licence`.
- An optional allow-list may **narrow further**, never widen. The NC/ND checks
  run first, so no caller can re-admit a non-commercial image.
- The gate runs **before** any image is fetched. After fetching, the temptation
  to keep what is already downloaded is too strong.
- Dropped records are counted and reported at the stage, so the drop rate is a
  number rather than an absence.

## Evidence

- COCO terms of use: annotations CC BY 4.0, images under Flickr terms, copyright
  not owned by the consortium. [V]
- Flickr30k: non-commercial research and education only, per its providers. [V]
- PixMo: ODC-BY-1.0 annotations, images distributed as URLs. [V]
- The Cauldron: CC BY 4.0 collection, 50 sub-datasets each with their own
  licence. [V]

## Consequences

COCO is the source for M2a because it is the only candidate that records a
licence per image, which is what makes the gate automatable. Flickr30k is
excluded from any commercial use; PixMo and Cauldron would need manual
provenance work before download.

## Measured outcome (2026-11-18)

Run on the full `instances_train2017.json` (118,287 images): **30,836 kept
(26.1%), 87,451 dropped (73.9%)** — 81,459 non-commercial and 5,992
no-derivatives. Survivors are CC BY 2.0 (19,470), CC BY-SA 2.0 (10,901) and
no-known-restrictions (465). Detail in `reports/m2a-coco-licence-audit.md`. [V]

The dataset card's CC BY 4.0 covered the annotations only; three quarters of the
photographs were unusable. Trusting the card would have ingested 118k images we
had no right to use.

Consequence: COCO supports the 20k iteration set but not the 131k reproduction on
one conversation per image.

## Open items

- [U] Whether the reproduction run uses multiple conversations per image or a
  second source. Owner: implementation agent. Closing step: a DECISIONS entry
  once the iteration run shows how many conversations per image are useful.