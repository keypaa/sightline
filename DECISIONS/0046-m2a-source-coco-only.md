# DECISION-0046: M2a source is COCO only for the 20k iteration

Date: 2026-11-21
Status: accepted
Work package: M2a

## Context

Baseten's paper (arXiv 2610.05897, section 3.1) says the SFT corpus is
"131,072 multi-turn conversations using images from 13 public sources". The
number of sources is a property of the recipe, so M2a has to decide what to do
with it.

The paper does not name the sources. Section 3.1 is one sentence; no appendix,
reference or linked artifact lists them. A search found nothing naming them
either. [V — arXiv HTML and PDF, read 2026-11-21]

The only image source under active audit is COCO: 80,111 usable images after the
signed licence policy in DECISIONS 0043, every one carrying `flickr_url` for
attribution.

## Decision

- **M2a's 20k iteration uses COCO only.** One source, one licence table, one
  fetch path.
- **"13 public sources" is recorded as unreproducible, not as a target.** Any
  13-source set we assembled would be our invention wearing Baseten's number,
  and the paper gives no basis for choosing them.
- FineVision is not adopted for this stage. It defers every sub-dataset licence
  to the user (185 subsets, 4.65 TB), and its decontamination set does not
  include MMMU-Pro or BLINK. [V — FineVision dataset card and article source;
  lmms-eval-embeddings file listing]

## Rationale

The iteration set's job is that a failure points at the pipeline
(`docs/sightline-project-document.md` section 8.0). Every additional source adds
a licence obligation and a fetch path to a run whose purpose is to have fewer
variables. Diversity is a real property of Baseten's data, but it is measurable
after the fact, whereas an unrecorded licence is not fixable after the fact.

COCO is also the only source where the licence is recorded per image and
machine-checkable, which is the property the project's evidence discipline
requires.

## Consequences

- The corpus has one `source` value. The manifest reports the distinct-source
  count, so "1" is visible rather than silently implied.
- The 131k reproduction cannot be reached from COCO at one conversation per
  image: 80,111 usable images against 131,072 conversations. The gap is closed
  by more than one conversation per image, or by a later source decision, not by
  this entry.
- G2 is a **qualitative** match: loss behaviour and vision-minus-blind, not
  Baseten's absolute scores. The paper's numbers also come from a corpus we
  cannot obtain, so no score comparison is available even in principle.
- Any later source addition needs its own DECISIONS entry with a per-source
  licence table, before download.

## What must be measured before M2 training

Each item names the number that closes it. All of these belong to the 20k
iteration, not to a later stage.

| # | Measure | Why it is not optional |
| --- | --- | --- |
| 1 | **Benchmark decontamination** — SSCD cosine >= 0.95 against MMMU-Pro and BLINK image sets, count and ids removed | FineVision's 66-benchmark dedup does **not** include either benchmark [V], so COCO carries no decontamination claim at all. Without this, G2 is contaminated by construction. |
| 2 | **Per-source and per-licence counts** in the final manifest | The paper records source, licence and checksum per example; the manifest must be able to reproduce that table. |
| 3 | **Category coverage** of the sampled 20k (COCO's 80 categories) | Diversity is the property "13 sources" was a proxy for. Measure it instead of assuming it; a narrow sample is evidence, not a guess. |
| 4 | **Perceptual-hash duplicates across splits** | Splits are by hash and source; a duplicate crossing train/test leaks. |
| 5 | **Conversations per image** actually achieved, and the token total | 131k needs ~1.6 per image from COCO. The measured number is the input to the reproduction decision. |
| 6 | **Annotator parse rate and grounded rate** against the frozen 0.90 pilot bar, with the prompt hash and chat-template hash | A new annotator is a new instrument; the 402 Kilo rows are deleted, not reused. |
| 7 | **Annotator throughput** — conversations/hour and tokens/conversation on this GPU | Decides whether 20k and later 131k are hours, days or weeks, and replaces the API-quota estimate. |
| 8 | **Image checksum verification rate** at fetch | A fetch that silently returns the wrong bytes invalidates the licence record attached to it. |

## Open items

- [U] Whether MMMU-Pro and BLINK image sets are obtainable for the dedup in item
  1, and at what revision. Closing step: locate the pinned test splits and record
  their hashes in the dedup run record.
- [U] The conversation count per image for the 131k reproduction. Closing step:
  item 5's measurement on the 20k run.
- [U] Whether a second source is needed at all. Closing step: item 3's coverage
  measurement, read against the G2 result.