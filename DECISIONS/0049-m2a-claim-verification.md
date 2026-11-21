# DECISIONS 0049: M2a grounding by claim verification, not an attribute map

Date: 2026-11-21
Status: accepted
Work package: M2a
Owner instruction: replace the attribute-map grounding instrument after a concrete
failure, promote the calibration UI to a committed tool, add Particle-1 as a second
verifier.

## Context

`DECISIONS/0048` built a 50-image COCO grounding set as a per-image attribute map
(`colour`, `shape`, `count`, `position`) produced by an independent vision model and
spot-checked by the owner.

A concrete counterexample showed the instrument is invalid, not merely noisy. COCO
image `000000127450.jpg` shows seven cars filling the frame and a red motorbike
jumping above them. The labeller produced `colour=red, shape=motorcycle, count=1,
position=top` — all four correct *about the motorbike*. The question asked for "the
most prominent object", and prominence is subjective, so the attribute has **no unique
correct answer**. An annotator that correctly answered "there are seven cars" would be
marked wrong.

Two further problems:

- **Two risks were conflated.** What matters for projector training is (a) whether the
  annotator states falsehoods and (b) whether its answers need the image at all.
- **50 images cannot resolve 0.90.** 45/50 gives a Wilson interval of roughly
  0.79–0.96, and claims within one image are correlated, so the real interval is
  wider.

## Decision

1. **The attribute map is retired as an invalid instrument.** This is a measurement
   change, not an annotator regression. The frozen 0.90 bar from `DECISIONS/0040` was
   set for a different instrument; it is logged here as a **deviation**, not a failure.
2. **Grounding becomes two separate tests:**
   - **claim contradiction rate** — each Q/A turn becomes one declarative claim naming
     its own referent, judged against the image;
   - **blind-answerability** — each question is given to a text-only model without the
     image, and the rate at which it reproduces the answer is reported.
3. **Two verifiers from families other than the annotator.** `opencode/hy4-flash-free`
   and `kilo/particle-ai/particle-1:free`. Each returns `true`, `false` or `can't-tell`
   plus a one-line evidence note. Nothing is marked wrong unless it is `false`.
4. **The annotator prompt is unchanged.** The annotator chooses its own referent; claims
   name it explicitly, so no parsing is needed. The pilot tests the production pipeline.
   Seeding referents for diversity is deferred until a measured coverage gap justifies
   it (`docs/sightline-project-document.md` section 8.0: a failure must point at the
   pipeline).
5. **COCO instance annotations are a flag, not a key.** For category and count claims,
   agreement or disagreement with the instances file is reported as supporting signal.
   Largest-instance-by-area is not used as the referent; it often selects a table or
   couch rather than the salient object.
6. **Planted errors measure the verifier.** True claims are automatically perturbed
   (number, colour, noun swapped) and must be rejected. This is the guard against
   verifier acquiescence, which the contradiction rate alone cannot detect.
7. **Counts of six or more go to human review.** Verifiers and annotators may share
   failure modes on large counts, and COCO counts are incomplete under crowd and
   occlusion.
8. **The 50 labelled images are burned.** They shaped the instrument and cannot be a
   test set. The pilot runs on a **fresh 500 images**, roughly 1,500+ claims.
9. **Thresholds are frozen before the pilot run:**
   - contradicted rate, upper bound of a 95% cluster bootstrap by image, **≤ 0.15**;
   - planted-error rejection **≥ 0.90**;
   - human–verifier agreement **≥ 0.85**.
10. **The owner's review time calibrates the verifier, not the rate.** The review set is
    sampled to cover verifier disagreement, agreement-true and agreement-false strata,
    and each item shows the verdict plus the evidence line.

## Why the prompt stays free

The pilot must measure the pipeline that will actually run. A prompt that names the
referent would test a different pipeline than the 20k corpus uses. If production later
adds referent seeding for diversity, the pilot prompt changes with it, under a new
DECISIONS entry with a before/after measurement.

## Consequences

- The grounded rate is no longer a single number over a fixed attribute map. The pilot
  reports contradiction rate, can't-tell share, blind-answerability and planted-error
  rejection, by claim type (existence, count, colour, spatial, text, action).
- The 20k corpus can be annotated and *filtered* on contradicted claims, rather than
  gating on a pilot estimate alone.
- The calibration UI becomes a committed tool (`tools/m2a_review.py`) because it is now
  an instrument with a version, not a throwaway page.

## Open items

- [U] Whether blind-answerability needs a threshold at all. Closing step: inspect the
  distribution on the fresh 500 and set one, rather than guessing now. Owner:
  Keylhan Paumard--André.
- [U] The human–verifier agreement figure will itself be wide at 50 items. Closing
  step: report the Wilson interval with the figure and treat it as calibration
  evidence, not as an estimate of verifier accuracy.