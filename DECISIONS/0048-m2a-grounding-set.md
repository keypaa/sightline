# DECISIONS 0048: M2a grounding set and labelling protocol

Date: 2026-11-21
Status: accepted
Work package: M2a
Owner instruction: hand-label COCO images for a stronger grounding gate; owner
spot-checks the labels; use hy4-flash-free via opencode while it is effectively
unmetered.

## Context

`DECISIONS/0046` measurement #6 requires the annotator's grounded rate against the
frozen 0.90 bar from `DECISIONS/0040`. That bar was measured on the **M1 synthetic
set**, where the generator knows the true answer. COCO carries captions, not
question/answer labels, so grounding **cannot** be measured on COCO without an
independent answer key.

A synthetic-only check leaves a real gap: a model can be well grounded on clean
rendered shapes and still be sloppy on photographs, which is the data M2a trains on.

## Decision

1. **A 50-image COCO grounding set is built and frozen.** Images are drawn from
   `experiments/m2a_iter20k/fetched.jsonl` with `sample_subset(seed=0)`; the chosen
   ids are written to a committed file, so the set cannot drift between runs.
2. **The labeller produces a per-image attribute table, not a conversation:**
   `{"colour": ..., "shape": ..., "count": ..., "position": ...}` — the same
   attributes the M2a prompt asks about. One JSON line per image.
3. **The labeller is `opencode/hy4-flash-free`, and it is not the annotator.**
   The annotator is Qwen3.6-27B-FP8. Different families, so agreement between them
   is informative rather than circular.
4. **The labeller never sees the annotator's answers.** It labels from the image
   and the attribute list only.
5. **The owner spot-checks the labels.** Each row carries `human_confirmed: true`
   or `false`. This is the only source of actual ground truth in the set.
6. **Two numbers are reported separately, never merged:**
   - agreement with the independent labeller, over all 50;
   - agreement on the human-confirmed subset.
   The 0.90 bar applies to the **human-confirmed** subset. Labeller agreement is
   reported as context and cannot move the bar.
7. **The protocol is frozen before any label is produced.** Changing the sample, the
   attributes, the prompt or the bar needs a new DECISIONS entry.

## What this is not

The labeller is a vision-language model and can be confidently wrong. Agreement with
it is **not** ground truth; it is a second opinion. The human spot-check is what
anchors the set, and its size bounds how much weight the grounded number can carry.

## Consequences

- The pilot gains a real COCO grounding path instead of reporting
  `grounding_measurable: false`.
- A COCO-specific grounding failure is caught before the 20k run rather than by a
  post-hoc spot-check.
- The labelling stage runs on this laptop through opencode; it does **not** need the
  GPU box. The Qwen3.6-27B annotator run remains GPU-dependent.

## Open items

- [U] Whether 50 images give a tight enough interval on the human-confirmed subset.
  Closing step: compute the Wilson interval on the confirmed subset after spot-checking
  and record whether it is narrow enough to distinguish 0.90 from chance.
  Owner: Keylhan Paumard--André.