# DECISION-0013: Right padding for training

Date: 2026-11-15
Status: accepted
Work package: WP2

## Context
The main document requires one padding side, tested, and the same code path for training and generation. Batches have different text lengths, so padding must not change logits at real positions.

## Options considered
1. Right padding.
2. Left padding.
3. Per-path padding (right for training, left for generation).

## Decision
Use right padding for training. Position ids are `0..L-1` at real tokens; padding gets position 0 and is masked out. The same mask and position-id helpers are used for generation in WP3.

## Evidence
- Main document section 7.1: pick a padding side once, test it, and use the same code path for training and generation. [V]
- M0 breakdown, WP2: positions and attention mask for batches with different text lengths, with right padding for training. [V]
- `test_padding_invariance` on the pinned reader: padded batch versus single sample agrees within `1e-4` (observed max difference `2.1e-5`). [V]

## Consequences
`make_attention_mask` and `make_position_ids` are padding-side agnostic; the default path is right padding. WP3 tests that generation uses the same path.

## Open items
none