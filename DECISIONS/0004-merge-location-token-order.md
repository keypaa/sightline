# DECISION-0004: Merge location and token order

Date: 2026-11-15
Status: accepted (verified in WP5)
Work package: WP0

## Context
The K3 tower groups 2x2 patch blocks; the projector consumes the concatenated block. The merge order and the projector's pre-norm differ between V2 and K2.6, so the token order must be tested rather than assumed.

## Options considered
1. Merge in the tower (`tpool_patch_merger`), then concatenate four patches row-major for the projector.
2. Merge inside the projector.
3. Column-major or another order.

## Decision
Use the tower's 2x2 row-major merge, then concatenate four patches for the projector (4096 wide for V2). Confirm by reading the vendored module and by a synthetic-grid token-order test in WP5.

## Evidence
- M0 breakdown, 2b decision 4: tower groups 2x2 blocks row-major; projector takes four patches concatenated; V2 4096, K2.6 4608. [V, via the document]
- K3 extraction manifest: `mm_projector.proj.0.weight` is 4096x4096, `mm_projector.proj.2.weight` is 7168x4096. [V, docs/manifests/k3_vision_f831ab66.manifest.json]
- K3 projector has no pre-norm and ends in RMSNorm; K2.6 starts with LayerNorm. [S, via the document]

## Consequences
The token order is identical in training and generation. WP2's splice tests use the derived token count; WP5 adds the merge-order test.

## Evidence from WP5
- The vendored `tpool_patch_merger` groups each 2x2 block in row-major order: a synthetic 4x4 grid maps output token (0,0) to patches 0, 1, 4, 5. [V, tests/unit/test_vision_k3_merge.py]
- A 448x448 image through the pinned tower produces 256 merged tokens of width 4096 (4 patches x 1024). [V, one-image tower run]
- The extracted projector tensors are 4096x4096 and 7168x4096, matching a 4096-wide merged token. [V, docs/manifests/k3_vision_f831ab66.manifest.json]

## Open items
- [U] Exact interpolation mode and merge order in the vendored `modeling_kimi_k3.py`. Closed: interpolation mode is `bilinear` (read from config) and the merge order is confirmed by the synthetic test. Owner: implementation agent. Closing step: WP5 tests.
- [U] 20-image forward parity on the GPU workstation. Owner: implementation agent. Closing step: run `tests/conformance/test_vision_k3_tower.py` with `-m slow` on the GPU box.