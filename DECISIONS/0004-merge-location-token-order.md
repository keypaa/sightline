# DECISION-0004: Merge location and token order

Date: 2026-11-15
Status: proposed (verify in WP5)
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

## Open items
- [U] Exact interpolation mode and merge order in the vendored `modeling_kimi_k3.py`. Owner: implementation agent. Closing step: WP5 read and synthetic-grid test.