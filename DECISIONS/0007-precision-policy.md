# DECISION-0007: Precision policy

Date: 2026-11-15
Status: accepted
Work package: WP0

## Context
The reader and encoder are frozen and ship in bf16. The projector is the only trained module, so its master weights and optimizer state need more precision than the forward pass.

## Options considered
1. bf16 everywhere.
2. Frozen modules in bf16; projector in fp32 with bf16 autocast; optimizer state in fp32.
3. fp32 everywhere.

## Decision
Keep frozen modules in their shipped bf16. Projector parameters are fp32 with bf16 autocast; optimizer state is fp32. A trained projector may be cast to bf16 for release.

## Evidence
- M0 breakdown, 2b decision 7. [V]
- Main document section 7.4: bf16 activations; projector weights and optimizer states in fp32. [V]
- K3 tower and projector tensors are BF16 in the extraction manifest. [V, docs/manifests/k3_vision_f831ab66.manifest.json]

## Consequences
The run record stores the dtype policy. WP7/WP8 tests check the projector dtype and the autocast path. Frozen-weight hashes are unaffected by the projector dtype.

## Open items
none