# DECISION-0029: K2.6 MoonViT-3D extraction on Hugging Face

Date: 2026-11-15
Status: accepted
Work package: WP6

## Context
The K2.6 MoonViT-3D extraction should be published for reproducibility, like the K3 extraction.

## Decision
Publish `keypa/MoonViT-3D-K2.6-7eb5002f` (public) with the 329 `vision_tower.` tensors, 6 `mm_projector.` tensors (942,287,840 bytes, all BF16), the manifest, the Modified MIT LICENSE and the model card.

## Evidence
- K3 repo: `keypa/MoonViT-V2-K3-f831ab66` (sha `a62a3622248bc80e0c52504d459dd7918e66d072`). [V]
- K2.6 repo: `keypa/MoonViT-3D-K2.6-7eb5002f`; 942 MB, 7.5 s upload, total 8.8 s, peak RSS 378.6 MB, 4.4 CPU-s. [V]
- Manifest: `docs/manifests/k26_vision_7eb5002f.manifest.json` with per-tensor sha256. [V]

## Consequences
The K2.6 tower is available for M2b and Modal; the extraction is reproducible. The K2.6 license is Modified MIT.

## Open items
none