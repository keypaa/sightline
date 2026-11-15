# DECISION-0017: K2.6 MoonViT extraction and vendoring

Date: 2026-11-15
Status: accepted; 20-image parity verified
Work package: WP6

## Context
WP6 needs the K2.6 MoonViT tower for the M2b encoder A/B and as the reproduction control.

## Decision
- Extract 329 `vision_tower.` and 6 `mm_projector.` tensors from `moonshotai/Kimi-K2.6` at `7eb5002f6aadc958aed6a9177b7ed26bb94011bb` into `artifacts/k26_vision_7eb5002f.safetensors` (942,287,840 bytes, all BF16) and `docs/manifests/k26_vision_7eb5002f.manifest.json`.
- Vendor the vision portion of `modeling_kimi_k25.py` as `src/sightline/vision_k25/modeling_k25_vision.py`, plus `configuration_kimi_k25.py`, `configuration_deepseek.py`, `media_utils.py`, `kimi_k25_vision_processing.py`, `kimi_k25_processor.py` and the Modified MIT `LICENSE`. The DeepSeek LM import is dropped; the K2.6 LM is out of scope.
- Use eager attention: the K2.6 module exposes only `flash_attention_2` and `eager` (no SDPA).
- The K2.6 position-embedding interpolation is bicubic (the class default); K3 uses bilinear.

## Evidence
- Extraction: 329 tower + 6 projector tensors, 0 missing, 942,287,840 bytes, 98.2 s, peak 1.86 GB, 13 CPU-s. [V]
- Tower load: 329 tensors, 0 missing, 0 unexpected. [V]
- 448x448: `pixel_values` `(1024, 3, 14, 14)`, `grid_thws` `[[1, 32, 32]]`, output `(256, 4, 1152)` -> `(256, 4608)`, finite; one CPU forward 16.8 s, peak 3.56 GB. [V]
- Parity: 20 images at 448/672/896 on Modal T4 vs `immanuelpeter/MoonViT-K2.6` (84ec70ac...): max_abs 0.0, min cosine 0.99999982, allclose true; 180.8 s, peak RSS 8.16 GB, CPU 177.3 s, max GPU 5.68 GB. [V]

## Consequences
The K2.6 tower is ready for M2b. The 6 projector tensors (54,277,888 parameters) are extracted but not wrapped; WP7 owns the projector. The K2.6 LM is not vendored.

## Open items
- [U] FlashAttention 2 vs eager on the K2.6 tower (the parity used eager). Owner: implementation agent. Closing step: GPU with a compatible FA2 wheel.
- [U] K2.6 projector wrapper (WP7). Owner: implementation agent.
- [U] Publish the K2.6 extraction to HF (optional). Owner: implementation agent. Closing step: ask the owner; push if approved.