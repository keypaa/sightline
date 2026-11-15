# DECISION-0015: Vendor the K3 MoonViT-V2 tower

Date: 2026-11-15
Status: accepted; 20-image CPU parity done, GPU/FA2 pending
Work package: WP5

## Context
WP5 needs the official K3 MoonViT-V2 tower so forward parity holds by construction. The full K3 repository is about 1.56 TB, but the vision code and the extracted tower are small.

## Options considered
1. Vendor the full `modeling_kimi_k3.py` (tower plus K3 LM).
2. Vendor only the vision portion of `modeling_kimi_k3.py`.
3. Rewrite the tower from the config and weights.

## Decision
Vendor the vision portion of `moonshotai/Kimi-K3` at revision `f831ab66814297da540d832a5235f8e904f29d06` as `src/sightline/vision_k3/modeling_k3_vision.py`, together with `configuration_kimi_k3.py`, `media_utils.py`, `kimi_k3_vision_processing.py` and the K3 `LICENSE`.

The full `modeling_kimi_k3.py` imports `modeling_kimi_linear.py`, which imports `transformers.utils.generic.OutputRecorder` and `check_model_inputs`; both are absent in transformers 5.12.1. The K3 language model is out of scope for M0, so the vision classes are vendored unchanged and the LM import is dropped. This is a recorded deviation from "vendor the whole file".

## Evidence
- The unmodified full file fails to import: `ImportError: cannot import name 'OutputRecorder' from 'transformers.utils.generic'`. [V]
- The vendored vision module imports and exposes `MoonViT3dPretrainedModel` and `VisionTowerConfig`. [V]
- Loading `artifacts/k3_vision_f831ab66.safetensors` into the tower: 165 tensors, 0 missing, 0 unexpected. [V]
- A 448x448 image through the official `KimiK3VisionProcessor`: `pixel_values` shape `(1024, 3, 14, 14)`, `grid_thws` `[[1, 32, 32]]`. [V]
- The tower output for that image is `(256, 4, 1024)`; the wrapper flattens it to `(256, 4096)`. [V]
- One-image fp32 forward peaks at 3.13 GB RSS. [V]
- The 20-image slow test passes with one image per forward: 6 tests, peak 3.14 GB. The run is CPU-bound and uses all 6 CPU cores; wall time was not recorded. [V]

## Consequences
The official tower code is used, not rewritten. The K3 LM is not vendored. The eager attention path is quadratic in packed tokens, so the wrapper runs one image per forward. The 20-image parity test is marked slow and skipped by default; set `SIGHTLINE_VISION_IMAGE_COUNT` to run fewer images.

## Open items
- [U] GPU forward parity and FlashAttention 2 on sm_120. The CPU run passes but is CPU-bound; the eager path is not the training path. Owner: implementation agent. Closing step: run the slow test on the GPU box with an sm_120 FlashAttention 2 wheel and record the result.
- [U] Tensor-by-tensor comparison with the earlier standalone extraction (`keypa/MoonViT-V2-Standalone`). The extraction manifest says the 165 tower tensors are byte-identical; the earlier file is not in this repository. Owner: implementation agent. Closing step: download the earlier file and compare, or cite the extraction result in the G0 report.