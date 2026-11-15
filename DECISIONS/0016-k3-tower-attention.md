# DECISION-0016: K3 tower attention has no SDPA path

Date: 2026-11-15
Status: accepted
Work package: WP5

## Context
DECISION-0009 chose PyTorch SDPA by default for the Qwen3 readers and the K3 tower. The vendored K3 tower does not expose SDPA.

## Options considered
1. Keep the DECISION-0009 wording and add an SDPA path to the vendored code.
2. Use the vendored eager path on CPU and FlashAttention 2 only when available.
3. Rewrite the attention layer to use SDPA.

## Decision
Use the vendored `eager` attention path on CPU. Use `flash_attention_2` only when the kernel is available and after a parity test. This supersedes the K3-tower part of DECISION-0009; the Qwen3 readers remain on SDPA.

## Evidence
- `VL_VISION_ATTENTION_FUNCTIONS = {"flash_attention_2": multihead_attention, "eager": eager_attention}`. There is no SDPA entry. [V, modeling_k3_vision.py]
- The K3 `vision_config._attn_implementation` is `flash_attention_2`; flash-attn is not installed, so the eager path is the only runnable path here. [V]
- `eager_attention` builds a full `(1, L, L)` boolean mask and a `(heads, L, L)` attention matrix. For 20 packed images, `L = 20 * 1024 = 20,480`, so the matrix would be about `12 * 20,480^2 * 4 bytes = 20 TB`. [V, arithmetic]

## Consequences
The wrapper runs one image per forward. Batching multiple images on the eager path is not possible. The 20-image parity test is marked slow. FlashAttention 2 would use `flash_attn_varlen_func` with `cu_seqlens` and avoid the dense mask, but it is deferred to the GPU.

## Open items
- [U] FlashAttention 2 on sm_120: install a compatible wheel, run the tower parity test, and record the result. Owner: implementation agent. Closing step: GPU box.