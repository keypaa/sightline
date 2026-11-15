# DECISION-0022: Projector shape and norm ratio

Date: 2026-11-15
Status: accepted
Work package: WP7

## Context
The projector maps merged MoonViT features to the reader's embedding width. Only it trains. Its shape, initialisation flag and norm-ratio definition must be fixed before WP8.

## Decision
- LayerNorm(input_width) -> Linear(input_width, hidden_width) -> GELU -> Linear(hidden_width, reader_width).
- `hidden_width` defaults to `input_width` (4096 for MoonViT-V2, 4608 for K2.6); `reader_width` comes from the reader config (2048 for Qwen3-1.7B).
- The parameter count is derived from the module. For 4096->4096->2048 it is 25,180,160; for 4608->4608->2048 it is 30,686,720; for a hidden width of 8192 it is 50,350,080.
- Initialisation: `random` by default. `embedding_alignment` is a later ablation (before M2a) and raises `NotImplementedError`.
- Norm ratio: mean projector-output norm / mean embedding-row norm. WP8 logs it at step 0 and every 100 steps.
- Dtype: fp32 projector parameters (DECISIONS 0007, WP8).

## Evidence
- Unit tests: parameter counts match the formula; output width 2048; 4D merged features accepted; random default; embedding alignment raises; norm ratio positive and finite. [V]
- K3 encoder integration: `(1, 256, 4096)` -> `(1, 256, 2048)`, all finite; peak 3.14 GB, CPU 64.9 s. [V]

## Consequences
The projector is ready for WP8. The hidden width is a design choice; the K2.6-style reference uses hidden_width = input_width. The embedding-alignment init is not implemented.

## Open items
- [U] Embedding-alignment init (before M2a). Owner: implementation agent. Closing step: count identical token strings between the K3 and Qwen3 tokenizers, fit the affine map, and add it as an ablation.
- [U] Hidden width for the M2a projector (4096 vs 8192). Owner: implementation agent. Closing step: M2a ablation; the current default is 4096 (input_width).