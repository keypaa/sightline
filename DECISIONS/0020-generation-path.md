# DECISION-0020: Generation path

Date: 2026-11-15
Status: accepted
Work package: WP3

## Context
Generation must use the same splice, mask and position path as training. The HF `generate()` with `inputs_embeds` uses the provided embeddings once and the KV cache for the rest; it returns only the new tokens. The ids path returns the prompt plus the new tokens.

## Decision
- Greedy decoding (`do_sample=False`), cache on by default.
- `generate_new_tokens` normalises the ids path to return only the new tokens; the `inputs_embeds` path already returns only the new tokens.
- The real reader's `generation_config.json` ships sampling on; evaluation overrides with greedy or explicitly seeded decoding and records the setting.
- The same splice, mask and position ids as training; the golden splice uses the reader's own embedding rows.

## Evidence
- Tiny model: `generate(inputs_embeds=...)` returns `(1, 5)`; `generate(input_ids=...)` returns `(1, 11)`; the new tokens match on the golden splice. Cache vs recompute identical. Padding invariance identical. [V]
- Real reader (Qwen3-1.7B, fp32, CPU): golden splice generation matches ids; cache vs recompute identical; peak 12.58 GB, CPU 131.5 s. [V]

## Consequences
Generation parity is verified. The same helper is used for training and evaluation. The `inputs_embeds` path must be given an attention mask for the prompt.

## Open items
- [U] bf16 generation parity on the RTX PRO 6000. Owner: implementation agent. Closing step: run the slow test on the GPU box.