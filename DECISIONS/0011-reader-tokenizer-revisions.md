# DECISION-0011: Reader and tokenizer revision pins

Date: 2026-11-15
Status: accepted
Work package: WP1

## Context
The reader, its tokenizer and the chat template must be pinned so every run is reproducible. Dimensions must come from the checkpoint's own config, not from recollection.

## Options considered
1. Follow the Hub `main` branch without a revision pin.
2. Pin a dated revision.
3. Pin the full commit sha for both the model and the tokenizer.

## Decision
Pin `Qwen/Qwen3-1.7B` at commit `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` for both the model and the tokenizer. Record the config values read from that revision in the injection card.

## Evidence
- `config.json` at that revision: hidden 2048, 28 layers, 16 attention heads, 8 KV heads, head_dim 128, vocab 151936, `tie_word_embeddings` true, bf16, max positions 40960, rope_theta 1e6. [V]
- `tokenizer_config.json`: `<|image_pad|>` id 151655. [V]
- Local snapshot `artifacts/models/Qwen3-1.7B/` downloaded at that revision; `config.json` sha256 `1ddb5b89...`, `tokenizer.json` sha256 `aeb13307...`. [V]
- Later readers, read from their configs: Qwen3-4B hidden 2560 / 36 layers / tied; Qwen3-8B hidden 4096 / 36; Qwen3-14B hidden 5120 / 40; Qwen3-32B hidden 5120 / 64. [V]
- The rendered golden prompt and its sha256 are in `DECISIONS/reader-qwen3-1p7b.md`. [V]

## Consequences
This closes the WP1 verification items in DECISION-0003 (placeholder id) and DECISION-0005 (rendered chat template). The projector output width for the M0 reader is 2048. The later-reader hidden sizes are recorded so projector widths are not guessed.

## Open items
- [U] Evaluation decoding settings (the checkpoint ships sampling on). Owner: implementation agent. Closing step: a DECISIONS file before the first WP10 run; until then evaluation uses greedy decoding and records it.