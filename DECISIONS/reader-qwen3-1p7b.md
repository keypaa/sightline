# Injection card: Qwen3-1.7B

Date: 2026-11-15
Status: accepted
Work package: WP1

## Pin

| Item | Value | Tag |
| --- | --- | --- |
| Reader repo | `Qwen/Qwen3-1.7B` | [V] |
| Model revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | [V] |
| Tokenizer revision | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | [V] |
| Local snapshot | `artifacts/models/Qwen3-1.7B/` (gitignored) | [V] |
| Model dtype | `bfloat16` (shipped) | [V, config.json] |

## Config read from the checkpoint's own config.json

| Field | Value | Tag |
| --- | --- | --- |
| `model_type` | `qwen3` | [V] |
| `hidden_size` | 2048 | [V] |
| `num_hidden_layers` | 28 | [V] |
| `num_attention_heads` | 16 | [V] |
| `num_key_value_heads` | 8 | [V] |
| `head_dim` | 128 | [V] |
| `vocab_size` | 151936 | [V] |
| `tie_word_embeddings` | true | [V] |
| `max_position_embeddings` | 40960 | [V] |
| `rope_theta` | 1000000 | [V] |
| `rms_norm_eps` | 1e-06 | [V] |
| `attention_bias` | false | [V] |

The projector output width must equal 2048. Because `tie_word_embeddings` is true, the input embedding matrix and the output head share one table; the golden splice feeds the reader's own embedding rows at placeholder positions.

`config.json` sha256: `1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197`.

## Tokenizer

| Item | Value | Tag |
| --- | --- | --- |
| Class | `Qwen2Tokenizer` | [V] |
| `<|image_pad|>` id | **151655** | [V] |
| `eos_token_id` | 151645 (`<|im_end|>`) | [V] |
| `pad_token_id` | 151643 (`<|endoftext|>`) | [V] |
| `tokenizer.json` sha256 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` | [V] |
| `tokenizer_config.json` sha256 | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` | [V] |

## Injection point

Projected patch vectors replace the embedding rows at the placeholder positions and enter through `inputs_embeds`. The Qwen3 forward accepts exactly one of `input_ids` or `inputs_embeds`; pass `inputs_embeds` and leave `input_ids` unset. No embedding lookup happens at placeholder positions.

## Positions

Plain 1D RoPE over the full spliced sequence: position ids are `0..T-1`. No MRoPE and no 2D/3D positions.

## Mask

Causal attention everywhere. Right padding for training (WP2 fixes the padding side and tests it); the same code path is used for generation (WP3).

## Side channels

None. Qwen3 dense has no token-id hash routing, expert routing by token id, or reserved-id palette. The placeholder id is never looked up.

## Chat template

Use the reader's own template with `enable_thinking=False`. The empty think block is part of the prompt, not the loss. Loss covers answer tokens plus the end token only.

Rendered golden prompt (`tests/unit/golden/qwen3-1p7b_prompt.txt`):

```text
<|im_start|>user
<image>What color is the shape?<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Golden sha256: `1d91a8fec65d03716b986d047a50d68711028dd97f742f687ef69bee7379c454`.

## Generation defaults

`generation_config.json` ships `do_sample: true`, `temperature: 0.6`, `top_k: 20`, `top_p: 0.95`. Evaluation must override these with greedy or explicitly seeded decoding; the run record stores the decoding settings. `generation_config.json` sha256: `2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2`.

## Verification (WP1)

- `test_reader_config_matches_pinned_checkpoint` passes against the local `config.json`. [V]
- `test_image_pad_token_id` returns 151655. [V]
- `test_special_token_ids` returns eos 151645 and pad 151643. [V]
- `test_chat_template_golden_string` matches the golden file byte for byte and its sha256. [V]
- `test_placeholder_not_in_plain_text` passes. [V]
- `test_text_only_forward_smoke` passes: logits shape `(1, 6, 151936)`, all finite, float32 on CPU. Load 3.37 s, forward 1.89 s, peak RSS 12.3 GiB. [V]

## Open items

none for M0. WP2 owns the placeholder-count and padding tests; WP3 owns the generation path; WP4 owns freezing.