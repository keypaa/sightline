# DECISION-0003: Qwen3 placeholder is `<|image_pad|>` (id 151655)

Date: 2026-11-15
Status: accepted
Work package: WP1

## Context
The placeholder positions receive projector output through `inputs_embeds`, so the token id at those positions is never looked up. Qwen3's tokenizer already contains an image pad token; its id must be confirmed in the pinned tokenizer.

## Options considered
1. Use the existing `<|image_pad|>` token.
2. Add a new special token (changes the vocabulary and the embedding table).
3. Use an unused ordinary token.

## Decision
Use `<|image_pad|>` with id 151655. Add a test that the pinned tokenizer maps the token to that id, and a WP2 test that normal user text cannot contain it.

## Evidence
- `tokenizer_config.json` at Qwen/Qwen3-1.7B revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e lists `151655: <|image_pad|>` in `added_tokens_decoder`. [V]
- M0 breakdown, 2b decision 3: confirm the id in the pinned tokenizer; with `inputs_embeds` the model takes exactly one of `input_ids` or `inputs_embeds`. [V]

## Consequences
No vocabulary change. The embedding matrix is untouched at placeholder positions. The golden splice test feeds the reader's own embedding rows there and must reproduce the plain text forward.

## Open items
- [U] Re-verify the id from the local pinned tokenizer snapshot in the WP1 test (the value above is from the Hub's `tokenizer_config.json`, which is primary but not the tokenizer file itself). Owner: implementation agent. Closing step: WP1 `test_image_pad_token_id`.