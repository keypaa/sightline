# DECISION-0005: Reader chat template, thinking off, answer-only loss

Date: 2026-11-15
Status: accepted (golden string pinned in WP1)
Work package: WP0

## Context
Training and generation must use the same chat template and thinking switch. Qwen3's template with `enable_thinking=False` adds an empty think block. Baseten put the empty block inside the supervised span; we do not.

## Options considered
1. Reader's own template, thinking off, empty block in the prompt, loss on answer and end token only.
2. Baseten's placement: empty block inside the supervised span.
3. A custom prompt format.

## Decision
Use the reader's own template with `enable_thinking=False`. The empty block is part of the prompt, not the loss. Loss covers answer tokens plus the end token only. A golden-string test pins the rendered prompt.

## Evidence
- M0 breakdown, 2b decision 5: Qwen3 template with thinking off adds an empty think block; Baseten put it inside the supervised span, a recorded deviation. [V, via the document]
- Main document section 7.2: pin non-thinking mode and write it into the run record. [V]

## Consequences
Training and evaluation render the same prompt. The WP2 label mask covers answer tokens plus the end token. The run record stores the chat-template hash and `enable_thinking: false`.

## Open items
- [U] The exact rendered prompt string. Owner: implementation agent. Closing step: WP1 golden-string test and `tests/unit/golden/qwen3-1p7b_prompt.txt`.