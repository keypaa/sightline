# DECISION-0028: Chunked-prefill tolerance

Date: 2026-11-15
Status: accepted
Work package: WP11

## Context
The conformance suite requires a chunked-prefill test: split the prefill with the KV cache versus one pass, and compare the last-position logits. The Qwen3-1.7B cached path differs from the single-pass path by a small numerical amount.

## Calibration

On the pinned Qwen3-1.7B, fp32, a 256-placeholder prompt split in half:

- max absolute logit difference at the last position: **2.288818359375e-05**
- 3.1% of the 151,936 logits differ; the rest are exact.

The test now uses a 64-placeholder prompt to keep the local peak memory lower; the chunking is still exercised.

## Decision
Freeze the chunked-prefill tolerance at **max abs <= 1e-4** (observed 2.29e-5). The test is `tests/conformance/test_qwen3_generation.py::test_chunked_prefill_matches_single`.

## Evidence
- Observed 2.288818359375e-05 with the 256-placeholder prompt. [V]
- The 64-placeholder prompt is a memory reduction, not a change to the tolerance. [V]

## Consequences
The chunked prefill is accepted within 1e-4. The tolerance is for the fp32 cached-versus-single path; the bf16 path is not tested here.

## Open items
none