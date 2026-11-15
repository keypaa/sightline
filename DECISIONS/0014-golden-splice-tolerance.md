# DECISION-0014: Golden-splice and padding tolerances

Date: 2026-11-15
Status: accepted
Work package: WP2

## Context
The conformance suite needs frozen tolerances. Thresholds are recorded before the acceptance run, not after.

## Calibration

`tools/calibrate_bf16_tolerance.py` on the pinned Qwen3-1.7B, CPU, one model at a time. Logits are compared at all 6 positions of the prompt; KL is in nats.

| Measurement | fp32 vs bf16 plain | bf16 splice vs plain |
| --- | --- | --- |
| Max absolute logit difference | 1.587961 | 0.0 |
| Max relative logit difference | 33699.9 | 0.0 |
| KL divergence, mean | 0.013842 | 0.0 |
| KL divergence, max | 0.078349 | 0.0 |
| KL divergence, final position | 0.001784 | 0.0 |
| Argmax mismatch | 0 / 6 (0.0%) | 0 / 6 (0.0%) |
| atol needed with rtol = 0 | 1.587961 | 0.0 |
| rtol needed with atol = 0 | 33699.9 | 0.0 |
| Peak RSS, fp32 load | 12,378.1 MB | — |

Other measurements:

| Measurement | Value |
| --- | --- |
| fp32 golden splice | exact (`torch.equal`) |
| Real-reader padding invariance, observed max abs diff | 2.1e-5 |

The max relative difference is dominated by logits near zero and is not a useful tolerance for logits. The meaningful precision checks are the argmax (no change at any position) and the KL divergence (mean 0.014 nats, final position 0.0018 nats). The bf16 splice path is exactly identical to the bf16 plain forward on this reader; the frozen tolerance is a floor for kernel and hardware variation.

## Decision

| Test | Pass criterion |
| --- | --- |
| fp32 golden splice | exact equality (`torch.equal`) |
| bf16 golden splice | max abs logit diff <= `1e-3` |
| Padding invariance | max abs logit diff <= `1e-4` |
| Tiny-model mechanics tests | max abs diff <= `1e-5` |

## Evidence
- Calibration output above, reproducible with `python tools/calibrate_bf16_tolerance.py`. [V]
- `tests/conformance/test_qwen3_conformance_fp32.py` and `_bf16.py` pass at these tolerances. [V]
- `tests/unit/test_backbone.py` uses a tiny Qwen3 model and passes at `1e-5`. [V]

## Consequences
The thresholds are frozen. Changing one needs a new DECISIONS entry. The bf16 tolerance should be rechecked on the GPU in WP10; if the observed difference exceeds `1e-3`, record a new entry rather than loosening it silently.

## Open items
- [U] bf16 golden-splice difference on the RTX PRO 6000 (CPU calibration shows 0.0). Owner: implementation agent. Closing step: run the bf16 conformance test on the GPU and record the result.