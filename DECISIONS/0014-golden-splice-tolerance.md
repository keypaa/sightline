# DECISION-0014: Golden-splice and padding tolerances

Date: 2026-11-15
Status: accepted
Work package: WP2

## Context
The conformance suite needs frozen tolerances. Thresholds are recorded before the acceptance run, not after.

## Calibration

`tools/calibrate_bf16_tolerance.py` on the pinned Qwen3-1.7B, CPU, one model at a time:

| Measurement | Value |
| --- | --- |
| Peak RSS, fp32 load | 12,625.8 MB |
| fp32 vs bf16 plain forward, max abs logit diff | 1.587961 |
| bf16 golden splice (plain vs spliced), max abs logit diff | 0.0 |
| fp32 golden splice | exact (`torch.equal`) |
| Real-reader padding invariance, observed max abs diff | 2.1e-5 |

The fp32-vs-bf16 difference is the precision gap, not a splice error. The bf16 splice path is exactly identical on this reader; the tolerance is a floor for kernel and hardware variation.

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