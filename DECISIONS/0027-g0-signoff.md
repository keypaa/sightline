# DECISION-0027: G0 sign-off

Date: 2026-11-15
Status: accepted
Work package: WP11

## Context
M0's G0 gate requires the golden splice, generation parity, parameter test, 32-example overfit and resume. All are now evidenced.

## Evidence
- Golden splice: `tests/conformance/test_qwen3_conformance_fp32.py` (exact), `_bf16.py` (<= 1e-3). [V]
- Generation parity: `tests/conformance/test_qwen3_generation.py` (golden splice, cache vs recompute, chunked prefill <= 1e-4). [V]
- Parameter test: `tests/conformance/test_freezing_qwen3.py` (frozen hashes unchanged; only projector gradients). [V]
- Resume: `tests/unit/test_train.py` (100 + 10 vs 110 within 1e-6). [V]
- Overfit: 3/3 seeds, 32/32 by step 125/125/150, controls 0% / 6.25% / 6.25%, vision-minus-blind 100%. [V]
- Report: `reports/g0.md`.

## Decision
Proposed: G0 is signed off. The owner approves by marking this entry accepted; M1 starts only after that.

## Consequences
M0 is complete; M1 (synthetic recognition) can start after the owner signs.

## Open items
- [U] bf16 golden-splice and FA2-vs-eager on the RTX PRO 6000 (batched GPU session). Owner: implementation agent. Closing step: GPU box.
- Closed: K3/K2.6 tower conformance on Modal T4 returncode 0 (all tower tests pass).