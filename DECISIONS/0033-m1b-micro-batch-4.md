# DECISION-0033: M1b micro-batch 4 after the A10G OOM

Date: 2026-11-15
Status: accepted
Work package: M1b

## Context
The M1b A10G OOMs at micro-batch 8 (19.73 GiB allocated) during the backward. The A10G has 24 GB.

## Decision
- Use **micro-batch 4** on the A10G; global batch 128 becomes 32 micro-batches per optimizer step.
- 2048 steps; the Modal function timeout is 14 hours.
- The 5-step calibration: **0.668 s per 4-example micro-batch**, peak RSS 9.5 GB, max GPU 13.3 GB.
- Projected: 32 × 0.668 s × 2048 ≈ **12.2 hours per seed**.
- The T4 remains unsuitable (OOM at micro-batch 8 and 4); the A10G is the M1b GPU.

## Evidence
- A10G OOM at micro-batch 8: 19.73 GiB allocated. [V]
- A10G micro-batch 4 calibration: 0.668 s, 9.5 GB RSS, 13.3 GB GPU. [V]
- Projected 12.2 h per seed. [I, arithmetic]

## Consequences
The M1b runs on the A10G with micro-batch 4. The three seeds can run sequentially (~36.6 h) or in parallel on three A10Gs (~12.2 h).

## Open items
none