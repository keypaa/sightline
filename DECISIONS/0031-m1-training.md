# DECISION-0031: M1 training recipe and Modal sizing

Date: 2026-11-15
Status: accepted
Work package: M1b

## Context
M1b trains the projector on the M1 synthetic data with the Baseten recipe. The local CPU is not suitable; Modal is the test harness.

## Decision
- Recipe: global batch 128, micro batch 8, AdamW (0.9, 0.95), weight decay 0.1, LR 3e-4, warmup 50, gradient clip 1.0, 2048 steps, bf16 autocast, fp32 projector and optimizer state.
- Modal: **A10G (24 GB)**, 4 CPU, 16 GB RAM. The T4 OOMs at micro-batch 8 (13.6 GiB) and micro-batch 4 (13.1 GiB), so A10G.
- The 5-step calibration: **0.633 s per 8-example micro-batch**, peak RSS 9.7 GB, max GPU 18.6 GB. The projected 2048-step time: 16 micro-batches per step × 0.633 s × 2048 ≈ **5.75 hours per seed**.
- The data is generated in the function; the K3 features are precomputed per run; the checkpoints go to the Modal Volume `sightline-m1`.
- Modal is test-only; the production code is local.

## Evidence
- T4 OOM at micro-batch 8: 13.62 GiB allocated. [V]
- T4 OOM at micro-batch 4: 13.13 GiB allocated. [V]
- A10G calibration: NVIDIA A10, 0.633 s per 8-example micro-batch, peak RSS 9.7 GB, max GPU 18.6 GB. [V]

## Consequences
The M1b runs on the A10G; the 2048-step runs are about 5.75 hours per seed. The T4 is not used for M1b. The three seeds can run sequentially (~17 h) or in parallel on three A10Gs.

## Open items
- [U] FlashAttention 2 vs eager on the GPU (the M1b uses eager). Owner: implementation agent. Closing step: GPU parity.