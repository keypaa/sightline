# DECISION-0023: Training loop, checkpoint and resume

Date: 2026-11-15
Status: accepted
Work package: WP8

## Context
WP8 needs the answer-only objective, optimizer and schedule, gradient accumulation, bf16 autocast, and a checkpoint that resumes exactly.

## Decision
- Answer-only cross-entropy; label `-100` is ignored; a prompt-only sample contributes a zero loss with a gradient path.
- AdamW (betas 0.9, 0.95), weight decay 0.1, base LR 3e-4, warmup 50 steps, cosine decay to 0, gradient clip 1.0.
- Global batch 128, micro batch 8; the loss is divided by the number of micro-batches. The accumulated gradient matches a single 128-batch (verified with 2x4 versus 1x8) within 1e-5.
- bf16 autocast on CUDA; fp32 projector and optimizer state; the reader and encoder are frozen in bf16.
- Checkpoint: projector, optimizer, scheduler, step, and RNG states (torch, cuda, numpy, random).
- Resume: stop at 100, resume, and the next 10 losses match a continuous 110-step run within 1e-6.
- Determinism: the same seed for 5 steps gives identical losses.

## Evidence
- 10 WP8 tests pass: answer-only loss, prompt-only zero, schedule, resume 10 losses, determinism, checkpoint keys, gradient accumulation. [V]
- Full suite: 67 passed, 86.7 s, peak 13.2 GB, 238.4 CPU-s. [V]

## Consequences
The training loop is ready for WP9/WP10. The bf16 autocast path is only exercised on the GPU; the CPU tests use float32. The Baseten reference is constant after warmup; WP8 uses cosine, and the constant schedule remains an ablation.

## Open items
- [U] bf16 autocast on sm_120. Owner: implementation agent. Closing step: WP10/GPU run.
- [U] Real-reader 100+10 resume on the GPU (optional; the tiny test covers the logic). Owner: implementation agent. Closing step: GPU box.