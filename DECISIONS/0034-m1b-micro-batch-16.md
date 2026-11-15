# DECISION-0034: M1b micro-batch 16 on the RTX PRO 6000

Date: 2026-11-15
Status: accepted
Work package: M1b

## Context

M1b runs on the RTX PRO 6000 Blackwell (102.0 GB reported by the driver, 97887 MiB
free, `sm_120` in `torch.cuda.get_arch_list()`). The recipe fixes the global batch at
128; the micro-batch is an execution knob that does not change the recipe, so it is a
candidate for optimization. Project rule 9 requires a recorded baseline before any
optimization, one change at a time.

## Measurement

`tools/run_m1.py --probe-steps 20` (2 untimed warmup steps, 20 timed steps), seed 0,
identical data order and projector init:

| micro-batch | step_s | per-example s | peak GPU | peak RSS | 2048 steps |
| --- | --- | --- | --- | --- | --- |
| 8 | 2.2365 | 0.01747 | 21711.3 MB | 10190.8 MB | 4580 s (1.27 h) |
| 16 | 2.1057 | 0.01645 | 32361.2 MB | 9315.7 MB | 4312 s (1.20 h) |

Delta: step time -5.9%, peak GPU +49%, peak RSS -8.6%.

## Decision

- Use **micro-batch 16** for the M1b runs (8 micro-batches per optimizer step,
  96 base batches over 1536 training examples, no partial batch).
- Do not go to 32. The gain from 8 to 16 is already only 5.9% and the per-example cost
  is flattening (0.01747 -> 0.01645 s); 32 would add ~55 GB of GPU memory for a further
  low-single-digit percent.
- The A10G numbers (DECISIONS 0031, 0033) are superseded for training; they remain the
  record of the test harness.

## Numerical equivalence

The two probes see the same 128 examples in the same order and the same init, so the
losses are directly comparable:

- first step: 16.190459 (mb8) vs 16.192950 (mb16), difference 0.0025
- step 20: 22.215820 (mb8) vs 22.212402 (mb16), difference 0.0034

The agreement to ~1.5e-4 relative is the expected float-accumulation-order noise, so
regrouping the global batch does not change the training signal. This is the evidence
that micro-batch is an execution knob here, not a recipe change.

## Evidence

- RTX PRO 6000 Blackwell Server Edition, 102.0 GB, `sm_120`, driver 595.71.05, CUDA 13.2. [V]
- mb8 probe: step_s 2.2365, peak GPU 21711.3 MB, peak RSS 10190.8 MB, elapsed 49.61 s. [V]
- mb16 probe: step_s 2.1057, peak GPU 32361.2 MB, peak RSS 9315.7 MB, elapsed 46.70 s. [V]
- Losses as quoted above, from the two probe run records. [V]
- 2048-step projections are arithmetic on the measured step_s. [I]

## Consequences

The 2048-step M1b run is ~1.2 h per seed at micro-batch 16, so the three seeds are
~3.6 h. The G1 recipe (global batch 128, 2048 steps, AdamW (0.9, 0.95), LR 3e-4,
warmup 50) is unchanged.

## Open items

- [U] The step-to-step loss in the full run is noisy because `cycle=True` serves a
  different 128 examples each step; a fixed-batch eval loss would be a cleaner curve.
  Owner: implementation agent. Closing step: decide whether to log a held-out loss
  every 100 steps in the M1c report.