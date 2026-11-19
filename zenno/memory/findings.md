# Findings

Running log. Newest first within each section. Evidence tags as in
[README.md](README.md).

## What works

### Logits trim via `logits_to_keep` — 11.5% faster step, numerically identical
Only the answer tokens carry a label, so projecting the vocabulary at all 271
positions wastes a `[128, 271, 151936]` bf16 tensor. Trimming to the last `k+1`
positions gave 2.398 → 2.121 s per step on the A10G, and a tiny-Qwen3
equivalence test showed identical losses to 1e-5 and identical projector
parameters to 1e-6. **[V]**
Why it matters: it is the rare optimization that is both free and provably
neutral.

### Host-side `n_supervised` removes 8 device syncs per step
`int((labels != -100).sum(dim=1).max())` on a GPU tensor syncs once per
micro-batch — 8 per step at micro-batch 4. Computing it on the host when the
batches are built removed 1.2% of step time and a queue drain per micro-batch.
**[V]**
Why: each sync serialises the CPU against the GPU, which is the same overhead
`torch.compile` exists to remove.

### `sdpa` over `eager` for the reader — 10.9% faster
2.383 s vs 2.121 s per step. The reader already defaults to `sdpa`; the
experiment was worth running to confirm we were not leaving it on the table.
**[V]**

### Profile before optimizing
An A10G profile showed **98.9% of the step is the frozen reader's forward and
backward** (27.1% / 71.8%), with projector, splice, loss and optimizer together
under 1.5%. **[V]**
Why it matters: it killed the `torch.compile` plan before any time was spent on
it. The step is GEMM-bound, so kernel fusion has little to fuse.

### A single linear layer reads every answer from the frozen features — 100% held-out
Mean-pool the 256 K3 patch vectors to 4096 numbers, fit one linear layer on 256
train examples, score on 256 held-out val examples: **color 1.000 (chance 0.125),
shape 1.000 (0.25), count 1.000 (0.25)**. Shuffled-label controls land at chance
for all three. **[V]**
**Why it matters:** this is the decisive result. The answers are in the frozen
features in linearly-decodable form. The failure is in the projector-to-reader
path, not in extraction. Full write-up in `reports/m1b-feature-probe.md`.

### A shuffle control is worth building into a probe
It caught an ordering-bias bug that would otherwise have produced a beautiful
100% on a partial label set. **[V]**
**Why:** a control that can fail tells you what an impressive number is actually
measuring.

### Gradient-norm logging is cheap and decisive
`clip_grad_norm_` already returns the pre-clip norm, so recording it costs
nothing. It falsified the "gradients do not reach the projector" hypothesis in
13 minutes: mean norm **829**, max 1264. **[V]**
Why: a hypothesis that can be killed for 13 minutes should never be allowed to
consume a 2048-step run.

### Seed-0-style training converges at `--lr 1e-3`, not at `3e-4`
300 steps at 1e-3 reached a final loss of **0.023** (mean 0.98 over the last 25);
the 2048-step run at 3e-4 oscillated between 0.33 and 4.0 and never settled.
**[V]**
Why: the recipe is not broken, the learning rate is a real lever. Note that a
single low loss value is not convergence — 0.02 was one step, not the average.

### Periodic atomic checkpoints make long runs survivable
`--save-every 200` with a write-to-`.tmp`-then-rename made an interrupted run
resumable and made diagnostics cheap to re-ask. **[V]**

### Colour generalises at LR 1e-3 — 85.5% on held-out test
2048 steps, micro-batch 16, seed 0: colour **0.855** on test against 0.125
chance, with blank, mismatched and no-tokens all at 0.000. Count reaches 0.953
on val. Shape stays at chance (0.260). **[V]**
**Why it matters:** the first real evidence that the frozen retrofit transmits
visual information. The earlier 3e-4 run was undertrained, not broken.
**Do not read the train-split numbers as generalisation**: train accuracy was
1.000 on all three tasks and shape generalises not at all.

## What does not work

### Single-token visual injection — clearly worse
Replacing 256 spliced patches with one mean-pooled token dropped teacher-forced
top-1 from 10/20 to **0/20** at 300 steps, and the loss from 0.023 to 3.82.
Greedy outputs degraded into meta-commentary and non-English tokens
(`'כמובן! כדי לספק תשובה'`), the signature of a reader with no usable visual
input. **[V]** `DECISIONS/0036`.
**Why:** one token is too narrow a channel for three tasks, so this kills the
single-token *design*, not the 256-token path.

### Matching the projector output RMS to the embedding RMS — inert
Setting `output_scale = 0.174` changed the mean loss from 20.56 to 19.84 and the
gradient norm from 829 to 803. **[V]**
**Why:** `Qwen3DecoderLayer.forward` begins with
`hidden_states = self.input_layernorm(hidden_states)`. That is an RMSNorm, so a
global scalar on the input embeddings is divided out at the first layer. The
lever cannot move the loss for this reader. Kept for readers without an input
RMSNorm; see `DECISIONS/0035`.

### Micro-batch 8 on a 24 GB A10G — OOM
OOMs at both micro-batch 8 and 4 before the logits trim, and still OOMs at 8
after it. Peak at micro-batch 4 was 18.6 GB. **[V]**
**Why:** memory is dominated by reader activations across 28 layers, not by the
vocabulary projection. Trimming freed 1.3 GB at micro-batch 4 — real, but not
enough to change the ceiling.

### `flash_attention_2` — redundant here
The reader already dispatches to PyTorch's flash kernels through `sdpa` at head
dimension 128. A separate wheel would add build cost on `sm_120` for no measured
gain. **[V]** for the sdpa default, **[I]** for the redundancy.

### `torch.compile` — low expected value
Profiling showed the step is GEMM-bound, so there is little for the compiler to
fuse. Not tested; deliberately not pursued. **[I]**

### Exact match as the only metric — scores zero on prose
A reader answering `'the shape is red'` scores zero against `'red'`. Seed 0's
1.4% colour score measured format compliance, not vision. **[V]**
Always run `--debug-generations` before believing an accuracy number.

### Slicing a subset from an ordered split
`items[:n]` on the M1 splits is biased because they are ordered by answer. It
gave a probe 16 classes instead of 8 and a chance rate of 6.25% instead of
12.5%. **[V]**
Use a seeded shuffle (`sample_subset`), not a slice and not a fixed stride — a
stride equal to the class period aliases onto a single class. See
`docs/measurement-pitfalls.md`.

### Combining `--no-eval` with `--debug-logits`
`--no-eval` returns before the diagnostic runs, silently skipping it. **[V]**
One or the other, not both.

### Hard-coding the colour chance rate as 1/6
There are 8 colours, so chance is 1/8. **[V]** Derive it from the data.

## Ruled out, with evidence

These are not "didn't work" — they are hypotheses that were tested and killed.
Recording them prevents re-testing.

- **Decoding / KV cache / prompt handling.** Greedy and teacher-forced top-1
  agree in 19 of 20 rows; the one mismatch is a three-way tie at 0.135/0.135/0.135.
  **[V]**
- **Memorisation as the explanation for seed 0.** Train-split accuracy was 0/7
  colour and 1/7 shape, not high. **[V]**
- **Evaluation misalignment.** The eval path shares prompt, splice and feature
  construction with training. **[V]**
- **Weak gradients.** Mean norm 829. **[V]**

## Closed questions

- **Do the frozen features carry the answers?** **Yes.** A linear probe gets
  100% held-out accuracy on colour, shape and count, with shuffle controls at
  chance. See `reports/m1b-feature-probe.md`. **[V]**

## Open

- **[U] Why count falls from 0.953 on val to 0.318 on test.** Train and val use
  counts 1–4, test uses 1–5, so count 5 is never trained; but that alone would
  leave the score near 0.76, so counts 1–4 also degrade. Owner: implementation
  agent. Closing step: per-count accuracy on both splits.
- **[U] Why shape generalises not at all (0.260 on test, chance 0.25) while colour
  reaches 0.855.** Owner: implementation agent. Closing step: a shape-only
  control against a colour-only control.
- **[U] Whether count 5 belongs in the test split** when train and val stop at 4.
  Needs a DECISIONS entry.
- **[U] Whether seeds 1 and 2 reproduce this at 1e-3.** Closing step: two more
  runs, then `tools/g1_report.py` over all three.
### A diagnostic that runs after training must be rehearsed before it

`debug_logits` was called without its new `spec` argument. The container trained
for ~43 minutes and only then raised `TypeError`. The runner now rehearses any
requested diagnostic once on the untrained projector with a single example and
prints `diagnostic self-check ok` before training starts. **[V]**
**Why:** a wiring error in the post-training path is indistinguishable from a
crash until the run is over. Failing in the first minute costs nothing.
