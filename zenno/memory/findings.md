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

### Colour and shape generalise at LR 1e-3 with real seeding
Seed 0, 2048 steps, held-out test: colour **0.8125** (chance 0.125), shape
**0.8066** (chance 0.25), all controls 0.000. Val: colour 0.697, shape 0.625.
Both are *higher* on test than on val. **[V]**
**Why it matters:** first solid evidence that the frozen retrofit transmits
visual information. The earlier seed-0/seed-1 spread (shape 0.260 vs 0.955) was
nondeterminism plus a seed-less init, not a property of the method.
**Do not read the train-split numbers as generalisation**: train accuracy is
1.000 on all three tasks.

### A task can fail on an unseen question phrasing while working on val
Count scores 0.969/0.797/0.617/0.203 for counts 1–4 on val and **0.000 on every
count on test**. Teacher-forced top-1 at the answer position is `'The'` (0.905),
not a digit: the model produces prose for `"Count the shapes."` but digits for
`"How many shapes do you see?"`. **[V]** `DECISIONS/0038`.
**Why:** the three splits use three different templates, so test measures
generalisation to an unseen phrasing. The failure is phrasing-specific behaviour,
not absent visual information.

### A weak class can hide inside a strong average
Magenta scores **0.078** on test and 0.172 on val while the other seven colours
sit at 0.80–0.98. **[V]** A per-task average of 0.81 conceals it; always read
`answer_breakdown`.

## What does not work

### The free opencode verifier quota cannot support a corpus-scale judging run
A 20+20 claim judge probe took **228 s** (~5.7 s/call at `--workers 1`) and the
last two calls returned HTTP 429 "Free usage limit reached. Connect a provider
or subscribe to OpenCode Go...". **[V]** `experiments/m2a_grounding/judge_probe.log`,
2026-11-23. 38 of 40 rows got verdicts.

Measured on those 38 rows: real claims 17 true / 3 false (15% contradicted,
n=20 — far too small for the 0.15 ceiling); planted errors 16 rejected of 18
valid (88.9%, below the 0.90 floor), 1 accepted and 3 `can't-tell`. **[V]**
These are probe-scale numbers, not a gate result; they show the instrument is
too noisy at n=20 and that the quota is the binding constraint.

**Consequence:** a 500-per-layer run needs ~1,000 calls (~95 min) and 1,500 per
layer needs ~3,000 (~285 min), both beyond the free quota. Do not launch one
until a provider decision is recorded. The 38 valid rows are resumable — `judge`
drops `error` rows and re-asks them.

### vLLM serving wedges silently after ~7 h of continuous annotation
The 20k run stopped advancing at 15,480/19,962 rows with no error line in
`m2a_iter20k_full.log`. All three server processes (launcher, EngineCore,
Rust frontend) were still alive under SIGTERM, GPU memory held at 96,281 MiB,
GPU utilisation 0%, and `GET /v1/models` returned HTTP code `000` (no response
at all). Clients blocked on the 900 s per-call timeout in
`src/sightline/m2a/local_transport.py`, so nothing was written and the
rate-limit guard never tripped — no `[retry]` line appeared. **[V]** GPU box,
2026-11-22: `curl -w` + `nvidia-smi` + `ps -p 693,961,982`, captured before
kill.

SIGTERM did not clear it; `kill -9` on all three PIDs dropped GPU memory to
0 MiB, and a clean re-serve reached `server_http=200`. The 15,480 parsed rows
were intact — resume is by parsed row and lost nothing. **[V]**

**Mitigation:** check `curl -s -m 10 -o /dev/null -w '%{http_code}' .../v1/models`
returns `200` before launching annotate, and again if progress stops. Run the
server with `nohup ... &` + `disown` on a single-terminal box so it survives
shell exit; save the PID so the wedged processes can be killed precisely.

**Why this mattered:** the first diagnosis was an O(n^2) annotator check
(`check_single_annotator` rescanned every prior row per write). That was a real
bug and was fixed in `e2db579`, but a quadratic check slows writes — it cannot
explain zero rows and no progress line. Total stall + live processes + no HTTP
response is the server, not the corpus code. **[I]** on that last inference,
supported by the observed metrics above.

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

- **[U] How many shape answers the parser loses to trailing content.** One debug
  example produced `'diamond\n```json\n{\n  "'` with `diamond` at 0.998, scored
  wrong. Owner: implementation agent. Closing step: re-score all three seeds under
  a first-line parser and report both numbers.
- **[U] Why magenta sits at 0.078 while other colours sit at 0.80–0.98.** Owner:
  implementation agent. Closing step: a magenta-only probe.
- **[U] Whether count generalises with one question template across all splits.**
  Closing step: a run with a consistent count template.
- **[U] Whether seeds 1 and 2 reproduce colour and shape at 1e-3.** Closing step:
  two more runs, then `tools/g1_report.py` over all three.
### A diagnostic that runs after training must be rehearsed before it

`debug_logits` was called without its new `spec` argument. The container trained
for ~43 minutes and only then raised `TypeError`. The runner now rehearses any
requested diagnostic once on the untrained projector with a single example and
prints `diagnostic self-check ok` before training starts. **[V]**
**Why:** a wiring error in the post-training path is indistinguishable from a
crash until the run is over. Failing in the first minute costs nothing.
