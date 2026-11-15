# Sightline — Project Document

*For Keylhan Paumard--André · 8 November 2026 · v1.0 · replaces the draft charter and the Vision-Adapter handoff*

**Evidence tags.** **\[V\]** read on a primary page this session. **\[S\]** search snippet or third-party mirror only. **\[I\]** my inference or arithmetic. **\[U\]** unknown, check before relying on it. Numbers marked \[I\] or \[U\] are planning aids, not facts.

**Status (2026-11-15):** M0 complete; G0 signed off in DECISIONS/0027. M1 next.

## 0. Summary

Sightline is a research and learning project. The question: can a small trained projector give a frozen, text-only open-weight LLM useful vision, and where does that stop? The long-term target is DeepSeek-V4-Flash-0731 with a Kimi MoonViT-class encoder. The immediate goal is different and more important: after the first three milestones you should understand how a multimodal model is trained, with a pipeline you trust and results you can defend.

Three findings reshape the plan:

1. **The earlier probe was invalid.** Qwen3.5-2B is natively multimodal, so it was not a text-only reader. Use Qwen3 (text-only) instead.
2. **DeepSeek V4.1 Flash exists and is natively multimodal.** The text-only DeepSeek you chose is now the older V4-Flash line (0731). That removes the product argument for the hero run, but it gives you a three-way comparison: retrofit (0731) versus continued training on the same architecture (Vision-Exp) versus native training (V4.1).
3. **Baseten published a reproducible recipe** (arXiv 2610.05897) with a text-only Qwen3 ladder (4B to 32B), exact hyperparameters, and a striking dynamic: the loss plateaus, then drops sharply. Your p05 run may have stopped before that transition. \[I\]

### Recommended decisions

| # | Decision | Recommendation | Revisit when |
| --- | --- | --- | --- |
| D1 | Probe backbones | Qwen3-1.7B for plumbing and debugging; Qwen3-4B as the calibration reader (smallest in Baseten’s ladder); 8B and 14B for the scale ladder | After M2 |
| D2 | Vision encoder | MoonViT-V2 for M0–M1; A/B against the Kimi-K2.6 MoonViT in M2 with a decision rule fixed in advance | End of M2 |
| D3 | Recipe baseline | Baseten’s Table 4 as the reference config, varied one factor at a time | Never changed silently |
| D4 | Success metric | Vision-minus-blind accuracy with paired bootstrap CIs, plus loss-gap tracking during training | — |
| D5 | Hero target | Keep V4-Flash-0731 as the text-only target, gated behind a feasibility spike and the scale ladder (M5) | After the spike |
| D6 | Scope | Recognition first, UI grounding after, RL only as a late optional milestone | After M4 |

## 1. Where we stand

### 1.1 Facts I checked

| Item | What I found | Tag |
| --- | --- | --- |
| Qwen3.5-2B | Natively multimodal: early text–vision fusion, vision encoder included, 24-layer hybrid of Gated DeltaNet and gated full-attention layers, Apache-2.0 | \[S\] |
| Qwen3-1.7B | Text-only causal LM, 28 layers, hidden size 2048 (read from a fine-tune repo’s config diff), Apache-2.0 | \[S\] |
| Qwen3-4B | Text-only, 36 layers per a third-party spec list | \[S\] |
| DeepSeek-V4-Flash-0731 | Text in, text out. 284B total and 13B active (from the V4.1 card’s comparison table). Released 31 July 2026, MIT. Safetensors about 167 GB, so the weights are packed at low precision | \[V\] for sizes, \[S\] for the rest |
| DeepSeek-V4-Flash-Vision-Exp | First multimodal V4 model: V4-Flash architecture plus visual modules and continued training. 32-layer vision tower, MIT. Size reported inconsistently (285B in a vLLM recipe, 305B elsewhere) | \[S\]; size not needed for any decision |
| DeepSeek-V4.1-Flash and -1017 | Natively multimodal MoE, 552B backbone, 763B parameters in the safetensors, 1M context, MIT. DeepSeek-ViT trained from scratch plus a two-layer MLP projector, trained jointly from the start of LM pretraining on 45T tokens. The -1017 revision is a minor update with better visual scores | \[V\] |
| MoonViT-V2 (Kimi K3) | 401M parameters, 27 layers, 1024 hidden, patch 14, 2×2 merge giving a 4096-dim token, trained from scratch jointly with the LM. **Your standalone extraction also carries K3’s own 46.1M projector (4096 to 7168)** | \[V\] (your repo card) |
| MoonViT (Kimi K2.6) | About 417M: the extracted tower file holds 833,765,768 bytes of BF16, and Baseten’s paper says 417M. The 466M in Baseten’s blog matches the encoder plus its 49.5M adapter \[I\]. 1152-dim, 27 layers, 4304 intermediate size, 16 heads \[V, config\] | \[V\] |
| Baseten paper (arXiv 2610.05897) | Frozen Kimi K2.6 encoder, about 50M-parameter MLP projector, readers Qwen3 4B/8B/14B/32B, GLM-5.2 and Nemotron-3-Ultra. Full recipe in Appendix A | \[V\] |

### 1.2 Corrections to earlier assumptions

- **Probe model.** Qwen3.5-2B is multimodal (you spotted this). Your p05 shuffled-versus-real gap was measured on a reader that already knows how to consume visual tokens, so it cannot answer the retrofit question in either direction.
- **“No visual capacities.”** True for V4-Flash-0731 only. The DeepSeek family now has two multimodal branches. I could not find V4.1 in my first search; you were right that it exists.
- **“Fits on a single GPU.”** I doubt it. About 167 GB of packed weights \[S\] exceeds a 141 GB GPU, and a 192 GB B200 would have almost no room left for activations and backward. Your REAP work already found that FP4 weights expand to BF16 at load without special kernels, and 284B parameters in BF16 is about 570 GB \[I\]. Plan for a multi-GPU node. -> depends on the GPU used, but yeah if the GPU doesn't support FP4 natively then yes it will dequantize on the fly and will take too much space which is very important to look at.&#32;
- **Charter details.** “304B-class probe” reversed the roles (the probe is the small model). “Batch 8” was probably a micro-batch: Baseten used 128 conversations per step, and a community replication used global batch 64 with micro-batch 8. \[S\]\[I\]

## 2. Purpose, claims and scope

### 2.1 Purpose

1. **Learn** how multimodal training works by building a minimal pipeline whose every part you can explain and test.
2. **Produce evidence** on when post-hoc visual retrofitting works, which capabilities transfer, and where it breaks.

If the two conflict, learning wins in M0–M2 and evidence wins afterwards.

### 2.2 What V4.1 changes

Nobody needs a retrofit to get a vision DeepSeek anymore: V4.1-Flash is open, MIT-licensed and natively multimodal \[V\]. Sightline is therefore a controlled instrument, not a product. That is a legitimate position, and it enables a comparison on one backbone family: **retrofit on 0731**, **continued training on the same architecture (Vision-Exp)**, and **native training from scratch (V4.1)**. If you ever write this up, that comparison is the hook.

**What the three-way comparison can and cannot claim.** The arms differ in data volume, training tokens, training stage, encoder and connector: V4.1 trains jointly from the start of LM pretraining on 45T tokens \[V\], Vision-Exp continues training an existing text architecture \[S\], and the retrofit trains about 50M projector parameters on about 10⁵ conversations \[I\]. So the comparison cannot rank quality or say which route is better. It can show a capability profile: vision-minus-blind per capability (recognition, OCR, grounding, UI state) on one frozen evaluation set, plus a text-regression check, reported per arm. Fix that evaluation set and each arm’s decoding settings before G8, and state in any write-up that the arms are not matched for data or compute.

### 2.3 Claims, in order

| Claim | Statement | A failure means |
| --- | --- | --- |
| C1 | Visual dependence: answers change with the image, not just the prompt | Stop and debug plumbing or data |
| C2 | Generalization to held-out images, layouts and combinations | The projector memorizes |
| C3 | Real-image recognition: Baseten-style QA beats the blind reader | Recipe or encoder problem |
| C4 | UI recognition and OCR on unseen layouts | UI needs different data or resolution |
| C5 | Grounding: pointing at elements | A frozen retrofit may be insufficient |
| C6 | Scale transfer: the ladder, then DeepSeek | Findings are scale-specific |

A failed claim is a valid result and stops downstream spending.

### 2.4 Non-goals until a gate says otherwise

Modifying LM weights or training an encoder (except in the optional M2c ladder), serving infrastructure, distributed-training frameworks, browser agents, parity claims with native models, and any claim about “all model pairs”.

## 3. Primer: how a multimodal model is trained

**The three parts.** An image encoder turns pixels into a grid of feature vectors. A connector (here an MLP projector) maps each feature into the language model’s input-embedding space. The LM then reads those vectors exactly as if they were word embeddings: the sequence is \[image vectors; question tokens; answer tokens\].

**What is trained.** Next-token cross-entropy on the answer tokens only. In a frozen setup the gradient still flows backward through every LM layer to reach the projector, so you pay for a full backward pass even though no LM weight changes: roughly 4N FLOPs per token instead of 6N \[I\].

**Three regimes, and where Sightline sits.**

| Regime | What trains | Examples | Gain | Cost |
| --- | --- | --- | --- | --- |
| Adapter-based | Projector only | BLIP-2, LLaVA stage 1, Baseten’s GLM work | Text behavior untouched, cheap, clean experiment | Capped by what fits through the projector. Baseten saw no multi-image reasoning emerge \[V\], and adapter-conveyed behavior acts like a prompt that can be overridden |
| Joint-based | Projector plus encoder and/or LM (full or LoRA) | LLaVA stage 2, many open VLMs | Stronger OCR, grounding and multi-image | Text drift or forgetting, more compute, harder to attribute effects |
| Native | Everything, from the start | Kimi K3, DeepSeek V4.1, Qwen3.5 | Best results | Enormous cost |

The boundaries blur \[I\]. Vision-Exp is an existing text architecture plus visual modules plus continued training, which is joint-style at near-native scale. Flamingo inserts new trainable layers into a frozen LM, so it is frozen but not projector-only. LLaVA itself has two stages, one in each of the first two regimes.

**What this means for Sightline.** The adapter-based regime is our constraint because it is the only one where the text model stays identical and the experiment stays controlled. We can also measure what that constraint costs. The optional M2c runs an unfreezing ladder at 1.7B or 4B: projector only, then plus LoRA on the LM, then plus the encoder. Each rung is compared on vision-minus-blind and on a text-regression check against the unmodified reader. Freezing is an experimental constraint, not an ideology.

**Why it can work at all.** Baseten’s interpretability section reports that image tokens enter the frozen reader nearly orthogonal to text tokens (cosine 0.02) and merge with the text subspace over depth (0.91 at the last layer). The projector does not write word-like vectors; it writes something the reader’s layers learn to translate. \[V\]

**Why it fails in practice.**

- *Text shortcuts.* The reader answers from priors and ignores the image. Low loss is not evidence of vision.
- *Format cues.* Visual tokens can teach format (“after an image, emit digits”) without carrying content. Your own p05 ablation showed this pattern.
- *Plumbing bugs.* Wrong placeholder expansion, a shifted label mask, padding side, or position ids. Training works, generation does not.
- *Train/inference mismatch.* Teacher-forced loss looks fine, but `generate()` with cached embeddings behaves differently.
- *Scale mismatch.* Projected vectors with a much larger norm than the embedding table can destabilize the first layers. Check the ratio.
- *Stopping inside the plateau.* See section 4.

**How you know it works.** Compare the same prompts with the correct image, a blank image, a mismatched image and no image, and report paired differences with confidence intervals. Everything else is secondary.

## 4. What Baseten’s paper gives us

Baseten’s Table 4 \[V\] is the reference recipe for Qwen3 readers:

| Setting | Value |
| --- | --- |
| Data | 131,072 multi-turn conversations over images from 13 public sources; QA text generated by Kimi-K3 from each image; loss on assistant tokens only |
| Global batch | 128 conversations |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 |
| LR | 3e-4 constant after 50 warmup steps; gradient clip 1.0 |
| Steps | 2,048 (2 epochs); bf16; max sequence 5,120 tokens |
| Projector | About 50M parameters, a per-patch MLP (pre-norm, linear, GELU, linear) |
| Assistant format | Empty reasoning block, so the model answers immediately |
| Init | Affine map from the encoder’s original companion model’s token embeddings to the target reader’s, folded into the projector’s last layer, fit over the shared vocabulary (about 95k tokens for Kimi to Qwen3) |

Findings worth internalizing:

- **Plateau, then a cliff.** Every Qwen3 run sat on a plateau, then fell sharply: 14B went from 0.75 to 0.50 in 27 steps. Transitions came around step 700 for 8B and 14B and around step 1,100 for 32B. MoE readers showed no plateau, but the paper says GLM and Nemotron used different trainers and rewards from the Qwen3 runs, so that comparison is confounded. \[V\]
- **The plateau is not “ignoring the image.”** At the plateau, removing the image raised loss from about 0.85 to 4.01, so the image was already used. The cliff sharpens it. \[V\]
- **Mask and placement barely matter at the end.** In a 2×2 ablation (causal or bidirectional image attention; in-message or prefix placement) all cells reached similar final scores. The default (causal, image tokens in place in the user message) transitioned first, at step 647, versus up to 1,230 for the others. \[V\]
- **Final SFT loss by reader size:** 0.230 (4B), 0.178 (8B), 0.163 (14B), 0.180 (32B). \[V\]
- **Scale saturates quickly.** MMMU-Pro after SFT plus RL: 31.9 (4B), 41.3 (8B), 45.3 (14B), 43.4 (32B); frontier MoE readers about 56.8. Reading approximate values off Figure 2 \[I\]: the blind floor is about 22–26 for the Qwen3 readers but about 40–42 for the MoE readers, so vision-minus-blind is roughly +10 (4B), +19 (8B), +19 (14B), +19 (32B), +14.5 (GLM) and +17 (Nemotron). Most of the gap between the dense plateau and the MoE readers is a higher blind floor, not more vision, which supports D4. Protocols differ across families.
- **Empty reasoning blocks suppress reasoning.** RL on a verifiable image task restored it, and the effect shrinks with reader size: MMMU-Pro before and after RL goes 26.5 to 31.9 (4B), 35.6 to 41.3 (8B), 44.5 to 45.3 (14B) and 44.4 to 43.4 (32B). In both SFT and RL only the projector trains (Table 5, “projector only”), so the LM cannot be damaged; what changes is the behavior the projector induces. \[V\]
- **Limits.** No cross-image reasoning emerges from single-image training. Object Localization was the smallest positive single-image gain over blind (+4.1), and Relative Reflectance was negative (−14.1). Single seeds throughout, one encoder family. \[V\]
- **Compute.** About 4,800 H100-hours, 940 H200-hours and 120 B200-hours across the whole paper, including the 743B runs. \[V\]
- **Targeted training works.** After projector training on 1,000 targeted examples (10 epochs, LR 1e-4), localization rose from 56.4 to 89.2 on the LIVR holdout, but only +15.6 on the BLINK task itself. So “grounding is hard” is too strong; “transfers poorly without targeted data” is closer. This is the reference for M4. \[V\]
- **RL recipe.** 64 prompts × 8 samples per update, LR 1e-5, 150 updates. The task set is 16,384 competition-style math problems, each rendered as an image, with a text prompt that only asks the model to solve the problem in the image. 150 × 64 is about 9,600 problems, so RL does not consume the whole set \[I\]. This is the reference for M6. \[V\]

## 5. Reading your own p05 run in this light

Treat all of this as hypotheses; I cannot see your code.

| # | Hypothesis | Evidence for | Test |
| --- | --- | --- | --- |
| H1 | Confounded reader: Qwen3.5-2B has native visual machinery and a hybrid linear-attention stack, so spliced embeddings may hit position or cache paths you did not intend | Garbage at generation despite good loss | Re-run on Qwen3 (plain RoPE, standard attention) |
| H2 | Stopped before the transition: 4,000 steps at batch 16 is 64,000 samples, about 500 steps at Baseten’s batch of 128. Their transitions came at roughly 650 to 1,230 steps (about 83k to 157k samples). Step equivalence is crude, but it is the right order of magnitude \[I\] | Your digit-NLL gap kept rising (+0.011, +0.047, +0.044, +0.064) | Train past about 150k samples at a matched effective batch |
| H3 | Task too hard first: coordinates from agentic UI data before basic recognition | Median coordinate error 401, 3 of 30 within 100; localization was Baseten’s smallest positive single-image gain (+4.1 over blind), though it is trainable with targeted data | Recognition first (M1, M2) |
| H4 | Data heterogeneity: sources behaved very differently (aitw +0.16, guiact-web-multi 0.00) | Your per-source ablation | Per-source gates before mixing |
| H5 | Small evaluation: 30 generated rows | Wide uncertainty | Denominators of 500 or more |

Your instincts were good: the zeros, shuffled and no-image ablation, per-source gaps and gap-versus-step tracking are exactly the diagnostics this project standardizes. Keep the p05 checkpoints as artifacts. Do not resume them.

## 6. Decisions in depth

### 6.1 Probe backbones

- **Qwen3-1.7B** (28 layers, hidden 2048 \[S\]): fast iteration, standard attention and RoPE, trivial splicing. Used for M0 and M1 and for every debugging session.
- **Qwen3-4B:** Baseten’s smallest reader, so published numbers exist to calibrate against. Used for M2.
- **Qwen3-8B and 14B:** the scale ladder before any DeepSeek spend.
- Verify each checkpoint’s config, tokenizer and chat template (thinking switch) on the primary card, and pin revisions.

| Reader | Layers | Hidden size \[S\] |
| --- | --- | --- |
| Qwen3-1.7B | 28 | 2048 |
| Qwen3-4B | 36 | 2560 |
| Qwen3-8B | 36 | 4096 |
| Qwen3-14B | 40 | 5120 |
| Qwen3-32B | 64 | 5120 |

The projector’s output width must equal the reader’s hidden size, so add it to each config-verification record. The 4B config also ties input and output embeddings \[S\].

### 6.2 Encoder: MoonViT-V2 (K3) versus MoonViT (K2.6)

|  | K2.6 MoonViT | K3 MoonViT-V2 |
| --- | --- | --- |
| Size and width | About 417M, 1152-dim, 27 layers, 4304 intermediate size \[V\] | 401M, 1024-dim, 27 layers, 4096-dim after the 2×2 merge \[V\] |
| Evidence for frozen projectors | Strong: Baseten’s whole recipe plus community replications | Almost none beyond a third-party retrieval projector \[S\] |
| Training | Jointly trained inside K2.6 | From scratch jointly with K3’s LM; Moonshot reports training-stability benefits \[S\] |
| UI relevance | Measured, not assumed: the UI probes in M3 and M4 answer it | Pretraining reportedly includes OCR, grounding with absolute and normalized coordinates, web pages, games, CAD \[S\] |
| Resolution | Native resolution | Native resolution, up to about 3584×3584 \[S\] |
| Tooling | Existing community graphs and mmproj conversions | Re-extracted from K3 at a pinned revision and byte-identical to your earlier extraction \[V\]; the vendored official module gives forward parity by construction (WP5 of the M0 breakdown) |
| License | Modified MIT \[S\] | Kimi K3 License, read in full on 13 November \[V\]; its conditions apply only to Model-as-a-Service businesses above $20M revenue and products above 100M monthly users |
| Init trick | Baseten’s embedding-alignment init was built for it | K3’s 46.1M projector is in your repo. Whether K3 and the reader share enough vocabulary is measured before M2a by counting identical token strings (Baseten’s Kimi-to-Qwen3 pairs shared 95,369 \[V\]); the init stays an ablation, not a default |

**Recommendation.** Do not decide by argument. Use V2 in M0 and M1 (your tooling, synthetic data, the encoder is not the variable). In M2, run both encoders under identical settings on Qwen3-4B. The K2.6 MoonViT is also your **reproduction control**: if your pipeline cannot reproduce Baseten’s qualitative result with their encoder, the bug is in your pipeline, not in the encoder choice. Pre-register the rule: pick the encoder with the higher paired held-out gap (vision-minus-blind) averaged over three seeds; a tie goes to V2 because it matches the long-term target.

### 6.3 Projector and initialization

- Start with Baseten’s shape: per-token pre-norm, linear to a hidden width, GELU, linear to the reader width. For MoonViT-V2 the input is 4096 per merged token. Derive the parameter count from config rather than fixing it. The charter’s 67.1M equals 2²⁶, about 4×4096², which would fit a 4096→8192→4096 MLP, but that is only my arithmetic \[I\].
- Initialization is an ablation, not a default: random versus Baseten-style embedding alignment. Every Qwen3 run used the embedding-alignment init and still plateaued, and I see no init-versus-random ablation in the paper, so a benefit is plausible but not shown. The fit residual is large (relative MSE 0.77–0.89, cosine 0.36–0.50) \[V\]. The MoE readers used different trainers, so they say nothing about init.
- Log the ratio of projector-output norm to embedding-table row norm at step 0 and every 100 steps.

### 6.4 Hero target and licenses

- **V4-Flash-0731** remains the cleanest text-only DeepSeek: MIT, 13B active.
- **Gate it.** The risk is engineering, not FLOPs: backward through packed FP4/FP8 MoE weights, activation memory, and kernels. Baseten had to add input-gradient formulas to quantized inference kernels for its interpretability work \[V\]. Run a one-day feasibility spike (section 12) before committing.
- **Native baselines.** V4.1-Flash(-1017) and Vision-Exp enter M5 as comparison rows, not targets.
- **Licenses.** DeepSeek MIT \[V\]; Qwen3 Apache-2.0 \[S\]; K2.6 Modified MIT \[S\]; the K3 License has commercial thresholds, so read the actual text before releasing any derived artifact. Record dataset licenses per source from day one.

## 7. Technical design

### 7.1 The token contract (the most expensive thing to get wrong)

- One image per sample for M0 to M3. Fixed resolution at first so the visual-token count is constant: M0 to M2 use a 448×448 canvas, which MoonViT-V2’s 28-pixel merged cells turn into 16×16 = 256 tokens \[I\]. Baseten’s interpretability images use a long side of at most 448 pixels \[V\]; Baseten’s paper does not state its training resolution; this is a recorded deviation (M0 breakdown, 2b), not an open question. Resolution becomes the studied variable at M3 (448, 672 and 896 pixels, which are 256, 576 and 1,024 tokens \[I\]).
- The user message contains a single placeholder token that is expanded to `n_img` slots before embedding. `n_img` is computed from the processor, never hard-coded.
- Slot positions receive projector outputs. All other positions receive the reader’s own embedding rows.
- Attention is causal everywhere (Baseten’s default, and it transitioned fastest). Positions are standard 1D positions over the full spliced sequence.
- Labels are −100 everywhere except the assistant answer tokens (and the end-of-turn token).
- Pick a padding side once, test it, and use the same code path for training and generation.

### 7.2 Prompt format

- Use the reader’s own chat template. For Qwen3, pin non-thinking mode (empty reasoning block) in training and evaluation, and write that into the run record.
- Keep answers short in M1. Longer descriptive answers arrive in M2, following Baseten’s data style.

### 7.3 Loss and live diagnostics

Answer-only cross-entropy. In addition, every 100 steps on a fixed 256-sample probe batch, log the **image-removal gap** (loss with no image minus loss with the correct image) and the **shuffle gap** (loss with a shuffled image minus loss with the correct one). This is your p05 digit-NLL gap, generalized and made standard. A rising gap on a flat loss is the earliest sign of the transition. Do not treat the absence of a plateau in M1 as a bug: the plateau-then-cliff comes from Baseten’s data mix \[V\], and trivial synthetic tasks may transition much earlier or never plateau \[I\].

### 7.4 Optimization defaults (hypotheses to vary one at a time)

| Setting | Start value | Source |
| --- | --- | --- |
| Global batch | 128 sequences via gradient accumulation | Baseten |
| Optimizer | AdamW (0.9, 0.95), weight decay 0.1 | Baseten |
| LR | 3e-4 constant after 50-step warmup; then try 5e-4 and 1e-3 | Baseten, your p05 |
| Gradient clip | 1.0 | Baseten |
| Precision | bf16 activations; projector weights and optimizer states in fp32 | your p05 practice |
| Steps | Well past the transition; minimum 2,048 | Baseten |
| Seeds | 3 for any claim | — |

### 7.5 Feature caching

The 401M encoder is frozen, so its output can be cached; the projector must never be cached. Estimate for a merged 4096-dim token in bf16 \[I\]: 8 KB per token, so at 256 tokens per image (448×448) about 2 MB per image and about 270 GB for 131k images; at 1,024 tokens (896×896) about 8 MB and about 1.1 TB. Cache only when profiling shows the encoder is the bottleneck, key every entry by preprocessing version, and test cached against live features for equality. Otherwise run the encoder online; it is small next to the reader’s forward and backward.

### 7.6 Injecting the projector output into each reader

The projector output always replaces the embedding rows at the placeholder positions, but what else must change depends on the reader. Qwen3 is the best-supported case. DeepSeek-V4 adds three mechanisms (hyper-connection streams, token-id routing, hybrid attention) that a plain splice may ignore. Treat every row as a hypothesis to test in M0 or M5b, not a settled fact.

| Reader | How the image enters | What else can matter | Evidence |
| --- | --- | --- | --- |
| Baseten’s readers (Qwen3 4B–32B, GLM-5.2, Nemotron-3-Ultra) | Projected patches replace the placeholder inside the user message and are concatenated with the embedded question as if they were token embeddings. Causal attention, the reader’s chat template, an empty reasoning block | Prefix placement and bidirectional image attention reach similar final scores but delay the loss transition (step 647 versus up to 1,230 at 14B). Implementation details for GLM and Nemotron are not in the paper | \[V\] paper section 2 and Appendix D; GLM and Nemotron implementation details are not needed for this plan |
| Qwen3 dense (1.7B to 32B) | Splice at the embedding output through inputs\_embeds, plain 1D RoPE, full attention in every layer (4B config) | Projector output width equals the reader’s hidden size (section 6.1). The 4B config ties input and output embeddings. Pin the chat template’s thinking switch | \[S\] public configs; inputs\_embeds worked on 1.7B in your earlier session, to re-verify in M0 |
| Qwen3.5-2B (the old probe) | Native multimodal reader with its own visual path and a hybrid linear-attention stack | Position and cache paths differ from a plain decoder, so p05 may have crossed them (H1) | \[S\]; H1 is \[I\] |
| Qwen3-VL (native reference) | Visual features are scattered into inputs\_embeds at the image\_token\_id positions with masked\_scatter; interleaved MRoPE gives them 3D positions. The forward pass accepts exactly one of input\_ids or inputs\_embeds, and the placeholder mask is built from input\_ids when given | DeepStack also adds visual features to the hidden states of the first three LLM layers, a channel a one-point injection lacks. A secondary summary says its first stage trains the merger only, with encoder and LLM frozen | \[V\] HF modeling-code excerpts (masked\_scatter, DeepStack scatter at the same positions) and technical report (MRoPE); \[S\] merger-only stage |
| DeepSeek-V4-Flash-0731 | Embeddings should enter as inputs\_embeds, before the 4-stream hyper-connection expansion; the vision variants’ vLLM wrapper does it this way | The first 3 of 43 layers \[V config\] route experts by a hash of the token id, so the id at each image position decides which experts it hits. Attention mixes sliding-window, compressed sparse and heavily compressed layers. The official reference code uses its own sparse-attention kernel, not stock FlashAttention \[V\]; your REAP notes say the HF-style path requires FlashAttention 2 (section 12) | \[V\] official 0731 config, vLLM wrapper source, WebBrain’s 0731 retrofit card (section 7.7); hash routing \[S\]; consequences \[I\] |
| DeepSeek-V4.1 and Vision-Exp (native) | Four learned vectors (image start, pad, newline, end) fill the structure positions; patch tokens are reordered into a block layout | The router receives raw token ids and applies a vision bias (bias\_vl) to image tokens. Sliding-window attention is widened over image spans, and the placeholder is padded to the compressor’s alignment. The text-only 0731 has none of these | \[V\] vLLM wrapper source; the 0731 config has no vision fields \[V\]; the checkpoint index has no vision or routing-bias tensors of any name (72,317 tensors, checked 14 November) \[V\] |
| Kimi K2.6 and K3 (the encoder’s own readers) | K2.5 and K2.6 \[V\]: the chat template emits media\_begin, the word image, media\_content, one media\_pad (id 163606 by default) and media\_end, and each pad expands to (h/2)×(w/2) tokens. K3 is not a reader in this plan, so its format is not needed | Matters because the encoder was trained with that format | \[V\] K2.5 and K2.6 chat template, vLLM and third-party code |

**Axes to test.**

- **Placement and mask.** Baseten’s 2×2 grid covers this for Qwen3 \[V\]. Default to in-message placement with causal attention.
- **Structure tokens.** Bare patches versus start, end and row-end markers, fixed or learned. Native V4.1 uses them \[V\]; untested for retrofits \[I\]. WebBrain uses bare patches \[V\]; Baseten’s paper does not say whether it uses markers, so marker use is tested in M1; a community GLM clone wraps the placeholders in the model’s own begin and end image tokens \[S\].
- **Scale.** Ratio of projector-output norm to embedding-row norm, already logged (section 6.3).
- **Positions and token order.** Plain 1D versus the reader’s native scheme; row-major versus the encoder’s merge order. Whatever is chosen must be identical in training and generation.
- **Injection site.** Input only (the baseline) versus DeepStack-style addition into early layers. This needs hooks and is a later experiment.
- **Token-id side channels (V4 only).** One placeholder id versus several reserved ids across image positions, which would spread hash routing. WebBrain’s 0731 retrofit uses a 64-ID palette for this reason \[V\]; the effect is measured in the M5b ablation.
- **Generation path.** Embeddings computed once, cache semantics, and the same code path as training.

**What M0 must produce.**

1. A reader-adapter interface (embed, splice, positions, mask, generate) with one implementation per reader family, all passing the same conformance tests: golden splice, generation parity, frozen-weight hash.
2. One injection card per reader in DECISIONS/: injection point, ids used, positions, mask, side channels, config values and evidence tags.
3. For DeepSeek-V4-Flash-0731, a reading task, not a run. From the checkpoint’s official inference code \[S\], answer how it consumes input\_ids when inputs\_embeds is supplied, whether the hash layers need ids, and how hyper-connections ingest the embedding. Also read WebBrain’s routing code and palette config (section 7.7) to see how its 64 IDs were chosen. Largely answered in section 7.7 (12 November); what remains is the palette file and an autograd-capable implementation. Finish this before renting anything for M5b.

### 7.7 Prior art and verification log (12 November 2026)

**Prior art that matters.**

- **WebBrain’s DeepSeek-V4-Flash-0731-Vision** (third party, experimental, not validated on this checkpoint by its own card \[V\]). Frozen K2.6 MoonViT-3d tower (329 tensors, 833,765,768 bytes, about 417M parameters \[I\]) and a trained PatchMerger projector (LayerNorm, 2×2 merge, Linear 4608→4608, GELU, Linear 4608→4096; 40,119,040 parameters), with the text backbone frozen. Image positions get deterministic routing IDs from a 64-ID palette so the hash-routed layers behave. One literal \<image> marker, at most 512 merged image tokens, served through a patched SGLang rather than stock Transformers \[V\].
- **Baseten’s GLM-5.2-Vision.** K2.6 MoonViT with a PatchMerger 1152→4608→6144 of about 49.5M parameters, up to 4,096 image tokens per image, served through an out-of-tree SGLang plugin \[V\]. The plugin code is not needed for this plan. A community clone (44.8M parameters) replaces <|image|> (id 154854) inside <|begin\_of\_image|> and <|end\_of\_image|> and normalizes click coordinates to \[0, 999\] \[S\].
- **What follows.** K2.6’s encoder emits 1152-dim patches before the merge, and the merge lives inside the projector \[V, two model cards\]. Decide where our 2×2 merge lives and record it. A V4-Flash retrofit already exists, so Sightline’s value is the controlled, instrumented study (ladder, diagnostics, honest failure analysis), not being first \[I\].

**What the two retrofits teach about injection.**

- **A plain inputs\_embeds path can silently do nothing.** At WebBrain’s pinned SGLang commit, the native DeepSeek V4 model accepted input\_embeds but ignored it at the embedding site. Their patch changes that one assignment and keeps input\_ids, so hash routing still works \[V, commit diff\]. For Sightline, the V4 conformance test must prove that changing the injected embeddings changes the logits \[I\].
- **The encoder side is reusable.** The tower and merger code comes from SGLang’s Kimi K2.5 model, so K2.6’s encoder follows the K2.5 architecture \[V\]. That code is a ready parity reference for the K2.6 extraction in M0 \[I\].
- **Image positions cycle through a fixed 64-ID palette** stored in the config. Their V4-Flash sibling card describes training sequences of up to 2,048 tokens with at most 512 merged image tokens \[V\].
- **Routing depends on position.** A second graft (jarrelscy, which says it copies WebBrain’s scheme) states that the IDs at image positions decide which experts process image tokens in layers 0 to 2, and that the palette rewrite is position-dependent, so an image prompt must stay in one prefill chunk in that vLLM graft \[S, search snippet\]. WebBrain’s SGLang routing code avoids the limit by computing the palette phase from the absolute position inside the image span, so chunked prefill yields identical IDs \[V\]. Add a chunked-versus-unchunked parity test \[I\].
- **Not found:** how the 64 IDs were chosen. The code and config show how the IDs are used (below) but give no selection criterion, and the palette file itself (checked 12 November) is a bare list of the same 64 IDs with no metadata \[V\]. The jarrelscy graft, which might document the method, is no longer public. The method is only partly reconstructed from the hash tables (below) \[I\]; the exact procedure stays unknown but is not needed, because our own palette replaces it. Candidate schemes for a small M5b ablation: one placeholder ID, the 64-ID cycle, and fixed random IDs per position \[I\].

**Read from WebBrain’s code and the official reference (12 November).**

1. **The official reference code answers the input\_ids questions** \[V\]. `Transformer.forward(input_ids)` embeds the IDs, then repeats the result 4 times along a stream axis (hc\_mult 4). Every block receives input\_ids and passes them to the MoE gate. Only the first 3 layers use them: the expert indices are `tid2eid[input_ids]`, a fixed, non-trainable int32 table of vocabulary size × 6 experts. So the expert set in layers 0 to 2 depends only on the token ID, while the mixing weights still come from the hidden state. There is no inputs\_embeds argument: injection means replacing the embedding call while still passing IDs, which is WebBrain’s one-line patch.
2. **The reference is inference-only** \[V\]: forward runs under inference\_mode with custom FP4 and FP8 kernels. Backward passes in M5b and M5c need another implementation, such as HF Transformers or NeMo AutoModel \[I\].
3. **WebBrain’s injection recipe** \[V\]. The prompt has one literal `<image>` marker and no begin or end markers. The processor expands it into n copies of the sentinel ID 129280 (one past the 129,280-token vocabulary). Embeddings at those positions come from the projector, and routing IDs there become `palette[(position − image_start) mod 64]`. This happens in prefill only; decode keeps ordinary IDs. Attention is stock: there is no widening of the sliding window over image spans, unlike the vision variants. The example prompt closes thinking with `</think>`.
4. **Image budget** \[V\]: the patch limit is 512 × 2 × 2 = 2,048 patches of 14 pixels, about 0.4 megapixels (roughly 633 × 633) at most \[I\]. Our 448 × 448 canvas (256 tokens) is half of that cap.
5. **Encoder and projector, checked against the config** \[V\]. K2.6 MoonViT has 27 layers, hidden size 1152, intermediate size 4304, 16 heads, patch size 14, a learned 64 × 64 position grid and a 2×2 merge. The projector’s 40,119,040 parameters equal LayerNorm(1152) plus Linear(4608→4608) plus Linear(4608→4096), with biases, which I reproduced exactly \[I\]. So the merge sits inside the projector after the pre-norm. The same dimensions give about 417M encoder parameters \[I\].
6. **The palette decoded with the 0731 tokenizer** \[V\]. IDs 0, 1 and 2 are begin, end and pad. IDs 8, 9, 10, 12, 74, 81, 110, 114 and 240 are single characters. The other 52 are ordinary subwords below ID 128,000. There is no semantic pattern, which fits a selection by expert assignment, and the hash tables confirm it (see the hash-table results below) \[V\]. The palette file is named for the base V4-Flash revision 60d8d707, but the palette is tuned to the 0731 hash tables (below), so the name does not point to a mismatch \[V\].
7. **Attention geometry** \[V config\]. Layers 0 and 1 use sliding-window attention only (window 128); deeper layers alternate compression ratios 4 and 128. In layers 0 and 1, image patches see only the previous 128 tokens. For sequences up to 2,048 tokens, ratio 4 gives at most 512 compressed entries, so the indexer’s top-512 selects everything and selection only starts to matter at longer contexts \[I\].
8. **Co-adaptation** \[I\]. WebBrain calls its projector routing-aware, so it was presumably trained with palette routing, and changing the palette would need retraining. Not verified.

**Also checked (12 November).**

- **The staging script adds no routing logic** \[V\]. It validates that every palette ID is below the vocabulary size, copies the palette into the config, sets the image sentinel to the vocabulary size (129,280), and appends the tower and projector tensor names to the index. The launch script uses tensor parallelism 4, a 4,096-token context, CUDA graphs disabled and one image per request, with FlashInfer FP4 backends on Blackwell.
- **The official checkpoint index** \[V\]: 48 shards, 72,317 tensors and 166,878,536,440 bytes in total, which matches the roughly 167 GB of packed weights quoted in section 15. The three hash tables sit in separate shards: layer 0 in shard 2, layer 1 in shard 3 and layer 2 in shard 4 (`layers.N.ffn.gate.tid2eid`). The embedding is in shard 1 and the output head in shard 45. Each table is small, so it can be read with HTTP range requests instead of downloading the shards.
- **Planned analysis once the tables are in hand** \[I\]. Done for 0731 and for the base V4-Flash tables, which are identical (next list).

**Hash-table results for 0731 (13 November).**

- **The tables** \[V\]. Layers 0 to 2 each hold an int64 table of 129,280 × 6. Every row has 6 distinct experts in 0 to 255, and vocabulary usage is balanced (about 3,030 token IDs per expert, spread about ±5%). The three layers are independent of each other (0.4% of entries equal, which is chance level), so a check that the layers “match” is expected to say False. A scan of all 43 layers found no table in layers 3 to 42.
- **The sentinel cannot route** \[I\]. The table has exactly vocabulary-size rows, so the image sentinel ID 129,280 is out of range. Any routing ID must be below it, which is why a palette or a single in-vocabulary ID is needed.
- **A single ID is a poor spread** \[V\]. With one ID, all 256 image positions hit the same 6 experts in each layer: 250 of 256 experts are never used and the busiest expert gets 42.7 times the mean load.
- **WebBrain’s palette is optimized for these tables** \[V\]. It reaches 246, 243 and 245 of 256 experts per layer, where random 64-ID sets average about 200 and never exceeded 221 in 20,000 draws. No expert is hit more than 3 times in any layer, while every random set in 5,000 draws exceeded that. The first 16 IDs have pairwise-disjoint expert sets in all three layers, and the first 12 are exactly what a greedy rule (the lowest ID that adds the most new experts) produces. The later IDs fit a greedy choice from a sparse candidate pool of roughly 10³ IDs \[I\]. The palette therefore matches the 0731 tables, which settles the worry about its file name.
- **Our own palette** \[V\]. A greedy search over IDs below 128,000 with a cap of 3 hits per expert gives 64 IDs that cover all 256 experts in all three layers with a maximum of 3 hits each (script make\_palette.py, output palette\_cap3\_0731.json). I recommend it as the default for M5b, because we know exactly how it is built, while WebBrain’s provenance is unknown \[I\]. Keep WebBrain’s palette, a single ID and a random-ID set as ablation arms. Whether any scheme changes image quality is measured in the M5b ablation.
- **Base comparison** \[V\]. The base V4-Flash and 0731 tables are identical in all three hash layers (checked 13 November on the shared arrays, 0 differing entries out of 774,720 per layer \[V\]). A palette built for one therefore applies to the other. Which checkpoint WebBrain used to build theirs does not matter for the tables; what remains unknown is only its selection method and its effect on quality.

**Verification log.**

| Claim | Status | Evidence |
| --- | --- | --- |
| 0731 has none of the vision machinery | Confirmed for the config: hidden size 4096, 43 layers, hc\_mult 4, 3 hash layers, sliding window 128, vocab 129,280, no vision fields. The Vision-Exp card says 0731 ignores multimodal elements. The checkpoint index has no routing-bias or vision tensors of any name (72,317 tensors checked) | \[V\] official config and checkpoint index |
| Placeholder ID matters on V4 | Supported by two independent 0731 retrofits, not proven. The official reference routes hash layers by token id, and both retrofits use WebBrain’s palette. How the palette IDs were chosen is not in what I read, and I found no ablation | \[V\] reference code excerpt, WebBrain card; effect size measured in M5b |
| K2.6 insertion format | Confirmed (see table in 7.6) | \[V\] chat template, vLLM and third-party code |
| K3 insertion format | Not needed: K3 is not a reader in this plan | Not applicable |
| GLM implementation | Partly: architecture and sizes from Baseten’s card; plugin code unread; token layout only from a community clone | \[V\] card; \[S\] clone; plugin code not needed |
| K3 License | Read in full (52 lines): MIT-style grant over the Software, which includes the weights; §2 Model-as-a-Service above $20M revenue over any 12 months needs a separate agreement; §3 attribution above 100M monthly users or $20M monthly revenue; §4 internal use exempt; §5 as-is, covering outputs, with no restriction on them. K2.6 is Modified MIT. The MoonViT-V2 weights sit in the K3 checkpoint, so they are covered \[I\] | \[V\] full license text; \[I\] weights scope |
| RTX PRO 6000 memory | Confirmed: 96 GB GDDR7 ECC, 1,792 GB/s, 600 W (Workstation Edition); the Max-Q variant is also 96 GB. Speed relative to an H100 not measured | \[V\] NVIDIA; speed measured in M0 (WP10) |
| Qwen3-VL placeholder detail | Confirmed: masked\_scatter at image\_token\_id positions (151655), DeepStack features scattered at the same positions; HF accepts exactly one of input\_ids or inputs\_embeds | \[V\] HF modeling-code excerpts |
| K2.6 encoder size | Model card says 400M, Baseten’s paper says 417M, and 833.8 MB of BF16 is about 417M; use 417M and verify from the checkpoint in M0 | \[V\] paper; \[I\] arithmetic |

## 8. Data plan

### 8.0 What data, and why

| Stage | Data | Why this data | Status |
| --- | --- | --- | --- |
| M1 | Synthetic rendered scenes: color, shape, count, relative position, short words | Exact labels, balanced answers and held-out compositions make blind chance exact | Generated by Sightline; nothing inherited |
| M2a, M2b | Real-image multi-turn QA in Baseten’s shape: public image sources, QA written by a strong VLM, 131,072 conversations (20k–40k for iteration) | Reproducing a published recipe means a failure points at the pipeline | To build; each source’s license is recorded before download (M2 entry criterion, section 16) |
| M3 | Synthetic HTML screens, then real screenshots | Known DOM boxes give exact labels; real screens test transfer | Synthetic to build; real screenshots come from keypa/vision-adapter-images, re-audited |
| M4 | Grounding data, synthetic first, then audited real UI sources | Localization transfers poorly without targeted data; Baseten’s 1,000-example setup is the reference | Coordinate convention decided at M4 |
| M6 | OpenR1-Math problems rendered as images | Rewards that cannot be earned without reading the image | Apache-2.0 split used by Baseten \[V\] |

**Decision, 10 November 2026: Sightline starts from a clean repository and inherits no preprocessed data.** The old embedding cache (keypa/vision-adapter-embeddings) is not used. My view is that dropping it is right \[I\]:

- It is derived data, MoonViT-V2 outputs for about 139k images (about 949 GB), and can be recomputed from the source images.
- Its validity depends on the standalone encoder matching the official code, which was never checked; the weights now match (13 November), but the cache is dropped anyway.
- The dataset card’s own numbers do not reconcile. A cap of 300k pixels at 28-pixel merged cells implies at most about 380 tokens per image, yet the card reports a p99 of 5,520 and a maximum of 16,653 tokens. I cannot tell from the card why.
- The card lists the license as unknown.
- An unverifiable preprocessing step would make any Sightline negative result ambiguous.

**Keep:** the source images (keypa/vision-adapter-images), which cannot be regenerated, and the p05 checkpoints as artifacts only. **Not inherited:** manifests, grids, labels and join logic. UI sources are re-audited one by one (section 8.3).

**Optional autopsy before deleting.** Copy one shard first (about 9 GB \[I\]; the card calls the first two shards smoke shards). After the M0 encoder parity test, recompute those rows with the new pipeline and compare. A match validates the old cache and narrows p05’s failure to the projector or injection path; a mismatch locates an old bug cheaply. One part is already settled (13 November): the old standalone MoonViT-V2 weights (keypa/MoonViT-V2-Standalone) have 165 tensors that are byte-identical to K3 shard 96 at revision f831ab66814297da540d832a5235f8e904f29d06 \[V\]. Weight extraction is therefore not a suspect for the old cache; the old encoder code and preprocessing remain unchecked \[I\].

### 8.1 Synthetic recognition data (M1)

- Programmatically rendered single-object and few-object scenes: color (6 to 8 classes), shape (4 to 6), count (1 to 5), relative position (left, right, above, below) and short rendered words.
- Templated prompts with several paraphrases. **Balance answers per prompt** so the best text-only strategy is exactly chance. Then the blind and majority baselines are chance by construction.
- Hold out whole compositions (for example, a red triangle never appears in training), plus backgrounds and positions, so generalization is measured on unseen combinations.
- Include matched pairs that differ in one property, and identical prompts with different correct answers.
- Every sample records generator version and seed, labels, split and template id.

### 8.2 Real-image QA (M2)

- Follow Baseten’s shape: images from public sources, multi-turn QA text generated by a strong VLM, loss on assistant turns only. They used 13 sources, 131,072 conversations and Kimi-K3 as the annotator. \[V\]
- Iterate on 20k to 40k conversations, then scale to 131k for the reproduction run (G2).
- Annotator options: Kimi K3 (as they did) or DeepSeek-V4.1-Flash(-1017), which is MIT-licensed and multimodal. Terms and cost are checked on a 100-example pilot before the full run (section 16). Record the annotator model, revision and prompt for every example. Use one annotator for both arms of M2b and never switch mid-comparison. An annotator that is also a G8 comparison row (V4.1) or that belongs to the encoder’s family (K3) leaves style traces in the data, so name that in any write-up.
- Candidate image sources, each verified for availability and license before download (section 16): COCO and Flickr-style photos, document, chart and text-in-image sets, and aggregated instruction collections such as The Cauldron or PixMo.
- Exclude every MMMU-Pro and BLINK image from training. Split train and validation by source and by perceptual hash.

### 8.3 UI data (M3, M4)

- **Synthetic screens first.** Render HTML/CSS pages with known DOM boxes, themes, resolutions and locales. Labels: element category, state (enabled, selected, disabled), visible text and bounding boxes. Ground truth is exact.
- **Real screenshots second.** Re-audit your earlier agentic sources one by one. Their gaps ranged from +0.16 to 0.00, so treat them as separate datasets, and drop or repair any source whose gap stays at zero after the join check.
- **Evaluation sets** for grounding: an established screenshot-grounding benchmark family such as ScreenSpot \[U: confirm current versions and licenses\], never mixed with training sources.
- **Coordinates** are decided at M4, not before. Integers on a normalized \[0, 1000) grid with a top-left origin is the convention used in one community retrofit \[S\]; K3 and V4.1 may use others, so check each model’s grounding convention. Keep the schema field open.

### 8.4 Hygiene rules

Every sample records source, revision, license, split, preprocessing version, prompt and answer. Split by template, source and time, not by image. Remove exact and perceptual duplicates across splits. Validate coordinates against image size. Hash manifests, and keep large files outside git.

## 9. Evaluation and gates

### 9.1 Conditions (every evaluation, identical prompts)

1. Correct image.
2. Blank image of the same size.
3. Mismatched image (only pairs whose correct answers differ).
4. No visual tokens and no placeholder (the blind reader).
5. Untrained projector.
6. Optional stress tests: shuffled patches, downscaling, noise (Baseten saw losses only at severe corruption).

Report blank and no-tokens separately. In p05 they behaved very differently from each other and from real images.

### 9.2 Statistics

Paired comparisons only. McNemar for binary outcomes, paired bootstrap (at least 10,000 resamples) for continuous ones. At least three seeds for any claim. Publish denominators and the parse-fallback rate. Single-seed results are exploratory.

### 9.3 Public benchmarks

MMMU-Pro and BLINK, following Baseten’s protocol: generative evaluation with the official parser, seeded random fallback counted and reported, and a blind control with identical settings. Report vision-minus-blind, not absolute scores alone: blind floors differ by family (see section 4). For SFT-only small readers, make a perception-style, likelihood-scored evaluation the primary measure, as Baseten used for BLINK in Table 3, and report MMMU-Pro as secondary, because generative non-thinking MMMU-Pro is a weak instrument there (about 33 at 14B, Table 13).

### 9.4 Gates (thresholds are placeholders; freeze them before each run)

| Gate | After | Pass criteria |
| --- | --- | --- |
| G0 Plumbing | M0 | Golden splice test passes; generation parity holds; parameter test passes; a 32-example overfit reaches 99% exact match within 500 steps; resume test passes |
| G1 Synthetic dependence | M1 | On at least 500 held-out items per task (color, shape, count): correct minus the best of blank, mismatched and no-tokens is at least 30 points on 2 of 3 tasks, with McNemar p below 0.01 and all 3 seeds positive. Control conditions sit within the 95% CI of chance |
| G2 Reproduction | M2a | Baseten recipe with the K2.6 encoder on Qwen3-4B: loss descends to the order of 0.2 (absolute values will differ with data); vision-minus-blind is positive with a paired CI that excludes zero on the perception-style likelihood-scored evaluation (section 9.3, the primary measure); the full MMMU-Pro set is reported as secondary and descriptive, because it is a weak instrument for SFT-only small readers |
| G3 Encoder A/B | M2b | Pre-registered rule from section 6.2 |
| G3b Unfreezing ladder (optional) | M2c | Each rung reports vision-minus-blind with a paired CI and a text-regression check against the unmodified reader. A finding about the cost of freezing, never a change to the default regime. Thresholds frozen before the run |
| G4 UI recognition | M3 | On held-out layouts, vision-minus-blind of at least 15 points on screen-state and text-in-screenshot tasks |
| G5 Grounding | M4 | Point-in-box hit rate on a held-out set beats center-click and random baselines with a CI. As a floor reference, one community projector-only retrofit reported 4.3% at a 50-unit tolerance \[S\] |
| G6 Ladder | M5a | The 4B to 14B trend resembles Baseten’s (rising, then saturating) and reasoning-mode behavior is checked |
| G7 Hero spike | M5b | Forward and backward to input embeddings works on one sample; throughput and memory are measured; a 32-example overfit decreases loss |
| G8 Hero run | M5c | Vision-minus-blind CI excludes zero; comparison rows against Vision-Exp and V4.1 are filled |

**G0 result (2026-11-15):** passed; see `reports/g0.md` and `DECISIONS/0027`.

### 9.5 Diagnostic guide

| Symptom | Likely cause | First check |
| --- | --- | --- |
| Loss falls, correct is about equal to blank and mismatched | Text shortcut or format cue (the p05 pattern) | Data balance; train past the plateau; gap curves |
| Correct beats blank but not mismatched | Using image presence, not content | Mismatch pairs actually differ in answer |
| Teacher-forced loss good, generation garbage | Parity bug: cache, positions, padding | Generation parity test |
| NaN or sudden loss spike | Precision, norm ratio, LR | Norm-ratio log; fp32 projector; lower LR |
| Train accuracy high, held-out low | Memorization | Held-out compositions; more diversity |
| High parse-fallback rate only with images | Template or format drift (Baseten saw 44.5% on multi-image) | Prompt template hash; training format coverage |

## 10. Roadmap

| Milestone | Goal | What you learn | Gate | Rough effort \[I\] |
| --- | --- | --- | --- | --- |
| M0 Skeleton | Splice, masks, freezing, generation and tests on Qwen3-1.7B with MoonViT-V2; build the K2.6 MoonViT extraction with a forward-parity test against the official remote code; one injection card per reader (section 7.6) | The token contract | G0 | 2 to 3 weeks (see the M0 breakdown) |
| M1 Synthetic | Paired sensitivity and generalization on controlled tasks | What a plateau looks like; reading gap curves | G1 | 1–2 weeks |
| M2a Reproduction | Baseten recipe on Qwen3-4B with the K2.6 encoder | Whether your pipeline matches published behavior | G2 | 2–3 weeks |
| M2b Encoder A/B | MoonViT-V2 versus K2.6 under identical settings | Isolating one variable | G3 | 1–2 weeks |
| M2c Unfreezing ladder (optional) | At 1.7B or 4B: projector only, then plus LoRA on the LM, then plus the encoder, under identical data and settings | What the frozen constraint costs, and how much text drift buys how much vision | G3b | 1–2 weeks |
| M3 UI recognition | Synthetic and real screenshots | How resolution and tokens limit detail | G4 | about 3 weeks |
| M4 Grounding | Pointing and region prediction, starting from Baseten’s targeted-training setup (1,000 examples, 10 epochs, LR 1e-4) | Why localization is the hard part | G5 | 3–4 weeks |
| M5a Ladder | Qwen3-8B and 14B | How the retrofit scales | G6 | 1–2 weeks |
| M5b Spike | Hero feasibility on a rented multi-GPU node | Backward through packed MoE | G7 | about 1 week |
| M5c Hero | V4-Flash-0731 plus native comparison rows | Whether the findings transfer | G8 | 2–3 weeks if the spike is clean; plan 6–9 weeks \[I\] |
| M6 Optional | RL on a rendered-math task to restore reasoning (Baseten: 64 prompts × 8 samples, LR 1e-5, 150 updates) | The RL loop on a frozen reader; closest to RL and training-infrastructure work | none yet | 3+ weeks |

At roughly 10 hours a week, M0 to M5c is about 6 to 7 months \[I\] (5 months before M2c was added and M5c was re-estimated at 6 to 9 weeks; the M0 breakdown now plans 2 to 3 weeks for M0). That assumption was answered in section 15: no fixed hours.

**M0 complete (2026-11-15):** all WP0–WP11 done; tags wp0-done … wp11-done.

**M0 task breakdown.** Work packages, the conformance suite, the debugging ladder and a kickoff prompt for the first coding session are in the companion document [Sightline M0: Plumbing task breakdown](https://claude.ai/artifact/8FHcv7ykwosmzLxj431don).

## 11. Repository and engineering practice

```
sightline/
  pyproject.toml  lockfile
  README.md  CHARTER.md  EXPERIMENTS.md  DECISIONS/
  src/sightline/
    config.py     typed dataclasses; resolved config dumped per run
    data/         synthetic.py  manifest.py  splits.py
    vision.py     encoder, preprocessing, optional cache
    projector.py
    backbone.py   tokenize, splice, masks, positions, generation path
    objective.py  answer-only loss
    train.py  evaluate.py  record.py
  tests/
  experiments/    one YAML per run, immutable once launched
  reports/        generated, small and diffable
```

Interfaces, kept small and explicit:

- `VisionEncoder.encode(images) -> features`
- `Projector(features) -> visual_embeds`
- `compose(input_ids, visual_embeds, spans) -> inputs_embeds, attention_mask, position_ids, labels`
- `evaluate(ckpt, manifest, conditions) -> per-sample paired results`

**Tests, in priority order.**

1. Golden splice: feeding the reader’s own embedding rows as fake visual embeddings reproduces the plain text forward.
2. Generation parity: first-token logits from teacher forcing equal those from `generate()`, and greedy continuation equals the teacher-forced argmax.
3. Label mask and truncation: only answer tokens are supervised, and truncation never drops a placeholder or a label.
4. Parameter test: only projector parameters receive gradients and change; the hash of frozen weights is unchanged.
5. Image sensitivity: changing only the image changes the projector output.
6. Resume: 2 steps, resume, 2 steps equals 4 steps on a tiny model.
7. Cache equality: cached features match live features.
8. Split hygiene: no template, source or perceptual-hash overlap between splits.

**Practices.** One change per experiment. Configs are immutable after launch. Never overwrite results; archive negative results with the same metadata as successes. Exploration happens in notebooks, but logic lives in `src`. Vision-Adapter code enters only with a test and a one-line justification. Keep three living documents (README, a short CHARTER, an EXPERIMENTS log) and put each expensive decision in a short ADR.

## 12. Compute and budget

**Assumptions.** One RTX PRO 6000 is available locally for the main training runs (it has 96 GB of GDDR7, confirmed on NVIDIA’s spec page \[V\]); Modal is for small debugging jobs and, much later, bigger GPUs. A single 80 GB GPU is the planning unit for M0 to M5a; multi-GPU is needed only for M5b and M5c. GPU-hours below are H100-equivalents, and the local card’s speed relative to an H100 is measured in M0 (WP10) and recorded.

**Back-of-envelope for one Qwen3-4B run** \[I\]: 2,048 steps × 128 sequences × about 600 tokens (256 visual plus about 350 text \[I\]) is about 157M tokens. At 4N FLOPs per token for a frozen reader that is about 2.5×10¹⁸ FLOPs, which is about 2 hours on one H100 at 35% utilization. Real runs are often 1× to 3× worse, so budget 2 to 6 H100-hours. Measure in M0 and replace these numbers.

| Phase | Hardware | Rough GPU-hours \[I\] |
| --- | --- | --- |
| M0–M1 | One 24–80 GB GPU | 5–20 in total |
| M2a | One 80 GB GPU | 2–6 per run |
| M2b (2 encoders × 3 seeds) | One 80 GB GPU | 12–36 |
| M2c (optional, 3 rungs) | One 80 GB GPU, checked in M2a | measured in M2a |
| M3–M4 | One 80 GB GPU | 20–60 |
| M5a | One 80 GB GPU | 40–120 |
| M5b spike | 8×B200, a few hours | 16–32 |
| M5c | Multi-GPU node | measured by the M5b spike |

**Hero spike (M5b).** Load V4-Flash-0731, inject random embeddings, run forward and backward to the input embeddings on one sample, and record peak memory, step time and any kernel failures. Your REAP notes already flag three traps that will reappear: weights expand from FP4 to BF16 at load, fused MoE kernels discard intermediates that hooks need, and, per those notes, V4’s custom attention requires FlashAttention 2 in the HF-style path (SDPA silently gives wrong outputs). The official reference code instead calls its own sparse\_attn kernel \[V\]. For the local card (compute capability 12.0), FlashAttention 2 has community builds, while FlashAttention 3 and 4 do not run on it \[S\]; rented Hopper or datacenter Blackwell GPUs are different cases. The candidate V4 backends, depending on hardware, are FlashInfer, FlashMLA, SDPA and FlashAttention 2, and FlashAttention 3 and 4 do not apply \[stated by you\]; the spike chooses. Set an abort criterion before renting.

**Hero cost floor and decision.** Counting 4 × active parameters per token, as for the small runs, V4-Flash-0731 needs about 4 × 13×10⁹ = 5.2×10¹⁰ FLOPs per token. The same 157M-token run is then about 8×10¹⁸ FLOPs, roughly 7 H100-hours at 35% utilization \[I\]. That is a floor: MoE routing, dequantization and kernel gaps usually cost several times more, and the real risk is memory, not FLOPs. The spike should measure tokens per second against this number. If throughput is more than about 10× worse (a placeholder threshold; freeze it before the spike), write up the Qwen3 ladder result instead of running M5c.

**Budget rule.** Write a GPU-hour cap before each phase. At 1.5× the cap, stop and write a note.

## 13. Risks

| Risk | Mitigation |
| --- | --- |
| A plumbing bug mimics success or failure | Golden splice and generation-parity tests before any training |
| Text shortcut or format cue | Balanced data, blank and no-tokens controls, gap curves |
| Stopping inside the plateau | Train past about 150k samples; watch the gap, not just the loss |
| Encoder choice confounds results | A/B with a reproduction control |
| Hero run infeasible (backward through packed MoE) | Spike first; the ladder is a worthwhile fallback result |
| Benchmark contamination | Exclude MMMU-Pro and BLINK images; hash manifests |
| LLM-generated QA leaks annotator bias | Record annotator and prompt; sample and audit |
| Licensing | Per-source records; read the K3 License before release |
| Scope creep | Gates and non-goals; one change per run |
| Time or compute drain | Caps per phase; weekly review |

## 14. Learning path

**Reading, in order.**

1. LLaVA, Visual Instruction Tuning (arXiv 2304.08485). The standard two-stage recipe. Skim.
2. Merullo et al., Linearly Mapping from Image to Text Space (arXiv 2209.15162). The closest classical precedent for a frozen LM with a learned map.
3. BLIP-2 (arXiv 2301.12597). A frozen encoder and frozen LM joined by a learned bridge.
4. Baseten, Fitting Vision Adapters at Frontier Scales (arXiv 2610.05897). Read sections 3 and 4.1 and appendices A and D closely.
5. Kimi K3 technical report (arXiv 2607.24653 per a Hugging Face mirror \[S\]). The vision sections on MoonViT-V2.
6. DeepSeek-V4.1 technical report (PDF on the model card). The multimodal architecture section.
7. Power et al., Grokking (arXiv 2201.02177). Intuition for plateau then cliff.

**Labs.** Each ends with a concrete done-criterion.

| Lab | What you do | Done when |
| --- | --- | --- |
| 1. Splice by hand | Feed embeddings through `inputs_embeds` on Qwen3-1.7B | The golden splice test passes |
| 2. Mask autopsy | Print tokens, labels and masks for 3 samples | You can predict them by eye before printing |
| 3. Overfit | One batch, then 32 examples | 99% exact match, projector-only updates verified |
| 4. Watch a plateau | Tiny synthetic task; plot loss and the gap curves | You can mark the transition step |
| 5. Counterfactual harness | Implement the five conditions; run on an untrained projector | You have chance-level baselines with CIs |
| 6. Embedding geometry | Fit an affine map between two models’ embeddings that share a tokenizer | You can explain Baseten’s relative-MSE result |
| 7. Mini ablation | Causal versus bidirectional image attention at 1.7B | You have a paired comparison with seeds |
| 8. Optional RL lab | Rendered-math RL on a frozen reader, at small scale | Reward rises and reasoning returns |

## 15. Questions and your answers

1. How many hours per week can you realistically give this, and are there fixed deadlines (school, work-study) the plan must respect?
2. What is the compute budget per phase, and which providers can you use?
3. How do you weight the goals: learning, a publishable result, or a public artifact (HF release, write-up)?
4. Do you already have a K2.6 MoonViT extraction, or should M0 include building one?
5. Are you comfortable using V4.1-Flash through an API to generate QA data, given cost and terms?
6. Do you plan to release checkpoints? That determines how soon the K3 License must be read closely.
7. Which of your earlier UI data sources have licenses clean enough to reuse?

**Your answers, 10 November 2026.**

| # | Answer | What it changes |
| --- | --- | --- |
| 1 | Open-ended (“whatever it takes”) | Estimates stay in weeks at about 10 hours a week as a planning assumption only \[I\]; no calendar deadline is encoded |
| 2 | RTX PRO 6000 for big training; Modal for small debugging; bigger GPUs on Modal much later | M0 to M5a run locally. The local card’s 96 GB \[V\] cannot hold V4-Flash’s roughly 167 GB of packed weights, so M5b and M5c rent bigger GPUs |
| 3 | Mostly learning; a public artifact or HF write-up possible later; interest in MLE or AIE roles | Learning wins in M0 to M2 (section 2.1). Keep M6 (RL) as the closest milestone to training-infrastructure work. Keep run records publishable |
| 4 | Only MoonViT-V2 exists; the K2.6 extraction is needed | M0 builds it with a forward-parity test before the M2b encoder A/B |
| 5 | API use is fine within limits: not hundreds of dollars, about 200M tokens a day acceptable | V4.1 or K3 can annotate. Assuming about 1.6k tokens per conversation (image plus generated QA \[I\]), 131k conversations is about 210M tokens, roughly a day at that cap; the 20k–40k iteration set is about 32–64M. Terms are checked in the M2 entry criteria (section 16) |
| 6 | Release likely if a write-up or HF artifact happens; you expect the K3 License to be mild | Read in full on 13 November (52 lines, GitHub) \[V\]. The “Software” it defines includes the model weights, so the MoonViT-V2 encoder is covered \[I\]. It grants use, copying, modification, fine-tuning and derivative works; the notice must travel with copies. A separate agreement is needed only for Model-as-a-Service businesses above $20M revenue over 12 months, and the name “Kimi K3” must be shown on products above 100M monthly users or $20M monthly revenue; internal use and use through Moonshot’s own products are exempt. The text has no clause restricting outputs or training other models \[V\]. A research projector and a write-up do not appear to trigger either condition \[I\]. This is not legal advice |
| 7 | Mostly Apache-2.0 or MIT; each source’s license is recorded before download (section 16) | Per-source license record from day one. The old embedding cache lists its license as unknown \[V\] |

## Appendix A: Sources opened this session

- Baseten, Fitting Vision Adapters at Frontier Scales: https://arxiv.org/pdf/2610.05897 \[V\]
- Baseten blog, GLM 5.2 with vision: https://www.baseten.co/blog/glm-52-with-vision/ \[S\]
- DeepSeek-V4.1-Flash: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash \[V\]
- DeepSeek-V4.1-Flash-1017: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 \[V\]
- DeepSeek-V4-Flash-Vision-Exp (vLLM recipe): https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \[S\]
- DeepSeek-V4-Flash-0731 coverage: https://techjacksolutions.com/ai-brief/deepseek-v4-flash-0731-mit-license-hugging-face/ \[S\]
- Your MoonViT-V2 extraction: https://huggingface.co/keypa/MoonViT-V2-Standalone \[V\]
- MoonViT-V2 card: https://huggingface.co/AI4Industry/MoonViT-V2 \[S\]
- Kimi K3: https://huggingface.co/moonshotai/Kimi-k3 \[S\]
- Kimi K2.6 (NVIDIA card): https://build.nvidia.com/moonshotai/kimi-k2.6/modelcard \[S\]
- Baseten GLM-5.2-Vision: https://huggingface.co/baseten/GLM-5.2-Vision-FP8 \[S\]
- Community projector-only retrofit: https://huggingface.co/0xSero/glm-local-vision-checkpoint \[S\]
- Qwen3.5-2B: https://modal.com/library/qwen/qwen3-5-2b \[S\]
- Qwen3-1.7B: https://partimus.com/en/ai-language-models/qwen3-1-7b/ \[S\]
- DeepSeek V4 vision wrapper (vLLM source docs): https://docs.vllm.ai/en/latest/api/vllm/models/deepseek\_v4/common/vl\_model/ \[V\]
- DeepSeek-V4.1-Flash vLLM recipe (search snippet): https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash \[S\]
- DeepSeek-V4 Flash, NeMo AutoModel notes (search snippet): https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/deepseek-v4-flash \[S\]
- Qwen3-VL technical report (search snippet): https://arxiv.org/pdf/2511.21631 \[S\]
- Qwen3 sizes, NVIDIA Megatron-Bridge docs: https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.models.qwen.qwen\_provider.html \[S\]
- Old embedding cache card: https://huggingface.co/datasets/keypa/vision-adapter-embeddings \[V\]

* WebBrain DeepSeek-V4-Flash-0731-Vision (K2.6 MoonViT retrofit): https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4 \[V\]
* Baseten GLM-5.2-Vision card: https://huggingface.co/baseten/GLM-5.2-Vision-FP8 \[V\]
* DeepSeek-V4-Flash-0731 config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json \[V\]
* Kimi K3 License (GitHub): https://github.com/MoonshotAI/Kimi-K3/blob/main/LICENSE \[V\]
* Kimi-K2.6 model card: https://huggingface.co/moonshotai/Kimi-K2.6 \[V\]
* NVIDIA RTX PRO 6000 Blackwell Workstation Edition: https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000.md \[V\]

- WebBrain SGLang compatibility commit (deployment notes, V4 embedding-site patch): https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-Vision-NVFP4/commit/bebc653f412567764f09f2332c905c345f1521a3 \[V\]
- jarrelscy projector-only graft onto 0731 (search snippet): https://huggingface.co/jarrelscy/deepseek-v4-flash-0731-vision \[S\]

## Appendix B: Run record (every run, written automatically)

Code revision (commit and dirty flag); resolved configuration; reader, encoder and tokenizer revisions; chat-template hash; data manifest hash and split ids; preprocessing version; seeds; hardware and library versions (torch, transformers, flash-attn); per-step metrics as JSONL; probe-gap curves; checkpoint and report hashes; wall-clock time and GPU-hours; outcome notes, including failures.

## 16. What is still open, and who closes it

Nothing below blocks M0. Every item has a place where it is closed and the evidence that closes it. Once coding starts, entries in DECISIONS/ override this document, and it is re-exported only when a milestone completes. If a new unknown appears during the work, it is added to DECISIONS as \[U\] with an owner and a closing step, and no work package ends with an unowned \[U\].

| # | Item | Closed by | When |
| --- | --- | --- | --- |
| 1 | K3 tower code details: tower class, merge order, position-embedding interpolation mode, attention switch | Vendor and read modeling\_kimi\_k3.py; token-order test; SDPA versus FlashAttention 2 parity (M0 breakdown, WP5) | M0 |
| 2 | Placeholder id and rendered chat template | Read from the pinned Qwen3 tokenizer; golden-string test (WP1, WP2) | M0 |
| 3 | Library versions | WP0 rule: newest stable torch whose arch list includes the local card, with a matching transformers | M0 |
| 4 | Local GPU speed, memory and GPU-hours | Measured in the M0 overfit (WP10) and in M2a, written to run records; replaces the H100-equivalent estimates | M0 and M2a |
| 5 | Bare patches versus markers | M1 ablation | M1 |
| 6 | Embedding-alignment init | Count identical token strings between the K3 tokenizer and each reader; run as an ablation, never as a default | Before M2a |
| 7 | Data sources and licenses | Per-source license table recorded before download; no source enters without a recorded license | M2 entry criterion |
| 8 | Annotator choice, API terms and cost | 100-example pilot; check terms; estimate tokens against the cap of about 200M tokens a day and no cost in the hundreds of dollars | M2 entry criterion |
| 9 | Baseten training resolution | Not stated in the paper; recorded deviation (448 canvas); resolution sweep in M3 | Closed as a deviation |
| 10 | DeepSeek V4: palette scheme effect, attention backend by hardware (FlashInfer, FlashMLA, SDPA, FlashAttention 2), autograd-capable implementation, memory and step time | M5b spike, with its abort criterion frozen before renting hardware | M5b |
| 11 | Old cache autopsy (optional) | Only if p05’s failure analysis is wanted: compare against the one kept shard after the M0 parity test | Optional, after M0 |
| 12 | Release of any artifact | Keep license notices (the K3 License has no output clause \[V\]), run records and per-source licenses with it | At release |

**M2 entry criteria.** The previous gates are passed with thresholds frozen; the per-source license table is complete; the annotator pilot is done and its terms and cost are recorded; the data manifest, its hash and the split rules are defined.
