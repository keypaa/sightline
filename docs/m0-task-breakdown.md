# Sightline M0: Plumbing task breakdown

Companion to [Sightline: Project Document v1.0](https://claude.ai/artifact/LyNX5csUp6CtmmPcHpL72m) · updated 14 November 2026 · Evidence tags as in the main document: \[V\] verified, \[S\] secondary source, \[I\] my inference, \[U\] unknown.

## 1. What M0 is for

M0 proves the plumbing, not the science. When it ends, a frozen Qwen3-1.7B reader and a frozen MoonViT-V2 encoder are connected through a trainable projector, every piece has a test, and a 32-example overfit works. Nothing in M0 says whether vision transfers; that is M1.

**Gate G0 (main document, section 9.4).** All of these, with thresholds frozen before the first run:

- The golden splice test passes.
- Generation parity holds.
- The parameter test passes.
- A 32-example overfit reaches 99% exact match within 500 steps.
- The resume test passes.

**Added since the first draft of the main document** (decisions of 10 to 13 November): the K2.6 MoonViT extraction with a forward-parity test, one injection card per reader, and the conformance suite in section 4. The palette tooling for DeepSeek V4 (make\_palette.py, palette\_cap3\_0731.json) is stored in the repo now and used in M5b.

**Out of scope for M0:** real-image QA data, any training of the reader or encoder, DeepSeek V4 runs, and the K2.6-versus-V2 comparison (M2b).

**Time.** The main document budgets about 2 weeks. With the K2.6 extraction and the conformance suite I would plan 2 to 3 weeks \[I\].

## 2. Decisions to record first

Each decision below is recorded in DECISIONS/ as a short dated file in WP0 and WP1. They were settled on 14 November; changing one later needs a new dated entry.

### 2b. Resolutions of 14 November

Your answers and the verification done since.  “Verify in WP” means a test or a read must confirm it before anything relies on it.

| # | Decision | Resolution | Status |
| --- | --- | --- | --- |
| 1 | Canvas | Keep 448×448 (256 tokens) for M0 to M2 as our own choice. Baseten’s paper does not state its training resolution, so M2 reproduces the recipe, not the resolution; record that deviation. The token count is derived from config and never hard-coded. M3 varies resolution. | Resolved |
| 2 | Image marker | One literal image marker expands to N placeholder positions, with bare patches by default (explained below). Markers become an M1 ablation, because Baseten’s usage is unknown and native models use markers. | Resolved |
| 3 | Placeholder id | Qwen3 tokenizers already contain the image\_pad token (id 151655 in the copies I read \[S\]); use it and confirm the id in the pinned tokenizer. With inputs\_embeds the model takes exactly one of input\_ids or inputs\_embeds \[S\], so no embedding lookup happens at those positions. Add a test that user text cannot contain the placeholder. | Verify in WP1 and WP2 |
| 4 | Merge location | The tower’s output stage groups 2×2 blocks in row-major order (tpool\_patch\_merger in K2.5 and K3 code \[S\]); the projector then takes four patches concatenated: 4096 wide for V2, 4608 for K2.6. The K3 projector has no pre-norm and ends in an RMSNorm; the K2.6 projector starts with a LayerNorm \[S\]. Shapes agree with the checkpoint \[V\]. | Verify by reading the vendored code and a token-order test |
| 5 | Chat template | Qwen3’s template with enable\_thinking=False adds an empty think block after the assistant header \[S\]. The empty block sits in the prompt, not in the loss, and the loss covers the answer and the end token. Baseten put the empty block inside the supervised span \[V\], so this is a recorded deviation. A golden-string test pins the rendered prompt. | Verify the rendered string in WP1 |
| 6 | Memorization | Built into the data and the tests (below). | Resolved |
| 7 | Precision | Frozen reader and encoder keep their shipped bf16. The projector keeps fp32 master weights with bf16 autocast; optimizer state in fp32. A trained projector can be cast to bf16 for release. | Resolved |
| 8 | Seeds | 0, 1 and 2. Never 42, which agents default to. A literal 42 used as a seed is a defect, and this is written in AGENTS.md. | Resolved |
| 9 | Attention | Qwen3 readers and the K3 tower: PyTorch SDPA by default, FlashAttention 2 only after a parity test, backend recorded in every run. Qwen3 uses head dimension 128 and 8 KV heads at every size, which FlashAttention 2 supports \[S, from your table; WP1 verifies from config.json\]. FlashAttention 3 and 4 do not run on this card, and DeepSeek-V4 does not use them either \[stated by you\]; V4’s head dimension is 512 \[V, config\], above the 256 limit I believe FlashAttention kernels have \[I\]. For V4 the candidate backends, depending on hardware, are FlashInfer, FlashMLA, SDPA and FlashAttention 2 \[stated by you\]; the M5b spike chooses. | Resolved |

**What “bare patches” means.** The prompt text contains one marker, for example `<|im_start|>user\n<image>What color is the shape?<|im_end|>`. Tokenization turns the marker into 256 copies of one placeholder id, and the model’s embedding lookup is skipped at those positions: our splice overwrites them with the projector’s output vectors. “Bare” means nothing else surrounds those 256 vectors. Native models usually wrap them in markers: Qwen uses vision\_start and vision\_end tokens, Kimi uses media\_begin, the word image, media\_content and media\_end, DeepSeek-V4.1 uses four learned vectors (start, pad, newline, end), and a community GLM clone uses begin and end image tokens \[S\]. WebBrain inserts bare patches \[V\]. Baseten’s paper describes replacing placeholder tokens inside the user message but says nothing about markers, so marker use is tested in M1; I had earlier written that both use bare patches, and that was an overstatement. Qwen3’s vision marker tokens exist in the vocabulary, but a text-only Qwen3 has probably never trained on them \[I\].

**Baseten’s resolution.** Not stated in the paper, so it is a recorded deviation. What the paper does give: SFT sequences are padded to 5,120 tokens \[V\], so image tokens per conversation stay under that; interpretability images use a long side of at most 448 pixels \[V\]; and Baseten’s GLM card says serving accepts up to 4,096 image tokens per image \[V\]. The K2.6 processor’s limits are dynamic \[S\], so training may have used a range of sizes. Options to learn more: the authors’ contact addresses are on the paper, and the GLM card’s processor configuration may show defaults. Neither blocks M0, because the canvas is a config value.

**Controls against memorization** (decision 6):

- Balance answers given the question: every question text appears with every answer equally often across different images, so text alone scores at chance. Questions must never uniquely identify the answer.
- For the 32-example overfit, use one question template (for example “What color is the shape?”) with answers that vary only by image, so the model cannot memorize a fixed phrase.
- After the overfit, evaluate the same examples with a blank image, a shuffled image (another example’s image) and a swapped pair. The overfit counts for G0 only if those collapse to near chance; the thresholds are frozen beforehand.
- Report the per-token answer loss gap between the real image and a blank image on held-out compositions (the digit-NLL gap idea from the old project), because a loss near zero can still hide a text-only shortcut.

**Changes to work packages.**

- WP1: confirm `<|image_pad|>` and its id in the pinned tokenizer; render the template with thinking off and store the golden string.
- WP2: placeholder-collision test and a test that no embedding lookup happens at placeholder positions.
- WP5: check the position-embedding interpolation mode (sources disagree, see WP5); add a merge-order test on a synthetic grid; cross-check the vendored tower against NVIDIA NeMo AutoModel’s K3 vision implementation, whose module names match \[S\]; compare SDPA against FlashAttention 2 for the tower, whose config requests flash\_attention\_2 \[V\].
- WP9 and WP10: apply the memorization controls above.
- Conformance suite: add counterfactual-image, template-golden-string, attention-backend-parity and projector-dtype tests.

## 3. Work packages

Order: WP0, then WP1; WP2, WP5 and WP6 can run in parallel; then WP3, WP4, WP7, WP8, WP9, WP10, WP11. WP12 is reading and can happen any time. Effort figures are my estimates \[I\].

| WP | Deliverable | Effort |
| --- | --- | --- |
| 0 | Repository skeleton and environment | 0.5 day |
| 1 | Reader pinned and verified, injection card for Qwen3-1.7B | 0.5 day |
| 2 | Splice, positions and masks | 1 to 2 days |
| 3 | Generation path | 1 day |
| 4 | Freezing and the parameter test | 0.5 day |
| 5 | MoonViT-V2 wrapper and parity | 2 to 3 days |
| 6 | K2.6 MoonViT extraction and parity | 2 to 3 days |
| 7 | Projector | 1 day |
| 8 | Objective, training loop, checkpoint and resume | 1 to 2 days |
| 9 | Minimal synthetic data | 1 day |
| 10 | The 32-example overfit (G0) | 1 to 3 days, depending on bugs |
| 11 | Conformance suite, cards, G0 report | 1 day |
| 12 | DeepSeek V4 reading and tooling | 1 to 2 days, no GPU |

### WP0. Repository skeleton and environment

- The starter files (AGENTS.md, CLAUDE.md, README.md, CHARTER.md, EXPERIMENTS.md, .gitignore, DECISIONS/, tools/, docs/) already exist in the repository. Create the rest of the layout from main document section 11: pyproject and lockfile, src/sightline/ with config, data, vision, projector, backbone, objective, train, evaluate and record, tests/unit/, tests/conformance/, experiments/ and reports/; artifacts/ and data/ are created locally and stay out of git.
- Typed config dataclasses; the resolved config is dumped into every run.
- Run record per section 11 of the main document: commit and dirty flag, resolved config, model and tokenizer revisions, chat-template hash, seeds, library versions, per-step metrics as JSONL, GPU-hours.
- Pin library versions in the lockfile. Versions are chosen in WP0 by this rule: the newest stable torch build whose get\_arch\_list includes the local GPU’s architecture, with a transformers version that loads the pinned Qwen3 checkpoint and the vendored K3 module; write both into the first DECISIONS file.
- **Done when:** pytest runs an empty suite and a dry-run writes a complete run record.

### WP1. Reader pinned and verified

- Pin the Qwen3-1.7B revision and tokenizer revision. Record layers, hidden size, head count, vocabulary size and whether input and output embeddings are tied, read from the checkpoint’s own config.json. The main document has 28 layers and hidden size 2048 \[S\]; confirm both.
- Record the hidden sizes for the later readers so projector widths are not guessed: 4B 2560, 8B 4096, 14B 5120, 32B 5120 \[S\].
- Write the injection card DECISIONS/reader-qwen3-1p7b.md: injection point (inputs\_embeds at the embedding output), placeholder id, positions (plain 1D RoPE), mask (causal), side channels (none), config values, evidence tags.
- **Done when:** the card exists and a text-only forward through the pinned reader runs.

### WP2. Splice, positions and masks (backbone.py)

- One function takes input\_ids, visual embeddings and the placeholder id and returns inputs\_embeds. It raises if the number of placeholder positions differs from the number of visual tokens, per sample.
- Positions and attention mask for batches with different text lengths, with right padding for training.
- **Done when:** the golden splice test passes (section 4) and the placeholder-count and padding tests pass.

### WP3. Generation path

- Compute visual embeddings once, splice, then decode with the KV cache from inputs\_embeds. Confirm what the library returns when generating from inputs\_embeds (I expect only the new tokens \[I\]; verify).
- **Done when:** greedy decoding with cache equals full recomputation token for token, and generation from spliced embeddings equals generation from ids on the golden splice.

### WP4. Freezing and the parameter test

- Reader and encoder in eval mode with requires\_grad false; only projector parameters are trainable.
- **Done when:** after 20 steps, only projector parameters have non-zero gradients and changed values, and the hash of every frozen weight is unchanged.

### WP5. MoonViT-V2 wrapper (vision.py)

- Fixed 448×448 canvas with one resize and pad policy, normalization values taken from the pinned processor config, not guessed.
- Output shape check: 256 merged tokens at the encoder’s output width \[I\]. A third-party description gives 27 blocks, hidden size 1024, 12 heads, MLP size 4096, patch size 14, RMSNorm, no biases and divided spatial and temporal position embeddings \[S\]. The blocks alone come to about 396M parameters \[I\], consistent with the roughly 401M a community model card reports \[S\]. Merge: the tower’s tpool\_patch\_merger groups each 2×2 block, and the projector takes 4096-wide tokens (1024 × 4) \[S\]. The projector has only proj.0 (4096 to 4096), proj.2 (4096 to 7168) and post\_norm (7168) \[V, tensor names and shapes\], which is 46,144,512 parameters \[I\]; with the 401M tower that matches the 447M a community MLX port reports \[S\]. Differences from K2.6 \[S\]: RMSNorm instead of LayerNorm, no biases, 2D RoPE, position-embedding interpolation whose mode (bilinear or bicubic) is read from the vendored code in WP5, and a different projector (that port lists six differences in all). Verify from the checkpoint’s vision\_config and record the merge in decision 4.
- Decision of 13 November: re-extract the tower and projector from moonshotai/Kimi-K3 at a pinned revision in the clean repository, and do not reuse the earlier extraction. Find the tensors by the prefixes vision\_tower. and mm\_projector. in the pinned model.safetensors.index.json and read them with HTTP range requests, so the checkpoint (about 1.4 to 1.6 TB \[S\]) is never downloaded. The K3 index you downloaded confirms 165 tower tensors, all in shard 96, and 3 projector tensors in shard 95, out of 96 shards, 497,220 tensors and 1,560,860,324,864 bytes in total \[V\]. In BF16 that is roughly 802 MB for the tower and 92 MB for the projector \[I\]. The script extract\_k3\_vision.py (saved with this project) does the extraction, checks the counts, copies raw bytes and writes a manifest with per-tensor sha256; you ran it against the real repository on 13 November (Result below) \[V\]. Implementation: vendor the official vision-tower module from the K3 repository at the pinned revision instead of rewriting it, so forward parity holds by construction, and keep the K3 License file with it. The tower class is MoonViT3dPretrainedModel in the repository’s custom code (modeling\_kimi\_k3.py), seen in a module print of the real model and in ports that transcribe it \[S\]; The file itself is read in WP5, where it is vendored. Its vision config requests flash\_attention\_2 \[V\], but a community MLX port runs Moonshot’s actual torch code on CPU for parity tests \[S\], so a non-flash path exists. WP5 confirms that the official module runs with SDPA on the local card (test in 2b).
- License, read in full on 13 November (52 lines) \[V\]: the “Software” it defines includes the model weights, so the encoder is covered \[I\]. It grants use, copying, modification, fine-tuning and derivative works. Keep the license notice with any copy of the weights or vendored code. The extra conditions apply only to Model-as-a-Service businesses above 20M US dollars over 12 months and to products with over 100M monthly users or 20M dollars monthly revenue; internal use is exempt, and the text has no clause restricting outputs. This is not legal advice.
- **Done when:** the extracted tensors equal the checkpoint’s tensors name by name; the earlier extraction, if you keep it, is compared tensor by tensor with the new one and the result is written in DECISIONS/ (identical validates it, any difference is a finding); forward outputs from our loading path equal the official module loaded straight from the pinned shards on at least 20 images within a frozen tolerance; and the shape and width are recorded. A community export script with the same tensor-count checks exists (immanuel-peter/vision-tower-bench, scripts/export\_moonvit\_v2.py \[S\]); use it as a cross-check, not a dependency.

**Result, 13 November \[V\].** The script ran against moonshotai/Kimi-K3 at revision f831ab66814297da540d832a5235f8e904f29d06: 165 tower and 3 projector tensors, all BF16, 894,717,952 bytes, 401,214,464 tower parameters and 46,144,512 projector parameters (the projector figure matches the prediction above). Compared with your earlier standalone extraction (keypa/MoonViT-V2-Standalone, moonvit\_v2.safetensors, 802 MB), all 165 tower tensors are byte-identical in name, shape, dtype and values. The 3 projector tensors are absent from the old file, as expected. So the earlier weights are validated; the earlier code and preprocessing are not reused, so they need no check; forward parity of the new code is WP5. Record the revision in the first DECISIONS file. The comparison run was made without a pinned revision, so its manifest says main; rerun once with both --revision and --compare to keep a single pinned manifest.

### WP6. K2.6 MoonViT extraction and parity

The reproduction control in M2 needs this encoder, and you only have V2.

- Source: moonshotai/Kimi-K2.6 at revision 7eb5002f6aadc958aed6a9177b7ed26bb94011bb, tensors with the prefix vision\_tower. WebBrain’s extraction has 329 tensors, all BF16, 833,765,768 bytes \[V\].
- Architecture from WebBrain’s config \[V\]: 27 layers, hidden size 1152, intermediate size 4304, 16 heads, patch size 14, a learned 64×64 position grid, 2×2 merge. That is about 417M parameters \[I\].
- Parity references: SGLang’s MoonViT3dPretrainedModel and K2VLMultiModalProjector, which WebBrain imports \[V\], the official remote code in the Kimi-K2.6 repository \[U, not read\], and a community packaging of the same 329-tensor tower with standalone modeling code, immanuelpeter/MoonViT-K2.6 \[S\]. If you use SGLang’s module, its weight loader renames wqkv to attn.qkv\_proj and wo to attn.proj \[V\].
- **Done when:** names, shapes and dtypes match the checkpoint (329 tensors), values match tensor by tensor (compare tensors, not file hashes, since file metadata can differ \[I\]), and forward outputs match a reference on at least 20 images at several resolutions within a frozen tolerance.

### WP7. Projector (projector.py)

- Follow the design in main document section 6.3. As a reference shape, K2.6-style is LayerNorm, 2×2 merge, Linear 4608 to 4608, GELU, Linear 4608 to the reader width; that is 40,119,040 parameters at width 4096 \[V\] and 30,679,808 at width 2048 \[I, arithmetic\].
- Initialization is a flag: random, or embedding alignment (an ablation, not a default; section 6.3 of the main document).
- Log the ratio of projector-output norm to embedding-row norm every N steps.
- **Done when:** the parameter count matches the formula, output width equals the reader’s hidden size, and the norm ratio is logged.

### WP8. Objective, training loop, checkpoint and resume

- Answer-only loss with a mask test (a prompt-only sample contributes zero loss).
- AdamW, warmup and cosine schedule, gradient accumulation, bf16 autocast, seeded data order.
- Checkpoints hold projector, optimizer, scheduler, RNG states and step.
- **Done when:** a run stopped at step 100 and resumed reproduces the next 10 losses within a frozen tolerance.

### WP9. Minimal synthetic data (data/synthetic.py)

- Just enough for the overfit: 32 rendered scenes (color, shape, count) with exact answers, deterministic by seed, with a hashed manifest. The full generator is M1.
- **Done when:** regenerating from the seed reproduces the manifest hash.

### WP10. The 32-example overfit (G0)

- Train on the 32 examples until 99% exact match; limit 500 steps.
- If it fails, use the debugging ladder in section 6 before changing hyperparameters.
- **Done when:** the run record shows 99% within 500 steps on at least two of three seeds \[I\]; the exact rule is frozen in DECISIONS before the first run.

### WP11. Conformance suite, cards and G0 report

- Collect the tests in section 4 under tests/conformance/ so every future reader adapter must pass the same suite.
- Write the G0 report (generated, small and diffable) and the first EXPERIMENTS entry.
- **Done when:** the whole suite is green and G0 is signed off in DECISIONS/.

### WP12. DeepSeek V4 reading and tooling (no GPU)

- Store in the repo: make\_palette.py, palette\_cap3\_0731.json, and the three tid2eid arrays with their sha256. Base V4-Flash and 0731 tables are identical \[V\].
- Reading tasks from main document sections 7.6 and 7.7: an autograd-capable implementation for backward passes (HF Transformers or NeMo AutoModel), since the reference code is inference-only \[V\].
- Closed: the 0731 checkpoint index has no vision or routing-bias tensors of any name \[V, checked 14 November\]; K3’s and GLM’s insertion formats are not needed because neither is a reader in this plan; and our own palette (make\_palette.py) replaces WebBrain’s unknown selection method.
- **Done when:** the files and hashes are in the repo and a short note lists which autograd-capable implementation you will test first.

## 4. Conformance suite

Thresholds are placeholders; freeze them before the first run.

| Test | What it proves | How | Pass |
| --- | --- | --- | --- |
| Golden splice | Splicing does not disturb the reader | Feed the reader’s own embedding rows as fake visual embeddings and compare logits with the plain text forward | Equal in fp32; small tolerance in bf16 |
| Logit change | The injected embeddings actually reach the model (the V4 failure mode where input\_embeds is ignored) | Perturb one visual embedding row and compare logits at later positions | Later logits change; unperturbed samples are identical |
| Placeholder count | Alignment cannot silently drift | Wrong number of visual tokens for a sample | Raises an error |
| Padding invariance | Batching does not change answers | Same sample alone and inside a padded batch | Logits equal within tolerance |
| Generation parity | Cache and recompute agree; ids and embeds agree | Greedy decode both ways | Identical tokens |
| Chunked prefill | Splitting a long prefill does not change logits | Prefill in chunks with cache versus in one pass | Equal within tolerance |
| Parameter test | Only the projector learns | Gradients and hashes after 20 steps | Frozen hashes unchanged |
| Resume | Checkpoints are complete | Stop at 100, resume, compare 10 losses | Within tolerance |
| Determinism | Runs are repeatable | Same seed, 5 steps | Same losses within tolerance |
| Routing IDs (V4 only) | Image positions get palette\[(position minus image start) mod 64\] in prefill and ordinary IDs in decode | Unit test on the routing function | Exact |

## 5. Environment checks

- The local card is an RTX PRO 6000 with 96 GB \[V\]. Confirm the installed torch build lists the card’s architecture in torch.cuda.get\_arch\_list(). New workstation architectures are a known setup trap for older builds \[I\]. FlashAttention 3 and 4 do not run on this architecture \[S\], and FlashAttention 2 needs a build or community wheel for it \[S\]. Use SDPA by default (decision 9).
- Record GPU-hours per run from the first run, so the compute table in the main document gets measured numbers.
- Debug on the local card; use Modal only for small jobs that need a different GPU.

## 6. Debugging ladder if the overfit fails

Work down the list and stop at the first failure.

1. The golden splice test passes on the actual batch.
2. A non-zero gradient reaches the projector, and nothing else changes.
3. The loss mask covers answer tokens only, and the labels are shifted correctly.
4. Placeholder alignment: no off-by-one between the text and visual positions.
5. The chat template and the thinking switch match between training and generation.
6. Projector output norm against embedding norm (look at the logged ratio).
7. Learning rate, then warmup.
8. Only then suspect the encoder: compare with a randomly initialized encoder on the same data; it should fail.

## 7. Remaining items and who closes them

- Closed by WP5: the vendored modeling\_kimi\_k3.py is read there, and the token-order and SDPA-versus-FlashAttention-2 tests settle the tower class, merge order and attention switch.
- Closed as a recorded deviation: the paper does not state it, and the canvas is a config value (2b, decision 1).
- Closed in 2b (decision 5); WP1 stores the rendered golden string.
- Closed by the WP0 rule for choosing versions.
- Effort and tolerance figures are estimates or placeholders by design: measured numbers replace the effort estimates after M0, and tolerances are frozen in DECISIONS before each first run (a rule in AGENTS.md).

## 8. Kickoff prompt for Claude Code

Paste this at the start of the first coding session.

> You are the implementation agent for Sightline, a research repository. Read AGENTS.md first, then docs/sightline-project-document.md and docs/m0-task-breakdown.md in full. We are doing M0 only, in this fresh repository, with no inherited code or data.
>
> Rules for everything you do:
>
> 1. One work package at a time, in the order of the M0 breakdown. Start each in plan mode: restate the goal, the tests and the acceptance criteria, and wait for my approval before editing.
> 2. Tests first. Write each test and its pass criterion before the code it covers, and freeze thresholds in a DECISIONS file before the first run.
> 3. Measure before moving on and before optimizing. After each work package, run its tests, record the numbers (step time, memory, tokens per second where relevant) in the run record and show me the output. Before any optimization, record a baseline and change one thing at a time.
> 4. Write a DECISIONS file whenever a default is chosen or changed. Use the evidence tags \[V\], \[S\], \[I\] and \[U\] in every note, say \[U\] when you do not know, and read checkpoint configs instead of recalling dimensions.
> 5. Seeds are 0, 1 and 2. Never 42.
> 6. Copy nothing from the old vision-adapter project. Do not start M1 until the G0 report is signed off.
> 7. Git: one branch per work package, small commits whose messages say why, and a tag when a work package's acceptance tests pass.
>
> Begin with WP0 and WP1 only. When done, show me the repository tree, the resolved config, the run record from a dry run, the pinned versions, and the injection card for Qwen3-1.7B. Ask me before any choice that is not in the documents.
