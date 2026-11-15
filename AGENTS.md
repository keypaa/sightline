# AGENTS.md: Sightline

Instructions for any coding agent working in this repository (Claude Code, Codex, others). Read all of it before acting.

## What this project is

Sightline is a clean-break research repository on post-hoc visual retrofitting: attach a frozen vision encoder (Kimi MoonViT) to a frozen text-only LLM through a small trainable projector, then measure honestly what vision capability results. Long-term target: Kimi K3's MoonViT-V2 to DeepSeek-V4-Flash-0731. The owner is learning, so understanding and honest evidence matter more than speed or headline numbers.

**Current stage: M0 only** (plumbing, Qwen3-1.7B reader). Do not start M1 until the G0 report is signed off in DECISIONS/.

## Where things are written down

- `docs/sightline-project-document.md`: main project document (plan, evidence, roadmap, gates).
- `docs/m0-task-breakdown.md`: M0 work packages (WP0 to WP12), conformance suite, debugging ladder.
- `DECISIONS/`: dated decision records. They override both documents. If two sources disagree, stop and ask.

## How to work

1. One work package at a time, in the order given in the M0 breakdown.
2. Tests first. Write the test and its pass criterion before the code it covers. Thresholds are frozen and recorded before the first run.
3. After each work package, show the owner what changed, the test output and the DECISIONS entry. Wait for approval before moving on when a decision was needed.
4. Any choice not already decided (versions, dtypes, tolerances, names) gets a DECISIONS file. Never silently pick a default.
5. No inherited code or data. The previous project (vision-adapter) is read-only reference: do not copy code, caches, manifests or labels from it. If something is reused, write a one-line justification and a test for it.
6. No new dependency without a DECISIONS entry.
7. Keep changes small and reviewable. No refactors outside the current work package.
8. Every run writes a run record (commit and dirty flag, resolved config, model and tokenizer revisions, chat-template hash, seeds, library versions, attention backend, dtypes, metrics as JSONL, GPU-hours). A result without a run record does not exist.
9. **Measure before moving on and before optimizing.** After each work package, run its tests and record the numbers (step time, memory, tokens per second where relevant) in the run record. Before any optimization, record a baseline, change one thing at a time and keep the baseline. An optimization without a before-and-after measurement is not accepted.
10. **Git.** One branch per work package (`wp0-skeleton`, `wp1-reader`, and so on), small commits whose messages say why, no weights, datasets or checkpoints committed, and a tag (for example `wp2-done`) when the work package's acceptance tests pass. Merge to main only after the owner approves.
11. **Open items.** A new unknown is written in DECISIONS as [U] with an owner and a closing step. No work package ends with an unowned [U].

## Evidence discipline

Tag every non-trivial claim in notes, comments and DECISIONS files:

- **[V]** verified: you read the primary source or ran the check yourself (cite file, commit or revision).
- **[S]** secondary: a summary, port or third-party report.
- **[I]** inference.
- **[U]** unknown.

Never invent numbers (model sizes, parameter counts, benchmark scores, API details, library behavior). If you do not know, write [U] and say how to find out. Read a checkpoint's own config.json instead of recalling dimensions. When two sources conflict, record both.

## Fixed design decisions (DECISIONS/ holds the current versions)

- **Reader (M0):** Qwen3-1.7B, frozen, eval mode, shipped bf16. Pin the revision in the first DECISIONS file.
- **Encoder:** Kimi K3 MoonViT-V2, frozen, eval mode, bf16. Weights are extracted from `moonshotai/Kimi-K3` at revision `f831ab66814297da540d832a5235f8e904f29d06` with `tools/extract_k3_vision.py` (165 tower tensors and 3 projector tensors, all BF16). Never download the full checkpoint (about 1.56 TB); use range requests.
- **Only the projector trains.** A frozen-weight hash test must pass before and after training.
- **Canvas:** 448×448 for M0 to M2. The visual-token count (expected 256 after the 2×2 merge) is derived from config. Never hard-code it.
- **Image marker:** one literal `<image>` in the user message expands to N placeholder positions; their embeddings are replaced with projector output through inputs_embeds. Default is bare patches (no begin or end markers). For Qwen3 the placeholder is the existing `<|image_pad|>` token; confirm its id in the pinned tokenizer.
- **Chat template:** the reader's own, with thinking off (`enable_thinking=False`, which adds an empty think block to the generation prompt). Loss on answer tokens plus the end token only. A golden-string test pins the rendered prompt.
- **Precision:** frozen modules in their shipped bf16; projector parameters in fp32 with bf16 autocast; fp32 optimizer state.
- **Seeds:** 0, 1 and 2. Never 42. Never rely on a library's default seed. A literal 42 used as a seed is a defect.
- **Attention:** for the Qwen3 readers and the K3 tower, PyTorch SDPA by default; FlashAttention 2 only after a parity test (Qwen3 uses head dimension 128 and 8 KV heads at every size, which FlashAttention 2 supports [S]). Record the backend in every run record. DeepSeek-V4 is not in M0: FlashAttention 3 and 4 do not apply to it, its candidate backends by hardware are FlashInfer, FlashMLA, SDPA and FlashAttention 2, and the M5b spike chooses.
- **Merge and token order:** the tower groups 2×2 blocks in row-major order and the projector takes four patches concatenated. Test the token order; do not assume it.

## Evidence that vision is used (non-negotiable)

Low loss, valid syntax or high exact-match on a small set is not evidence of vision.

- Answers must be balanced given the question text, so text alone cannot beat chance. Never let the question text uniquely identify the answer.
- Every evaluation reports the blank-image, shuffled-image and swapped-image controls and vision-minus-blind with a paired confidence interval.
- The 32-example overfit counts for G0 only if those controls collapse to near chance afterwards.
- Report negative and null results with the same care as positive ones. Freeze thresholds before running.

## Environment notes

- Local GPU: RTX PRO 6000 (compute capability 12.0, 96 GB) [S]. Before any training, check `torch.cuda.get_arch_list()` includes it.
- FlashAttention 3 and 4 do not run on this architecture [S]. FlashAttention 2 needs an sm_120 build or wheel [S]. Default to SDPA.
- DeepSeek-V4's official code uses its own sparse-attention kernel, not stock FlashAttention, and its head dimension is 512. Not in scope for M0.

## Licensing

The Kimi K3 License covers the K3 weights and code, including the vision tower. Keep its notice with any copy of K3 weights or vendored K3 code. Record each data source's license.

## When blocked or unsure

Stop. Write the question, mark what is [U], and propose two options with their consequences. Do not guess and continue.

## Repository layout (target)

```
CLAUDE.md  AGENTS.md  README.md  CHARTER.md  EXPERIMENTS.md
DECISIONS/          dated decision records and injection cards
docs/               the two project documents
src/sightline/      config, data, vision, projector, backbone, objective, train, evaluate, record
tests/              unit tests and tests/conformance/
tools/              extract_k3_vision.py, make_palette.py (for later V4 work)
experiments/        one directory per experiment, config plus run records
reports/            generated reports (small, diffable)
```

## Commands

- Environment: `uv sync` (reads `uv.lock`; Python 3.12+; torch 2.13.0+cu130). Without sync, prefix Python commands with `PYTHONPATH=src`.
- Tests: `pytest` (config in `pyproject.toml`; `pythonpath = ["src"]`). Tower tests are `slow`-marked and skipped by default; run with `pytest -m slow`.
- Lint: `ruff check src tests modal`.
- Format: `ruff format --check src tests modal` (write with `ruff format src tests modal`).
- Dry run: `python -m sightline.record --dry-run --out experiments/_dryrun`.
- GPU check: `python -c "import torch; print(torch.cuda.get_arch_list(), torch.cuda.is_available())"`.
- K2.6 extraction: `python tools/extract_k3_vision.py --repo moonshotai/Kimi-K2.6 --revision 7eb5002f6aadc958aed6a9177b7ed26bb94011bb --prefixes vision_tower.,mm_projector. --expect-tower 329 --expect-projector 6 --out artifacts/k26_vision_7eb5002f.safetensors --manifest docs/manifests/k26_vision_7eb5002f.manifest.json`.
- Modal parity: `python modal/wp6_parity.py --resolutions 448 672 896 --count 20`.

## Style

Concise commit messages that say why. Type hints. Comments explain intent, not mechanics. Metrics go to JSONL, not print statements. No emojis.
