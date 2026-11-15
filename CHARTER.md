# CHARTER

Short and stable. Changing anything here needs a new dated entry in DECISIONS/.

## The question

Can a small trained projector give a frozen, text-only open-weight LLM useful vision, and where does that stop?

## Purpose

Sightline is a research and learning project. The immediate goal matters more than the long-term target: after the first three milestones, the owner should understand how a multimodal model is trained, with a pipeline they trust and results they can defend. The long-term target is DeepSeek-V4-Flash-0731 with a Kimi MoonViT-class encoder.

## Non-goals until a gate says otherwise

- Modifying LM weights or training an encoder (except in the optional M2c unfreezing ladder).
- Serving infrastructure.
- Distributed-training frameworks.
- Browser agents.
- Parity claims with native multimodal models.
- Any claim about "all model pairs".

## Milestones and gates

Gate definitions live in docs/sightline-project-document.md, section 9.4. Effort estimates and the plan for each milestone live there too.

| Milestone | Goal | Gate |
| --- | --- | --- |
| M0 Skeleton | Splice, masks, freezing, generation and tests on Qwen3-1.7B with MoonViT-V2; K2.6 MoonViT extraction; one injection card per reader | G0 |
| M1 Synthetic | Paired sensitivity and generalization on controlled tasks | G1 |
| M2a Reproduction | Baseten recipe on Qwen3-4B with the K2.6 encoder | G2 |
| M2b Encoder A/B | MoonViT-V2 versus K2.6 under identical settings | G3 |
| M2c Unfreezing ladder (optional) | Projector only, plus LoRA on the LM, plus the encoder | G3b |
| M3 UI recognition | Synthetic and real screenshots | G4 |
| M4 Grounding | Pointing and region prediction | G5 |
| M5a Ladder | Qwen3-8B and 14B | G6 |
| M5b Spike | Hero feasibility on a rented multi-GPU node | G7 |
| M5c Hero | DeepSeek-V4-Flash-0731 plus native comparison rows | G8 |
| M6 Optional | RL on a rendered-math task to restore reasoning | none yet |

Current stage: M0 complete (G0 signed off 2026-11-15); M1 next.

## Honesty rules

- Negative and null results are results. Archive them with the same metadata as successes.
- Thresholds are frozen in DECISIONS/ before a run, never after.
- One change per experiment. Configs are immutable after launch. Never overwrite results.
- Vision is claimed only from vision-minus-blind with a paired confidence interval and the blank, shuffled and swapped-image controls. Low loss is not evidence.
- Every non-trivial claim carries an evidence tag: [V] verified, [S] secondary, [I] inference, [U] unknown.
- Exploration may happen in notebooks, but logic lives in src/. Code from the old vision-adapter project enters only with a test and a one-line justification (default: it does not enter).

## Working agreements

- DECISIONS/ overrides the documents in docs/. If two sources disagree, stop and ask.
- Rules for coding agents are in AGENTS.md.
- The Kimi K3 License applies to the K3 weights and any vendored K3 code; keep its notice with them.
