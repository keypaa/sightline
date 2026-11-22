# M2a annotator concurrency probe (2026-11-22)

## Question

At what client concurrency should the 20k annotation run execute?

## Setup [V]

GPU box: 1× RTX PRO 6000, vLLM 0.34.0, Qwen/Qwen3.6-27B-FP8 at revision
`a56219b8202864dc6d3e565be5ff1eea01f54cb4`, FP8, MTP 3, `--max-model-len 8192`,
`VLLM_USE_FLASHINFER_SAMPLER=0`. Reporter commit `20fe095` on `m2a-annotation`.

`PYTHONPATH=src python3 -m sightline.m2a.verify` was not involved. The measurement is
`tools/build_m2a.py annotate --limit 200` against 200 staged COCO images, timed with
`date +%s.%N` in each probe root (`experiments/m2a_probe/c8`, `c16`, `c32`).

## Results [V]

| workers | rows | wall (s) | conversations/hour | output tok/s |
| --- | --- | --- | --- | --- |
| 8 | 200 | 86.8 | 8,298 | 254.3 |
| 16 | 200 | 32.9 | 21,900 | 671.0 |
| 32 | 200 | 30.3 | 23,736 | 722.3 |

All three runs completed `kept=200 dropped=0` with `unparsed=0` and exit 0. Chat
template SHA was `1b8361565cc35d47` (recomputed on the box, matching the earlier value).

## Reading

- Scale from 8 to 16 is **2.64×** in throughput; from 16 to 32 only **1.08×**. The knee
  is at 16. The frozen rule in the recipe says pick the lowest concurrency within ~10%
  of the best: 21,900 / 23,736 = **92.3%**, so 16 qualifies.
- Server headroom at c=32 was substantial: ~700-760 tok/s generation, KV cache below
  20%. The c=16/c=32 difference is client-side scheduling overhead and diminishing
  returns, not a hardware ceiling.
- Output tokens per conversation from the probe: 671.0 tok/s × 3600 / 21,900 conv/h =
  **110.3**, consistent with the 500-image pilot's 109.6 [V]. That agreement is a
  useful check that the probe and the pilot measure the same annotator.
- The earlier 3,276 conv/hour [V — `experiments/m2a_pilot/summary.json`] was the pilot
  running sequentially. It was never a measure of the GPU's throughput.

## Projection [I]

At 21,900 conv/hour, 19,962 conversations take **~3.3 hours**. At 23,736, ~3.0
hours — a 15-minute difference, not worth doubling client concurrency.

## Earlier failures in this probe

Three attempts before this one failed with `vLLM returned no content`. The published
diagnostic [V] sent 32 concurrent identical requests and got 0 empty replies, all
`finish_reason: stop`, so the server was healthy; the failures were client-side error
handling. Fixed in commits `03543b0`, `9af0516`, `cbb2f26`, `20fe095`, all recorded in
the branch history.

## Consequence

Decision recorded in `DECISIONS/0051-m2a-annotator-concurrency.md`.
