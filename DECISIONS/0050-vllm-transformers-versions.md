# DECISIONS 0050: vLLM and transformers for the M2a local annotator

Date: 2026-11-21
Status: accepted
Work package: M2a
Owner instruction: record the two libraries and their versions with the working recipe.

## Context

M2a annotation moved from the Kilo API transport to a local vLLM server, because the
free API quota (~200 requests/hour per IP [V — Kilo docs]) makes 20k conversations a
multi-week job. The annotator is `Qwen/Qwen3.6-27B-FP8` served on the local
RTX PRO 6000.

Two dependencies are new to the project, and `AGENTS.md` rule 6 requires a DECISIONS
entry for each.

## Decision

| dependency | floor | why this floor |
| --- | --- | --- |
| `vllm` | **>= 0.17.0** | the floor the Qwen3.6 recipe states; older versions do not parse the model's `Qwen3_5Config` |
| `transformers` | **>= 5.8.0** | matches the version `config.json` was written by, and provides the Qwen3-VL processor classes |

Both are installed with:

```bash
uv pip install -U vllm --torch-backend auto
uv pip install -U "transformers>=5.8.0"
```

Neither is added to `pyproject.toml`. They are **serving-box only**: the annotation
client talks HTTP, so the repo's own test suite does not import vLLM and CI stays
light. Recording them here rather than as project dependencies keeps the training
environment and the serving environment separate.

## The verified serving configuration

The full command, including the flags needed on this GPU, is in
`docs/m2a-annotator-recipe.md`. Two settings there are not optional on this box:

- `VLLM_USE_FLASHINFER_SAMPLER=0` — FlashInfer's JIT sampler fails to compile against
  the pip-installed CUDA 13 headers with
  `"CUDA compiler and CUDA toolkit headers are incompatible"`. The flag falls back to
  vLLM's native PyTorch sampler.
- `OMP_NUM_THREADS=1` — the default (20) causes spin-wait contention during serving.

Both were found by running the recipe, not by reading it. They are recorded so the
next operator does not rediscover them.

## Consequences

- The serving box needs vLLM >= 0.17.0 and transformers >= 5.8.0. The repo's own
  environment (`uv sync`) does not install them.
- A vLLM or transformers upgrade can change kernel selection or parsing behaviour.
  The **engine version is recorded in every pilot row and run record**, so a result
  can be tied to the version that produced it.
- The recipe document is the reproduction path; this entry is the authority for the
  version floors.

## Open items

- [U] Whether the FlashInfer sampler can be restored with a matching CUDA toolkit
  instead of being disabled. Closing step: install a CUDA toolkit whose headers match
  the nvcc the wheel ships, then re-run without the flag and compare throughput.
  Owner: Keylhan Paumard--André.