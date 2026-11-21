# M2a annotator recipe: Qwen3.6-27B-FP8 on one RTX PRO 6000

Reproduction record for the local annotation server. Every value here was run on the
GPU box on 2026-11-21, not read from a table. The authority for version floors is
`DECISIONS/0050`; the authority for the corpus and thresholds is `DECISIONS/0046`
and `DECISIONS/0049`.

## Why this recipe exists

The API transport (Kilo free tier) is capped at roughly 200 requests per hour per IP
[V — Kilo docs], and two runs died near 145 calls. At that rate 20k conversations is a
multi-week job. A local vLLM server removes the quota entirely.

## 1. Install (serving box only)

```bash
uv pip install -U vllm --torch-backend auto
uv pip install -U "transformers>=5.8.0"
```

These are not project dependencies. The annotation client speaks HTTP, so the repo's
test environment never imports vLLM.

## 2. Serve

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_RUST_FRONTEND=1
export OMP_NUM_THREADS=1

vllm serve Qwen/Qwen3.6-27B-FP8 \
  --tensor-parallel-size 1 --max-num-seqs 512 \
  --reasoning-parser qwen3 --mm-encoder-tp-mode data \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

### Two flags that are not optional on this GPU

| flag | why |
| --- | --- |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | FlashInfer's JIT sampler fails to build against the pip CUDA 13 headers: `CUDA compiler and CUDA toolkit headers are incompatible`. The flag falls back to vLLM's native PyTorch sampler. The model loads fine either way; the failure happens at the first sampling step. |
| `OMP_NUM_THREADS=1` | The default (20) makes Torch warn about spin-wait contention and cgroup throttling during serving. |

Both were found by running the recipe. Without them the server starts and then dies
during warm-up, which looks like a model problem but is not.

### Expected healthy startup

```
Model loading took 28.94 GiB memory
GPU KV cache size: 862,563 tokens
init engine (profile, create kv cache, warmup model) took 173.70 s
OpenAI server is ready to accept requests bind_address="0.0.0.0:8000"
```

### Throughput observed

| metric | value |
| --- | --- |
| single-stream generation | ~92.7 tok/s |
| speculative decoding acceptance | 3.77 tokens mean, ~92% draft acceptance |
| per-request latency, image + prompt | ~0.7–1.6 s |

The pilot is sequential by design, because it measures per-conversation quality rather
than throughput. A 20k run would need concurrency.

## 3. Run the pilot

```bash
PYTHONPATH=src python3 tools/m2a_annotator_pilot.py \
  --images 500 \
  --exclude experiments/m2a_grounding/labels.jsonl \
  --host http://localhost:8000/v1 \
  --model Qwen/Qwen3.6-27B-FP8 \
  --revision a56219b8202864dc6d3e565be5ff1eea01f54cb4 \
  --quantization fp8 \
  --engine-version "$(python3 -c 'import vllm; print(vllm.__version__)')" \
  --chat-template-sha256 "$(python3 -c "
from transformers import AutoTokenizer; import hashlib
t=AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B-FP8')
print(hashlib.sha256(t.chat_template.encode()).hexdigest()[:16])")"
```

### Why each argument is there

| argument | value | source |
| --- | --- | --- |
| `--images 500` | fresh sample, large enough to resolve the rates | `DECISIONS/0049` |
| `--exclude` | the 50 burned images that shaped the instrument | `DECISIONS/0049` |
| `--model` | the pinned annotator | `DECISIONS/0040` |
| `--revision` | `a56219b8202864dc6d3e565be5ff1eea01f54cb4` — the checkpoint's own commit | Hugging Face API, read 2026-11-21 |
| `--quantization` | `fp8` | the served checkpoint |
| `--engine-version` | resolved from the installed vLLM | recorded, not assumed |
| `--chat-template-sha256` | resolved from the tokenizer | recorded, not assumed |

The four provenance flags are **required**. The pilot refuses to write a row without
them, because a row that cannot be reproduced is worse than no row.

### What it writes

| file | contents |
| --- | --- |
| `experiments/m2a_pilot/pilot.jsonl` | one row per conversation, with full provenance |
| `experiments/m2a_pilot/summary.json` | parse rate, token means, wall time, throughput |
| `experiments/m2a_pilot/run_record.json` | git state, config, and the summary together |
| `reports/m2a-annotator-pilot-local.md` | the human-readable run record |

### Reading the exit code

| code | meaning |
| --- | --- |
| 0 | parse bar met (and grounding measured and met, when measurable) |
| 1 | a bar was missed, or the run stopped early |
| 2 | refused to run: no input, or everything excluded |

**Grounding is not measurable on COCO** and is reported as `grounding_skipped`, not as
a failure (`DECISIONS/0049`). The gate that matters here is claim contradiction,
measured afterwards by `sightline.m2a.verify`.

## 4. After the pilot: claim verification

Runs on the laptop or the box, no GPU needed:

```bash
PYTHONPATH=src python3 -m sightline.m2a.verify extract \
  --conversations experiments/m2a_pilot/pilot.jsonl

PYTHONPATH=src python3 -m sightline.m2a.verify plant \
  --claims experiments/m2a_grounding/claims.jsonl
```

Frozen thresholds (`DECISIONS/0049`): contradicted rate ≤ 0.15, planted-error
rejection ≥ 0.90, human agreement ≥ 0.85.

## 5. Reproducing this later

Record with every run: the vLLM version, the transformers version, the model revision,
the four provenance flags, and the two environment flags above. The engine version is
already in every pilot row and in `run_record.json`.

If vLLM or transformers is upgraded, re-run the 3-image smoke test before trusting a
full run — kernel selection and chat-template rendering can both change.