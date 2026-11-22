# M2a annotator recipe — end to end

Reproduction record for the local annotation server and the corpus build. Every value
here was run on the GPU box on 2026-11-21, not read from a table.

Authorities: `DECISIONS/0040` (annotator contract), `0046` (corpus source), `0049`
(claim verification), `0050` (dependencies), and the schema spec
`zenno/specs/2026-11-21-m2a-corpus-schemas.md`.

## 0. What this produces

```
fetched.jsonl → conversations.jsonl → claims.jsonl → verdicts.jsonl
                                                   → blind.jsonl
                                                   → decisions.jsonl
                                                   → corpus-manifest.json
```

Each row carries the identity of what produced it. `claim_id` is a content hash and
joins every layer. One annotator (Qwen3.6-27B-FP8), one verifier (hy4-flash), one blind
reader (hy4-flash), one human review.

## 1. Clone and install (GPU box)

From a clean directory on the GPU box:

```bash
git clone https://github.com/keypaa/sightline.git
cd sightline
git checkout m2a-annotation
git pull origin m2a-annotation

uv sync
uv pip install -U vllm --torch-backend auto
uv pip install -U "transformers>=5.8.0"

git rev-parse HEAD          # record this with the run
```

The branch matters: `main` does not contain the corpus pipeline. `uv sync` installs the
project; the two `uv pip install` lines add the serving-only dependencies.

Floors are pinned in `DECISIONS/0050`: vLLM ≥ 0.17.0, transformers ≥ 5.8.0. These are
**serving-box only** — out of `pyproject.toml` — because the client speaks HTTP. The
verified engine version was **0.34.0**; record whatever is installed with
`python3 -c 'import vllm; print(vllm.__version__)'`.

## 2. Stage the images on the GPU box

The image cache is **not** committed, but the manifest is — and it carries each image's
COCO URL and sha256. The cache is reconstructed from it, so nothing is transferred by
hand:

```bash
cd /path/to/sightline

# The 200 images the c=8/16/32 probes will read
PYTHONPATH=src python3 tools/build_m2a.py stage-images \
  --manifest experiments/m2a_iter20k/fetched.jsonl \
  --cache artifacts/m2a_images --workers 8 --limit 200

# Later, before the full run: all 19,962
PYTHONPATH=src python3 tools/build_m2a.py stage-images \
  --manifest experiments/m2a_iter20k/fetched.jsonl \
  --cache artifacts/m2a_images --workers 16
```

`stage-images` is idempotent. A file already on disk with the manifest's sha256 is not
downloaded; a file whose bytes disagree is deleted and re-fetched. Verified locally:
staging 200 images over a complete cache reported `present=200 downloaded=0 failed=0`.

It exits 0 when every image is present and correct, 1 if any failed, 2 if the manifest
is missing. Re-run it safely at any time — a partial download is not a problem.

## 3. Serve

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0     # DO NOT REMOVE — see below
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_RUST_FRONTEND=1
export OMP_NUM_THREADS=1

vllm serve Qwen/Qwen3.6-27B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 512 \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

### `VLLM_USE_FLASHINFER_SAMPLER=0` is not optional

FlashInfer's JIT sampler fails to compile against the pip CUDA 13 headers:

```
error: "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
```

The model loads fine; the server dies at the first sampling step during warm-up. The flag
falls back to vLLM's native PyTorch sampler. `OMP_NUM_THREADS=1` clears a spin-wait
contention warning from the default of 20.

`--max-model-len 8192` matters for throughput: our prompts are ~460 input + ~110 output
tokens. Serving a 262K context reserves capacity for length we never use.

### Expected healthy startup

```
Model loading took 28.94 GiB memory
GPU KV cache size: 862,563 tokens
init engine (profile, create kv cache, warmup model) took ~174 s
OpenAI server is ready to accept requests bind_address="0.0.0.0:8000"
```

## 4. Find your concurrency before the full run

Published numbers for this model on one RTX PRO 6000, 8K context [S — LocalMaxxing]:

| concurrency | aggregate tok/s |
|---|---|
| 1 | 78 |
| 8 | 483 |
| 16 | 770 |
| 32 | 1,103 |
| 64 | 1,425 |

Our pilot ran sequentially at **99.7 tok/s**, 3,276 conversations/hour. The annotate
command is concurrent (`--workers`); the pilot simply didn't use it. The projection at
c=32 is ~36,000 conversations/hour — roughly 20k in 35 minutes.

**Measure before committing** (rule 9). `annotate` reads `fetched.jsonl` from its
`--root`, so each probe needs its own root. The `cp` below copies that small committed
**manifest** — not images; the images were staged by section 2 into the shared cache.
Wall time is not in the run record, so `date` brackets each run and writes
`WALL_SECONDS` to a file. This works on boxes without `/usr/bin/time`.

First compute the template hash once and export it:

```bash
export TEMPLATE_SHA="$(python3 -c "
from transformers import AutoTokenizer; import hashlib
t=AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B-FP8')
print(hashlib.sha256(t.chat_template.encode()).hexdigest()[:16])")"
echo "$TEMPLATE_SHA"
```

Then give each probe its own root and run it:

```bash
for W in 8 16 32; do
  mkdir -p experiments/m2a_probe/c$W
  cp experiments/m2a_iter20k/fetched.jsonl experiments/m2a_probe/c$W/fetched.jsonl
done

for W in 8 16 32; do
  echo "=== concurrency $W ==="
  START=$(date +%s.%N)
  PYTHONPATH=src python3 tools/build_m2a.py --root experiments/m2a_probe/c$W annotate \
      --workers $W --limit 200 \
      --host http://localhost:8000/v1 \
      --model Qwen/Qwen3.6-27B-FP8 \
      --revision a56219b8202864dc6d3e565be5ff1eea01f54cb4 \
      --quantization fp8 --engine-version 0.34.0 \
      --chat-template-sha256 "$TEMPLATE_SHA" \
    | tee experiments/m2a_probe/c$W/log.txt
  END=$(date +%s.%N)
  python3 -c "print(f'WALL_SECONDS={$END - $START:.3f}')" \
    > experiments/m2a_probe/c$W/time.txt
done
```

### Reading the score

For each concurrency, from the files just written:

```bash
for W in 8 16 32; do
python3 - "$W" <<'PY'
import json, sys
from pathlib import Path
w = sys.argv[1]
root = Path(f"experiments/m2a_probe/c{w}")
wall = float([l for l in (root/"time.txt").read_text().splitlines()
              if l.startswith("WALL_SECONDS=")][0].split("=")[1])
rows = [json.loads(l) for l in (root/"conversations.jsonl").read_text().splitlines() if l.strip()]
out = sum(r.get("tokens_out") or 0 for r in rows)
print(f"c={w}: n={len(rows)} wall={wall:.1f}s "
      f"conv/hour={len(rows)*3600/wall:.0f} output_tok/s={out/wall:.1f}")
PY
done
```

Pick the knee: the lowest concurrency whose conv/hour is within ~10% of the best.
Higher c adds latency and risk without much throughput. Record all three numbers in
the run record before choosing — do not pick 32 because the table says so.

## 5. Run the 20k annotation

Use the `$TEMPLATE_SHA` exported in section 4 and the `--workers` value you selected
from the probe. Do not reuse the hardcoded values from an earlier run.

```bash
cd /path/to/sightline

PYTHONPATH=src python3 tools/build_m2a.py --root experiments/m2a_iter20k annotate \
  --workers <WINNER_FROM_PROBE> \
  --host http://localhost:8000/v1 \
  --model Qwen/Qwen3.6-27B-FP8 \
  --revision a56219b8202864dc6d3e565be5ff1eea01f54cb4 \
  --quantization fp8 \
  --engine-version 0.34.0 \
  --chat-template-sha256 "$TEMPLATE_SHA" \
  --seed 0
```

The four provenance flags are **required** — argparse exits 2 without them. A row that
cannot be reproduced is worse than no row.

What it writes:

| file | contents |
| --- | --- |
| `experiments/m2a_iter20k/conversations.jsonl` | one row per image, full provenance |
| `experiments/m2a_iter20k/annotate_failures.jsonl` | per-run audit of unusable replies |

**Exit codes:** 0 = all pending done; 1 = partial results kept, re-run to resume; 2 =
refused to run. Resume is by id and safe — only `parsed=True` rows count as done.
`order.jsonl` fixes the corpus permutation; re-running does not re-shuffle.

## 6. Verify (no GPU needed, run on the laptop)

Copy `conversations.jsonl` back to the laptop. All 19,994 images are already in
`artifacts/m2a_images/`, so no image transfer is needed.

```bash
PYTHONPATH=src python3 -m sightline.m2a.verify extract \
  --conversations experiments/m2a_iter20k/conversations.jsonl \
  --out experiments/m2a_grounding/claims.jsonl

PYTHONPATH=src python3 -m sightline.m2a.verify plant \
  --claims experiments/m2a_grounding/claims.jsonl \
  --out experiments/m2a_grounding/planted.jsonl

# Judge real + planted with one verifier. Resumable: re-run after a quota interruption.
PYTHONPATH=src python3 -m sightline.m2a.verify judge \
  --claims experiments/m2a_grounding/claims.jsonl \
  --planted experiments/m2a_grounding/planted.jsonl \
  --out experiments/m2a_grounding/verdicts.jsonl \
  --workers 1

PYTHONPATH=src python3 -m sightline.m2a.verify blind \
  --claims experiments/m2a_grounding/claims.jsonl \
  --out experiments/m2a_grounding/blind.jsonl
```

`judge` writes **flat rows**: one per claim per verifier, appended as each completes. A
failed call is written with `error` set, never counted as `can't-tell`. A verdict from a
different judging prompt is not reused on resume.

## 7. Human review

```bash
PYTHONPATH=src python3 tools/m2a_review.py \
  --claims experiments/m2a_grounding/verdicts.jsonl \
  --images-dir artifacts/m2a_images \
  --out /tmp/opencode/m2a_review.html \
  --sample 50 --seed 0
```

The page is self-contained. Checkboxes persist in browser local storage. **Export
decisions** downloads a JSON file; fold it in with:

```bash
PYTHONPATH=src python3 tools/m2a_review.py \
  --merge ~/Downloads/m2a_review_decisions.json \
  --claims experiments/m2a_grounding/claims.jsonl \
  --decisions-out experiments/m2a_grounding/decisions.jsonl \
  --reviewer owner
```

Decisions whose `(image_id, turn, claim_id)` cannot be resolved are rejected, not
written. `truth` is optional — fill it when a verdict is wrong.

## 8. Gates

```bash
PYTHONPATH=src python3 -m sightline.m2a.verify report \
  --verdicts experiments/m2a_grounding/verdicts.jsonl \
  --blind experiments/m2a_grounding/blind.jsonl

PYTHONPATH=src python3 tools/build_m2a.py \
  --root experiments/m2a_iter20k corpus-manifest
```

Frozen in `DECISIONS/0049`:

| gate | threshold |
| --- | --- |
| contradicted rate, upper 95% | ≤ 0.15 |
| planted-error rejection | ≥ 0.90 |
| human agreement | ≥ 0.85 |

The contradiction rate is over judged claims only; rows with `error` are gaps. The
manifest names the git commit that built it and emits `n_judged` and `n_planted` so the
denominators are visible.

**Judge the run on the contradiction rate, not the old grounded bar.** The 0.90 from
`DECISIONS/0040` belonged to the retired attribute-map instrument.

## 9. Before a full 131k run

Two things are still `[U]` in `DECISIONS/0046`:

- MMMU-Pro and BLINK image sets at a pinned revision, for SSCD decontamination.
- The number of conversations per image actually achieved by the 20k run.

Do not launch 131k until both are closed.

## Troubleshooting

| symptom | cause |
| --- | --- |
| server dies during warm-up with "CUDA compiler and CUDA toolkit headers are incompatible" | `VLLM_USE_FLASHINFER_SAMPLER=0` was dropped |
| `argparse: the following arguments are required: --revision` | a provenance flag is missing; supply the real value |
| `contradiction_rate: null` in report | no verdict rows, or every one has `error` set |
| judge produces no file | a failed call with no `error` row — check the build; errors must be persisted |
| resume re-judges everything | claims were extracted before `claim_id` was added; re-run `extract` |
| human decisions all rejected | old review page without `claim_id`/`turn`; regenerate the page |
