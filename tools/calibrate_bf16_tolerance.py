#!/usr/bin/env python3
"""Calibrate the bf16 golden-splice tolerance on the pinned reader.

Loads the reader in fp32, records the logits and frees it; then loads bf16,
records the plain and spliced logits, and prints a fuller comparison than a
single max-abs difference:

  - max absolute and relative logit difference
  - KL divergence (expected || actual), mean, max and final position
  - argmax mismatch count and rate
  - the atol/rtol that each comparison would need

The frozen tolerance is recorded in DECISIONS/0014.

Run from the repository root:

    python tools/calibrate_bf16_tolerance.py
"""

from __future__ import annotations

import gc
import json
import resource
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sightline.backbone import splice

MODEL_ROOT = Path("artifacts/models/Qwen3-1.7B")
IMAGE_PAD = 151655
TEXT = "What color is the shape?"


def load(dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(str(MODEL_ROOT))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_ROOT), dtype=dtype)
    model.eval()
    return tok, model


def make_ids(tok) -> torch.Tensor:
    ids = tok(TEXT, return_tensors="pt")["input_ids"]
    ids[0, 2] = IMAGE_PAD
    return ids


def compare(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    expected = expected.float()
    actual = actual.float()
    abs_diff = (actual - expected).abs()
    rel_diff = abs_diff / (expected.abs() + 1e-12)

    log_p = torch.log_softmax(expected, dim=-1)
    log_q = torch.log_softmax(actual, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)

    am_expected = expected.argmax(-1)
    am_actual = actual.argmax(-1)
    mismatch = am_expected != am_actual

    return {
        "max_abs_diff": abs_diff.max().item(),
        "max_rel_diff": rel_diff.max().item(),
        "kl_mean": kl.mean().item(),
        "kl_max": kl.max().item(),
        "kl_final": kl[0, -1].item(),
        "argmax_mismatch_count": int(mismatch.sum()),
        "argmax_mismatch_total": int(mismatch.numel()),
        "argmax_mismatch_rate": mismatch.float().mean().item(),
        "atol_for_rtol_0": abs_diff.max().item(),
        "rtol_for_atol_0": rel_diff.max().item(),
    }


def main() -> int:
    if not (MODEL_ROOT / "config.json").exists():
        print(f"missing {MODEL_ROOT}/config.json; run the WP1 snapshot step", file=sys.stderr)
        return 2

    tok, model = load(torch.float32)
    ids = make_ids(tok)
    with torch.no_grad():
        fp32_logits = model(input_ids=ids).logits.clone()
    peak_fp32 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    del model, tok, ids
    gc.collect()

    tok, model = load(torch.bfloat16)
    ids = make_ids(tok)
    with torch.no_grad():
        plain = model(input_ids=ids).logits.clone()
        embed = model.get_input_embeddings()
        visual = embed(ids[:, 2:3])
        spliced = model(inputs_embeds=splice(ids, visual, IMAGE_PAD, embed.weight)).logits.clone()
    peak_after_bf16 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    plain_vs_fp32 = compare(fp32_logits, plain)
    splice_vs_plain = compare(plain, spliced)

    print(f"peak_fp32_mb={peak_fp32:.1f}")
    print(f"peak_after_bf16_mb={peak_after_bf16:.1f}")
    print("fp32_vs_bf16_plain " + json.dumps(plain_vs_fp32, sort_keys=True))
    print("bf16_splice_vs_plain " + json.dumps(splice_vs_plain, sort_keys=True))
    print("frozen_bf16_tolerance " + json.dumps({"atol": 1e-3, "rtol": 0.0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
