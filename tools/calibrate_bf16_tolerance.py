#!/usr/bin/env python3
"""Calibrate the bf16 golden-splice tolerance on the pinned reader.

Loads the reader in fp32, records the logits and frees it; then loads bf16,
records the plain and spliced logits, and prints the maximum absolute
differences. The frozen tolerance is recorded in DECISIONS/0014.

Run from the repository root:

    python tools/calibrate_bf16_tolerance.py
"""

from __future__ import annotations

import gc
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

    plain_diff = (fp32_logits.float() - plain.float()).abs().max().item()
    splice_diff = (plain.float() - spliced.float()).abs().max().item()
    suggested = max(2.0 * splice_diff, 1e-3)

    print(f"peak_fp32_mb={peak_fp32:.1f}")
    print(f"peak_after_bf16_mb={peak_after_bf16:.1f}")
    print(f"fp32_vs_bf16_plain_max_abs_diff={plain_diff:.6e}")
    print(f"bf16_golden_splice_max_abs_diff={splice_diff:.6e}")
    print(f"suggested_bf16_tolerance={suggested:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
