from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

MODEL_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "Qwen3-1.7B"
GOLDEN = Path(__file__).parent / "golden" / "qwen3-1p7b_prompt.txt"
GOLDEN_SHA256 = "1d91a8fec65d03716b986d047a50d68711028dd97f742f687ef69bee7379c454"

pytestmark = pytest.mark.skipif(
    not (MODEL_ROOT / "config.json").exists(),
    reason="Qwen3-1.7B snapshot not downloaded; run the WP1 snapshot step",
)


def test_reader_config_matches_pinned_checkpoint() -> None:
    cfg = json.loads((MODEL_ROOT / "config.json").read_text())
    assert cfg["model_type"] == "qwen3"
    assert cfg["hidden_size"] == 2048
    assert cfg["num_hidden_layers"] == 28
    assert cfg["num_attention_heads"] == 16
    assert cfg["num_key_value_heads"] == 8
    assert cfg["head_dim"] == 128
    assert cfg["vocab_size"] == 151936
    assert cfg["tie_word_embeddings"] is True
    assert cfg["torch_dtype"] == "bfloat16"
    assert cfg["max_position_embeddings"] == 40960
    assert cfg["rope_theta"] == 1000000


def test_image_pad_token_id() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_ROOT))
    assert tok.convert_tokens_to_ids("<|image_pad|>") == 151655


def test_special_token_ids() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_ROOT))
    assert tok.eos_token_id == 151645
    assert tok.pad_token_id == 151643


def test_chat_template_golden_string() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_ROOT))
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "<image>What color is the shape?"}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    assert rendered == GOLDEN.read_text()
    assert hashlib.sha256(rendered.encode()).hexdigest() == GOLDEN_SHA256


def test_placeholder_not_in_plain_text() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_ROOT))
    ids = tok("What color is the shape?")["input_ids"]
    assert 151655 not in ids


def test_text_only_forward_smoke() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_ROOT))
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_ROOT), dtype=dtype)
    model.eval()
    batch = tok("What color is the shape?", return_tensors="pt")
    with torch.no_grad():
        out = model(**batch)
    assert out.logits.shape == (1, batch["input_ids"].shape[1], 151936)
    assert torch.isfinite(out.logits).all()
