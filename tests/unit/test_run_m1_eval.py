"""The fixed evaluation subset must be unbiased and the accuracy loop must run."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, Qwen3Config

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "run_m1.py"

VOCAB = 128
HIDDEN = 32
PLACEHOLDER = 5


def load_runner():
    spec = importlib.util.spec_from_file_location("run_m1_eval", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tiny_model() -> AutoModelForCausalLM:
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    return AutoModelForCausalLM.from_config(cfg).eval()


def test_sample_subset_is_reproducible_and_covers_the_list() -> None:
    runner = load_runner()
    ordered = [f"c{i % 8}" for i in range(64)]

    picked = runner.sample_subset(ordered, 16, seed=0)

    assert len(picked) == 16
    assert picked == runner.sample_subset(ordered, 16, seed=0)
    assert len(set(picked)) > 1


def test_exact_match_reports_every_task() -> None:
    runner = load_runner()
    runner.IMAGE_PAD = PLACEHOLDER
    torch.manual_seed(0)

    class StubTok:
        eos_token_id = 0
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens: bool = True) -> str:
            return "red"

    model = tiny_model()
    projector = runner.Projector(input_width=HIDDEN, reader_width=HIDDEN)

    examples = [
        SimpleNamespace(task="color", answer="red"),
        SimpleNamespace(task="shape", answer="square"),
    ]
    prompts = [
        torch.tensor([[PLACEHOLDER, 11, 12, 13]]),
        torch.tensor([[PLACEHOLDER, 11, 12, 14]]),
    ]
    features = torch.randn(2, 1, HIDDEN)

    accuracy = runner.exact_match(model, StubTok(), projector, examples, features, prompts, "cpu")

    assert accuracy == {"color": 1.0, "shape": 0.0}
