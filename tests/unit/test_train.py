from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from sightline.projector import Projector
from sightline.train import Batch, Trainer

VOCAB = 128
HIDDEN = 32
PLACEHOLDER = 5


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


def make_batches(seed: int, n_micro: int, micro_batch: int = 4, seq_len: int = 6) -> list[Batch]:
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_micro):
        input_ids = torch.randint(0, VOCAB, (micro_batch, seq_len), generator=generator)
        input_ids[input_ids == PLACEHOLDER] = 0
        input_ids[:, 2] = PLACEHOLDER
        features = torch.randn(micro_batch, 1, HIDDEN, generator=generator)
        labels = input_ids.clone()
        labels[:, :3] = -100
        batches.append(Batch(input_ids, features, labels))
    return batches


def make_trainer(model: AutoModelForCausalLM, batches: list[Batch]) -> Trainer:
    torch.manual_seed(123)
    projector = Projector(input_width=HIDDEN, reader_width=HIDDEN)
    return Trainer(
        model,
        projector,
        batches,
        micro_batches_per_step=2,
        base_lr=1e-3,
        warmup_steps=5,
        max_steps=110,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        grad_clip=1.0,
        placeholder_id=PLACEHOLDER,
    )


def test_resume_reproduces_next_10_losses(tmp_path: Path) -> None:
    model = tiny_model()
    batches = make_batches(seed=0, n_micro=110 * 2)

    continuous = make_trainer(model, batches)
    continuous.train(max_steps=110)

    trainer = make_trainer(model, batches)
    trainer.train(max_steps=100)
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)

    resumed = make_trainer(model, batches)
    resumed.load_checkpoint(checkpoint)
    resumed.train(max_steps=110)

    assert len(resumed.losses) == 10
    for got, expected in zip(resumed.losses, continuous.losses[100:110], strict=True):
        assert got == pytest.approx(expected, abs=1e-6)


def test_determinism() -> None:
    model = tiny_model()
    batches = make_batches(seed=1, n_micro=5 * 2)
    first = make_trainer(model, batches)
    second = make_trainer(model, batches)
    first.train(max_steps=5)
    second.train(max_steps=5)
    assert first.losses == second.losses


def test_checkpoint_contains_required_state(tmp_path: Path) -> None:
    model = tiny_model()
    batches = make_batches(seed=2, n_micro=2 * 2)
    trainer = make_trainer(model, batches)
    trainer.train(max_steps=2)
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    state = torch.load(checkpoint, weights_only=False)
    for key in ("projector", "optimizer", "scheduler", "step", "rng"):
        assert key in state
    assert state["step"] == 2


def test_gradient_accumulation_matches_single_batch() -> None:
    model = tiny_model()
    generator = torch.Generator().manual_seed(7)
    input_ids = torch.randint(0, VOCAB, (8, 6), generator=generator)
    input_ids[input_ids == PLACEHOLDER] = 0
    input_ids[:, 2] = PLACEHOLDER
    features = torch.randn(8, 1, HIDDEN, generator=generator)
    labels = input_ids.clone()
    labels[:, :3] = -100
    full = Batch(input_ids, features, labels)

    torch.manual_seed(123)
    single_projector = Projector(input_width=HIDDEN, reader_width=HIDDEN)
    single = Trainer(
        model,
        single_projector,
        [full],
        micro_batches_per_step=1,
        base_lr=1e-3,
        warmup_steps=0,
        max_steps=1,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        grad_clip=1.0,
        placeholder_id=PLACEHOLDER,
    )
    single.train(max_steps=1)
    single_grad = single_projector.fc1.weight.grad.clone()

    small = [Batch(input_ids[i : i + 4], features[i : i + 4], labels[i : i + 4]) for i in (0, 4)]
    torch.manual_seed(123)
    accumulated_projector = Projector(input_width=HIDDEN, reader_width=HIDDEN)
    accumulated = Trainer(
        model,
        accumulated_projector,
        small,
        micro_batches_per_step=2,
        base_lr=1e-3,
        warmup_steps=0,
        max_steps=1,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        grad_clip=1.0,
        placeholder_id=PLACEHOLDER,
    )
    accumulated.train(max_steps=1)
    accumulated_grad = accumulated_projector.fc1.weight.grad.clone()

    assert torch.allclose(single_grad, accumulated_grad, atol=1e-5)


def test_cycle_reuses_batches() -> None:
    model = tiny_model()
    projector = Projector(input_width=HIDDEN, reader_width=HIDDEN)
    batches = make_batches(seed=3, n_micro=2)
    trainer = Trainer(
        model,
        projector,
        batches,
        micro_batches_per_step=2,
        base_lr=1e-3,
        warmup_steps=0,
        max_steps=4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        grad_clip=1.0,
        placeholder_id=PLACEHOLDER,
        cycle=True,
    )
    trainer.train(max_steps=4)
    assert trainer.step == 4
    assert len(trainer.losses) == 4
    assert all(math.isfinite(loss) for loss in trainer.losses)


def test_probe_reports_step_time_and_memory() -> None:
    model = tiny_model()
    batches = make_batches(seed=5, n_micro=4)
    trainer = make_trainer(model, batches)
    trainer.max_steps = 32
    trainer.cycle = True

    report = trainer.probe(3, warmup=2)

    assert trainer.step == 5
    assert len(trainer.losses) == 5
    assert report["steps"] == 3
    assert report["warmup"] == 2
    assert report["step_s"] > 0.0
    assert report["loss_first"] == trainer.losses[2]
    assert report["loss_last"] == trainer.losses[4]
    assert report["peak_gpu_memory_mb"] is None

    with pytest.raises(ValueError):
        trainer.probe(0)


def test_on_step_callback_reports_progress() -> None:
    model = tiny_model()
    batches = make_batches(seed=7, n_micro=4)
    seen: list[dict] = []

    trainer = make_trainer(model, batches)
    trainer.cycle = True
    trainer.on_step = seen.append
    trainer.train(max_steps=3)

    assert [s["step"] for s in seen] == [1, 2, 3]
    assert all(math.isfinite(s["loss"]) for s in seen)
    assert all(s["step_s"] > 0.0 for s in seen)
    assert [s["step_times"] for s in seen] == [1, 2, 3]
    assert len(trainer.step_times) == 3


def test_load_state_dict_accepts_a_checkpoint_without_rng() -> None:
    model = tiny_model()
    batches = make_batches(seed=8, n_micro=4)
    donor = make_trainer(model, batches)
    donor.cycle = True
    donor.train(max_steps=2)
    state = {k: v for k, v in donor.state_dict().items() if k != "rng"}

    fresh = make_trainer(model, batches)
    fresh.cycle = True
    fresh.load_state_dict(state)

    assert fresh.step == 2
    for trained, restored in zip(
        donor.projector.state_dict().values(),
        fresh.projector.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(trained, restored)
