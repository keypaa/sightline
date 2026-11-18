"""The probe must recover an answer that is genuinely in the features.

A probe that cannot decode a synthetic, obviously-present signal would make a
"features do not carry colour" conclusion meaningless.
"""

from __future__ import annotations

import torch

from tools.probe_features import accuracy, pool, train_probe


def make_data(n: int, n_classes: int, width: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % n_classes
    x = torch.randn(n, width, generator=generator) * 0.1
    x[torch.arange(n), labels] += 4.0
    return x, labels


def test_pool_averages_the_token_axis() -> None:
    features = torch.ones(2, 4, 8)
    assert pool(features).shape == (2, 8)
    assert torch.allclose(pool(features), torch.ones(2, 8))


def test_probe_recovers_a_signal_that_is_present() -> None:
    x, y = make_data(400, 8, 16)

    model = train_probe(x, y, steps=400, lr=1e-2)
    score = accuracy(model, x, y)

    assert score > 0.9


def test_probe_beats_chance_by_a_margin_on_noisy_data() -> None:
    x, y = make_data(400, 8, 16)
    x = x + torch.randn_like(x) * 1.0

    model = train_probe(x, y, steps=400, lr=1e-2)
    score = accuracy(model, x, y)

    assert score > 0.5


def test_probe_with_hidden_layer_still_recovers_the_signal() -> None:
    x, y = make_data(400, 8, 16)

    model = train_probe(x, y, hidden=32, steps=400, lr=1e-2)
    score = accuracy(model, x, y)

    assert score > 0.9


def test_label_tables_are_per_task() -> None:
    from types import SimpleNamespace

    from tools.probe_features import label_tables

    examples = [
        SimpleNamespace(task="color", answer="red"),
        SimpleNamespace(task="color", answer="blue"),
        SimpleNamespace(task="shape", answer="square"),
        SimpleNamespace(task="count", answer="1"),
        SimpleNamespace(task="count", answer="2"),
    ]

    tables = label_tables(examples)

    assert len(tables["color"]) == 2
    assert len(tables["shape"]) == 1
    assert len(tables["count"]) == 2


def test_sample_evenly_spreads_across_ordered_items() -> None:
    from tools.probe_features import sample_evenly

    ordered = [f"c{i % 8}" for i in range(64)]

    picked = sample_evenly(ordered, 8)

    assert len(picked) == 8
    assert len(set(picked)) == 8
    assert picked[0] == "c0"
