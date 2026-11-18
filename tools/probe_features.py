#!/usr/bin/env python3
"""Is the answer linearly decodable from the frozen visual features?

The seed-0 diagnostic showed the model fits count perfectly but sits near chance
on colour and shape, with greedy and teacher forcing agreeing. That means the
failure is in the visual pathway, not decoding. This probe separates two very
different causes:

- the features carry the answer, and the projector/reader path is losing it;
- the features do not carry the answer, in which case extraction or pooling is
  broken and no training run could have worked.

It fits a linear classifier on mean-pooled frozen features and reports held-out
accuracy against the chance rate. No reader, no projector, no training loop.

    PYTHONPATH=src python tools/probe_features.py --per-task 512
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from sightline.data.m1_synthetic import generate_m1_dataset
from sightline.vision_k3_encoder import K3VisionEncoder

K3_WEIGHTS = Path("artifacts/k3_vision_f831ab66.safetensors")
CLASSES = {"color": 8, "shape": 4, "count": 4}


def pool(features: torch.Tensor) -> torch.Tensor:
    """Mean-pool the token axis: [n, tokens, width] -> [n, width]."""

    return features.float().mean(dim=1)


def train_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    hidden: int | None = None,
    steps: int = 600,
    lr: float = 1e-3,
    seed: int = 0,
) -> nn.Module:
    """Fit a linear (or one-hidden-layer) classifier on pooled features."""

    torch.manual_seed(seed)
    width = x.shape[1]
    n_classes = int(y.max()) + 1
    layers: list[nn.Module] = []
    if hidden:
        layers += [nn.Linear(width, hidden), nn.GELU()]
        width = hidden
    layers.append(nn.Linear(width, n_classes))
    model = nn.Sequential(*layers)

    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    x = (x - mean) / std

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    model.mean = mean
    model.std = std
    model.eval()
    return model


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        z = (x - model.mean) / model.std
        return float((model(z).argmax(dim=1) == y).float().mean())


def encode_split(encoder: K3VisionEncoder, examples: list) -> torch.Tensor:
    return torch.cat([encoder.encode([e.image]) for e in examples], dim=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-task", type=int, default=512, help="examples per split")
    parser.add_argument("--hidden", type=int, default=0, help="hidden width; 0 is linear")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")

    started = time.perf_counter()
    encoder = K3VisionEncoder(
        weights=str(K3_WEIGHTS),
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    dataset = generate_m1_dataset(seed=0)

    labels = {"color": {}, "shape": {}, "count": {}}
    for name in labels:
        for example in dataset.train:
            labels[name].setdefault(example.answer, len(labels[name]))

    results = {}
    for task in ("color", "shape", "count"):
        train = [e for e in dataset.train if e.task == task][: args.per_task]
        val = [e for e in dataset.val if e.task == task][: args.per_task]

        t = time.perf_counter()
        x_train = pool(encode_split(encoder, train))
        x_val = pool(encode_split(encoder, val))
        encode_s = time.perf_counter() - t

        table = labels[task]
        y_train = torch.tensor([table[e.answer] for e in train])
        y_val = torch.tensor([table[e.answer] for e in val])

        model = train_probe(
            x_train, y_train, hidden=args.hidden or None, steps=args.steps, seed=args.seed
        )
        train_acc = accuracy(model, x_train, y_train)
        val_acc = accuracy(model, x_val, y_val)
        chance = 1.0 / len(table)
        results[task] = {
            "n_train": len(train),
            "n_val": len(val),
            "classes": len(table),
            "chance": round(chance, 4),
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "val_minus_chance": round(val_acc - chance, 4),
            "encode_s": round(encode_s, 1),
        }
        print(json.dumps({"task": task, **results[task]}), flush=True)

    print(
        json.dumps(
            {
                "hidden": args.hidden or None,
                "steps": args.steps,
                "elapsed_s": round(time.perf_counter() - started, 1),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
