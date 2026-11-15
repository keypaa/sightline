"""Freezing and parameter-hash utilities.

The reader and encoder are frozen (eval mode, ``requires_grad=False``); only the
projector trains. Frozen hashes are SHA-256 of the parameter bytes.
"""

from __future__ import annotations

import hashlib

import torch
from torch import nn


def freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def parameter_hashes(module: nn.Module) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, parameter in module.named_parameters():
        raw = parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return hashes


def assert_frozen(module: nn.Module, before: dict[str, str]) -> None:
    after = parameter_hashes(module)
    if after == before:
        return
    changed = [name for name in before if before[name] != after.get(name)]
    missing = [name for name in before if name not in after]
    raise AssertionError(f"frozen parameters changed: {changed[:5]}; missing: {missing[:5]}")
