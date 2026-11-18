"""VisualSpec must keep the placeholder count and feature shape in step.

The single-token ablation changes both at once; if they disagree, splice raises
or silently pairs the wrong features with the wrong question.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "run_m1.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_m1_visual", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_patches_mode_keeps_every_token() -> None:
    runner = load_runner()
    visual = runner.VISUAL_SPECS[256]

    features = torch.zeros(2, 256, 4096)

    assert visual.placeholder.count("<|image_pad|>") == 256
    assert visual.prepare(features).shape == (2, 256, 4096)


def test_mean_mode_produces_one_token_of_the_same_width() -> None:
    runner = load_runner()
    visual = runner.VISUAL_SPECS[1]

    features = torch.arange(2 * 256 * 4096, dtype=torch.float32).reshape(2, 256, 4096)

    prepared = visual.prepare(features)

    assert visual.placeholder.count("<|image_pad|>") == 1
    assert prepared.shape == (2, 4096)
    assert torch.allclose(prepared[0], features[0].mean(dim=0))
