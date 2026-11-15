from __future__ import annotations

import torch

from sightline.vision_k3.modeling_k3_vision import tpool_patch_merger


def test_merge_order_is_row_major() -> None:
    x = torch.arange(16, dtype=torch.float32).view(16, 1)
    grid = torch.tensor([[1, 4, 4]])
    out = tpool_patch_merger(x, grid, merge_kernel_size=(2, 2))
    assert isinstance(out, list) and len(out) == 1
    merged = out[0]
    assert merged.shape == (4, 4, 1)
    expected = [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15],
    ]
    for row, values in enumerate(expected):
        assert merged[row].flatten().tolist() == values


def test_merge_448_grid_has_256_tokens() -> None:
    x = torch.zeros(32 * 32, 8)
    out = tpool_patch_merger(x, torch.tensor([[1, 32, 32]]), merge_kernel_size=(2, 2))
    assert out[0].shape == (256, 4, 8)