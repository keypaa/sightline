"""Shared MoonViT encoder wrapper.

The vendored eager attention builds a full ``(L, L)`` mask over the packed token
sequence, so a multi-image batch would be quadratic in the total token count.
``encode`` therefore runs one image per forward; the output is still
``(batch, tokens, 4 * hidden)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image


class MoonViTEncoder:
    def __init__(
        self,
        model: torch.nn.Module,
        processor,
        merge_kernel_size: tuple[int, int],
        patch_size: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model.to(device=self.device, dtype=dtype)
        self.model.eval()
        self.processor = processor
        self.merge_kernel_size = merge_kernel_size
        self.patch_size = patch_size

    def n_visual_tokens(self, grid_thws: torch.Tensor) -> int:
        """Return the merged token count for the first image in ``grid_thws``."""

        _, height, width = (int(v) for v in grid_thws[0].tolist())
        kernel_h, kernel_w = self.merge_kernel_size
        return (height // kernel_h) * (width // kernel_w)

    def preprocess(self, images: Sequence[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        medias = [{"type": "image", "image": image.convert("RGB")} for image in images]
        batch = self.processor.preprocess(medias, return_tensors="pt")
        pixel_values = batch["pixel_values"].to(device=self.device, dtype=self.dtype)
        grid_thws = batch["grid_thws"].to(device=self.device)
        return pixel_values, grid_thws

    def encode(self, images: Sequence[Image.Image]) -> torch.Tensor:
        outputs = []
        for image in images:
            pixel_values, grid_thws = self.preprocess([image])
            with torch.no_grad():
                features = self.model(pixel_values, grid_thws)
            outputs.append(features[0].reshape(features[0].shape[0], -1))
        return torch.stack(outputs, dim=0)
